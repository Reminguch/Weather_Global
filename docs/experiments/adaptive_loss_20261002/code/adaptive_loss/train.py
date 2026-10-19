"""Bounded matched fixed / calibrated / adaptive K=2 experiments.

Independent entry point. No changes to production training defaults. All
checkpoint ranking uses the same fixed physical metric, including update zero.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def source_hashes():
    from src.models.neuralgcm_residual.io import sha256
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'third_party').rglob('*.py'))
    paths+=list((ROOT/'experimental/adaptive_loss').glob('*.py'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resources',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--lifecycle-report',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('fixed','calibrated','adaptive'),default='fixed')
    p.add_argument('--updates',type=int,default=100)
    p.add_argument('--validate-every',type=int,default=20)
    p.add_argument('--validation-origins',type=int,default=8)
    p.add_argument('--validation-offset',type=int,default=80)
    p.add_argument('--calibration-batches',type=int,default=8)
    p.add_argument('--window',type=int,default=100)
    p.add_argument('--interval',type=int,default=20)
    p.add_argument('--peak-lr',type=float,default=.002)
    p.add_argument('--warmup',type=int,default=2000)
    p.add_argument('--max-seconds',type=float,default=3000)
    p.add_argument('--stop-after',type=int)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.updates,a.validate_every,a.validation_origins,a.calibration_batches,a.warmup)<1 or a.validation_origins%2:
        p.error('Positive budgets and even validation origin count required')
    if a.validation_offset<0 or not 0<a.peak_lr<1 or not 0<a.max_seconds<3500:
        p.error('Invalid offset, learning rate, or runtime')
    if a.stop_after is not None and not 1<=a.stop_after<=a.updates:
        p.error('Invalid stop-after')
    from src.models.neuralgcm_residual.paper_execution import configure_paper_environment
    configure_paper_environment()
    import jax
    import numpy as np
    from src.models.neuralgcm_residual.runtime import build_runtime
    from src.models.neuralgcm_residual.config import RunConfig,Architecture,BALANCED_LOSS_NAME
    from src.models.neuralgcm_residual.data import seasonal_order,eligible_origins
    from src.models.neuralgcm_residual.pressure_weighted_loss import PressureWeightedLoss
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest
    from src.models.neuralgcm_residual.io import read_json,write_json,digest,sha256,versions
    from src.models.neuralgcm_residual.paper_lifecycle import lock_run
    from src.models.neuralgcm_residual.physical_selection import headline_rmse_ratio,HEADLINES
    from experimental.adaptive_loss.controller import WindowBalancer,BalanceConfig
    from experimental.adaptive_loss.objective import GroupedLoss,calibrate
    from experimental.adaptive_loss.trainer import AdaptiveTrainer,finite_tree,tree_norm
    from experimental.adaptive_loss.checkpoint import validate,commit,materialize
    if not os.environ.get('SLURM_JOB_ID') or not all(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('A Slurm GPU allocation is required')
    hashes=source_hashes()
    gate=read_json(a.preflight)
    if not gate.get('passed') or not gate.get('pilot') or gate['resources_sha256']!=sha256(a.resources):
        raise ValueError('Missing/mismatched independent preflight')
    for name in ('controller.py','objective.py','trainer.py'):
        path='experimental/adaptive_loss/'+name
        if gate['source_hashes'][path]!=hashes[path]:
            raise ValueError('Numerical code differs from preflight: '+name)
    if not a.pilot:
        if a.lifecycle_report is None:
            raise ValueError('A bounded trial requires the separate-process lifecycle smoke')
        lifecycle=read_json(a.lifecycle_report)
        if not lifecycle.get('passed') or lifecycle['source_hashes']!=hashes:
            raise ValueError('Missing/mismatched lifecycle smoke')
    lock=lock_run(a.output,resume=a.resume)
    started=time.monotonic()
    log=lambda **x:print(json.dumps(x,allow_nan=False),flush=True)
    resources,stats=read_json(a.resources),read_json(a.statistics)
    if ('pilot' in stats.get('scope','').lower())!=a.pilot:
        raise ValueError('Pilot/trial statistics scope mismatch')
    if not a.pilot and stats['samples']!=60:
        raise ValueError('Trial requires training-only full calibration statistics')
    r=build_runtime(RunConfig(architecture=Architecture(width=128,d_inner=16),
        loss=BALANCED_LOSS_NAME,correction_policy='no_pressure_zero_mean_v2'),resources,stage='finetune')
    if stats['dataset_id']!=r.store.identity or stats['checkpoint_sha256']!=resources['checkpoint_sha256']:
        raise ValueError('Statistics dataset/checkpoint mismatch')
    before=backbone_parameter_digest(r.backbone.model)
    grouped=GroupedLoss(r.backbone.model,stats,2)
    fixed=PressureWeightedLoss(r.backbone.model,stats,2,upper_pressure_weight=1e-5)
    trainer=AdaptiveTrainer(r,fixed,grouped,peak_learning_rate=a.peak_lr,warmup_steps=a.warmup)
    bc=BalanceConfig(window=a.window,interval=a.interval)
    all_origins=seasonal_order(eligible_origins(r.store.times,'train',2,warm_hours=0,daily=False))
    calibration_origins=[all_origins[i] for i in np.linspace(0,len(all_origins)-1,2*a.calibration_batches,dtype=int)]
    origins=[o for o in all_origins if o not in set(calibration_origins)]
    validation_origins=seasonal_order(eligible_origins(r.store.times,'val',2,warm_hours=0))[
        a.validation_offset:a.validation_offset+a.validation_origins]
    if len(set(calibration_origins))!=2*a.calibration_batches or len(validation_origins)!=a.validation_origins or len(origins)<2:
        raise ValueError('Insufficient split-safe origins')
    metadata=dict(version='adaptive_loss_bounded_trial_v1',mode=a.mode,pilot=a.pilot,K=2,
        architecture=dict(width=128,d_inner=16),seed=22,batch_size=2,updates=a.updates,
        beta=1e-5,balance=asdict(bc),group_names=grouped.names,priors=grouped.priors.tolist(),
        calibration_origins=calibration_origins,origin_order_sha256=digest(origins),
        validation_origins=validation_origins,validate_every=a.validate_every,
        optimizer=dict(peak_lr=a.peak_lr,warmup=a.warmup),statistics_sha256=sha256(a.statistics),
        resources_sha256=sha256(a.resources),source_hashes=hashes,libraries=versions(),
        selection='fixed mean headline RMSE ratio; worst headline ratio <=1.05; initial included',
        physical_feedback='stop_gradient',memory_gradient='within-episode BPTT')
    identity=digest(metadata)
    write_json(a.output/'config.json',metadata,immutable=True)
    write_json(a.output/'origin_order.json',origins,immutable=True)
    optimizer_template=trainer.optimizer.init(r.params)

    def evaluate(params,update,baseline=None):
        reports=[];values=[]
        for i in range(0,len(validation_origins),2):
            v,d,_,_=trainer.batch(params,jax.random.PRNGKey(0),validation_origins[i:i+2],gradients=False)
            reports.append(d);values.append(v)
        result=dict(update=update,loss=float(np.mean(values)),
            pressure_hpa=np.asarray(r.backbone.model.data_coords.vertical.centers).tolist(),
            lead_hours=[6,12],
            physical_rmse={n:np.sqrt(np.mean([d['physical_mse'][n] for d in reports],axis=0)).tolist()
                           for n in reports[0]['physical_mse']})
        if not np.isfinite(result['loss']) or any(not np.isfinite(x).all() for x in result['physical_rmse'].values()):
            raise FloatingPointError('Nonfinite fixed validation')
        if baseline is None:
            result.update(selection_score=1.,worst_headline_ratio=1.,eligible=True)
        else:
            score=headline_rmse_ratio(result,baseline)
            pressure=np.asarray(r.backbone.model.data_coords.vertical.centers)
            ratios=[]
            for name,level in HEADLINES:
                index=int(np.flatnonzero(pressure==level)[0])
                ratios.extend(np.asarray(result['physical_rmse'][name])[:,index]/
                              np.maximum(np.asarray(baseline['physical_rmse'][name])[:,index],1e-12))
            worst=float(max(ratios))
            result.update(selection_score=score,worst_headline_ratio=worst,eligible=worst<=1.05)
        return result

    if a.resume:
        with (a.output/'last.pkl').open('rb') as f:state=pickle.load(f)
        validate(state,identity,r.params,optimizer_template,a.updates)
        materialize(a.output,state)
        log(resumed_update=state['update'])
    else:
        samples=[]
        for i in range(0,len(calibration_origins),2):
            _,_,_,_=trainer.batch(r.params,jax.random.PRNGKey(22),calibration_origins[i:i+2],gradients=False,capture=True)
            samples.append(np.asarray(trainer.group_values(trainer.captured['prediction'],trainer.captured['target'])))
            log(stage='calibration',batches=i//2+1)
        calibration=calibrate(samples,grouped.priors)
        write_json(a.output/'calibration.json',dict(calibration,origins=calibration_origins,split='train'),immutable=True)
        baseline=evaluate(r.params,0)
        controller=WindowBalancer(grouped.names,bc,excluded=grouped.excluded) if a.mode=='adaptive' else None
        state=dict(schema='adaptive_resume_v1',identity=identity,update=0,params=r.params,
            optimizer=optimizer_template,rng=jax.random.PRNGKey(22),
            controller=None if controller is None else controller.state_dict(),
            calibration=calibration,baseline=baseline,metrics=[],validations=[baseline],
            best=dict(update=0,selection_score=1.,params=jax.device_get(r.params)))
        validate(state,identity,r.params,optimizer_template,a.updates)
        commit(a.output,state)
        log(stage='initial_validation',selection_score=1.,fixed_loss=baseline['loss'])
    base=np.asarray(state['calibration']['coefficients'],np.float32)
    end=a.updates if a.stop_after is None else a.stop_after
    if end<state['update']:raise ValueError('Requested stop precedes restored checkpoint')
    for update in range(state['update'],end):
        if time.monotonic()-started>=a.max_seconds:
            commit(a.output,state);log(continuation_required=True,update=state['update']);return 75
        tick=time.monotonic()
        batch=[origins[(update*2+i)%len(origins)] for i in range(2)]
        controller=WindowBalancer.from_state(state['controller']) if state['controller'] is not None else None
        weights=controller.weights.copy() if controller is not None else np.ones(len(base))
        coefficients=np.ones(len(base),np.float32) if a.mode=='fixed' else base*weights
        due=controller is not None and controller.needs_probe()
        try:
            value,detail,rng,gradient=trainer.training_batch(state['params'],state['rng'],batch,coefficients,
                probe_coefficients=base if due else None)
            params,optimizer,_=trainer.apply_update(state['params'],state['optimizer'],gradient)
            finite_tree((params,optimizer),'optimizer output')
            if controller is not None:controller.observe(detail['groups']*base,detail['probe_norms'])
        except Exception as error:
            commit(a.output,state)
            write_json(a.output/'failure.json',dict(attempted_update=update+1,origins=batch,
                last_committed_update=state['update'],error_type=type(error).__name__,message=str(error)))
            raise
        row=dict(update=update+1,origins=batch,training_loss=value,fixed_loss=detail['fixed_loss'],
            calibrated_fixed_loss=float(np.dot(detail['groups'],base)),gradient_norm=tree_norm(gradient),
            groups=detail['groups'].tolist(),weights_used=weights.tolist(),
            weights_next=weights.tolist() if controller is None else controller.weights.tolist(),
            probe_norms=None if detail['probe_norms'] is None else detail['probe_norms'].tolist(),
            seconds=time.monotonic()-tick)
        state.update(update=update+1,params=params,optimizer=optimizer,rng=rng,
            controller=None if controller is None else controller.state_dict())
        state['metrics'].append(row)
        log(update=update+1,training_loss=value,fixed_loss=row['fixed_loss'],probe=due,seconds=row['seconds'])
        if (update+1)%a.validate_every==0 or update+1==a.updates:
            report=evaluate(params,update+1,state['baseline'])
            state['validations'].append(report)
            if report['eligible'] and report['selection_score']<state['best']['selection_score']:
                state['best']=dict(update=update+1,selection_score=report['selection_score'],params=jax.device_get(params))
            log(validation_update=update+1,selection_score=report['selection_score'],
                worst_headline_ratio=report['worst_headline_ratio'],best_update=state['best']['update'])
        if (update+1)%20==0 or update+1==end or (update+1)%a.validate_every==0:
            validate(state,identity,r.params,optimizer_template,a.updates)
            commit(a.output,state)
    if before!=backbone_parameter_digest(r.backbone.model):raise AssertionError('Backbone parameters changed')
    if state['update']==a.updates:
        write_json(a.output/'DONE.json',dict(identity=identity,updates=state['update'],mode=a.mode,
            best_update=state['best']['update'],best_score=state['best']['selection_score'],
            backbone_frozen=True,seconds=time.monotonic()-started,
            slurm_job_id=os.environ['SLURM_JOB_ID']),immutable=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
