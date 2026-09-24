#!/usr/bin/env python3
"""Matched K=1/K=2 residual-only training with the five-term NGCM loss.

The physical backbone is frozen and its feedback is stop-gradient. Only the
Mamba recurrence carries cross-step derivatives, as explicitly requested.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke','statistics','train'),required=True)
    parser.add_argument('--resource-plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--statistics',type=Path)
    parser.add_argument('--steps',type=int,choices=(1,2),default=2)
    parser.add_argument('--width',type=int,choices=(128,256),default=128)
    parser.add_argument('--d-inner',type=int,choices=(16,32),default=16)
    parser.add_argument('--updates',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--validate-every',type=int,default=100)
    parser.add_argument('--validation-origins',type=int,default=16)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after',type=int,help='Checkpoint and exit at this absolute update without changing the run identity')
    args = parser.parse_args()
    if min(args.updates,args.validate_every,args.validation_origins) < 1:
        parser.error('Updates, validation interval and validation origins must be positive')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.updates:
        parser.error('--stop-after must lie within the configured update budget')
    from src.models.neuralgcm_residual.paper_execution import configure_paper_environment, execution_metadata
    configure_paper_environment()
    import jax
    import numpy as np
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.data import PreparedStore, eligible_origins, seasonal_order
    from src.models.neuralgcm_residual.config import RunConfig, Architecture, BALANCED_LOSS_NAME
    from src.models.neuralgcm_residual.runtime import build_runtime
    from src.models.neuralgcm_residual.paper_statistics import fit_statistics
    from src.models.neuralgcm_residual.paper_loss import PaperLoss
    from src.models.neuralgcm_residual.trajectory_training import TrajectoryTrainer
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest,tree_comparison
    from src.models.neuralgcm_residual.model import zero_memory
    from src.models.neuralgcm_residual.io import (read_json,write_json,write_pickle,append_jsonl,
                                                 digest,sha256,versions)
    import pickle
    if not all(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Run this worker in a Slurm GPU allocation')
    if args.batch_size < 2:
        raise ValueError('Use at least two independent origins for the batch-bias term')
    resources = read_json(args.resource_plan)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    log = lambda item: print(json.dumps(item,allow_nan=False),flush=True)
    started = time.perf_counter()
    if args.phase=='statistics':
        b = FrozenBackbone.load(resources['checkpoint'],resources['checkpoint_sha256'])
        store = PreparedStore(resources['prepared'])
        stats = fit_statistics(b.model,store,60,progress=log)
        stats['checkpoint_sha256'] = resources['checkpoint_sha256']
        stats['scope'] = 'shared training-only loss statistics for matched K=1/K=2 matrix'
        write_json(args.output/'statistics.json',stats,immutable=True)
        log({'statistics_ready':str(args.output/'statistics.json'),'seconds':time.perf_counter()-started})
        return

    config = RunConfig(architecture=Architecture(width=args.width,d_inner=args.d_inner),
        loss=BALANCED_LOSS_NAME,correction_policy='no_pressure_zero_mean_v2')
    runtime = build_runtime(config,resources,stage='finetune')
    b = runtime.backbone
    before = backbone_parameter_digest(b.model)
    if args.phase=='smoke':
        stats = fit_statistics(b.model,runtime.store,10,progress=log)
        stats['scope'] = 'independent pilot only; never production loss statistics'
        write_json(args.output/'pilot_statistics.json',stats,immutable=True)
    else:
        if args.statistics is None:
            raise ValueError('Production requires the shared new loss-statistics artifact')
        stats = read_json(args.statistics)
        if stats['samples']!=60 or stats['dataset_id']!=runtime.store.identity:
            raise ValueError('Production loss statistics must use all 60 selected training snapshots')
        if stats['checkpoint_sha256']!=resources['checkpoint_sha256']:
            raise ValueError('Native loss statistics were fitted for a different checkpoint')
    # Identical eligible origin pool for both K, including the 12 h target.
    origins = seasonal_order(eligible_origins(runtime.store.times,'train',2,warm_hours=0,daily=False))
    validation = seasonal_order(eligible_origins(runtime.store.times,'val',2,warm_hours=0))[:args.validation_origins]
    params,key = runtime.params,jax.random.PRNGKey(22)
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for p in
                    sorted((ROOT/'src/models/neuralgcm_residual').glob('*.py'))}
    source_hashes[str(Path(__file__).relative_to(ROOT))] = sha256(__file__)

    if args.phase=='smoke':
        checks = {}
        # Zero-head identity on actual corrected feedback, not independent K=1 records.
        inp,forcing = runtime.store.inputs_and_forcing(b.model,origins[0])
        state,baseline = b.encode(inp,forcing),b.encode(inp,forcing)
        h = zero_memory(runtime.memory)
        for lead in (1,2):
            delta,h = runtime.branch.apply(params,h,key,
                runtime.normalization.inputs(runtime.adapter.features(state)),runtime.known(state,forcing))
            np.testing.assert_array_equal(delta,0)
            state = runtime.adapter.apply_increment(b.advance_6h(state,forcing),runtime.normalization.increment(delta))
            baseline = b.advance_6h(baseline,forcing)
            checks[f'zero_head_{lead*6}h'] = tree_comparison(state,baseline,exact=True)
        for k in (1,2):
            loss = PaperLoss(b.model,stats,k)
            trainer = TrajectoryTrainer(runtime,loss)
            p,opt,rng = params,trainer.optimizer.init(params),key
            values = []
            for update in range(3):
                tick = time.perf_counter()
                value,details,rng,g = trainer.batch(p,rng,origins[:args.batch_size])
                p,opt,norm = trainer.apply_update(p,opt,g)
                values.append(value)
                log({'phase':'smoke','K':k,'update':update+1,'loss':value,'gradient_norm':float(norm),
                     'seconds':time.perf_counter()-tick})
            delta = max(float(np.max(np.abs(x-y))) for x,y in zip(
                jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(p),strict=True))
            if delta<=0 or not np.isfinite(values).all():
                raise AssertionError('Smoke failed to update residual parameters')
            checks[f'K{k}'] = dict(losses=values,max_parameter_change=delta,
                                   terms={n:float(v) for n,v in details['terms'].items()})
        assert before==backbone_parameter_digest(b.model),'Frozen backbone parameters changed'
        write_json(args.output/'report.json',dict(passed=True,scope='independent real ERA5 pilot',
            production_statistics=False,checks=checks,backbone_frozen=True,physical_feedback='stop_gradient',
            source_hashes=source_hashes,seconds=time.perf_counter()-started,libraries=versions(),
            slurm_job_id=os.environ.get('SLURM_JOB_ID')))
        log({'smoke_passed':True,'seconds':time.perf_counter()-started})
        return

    loss = PaperLoss(b.model,stats,args.steps)
    trainer = TrajectoryTrainer(runtime,loss)
    metadata = dict(version='ngcm_residual_paper_sg_k1k2_v1',K=args.steps,forecast_hours=6*args.steps,
        architecture=asdict(config.architecture),seed=22,batch_size=args.batch_size,updates=args.updates,
        physical_feedback='stop_gradient',recurrent_gradient='full within K-step episode',
        objective=loss.metadata,statistics_sha256=sha256(args.statistics),
        input_increment_statistics_sha256=sha256(resources['statistics']),
        checkpoint_sha256=resources['checkpoint_sha256'],dataset_id=runtime.store.identity,
        origin_order_sha256=digest(origins),validation_origins=validation,
        optimizer=dict(name='Adam',peak_lr=.002,warmup=2000,b1=.9,b2=.95,eps=1e-6,
                       weight_decay=0.,decay_start=15000,decay_rate=.5,decay_steps=10000),
        trainable='residual Mamba only',correction_policy=config.correction_policy,
        forcing_policy=config.forcing_policy,source_hashes=source_hashes,libraries=versions(),
        execution_environment=execution_metadata())
    identity = digest(metadata)
    write_json(args.output/'config.json',metadata,immutable=True)
    write_json(args.output/'origin_order.json',origins,immutable=True)
    opt,start,best = trainer.optimizer.init(params),0,float('inf')
    if args.resume:
        with (args.output/'last.pkl').open('rb') as f: saved=pickle.load(f)
        if saved['identity']!=identity: raise ValueError('Exact-resume identity mismatch')
        params,opt,key,start,best = (saved[n] for n in ('params','optimizer','rng','update','best'))
        # Discard only uncommitted metric tails, preserving them for audit.
        for name in ('metrics.jsonl','validation.jsonl'):
            path=args.output/name
            if path.exists():
                lines=path.read_text().splitlines()
                keep=[line for line in lines if json.loads(line)['update']<=start]
                if len(keep)!=len(lines):
                    (args.output/(name+'.uncommitted')).write_text('\n'.join(lines[len(keep):])+'\n')
                    path.write_text('\n'.join(keep)+'\n')
    def save(update):
        write_pickle(args.output/'last.pkl',dict(identity=identity,params=jax.device_get(params),
            optimizer=jax.device_get(opt),rng=jax.device_get(key),update=update,best=best))
    if not args.resume:
        save(0)
    def validate(p):
        scores,counts,reports=[],[],[]
        for i in range(0,len(validation),args.batch_size):
            batch=validation[i:i+args.batch_size]
            val,detail,_,_=trainer.batch(p,jax.random.PRNGKey(0),batch,gradients=False)
            scores.append(val);counts.append(len(batch));reports.append(detail)
        result=dict(loss=float(np.average(scores,weights=counts)),
                    terms={n:float(np.average([x['terms'][n] for x in reports],weights=counts))
                           for n in reports[0]['terms']},
                    physical_rmse={n:np.sqrt(np.average([x['physical_mse'][n] for x in reports],
                                   axis=0,weights=counts)).tolist() for n in reports[0]['physical_mse']})
        if not np.isfinite(result['loss']):
            raise FloatingPointError('Nonfinite validation loss; no checkpoint selection')
        return result
    baseline_path=args.output/'baseline_validation.json'
    if baseline_path.exists():
        baseline=read_json(baseline_path)
    else:
        baseline=validate(runtime.params)
        baseline.update(origins=validation,pressure_hpa=np.asarray(b.model.data_coords.vertical.centers).tolist(),
                        lead_hours=list(range(6,6*args.steps+1,6)),identity=identity)
        write_json(baseline_path,baseline,immutable=True)
    if baseline['identity']!=identity:raise ValueError('Baseline validation identity mismatch')
    end = args.updates if args.stop_after is None else args.stop_after
    if end < start:
        raise ValueError('--stop-after precedes the restored checkpoint')
    for update in range(start,end):
        tick=time.perf_counter()
        batch=[origins[(update*args.batch_size+i)%len(origins)] for i in range(args.batch_size)]
        value,details,key,g = trainer.batch(params,key,batch)
        params,opt,norm=trainer.apply_update(params,opt,g)
        row=dict(update=update+1,K=args.steps,loss=value,gradient_norm=float(norm),
                 seconds=time.perf_counter()-tick,origins=batch,
                 terms={n:float(v) for n,v in details['terms'].items()},
                 fields={n:float(v) for n,v in details['fields'].items()})
        append_jsonl(args.output/'metrics.jsonl',row)
        log({n:row[n] for n in ('update','K','loss','gradient_norm','seconds')})
        if (update+1)%args.validate_every==0 or update+1==args.updates:
            report=validate(params)
            score=report['loss']
            report.update(update=update+1,K=args.steps,baseline_loss=baseline['loss'],
                          reduction_pct=100*(1-score/baseline['loss']))
            append_jsonl(args.output/'validation.jsonl',report)
            if score<best:
                best=score
                write_pickle(args.output/'best.pkl',dict(identity=identity,params=jax.device_get(params),
                    update=update+1,validation_loss=score))
            log({'validation_update':update+1,'loss':score,'best':best})
        if (update+1)%10==0 or update+1==end:
            save(update+1)
    assert before==backbone_parameter_digest(b.model),'Frozen backbone parameters changed'
    if end < args.updates:
        log({'checkpointed_update':end,'configured_updates':args.updates,'backbone_frozen':True})
        return
    write_json(args.output/'DONE.json',dict(identity=identity,updates=args.updates,best_validation_loss=best,
        backbone_frozen=True,seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
