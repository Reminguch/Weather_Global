#!/usr/bin/env python3
"""Detailed real-model checks and separate-process exact resume, K=1 and K=2."""
from pathlib import Path
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('all','reference','resume'),default='all')
    p.add_argument('--resource-plan',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--d-inner',type=int,default=16)
    p.add_argument('--steps',type=int,choices=(1,2),default=2)
    p.add_argument('--updates',type=int,default=20)
    a=p.parse_args()
    if a.stage=='all':
        a.output.mkdir(parents=True,exist_ok=False)
        reports={}
        for k in (1,2):
            for phase in ('reference','resume'):
                subprocess.run([sys.executable,'-u',__file__,'--stage',phase,
                    '--resource-plan',str(a.resource_plan),'--statistics',str(a.statistics),
                    '--output',str(a.output/f'K{k}'),'--steps',str(k),'--width',str(a.width),
                    '--d-inner',str(a.d_inner),'--updates',str(a.updates)],check=True)
            reports[f'K{k}']=json.loads((a.output/f'K{k}'/'report.json').read_text())
        (a.output/'report.json').write_text(json.dumps(dict(passed=True,runs=reports,
            slurm_job_id=os.environ.get('SLURM_JOB_ID')),indent=2)+'\n')
        print(json.dumps({'detailed_passed':True,'width':a.width,'d_inner':a.d_inner}),flush=True)
        return

    from src.models.neuralgcm_residual.paper_execution import configure_paper_environment, execution_metadata
    configure_paper_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.config import RunConfig,Architecture,BALANCED_LOSS_NAME,FIELDS
    from src.models.neuralgcm_residual.runtime import build_runtime
    from src.models.neuralgcm_residual.paper_loss import PaperLoss,COEFFICIENTS,amplitude
    from src.models.neuralgcm_residual.trajectory_training import TrajectoryTrainer,stack
    from src.models.neuralgcm_residual.data import eligible_origins,seasonal_order
    from src.models.neuralgcm_residual.checks import backbone_parameter_digest,tree_comparison
    from src.models.neuralgcm_residual.model import zero_memory
    from src.models.neuralgcm_residual.native_state import field_value,replace,physical_state
    from src.models.neuralgcm_residual.io import read_json,write_json,write_pickle,sha256,digest
    if not all(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU smoke requires Slurm GPU')
    a.output.mkdir(parents=True,exist_ok=a.stage=='resume')
    resources,statistics=read_json(a.resource_plan),read_json(a.statistics)
    config=RunConfig(architecture=Architecture(width=a.width,d_inner=a.d_inner),
                    loss=BALANCED_LOSS_NAME,correction_policy='no_pressure_zero_mean_v2')
    r=build_runtime(config,resources,stage='finetune')
    loss=PaperLoss(r.backbone.model,statistics,a.steps)
    trainer=TrajectoryTrainer(r,loss)
    before=backbone_parameter_digest(r.backbone.model)
    origins=seasonal_order(eligible_origins(r.store.times,'train',2,warm_hours=0,daily=False))[:2]
    key=jax.random.PRNGKey(22)
    params=r.params
    optimizer=trainer.optimizer.init(params)
    checks={}
    started=time.perf_counter()
    metadata=dict(K=a.steps,width=a.width,d_inner=a.d_inner,updates=a.updates,origins=origins,
        statistics_sha256=sha256(a.statistics),resource_plan_sha256=sha256(a.resource_plan),
        execution_environment=execution_metadata(),
        source={str(x.relative_to(ROOT)):sha256(x) for x in sorted((ROOT/'src/models/neuralgcm_residual').glob('*.py'))})
    identity=digest(metadata)

    if a.stage=='reference':
        print(json.dumps({'stage':'reference','K':a.steps,'check':'zero_head_identity'}),flush=True)
        inp,forcing=r.store.inputs_and_forcing(r.backbone.model,origins[0])
        initial=r.backbone.encode(inp,forcing)
        s,baseline,h=initial,initial,zero_memory(r.memory)
        for lead in range(1,a.steps+1):
            d,h=r.branch.apply(params,h,key,r.normalization.inputs(r.adapter.features(s)),r.known(s,forcing))
            np.testing.assert_array_equal(d,0)
            s=r.adapter.apply_increment(r.backbone.advance_6h(s,forcing),r.normalization.increment(d))
            baseline=r.backbone.advance_6h(baseline,forcing)
            checks[f'zero_identity_{lead*6}h']=tree_comparison(s,baseline,exact=True)
            r.backbone.assert_time(initial,s,lead)
        # Nonzero output head and incoming memory exercise recurrent derivatives.
        probe={name:dict(values) for name,values in params.items()}
        for name in probe:
            if 'native_zero_head' in name:
                probe[name]['b']=jnp.full_like(probe[name]['b'],1e-3)
                probe[name]['w']=jax.random.normal(jax.random.PRNGKey(41),probe[name]['w'].shape)*1e-4
        print(json.dumps({'stage':'reference','K':a.steps,'check':'nonzero_probe_batch_gradient'}),flush=True)
        memory=jax.tree_util.tree_map(lambda x:jnp.full_like(x,1e-3),r.memory)
        value,details,_,gradient=trainer.batch(probe,key,origins,capture=True,initial_memory=memory)
        debug=trainer.captured

        # Independent NumPy evaluation of all five terms, including batch bias.
        terms={k:0. for k in COEFFICIENTS}
        for space,names in loss.fields.items():
            grid=loss.grids[space];area=4*np.pi*grid.radius**2
            for name in names:
                scale=amplitude(name)/np.asarray(statistics['scales'][space][name],np.float64)[None,None,:,None,None]
                pred=np.asarray(debug['prediction'][space][name],np.float64)*scale*grid.mask
                truth=np.asarray(debug['target'][space][name],np.float64)*scale*grid.mask
                tw=np.asarray(loss.time)[None,:,None,None,None]
                e=(pred-truth)*tw*np.asarray(loss.filters[space])
                terms[space]+=np.mean(np.sum(e*e,axis=(-2,-1))/area)
                e=(np.sqrt(np.sum(pred*pred,axis=-2))-np.sqrt(np.sum(truth*truth,axis=-2)))[...,:43]
                e=e*np.asarray(loss.spectral_time)[None,:,None,None]
                terms[space+'_spectrum']+=np.mean(np.sum(e*e,axis=-1)/area)
                if space=='data':
                    e=np.mean((np.abs(pred)-np.abs(truth))*tw,axis=(0,1))
                    terms['bias']+=np.mean(np.sum(e*e,axis=(-2,-1))/area)
        for term,coefficient in COEFFICIENTS.items():
            np.testing.assert_allclose(details['terms'][term],coefficient*terms[term],rtol=3e-5,atol=2e-5)
        checks['numpy_five_term_parity']={k:float(v) for k,v in details['terms'].items()}
        hidden=sum(float(np.sum(np.asarray(v)**2)) for name,values in gradient.items()
                   if 'native_zero_head' not in name for v in values.values())
        if hidden<=0:raise AssertionError('No hidden residual-network gradient with nonzero head')
        checks['nonzero_hidden_gradient_squared_norm']=hidden
        print(json.dumps({'stage':'reference','K':a.steps,'check':'independent_recurrent_gradient',
                          'hidden_gradient_squared_norm':hidden}),flush=True)

        # Direct autodiff of the complete recorded residual recurrence is
        # independent of the custom host-tape reverse implementation.
        def reference(p):
            predictions=[]
            for tape in debug['tapes']:
                h=debug['initial_memory'];sequence=[]
                for _,sub,record in tape:
                    out,h,_=trainer.forward(p,h,sub,*record)
                    sequence.append(out)
                predictions.append(stack(sequence))
            return loss(stack(predictions),debug['target'])
        direct=jax.jit(jax.grad(reference))(probe)
        ga=np.concatenate([np.asarray(x).ravel().astype(np.float64) for x in jax.tree_util.tree_leaves(gradient)])
        gb=np.concatenate([np.asarray(x).ravel().astype(np.float64) for x in jax.tree_util.tree_leaves(direct)])
        relative=float(np.linalg.norm(ga-gb)/max(np.linalg.norm(gb),1e-30))
        cosine=float(np.dot(ga,gb)/(np.linalg.norm(ga)*np.linalg.norm(gb)))
        if relative>2e-5 or cosine<.99999:raise AssertionError((relative,cosine))
        checks['independent_recurrent_gradient']=dict(relative_l2=relative,cosine=cosine)
        h,sub,record=debug['tapes'][0][0]
        def physical_probe(eps):
            features,known,base,forcing=record
            core=physical_state(base)
            core=replace(core,temperature_variation=core.temperature_variation+eps)
            perturbed=replace(base,state=core) if hasattr(base,'state') else core
            out,_,_=trainer.forward(probe,h,sub,features+eps,known,perturbed,forcing)
            return jnp.sum(out['data']['temperature'])
        stopped=float(jax.jit(jax.grad(physical_probe))(jnp.float32(0)))
        if stopped!=0.:raise AssertionError('Physical feedback gradient was not stopped')
        checks['physical_feedback_gradient']=stopped
        print(json.dumps({'stage':'reference','K':a.steps,'gradient_relative_l2':relative,
                          'gradient_cosine':cosine,'physical_feedback_gradient':stopped}),flush=True)
        if a.steps==2:
            _,_,corrected=trainer.forward(probe,h,sub,*record)
            expected=r.normalization.inputs(r.adapter.features(corrected))
            actual=debug['tapes'][0][1][2][0]
            checks['second_step_corrected_features']=tree_comparison(expected,actual,exact=True)
        del direct,gradient,debug,ga,gb
        trainer.captured=None
        jax.clear_caches()
        start,history=0,[]
        write_json(a.output/'checks.json',checks)
    else:
        with (a.output/'midpoint.pkl').open('rb') as f:saved=pickle.load(f)
        if saved['identity']!=identity:raise AssertionError('Resume source/config changed')
        params,optimizer,key,start,history=(saved[k] for k in ('params','optimizer','rng','update','history'))

    for step in range(start,a.updates):
        tick=time.perf_counter()
        value,details,key,g=trainer.batch(params,key,origins)
        params,optimizer,norm=trainer.apply_update(params,optimizer,g)
        row=dict(update=step+1,loss=value,gradient_norm=float(norm),terms={k:float(v) for k,v in details['terms'].items()})
        history.append(row)
        print(json.dumps(dict(stage=a.stage,K=a.steps,**row,seconds=time.perf_counter()-tick)),flush=True)
        if a.stage=='reference' and step+1==a.updates//2:
            write_pickle(a.output/'midpoint.pkl',dict(identity=identity,params=jax.device_get(params),
                optimizer=jax.device_get(optimizer),rng=jax.device_get(key),update=step+1,history=history.copy()))
    final=dict(params=jax.device_get(params),optimizer=jax.device_get(optimizer),rng=jax.device_get(key),history=history)
    if before!=backbone_parameter_digest(r.backbone.model):raise AssertionError('Backbone weights changed')
    if a.stage=='reference':
        write_pickle(a.output/'reference.pkl',final)
        write_json(a.output/'reference_scope.json',dict(identity=identity,seconds=time.perf_counter()-started))
    else:
        with (a.output/'reference.pkl').open('rb') as f:reference=pickle.load(f)
        for name in ('params','optimizer','rng'):
            tree_comparison(final[name],reference[name],exact=True)
        if final['history']!=reference['history']:raise AssertionError('Resumed metrics differ')
        movement=max(float(np.max(np.abs(x-y))) for x,y in zip(jax.tree_util.tree_leaves(final['params']),
            jax.tree_util.tree_leaves(r.params),strict=True))
        if movement<=0:raise AssertionError('No residual parameter update')
        write_json(a.output/'report.json',dict(passed=True,identity=identity,checks=read_json(a.output/'checks.json'),
            K=a.steps,updates=a.updates,resume_at=a.updates//2,exact_separate_process_resume=True,
            frozen_backbone=True,max_parameter_change=movement,final_loss=history[-1]['loss'],
            peak_device_memory=jax.devices()[0].memory_stats(),seconds=time.perf_counter()-started))


if __name__=='__main__':main()
