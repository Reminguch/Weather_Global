#!/usr/bin/env python3
"""Isolate the injected degree-zero divergence/vorticity modes without retraining."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    raw=json.loads((args.experiment_root/'manifest.json').read_text());source=Path(raw['source_root'])
    for path in (source/'third_party/graphcast',source/'third_party/neuralgcm',source):sys.path.insert(0,str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.launcher import load_experiment,resolve_resources
    from src.models.neuralgcm_residual.config import load_config
    from src.models.neuralgcm_residual.runtime import build_runtime,runtime_identities
    from src.models.neuralgcm_residual.cache import CacheReader
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint
    from src.models.neuralgcm_residual.native_state import physical_state,replace,field_value
    from src.models.neuralgcm_residual.model import zero_memory
    from src.models.neuralgcm_residual.io import sha256,versions
    if any(d.platform!='gpu' for d in jax.devices()):raise RuntimeError('GPU required')
    manifest=load_experiment(args.experiment_root)
    if versions()!=manifest['libraries']:raise ValueError('Library mismatch')
    config=load_config(manifest['configs'][args.run_id]['path'])
    resources=resolve_resources(manifest,config.resolution_id)
    rt=build_runtime(config,resources,stage='pretrain');b,a,n=rt.backbone,rt.adapter,rt.normalization
    reader=CacheReader(resources['cache_root'],rt.store,'val')
    identities=runtime_identities(rt,config,manifest['source_id'],reader.manifest['producer_id'])
    checkpoint=args.experiment_root/'runs'/args.run_id/'seed22/pretrain/checkpoint_pass_05.pkl'
    saved=load_checkpoint(checkpoint,stage='pretrain',identities=identities);params=saved['params'];del saved
    started=time.perf_counter()
    report=dict(run_id=args.run_id,checkpoint_sha256=sha256(checkpoint),identities=identities,
                script_sha256=sha256(__file__),job_id=os.environ.get('SLURM_JOB_ID'),
                devices=[d.device_kind for d in jax.devices()],rollouts=[],first_injections=[])

    def plain(v):
        if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [plain(x) for x in v]
        if isinstance(v,np.ndarray):return plain(v.tolist())
        if isinstance(v,np.generic):return plain(v.item())
        if isinstance(v,float) and not np.isfinite(v):return str(v)
        return v

    def emit(kind,data):
        entry=plain(dict(kind=kind,elapsed=time.perf_counter()-started,**data))
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
        print(json.dumps(entry,allow_nan=False),flush=True)

    def save():
        temp=args.output/'report.tmp';temp.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n');temp.replace(args.output/'report.json')

    @jax.jit
    def branch(state,memory,key,forcing):return rt.branch.apply(params,memory,key,n.inputs(a.features(state)),rt.known(state,forcing))

    @jax.jit
    def apply(state,increment,remove_div,remove_vor):
        new=a.apply_increment(state,increment)
        old_core,core=physical_state(state),physical_state(new)
        # Preserve the solver's pre-correction l=0 coefficients exactly.
        div=core.divergence.at[...,0].set(jnp.where(remove_div,old_core.divergence[...,0],core.divergence[...,0]))
        vor=core.vorticity.at[...,0].set(jnp.where(remove_vor,old_core.vorticity[...,0],core.vorticity[...,0]))
        core=replace(core,divergence=div,vorticity=vor)
        return replace(new,state=core) if hasattr(new,'state') else core

    weights=np.polynomial.legendre.leggauss(len(a.grid.latitudes))[1];weights=jnp.asarray(weights/weights.sum())
    @jax.jit
    def stats(state,forcing,truth):
        decoded=b.decode(state,forcing);fields=rt.trainer.weather_loss.field_scores(decoded,truth)
        ps=b.model.from_nondim_units(jnp.exp(a.grid.to_nodal(field_value(state,'log_surface_pressure'))),'Pa')
        div=b.model.from_nondim_units(a.grid.to_nodal(field_value(state,'divergence')),'1/s')
        vor=b.model.from_nondim_units(a.grid.to_nodal(field_value(state,'vorticity')),'1/s')
        mean=lambda x:jnp.mean(jnp.sum(x*weights,axis=-1),axis=-1)
        return dict(loss=rt.trainer.weather_loss(decoded,truth),fields=fields,
                    ps_area_mean=jnp.mean(mean(ps)),ps_min=jnp.min(ps),ps_max=jnp.max(ps),
                    divergence_area_mean_by_sigma=mean(div),vorticity_area_mean_by_sigma=mean(vor),
                    divergence_l0=field_value(state,'divergence')[...,0],
                    vorticity_l0=field_value(state,'vorticity')[...,0])

    ones=jnp.ones(a.output_size,jnp.float32)
    no_ps=ones.at[-1].set(0)
    div_only=jnp.zeros_like(ones).at[32:64].set(1)
    # Widths above are asserted against the actual pinned 32-level native schema.
    assert a.channels[0][2]==a.channels[1][2]==32 and a.channels[-1][0]=='log_surface_pressure'
    variants=[('baseline',ones*0,False,False),('full',ones,False,False),
              ('remove_div_l0',ones,True,False),('remove_vor_l0',ones,False,True),
              ('remove_both_l0',ones,True,True),('no_pressure_remove_div_l0',no_ps,True,False),
              ('no_pressure_remove_both_l0',no_ps,True,True),
              ('divergence_only',div_only,False,False),('divergence_only_remove_l0',div_only,True,False)]
    all_origins=json.loads(Path(resources['validation_origins']).read_text())['origins']
    for origin in (all_origins[0],all_origins[2]):
        inputs,forcing=rt.store.inputs_and_forcing(b.model,origin);initial=b.encode(inputs,forcing)
        truths=[rt.store.frame(np.datetime64(origin,'h')+np.timedelta64(6*i,'h')) for i in range(1,21)]
        for mode,mask,remove_div,remove_vor in variants:
            state,memory,key=initial,zero_memory(rt.memory),jax.random.PRNGKey(22);steps=[]
            for lead in range(1,21):
                old=state;advanced=b.advance_6h(old,forcing)
                if mode=='baseline':state=advanced
                else:
                    key,sub=jax.random.split(key);delta,memory=branch(old,memory,sub,forcing)
                    state=apply(advanced,n.increment(delta)*mask,remove_div,remove_vor)
                result=plain(jax.device_get(stats(state,forcing,truths[lead-1])))
                steps.append(dict(lead_hours=lead*6,**result))
                emit('lead',dict(origin=origin,mode=mode,lead_hours=lead*6,loss=result['loss'],
                                 ps_area_mean=result['ps_area_mean'],
                                 divergence_area_mean_by_sigma=result['divergence_area_mean_by_sigma']))
                if lead==1:
                    report['first_injections'].append(dict(origin=origin,mode=mode,
                        before=plain(jax.device_get(stats(advanced,forcing,truths[0]))),after=result))
                if not isinstance(result['loss'],(float,int)):break
            report['rollouts'].append(dict(origin=origin,mode=mode,steps=steps));save()
    assert sha256(checkpoint)==report['checkpoint_sha256']
    report.update(completed=True,elapsed_seconds=time.perf_counter()-started,production_checkpoint_unchanged=True);save()
    emit('completed',dict(report=str(args.output/'report.json')))


if __name__=='__main__':main()
