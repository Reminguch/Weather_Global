#!/usr/bin/env python3
"""Isolate pressure-feedback instability, keeping production weights immutable."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((args.experiment_root/'manifest.json').read_text())
    source = Path(raw['source_root'])
    for path in (source/'third_party/graphcast', source/'third_party/neuralgcm', source):
        sys.path.insert(0, str(path))
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.launcher import load_experiment, resolve_resources
    from src.models.neuralgcm_residual.config import load_config, FIELDS
    from src.models.neuralgcm_residual.runtime import build_runtime, runtime_identities
    from src.models.neuralgcm_residual.cache import CacheReader
    from src.models.neuralgcm_residual.checkpoint import load_checkpoint
    from src.models.neuralgcm_residual.model import zero_memory
    from src.models.neuralgcm_residual.native_state import field_value
    from src.models.neuralgcm_residual.io import sha256, versions
    if any(d.platform != 'gpu' for d in jax.devices()): raise RuntimeError('GPU required')
    started = time.perf_counter()
    manifest = load_experiment(args.experiment_root)
    if versions() != manifest['libraries']: raise ValueError('Different libraries')
    config = load_config(manifest['configs'][args.run_id]['path'])
    resources = resolve_resources(manifest, config.resolution_id)
    rt = build_runtime(config, resources, stage='pretrain')
    b, a, n = rt.backbone, rt.adapter, rt.normalization
    reader = CacheReader(resources['cache_root'], rt.store, 'val')
    ids = runtime_identities(rt, config, manifest['source_id'], reader.manifest['producer_id'])
    checkpoint = args.experiment_root/'runs'/args.run_id/'seed22/pretrain/checkpoint_pass_05.pkl'
    saved = load_checkpoint(checkpoint, stage='pretrain', identities=ids)
    params = saved['params']
    del saved
    report = dict(run_id=args.run_id, checkpoint_sha256=sha256(checkpoint), identities=ids,
                  script_sha256=sha256(__file__), job_id=os.environ.get('SLURM_JOB_ID'),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  devices=[d.device_kind for d in jax.devices()],rollouts=[], pressure_sweeps=[])

    def plain(v):
        if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [plain(x) for x in v]
        if isinstance(v,np.ndarray):return plain(v.tolist())
        if isinstance(v,np.generic):return plain(v.item())
        if isinstance(v,float) and not np.isfinite(v):return str(v)
        return v

    def emit(kind, data):
        data=plain(dict(kind=kind,elapsed=time.perf_counter()-started,**data))
        with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')
        print(json.dumps(data,allow_nan=False),flush=True)

    def save():
        temp=args.output/'report.tmp';temp.write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n')
        temp.replace(args.output/'report.json')

    @jax.jit
    def branch(memory,key,state,forcing):
        return rt.branch.apply(params,memory,key,n.inputs(a.features(state)),rt.known(state,forcing))

    apply=jax.jit(a.apply_increment)

    @jax.jit
    def diagnostics(state, forcing, truth):
        decoded=b.decode(state,forcing)
        fields=rt.trainer.weather_loss.field_scores(decoded,truth)
        logp=a.grid.to_nodal(field_value(state,'log_surface_pressure'))
        ps=b.model.from_nondim_units(jnp.exp(logp),'Pa')
        temperature=b.model.from_nondim_units(a.grid.to_nodal(field_value(state,'temperature_variation')),'K')
        return dict(loss=rt.trainer.weather_loss(decoded,truth),fields=fields,
                    ps_min=jnp.min(ps),ps_max=jnp.max(ps),ps_mean=jnp.mean(ps),
                    temp_variation_min=jnp.min(temperature),temp_variation_max=jnp.max(temperature),
                    all_native_finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(state)])))

    ps_mask=np.zeros(a.output_size,np.float32);ps_mask[-1]=1
    ps_mask=jnp.asarray(ps_mask)
    origins=json.loads(Path(resources['validation_origins']).read_text())['origins'][:4]
    emit('loaded',dict(origins=origins))
    for origin in origins:
        inputs,forcing=rt.store.inputs_and_forcing(b.model,origin)
        initial=b.encode(inputs,forcing)
        truths=[rt.store.frame(np.datetime64(origin,'h')+np.timedelta64(6*i,'h')) for i in range(1,21)]
        variants=[('baseline',jnp.zeros_like(ps_mask)),('full',jnp.ones_like(ps_mask)),
                  ('no_pressure',1-ps_mask),('pressure_only',ps_mask),
                  ('pressure_scale_0p01',1-.99*ps_mask),('pressure_clip_3sigma',jnp.ones_like(ps_mask))]
        for name,mask in variants:
            state,memory,key=initial,zero_memory(rt.memory),jax.random.PRNGKey(22)
            steps=[]
            for lead in range(1,21):
                old=state;state=b.advance_6h(old,forcing)
                if name!='baseline':
                    key,sub=jax.random.split(key)
                    delta,memory=branch(memory,sub,old,forcing)
                    if name=='pressure_clip_3sigma':delta=delta.at[...,-1].set(jnp.clip(delta[...,-1],-3,3))
                    state=apply(state,n.increment(delta)*mask)
                result=plain(jax.device_get(diagnostics(state,forcing,truths[lead-1])))
                steps.append(dict(lead_hours=lead*6,**result))
                emit('lead',dict(origin=origin,mode=name,lead_hours=lead*6,**result))
                if not isinstance(result['loss'],(float,int)) or not result['all_native_finite']:break
            report['rollouts'].append(dict(origin=origin,mode=name,steps=steps));save()
        base6=b.advance_6h(initial,forcing)
        _,sub=jax.random.split(jax.random.PRNGKey(22))
        delta,_=branch(zero_memory(rt.memory),sub,initial,forcing)
        increment=n.increment(delta)*ps_mask
        for scale in (0.,.01,.1,.25,.5,1.):
            state6=apply(base6,increment*scale);state12=b.advance_6h(state6,forcing)
            sample=dict(origin=origin,pressure_scale=scale,
                        at6=plain(jax.device_get(diagnostics(state6,forcing,truths[0]))),
                        at12=plain(jax.device_get(diagnostics(state12,forcing,truths[1]))))
            report['pressure_sweeps'].append(sample);emit('sweep',sample)
        # Derivative along the learned pressure correction, from a baseline state.
        # Negative cloud-loss derivative verifies the pressure move is rewarded
        # by the current objective even if other weather fields deteriorate.
        def field_vector(alpha):
            state=apply(base6,increment*alpha)
            scores=rt.trainer.weather_loss.field_scores(b.decode(state,forcing),truths[0])
            return jnp.stack([scores[k] for k in FIELDS])
        derivative=jax.jit(lambda alpha:jax.jvp(field_vector,(alpha,),(jnp.float32(1),)))
        gradient_points=[]
        for alpha in (0.,1.):
            values,gradient=jax.device_get(derivative(jnp.float32(alpha)))
            gradient_points.append(dict(alpha=alpha,loss_by_field=dict(zip(FIELDS,values)),
                                        derivative_by_field=dict(zip(FIELDS,gradient)),
                                        aggregate_derivative=float(np.mean(gradient))))
        report.setdefault('pressure_objective_derivatives',[]).append(dict(origin=origin,points=gradient_points))
        emit('pressure_derivative',dict(origin=origin,points=gradient_points));save()
    assert sha256(checkpoint)==report['checkpoint_sha256']
    report.update(completed=True,elapsed_seconds=time.perf_counter()-started,
                  production_checkpoint_unchanged=True);save()
    emit('completed',dict(report=str(args.output/'report.json')))


if __name__=='__main__':main()
