#!/usr/bin/env python3
"""One real ERA5 cached/live decoded-loss update; never releases production jobs."""
from pathlib import Path
import sys
import argparse
import os
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resolution', type=float, choices=(2.8,1.4), default=2.8)
    args=p.parse_args()
    from src.models.neuralgcm_residual.numerics import configure_environment
    configure_environment()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from src.models.neuralgcm_residual.backbone import FrozenBackbone
    from src.models.neuralgcm_residual.data import PreparedStore
    from src.models.neuralgcm_residual.cache import live_record, valid_k1_origins
    from src.models.neuralgcm_residual.native_state import NativeAdapter
    from src.models.neuralgcm_residual.normalization import fit_statistics
    from src.models.neuralgcm_residual.features import KnownFeatures
    from src.models.neuralgcm_residual.config import RunConfig,Architecture
    from src.models.neuralgcm_residual.model import make_branch,zero_memory
    from src.models.neuralgcm_residual.loss import WeatherLoss
    from src.models.neuralgcm_residual.kernels import DecodedTrainer,make_optimizer
    from src.models.neuralgcm_residual.checks import tree_comparison,backbone_parameter_digest,TOLERANCES
    from src.models.neuralgcm_residual.io import write_json,sha256
    from src.models.neuralgcm_residual.checkpoint import save_checkpoint,load_checkpoint
    source_hashes={p.name:sha256(p) for p in (ROOT/'src/models/neuralgcm_residual').glob('*.py')}
    b=FrozenBackbone.load(args.checkpoint)
    store=PreparedStore(args.prepared)
    origin=valid_k1_origins(store,'train')[0]
    print('real ERA5 origin',origin,'devices',jax.devices(),flush=True)
    record=live_record(b,store,origin)
    adapter=NativeAdapter(b.model,record['origin_state'])
    record['target']=store.frame(record['valid'])
    record['truth']=store.frame(origin)
    inputs,f=store.inputs_and_forcing(b.model,record['valid'])
    record['next_truth_state']=b.encode(inputs,f)
    norm=fit_statistics([record],adapter,b.model.data_coords.horizontal.latitudes,
                        dataset_id=store.identity,output=args.output/'pilot_only_statistics.json')
    known=KnownFeatures(b.model)
    cfg=RunConfig(resolution=args.resolution,architecture=Architecture(width=128 if args.resolution==2.8 else 256,
                                                                    d_inner=16 if args.resolution==2.8 else 32))
    branch=make_branch(cfg.architecture,adapter.grid.latitudes,adapter.grid.longitudes,adapter.output_size)
    x=norm.inputs(adapter.features(record['origin_state']))
    known_input=norm.known(known(record['origin_state'],record['forcing']))
    params,memory=branch.init(jax.random.PRNGKey(22),x,known_input)
    for name in params:
        if 'native_zero_head' in name:
            params[name]=dict(params[name],w=jnp.full_like(params[name]['w'],1e-5),b=jnp.full_like(params[name]['b'],1e-3))
    memory=jax.tree_util.tree_map(lambda x:jnp.full_like(x,1e-3),memory)
    loss=WeatherLoss(b.model.data_coords.horizontal.latitudes,b.model.data_coords.vertical.centers,norm.manifest['loss_scales'])
    trainer=DecodedTrainer(branch,adapter,norm,b,loss,make_optimizer(cfg.pretrain_optimizer))
    frozen=backbone_parameter_digest(b.model)
    cached=trainer.record(record['origin_state'],record['baseline_state'],record['forcing'],record['target'],known_input)
    before=time.perf_counter()
    a=trainer.cached_gradients(params,memory,jax.random.PRNGKey(22),[cached])
    print('cached backward compiled and completed',float(a[0]),flush=True)
    fresh=live_record(b,store,origin)
    live=trainer.record(fresh['origin_state'],fresh['baseline_state'],fresh['forcing'],record['target'],
                        norm.known(known(fresh['origin_state'],fresh['forcing'])))
    c=trainer.cached_gradients(params,memory,jax.random.PRNGKey(22),[live])
    repeated=trainer.cached_gradients(params,memory,jax.random.PRNGKey(22),[cached])
    def diagnose(left, right):
        rows=[]
        for (path,u),v in zip(jax.tree_util.tree_flatten_with_path(left)[0],jax.tree_util.tree_leaves(right),strict=True):
            u,v=np.asarray(u,dtype=np.float64),np.asarray(v,dtype=np.float64)
            rows.append({'path':jax.tree_util.keystr(path),'max_abs':float(np.max(np.abs(u-v),initial=0)),
                         'relative_l2':float(np.linalg.norm(u-v)/max(np.linalg.norm(u),1e-30)),
                         'exact':bool(np.array_equal(u,v))})
        return rows
    diagnostics={
        'origin':diagnose(record['origin_state'],fresh['origin_state']),
        'baseline':diagnose(record['baseline_state'],fresh['baseline_state']),
        'forcing':diagnose(record['forcing'],fresh['forcing']),
        'features':diagnose(cached[:2],live[:2]),
        'memory':diagnose(a[1],c[1]),'gradients':diagnose(a[3],c[3]),
        'repeat_memory':diagnose(a[1],repeated[1]),'repeat_gradients':diagnose(a[3],repeated[3])}
    write_json(args.output/'diagnostics.json',diagnostics,immutable=True)
    print('diagnostics', {k: {'max_abs':max(x['max_abs'] for x in v),'max_relative_l2':max(x['relative_l2'] for x in v)} for k,v in diagnostics.items()},flush=True)
    opt=trainer.optimizer.init(params)
    update_a=trainer.checked_update(params,opt,a[3]);update_b=trainer.checked_update(params,opt,c[3])
    jax.block_until_ready(update_b)
    ids={'backbone':b.checkpoint_sha256,'dataset':store.identity,'config':cfg.identity,'source':source_hashes}
    cursor={'epoch':0,'segment':0,'chunk':1,'update':1}
    checkpoint=args.output/'resume.pkl'
    save_checkpoint(checkpoint,stage='pretrain',identities=ids,params=update_a[0],optimizer=update_a[1],
                    memory=a[1],rng=a[2],cursor=cursor)
    restored=load_checkpoint(checkpoint,stage='pretrain',identities=ids)
    resume_parity=tree_comparison((update_a[0],update_a[1],a[1],a[2]),
                                 tuple(restored[k] for k in ('params','optimizer','memory','rng')),exact=True)
    if restored['cursor']!=cursor: raise AssertionError('Resume cursor differs')
    uninterrupted=trainer.cached_gradients(update_a[0],a[1],a[2],[cached])
    resumed=trainer.cached_gradients(restored['params'],restored['memory'],restored['rng'],[cached])
    continued=tree_comparison(uninterrupted,resumed,exact=True)
    report={'kind':'real_era5_one_update_smoke','not_a_production_gate':True,'origin':origin,
            'checkpoint_sha256':b.checkpoint_sha256,'native_schema':adapter.identity,'dataset_id':store.identity,
            'run_id':cfg.run_id,'tolerances':TOLERANCES,'source_hashes':source_hashes,
            'numerics_environment':{k:os.environ.get(k) for k in ('XLA_FLAGS','JAX_DEFAULT_MATMUL_PRECISION')},
            'loss':float(a[0]),'loss_comparison':tree_comparison(a[0],c[0]),
            'memory_comparison':tree_comparison(a[1],c[1]),'gradient_comparison':tree_comparison(a[3],c[3]),
            'update_comparison':tree_comparison(update_a,update_b),'gradient_norm':float(update_a[2]),
            'resume_state':resume_parity,'resume_continuation':continued,
            'seconds_with_compile':time.perf_counter()-before,
            'frozen_parameters_unchanged':backbone_parameter_digest(b.model)==frozen,
            'devices':[str(d) for d in jax.devices()],'gpu_allocator_stats':[d.memory_stats() for d in jax.devices()],
            'passed':True}
    if not report['frozen_parameters_unchanged']:raise AssertionError('Backbone changed')
    write_json(args.output/'report.json',report,immutable=True)
    print('PASS',report,flush=True)


if __name__=='__main__':main()
