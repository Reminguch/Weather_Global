#!/usr/bin/env python3
"""Separate serialization, repeated execution, host input and compilation drift."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from residualmamba_ngcm.config import Config
from residualmamba_ngcm.model import make_model,read_model_config
from residualmamba_ngcm.train import chunk_loss
from residualmamba_ngcm.artifacts import load_checkpoint,save_checkpoint,write_json,source_identity
from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer


def compare(left,right):
    a,ta=jax.tree_util.tree_flatten_with_path(left)
    b,tb=jax.tree_util.tree_flatten_with_path(right)
    if ta!=tb:raise ValueError('Tree structure changed')
    rows=[]
    for (path,x),(_,y) in zip(a,b):
        x,y=np.asarray(x),np.asarray(y)
        if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Array shape/dtype changed')
        delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
        maximum=float(delta.max(initial=0))
        if not np.isfinite(maximum):raise FloatingPointError('Nonfinite difference')
        if x.tobytes(order='C')!=y.tobytes(order='C'):
            rows.append({'path':jax.tree_util.keystr(path),'max_abs':maximum,
                'relative_l2':float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(x.astype(np.float64).ravel()),1e-30)),
                'reference_max_abs':float(np.abs(x).max(initial=0))})
    rows.sort(key=lambda r:r['max_abs'],reverse=True)
    return {'bitwise_equal':not rows,'max_abs':rows[0]['max_abs'] if rows else 0.,'changed_leaves':len(rows),'worst_leaves':rows[:5]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--gc-checkpoint',required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--compare-process',type=Path)
    a=p.parse_args();root=a.output_root;root.mkdir(parents=True,exist_ok=True)
    cfg=Config();lat=np.arcsin(np.polynomial.legendre.leggauss(64)[0]);lon=np.arange(128)*2*np.pi/128
    base=read_model_config(a.gc_checkpoint,'e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29')
    model=make_model(base,cfg,lat,lon)
    optimizer,_=build_optimizer(cfg.v24())
    checkpoint=load_checkpoint(a.checkpoint)
    host={'params':checkpoint['params'],'memory':checkpoint['memory'],'opt_state':checkpoint['opt_state']}
    device=jax.device_put(host)
    key=jax.device_put(checkpoint['rng'])
    x=np.random.default_rng(22).normal(0,.1,(128,64,796)).astype(np.float32)
    batch={'features':jnp.broadcast_to(jnp.asarray(x),(24,*x.shape)),
           'target':jnp.full((24,7,29,128,64),.1,dtype=jnp.float32)}
    def step(state,batch,key):
        (loss,h),g=jax.value_and_grad(lambda params:chunk_loss(model.apply,params,state['memory'],batch,key,lat),has_aux=True)(state['params'])
        updates,s=optimizer.update(g,state['opt_state'],state['params'])
        return {'params':optax.apply_updates(state['params'],updates),'memory':h,'opt_state':s,
                'loss':loss,'grad_norm':optax.global_norm(g)}
    update=jax.jit(step)
    cache_sizes=[]
    def execute(state):
        start=time.monotonic();out=jax.device_get(update(state,batch,key))
        cache_sizes.append(update._cache_size())
        return out,time.monotonic()-start
    print('first execution',flush=True)
    first,t1=execute(device)
    print('repeat same device inputs',flush=True)
    repeat,t2=execute(device)
    print('same values from host inputs',flush=True)
    resumed,t3=execute(host)
    print('serialize/load/place state, then execute',flush=True)
    path=root/'roundtrip.pkl'
    save_checkpoint(path,{**host,'step':2,'identity':'restart_probe','rng':np.asarray(key),'cursor':checkpoint['cursor']})
    loaded=load_checkpoint(path,'restart_probe')
    loaded={k:loaded[k] for k in host}
    after_load,t4=execute(jax.device_put(loaded))
    report={'source_identity':source_identity(),'device':jax.devices()[0].device_kind,
        'flags':os.environ.get('XLA_FLAGS',''),'precision':str(jax.config.jax_default_matmul_precision),
        'serialization':compare(host,loaded),'initial_device_values':compare(host,jax.device_get(device)),
        'repeat_same_executable':compare(first,repeat),'host_inputs':compare(first,resumed),
        'serialized_then_placed':compare(first,after_load),'seconds':[t1,t2,t3,t4]}
    report['jit_cache_sizes_before_recompile']=list(cache_sizes)
    for group in ('params','memory','opt_state','loss','grad_norm'):
        report[group]={'repeat':compare(first[group],repeat[group]),'resume':compare(first[group],resumed[group])}
    print('clear executable cache and recompile',flush=True)
    update.clear_cache()
    recompiled,t5=execute(device)
    report['recompile']=compare(first,recompiled);report['seconds'].append(t5)
    save_checkpoint(root/'reference.pkl',{'result':first,'step':3,'identity':'restart_probe'})
    if a.compare_process:
        other=load_checkpoint(a.compare_process,'restart_probe')
        report['fresh_process']=compare(other['result'],first)
    write_json(root/'result.json',report,immutable=True)
    print({k:v for k,v in report.items() if k in ('seconds','flags','serialization','repeat_same_executable','host_inputs','serialized_then_placed','recompile','fresh_process')},flush=True)

if __name__=='__main__':main()
