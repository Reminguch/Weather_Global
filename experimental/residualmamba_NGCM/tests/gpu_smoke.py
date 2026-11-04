#!/usr/bin/env python3
"""Actual 8192-node, mesh-4 width-512 model smoke; no production ERA5 claims."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from residualmamba_ngcm.config import Config
from residualmamba_ngcm.model import make_model,read_model_config,zero_memory
from residualmamba_ngcm.train import chunk_loss
from residualmamba_ngcm.loss import correct,weighted_mse
from residualmamba_ngcm.artifacts import sha256,write_json,source_identity,versions,save_checkpoint,load_checkpoint
from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--gc-checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();started=time.monotonic()
    if jax.devices()[0].platform!='gpu':raise RuntimeError('This is a real GPU smoke test')
    cfg=Config();lat=np.arcsin(np.polynomial.legendre.leggauss(64)[0]);lon=np.arange(128)*2*np.pi/128
    base=read_model_config(a.gc_checkpoint,'e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29')
    model=make_model(base,cfg,lat,lon)
    rng=np.random.default_rng(22)
    x=rng.normal(0,.1,(128,64,796)).astype(np.float32)
    print('initializing full GraphCast-Mamba',flush=True)
    params,memory=jax.jit(model.init)(jax.random.PRNGKey(22),x)
    memory=zero_memory(memory)
    out,next_memory=jax.jit(model.apply)(params,memory,jax.random.PRNGKey(22),x)
    assert out.shape==(7,29,128,64)
    np.testing.assert_array_equal(np.asarray(out),0)
    assert any(np.any(np.asarray(v)!=0) for v in jax.tree_util.tree_leaves(next_memory))
    b=rng.normal(size=(7,37,128,64)).astype(np.float32);b[:,:,0,0]=-0.
    stats={'diff_std':np.ones((7,37),np.float32)}
    y=np.asarray(correct(b,out,stats))
    np.testing.assert_array_equal(b.view(np.uint32),y.view(np.uint32))
    batch={'features':jnp.broadcast_to(jnp.asarray(x),(24,*x.shape)),
           'target':jnp.full((24,7,29,128,64),.1,dtype=jnp.float32)}
    optimizer,schedule=build_optimizer(cfg.v24());state=optimizer.init(params)
    def update(p,h,s,batch,key):
        (loss,new_h),grad=jax.value_and_grad(lambda par:chunk_loss(model.apply,par,h,batch,key,lat),has_aux=True)(p)
        updates,s=optimizer.update(grad,s,p)
        return optax.apply_updates(p,updates),new_h,s,loss,optax.global_norm(grad)
    update=jax.jit(update)
    records=[]
    checkpoint_path=Path(a.output).with_suffix('.checkpoint.pkl')
    smoke_identity='smoke/'+source_identity()
    for step in range(3):
        print(f'full 24-step BPTT update {step+1}',flush=True)
        params,memory,state,loss,norm=update(params,memory,state,batch,jax.random.PRNGKey(22+step))
        loss,norm=float(loss),float(norm)
        if not np.isfinite(loss+norm):raise FloatingPointError('Nonfinite model smoke update')
        records.append({'step':step+1,'loss':loss,'gradient_norm':norm,'lr':float(schedule(step))})
        print(records[-1],flush=True)
        if step==1:
            save_checkpoint(checkpoint_path,{'params':params,'memory':memory,'opt_state':state,
                'rng':jax.random.PRNGKey(24),'cursor':{'epoch':0,'segment_index':0,'segment_offset':48},
                'step':2,'identity':smoke_identity})
    restored=load_checkpoint(checkpoint_path,smoke_identity)
    restored_params,restored_memory,restored_opt,restored_key=jax.device_put(
        (restored['params'],restored['memory'],restored['opt_state'],restored['rng']))
    resumed=update(restored_params,restored_memory,restored_opt,batch,restored_key)
    differences=[float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))
        for a,b in zip(jax.tree_util.tree_leaves((params,memory,state)),jax.tree_util.tree_leaves(resumed[:3]))]
    restart_delta=max(differences,default=0.)
    restart_bitwise=all(np.asarray(a).dtype==np.asarray(b).dtype and
        np.asarray(a).shape==np.asarray(b).shape and np.asarray(a).tobytes()==np.asarray(b).tobytes()
        for a,b in zip(jax.tree_util.tree_leaves((params,memory,state)),jax.tree_util.tree_leaves(resumed[:3])))
    if not np.isfinite(restart_delta):raise FloatingPointError('Nonfinite resumed GPU update')
    finite=all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves((params,memory,state)))
    if not finite:raise FloatingPointError('Nonfinite smoke optimizer/parameters/memory')
    write_json(a.output,{'status':'passed','scope':'full architecture, synthetic features, 3 optimizer updates each BPTT24',
        'production_data_validation':False,'device':str(jax.devices()[0]),'source_identity':source_identity(),
        'libraries':versions(),'input_channels':796,'output_channels':203,'width':512,'mesh_size':4,
        'zero_head_bitwise':True,'upper_levels_bitwise':True,'memory_nonzero':True,'updates':records,
        'restart_max_abs_delta':restart_delta,'restart_bitwise':restart_bitwise,
        'seconds':time.monotonic()-started,'parameters':sum(x.size for x in jax.tree_util.tree_leaves(params)),
        'memory_bytes':sum(x.size*x.dtype.itemsize for x in jax.tree_util.tree_leaves(memory))},immutable=True)
    print('GPU smoke passed',flush=True)

if __name__=='__main__':main()
