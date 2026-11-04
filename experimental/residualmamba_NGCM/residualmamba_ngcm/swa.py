"""Fixed uniform parameter-only SWA: 13 snapshots, 2000 through 8000 inclusive."""
from pathlib import Path
import jax
import numpy as np
from .artifacts import load_checkpoint,save_checkpoint,sha256


def average(root,config,output):
    paths=[Path(root)/f'step_{step:05d}.pkl' for step in config.swa_steps]
    mean=None
    identity=None
    for n,path in enumerate(paths,1):
        value=load_checkpoint(path,identity)
        identity=value['identity']
        if value['step']!=config.swa_steps[n-1]:
            raise ValueError('SWA checkpoint step mismatch')
        params=value['params']
        if mean is None:
            mean=jax.tree_util.tree_map(lambda x:np.asarray(x,dtype=np.float64),params)
        else:
            mean=jax.tree_util.tree_map(lambda a,b:a+(b-a)/n,mean,params)
    params=jax.tree_util.tree_map(lambda x:x.astype(np.float32),mean)
    save_checkpoint(output,{'params':params,'step':8000,'identity':identity,'kind':'fixed_swa',
        'source_checkpoints':{p.name:sha256(p) for p in paths},'steps':list(config.swa_steps)})
