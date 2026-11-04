#!/usr/bin/env python3
"""Exercise the production trainer in separate full/partial/resume processes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import jax
import numpy as np
from residualmamba_ngcm.config import Config
from residualmamba_ngcm.model import make_model,read_model_config
from residualmamba_ngcm.train import run
from residualmamba_ngcm.artifacts import load_checkpoint,write_json,source_identity,versions
from restart_probe import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('full','partial','resume','check'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--gc-checkpoint',required=True)
    a=p.parse_args();cfg=Config();identity='production_trainer_gpu_restart_smoke/'+source_identity()
    if a.mode=='check':
        full=load_checkpoint(a.root/'full/step_00003.pkl',identity)
        resumed=load_checkpoint(a.root/'split/step_00003.pkl',identity)
        result={name:compare(full[name],resumed[name]) for name in ('params','memory','opt_state','rng','cursor')}
        if not all(v['bitwise_equal'] for v in result.values()):
            write_json(a.root/'comparison.json',{'status':'different','comparisons':result},immutable=True)
            raise AssertionError('Production trainer differs after a fresh-process restart')
        write_json(a.root/'comparison.json',{'status':'passed','comparisons':result,
            'source_identity':source_identity(),'libraries':versions(),
            'scope':'production train.run; uninterrupted 3 vs 2+1 in separate Python processes',
            'device':jax.devices()[0].device_kind},immutable=True)
        print('Production trainer fresh-process resume is bitwise identical',flush=True)
        return
    if jax.devices()[0].platform!='gpu':raise RuntimeError('GPU allocation required')
    latitude=np.arcsin(np.polynomial.legendre.leggauss(64)[0]);longitude=np.arange(128)*2*np.pi/128
    base=read_model_config(a.gc_checkpoint,'e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29')
    model=make_model(base,cfg,latitude,longitude)
    x=np.random.default_rng(22).normal(0,.1,(128,64,796)).astype(np.float32)
    batch={'features':np.broadcast_to(x,(24,*x.shape)),
           'target':np.full((24,7,29,128,64),.1,np.float32)}
    segments=(np.arange(96),np.arange(96,192))
    output=a.root/('full' if a.mode=='full' else 'split')
    resume=output/'step_00002.pkl' if a.mode=='resume' else None
    run(model,cfg,segments,(),lambda _:batch,latitude,output,identity,
        steps=2 if a.mode=='partial' else 3,resume=resume)

if __name__=='__main__':main()
