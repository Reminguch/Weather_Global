#!/usr/bin/env python3
"""Add 100-update saves by re-entering the unchanged qualified trainer."""
from pathlib import Path
import argparse
import time
import json
from dense_eval import configure


def train(arm_root,endpoint,deadline):
    base,plan,parent,pp,store,geometry,stats,identity,cfg=configure(arm_root)
    from residualmamba_ngcm.model import make_model,read_model_config
    from residualmamba_ngcm.train import run
    from residualmamba_ngcm.timeline import split_segments
    if 'parent_root' in plan:
        import tail1_2k as t1
        load=t1.make_load(arm_root,store,geometry,stats)
    else:load=base.loader(arm_root,store,geometry)[1]
    model=make_model(read_model_config(pp['gc_checkpoint'],base.GC_SHA),cfg,geometry['latitude'],geometry['longitude'])
    segments=split_segments(store.times,'train')[0]
    while time.monotonic()+180<deadline:
        resume=base.checkpoint(arm_root)
        if resume:
            from residualmamba_ngcm.artifacts import load_checkpoint
            step=load_checkpoint(resume,identity)['step']
            if step>=endpoint:return
        result=run(model,cfg,segments,(),load,geometry['latitude'],arm_root/'checkpoints',identity,
            steps=endpoint,resume=resume,defer_validation=True,
            stop_after_update=lambda row:row['step']%100==0 or time.monotonic()+max(180,2*row['seconds'])>=deadline)
        print(json.dumps({'dense_checkpoint':result}),flush=True)
        if result['step']>=endpoint:return


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm-root',type=Path,required=True);p.add_argument('--endpoint',type=int,required=True)
    a=p.parse_args()
    if a.endpoint not in (1000,2000):raise ValueError('Expected diagnostic endpoint')
    configure(a.arm_root)
    import jax
    if jax.devices()[0].device_kind!='NVIDIA H200':raise RuntimeError('H200 required')
    train(a.arm_root,a.endpoint,time.monotonic()+3000)

if __name__=='__main__':main()
