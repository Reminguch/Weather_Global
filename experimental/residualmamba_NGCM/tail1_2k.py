#!/usr/bin/env python3
"""Independent ar_tail_k=1 diagnostic, reusing verified truth/statistics only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
from dataclasses import asdict
import json
import time
import numpy as np
import diagnostic_2k as base
import tail1_extension as ext
from residualmamba_ngcm.artifacts import (read_json,write_json,sha256,digest,source_identity,
    versions,load_arrays,load_checkpoint)
from residualmamba_ngcm.model import make_model,read_model_config
from residualmamba_ngcm.timeline import split_segments
from residualmamba_ngcm.train import run,validate


def context(root):
    plan=read_json(root/'plan.json')
    if plan['source_identity']!=source_identity() or plan['libraries']!=versions():
        raise ValueError('Core source/environment changed')
    for name,value in plan['code_hashes'].items():
        if sha256(base.ROOT/name)!=value:raise ValueError('Extension code changed: '+name)
    parent=Path(plan['parent_root'])
    for name,value in plan['parent_artifacts'].items():
        if sha256(parent/name)!=value:raise ValueError('Parent artifact changed: '+name)
    pp=read_json(parent/'plan.json')
    setup,store,geometry=base.context(parent,pp)
    stats,meta=load_arrays(parent/'stats.npz')
    if meta['store_identity']!=store.identity or meta['preflight_hash']!=sha256(parent/'setup.json'):
        raise ValueError('Statistics provenance mismatch')
    return plan,pp,setup,store,geometry,stats


def make_load(root,store,geometry,stats,cache_name='cache'):
    return lambda a0:ext.load_chunk(store,root/cache_name,a0,stats,geometry)


def model_for(pp,geometry):
    return make_model(read_model_config(pp['gc_checkpoint'],base.GC_SHA),ext.Tail1Config(),
                      geometry['latitude'],geometry['longitude'])


def training_id(root):
    required=('plan.json','qa/COMPLETE.json','cache/READY.json','cache-verified.json')
    return digest({n:sha256(root/n) for n in required})


def verification(root,pp,store,stats,cache_root,indices):
    from residualmamba_ngcm.cache import reference_native_forecast
    bb=base.backbone(pp);contract=read_json(cache_root/'contract.json');rows=[]
    for a0 in indices:
        a,meta=load_arrays(ext.path_for(cache_root,a0));ext.validate(a['baseline'],meta,store.times,a0,contract)
        tail=reference_native_forecast(bb,store,store.times[a0+22],2)
        scale=stats['std'][None,:,:,None,None]
        td=float(np.max(np.abs(a['baseline'][22:]-tail)/scale))
        # Also exercise independent encode/advance/decode for prefix origins.
        prefix=reference_native_forecast(bb,store,store.times[a0+11],1)
        pd=float(np.max(np.abs(a['baseline'][11:12]-prefix)/scale))
        if not np.isfinite(td+pd):raise FloatingPointError('Nonfinite cache verification')
        rows.append({'a0':a0,'tail_max_diff_in_std':td,'prefix_max_diff_in_std':pd,
                     'warning':max(td,pd)>1e-5})
    return rows


def qa_train(root,pp,store,geometry,stats,mode):
    import jax
    bb=base.backbone(pp);cache=root/'qa/cache'
    contract=ext.contract_for(read_json(root/'plan.json'),bb,store,sha256(root/'plan.json'),jax.devices()[0].device_kind)
    for a0 in (4,28,52):ext.build_one(bb,store,cache,a0,contract)
    if not (root/'qa/native.json').exists():
        write_json(root/'qa/native.json',verification(root,pp,store,stats,cache,[4]),immutable=True)
    del bb
    model=model_for(pp,geometry);segments=split_segments(store.times,'train')[0]
    directory=root/'qa'/('full' if mode=='qa-full' else 'split')
    run(model,ext.Tail1Config(),segments,(),make_load(root,store,geometry,stats,'qa/cache'),
        geometry['latitude'],directory,digest({'plan':sha256(root/'plan.json'),'scope':'qa'}),
        steps=3,resume=directory/'step_00002.pkl' if mode=='qa-resume' else None,
        stop_after_update=(lambda row:row['step']==2) if mode=='qa-split' else None,defer_validation=True)


def qa_compare(root):
    import jax
    a=load_checkpoint(root/'qa/full/step_00003.pkl');b=load_checkpoint(root/'qa/split/step_00003.pkl')
    result={}
    for name in ('params','memory','opt_state','rng','cursor'):
        aa,ta=jax.tree_util.tree_flatten(a[name]);bb,tb=jax.tree_util.tree_flatten(b[name])
        if ta!=tb:raise ValueError('Resume tree mismatch')
        diff=max([float(np.max(np.abs(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)))) for x,y in zip(aa,bb)] or [0])
        if not np.isfinite(diff):raise FloatingPointError('Nonfinite restart comparison')
        result[name]={'max_abs_diff':diff,'bitwise':all(np.array_equal(x,y) for x,y in zip(aa,bb))}
    write_json(root/'qa/COMPLETE.json',{'source_identity':source_identity(),'groups':result,
        'native':read_json(root/'qa/native.json'),'finite':True,
        'restart_warning':any(not r['bitwise'] for r in result.values()),
        'scope':'full architecture, real tail1 data, continuous 3 vs callback stop 2 and fresh-process resume'},immutable=True)


def validation(root,plan,pp,store,geometry,stats,endpoint,deadline):
    import jax
    from residualmamba_ngcm.cache import load_chunk
    from residualmamba_ngcm.config import Config
    saved=load_checkpoint(root/f'checkpoints/step_{endpoint:05d}.pkl',training_id(root))
    params,memory=jax.device_put((saved['params'],saved['memory']));model=model_for(pp,geometry)
    segments=split_segments(store.times,'val')[0];ck=sha256(root/f'checkpoints/step_{endpoint:05d}.pkl')
    for label in ('tail1','common_tail20'):
        directory=root/f'validation_{endpoint:05d}'/label
        if label=='tail1':load=make_load(root,store,geometry,stats)
        else:load=lambda a0:load_chunk(store,Path(plan['parent_root'])/'cache',a0,stats,Config(),geometry['static'],geometry['latitude'],geometry['longitude'])
        for i,seg in enumerate(segments):
            path=directory/f'segment_{i:02d}.json'
            if path.exists():continue
            if time.monotonic()+180>=deadline:return
            value=validate(model,params,memory,[seg],load,geometry['latitude'],ext.Tail1Config())
            # Qualified validation returns correct per-step values. Replace its
            # v1-only presentation of prefix/tail lengths for this extension.
            prefix=22 if label=='tail1' else 3
            value.update(one_step_prefix_mean=float(np.mean(value['per_step'][:prefix])),
                tail_leads_hours=list(range(6,(24-prefix)*6+1,6)),tail=value['per_step'][prefix:])
            write_json(path,{**value,'checkpoint_sha256':ck},immutable=True)
            print(f'validation {endpoint} {label} {i+1}/15',flush=True)
        rows=[read_json(directory/f'segment_{i:02d}.json') for i in range(15)]
        if any(r['checkpoint_sha256']!=ck for r in rows):raise ValueError('Changed validation checkpoint')
        value=np.mean([r['per_step'] for r in rows],axis=0);baseline=np.mean([r['baseline_per_step'] for r in rows],axis=0)
        write_json(directory/'summary.json',{'step':endpoint,'evaluation':label,'per_step':value.tolist(),
            'baseline_per_step':baseline.tolist(),'mean_loss':float(value.mean()),
            'baseline_mean_loss':float(baseline.mean()),'improvement_percent':float(100*(1-value.mean()/baseline.mean())),
            'segments':15,'checkpoint_sha256':ck},immutable=True) if not (directory/'summary.json').exists() else None
    write_json(root/f'validation_{endpoint:05d}/COMPLETE.json',{'step':endpoint,'checkpoint_sha256':ck},immutable=True)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows={int(r['step']):r for line in (root/'checkpoints/training.jsonl').read_text().splitlines() if (r:=json.loads(line))}
    if set(rows)!=set(range(1,2001)):raise ValueError('Incomplete 2k training')
    load_checkpoint(root/'checkpoints/step_02000.pkl',training_id(root))
    losses=np.array([rows[i]['loss'] for i in range(1,2001)]);first={};matched=[]
    for i in range(1,2001):
        row=rows[i];first.setdefault(row['a0'],row['loss'])
        if i>1696:matched.append((first[row['a0']],row['loss']))
    matched=np.asarray(matched);summary={'steps':2000,'ar_tail_k':1,
        'matched_training_reduction_percent':float(100*(1-matched[:,1].mean()/matched[:,0].mean())),
        'validation':[read_json(root/f'validation_{i:05d}/{label}/summary.json') for i in (1000,2000) for label in ('tail1','common_tail20')],
        'interpretation':'Training losses across tail lengths have different difficulty. Compare common_tail20 validation for horizon generalization; this is not a memory ablation.'}
    d=root/'report';d.mkdir(exist_ok=True);write_json(d/'summary.json',summary,immutable=True)
    fig,ax=plt.subplots(figsize=(10,4));ax.plot(np.arange(1,2001),losses,alpha=.3)
    ax.plot(np.arange(100,2001),np.convolve(losses,np.ones(100)/100,'valid'));ax.set(xlabel='Updates',ylabel='Tail1 training loss');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(d/'loss-curves.png',dpi=160);plt.close(fig)
    (d/'REPORT.md').write_text('# ar_tail_k=1 independent 2k diagnostic\n\n'+summary['interpretation']+'\n\n![Loss](loss-curves.png)\n')
    write_json(d/'COMPLETE.json',{'steps':2000,'summary_sha256':sha256(d/'summary.json')},immutable=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('--root',type=Path,required=True)
    p.add_argument('--lane',type=int);p.add_argument('--endpoint',type=int);a=p.parse_args();root=a.root.resolve()
    if not str(root).startswith('/scratch/'):raise ValueError('Scratch required')
    plan,pp,setup,store,geometry,stats=context(root);deadline=time.monotonic()+3000
    if a.stage!='report':
        import jax
        if jax.devices()[0].device_kind!='NVIDIA H200':raise RuntimeError('H200 allocation required')
    if a.stage.startswith('qa-'):
        if a.stage=='qa-compare':qa_compare(root)
        else:qa_train(root,pp,store,geometry,stats,a.stage)
        return
    if a.stage=='cache':
        import jax
        if a.lane not in (0,1):raise ValueError('Two disjoint lanes required')
        bb=base.backbone(pp);contract=ext.contract_for(plan,bb,store,sha256(root/'plan.json'),jax.devices()[0].device_kind)
        for a0 in setup['cache_indices'][a.lane::2]:
            if time.monotonic()+180>=deadline:return
            ext.build_one(bb,store,root/'cache',a0,contract)
        write_json(root/f'cache-lane-{a.lane}.json',{'complete':True},immutable=True);return
    if a.stage=='checks':
        contract=read_json(root/'cache/contract.json');hashes={}
        for a0 in setup['cache_indices']:
            arrays,meta=load_arrays(ext.path_for(root/'cache',a0));ext.validate(arrays['baseline'],meta,store.times,a0,contract);hashes[str(a0)]=meta['sha256']
        if not (root/'cache/READY.json').exists():write_json(root/'cache/READY.json',{'chunks':hashes,'contract_sha256':sha256(root/'cache/contract.json')},immutable=True)
        rng=np.random.default_rng(22);selected=list(map(int,rng.choice(setup['cache_indices'][:424],3,replace=False)))+[setup['cache_indices'][450]]
        write_json(root/'cache-verified.json',{'records':verification(root,pp,store,stats,root/'cache',selected)},immutable=True);return
    if a.stage=='train':
        model=model_for(pp,geometry);segments=split_segments(store.times,'train')[0]
        result=run(model,ext.Tail1Config(),segments,(),make_load(root,store,geometry,stats),geometry['latitude'],
            root/'checkpoints',training_id(root),steps=a.endpoint,resume=base.checkpoint(root),defer_validation=True,
            stop_after_update=lambda row:time.monotonic()+max(180,2*row['seconds'])>=deadline)
        print(json.dumps(result),flush=True);return
    if a.stage=='validate':validation(root,plan,pp,store,geometry,stats,a.endpoint,deadline);return
    if a.stage=='report':report(root);return
    raise ValueError('Unknown stage '+a.stage)

if __name__=='__main__':main()
