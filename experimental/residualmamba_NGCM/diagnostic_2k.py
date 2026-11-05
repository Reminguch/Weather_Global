#!/usr/bin/env python3
"""Bounded, explicitly non-production 2k loss diagnostic using the qualified core."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse
from dataclasses import asdict
import json
import shutil
import time
import numpy as np
from residualmamba_ngcm import ROOT
from residualmamba_ngcm.artifacts import (source_identity,versions,sha256,read_json,write_json,
    save_arrays,load_arrays,digest,load_checkpoint)
from residualmamba_ngcm.config import Config

NGCM_SHA='bdec1b4612c7385fc492aa031db252c66fb74b788a7efbb118b8b60b06644d3e'
GC_SHA='e9438d8ad31ca6e1d3397a33b2508f4bbb6ec16aed84629f9a64712bf756bc29'


def checked_plan(root,require_qualification=True):
    from residualmamba_ngcm.runtime import require_production_policy
    require_production_policy()
    plan=read_json(root/'plan.json')
    if plan['source_identity']!=source_identity() or plan['libraries']!=versions():
        raise ValueError('Qualified source or execution environment changed')
    if plan['config']!=json.loads(json.dumps(asdict(Config()))):raise ValueError('Configuration changed')
    for name,expected in plan['orchestration_hashes'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('Orchestration source changed: '+name)
    if require_qualification:
        evidence=read_json(root/'qualification/qualification.json')
        if evidence['status']!='passed' or evidence['source_identity']!=plan['source_identity']:
            raise ValueError('Qualification missing or belongs to another source')
    return plan


def store_for(plan):
    from src.models.neuralgcm_residual.data import PreparedStore,require_complete_experiment_data
    directory=Path(plan['download_root']);complete=read_json(directory/'COMPLETE.json')
    store=PreparedStore(directory/'res2p8');require_complete_experiment_data(store)
    if complete['frames']!=13148 or complete['store_identity']!=store.identity:raise ValueError('Download verification mismatch')
    return store


def backbone(plan):
    from residualmamba_ngcm.ngcm import load
    return load(plan['checkpoint'],NGCM_SHA,Path(plan['download_root'])/'res2p8')[0]


def setup(root,plan):
    from residualmamba_ngcm.features import static_fields
    from residualmamba_ngcm.model import read_model_config
    from residualmamba_ngcm.timeline import split_segments
    if shutil.disk_usage(root).free<200*1024**3:raise RuntimeError('Need 200 GiB scratch headroom for this diagnostic')
    store=store_for(plan);bb=backbone(plan);read_model_config(plan['gc_checkpoint'],GC_SHA)
    grid=bb.model.data_coords.horizontal
    path=root/'geometry.npz'
    if not path.exists():
        save_arrays(path,{'latitude':np.asarray(grid.latitudes),'longitude':np.asarray(grid.longitudes),
                         'static':static_fields(bb.model)}, {'checkpoint_sha256':NGCM_SHA})
    train=split_segments(store.times,'train')[0];val=split_segments(store.times,'val')[0]
    if (len(train),len(val))!=(106,15):raise ValueError('Unexpected complete segments')
    indices=[int(s[o]) for segments in (train,val) for s in segments for o in range(0,96,24)]
    write_json(root/'setup.json',{'scope':plan['scope'],'production':False,
        'source_identity':source_identity(),'store_identity':store.identity,'plan_sha256':sha256(root/'plan.json'),
        'geometry_sha256':sha256(path),'checkpoint_sha256':NGCM_SHA,'gc_sha256':GC_SHA,
        'cache_location_exception':'MENGDIW scratch for the user-requested 2k diagnostic; DABANIN is inaccessible',
        'annual_baseline_audit':'deferred; required for a future full production experiment',
        'cache_indices':indices,'train_segments':106,'val_segments':15},immutable=True)


def context(root,plan):
    record=read_json(root/'setup.json');store=store_for(plan)
    if record['plan_sha256']!=sha256(root/'plan.json') or record['store_identity']!=store.identity:
        raise ValueError('Setup provenance mismatch')
    geometry,_=load_arrays(root/'geometry.npz')
    if sha256(root/'geometry.npz')!=record['geometry_sha256']:raise ValueError('Geometry changed')
    return record,store,geometry


def loader(root,store,geometry):
    from residualmamba_ngcm.cache import load_chunk
    from residualmamba_ngcm.timeline import split_segments
    stats,meta=load_arrays(root/'stats.npz')
    if meta['preflight_hash']!=sha256(root/'setup.json') or meta['store_identity']!=store.identity:
        raise ValueError('Statistics provenance mismatch')
    load=lambda a0:load_chunk(store,root/'cache',a0,stats,Config(),geometry['static'],geometry['latitude'],geometry['longitude'])
    return stats,load,split_segments(store.times,'train')[0],split_segments(store.times,'val')[0]


def training_identity(root,plan):
    record=read_json(root/'diagnostic-training.json')
    expected={name:sha256(root/name) for name in ('setup.json','stats.npz','cache/READY.json','verify-cache.json','census.json','census-decision.json')}
    expected['qualification']=sha256(root/'qualification/qualification.json')
    if record['artifacts']!=expected or record['identity']!=digest(expected):raise ValueError('Training prerequisites changed')
    return record['identity']


def checkpoint(root):
    paths=sorted((root/'checkpoints').glob('step_*.pkl'))
    return paths[-1] if paths else None


def checks(root,plan,store,geometry):
    from residualmamba_ngcm.cache import verify
    from residualmamba_ngcm.train import census
    stats,load,train,val=loader(root,store,geometry)
    if not (root/'verify-cache.json').exists():verify(backbone(plan),store,root/'cache',stats,root/'verify-cache.json')
    upstream={'stats':sha256(root/'stats.npz'),'cache':sha256(root/'cache/READY.json')}
    if not (root/'census.json').exists():census(val,load,geometry['latitude'],root/'census.json',upstream)
    measured=read_json(root/'census.json')
    if measured['upstream']!=upstream:raise ValueError('Census provenance mismatch')
    write_json(root/'census-decision.json',{'census_sha256':sha256(root/'census.json'),
        'action':'retain_v1_scales','rationale':'User requested a bounded 2000-update loss diagnostic. Retain the predeclared scales and cloud floors so validation does not tune the objective. Record concentration as a warning and let finite training reach the requested endpoint.',
        'concentration_warning':measured['decision_required']},immutable=True)
    artifacts={name:sha256(root/name) for name in ('setup.json','stats.npz','cache/READY.json','verify-cache.json','census.json','census-decision.json')}
    artifacts['qualification']=sha256(root/'qualification/qualification.json')
    write_json(root/'diagnostic-training.json',{'identity':digest(artifacts),'artifacts':artifacts,
        'target_steps':2000,'scope':plan['scope'],'production':False},immutable=True)


def train(root,plan,store,geometry,endpoint,deadline):
    from residualmamba_ngcm.model import make_model,read_model_config
    from residualmamba_ngcm.train import run
    identity=training_identity(root,plan)
    stats,load,segments,val=loader(root,store,geometry)
    model=make_model(read_model_config(plan['gc_checkpoint'],GC_SHA),Config(),geometry['latitude'],geometry['longitude'])
    result=run(model,Config(),segments,val,load,geometry['latitude'],root/'checkpoints',identity,
        steps=endpoint,resume=checkpoint(root),defer_validation=True,
        stop_after_update=lambda record:time.monotonic()+max(180,2*record['seconds'])>=deadline)
    print(json.dumps(result),flush=True)


def validation(root,plan,store,geometry,endpoint,deadline):
    import jax
    from residualmamba_ngcm.model import make_model,read_model_config
    from residualmamba_ngcm.train import validate
    identity=training_identity(root,plan)
    saved=load_checkpoint(root/f'checkpoints/step_{endpoint:05d}.pkl',identity)
    params,memory=jax.device_put((saved['params'],saved['memory']))
    stats,load,segments,val=loader(root,store,geometry)
    model=make_model(read_model_config(plan['gc_checkpoint'],GC_SHA),Config(),geometry['latitude'],geometry['longitude'])
    directory=root/f'validation_{endpoint:05d}'
    for i,segment in enumerate(val):
        path=directory/f'segment_{i:02d}.json'
        if path.exists():continue
        if time.monotonic()+180>=deadline:return
        result=validate(model,params,memory,[segment],load,geometry['latitude'],Config())
        write_json(path,{**result,'checkpoint_sha256':sha256(root/f'checkpoints/step_{endpoint:05d}.pkl')},immutable=True)
        print(f'validation step={endpoint} segment={i+1}/15',flush=True)
    rows=[read_json(directory/f'segment_{i:02d}.json') for i in range(15)]
    checkpoint_sha=sha256(root/f'checkpoints/step_{endpoint:05d}.pkl')
    if any(r['checkpoint_sha256']!=checkpoint_sha for r in rows):raise ValueError('Validation checkpoint changed')
    value=np.mean([r['per_step'] for r in rows],axis=0)
    baseline=np.mean([r['baseline_per_step'] for r in rows],axis=0)
    report={'step':endpoint,'segments':15,'per_step':value.tolist(),'baseline_per_step':baseline.tolist(),
            'one_step_prefix_mean':float(value[:3].mean()),'tail_leads_hours':list(range(6,127,6)),
            'tail':value[3:].tolist(),'selection':'monitor_only','checkpoint_sha256':checkpoint_sha,
            'mean_loss':float(value.mean()),'baseline_mean_loss':float(baseline.mean()),
            'improvement_percent':float(100*(1-value.mean()/baseline.mean()))}
    write_json(root/f'checkpoints/validation_{endpoint:05d}.json',report,immutable=True)


def report(root,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(line) for line in (root/'checkpoints/training.jsonl').read_text().splitlines()]
    selected={int(r['step']):r for r in rows if int(r['step'])<=2000}
    if set(selected)!=set(range(1,2001)):raise ValueError('Report requires all 2000 committed updates')
    saved=load_checkpoint(root/'checkpoints/step_02000.pkl',training_identity(root,plan))
    if saved['step']!=2000:raise ValueError('Missing final checkpoint')
    ordered=[selected[i] for i in range(1,2001)];loss=np.array([r['loss'] for r in ordered]);steps=np.arange(1,2001)
    first={};ratios=[];matched=[]
    for r in ordered:
        a=r['a0'];first.setdefault(a,r)
        if r['step']!=first[a]['step'] and first[a]['loss']>0:
            ratios.append((r['step'],r['loss']/first[a]['loss']))
    for r in ordered[1696:]:
        f=first[r['a0']]
        matched.append({'a0':r['a0'],'first_step':f['step'],'last_step':r['step'],
                        'first_loss':f['loss'],'last_loss':r['loss']})
    first_mean=float(np.mean([r['first_loss'] for r in matched]));last_mean=float(np.mean([r['last_loss'] for r in matched]))
    validations=[read_json(root/f'checkpoints/validation_{i:05d}.json') for i in (1000,2000)]
    summary={'steps':2000,'duplicate_log_rows_after_recovery':len(rows)-len(selected),
        'raw_first_250_mean':float(loss[:250].mean()),'raw_last_250_mean':float(loss[-250:].mean()),
        'matched_anchor_count':len(matched),'matched_first_mean':first_mean,'matched_last_mean':last_mean,
        'matched_loss_reduction_percent':100*(1-last_mean/first_mean),
        'validation':validations,'production':False,'scope':plan['scope'],
        'interpretation':'Matched anchors compare the same weather chunks across epochs; first encounters already include training. Training loss and validation objective are not full forecast-skill evaluation.'}
    directory=root/'report';directory.mkdir(exist_ok=True)
    write_json(directory/'summary.json',summary,immutable=True)
    write_json(directory/'matched-anchors.json',matched,immutable=True)
    fig,axes=plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
    axes[0].plot(steps,loss,alpha=.3,lw=.7,label='Training loss')
    axes[0].plot(steps[99:],np.convolve(loss,np.ones(100)/100,mode='valid'),label='100-update mean')
    axes[0].axvline(200,color='grey',ls='--',label='Warmup ends');axes[0].set_ylabel('Loss');axes[0].legend()
    ratio=np.asarray(ratios);axes[1].plot(ratio[:,0],ratio[:,1],alpha=.35,lw=.7)
    axes[1].plot(ratio[99:,0],np.convolve(ratio[:,1],np.ones(100)/100,mode='valid'))
    axes[1].axhline(1,color='grey',ls='--');axes[1].set_ylabel('Loss / first same-anchor loss')
    axes[2].plot([1000,2000],[v['mean_loss'] for v in validations],marker='o',label='Residual validation')
    axes[2].plot([1000,2000],[v['baseline_mean_loss'] for v in validations],ls='--',label='Official NeuralGCM zero head')
    axes[2].set_ylabel('Fixed validation objective');axes[2].legend()
    for axis in axes:axis.set_xlabel('Optimizer updates');axis.grid(alpha=.2)
    fig.savefig(directory/'loss-curves.png',dpi=160);plt.close(fig)
    (directory/'REPORT.md').write_text('# 2,000-update loss diagnostic\n\n'
        f'Completed 2,000 updates. Matched-anchor training loss reduction: {summary["matched_loss_reduction_percent"]:.4f}% '
        f'across {len(matched)} chunks. A positive value means lower loss.\n\n'
        f'Validation relative to the zero head: step 1,000 {validations[0]["improvement_percent"]:.4f}%; '
        f'step 2,000 {validations[1]["improvement_percent"]:.4f}%.\n\n'
        'Raw chronological losses also reflect seasonal and chunk difficulty changes. The matched comparison and the fixed validation set help distinguish those effects from learning. '
        'This bounded diagnostic uses MENGDIW scratch; the full year-by-year native audit and production forecast evaluation remain deferred.\n\n'
        '![Loss curves](loss-curves.png)\n')
    write_json(directory/'COMPLETE.json',{'steps':2000,'summary_sha256':sha256(directory/'summary.json'),
        'checkpoint_sha256':sha256(root/'checkpoints/step_02000.pkl')},immutable=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('identity','qualify','setup','stats','cache','cache-finalize','checks','train','validate','report'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--lane',type=int);p.add_argument('--endpoint',type=int)
    a=p.parse_args();root=a.root.resolve();deadline=time.monotonic()+3000
    if not str(root).startswith('/scratch/'):p.error('Use scratch')
    if a.stage=='identity':
        print(json.dumps({'source_identity':source_identity(),'libraries':versions(),'config':asdict(Config())}));return
    plan=checked_plan(root,require_qualification=a.stage in ('checks','train','validate','report'))
    if a.stage=='qualify':
        from residualmamba_ngcm.pipeline import qualify
        qa=Path(plan['qa_root'])
        qualify(root/'qualification',qa/'gpu-smoke.json',qa/'native/result.json',qa/'trainer/comparison.json');return
    if a.stage=='setup':setup(root,plan);return
    record,store,geometry=context(root,plan)
    if a.stage=='stats':
        from residualmamba_ngcm.stats import compute
        compute(backbone(plan),store,Config(),root/'stats.npz',sha256(root/'setup.json'));return
    if a.stage in ('cache','cache-finalize'):
        import jax
        from residualmamba_ngcm.cache import build,cache_path
        if jax.devices()[0].device_kind!='NVIDIA H200':raise RuntimeError('H200 allocation required')
        bb=backbone(plan)
        if a.stage=='cache-finalize':build(bb,store,root/'cache',sha256(root/'setup.json'),[],production=False);return
        if a.lane not in (0,1,2):p.error('--lane must be 0..2')
        indices=record['cache_indices'][a.lane::3]
        for index in indices:
            if time.monotonic()+180>=deadline:return
            build(bb,store,root/'cache',sha256(root/'setup.json'),[index],production=False)
        write_json(root/f'cache-lane-{a.lane}.json',{'indices':indices,'source_identity':source_identity()},immutable=True);return
    if a.stage=='checks':checks(root,plan,store,geometry);return
    if a.stage in ('train','validate'):
        import jax
        if jax.devices()[0].device_kind!='NVIDIA H200':raise RuntimeError('H200 allocation required')
        if a.endpoint not in (1000,2000):p.error('--endpoint must be 1000 or 2000')
        (train if a.stage=='train' else validation)(root,plan,store,geometry,a.endpoint,deadline);return
    report(root,plan)


if __name__=='__main__':main()
