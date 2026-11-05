#!/usr/bin/env python3
"""Bounded v2 diagnostic using unchanged qualified core and authenticated v1 inputs."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from residualmamba_ngcm import ROOT
from residualmamba_ngcm.config import Config,SCORED_FIELDS
from residualmamba_ngcm.artifacts import (read_json,write_json,sha256,digest,source_identity,
    load_arrays,load_checkpoint)
from residualmamba_ngcm.pipeline import context as pipeline_context
sys.path.insert(0,str(ROOT/'scripts/training'))
from v2_ngcm_controller import pending_checkpoints

FILES=('experimental/residualmamba_NGCM/v2_diagnostic.py','scripts/training/v2_ngcm_controller.py',
       'scripts/training/ngcm_2k_controller.py','experimental/residualmamba_NGCM/dense_eval.py',
       'experimental/residualmamba_NGCM/plot_checkpoint_losses.py',
       'experimental/residualmamba_NGCM/V2_DIAGNOSTIC_2K.md',
       'experimental/residualmamba_NGCM/tail1_extension.py',
       'experimental/residualmamba_NGCM/v2_tail1_inputs.py')
PREREQUISITES=('preflight.json','stats.npz','verify-cache.json','census.json','census-decision.json',
               'qualification/qualification.json')


def prepare(root,donor,qa,tail1_donor=None):
    from residualmamba_ngcm.reuse import import_inputs
    from residualmamba_ngcm.artifacts import versions
    import subprocess
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit the tested launcher before preparing its immutable snapshot')
    manifest=import_inputs(donor,root,Config())
    for sub in ('controller/jobs','logs','qualification','report','eval'):(root/sub).mkdir(parents=True,exist_ok=True)
    qualification=read_json(qa/'qualification/qualification.json')
    if qualification['source_identity']!=source_identity() or qualification['status']!='passed':
        raise ValueError('v2 GPU qualification does not match the core')
    shutil.copyfile(qa/'qualification/qualification.json',root/'qualification/qualification.json')
    source=ROOT/'experimental/residualmamba_NGCM/validation/20261105_v2/cache-regression.json'
    census=read_json(source)
    if (census['source_identity']!=source_identity() or census['chunks']!=60 or
        census['stats_sha256']!=sha256(root/'stats.npz') or
        census['cache_ready_sha256']!=sha256(Path(manifest['cache'])/'READY.json')):
        raise ValueError('v2 census provenance mismatch')
    shutil.copyfile(source,root/'census.json')
    a=np.asarray(census['loss_by_step_field_level'])
    classes=np.concatenate((a[:3].mean(axis=0,keepdims=True),a[3:]),axis=0)
    share=classes/classes.sum(axis=(1,2),keepdims=True)
    write_json(root/'census-decision.json',{'action':'retain_declared_scales',
        'census_sha256':sha256(root/'census.json'),'maximum_step_class_field_level_share':float(share.max()),
        'rationale':'Use the declared v2 five-field loss and unchanged train-only scales. Record any concentration as a warning, never tune on validation.'},immutable=True)
    if tail1_donor is not None:
        from v2_tail1_inputs import verify_donor
        write_json(root/'tail1-inputs.json',verify_donor(tail1_donor,manifest),immutable=True)
    parent=read_json(donor/'plan.json')
    plan={'root':str(root),'python':parent['python'],'download_root':parent['download_root'],
        'target_steps':2000,'qa_job':16928171,'code_snapshot':str(root/'code'),
        'source_identity':source_identity(),'libraries':versions(),
        'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'orchestration_hashes':{name:sha256(ROOT/name) for name in FILES},
        'prerequisites':{name:sha256(root/name) for name in
            PREREQUISITES+(('tail1-inputs.json',) if tail1_donor is not None else ())},
        'ar_tail_k':1 if tail1_donor is not None else 20,
        'arm':'tail1' if tail1_donor is not None else 'tail20',
        'scope':'Bounded 2000-update v2 diagnostic; production annual baseline audit remains deferred',
        'validation':'all 15 fixed 2022 segments / 60 tail20 chunks; every available saved checkpoint',
        'checkpoint_spacing':100,'fresh_initialization':True}
    plan['identity']=digest({'prerequisites':plan['prerequisites'],'source':source_identity(),
                             'orchestration':plan['orchestration_hashes'],'target_steps':2000,
                             'ar_tail_k':plan['ar_tail_k'],'arm':plan['arm']})
    write_json(root/'plan.json',plan,immutable=True)
    write_json(root/'eval/plan.json',{'arms':{plan['arm']:str(root)},'schema_version':2},immutable=True)
    print(json.dumps({'prepared':str(root),'identity':plan['identity'],'target_steps':2000}),flush=True)


def training_config(plan):
    if plan.get('ar_tail_k',20)==1:
        from tail1_extension import Tail1Config
        return Tail1Config()
    if plan.get('ar_tail_k',20)!=20:raise ValueError('Unknown tail length')
    return Config()


def context(root,evaluation=False):
    plan=read_json(root/'plan.json')
    for name,value in plan['orchestration_hashes'].items():
        if sha256(ROOT/name)!=value:raise ValueError('Worker source changed: '+name)
    if any(sha256(root/name)!=value for name,value in plan['prerequisites'].items()):
        raise ValueError('Training prerequisites changed')
    expected=digest({'prerequisites':plan['prerequisites'],'source':source_identity(),
                     'orchestration':plan['orchestration_hashes'],'target_steps':2000,
                             'ar_tail_k':plan['ar_tail_k'],'arm':plan['arm']})
    if plan['identity']!=expected:raise ValueError('Run identity changed')
    manifest,store,geometry=pipeline_context(root,Config())
    stats,_=load_arrays(root/'stats.npz')
    from residualmamba_ngcm.cache import load_chunk
    load=lambda a:load_chunk(store,manifest['cache'],a,stats,Config(),geometry['static'],geometry['latitude'],geometry['longitude'])
    if plan.get('ar_tail_k',20)==1:
        from v2_tail1_inputs import check_receipts
        receipt=read_json(root/'tail1-inputs.json');check_receipts(receipt)
        if not evaluation:
            from tail1_extension import load_chunk as load_tail1
            load=lambda a:load_tail1(store,Path(receipt['cache']),a,stats,geometry)
    from residualmamba_ngcm.model import make_model,read_model_config
    model=make_model(read_model_config(manifest['gc_checkpoint'],manifest['gc_sha256']),
                     Config(),geometry['latitude'],geometry['longitude'])
    return plan,store,geometry,model,load


def train(root,deadline):
    from residualmamba_ngcm.train import run
    from residualmamba_ngcm.timeline import split_segments
    plan,store,geometry,model,load=context(root)
    segments=split_segments(store.times,'train')[0]
    while time.monotonic()+180<deadline:
        paths=sorted(p for p in (root/'checkpoints').glob('step_*.pkl') if p.with_suffix('.json').exists())
        resume=paths[-1] if paths else None
        if resume and load_checkpoint(resume,plan['identity'])['step']>=2000:return
        result=run(model,training_config(plan),segments,(),load,geometry['latitude'],root/'checkpoints',plan['identity'],
            steps=2000,resume=resume,defer_validation=True,
            stop_after_update=lambda r:r['step']%100==0 or time.monotonic()+max(180,2*r['seconds'])>=deadline)
        if result['step']>=2000:return


def validate(root,deadline):
    import jax
    from residualmamba_ngcm.timeline import split_segments
    from dense_eval import predictor,evaluate_segment
    paths=pending_checkpoints(root)
    if not paths:return
    checkpoint=paths[0];plan,store,geometry,model,load=context(root,evaluation=True)
    saved=load_checkpoint(checkpoint,plan['identity']);params,memory=jax.device_put((saved['params'],saved['memory']))
    step=saved['step'];cksha=sha256(checkpoint);directory=root/'eval'/plan.get('arm','tail20')/checkpoint.stem
    predict=predictor(model,geometry['latitude'],22)
    segments=split_segments(store.times,'val')[0]
    for i,segment in enumerate(segments):
        path=directory/f'segment_{i:02d}.json'
        if path.exists():
            record=read_json(path)
            if record['checkpoint_sha256']!=cksha or record['identity']!=plan['identity']:raise ValueError('Mixed eval inputs')
            continue
        if time.monotonic()+180>=deadline:return
        loss,baseline=evaluate_segment(predict,params,memory,segment,load,geometry['latitude'],22)
        if not np.isfinite(loss).all() or not np.isfinite(baseline).all():raise FloatingPointError('Nonfinite validation')
        write_json(path,{'identity':plan['identity'],'checkpoint_sha256':cksha,
            'loss_by_step_field_level':loss.tolist(),'baseline_by_step_field_level':baseline.tolist()},immutable=True)
        print(f'v2 eval step={step} segment={i+1}/15',flush=True)
    rows=[read_json(directory/f'segment_{i:02d}.json') for i in range(len(segments))]
    loss=np.mean([r['loss_by_step_field_level'] for r in rows],axis=0)
    baseline=np.mean([r['baseline_by_step_field_level'] for r in rows],axis=0)
    fields=loss.mean(axis=0).sum(axis=-1);native=baseline.mean(axis=0).sum(axis=-1)
    value=float(fields.sum());ref=float(native.sum())
    result={'schema_version':2,'identity':plan['identity'],'step':step,'arm':plan.get('arm','tail20'),'segments':15,'chunks':60,
        'checkpoint_sha256':cksha,'mean_loss':value,'baseline_mean_loss':ref,'improvement_percent':100*(1-value/ref),
        'per_step':loss.sum(axis=(1,2)).tolist(),'baseline_per_step':baseline.sum(axis=(1,2)).tolist(),
        'per_field':dict(zip(SCORED_FIELDS,fields.tolist())), 'baseline_per_field':dict(zip(SCORED_FIELDS,native.tolist())),
        'field_share':dict(zip(SCORED_FIELDS,(fields/value if value else np.zeros(5)).tolist())),
        'baseline_field_share':dict(zip(SCORED_FIELDS,(native/ref).tolist())),
        'selection':'monitor only; no validation-based checkpoint selection'}
    write_json(directory/'summary.json',result,immutable=True)
    from plot_checkpoint_losses import draw
    draw(root/'eval',root/'eval/separate_plots')
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','train','validate','report','check'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--donor-root',type=Path)
    parser.add_argument('--qa-root',type=Path);parser.add_argument('--tail1-donor-root',type=Path);parser.add_argument('--sample',action='store_true');parser.add_argument('--endpoint',type=int,default=2000)
    a=parser.parse_args();root=a.root.resolve()
    if not str(root).startswith('/scratch/'):raise ValueError('Use scratch')
    if a.endpoint!=2000:raise ValueError('Bounded run endpoint is 2000')
    if a.stage=='prepare':
        if not a.donor_root or not a.qa_root:parser.error('prepare requires --donor-root and --qa-root')
        prepare(root,a.donor_root.resolve(),a.qa_root.resolve(),a.tail1_donor_root);return
    if a.stage=='check':
        plan,store,geometry,model,load=context(root)
        if a.sample:
            from residualmamba_ngcm.loss import contributions
            batch=load(4)
            if batch['features'].shape!=(24,128,64,796) or batch['target'].shape!=(24,5,29,128,64):
                raise ValueError('Unexpected v2 training batch')
            if not all(np.isfinite(x).all() for x in batch.values()):raise ValueError('Nonfinite input')
            scores=np.mean([np.asarray(contributions(np.zeros_like(t),t,geometry['latitude'])) for t in batch['target']],axis=0)
            write_json(root/'loader-smoke.json',{'ar_tail_k':training_config(plan).ar_tail_k,
                'schema_version':2,'target_shape':list(batch['target'].shape),'finite':True,
                'zero_head_loss_first_train_chunk':float(scores.sum()),
                'per_field':dict(zip(SCORED_FIELDS,scores.sum(axis=-1).tolist())),
                'scope':'actual reused cache and ERA5 features, first full 24-step chunk'},immutable=True)
        print('v2 run context and reused inputs verified');return
    if a.stage in ('train','validate'):
        import jax
        if jax.devices()[0].device_kind!='NVIDIA H200':raise RuntimeError('H200 required')
        (train if a.stage=='train' else validate)(root,time.monotonic()+3000);return
    from plot_checkpoint_losses import draw
    draw(root/'eval',root/'eval/separate_plots')
    from v2_ngcm_controller import controller
    if controller.latest_step(root)!=2000 or pending_checkpoints(root):raise ValueError('Incomplete run')
    write_json(root/'report/COMPLETE.json',{'step':2000,'schema_version':2,
        'scope':'Training diagnostic complete; scientific skill depends on per-field validation'},immutable=True)


if __name__=='__main__':main()
