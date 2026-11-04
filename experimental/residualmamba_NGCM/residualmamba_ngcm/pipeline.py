"""Stage provenance and production prerequisites, separate from scientific outcomes."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .artifacts import (read_json,write_json,sha256,digest,source_identity,versions,save_arrays,load_arrays)
from .config import Config
from .ngcm import load
from .timeline import split_segments
from .features import static_fields,layout_identity
from .model import read_model_config


def preflight(args,config):
    from .runtime import require_production_policy
    require_production_policy()
    bb,store=load(args.checkpoint,args.checkpoint_sha256,args.store)
    read_model_config(args.gc_checkpoint,args.gc_sha256)
    segments={split:split_segments(store.times,split) for split in ('train','val')}
    if [len(segments[s][0]) for s in ('train','val')]!=[106,15]:
        raise ValueError('Expected exactly 106 train and 15 validation segments')
    cache=Path(args.cache).resolve()
    if not str(cache).startswith('/scratch/gpfs/DABANIN/'):
        raise ValueError('Production cache must be on DABANIN scratch, as specified')
    import shutil
    cache.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(cache).free<110*1024**3:
        raise ValueError('Need at least 110 GiB free for the baseline cache')
    probe=cache/'.write_probe'
    probe.write_bytes(b'');probe.unlink()
    # Filesystem free space is not an account quota. Require an explicit recorded audit.
    quota=read_json(args.quota_audit)
    if quota.get('cache_path')!=str(cache) or quota.get('available_bytes',0)<110*1024**3:
        raise ValueError('Quota audit must identify this cache path and at least 110 GiB available')
    grid=bb.model.data_coords.horizontal
    root=Path(args.run_root)
    save_arrays(root/'geometry.npz',{'latitude':np.asarray(grid.latitudes),'longitude':np.asarray(grid.longitudes),
        'static':static_fields(bb.model)}, {'checkpoint_sha256':bb.checkpoint_sha256})
    record={'stage':'preflight','config':asdict(config),'source_identity':source_identity(),
        'libraries':versions(),'store':str(Path(args.store).resolve()),'store_identity':store.identity,
        'checkpoint':str(Path(args.checkpoint).resolve()),'checkpoint_sha256':args.checkpoint_sha256,
        'gc_checkpoint':str(Path(args.gc_checkpoint).resolve()),'gc_sha256':args.gc_sha256,
        'cache':str(cache),'quota_audit_sha256':sha256(args.quota_audit),
        'feature_layout_hash':layout_identity(config.arm),'geometry_sha256':sha256(root/'geometry.npz'),
        'backbone':bb.manifest(),'segments':{s:{'count':len(v[0]),'omitted_anchors':v[1]} for s,v in segments.items()}}
    write_json(root/'preflight.json',record,immutable=True)
    return record


def context(root,config):
    from .runtime import require_production_policy
    require_production_policy()
    root=Path(root)
    manifest=read_json(root/'preflight.json')
    # JSON round-trip turns tuples into lists; digest canonically compares values.
    if digest(manifest['config'])!=digest(asdict(config)) or manifest['source_identity']!=source_identity():
        raise ValueError('Protocol or source changed; create a new run root')
    if manifest['libraries']!=versions():raise ValueError('Library versions changed')
    from src.models.neuralgcm_residual.data import PreparedStore,require_complete_experiment_data
    store=PreparedStore(manifest['store'])
    require_complete_experiment_data(store)
    if store.identity!=manifest['store_identity']:raise ValueError('Prepared store identity changed')
    geometry,_=load_arrays(root/'geometry.npz')
    if sha256(root/'geometry.npz')!=manifest['geometry_sha256']:raise ValueError('Geometry changed')
    return manifest,store,geometry


def training_identity(root,config):
    root=Path(root)
    manifest,store,geometry=context(root,config)
    stats,sm=load_arrays(root/'stats.npz')
    if sm['store_identity']!=store.identity or sm['preflight_hash']!=sha256(root/'preflight.json'):
        raise ValueError('Statistics provenance mismatch')
    for t in sm['train_times']+sm['extra_times']+[t for pair in sm['diff_pairs'] for t in pair]:
        if not 2015<=int(t[:4])<=2021:raise ValueError('Statistics include non-training frames')
    if any(np.datetime64(b)-np.datetime64(a)!=np.timedelta64(6,'h') for a,b in sm['diff_pairs']):
        raise ValueError('Statistics differences are not adjacent 6h differences')
    cache=Path(manifest['cache'])
    ready=read_json(cache/'READY.json')
    contract=read_json(cache/'contract.json')
    expected_cache={'store_identity':store.identity,'checkpoint_sha256':manifest['checkpoint_sha256'],
                    'source_identity':source_identity(),'libraries':versions()}
    if ready['contract_hash']!=sha256(cache/'contract.json') or any(contract[k]!=v for k,v in expected_cache.items()):
        raise ValueError('Cache contract mismatch')
    verification=read_json(root/'verify-cache.json')
    if verification['cache_ready_hash']!=sha256(cache/'READY.json') or not verification['finite']:
        raise ValueError('Missing/mismatched cache verification')
    census=read_json(root/'census.json')
    upstream={'stats':sha256(root/'stats.npz'),'cache':sha256(cache/'READY.json')}
    if census['upstream']!=upstream:raise ValueError('Census must use this cache and statistics')
    if census['decision_required']:
        decision=read_json(root/'census-decision.json')
        if decision.get('census_sha256')!=sha256(root/'census.json') or decision.get('action')!='retain_v1_scales' or not decision.get('rationale','').strip():
            raise ValueError('Record whether to retain the v1 scales before training. Changed floors require a versioned spec and new statistics.')
    tests=read_json(root/'qualification.json')
    if tests.get('source_identity')!=source_identity() or tests.get('status')!='passed':
        raise ValueError('A matching successful unit + real-model qualification is required')
    audit=read_json(root/'baseline-audit/summary.json')
    if audit['upstream']!={'preflight':sha256(root/'preflight.json'),'stats':sha256(root/'stats.npz')} or set(audit['years'])!=set(map(str,range(2015,2024))):
        raise ValueError('The specification requires the native baseline J_k audit for 2015–2023 before training')
    # These are execution/integrity checks. Finite drift and weak skill are warnings.
    return digest({'preflight':sha256(root/'preflight.json'),**upstream,
                   'verification':sha256(root/'verify-cache.json'),'census':sha256(root/'census.json'),
                   'census_decision':sha256(root/'census-decision.json') if census['decision_required'] else None,
                   'qualification':sha256(root/'qualification.json'),
                   'baseline_audit':sha256(root/'baseline-audit/summary.json')}),manifest,store,geometry,stats


def qualify(root,gpu_smoke,native_smoke,resume_smoke):
    """Generate qualification from executed tests, never from a hand-written pass flag."""
    from .runtime import require_production_policy
    policy=require_production_policy()
    import os
    import subprocess
    import sys
    from . import ROOT
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    gpu,native,resume=read_json(gpu_smoke),read_json(native_smoke),read_json(resume_smoke)
    identity=source_identity()
    for name,evidence in [('GPU',gpu),('native',native),('fresh-process resume',resume)]:
        if evidence.get('status')!='passed' or evidence.get('source_identity')!=identity:
            raise ValueError(f'{name} evidence is missing or belongs to different source')
    if not gpu.get('zero_head_bitwise') or not gpu.get('upper_levels_bitwise') or not native.get('finite'):
        raise ValueError('Incomplete real-model smoke evidence')
    for evidence in (gpu,resume):
        if evidence.get('libraries',{}).get('gpu_execution')!=policy:
            raise ValueError('GPU evidence must use the production execution policy')
    groups=('params','memory','opt_state','rng','cursor')
    if not gpu.get('restart_bitwise') or not all(resume.get('comparisons',{}).get(k,{}).get('bitwise_equal') for k in groups):
        raise ValueError('The requested deterministic GPU restart fix has not passed its regression test')
    env=dict(os.environ,JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',PYTEST_ADDOPTS='-p no:cacheprovider')
    log=root/'qualification-pytest.log'
    with log.open('w') as f:
        subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'experimental/residualmamba_NGCM/tests'),
            '--basetemp='+str(root/'pytest-tmp')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    write_json(root/'qualification.json',{'status':'passed','source_identity':identity,
        'gpu_smoke_sha256':sha256(gpu_smoke),'native_smoke_sha256':sha256(native_smoke),
        'resume_smoke_sha256':sha256(resume_smoke),
        'unit_test_log_sha256':sha256(log),'scope':'unit tests and real-model smoke; cache/census checked separately'},immutable=True)
