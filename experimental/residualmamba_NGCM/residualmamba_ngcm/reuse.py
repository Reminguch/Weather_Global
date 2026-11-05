"""Explicit import of qualified v1 inputs into a NEW v2 run, never checkpoints."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .artifacts import read_json, write_json, load_arrays, sha256, digest, source_identity, versions
from .features import layout_identity
from .config import Config

V1_SOURCE = 'ba35cf865c859861b932842b7fb02319ac9d66d3a51b8b85567095a967bb10aa'


def import_inputs(donor, root, config):
    """Reuse the completed 20261105 tail20 diagnostic's authenticated raw inputs.

    The donor is explicitly pinned to the qualified v1 implementation. Other
    producers need a separately reviewed migration. Raw cache arrays stay in
    place, preserving their original receipts and producer source identity.
    """
    donor,root=Path(donor).resolve(),Path(root).resolve()
    if not str(root).startswith('/scratch/') or donor==root:
        raise ValueError('Import requires a NEW scratch run root')
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('v2 import will not overwrite or resume an existing run')
    plan=read_json(donor/'plan.json'); setup=read_json(donor/'setup.json')
    expected=asdict(Config());expected.update(schema_version=1,output_channels=203,cloud_alpha=.05)
    if digest(plan['config'])!=digest(expected) or plan['source_identity']!=V1_SOURCE:
        raise ValueError('Only the pinned tail20 v1 protocol is eligible for this import')
    if setup['plan_sha256']!=sha256(donor/'plan.json') or setup['source_identity']!=V1_SOURCE:
        raise ValueError('Donor setup identity mismatch')
    if plan['libraries']!=versions():
        raise ValueError('Use the qualified donor environment and execution policy')
    stats,sm=load_arrays(donor/'stats.npz'); _,gm=load_arrays(donor/'geometry.npz')
    contract=read_json(donor/'cache/contract.json');ready=read_json(donor/'cache/READY.json')
    verification=read_json(donor/'verify-cache.json')
    if (sm['preflight_hash']!=sha256(donor/'setup.json') or
        sm['store_identity']!=setup['store_identity'] or
        gm['sha256']!=setup['geometry_sha256'] or
        contract['source_identity']!=V1_SOURCE or contract['store_identity']!=setup['store_identity'] or
        contract['checkpoint_sha256']!=setup['checkpoint_sha256'] or
        ready['contract_hash']!=sha256(donor/'cache/contract.json') or
        set(ready['chunks'])!=set(map(str,setup['cache_indices'])) or len(ready['chunks'])!=484 or
        verification['cache_ready_hash']!=sha256(donor/'cache/READY.json') or not verification['finite']):
        raise ValueError('Donor input receipts do not agree')
    for scale in ('std','diff_std'):
        if not np.isfinite(stats[scale]).all() or np.any(stats[scale]<=0):
            raise ValueError('Invalid donor normalization scales')
    from src.models.neuralgcm_residual.data import PreparedStore,require_complete_experiment_data
    store=PreparedStore(Path(plan['download_root'])/'res2p8');require_complete_experiment_data(store)
    if store.identity!=setup['store_identity']:
        raise ValueError('Prepared data changed')
    for key in ('checkpoint','gc_checkpoint'):
        expected_hash=setup['checkpoint_sha256' if key=='checkpoint' else 'gc_sha256']
        if sha256(plan[key])!=expected_hash:raise ValueError('Backbone/config checkpoint changed')
    # Full chunk byte hashes remain checked on load; READY membership and receipt
    # hashes are checked here without copying or recomputing ~100 GB of forecasts.
    for anchor,expected_hash in ready['chunks'].items():
        meta=read_json(donor/'cache'/f'chunk_{int(anchor):05d}.json')
        if meta['sha256']!=expected_hash or any(meta[k]!=v for k,v in contract.items()):
            raise ValueError('Cache chunk receipt mismatch')
    files=('plan.json','setup.json','stats.npz','stats.json','geometry.npz','geometry.json',
           'cache/contract.json','cache/READY.json','verify-cache.json')
    reused={'donor_root':str(donor),'files':{name:sha256(donor/name) for name in files},
            'producer_source_identity':V1_SOURCE,'stats_preflight_hash':sha256(donor/'setup.json')}
    root.mkdir(parents=True,exist_ok=True)
    for name in ('stats.npz','stats.json','geometry.npz','geometry.json','verify-cache.json'):
        (root/name).symlink_to(donor/name)
    manifest={'stage':'reuse-v1','config':asdict(config),'source_identity':source_identity(),
        'libraries':versions(),'store':str(store.root) if hasattr(store,'root') else str(Path(plan['download_root'])/'res2p8'),
        'store_identity':store.identity,'checkpoint':plan['checkpoint'],
        'checkpoint_sha256':setup['checkpoint_sha256'],'gc_checkpoint':plan['gc_checkpoint'],
        'gc_sha256':setup['gc_sha256'],'cache':str(donor/'cache'),
        'feature_layout_hash':layout_identity(config.arm),'geometry_sha256':setup['geometry_sha256'],
        'reused_inputs':reused,'production':False,
        'scope':'Imported inputs only. New v2 census, qualification and baseline audit are required for production.',
        'cache_location_exception':setup['cache_location_exception']}
    write_json(root/'preflight.json',manifest,immutable=True)
    return manifest


def check_receipts(manifest):
    reused=manifest.get('reused_inputs')
    if reused is None:return
    if reused['producer_source_identity']!=V1_SOURCE:
        raise ValueError('Unrecognized reused input producer')
    for name,expected in reused['files'].items():
        if sha256(Path(reused['donor_root'])/name)!=expected:
            raise ValueError('Reused input changed: '+name)
