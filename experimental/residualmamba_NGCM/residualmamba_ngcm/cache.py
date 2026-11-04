"""Immutable baseline chunks; checkpoint/data/code/library identities are mandatory."""
from pathlib import Path
import numpy as np
from .artifacts import save_arrays,load_arrays,source_identity,versions,write_json,sha256,digest
from .ngcm import chunk_baseline,rollout
from .timeline import split_segments,origin_index,window,step_times,lead_hours,stack_weather
from .features import ancillary,assemble
from .loss import target_norm


def cache_path(root,a0):
    return Path(root)/f'chunk_{a0:05d}.npz'


def validate_chunk(baseline,meta,times,a0):
    if meta['a0']!=a0 or meta['valid_times']!=[str(t) for t in times[a0+1:a0+25]]:
        raise ValueError('Cache anchor/valid times disagree with the store')
    if meta['origin_times']!=[str(times[origin_index(a0,i)]) for i in range(24)]:
        raise ValueError('Cache origins disagree with the native continuation policy')
    if baseline.shape!=(24,7,37,128,64) or baseline.dtype!=np.float32 or not np.isfinite(baseline).all():
        raise ValueError('Invalid cached baseline shape, precision or values')


def build(bb,store,root,preflight_hash,indices=None,production=True):
    import jax
    root = Path(root).resolve()
    if production and not str(root).startswith('/scratch/gpfs/DABANIN/'):
        raise ValueError('The specification requires the production baseline cache on DABANIN scratch')
    identity = {'store_identity':store.identity,'checkpoint_sha256':bb.checkpoint_sha256,
                'source_identity':source_identity(),'libraries':versions(),'preflight_hash':preflight_hash,
                'device_kind':jax.devices()[0].device_kind}
    write_json(root/'contract.json',identity,immutable=True)
    all_indices = [int(seg[off]) for split in ('train','val') for seg in split_segments(store.times,split)[0]
                   for off in range(0,96,24)]
    selected = all_indices if indices is None else indices
    for a0 in selected:
        if a0 not in all_indices:
            raise ValueError(f'Anchor {a0} is outside the declared complete segments')
        path = cache_path(root,a0)
        if path.exists():
            arrays,meta = load_arrays(path)
            validate_chunk(arrays['baseline'],meta,store.times,a0)
            if any(meta[k]!=v for k,v in identity.items()):
                raise ValueError('Cannot mix baseline cache identities')
            continue
        baseline = chunk_baseline(bb,store,store.times,a0)
        meta = {**identity,'a0':a0,'a0_time':str(store.times[a0]),'valid_times':[str(t) for t in store.times[a0+1:a0+25]],
                'origin_times':[str(store.times[origin_index(a0,i)]) for i in range(24)],
                'shape':list(baseline.shape),'dtype':str(baseline.dtype)}
        save_arrays(path,{'baseline':baseline},meta)
        print(f'cached a0={a0}',flush=True)
    if all(cache_path(root,a).exists() and cache_path(root,a).with_suffix('.json').exists() for a in all_indices):
        # Hash every artifact again; existence alone never makes a cache ready.
        hashes = {}
        for a in all_indices:
            arrays,meta = load_arrays(cache_path(root,a))
            validate_chunk(arrays['baseline'],meta,store.times,a)
            if any(meta[k]!=v for k,v in identity.items()):
                raise ValueError('Mixed cache identity')
            hashes[str(a)] = meta['sha256']
        write_json(root/'READY.json',{'contract_hash':sha256(root/'contract.json'),'chunks':hashes},immutable=True)


def verify(bb,store,root,stats,path):
    import jax
    rng = np.random.default_rng(22)
    train = [int(s[o]) for s in split_segments(store.times,'train')[0] for o in range(0,96,24)]
    val = [int(s[o]) for s in split_segments(store.times,'val')[0] for o in range(0,96,24)]
    selected = list(rng.choice(train,3,replace=False))+list(rng.choice(val,1,replace=False))
    records = []
    for a0 in selected:
        a0 = int(a0)
        cached,meta = load_arrays(cache_path(root,a0))
        validate_chunk(cached['baseline'],meta,store.times,a0)
        if meta['checkpoint_sha256']!=bb.checkpoint_sha256 or meta['store_identity']!=store.identity:
            raise ValueError('Verification must use the same official checkpoint and prepared store')
        live = chunk_baseline(bb,store,store.times,a0)
        tail = reference_native_forecast(bb,store,store.times[a0+3],21)
        scale = stats['std'][None,:,:,None,None]
        delta = float(np.max(np.abs(cached['baseline']-live)/scale))
        tail_delta = float(np.max(np.abs(cached['baseline'][3:]-tail)/scale))
        if not np.isfinite(delta+tail_delta):
            raise FloatingPointError('Nonfinite cache verification')
        records.append({'a0':a0,'max_diff_in_std':delta,'tail_diff_in_std':tail_delta,
                        'producer_device':meta['device_kind'],'verification_device':jax.devices()[0].device_kind,
                        'bitwise':bool(np.array_equal(cached['baseline'],live)),
                        'within_reference_bound':max(delta,tail_delta)<=1e-5})
    # AGENTS.md: finite numerical repeatability differences are warnings, not aborts.
    result = {'records':records,'cache_ready_hash':sha256(Path(root)/'READY.json'),
              'finite':True,'source_identity':source_identity(),
              'warnings':[r for r in records if not r['within_reference_bound']]}
    write_json(path,result,immutable=True)
    return result


def reference_native_forecast(bb,store,origin,steps):
    """Exercise the existing evaluator with params=None; collect its decoded fields.

    The old trainer/loss never participates. This is an independent native
    rollout definition used ONLY to test compatibility with the old baseline.
    """
    from types import SimpleNamespace
    from src.models.neuralgcm_residual.evaluate import forecast_origin
    decoded=[]
    def collect(prediction,truth):
        decoded.append(stack_weather(prediction))
        return 0.0
    runtime=SimpleNamespace(backbone=bb,store=store,memory={},
                            trainer=SimpleNamespace(weather_loss=collect))
    forecast_origin(runtime,None,origin,steps=steps)
    # That evaluator scores corrected (identical to native) and native per lead.
    return np.stack(decoded[1::2])


def load_chunk(store,root,a0,stats,config,static,latitude,longitude):
    arrays,meta = load_arrays(cache_path(root,a0))
    contract=__import__('json').loads((Path(root)/'contract.json').read_text())
    if any(meta[k]!=v for k,v in contract.items()):
        raise ValueError('Chunk producer differs from the cache contract')
    if meta['store_identity']!=store.identity or meta['a0']!=a0:
        raise ValueError('Cache belongs to a different store or anchor')
    baseline = arrays['baseline']
    times = store.times
    validate_chunk(baseline,meta,times,a0)
    prefix = [stack_weather(store.frame(t)) for t in times[a0-1:a0+4]]
    targets = [stack_weather(store.frame(t)) for t in times[a0+1:a0+25]]
    # Forcing arrays come from the exact same lagged frame as NGCM; no model import/trace.
    forcing = {o:store.frame(times[o]-np.timedelta64(24,'h')) for o in {origin_index(a0,i) for i in range(24)}}
    features,normalized = [],[]
    for i in range(24):
        older,newer = window(prefix,baseline,i)
        extra = ancillary(step_times(times,a0,i),forcing[origin_index(a0,i)],static,latitude,longitude)
        features.append(assemble(older,newer,baseline[i],extra,stats,config,lead_hours(i)))
        normalized.append(target_norm(targets[i],baseline[i],stats))
    return {'features':np.stack(features),'target':np.stack(normalized).astype(np.float32)}
