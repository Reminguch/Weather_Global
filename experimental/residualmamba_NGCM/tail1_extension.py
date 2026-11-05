"""Explicit ar_tail_k=1 extension; the qualified v1 core stays unchanged."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
from residualmamba_ngcm.config import Config
from residualmamba_ngcm.artifacts import (save_arrays, load_arrays, write_json,
    read_json, sha256, source_identity, versions)
from residualmamba_ngcm.timeline import stack_weather, step_times
from residualmamba_ngcm.ngcm import rollout
from residualmamba_ngcm.features import ancillary, assemble
from residualmamba_ngcm.loss import target_norm


@dataclass(frozen=True)
class Tail1Config(Config):
    ar_tail_k: int = 1
    truth_prefix_steps: int = 23

    def __post_init__(self):
        expected = asdict(Config())
        expected.update(ar_tail_k=1, truth_prefix_steps=23)
        if asdict(self) != expected:
            raise ValueError('Tail1 changes only ar_tail_k and its derived truth prefix')

    def v24(self):
        return replace(Config().v24(), ar_tail_k=1)


def origin_index(a0, i):
    if not 0 <= i < 24:
        raise ValueError('Chunk step must be 0..23')
    return a0 + min(i, 22)


def window(prefix, baseline, i):
    # prefix includes truth x[a0-1] ... x[a0+22]. At i=23 only the
    # newer input is predicted, and it is the first forecast from a0+22.
    def frame(j):
        return prefix[j] if j <= 23 else baseline[j-2]
    return frame(i), frame(i+1)


def chunk_baseline(bb, store, a0):
    output = [next(rollout(bb, store, store.times[a0+i], 1))[0] for i in range(22)]
    output.extend(b for b, _ in rollout(bb, store, store.times[a0+22], 2))
    return np.stack(output)


def path_for(root, a0):
    return Path(root) / f'chunk_{a0:05d}.npz'


def validate(baseline, meta, times, a0, contract):
    if any(meta.get(k) != v for k, v in contract.items()):
        raise ValueError('Tail1 cache producer identity mismatch')
    if meta['a0'] != a0 or meta['valid_times'] != [str(t) for t in times[a0+1:a0+25]]:
        raise ValueError('Tail1 valid times mismatch')
    if meta['origin_times'] != [str(times[origin_index(a0, i)]) for i in range(24)]:
        raise ValueError('Tail1 origin policy mismatch')
    if baseline.shape != (24,7,37,128,64) or baseline.dtype != np.float32 or not np.isfinite(baseline).all():
        raise ValueError('Invalid tail1 baseline')


def contract_for(plan, bb, store, plan_hash, device_kind):
    return {'ar_tail_k':1, 'truth_prefix_steps':23, 'store_identity':store.identity,
            'checkpoint_sha256':bb.checkpoint_sha256, 'source_identity':source_identity(),
            'extension_sha256':plan['extension_sha256'], 'plan_sha256':plan_hash,
            'libraries':versions(), 'device_kind':device_kind}


def build_one(bb, store, root, a0, contract):
    path = path_for(root, a0)
    write_json(Path(root)/'contract.json', contract, immutable=True)
    if path.exists() and path.with_suffix('.json').exists():
        arrays, meta = load_arrays(path)
        validate(arrays['baseline'], meta, store.times, a0, contract)
        return
    if path.exists():
        raise ValueError('Uncommitted cache payload requires recovery: '+str(path))
    baseline = chunk_baseline(bb, store, a0)
    meta = {**contract, 'a0':int(a0),
            'valid_times':[str(t) for t in store.times[a0+1:a0+25]],
            'origin_times':[str(store.times[origin_index(a0,i)]) for i in range(24)]}
    validate(baseline, meta, store.times, a0, contract)
    save_arrays(path, {'baseline':baseline}, meta)
    print(f'tail1 cached a0={a0}', flush=True)


def load_chunk(store, root, a0, stats, geometry):
    arrays, meta = load_arrays(path_for(root, a0))
    contract = read_json(Path(root)/'contract.json')
    if contract.get('ar_tail_k') != 1 or contract['store_identity'] != store.identity:
        raise ValueError('Not this store / ar_tail_k=1 cache')
    baseline = arrays['baseline']
    validate(baseline, meta, store.times, a0, contract)
    times = store.times
    prefix = [stack_weather(store.frame(t)) for t in times[a0-1:a0+23]]
    targets = [stack_weather(store.frame(t)) for t in times[a0+1:a0+25]]
    forcing = {o:store.frame(times[o]-np.timedelta64(24,'h'))
               for o in {origin_index(a0,i) for i in range(24)}}
    features, normalized = [], []
    for i in range(24):
        older, newer = window(prefix, baseline, i)
        extra = ancillary(step_times(times,a0,i), forcing[origin_index(a0,i)],
                          geometry['static'], geometry['latitude'], geometry['longitude'])
        features.append(assemble(older,newer,baseline[i],extra,stats,Tail1Config(),6 if i<23 else 12))
        normalized.append(target_norm(targets[i],baseline[i],stats))
    return {'features':np.stack(features), 'target':np.stack(normalized).astype(np.float32)}
