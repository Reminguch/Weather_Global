"""Frozen, timestamp-checked shared training order for scalar comparison arms."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_segments(split, anchors, times, *, segment_steps, input_steps=2,
                      train_start='2015-01-01T00:00:00', train_end='2021-12-31T18:00:00'):
    """Partition contiguous valid anchors; inspect every input and target time."""
    split, anchors = np.asarray(split, np.int64), np.asarray(anchors, np.int64)
    times = np.asarray(times).astype('datetime64[ns]')
    if split.ndim != 1 or anchors.ndim != 1 or times.ndim != 1 or segment_steps < 1:
        raise ValueError('Invalid training manifest array shapes or segment length')
    if times.size < 2 or np.any(np.isnat(times)) or np.any(np.diff(times) != np.timedelta64(6, 'h')):
        raise ValueError('Prepared time coordinate must be complete consecutive six-hour data')
    if split.size == 0 or np.any(np.diff(split) <= 0) or split.min() < 0 or split.max() >= len(anchors):
        raise ValueError('Training split must contain unique increasing valid anchor indices')
    raw = anchors[split]
    valid = (raw >= input_steps - 1) & (raw + 1 < len(times))
    safe = np.clip(raw, input_steps - 1, len(times) - 2)
    valid &= (times[safe - input_steps + 1] >= np.datetime64(train_start, 'ns'))
    valid &= (times[safe + 1] <= np.datetime64(train_end, 'ns'))
    selected = split[valid]
    if not len(selected):
        raise ValueError('No training anchors meet complete input/target timestamp bounds')
    boundaries = np.flatnonzero(np.diff(anchors[selected]) != 1) + 1
    segments = []
    for run in np.split(selected, boundaries):
        for offset in range(0, len(run) - segment_steps + 1, segment_steps):
            segments.append(run[offset:offset + segment_steps].copy())
    if not segments:
        raise ValueError('No complete eligible training segment')
    return tuple(segments)


def build_order(segments, *, seed, max_steps, batch_size, segment_steps, bptt_steps):
    if min(max_steps, batch_size, segment_steps, bptt_steps) <= 0 or segment_steps % bptt_steps:
        raise ValueError('Invalid order dimensions')
    groups = len(segments) // batch_size
    if not groups:
        raise ValueError('Not enough segments for a complete batch')
    chunks = segment_steps // bptt_steps
    epochs, updates = [], []
    for epoch in range(math.ceil(max_steps / (groups * chunks))):
        permutation = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch]))).permutation(len(segments))
        epochs.append(permutation.tolist())
        for group in range(groups):
            ids = permutation[group * batch_size:(group + 1) * batch_size].tolist()
            for offset in range(0, segment_steps, bptt_steps):
                if len(updates) >= max_steps:
                    break
                updates.append({'epoch': epoch, 'segment_index': group * batch_size,
                                'segment_offset': offset, 'segment_ids': ids})
    return epochs, updates


def create_train_order_manifest(anchor_root, prepared_root, path, *, seed=22,
                                max_steps=2500, batch_size=4, segment_steps=96,
                                bptt_steps=24, train_end='2021-12-31T18:00:00'):
    anchor_root, prepared_root, path = Path(anchor_root), Path(prepared_root), Path(path)
    files = {'anchor_indices': anchor_root / 'anchors/anchor_indices.npy',
             'split_train': anchor_root / 'anchors/split_train.npy',
             'time': prepared_root / 'coords/time.npy'}
    arrays = {key: np.load(value, allow_pickle=False) for key, value in files.items()}
    segments = eligible_segments(arrays['split_train'], arrays['anchor_indices'], arrays['time'],
                                 segment_steps=segment_steps, train_end=train_end)
    epochs, updates = build_order(segments, seed=seed, max_steps=max_steps, batch_size=batch_size,
                                  segment_steps=segment_steps, bptt_steps=bptt_steps)
    raw = np.concatenate([arrays['anchor_indices'][segment] for segment in segments])
    payload = {'schema_version': 1, 'seed': seed, 'max_steps': max_steps, 'batch_size': batch_size,
               'segment_steps': segment_steps, 'bptt_steps': bptt_steps,
               'train_start': '2015-01-01T00:00:00', 'train_end': train_end,
               'numpy_version': np.__version__, 'sampler': 'PCG64(SeedSequence([seed,epoch]))',
               'sources': {key: {'path': str(value.resolve()), 'sha256': file_sha256(value)}
                           for key, value in files.items()},
               'segment_count': len(segments), 'segments': [segment.tolist() for segment in segments],
               'first_input_time': str(arrays['time'][raw.min() - 1]),
               'last_target_time': str(arrays['time'][raw.max() + 1]),
               'dropped_batch_segments_per_epoch': len(segments) % batch_size,
               'epoch_permutations': epochs, 'updates': updates}
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f'Existing training order differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def load_train_order_manifest(path, *, config, anchors, times, split):
    payload = json.loads(Path(path).read_text())
    if payload.get('schema_version') != 1:
        raise ValueError('Unknown training order schema')
    expected = {'seed': config.seed, 'max_steps': config.max_steps,
                'batch_size': config.distributed.per_device_batch_size,
                'segment_steps': config.segment_steps, 'bptt_steps': config.bptt_steps}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('Training order dimensions/seed differ from run configuration')
    if config.distributed.mode != 'single':
        raise ValueError('Frozen segment replay currently requires single-device training')
    for record in payload['sources'].values():
        if file_sha256(record['path']) != record['sha256']:
            raise ValueError(f"Training order source changed: {record['path']}")
    segments = eligible_segments(split, anchors, times, segment_steps=config.segment_steps,
                                 train_start=payload['train_start'], train_end=payload['train_end'])
    if [segment.tolist() for segment in segments] != payload['segments']:
        raise ValueError('Training order segments do not match validated source timestamps')
    epochs, updates = build_order(segments, **expected)
    if epochs != payload['epoch_permutations'] or updates != payload['updates']:
        raise ValueError('Training order mapping does not match the recorded seeded permutation')
    return segments, payload


def segment_ids_at_cursor(payload, cursor):
    batch, length, chunk = payload['batch_size'], payload['segment_steps'], payload['bptt_steps']
    groups = payload['segment_count'] // batch
    index = (cursor.epoch * groups + cursor.segment_index // batch) * (length // chunk) + cursor.segment_offset // chunk
    if not 0 <= index < len(payload['updates']):
        raise ValueError('Training cursor lies outside the frozen update mapping')
    entry = payload['updates'][index]
    if any(entry[key] != getattr(cursor, key) for key in ('epoch', 'segment_index', 'segment_offset')):
        raise ValueError('Training cursor is not aligned with frozen update mapping')
    return tuple(entry['segment_ids'])
