from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.mamba.v24_Ilya.training.segment_order import (
    build_order, create_train_order_manifest, eligible_segments,
    load_train_order_manifest, segment_ids_at_cursor,
)


def test_timestamp_bounds_exclude_cross_year_targets_and_split_gaps():
    times = np.arange(np.datetime64('2021-12-25'), np.datetime64('2022-01-03'), np.timedelta64(6, 'h'))
    anchors = np.arange(1, len(times) - 1)
    split = np.arange(len(anchors))
    segments = eligible_segments(split, anchors, times, segment_steps=4)
    assert all(np.all(times[anchors[segment] + 1] <= np.datetime64('2021-12-31T18')) for segment in segments)
    split = np.delete(split, 10)
    segments = eligible_segments(split, anchors, times, segment_steps=4)
    assert all(np.all(np.diff(anchors[segment]) == 1) for segment in segments)


def test_seeded_order_replays_cursor_and_drops_incomplete_batches():
    segments = tuple(np.arange(i * 8, (i + 1) * 8) for i in range(11))
    epochs, updates = build_order(segments, seed=22, max_steps=20, batch_size=4, segment_steps=8, bptt_steps=2)
    payload = dict(batch_size=4, segment_steps=8, bptt_steps=2, segment_count=11, updates=updates)
    for entry in updates:
        cursor = SimpleNamespace(**{k: entry[k] for k in ('epoch', 'segment_index', 'segment_offset')})
        assert segment_ids_at_cursor(payload, cursor) == tuple(entry['segment_ids'])
    assert len(updates) == 20
    assert updates[0]['segment_ids'] == updates[3]['segment_ids']
    assert epochs[0] != epochs[1]
    assert epochs == build_order(segments, seed=22, max_steps=20, batch_size=4, segment_steps=8, bptt_steps=2)[0]
    assert epochs != build_order(segments, seed=23, max_steps=20, batch_size=4, segment_steps=8, bptt_steps=2)[0]


def test_manifest_hash_and_sampler_dimensions_are_integrity_checks(tmp_path):
    anchor_root, prepared = tmp_path / 'anchors', tmp_path / 'prepared'
    (anchor_root / 'anchors').mkdir(parents=True)
    (prepared / 'coords').mkdir(parents=True)
    times = np.arange(np.datetime64('2021-01-01'), np.datetime64('2021-02-01'), np.timedelta64(6, 'h'))
    anchors = np.arange(1, len(times) - 1)
    split = np.arange(len(anchors))
    np.save(anchor_root / 'anchors/anchor_indices.npy', anchors)
    np.save(anchor_root / 'anchors/split_train.npy', split)
    np.save(prepared / 'coords/time.npy', times)
    path = tmp_path / 'order.json'
    payload = create_train_order_manifest(anchor_root, prepared, path, max_steps=30, segment_steps=8, bptt_steps=2)
    config = SimpleNamespace(seed=22, max_steps=30, segment_steps=8, bptt_steps=2,
                             distributed=SimpleNamespace(mode='single', per_device_batch_size=4))
    segments, loaded = load_train_order_manifest(path, config=config, anchors=anchors, times=times, split=split)
    assert loaded == payload
    assert len(segments) == payload['segment_count']
    np.save(anchor_root / 'anchors/split_train.npy', split[:-1])
    with pytest.raises(ValueError, match='source changed'):
        load_train_order_manifest(path, config=config, anchors=anchors, times=times, split=split)


def test_incomplete_or_duplicate_time_coordinates_fail():
    times = np.asarray(['2021-01-01T00', '2021-01-01T06', '2021-01-01T18'], dtype='datetime64[h]')
    with pytest.raises(ValueError, match='consecutive'):
        eligible_segments(np.arange(2), np.arange(2), times, segment_steps=1)
