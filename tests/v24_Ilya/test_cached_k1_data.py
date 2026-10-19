"""Checks for k1 sequence ownership, padded coverage and resumable ordering."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.training.cached_data import CachedSequenceBatch
from src.models.mamba.v24_Ilya.training.cached_k1_data import CachedK1Cursor, CachedK1Data


def _field(value: float, *, input_field: bool = False) -> xr.Dataset:
    time = np.array([-6, 0] if input_field else [6], dtype="timedelta64[h]")
    return xr.Dataset(
        {"u": (("batch", "time", "lat", "lon"),
               np.full((1, len(time), 1, 2), value, np.float32)),
         "land": (("lat", "lon"), np.ones((1, 2), np.float32))},
        coords={"batch": [0], "time": time,
                "lat": np.array([0], np.float32), "lon": np.array([0, 1], np.float32)})


class VerifiedFixture:
    """The producer API returns independently owned singleton-batch datasets."""

    def __init__(self, train_segments=5, val_segments=2):
        self.common = SimpleNamespace(ar_tail_k=0, feedback_mode="baseline",
                                      bptt_steps=24, segment_steps=96,
                                      distributed=SimpleNamespace(global_batch_size=1))
        self.training_data = SimpleNamespace(segments=tuple(range(train_segments)),
                                             validation_segments=tuple(range(val_segments)))
        self.manifest = {"partial": False,
                         "compatibility": {"ar_tail_k": 0, "feedback_mode": "baseline",
                                           "bptt_steps": 24, "truth_prefix_steps": 24},
                         "chunks": [{"split": split, "segment_id": segment,
                                     "segment_offset": offset}
                                    for split, count in (("train", train_segments),
                                                         ("val", val_segments))
                                    for segment in range(count) for offset in (0, 24, 48, 72)]}
        self.identity = {"manifest_sha256": "immutable-producer"}
        self.returned = []

    def _chunk(self, segment, offset, split, fields):
        base = (10000 if split == "val" else 0) + segment * 96 + offset
        inputs = tuple(_field(base + index, input_field=True) for index in range(24))
        targets = tuple(_field(base + index + 1) for index in range(24))
        forcings = tuple(_field(base + index + 2) for index in range(24))
        result = CachedSequenceBatch(
            inputs, targets, forcings, targets if fields else None,
            forcings if fields else None, segment * 4 + offset // 24,
            split, segment, offset, offset // 24, np.arange(base, base + 24),
            tuple(str(base + index) for index in range(24)), None)
        self.returned.append(result)
        return result

    def build_chunk(self, cursor, *, include_validation_fields=False):
        return self._chunk(cursor.segment_index, cursor.segment_offset, "train",
                           include_validation_fields)

    def load_validation_chunk(self, segment_id, segment_offset, *, include_validation_fields=True):
        return self._chunk(segment_id, segment_offset, "val", include_validation_fields)


def test_owned_batch_preserves_lane_values_coordinates_and_dtypes():
    producer = VerifiedFixture()
    data = CachedK1Data(producer, batch_size=3, seed=22, shuffle=False)
    batch = data.build_batch(CachedK1Cursor())
    np.testing.assert_array_equal(batch.segment_ids, [0, 1, 2])
    np.testing.assert_array_equal(batch.valid_mask, [1, 1, 1])
    np.testing.assert_array_equal(batch.inputs[7].u[:, -1, 0, 0], [7, 103, 199])
    assert batch.inputs[7].sizes["batch"] == 3
    assert batch.inputs[7].lat.dtype == np.float32
    assert batch.inputs[7].u.dtype == np.float32
    assert batch.inputs[7].u.data.flags.c_contiguous
    assert batch.inputs[7].u.dims == ("batch", "time", "lat", "lon")
    assert batch.valid_sequences == 3 and batch.valid_weather_steps == 72
    assert batch.reset_state and batch.next_cursor == CachedK1Cursor(chunk_index=1)
    for lane in producer.returned:
        assert not np.shares_memory(batch.inputs[7].u.values, lane.inputs[7].u.values)
        lane.inputs[7].u.data.fill(-999)
        # Xarray's indexed coordinates can be read-only; replace their binding.
        lane.inputs[7].coords["lat"] = np.array([-999], dtype=np.float32)
    np.testing.assert_array_equal(batch.inputs[7].u[:, -1, 0, 0], [7, 103, 199])
    np.testing.assert_array_equal(batch.inputs[7].lat, [0])
    assert producer.common.distributed.global_batch_size == 1
    assert producer.identity == {"manifest_sha256": "immutable-producer"}


def test_group_keeps_segments_for_four_chunks_then_resets_and_pads():
    data = CachedK1Data(VerifiedFixture(), batch_size=3, seed=22, shuffle=False)
    batches = list(data.iter_batches(CachedK1Cursor(), max_batches=8))
    assert [b.reset_state for b in batches] == [True, False, False, False] * 2
    for index, batch in enumerate(batches):
        group = index // 4
        np.testing.assert_array_equal(batch.segment_ids, [0, 1, 2] if group == 0 else [3, 4, -1])
        assert batch.chunk_index == index % 4
        for lane, segment in enumerate(batch.segment_ids):
            if segment >= 0:
                np.testing.assert_array_equal(batch.raw_anchor_indices[lane],
                    np.arange(segment * 96 + (index % 4) * 24,
                              segment * 96 + (index % 4) * 24 + 24))
            else:
                assert np.all(batch.raw_anchor_indices[lane] == -1)
                assert batch.chunk_ids[lane] == -1 and batch.timestamps[lane] == ()
                assert batch.valid_mask[lane] == 0
                np.testing.assert_array_equal(batch.inputs[0].u[lane], batch.inputs[0].u[lane - 1])
    assert sum(batch.valid_weather_steps for batch in batches) == 5 * 96
    assert batches[-1].next_cursor == CachedK1Cursor(epoch=1)


@pytest.mark.parametrize("batch_size,steps,updates", [(8, 56, 1400), (4, 108, 2700)])
def test_production_106_segment_schedule_covers_every_segment_and_25_epochs(batch_size, steps, updates):
    data = CachedK1Data(VerifiedFixture(106, 15), batch_size=batch_size, seed=22)
    assert data.steps_per_epoch == steps
    assert data.completed_updates(CachedK1Cursor(epoch=25)) == updates
    for epoch in (0, 1, 24):
        order = data.segment_order(epoch)
        np.testing.assert_array_equal(np.sort(order), np.arange(106))
        for group in range(data.groups_per_epoch):
            for chunk in range(4):
                cursor = CachedK1Cursor(epoch, group, chunk)
                assert data.cursor_for_update(data.completed_updates(cursor)) == cursor
    assert len(data.segment_order(0)[(data.groups_per_epoch - 1) * batch_size:]) == 2
    np.testing.assert_array_equal(data.segment_order(0), data.segment_order(0))
    assert not np.array_equal(data.segment_order(0), data.segment_order(1))


def test_resume_reconstructs_identical_midsegment_lanes_and_targets():
    first = CachedK1Data(VerifiedFixture(), batch_size=3, seed=72)
    cursor = first.cursor_for_update(13)
    resumed = CachedK1Cursor.from_mapping(cursor.to_dict())
    second = CachedK1Data(VerifiedFixture(), batch_size=3, seed=72)
    a, b = first.build_batch(cursor), second.build_batch(resumed)
    np.testing.assert_array_equal(a.segment_ids, b.segment_ids)
    np.testing.assert_array_equal(a.raw_anchor_indices, b.raw_anchor_indices)
    assert a.next_cursor == b.next_cursor and not a.reset_state
    for left, right in zip(a.targets, b.targets):
        xr.testing.assert_identical(left, right)


def test_validation_uses_own_order_padding_and_fields():
    data = CachedK1Data(VerifiedFixture(5, 2), batch_size=3, seed=22)
    batch = data.build_validation_batch(0, 2)
    assert batch.split == "val" and not batch.reset_state and batch.next_cursor is None
    np.testing.assert_array_equal(batch.segment_ids, [0, 1, -1])
    np.testing.assert_array_equal(batch.valid_mask, [1, 1, 0])
    assert batch.truths is not None and batch.baselines is not None
    assert float(batch.truths[0].u[0, 0, 0, 0]) == 10049
    assert data.completed_updates(CachedK1Cursor()) == 0


@pytest.mark.parametrize("change,match", [
    (lambda p: p.manifest["compatibility"].update(ar_tail_k=20), "genuine"),
    (lambda p: p.manifest["compatibility"].update(truth_prefix_steps=4), "observed"),
    (lambda p: p.manifest.update(partial=True), "complete"),
    (lambda p: p.manifest["chunks"].pop(), "coverage"),
    (lambda p: p.manifest["chunks"].append(p.manifest["chunks"][0]), "coverage"),
    (lambda p: setattr(p.common.distributed, "global_batch_size", 4), "original batch"),
])
def test_rejects_incompatible_or_incomplete_producer(change, match):
    producer = VerifiedFixture()
    change(producer)
    with pytest.raises(ValueError, match=match):
        CachedK1Data(producer, batch_size=3, seed=22)


@pytest.mark.parametrize("value", [-1, 0.5, True, "1"])
def test_cursor_rejects_noninteger_or_negative_fields(value):
    with pytest.raises(ValueError):
        CachedK1Cursor(epoch=value)


def test_cursor_bounds_and_empty_iteration():
    data = CachedK1Data(VerifiedFixture(), batch_size=3, seed=22)
    assert list(data.iter_batches(CachedK1Cursor(), max_batches=0)) == []
    with pytest.raises(ValueError, match="group_index"):
        data.build_batch(CachedK1Cursor(group_index=2))
    with pytest.raises(ValueError, match="chunk_index"):
        data.build_batch(CachedK1Cursor(chunk_index=4))
    with pytest.raises(ValueError):
        CachedK1Cursor.from_mapping({"epoch": 0, "group_index": 0})
