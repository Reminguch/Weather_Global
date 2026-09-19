from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import jax

from src.models.mamba.v24_Ilya import baseline_cache as cache
from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.cached_config import (
    CachedExecutionConfig, CachedTrainingConfig, load_cached_training_config,
)
from src.models.mamba.v24_Ilya.training.cached_data import (
    CachedTrainingData, open_cached_training_data,
)
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig
from src.models.mamba.v24_Ilya.training.data import TrainingCursor, V24IlyaTrainingData


def _frame(value, forcing=None):
    data = {"u": (("batch", "time", "lat", "lon"), np.full((1, 1, 1, 2), value, np.float32))}
    if forcing is not None:
        data["f"] = (("batch", "time", "lat", "lon"), np.full((1, 1, 1, 2), forcing, np.float32))
    return xr.Dataset(data, coords={"batch": [0], "time": np.array([6], dtype="timedelta64[h]"), "lat": np.array([0.], np.float32), "lon": np.array([0., 1.], np.float32)})


@pytest.fixture
def toy_cache(tmp_path, monkeypatch):
    common = V24IlyaTrainConfig(
        prepared_root=tmp_path / "prepared", anchor_manifest_root=tmp_path / "anchors",
        baseline_checkpoint=tmp_path / "baseline.npz", output_root=tmp_path / "output",
        run_name="toy", architecture=V24IlyaArchitectureConfig(),
        segment_steps=8, bptt_steps=4, ar_tail_k=2, loss_mode="all_steps",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(common.to_dict()))
    root = tmp_path / "cache"
    root.mkdir()
    times = np.datetime64("2020-01-01T00") + np.arange(40) * np.timedelta64(6, "h")
    data = SimpleNamespace(
        store=SimpleNamespace(time=SimpleNamespace(values=times)),
        time_step=pd.Timedelta("6h"), anchor_indices=np.arange(1, 33),
        train_split=np.arange(16), val_split=np.arange(16, 24),
        segments=(np.arange(8), np.arange(8, 16)),
        validation_segments=(np.arange(16, 24),),
    )
    data.validate_cursor = lambda cursor, config: V24IlyaTrainingData.validate_cursor(data, cursor, config)
    chunks, excluded = cache.enumerate_chunks(data, common)
    compatibility = {"bptt_steps": 4, "truth_prefix_steps": 2}
    manifest = {
        "format": cache.FORMAT, "compatibility": compatibility,
        "compatibility_sha256": cache.digest(compatibility),
        "variables": {"u": {"dims": ["lat", "lon"], "shape": [1, 2], "dtype": "float32"}},
        "coordinates": {"lat": [0.], "lon": [0., 1.]},
        "chunks": cache.select_chunks(chunks, None, 1), "num_shards": 1,
        "excluded_tail_anchors": excluded, "full_chunk_count": len(chunks), "partial": False,
    }
    manifest["manifest_sha256"] = cache.digest(manifest)
    truth_slab = np.empty((4, 1, 1, 1, 2), dtype=np.float32)
    truth_change = {"offset": 0.0}

    def load(context, item):
        start = item["raw_anchor_indices"][0]
        truths = []
        for index in range(4):
            truth_slab[index].fill(start + index + 2 + truth_change["offset"])
            truth = _frame(0.)
            truth["u"].data = truth_slab[index]
            truths.append(truth)
        return SimpleNamespace(
            input_frames=tuple(_frame(start + i, start + i + 100) for i in range(3)),
            static_inputs=xr.Dataset({"land": (("batch", "lat", "lon"), np.ones((1, 1, 2), np.float32))}),
            truths=tuple(truths),
            forcings=tuple(_frame(0, start + i + 102).drop_vars("u") for i in range(4)),
            raw_anchor_indices=np.array(item["raw_anchor_indices"]),
        )

    def predict(inputs, truth, forcing):
        result = truth.copy(deep=True)
        result["u"].data[:] = inputs["u"].isel(time=-1).values[:, None] + 10
        return result

    context = (common, None, None, None, data, None)
    monkeypatch.setattr(cache, "load_chunk", load)
    monkeypatch.setattr(cache, "make_predictor", lambda _: predict)
    cache.generate_shard(context, manifest, root, 0)
    cache.verify_cache(root)
    monkeypatch.setattr(cache, "make_predictor", lambda _: pytest.fail("Reader initialized baseline"))
    reader = cache.BaselineCacheReader(root, manifest["compatibility_sha256"])
    adapter = CachedTrainingData(context, reader)
    config = load_cached_training_config(config_path, cache_root=root)
    return SimpleNamespace(
        root=root, context=context, manifest=manifest, load=load, reader=reader,
        adapter=adapter, config=config, slab=truth_slab, truth_change=truth_change,
        predict=predict,
    )


def test_owned_sequences_match_reader_after_workspace_reuse(toy_cache):
    toy = toy_cache
    item = toy.manifest["chunks"][0]
    reference_chunk = toy.load(toy.context, item)
    reference = [(a.copy(deep=True), b.copy(deep=True), c.copy(deep=True), d.copy(deep=True))
                 for a, b, c, d in toy.reader.iter_chunk(0, reference_chunk, toy.context[4].time_step)]
    first = toy.adapter.build_chunk(TrainingCursor(), include_validation_fields=True)
    original_truths = tuple(truth.copy(deep=True) for truth in first.truths)
    second = toy.adapter.build_chunk(first.next_cursor)
    assert second.truths is None and second.baselines is None
    for index, (inputs, baseline, target, forcing) in enumerate(reference):
        xr.testing.assert_identical(first.inputs[index], inputs)
        xr.testing.assert_identical(first.baselines[index], baseline)
        xr.testing.assert_identical(first.targets[index], target)
        xr.testing.assert_identical(first.forcings[index], forcing)
        xr.testing.assert_identical(first.truths[index], original_truths[index])
        assert not np.shares_memory(first.truths[index].u.data, toy.slab)
        assert first.inputs[index].u.data.flags.c_contiguous
        assert first.targets[index].u.dtype == np.float32


def test_chunk_restarts_and_cursor_segment_epoch_boundaries(toy_cache):
    batches = list(toy_cache.adapter.iter_chunks(TrainingCursor(), max_chunks=5, prefetch_batches=0))
    assert [batch.chunk_id for batch in batches] == [0, 1, 2, 3, 0]
    assert [batch.reset_state for batch in batches] == [True, False, True, False, True]
    assert batches[3].next_cursor == TrainingCursor(epoch=1)
    for batch in batches:
        start = batch.raw_anchor_indices[0]
        assert [float(inputs.u.isel(time=-1).mean()) for inputs in batch.inputs] == [start + 1, start + 2, start + 12, start + 22]
        assert [float(forcing.f.mean()) for forcing in batch.forcings] == [start + 102, start + 103, start + 104, start + 105]


def test_future_truth_never_changes_weather_inputs(toy_cache):
    before = toy_cache.adapter.build_chunk(TrainingCursor(), include_validation_fields=True)
    toy_cache.truth_change["offset"] = 123.0
    after = toy_cache.adapter.build_chunk(TrainingCursor(), include_validation_fields=True)
    for index in range(4):
        xr.testing.assert_identical(before.inputs[index], after.inputs[index])
        xr.testing.assert_identical(before.baselines[index], after.baselines[index])
        np.testing.assert_array_equal(after.targets[index].u - before.targets[index].u, 123.)


def test_validation_identity_and_training_cursor_are_separate(toy_cache):
    validation = toy_cache.adapter.load_validation_chunk(0, 0)
    assert validation.split == "val" and validation.chunk_id == 4
    assert validation.next_cursor is None and validation.reset_state
    assert validation.truths is not None and validation.baselines is not None
    training = toy_cache.adapter.build_chunk(TrainingCursor())
    assert training.chunk_id == 0 and training.next_cursor == TrainingCursor(segment_offset=4)
    assert set(training.timestamps).isdisjoint(validation.timestamps)
    with pytest.raises(ValueError, match="No cached"):
        toy_cache.adapter.load_validation_chunk(0, 1)


def test_one_batch_prefetch_preserves_owned_values_and_order(toy_cache):
    serial = list(toy_cache.adapter.iter_chunks(TrainingCursor(), max_chunks=4, prefetch_batches=0))
    prefetched = list(toy_cache.adapter.iter_chunks(TrainingCursor(), max_chunks=4, prefetch_batches=1))
    assert [batch.chunk_id for batch in prefetched] == [0, 1, 2, 3]
    for left, right in zip(serial, prefetched):
        assert left.next_cursor == right.next_cursor
        for left_target, right_target in zip(left.targets, right.targets):
            xr.testing.assert_identical(left_target, right_target)
    iterator = toy_cache.adapter.iter_chunks(TrainingCursor(), prefetch_batches=1)
    assert next(iterator).chunk_id == 0
    iterator.close()
    assert list(toy_cache.adapter.iter_chunks(TrainingCursor(), max_chunks=0)) == []


def test_open_validates_current_identity_and_ready_counts(toy_cache, monkeypatch):
    toy = toy_cache
    monkeypatch.setattr(cache, "load_context", lambda _: toy.context)
    monkeypatch.setattr(cache, "build_manifest", lambda context, num_shards: toy.manifest)
    opened = open_cached_training_data(toy.config)
    assert opened.identity["manifest_sha256"] == toy.manifest["manifest_sha256"]
    assert opened.identity["train_chunks"] == 4
    assert opened.identity["validation_chunks"] == 2
    ready_path = toy.root / "READY.json"
    ready = json.loads(ready_path.read_text())
    ready["predictions"] -= 1
    ready_path.write_text(json.dumps(ready))
    with pytest.raises(ValueError, match="READY marker coverage"):
        open_cached_training_data(toy.config)


def test_open_rejects_source_change_pin_and_missing_ready(toy_cache, monkeypatch):
    toy = toy_cache
    monkeypatch.setattr(cache, "load_context", lambda _: toy.context)
    monkeypatch.setattr(cache, "build_manifest", lambda context, num_shards: {**toy.manifest, "num_shards": 2})
    with pytest.raises(ValueError, match="current source"):
        open_cached_training_data(toy.config)
    pinned = dataclasses.replace(toy.config, execution=dataclasses.replace(toy.config.execution, expected_manifest_sha256="0" * 64))
    with pytest.raises(ValueError, match="expected_manifest_sha256"):
        open_cached_training_data(pinned)
    monkeypatch.setattr(cache, "build_manifest", lambda context, num_shards: toy.manifest)
    (toy.root / "READY.json").unlink()
    with pytest.raises(FileNotFoundError):
        open_cached_training_data(toy.config)


def test_execution_settings_do_not_change_common_objective(toy_cache):
    config = load_cached_training_config(
        toy_cache.config.config_path, cache_root=toy_cache.root,
        prefetch_batches=1,
    )
    assert config.common == toy_cache.config.common
    assert config.common.bptt_steps == 4 and config.common.truth_prefix_steps == 2
    assert config.execution.to_dict()["backend"] == "cached_stepwise"
    assert toy_cache.config.execution.prefetch_batches == 0
    changed = dataclasses.replace(config.common, feedback_mode="closed_loop_sg")
    with pytest.raises(ValueError, match="feedback_mode"):
        CachedTrainingConfig(changed, config.execution, config.config_path)
    with pytest.raises(ValueError, match="all_steps"):
        CachedTrainingConfig(dataclasses.replace(config.common, loss_mode="last_step"), config.execution, config.config_path)


@pytest.mark.parametrize("overrides", [
    {"backend": "cached_sequence"}, {"prefetch_batches": True}, {"prefetch_batches": 2},
    {"expected_manifest_sha256": "unvalidated"},
])
def test_execution_rejects_invalid_settings(overrides):
    with pytest.raises(ValueError):
        CachedExecutionConfig(cache_root=Path("cache"), **overrides)


def test_validation_stream_matches_owned_view_and_releases_lock(toy_cache):
    owned = toy_cache.adapter.load_validation_chunk(0, 0)
    rows = toy_cache.adapter.iter_validation_steps(0, 0)
    for index, row in enumerate(rows):
        assert row.step_index == index
        xr.testing.assert_identical(row.inputs, owned.inputs[index])
        xr.testing.assert_identical(row.target, owned.targets[index])
        xr.testing.assert_identical(row.truth, owned.truths[index])
        xr.testing.assert_identical(row.baseline, owned.baselines[index])
        assert np.shares_memory(row.truth.u.data, toy_cache.slab)
    rows = toy_cache.adapter.iter_validation_steps(0, 0)
    next(rows)
    rows.close()
    assert toy_cache.adapter.build_chunk(TrainingCursor()).chunk_id == 0


def test_cached_spatial_coordinates_preserve_online_jax_tree_identity(toy_cache):
    toy = toy_cache

    def signature(dataset):
        # Coordinate values/dtypes live in the static tree definition, not the
        # array leaves: numeric weather equality alone misses this regression.
        leaves, definition = jax.tree_util.tree_flatten(dataset)
        return str(definition), tuple(np.asarray(leaf).tobytes() for leaf in leaves)

    item = toy.manifest["chunks"][0]
    chunk = toy.load(toy.context, item)
    raw = list(toy.reader.iter_chunk(item["id"], chunk, toy.context[4].time_step))
    assert raw[-1][0].lat.dtype == np.float64  # Current immutable producer reader.
    live = list(cache.iter_trajectory(chunk, toy.predict, 2, toy.context[4].time_step))
    owned = toy.adapter.build_chunk(TrainingCursor(), include_validation_fields=True)
    for index, inputs, baseline in live:
        target, forcing = chunk.truths[index] - baseline, chunk.forcings[index]
        for actual, expected in ((owned.inputs[index], inputs), (owned.targets[index], target),
                                 (owned.baselines[index], baseline), (owned.forcings[index], forcing)):
            assert actual.lat.dtype == expected.lat.dtype == np.float32
            assert actual.lon.dtype == expected.lon.dtype == np.float32
            assert signature(actual) == signature(expected)
        assert owned.inputs[index].sizes["time"] == 2
        assert owned.targets[index].sizes["time"] == 1

    val_item = next(item for item in toy.manifest["chunks"] if item["split"] == "val")
    val_chunk = toy.load(toy.context, val_item)
    val_live = list(cache.iter_trajectory(val_chunk, toy.predict, 2, toy.context[4].time_step))
    for row, (index, inputs, baseline) in zip(toy.adapter.iter_validation_steps(0, 0), val_live, strict=True):
        target, forcing = val_chunk.truths[index] - baseline, val_chunk.forcings[index]
        for actual, expected in ((row.inputs, inputs), (row.target, target),
                                 (row.baseline, baseline), (row.forcing, forcing)):
            assert signature(actual) == signature(expected)
