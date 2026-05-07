from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
for path in (ROOT, GRAPHCAST_LOCAL):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.models.graphcast.training.core.batching import (  # noqa: E402
    build_batch_from_indices,
    build_batch_from_indices_direct,
    build_batch_from_indices_vectorized,
    select_batch_builders,
)
from src.models.graphcast.training.core.logging import build_batch_builder_metadata  # noqa: E402
from src.models.graphcast.training.core.prepared_array import (  # noqa: E402
    PREPARED_ARRAY_FORMAT_VERSION,
    PreparedArrayStore,
)
from src.models.graphcast.training.core.prepared_block_batches import PreparedBlockBatchLoader  # noqa: E402
from src.models.graphcast.training.core.segments import SegmentBlockBatchLoader, SegmentChunk  # noqa: E402


def _task_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        input_variables=("temperature", "land_sea_mask"),
        target_variables=("temperature",),
        forcing_variables=("toa_incident_solar_radiation",),
        pressure_levels=(500, 850),
        input_duration="12h",
    )


def _make_dataset() -> xr.Dataset:
    batch = np.array([0], dtype=np.int64)
    time = np.array("2021-01-01T00:00:00", dtype="datetime64[ns]") + np.arange(8) * np.timedelta64(6, "h")
    level = np.array([500, 850], dtype=np.int64)
    lat = np.array([-10.0, 10.0], dtype=np.float32)
    lon = np.array([0.0, 120.0, 240.0], dtype=np.float32)

    temp = np.arange(batch.size * time.size * level.size * lat.size * lon.size, dtype=np.float32).reshape(
        batch.size, time.size, level.size, lat.size, lon.size
    )
    forcing = np.arange(batch.size * time.size * lat.size * lon.size, dtype=np.float32).reshape(
        batch.size, time.size, lat.size, lon.size
    )
    static = np.arange(lat.size * lon.size, dtype=np.float32).reshape(lat.size, lon.size)

    return xr.Dataset(
        data_vars={
            "temperature": (("batch", "time", "level", "lat", "lon"), temp),
            "toa_incident_solar_radiation": (("batch", "time", "lat", "lon"), forcing),
            "land_sea_mask": (("lat", "lon"), static),
        },
        coords={
            "batch": batch,
            "time": time,
            "datetime": ("time", time),
            "level": level,
            "lat": lat,
            "lon": lon,
        },
    )


def _assert_datasets_match(actual: xr.Dataset, expected: xr.Dataset) -> None:
    assert set(actual.data_vars) == set(expected.data_vars)
    assert dict(actual.sizes) == dict(expected.sizes)
    for name in actual.data_vars:
        assert actual[name].dims == expected[name].dims
        np.testing.assert_allclose(np.asarray(actual[name].values), np.asarray(expected[name].values))
        for coord_name in actual[name].coords:
            np.testing.assert_array_equal(
                np.asarray(actual[name].coords[coord_name].values),
                np.asarray(expected[name].coords[coord_name].values),
            )


def test_vectorized_batch_builder_matches_legacy_for_multiple_indices() -> None:
    ds = _make_dataset()
    cfg = _task_cfg()

    legacy = build_batch_from_indices(
        ds,
        indices=[1, 4],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )
    vectorized = build_batch_from_indices_vectorized(
        ds,
        indices=[1, 4],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )

    for actual, expected in zip(vectorized, legacy):
        _assert_datasets_match(actual, expected)


def test_vectorized_batch_builder_matches_legacy_for_single_index() -> None:
    ds = _make_dataset()
    cfg = _task_cfg()

    legacy = build_batch_from_indices(
        ds,
        indices=[3],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )
    vectorized = build_batch_from_indices_vectorized(
        ds,
        indices=[3],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )

    for actual, expected in zip(vectorized, legacy):
        _assert_datasets_match(actual, expected)


def test_direct_batch_builder_matches_legacy_for_multiple_indices() -> None:
    ds = _make_dataset()
    cfg = _task_cfg()

    legacy = build_batch_from_indices(
        ds,
        indices=[1, 4],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )
    direct = build_batch_from_indices_direct(
        ds,
        indices=[1, 4],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )

    for actual, expected in zip(direct, legacy):
        _assert_datasets_match(actual, expected)


def test_direct_batch_builder_matches_legacy_for_single_index() -> None:
    ds = _make_dataset()
    cfg = _task_cfg()

    legacy = build_batch_from_indices(
        ds,
        indices=[3],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )
    direct = build_batch_from_indices_direct(
        ds,
        indices=[3],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )

    for actual, expected in zip(direct, legacy):
        _assert_datasets_match(actual, expected)


def test_direct_batch_builder_works_without_source_batch_dim() -> None:
    ds = _make_dataset().isel(batch=0, drop=True)
    cfg = _task_cfg()

    inputs, targets, forcings = build_batch_from_indices_direct(
        ds,
        indices=[2, 5],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )

    assert inputs["land_sea_mask"].dims == ("batch", "lat", "lon")
    assert inputs["temperature"].dims == ("batch", "time", "level", "lat", "lon")
    assert targets["temperature"].sizes["level"] == 2
    assert forcings["toa_incident_solar_radiation"].dims == ("batch", "time", "lat", "lon")


def test_numpy_builder_requires_active_cache() -> None:
    ds = _make_dataset()
    with pytest.raises(ValueError, match="--batch-builder numpy requires an active full-RAM train cache"):
        select_batch_builders(
            ds,
            ds,
            requested="numpy",
            should_cache_train=False,
            task_cfg=_task_cfg(),
        )


def test_direct_builder_reads_tiny_zarr(tmp_path) -> None:
    pytest.importorskip("zarr")
    ds = _make_dataset().isel(batch=0, drop=True)
    store_path = tmp_path / "res1"
    ds.to_zarr(store_path, mode="w", consolidated=True)
    opened = xr.open_zarr(store_path, consolidated=True)

    expected = build_batch_from_indices_direct(
        ds,
        indices=[2, 5],
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    actual = build_batch_from_indices_direct(
        opened,
        indices=[2, 5],
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )

    for actual_ds, expected_ds in zip(actual, expected):
        _assert_datasets_match(actual_ds, expected_ds)


def _write_tiny_prepared_array_store(tmp_path: Path, ds: xr.Dataset) -> PreparedArrayStore:
    store = tmp_path / "res1"
    (store / "coords").mkdir(parents=True)
    (store / "vars").mkdir()
    source = ds.isel(batch=0, drop=True)
    for coord in ("time", "lat", "lon", "level"):
        np.save(store / "coords" / f"{coord}.npy", np.asarray(source.coords[coord].values))
    variables = {}
    for name in ("temperature", "toa_incident_solar_radiation", "land_sea_mask"):
        values = np.asarray(source[name].values)
        np.save(store / "vars" / f"{name}.npy", values)
        variables[name] = {
            "dims": list(source[name].dims),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
    metadata = {
        "prepared_array_format_version": PREPARED_ARRAY_FORMAT_VERSION,
        "resolution": 1.0,
        "pressure_levels": [500, 850],
        "task_input_variables": ["temperature", "land_sea_mask"],
        "task_target_variables": ["temperature"],
        "task_forcing_variables": ["toa_incident_solar_radiation"],
        "variables": variables,
    }
    (store / "metadata.json").write_text(__import__("json").dumps(metadata))
    return PreparedArrayStore(store)


def test_prepared_array_batch_builder_matches_direct(tmp_path) -> None:
    ds = _make_dataset()
    cfg = _task_cfg()
    store = _write_tiny_prepared_array_store(tmp_path, ds)

    expected = build_batch_from_indices_direct(
        ds,
        indices=[2, 5],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )
    actual = store.build_batch_from_indices(
        indices=[2, 5],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )

    for actual_ds, expected_ds in zip(actual, expected):
        _assert_datasets_match(actual_ds, expected_ds)


def test_select_batch_builders_uses_prepared_array_without_full_cache(tmp_path) -> None:
    store = _write_tiny_prepared_array_store(tmp_path, _make_dataset())
    selection = select_batch_builders(
        store,
        store,
        requested="numpy",
        should_cache_train=False,
        task_cfg=_task_cfg(),
    )

    assert selection.effective_train_batch_builder == "prepared_array"
    assert selection.effective_eval_batch_builder == "prepared_array"
    assert selection.numpy_cache_active is False


def test_prepared_array_segment_block_loader_matches_direct(tmp_path) -> None:
    ds = _make_dataset()
    cfg = _task_cfg()
    store = _write_tiny_prepared_array_store(tmp_path, ds)
    segments = [np.asarray([1, 2, 3], dtype=np.int64), np.asarray([4, 5, 6], dtype=np.int64)]
    chunk = SegmentChunk(
        chunk_indices=(np.asarray([1, 4], dtype=np.int64), np.asarray([2, 5], dtype=np.int64)),
        reset_mask=np.asarray([True, True]),
        lane_segment_ids=np.asarray([0, 1], dtype=np.int64),
        lane_offsets=np.asarray([0, 0], dtype=np.int64),
        epoch=0,
    )
    loader = SegmentBlockBatchLoader(
        store,
        segments,
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
        label="test-prepared-array-segment",
    )

    inputs, targets, forcings, stats = loader.load_chunk(chunk)

    assert stats.cache_misses == 2
    for bptt_i, step_indices in enumerate(chunk.chunk_indices):
        expected = build_batch_from_indices_direct(
            ds,
            indices=step_indices,
            input_steps=2,
            target_steps=1,
            task_cfg=cfg,
            dt=pd.Timedelta("6h"),
        )
        for actual_ds, expected_ds in zip((inputs[bptt_i], targets[bptt_i], forcings[bptt_i]), expected):
            _assert_datasets_match(actual_ds, expected_ds)


def test_prepared_block_loader_load_chunk_matches_direct(tmp_path) -> None:
    ds = _make_dataset()
    cfg = _task_cfg()
    store = _write_tiny_prepared_array_store(tmp_path, ds)
    segments = [np.asarray([1, 2, 3], dtype=np.int64), np.asarray([4, 5, 6], dtype=np.int64)]
    chunk = SegmentChunk(
        chunk_indices=(np.asarray([1, 4], dtype=np.int64), np.asarray([2, 5], dtype=np.int64)),
        reset_mask=np.asarray([True, True]),
        lane_segment_ids=np.asarray([0, 1], dtype=np.int64),
        lane_offsets=np.asarray([0, 0], dtype=np.int64),
        epoch=0,
    )
    loader = PreparedBlockBatchLoader(
        store,
        segments,
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
        label="test-prepared-block",
    )

    inputs, targets, forcings, stats = loader.load_chunk(chunk)

    assert stats.cache_misses == 2
    for bptt_i, step_indices in enumerate(chunk.chunk_indices):
        expected = build_batch_from_indices_direct(
            ds,
            indices=step_indices,
            input_steps=2,
            target_steps=1,
            task_cfg=cfg,
            dt=pd.Timedelta("6h"),
        )
        for actual_ds, expected_ds in zip((inputs[bptt_i], targets[bptt_i], forcings[bptt_i]), expected):
            _assert_datasets_match(actual_ds, expected_ds)


def test_prepared_block_loader_iter_chunk_batches_matches_load_chunk(tmp_path) -> None:
    ds = _make_dataset()
    cfg = _task_cfg()
    store = _write_tiny_prepared_array_store(tmp_path, ds)
    segments = [np.asarray([1, 2, 3], dtype=np.int64), np.asarray([4, 5, 6], dtype=np.int64)]
    chunk = SegmentChunk(
        chunk_indices=(np.asarray([1, 4], dtype=np.int64), np.asarray([2, 5], dtype=np.int64)),
        reset_mask=np.asarray([True, True]),
        lane_segment_ids=np.asarray([0, 1], dtype=np.int64),
        lane_offsets=np.asarray([0, 0], dtype=np.int64),
        epoch=0,
    )
    load_chunk_loader = PreparedBlockBatchLoader(
        store,
        segments,
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
        label="test-prepared-block-load",
    )
    iter_loader = PreparedBlockBatchLoader(
        store,
        segments,
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
        label="test-prepared-block-iter",
    )

    expected_inputs, expected_targets, expected_forcings, _ = load_chunk_loader.load_chunk(chunk)
    actual_batches = list(iter_loader.iter_chunk_batches(chunk))

    assert len(actual_batches) == len(chunk.chunk_indices)
    for bptt_i, (actual, stats) in enumerate(actual_batches):
        assert stats.cache_misses == 2
        for actual_ds, expected_ds in zip(
            actual,
            (expected_inputs[bptt_i], expected_targets[bptt_i], expected_forcings[bptt_i]),
        ):
            _assert_datasets_match(actual_ds, expected_ds)


def test_vectorized_batch_builder_avoids_vectorized_time_indexing(monkeypatch) -> None:
    ds = _make_dataset()
    cfg = _task_cfg()
    original_isel = xr.Dataset.isel

    def guarded_isel(self, indexers=None, drop=False, missing_dims="raise", **indexers_kwargs):
        merged_indexers = {}
        if indexers is not None:
            merged_indexers.update(indexers)
        merged_indexers.update(indexers_kwargs)
        if isinstance(merged_indexers.get("time"), xr.DataArray):
            raise TypeError("Vectorized indexing is not supported")
        return original_isel(self, indexers=indexers, drop=drop, missing_dims=missing_dims, **indexers_kwargs)

    monkeypatch.setattr(xr.Dataset, "isel", guarded_isel)

    inputs, targets, forcings = build_batch_from_indices_vectorized(
        ds,
        indices=[2, 5],
        input_steps=2,
        target_steps=1,
        task_cfg=cfg,
        dt=pd.Timedelta("6h"),
    )

    assert inputs.sizes["batch"] == 2
    assert targets.sizes["batch"] == 2
    assert forcings.sizes["batch"] == 2


def test_batch_builder_metadata_marks_numpy_fallback() -> None:
    metadata = build_batch_builder_metadata(
        requested_batch_builder="numpy",
        effective_train_batch_builder="vectorized",
        effective_eval_batch_builder="vectorized",
        numpy_cache_active=False,
    )

    assert metadata["requested_batch_builder"] == "numpy"
    assert metadata["effective_train_batch_builder"] == "vectorized"
    assert metadata["effective_eval_batch_builder"] == "vectorized"
    assert metadata["numpy_cache_active"] is False
    assert metadata["used_fallback"] is True


def test_batch_builder_metadata_preserves_non_fallback_modes() -> None:
    vectorized = build_batch_builder_metadata(
        requested_batch_builder="vectorized",
        effective_train_batch_builder="vectorized",
        effective_eval_batch_builder="vectorized",
        numpy_cache_active=False,
    )
    legacy = build_batch_builder_metadata(
        requested_batch_builder="legacy",
        effective_train_batch_builder="legacy",
        effective_eval_batch_builder="legacy",
        numpy_cache_active=False,
    )
    cached_numpy = build_batch_builder_metadata(
        requested_batch_builder="numpy",
        effective_train_batch_builder="numpy",
        effective_eval_batch_builder="numpy",
        numpy_cache_active=True,
    )

    assert vectorized["used_fallback"] is False
    assert legacy["used_fallback"] is False
    assert cached_numpy["used_fallback"] is False
    assert cached_numpy["numpy_cache_active"] is True
