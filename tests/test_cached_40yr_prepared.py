from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data_operations.preprocessing import prepare_cached_40yr as preparation
from src.models.graphcast.training.core import prepared_array


@pytest.fixture
def source(tmp_path, monkeypatch):
    times = pd.date_range("1982-01-01", "1984-12-31 18:00", freq="6h")
    data = xr.Dataset({
        "temperature": (("time", "level", "lat", "lon"),
                        np.arange(len(times) * 12, dtype=np.float32).reshape(len(times), 2, 2, 3)),
        "land_sea_mask": (("lat", "lon"), np.ones((2, 3), dtype=np.float32)),
        "toa_incident_solar_radiation": (("time", "lat", "lon"), np.ones((len(times), 2, 3), dtype=np.float32)),
    }, coords={"time": times, "level": [500, 850], "lat": [-1, 1], "lon": [0, 1, 2]})
    root = tmp_path / "source.zarr"
    data.chunk({"time": 120}).to_zarr(root, consolidated=True, zarr_format=2)
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"synthetic checkpoint")
    task = SimpleNamespace(input_variables=("temperature", "land_sea_mask"), target_variables=("temperature",),
                           forcing_variables=("toa_incident_solar_radiation",), pressure_levels=(500, 850), input_duration="12h")
    fake = SimpleNamespace(model_config=SimpleNamespace(resolution=1.0), task_config=task)
    monkeypatch.setattr(preparation, "load_graphcast_checkpoint", lambda _: fake)
    monkeypatch.setattr(preparation, "_check_grid", lambda *_: None)
    return root, checkpoint, task


def test_monthly_resume_checks_completed_data_and_leap_year(source, tmp_path, monkeypatch):
    root, checkpoint, task = source
    output = tmp_path / "prepared"
    original = preparation._json

    def interrupt(path, value):
        if path.name == "progress.json" and "1982-02" in value["months"]:
            raise InterruptedError("synthetic stop before month commit")
        original(path, value)

    with monkeypatch.context() as patched:
        patched.setattr(preparation, "_json", interrupt)
        with pytest.raises(InterruptedError):
            preparation.prepare_year(root, output, 1982, checkpoint, chunk_time=17)
    assert not (output / "READY.json").exists()
    assert not (output / "metadata.json").exists()
    assert (output / ".incomplete").exists()
    saved = json.loads((output / "years/1982/progress.json").read_text())
    assert set(saved["months"]) == {"1982-01"}
    completed = preparation.prepare_year(root, output, 1982, checkpoint, chunk_time=11)
    assert len(completed["months"]) == 12
    assert completed["months"]["1982-01"] == saved["months"]["1982-01"]
    repeated = preparation.prepare_year(root, output, 1982, checkpoint)
    assert repeated == completed
    leap = preparation.prepare_year(root, output, 1984, checkpoint)
    assert leap["months"]["1984-02"]["stop"] - leap["months"]["1984-02"]["start"] == 29 * 4
    assert leap["variables"]["temperature"]["shape"][0] == 366 * 4
    assert leap["variables"]["temperature"]["dtype"] == "float32"


def test_resume_rejects_corrupted_month(source, tmp_path, monkeypatch):
    root, checkpoint, _ = source
    output = tmp_path / "prepared"
    original = preparation._json

    def interrupt(path, value):
        if path.name == "progress.json" and "1982-02" in value["months"]:
            raise InterruptedError
        original(path, value)

    with monkeypatch.context() as patched:
        patched.setattr(preparation, "_json", interrupt)
        with pytest.raises(InterruptedError):
            preparation.prepare_year(root, output, 1982, checkpoint)
    values = np.load(output / "years/1982/vars/temperature.npy", mmap_mode="r+")
    values[0] += 1
    values.flush()
    values._mmap.close()
    with pytest.raises(ValueError, match="checksum mismatch"):
        preparation.prepare_year(root, output, 1982, checkpoint)
    assert not (output / "READY.json").exists()


def test_finalize_rejects_partial_campaign_before_publishing(tmp_path):
    with pytest.raises(ValueError, match="every year"):
        preparation.finalize_prepared(tmp_path / "source", tmp_path / "output", [1982, 1984], tmp_path / "checkpoint")
    assert not (tmp_path / "output").exists()


def test_grid_and_finite_checks_reject_incompatible_source(tmp_path):
    levels = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    checkpoint = SimpleNamespace(model_config=SimpleNamespace(resolution=1.0),
                                 task_config=SimpleNamespace(pressure_levels=levels))
    dataset = xr.Dataset(coords={"level": list(levels), "lat": np.arange(90, -91, -1), "lon": np.arange(360)})
    preparation._check_grid(dataset, checkpoint)
    with pytest.raises(ValueError, match="181 x 360"):
        preparation._check_grid(dataset.isel(lat=slice(1, None)), checkpoint)
    with pytest.raises(ValueError, match="13 pressure"):
        preparation._check_grid(dataset.isel(level=slice(None, -1)), checkpoint)
    path = tmp_path / "nonfinite.npy"
    np.save(path, np.asarray([1, np.nan, 3], dtype=np.float32))
    with pytest.raises(ValueError, match="Nonfinite"):
        preparation._slice_digest(path, 0, 3, 2)


def test_finalization_validates_cross_year_batches(source, tmp_path, monkeypatch):
    root, checkpoint, task = source
    output = tmp_path / "prepared"
    # Exercise publication on three tiny synthetic years without weakening the
    # production requirement (the separate test above checks the real YEARS).
    monkeypatch.setattr(preparation, "YEARS", (1982, 1983, 1984))
    for year in preparation.YEARS:
        preparation.prepare_year(root, output, year, checkpoint)
    ready = preparation.finalize_prepared(root, output, preparation.YEARS, checkpoint)
    assert ready["time_steps"] == (365 + 365 + 366) * 4
    assert not (output / ".incomplete").exists()
    store = prepared_array.PreparedArrayStore(output, bounded_reads=True)
    inputs, truth, _ = store.build_batch_from_indices(indices=[365 * 4], input_steps=2, target_steps=1,
                                                      task_cfg=task, dt=pd.Timedelta(hours=6))
    assert inputs.sizes["time"] == 2 and truth.sizes["time"] == 1
    before = {path: path.stat().st_mtime_ns for path in output.rglob("*.npy")}
    assert preparation.finalize_prepared(root, output, preparation.YEARS, checkpoint) == ready
    assert before == {path: path.stat().st_mtime_ns for path in output.rglob("*.npy")}


def test_owned_reads_release_every_mapping_and_cross_shards(tmp_path, monkeypatch):
    original_load = np.load
    mappings = []

    def tracked(*args, **kwargs):
        result = original_load(*args, **kwargs)
        if isinstance(result, np.memmap):
            mappings.append(result._mmap)
        return result

    monkeypatch.setattr(np, "load", tracked)
    shards = []
    expected = np.arange(24, dtype=np.float32).reshape(8, 3)
    for index in range(2):
        path = tmp_path / f"{index}.npy"
        np.save(path, expected[index * 4:(index + 1) * 4])
        shard = prepared_array._PreparedArrayShard(index * 4, (index + 1) * 4, prepared_array.PreparedFileArray(path))
        shards.append(shard)
    data = prepared_array.PreparedShardedArray(shards, expected.shape, expected.dtype)
    selected = data.take(np.asarray([[3, 4], [7, 0]]))
    np.testing.assert_array_equal(selected, expected[[[3, 4], [7, 0]]])
    assert mappings and all(mapping.closed for mapping in mappings)
    selected[0, 0, 0] = -10
    np.testing.assert_array_equal(data.take(np.asarray([3])), expected[[3]])
    assert all(mapping.closed for mapping in mappings)
    with pytest.raises(IndexError):
        shards[0].data.take(np.asarray([100]))
    assert all(mapping.closed for mapping in mappings)


def test_bounded_store_environment_and_split_preserve_opt_in(tmp_path, monkeypatch):
    root = tmp_path / "v1"
    (root / "coords").mkdir(parents=True)
    (root / "vars").mkdir()
    np.save(root / "coords/time.npy", pd.date_range("1982-01-01", periods=8, freq="6h").values)
    np.save(root / "vars/temperature.npy", np.arange(8, dtype=np.float32))
    (root / "metadata.json").write_text(json.dumps({"prepared_array_format_version": 1,
        "variables": {"temperature": {"dims": ["time"], "shape": [8], "dtype": "float32"}}}))
    monkeypatch.setenv("WEATHER_PREPARED_BOUNDED_READS", "1")
    store = prepared_array.PreparedArrayStore(root)
    assert isinstance(store.data_vars["temperature"].data, prepared_array.PreparedFileArray)
    assert store.split_by_time_indices(np.asarray([1, 2])).bounded_reads
    default = prepared_array.PreparedArrayStore(root, bounded_reads=False)
    assert isinstance(default.data_vars["temperature"].data, np.memmap)
