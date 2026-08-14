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

from src.data_operations.preprocessing.build_graphcast37_prepared_stream import (  # noqa: E402
    PreparedStoreV2Writer,
    _month_windows,
    _time_shards,
    remove_staged_if_verified,
)
from src.models.graphcast.training.core.prepared_array import PreparedArrayStore  # noqa: E402


def _task_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        input_variables=("temperature", "land_sea_mask"),
        target_variables=("temperature",),
        forcing_variables=("toa_incident_solar_radiation",),
        pressure_levels=(500, 850),
        input_duration="12h",
    )


def _dataset(times: pd.DatetimeIndex) -> xr.Dataset:
    level = np.asarray([500, 850], dtype=np.int64)
    lat = np.asarray([-10.0, 10.0], dtype=np.float32)
    lon = np.asarray([0.0, 120.0, 240.0], dtype=np.float32)
    temperature = np.arange(len(times) * len(level) * len(lat) * len(lon), dtype=np.float32).reshape(
        len(times), len(level), len(lat), len(lon)
    )
    forcing = np.arange(len(times) * len(lat) * len(lon), dtype=np.float32).reshape(
        len(times), len(lat), len(lon)
    )
    static = np.arange(len(lat) * len(lon), dtype=np.float32).reshape(len(lat), len(lon))
    return xr.Dataset(
        {
            "temperature": (("time", "level", "lat", "lon"), temperature),
            "toa_incident_solar_radiation": (("time", "lat", "lon"), forcing),
            "land_sea_mask": (("lat", "lon"), static),
        },
        coords={"time": times, "datetime": ("time", times), "level": level, "lat": lat, "lon": lon},
    )


def _writer(root: Path, times: pd.DatetimeIndex, *, resume: bool) -> PreparedStoreV2Writer:
    return PreparedStoreV2Writer(
        root=root,
        expected_times=times,
        task_cfg=_task_cfg(),
        checkpoint_path=Path("graphcast37.npz"),
        checkpoint_sha256="abc123",
        source_uri="synthetic.zarr",
        resume=resume,
    )


def test_writer_resumes_and_finalizes_cross_year_store(tmp_path: Path) -> None:
    times = pd.date_range("2021-12-31 00:00", "2022-01-01 18:00", freq="6h")
    windows = _month_windows(times, _time_shards(times))
    assert [window.key for window in windows] == ["2021-12", "2022-01"]

    root = tmp_path / "res0p25"
    writer = _writer(root, times, resume=True)
    first = windows[0]
    writer.write_window(_dataset(times[first.global_start : first.global_stop]), first, chunk_time=2)
    with pytest.raises(RuntimeError, match="incomplete"):
        PreparedArrayStore(root)

    resumed = _writer(root, times, resume=True)
    second = windows[1]
    resumed.write_window(_dataset(times[second.global_start : second.global_stop]), second, chunk_time=2)
    store = resumed.finalize()

    assert store.sizes["time"] == len(times)
    assert not (root / ".incomplete").exists()
    assert [shard["year"] for shard in store.metadata["time_shards"]] == [2021, 2022]


def test_writer_rewrite_of_completed_window_is_deterministic(tmp_path: Path) -> None:
    times = pd.date_range("2021-01-01 00:00", periods=4, freq="6h")
    window = _month_windows(times, _time_shards(times))[0]
    writer = _writer(tmp_path / "res0p25", times, resume=True)
    data = _dataset(times)

    writer.write_window(data, window, chunk_time=2)
    before = np.load(tmp_path / "res0p25" / "years" / "2021" / "vars" / "temperature.npy").copy()
    writer.write_window(data, window, chunk_time=1)
    after = np.load(tmp_path / "res0p25" / "years" / "2021" / "vars" / "temperature.npy")

    np.testing.assert_array_equal(before, after)


def test_writer_rejects_incompatible_resume(tmp_path: Path) -> None:
    times = pd.date_range("2021-01-01 00:00", periods=4, freq="6h")
    window = _month_windows(times, _time_shards(times))[0]
    root = tmp_path / "res0p25"
    writer = _writer(root, times, resume=True)
    writer.write_window(_dataset(times), window, chunk_time=2)

    with pytest.raises(ValueError, match="different checkpoint"):
        PreparedStoreV2Writer(
            root=root,
            expected_times=times,
            task_cfg=_task_cfg(),
            checkpoint_path=Path("other.npz"),
            checkpoint_sha256="different",
            source_uri="synthetic.zarr",
            resume=True,
        )


def test_remove_staged_requires_successful_verification(tmp_path: Path) -> None:
    staged = tmp_path / "2021-01.zarr"
    staged.mkdir()
    (staged / "sentinel").write_text("keep until verified")

    remove_staged_if_verified(staged, verified=False, delete_staged=True)
    assert staged.exists()

    remove_staged_if_verified(staged, verified=True, delete_staged=True)
    assert not staged.exists()


def test_month_windows_are_ordered_and_non_overlapping() -> None:
    times = pd.date_range("2021-11-01 00:00", "2022-02-28 18:00", freq="6h")
    windows = _month_windows(times, _time_shards(times))

    assert [window.key for window in windows] == ["2021-11", "2021-12", "2022-01", "2022-02"]
    assert all(left.global_stop == right.global_start for left, right in zip(windows, windows[1:]))
