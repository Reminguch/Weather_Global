from __future__ import annotations

import json
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
    _validate_manifest,
    remove_staged_if_verified,
)
from src.data_operations.staging.stage_wb2_graphcast37_window import (  # noqa: E402
    _validate_hourly_source_time,
)
from src.models.graphcast.training.core.prepared_array import (  # noqa: E402
    PreparedArrayStore,
    PreparedShardedArray,
)


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


def _writer(
    root: Path,
    times: pd.DatetimeIndex,
    *,
    resume: bool,
    extend_existing: bool = False,
) -> PreparedStoreV2Writer:
    return PreparedStoreV2Writer(
        root=root,
        expected_times=times,
        task_cfg=_task_cfg(),
        checkpoint_path=Path("graphcast37.npz"),
        checkpoint_sha256="abc123",
        source_uri="synthetic.zarr",
        resume=resume,
        extend_existing=extend_existing,
    )


def _write_month_statuses(
    root: Path,
    windows,
    statuses: dict[str, str],
) -> None:
    manifest = {
        "status": "incomplete",
        "months": {
            window.key: {
                "status": statuses[window.key],
                "start": str(window.start.to_datetime64()),
                "end": str(window.end.to_datetime64()),
                "time_count": window.count,
            }
            for window in windows
        },
    }
    (root / "build_manifest.json").write_text(json.dumps(manifest))


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


def test_completed_partial_range_uses_only_intersecting_shard_and_global_indices(
    tmp_path: Path,
) -> None:
    times = pd.date_range("2021-12-31 00:00", "2022-01-01 18:00", freq="6h")
    windows = _month_windows(times, _time_shards(times))
    root = tmp_path / "res0p25"
    writer = _writer(root, times, resume=True)
    first, second = windows
    writer.write_window(
        _dataset(times[first.global_start:first.global_stop]),
        first,
        chunk_time=2,
    )
    _write_month_statuses(
        root, windows, {first.key: "complete", second.key: "pending"}
    )

    partial = PreparedArrayStore(
        root,
        time_start=first.start,
        time_end=first.end,
        allow_incomplete=True,
    )
    assert partial.sizes["time"] == first.global_stop - first.global_start
    assert (root / ".incomplete").exists()
    temperature = partial.data_vars["temperature"].data
    assert isinstance(temperature, PreparedShardedArray)
    assert [(shard.start, shard.stop) for shard in temperature.shards] == [
        (first.global_start, first.global_stop)
    ]
    inputs, targets, _ = partial.build_batch_from_indices(
        indices=[1],
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    expected = _dataset(times[first.global_start:first.global_stop])
    np.testing.assert_array_equal(
        inputs["temperature"].values[0],
        expected["temperature"].values[:2],
    )
    np.testing.assert_array_equal(
        targets["temperature"].values[0],
        expected["temperature"].values[2:3],
    )

    with pytest.raises(RuntimeError, match="status='pending'"):
        PreparedArrayStore(
            root,
            time_start=second.start,
            time_end=second.end,
            allow_incomplete=True,
        )

    with pytest.raises(RuntimeError, match="status='pending'"):
        PreparedArrayStore(
            root,
            time_indices=np.arange(first.global_start, first.global_stop),
            time_start=first.start,
            time_end=second.end,
            allow_incomplete=True,
        )

    writer.write_window(
        _dataset(times).isel(
            time=slice(second.global_start, second.global_stop)
        ),
        second,
        chunk_time=2,
    )
    _write_month_statuses(
        root, windows, {first.key: "complete", second.key: "complete"}
    )
    second_partial = PreparedArrayStore(
        root,
        time_start=second.start,
        time_end=second.end,
        allow_incomplete=True,
    )
    second_temperature = second_partial.data_vars["temperature"].data
    assert isinstance(second_temperature, PreparedShardedArray)
    assert [(shard.start, shard.stop) for shard in second_temperature.shards] == [
        (second.global_start, second.global_stop)
    ]
    second_inputs, _, _ = second_partial.build_batch_from_indices(
        indices=[1],
        input_steps=2,
        target_steps=1,
        task_cfg=_task_cfg(),
        dt=pd.Timedelta("6h"),
    )
    np.testing.assert_array_equal(
        second_inputs["temperature"].values[0],
        _dataset(times)["temperature"].values[
            second.global_start:second.global_start + 2
        ],
    )


def test_incomplete_partial_range_rejects_staged_month(tmp_path: Path) -> None:
    times = pd.date_range("2021-12-31 00:00", "2022-01-01 18:00", freq="6h")
    windows = _month_windows(times, _time_shards(times))
    root = tmp_path / "res0p25"
    writer = _writer(root, times, resume=True)
    writer.write_window(_dataset(times[:4]), windows[0], chunk_time=2)
    _write_month_statuses(
        root, windows, {windows[0].key: "complete", windows[1].key: "staged"}
    )

    with pytest.raises(RuntimeError, match="status='staged'"):
        PreparedArrayStore(
            root,
            time_start=windows[1].start,
            time_end=windows[1].end,
            allow_incomplete=True,
        )


def test_incomplete_partial_range_rejects_missing_intersecting_shard(
    tmp_path: Path,
) -> None:
    times = pd.date_range("2021-01-01 00:00", periods=4, freq="6h")
    window = _month_windows(times, _time_shards(times))[0]
    root = tmp_path / "res0p25"
    writer = _writer(root, times, resume=True)
    writer.write_window(_dataset(times), window, chunk_time=2)
    _write_month_statuses(root, [window], {window.key: "complete"})
    (root / "years/2021/vars/temperature.npy").unlink()

    with pytest.raises(FileNotFoundError, match="requires missing shard"):
        PreparedArrayStore(
            root,
            time_start=window.start,
            time_end=window.end,
            allow_incomplete=True,
        )


def test_writer_prepends_whole_years_without_rewriting_existing_shards(tmp_path: Path) -> None:
    old_times = pd.date_range("2021-01-01 00:00", "2022-12-31 18:00", freq="6h")
    root = tmp_path / "res0p25"
    writer = _writer(root, old_times, resume=True)
    for window in _month_windows(old_times, _time_shards(old_times)):
        writer.write_window(
            _dataset(old_times[window.global_start : window.global_stop]),
            window,
            chunk_time=8,
        )
    writer.finalize()

    preserved_path = root / "years/2021/vars/temperature.npy"
    preserved_bytes = preserved_path.read_bytes()
    extended_times = pd.date_range("2019-01-01 00:00", "2022-12-31 18:00", freq="6h")

    with pytest.raises(ValueError, match="extend-existing"):
        _writer(root, extended_times, resume=True)
    assert not (root / ".incomplete").exists()

    extended = _writer(root, extended_times, resume=True, extend_existing=True)
    assert (root / ".incomplete").exists()
    assert [shard["year"] for shard in extended.metadata["time_shards"]] == [
        2019,
        2020,
        2021,
        2022,
    ]
    assert extended.metadata["time_shards"][2]["start"] == 2924
    assert extended.metadata["variables"]["temperature"]["shape"][0] == len(extended_times)

    for window in _month_windows(extended_times, _time_shards(extended_times)):
        if window.year >= 2021:
            continue
        extended.write_window(
            _dataset(extended_times[window.global_start : window.global_stop]),
            window,
            chunk_time=8,
        )
    store = extended.finalize()

    assert store.sizes["time"] == 5844
    assert store.metadata["time_shards"][1]["stop"] - store.metadata["time_shards"][1]["start"] == 1464
    assert preserved_path.read_bytes() == preserved_bytes
    assert not (root / ".incomplete").exists()


def test_manifest_extension_is_idempotent_and_preserves_completed_months() -> None:
    old_times = pd.date_range("2021-01-01 00:00", "2022-12-31 18:00", freq="6h")
    old_windows = _month_windows(old_times, _time_shards(old_times))
    manifest = {
        "status": "complete",
        "source_uri": "synthetic.zarr",
        "checkpoint_sha256": "abc123",
        "time_start": str(old_times[0].to_datetime64()),
        "time_end": str(old_times[-1].to_datetime64()),
        "expected_time_count": len(old_times),
        "months": {
            window.key: {
                "status": "complete",
                "start": str(window.start.to_datetime64()),
                "end": str(window.end.to_datetime64()),
                "time_count": window.count,
            }
            for window in old_windows
        },
    }
    requested = pd.date_range("2019-01-01 00:00", "2022-12-31 18:00", freq="6h")
    requested_windows = _month_windows(requested, _time_shards(requested))
    args = SimpleNamespace(source_uri="synthetic.zarr", extend_existing=True)

    changed = _validate_manifest(
        manifest,
        args=args,
        times=requested,
        sha256="abc123",
        windows=requested_windows,
        space_report={"minimum_free_tib": 3.0},
    )
    assert changed
    assert manifest["status"] == "incomplete"
    assert manifest["expected_time_count"] == 5844
    assert manifest["months"]["2019-01"]["status"] == "pending"
    assert manifest["months"]["2020-02"]["time_count"] == 116
    assert manifest["months"]["2021-01"]["status"] == "complete"
    assert len(manifest["extensions"]) == 1

    changed_again = _validate_manifest(
        manifest,
        args=args,
        times=requested,
        sha256="abc123",
        windows=requested_windows,
        space_report={"minimum_free_tib": 3.0},
    )
    assert not changed_again
    assert len(manifest["extensions"]) == 1


def test_hourly_precipitation_padding_rejects_missing_or_duplicate_hours() -> None:
    start = pd.Timestamp("2018-12-31 19:00")
    end = pd.Timestamp("2019-01-01 06:00")
    complete = pd.date_range(start, end, freq="1h")
    _validate_hourly_source_time(complete.values, start=start, end=end)

    missing = complete.delete(3)
    with pytest.raises(ValueError, match="incomplete or out of order"):
        _validate_hourly_source_time(missing.values, start=start, end=end)

    duplicated = complete.insert(4, complete[3])
    with pytest.raises(ValueError, match="duplicates=1"):
        _validate_hourly_source_time(duplicated.values, start=start, end=end)
