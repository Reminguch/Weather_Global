"""Scientific alignment and restart checks for the independent ERA5 test store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from src.data_operations.staging import stage_arco_era5_res1 as staging


LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
START = pd.Timestamp("2023-01-01 00:00")
END = pd.Timestamp("2023-01-01 18:00")


def _hourly_source() -> xr.Dataset:
    """Use a tiny 0.25-degree domain with distinguishable hours and grid cells."""
    times = pd.date_range(START - pd.Timedelta(hours=5), END, freq="h")
    levels = np.array(sorted([*LEVELS_13, 70, 975]), dtype=np.int32)
    latitude = np.arange(2.0, -0.01, -0.25, dtype=np.float32)
    longitude = np.arange(0.0, 3.0, 0.25, dtype=np.float32)
    time_signal = np.arange(len(times), dtype=np.float32)[:, None, None]
    spatial = (
        np.arange(len(latitude), dtype=np.float32)[:, None] * 100
        + np.arange(len(longitude), dtype=np.float32)[None, :]
    )
    surface = time_signal * 10000 + spatial[None, :, :]
    upper = surface[:, None, :, :] + levels[None, :, None, None] * 10
    data_vars = {
        name: (("time", "latitude", "longitude"), surface.copy())
        for name in (
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "toa_incident_solar_radiation",
        )
    }
    data_vars.update(
        {
            name: (("time", "level", "latitude", "longitude"), upper.copy())
            for name in (
                "temperature",
                "geopotential",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
                "specific_humidity",
            )
        }
    )
    data_vars["total_precipitation"] = (
        ("time", "latitude", "longitude"),
        np.broadcast_to((time_signal + 1) / 1000, surface.shape).copy(),
    )
    data_vars["geopotential_at_surface"] = (("latitude", "longitude"), spatial.copy())
    data_vars["land_sea_mask"] = (("latitude", "longitude"), (spatial % 2).copy())
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "level": levels, "latitude": latitude, "longitude": longitude},
    )
    ds["total_precipitation"].attrs["units"] = "m"
    return ds


def _write_source(tmp_path: Path, ds: xr.Dataset | None = None) -> Path:
    source = tmp_path / "source.zarr"
    dataset = _hourly_source() if ds is None else ds
    encoding = {
        name: {"chunks": (1, *array.shape[1:])}
        for name, array in dataset.data_vars.items()
        if array.dims[0] == "time"
    }
    encoding["time"] = {
        "units": "hours since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    dataset.to_zarr(source, mode="w", consolidated=True, zarr_format=2, encoding=encoding)
    return source


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture()
def small_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass only the production global-grid dimensions, retaining real I/O."""
    def validate_small_layout(source):
        levels = source.read_vector("level")
        return (
            source.read_vector("latitude")[::4],
            source.read_vector("longitude")[::4],
            np.asarray([np.flatnonzero(levels == level).item() for level in LEVELS_13]),
        )

    monkeypatch.setattr(staging.ChunkSource, "validate_layout", validate_small_layout)


def _stage_kwargs(tmp_path: Path, ds: xr.Dataset | None = None) -> dict:
    dataset = _hourly_source() if ds is None else ds
    source = _write_source(tmp_path, dataset)
    reference_path = tmp_path / "reference.zarr"
    reference = (
        dataset[list(staging.STATIC_VARIABLES)]
        .isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))
        .rename(latitude="lat", longitude="lon")
        .assign_coords(level=np.asarray(LEVELS_13, dtype=np.int32))
    )
    reference.to_zarr(reference_path, mode="w", consolidated=True, zarr_format=2)
    return {
        "uri": str(source),
        "output": tmp_path / "evaluation.zarr",
        "start_time": START.isoformat(),
        "end_time": END.isoformat(),
        "workers": 1,
        "reference_store": reference_path,
    }


def test_source_selects_exact_points_levels_and_five_hour_precipitation_padding(
    tmp_path: Path, small_grid: None
) -> None:
    dataset = _hourly_source()
    source = staging.ChunkSource(str(_write_source(tmp_path, dataset)))
    lat, lon, level_indices = source.validate_layout()
    times = staging.expected_time_grid(START.isoformat(), END.isoformat())
    hourly_indices = source.hourly_indices(times)

    np.testing.assert_array_equal(hourly_indices, [5, 11, 17, 23])
    np.testing.assert_array_equal(lat, [2, 1, 0])
    np.testing.assert_array_equal(lon, [0, 1, 2])
    np.testing.assert_array_equal(source.read_vector("level")[level_indices], LEVELS_13)
    expected = dataset.temperature.sel(time=START, level=LEVELS_13).isel(
        latitude=slice(None, None, 4), longitude=slice(None, None, 4)
    )
    np.testing.assert_array_equal(source.field("temperature", 5, level_indices), expected.values)

    totals = []
    for hour in hourly_indices:
        hourly = np.stack(
            [source.field("total_precipitation", int(index), level_indices) for index in range(hour - 5, hour + 1)]
        )
        totals.append(staging.sum_hourly_precipitation(hourly)[0, 0])
    np.testing.assert_allclose(totals, [0.021, 0.057, 0.093, 0.129], rtol=1e-6)


def test_pilot_remains_partial_and_resume_publishes_verified_complete_store(
    tmp_path: Path, small_grid: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = _stage_kwargs(tmp_path)
    output = kwargs["output"]
    partial = output.with_name(f".{output.name}.partial")
    progress_path = output.with_name(f"{output.name}.stage_progress.json")

    first = staging.stage_arco_res1(**kwargs, max_steps=1)
    assert first["status"] == "partial"
    assert first["completed_count"] == 1
    assert partial.is_dir()
    assert not output.exists()
    assert set(_read_json(progress_path)["completed"]) == {"0"}
    with pytest.raises(FileExistsError, match="resume"):
        staging.stage_arco_res1(**kwargs)

    # A resume may verify the completed local frame but must not fetch it again.
    original_field = staging.ChunkSource.field
    fetched_hours = []

    def tracked_field(self, name, hour_index, level_indices):
        fetched_hours.append(hour_index)
        return original_field(self, name, hour_index, level_indices)

    monkeypatch.setattr(staging.ChunkSource, "field", tracked_field)
    complete = staging.stage_arco_res1(**kwargs, resume=True)
    assert complete["status"] == "complete"
    assert complete["completed_count"] == 4
    assert min(fetched_hours) == 6
    assert output.is_dir()
    assert not partial.exists()
    assert _read_json(progress_path)["status"] == "complete"

    with xr.open_zarr(output, consolidated=True) as actual:
        np.testing.assert_array_equal(actual.time.values, pd.date_range(START, END, freq="6h").values)
        np.testing.assert_array_equal(actual.level.values, LEVELS_13)
        assert actual.temperature.shape == (4, 13, 3, 3)
        np.testing.assert_allclose(
            actual.total_precipitation_6hr[:, 0, 0], [0.021, 0.057, 0.093, 0.129], rtol=1e-6
        )
        assert actual.total_precipitation_6hr.attrs["units"] == "m"
        assert "toa_incident_solar_radiation" not in actual
    group = zarr.open_group(output, mode="r")
    assert group["temperature"].chunks == (1, 13, 3, 3)
    with pytest.raises(FileExistsError, match="Completed output"):
        staging.stage_arco_res1(**kwargs, resume=True)
    assert _read_json(progress_path) == complete


@pytest.mark.parametrize("bad_variable", ["temperature", "total_precipitation"])
def test_nonfinite_source_cannot_publish_or_mark_timestamp_complete(
    tmp_path: Path, small_grid: None, bad_variable: str
) -> None:
    dataset = _hourly_source()
    if bad_variable == "temperature":
        dataset[bad_variable].values[5, 0, 0, 0] = np.nan
    else:
        dataset[bad_variable].values[2, 0, 0] = np.nan
    kwargs = _stage_kwargs(tmp_path, dataset)
    with pytest.raises(RuntimeError, match="Staging failed") as error:
        staging.stage_arco_res1(**kwargs)
    assert isinstance(error.value.__cause__, ValueError)
    assert not kwargs["output"].exists()
    progress_path = kwargs["output"].with_name("evaluation.zarr.stage_progress.json")
    progress = _read_json(progress_path)
    assert progress["completed"] == {}
    assert progress["status"] == "partial"


def test_missing_source_chunk_cannot_be_interpreted_as_zero_rain(
    tmp_path: Path, small_grid: None
) -> None:
    kwargs = _stage_kwargs(tmp_path)
    (Path(kwargs["uri"]) / "total_precipitation" / "2.0.0").unlink()
    with pytest.raises(RuntimeError, match="Staging failed") as error:
        staging.stage_arco_res1(**kwargs)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("problem", ["missing_padding", "missing_hour", "beyond_end", "unfinalized"])
def test_source_coverage_is_checked_before_any_output_is_created(
    tmp_path: Path, small_grid: None, problem: str
) -> None:
    dataset = _hourly_source()
    if problem == "missing_padding":
        dataset = dataset.isel(time=slice(1, None))
    elif problem == "missing_hour":
        dataset = dataset.isel(time=[index for index in range(24) if index != 3])
    elif problem == "unfinalized":
        dataset.attrs["valid_time_stop"] = "2022-12-31"
    kwargs = _stage_kwargs(tmp_path, dataset)
    if problem == "beyond_end":
        kwargs["end_time"] = "2023-01-02T00:00:00"
    with pytest.raises(ValueError, match="coordinate range|hourly sequence|coverage"):
        staging.stage_arco_res1(**kwargs)
    assert not kwargs["output"].exists()
    assert not kwargs["output"].with_name(".evaluation.zarr.partial").exists()


def test_resume_rejects_a_corrupted_completed_frame(tmp_path: Path, small_grid: None) -> None:
    kwargs = _stage_kwargs(tmp_path)
    staging.stage_arco_res1(**kwargs, max_steps=1)
    partial = kwargs["output"].with_name(".evaluation.zarr.partial")
    group = zarr.open_group(partial, mode="r+")
    group["temperature"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        staging.stage_arco_res1(**kwargs, resume=True)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("values", [np.ones((5, 1)), np.array([1, 2, 3, np.nan, 5, 6])])
def test_precipitation_sum_requires_all_six_finite_hourly_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="six finite"):
        staging.sum_hourly_precipitation(values)
