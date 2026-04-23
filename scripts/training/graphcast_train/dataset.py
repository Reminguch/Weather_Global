from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from . import bootstrap as _bootstrap  # noqa: F401
from .config import GRAPHCAST_VARS, RunConfig
from .model import data_utils, gc
from src.data.graphcast_dataset import open_graphcast_era5


def _ensure_datetime_coord(ds: xr.Dataset) -> xr.Dataset:
    def _with_datetime_coord(dataset: xr.Dataset, time_values: np.ndarray) -> xr.Dataset:
        if "batch" in dataset.dims:
            bt = np.broadcast_to(np.asarray(time_values)[None, :], (dataset.sizes["batch"], len(time_values)))
            return dataset.assign_coords(datetime=(("batch", "time"), bt))
        return dataset.assign_coords(datetime=("time", time_values))

    if np.issubdtype(ds.time.dtype, np.datetime64):
        if "datetime" not in ds.coords:
            ds = _with_datetime_coord(ds, ds.time.values)
        elif "batch" in ds.dims and ds.coords["datetime"].dims == ("time",):
            ds = _with_datetime_coord(ds, ds.time.values)
        return ds

    decoded = xr.decode_cf(ds)
    if "datetime" not in decoded.coords:
        decoded = _with_datetime_coord(decoded, decoded.time.values)
    elif "batch" in decoded.dims and decoded.coords["datetime"].dims == ("time",):
        decoded = _with_datetime_coord(decoded, decoded.time.values)
    return decoded


def prepare_dataset_for_task(ds: xr.Dataset, task_cfg: gc.TaskConfig) -> xr.Dataset:
    ds = _ensure_datetime_coord(ds)
    forcing_vars = set(task_cfg.forcing_variables)
    static_input_vars = (
        set(task_cfg.input_variables)
        - set(task_cfg.target_variables)
        - set(task_cfg.forcing_variables)
    )

    if forcing_vars & {"year_progress_sin", "year_progress_cos", "day_progress_sin", "day_progress_cos"}:
        data_utils.add_derived_vars(ds)
    if "toa_incident_solar_radiation" in forcing_vars and "toa_incident_solar_radiation" not in ds.data_vars:
        data_utils.add_tisr_var(ds)

    for name in sorted(static_input_vars):
        if name in ds.data_vars and "time" in ds[name].dims:
            ds[name] = ds[name].isel(time=0, drop=True)
    return ds


def _estimate_dataset_nbytes(ds: xr.Dataset, variable_names: Iterable[str]) -> int:
    total = 0
    for name in variable_names:
        if name not in ds.data_vars:
            continue
        var = ds[name]
        dtype = np.dtype(var.dtype)
        if dtype == object:
            continue
        total += int(var.size) * int(dtype.itemsize)
    return total


def _load_static_vars(ds: xr.Dataset, label: str) -> xr.Dataset:
    static_vars = [name for name, var in ds.data_vars.items() if "time" not in var.dims]
    if not static_vars:
        return ds
    print(f"[cache] loading static vars for {label}: {', '.join(static_vars)}")
    loaded = ds[static_vars].load()
    for name in static_vars:
        ds[name] = loaded[name]
    return ds


def _training_cache_decision(
    train_ds: xr.Dataset,
    cfg: RunConfig,
    task_cfg: gc.TaskConfig,
) -> tuple[bool, float]:
    if cfg.data_cache_mode == "never":
        return False, 0.0
    task_vars = set(task_cfg.input_variables) | set(task_cfg.target_variables) | set(task_cfg.forcing_variables)
    estimate_gib = _estimate_dataset_nbytes(train_ds, task_vars) / (1024**3)
    should_cache = cfg.data_cache_mode == "always" or estimate_gib <= cfg.data_cache_max_gib
    return should_cache, estimate_gib


def maybe_cache_training_data(
    train_ds: xr.Dataset,
    eval_ds: xr.Dataset,
    cfg: RunConfig,
    task_cfg: gc.TaskConfig,
) -> tuple[xr.Dataset, xr.Dataset]:
    train_ds = _load_static_vars(train_ds, "train")
    eval_ds = _load_static_vars(eval_ds, "eval")

    if cfg.data_cache_mode == "never":
        print("[cache] data_cache_mode=never; streaming time-varying train data.")
        return train_ds, eval_ds

    should_cache, estimate_gib = _training_cache_decision(train_ds, cfg, task_cfg)

    print(
        "[cache] train estimate "
        f"{estimate_gib:.2f} GiB for task vars; mode={cfg.data_cache_mode}, "
        f"max={cfg.data_cache_max_gib:.2f} GiB"
    )
    if not should_cache:
        print("[cache] train split exceeds cache cap; streaming time-varying train data.")
        return train_ds, eval_ds

    print("[cache] loading prepared train split into RAM.")
    t0 = time.time()
    train_ds = train_ds.load()
    print(f"[cache] train split loaded in {time.time() - t0:.1f}s.")
    return train_ds, eval_ds


def _assert_local_path(path: str, arg_name: str) -> None:
    if "://" in path:
        raise ValueError(f"{arg_name} must be a local path; remote URIs are disabled: {path}")


def _infer_base_resolution_deg(ds: xr.Dataset) -> float:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("Expected `lat` and `lon` coordinates in local dataset.")
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("Need at least two lat/lon coordinates to infer base resolution.")
    lat_d = np.abs(np.diff(lat))
    lon_d = np.abs(np.diff(lon))
    lat_d = lat_d[lat_d > 0]
    lon_d = lon_d[lon_d > 0]
    if lat_d.size == 0 or lon_d.size == 0:
        raise ValueError("Unable to infer base resolution from lat/lon coordinates.")
    lat_res = float(np.median(lat_d))
    lon_res = float(np.median(lon_d))
    if not np.isclose(lat_res, lon_res, atol=1e-6):
        raise ValueError(f"Lat/lon spacing mismatch: lat={lat_res}, lon={lon_res}")
    return lat_res


def _build_no_pole_latitudes(resolution: float) -> np.ndarray:
    """Build colatitude grid x, x+res, ..., 180-x and map to latitude 90-colat."""
    n_steps = int(180.0 // resolution)
    x = (180.0 - n_steps * resolution) / 2.0
    colat = x + np.arange(n_steps + 1, dtype=np.float64) * resolution
    return (90.0 - colat).astype(np.float32)


def _open_local_splits(cfg: RunConfig) -> tuple[xr.Dataset, xr.Dataset]:
    _assert_local_path(cfg.data_path, "--data-path")
    print(f"Opening local dataset: {cfg.data_path}")
    ds = open_graphcast_era5(cfg.data_path)
    ds = _ensure_datetime_coord(ds)

    if "time" not in ds.coords:
        raise KeyError("Expected `time` coordinate in local dataset.")

    available_vars = [name for name in GRAPHCAST_VARS if name in ds.data_vars]
    if not available_vars:
        raise ValueError("None of the required GraphCast variables were found in local dataset.")
    ds = ds[available_vars]

    base_res = _infer_base_resolution_deg(ds)
    ratio = cfg.resolution / base_res
    stride = int(round(ratio))
    if not np.isclose(ratio, stride, atol=1e-6):
        raise ValueError(
            f"resolution={cfg.resolution} is not an integer multiple of base grid {base_res}deg."
        )
    if stride <= 0:
        raise ValueError(f"Invalid resolution stride: {stride}")

    lat_divides_180 = np.isclose(np.mod(180.0, cfg.resolution), 0.0, atol=1e-6)
    if lat_divides_180:
        ds = ds.isel(lat=slice(0, None, stride))
    else:
        n_steps = int(180.0 // cfg.resolution)
        x = (180.0 - n_steps * cfg.resolution) / 2.0
        lat_targets = _build_no_pole_latitudes(cfg.resolution)
        print(
            "Using no-pole latitude grid because 180 is not divisible by resolution: "
            f"resolution={cfg.resolution}, x={x}, lat_count={lat_targets.size}"
        )
        # Match requested no-pole latitude layout while keeping monotonic descending order.
        ds = ds.sel(lat=lat_targets, method="nearest").sortby("lat", ascending=False)

    ds = ds.isel(lon=slice(0, None, stride))

    for name in list(ds.data_vars):
        if ds[name].dtype.kind == "f" and ds[name].dtype != np.float32:
            ds[name] = ds[name].astype(np.float32)

    time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    years = sorted(set(time_index.year.astype(int).tolist()))
    if cfg.val_year not in years:
        raise ValueError(
            f"Requested val year {cfg.val_year} not present in local dataset years: {years}"
        )

    train_years = [y for y in years if y != cfg.val_year]
    if cfg.train_start_year is not None:
        train_years = [y for y in train_years if cfg.train_start_year <= y <= cfg.train_end_year]

    if not train_years:
        raise ValueError("No train years left after excluding val year and applying train-year bounds.")

    train_mask = np.isin(time_index.year, np.asarray(train_years))
    val_mask = time_index.year == cfg.val_year
    train_raw = ds.isel(time=np.where(train_mask)[0])
    val_raw = ds.isel(time=np.where(val_mask)[0])

    if train_raw.sizes.get("time", 0) == 0:
        raise ValueError("Empty train split after year selection.")
    if val_raw.sizes.get("time", 0) == 0:
        raise ValueError("Empty validation split after year selection.")

    train_times = pd.DatetimeIndex(pd.to_datetime(train_raw.time.values))
    val_times = pd.DatetimeIndex(pd.to_datetime(val_raw.time.values))
    if train_times.intersection(val_times).size > 0:
        raise ValueError("Train/validation overlap detected in year split.")

    print(
        "Data split: "
        f"train_years={train_years[0]}-{train_years[-1]} (excluding {cfg.val_year}), "
        f"val_year={cfg.val_year}, train_time={train_raw.sizes['time']}, val_time={val_raw.sizes['time']}, "
        f"base_res={base_res}, target_res={cfg.resolution}, stride={stride}"
    )
    return train_raw, val_raw
