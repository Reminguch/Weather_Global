from __future__ import annotations

import time
from pathlib import Path
import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import bootstrap as _bootstrap  # noqa: F401
from .config import GRAPHCAST_VARS, RunConfig
from .model import data_utils, gc
from .prepared_array import PreparedArrayStore, is_prepared_array_store
from src.data_operations.loaders.graphcast_dataset import open_graphcast_era5


PREPARED_FORMAT_VERSION = 1


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


def resolution_tag(resolution: float) -> str:
    value = float(resolution)
    if np.isclose(value, round(value), atol=1e-6):
        return f"res{int(round(value))}"
    return f"res{str(value).replace('.', 'p')}"


def prepared_store_path(cfg: RunConfig) -> Path:
    return Path(cfg.prepared_data_root) / resolution_tag(cfg.resolution)


def _training_cache_decision(
    train_ds: xr.Dataset,
    cfg: RunConfig,
    task_cfg: gc.TaskConfig,
) -> tuple[bool, float]:
    if is_prepared_array_store(train_ds):
        task_vars = set(task_cfg.input_variables) | set(task_cfg.target_variables) | set(task_cfg.forcing_variables)
        estimate_gib = train_ds.estimate_task_nbytes(task_vars) / (1024**3)
        return False, estimate_gib
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
    if is_prepared_array_store(train_ds):
        print("[cache] prepared_array uses memmap streaming; full train RAM cache disabled.")
        return train_ds, eval_ds
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


def select_resolution(ds: xr.Dataset, resolution: float) -> tuple[xr.Dataset, float, int]:
    base_res = _infer_base_resolution_deg(ds)
    ratio = resolution / base_res
    stride = int(round(ratio))
    if not np.isclose(ratio, stride, atol=1e-6):
        raise ValueError(
            f"resolution={resolution} is not an integer multiple of base grid {base_res}deg."
        )
    if stride <= 0:
        raise ValueError(f"Invalid resolution stride: {stride}")

    lat_divides_180 = np.isclose(np.mod(180.0, resolution), 0.0, atol=1e-6)
    if lat_divides_180:
        ds = ds.isel(lat=slice(0, None, stride))
    else:
        n_steps = int(180.0 // resolution)
        x = (180.0 - n_steps * resolution) / 2.0
        lat_targets = _build_no_pole_latitudes(resolution)
        print(
            "Using no-pole latitude grid because 180 is not divisible by resolution: "
            f"resolution={resolution}, x={x}, lat_count={lat_targets.size}"
        )
        ds = ds.sel(lat=lat_targets, method="nearest").sortby("lat", ascending=False)

    ds = ds.isel(lon=slice(0, None, stride))

    for name in list(ds.data_vars):
        if ds[name].dtype.kind == "f" and ds[name].dtype != np.float32:
            ds[name] = ds[name].astype(np.float32)
    return ds, base_res, stride


def _split_by_year(ds: xr.Dataset, cfg: RunConfig, *, label: str) -> tuple[xr.Dataset, xr.Dataset, list[int]]:
    if "time" not in ds.coords:
        raise KeyError(f"Expected `time` coordinate in {label} dataset.")

    time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    years = sorted(set(time_index.year.astype(int).tolist()))
    if cfg.val_year not in years:
        raise ValueError(
            f"Requested val year {cfg.val_year} not present in {label} dataset years: {years}"
        )

    train_years = [y for y in years if y != cfg.val_year]
    if cfg.train_start_year is not None:
        train_years = [y for y in train_years if cfg.train_start_year <= y <= cfg.train_end_year]

    if not train_years:
        raise ValueError(f"No train years left in {label} data after year selection.")

    train_mask = np.isin(time_index.year, np.asarray(train_years))
    val_mask = time_index.year == cfg.val_year
    train_ds = ds.isel(time=np.where(train_mask)[0])
    val_ds = ds.isel(time=np.where(val_mask)[0])

    if train_ds.sizes.get("time", 0) == 0:
        raise ValueError("Empty train split after year selection.")
    if val_ds.sizes.get("time", 0) == 0:
        raise ValueError("Empty validation split after year selection.")

    train_times = pd.DatetimeIndex(pd.to_datetime(train_ds.time.values))
    val_times = pd.DatetimeIndex(pd.to_datetime(val_ds.time.values))
    if train_times.intersection(val_times).size > 0:
        raise ValueError("Train/validation overlap detected in year split.")
    return train_ds, val_ds, train_years


def _split_prepared_array_by_year(
    store: PreparedArrayStore,
    cfg: RunConfig,
    *,
    label: str,
) -> tuple[PreparedArrayStore, PreparedArrayStore, list[int]]:
    time_index = pd.DatetimeIndex(pd.to_datetime(store.time.values))
    years = sorted(set(time_index.year.astype(int).tolist()))
    if cfg.val_year not in years:
        raise ValueError(f"Requested val year {cfg.val_year} not present in {label} dataset years: {years}")

    train_years = [y for y in years if y != cfg.val_year]
    if cfg.train_start_year is not None:
        train_years = [y for y in train_years if cfg.train_start_year <= y <= cfg.train_end_year]
    if not train_years:
        raise ValueError(f"No train years left in {label} data after year selection.")

    train_idx = np.where(np.isin(time_index.year, np.asarray(train_years)))[0]
    val_idx = np.where(time_index.year == cfg.val_year)[0]
    if train_idx.size == 0:
        raise ValueError("Empty train split after year selection.")
    if val_idx.size == 0:
        raise ValueError("Empty validation split after year selection.")
    return (
        store.split_by_time_indices(train_idx, label=f"{label}-train"),
        store.split_by_time_indices(val_idx, label=f"{label}-eval"),
        train_years,
    )


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

    ds, base_res, stride = select_resolution(ds, cfg.resolution)
    train_raw, val_raw, train_years = _split_by_year(ds, cfg, label="local")

    print(
        "Data split: "
        f"train_years={train_years[0]}-{train_years[-1]} (excluding {cfg.val_year}), "
        f"val_year={cfg.val_year}, train_time={train_raw.sizes['time']}, val_time={val_raw.sizes['time']}, "
        f"base_res={base_res}, target_res={cfg.resolution}, stride={stride}"
    )
    return train_raw, val_raw


def _task_var_names(task_cfg: gc.TaskConfig) -> set[str]:
    return set(task_cfg.input_variables) | set(task_cfg.target_variables) | set(task_cfg.forcing_variables)


def _attr_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _validate_six_hour_sorted_time(ds: xr.Dataset) -> None:
    if "time" not in ds.coords:
        raise KeyError("Prepared store is missing `time` coordinate.")
    time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    if len(time_index) < 2:
        raise ValueError("Prepared store must contain at least two time steps.")
    if not time_index.is_monotonic_increasing:
        raise ValueError("Prepared store time coordinate must be sorted ascending.")
    deltas = np.diff(time_index.values).astype("timedelta64[ns]")
    expected = np.array(np.timedelta64(6, "h")).astype("timedelta64[ns]")
    if not np.all(deltas == expected):
        raise ValueError("Prepared store time grid must be strictly 6-hourly.")


def validate_prepared_dataset(ds: xr.Dataset, cfg: RunConfig, task_cfg: gc.TaskConfig) -> None:
    version = ds.attrs.get("prepared_format_version")
    if version is not None and int(version) != PREPARED_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported prepared_format_version={version}; expected {PREPARED_FORMAT_VERSION}."
        )
    resolution = ds.attrs.get("resolution")
    if resolution is not None and not np.isclose(float(resolution), cfg.resolution, atol=1e-6):
        raise ValueError(
            f"Prepared store resolution={resolution} does not match requested {cfg.resolution}."
        )

    prepared_levels = _attr_list(ds.attrs.get("pressure_levels"))
    if prepared_levels:
        prepared_levels_int = [int(level) for level in prepared_levels]
        requested_levels = [int(level) for level in task_cfg.pressure_levels]
        if prepared_levels_int != requested_levels:
            raise ValueError(
                "Prepared store pressure_levels do not match checkpoint task: "
                f"prepared={prepared_levels_int}, requested={requested_levels}."
            )

    missing = sorted(name for name in _task_var_names(task_cfg) if name not in ds.data_vars)
    if missing:
        raise ValueError(f"Prepared store is missing task variables: {missing}")
    attr_checks = [
        ("task_input_variables", list(task_cfg.input_variables)),
        ("task_target_variables", list(task_cfg.target_variables)),
        ("task_forcing_variables", list(task_cfg.forcing_variables)),
    ]
    for attr_name, expected in attr_checks:
        prepared_values = _attr_list(ds.attrs.get(attr_name))
        if prepared_values and list(prepared_values) != expected:
            raise ValueError(
                f"Prepared store {attr_name} does not match checkpoint task: "
                f"prepared={prepared_values}, requested={expected}."
            )

    _validate_six_hour_sorted_time(ds)
    _split_by_year(ds, cfg, label="prepared")


def _open_prepared_splits(cfg: RunConfig, task_cfg: gc.TaskConfig) -> tuple[xr.Dataset, xr.Dataset]:
    store_path = prepared_store_path(cfg)
    _assert_local_path(str(store_path), "--prepared-data-root")
    if not store_path.exists():
        raise FileNotFoundError(
            f"Prepared data store not found: {store_path}. "
            "Run python -m src.data_operations.preprocessing.prepare_graphcast_training_store first."
        )
    print(f"Opening prepared dataset: {store_path}")
    ds = xr.open_zarr(store_path, consolidated=True)
    ds = _ensure_datetime_coord(ds)
    validate_prepared_dataset(ds, cfg, task_cfg)
    train_ds, val_ds, train_years = _split_by_year(ds, cfg, label="prepared")
    print(
        "Prepared data split: "
        f"train_years={train_years[0]}-{train_years[-1]} (excluding {cfg.val_year}), "
        f"val_year={cfg.val_year}, train_time={train_ds.sizes['time']}, val_time={val_ds.sizes['time']}, "
        f"store={store_path}"
    )
    return train_ds, val_ds


def _open_prepared_array_splits(cfg: RunConfig, task_cfg: gc.TaskConfig) -> tuple[PreparedArrayStore, PreparedArrayStore]:
    store_path = prepared_store_path(cfg)
    _assert_local_path(str(store_path), "--prepared-data-root")
    if not store_path.exists():
        raise FileNotFoundError(
            f"Prepared array store not found: {store_path}. "
            "Run python -m src.data_operations.preprocessing.prepare_graphcast_streaming_store first."
        )
    print(f"Opening prepared array dataset: {store_path}")
    store = PreparedArrayStore(store_path, label="prepared-array")
    store.validate(resolution=cfg.resolution, task_cfg=task_cfg)
    train_ds, val_ds, train_years = _split_prepared_array_by_year(store, cfg, label="prepared_array")
    print(
        "Prepared array data split: "
        f"train_years={train_years[0]}-{train_years[-1]} (excluding {cfg.val_year}), "
        f"val_year={cfg.val_year}, train_time={train_ds.sizes['time']}, val_time={val_ds.sizes['time']}, "
        f"store={store_path}"
    )
    return train_ds, val_ds


def open_training_splits(cfg: RunConfig, task_cfg: gc.TaskConfig) -> tuple[xr.Dataset, xr.Dataset]:
    if cfg.data_source == "raw":
        train_ds, eval_ds = _open_local_splits(cfg)
        return prepare_dataset_for_task(train_ds, task_cfg), prepare_dataset_for_task(eval_ds, task_cfg)
    if cfg.data_source == "prepared":
        return _open_prepared_splits(cfg, task_cfg)
    if cfg.data_source == "prepared_array":
        return _open_prepared_array_splits(cfg, task_cfg)
    raise ValueError(f"Unknown data_source={cfg.data_source!r}; expected raw, prepared, or prepared_array.")
