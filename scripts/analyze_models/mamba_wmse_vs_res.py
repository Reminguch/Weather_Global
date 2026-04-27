#!/usr/bin/env python3
"""Compute normalized grid-WMSE vs resolution for baseline vs mamba d=16 vs mamba d=32.

Compares:
- baseline: graphcast_res{res}_stream/res{res}_m4_w1024_mp2_h6_bs4/ckpt_best.npz
- mamba di16: graphcast_stream_frozen_residual_mamba/residual_mamba_int_res{res}_..._di16_.../ckpt_best.npz
- mamba di32: graphcast_stream_frozen_residual_mamba/residual_mamba_int_res{res}_..._di32_.../ckpt_best.npz

Outputs:
- CSV: plots/analyze_models/data/mamba_res/mamba_wmse_vs_res.csv
- PNGs: plots/analyze_models/images/mamba_res/mamba_grid15_wmse_vs_res_lead{N}d.png
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if GRAPHCAST_LOCAL.exists() and str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))

from src.data.graphcast_dataset import open_graphcast_era5


def _require_graphcast() -> None:
    try:
        import graphcast  # noqa: F401
    except Exception as exc:
        raise ImportError("graphcast is required. Activate env via scripts/graphcast_env.sh.") from exc


_require_graphcast()
from graphcast import checkpoint, data_utils, graphcast, losses as gc_losses

from scripts.analyze_models.graphcast_analysis_utils import (
    build_run_jitted,
    build_truth_anchored_residual_runner,
    suppress_graphcast_future_warnings,
)


DATASET_PATH = "data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc"
STATS_DIR = "data/graphcast/graphcast/stats"
MAMBA_CKPT_ROOT = "artifacts/checkpoints/graphcast_stream_frozen_residual_mamba"
OUTPUT_DATA_DIR = "plots/analyze_models/data/mamba_res"
OUTPUT_IMAGE_DIR = "plots/analyze_models/images/mamba_res"

LEAD_DAYS = [1, 2, 4]
N_EVAL_DAYS = 365
HOURS_PER_STEP = 6
N_INPUT_STEPS = 2
N_EXTRA_STEPS = 14
WINDOW_BATCH_SIZE = 8
WARMUP_STEPS = 24
TRUNK_STEPS = 32
NYC_LAT = 30.0
NYC_LON = 270.0
RES_GRID_STRIDE = 15

GRAPHCAST_PER_VARIABLE_WEIGHTS: dict[str, float] = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute normalized WMSE vs res for baseline vs mamba di16/di32.")
    p.add_argument("--n-eval-days", type=int, default=N_EVAL_DAYS)
    p.add_argument("--window-batch-size", type=int, default=WINDOW_BATCH_SIZE)
    p.add_argument("--lead-days", type=int, nargs="+", default=LEAD_DAYS)
    p.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit res list. Defaults to all resolutions with both baseline and mamba checkpoints.",
    )
    p.add_argument("--output-data-dir", type=Path, default=None)
    p.add_argument("--output-image-dir", type=Path, default=None)
    p.add_argument("--output-csv-name", type=str, default="mamba_wmse_vs_res.csv")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--trunk-steps", type=int, default=TRUNK_STEPS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers (copied from nyc_width_error_by_res_mp.py)
# ---------------------------------------------------------------------------

def _load_stats(stats_dir: Path) -> dict[str, xarray.Dataset]:
    return {
        "diffs_stddev_by_level": xarray.open_dataset(stats_dir / "diffs_stddev_by_level.nc").compute(),
        "mean_by_level": xarray.open_dataset(stats_dir / "mean_by_level.nc").compute(),
        "stddev_by_level": xarray.open_dataset(stats_dir / "stddev_by_level.nc").compute(),
    }


def _ensure_datetime_coord(ds: xarray.Dataset) -> xarray.Dataset:
    if np.issubdtype(ds["time"].dtype, np.integer) or ds["time"].dtype.kind in "iu":
        times = pd.to_datetime(ds["time"].values, unit="h", origin="1959-01-01")
        ds = ds.assign_coords(time=("time", times))
    if "batch" in ds.dims:
        dt = xarray.DataArray(ds["time"].values, dims=["time"]).expand_dims(batch=ds.sizes["batch"])
        ds = ds.assign_coords(datetime=dt)
    else:
        ds = ds.assign_coords(datetime=("time", ds["time"].values))
    return ds


def _prepare_dataset(max_lead_steps: int, n_eval_days: int) -> tuple[xarray.Dataset, int, int]:
    n_steps_per_day = 24 // HOURS_PER_STEP
    n_target_times = n_eval_days * n_steps_per_day
    n_context_steps = N_INPUT_STEPS + max_lead_steps
    n_total_steps = n_context_steps + n_target_times + N_EXTRA_STEPS

    ds = open_graphcast_era5(DATASET_PATH, time_slice=slice(-n_total_steps, None)).compute()
    if ds.sizes.get("lat") == 721 and ds.sizes.get("lon") == 1440:
        ds = ds.isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
    ds = _ensure_datetime_coord(ds)

    n_steps = ds.sizes["time"]
    if n_steps < n_context_steps + n_target_times:
        n_target_times = n_steps - n_context_steps
    if n_target_times <= 0:
        raise ValueError(f"Dataset has only {n_steps} steps; not enough context for lead={max_lead_steps}.")
    start_target_idx = n_steps - n_target_times
    return ds, start_target_idx, n_steps


# ---------------------------------------------------------------------------
# Metric helpers (copied from nyc_width_error_by_res_mp.py)
# ---------------------------------------------------------------------------

def _latitude_weights_with_fallback(data: xarray.DataArray) -> xarray.DataArray:
    try:
        return gc_losses.normalized_latitude_weights(data)
    except Exception:
        latitude = data.coords["lat"]
        lat_vals = np.asarray(latitude.values, dtype=np.float64)
        if lat_vals.ndim != 1 or lat_vals.size == 0:
            raise
        if lat_vals.size == 1:
            return xarray.DataArray(np.ones(1, dtype=np.float32), coords=latitude.coords, dims=latitude.dims)

        edges = np.empty(lat_vals.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (lat_vals[:-1] + lat_vals[1:])
        edges[0] = lat_vals[0] - (lat_vals[1] - lat_vals[0]) / 2.0
        edges[-1] = lat_vals[-1] + (lat_vals[-1] - lat_vals[-2]) / 2.0
        edges = np.clip(edges, -90.0, 90.0)
        weights_np = np.abs(np.sin(np.deg2rad(edges[:-1])) - np.sin(np.deg2rad(edges[1:])))
        weights = xarray.DataArray(weights_np, coords=latitude.coords, dims=latitude.dims).astype(np.float32)
        return weights / weights.mean(skipna=False)


def _normalized_weighted_mse_allvars(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    *,
    per_variable_weights: dict[str, float],
    use_latitude_weights: bool,
    diffs_stddev_by_level: xarray.Dataset | None = None,
) -> xarray.DataArray:
    per_var_losses: list[xarray.DataArray] = []
    per_var_weights: list[float] = []
    for name, target in targets.data_vars.items():
        if name not in predictions:
            continue
        prediction = predictions[name]
        loss = (prediction - target) ** 2
        if diffs_stddev_by_level is not None and name in diffs_stddev_by_level:
            scale = diffs_stddev_by_level[name].astype(loss.dtype)
            loss = loss / (scale ** 2)
        if use_latitude_weights and "lat" in loss.dims:
            lat_w = _latitude_weights_with_fallback(target).astype(loss.dtype)
            loss = loss.weighted(lat_w).mean("lat", skipna=False)
        if "lon" in loss.dims:
            loss = loss.mean("lon", skipna=False)
        if "level" in loss.dims:
            loss = loss.mean("level", skipna=False)
        reduce_dims = [d for d in loss.dims if d not in ("batch",)]
        if reduce_dims:
            loss = loss.mean(reduce_dims, skipna=False)
        per_var_losses.append(loss)
        per_var_weights.append(float(per_variable_weights.get(name, 1.0)))

    if not per_var_losses:
        raise ValueError("No overlapping prediction/target variables found.")

    total_var_weight = float(np.sum(per_var_weights))
    if total_var_weight <= 0.0:
        raise ValueError("Total variable weight must be positive.")

    weighted_losses = [loss * w for loss, w in zip(per_var_losses, per_var_weights)]
    return xarray.concat(weighted_losses, dim="variable", join="exact").sum("variable", skipna=False) / total_var_weight


# ---------------------------------------------------------------------------
# Model loading helpers (copied from nyc_width_error_by_res_mp.py)
# ---------------------------------------------------------------------------

def _build_batch(
    ds_res: xarray.Dataset,
    idx_batch: list[int],
    task_cfg,
    max_lead_steps: int,
    lat_idx: int,
    lon_idx: int,
):
    lead_slice = slice(f"{HOURS_PER_STEP}h", f"{max_lead_steps * HOURS_PER_STEP}h")
    inputs_list, targets_list, forcings_list, real_list = [], [], [], []
    for target_idx in idx_batch:
        window = ds_res.isel(time=slice(target_idx - (N_INPUT_STEPS + max_lead_steps) + 1, target_idx + 1))
        if window.sizes["time"] < N_INPUT_STEPS + max_lead_steps:
            continue

        in_i, tgt_i, forc_i = data_utils.extract_inputs_targets_forcings(
            window, target_lead_times=lead_slice, **dataclasses.asdict(task_cfg)
        )
        for name in getattr(graphcast, "STATIC_VARS", ("geopotential_at_surface", "land_sea_mask")):
            if name in in_i and "time" in in_i[name].dims:
                in_i = in_i.assign({name: in_i[name].isel(time=0, drop=True)})

        inputs_list.append(in_i.isel(batch=0, drop=True))
        targets_list.append(tgt_i.isel(batch=0, drop=True))
        forcings_list.append(forc_i.isel(batch=0, drop=True))
        real_list.append(
            np.asarray(
                ds_res["2m_temperature"].isel(
                    time=slice(target_idx - max_lead_steps + 1, target_idx + 1),
                    batch=0,
                    lat=lat_idx,
                    lon=lon_idx,
                ).values
            )
        )

    if not inputs_list:
        return None
    b = len(inputs_list)
    inputs_b = xarray.concat(inputs_list, dim="batch").assign_coords(batch=np.arange(b))
    targets_b = xarray.concat(targets_list, dim="batch").assign_coords(batch=np.arange(b))
    forcings_b = xarray.concat(forcings_list, dim="batch").assign_coords(batch=np.arange(b))
    real_b = np.stack(real_list, axis=0)
    return inputs_b, targets_b, forcings_b, real_b


def _build_warm_chunk_batch(
    ds_res: xarray.Dataset,
    chunk_start_indices: list[int],
    task_cfg,
    total_horizon_steps: int,
):
    lead_slice = slice(f"{HOURS_PER_STEP}h", f"{total_horizon_steps * HOURS_PER_STEP}h")
    inputs_list, targets_list, forcings_list = [], [], []
    for chunk_start in chunk_start_indices:
        window = ds_res.isel(time=slice(chunk_start, chunk_start + N_INPUT_STEPS + total_horizon_steps))
        if window.sizes["time"] < N_INPUT_STEPS + total_horizon_steps:
            continue

        in_i, tgt_i, forc_i = data_utils.extract_inputs_targets_forcings(
            window, target_lead_times=lead_slice, **dataclasses.asdict(task_cfg)
        )
        for name in getattr(graphcast, "STATIC_VARS", ("geopotential_at_surface", "land_sea_mask")):
            if name in in_i and "time" in in_i[name].dims:
                in_i = in_i.assign({name: in_i[name].isel(time=0, drop=True)})

        inputs_list.append(in_i.isel(batch=0, drop=True))
        targets_list.append(tgt_i.isel(batch=0, drop=True))
        forcings_list.append(forc_i.isel(batch=0, drop=True))

    if not inputs_list:
        return None
    b = len(inputs_list)
    inputs_b = xarray.concat(inputs_list, dim="batch").assign_coords(batch=np.arange(b))
    targets_b = xarray.concat(targets_list, dim="batch").assign_coords(batch=np.arange(b))
    forcings_b = xarray.concat(forcings_list, dim="batch").assign_coords(batch=np.arange(b))
    return inputs_b, targets_b, forcings_b


def _empty_metric_accumulator(max_lead_steps: int) -> dict[str, np.ndarray]:
    return {
        "abs_err_sum": np.zeros(max_lead_steps, dtype=float),
        "counts": np.zeros(max_lead_steps, dtype=int),
        "point_weighted_mse_sum": np.zeros(max_lead_steps, dtype=float),
        "grid_weighted_mse_sum": np.zeros(max_lead_steps, dtype=float),
        "weighted_mse_counts": np.zeros(max_lead_steps, dtype=int),
    }


def _accumulate_metrics(
    acc: dict[str, np.ndarray],
    pred_b: xarray.Dataset,
    target_b: xarray.Dataset,
    *,
    lat_idx: int,
    lon_idx: int,
    res_grid_lats: xarray.DataArray,
    res_grid_lons: xarray.DataArray,
    stats: dict[str, xarray.Dataset],
) -> None:
    n = pred_b.sizes.get("time", 0)
    if n == 0:
        return

    pred_bt = np.asarray(pred_b["2m_temperature"].isel(lat=lat_idx, lon=lon_idx).transpose("batch", "time").values)
    real_bt = np.asarray(target_b["2m_temperature"].isel(lat=lat_idx, lon=lon_idx).transpose("batch", "time").values)
    err_bt = np.abs(pred_bt[:, :n] - real_bt[:, :n])
    acc["abs_err_sum"][:n] += np.sum(err_bt, axis=0)
    acc["counts"][:n] += err_bt.shape[0]

    for step_i in range(n):
        pred_step = pred_b.isel(time=step_i)
        target_step = target_b.isel(time=step_i)

        point_pred = pred_step.isel(lat=lat_idx, lon=lon_idx)
        point_target = target_step.isel(lat=lat_idx, lon=lon_idx)
        point_loss_batch = _normalized_weighted_mse_allvars(
            point_pred,
            point_target,
            per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
            use_latitude_weights=False,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        )

        grid_pred = pred_step.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")
        grid_target = target_step.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")
        grid_loss_batch = _normalized_weighted_mse_allvars(
            grid_pred,
            grid_target,
            per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
            use_latitude_weights=True,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        )

        batch_count = int(point_loss_batch.sizes.get("batch", 1))
        acc["point_weighted_mse_sum"][step_i] += float(np.asarray(point_loss_batch.values).sum())
        acc["grid_weighted_mse_sum"][step_i] += float(np.asarray(grid_loss_batch.values).sum())
        acc["weighted_mse_counts"][step_i] += batch_count


def _finalize_metrics(
    acc: dict[str, np.ndarray],
    lead_days: list[int],
    lead_steps: list[int],
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, int]]:
    max_lead_steps = len(acc["abs_err_sum"])
    mae_curve = np.divide(acc["abs_err_sum"], acc["counts"], out=np.full(max_lead_steps, np.nan), where=acc["counts"] > 0)
    point_wmse_curve = np.divide(
        acc["point_weighted_mse_sum"],
        acc["weighted_mse_counts"],
        out=np.full(max_lead_steps, np.nan),
        where=acc["weighted_mse_counts"] > 0,
    )
    grid_wmse_curve = np.divide(
        acc["grid_weighted_mse_sum"],
        acc["weighted_mse_counts"],
        out=np.full(max_lead_steps, np.nan),
        where=acc["weighted_mse_counts"] > 0,
    )

    mae_by_day: dict[int, float] = {}
    point_wmse_by_day: dict[int, float] = {}
    grid_wmse_by_day: dict[int, float] = {}
    n_by_day: dict[int, int] = {}
    for d, s in zip(lead_days, lead_steps):
        mae_by_day[d] = float(mae_curve[s - 1])
        point_wmse_by_day[d] = float(point_wmse_curve[s - 1])
        grid_wmse_by_day[d] = float(grid_wmse_curve[s - 1])
        n_by_day[d] = int(acc["counts"][s - 1])
    return mae_by_day, point_wmse_by_day, grid_wmse_by_day, n_by_day


def _evaluate_checkpoint(
    ckpt_path: Path,
    stats: dict[str, xarray.Dataset],
    ds_nyc: xarray.Dataset,
    start_target_idx: int,
    n_steps: int,
    lead_days: list[int],
    lead_steps: list[int],
    seed_base: int,
    window_batch_size: int = WINDOW_BATCH_SIZE,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, int]]:
    max_lead_steps = max(lead_steps)
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    run_jitted, task_cfg, model_cfg, _run_cfg = build_run_jitted(ckpt_obj, stats, ckpt_path)

    stride = max(1, int(round(float(getattr(model_cfg, "resolution", 1.0)))))
    ds_res = ds_nyc.isel(lat=slice(0, None, stride), lon=slice(0, None, stride)) if stride > 1 else ds_nyc
    lat_idx = int(np.argmin(np.abs(ds_res["lat"].values - NYC_LAT)))
    lon_idx = int(np.argmin(np.abs(ds_res["lon"].values - NYC_LON)))
    res_grid_lats = xarray.DataArray(np.asarray(ds_nyc["lat"].values)[::RES_GRID_STRIDE], dims=["lat"])
    res_grid_lons = xarray.DataArray(np.asarray(ds_nyc["lon"].values)[::RES_GRID_STRIDE], dims=["lon"])

    min_target_idx = N_INPUT_STEPS + max_lead_steps - 1
    target_indices = np.arange(max(start_target_idx, min_target_idx), n_steps, dtype=int).tolist()
    n_batches = (len(target_indices) + window_batch_size - 1) // window_batch_size

    acc = _empty_metric_accumulator(max_lead_steps)

    for b_i in range(n_batches):
        i0 = b_i * window_batch_size
        i1 = min((b_i + 1) * window_batch_size, len(target_indices))
        batch = _build_batch(ds_res, target_indices[i0:i1], task_cfg, max_lead_steps, lat_idx, lon_idx)
        if batch is None:
            continue
        inputs_b, targets_b, forcings_b, real_b = batch
        with suppress_graphcast_future_warnings():
            pred_b = run_jitted(
                rng=jax.random.PRNGKey(seed_base + b_i),
                inputs=inputs_b,
                targets_template=targets_b * np.nan,
                forcings=forcings_b,
            )
        _accumulate_metrics(
            acc,
            pred_b.isel(time=slice(0, n)),
            targets_b.isel(time=slice(0, n)),
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            res_grid_lats=res_grid_lats,
            res_grid_lons=res_grid_lons,
            stats=stats,
        )

    return _finalize_metrics(acc, lead_days, lead_steps)


def _evaluate_checkpoint_truth_anchored(
    ckpt_path: Path,
    stats: dict[str, xarray.Dataset],
    ds_nyc: xarray.Dataset,
    start_target_idx: int,
    n_steps: int,
    lead_days: list[int],
    lead_steps: list[int],
    seed_base: int,
    *,
    warmup_steps: int,
    trunk_steps: int,
    window_batch_size: int = WINDOW_BATCH_SIZE,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, int]]:
    max_lead_steps = max(lead_steps)
    total_horizon_steps = warmup_steps + trunk_steps + max_lead_steps
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    runner = build_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path)
    task_cfg = runner["task_cfg"]
    model_cfg = runner["model_cfg"]

    stride = max(1, int(round(float(getattr(model_cfg, "resolution", 1.0)))))
    ds_res = ds_nyc.isel(lat=slice(0, None, stride), lon=slice(0, None, stride)) if stride > 1 else ds_nyc
    lat_idx = int(np.argmin(np.abs(ds_res["lat"].values - NYC_LAT)))
    lon_idx = int(np.argmin(np.abs(ds_res["lon"].values - NYC_LON)))
    res_grid_lats = xarray.DataArray(np.asarray(ds_nyc["lat"].values)[::RES_GRID_STRIDE], dims=["lat"])
    res_grid_lons = xarray.DataArray(np.asarray(ds_nyc["lon"].values)[::RES_GRID_STRIDE], dims=["lon"])

    min_chunk_start = max(0, start_target_idx - (N_INPUT_STEPS + warmup_steps))
    max_chunk_start = n_steps - (N_INPUT_STEPS + total_horizon_steps)
    if max_chunk_start < min_chunk_start:
        raise ValueError(
            f"Not enough timesteps for truth-anchored eval: need {N_INPUT_STEPS + total_horizon_steps}, have {n_steps}"
        )
    chunk_start_indices = np.arange(min_chunk_start, max_chunk_start + 1, trunk_steps, dtype=int).tolist()
    n_batches = (len(chunk_start_indices) + window_batch_size - 1) // window_batch_size

    acc = _empty_metric_accumulator(max_lead_steps)
    for b_i in range(n_batches):
        i0 = b_i * window_batch_size
        i1 = min((b_i + 1) * window_batch_size, len(chunk_start_indices))
        batch = _build_warm_chunk_batch(ds_res, chunk_start_indices[i0:i1], task_cfg, total_horizon_steps)
        if batch is None:
            continue
        inputs_b, targets_b, forcings_b = batch
        with suppress_graphcast_future_warnings():
            context = runner["initialize_context"](
                inputs=inputs_b,
                targets_template=targets_b * np.nan,
                forcings=forcings_b,
            )
            step_keys = jax.random.split(
                jax.random.PRNGKey(seed_base + b_i),
                2 * warmup_steps + trunk_steps * (2 + max_lead_steps * 2),
            )
            key_i = 0
            for step_i in range(warmup_steps):
                _truth_pred, context = runner["truth_step"](
                    rng=(step_keys[key_i], step_keys[key_i + 1]),
                    context=context,
                    target_step=targets_b.isel(time=slice(step_i, step_i + 1)),
                    forcings_step=forcings_b.isel(time=slice(step_i, step_i + 1)),
                )
                key_i += 2

            for anchor_i in range(trunk_steps):
                branch_start = warmup_steps + anchor_i
                branch_targets = targets_b.isel(time=slice(branch_start, branch_start + max_lead_steps))
                branch_forcings = forcings_b.isel(time=slice(branch_start, branch_start + max_lead_steps))
                branch_rng = step_keys[key_i]
                branch_pred = runner["branch_rollout"](
                    rng=branch_rng,
                    context=context,
                    targets_template=branch_targets * np.nan,
                    forcings=branch_forcings,
                )
                _accumulate_metrics(
                    acc,
                    branch_pred,
                    branch_targets,
                    lat_idx=lat_idx,
                    lon_idx=lon_idx,
                    res_grid_lats=res_grid_lats,
                    res_grid_lons=res_grid_lons,
                    stats=stats,
                )
                key_i += 1
                _truth_pred, context = runner["truth_step"](
                    rng=(step_keys[key_i], step_keys[key_i + 1]),
                    context=context,
                    target_step=targets_b.isel(time=slice(branch_start, branch_start + 1)),
                    forcings_step=forcings_b.isel(time=slice(branch_start, branch_start + 1)),
                )
                key_i += 2

    return _finalize_metrics(acc, lead_days, lead_steps)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _discover_mamba_checkpoints(resolutions: list[int]) -> list[dict]:
    root = ROOT / MAMBA_CKPT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Mamba checkpoint root not found: {root}")
    entries: list[dict] = []
    for ckpt in sorted(root.glob("residual_mamba_int_res*/ckpt_best.npz")):
        run_dir = ckpt.parent
        m_res = re.search(r"_res(\d+)_", run_dir.name)
        m_di = re.search(r"_di(\d+)_", run_dir.name)
        if not m_res or not m_di:
            print(f"[warn] cannot parse res/di from: {run_dir.name}")
            continue
        res = int(m_res.group(1))
        di = int(m_di.group(1))
        if res not in resolutions:
            continue
        entries.append({"model_type": "mamba", "res": res, "di": di, "ckpt_path": ckpt, "run_name": run_dir.name})
    return sorted(entries, key=lambda e: (e["di"], e["res"]))


def _load_run_config(ckpt_path: Path) -> dict:
    run_cfg_path = ckpt_path.parent / "run_config.json"
    if not run_cfg_path.exists():
        return {}
    with run_cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_baseline_checkpoints_from_mamba(
    mamba_entries: list[dict],
    resolutions: list[int],
) -> list[dict]:
    entries_by_res: dict[int, dict] = {}
    for entry in mamba_entries:
        ckpt_path = entry["ckpt_path"]
        run_cfg = _load_run_config(ckpt_path)
        baseline_ckpt_path = run_cfg.get("residual_training", {}).get("baseline_checkpoint")
        if not baseline_ckpt_path:
            print(f"[warn] missing baseline_checkpoint in {ckpt_path.parent / 'run_config.json'}")
            continue
        baseline_ckpt = Path(baseline_ckpt_path)
        if not baseline_ckpt.is_absolute():
            baseline_ckpt = ROOT / baseline_ckpt

        res = int(entry["res"])
        prev = entries_by_res.get(res)
        if prev is None:
            entries_by_res[res] = {
                "model_type": "baseline",
                "res": res,
                "di": None,
                "ckpt_path": baseline_ckpt,
                "run_name": baseline_ckpt.parent.name,
            }
            continue

        if prev["ckpt_path"] != baseline_ckpt:
            raise ValueError(
                f"Conflicting baseline checkpoints for res={res}: "
                f"{prev['ckpt_path']} vs {baseline_ckpt}"
            )

    missing_res = sorted(set(resolutions) - set(entries_by_res))
    for res in missing_res:
        print(f"[warn] no mamba-linked baseline checkpoint found for res={res}")
    return [entries_by_res[res] for res in sorted(entries_by_res)]


def _discover_all_available_resolutions() -> list[int]:
    """Return res values that exist in both baseline and mamba checkpoint sets."""
    baseline_res = {
        int(m.group(1))
        for root in ROOT.glob("artifacts/checkpoints/graphcast_res*_stream")
        if (m := re.match(r"graphcast_res(\d+)_stream", root.name))
    }
    mamba_res = {
        int(m.group(1))
        for run_dir in (ROOT / MAMBA_CKPT_ROOT).glob("residual_mamba_int_res*")
        if run_dir.is_dir() and (m := re.search(r"_res(\d+)_", run_dir.name))
    }
    return sorted(baseline_res & mamba_res)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_wmse_vs_res(
    df: pd.DataFrame,
    lead_day: int,
    metric_col: str,
    metric_label: str,
    out_path: Path,
    *,
    baseline_metric_col: str | None = None,
) -> None:
    sub = df[df["lead_days"] == lead_day].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    baseline_metric_col = baseline_metric_col or metric_col

    baseline = sub[sub["model_type"] == "baseline"].sort_values("res")
    if not baseline.empty:
        ax.plot(
            baseline["res"],
            baseline[baseline_metric_col],
            marker="o",
            color="steelblue",
            label="baseline (w1024)",
        )

    for di, color, label in [(16, "darkorange", "mamba di=16"), (32, "crimson", "mamba di=32")]:
        mamba = sub[(sub["model_type"] == "mamba") & (sub["di"] == di)].sort_values("res")
        if not mamba.empty:
            ax.plot(mamba["res"], mamba[metric_col], marker="s", color=color, label=label)

    ax.set_xlabel("Resolution group (res)")
    ax.set_ylabel(metric_label)
    ax.set_title(f"15-grid Weighted MSE (normalized) vs res | lead={lead_day}d")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved image: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    lead_days = args.lead_days
    lead_steps = [int((24 * d) // HOURS_PER_STEP) for d in lead_days]
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.trunk_steps <= 0:
        raise ValueError("--trunk-steps must be positive.")
    resolutions = args.resolutions if args.resolutions is not None else _discover_all_available_resolutions()
    if not resolutions:
        raise RuntimeError("No shared baseline/mamba resolutions found.")

    ds_nyc, start_target_idx, n_steps = _prepare_dataset(max(lead_steps), args.n_eval_days)
    stats = _load_stats(ROOT / STATS_DIR)

    mamba_models = _discover_mamba_checkpoints(resolutions)
    baseline_models = _discover_baseline_checkpoints_from_mamba(mamba_models, resolutions)
    models = baseline_models + mamba_models
    if not models:
        raise RuntimeError("No checkpoints found for any model group.")

    print(f"Evaluating {len(models)} checkpoints across {len(resolutions)} resolutions: {resolutions}")

    rows: list[dict] = []
    for i, entry in enumerate(models, start=1):
        ckpt_path: Path = entry["ckpt_path"]
        run_name: str = entry["run_name"]
        print(f"\n[{i}/{len(models)}] {run_name}")
        cold_mae_by_day, cold_point_wmse_by_day, cold_grid_wmse_by_day, n_by_day = _evaluate_checkpoint(
            ckpt_path, stats, ds_nyc, start_target_idx, n_steps,
            lead_days, lead_steps,
            seed_base=100000 * i,
            window_batch_size=args.window_batch_size,
        )
        if entry["model_type"] == "mamba":
            warm_mae_by_day, warm_point_wmse_by_day, warm_grid_wmse_by_day, warm_n_by_day = _evaluate_checkpoint_truth_anchored(
                ckpt_path,
                stats,
                ds_nyc,
                start_target_idx,
                n_steps,
                lead_days,
                lead_steps,
                seed_base=200000 * i,
                warmup_steps=args.warmup_steps,
                trunk_steps=args.trunk_steps,
                window_batch_size=args.window_batch_size,
            )
        else:
            warm_mae_by_day = {d: np.nan for d in lead_days}
            warm_point_wmse_by_day = {d: np.nan for d in lead_days}
            warm_grid_wmse_by_day = {d: np.nan for d in lead_days}
            warm_n_by_day = {d: 0 for d in lead_days}
        for d in lead_days:
            rows.append({
                "model_type": entry["model_type"],
                "di": entry["di"],
                "res": entry["res"],
                "lead_days": d,
                "mae_c": cold_mae_by_day[d],
                "point_weighted_mse_allvars_normalized": cold_point_wmse_by_day[d],
                "grid15_weighted_mse_allvars_normalized": cold_grid_wmse_by_day[d],
                "mae_c_cold": cold_mae_by_day[d],
                "point_weighted_mse_allvars_normalized_cold": cold_point_wmse_by_day[d],
                "grid15_weighted_mse_allvars_normalized_cold": cold_grid_wmse_by_day[d],
                "n_points": n_by_day[d],
                "n_points_cold": n_by_day[d],
                "mae_c_warm": warm_mae_by_day[d],
                "point_weighted_mse_allvars_normalized_warm": warm_point_wmse_by_day[d],
                "grid15_weighted_mse_allvars_normalized_warm": warm_grid_wmse_by_day[d],
                "n_points_warm": warm_n_by_day[d],
                "warmup_steps": args.warmup_steps if entry["model_type"] == "mamba" else np.nan,
                "trunk_steps": args.trunk_steps if entry["model_type"] == "mamba" else np.nan,
                "warm_eval_mode": "truth_anchored_branch" if entry["model_type"] == "mamba" else "cold_only_baseline",
                "run_name": run_name,
                "ckpt_path": str(ckpt_path),
            })
        print(
            "  " + ", ".join(
                f"{d}d cold={cold_grid_wmse_by_day[d]:.5f}"
                + (
                    f" warm={warm_grid_wmse_by_day[d]:.5f}"
                    if entry["model_type"] == "mamba"
                    else ""
                )
                for d in lead_days
            )
        )

    df = pd.DataFrame(rows).sort_values(["model_type", "di", "res", "lead_days"]).reset_index(drop=True)

    data_dir = args.output_data_dir if args.output_data_dir is not None else ROOT / OUTPUT_DATA_DIR
    image_dir = args.output_image_dir if args.output_image_dir is not None else ROOT / OUTPUT_IMAGE_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / args.output_csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    if not args.skip_plots:
        for d in lead_days:
            out_png = image_dir / f"mamba_grid15_wmse_vs_res_lead{d}d.png"
            _plot_wmse_vs_res(
                df, lead_day=d,
                metric_col="grid15_weighted_mse_allvars_normalized_cold",
                metric_label="15-grid Weighted MSE (all vars, normalized) [cold]",
                out_path=out_png,
            )
            warm_png = image_dir / f"mamba_grid15_wmse_vs_res_lead{d}d_warm.png"
            _plot_wmse_vs_res(
                df,
                lead_day=d,
                metric_col="grid15_weighted_mse_allvars_normalized_warm",
                baseline_metric_col="grid15_weighted_mse_allvars_normalized_cold",
                metric_label="15-grid Weighted MSE (all vars, normalized) [warm]",
                out_path=warm_png,
            )


if __name__ == "__main__":
    main()
