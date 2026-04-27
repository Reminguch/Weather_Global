#!/usr/bin/env python3
"""NYC point-error metric extraction for one resolution + one mp.

Usage:
  python scripts/analyze_models/nyc_width_error_by_res_mp.py --res 4 --mp 1
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

import jax
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
    except Exception as exc:  # pragma: no cover
        raise ImportError("graphcast is required. Activate env via scripts/graphcast_env.sh.") from exc


_require_graphcast()
from graphcast import checkpoint, data_utils, graphcast, losses as gc_losses

from scripts.analyze_models.graphcast_analysis_utils import (
    build_run_jitted,
    suppress_graphcast_future_warnings,
)


# Fixed defaults to keep the script compact and easy to read.
DATASET_PATH = "data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc"
STATS_DIR = "data/graphcast/graphcast/stats"
OUTPUT_BASE_DIR = "plots/analyze_models"
OUTPUT_DATA_SUBDIR = "data"
LEAD_DAYS = [1, 2, 4]
N_EVAL_DAYS = 365
HOURS_PER_STEP = 6
N_INPUT_STEPS = 2
N_EXTRA_STEPS = 14
WINDOW_BATCH_SIZE = 8
# Target point near New Orleans: 30N, 90W (270E).
NYC_LAT = 30.0
NYC_LON = 270.0  # 0..360 convention
RES_GRID_STRIDE = 15

# GraphCast per-variable loss weights used in GraphCast.loss_and_predictions.
GRAPHCAST_PER_VARIABLE_WEIGHTS: dict[str, float] = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute NYC MAE metrics per width for one resolution and mp; writes CSV used by MAE-vs-res plots."
    )
    p.add_argument("--res", type=int, required=True, help="Resolution group (e.g., 1,2,4,9,12,15,18,30).")
    p.add_argument("--mp", type=int, required=True, help="mp value to filter runs (e.g., 1 or 2).")
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="If set, evaluate only this width. Useful for per-width SLURM array jobs to avoid "
             "repeated JIT recompilation across widths on the same GPU.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=WINDOW_BATCH_SIZE,
        help=f"Inference batch size (default: {WINDOW_BATCH_SIZE}). "
             "Increase for small/narrow models on large-memory GPUs.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Optional checkpoint root containing res{res}_* run directories. "
             "Defaults to artifacts/checkpoints/graphcast_res{res}_stream.",
    )
    p.add_argument(
        "--output-data-dir",
        type=Path,
        default=None,
        help="Optional output directory for CSVs. Defaults to plots/analyze_models/data.",
    )
    return p.parse_args()


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


def _prepare_dataset(max_lead_steps: int) -> tuple[xarray.Dataset, int, int]:
    n_steps_per_day = 24 // HOURS_PER_STEP
    n_target_times = N_EVAL_DAYS * n_steps_per_day
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


def _extract_width(run_name: str) -> int | None:
    m = re.search(r"_w(\d+)_", run_name)
    return int(m.group(1)) if m else None


def _extract_ckpt_step(p: Path) -> int:
    m = re.search(r"ckpt_step(\d+)$", p.stem)
    return int(m.group(1)) if m else -1


def _discover_best_by_width(res: int, mp: int, checkpoint_root: Path | None = None) -> list[tuple[int, Path]]:
    root = checkpoint_root if checkpoint_root is not None else ROOT / f"artifacts/checkpoints/graphcast_res{res}_stream"
    if not root.is_absolute():
        root = ROOT / root
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")

    best_by_width: dict[int, Path] = {}
    for run_dir in sorted(root.glob(f"res{res}_*")):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if f"_mp{mp}_" not in run_name:
            continue

        width = _extract_width(run_name)
        if width is None:
            print(f"[warn] skipping run without width token: {run_name}")
            continue

        ckpt_best = run_dir / "ckpt_best.npz"
        if not ckpt_best.exists():
            print(f"[warn] missing ckpt_best.npz: {run_dir}")
            continue

        prev = best_by_width.get(width)
        if prev is None or ckpt_best.stat().st_mtime > prev.stat().st_mtime:
            best_by_width[width] = ckpt_best

    rows = sorted(best_by_width.items(), key=lambda x: x[0])
    if not rows:
        raise RuntimeError(f"No checkpoints found for res={res}, mp={mp} under {root}")
    return rows


def _latitude_weights_with_fallback(data: xarray.DataArray) -> xarray.DataArray:
    """GraphCast latitude weights with a robust fallback for unusual grids."""
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
    """Normalized weighted MSE over variables.

    Normalization guarantees the output is:
    - per-step: caller passes one forecast step (`time` already sliced),
    - per-area-point: spatial dimensions are averaged (lat/lon, with optional lat weighting),
    - per-atmospheric-level: `level` is averaged,
    - per-variable: weighted mean over variables (dividing by total variable weight).

    When `diffs_stddev_by_level` is provided the squared error is divided by the
    per-variable (and per-level) variance before averaging, matching the scale of
    the training loss which is computed on normalized residuals.
    """
    per_var_losses: list[xarray.DataArray] = []
    per_var_weights: list[float] = []
    for name, target in targets.data_vars.items():
        if name not in predictions:
            continue
        prediction = predictions[name]
        loss = (prediction - target) ** 2
        # Divide by diffs_stddev² to match the training-loss normalization scale.
        if diffs_stddev_by_level is not None and name in diffs_stddev_by_level:
            scale = diffs_stddev_by_level[name].astype(loss.dtype)
            loss = loss / (scale ** 2)
        if use_latitude_weights and "lat" in loss.dims:
            # Latitude-weighted mean over latitude, then ordinary means elsewhere.
            lat_w = _latitude_weights_with_fallback(target).astype(loss.dtype)
            loss = loss.weighted(lat_w).mean("lat", skipna=False)
        if "lon" in loss.dims:
            loss = loss.mean("lon", skipna=False)
        if "level" in loss.dims:
            # Per-level normalization: average across atmospheric levels.
            loss = loss.mean("level", skipna=False)
        reduce_dims = [d for d in loss.dims if d not in ("batch",)]
        if reduce_dims:
            loss = loss.mean(reduce_dims, skipna=False)
        per_var_losses.append(loss)
        per_var_weights.append(float(per_variable_weights.get(name, 1.0)))

    if not per_var_losses:
        raise ValueError("No overlapping prediction/target variables found for weighted loss.")

    total_var_weight = float(np.sum(per_var_weights))
    if total_var_weight <= 0.0:
        raise ValueError("Total variable weight must be positive.")

    weighted_losses = [loss * w for loss, w in zip(per_var_losses, per_var_weights)]
    return xarray.concat(weighted_losses, dim="variable", join="exact").sum("variable", skipna=False) / total_var_weight


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

    abs_err_sum = np.zeros(max_lead_steps, dtype=float)
    counts = np.zeros(max_lead_steps, dtype=int)
    point_weighted_mse_sum = np.zeros(max_lead_steps, dtype=float)
    grid_weighted_mse_sum = np.zeros(max_lead_steps, dtype=float)
    weighted_mse_counts = np.zeros(max_lead_steps, dtype=int)
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
        pred_bt = np.asarray(pred_b["2m_temperature"].isel(lat=lat_idx, lon=lon_idx).transpose("batch", "time").values)
        n = min(max_lead_steps, pred_bt.shape[1], real_b.shape[1])
        if n == 0:
            continue
        err_bt = np.abs(pred_bt[:, :n] - real_b[:, :n])  # K diff == C diff
        abs_err_sum[:n] += np.sum(err_bt, axis=0)
        counts[:n] += err_bt.shape[0]

        for step_i in range(n):
            pred_step = pred_b.isel(time=step_i)
            target_step = targets_b.isel(time=step_i)

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
            point_weighted_mse_sum[step_i] += float(np.asarray(point_loss_batch.values).sum())
            grid_weighted_mse_sum[step_i] += float(np.asarray(grid_loss_batch.values).sum())
            weighted_mse_counts[step_i] += batch_count

    mae_curve = np.divide(abs_err_sum, counts, out=np.full(max_lead_steps, np.nan), where=counts > 0)
    point_weighted_mse_curve = np.divide(
        point_weighted_mse_sum,
        weighted_mse_counts,
        out=np.full(max_lead_steps, np.nan),
        where=weighted_mse_counts > 0,
    )
    grid_weighted_mse_curve = np.divide(
        grid_weighted_mse_sum,
        weighted_mse_counts,
        out=np.full(max_lead_steps, np.nan),
        where=weighted_mse_counts > 0,
    )
    mae_by_day: dict[int, float] = {}
    point_weighted_mse_by_day: dict[int, float] = {}
    grid_weighted_mse_by_day: dict[int, float] = {}
    n_by_day: dict[int, int] = {}
    for d, s in zip(lead_days, lead_steps):
        mae_by_day[d] = float(mae_curve[s - 1])
        point_weighted_mse_by_day[d] = float(point_weighted_mse_curve[s - 1])
        grid_weighted_mse_by_day[d] = float(grid_weighted_mse_curve[s - 1])
        n_by_day[d] = int(counts[s - 1])
    return mae_by_day, point_weighted_mse_by_day, grid_weighted_mse_by_day, n_by_day


def main() -> None:
    args = parse_args()
    lead_days = LEAD_DAYS
    lead_steps = [int((24 * d) // HOURS_PER_STEP) for d in lead_days]

    ds_nyc, start_target_idx, n_steps = _prepare_dataset(max(lead_steps))
    stats = _load_stats(ROOT / STATS_DIR)
    by_width = _discover_best_by_width(args.res, args.mp, args.checkpoint_root)

    if args.width is not None:
        by_width = [(w, p) for w, p in by_width if w == args.width]
        if not by_width:
            raise RuntimeError(f"No checkpoint found for res={args.res}, mp={args.mp}, width={args.width}")

    window_batch_size = args.batch_size

    rows: list[dict[str, object]] = []
    for i, (width, ckpt_path) in enumerate(by_width, start=1):
        run_name = ckpt_path.parent.name
        ckpt_step = int(_extract_ckpt_step(ckpt_path))
        mae_by_day, point_wmse_by_day, grid_wmse_by_day, n_by_day = _evaluate_checkpoint(
            ckpt_path, stats, ds_nyc, start_target_idx, n_steps, lead_days, lead_steps,
            seed_base=100000 * i, window_batch_size=window_batch_size,
        )
        print(
            f"[{i}/{len(by_width)}] {run_name} w={width} ckpt=ckpt_best.npz "
            + ", ".join(
                f"{d}d: mae_t2m={mae_by_day[d]:.3f}, point_wmse_norm={point_wmse_by_day[d]:.5f}, "
                f"grid15_wmse_norm={grid_wmse_by_day[d]:.5f}"
                for d in lead_days
            )
        )
        for d in lead_days:
            point_norm = float(point_wmse_by_day[d])
            grid_norm = float(grid_wmse_by_day[d])
            rows.append(
                {
                    "res": int(args.res),
                    "mp": int(args.mp),
                    "width": int(width),
                    "lead_days": int(d),
                    "mae_c": float(mae_by_day[d]),
                    # Backward-compatible legacy column names.
                    "point_weighted_mse_allvars": point_norm,
                    "grid15_weighted_mse_allvars": grid_norm,
                    # Preferred explicit names.
                    "point_weighted_mse_allvars_normalized": point_norm,
                    "grid15_weighted_mse_allvars_normalized": grid_norm,
                    "n_points": int(n_by_day[d]),
                    "ckpt_step": int(ckpt_step),
                    "run_name": run_name,
                    "ckpt_path": str(ckpt_path),
                }
            )

    df = pd.DataFrame(rows).sort_values(["res", "mp", "width", "lead_days"]).reset_index(drop=True)
    data_dir = args.output_data_dir if args.output_data_dir is not None else ROOT / OUTPUT_BASE_DIR / OUTPUT_DATA_SUBDIR
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.width is not None:
        csv_path = data_dir / f"nyc_width_error_res{args.res}_mp{args.mp}_w{args.width}.csv"
    else:
        csv_path = data_dir / f"nyc_width_error_res{args.res}_mp{args.mp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV: {csv_path}")
    print("Per-(mp,lead_days) MAE-vs-res plots are generated by:")
    print("  python scripts/analyze_models/plot_nyc_width_error_mae_vs_res.py")


if __name__ == "__main__":
    main()
