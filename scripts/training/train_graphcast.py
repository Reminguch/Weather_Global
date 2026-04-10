#!/usr/bin/env python3
"""Train GraphCast on local ERA5 data at 2.0-degree resolution.

This script:
 - loads local ERA5 dataset
 - splits train/val by year (train: available years except val_year, val: val_year)
 - downsamples grid from base local resolution to target resolution (default 2.0deg)
 - trains one-step (+6h) objective with fixed max optimizer steps
 - logs losses, timings, GPU/CPU memory, checkpoints, and run metadata
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import xarray as xr

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
        raise ImportError(
            "graphcast is required. Activate env via scripts/graphcast_env.sh or install graphcast+jax+haiku."
        ) from exc


_require_graphcast()
import haiku as hk
import jax
import jax.numpy as jnp
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast as gc,
    losses as gc_losses,
    normalization,
    xarray_jax,
)

# Silence known upstream xarray FutureWarning emitted inside graphcast.autoregressive.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"graphcast\.autoregressive",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The return type of `Dataset\.dims` will be changed.*",
)


_ORIG_NORMALIZED_LATITUDE_WEIGHTS = gc_losses.normalized_latitude_weights


def _fallback_normalized_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    """Area weights for any uniformly spaced latitude vector (with or without poles)."""
    latitude = data.coords["lat"]
    lat_vals = np.asarray(latitude.values, dtype=np.float64)
    if lat_vals.ndim != 1 or lat_vals.size < 2:
        raise ValueError(f"Expected 1D latitude with at least 2 points; got shape={lat_vals.shape}")

    diffs = np.diff(lat_vals)
    if not np.all(np.isclose(diffs, diffs[0], atol=1e-6)):
        raise ValueError(f"Latitude vector is not uniformly spaced: {latitude}")
    delta = float(diffs[0])

    edges = np.empty(lat_vals.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lat_vals[:-1] + lat_vals[1:])
    edges[0] = lat_vals[0] - (delta / 2.0)
    edges[-1] = lat_vals[-1] + (delta / 2.0)
    edges = np.clip(edges, -90.0, 90.0)

    weights_np = np.abs(np.sin(np.deg2rad(edges[:-1])) - np.sin(np.deg2rad(edges[1:])))
    weights = xr.DataArray(weights_np, coords=latitude.coords, dims=latitude.dims).astype(np.float32)
    return weights / weights.mean(skipna=False)


def _normalized_latitude_weights_with_fallback(data: xr.DataArray) -> xr.DataArray:
    try:
        return _ORIG_NORMALIZED_LATITUDE_WEIGHTS(data)
    except ValueError as exc:
        if "does not start/end" not in str(exc):
            raise
        return _fallback_normalized_latitude_weights(data)


gc_losses.normalized_latitude_weights = _normalized_latitude_weights_with_fallback


DEFAULT_DATA_PATH = "data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
DEFAULT_CKPT = (
    "data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
DEFAULT_STATS_DIR = "data/graphcast/graphcast/stats"
DEFAULT_OUT_DIR = "artifacts/checkpoints/graphcast_res2_stream"

GRAPHCAST_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "geopotential_at_surface",
    "land_sea_mask",
    # Optional in source data.
    "toa_incident_solar_radiation",
]


@dataclasses.dataclass
class RunConfig:
    data_path: str
    resolution: float
    mesh_size: int
    width: int
    processor_msg_steps: int
    val_year: int
    train_start_year: int | None
    train_end_year: int | None
    ckpt_in: str
    stats_dir: str
    out_dir: str
    run_name: str
    batch_size: int
    max_steps: int
    eval_every: int
    eval_batch_size: int
    checkpoint_every: int
    lr: float
    weight_decay: float
    seed: int
    precision: str
    resume_step: int | None
    input_duration: str | None
    temporal_backbone: str
    temporal_location: str
    temporal_hidden_size: int
    temporal_layers: int
    temporal_dropout: float
    temporal_stateful: bool
    target_steps: int
    sequential_segment_steps: int | None
    eval_only: bool


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train GraphCast at 2.0deg from local ERA5 data.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Local dataset path (.zarr or .nc).")
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--width", type=int, choices=[128, 256, 512, 1024], default=128)
    parser.add_argument("--processor-msg-steps", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--val-year", type=int, default=2021, help="Validation year (excluded from train split).")
    parser.add_argument("--train-start-year", type=int, default=None, help="Optional lower bound for train years.")
    parser.add_argument("--train-end-year", type=int, default=None, help="Optional upper bound for train years.")
    parser.add_argument("--ckpt-in", default=DEFAULT_CKPT)
    parser.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="res2_m4_w128_mp1_h6_bs4")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume-step", type=int, default=None, help="Resume from this step (load params from --ckpt-in).")
    parser.add_argument(
        "--input-duration",
        default=None,
        help="Override task input duration (e.g. 12h/24h/36h/48h). Default: use checkpoint task config.",
    )
    parser.add_argument(
        "--temporal-backbone",
        choices=["none", "mamba"],
        default="none",
        help="Temporal module type. 'none' preserves existing GraphCast behavior.",
    )
    parser.add_argument(
        "--temporal-location",
        choices=["mesh_post_encoder", "mesh_processor_interleaved", "mesh_post_encoder_residual"],
        default="mesh_post_encoder",
        help="Where to insert temporal module when enabled.",
    )
    parser.add_argument("--temporal-hidden-size", type=int, default=128)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-stateful", action="store_true", default=False,
                        help="Use stateful Mamba (preserves SSM state across autoregressive steps).")
    parser.add_argument("--target-steps", type=int, default=1,
                        help="Number of autoregressive target steps (default 1 = 6h single step).")
    parser.add_argument("--sequential-segment-steps", type=int, default=None,
                        help="Enable chunked sequential sampling: segment length in time steps. "
                             "E.g. 120 = 30 days. Segments are shuffled across epochs, sequential within. "
                             "Mamba state carries across samples within a segment (truncated BPTT).")
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="Skip training, only run eval on the loaded checkpoint.")
    args = parser.parse_args()

    if not args.eval_only and args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.train_start_year is not None and args.train_end_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_end_year is not None and args.train_start_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_start_year is not None and args.train_start_year > args.train_end_year:
        raise ValueError("--train-start-year must be <= --train-end-year")
    if args.resume_step is not None and args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if args.temporal_hidden_size <= 0:
        raise ValueError("--temporal-hidden-size must be > 0")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be > 0")
    if not (0.0 <= args.temporal_dropout < 1.0):
        raise ValueError("--temporal-dropout must be in [0, 1)")

    return RunConfig(
        data_path=args.data_path,
        resolution=args.resolution,
        mesh_size=args.mesh_size,
        width=args.width,
        processor_msg_steps=args.processor_msg_steps,
        val_year=args.val_year,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        ckpt_in=args.ckpt_in,
        stats_dir=args.stats_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        precision=args.precision,
        resume_step=args.resume_step,
        input_duration=args.input_duration,
        temporal_backbone=args.temporal_backbone,
        temporal_location=args.temporal_location,
        temporal_hidden_size=args.temporal_hidden_size,
        temporal_layers=args.temporal_layers,
        temporal_dropout=args.temporal_dropout,
        temporal_stateful=args.temporal_stateful,
        target_steps=args.target_steps,
        sequential_segment_steps=args.sequential_segment_steps,
        eval_only=args.eval_only,
    )


def load_stats(stats_dir: Path) -> dict[str, xr.Dataset]:
    def open_nc(name: str) -> xr.Dataset:
        return xr.open_dataset(stats_dir / f"{name}.nc")

    return {
        "stddev_by_level": open_nc("stddev_by_level"),
        "mean_by_level": open_nc("mean_by_level"),
        "diffs_stddev_by_level": open_nc("diffs_stddev_by_level"),
    }


def load_graphcast_checkpoint(path: Path) -> gc.CheckPoint:
    with path.open("rb") as f:
        return checkpoint.load(f, gc.CheckPoint)


def build_predictor(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    stats: dict[str, xr.Dataset],
    *,
    use_bf16: bool,
    gradient_checkpointing: bool,
    temporal_backbone: str,
    temporal_location: str,
    temporal_hidden_size: int,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool = False,
):
    predictor = gc.GraphCast(model_cfg, task_cfg)
    if hasattr(predictor, "_temporal_backbone"):
        predictor._temporal_backbone = temporal_backbone
        predictor._temporal_location = temporal_location
        predictor._temporal_stateful = temporal_stateful
        predictor._temporal_hidden_size = temporal_hidden_size
        predictor._temporal_layers = temporal_layers
        predictor._temporal_dropout = temporal_dropout
    if use_bf16:
        predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=gradient_checkpointing)
    return predictor


def validate_stats_coverage(task_cfg: gc.TaskConfig, stats: dict[str, xr.Dataset]) -> None:
    required_inputs = set(task_cfg.input_variables) | set(task_cfg.forcing_variables)
    required_targets = set(task_cfg.target_variables)

    stddev_vars = set(stats["stddev_by_level"].data_vars)
    mean_vars = set(stats["mean_by_level"].data_vars)
    diffs_vars = set(stats["diffs_stddev_by_level"].data_vars)

    missing_stddev = sorted(required_inputs - stddev_vars)
    missing_mean = sorted(required_inputs - mean_vars)
    missing_diffs = sorted(required_targets - diffs_vars)

    if missing_stddev or missing_mean or missing_diffs:
        raise ValueError(
            "Normalization stats missing required variables: "
            f"stddev_missing={missing_stddev}, "
            f"mean_missing={missing_mean}, "
            f"diffs_stddev_missing={missing_diffs}"
        )


def _to_graphcast_layout(ds: xr.Dataset) -> xr.Dataset:
    """Add batch dimension and expand static 2D vars to (batch=1, time, lat, lon)."""
    out_vars: dict[str, xr.DataArray] = {}
    time_coord = ds.coords["time"]

    for name in list(ds.data_vars):
        var = ds[name]
        dims = list(var.dims)

        if "batch" in dims:
            out_vars[name] = var
            continue

        if "time" in dims:
            out_vars[name] = var.expand_dims(batch=[0]).transpose("batch", *dims)
        else:
            expanded = var.expand_dims(batch=[0], time=time_coord)
            out_vars[name] = expanded.transpose("batch", "time", "lat", "lon")

    coords = dict(ds.coords)
    if "batch" not in coords:
        coords["batch"] = [0]

    return xr.Dataset(data_vars=out_vars, coords=coords, attrs=ds.attrs)


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


def infer_time_step(ds: xr.Dataset) -> pd.Timedelta:
    if ds.sizes["time"] < 2:
        raise ValueError("Dataset must contain at least two time steps.")
    delta = pd.Timedelta(ds.time.values[1] - ds.time.values[0])
    if delta <= pd.Timedelta(0):
        raise ValueError(f"Invalid non-positive time step: {delta}.")
    return delta


def input_steps_from_duration(input_duration: str, dt: pd.Timedelta) -> int:
    duration = pd.Timedelta(input_duration)
    if duration % dt != pd.Timedelta(0):
        raise ValueError(f"input_duration={duration} is not divisible by dt={dt}.")
    steps = int(duration // dt)
    if steps < 1:
        raise ValueError(f"input_duration={duration} produced invalid input steps={steps}.")
    return steps


def lead_times(target_steps: int, dt: pd.Timedelta) -> Sequence[pd.Timedelta]:
    return [dt * (i + 1) for i in range(target_steps)]


def build_sequential_segments(
    indices: np.ndarray,
    segment_steps: int,
) -> list[np.ndarray]:
    """Split sorted indices into contiguous segments of approximately segment_steps.

    Each segment contains consecutive indices (sequential time windows).
    Indices that don't form contiguous runs are split at gaps.
    """
    if len(indices) == 0:
        return []
    sorted_idx = np.sort(indices)
    # Find gaps: where consecutive indices differ by more than 1
    gaps = np.where(np.diff(sorted_idx) > 1)[0] + 1
    # Split into contiguous runs
    runs = np.split(sorted_idx, gaps)
    # Further split long runs into chunks of segment_steps
    segments = []
    for run in runs:
        for i in range(0, len(run), segment_steps):
            chunk = run[i : i + segment_steps]
            if len(chunk) > 0:
                segments.append(chunk)
    return segments


def valid_final_input_indices(total_time_steps: int, input_steps: int, target_steps: int) -> np.ndarray:
    start = input_steps - 1
    stop = total_time_steps - target_steps
    if stop <= start:
        return np.array([], dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


def build_single_sample(
    ds: xr.Dataset,
    *,
    final_input_idx: int,
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    window_start = final_input_idx - input_steps + 1
    window_stop = final_input_idx + target_steps
    if window_start < 0 or window_stop >= ds.sizes["time"]:
        raise IndexError(
            f"Requested sample idx={final_input_idx} outside valid range for "
            f"input_steps={input_steps}, target_steps={target_steps}, total={ds.sizes['time']}."
        )

    window = ds.isel(time=slice(window_start, window_stop + 1))
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        window,
        input_variables=task_cfg.input_variables,
        target_variables=task_cfg.target_variables,
        forcing_variables=task_cfg.forcing_variables,
        pressure_levels=task_cfg.pressure_levels,
        input_duration=task_cfg.input_duration,
        target_lead_times=lead_times(target_steps, dt),
    )
    return inputs, targets, forcings


def build_batch_from_indices(
    ds: xr.Dataset,
    *,
    indices: Iterable[int],
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    inputs_list = []
    targets_list = []
    forcings_list = []

    for idx in indices:
        inputs, targets, forcings = build_single_sample(
            ds,
            final_input_idx=int(idx),
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )
        inputs_list.append(inputs.isel(batch=0, drop=True))
        targets_list.append(targets.isel(batch=0, drop=True))
        forcings_list.append(forcings.isel(batch=0, drop=True))

    batch_inputs = xr.concat(inputs_list, dim="batch").assign_coords(batch=np.arange(len(inputs_list)))
    batch_targets = xr.concat(targets_list, dim="batch").assign_coords(batch=np.arange(len(targets_list)))
    batch_forcings = xr.concat(forcings_list, dim="batch").assign_coords(batch=np.arange(len(forcings_list)))
    return batch_inputs.load(), batch_targets.load(), batch_forcings.load()


def scalarize_loss(loss_da: xr.DataArray) -> jax.Array:
    return jnp.mean(xarray_jax.unwrap_data(loss_da))


def run_eval(
    transformed,
    params: hk.Params,
    state: hk.State,
    rng: jax.Array,
    eval_ds: xr.Dataset,
    eval_indices: np.ndarray,
    *,
    eval_batch_size: int,
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
    progress_label: str = "eval",
    transformed_predict=None,
) -> dict[str, float]:
    # Reset SSM hidden states so eval starts from clean zeros
    state = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros_like(leaf) if isinstance(leaf, jax.Array) else leaf,
        state,
    )

    losses: list[float] = []
    # Per-variable accumulators for RMSE and MAE
    var_se_sums: dict[str, float] = {}   # sum of squared errors
    var_ae_sums: dict[str, float] = {}   # sum of absolute errors
    var_counts: dict[str, int] = {}      # number of elements
    compute_metrics = transformed_predict is not None

    n_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    t_eval0 = time.time()

    for batch_i, i in enumerate(range(0, len(eval_indices), eval_batch_size), start=1):
        idx = eval_indices[i : i + eval_batch_size]
        inputs, targets, forcings = build_batch_from_indices(
            eval_ds,
            indices=idx,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )

        rng, key = jax.random.split(rng)

        if compute_metrics:
            # Get loss from original transformed
            (loss_and_diag, _state_after) = transformed.apply(
                params, state, key, inputs, targets, forcings, False)
            loss = float(scalarize_loss(loss_and_diag[0]))
            # Get predictions from predict fn
            (preds, _pred_state) = transformed_predict.apply(
                params, state, key, inputs, targets, forcings, False)
            # Accumulate per-variable SE and AE in original (denormalized) space
            for var_name in preds.data_vars:
                if var_name not in targets.data_vars:
                    continue
                # Align dimensions: preds and targets may have different dim order
                # when target_steps > 1 (e.g. batch,time,lat,lon vs time,batch,lat,lon)
                common_dims = [d for d in targets[var_name].dims if d in preds[var_name].dims]
                pred_aligned = preds[var_name].transpose(*common_dims)
                tgt_aligned = targets[var_name].transpose(*common_dims)
                pred_vals = np.asarray(pred_aligned.values, dtype=np.float32)
                tgt_vals = np.asarray(tgt_aligned.values, dtype=np.float32)
                diff = pred_vals - tgt_vals
                n = diff.size
                var_se_sums[var_name] = var_se_sums.get(var_name, 0.0) + float(np.sum(diff ** 2))
                var_ae_sums[var_name] = var_ae_sums.get(var_name, 0.0) + float(np.sum(np.abs(diff)))
                var_counts[var_name] = var_counts.get(var_name, 0) + n
        else:
            (loss_and_diag, _state_after) = transformed.apply(
                params, state, key, inputs, targets, forcings, False)
            loss = float(scalarize_loss(loss_and_diag[0]))

        losses.append(loss)

        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_loss {loss:.6f}"
            )

    results: dict[str, float] = {"total": float(np.mean(losses))}

    if compute_metrics and var_counts:
        # Compute and print per-variable RMSE and MAE
        total_se, total_ae, total_n = 0.0, 0.0, 0
        print(f"[{progress_label}] === Per-variable metrics ===")
        for var_name in sorted(var_counts.keys()):
            n = var_counts[var_name]
            rmse = float(np.sqrt(var_se_sums[var_name] / n))
            mae = float(var_ae_sums[var_name] / n)
            results[f"{var_name}_RMSE"] = rmse
            results[f"{var_name}_MAE"] = mae
            print(f"[{progress_label}]   {var_name}: RMSE={rmse:.4f}  MAE={mae:.4f}")
            total_se += var_se_sums[var_name]
            total_ae += var_ae_sums[var_name]
            total_n += n
        overall_rmse = float(np.sqrt(total_se / total_n))
        overall_mae = float(total_ae / total_n)
        results["overall_RMSE"] = overall_rmse
        results["overall_MAE"] = overall_mae
        print(f"[{progress_label}]   OVERALL: RMSE={overall_rmse:.4f}  MAE={overall_mae:.4f}")

    return results


def save_checkpoint(
    out_dir: Path,
    *,
    params: hk.Params,
    step: int,
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    description: str,
    license_text: str,
) -> None:
    path = out_dir / f"ckpt_step{step}.npz"
    ckpt_out = gc.CheckPoint(
        params=params,
        model_config=model_cfg,
        task_config=task_cfg,
        description=description,
        license=license_text,
    )
    with path.open("wb") as f:
        checkpoint.dump(f, ckpt_out)
    print(f"saved checkpoint: {path}")


def save_logs(
    out_dir: Path,
    train_losses: list[tuple[int, float]],
    eval_losses: list[tuple[int, float]],
    eval_details: list[dict[str, Any]],
    step_times: list[tuple[int, float]],
    mem_usage: list[tuple[int, float]],
    actual_usage: list[dict[str, Any]],
    epoch_summaries: list[dict[str, Any]],
) -> None:
    with (out_dir / "train_loss.json").open("w", encoding="utf-8") as f:
        json.dump(train_losses, f)
    with (out_dir / "eval_loss.json").open("w", encoding="utf-8") as f:
        json.dump(eval_losses, f)
    with (out_dir / "eval_details.json").open("w", encoding="utf-8") as f:
        json.dump(eval_details, f, indent=2)
    with (out_dir / "step_times.json").open("w", encoding="utf-8") as f:
        json.dump(step_times, f)
    with (out_dir / "memory_gib.json").open("w", encoding="utf-8") as f:
        json.dump(mem_usage, f)
    with (out_dir / "actual_usage.json").open("w", encoding="utf-8") as f:
        json.dump(actual_usage, f, indent=2)

    rss_vals = [float(x["proc_rss_gib"]) for x in actual_usage if x.get("proc_rss_gib") is not None]
    gpu_vals = [float(x["gpu_mem_gib"]) for x in actual_usage if x.get("gpu_mem_gib") is not None]
    with (out_dir / "actual_usage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rss_gib_peak": float(np.max(rss_vals)) if rss_vals else None,
                "rss_gib_avg": float(np.mean(rss_vals)) if rss_vals else None,
                "gpu_mem_gib_peak": float(np.max(gpu_vals)) if gpu_vals else None,
                "gpu_mem_gib_avg": float(np.mean(gpu_vals)) if gpu_vals else None,
                "samples": len(actual_usage),
            },
            f,
            indent=2,
        )
    with (out_dir / "epoch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(epoch_summaries, f, indent=2)


def _load_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _load_step_value_pairs(path: Path) -> list[tuple[int, float]]:
    data = _load_json_list(path)
    out: list[tuple[int, float]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((int(item[0]), float(item[1])))
    return out


def _load_train_losses(path: Path) -> list[tuple[int, float]]:
    data = _load_json_list(path)
    if not data:
        return []
    first = data[0]
    if isinstance(first, (list, tuple)):
        out: list[tuple[int, float]] = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((int(item[0]), float(item[1])))
        return out
    return [(i + 1, float(loss)) for i, loss in enumerate(data)]


def _filter_pairs_upto_step(data: list[tuple[int, float]], max_step: int) -> list[tuple[int, float]]:
    return [(int(step), float(value)) for step, value in data if int(step) <= max_step]


def _load_dict_series_upto_step(path: Path, max_step: int) -> list[dict[str, Any]]:
    data = _load_json_list(path)
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if step is None:
            continue
        if int(step) <= max_step:
            out.append(item)
    return out


def _read_proc_mem_gib() -> tuple[float | None, float | None]:
    """Return (current_rss_gib, peak_hwm_gib) from /proc/self/status."""
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as f:
            lines = f.readlines()
        rss_kib: int | None = None
        hwm_kib: int | None = None
        for line in lines:
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kib = int(line.split()[1])
        rss_gib = float(rss_kib) / (1024**2) if rss_kib is not None else None
        hwm_gib = float(hwm_kib) / (1024**2) if hwm_kib is not None else None
        return rss_gib, hwm_gib
    except Exception:
        return None, None


def _read_gpu_mem_by_device() -> tuple[list[dict[str, float | int]], float | None]:
    """Return per-device GPU memory stats and total used GiB from nvidia-smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        devices: list[dict[str, float | int]] = []
        total_mib = 0.0
        for raw in proc.stdout.splitlines():
            row = raw.strip()
            if not row:
                continue
            parts = [x.strip() for x in row.split(",") if x.strip() != ""]
            if len(parts) < 3:
                continue
            index = int(parts[0])
            used_mib = float(parts[1])
            total_dev_mib = float(parts[2])
            total_mib += used_mib
            devices.append(
                {
                    "index": index,
                    "used_gib": used_mib / 1024.0,
                    "total_gib": total_dev_mib / 1024.0,
                }
            )
        return devices, (total_mib / 1024.0 if devices else None)
    except Exception:
        return [], None


def sample_actual_usage(step: int) -> dict[str, Any]:
    proc_rss_gib, proc_hwm_gib = _read_proc_mem_gib()
    gpu_devices, gpu_mem_gib = _read_gpu_mem_by_device()
    return {
        "step": step,
        "timestamp": time.time(),
        "proc_rss_gib": proc_rss_gib,
        "proc_hwm_gib": proc_hwm_gib,
        "gpu_mem_gib": gpu_mem_gib,
        "gpu_mem_total_gib": gpu_mem_gib,
        "gpu_devices": gpu_devices,
    }


def plot_loss_curves(
    out_dir: Path,
    train_losses: list[tuple[int, float]],
    eval_losses: list[tuple[int, float]],
) -> None:
    if not train_losses and not eval_losses:
        return

    plt.figure()
    y_vals: list[float] = []

    if train_losses:
        train_steps, train_vals = zip(*train_losses)
        plt.plot(train_steps, train_vals, label="train loss", alpha=0.6)
        y_vals.extend(train_vals)

    if eval_losses:
        eval_steps, eval_vals = zip(*eval_losses)
        plt.plot(eval_steps, eval_vals, marker="o", label="val loss")
        y_vals.extend(eval_vals)

    y_max = max(y_vals) * 2.0 if y_vals else 0.0
    if y_max > 0:
        plt.ylim(0.0, y_max)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Train and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss.png")
    plt.close()


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


def _write_run_config(out_dir: Path, cfg: RunConfig, model_cfg: gc.ModelConfig, task_cfg: gc.TaskConfig) -> None:
    payload = {
        "data_path": cfg.data_path,
        "val_year": cfg.val_year,
        "train_start_year": cfg.train_start_year,
        "train_end_year": cfg.train_end_year,
        "batch_size": cfg.batch_size,
        "max_steps": cfg.max_steps,
        "eval_every": cfg.eval_every,
        "eval_batch_size": cfg.eval_batch_size,
        "checkpoint_every": cfg.checkpoint_every,
        "seed": cfg.seed,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "precision": cfg.precision,
        "temporal_config": {
            "backbone": cfg.temporal_backbone,
            "location": cfg.temporal_location,
            "hidden_size": cfg.temporal_hidden_size,
            "layers": cfg.temporal_layers,
            "dropout": cfg.temporal_dropout,
        },
        "model_config": dataclasses.asdict(model_cfg),
        "task_config": dataclasses.asdict(task_cfg),
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_in = load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)

    model_cfg = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.processor_msg_steps,
        hidden_layers=1,
        mesh2grid_edge_normalization_factor=None,
    )

    norm_stats = load_stats(Path(cfg.stats_dir))
    validate_stats_coverage(task_cfg, norm_stats)

    train_ds, eval_ds = _open_local_splits(cfg)

    train_ds = prepare_dataset_for_task(train_ds, task_cfg)
    eval_ds = prepare_dataset_for_task(eval_ds, task_cfg)

    dt_train = infer_time_step(train_ds)
    dt_eval = infer_time_step(eval_ds)
    if dt_train != dt_eval:
        raise ValueError(f"Train/eval time step mismatch: train={dt_train}, eval={dt_eval}")

    input_steps = input_steps_from_duration(task_cfg.input_duration, dt_train)
    if cfg.temporal_backbone != "none" and input_steps < 2:
        raise ValueError("Temporal module requires at least 2 input steps.")
    if cfg.temporal_backbone != "none":
        print(
            "Temporal module configured "
            f"(backbone={cfg.temporal_backbone}, location={cfg.temporal_location}, "
            f"hidden={cfg.temporal_hidden_size}, layers={cfg.temporal_layers}, dropout={cfg.temporal_dropout})"
        )
    target_steps = cfg.target_steps

    train_final_indices = valid_final_input_indices(train_ds.sizes["time"], input_steps, target_steps)
    eval_final_indices = valid_final_input_indices(eval_ds.sizes["time"], input_steps, target_steps)
    if len(train_final_indices) == 0:
        raise ValueError("No train samples after applying input/target window requirements.")
    if len(eval_final_indices) == 0:
        raise ValueError("No eval samples after applying input/target window requirements.")

    print(
        "Prepared dataset windows: "
        f"train_time={train_ds.sizes['time']}, eval_time={eval_ds.sizes['time']}, "
        f"train_samples={len(train_final_indices)}, eval_samples={len(eval_final_indices)}, "
        f"input_steps={input_steps}, target_steps={target_steps}"
    )

    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=True,
            temporal_backbone=cfg.temporal_backbone,
            temporal_location=cfg.temporal_location,
            temporal_hidden_size=cfg.temporal_hidden_size,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=cfg.temporal_dropout,
            temporal_stateful=cfg.temporal_stateful,
        )
        return predictor.loss(inputs, targets, forcings)

    def predict_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone=cfg.temporal_backbone,
            temporal_location=cfg.temporal_location,
            temporal_hidden_size=cfg.temporal_hidden_size,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=cfg.temporal_dropout,
            temporal_stateful=cfg.temporal_stateful,
        )
        return predictor(inputs, targets_template=targets, forcings=forcings)

    transformed = hk.transform_with_state(forward_fn)
    transformed_predict = hk.transform_with_state(predict_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    sample_inputs, sample_targets, sample_forcings = build_batch_from_indices(
        train_ds,
        indices=[int(train_final_indices[0])],
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
    )

    params, state = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, True)
    if cfg.resume_step is not None:
        params = ckpt_in.params
        print(f"Resuming from step {cfg.resume_step} (params loaded from {cfg.ckpt_in})")
    elif cfg.eval_only:
        params = ckpt_in.params
        print(f"Eval-only: loaded params from {cfg.ckpt_in}")

    if cfg.eval_only:
        print("Eval-only mode: running eval on loaded checkpoint, no training.")
        eval_metrics = run_eval(
            transformed,
            params,
            state,
            rng,
            eval_ds,
            eval_final_indices,
            eval_batch_size=cfg.eval_batch_size,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt_train,
            progress_label="eval-only",
            transformed_predict=transformed_predict,
        )
        print(f"[eval-only] total {eval_metrics['total']:.6f}")
        return

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)

    _write_run_config(out_dir, cfg, model_cfg, task_cfg)

    def _reset_ssm_state(state: hk.State) -> hk.State:
        """Zero out SSM hidden states so each sample starts fresh."""
        return jax.tree_util.tree_map(
            lambda leaf: jnp.zeros_like(leaf) if isinstance(leaf, jax.Array) else leaf,
            state,
        )

    def _stop_grad_state(state: hk.State) -> hk.State:
        """Detach SSM state from computation graph (truncated BPTT).

        The state values are preserved for the next sample, but gradients
        are cut so backprop only flows through the current sample's
        target_steps.
        """
        return jax.tree_util.tree_map(
            lambda leaf: jax.lax.stop_gradient(leaf) if isinstance(leaf, jax.Array) else leaf,
            state,
        )

    use_sequential = cfg.sequential_segment_steps is not None

    @functools.partial(jax.jit, static_argnames=("reset_state",))
    def train_step(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        rng_key: jax.Array,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        forcings: xr.Dataset,
        reset_state: bool = True,
    ):
        # reset_state=True: zero out SSM state (random sampling or segment boundary)
        # reset_state=False: carry state with stop_gradient (truncated BPTT within segment)
        if reset_state:
            state = _reset_ssm_state(state)
        else:
            state = _stop_grad_state(state)

        def loss_fn(p, s, key):
            (loss_and_diag, new_state) = transformed.apply(p, s, key, inputs, targets, forcings, True)
            loss = scalarize_loss(loss_and_diag[0])
            return loss, new_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, new_opt_state, loss

    step = cfg.resume_step if cfg.resume_step is not None else 0

    train_losses: list[tuple[int, float]] = []
    eval_losses: list[tuple[int, float]] = []
    eval_details: list[dict[str, Any]] = []
    step_times: list[tuple[int, float]] = []
    mem_usage: list[tuple[int, float]] = []
    actual_usage: list[dict[str, Any]] = []
    epoch_summaries: list[dict[str, Any]] = []

    if cfg.resume_step is not None:
        train_losses = _filter_pairs_upto_step(_load_train_losses(out_dir / "train_loss.json"), cfg.resume_step)
        eval_losses = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "eval_loss.json"), cfg.resume_step)
        step_times = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "step_times.json"), cfg.resume_step)
        mem_usage = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "memory_gib.json"), cfg.resume_step)
        eval_details = _load_dict_series_upto_step(out_dir / "eval_details.json", cfg.resume_step)
        actual_usage = _load_dict_series_upto_step(out_dir / "actual_usage.json", cfg.resume_step)
        epoch_summaries = _load_json_list(out_dir / "epoch_summary.json")
        print(
            "Loaded existing logs for resume: "
            f"train={len(train_losses)}, eval={len(eval_losses)}, "
            f"step_times={len(step_times)}, mem={len(mem_usage)}, actual={len(actual_usage)}"
        )

    np_rng = np.random.default_rng(cfg.seed)

    # Build sampling structure
    if use_sequential:
        segments = build_sequential_segments(train_final_indices, cfg.sequential_segment_steps)
        np_rng.shuffle(segments)
        seg_idx = 0  # current segment
        seg_cursor = 0  # position within current segment
        print(
            f"Sequential sampling: {len(segments)} segments of ~{cfg.sequential_segment_steps} steps "
            f"({cfg.sequential_segment_steps * 6 / 24:.0f} days each)"
        )
    else:
        current_indices = train_final_indices.copy()
        np_rng.shuffle(current_indices)
    cursor = 0
    pass_idx = 1
    pass_start_step = step + 1
    pass_loss_accum: list[float] = []

    while step < cfg.max_steps:
        if use_sequential:
            # Sequential segment sampling
            need_new_segment = seg_idx >= len(segments) or seg_cursor >= len(segments[seg_idx])
            if seg_idx >= len(segments):
                # All segments exhausted -> new epoch
                epoch_summaries.append(
                    {
                        "pass": pass_idx,
                        "steps": step - pass_start_step + 1,
                        "train_loss_mean": float(np.mean(pass_loss_accum)) if pass_loss_accum else float("nan"),
                        "time_per_step_mean": float(
                            np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                        ),
                        "mem_gib_max": float(
                            np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])
                        ),
                    }
                )
                pass_idx += 1
                pass_start_step = step + 1
                pass_loss_accum = []
                segments = build_sequential_segments(train_final_indices, cfg.sequential_segment_steps)
                np_rng.shuffle(segments)
                seg_idx = 0
                seg_cursor = 0
            elif seg_cursor >= len(segments[seg_idx]):
                # Current segment exhausted -> move to next segment (reset state)
                seg_idx += 1
                seg_cursor = 0
                if seg_idx >= len(segments):
                    continue  # will trigger new epoch on next iteration

            # At start of segment, reset state; otherwise carry with stop_gradient
            reset_state = (seg_cursor == 0)
            seg = segments[seg_idx]
            batch_idx = seg[seg_cursor : seg_cursor + cfg.batch_size]
            seg_cursor += cfg.batch_size
        else:
            # Original random sampling
            if cursor >= len(current_indices):
                epoch_summaries.append(
                    {
                        "pass": pass_idx,
                        "steps": step - pass_start_step + 1,
                        "train_loss_mean": float(np.mean(pass_loss_accum)) if pass_loss_accum else float("nan"),
                        "time_per_step_mean": float(
                            np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                        ),
                        "mem_gib_max": float(
                            np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])
                        ),
                    }
                )
                pass_idx += 1
                pass_start_step = step + 1
                pass_loss_accum = []
                current_indices = train_final_indices.copy()
                np_rng.shuffle(current_indices)
                cursor = 0

            batch_idx = current_indices[cursor : cursor + cfg.batch_size]
            cursor += cfg.batch_size
            reset_state = True  # always reset in random mode

        batch_inputs, batch_targets, batch_forcings = build_batch_from_indices(
            train_ds,
            indices=batch_idx,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt_train,
        )

        rng, step_key = jax.random.split(rng)
        t0 = time.time()
        params, state, opt_state, loss = train_step(
            params,
            state,
            opt_state,
            step_key,
            batch_inputs,
            batch_targets,
            batch_forcings,
            reset_state=reset_state,
        )
        step_time = time.time() - t0

        step += 1
        loss_f = float(loss)
        train_losses.append((step, loss_f))
        pass_loss_accum.append(loss_f)
        step_times.append((step, step_time))

        usage = sample_actual_usage(step=step)
        actual_usage.append(usage)
        if usage.get("gpu_mem_gib") is not None:
            mem_usage.append((step, float(usage["gpu_mem_gib"])))

        if step % 10 == 0:
            print(f"step {step}/{cfg.max_steps} loss {loss_f:.6f}")

        if step % cfg.eval_every == 0:
            eval_metrics = run_eval(
                transformed,
                params,
                state,
                rng,
                eval_ds,
                eval_final_indices,
                eval_batch_size=cfg.eval_batch_size,
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt_train,
                progress_label=f"eval@step{step}",
                transformed_predict=transformed_predict,
            )
            eval_losses.append((step, eval_metrics["total"]))
            eval_details.append(
                {
                    "step": step,
                    "total": eval_metrics["total"],
                }
            )
            print(f"[eval] step {step} total {eval_metrics['total']:.6f}")
            save_logs(
                out_dir,
                train_losses,
                eval_losses,
                eval_details,
                step_times,
                mem_usage,
                actual_usage,
                epoch_summaries,
            )

        if step % cfg.checkpoint_every == 0:
            save_checkpoint(
                out_dir,
                params=params,
                step=step,
                model_cfg=model_cfg,
                task_cfg=task_cfg,
                description=ckpt_in.description,
                license_text=ckpt_in.license,
            )

    if pass_loss_accum:
        epoch_summaries.append(
            {
                "pass": pass_idx,
                "steps": step - pass_start_step + 1,
                "train_loss_mean": float(np.mean(pass_loss_accum)),
                "time_per_step_mean": float(
                    np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                ),
                "mem_gib_max": float(
                    np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])
                ),
            }
        )

    final_eval = run_eval(
        transformed,
        params,
        state,
        rng,
        eval_ds,
        eval_final_indices,
        eval_batch_size=cfg.eval_batch_size,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
        progress_label="eval@final",
        transformed_predict=transformed_predict,
    )
    eval_losses.append((step, final_eval["total"]))
    eval_details.append(
        {
            "step": step,
            "final": True,
            "total": final_eval["total"],
        }
    )

    save_checkpoint(
        out_dir,
        params=params,
        step=step,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        description=ckpt_in.description,
        license_text=ckpt_in.license,
    )
    save_logs(
        out_dir,
        train_losses,
        eval_losses,
        eval_details,
        step_times,
        mem_usage,
        actual_usage,
        epoch_summaries,
    )
    plot_loss_curves(out_dir, train_losses, eval_losses)

    rss_vals = [float(x["proc_rss_gib"]) for x in actual_usage if x.get("proc_rss_gib") is not None]
    gpu_vals = [float(x["gpu_mem_gib"]) for x in actual_usage if x.get("gpu_mem_gib") is not None]
    print(
        "Actual usage summary: "
        f"rss_peak={float(np.max(rss_vals)) if rss_vals else float('nan'):.3f} GiB, "
        f"rss_avg={float(np.mean(rss_vals)) if rss_vals else float('nan'):.3f} GiB, "
        f"gpu_peak={float(np.max(gpu_vals)) if gpu_vals else float('nan'):.3f} GiB, "
        f"gpu_avg={float(np.mean(gpu_vals)) if gpu_vals else float('nan'):.3f} GiB"
    )
    print(f"Done. Final eval total {final_eval['total']:.6f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
