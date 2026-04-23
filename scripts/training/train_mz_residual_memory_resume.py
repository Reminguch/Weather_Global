#!/usr/bin/env python3
"""Resumable variant of train_mz_residual_memory.py.

Adds two CLI flags to support continuing an earlier run:
  --resume-from <path>   : path to an mz_residual_stepN.pkl checkpoint.
  --resume-step <int>    : the step number that checkpoint corresponds to;
                           training continues at resume_step + 1.

Caveats:
  * Adam optimizer moments are NOT persisted in the old checkpoint format,
    so they restart from zero on resume. Expect a small grad-norm transient
    (<~50 steps) while the momenta warm up.
  * train_log.json and eval_log.json are APPENDED (not overwritten) when
    --resume-from is given, so the full curve is preserved.
  * Everything else matches train_mz_residual_memory.py. This file is kept
    separate so it cannot break currently-running non-resumable jobs.

The underlying task is the same as train_mz_residual_memory.py:
frozen GraphCast provides the Markov one-step prediction; a small Mamba-style
temporal block learns the residual correction; each contiguous time segment
is treated as one sample; only selected resolved variables are corrected
(geopotential, mean sea level pressure, u wind, v wind).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_graphcast as base_train

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import xarray as xr
from src.models.mz_residual_mamba import MZResidualConfig
from src.models.mz_residual_mamba import MZResidualMamba
from src.models.mz_meshed import (
    MZResidualMeshedConfig,
    MZResidualMeshedMamba,
    build_grid_mesh_projections,
)
from src.models.mz_residual_mamba import shift_residual_history


DEFAULT_BASELINE_CKPT = (
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/lagrangian_lab/"
    "artifacts/checkpoints/long_iid_mamba/bfix_r4_m3_in32_t1/ckpt_step4000.npz"
)
DEFAULT_OUT_DIR = "artifacts/checkpoints/mz_residual_memory"
RESOLVED_VARIABLES = (
    "mean_sea_level_pressure",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
)


@dataclasses.dataclass
class RunConfig:
    data_path: str
    baseline_ckpt: str
    stats_dir: str
    out_dir: str
    run_name: str
    resolution: float
    mesh_size: int
    val_year: int
    train_start_year: int | None
    train_end_year: int | None
    input_duration: str | None
    segment_steps: int
    max_steps: int
    eval_every: int
    eval_max_segments: int
    checkpoint_every: int
    hidden_size: int
    layers: int
    dropout: float
    a_log_init: float
    residual_loss_weight: float
    lr: float
    weight_decay: float
    grad_clip: float
    warmup_steps: int
    normalize_loss: bool
    standardize_input: bool
    baseline_precision: str
    # --- autoregressive / scheduled-sampling controls (Planned fix, Steps 1-4) ---
    train_mode: str            # "teacher" | "ar" | "mixed" | "target_rollout"
    tf_prob_start: float       # teacher-forcing prob at step 0 (mixed mode)
    tf_prob_end: float         # teacher-forcing prob at tf_prob_schedule_end
    tf_prob_schedule_end: int  # step after which tf_prob stays at tf_prob_end
    eval_autoregressive: bool  # also report closed-loop AR metrics at eval time
    residual_clip: float       # <=0 disables
    residual_shrinkage: float  # 1.0 = off
    target_steps: int          # forecast horizon K per sample. K=1 is one-step
                               # teacher mode; K>1 requires train_mode=target_rollout.
    seed: int
    precision: str
    # --- meshed MZ variant (grid -> mesh -> Mamba -> mesh -> grid) -----------
    meshed: bool               # if True, use MZResidualMeshedMamba instead of
                               # per-grid-point MZResidualMamba
    mz_mesh_size: int          # icosphere splits for MZ's internal mesh
                               # (independent of baseline's mesh_size)
    n_grid_neighbors: int      # grid->mesh KNN width
    n_mesh_neighbors: int      # mesh->grid KNN width
    # --- resume from existing checkpoint ------------------------------------
    resume_from: str | None    # path to an mz_residual_stepN.pkl to resume from
    resume_step: int           # step that checkpoint corresponds to


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train MZ-lite residual memory on top of frozen GraphCast.")
    parser.add_argument("--data-path", default=base_train.DEFAULT_DATA_PATH)
    parser.add_argument("--baseline-ckpt", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="mz_r4_m3_i32_seg32_h16")
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--mesh-size", type=int, default=3)
    parser.add_argument("--val-year", type=int, default=2022,
                        help="Year held out for validation + final inference. "
                             "Default 2022 to match the project's canonical split. "
                             "Used to default to 2021, which with the local 3-year "
                             "dataset silently made train=2020+2022 (a non-adjacent "
                             "split) and caused NaN at the year boundary.")
    parser.add_argument("--train-start-year", type=int, default=None,
                        help="Lower bound for train years (inclusive). None = all "
                             "available years except val-year.")
    parser.add_argument("--train-end-year", type=int, default=None,
                        help="Upper bound for train years (inclusive). None = all "
                             "available years except val-year.")
    parser.add_argument("--input-duration", default="192h")
    parser.add_argument("--segment-steps", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-max-segments", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--a-log-init", type=float, default=-0.1)
    parser.add_argument("--residual-loss-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Global-norm gradient clip. Set <=0 to disable.")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear warmup from 0 to --lr over this many steps. 0 disables.")
    parser.add_argument("--no-normalize-loss", action="store_true", default=False,
                        help="Disable per-variable std normalization of the loss (debug only).")
    parser.add_argument("--no-standardize-input", action="store_true", default=False,
                        help="Disable z-scoring of current_state and prev_residual inputs (debug only).")
    parser.add_argument("--baseline-precision", choices=["bf16", "fp32"], default="fp32",
                        help="Precision for the frozen baseline forward pass. Default fp32 to avoid "
                             "bf16 quantisation noise (~780 Pa on pressure) swamping the residual signal.")
    # --- Autoregressive / scheduled-sampling (Planned fix, Steps 1-4) ---
    parser.add_argument("--train-mode",
                        choices=["teacher", "ar", "mixed", "target_rollout"],
                        default="teacher",
                        help="'teacher' (default): parallel teacher-forced 1-step; "
                             "honest for operational NWP with continuous assimilation "
                             "(target_steps must equal 1). "
                             "'ar': pure autoregressive over the full segment (debug). "
                             "'mixed': scheduled sampling with Bernoulli tf_prob. "
                             "'target_rollout': K-step intra-sample AR rollout "
                             "(uses --target-steps K); step 1 of each anchor is "
                             "teacher-forced (observable at deployment) and "
                             "steps 2..K are self-fed. This is the right mode for "
                             "K>1 forecast horizons.")
    parser.add_argument("--target-steps", type=int, default=1,
                        help="Forecast horizon K per anchor. K=1 = single-step "
                             "assimilated forecast (use --train-mode teacher). "
                             "K>1 requires --train-mode target_rollout; the inner "
                             "K-step rollout is autoregressive (baseline feeds its "
                             "own output; MZ feeds its own r_hat).")
    parser.add_argument("--tf-prob-start", type=float, default=1.0,
                        help="Teacher-forcing probability at step 0 (mixed mode only).")
    parser.add_argument("--tf-prob-end", type=float, default=0.0,
                        help="Teacher-forcing probability after schedule end (mixed mode only).")
    parser.add_argument("--tf-prob-schedule-end", type=int, default=-1,
                        help="Step at which tf_prob reaches tf_prob_end (mixed mode). "
                             "-1 means use max_steps. Ignored in teacher/ar modes.")
    parser.add_argument("--no-eval-autoregressive", action="store_true", default=False,
                        help="Skip the autoregressive closed-loop eval metrics.")
    parser.add_argument("--residual-clip", type=float, default=0.0,
                        help="Clip the emitted (normalised) residual to [-c, c] at inference. "
                             "0 or negative disables clipping.")
    parser.add_argument("--residual-shrinkage", type=float, default=1.0,
                        help="Multiply predicted residual by this factor before adding to the "
                             "baseline (0 < beta <= 1). 1.0 = off.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--meshed", action="store_true", default=False,
                        help="Use the Grid->Mesh->Mamba->Mesh->Grid variant (MZResidualMeshedMamba).")
    parser.add_argument("--mz-mesh-size", type=int, default=3,
                        help="Icosphere splits for MZ's internal mesh (ignored unless --meshed).")
    parser.add_argument("--n-grid-neighbors", type=int, default=6,
                        help="K for the grid->mesh KNN aggregation (meshed only).")
    parser.add_argument("--n-mesh-neighbors", type=int, default=3,
                        help="K for the mesh->grid KNN aggregation (meshed only).")
    parser.add_argument("--resume-from", default=None,
                        help="Path to an mz_residual_stepN.pkl to resume training from.")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step number the --resume-from checkpoint corresponds to. "
                             "Training will continue at resume_step + 1 and run until --max-steps.")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="Precision hint (kept for compatibility; MZ network currently runs in fp32 either way).")
    args = parser.parse_args()

    if args.segment_steps < 4:
        raise ValueError("--segment-steps must be >= 4")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be > 0")
    if args.eval_max_segments <= 0:
        raise ValueError("--eval-max-segments must be > 0")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be > 0")
    if args.layers <= 0:
        raise ValueError("--layers must be > 0")
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError("--dropout must be in [0, 1)")
    if (args.train_start_year is None) ^ (args.train_end_year is None):
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if not (0.0 <= args.tf_prob_start <= 1.0) or not (0.0 <= args.tf_prob_end <= 1.0):
        raise ValueError("--tf-prob-start and --tf-prob-end must be in [0, 1].")
    if not (0.0 < args.residual_shrinkage <= 1.0):
        raise ValueError("--residual-shrinkage must be in (0, 1].")
    if args.target_steps < 1:
        raise ValueError("--target-steps must be >= 1")
    if args.target_steps > 1 and args.train_mode != "target_rollout":
        raise ValueError(
            f"--target-steps={args.target_steps} requires --train-mode target_rollout "
            "(step 2..K are unobservable at deployment so must self-feed)."
        )
    if args.train_mode == "target_rollout" and args.target_steps < 2:
        raise ValueError(
            "--train-mode target_rollout only makes sense with --target-steps >= 2."
        )

    return RunConfig(
        data_path=args.data_path,
        baseline_ckpt=args.baseline_ckpt,
        stats_dir=args.stats_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        resolution=args.resolution,
        mesh_size=args.mesh_size,
        val_year=args.val_year,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        input_duration=args.input_duration,
        segment_steps=args.segment_steps,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_max_segments=args.eval_max_segments,
        checkpoint_every=args.checkpoint_every,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
        a_log_init=args.a_log_init,
        residual_loss_weight=args.residual_loss_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        normalize_loss=(not args.no_normalize_loss),
        standardize_input=(not args.no_standardize_input),
        baseline_precision=args.baseline_precision,
        train_mode=args.train_mode,
        tf_prob_start=args.tf_prob_start,
        tf_prob_end=args.tf_prob_end,
        tf_prob_schedule_end=(args.tf_prob_schedule_end if args.tf_prob_schedule_end > 0 else args.max_steps),
        eval_autoregressive=(not args.no_eval_autoregressive),
        residual_clip=args.residual_clip,
        residual_shrinkage=args.residual_shrinkage,
        target_steps=args.target_steps,
        seed=args.seed,
        precision=args.precision,
        meshed=args.meshed,
        mz_mesh_size=args.mz_mesh_size,
        n_grid_neighbors=args.n_grid_neighbors,
        n_mesh_neighbors=args.n_mesh_neighbors,
        resume_from=args.resume_from,
        resume_step=args.resume_step,
    )


def _filter_time_continuous_indices(
    ds: xr.Dataset,
    indices: np.ndarray,
    *,
    input_steps: int,
    target_steps: int,
    dt,
) -> np.ndarray:
    """Keep only final-input indices whose full [idx-input+1 .. idx+target] window
    consists of consecutive timestamps spaced exactly by `dt`.

    When the training split concatenates non-adjacent years (2020 + 2022 here),
    the array index axis is continuous but the time coordinate has a gap. Samples
    whose window straddles that gap feed the frozen baseline forcings that jump
    in time (solar radiation, year_progress, accumulated precip) and trigger
    Inf/NaN in the baseline prediction. Filter them out here so downstream
    training never sees such samples.
    """
    import pandas as pd  # local import to avoid polluting module namespace
    if len(indices) == 0:
        return np.asarray([], dtype=np.int64)
    time_values = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    expected_dt = pd.Timedelta(dt)
    good: list[int] = []
    n_time = len(time_values)
    for idx in np.asarray(indices, dtype=np.int64):
        start = int(idx) - input_steps + 1
        stop = int(idx) + target_steps
        if start < 0 or stop >= n_time:
            continue
        window = time_values[start : stop + 1]
        diffs = np.diff(window)
        if np.all(diffs == expected_dt):
            good.append(int(idx))
    return np.asarray(good, dtype=np.int64)


def _time_continuous_segments(
    ds: xr.Dataset,
    indices: np.ndarray,
    segment_steps: int,
    dt,
) -> list[np.ndarray]:
    """Group time-continuous indices into segments of length `segment_steps`.

    Splits whenever consecutive indices correspond to a time delta different
    from `dt` (e.g. across a year boundary in a concatenated train split).
    """
    import pandas as pd
    if len(indices) == 0:
        return []
    sorted_idx = np.sort(np.asarray(indices, dtype=np.int64))
    time_values = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    expected_dt = pd.Timedelta(dt)
    time_at_idx = time_values[sorted_idx]
    diffs = np.diff(time_at_idx)
    # Split positions: wherever two consecutive valid indices are not dt apart
    # (covers both index gaps AND time gaps from concatenated years).
    gap_mask = diffs != expected_dt
    gap_positions = np.where(gap_mask)[0] + 1
    runs = np.split(sorted_idx, gap_positions)
    segments: list[np.ndarray] = []
    for run in runs:
        for i in range(0, len(run), segment_steps):
            chunk = run[i : i + segment_steps]
            if len(chunk) > 0:
                segments.append(chunk)
    return segments


def _resolved_feature_layout(task_cfg) -> tuple[tuple[str, ...], dict[str, slice], int]:
    layout: list[str] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for name in RESOLVED_VARIABLES:
        width = len(task_cfg.pressure_levels) if name in {
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
        } else 1
        slices[name] = slice(cursor, cursor + width)
        layout.append(name)
        cursor += width
    return tuple(layout), slices, cursor


def _stack_per_channel(
    task_cfg,
    ds: xr.Dataset,
    feature_order: tuple[str, ...],
) -> np.ndarray:
    """Stack per-(variable, level) stats into a shape (F,) vector matching feature_order."""
    chunks: list[np.ndarray] = []
    for name in feature_order:
        da = ds[name]
        if "level" in da.dims:
            arr = da.sel(level=list(task_cfg.pressure_levels)).values.astype(np.float32)
        else:
            arr = np.asarray([float(da.values)], dtype=np.float32)
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)


def _build_diffs_stddev_vector(
    task_cfg,
    norm_stats: dict[str, xr.Dataset],
    feature_order: tuple[str, ...],
) -> np.ndarray:
    """Per-channel one-step diff stddev (used to standardise the loss and the
    `prev_residual` input, and to de-normalise the MZ network output)."""
    return _stack_per_channel(task_cfg, norm_stats["diffs_stddev_by_level"], feature_order)


def _build_mean_stddev_vectors(
    task_cfg,
    norm_stats: dict[str, xr.Dataset],
    feature_order: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel absolute mean and stddev for z-scoring the `current_state`
    input so that the MZ network sees O(1) inputs per variable."""
    mean_f = _stack_per_channel(task_cfg, norm_stats["mean_by_level"], feature_order)
    std_f = _stack_per_channel(task_cfg, norm_stats["stddev_by_level"], feature_order)
    return mean_f, std_f


# Matches GraphCast's per-variable loss weights
# (third_party/graphcast/graphcast/graphcast.py:477-490). Variables not listed
# default to 1.0. Among our resolved set only mean_sea_level_pressure is
# downweighted.
_PER_VARIABLE_LOSS_WEIGHTS: dict[str, float] = {
    "mean_sea_level_pressure": 0.1,
}


def _build_latitude_weights(lat_deg_np: np.ndarray) -> np.ndarray:
    """Area-proportional latitude weights, normalised to mean 1.

    Uses cos(lat) which is the small-angle limit of the sphere-slice area
    and exactly matches GraphCast's off-pole convention. Pole rows (if any)
    get weight 0; that is slightly looser than GraphCast's special pole
    weighting but is a negligible correction in practice.
    """
    cos_lat = np.cos(np.deg2rad(lat_deg_np)).astype(np.float32)
    return cos_lat / float(cos_lat.mean())


def _build_per_channel_loss_weights(
    task_cfg,
    feature_order: tuple[str, ...],
    feature_slices: dict[str, slice],
) -> np.ndarray:
    """Per-channel loss weight vector of shape (F,).

    Combines:
      * per-variable weights (matching GraphCast): msl_pressure gets 0.1,
        others get 1.0.
      * pressure-level weights for 3D vars: `level / mean(level)` on the
        task pressure levels (normalised to mean 1). Surface vars get 1.
    """
    n_ch = sum((sl.stop - sl.start) for sl in feature_slices.values())
    w = np.zeros(n_ch, dtype=np.float32)
    levels = np.asarray(task_cfg.pressure_levels, dtype=np.float32)
    level_w_norm = levels / float(levels.mean())
    for name in feature_order:
        sl = feature_slices[name]
        w_var = _PER_VARIABLE_LOSS_WEIGHTS.get(name, 1.0)
        n = sl.stop - sl.start
        if n == 1:
            w[sl] = w_var
        else:
            if n != len(level_w_norm):
                raise ValueError(
                    f"Variable {name} has {n} channels but there are "
                    f"{len(level_w_norm)} task pressure levels."
                )
            w[sl] = w_var * level_w_norm
    return w


def _extract_feature_block(
    ds: xr.Dataset,
    *,
    time_index: int,
    task_cfg,
    feature_order: tuple[str, ...],
) -> jax.Array:
    arrays = []
    for name in feature_order:
        if name not in ds:
            raise KeyError(f"Missing resolved variable {name}")
        da = ds[name]
        if "time" in da.dims:
            da = da.isel(time=time_index)
        if "batch" not in da.dims:
            raise ValueError(f"Expected batch dimension for {name}, got dims={da.dims}")
        if "level" in da.dims:
            da = da.transpose("batch", "lat", "lon", "level")
            arr = np.asarray(da.values, dtype=np.float32)
        else:
            da = da.transpose("batch", "lat", "lon")
            arr = np.asarray(da.values, dtype=np.float32)[..., None]
        arrays.append(arr)
    return jnp.asarray(np.concatenate(arrays, axis=-1), dtype=jnp.float32)


def _compute_group_metrics(
    pred_tblnf: np.ndarray,
    truth_tblnf: np.ndarray,
    slices: dict[str, slice],
    prefix: str,
) -> dict[str, float]:
    results: dict[str, float] = {}
    total_sq = 0.0
    total_abs = 0.0
    total_n = 0
    diff = pred_tblnf - truth_tblnf
    for name, sl in slices.items():
        var_diff = diff[..., sl]
        n = int(var_diff.size)
        rmse = float(np.sqrt(np.mean(np.square(var_diff))))
        mae = float(np.mean(np.abs(var_diff)))
        results[f"{prefix}_{name}_RMSE"] = rmse
        results[f"{prefix}_{name}_MAE"] = mae
        total_sq += float(np.sum(np.square(var_diff)))
        total_abs += float(np.sum(np.abs(var_diff)))
        total_n += n
    results[f"{prefix}_overall_RMSE"] = float(np.sqrt(total_sq / total_n))
    results[f"{prefix}_overall_MAE"] = float(total_abs / total_n)
    return results


def _to_jax_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Wrap dataset data vars as JAX-backed xarray values for GraphCast/Haiku."""
    data_vars = {}
    for name, da in ds.data_vars.items():
        var = da.variable
        data_vars[name] = (var.dims, jnp.asarray(np.asarray(var.data)))
    coords = {name: coord.variable for name, coord in ds.coords.items()}
    return base_train.xarray_jax.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=ds.attrs,
    )


def _segment_to_tensors(
    baseline_predict,
    baseline_params,
    baseline_state,
    rng,
    ds: xr.Dataset,
    segment_indices: np.ndarray,
    *,
    input_steps: int,
    task_cfg,
    dt,
    feature_order: tuple[str, ...],
    target_steps: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build per-segment tensors with an arbitrary forecast horizon K=target_steps.

    For each of the ``len(segment_indices)`` = S anchors:
      * input window: ``input_steps`` real-obs frames ending at the anchor time,
      * target: K frames produced by the frozen baseline's K-step autoregressive
        rollout (step 1 uses real history; step s>1 uses baseline's own output
        from step s-1).

    The S anchors × K steps are flattened to a single length-T axis with
    ``T = S * K`` and time index ``T = k * K + s`` (s in 0..K-1).

    Returns four tensors:
      * ``seq_inputs``  [T, 1, lat, lon, 2F]  — current_state concatenated with
        teacher-forcing prev_residual (which is only populated at step 1 of
        each anchor; other entries are zero, will be ignored by the mask).
      * ``baseline_next`` [T, 1, lat, lon, F]
      * ``truth_next``    [T, 1, lat, lon, F]
      * ``tf_mask`` [T]  — 1.0 at anchor-step-1 positions (observable at
        deployment), 0.0 at intra-sample steps s>1 (must self-feed). For K=1
        this is all ones, giving back the original teacher-forced behaviour.
    """
    if segment_indices.ndim != 1:
        raise ValueError(f"Expected 1D segment indices, got shape={segment_indices.shape}")
    if len(segment_indices) < 2:
        raise ValueError("Segment must contain at least 2 steps.")
    if not np.all(np.diff(segment_indices) == 1):
        raise ValueError(
            "Expected contiguous sequential segment indices; got "
            f"{segment_indices[:8]}..."
        )
    if target_steps < 1:
        raise ValueError(f"target_steps must be >= 1, got {target_steps}")
    S = int(len(segment_indices))
    K = int(target_steps)

    inputs, targets, forcings = base_train.build_batch_from_indices(
        ds,
        indices=segment_indices,
        input_steps=input_steps,
        target_steps=K,
        task_cfg=task_cfg,
        dt=dt,
    )
    inputs_jax = _to_jax_dataset(inputs)
    targets_jax = _to_jax_dataset(targets)
    forcings_jax = _to_jax_dataset(forcings)

    preds, _ = baseline_predict.apply(
        baseline_params,
        baseline_state,
        rng,
        inputs_jax,
        targets_jax,
        forcings_jax,
        False,
    )

    current_state = _extract_feature_block(       # [S, lat, lon, F]
        inputs, time_index=-1,
        task_cfg=task_cfg, feature_order=feature_order,
    )
    # Pull all K target times for truth and baseline.
    truth_per_step = [                            # list of [S, lat, lon, F]
        _extract_feature_block(
            targets, time_index=s,
            task_cfg=task_cfg, feature_order=feature_order,
        )
        for s in range(K)
    ]
    baseline_per_step = [
        _extract_feature_block(
            preds, time_index=s,
            task_cfg=task_cfg, feature_order=feature_order,
        )
        for s in range(K)
    ]
    truth_SK = jnp.stack(truth_per_step, axis=1)        # [S, K, lat, lon, F]
    baseline_SK = jnp.stack(baseline_per_step, axis=1)  # [S, K, lat, lon, F]

    # Flatten (anchor, step) -> T = S*K in row-major order: anchor 0 all K
    # steps, then anchor 1 all K steps, etc.
    new_shape = (S * K,) + truth_SK.shape[2:]
    truth_flat = truth_SK.reshape(new_shape)
    baseline_flat = baseline_SK.reshape(new_shape)

    # Build per-(anchor, step) "current state" tensor. This is the state
    # estimate the MZ network sees as its anchor input at each intra-sample
    # step. Physically, at step s of anchor k we want an estimate of the state
    # at time tau_k + 6h*(s-1) -- the state FROM WHICH the next prediction is
    # being launched.
    #
    #   step s = 0 : real observation at the anchor's last input frame
    #                (obs_{tau_k} -- always available at deployment)
    #   step s > 0 : baseline's own prediction at step s-1 of this anchor
    #                (baseline_SK[:, s-1, ...]), which is already produced by
    #                the K-step autoregressive baseline rollout. This is a
    #                proxy for the unobservable state at tau_k + 6h*(s-1).
    #                Using baseline's output (not MZ's corrected output) keeps
    #                the current_state input decoupled from the residual-head
    #                self-feedback loop and is precomputable in this function.
    # For K=1 this reduces to the original behaviour (current_state[k,0] = obs).
    current_state_SK = jnp.zeros(
        (S, K) + current_state.shape[1:],
        dtype=current_state.dtype,
    )
    current_state_SK = current_state_SK.at[:, 0, ...].set(current_state)
    if K > 1:
        # For s in 1..K-1, current_state[:, s, ...] = baseline[:, s-1, ...].
        current_state_SK = current_state_SK.at[:, 1:, ...].set(
            baseline_SK[:, :-1, ...].astype(current_state.dtype)
        )
    current_state_flat = current_state_SK.reshape(new_shape)

    # Teacher-forcing prev_residual. The only positions where the model is
    # allowed to look at a ground-truth residual are the first step of each
    # anchor (intra-sample step 1), and the value used is the previous anchor's
    # step-1 residual (observable at deployment since truth at t_anchor_k was
    # observed at cycle k and anchor k-1 had predicted it as its step 1).
    #
    # Anchor 0 gets zeros (no previous anchor), matching shift_residual_history.
    residual_step1 = truth_SK[:, 0, ...] - baseline_SK[:, 0, ...]   # [S, lat, lon, F]
    residual_step1_shifted = jnp.concatenate(
        [jnp.zeros_like(residual_step1[:1]), residual_step1[:-1]], axis=0
    )  # [S, lat, lon, F]; position k = anchor (k-1)'s step-1 residual
    teacher_prev_residual_SK = jnp.zeros(
        (S, K) + residual_step1_shifted.shape[1:],
        dtype=residual_step1_shifted.dtype,
    )
    teacher_prev_residual_SK = teacher_prev_residual_SK.at[:, 0, ...].set(
        residual_step1_shifted
    )
    teacher_prev_residual_flat = teacher_prev_residual_SK.reshape(new_shape)

    # Per-step teacher-forcing mask. 1.0 at each anchor's first step
    # (observable at deployment), 0.0 elsewhere (intra-sample rollout, must
    # self-feed). K=1 -> all ones, equivalent to the original teacher regime.
    tf_mask = jnp.zeros((S * K,), dtype=jnp.float32)
    tf_mask = tf_mask.at[::K].set(1.0)

    # Add the length-1 batch axis expected by the MZ module: [T, B=1, lat, lon, F].
    def _addB(x):
        return x[:, None, ...]

    seq_inputs = jnp.concatenate(
        [_addB(current_state_flat), _addB(teacher_prev_residual_flat)],
        axis=-1,
    )
    return seq_inputs, _addB(baseline_flat), _addB(truth_flat), tf_mask


def _write_run_config(out_dir: Path, cfg: RunConfig, task_cfg, feature_slices: dict[str, slice]) -> None:
    payload = {
        "config": dataclasses.asdict(cfg),
        "resolved_variables": list(RESOLVED_VARIABLES),
        "feature_slices": {k: [v.start, v.stop] for k, v in feature_slices.items()},
        "task_config": dataclasses.asdict(task_cfg),
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_memory_checkpoint(out_dir: Path, step: int, params: hk.Params) -> None:
    path = out_dir / f"mz_residual_step{step}.pkl"
    with path.open("wb") as f:
        pickle.dump(params, f)
    print(f"saved residual-memory checkpoint: {path}")


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    ckpt = base_train.load_graphcast_checkpoint(Path(cfg.baseline_ckpt))
    task_cfg = ckpt.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg = dataclasses.replace(
        ckpt.model_config,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
    )

    norm_stats = base_train.load_stats(Path(cfg.stats_dir))
    base_train.validate_stats_coverage(task_cfg, norm_stats)

    class _SplitCfg:
        data_path = cfg.data_path
        resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = cfg.train_start_year
        train_end_year = cfg.train_end_year

    train_ds, eval_ds = base_train._open_local_splits(_SplitCfg)
    train_ds = base_train.prepare_dataset_for_task(train_ds, task_cfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)

    dt = base_train.infer_time_step(train_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    train_indices_raw = base_train.valid_final_input_indices(train_ds.sizes["time"], input_steps, 1)
    eval_indices_raw = base_train.valid_final_input_indices(eval_ds.sizes["time"], input_steps, 1)
    # The training set concatenates non-adjacent years (e.g. 2020 and 2022
    # with 2021 held out for validation). The resulting time axis has a
    # ~1-year jump at the boundary, but array indices remain consecutive.
    # Any sample whose full input+target window crosses that jump feeds
    # discontinuous forcings (year_progress / solar radiation / accumulated
    # precip) to the frozen baseline, which then produces Inf/NaN. Filter
    # those samples out based on actual time-stamp deltas.
    train_indices = _filter_time_continuous_indices(
        train_ds, train_indices_raw, input_steps=input_steps, target_steps=1, dt=dt
    )
    eval_indices = _filter_time_continuous_indices(
        eval_ds, eval_indices_raw, input_steps=input_steps, target_steps=1, dt=dt
    )
    train_dropped = len(train_indices_raw) - len(train_indices)
    eval_dropped = len(eval_indices_raw) - len(eval_indices)
    if train_dropped or eval_dropped:
        print(
            f"[mz-setup] dropped {train_dropped} train and {eval_dropped} eval indices "
            f"whose input+target window straddles a time gap (likely year boundary in "
            f"the concatenated train split)."
        )
    train_segments = [
        seg for seg in _time_continuous_segments(train_ds, train_indices, cfg.segment_steps, dt)
        if len(seg) == cfg.segment_steps
    ]
    eval_segments = [
        seg for seg in _time_continuous_segments(eval_ds, eval_indices, cfg.segment_steps, dt)
        if len(seg) == cfg.segment_steps
    ]
    if not train_segments:
        raise ValueError("No full-length train segments available.")
    if not eval_segments:
        raise ValueError("No full-length eval segments available.")

    feature_order, feature_slices, feature_dim = _resolved_feature_layout(task_cfg)
    diffs_std_f_np = _build_diffs_stddev_vector(task_cfg, norm_stats, feature_order)
    mean_f_np, std_f_np = _build_mean_stddev_vectors(task_cfg, norm_stats, feature_order)
    lat_weights_np = _build_latitude_weights(np.asarray(train_ds.lat.values, dtype=np.float32))
    per_channel_loss_w_np = _build_per_channel_loss_weights(
        task_cfg, feature_order, feature_slices
    )
    lat_weights = jnp.asarray(lat_weights_np, dtype=jnp.float32)
    per_channel_loss_w = jnp.asarray(per_channel_loss_w_np, dtype=jnp.float32)
    loss_norm_f = (
        jnp.asarray(diffs_std_f_np, dtype=jnp.float32)
        if cfg.normalize_loss
        else jnp.ones((feature_dim,), dtype=jnp.float32)
    )
    # Input z-score: (current_state - mean) / std; prev_residual is zero-mean
    # by construction, so scale it by diffs_stddev only. De-normalise the
    # network's predicted residual by multiplying by diffs_stddev before
    # adding to the baseline prediction.
    if cfg.standardize_input:
        input_mean_f = jnp.asarray(mean_f_np, dtype=jnp.float32)
        input_std_f = jnp.asarray(std_f_np, dtype=jnp.float32)
        output_denorm_f = jnp.asarray(diffs_std_f_np, dtype=jnp.float32)
        residual_input_std_f = jnp.asarray(diffs_std_f_np, dtype=jnp.float32)
    else:
        input_mean_f = jnp.zeros((feature_dim,), dtype=jnp.float32)
        input_std_f = jnp.ones((feature_dim,), dtype=jnp.float32)
        output_denorm_f = jnp.ones((feature_dim,), dtype=jnp.float32)
        residual_input_std_f = jnp.ones((feature_dim,), dtype=jnp.float32)
    print(
        "MZ-lite setup: "
        f"resolved={feature_order}, feature_dim={feature_dim}, "
        f"train_segments={len(train_segments)}, eval_segments={len(eval_segments)}, "
        f"segment_steps={cfg.segment_steps}, input_steps={input_steps}, "
        f"normalize_loss={cfg.normalize_loss}, standardize_input={cfg.standardize_input}, "
        f"baseline_precision={cfg.baseline_precision}, "
        f"grad_clip={cfg.grad_clip}, warmup_steps={cfg.warmup_steps}"
    )
    if cfg.normalize_loss:
        print("  per-channel diff stddev:", diffs_std_f_np.tolist())
    if cfg.standardize_input:
        print("  per-channel mean (for input z-score):", mean_f_np.tolist())
        print("  per-channel std  (for input z-score):", std_f_np.tolist())
    print(
        f"  lat-weight  min/mean/max = "
        f"{float(lat_weights_np.min()):.3f}/{float(lat_weights_np.mean()):.3f}/{float(lat_weights_np.max()):.3f}"
    )
    print(
        "  per-channel loss weight (per-var * level/mean(level)):",
        per_channel_loss_w_np.tolist(),
    )

    def baseline_predict_fn(inputs, targets, forcings, is_training):
        predictor = base_train.build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.baseline_precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone="none",
            temporal_location="mesh_post_encoder",
            temporal_hidden_size=model_cfg.latent_size,
            temporal_layers=1,
            temporal_dropout=0.0,
        )
        return predictor(
            inputs,
            targets_template=targets,
            forcings=forcings,
            is_training=is_training,
        )

    baseline_predict = hk.transform_with_state(baseline_predict_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        train_ds,
        indices=[int(train_indices[0])],
        input_steps=input_steps,
        target_steps=1,
        task_cfg=task_cfg,
        dt=dt,
    )
    sample_inputs = _to_jax_dataset(sample_inputs)
    sample_targets = _to_jax_dataset(sample_targets)
    sample_forcings = _to_jax_dataset(sample_forcings)
    _baseline_params_unused, baseline_state = baseline_predict.init(
        rng, sample_inputs, sample_targets, sample_forcings, False
    )
    baseline_params = ckpt.params

    seq_inputs0, baseline_next0, truth_next0, tf_mask0 = _segment_to_tensors(
        baseline_predict,
        baseline_params,
        baseline_state,
        rng,
        train_ds,
        train_segments[0],
        input_steps=input_steps,
        task_cfg=task_cfg,
        dt=dt,
        feature_order=feature_order,
        target_steps=cfg.target_steps,
    )

    # ---- MZ model factory: either per-grid-point Mamba or meshed variant ---
    if cfg.meshed:
        lat_deg = np.asarray(eval_ds.lat.values, dtype=np.float64)
        lon_deg = np.asarray(eval_ds.lon.values, dtype=np.float64)
        proj_arrays, n_mesh_nodes = build_grid_mesh_projections(
            lat_deg=lat_deg, lon_deg=lon_deg, mesh_size=cfg.mz_mesh_size,
            n_grid_neighbors=cfg.n_grid_neighbors,
            n_mesh_neighbors=cfg.n_mesh_neighbors,
        )
        print(
            f"[meshed] mz_mesh_size={cfg.mz_mesh_size}  n_mesh_nodes={n_mesh_nodes}  "
            f"n_grid_pts={len(lat_deg) * len(lon_deg)}  "
            f"KNN g2m={cfg.n_grid_neighbors} m2g={cfg.n_mesh_neighbors}"
        )
        mz_cfg_meshed = MZResidualMeshedConfig(
            input_size=feature_dim * 2,
            output_size=feature_dim,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            dropout=cfg.dropout,
            a_log_init=cfg.a_log_init,
        )

        def _build_mz_model():
            return MZResidualMeshedMamba(
                mz_cfg_meshed,
                n_mesh_nodes=n_mesh_nodes,
                **proj_arrays,
            )

        mz_cfg = mz_cfg_meshed  # used for attribute access below (input_size/output_size)
    else:
        mz_cfg = MZResidualConfig(
            input_size=feature_dim * 2,
            output_size=feature_dim,
            hidden_size=cfg.hidden_size,
            layers=cfg.layers,
            dropout=cfg.dropout,
            a_log_init=cfg.a_log_init,
        )

        def _build_mz_model():
            return MZResidualMamba(mz_cfg)

    def _normalize_inputs(seq_inputs):
        """Split raw seq_inputs into (current_state_n, true_prev_residual_n) both normalised."""
        current_state, prev_residual = jnp.split(seq_inputs, 2, axis=-1)
        current_state_n = (
            current_state - input_mean_f[None, None, None, None, :]
        ) / input_std_f[None, None, None, None, :]
        prev_residual_n = prev_residual / residual_input_std_f[None, None, None, None, :]
        return current_state_n, prev_residual_n

    def _finalise_loss(pred_residual, baseline_next, truth_next):
        """GraphCast-style lat/level/per-var weighted MSE in normalised residual units."""
        corrected = baseline_next + pred_residual
        residual_target = truth_next - baseline_next
        inv_norm = 1.0 / loss_norm_f[None, None, None, None, :]
        norm_diff = (corrected - truth_next) * inv_norm
        norm_res_diff = (pred_residual - residual_target) * inv_norm
        w_lat = lat_weights[None, None, :, None, None]
        w_chan = per_channel_loss_w[None, None, None, None, :]
        weighted_sq_state = jnp.square(norm_diff) * w_lat * w_chan
        weighted_sq_res = jnp.square(norm_res_diff) * w_lat * w_chan
        state_loss = jnp.mean(weighted_sq_state)
        residual_loss = jnp.mean(weighted_sq_res)
        total_loss = state_loss  # algebraically identical to residual_loss
        return corrected, total_loss, state_loss, residual_loss

    def _pred_to_real(pred_residual_n):
        """De-normalise residual back to physical units and apply optional shrinkage."""
        pred_residual = pred_residual_n * output_denorm_f[None, None, None, None, :]
        if cfg.residual_shrinkage < 1.0:
            pred_residual = pred_residual * cfg.residual_shrinkage
        return pred_residual

    # Option-2 feedback constants (used in target_rollout and AR eval paths).
    # At intra-sample step s>0 the network's "current state" input becomes
    # (baseline_raw_{s-1} + pred_residual_raw_{s-1} - mean) / std
    # = baseline_absolute_n[s-1] + pred_residual_n[s-1] * (output_denorm / input_std)
    residual_to_state_rescale_f = output_denorm_f / input_std_f

    def residual_objective(seq_inputs, baseline_next, truth_next, tf_mask, is_training, tf_prob):
        """Training forward.

        Dispatches between teacher / ar / mixed / target_rollout modes at
        graph-build time by reading ``cfg.train_mode``.
          * ``tf_prob`` is the scalar scheduled-sampling probability (only
            used in ``mixed`` mode).
          * ``tf_mask`` is the per-step deterministic teacher-forcing mask
            ([T] of 0/1) produced by ``_segment_to_tensors``. Only used in
            ``target_rollout`` mode.
        Eval paths are defined separately.
        """
        current_state_n, true_prev_residual_n = _normalize_inputs(seq_inputs)
        model = _build_mz_model()

        if cfg.train_mode == "teacher":
            # Original path: parallel scan, teacher-forced prev_residual.
            # Only valid for target_steps=1; validated in parse_args.
            seq_inputs_n = jnp.concatenate([current_state_n, true_prev_residual_n], axis=-1)
            pred_residual_n = model(seq_inputs_n, is_training=is_training)
        elif cfg.train_mode == "target_rollout":
            # K-step intra-sample AR: per-step mask is 1 at anchor-step-1
            # positions (observable) and 0 at intra-sample s>1 (self-feed).
            # Option-2 state feedback: MZ's own corrected output from step
            # s-1 is fed in as the "current state" at step s (normalized).
            baseline_absolute_n = (
                baseline_next - input_mean_f[None, None, None, None, :]
            ) / input_std_f[None, None, None, None, :]
            pred_residual_n = model.rollout_ar(
                current_state_n,
                is_training=is_training,
                true_prev_residual_n_tblnf=true_prev_residual_n,
                tf_mask_per_step=tf_mask,
                residual_clip=(cfg.residual_clip if cfg.residual_clip > 0 else None),
                baseline_absolute_n_tblnf=baseline_absolute_n,
                residual_to_state_rescale_f=residual_to_state_rescale_f,
            )
        else:
            # AR or mixed: scalar Bernoulli tf_prob (no per-step mask).
            pred_residual_n = model.rollout_ar(
                current_state_n,
                is_training=is_training,
                true_prev_residual_n_tblnf=true_prev_residual_n,
                teacher_forcing_prob=tf_prob,
                residual_clip=(cfg.residual_clip if cfg.residual_clip > 0 else None),
            )
        pred_residual = _pred_to_real(pred_residual_n)
        corrected, total_loss, state_loss, residual_loss = _finalise_loss(
            pred_residual, baseline_next, truth_next
        )
        return {
            "total_loss": total_loss,
            "state_loss": state_loss,
            "residual_loss": residual_loss,
            "corrected": corrected,
            "pred_residual": pred_residual,
        }

    def residual_objective_ar_eval(seq_inputs, baseline_next, truth_next):
        """Closed-loop AR evaluation: prev_residual always comes from the
        model's own output. Also uses Option-2 state feedback so that the
        intra-sample ``current_state`` at step s>0 is the previous step's
        corrected prediction (baseline_{s-1} + r_hat_{s-1}) in normalized
        units, matching what a deployed system would feed forward."""
        current_state_n, _ = _normalize_inputs(seq_inputs)
        model = _build_mz_model()
        baseline_absolute_n = (
            baseline_next - input_mean_f[None, None, None, None, :]
        ) / input_std_f[None, None, None, None, :]
        pred_residual_n = model.rollout_ar(
            current_state_n,
            is_training=False,
            true_prev_residual_n_tblnf=None,     # force pure autoregression
            teacher_forcing_prob=0.0,
            residual_clip=(cfg.residual_clip if cfg.residual_clip > 0 else None),
            baseline_absolute_n_tblnf=baseline_absolute_n,
            residual_to_state_rescale_f=residual_to_state_rescale_f,
        )
        pred_residual = _pred_to_real(pred_residual_n)
        corrected, total_loss, state_loss, residual_loss = _finalise_loss(
            pred_residual, baseline_next, truth_next
        )
        return {
            "total_loss": total_loss,
            "state_loss": state_loss,
            "residual_loss": residual_loss,
            "corrected": corrected,
            "pred_residual": pred_residual,
        }

    residual_model = hk.transform(residual_objective)
    residual_model_ar_eval = hk.transform(residual_objective_ar_eval)
    # Init: run in whichever mode is needed to create ALL params.
    # With train_mode=teacher, __call__ creates in_proj/out_proj/... on the SSM
    # block; rollout_ar reuses them at eval time thanks to matching names.
    # With train_mode in {ar, mixed, target_rollout}, rollout_ar itself creates
    # all params.
    mem_params = residual_model.init(
        rng, seq_inputs0, baseline_next0, truth_next0, tf_mask0, True, 1.0
    )

    # ---- Resume from existing checkpoint, if requested -----------------------
    if cfg.resume_from is not None:
        ckpt_path = Path(cfg.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume-from path not found: {ckpt_path}")
        with ckpt_path.open("rb") as f:
            loaded_params = pickle.load(f)
        # Shape-check by recursing leaf pairs; haiku trees are dicts of dicts.
        loaded_leaves = jax.tree_util.tree_leaves(loaded_params)
        init_leaves = jax.tree_util.tree_leaves(mem_params)
        if len(loaded_leaves) != len(init_leaves):
            raise ValueError(
                f"resume mismatch: checkpoint has {len(loaded_leaves)} param leaves, "
                f"current model has {len(init_leaves)}. Are configs aligned?"
            )
        for i, (lp, ip) in enumerate(zip(loaded_leaves, init_leaves)):
            if lp.shape != ip.shape:
                raise ValueError(
                    f"resume mismatch at leaf {i}: ckpt shape {lp.shape} != "
                    f"init shape {ip.shape}"
                )
        mem_params = loaded_params
        print(
            f"[resume] loaded {len(loaded_leaves)} param leaves from {ckpt_path}; "
            f"training continues at step {cfg.resume_step + 1} -> {cfg.max_steps}"
        )

    if cfg.warmup_steps > 0:
        lr_schedule = optax.linear_schedule(
            init_value=0.0, end_value=cfg.lr, transition_steps=cfg.warmup_steps
        )
    else:
        lr_schedule = cfg.lr
    opt_chain = []
    if cfg.grad_clip > 0:
        opt_chain.append(optax.clip_by_global_norm(cfg.grad_clip))
    opt_chain.append(optax.adamw(lr_schedule, weight_decay=cfg.weight_decay))
    opt = optax.chain(*opt_chain) if len(opt_chain) > 1 else opt_chain[0]
    opt_state = opt.init(mem_params)

    @jax.jit
    def train_step(params, opt_state, rng_key, seq_inputs, baseline_next, truth_next, tf_mask, tf_prob):
        def loss_fn(p):
            outputs = residual_model.apply(
                p, rng_key, seq_inputs, baseline_next, truth_next, tf_mask, True, tf_prob
            )
            return outputs["total_loss"], outputs

        (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (
            new_params,
            new_opt_state,
            loss,
            outputs["state_loss"],
            outputs["residual_loss"],
            grad_norm,
        )

    @jax.jit
    def eval_step_teacher(params, rng_key, seq_inputs, baseline_next, truth_next, tf_mask):
        """Eval in the same mode as training:
          * train_mode=teacher       -> parallel teacher-forced scan
          * train_mode=target_rollout -> per-step-mask K-step rollout
          * train_mode in {ar, mixed} -> pure teacher (tf_prob=1) eval (optimistic)
        """
        return residual_model.apply(
            params, rng_key, seq_inputs, baseline_next, truth_next, tf_mask, False, 1.0
        )

    @jax.jit
    def eval_step_ar(params, rng_key, seq_inputs, baseline_next, truth_next):
        """Closed-loop autoregressive eval (deployment-honest, ignores tf_mask)."""
        return residual_model_ar_eval.apply(
            params, rng_key, seq_inputs, baseline_next, truth_next
        )

    # Back-compat alias used by the pre-flight probe below.
    eval_step = eval_step_teacher

    def tf_prob_for_step(step: int) -> float:
        """Scheduled-sampling teacher-forcing probability at a given training step.

        Note: in ``target_rollout`` mode tf_prob is ignored (the per-step
        deterministic mask from _segment_to_tensors takes over); we return 1.0
        here just so the jit'd train_step has a valid scalar to pass through.
        """
        if cfg.train_mode == "teacher" or cfg.train_mode == "target_rollout":
            return 1.0
        if cfg.train_mode == "ar":
            return 0.0
        # mixed: linear anneal from tf_prob_start at step 0 to tf_prob_end at
        # tf_prob_schedule_end; constant thereafter.
        end = max(1, cfg.tf_prob_schedule_end)
        frac = min(1.0, step / end)
        return float(cfg.tf_prob_start + (cfg.tf_prob_end - cfg.tf_prob_start) * frac)

    _write_run_config(out_dir, cfg, task_cfg, feature_slices)

    train_log: list[dict[str, Any]] = []
    eval_log: list[dict[str, Any]] = []
    # If resuming, load prior logs so new entries get appended to the full curve.
    if cfg.resume_from is not None:
        train_log_path = out_dir / "train_log.json"
        eval_log_path = out_dir / "eval_log.json"
        if train_log_path.exists():
            with train_log_path.open("r", encoding="utf-8") as f:
                train_log = json.load(f)
            # Trim any entries beyond the resume step (shouldn't happen, but defensive)
            train_log = [e for e in train_log if e.get("step", 0) <= cfg.resume_step]
            print(f"[resume] loaded {len(train_log)} existing train_log entries")
        if eval_log_path.exists():
            with eval_log_path.open("r", encoding="utf-8") as f:
                eval_log = json.load(f)
            eval_log = [e for e in eval_log if e.get("step", 0) <= cfg.resume_step]
            print(f"[resume] loaded {len(eval_log)} existing eval_log entries")
    segments_shuffled = list(train_segments)
    random.shuffle(segments_shuffled)
    seg_ptr = 0

    # --- Pre-flight NaN probe on the first few segments ---
    # Reruns the pipeline (_segment_to_tensors + one forward of residual_model
    # in eval mode, no gradient step) and prints per-stage finiteness and
    # dynamic range. Catches bad segments / bad baseline outputs before the
    # training loop has a chance to poison mem_params with NaN.
    def _nan_probe(tag: str, x) -> dict:
        arr = np.asarray(x)
        return {
            "nan": bool(np.any(np.isnan(arr))),
            "inf": bool(np.any(np.isinf(arr))),
            "min": float(np.nanmin(arr)) if arr.size else float("nan"),
            "max": float(np.nanmax(arr)) if arr.size else float("nan"),
            "abs_max": float(np.nanmax(np.abs(arr))) if arr.size else float("nan"),
        }

    print("[mz-preflight] probing first segments for NaN/Inf before training...")
    probe_n = min(8, len(segments_shuffled))
    for probe_i in range(probe_n):
        seg = segments_shuffled[probe_i]
        rng, key = jax.random.split(rng)
        seq_inputs_p, baseline_next_p, truth_next_p, tf_mask_p = _segment_to_tensors(
            baseline_predict,
            baseline_params,
            baseline_state,
            key,
            train_ds,
            seg,
            input_steps=input_steps,
            task_cfg=task_cfg,
            dt=dt,
            feature_order=feature_order,
            target_steps=cfg.target_steps,
        )
        outputs_p = eval_step(mem_params, key, seq_inputs_p, baseline_next_p, truth_next_p, tf_mask_p)
        seq_stats = _nan_probe("seq_inputs", seq_inputs_p)
        base_stats = _nan_probe("baseline_next", baseline_next_p)
        truth_stats = _nan_probe("truth_next", truth_next_p)
        pred_stats = _nan_probe("pred_residual", outputs_p["pred_residual"])
        corr_stats = _nan_probe("corrected", outputs_p["corrected"])
        loss_f = float(np.asarray(outputs_p["total_loss"]))
        print(
            f"  probe {probe_i}: seg_len={len(seg)} seg_first={int(seg[0])} seg_last={int(seg[-1])}  "
            f"loss={loss_f:.4f}"
        )
        for nm, st in [("seq_inputs", seq_stats), ("baseline_next", base_stats),
                       ("truth_next", truth_stats), ("pred_residual", pred_stats),
                       ("corrected", corr_stats)]:
            flag = ""
            if st["nan"]: flag += " NaN!"
            if st["inf"]: flag += " Inf!"
            print(f"    {nm:16s} min={st['min']:+.3e} max={st['max']:+.3e} |.|max={st['abs_max']:.3e}{flag}")
        if seq_stats["nan"] or seq_stats["inf"] or base_stats["nan"] or base_stats["inf"] \
           or truth_stats["nan"] or truth_stats["inf"] or pred_stats["nan"] or pred_stats["inf"]:
            raise RuntimeError(
                f"[mz-preflight] Segment {probe_i} produced NaN/Inf upstream of the "
                "optimiser. Fix the data / baseline before launching a full train run."
            )
    print("[mz-preflight] all probed segments finite. starting training loop.")

    def run_eval(step: int) -> dict[str, float]:
        """Evaluate on held-out segments in both teacher-forced and AR modes.

        Metrics are emitted with prefixes:
          * ``baseline_*``         — frozen baseline (constant across steps),
          * ``corrected_*``        — teacher-forced MZ (original, optimistic),
          * ``corrected_ar_*``     — closed-loop autoregressive MZ (honest).
        The ``total_loss``/``state_loss``/``residual_loss`` scalars report the
        teacher-forced numbers (kept for log compatibility); their AR
        counterparts are logged as ``total_loss_ar`` etc.
        """
        nonlocal rng
        chosen_segments = eval_segments[: min(cfg.eval_max_segments, len(eval_segments))]
        tf_totals: list[float] = []
        tf_state: list[float] = []
        tf_res: list[float] = []
        ar_totals: list[float] = []
        ar_state: list[float] = []
        ar_res: list[float] = []
        corr_tf_metrics_accum: list[dict[str, float]] = []
        corr_ar_metrics_accum: list[dict[str, float]] = []
        baseline_metrics_accum: list[dict[str, float]] = []
        t0 = time.time()

        for seg_i, seg in enumerate(chosen_segments, start=1):
            rng, key = jax.random.split(rng)
            seq_inputs, baseline_next, truth_next, tf_mask = _segment_to_tensors(
                baseline_predict,
                baseline_params,
                baseline_state,
                key,
                eval_ds,
                seg,
                input_steps=input_steps,
                task_cfg=task_cfg,
                dt=dt,
                feature_order=feature_order,
                target_steps=cfg.target_steps,
            )
            # Teacher-forced eval (uses true previous residual; optimistic).
            tf_outputs = eval_step_teacher(mem_params, key, seq_inputs, baseline_next, truth_next, tf_mask)
            tf_corrected = np.asarray(tf_outputs["corrected"], dtype=np.float32)
            tf_totals.append(float(tf_outputs["total_loss"]))
            tf_state.append(float(tf_outputs["state_loss"]))
            tf_res.append(float(tf_outputs["residual_loss"]))
            corr_tf_metrics_accum.append(
                _compute_group_metrics(tf_corrected, np.asarray(truth_next, dtype=np.float32),
                                       feature_slices, "corrected")
            )

            # Closed-loop autoregressive eval (honest; prev_residual = model's own r_hat).
            if cfg.eval_autoregressive:
                ar_outputs = eval_step_ar(mem_params, key, seq_inputs, baseline_next, truth_next)
                ar_corrected = np.asarray(ar_outputs["corrected"], dtype=np.float32)
                ar_totals.append(float(ar_outputs["total_loss"]))
                ar_state.append(float(ar_outputs["state_loss"]))
                ar_res.append(float(ar_outputs["residual_loss"]))
                corr_ar_metrics_accum.append(
                    _compute_group_metrics(ar_corrected, np.asarray(truth_next, dtype=np.float32),
                                           feature_slices, "corrected_ar")
                )

            baseline_metrics_accum.append(
                _compute_group_metrics(np.asarray(baseline_next, dtype=np.float32),
                                       np.asarray(truth_next, dtype=np.float32),
                                       feature_slices, "baseline")
            )
            if seg_i == 1 or seg_i % 4 == 0 or seg_i == len(chosen_segments):
                msg = (f"[mz-eval@step{step}] segment {seg_i}/{len(chosen_segments)} "
                       f"elapsed {time.time() - t0:.1f}s  tf_total={tf_totals[-1]:.6f}")
                if cfg.eval_autoregressive:
                    msg += f"  ar_total={ar_totals[-1]:.6f}"
                print(msg)

        results: dict[str, float] = {
            "step": step,
            "total_loss": float(np.mean(tf_totals)),        # backward-compat = teacher
            "state_loss": float(np.mean(tf_state)),
            "residual_loss": float(np.mean(tf_res)),
        }
        if cfg.eval_autoregressive:
            results["total_loss_ar"] = float(np.mean(ar_totals))
            results["state_loss_ar"] = float(np.mean(ar_state))
            results["residual_loss_ar"] = float(np.mean(ar_res))

        for k in corr_tf_metrics_accum[0]:
            results[k] = float(np.mean([m[k] for m in corr_tf_metrics_accum]))
        for k in baseline_metrics_accum[0]:
            results[k] = float(np.mean([m[k] for m in baseline_metrics_accum]))
        if cfg.eval_autoregressive and corr_ar_metrics_accum:
            for k in corr_ar_metrics_accum[0]:
                results[k] = float(np.mean([m[k] for m in corr_ar_metrics_accum]))

        line = (f"[mz-eval] step {step}  "
                f"baseline_MAE={results['baseline_overall_MAE']:.4f}  "
                f"tf_MAE={results['corrected_overall_MAE']:.4f}")
        if cfg.eval_autoregressive:
            line += f"  ar_MAE={results['corrected_ar_overall_MAE']:.4f}"
            gap = results['corrected_ar_overall_MAE'] - results['corrected_overall_MAE']
            line += f"  AR-TF gap={gap:+.4f}"
        print(line)
        return results

    start_step = cfg.resume_step + 1 if cfg.resume_from is not None else 1
    if cfg.resume_from is not None and start_step > cfg.max_steps:
        print(
            f"[resume] resume_step={cfg.resume_step} >= max_steps={cfg.max_steps}; "
            "nothing to do"
        )
        return
    for step in range(start_step, cfg.max_steps + 1):
        if seg_ptr >= len(segments_shuffled):
            random.shuffle(segments_shuffled)
            seg_ptr = 0
        seg = segments_shuffled[seg_ptr]
        seg_ptr += 1

        rng, key = jax.random.split(rng)
        t_step0 = time.time()
        seq_inputs, baseline_next, truth_next, tf_mask = _segment_to_tensors(
            baseline_predict,
            baseline_params,
            baseline_state,
            key,
            train_ds,
            seg,
            input_steps=input_steps,
            task_cfg=task_cfg,
            dt=dt,
            feature_order=feature_order,
            target_steps=cfg.target_steps,
        )
        tf_prob = tf_prob_for_step(step)
        mem_params, opt_state, loss, state_loss, residual_loss, grad_norm = train_step(
            mem_params, opt_state, key, seq_inputs, baseline_next, truth_next,
            tf_mask, jnp.asarray(tf_prob, dtype=jnp.float32),
        )
        step_sec = time.time() - t_step0
        train_log.append(
            {
                "step": step,
                "loss": float(loss),
                "state_loss": float(state_loss),
                "residual_loss": float(residual_loss),
                "grad_norm": float(grad_norm),
                "tf_prob": float(tf_prob),
                "step_seconds": step_sec,
            }
        )
        if step == 1 or step % 10 == 0:
            print(
                f"step {step}/{cfg.max_steps} loss {float(loss):.6f} "
                f"state_loss {float(state_loss):.6f} residual_loss {float(residual_loss):.6f} "
                f"grad_norm {float(grad_norm):.3e} tf_prob {float(tf_prob):.3f} "
                f"step_time {step_sec:.2f}s"
            )

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            results = run_eval(step)
            eval_log.append(results)
            with (out_dir / "train_log.json").open("w", encoding="utf-8") as f:
                json.dump(train_log, f, indent=2)
            with (out_dir / "eval_log.json").open("w", encoding="utf-8") as f:
                json.dump(eval_log, f, indent=2)

        if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
            _save_memory_checkpoint(out_dir, step, mem_params)


if __name__ == "__main__":
    main()
