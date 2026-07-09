#!/usr/bin/env python3
"""v4 trainer: K=1 stateful-cycling residual Mamba with truncated BPTT.

Design:
  * Frozen GraphCast baseline emits one-step prediction (same as v1/v3).
  * Trainable residual Mamba head predicts truth - baseline at +6h.
  * No K>1 self-fed rollout. target_steps == 1 always.
  * Training proceeds in chunked TBPTT over long contiguous segments:
      - Each "segment" is len_segment consecutive 6h cycles (e.g. 32).
      - num_lanes independent segments are processed in parallel batch-wise.
      - Each train step processes ONE chunk of bptt_steps cycles per lane.
      - SSM hidden state carries across chunks within a segment (value
        preserved, but jax.lax.stop_gradient cuts the backward graph at
        chunk boundaries -> truncated BPTT of depth bptt_steps).
      - When a lane finishes its segment, hidden is reset to zero and a
        new random segment is assigned to that lane.
  * Data: NumpyBatchCache pre-extracts task variables from the prepared
    train/eval xarray Datasets into raw numpy arrays (level subset +
    static-time-drop applied once). Per-chunk batch builds are pure
    np.take + ThreadPoolExecutor parallel over bptt_steps positions.
    Prefetcher overlaps next-chunk data prep with current-chunk GPU train.

Why this design:
  v1 K=1 trains with hidden init=0 every train step (fresh segment of 16,
  full BPTT). SSM weights only ever get gradient signal at hidden depth
  0..16. Eval at K=20+ is OOD for the SSM.
  v4 K=1 keeps hidden across len_segment=32 (or longer), gradient still
  only spans bptt_steps within each train step, so SSM weights see
  hidden at depths 8, 16, 24, 32 across consecutive train steps within
  one segment. Eval rollout K=20+ should be in-distribution for the SSM.

Reuses v3 model factory + group loss + lat weighting + specialist heads.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import pickle
import random
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "scripts" / "training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

V3_DIR = ROOT / "scripts" / "training" / "full_mamba_v3"
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))

import train_graphcast as base_train

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import xarray as xr

# Import v3 helpers verbatim (model factory, feature layout, normalisation
# stats, etc.) so v4 stays compatible with v3 ckpts param tree.
import train_mz_fullmamba_v3 as v3

from src.models.full_mamba import (
    MZResidualFullMambaConfig,
    MZResidualFullMambaMeshed,
)
from src.models.mz_meshed import build_grid_mesh_projections


# ============================================================================
# RunConfig
# ============================================================================


@dataclasses.dataclass
class RunConfigV4:
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
    # K=1 stateful TBPTT segment params
    len_segment: int
    bptt_steps: int
    num_lanes: int
    chunk_load_workers: int
    data_cache_mode: str        # "auto" | "always" | "never"
    data_cache_max_gib: float
    baseline_batch_chunk: int   # mini-batch size for batched baseline forward
    # Optimizer
    max_steps: int
    eval_every: int
    eval_max_segments: int
    checkpoint_every: int
    hidden_size: int
    layers: int
    dropout: float
    lr: float
    weight_decay: float
    grad_clip: float
    warmup_steps: int
    normalize_loss: bool
    standardize_input: bool
    baseline_precision: str
    seed: int
    precision: str
    # Mesh / Mamba
    mz_mesh_size: int
    n_grid_neighbors: int
    n_mesh_neighbors: int
    d_state: int
    expand: int
    a_log_init_min: float
    a_log_init_max: float
    full_variables: bool
    use_specialist_heads: bool
    # Optional three-group loss (default off; legacy single-mean is fine for K=1)
    use_group_loss: bool
    upper_loss_weight: float
    mslp_loss_weight: float
    small_surface_loss_weight: float
    # Resume
    resume_from: str | None
    resume_step: int
    allow_partial_resume: bool


def parse_args() -> RunConfigV4:
    p = argparse.ArgumentParser(description="v4: K=1 stateful TBPTT residual Mamba.")
    p.add_argument("--data-path", default=base_train.DEFAULT_DATA_PATH)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
    p.add_argument("--out-dir", default="artifacts/checkpoints/mz_residual_memory_v4")
    p.add_argument("--run-name", required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--train-start-year", type=int, default=None)
    p.add_argument("--train-end-year", type=int, default=None)
    p.add_argument("--input-duration", default="12h")
    # K=1 stateful TBPTT
    p.add_argument("--len-segment", type=int, default=32,
                   help="Consecutive 6h cycles in one training segment.")
    p.add_argument("--bptt-steps", type=int, default=8,
                   help="Truncated BPTT chunk length. Must divide --len-segment.")
    p.add_argument("--num-lanes", type=int, default=8,
                   help="Independent segments processed in parallel as batch dim.")
    p.add_argument("--chunk-load-workers", type=int, default=6)
    p.add_argument("--data-cache-mode", choices=["auto", "always", "never"], default="auto",
                   help="Controls eager xr.Dataset.load() before NumpyBatchCache build. "
                        "'never' skips the upfront load (lazy zarr access during cache build). "
                        "NumpyBatchCache itself ALWAYS materialises arrays via np.asarray(); "
                        "this flag does not avoid that.")
    p.add_argument("--data-cache-max-gib", type=float, default=80.0,
                   help="(Reserved for auto-mode size gating; not yet enforced.)")
    p.add_argument("--baseline-batch-chunk", type=int, default=8,
                   help="Mini-batch size for batched baseline (GraphCast) forward. "
                        "T·B anchors are split into ceil(T·B / chunk) calls. "
                        "Default 8 fits 80GB at 1° GraphCast; raise to trade memory "
                        "for fewer calls.")
    # Optimizer
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=200,
                   help="Currently a LOG-flush cadence: writes train_log.json. "
                        "There is no inline eval; use scripts/eval_per_level.py "
                        "offline on saved ckpts.")
    p.add_argument("--eval-max-segments", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--no-normalize-loss", action="store_true", default=False)
    p.add_argument("--no-standardize-input", action="store_true", default=False)
    p.add_argument("--baseline-precision", choices=["bf16", "fp32"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    # Mesh / Mamba
    p.add_argument("--mz-mesh-size", type=int, default=5)
    p.add_argument("--n-grid-neighbors", type=int, default=6)
    p.add_argument("--n-mesh-neighbors", type=int, default=3)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--a-log-init-min", type=float, default=-3.0)
    p.add_argument("--a-log-init-max", type=float, default=-0.1)
    p.add_argument("--full-variables", action="store_true", default=False)
    p.add_argument("--specialist-heads", action="store_true", default=False)
    p.add_argument("--use-group-loss", action="store_true", default=False)
    p.add_argument("--upper-loss-weight", type=float, default=1.0)
    p.add_argument("--mslp-loss-weight", type=float, default=1.0)
    p.add_argument("--small-surface-loss-weight", type=float, default=0.1)
    # Resume
    p.add_argument("--resume-from", default=None)
    p.add_argument("--resume-step", type=int, default=0)
    p.add_argument("--allow-partial-resume", action="store_true", default=False)
    args = p.parse_args()

    if args.len_segment % args.bptt_steps != 0:
        raise ValueError(
            f"--len-segment={args.len_segment} must be divisible by "
            f"--bptt-steps={args.bptt_steps}.")
    if args.bptt_steps < 1 or args.num_lanes < 1:
        raise ValueError("--bptt-steps and --num-lanes must be >= 1.")
    if args.use_group_loss and not args.full_variables:
        raise ValueError("--use-group-loss requires --full-variables.")
    if args.specialist_heads and not args.full_variables:
        raise ValueError("--specialist-heads requires --full-variables.")

    return RunConfigV4(
        data_path=args.data_path, baseline_ckpt=args.baseline_ckpt,
        stats_dir=args.stats_dir, out_dir=args.out_dir, run_name=args.run_name,
        resolution=args.resolution, mesh_size=args.mesh_size, val_year=args.val_year,
        train_start_year=args.train_start_year, train_end_year=args.train_end_year,
        input_duration=args.input_duration,
        len_segment=args.len_segment, bptt_steps=args.bptt_steps,
        num_lanes=args.num_lanes, chunk_load_workers=args.chunk_load_workers,
        data_cache_mode=args.data_cache_mode, data_cache_max_gib=args.data_cache_max_gib,
        baseline_batch_chunk=args.baseline_batch_chunk,
        max_steps=args.max_steps, eval_every=args.eval_every,
        eval_max_segments=args.eval_max_segments, checkpoint_every=args.checkpoint_every,
        hidden_size=args.hidden_size, layers=args.layers, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        normalize_loss=(not args.no_normalize_loss),
        standardize_input=(not args.no_standardize_input),
        baseline_precision=args.baseline_precision, seed=args.seed, precision=args.precision,
        mz_mesh_size=args.mz_mesh_size, n_grid_neighbors=args.n_grid_neighbors,
        n_mesh_neighbors=args.n_mesh_neighbors,
        d_state=args.d_state, expand=args.expand,
        a_log_init_min=args.a_log_init_min, a_log_init_max=args.a_log_init_max,
        full_variables=args.full_variables, use_specialist_heads=args.specialist_heads,
        use_group_loss=args.use_group_loss,
        upper_loss_weight=args.upper_loss_weight,
        mslp_loss_weight=args.mslp_loss_weight,
        small_surface_loss_weight=args.small_surface_loss_weight,
        resume_from=args.resume_from, resume_step=args.resume_step,
        allow_partial_resume=args.allow_partial_resume,
    )


# ============================================================================
# NumpyBatchCache: pre-extract task variables from xarray to raw numpy.
# Per-step batch build = np.take(cached_array, gather_indices, axis=time_axis).
# ============================================================================


@dataclasses.dataclass(frozen=True)
class _CachedVar:
    data: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]


def _timedelta_coords(count: int, start_steps: int, dt: pd.Timedelta) -> np.ndarray:
    return np.array([(start_steps + i) * dt for i in range(count)],
                    dtype="timedelta64[ns]")


class NumpyBatchCache:
    """Holds task variables as raw numpy arrays for fast np.take indexing."""

    def __init__(self, ds: xr.Dataset, task_cfg, *, label: str) -> None:
        t0 = time.time()
        self._label = label
        self._vars: dict[str, _CachedVar] = {}
        self._sizes = dict(ds.sizes)
        self._time_size = int(ds.sizes["time"])
        self._coord_values = {
            n: np.asarray(c.values)
            for n, c in ds.coords.items() if n in {"lat", "lon", "level"}
        }
        self._pressure_levels = np.asarray(task_cfg.pressure_levels)
        for name, var in ds.data_vars.items():
            data = np.asarray(var.values)
            dims = tuple(var.dims)
            coords = {d: np.asarray(var.coords[d].values)
                      for d in dims if d in var.coords}
            if "batch" in dims:
                axis = dims.index("batch")
                if data.shape[axis] != 1:
                    raise ValueError(
                        f"[numpy-cache:{label}] non-singleton batch dim for {name}: "
                        f"shape={data.shape}")
                data = np.squeeze(data, axis=axis)
                dims = tuple(d for d in dims if d != "batch")
                coords.pop("batch", None)
            if "level" in dims:
                level_values = coords.get("level", self._coord_values.get("level"))
                if level_values is None:
                    raise ValueError(f"[numpy-cache:{label}] {name} no level coord.")
                missing = [lv for lv in self._pressure_levels
                           if lv not in set(level_values.tolist())]
                if missing:
                    raise ValueError(
                        f"[numpy-cache:{label}] {name} missing levels: {missing}")
                lvl_idx = np.asarray([
                    int(np.where(level_values == lv)[0][0])
                    for lv in self._pressure_levels])
                axis = dims.index("level")
                data = np.take(data, lvl_idx, axis=axis)
                coords["level"] = self._pressure_levels
            self._vars[name] = _CachedVar(data=data, dims=dims, coords=coords)
        total_gib = sum(v.data.nbytes for v in self._vars.values()) / 1024**3
        print(f"[numpy-cache:{label}] {len(self._vars)} vars, "
              f"{total_gib:.2f} GiB in {time.time() - t0:.1f}s")

    def build_batch_from_indices(
        self, *, indices: Iterable[int],
        input_steps: int, target_steps: int,
        task_cfg, dt: pd.Timedelta,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        batch_indices = np.asarray(list(indices), dtype=np.int64)
        if batch_indices.size == 0:
            raise ValueError("empty batch.")
        window_offsets = np.arange(-(input_steps - 1), target_steps + 1, dtype=np.int64)
        window_indices = batch_indices[:, None] + window_offsets[None, :]
        if window_indices.min() < 0 or window_indices.max() >= self._time_size:
            raise IndexError(
                f"batch window outside [0, {self._time_size}): "
                f"min={window_indices.min()} max={window_indices.max()}")

        input_pos = np.arange(input_steps, dtype=np.int64)
        target_pos = np.arange(input_steps, input_steps + target_steps, dtype=np.int64)
        input_time = _timedelta_coords(input_steps, -(input_steps - 1), dt)
        target_time = _timedelta_coords(target_steps, 1, dt)
        full_pos = np.arange(input_steps + target_steps, dtype=np.int64)
        full_time = np.concatenate([input_time, target_time])
        batch_coord = np.arange(batch_indices.size)

        def _build(name: str, positions: np.ndarray, time_coord: np.ndarray):
            cached = self._vars[name]
            dims = cached.dims
            if "time" in dims:
                t_axis = dims.index("time")
                gather = window_indices[:, positions]
                data = np.take(cached.data, gather, axis=t_axis)
                rest = tuple(d for d in dims if d != "time")
                out_dims = ("batch", "time", *rest)
                coords: dict[str, Any] = {"batch": batch_coord, "time": time_coord}
                for d in rest:
                    if d in cached.coords:
                        coords[d] = cached.coords[d]
                    elif d in self._coord_values:
                        coords[d] = self._coord_values[d]
                return xr.DataArray(data, dims=out_dims, coords=coords)
            data = cached.data
            data = np.broadcast_to(data[None, ...], (batch_indices.size, *data.shape))
            out_dims = ("batch", *dims)
            coords = {"batch": batch_coord}
            for d in dims:
                if d in cached.coords:
                    coords[d] = cached.coords[d]
                elif d in self._coord_values:
                    coords[d] = self._coord_values[d]
            return xr.DataArray(data, dims=out_dims, coords=coords)

        in_vars = list(task_cfg.input_variables)
        tg_vars = list(task_cfg.target_variables)
        fc_vars = list(task_cfg.forcing_variables)

        # CRITICAL: GraphCast expects `forcings` to be ONLY at target time
        # positions (length = target_steps), NOT the full input+target window.
        # Mismatching this triggers "conflicting sizes for dimension 'time'"
        # in the autoregressive scan unflatten path. Verified empirically
        # against base_train.build_batch_from_indices output (xr-contract probe).
        inputs = xr.Dataset({n: _build(n, input_pos, input_time)
                             for n in in_vars if n in self._vars})
        targets = xr.Dataset({n: _build(n, target_pos, target_time)
                              for n in tg_vars if n in self._vars})
        forcings = xr.Dataset({n: _build(n, target_pos, target_time)
                               for n in fc_vars if n in self._vars})
        return inputs, targets, forcings


# ============================================================================
# LaneScheduler: assign shuffled segments to num_lanes parallel batch lanes
# ============================================================================


class LaneScheduler:
    def __init__(self, segments: list[np.ndarray], *, num_lanes: int,
                 bptt_steps: int, seed: int) -> None:
        if not segments:
            raise ValueError("no training segments.")
        self._segments = segments
        self._num_lanes = num_lanes
        self._bptt_steps = bptt_steps
        self._rng = np.random.default_rng(seed)
        self._active: list[np.ndarray | None] = [None] * num_lanes
        self._offsets = np.zeros(num_lanes, dtype=np.int64)
        self.epoch = 0
        self._order = np.arange(len(segments), dtype=np.int64)
        self._cursor = len(segments)

    def _reshuffle(self):
        self._order = np.arange(len(self._segments), dtype=np.int64)
        self._rng.shuffle(self._order)
        self._cursor = 0
        self.epoch += 1

    def _next_segment(self) -> np.ndarray:
        if self._cursor >= len(self._order):
            self._reshuffle()
        seg = self._segments[int(self._order[self._cursor])]
        self._cursor += 1
        return seg

    def next_chunk(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (chunk_indices [bptt_steps, num_lanes], reset_mask [num_lanes])."""
        reset_mask = np.zeros(self._num_lanes, dtype=bool)
        chunk = np.zeros((self._bptt_steps, self._num_lanes), dtype=np.int64)
        for lane in range(self._num_lanes):
            seg = self._active[lane]
            offset = int(self._offsets[lane])
            if seg is None or offset + self._bptt_steps > len(seg):
                seg = self._next_segment()
                self._active[lane] = seg
                offset = 0
                self._offsets[lane] = 0
                reset_mask[lane] = True
            for i in range(self._bptt_steps):
                chunk[i, lane] = int(seg[offset + i])
            self._offsets[lane] = offset + self._bptt_steps
        return chunk, reset_mask


# ============================================================================
# Segment construction: fixed-length contiguous runs of valid indices
# ============================================================================


def build_full_segments(indices: np.ndarray, len_segment: int) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    sorted_idx = np.sort(indices)
    gaps = np.where(np.diff(sorted_idx) > 1)[0] + 1
    runs = np.split(sorted_idx, gaps)
    segments: list[np.ndarray] = []
    for run in runs:
        for start in range(0, len(run) - len_segment + 1, len_segment):
            seg = run[start : start + len_segment]
            if len(seg) == len_segment:
                segments.append(seg)
    return segments


# ============================================================================
# Build chunk tensors using NumpyBatchCache + parallel workers
# ============================================================================


def _build_chunk_tensors(
    cache: NumpyBatchCache,
    baseline_predict, baseline_params, baseline_state,
    rng,
    chunk_indices: np.ndarray,         # [T=bptt_steps, B=num_lanes]
    *,
    input_steps: int, task_cfg, dt,
    feature_order: tuple[str, ...],
    executor: concurrent.futures.ThreadPoolExecutor,
    baseline_batch_chunk: int = 8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (current_state, baseline_next, truth_next) each [T, B, lat, lon, F].

    OPTIMIZATION vs sequential T calls: batch the T·B anchors and call
    baseline_predict.apply on mini-batches of size baseline_batch_chunk. With
    default T=8, B=4, chunk=8 → T·B=32 split into 4 calls instead of T=8.
    Single-call (T·B=32 in one forward) OOMs at 1° GraphCast on 80GB; chunk=8
    fits with margin.
    """
    T, B = chunk_indices.shape
    flat_n = T * B
    chunk = max(1, min(baseline_batch_chunk, flat_n))

    def _one_pos(t: int):
        return cache.build_batch_from_indices(
            indices=chunk_indices[t],
            input_steps=input_steps, target_steps=1,
            task_cfg=task_cfg, dt=dt,
        )
    futures = [executor.submit(_one_pos, t) for t in range(T)]
    per_pos = [f.result() for f in futures]   # list of (inputs, targets, forcings)

    # xr.concat along "batch" stacks the T per-position datasets into one
    # batch=T·B. assign_coords overrides duplicate batch indices (each per-pos
    # has batch=[0..B-1]; need fresh contiguous 0..T·B-1).
    flat_inputs = xr.concat([p[0] for p in per_pos], dim="batch").assign_coords(
        batch=np.arange(flat_n))
    flat_targets = xr.concat([p[1] for p in per_pos], dim="batch").assign_coords(
        batch=np.arange(flat_n))
    flat_forcings = xr.concat([p[2] for p in per_pos], dim="batch").assign_coords(
        batch=np.arange(flat_n))

    # Mini-batched baseline forward: split [0, flat_n) into windows of size
    # `chunk` and concatenate predictions along batch axis. Each call holds at
    # most `chunk` GraphCast forwards in GPU memory at once.
    pred_blocks = []
    for start in range(0, flat_n, chunk):
        stop = min(start + chunk, flat_n)
        sl_inputs = flat_inputs.isel(batch=slice(start, stop))
        sl_targets = flat_targets.isel(batch=slice(start, stop))
        sl_forcings = flat_forcings.isel(batch=slice(start, stop))
        sl_inputs_jax = v3._to_jax_dataset(sl_inputs)
        sl_targets_jax = v3._to_jax_dataset(sl_targets)
        sl_forcings_jax = v3._to_jax_dataset(sl_forcings)
        rng, key = jax.random.split(rng)
        sl_preds, _ = baseline_predict.apply(
            baseline_params, baseline_state, key,
            sl_inputs_jax, sl_targets_jax, sl_forcings_jax, False,
        )
        pred_blocks.append(v3._extract_feature_block(
            sl_preds, time_index=0,
            task_cfg=task_cfg, feature_order=feature_order))

    # Each block: [chunk_n, lat, lon, F] (last block may be smaller).
    baseline_flat = jnp.concatenate(pred_blocks, axis=0)

    # Truth and current_state are read directly off the xarray objects (no
    # GraphCast call); cheap, do once on the full flat batch.
    truth_flat = v3._extract_feature_block(
        flat_targets, time_index=0,
        task_cfg=task_cfg, feature_order=feature_order)
    cs_flat = v3._extract_feature_block(
        flat_inputs, time_index=-1,
        task_cfg=task_cfg, feature_order=feature_order)

    # Reshape [T·B, lat, lon, F] -> [T, B, lat, lon, F]. xr.concat dim="batch"
    # stacks T position-blocks contiguously, so flat[t·B + lane] = (t, lane),
    # which is exactly what reshape(T, B, ...) produces in row-major order.
    truth = truth_flat.reshape(T, B, *truth_flat.shape[1:])
    baseline = baseline_flat.reshape(T, B, *baseline_flat.shape[1:])
    current_state = cs_flat.reshape(T, B, *cs_flat.shape[1:])
    return current_state, baseline, truth


# ============================================================================
# Main
# ============================================================================


def _save_ckpt(out_dir: Path, step: int, params: hk.Params) -> None:
    p = out_dir / f"mz_residual_step{step}.pkl"
    with p.open("wb") as f:
        pickle.dump(params, f)
    print(f"saved ckpt: {p}")


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.full_variables:
        v3.RESOLVED_VARIABLES = v3.RESOLVED_VARIABLES_FULL
    else:
        v3.RESOLVED_VARIABLES = v3.RESOLVED_VARIABLES_MIN
    print(f"[vars] full_variables={cfg.full_variables}  n_vars={len(v3.RESOLVED_VARIABLES)}")
    print(f"[v4] len_segment={cfg.len_segment} bptt_steps={cfg.bptt_steps} "
          f"num_lanes={cfg.num_lanes} chunks/segment={cfg.len_segment // cfg.bptt_steps}")

    ckpt = base_train.load_graphcast_checkpoint(Path(cfg.baseline_ckpt))
    task_cfg = ckpt.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg = dataclasses.replace(
        ckpt.model_config, resolution=cfg.resolution, mesh_size=cfg.mesh_size)
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

    # NOTE: NumpyBatchCache.__init__ calls np.asarray(var.values) which itself
    # triggers materialisation of any lazy zarr-backed xarray var. Calling
    # ds.load() upfront duplicates memory (xarray copy + numpy copy held
    # simultaneously) and OOMs on 160G nodes for 2-year 1° 11-var data. We
    # skip the eager load and let the cache build do the materialisation
    # incrementally, then drop train_ds so GC reclaims the xarray copies.
    # eval_ds stays lazy: we only need its time coord and lat/lon for
    # segment construction and mesh projection. No eval_cache during training;
    # offline eval uses scripts/eval_per_level.py on saved ckpts.

    dt = base_train.infer_time_step(train_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    # Capture all train/eval metadata BEFORE building the cache + nulling
    # train_ds. After the cache is built we want to reclaim the xarray copies.
    lat_deg = np.asarray(eval_ds.lat.values, dtype=np.float64)
    lon_deg = np.asarray(eval_ds.lon.values, dtype=np.float64)
    n_lat = int(eval_ds.sizes["lat"])
    n_lon = int(eval_ds.sizes["lon"])

    train_indices_raw = base_train.valid_final_input_indices(
        int(train_ds.sizes["time"]), input_steps, 1)
    eval_indices_raw = base_train.valid_final_input_indices(
        int(eval_ds.sizes["time"]), input_steps, 1)
    train_indices = v3._filter_time_continuous_indices(
        train_ds, train_indices_raw, input_steps=input_steps, target_steps=1, dt=dt)
    eval_indices = v3._filter_time_continuous_indices(
        eval_ds, eval_indices_raw, input_steps=input_steps, target_steps=1, dt=dt)
    train_segments = build_full_segments(train_indices, cfg.len_segment)
    eval_segments = build_full_segments(eval_indices, cfg.len_segment)
    print(f"[segments] train={len(train_segments)} eval={len(eval_segments)} "
          f"len={cfg.len_segment}")
    if not train_segments:
        raise ValueError("No train segments; reduce --len-segment.")

    feature_order, feature_slices, feature_dim = v3._resolved_feature_layout(task_cfg)
    diffs_std_f_np = v3._build_diffs_stddev_vector(task_cfg, norm_stats, feature_order)
    mean_f_np, std_f_np = v3._build_mean_stddev_vectors(task_cfg, norm_stats, feature_order)
    # Lat coord is shared between train + eval; use eval_ds (kept lazy).
    lat_weights_np = v3._build_latitude_weights(np.asarray(eval_ds.lat.values, dtype=np.float32))

    # Now build the train cache (materialises numpy arrays from xarray) and
    # drop train_ds so its in-memory copy can be GC'd. eval_ds stays lazy
    # through the rest of training (we never read its data, only its coords).
    print(f"[data] building NumpyBatchCache for train (mode={cfg.data_cache_mode})...")
    train_cache = NumpyBatchCache(train_ds, task_cfg, label="v4-train")
    # train_ds is dropped LATER, after baseline-init's sample batch is built
    # using base_train.build_batch_from_indices (the proven xarray path,
    # whose time-coord layout is what GraphCast's autoregressive scan
    # expects on init). Our NumpyBatchCache is correct for the chunk-time
    # train forward but the init path is more sensitive to coord conventions.
    per_channel_loss_w_np = v3._build_per_channel_loss_weights(
        task_cfg, feature_order, feature_slices)
    lat_weights = jnp.asarray(lat_weights_np, dtype=jnp.float32)
    per_channel_loss_w = jnp.asarray(per_channel_loss_w_np, dtype=jnp.float32)
    loss_norm_f = (jnp.asarray(diffs_std_f_np, dtype=jnp.float32)
                   if cfg.normalize_loss
                   else jnp.ones((feature_dim,), dtype=jnp.float32))
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

    # Three-group indices
    _SMALL = {"2m_temperature", "10m_u_component_of_wind",
              "10m_v_component_of_wind", "total_precipitation_6hr"}
    upper_idx_list, mslp_idx_list, small_idx_list = [], [], []
    for n in feature_order:
        sl = feature_slices[n]
        ix = list(range(sl.start, sl.stop))
        if n == "mean_sea_level_pressure":
            mslp_idx_list.extend(ix)
        elif n in _SMALL:
            small_idx_list.extend(ix)
        else:
            upper_idx_list.extend(ix)
    upper_idx_jnp = jnp.asarray(upper_idx_list or list(range(feature_dim)), dtype=jnp.int32)
    mslp_idx_jnp = jnp.asarray(mslp_idx_list, dtype=jnp.int32) if mslp_idx_list else None
    small_idx_jnp = jnp.asarray(small_idx_list, dtype=jnp.int32) if small_idx_list else None

    # Frozen baseline GraphCast
    def baseline_predict_fn(inputs, targets, forcings, is_training):
        predictor = base_train.build_predictor(
            model_cfg, task_cfg, norm_stats,
            use_bf16=(cfg.baseline_precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone="none", temporal_location="mesh_post_encoder",
            temporal_hidden_size=model_cfg.latent_size,
            temporal_layers=1, temporal_dropout=0.0)
        return predictor(inputs, targets_template=targets, forcings=forcings,
                         is_training=is_training)
    baseline_predict = hk.transform_with_state(baseline_predict_fn)
    rng = jax.random.PRNGKey(cfg.seed)
    # Use base_train's xarray builder (still alive) for the init batch — its
    # time-coord layout matches what GraphCast's autoregressive scan expects.
    s_in, s_tg, s_fc = base_train.build_batch_from_indices(
        train_ds, indices=[int(train_indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
    s_in = v3._to_jax_dataset(s_in); s_tg = v3._to_jax_dataset(s_tg); s_fc = v3._to_jax_dataset(s_fc)
    _, baseline_state = baseline_predict.init(rng, s_in, s_tg, s_fc, False)
    baseline_params = ckpt.params

    # XR-CONTRACT DEBUG PROBE: print metadata for ORIG base_train path vs
    # CACHE NumpyBatchCache path on the same anchor index, then exit. Run
    # while train_ds is still alive (we drop it below for normal training
    # but the smoke run flag short-circuits there).
    _DEBUG_XR_CONTRACT = False  # flip to True to print ORIG vs CACHE diff and exit
    if _DEBUG_XR_CONTRACT and train_ds is not None:
        def _print_xr_contract(tag, inputs, targets, forcings):
            print(f"\n========== XR CONTRACT: {tag} ==========")
            for name, ds in [("inputs", inputs), ("targets", targets), ("forcings", forcings)]:
                print(f"\n[{tag}] {name}")
                print(f"  sizes: {dict(ds.sizes)}")
                print(f"  dims:  {dict(ds.dims)}")
                if "time" in ds.coords:
                    tv = ds.time.values
                    print(f"  time.values: {tv}")
                    print(f"  time.shape:  {tv.shape}")
                    print(f"  time.dtype:  {tv.dtype}")
                    print(f"  time.dims:   {ds.time.dims}")
                else:
                    print("  time: <NO TIME COORD>")
            for var in ["geopotential", "mean_sea_level_pressure"]:
                print(f"\n[{tag}] data var: {var}")
                for ds_name, ds in [("inputs", inputs), ("targets", targets), ("forcings", forcings)]:
                    if var not in ds:
                        print(f"  {ds_name}: <not present>")
                        continue
                    da = ds[var]
                    print(f"  {ds_name}: dims={da.dims} shape={da.shape} coords={list(da.coords)}")
                    if "time" in da.coords:
                        print(f"    time.values: {da.time.values}  shape: {da.time.values.shape}  dtype: {da.time.values.dtype}  dims: {da.time.dims}")
                    if "level" in da.coords:
                        print(f"    level.values: {da.level.values}  dtype: {da.level.values.dtype}")
            print(f"========== END XR CONTRACT: {tag} ==========\n")

        idx0 = int(train_indices[0])
        print("\n[xr-contract] ORIG base_train.build_batch_from_indices...")
        orig_in, orig_tg, orig_fc = base_train.build_batch_from_indices(
            train_ds, indices=[idx0],
            input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
        _print_xr_contract("ORIG base_train", orig_in, orig_tg, orig_fc)

        print("[xr-contract] CACHE NumpyBatchCache.build_batch_from_indices...")
        cache_in, cache_tg, cache_fc = train_cache.build_batch_from_indices(
            indices=[idx0],
            input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
        _print_xr_contract("CACHE numpy", cache_in, cache_tg, cache_fc)

        def _compare_time(orig, cache, name):
            print(f"\n--- compare {name}.time ---")
            print(f"  orig:  values={orig.time.values}  shape={orig.time.values.shape}  dtype={orig.time.values.dtype}  dims={orig.time.dims}")
            print(f"  cache: values={cache.time.values}  shape={cache.time.values.shape}  dtype={cache.time.values.dtype}  dims={cache.time.dims}")
            try:
                np.testing.assert_array_equal(orig.time.values, cache.time.values)
                print(f"  time values: OK")
            except Exception as e:
                print(f"  time values MISMATCH: {repr(e)[:200]}")
        _compare_time(orig_in, cache_in, "inputs")
        _compare_time(orig_tg, cache_tg, "targets")
        _compare_time(orig_fc, cache_fc, "forcings")
        print("[xr-contract] DONE — exiting before chunk init.")
        sys.stdout.flush()
        sys.exit(0)

    # Now drop train_ds (cache holds the data); peak host RAM should drop
    # back to ~cache size + JAX device overhead.
    train_ds = None
    import gc
    gc.collect()

    # MZ residual Mamba head (lat_deg / lon_deg captured earlier)
    proj_arrays, n_mesh_nodes = build_grid_mesh_projections(
        lat_deg=lat_deg, lon_deg=lon_deg, mesh_size=cfg.mz_mesh_size,
        n_grid_neighbors=cfg.n_grid_neighbors, n_mesh_neighbors=cfg.n_mesh_neighbors)
    print(f"[meshed] n_mesh_nodes={n_mesh_nodes}")

    if cfg.use_specialist_heads:
        surface_names = {"2m_temperature", "mean_sea_level_pressure",
                         "10m_u_component_of_wind", "10m_v_component_of_wind",
                         "total_precipitation_6hr"}
        upper_h, surface_h = [], []
        for n in feature_order:
            sl = feature_slices[n]
            (surface_h if n in surface_names else upper_h).extend(range(sl.start, sl.stop))
        upper_idx_t = tuple(upper_h); surface_idx_t = tuple(surface_h)
    else:
        upper_idx_t, surface_idx_t = (), ()

    mz_cfg = MZResidualFullMambaConfig(
        input_size=feature_dim * 2, output_size=feature_dim,
        hidden_size=cfg.hidden_size, d_state=cfg.d_state, expand=cfg.expand,
        layers=cfg.layers, dropout=cfg.dropout,
        a_log_init_min=cfg.a_log_init_min, a_log_init_max=cfg.a_log_init_max,
        processor_layers=0,
        use_specialist_heads=cfg.use_specialist_heads,
        upper_channel_indices=upper_idx_t,
        surface_channel_indices=surface_idx_t,
    )

    def _build_mz():
        return MZResidualFullMambaMeshed(
            mz_cfg, n_mesh_nodes=n_mesh_nodes,
            mesh_senders=None, mesh_receivers=None, **proj_arrays)

    residual_to_state_rescale_f = output_denorm_f / input_std_f

    def _normalize_inputs(seq_inputs):
        cs, pr = jnp.split(seq_inputs, 2, axis=-1)
        cs_n = (cs - input_mean_f[None, None, None, None, :]) / input_std_f[None, None, None, None, :]
        pr_n = pr / residual_input_std_f[None, None, None, None, :]
        return cs_n, pr_n

    def _pred_to_real(pred_n):
        return pred_n * output_denorm_f[None, None, None, None, :]

    def _finalise_loss(pred_residual, baseline_next, truth_next):
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
        weighted_sq_group = jnp.square(norm_diff) * w_lat
        def _gm(arr, idx):
            if idx is None:
                return jnp.asarray(0.0, dtype=arr.dtype)
            return jnp.mean(jnp.take(arr, idx, axis=-1))
        loss_upper = _gm(weighted_sq_group, upper_idx_jnp)
        loss_mslp = _gm(weighted_sq_group, mslp_idx_jnp)
        loss_small = _gm(weighted_sq_group, small_idx_jnp)
        if cfg.use_group_loss:
            den = (cfg.upper_loss_weight + cfg.mslp_loss_weight + cfg.small_surface_loss_weight)
            total_loss = (cfg.upper_loss_weight * loss_upper
                          + cfg.mslp_loss_weight * loss_mslp
                          + cfg.small_surface_loss_weight * loss_small) / den
        else:
            total_loss = state_loss
        return (corrected, total_loss, state_loss, residual_loss,
                loss_upper, loss_mslp, loss_small)

    def chunk_objective(seq_inputs, baseline_next, truth_next, h_in, is_training):
        cs_n, tpr_n = _normalize_inputs(seq_inputs)
        m = _build_mz()
        baseline_abs_n = (baseline_next - input_mean_f[None, None, None, None, :]) / input_std_f[None, None, None, None, :]
        T = cs_n.shape[0]
        tf_mask = jnp.ones((T,), dtype=jnp.float32)
        pred_residual_n, h_new = m.rollout_ar(
            cs_n, is_training=is_training,
            true_prev_residual_n_tblnf=tpr_n,
            tf_mask_per_step=tf_mask,
            baseline_absolute_n_tblnf=baseline_abs_n,
            residual_to_state_rescale_f=residual_to_state_rescale_f,
            allow_tf_at_t0=True,
            initial_h_states=h_in,
            return_final_h_states=True,
        )
        pred_residual = _pred_to_real(pred_residual_n)
        (corrected, total_loss, state_loss, residual_loss,
         loss_upper, loss_mslp, loss_small) = _finalise_loss(
            pred_residual, baseline_next, truth_next)
        return {
            "total_loss": total_loss, "state_loss": state_loss,
            "residual_loss": residual_loss, "loss_upper": loss_upper,
            "loss_mslp": loss_mslp, "loss_small_surface": loss_small,
            "corrected": corrected, "h_new": h_new,
        }

    residual_model = hk.transform(chunk_objective)

    # Init
    init_chunk = np.zeros((cfg.bptt_steps, 1), dtype=np.int64)
    for t in range(cfg.bptt_steps):
        init_chunk[t, 0] = int(train_indices[t]) if t < len(train_indices) else int(train_indices[0])
    rng, k1 = jax.random.split(rng)
    load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.chunk_load_workers)

    # =====================================================================
    # XR-CONTRACT DEBUG PROBE (v4 cache vs base_train original)
    # Diagnoses why cache-built dataset trips GraphCast autoregressive scan
    # unflatten. Prints metadata for both paths on the same anchor index.
    # train_ds is still alive here (we kept it for baseline init); we use it
    # for the ORIG path comparison, then drop it.
    # =====================================================================
    if train_ds is not None:
        def _print_xr_contract(tag, inputs, targets, forcings):
            print(f"\n========== XR CONTRACT: {tag} ==========")
            for name, ds in [("inputs", inputs), ("targets", targets), ("forcings", forcings)]:
                print(f"\n[{tag}] {name}")
                print(f"  sizes: {dict(ds.sizes)}")
                print(f"  dims:  {dict(ds.dims)}")
                if "time" in ds.coords:
                    tv = ds.time.values
                    print(f"  time.values: {tv}")
                    print(f"  time.shape:  {tv.shape}")
                    print(f"  time.dtype:  {tv.dtype}")
                    print(f"  time.dims:   {ds.time.dims}")
                else:
                    print("  time: <NO TIME COORD>")
            for var in ["geopotential", "mean_sea_level_pressure"]:
                print(f"\n[{tag}] data var: {var}")
                for ds_name, ds in [("inputs", inputs), ("targets", targets), ("forcings", forcings)]:
                    if var not in ds:
                        print(f"  {ds_name}: <not present>")
                        continue
                    da = ds[var]
                    print(f"  {ds_name}: dims={da.dims}  shape={da.shape}  coords={list(da.coords)}")
                    if "time" in da.coords:
                        print(f"    time.values: {da.time.values}  shape: {da.time.values.shape}  dtype: {da.time.values.dtype}  dims: {da.time.dims}")
                    if "level" in da.coords:
                        print(f"    level.values: {da.level.values}  dtype: {da.level.values.dtype}")
            print(f"========== END XR CONTRACT: {tag} ==========\n")

        idx0 = int(train_indices[0])
        print("\n[xr-contract] running ORIG base_train.build_batch_from_indices...")
        orig_in, orig_tg, orig_fc = base_train.build_batch_from_indices(
            train_ds, indices=[idx0],
            input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
        _print_xr_contract("ORIG base_train", orig_in, orig_tg, orig_fc)

        print("[xr-contract] running CACHE NumpyBatchCache.build_batch_from_indices...")
        cache_in, cache_tg, cache_fc = train_cache.build_batch_from_indices(
            indices=[idx0],
            input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
        _print_xr_contract("CACHE numpy", cache_in, cache_tg, cache_fc)

        def _compare_time(orig, cache, name):
            print(f"\n--- compare {name}.time ---")
            print(f"  orig:  values={orig.time.values}  shape={orig.time.values.shape}  dtype={orig.time.values.dtype}  dims={orig.time.dims}")
            print(f"  cache: values={cache.time.values}  shape={cache.time.values.shape}  dtype={cache.time.values.dtype}  dims={cache.time.dims}")
            try:
                np.testing.assert_array_equal(orig.time.values, cache.time.values)
                print(f"  time values: OK")
            except Exception as e:
                print(f"  time values MISMATCH: {repr(e)[:200]}")
        _compare_time(orig_in, cache_in, "inputs")
        _compare_time(orig_tg, cache_tg, "targets")
        _compare_time(orig_fc, cache_fc, "forcings")
        print("[xr-contract] DONE — exiting before chunk init to keep smoke fast.")
        sys.stdout.flush()
        sys.exit(0)
    # =====================================================================

    cs_init, base_init, truth_init = _build_chunk_tensors(
        train_cache, baseline_predict, baseline_params, baseline_state, k1,
        init_chunk, input_steps=input_steps, task_cfg=task_cfg, dt=dt,
        feature_order=feature_order, executor=load_executor,
        baseline_batch_chunk=cfg.baseline_batch_chunk)
    seq_inputs_init = jnp.concatenate([cs_init, jnp.zeros_like(cs_init)], axis=-1)
    rng, k2 = jax.random.split(rng)
    mem_params = residual_model.init(
        k2, seq_inputs_init, base_init, truth_init, None, True)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(mem_params))
    print(f"[init] residual model params = {n_params:,}")

    # Resume
    if cfg.resume_from is not None:
        cp = Path(cfg.resume_from)
        if not cp.exists():
            raise FileNotFoundError(f"--resume-from missing: {cp}")
        with cp.open("rb") as f:
            loaded = pickle.load(f)
        if cfg.allow_partial_resume:
            init_flat = hk.data_structures.to_mutable_dict(mem_params)
            loaded_flat = hk.data_structures.to_mutable_dict(loaded)
            n_copy = n_init_only = n_ckpt_only = 0
            for mk in set(init_flat) & set(loaded_flat):
                for pk in init_flat[mk]:
                    if pk in loaded_flat[mk] and loaded_flat[mk][pk].shape == init_flat[mk][pk].shape:
                        init_flat[mk][pk] = loaded_flat[mk][pk]
                        n_copy += 1
                    else:
                        n_init_only += 1
                for pk in loaded_flat[mk]:
                    if pk not in init_flat[mk]:
                        n_ckpt_only += 1
            mem_params = hk.data_structures.to_immutable_dict(init_flat)
            print(f"[resume:partial] copy={n_copy} init_only={n_init_only} ckpt_only={n_ckpt_only}")
        else:
            mem_params = loaded
            print(f"[resume] strict load from {cp}")

    # Optimizer
    if cfg.warmup_steps > 0:
        lr_schedule = optax.linear_schedule(
            init_value=0.0, end_value=cfg.lr, transition_steps=cfg.warmup_steps)
    else:
        lr_schedule = cfg.lr
    opt_chain = []
    if cfg.grad_clip > 0:
        opt_chain.append(optax.clip_by_global_norm(cfg.grad_clip))
    opt_chain.append(optax.adamw(lr_schedule, weight_decay=cfg.weight_decay))
    opt = optax.chain(*opt_chain) if len(opt_chain) > 1 else opt_chain[0]
    opt_state = opt.init(mem_params)

    @jax.jit
    def train_chunk(params, opt_state, key, seq_inputs, baseline_next, truth_next, h_in):
        def loss_fn(p):
            outs = residual_model.apply(p, key, seq_inputs, baseline_next, truth_next, h_in, True)
            return outs["total_loss"], outs
        (loss, outs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        h_stop = jax.tree_util.tree_map(jax.lax.stop_gradient, outs["h_new"])
        return (new_params, new_opt_state, loss, outs["state_loss"],
                outs["residual_loss"], outs["loss_upper"], outs["loss_mslp"],
                outs["loss_small_surface"], grad_norm, h_stop)

    scheduler = LaneScheduler(
        train_segments, num_lanes=cfg.num_lanes,
        bptt_steps=cfg.bptt_steps, seed=cfg.seed)

    M = n_mesh_nodes
    D_inner = cfg.hidden_size * cfg.expand
    lane_h: list[jax.Array] = [
        jnp.zeros((cfg.num_lanes * M, D_inner, cfg.d_state), dtype=jnp.float32)
        for _ in range(cfg.layers)
    ]
    # Per-lane previous-residual carry: at chunk t=0 of each lane, prev_residual
    # is this lane's last-cycle residual (truth − baseline at the previous 6h
    # cycle). Carried across chunks within a segment, reset to zero on lane
    # reset (alongside lane_h). Stored in REAL units; _normalize_inputs divides
    # by residual_input_std_f when fed into the model. This recovers v1-style
    # teacher-forced previous-residual input + hidden cycling memory together.
    # n_lat / n_lon captured earlier from eval_ds before cache build.
    lane_prev_residual: jax.Array = jnp.zeros(
        (cfg.num_lanes, n_lat, n_lon, feature_dim), dtype=jnp.float32)

    cfg_dict = dataclasses.asdict(cfg)
    # eval_per_level.py expects these architecture flags. v4 IS the full_mamba
    # meshed model by construction; surface them so the eval script can
    # dispatch correctly without an extra CLI override.
    cfg_dict["meshed"] = True
    cfg_dict["full_mamba"] = True
    # eval_per_level.py also reads these v3-era fields. Map v4 semantics:
    cfg_dict["segment_steps"] = cfg.len_segment   # anchors per segment
    cfg_dict["target_steps"] = 1                  # v4 is K=1 always
    cfg_dict["anchor_as_batch"] = True            # match v3 PARITY layout
    cfg_dict["anchor_as_batch_min_k"] = 2         # forced-True when warm>0 anyway
    payload = {
        "config": cfg_dict,
        "resolved_variables": list(v3.RESOLVED_VARIABLES),
        "feature_slices": {k: [v.start, v.stop] for k, v in feature_slices.items()},
        "task_config": dataclasses.asdict(task_cfg),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(payload, f, indent=2)

    train_log: list[dict[str, Any]] = []
    eval_log: list[dict[str, Any]] = []
    if cfg.resume_from is not None:
        for fn in ("train_log.json", "eval_log.json"):
            p = out_dir / fn
            if p.exists():
                with p.open() as f:
                    pl = json.load(f)
                if fn == "train_log.json":
                    train_log = [e for e in pl if e.get("step", 0) <= cfg.resume_step]
                else:
                    eval_log = [e for e in pl if e.get("step", 0) <= cfg.resume_step]

    prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _prep_next_chunk():
        nonlocal rng
        chunk_idx, reset_mask = scheduler.next_chunk()
        rng, prep_key = jax.random.split(rng)
        cs, base_, truth = _build_chunk_tensors(
            train_cache, baseline_predict, baseline_params, baseline_state, prep_key,
            chunk_idx, input_steps=input_steps, task_cfg=task_cfg, dt=dt,
            feature_order=feature_order, executor=load_executor,
            baseline_batch_chunk=cfg.baseline_batch_chunk)
        # Return raw cs (no seq_inputs concat). Main loop builds seq_inputs
        # AFTER applying lane_prev_residual carry (which the prefetch worker
        # doesn't have access to since it depends on previous chunk output).
        return cs, base_, truth, reset_mask, scheduler.epoch

    pending = prefetch_executor.submit(_prep_next_chunk)

    start_step = cfg.resume_step + 1 if cfg.resume_from is not None else 1
    print(f"[train] start step {start_step}, target {cfg.max_steps}")
    for step in range(start_step, cfg.max_steps + 1):
        t0_data = time.time()
        cs, baseline_next, truth_next, reset_mask, _ = pending.result()
        data_wait = time.time() - t0_data
        next_pending = prefetch_executor.submit(_prep_next_chunk) if step + 1 <= cfg.max_steps else None

        # Lane reset: zero hidden state AND prev_residual carry for any lane
        # that started a fresh segment this step.
        if reset_mask.any():
            mask_jnp = jnp.asarray(reset_mask, dtype=jnp.bool_)
            new_lane_h: list[jax.Array] = []
            for h in lane_h:
                h_r = h.reshape(cfg.num_lanes, M, *h.shape[1:])
                mask_b = mask_jnp.reshape(cfg.num_lanes, *([1] * (h_r.ndim - 1)))
                h_r = jnp.where(mask_b, jnp.zeros_like(h_r), h_r)
                new_lane_h.append(h_r.reshape(h.shape))
            lane_h = new_lane_h
            # Reset prev_residual on the same mask (broadcast over lat,lon,F).
            mask_b4 = mask_jnp.reshape(cfg.num_lanes, 1, 1, 1)
            lane_prev_residual = jnp.where(
                mask_b4, jnp.zeros_like(lane_prev_residual), lane_prev_residual)

        # Build prev_residual tensor for the chunk: at t=0 use the lane's
        # carried previous-residual; at t=1..T-1 use the prior chunk position's
        # truth − baseline (within-chunk teacher-forced). Both ingredients are
        # deployment-observable (real obs at the corresponding 6h cycle).
        # resid: [T, B, lat, lon, F] in REAL units.
        resid = truth_next - baseline_next
        prev_residual_chunk = jnp.concatenate(
            [lane_prev_residual[None, ...], resid[:-1]], axis=0)
        seq_inputs = jnp.concatenate([cs, prev_residual_chunk], axis=-1)

        # Diagnostic: hidden-state norm BEFORE the chunk forward.
        # Should be 0 on the very first chunk of any lane (reset_mask=True),
        # and non-zero from chunk 2 onward of the same segment (TBPTT carry).
        # If h_norm_before stays 0 step after step, hidden is being reset
        # every time and TBPTT is silently broken.
        h_norm_before = float(sum(
            float(jnp.sqrt(jnp.sum(jnp.square(h)))) for h in lane_h))
        prev_resid_norm_before = float(jnp.sqrt(jnp.sum(jnp.square(lane_prev_residual))))

        rng, step_key = jax.random.split(rng)
        t0 = time.time()
        (mem_params, opt_state, loss, state_loss, residual_loss,
         loss_upper, loss_mslp, loss_small, grad_norm, lane_h) = train_chunk(
            mem_params, opt_state, step_key,
            seq_inputs, baseline_next, truth_next, lane_h)
        gpu_time = time.time() - t0

        # Carry the LAST cycle's residual to next chunk's t=0. stop_gradient
        # to keep the backward graph confined to within-chunk TBPTT.
        lane_prev_residual = jax.lax.stop_gradient(resid[-1])

        # h_norm AFTER the chunk forward (in stop_gradient'd next-chunk init).
        h_norm_after = float(sum(
            float(jnp.sqrt(jnp.sum(jnp.square(h)))) for h in lane_h))

        train_log.append({
            "step": step, "loss": float(loss),
            "state_loss": float(state_loss),
            "residual_loss": float(residual_loss),
            "loss_upper": float(loss_upper),
            "loss_mslp": float(loss_mslp),
            "loss_small_surface": float(loss_small),
            "grad_norm": float(grad_norm),
            "data_wait_s": data_wait,
            "gpu_train_s": gpu_time,
            "lane_resets": int(reset_mask.sum()),
            "h_norm_before": h_norm_before,
            "h_norm_after": h_norm_after,
            "prev_resid_norm_before": prev_resid_norm_before,
        })
        # Verbose log first 6 chunks (covers full 1st segment under
        # len_segment=32, bptt_steps=8: 4 chunks per lane segment) then every 10.
        if step == start_step or step <= 6 or step % 10 == 0:
            print(f"step {step}/{cfg.max_steps} loss {float(loss):.5f} "
                  f"state {float(state_loss):.5f} grad_norm {float(grad_norm):.4f} "
                  f"h_before {h_norm_before:.3e} h_after {h_norm_after:.3e} "
                  f"prev_r {prev_resid_norm_before:.3e} "
                  f"data {data_wait:.2f}s gpu {gpu_time:.2f}s "
                  f"resets {int(reset_mask.sum())}/{cfg.num_lanes}")

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_log, f, indent=2)

        if step % cfg.checkpoint_every == 0 or step == cfg.max_steps:
            _save_ckpt(out_dir, step, mem_params)

        pending = next_pending if next_pending is not None else pending

    prefetch_executor.shutdown(wait=False)
    load_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
