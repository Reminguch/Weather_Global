#!/usr/bin/env python3
"""Train GraphCast/Mamba on shuffled chronological segments with chunked BPTT."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_graphcast as base  # noqa: E402


@dataclasses.dataclass(frozen=True)
class SegmentRunConfig:
    base_cfg: base.RunConfig
    len_segment: int
    bptt_steps: int


class SegmentBatchScheduler:
    """Assign shuffled chronological segments to independent batch lanes."""

    def __init__(
        self,
        segments: list[np.ndarray],
        *,
        batch_size: int,
        bptt_steps: int,
        seed: int,
    ) -> None:
        if not segments:
            raise ValueError("No training segments available.")
        self._segments = segments
        self._batch_size = batch_size
        self._bptt_steps = bptt_steps
        self._rng = np.random.default_rng(seed)
        self._active: list[np.ndarray | None] = [None] * batch_size
        self._offsets = np.zeros(batch_size, dtype=np.int64)
        self.epoch = 0
        self._order = np.arange(len(segments), dtype=np.int64)
        self._cursor = len(segments)

    def _reshuffle(self) -> None:
        self._order = np.arange(len(self._segments), dtype=np.int64)
        self._rng.shuffle(self._order)
        self._cursor = 0
        self.epoch += 1

    def _next_segment(self) -> np.ndarray:
        if self._cursor >= len(self._order):
            self._reshuffle()
        segment = self._segments[int(self._order[self._cursor])]
        self._cursor += 1
        return segment

    def next_chunk(self) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """Return bptt_steps arrays of final-input indices plus lane reset mask."""
        reset_mask = np.zeros(self._batch_size, dtype=np.bool_)
        per_step: list[list[int]] = [[] for _ in range(self._bptt_steps)]

        for lane in range(self._batch_size):
            segment = self._active[lane]
            offset = int(self._offsets[lane])
            if segment is None or offset + self._bptt_steps > len(segment):
                segment = self._next_segment()
                self._active[lane] = segment
                offset = 0
                self._offsets[lane] = 0
                reset_mask[lane] = True

            for bptt_i in range(self._bptt_steps):
                per_step[bptt_i].append(int(segment[offset + bptt_i]))
            self._offsets[lane] = offset + self._bptt_steps

        return tuple(np.asarray(step_indices, dtype=np.int64) for step_indices in per_step), reset_mask


def parse_args() -> SegmentRunConfig:
    parser = argparse.ArgumentParser(
        description="Train GraphCast on shuffled chronological segments with chunked BPTT."
    )
    parser.add_argument("--data-path", default=base.DEFAULT_DATA_PATH)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--width", type=int, choices=[128, 256, 512, 1024], default=128)
    parser.add_argument("--processor-msg-steps", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--val-year", type=int, default=2021)
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--ckpt-in", default=base.DEFAULT_CKPT)
    parser.add_argument("--stats-dir", default=base.DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", default="artifacts/checkpoints/graphcast_mamba_interleaved_segments")
    parser.add_argument("--run-name", default="segments_res2_m4_w128_mp1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000, help="Optimizer updates, not forecast windows.")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--input-duration", default=None)
    parser.add_argument("--target-steps", type=int, default=1)
    parser.add_argument("--len-segment", type=int, default=30)
    parser.add_argument("--bptt-steps", type=int, default=6)
    parser.add_argument("--temporal-backbone", choices=["none", "mamba"], default="none")
    parser.add_argument(
        "--temporal-location",
        choices=["mesh_post_encoder", "mesh_processor_interleaved", "mesh_post_encoder_residual"],
        default="mesh_post_encoder",
    )
    parser.add_argument("--temporal-hidden-size", type=int, default=128)
    parser.add_argument("--temporal-d-inner", type=int, default=None)
    parser.add_argument("--temporal-d-state", type=int, default=16)
    parser.add_argument("--temporal-d-conv", type=int, default=4)
    parser.add_argument("--temporal-dt-rank", default="auto")
    parser.add_argument("--temporal-bias", action="store_true", default=False)
    parser.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias", action="store_false", default=True)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-stateful", action="store_true", default=False)
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.len_segment <= 0:
        raise ValueError("--len-segment must be > 0")
    if args.bptt_steps <= 0:
        raise ValueError("--bptt-steps must be > 0")
    if args.len_segment % args.bptt_steps != 0:
        raise ValueError("--bptt-steps must divide --len-segment")
    if args.target_steps != 1:
        raise ValueError("Segment BPTT training currently requires --target-steps 1.")
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
    if args.temporal_d_inner is not None and args.temporal_d_inner <= 0:
        raise ValueError("--temporal-d-inner must be > 0")
    if args.temporal_d_state <= 0:
        raise ValueError("--temporal-d-state must be > 0")
    if args.temporal_d_conv <= 0:
        raise ValueError("--temporal-d-conv must be > 0")
    if args.temporal_dt_rank != "auto" and int(args.temporal_dt_rank) <= 0:
        raise ValueError("--temporal-dt-rank must be 'auto' or a positive integer")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be > 0")
    if not (0.0 <= args.temporal_dropout < 1.0):
        raise ValueError("--temporal-dropout must be in [0, 1)")

    base_cfg = base.RunConfig(
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
        temporal_d_inner=args.temporal_d_inner,
        temporal_d_state=args.temporal_d_state,
        temporal_d_conv=args.temporal_d_conv,
        temporal_dt_rank=args.temporal_dt_rank,
        temporal_bias=args.temporal_bias,
        temporal_conv_bias=args.temporal_conv_bias,
        temporal_layers=args.temporal_layers,
        temporal_dropout=args.temporal_dropout,
        temporal_stateful=args.temporal_stateful,
        target_steps=args.target_steps,
        sequential_segment_steps=None,
        eval_only=False,
    )
    return SegmentRunConfig(base_cfg=base_cfg, len_segment=args.len_segment, bptt_steps=args.bptt_steps)


def valid_contiguous_final_input_indices(
    ds: xr.Dataset,
    *,
    input_steps: int,
    target_steps: int,
    dt: pd.Timedelta,
) -> np.ndarray:
    """Final input indices whose full input+target window has no time gaps."""
    time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    candidates = base.valid_final_input_indices(len(time_index), input_steps, target_steps)
    valid: list[int] = []
    expected_count = input_steps + target_steps
    for idx in candidates:
        start = int(idx) - input_steps + 1
        stop = int(idx) + target_steps
        window = time_index[start : stop + 1]
        if len(window) != expected_count:
            continue
        if all((window[i + 1] - window[i]) == dt for i in range(len(window) - 1)):
            valid.append(int(idx))
    return np.asarray(valid, dtype=np.int64)


def build_full_segments(indices: np.ndarray, len_segment: int) -> list[np.ndarray]:
    """Split consecutive valid indices into full, chronological segments."""
    if len(indices) == 0:
        return []
    sorted_idx = np.sort(indices)
    gaps = np.where(np.diff(sorted_idx) > 1)[0] + 1
    runs = np.split(sorted_idx, gaps)
    segments: list[np.ndarray] = []
    for run in runs:
        for start in range(0, len(run) - len_segment + 1, len_segment):
            segment = run[start : start + len_segment]
            if len(segment) == len_segment:
                segments.append(segment)
    return segments


def _map_temporal_state_leaves(state: hk.State, fn) -> hk.State:
    mutable_state = hk.data_structures.to_mutable_dict(state)
    for module_state in mutable_state.values():
        for state_name, leaf in module_state.items():
            if not isinstance(leaf, jax.Array):
                continue
            if state_name.endswith("_ssm_state") or state_name.endswith("_conv_cache"):
                module_state[state_name] = fn(leaf)
    return hk.data_structures.to_immutable_dict(mutable_state)


def _reset_temporal_state_lanes(state: hk.State, reset_mask: jax.Array) -> hk.State:
    reset_mask = jnp.asarray(reset_mask, dtype=bool)

    def reset_leaf(leaf: jax.Array) -> jax.Array:
        if leaf.ndim > 0 and leaf.shape[0] == reset_mask.shape[0]:
            mask_shape = (reset_mask.shape[0],) + (1,) * (leaf.ndim - 1)
            return jnp.where(reset_mask.reshape(mask_shape), jnp.zeros_like(leaf), leaf)
        return jnp.where(jnp.any(reset_mask), jnp.zeros_like(leaf), leaf)

    return _map_temporal_state_leaves(state, reset_leaf)


def _stop_gradient_temporal_state(state: hk.State) -> hk.State:
    return _map_temporal_state_leaves(state, jax.lax.stop_gradient)


def _write_segment_run_config(
    out_dir: Path,
    *,
    segment_cfg: SegmentRunConfig,
    model_cfg: base.gc.ModelConfig,
    task_cfg: base.gc.TaskConfig,
) -> None:
    base._write_run_config(out_dir, segment_cfg.base_cfg, model_cfg, task_cfg)
    path = out_dir / "run_config.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["segment_training"] = {
        "len_segment": segment_cfg.len_segment,
        "bptt_steps": segment_cfg.bptt_steps,
        "shuffle_segments": True,
        "drop_short_tail_segments": True,
        "max_steps_unit": "optimizer_updates",
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_chunk_batches(
    train_ds: xr.Dataset,
    chunk_indices: Iterable[np.ndarray],
    *,
    input_steps: int,
    target_steps: int,
    task_cfg: base.gc.TaskConfig,
    dt: pd.Timedelta,
) -> tuple[tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], tuple[xr.Dataset, ...]]:
    inputs = []
    targets = []
    forcings = []
    for step_indices in chunk_indices:
        batch_inputs, batch_targets, batch_forcings = base.build_batch_from_indices(
            train_ds,
            indices=step_indices,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )
        inputs.append(batch_inputs)
        targets.append(batch_targets)
        forcings.append(batch_forcings)
    return tuple(inputs), tuple(targets), tuple(forcings)


def run_eval_fresh_state(
    transformed,
    params: hk.Params,
    rng: jax.Array,
    eval_ds: xr.Dataset,
    eval_indices: np.ndarray,
    *,
    eval_batch_size: int,
    input_steps: int,
    target_steps: int,
    task_cfg: base.gc.TaskConfig,
    dt: pd.Timedelta,
    progress_label: str,
) -> dict[str, float]:
    losses: list[float] = []
    state_by_batch_size: dict[int, hk.State] = {}
    n_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    t_eval0 = time.time()
    for batch_i, i in enumerate(range(0, len(eval_indices), eval_batch_size), start=1):
        idx = eval_indices[i : i + eval_batch_size]
        inputs, targets, forcings = base.build_batch_from_indices(
            eval_ds,
            indices=idx,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )
        rng, init_key, apply_key = jax.random.split(rng, 3)
        batch_size = len(idx)
        if batch_size not in state_by_batch_size:
            _, state_by_batch_size[batch_size] = transformed.init(init_key, inputs, targets, forcings, False)
        eval_state = state_by_batch_size[batch_size]
        (loss_and_diag, _) = transformed.apply(params, eval_state, apply_key, inputs, targets, forcings, False)
        loss = float(base.scalarize_loss(loss_and_diag[0]))
        losses.append(loss)
        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_loss {loss:.6f}"
            )
    return {"total": float(np.mean(losses))}


def main() -> None:
    segment_cfg = parse_args()
    cfg = segment_cfg.base_cfg
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_in = base.load_graphcast_checkpoint(Path(cfg.ckpt_in))
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

    norm_stats = base.load_stats(Path(cfg.stats_dir))
    base.validate_stats_coverage(task_cfg, norm_stats)
    train_ds, eval_ds = base._open_local_splits(cfg)
    train_ds = base.prepare_dataset_for_task(train_ds, task_cfg)
    eval_ds = base.prepare_dataset_for_task(eval_ds, task_cfg)

    dt_train = base.infer_time_step(train_ds)
    dt_eval = base.infer_time_step(eval_ds)
    if dt_train != dt_eval:
        raise ValueError(f"Train/eval time step mismatch: train={dt_train}, eval={dt_eval}")

    input_steps = base.input_steps_from_duration(task_cfg.input_duration, dt_train)
    if input_steps < 2:
        raise ValueError("Segment training expects at least two input frames.")
    target_steps = cfg.target_steps

    train_final_indices = valid_contiguous_final_input_indices(
        train_ds,
        input_steps=input_steps,
        target_steps=target_steps,
        dt=dt_train,
    )
    eval_final_indices = valid_contiguous_final_input_indices(
        eval_ds,
        input_steps=input_steps,
        target_steps=target_steps,
        dt=dt_train,
    )
    segments = build_full_segments(train_final_indices, segment_cfg.len_segment)
    if not segments:
        raise ValueError(
            "No full training segments after timestamp-contiguous filtering. "
            f"len_segment={segment_cfg.len_segment}, valid_windows={len(train_final_indices)}"
        )
    print(
        "Prepared segment windows: "
        f"train_windows={len(train_final_indices)}, eval_windows={len(eval_final_indices)}, "
        f"segments={len(segments)}, len_segment={segment_cfg.len_segment}, "
        f"bptt_steps={segment_cfg.bptt_steps}, input_steps={input_steps}, target_steps={target_steps}"
    )

    def forward_fn(inputs, targets, forcings, is_training):
        predictor = base.build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=True,
            temporal_backbone=cfg.temporal_backbone,
            temporal_location=cfg.temporal_location,
            temporal_hidden_size=cfg.temporal_hidden_size,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank=cfg.temporal_dt_rank,
            temporal_bias=cfg.temporal_bias,
            temporal_conv_bias=cfg.temporal_conv_bias,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=cfg.temporal_dropout,
            temporal_stateful=cfg.temporal_stateful,
        )
        return predictor.loss(inputs, targets, forcings)

    transformed = hk.transform_with_state(forward_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    init_indices = [int(segments[lane % len(segments)][0]) for lane in range(cfg.batch_size)]
    sample_inputs, sample_targets, sample_forcings = base.build_batch_from_indices(
        train_ds,
        indices=init_indices,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
    )
    params, state = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, True)
    if cfg.resume_step is not None:
        params = ckpt_in.params
        print(f"Resuming from step {cfg.resume_step} (params loaded from {cfg.ckpt_in})")

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)
    _write_segment_run_config(out_dir, segment_cfg=segment_cfg, model_cfg=model_cfg, task_cfg=task_cfg)

    @functools.partial(jax.jit)
    def train_chunk(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        rng_key: jax.Array,
        chunk_inputs: tuple[xr.Dataset, ...],
        chunk_targets: tuple[xr.Dataset, ...],
        chunk_forcings: tuple[xr.Dataset, ...],
        reset_mask: jax.Array,
    ):
        state = _reset_temporal_state_lanes(state, reset_mask)

        def loss_fn(p, s, key):
            current_state = s
            losses = []
            keys = jax.random.split(key, segment_cfg.bptt_steps)
            for bptt_i in range(segment_cfg.bptt_steps):
                (loss_and_diag, current_state) = transformed.apply(
                    p,
                    current_state,
                    keys[bptt_i],
                    chunk_inputs[bptt_i],
                    chunk_targets[bptt_i],
                    chunk_forcings[bptt_i],
                    True,
                )
                losses.append(base.scalarize_loss(loss_and_diag[0]))
            return jnp.mean(jnp.stack(losses)), current_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, _stop_gradient_temporal_state(new_state), new_opt_state, loss

    step = cfg.resume_step if cfg.resume_step is not None else 0
    train_losses: list[tuple[int, float]] = []
    eval_losses: list[tuple[int, float]] = []
    eval_details: list[dict[str, Any]] = []
    step_times: list[tuple[int, float]] = []
    mem_usage: list[tuple[int, float]] = []
    actual_usage: list[dict[str, Any]] = []
    epoch_summaries: list[dict[str, Any]] = []

    if cfg.resume_step is not None:
        train_losses = base._filter_pairs_upto_step(base._load_train_losses(out_dir / "train_loss.json"), cfg.resume_step)
        eval_losses = base._filter_pairs_upto_step(base._load_step_value_pairs(out_dir / "eval_loss.json"), cfg.resume_step)
        step_times = base._filter_pairs_upto_step(base._load_step_value_pairs(out_dir / "step_times.json"), cfg.resume_step)
        mem_usage = base._filter_pairs_upto_step(base._load_step_value_pairs(out_dir / "memory_gib.json"), cfg.resume_step)
        eval_details = base._load_dict_series_upto_step(out_dir / "eval_details.json", cfg.resume_step)
        actual_usage = base._load_dict_series_upto_step(out_dir / "actual_usage.json", cfg.resume_step)
        epoch_summaries = base._load_json_list(out_dir / "epoch_summary.json")

    best_eval_step: int | None = None
    best_eval_loss = float("inf")
    if eval_losses:
        best_eval_step, best_eval_loss = min(eval_losses, key=lambda x: (x[1], x[0]))
        print(f"[best:init] step {best_eval_step} val {best_eval_loss:.6f}")

    def maybe_save_best_checkpoint(eval_step: int, eval_total: float) -> None:
        nonlocal best_eval_step, best_eval_loss
        if eval_total >= best_eval_loss:
            return
        best_eval_step = int(eval_step)
        best_eval_loss = float(eval_total)
        base.save_checkpoint(
            out_dir,
            params=params,
            step=eval_step,
            model_cfg=model_cfg,
            task_cfg=task_cfg,
            description=ckpt_in.description,
            license_text=ckpt_in.license,
            filename="ckpt_best.npz",
        )
        with (out_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_eval_step": best_eval_step,
                    "best_eval_loss": best_eval_loss,
                    "best_checkpoint": "ckpt_best.npz",
                    "match_type": "exact",
                },
                f,
                indent=2,
            )
        print(f"[best] updated step {best_eval_step} val {best_eval_loss:.6f}")

    scheduler = SegmentBatchScheduler(
        segments,
        batch_size=cfg.batch_size,
        bptt_steps=segment_cfg.bptt_steps,
        seed=cfg.seed,
    )
    pass_start_step = step + 1
    pass_loss_accum: list[float] = []
    observed_epoch = scheduler.epoch

    while step < cfg.max_steps:
        chunk_indices, reset_mask_np = scheduler.next_chunk()
        if scheduler.epoch != observed_epoch and pass_loss_accum:
            epoch_summaries.append(
                {
                    "pass": observed_epoch,
                    "steps": step - pass_start_step + 1,
                    "train_loss_mean": float(np.mean(pass_loss_accum)),
                    "time_per_step_mean": float(
                        np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                    ),
                    "mem_gib_max": float(np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])),
                }
            )
            observed_epoch = scheduler.epoch
            pass_start_step = step + 1
            pass_loss_accum = []

        chunk_inputs, chunk_targets, chunk_forcings = _build_chunk_batches(
            train_ds,
            chunk_indices,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt_train,
        )

        rng, step_key = jax.random.split(rng)
        t0 = time.time()
        params, state, opt_state, loss = train_chunk(
            params,
            state,
            opt_state,
            step_key,
            chunk_inputs,
            chunk_targets,
            chunk_forcings,
            jnp.asarray(reset_mask_np),
        )
        step_time = time.time() - t0
        step += 1
        loss_f = float(loss)
        train_losses.append((step, loss_f))
        pass_loss_accum.append(loss_f)
        step_times.append((step, step_time))

        usage = base.sample_actual_usage(step=step)
        actual_usage.append(usage)
        if usage.get("gpu_mem_gib") is not None:
            mem_usage.append((step, float(usage["gpu_mem_gib"])))

        if step % 10 == 0:
            print(
                f"step {step}/{cfg.max_steps} loss {loss_f:.6f} "
                f"segment_epoch {scheduler.epoch} reset_lanes {int(reset_mask_np.sum())}"
            )

        if step % cfg.eval_every == 0:
            eval_metrics = run_eval_fresh_state(
                transformed,
                params,
                rng,
                eval_ds,
                eval_final_indices,
                eval_batch_size=cfg.eval_batch_size,
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt_train,
                progress_label=f"eval@step{step}",
            )
            eval_losses.append((step, eval_metrics["total"]))
            maybe_save_best_checkpoint(step, float(eval_metrics["total"]))
            eval_details.append({"step": step, "total": eval_metrics["total"]})
            print(f"[eval] step {step} total {eval_metrics['total']:.6f}")
            base.plot_loss_curves(out_dir, train_losses, eval_losses)
            base.save_logs(
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
            base.save_checkpoint(
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
                "pass": observed_epoch,
                "steps": step - pass_start_step + 1,
                "train_loss_mean": float(np.mean(pass_loss_accum)),
                "time_per_step_mean": float(
                    np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                ),
                "mem_gib_max": float(np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])),
            }
        )

    final_eval = run_eval_fresh_state(
        transformed,
        params,
        rng,
        eval_ds,
        eval_final_indices,
        eval_batch_size=cfg.eval_batch_size,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
        progress_label="eval@final",
    )
    eval_losses.append((step, final_eval["total"]))
    maybe_save_best_checkpoint(step, float(final_eval["total"]))
    eval_details.append({"step": step, "final": True, "total": final_eval["total"]})

    base.save_checkpoint(
        out_dir,
        params=params,
        step=step,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        description=ckpt_in.description,
        license_text=ckpt_in.license,
    )
    base.save_logs(
        out_dir,
        train_losses,
        eval_losses,
        eval_details,
        step_times,
        mem_usage,
        actual_usage,
        epoch_summaries,
    )
    base.plot_loss_curves(out_dir, train_losses, eval_losses)
    print(f"Done. Final eval total {final_eval['total']:.6f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
