from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Iterable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from .batching import BatchBuilder, build_batch_from_indices_vectorized, valid_final_input_indices
from .config import RunConfig
from .logging import _write_run_config
from .model import gc, scalarize_loss


@dataclasses.dataclass(frozen=True)
class SegmentRunConfig:
    base_cfg: RunConfig
    len_segment: int
    bptt_steps: int
    chunk_load_workers: int


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


def valid_contiguous_final_input_indices(
    ds: xr.Dataset,
    *,
    input_steps: int,
    target_steps: int,
    dt: pd.Timedelta,
) -> np.ndarray:
    """Final input indices whose full input+target window has no time gaps."""
    time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    candidates = valid_final_input_indices(len(time_index), input_steps, target_steps)
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
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    numpy_cache_active: bool = False,
    train_cache_estimate_gib: float | None = None,
    effective_train_batch_builder: str | None = None,
    effective_eval_batch_builder: str | None = None,
) -> None:
    _write_run_config(
        out_dir,
        segment_cfg.base_cfg,
        model_cfg,
        task_cfg,
        numpy_cache_active=numpy_cache_active,
        train_cache_estimate_gib=train_cache_estimate_gib,
        effective_train_batch_builder=effective_train_batch_builder,
        effective_eval_batch_builder=effective_eval_batch_builder,
    )
    path = out_dir / "run_config.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["segment_training"] = {
        "len_segment": segment_cfg.len_segment,
        "bptt_steps": segment_cfg.bptt_steps,
        "chunk_load_workers": segment_cfg.chunk_load_workers,
        "prefetch_chunks": 1,
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
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
    batch_builder: BatchBuilder = build_batch_from_indices_vectorized,
    chunk_load_workers: int = 1,
    load_executor: concurrent.futures.Executor | None = None,
) -> tuple[tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], tuple[xr.Dataset, ...]]:
    chunk_indices = tuple(chunk_indices)

    def load_one(step_indices: np.ndarray) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        return batch_builder(
            train_ds,
            indices=step_indices,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )

    if chunk_load_workers <= 1 or len(chunk_indices) <= 1:
        results = [load_one(step_indices) for step_indices in chunk_indices]
    elif load_executor is not None:
        futures = [load_executor.submit(load_one, step_indices) for step_indices in chunk_indices]
        results = [future.result() for future in futures]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_load_workers) as executor:
            futures = [executor.submit(load_one, step_indices) for step_indices in chunk_indices]
            results = [future.result() for future in futures]

    inputs = []
    targets = []
    forcings = []
    for batch_inputs, batch_targets, batch_forcings in results:
        inputs.append(batch_inputs)
        targets.append(batch_targets)
        forcings.append(batch_forcings)
    return tuple(inputs), tuple(targets), tuple(forcings)


def _save_chunk_timing_logs(out_dir: Path, chunk_timing: list[dict[str, Any]]) -> None:
    with (out_dir / "chunk_timing.json").open("w", encoding="utf-8") as f:
        json.dump(chunk_timing, f, indent=2)

    def vals(name: str) -> list[float]:
        return [float(item[name]) for item in chunk_timing if item.get(name) is not None]

    data_wait = vals("data_wait_s")
    gpu_train = vals("gpu_train_s")
    iteration_wall = vals("iteration_wall_s")
    with (out_dir / "chunk_timing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": len(chunk_timing),
                "data_wait_s_avg": float(np.mean(data_wait)) if data_wait else None,
                "data_wait_s_peak": float(np.max(data_wait)) if data_wait else None,
                "gpu_train_s_avg": float(np.mean(gpu_train)) if gpu_train else None,
                "gpu_train_s_peak": float(np.max(gpu_train)) if gpu_train else None,
                "iteration_wall_s_avg": float(np.mean(iteration_wall)) if iteration_wall else None,
                "iteration_wall_s_peak": float(np.max(iteration_wall)) if iteration_wall else None,
            },
            f,
            indent=2,
        )


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
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
    progress_label: str,
    batch_builder: BatchBuilder = build_batch_from_indices_vectorized,
) -> dict[str, float]:
    losses: list[float] = []
    state_by_batch_size: dict[int, hk.State] = {}
    eval_fn_by_batch_size: dict[int, callable] = {}
    n_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    t_eval0 = time.time()
    for batch_i, i in enumerate(range(0, len(eval_indices), eval_batch_size), start=1):
        idx = eval_indices[i : i + eval_batch_size]
        inputs, targets, forcings = batch_builder(
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
        if batch_size not in eval_fn_by_batch_size:
            eval_state = state_by_batch_size[batch_size]

            @jax.jit
            def eval_batch(params, key, inputs, targets, forcings):
                (loss_and_diag, _) = transformed.apply(params, eval_state, key, inputs, targets, forcings, False)
                return scalarize_loss(loss_and_diag[0])

            eval_fn_by_batch_size[batch_size] = eval_batch
        loss = float(eval_fn_by_batch_size[batch_size](params, apply_key, inputs, targets, forcings))
        losses.append(loss)
        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_loss {loss:.6f}"
            )
    return {"total": float(np.mean(losses))}
