from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
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
from .eval_selection import (
    EVAL_SUBSET_STRATIFIED_FIXED,
    select_eval_subset,
)
from .logging import _write_run_config
from .model import gc, scalarize_loss, xarray_jax
from .prepared_array import PreparedArrayStore, is_prepared_array_store
from .prepared_block_batches import PreparedBlockBatchLoader


AR_LOSS_MODE_TAIL_UNIFORM = "tail_uniform"
AR_LOSS_MODE_ALL_BPTT_UNIFORM = "all_bptt_uniform"
AR_LOSS_MODE_CHOICES = (AR_LOSS_MODE_TAIL_UNIFORM, AR_LOSS_MODE_ALL_BPTT_UNIFORM)
AR_EVAL_LOSS_MODE = AR_LOSS_MODE_TAIL_UNIFORM


@dataclasses.dataclass(frozen=True)
class SegmentRunConfig:
    base_cfg: RunConfig
    len_segment: int
    bptt_steps: int
    chunk_load_workers: int
    segment_prefetch_depth: int = 2
    use_segment_block_loader: bool = True
    filter_nan_segments: bool = True
    eval_num_segments: int | None = 16
    final_eval_num_segments: int | None = None
    eval_subset_policy: str = EVAL_SUBSET_STRATIFIED_FIXED
    eval_rotating_diagnostics: bool = True
    autoregressive_loss_mode: str = AR_LOSS_MODE_TAIL_UNIFORM


@dataclasses.dataclass(frozen=True)
class SegmentChunk:
    chunk_indices: tuple[np.ndarray, ...]
    reset_mask: np.ndarray
    lane_segment_ids: np.ndarray
    lane_offsets: np.ndarray
    epoch: int


@dataclasses.dataclass(frozen=True)
class SegmentLoadStats:
    load_s: float
    cache_hits: int = 0
    cache_misses: int = 0
    loaded_gib: float = 0.0
    loader: str = "batch_builder"


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
        self._active_segment_ids = np.full(batch_size, -1, dtype=np.int64)
        self._offsets = np.zeros(batch_size, dtype=np.int64)
        self.epoch = 0
        self._order = np.arange(len(segments), dtype=np.int64)
        self._cursor = len(segments)

    def _reshuffle(self) -> None:
        self._order = np.arange(len(self._segments), dtype=np.int64)
        self._rng.shuffle(self._order)
        self._cursor = 0
        self.epoch += 1

    def _next_segment(self) -> tuple[int, np.ndarray]:
        if self._cursor >= len(self._order):
            self._reshuffle()
        segment_id = int(self._order[self._cursor])
        segment = self._segments[segment_id]
        self._cursor += 1
        return segment_id, segment

    def next_chunk(self) -> SegmentChunk:
        """Return bptt_steps arrays of final-input indices plus lane reset mask."""
        reset_mask = np.zeros(self._batch_size, dtype=np.bool_)
        per_step: list[list[int]] = [[] for _ in range(self._bptt_steps)]
        lane_segment_ids = np.empty(self._batch_size, dtype=np.int64)
        lane_offsets = np.empty(self._batch_size, dtype=np.int64)

        for lane in range(self._batch_size):
            segment = self._active[lane]
            offset = int(self._offsets[lane])
            if segment is None or offset + self._bptt_steps > len(segment):
                segment_id, segment = self._next_segment()
                self._active[lane] = segment
                self._active_segment_ids[lane] = segment_id
                offset = 0
                self._offsets[lane] = 0
                reset_mask[lane] = True

            lane_segment_ids[lane] = self._active_segment_ids[lane]
            lane_offsets[lane] = offset
            for bptt_i in range(self._bptt_steps):
                per_step[bptt_i].append(int(segment[offset + bptt_i]))
            self._offsets[lane] = offset + self._bptt_steps

        return SegmentChunk(
            chunk_indices=tuple(np.asarray(step_indices, dtype=np.int64) for step_indices in per_step),
            reset_mask=reset_mask,
            lane_segment_ids=lane_segment_ids,
            lane_offsets=lane_offsets,
            epoch=self.epoch,
        )

    def next_chunk_arrays(self) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        chunk = self.next_chunk()
        return chunk.chunk_indices, chunk.reset_mask


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


def segment_midpoint_times(ds: xr.Dataset, segments: list[np.ndarray]) -> pd.DatetimeIndex | None:
    if not hasattr(ds, "time"):
        return None
    try:
        time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
        return pd.DatetimeIndex([time_index[int(segment[len(segment) // 2])] for segment in segments])
    except (IndexError, TypeError, ValueError):
        return None


def iter_eval_segment_chunks(
    segments: list[np.ndarray],
    *,
    batch_size: int,
    bptt_steps: int,
    segment_ids: np.ndarray | None = None,
) -> Iterable[tuple[tuple[np.ndarray, ...], np.ndarray]]:
    """Yield deterministic eval chunks over full segments without wraparound.

    Segments are traversed in chronological order. Each yielded item contains
    `bptt_steps` arrays of final-input indices, one per step in the chunk, plus
    a reset mask for the active lanes. The reset mask is all-ones for the first
    chunk of each lane group and all-zeros afterwards so temporal state is
    preserved within an eval segment and reset only at segment boundaries.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if bptt_steps <= 0:
        raise ValueError("bptt_steps must be > 0")
    if not segments:
        return

    segment_len = len(segments[0])
    if segment_len == 0:
        raise ValueError("Eval segments must be non-empty.")
    if segment_len % bptt_steps != 0:
        raise ValueError(
            f"Eval segment length {segment_len} must be divisible by bptt_steps={bptt_steps}."
        )
    if any(len(segment) != segment_len for segment in segments):
        raise ValueError("All eval segments must have the same full length.")

    for chunk in iter_eval_segment_chunk_infos(
        segments,
        batch_size=batch_size,
        bptt_steps=bptt_steps,
        segment_ids=segment_ids,
    ):
        yield chunk.chunk_indices, chunk.reset_mask


def iter_eval_segment_chunk_infos(
    segments: list[np.ndarray],
    *,
    batch_size: int,
    bptt_steps: int,
    segment_ids: np.ndarray | None = None,
) -> Iterable[SegmentChunk]:
    """Yield deterministic eval chunks with lane segment metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if bptt_steps <= 0:
        raise ValueError("bptt_steps must be > 0")
    if not segments:
        return

    segment_len = len(segments[0])
    if segment_len == 0:
        raise ValueError("Eval segments must be non-empty.")
    if segment_len % bptt_steps != 0:
        raise ValueError(
            f"Eval segment length {segment_len} must be divisible by bptt_steps={bptt_steps}."
        )
    if any(len(segment) != segment_len for segment in segments):
        raise ValueError("All eval segments must have the same full length.")
    if segment_ids is None:
        segment_ids = np.arange(len(segments), dtype=np.int64)
    else:
        segment_ids = np.asarray(segment_ids, dtype=np.int64)
        if segment_ids.shape != (len(segments),):
            raise ValueError("segment_ids must have one entry per eval segment.")

    for start in range(0, len(segments), batch_size):
        segment_group = segments[start : start + batch_size]
        segment_id_group = segment_ids[start : start + len(segment_group)]
        lane_count = len(segment_group)
        for offset in range(0, segment_len, bptt_steps):
            reset_mask = np.zeros(lane_count, dtype=np.bool_)
            if offset == 0:
                reset_mask[:] = True
            chunk_indices = tuple(
                np.asarray(
                    [int(segment[offset + bptt_i]) for segment in segment_group],
                    dtype=np.int64,
                )
                for bptt_i in range(bptt_steps)
            )
            yield SegmentChunk(
                chunk_indices=chunk_indices,
                reset_mask=reset_mask,
                lane_segment_ids=np.asarray(segment_id_group, dtype=np.int64),
                lane_offsets=np.full(lane_count, offset, dtype=np.int64),
                epoch=0,
            )


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


def _loss_by_lane(loss_da: xr.DataArray) -> jax.Array:
    loss = xarray_jax.unwrap_data(loss_da)
    if "batch" not in loss_da.dims:
        return loss[None]
    batch_axis = loss_da.dims.index("batch")
    if batch_axis != 0:
        loss = jnp.moveaxis(loss, batch_axis, 0)
    reduce_axes = tuple(range(1, loss.ndim))
    if reduce_axes:
        loss = jnp.mean(loss, axis=reduce_axes)
    return loss


def _reset_dataset_lanes(current: xr.Dataset, reset_values: xr.Dataset, reset_mask: jax.Array) -> xr.Dataset:
    reset_mask = jnp.asarray(reset_mask, dtype=bool)
    data_vars = {}
    for name, current_da in current.data_vars.items():
        if name not in reset_values:
            data_vars[name] = current_da
            continue
        if "batch" not in current_da.dims:
            data_vars[name] = current_da
            continue
        batch_axis = current_da.dims.index("batch")
        current_data = xarray_jax.unwrap_data(current_da)
        reset_data = xarray_jax.unwrap_data(reset_values[name])
        mask_shape = [1] * current_data.ndim
        mask_shape[batch_axis] = reset_mask.shape[0]
        data = jnp.where(jnp.reshape(reset_mask, mask_shape), reset_data, current_data)
        data_vars[name] = xr.DataArray(
            xarray_jax.wrap(data),
            dims=current_da.dims,
            coords=current_da.coords,
            attrs=current_da.attrs,
            name=current_da.name,
        )
    return xr.Dataset(data_vars, coords=current.coords)


def _stop_gradient_dataset(dataset: xr.Dataset) -> xr.Dataset:
    data_vars = {}
    for name, data_array in dataset.data_vars.items():
        data = jax.lax.stop_gradient(xarray_jax.unwrap_data(data_array))
        data_vars[name] = xr.DataArray(
            xarray_jax.wrap(data),
            dims=data_array.dims,
            coords=data_array.coords,
            attrs=data_array.attrs,
            name=data_array.name,
        )
    return xr.Dataset(data_vars, coords=dataset.coords)


def _constant_inputs(inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset) -> xr.Dataset:
    constant_inputs = inputs.drop_vars(targets_template.keys(), errors="ignore")
    constant_inputs = constant_inputs.drop_vars(forcings.keys(), errors="ignore")
    for name, var in constant_inputs.items():
        if "time" in var.dims:
            raise ValueError(
                f"Time-dependent input variable {name} must either be a forcing variable or target variable."
            )
    return constant_inputs


def _advance_autoregressive_inputs(
    inputs: xr.Dataset,
    predictions: xr.Dataset,
    forcings: xr.Dataset,
) -> xr.Dataset:
    constant_inputs = _constant_inputs(inputs, predictions, forcings)
    rolling_inputs = inputs.drop_vars(constant_inputs.keys())
    num_inputs = rolling_inputs.sizes["time"]
    next_frame = xr.merge([predictions, forcings])
    predicted_or_forced_inputs = next_frame[list(rolling_inputs.keys())]
    updated = (
        xr.concat([rolling_inputs, predicted_or_forced_inputs], dim="time")
        .tail(time=num_inputs)
        .assign_coords(time=rolling_inputs.coords["time"])
    )
    return xr.merge([constant_inputs, updated])


def _chunk_ar_truth_prefix(target_steps: int, bptt_steps: int) -> int:
    """Return truth-fed prefix length for chunk-local corrected AR training."""
    if target_steps <= 1:
        return bptt_steps
    if target_steps >= bptt_steps:
        raise ValueError(
            "Chunk-local autoregressive segment training requires "
            f"--target-steps < --bptt-steps, got target_steps={target_steps}, "
            f"bptt_steps={bptt_steps}."
        )
    return bptt_steps - target_steps


def include_bptt_loss_step(loss_mode: str, bptt_i: int, truth_prefix_steps: int) -> bool:
    """Return whether a BPTT lane loss contributes to the AR training objective."""
    if loss_mode == AR_LOSS_MODE_TAIL_UNIFORM:
        return bptt_i >= truth_prefix_steps
    if loss_mode == AR_LOSS_MODE_ALL_BPTT_UNIFORM:
        return True
    raise ValueError(f"Unknown autoregressive loss mode: {loss_mode!r}")


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
    finite_segment_filter_stats: dict[str, Any] | None = None,
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
        "prefetch_chunks": segment_cfg.segment_prefetch_depth,
        "segment_block_loader": segment_cfg.use_segment_block_loader,
        "filter_nan_segments": segment_cfg.filter_nan_segments,
        "eval_num_segments": segment_cfg.eval_num_segments,
        "final_eval_num_segments": segment_cfg.final_eval_num_segments,
        "eval_subset_policy": segment_cfg.eval_subset_policy,
        "eval_rotating_diagnostics": segment_cfg.eval_rotating_diagnostics,
        "ar_gradient_alignment_diagnostics": bool(
            getattr(segment_cfg, "ar_gradient_alignment_diagnostics", False)
        ),
        "ar_gradient_alignment_every": getattr(segment_cfg, "ar_gradient_alignment_every", None),
        "ar_gradient_alignment_num_chunks": int(
            getattr(segment_cfg, "ar_gradient_alignment_num_chunks", 1)
        ),
        "shuffle_segments": True,
        "drop_short_tail_segments": True,
        "max_steps_unit": "optimizer_updates",
    }
    if segment_cfg.base_cfg.target_steps > 1:
        truth_prefix_steps = _chunk_ar_truth_prefix(
            int(segment_cfg.base_cfg.target_steps),
            int(segment_cfg.bptt_steps),
        )
        loss_mode = getattr(segment_cfg, "autoregressive_loss_mode", AR_LOSS_MODE_TAIL_UNIFORM)
        payload["autoregressive_training"] = {
            "enabled": True,
            "mode": "chunk_local_corrected_ar_tail",
            "loss_mode": loss_mode,
            "eval_loss_mode": loss_mode,
            "target_steps": int(segment_cfg.base_cfg.target_steps),
            "truth_prefix_steps": int(truth_prefix_steps),
            "ar_tail_steps": int(segment_cfg.base_cfg.target_steps),
            "feedback_gradient": "full_bptt_chunk_detached",
            "state_carry_steps": "temporal_state_stop_gradient_chunks",
            "physical_state_carry": "truth_reset_each_chunk",
        }
    else:
        payload["single_step_training"] = {
            "enabled": True,
            "mode": "teacher_forced_one_step_uniform_bptt",
            "loss_path": "loss_and_predictions",
            "single_step_loss_mode": AR_LOSS_MODE_ALL_BPTT_UNIFORM,
            "target_steps": int(segment_cfg.base_cfg.target_steps),
            "truth_prefix_steps": int(segment_cfg.bptt_steps),
            "feedback_gradient": "none_truth_fed_bptt",
            "state_carry_steps": "temporal_state_stop_gradient_chunks",
            "physical_state_carry": "truth_reset_each_bptt_step",
        }
    if finite_segment_filter_stats is not None:
        payload["segment_training"]["finite_segment_filter"] = finite_segment_filter_stats
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@dataclasses.dataclass(frozen=True)
class _BlockVar:
    data: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class _LaneBlock:
    segment_id: int
    block_start: int
    block_stop: int
    vars: dict[str, _BlockVar]
    bytes_loaded: int


def _drop_source_batch(ds: xr.Dataset) -> xr.Dataset:
    if "batch" not in ds.dims:
        return ds
    if ds.sizes["batch"] != 1:
        raise ValueError(f"Expected source dataset batch size 1, got {ds.sizes['batch']}.")
    return ds.isel(batch=0, drop=True)


def _select_pressure_levels(var: xr.DataArray, task_cfg: gc.TaskConfig) -> xr.DataArray:
    if "level" not in var.dims:
        return var
    return var.sel(level=list(task_cfg.pressure_levels))


def _timedelta_coords(count: int, start_steps: int, dt: pd.Timedelta) -> np.ndarray:
    step_ns = int(dt / pd.Timedelta(1, "ns"))
    offsets = (start_steps + np.arange(count, dtype=np.int64)) * np.timedelta64(step_ns, "ns")
    return offsets.astype("timedelta64[ns]")


class SegmentBlockBatchLoader:
    """Build segment BPTT chunks from per-lane contiguous data blocks."""

    def __init__(
        self,
        ds: xr.Dataset,
        segments: list[np.ndarray],
        *,
        input_steps: int,
        target_steps: int,
        task_cfg: gc.TaskConfig,
        dt: pd.Timedelta,
        load_executor: concurrent.futures.Executor | None = None,
        max_workers: int = 1,
        label: str = "segment-block",
    ) -> None:
        self._array_store = ds if is_prepared_array_store(ds) else None
        self._prepared_loader = (
            PreparedBlockBatchLoader(
                self._array_store,
                segments,
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt,
                load_executor=load_executor,
                max_workers=max_workers,
                label=label,
            )
            if self._array_store is not None
            else None
        )
        self._source = None if self._array_store is not None else _drop_source_batch(ds)
        self._segments = segments
        self._input_steps = int(input_steps)
        self._target_steps = int(target_steps)
        self._task_cfg = task_cfg
        self._dt = dt
        self._load_executor = load_executor
        self._max_workers = max(1, int(max_workers))
        self._label = label
        self._cache: dict[int, _LaneBlock] = {}
        self._cache_segment_ids: dict[int, int] = {}
        self._task_vars = tuple(
            sorted(set(task_cfg.input_variables) | set(task_cfg.target_variables) | set(task_cfg.forcing_variables))
        )
        self._coord_values = (
            {}
            if self._source is None
            else {
                name: np.asarray(coord.values)
                for name, coord in self._source.coords.items()
                if name in {"lat", "lon", "level"}
            }
        )
        self.last_stats = SegmentLoadStats(load_s=0.0, loader=self._label)

    def load_chunk(
        self,
        chunk: SegmentChunk,
    ) -> tuple[tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], SegmentLoadStats]:
        if self._prepared_loader is not None:
            return self._prepared_loader.load_chunk(chunk)

        t0 = time.time()
        hits = 0
        misses: list[tuple[int, int]] = []
        for lane, segment_id in enumerate(chunk.lane_segment_ids):
            cached = self._cache.get(lane)
            if cached is not None and self._cache_segment_ids.get(lane) == int(segment_id):
                hits += 1
            else:
                misses.append((lane, int(segment_id)))

        loaded_bytes = 0
        if misses:
            if self._load_executor is not None and self._max_workers > 1:
                futures = [
                    self._load_executor.submit(self._load_lane_block, segment_id)
                    for _, segment_id in misses
                ]
                loaded = [future.result() for future in futures]
            else:
                loaded = [self._load_lane_block(segment_id) for _, segment_id in misses]
            for (lane, segment_id), block in zip(misses, loaded):
                self._cache[lane] = block
                self._cache_segment_ids[lane] = segment_id
                loaded_bytes += block.bytes_loaded

        inputs: list[xr.Dataset] = []
        targets: list[xr.Dataset] = []
        forcings: list[xr.Dataset] = []
        for bptt_i, step_indices in enumerate(chunk.chunk_indices):
            batch_inputs, batch_targets, batch_forcings = self._build_step_batch(step_indices)
            inputs.append(batch_inputs)
            targets.append(batch_targets)
            forcings.append(batch_forcings)

        stats = SegmentLoadStats(
            load_s=time.time() - t0,
            cache_hits=hits,
            cache_misses=len(misses),
            loaded_gib=loaded_bytes / (1024**3),
            loader=self._label,
        )
        self.last_stats = stats
        return tuple(inputs), tuple(targets), tuple(forcings), stats

    def _load_lane_block(self, segment_id: int) -> _LaneBlock:
        segment = self._segments[int(segment_id)]
        block_start = int(segment[0]) - self._input_steps + 1
        block_stop = int(segment[-1]) + self._target_steps + 1
        assert self._source is not None
        if block_start < 0 or block_stop > self._source.sizes["time"]:
            raise IndexError(
                f"[{self._label}] segment {segment_id} requests time slice "
                f"{block_start}:{block_stop}, outside 0:{self._source.sizes['time']}."
            )

        vars_loaded: dict[str, _BlockVar] = {}
        bytes_loaded = 0
        for name in self._task_vars:
            if name not in self._source.data_vars:
                raise KeyError(f"[{self._label}] variable {name!r} is not present in dataset.")
            var = _select_pressure_levels(self._source[name], self._task_cfg)
            dims = tuple(var.dims)
            coords = {
                dim: np.asarray(var.coords[dim].values)
                for dim in dims
                if dim in var.coords and dim != "time"
            }
            if "time" in dims:
                var = var.isel(time=slice(block_start, block_stop))
            data = np.asarray(var.values)
            bytes_loaded += int(data.nbytes)
            vars_loaded[name] = _BlockVar(data=data, dims=dims, coords=coords)
        return _LaneBlock(
            segment_id=int(segment_id),
            block_start=block_start,
            block_stop=block_stop,
            vars=vars_loaded,
            bytes_loaded=bytes_loaded,
        )

    def _build_step_batch(self, final_indices: np.ndarray) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        final_indices = np.asarray(final_indices, dtype=np.int64)
        batch_coord = np.arange(final_indices.size)
        input_time = _timedelta_coords(self._input_steps, -(self._input_steps - 1), self._dt)
        target_time = _timedelta_coords(self._target_steps, 1, self._dt)
        input_offsets = np.arange(-(self._input_steps - 1), 1, dtype=np.int64)
        target_offsets = np.arange(1, self._target_steps + 1, dtype=np.int64)

        def build_var(name: str, offsets: np.ndarray, *, is_input_time: bool) -> xr.DataArray:
            lane_arrays = []
            sample_dims: tuple[str, ...] | None = None
            sample_coords: dict[str, np.ndarray] = {}
            for lane, final_idx in enumerate(final_indices):
                block = self._cache.get(lane)
                if block is None:
                    raise RuntimeError(f"[{self._label}] missing cached block for lane {lane}.")
                cached = block.vars[name]
                dims = cached.dims
                if "time" in dims:
                    time_axis = dims.index("time")
                    local_positions = int(final_idx) + offsets - block.block_start
                    if local_positions.min() < 0 or local_positions.max() >= cached.data.shape[time_axis]:
                        raise IndexError(
                            f"[{self._label}] lane {lane} var {name!r} local positions "
                            f"{local_positions.tolist()} outside cached block shape {cached.data.shape}."
                        )
                    data = np.take(cached.data, local_positions, axis=time_axis)
                    if time_axis != 0:
                        data = np.moveaxis(data, time_axis, 0)
                    remaining_dims = tuple(dim for dim in dims if dim != "time")
                    dims_out = ("time", *remaining_dims)
                    coords = {
                        "time": input_time if is_input_time else target_time,
                        **{
                            dim: cached.coords.get(dim, self._coord_values.get(dim))
                            for dim in remaining_dims
                            if dim in cached.coords or dim in self._coord_values
                        },
                    }
                else:
                    data = cached.data
                    dims_out = dims
                    coords = {
                        dim: cached.coords.get(dim, self._coord_values.get(dim))
                        for dim in dims
                        if dim in cached.coords or dim in self._coord_values
                    }
                if sample_dims is None:
                    sample_dims = dims_out
                    sample_coords = coords
                lane_arrays.append(data)

            if sample_dims is None:
                raise RuntimeError(f"[{self._label}] no arrays built for variable {name!r}.")
            stacked = np.stack(lane_arrays, axis=0)
            coords_out: dict[str, Any] = {"batch": batch_coord}
            coords_out.update(sample_coords)
            return xr.DataArray(stacked, dims=("batch", *sample_dims), coords=coords_out)

        inputs = xr.Dataset(
            {
                name: build_var(name, input_offsets, is_input_time=True)
                for name in self._task_cfg.input_variables
            },
            coords={"batch": batch_coord, "time": input_time},
        )
        targets = xr.Dataset(
            {
                name: build_var(name, target_offsets, is_input_time=False)
                for name in self._task_cfg.target_variables
            },
            coords={"batch": batch_coord, "time": target_time},
        )
        forcing_vars = {
            name: build_var(name, target_offsets, is_input_time=False)
            for name in self._task_cfg.forcing_variables
        }
        forcing_coords = {"batch": batch_coord, "time": target_time} if forcing_vars else {"batch": batch_coord}
        forcings = xr.Dataset(forcing_vars, coords=forcing_coords)
        return inputs, targets, forcings


def filter_finite_segments(
    ds: xr.Dataset,
    segments: list[np.ndarray],
    *,
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    label: str,
    max_report: int = 8,
    cache_dir: Path | None = Path("artifacts/cache/finite_segments"),
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Drop segments whose full input/target halo contains nonfinite task data."""
    if not segments:
        return segments, {
            "label": label,
            "enabled": True,
            "input_segments": 0,
            "kept_segments": 0,
            "dropped_segments": 0,
            "dropped_examples": [],
        }

    task_vars = tuple(
        sorted(set(task_cfg.input_variables) | set(task_cfg.target_variables) | set(task_cfg.forcing_variables))
    )
    if is_prepared_array_store(ds):
        kept: list[np.ndarray] = []
        dropped_examples: list[dict[str, Any]] = []
        t0 = time.time()
        for segment_id, segment in enumerate(segments):
            block_start = int(segment[0]) - input_steps + 1
            block_stop = int(segment[-1]) + target_steps + 1
            block = ds.load_time_block(block_start, block_stop, task_cfg=task_cfg)
            bad_vars = [
                name
                for name in task_vars
                if not np.isfinite(block.vars[name].data).all()
            ]
            if bad_vars:
                if len(dropped_examples) < max_report:
                    dropped_examples.append(
                        {
                            "segment_id": int(segment_id),
                            "start_index": int(segment[0]),
                            "end_index": int(segment[-1]),
                            "bad_variables": bad_vars,
                        }
                    )
                continue
            kept.append(segment)
        stats = {
            "label": label,
            "enabled": True,
            "input_segments": len(segments),
            "kept_segments": len(kept),
            "dropped_segments": len(segments) - len(kept),
            "elapsed_s": time.time() - t0,
            "cache_hit": False,
            "cache_path": None,
            "dropped_examples": dropped_examples,
        }
        print(
            f"[finite-segments:{label}] kept {len(kept)}/{len(segments)} segments "
            f"(dropped {len(segments) - len(kept)}) in {stats['elapsed_s']:.1f}s."
        )
        if dropped_examples:
            print(f"[finite-segments:{label}] examples: {dropped_examples}")
        return kept, stats

    source = _drop_source_batch(ds)
    cache_path: Path | None = None
    if cache_dir is not None:
        time_values = np.asarray(source.coords["time"].values) if "time" in source.coords else np.asarray([])
        segment_bounds = tuple((int(segment[0]), int(segment[-1]), int(len(segment))) for segment in segments)
        payload = {
            "schema": 1,
            "label": label,
            "sizes": {name: int(size) for name, size in sorted(source.sizes.items())},
            "time_start": str(time_values[0]) if time_values.size else None,
            "time_stop": str(time_values[-1]) if time_values.size else None,
            "input_steps": int(input_steps),
            "target_steps": int(target_steps),
            "task_vars": task_vars,
            "pressure_levels": tuple(int(level) for level in task_cfg.pressure_levels),
            "segments": segment_bounds,
        }
        key = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:20]
        cache_path = Path(cache_dir) / f"{label}_{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            kept_ids = {int(segment_id) for segment_id in cached["kept_segment_ids"]}
            kept = [segment for segment_id, segment in enumerate(segments) if segment_id in kept_ids]
            stats = dict(cached["stats"])
            stats["cache_hit"] = True
            stats["cache_path"] = str(cache_path)
            stats["elapsed_s"] = 0.0
            print(
                f"[finite-segments:{label}] loaded cached mask {len(kept)}/{len(segments)} "
                f"from {cache_path}."
            )
            if stats.get("dropped_examples"):
                print(f"[finite-segments:{label}] cached examples: {stats['dropped_examples']}")
            return kept, stats

    kept: list[np.ndarray] = []
    kept_ids: list[int] = []
    dropped_examples: list[dict[str, Any]] = []
    t0 = time.time()

    static_bad: list[str] = []
    for name in task_vars:
        if name not in source.data_vars:
            raise KeyError(f"[finite-segments:{label}] variable {name!r} is not present in dataset.")
        var = _select_pressure_levels(source[name], task_cfg)
        if "time" not in var.dims:
            values = np.asarray(var.values)
            if not np.isfinite(values).all():
                static_bad.append(name)
    if static_bad:
        raise ValueError(f"[finite-segments:{label}] static variables contain nonfinite values: {static_bad}")

    for segment_id, segment in enumerate(segments):
        block_start = int(segment[0]) - input_steps + 1
        block_stop = int(segment[-1]) + target_steps + 1
        bad_vars: list[str] = []
        for name in task_vars:
            var = _select_pressure_levels(source[name], task_cfg)
            if "time" not in var.dims:
                continue
            values = np.asarray(var.isel(time=slice(block_start, block_stop)).values)
            if not np.isfinite(values).all():
                bad_vars.append(name)
        if bad_vars:
            if len(dropped_examples) < max_report:
                dropped_examples.append(
                    {
                        "segment_id": int(segment_id),
                        "start_index": int(segment[0]),
                        "end_index": int(segment[-1]),
                        "bad_variables": bad_vars,
                    }
                )
            continue
        kept.append(segment)
        kept_ids.append(int(segment_id))

    stats = {
        "label": label,
        "enabled": True,
        "input_segments": len(segments),
        "kept_segments": len(kept),
        "dropped_segments": len(segments) - len(kept),
        "elapsed_s": time.time() - t0,
        "cache_hit": False,
        "cache_path": str(cache_path) if cache_path is not None else None,
        "dropped_examples": dropped_examples,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "kept_segment_ids": kept_ids,
                    "stats": stats,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(
        f"[finite-segments:{label}] kept {len(kept)}/{len(segments)} segments "
        f"(dropped {len(segments) - len(kept)}) in {stats['elapsed_s']:.1f}s."
    )
    if dropped_examples:
        print(f"[finite-segments:{label}] examples: {dropped_examples}")
    return kept, stats


def _build_chunk_batches(
    ds: xr.Dataset,
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
            ds,
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
    load = vals("load_s")
    gpu_train = vals("gpu_train_s")
    iteration_wall = vals("iteration_wall_s")
    loaded_gib = vals("loaded_gib")
    with (out_dir / "chunk_timing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "samples": len(chunk_timing),
                "load_s_avg": float(np.mean(load)) if load else None,
                "load_s_peak": float(np.max(load)) if load else None,
                "data_wait_s_avg": float(np.mean(data_wait)) if data_wait else None,
                "data_wait_s_peak": float(np.max(data_wait)) if data_wait else None,
                "gpu_train_s_avg": float(np.mean(gpu_train)) if gpu_train else None,
                "gpu_train_s_peak": float(np.max(gpu_train)) if gpu_train else None,
                "iteration_wall_s_avg": float(np.mean(iteration_wall)) if iteration_wall else None,
                "iteration_wall_s_peak": float(np.max(iteration_wall)) if iteration_wall else None,
                "loaded_gib_total": float(np.sum(loaded_gib)) if loaded_gib else None,
            },
            f,
            indent=2,
        )


def run_eval_segments(
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
    len_segment: int,
    bptt_steps: int,
    progress_label: str,
    batch_builder: BatchBuilder = build_batch_from_indices_vectorized,
    chunk_load_workers: int = 1,
    load_executor: concurrent.futures.Executor | None = None,
    segment_loader: SegmentBlockBatchLoader | None = None,
    rolling_ar: bool = False,
    rolling_loss_mode: str = AR_EVAL_LOSS_MODE,
    load_target_steps: int | None = None,
    max_segments: int | None = None,
    subset_policy: str = EVAL_SUBSET_STRATIFIED_FIXED,
    subset_role: str = "fixed_checkpoint",
    subset_fold: int | None = None,
) -> dict[str, float]:
    target_load_steps = load_target_steps if load_target_steps is not None else (1 if rolling_ar else target_steps)
    eval_segments = build_full_segments(eval_indices, len_segment)
    if not eval_segments:
        raise ValueError(
            "No full eval segments after timestamp-contiguous filtering. "
            f"len_segment={len_segment}, valid_windows={len(eval_indices)}"
        )
    available_segments = len(eval_segments)
    selection = select_eval_subset(
        np.arange(available_segments, dtype=np.int64),
        max_segments,
        times=segment_midpoint_times(eval_ds, eval_segments),
        policy=subset_policy,
        role=subset_role,
        fold=subset_fold,
    )
    eval_segments = [eval_segments[int(position)] for position in selection.positions.tolist()]
    if not eval_segments:
        raise ValueError("No eval segments selected.")
    truth_prefix_steps = _chunk_ar_truth_prefix(target_steps, bptt_steps) if rolling_ar else bptt_steps
    if selection.capped:
        readable_policy = selection.policy.replace("_", " ")
        print(
            f"[{progress_label}] using {readable_policy} "
            f"{len(eval_segments)}/{available_segments} validation segments"
        )

    state_by_lane_count: dict[int, hk.State] = {}
    eval_chunk_fn_by_lane_count: dict[int, callable] = {}
    total_weighted_loss = 0.0
    total_windows = 0
    segment_len = len(eval_segments[0])
    n_lane_groups = (len(eval_segments) + eval_batch_size - 1) // eval_batch_size
    n_chunks_per_group = segment_len // bptt_steps
    n_chunks = n_lane_groups * n_chunks_per_group
    t_eval0 = time.time()

    for chunk_i, chunk in enumerate(
        iter_eval_segment_chunk_infos(
            eval_segments,
            batch_size=eval_batch_size,
            bptt_steps=bptt_steps,
            segment_ids=selection.item_ids,
        ),
        start=1,
    ):
        if segment_loader is not None:
            chunk_inputs, chunk_targets, chunk_forcings, _ = segment_loader.load_chunk(chunk)
        else:
            chunk_inputs, chunk_targets, chunk_forcings = _build_chunk_batches(
                eval_ds,
                chunk.chunk_indices,
                input_steps=input_steps,
                target_steps=target_load_steps,
                task_cfg=task_cfg,
                dt=dt,
                batch_builder=batch_builder,
                chunk_load_workers=chunk_load_workers,
                load_executor=load_executor,
            )
        lane_count = int(len(chunk.reset_mask))
        rng, init_key, apply_key = jax.random.split(rng, 3)

        if lane_count not in state_by_lane_count:
            _, state_by_lane_count[lane_count] = transformed.init(
                init_key,
                chunk_inputs[0],
                chunk_targets[0],
                chunk_forcings[0],
                False,
            )

        if lane_count not in eval_chunk_fn_by_lane_count:
            if rolling_ar:
                @jax.jit
                def eval_chunk(
                    params,
                    state,
                    key,
                    chunk_inputs,
                    chunk_targets,
                    chunk_forcings,
                    reset_mask,
                ):
                    current_state = _reset_temporal_state_lanes(state, reset_mask)
                    current_inputs = chunk_inputs[0]
                    weighted_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                    valid_count = jnp.asarray(0.0, dtype=jnp.float32)
                    keys = jax.random.split(key, len(chunk_inputs))
                    for bptt_i in range(len(chunk_inputs)):
                        if bptt_i < truth_prefix_steps:
                            current_inputs = chunk_inputs[bptt_i]
                        (loss_and_diag, predictions), current_state = transformed.apply(
                            params,
                            current_state,
                            keys[bptt_i],
                            current_inputs,
                            chunk_targets[bptt_i],
                            chunk_forcings[bptt_i],
                            False,
                        )
                        if include_bptt_loss_step(
                            rolling_loss_mode,
                            bptt_i,
                            truth_prefix_steps,
                        ):
                            loss_by_lane = _loss_by_lane(loss_and_diag[0])
                            weighted_loss_sum = weighted_loss_sum + jnp.sum(loss_by_lane)
                            valid_count = valid_count + jnp.asarray(loss_by_lane.size, dtype=loss_by_lane.dtype)
                        if bptt_i < len(chunk_inputs) - 1:
                            if bptt_i + 1 < truth_prefix_steps:
                                current_inputs = chunk_inputs[bptt_i + 1]
                            else:
                                current_inputs = _advance_autoregressive_inputs(
                                    current_inputs,
                                    predictions,
                                    chunk_forcings[bptt_i],
                                )
                    return (
                        current_state,
                        weighted_loss_sum,
                        valid_count,
                    )
            else:
                @jax.jit
                def eval_chunk(params, state, key, chunk_inputs, chunk_targets, chunk_forcings, reset_mask):
                    current_state = _reset_temporal_state_lanes(state, reset_mask)
                    losses = []
                    keys = jax.random.split(key, len(chunk_inputs) * 2)
                    for bptt_i in range(len(chunk_inputs)):
                        loss_key = keys[2 * bptt_i]
                        carry_key = keys[2 * bptt_i + 1]
                        state_before_rollout = current_state
                        (loss_and_diag, rollout_state) = transformed.apply(
                            params,
                            state_before_rollout,
                            loss_key,
                            chunk_inputs[bptt_i],
                            chunk_targets[bptt_i],
                            chunk_forcings[bptt_i],
                            False,
                        )
                        losses.append(scalarize_loss(loss_and_diag[0]))
                        if chunk_targets[bptt_i].sizes["time"] > 1:
                            first_targets = chunk_targets[bptt_i].isel(time=slice(0, 1))
                            first_forcings = chunk_forcings[bptt_i].isel(time=slice(0, 1))
                            (_carry_loss_and_diag, current_state) = transformed.apply(
                                params,
                                state_before_rollout,
                                carry_key,
                                chunk_inputs[bptt_i],
                                first_targets,
                                first_forcings,
                                False,
                            )
                        else:
                            current_state = rollout_state
                    return current_state, jnp.stack(losses)

            eval_chunk_fn_by_lane_count[lane_count] = eval_chunk

        if rolling_ar:
            (
                next_state,
                chunk_loss_sum,
                chunk_valid_count,
            ) = eval_chunk_fn_by_lane_count[lane_count](
                params,
                state_by_lane_count[lane_count],
                apply_key,
                chunk_inputs,
                chunk_targets,
                chunk_forcings,
                jnp.asarray(chunk.reset_mask),
            )
            state_by_lane_count[lane_count] = next_state
            chunk_loss_sum_f = float(jax.device_get(chunk_loss_sum))
            chunk_valid_count_f = float(jax.device_get(chunk_valid_count))
            total_weighted_loss += chunk_loss_sum_f
            total_windows += int(chunk_valid_count_f)
            current_loss = chunk_loss_sum_f / max(chunk_valid_count_f, 1.0)
        else:
            next_state, chunk_losses = eval_chunk_fn_by_lane_count[lane_count](
                params,
                state_by_lane_count[lane_count],
                apply_key,
                chunk_inputs,
                chunk_targets,
                chunk_forcings,
                jnp.asarray(chunk.reset_mask),
            )
            state_by_lane_count[lane_count] = next_state

            chunk_losses_np = np.asarray(jax.device_get(chunk_losses), dtype=np.float64)
            total_weighted_loss += float(chunk_losses_np.sum()) * lane_count
            total_windows += lane_count * len(chunk_inputs)
            current_loss = float(chunk_losses_np.mean())

        if chunk_i == 1 or chunk_i % 10 == 0 or chunk_i == n_chunks:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] chunk {chunk_i}/{n_chunks} "
                f"elapsed {elapsed:.1f}s current_loss {current_loss:.6f}"
            )

    if total_windows <= 0:
        raise ValueError("No eval windows were available for the chunk-local autoregressive tail loss.")

    return {
        "total": float(total_weighted_loss / total_windows),
        "segments": float(len(eval_segments)),
        "chunks": float(n_chunks),
        **selection.metadata(item_name="segments"),
    }


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
