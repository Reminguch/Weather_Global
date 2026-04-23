from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .model import data_utils, gc


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


def _common_window_time(window_len: int, dt: pd.Timedelta) -> np.ndarray:
    step_ns = int(dt / pd.Timedelta(1, "ns"))
    if step_ns <= 0:
        raise ValueError(f"Invalid non-positive time step: {dt}")
    start = np.datetime64("2000-01-01T00:00:00", "ns")
    offsets = np.arange(window_len, dtype=np.int64) * np.timedelta64(step_ns, "ns")
    return start + offsets


def _ensure_batch_dim_for_static_vars(ds: xr.Dataset, batch_size: int) -> xr.Dataset:
    if "batch" not in ds.coords:
        ds = ds.assign_coords(batch=np.arange(batch_size))
    batch_values = np.asarray(ds.coords["batch"].values)
    for name in list(ds.data_vars):
        if "batch" in ds[name].dims:
            continue
        ds[name] = ds[name].expand_dims(batch=batch_values, axis=0)
    return ds


def build_batch_from_indices_vectorized(
    ds: xr.Dataset,
    *,
    indices: Iterable[int],
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    batch_indices = np.asarray(list(indices), dtype=np.int64)
    if batch_indices.size == 0:
        raise ValueError("Cannot build an empty batch.")

    window_offsets = np.arange(-(input_steps - 1), target_steps + 1, dtype=np.int64)
    window_indices = batch_indices[:, None] + window_offsets[None, :]
    if window_indices.min() < 0 or window_indices.max() >= ds.sizes["time"]:
        raise IndexError(
            f"Requested batch outside valid time range: min={window_indices.min()}, "
            f"max={window_indices.max()}, total={ds.sizes['time']}."
        )

    # The source datasets carry a singleton batch dimension. Drop it before
    # vectorized time indexing so the gathered dimension becomes the training
    # batch dimension rather than colliding with the source batch axis.
    if "batch" in ds.dims:
        if ds.sizes["batch"] != 1:
            raise ValueError(f"Expected source dataset batch size 1, got {ds.sizes['batch']}.")
        source = ds.isel(batch=0, drop=True)
    else:
        source = ds

    window_time = _common_window_time(len(window_offsets), dt)
    time_indexer = xr.DataArray(
        window_indices,
        dims=("batch", "time"),
        coords={"batch": np.arange(batch_indices.size)},
    )
    gathered = source.isel(time=time_indexer)

    if "datetime" in gathered.coords:
        datetime_values = np.asarray(gathered.coords["datetime"].values)
    else:
        datetime_values = np.asarray(gathered.coords["time"].values)
    gathered = gathered.assign_coords(
        batch=np.arange(batch_indices.size),
        time=("time", window_time),
        datetime=(("batch", "time"), datetime_values),
    )

    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        gathered,
        input_variables=task_cfg.input_variables,
        target_variables=task_cfg.target_variables,
        forcing_variables=task_cfg.forcing_variables,
        pressure_levels=task_cfg.pressure_levels,
        input_duration=task_cfg.input_duration,
        target_lead_times=lead_times(target_steps, dt),
    )
    inputs = _ensure_batch_dim_for_static_vars(inputs, batch_indices.size)
    targets = _ensure_batch_dim_for_static_vars(targets, batch_indices.size)
    forcings = _ensure_batch_dim_for_static_vars(forcings, batch_indices.size)
    return inputs.load(), targets.load(), forcings.load()


def _timedelta_coords(count: int, start_steps: int, dt: pd.Timedelta) -> np.ndarray:
    step_ns = int(dt / pd.Timedelta(1, "ns"))
    offsets = (start_steps + np.arange(count, dtype=np.int64)) * np.timedelta64(step_ns, "ns")
    return offsets.astype("timedelta64[ns]")


@dataclasses.dataclass(frozen=True)
class _CachedVar:
    data: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]


class NumpyBatchCache:
    """Fast batch builder that gathers prepared GraphCast variables with numpy."""

    def __init__(self, ds: xr.Dataset, task_cfg: gc.TaskConfig, *, label: str) -> None:
        t0 = time.time()
        self._label = label
        self._vars: dict[str, _CachedVar] = {}
        self._sizes = dict(ds.sizes)
        self._time_size = int(ds.sizes["time"])
        self._coord_values = {
            name: np.asarray(coord.values)
            for name, coord in ds.coords.items()
            if name in {"lat", "lon", "level"}
        }
        self._pressure_levels = np.asarray(task_cfg.pressure_levels)
        for name, var in ds.data_vars.items():
            data = np.asarray(var.values)
            dims = tuple(var.dims)
            coords = {
                dim: np.asarray(var.coords[dim].values)
                for dim in dims
                if dim in var.coords
            }
            if "batch" in dims:
                batch_axis = dims.index("batch")
                if data.shape[batch_axis] != 1:
                    raise ValueError(
                        f"[numpy-cache:{label}] expected singleton source batch for {name}, "
                        f"got shape {data.shape}."
                    )
                data = np.squeeze(data, axis=batch_axis)
                dims = tuple(dim for dim in dims if dim != "batch")
                coords.pop("batch", None)
            if "level" in dims:
                level_values = coords.get("level", self._coord_values.get("level"))
                if level_values is None:
                    raise ValueError(f"[numpy-cache:{label}] variable {name} has no level coordinate.")
                missing = [level for level in self._pressure_levels if level not in set(level_values.tolist())]
                if missing:
                    raise ValueError(f"[numpy-cache:{label}] missing pressure levels for {name}: {missing}")
                level_indices = np.asarray([int(np.where(level_values == level)[0][0]) for level in self._pressure_levels])
                axis = dims.index("level")
                data = np.take(data, level_indices, axis=axis)
                coords["level"] = self._pressure_levels
            self._vars[name] = _CachedVar(data=data, dims=dims, coords=coords)
        total_gib = sum(var.data.nbytes for var in self._vars.values()) / (1024**3)
        print(
            f"[numpy-cache] built {label} cache with {len(self._vars)} vars, "
            f"{total_gib:.2f} GiB in {time.time() - t0:.1f}s."
        )

    @property
    def active(self) -> bool:
        return True

    def build_batch_from_indices(
        self,
        *,
        indices: Iterable[int],
        input_steps: int,
        target_steps: int,
        task_cfg: gc.TaskConfig,
        dt: pd.Timedelta,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        batch_indices = np.asarray(list(indices), dtype=np.int64)
        if batch_indices.size == 0:
            raise ValueError("Cannot build an empty batch.")

        window_offsets = np.arange(-(input_steps - 1), target_steps + 1, dtype=np.int64)
        window_indices = batch_indices[:, None] + window_offsets[None, :]
        if window_indices.min() < 0 or window_indices.max() >= self._time_size:
            raise IndexError(
                f"Requested batch outside valid time range: min={window_indices.min()}, "
                f"max={window_indices.max()}, total={self._time_size}."
            )

        input_time = _timedelta_coords(input_steps, -(input_steps - 1), dt)
        target_time = _timedelta_coords(target_steps, 1, dt)
        input_pos = np.arange(input_steps, dtype=np.int64)
        target_pos = np.arange(input_steps, input_steps + target_steps, dtype=np.int64)

        batch_coord = np.arange(batch_indices.size)

        def build_var(name: str, positions: np.ndarray, *, expand_static_time: np.ndarray | None) -> xr.DataArray:
            if name not in self._vars:
                raise KeyError(f"[numpy-cache:{self._label}] variable {name!r} is not cached.")
            cached = self._vars[name]
            dims = cached.dims
            if "time" in dims:
                time_axis = dims.index("time")
                gather_indices = window_indices[:, positions]
                data = np.take(cached.data, gather_indices, axis=time_axis)
                remaining_dims = tuple(dim for dim in dims if dim != "time")
                dims_out = ("batch", "time", *remaining_dims)
                coords: dict[str, Any] = {"batch": batch_coord}
                coords["time"] = input_time if len(positions) == input_steps and positions[0] == 0 else target_time
                for dim in remaining_dims:
                    if dim in cached.coords:
                        coords[dim] = cached.coords[dim]
                    elif dim in self._coord_values:
                        coords[dim] = self._coord_values[dim]
                return xr.DataArray(data, dims=dims_out, coords=coords)

            data = cached.data
            dims_out = dims
            if expand_static_time is not None:
                data = np.broadcast_to(data[None, ...], (batch_indices.size, *data.shape))
                dims_out = ("batch", *dims)
            coords = {"batch": batch_coord} if "batch" in dims_out else {}
            for dim in dims:
                if dim in cached.coords:
                    coords[dim] = cached.coords[dim]
                elif dim in self._coord_values:
                    coords[dim] = self._coord_values[dim]
            return xr.DataArray(data, dims=dims_out, coords=coords)

        inputs = xr.Dataset(
            {
                name: build_var(name, input_pos, expand_static_time=input_time)
                for name in task_cfg.input_variables
            },
            coords={"batch": batch_coord, "time": input_time},
        )
        targets = xr.Dataset(
            {
                name: build_var(name, target_pos, expand_static_time=None)
                for name in task_cfg.target_variables
            },
            coords={"batch": batch_coord, "time": target_time},
        )
        forcing_vars = {
            name: build_var(name, target_pos, expand_static_time=None)
            for name in task_cfg.forcing_variables
        }
        forcing_coords = {"batch": batch_coord, "time": target_time} if forcing_vars else {"batch": batch_coord}
        forcings = xr.Dataset(forcing_vars, coords=forcing_coords)
        return inputs, targets, forcings



BatchBuilder = Callable[..., tuple[xr.Dataset, xr.Dataset, xr.Dataset]]
