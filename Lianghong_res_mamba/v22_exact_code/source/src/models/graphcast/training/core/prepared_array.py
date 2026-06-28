from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .model import gc


PREPARED_ARRAY_FORMAT_VERSION = 1


@dataclasses.dataclass(frozen=True)
class PreparedArrayVar:
    data: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class PreparedArrayBlock:
    block_start: int
    block_stop: int
    vars: dict[str, PreparedArrayVar]
    bytes_loaded: int


def _timedelta_coords(count: int, start_steps: int, dt: pd.Timedelta) -> np.ndarray:
    step_ns = int(dt / pd.Timedelta(1, "ns"))
    offsets = (start_steps + np.arange(count, dtype=np.int64)) * np.timedelta64(step_ns, "ns")
    return offsets.astype("timedelta64[ns]")


class PreparedArrayStore:
    """Memmap-backed prepared GraphCast arrays with GraphCast batch builders."""

    def __init__(
        self,
        root: str | Path,
        *,
        time_indices: np.ndarray | None = None,
        label: str = "prepared-array",
    ) -> None:
        self.root = Path(root)
        self.label = label
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Prepared array metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.coords = {
            path.stem: np.load(path, mmap_mode="r")
            for path in sorted((self.root / "coords").glob("*.npy"))
        }
        self._vars: dict[str, PreparedArrayVar] = {}
        for name, info in self.metadata["variables"].items():
            data = np.load(self.root / "vars" / f"{name}.npy", mmap_mode="r")
            dims = tuple(info["dims"])
            coords = {
                dim: np.asarray(self.coords[dim])
                for dim in dims
                if dim in self.coords
            }
            self._vars[name] = PreparedArrayVar(data=data, dims=dims, coords=coords)
        full_time_size = int(np.asarray(self.coords["time"]).shape[0])
        self._time_indices = (
            np.arange(full_time_size, dtype=np.int64)
            if time_indices is None
            else np.asarray(time_indices, dtype=np.int64)
        )
        self.sizes = {
            "time": int(self._time_indices.size),
            **{
                name: int(np.asarray(values).shape[0])
                for name, values in self.coords.items()
                if name != "time"
            },
        }
        self.time = SimpleNamespace(values=np.asarray(self.coords["time"])[self._time_indices])

    @property
    def data_vars(self) -> dict[str, PreparedArrayVar]:
        return self._vars

    def split_by_time_indices(self, time_indices: np.ndarray, *, label: str | None = None) -> "PreparedArrayStore":
        return PreparedArrayStore(
            self.root,
            time_indices=np.asarray(self._time_indices)[np.asarray(time_indices, dtype=np.int64)],
            label=label or self.label,
        )

    def validate(self, *, resolution: float, task_cfg: gc.TaskConfig) -> None:
        version = int(self.metadata.get("prepared_array_format_version", -1))
        if version != PREPARED_ARRAY_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported prepared_array_format_version={version}; "
                f"expected {PREPARED_ARRAY_FORMAT_VERSION}."
            )
        stored_resolution = self.metadata.get("resolution")
        if stored_resolution is not None and not np.isclose(float(stored_resolution), resolution, atol=1e-6):
            raise ValueError(
                f"Prepared array store resolution={stored_resolution} does not match requested {resolution}."
            )
        stored_levels = [int(level) for level in self.metadata.get("pressure_levels", [])]
        requested_levels = [int(level) for level in task_cfg.pressure_levels]
        if stored_levels and stored_levels != requested_levels:
            raise ValueError(
                "Prepared array store pressure_levels do not match checkpoint task: "
                f"prepared={stored_levels}, requested={requested_levels}."
            )
        required = set(task_cfg.input_variables) | set(task_cfg.target_variables) | set(task_cfg.forcing_variables)
        missing = sorted(name for name in required if name not in self._vars)
        if missing:
            raise ValueError(f"Prepared array store is missing task variables: {missing}")
        time_values = pd.DatetimeIndex(pd.to_datetime(self.time.values))
        if len(time_values) < 2:
            raise ValueError("Prepared array store must contain at least two time steps.")
        if not time_values.is_monotonic_increasing:
            raise ValueError("Prepared array store time coordinate must be sorted ascending.")
        deltas = np.diff(time_values.values).astype("timedelta64[ns]")
        expected = np.array(np.timedelta64(6, "h")).astype("timedelta64[ns]")
        if not np.all(deltas == expected):
            raise ValueError("Prepared array store time grid must be strictly 6-hourly.")

    def estimate_task_nbytes(self, variable_names: Iterable[str]) -> int:
        total = 0
        for name in variable_names:
            var = self._vars.get(name)
            if var is None:
                continue
            if "time" in var.dims:
                time_axis = var.dims.index("time")
                per_time = int(var.data.nbytes // max(1, var.data.shape[time_axis]))
                total += per_time * int(self._time_indices.size)
            else:
                total += int(var.data.nbytes)
        return total

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
        local_windows = batch_indices[:, None] + window_offsets[None, :]
        if local_windows.min() < 0 or local_windows.max() >= self._time_indices.size:
            raise IndexError(
                f"Requested batch outside valid time range: min={local_windows.min()}, "
                f"max={local_windows.max()}, total={self._time_indices.size}."
            )
        global_windows = self._time_indices[local_windows]
        return self._build_from_window_indices(
            global_windows=global_windows,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )

    def load_time_block(
        self,
        start: int,
        stop: int,
        *,
        task_cfg: gc.TaskConfig,
    ) -> PreparedArrayBlock:
        if start < 0 or stop > self._time_indices.size:
            raise IndexError(f"Requested block {start}:{stop} outside split time size {self._time_indices.size}.")
        global_start = int(self._time_indices[start])
        global_stop = int(self._time_indices[stop - 1]) + 1
        if not np.array_equal(self._time_indices[start:stop], np.arange(global_start, global_stop, dtype=np.int64)):
            raise ValueError("Prepared array time block loading requires contiguous underlying time indices.")
        task_vars = sorted(set(task_cfg.input_variables) | set(task_cfg.target_variables) | set(task_cfg.forcing_variables))
        vars_loaded: dict[str, PreparedArrayVar] = {}
        bytes_loaded = 0
        for name in task_vars:
            source = self._vars[name]
            data = source.data
            if "time" in source.dims:
                axis = source.dims.index("time")
                data = np.asarray(np.take(data, np.arange(global_start, global_stop, dtype=np.int64), axis=axis))
            else:
                data = np.asarray(data)
            bytes_loaded += int(data.nbytes)
            vars_loaded[name] = PreparedArrayVar(data=data, dims=source.dims, coords=source.coords)
        return PreparedArrayBlock(
            block_start=start,
            block_stop=stop,
            vars=vars_loaded,
            bytes_loaded=bytes_loaded,
        )

    def build_step_from_blocks(
        self,
        blocks: list[PreparedArrayBlock],
        final_indices: np.ndarray,
        *,
        input_steps: int,
        target_steps: int,
        task_cfg: gc.TaskConfig,
        dt: pd.Timedelta,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        final_indices = np.asarray(final_indices, dtype=np.int64)
        if len(blocks) != final_indices.size:
            raise ValueError(f"Expected {final_indices.size} blocks, got {len(blocks)}.")
        window_offsets = np.arange(-(input_steps - 1), target_steps + 1, dtype=np.int64)
        local_windows = final_indices[:, None] + window_offsets[None, :]
        if local_windows.min() < 0 or local_windows.max() >= self._time_indices.size:
            raise IndexError("Segment step outside split time range.")
        return self._build_from_window_indices(
            local_windows=local_windows,
            blocks=blocks,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )

    def _build_from_window_indices(
        self,
        *,
        input_steps: int,
        target_steps: int,
        task_cfg: gc.TaskConfig,
        dt: pd.Timedelta,
        global_windows: np.ndarray | None = None,
        local_windows: np.ndarray | None = None,
        blocks: list[PreparedArrayBlock] | None = None,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        if global_windows is None and local_windows is None:
            raise ValueError("Either global_windows or local_windows must be provided.")
        batch_size = int((global_windows if global_windows is not None else local_windows).shape[0])
        input_time = _timedelta_coords(input_steps, -(input_steps - 1), dt)
        target_time = _timedelta_coords(target_steps, 1, dt)
        input_pos = np.arange(input_steps, dtype=np.int64)
        target_pos = np.arange(input_steps, input_steps + target_steps, dtype=np.int64)
        batch_coord = np.arange(batch_size)

        def build_var(name: str, positions: np.ndarray, *, is_input_time: bool, expand_static: bool) -> xr.DataArray:
            source = self._vars[name]
            dims = source.dims
            if "time" in dims:
                time_axis = dims.index("time")
                if blocks is None:
                    assert global_windows is not None
                    gather = global_windows[:, positions]
                    if time_axis == 0:
                        data = np.take(source.data, gather, axis=time_axis)
                    else:
                        lane_arrays = []
                        for lane in range(batch_size):
                            data_lane = np.take(source.data, gather[lane], axis=time_axis)
                            data_lane = np.moveaxis(data_lane, time_axis, 0)
                            lane_arrays.append(data_lane)
                        data = np.stack(lane_arrays, axis=0)
                else:
                    assert local_windows is not None
                    lane_arrays = []
                    for lane, block in enumerate(blocks):
                        cached = block.vars[name]
                        local_positions = local_windows[lane, positions] - block.block_start
                        data_lane = np.take(cached.data, local_positions, axis=time_axis)
                        if time_axis != 0:
                            data_lane = np.moveaxis(data_lane, time_axis, 0)
                        lane_arrays.append(data_lane)
                    data = np.stack(lane_arrays, axis=0)
                remaining_dims = tuple(dim for dim in dims if dim != "time")
                dims_out = ("batch", "time", *remaining_dims)
                coords: dict[str, Any] = {
                    "batch": batch_coord,
                    "time": input_time if is_input_time else target_time,
                }
                for dim in remaining_dims:
                    if dim in source.coords:
                        coords[dim] = source.coords[dim]
                    elif dim in self.coords:
                        coords[dim] = np.asarray(self.coords[dim])
                return xr.DataArray(data, dims=dims_out, coords=coords)

            data = source.data if blocks is None else blocks[0].vars[name].data
            dims_out = dims
            if expand_static:
                data = np.broadcast_to(np.asarray(data)[None, ...], (batch_size, *data.shape))
                dims_out = ("batch", *dims)
            coords = {"batch": batch_coord} if "batch" in dims_out else {}
            for dim in dims:
                if dim in source.coords:
                    coords[dim] = source.coords[dim]
                elif dim in self.coords:
                    coords[dim] = np.asarray(self.coords[dim])
            return xr.DataArray(data, dims=dims_out, coords=coords)

        inputs = xr.Dataset(
            {
                name: build_var(name, input_pos, is_input_time=True, expand_static=True)
                for name in task_cfg.input_variables
            },
            coords={"batch": batch_coord, "time": input_time},
        )
        targets = xr.Dataset(
            {
                name: build_var(name, target_pos, is_input_time=False, expand_static=False)
                for name in task_cfg.target_variables
            },
            coords={"batch": batch_coord, "time": target_time},
        )
        forcing_vars = {
            name: build_var(name, target_pos, is_input_time=False, expand_static=False)
            for name in task_cfg.forcing_variables
        }
        forcing_coords = {"batch": batch_coord, "time": target_time} if forcing_vars else {"batch": batch_coord}
        forcings = xr.Dataset(forcing_vars, coords=forcing_coords)
        return inputs, targets, forcings


def is_prepared_array_store(obj: object) -> bool:
    return isinstance(obj, PreparedArrayStore)
