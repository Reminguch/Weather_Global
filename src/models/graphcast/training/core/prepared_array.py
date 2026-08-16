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
PREPARED_ARRAY_SHARDED_FORMAT_VERSION = 2
SUPPORTED_PREPARED_ARRAY_FORMAT_VERSIONS = (
    PREPARED_ARRAY_FORMAT_VERSION,
    PREPARED_ARRAY_SHARDED_FORMAT_VERSION,
)


@dataclasses.dataclass(frozen=True)
class _PreparedArrayShard:
    start: int
    stop: int
    data: np.ndarray


class PreparedShardedArray:
    """A time-axis array assembled lazily from consecutive memmap shards."""

    def __init__(
        self,
        shards: list[_PreparedArrayShard],
        shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        require_full_coverage: bool = True,
    ) -> None:
        if not shards:
            raise ValueError("Prepared sharded array requires at least one shard.")
        self.shards = tuple(shards)
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape, dtype=np.int64)) * int(self.dtype.itemsize)
        expected_start = 0 if require_full_coverage else shards[0].start
        for shard in self.shards:
            if shard.start != expected_start:
                raise ValueError(
                    f"Prepared array shards must be consecutive: expected start {expected_start}, "
                    f"found {shard.start}."
                )
            if shard.stop <= shard.start:
                raise ValueError(f"Invalid prepared array shard range {shard.start}:{shard.stop}.")
            expected_shape = (shard.stop - shard.start, *self.shape[1:])
            if tuple(shard.data.shape) != expected_shape:
                raise ValueError(
                    f"Prepared array shard shape mismatch for {shard.start}:{shard.stop}: "
                    f"expected {expected_shape}, found {tuple(shard.data.shape)}."
                )
            if np.dtype(shard.data.dtype) != self.dtype:
                raise ValueError(
                    f"Prepared array shard dtype mismatch: expected {self.dtype}, found {shard.data.dtype}."
                )
            expected_start = shard.stop
        if require_full_coverage and expected_start != self.shape[0]:
            raise ValueError(
                f"Prepared array shards cover {expected_start} time steps, expected {self.shape[0]}."
            )

    def take(self, indices: np.ndarray, *, axis: int = 0) -> np.ndarray:
        if axis != 0:
            raise ValueError(f"Prepared sharded arrays only support time axis 0, got axis={axis}.")
        requested = np.asarray(indices, dtype=np.int64)
        normalized = requested.copy()
        normalized[normalized < 0] += self.shape[0]
        if normalized.size and (normalized.min() < 0 or normalized.max() >= self.shape[0]):
            raise IndexError(f"Prepared array index outside 0:{self.shape[0]}.")

        flat = normalized.reshape(-1)
        output = np.empty((flat.size, *self.shape[1:]), dtype=self.dtype)
        assigned = np.zeros(flat.size, dtype=bool)
        for shard in self.shards:
            positions = np.flatnonzero((flat >= shard.start) & (flat < shard.stop))
            if positions.size == 0:
                continue
            output[positions] = np.take(shard.data, flat[positions] - shard.start, axis=0)
            assigned[positions] = True
        if not np.all(assigned):
            missing = flat[~assigned]
            raise IndexError(f"Prepared array indices are not covered by a shard: {missing.tolist()}.")
        return output.reshape((*requested.shape, *self.shape[1:]))


def _take_array(data: np.ndarray | PreparedShardedArray, indices: np.ndarray, *, axis: int) -> np.ndarray:
    if isinstance(data, PreparedShardedArray):
        return data.take(indices, axis=axis)
    return np.take(data, indices, axis=axis)


@dataclasses.dataclass(frozen=True)
class PreparedArrayVar:
    data: np.ndarray | PreparedShardedArray
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
        time_start: str | np.datetime64 | pd.Timestamp | None = None,
        time_end: str | np.datetime64 | pd.Timestamp | None = None,
        allow_incomplete: bool = False,
        label: str = "prepared-array",
    ) -> None:
        self.root = Path(root)
        self.label = label
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Prepared array metadata not found: {metadata_path}")
        source_incomplete = (self.root / ".incomplete").exists()
        if source_incomplete and not allow_incomplete:
            raise RuntimeError(f"Prepared array store is incomplete: {self.root}")
        if (time_start is None) != (time_end is None):
            raise ValueError("Provide both time_start and time_end, or neither.")
        if allow_incomplete and time_start is None:
            raise ValueError(
                "allow_incomplete=True requires explicit time_start and time_end bounds."
            )
        self.metadata = json.loads(metadata_path.read_text())
        version = int(self.metadata.get("prepared_array_format_version", -1))
        if version not in SUPPORTED_PREPARED_ARRAY_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported prepared_array_format_version={version}; "
                f"expected one of {SUPPORTED_PREPARED_ARRAY_FORMAT_VERSIONS}."
            )
        self.coords = {
            path.stem: np.load(path, mmap_mode="r")
            for path in sorted((self.root / "coords").glob("*.npy"))
        }
        if "time" not in self.coords:
            raise ValueError(f"Prepared array store has no time coordinate: {self.root}")
        full_time_values = np.asarray(self.coords["time"]).astype("datetime64[ns]")
        full_time_size = int(full_time_values.shape[0])
        selected_by_range = np.arange(full_time_size, dtype=np.int64)
        normalized_start = None
        normalized_end = None
        if time_start is not None:
            normalized_start = np.datetime64(pd.Timestamp(time_start), "ns")
            normalized_end = np.datetime64(pd.Timestamp(time_end), "ns")
            if normalized_end < normalized_start:
                raise ValueError("time_end must be on or after time_start.")
            selected_by_range = np.flatnonzero(
                (full_time_values >= normalized_start)
                & (full_time_values <= normalized_end)
            ).astype(np.int64)
            if selected_by_range.size == 0:
                raise ValueError(
                    f"Prepared array time range {normalized_start}..{normalized_end} is empty."
                )
            if (
                full_time_values[selected_by_range[0]] != normalized_start
                or full_time_values[selected_by_range[-1]] != normalized_end
            ):
                raise ValueError(
                    "time_start and time_end must exactly match prepared time coordinates."
                )

        if time_indices is None:
            self._time_indices = selected_by_range
        else:
            requested_indices = np.asarray(time_indices, dtype=np.int64)
            if requested_indices.ndim != 1:
                raise ValueError("time_indices must be one-dimensional.")
            if requested_indices.size and (
                requested_indices.min() < 0 or requested_indices.max() >= full_time_size
            ):
                raise IndexError(f"time_indices must lie within 0:{full_time_size}.")
            if time_start is not None and not np.all(
                np.isin(requested_indices, selected_by_range)
            ):
                raise ValueError("time_indices must lie within time_start..time_end.")
            self._time_indices = requested_indices
        if self._time_indices.size == 0:
            raise ValueError("Prepared array selection contains no time steps.")

        completed_months: dict[str, str] = {}
        build_manifest_path = self.root / "build_manifest.json"
        if source_incomplete:
            if not build_manifest_path.is_file():
                raise FileNotFoundError(
                    "Incomplete prepared store is missing build_manifest.json: "
                    f"{build_manifest_path}"
                )
            build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
            month_records = build_manifest.get("months")
            if not isinstance(month_records, dict):
                raise ValueError(
                    f"Malformed prepared build manifest months: {build_manifest_path}"
                )
            selected_months = sorted(
                set(
                    pd.DatetimeIndex(full_time_values[selected_by_range]).strftime("%Y-%m")
                )
            )
            for month in selected_months:
                record = month_records.get(month)
                status = record.get("status") if isinstance(record, dict) else None
                completed_months[month] = str(status)
                if status != "complete":
                    raise RuntimeError(
                        f"Prepared month {month} is not complete in {build_manifest_path}: "
                        f"status={status!r}"
                    )

        self._requested_time_start = (
            str(normalized_start) if normalized_start is not None else None
        )
        self._requested_time_end = (
            str(normalized_end) if normalized_end is not None else None
        )
        self._allow_incomplete = bool(allow_incomplete)
        selected_times = full_time_values[self._time_indices]
        self.selection_metadata = {
            "time_start": str(selected_times[0]),
            "time_end": str(selected_times[-1]),
            "selected_time_steps": int(selected_times.size),
            "source_incomplete": bool(source_incomplete),
            "allow_incomplete": bool(allow_incomplete),
            "completed_months": completed_months,
            "build_manifest": str(build_manifest_path) if source_incomplete else None,
        }
        self._vars: dict[str, PreparedArrayVar] = {}
        for name, info in self.metadata["variables"].items():
            dims = tuple(info["dims"])
            if version == PREPARED_ARRAY_SHARDED_FORMAT_VERSION and "time" in dims:
                if dims.index("time") != 0:
                    raise ValueError(
                        f"Format-v2 variable {name!r} must use time axis 0, found dims={dims}."
                    )
                shards = []
                for shard_info in self.metadata.get("time_shards", []):
                    start = int(shard_info["start"])
                    stop = int(shard_info["stop"])
                    if not np.any(
                        (self._time_indices >= start) & (self._time_indices < stop)
                    ):
                        continue
                    shard_root = self.root / str(shard_info["path"])
                    shard_path = shard_root / "vars" / f"{name}.npy"
                    if not shard_path.is_file():
                        raise FileNotFoundError(
                            f"Prepared array selection requires missing shard: {shard_path}"
                        )
                    shard_data = np.load(shard_path, mmap_mode="r")
                    shards.append(_PreparedArrayShard(start=start, stop=stop, data=shard_data))
                data = PreparedShardedArray(
                    shards,
                    shape=tuple(int(value) for value in info["shape"]),
                    dtype=np.dtype(info["dtype"]),
                    require_full_coverage=(time_start is None and time_indices is None),
                )
            else:
                data = np.load(self.root / "vars" / f"{name}.npy", mmap_mode="r")
            coords = {
                dim: np.asarray(self.coords[dim])
                for dim in dims
                if dim in self.coords
            }
            self._vars[name] = PreparedArrayVar(data=data, dims=dims, coords=coords)
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
            time_start=self._requested_time_start,
            time_end=self._requested_time_end,
            allow_incomplete=self._allow_incomplete,
            label=label or self.label,
        )

    def validate(self, *, resolution: float, task_cfg: gc.TaskConfig) -> None:
        version = int(self.metadata.get("prepared_array_format_version", -1))
        if version not in SUPPORTED_PREPARED_ARRAY_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported prepared_array_format_version={version}; "
                f"expected one of {SUPPORTED_PREPARED_ARRAY_FORMAT_VERSIONS}."
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
                data = np.asarray(
                    _take_array(data, np.arange(global_start, global_stop, dtype=np.int64), axis=axis)
                )
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
                        data = _take_array(source.data, gather, axis=time_axis)
                    else:
                        lane_arrays = []
                        for lane in range(batch_size):
                            data_lane = _take_array(source.data, gather[lane], axis=time_axis)
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
