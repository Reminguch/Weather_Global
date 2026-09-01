"""Selective prepared-array reads for v24_Ilya endpoint BPTT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class FrameDataReport:
    """Logical materialization performed for one BPTT chunk."""

    teacher_input_windows: int
    unique_teacher_frames: int
    forcing_steps: int
    truth_target_steps: int
    bytes_loaded: int


@dataclass(frozen=True)
class EndpointFrameBatch:
    """Unique-frame representation consumed by the explicit reverse objective."""

    input_frames: tuple[xr.Dataset, ...]
    static_inputs: xr.Dataset
    truths: tuple[xr.Dataset, ...]
    forcings: tuple[xr.Dataset, ...]
    report: FrameDataReport


@dataclass
class EndpointFrameWorkspace:
    """Reusable owned FP32 truth slabs, one independent slot per replica."""

    _truth_buffers: dict[tuple[int, str, tuple[int, ...], str], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def take_truth(
        self,
        *,
        slot: int,
        name: str,
        source: Any,
        indices: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        if slot < 0:
            raise ValueError("truth workspace slot must be non-negative")
        source_array = np.asarray(source)
        shape = list(source_array.shape)
        shape[axis] = int(indices.size)
        output_shape = tuple(shape)
        dtype = (
            np.dtype(np.float32)
            if np.issubdtype(source_array.dtype, np.inexact)
            else np.dtype(source_array.dtype)
        )
        key = (slot, name, output_shape, dtype.str)
        output = self._truth_buffers.get(key)
        if output is None:
            output = np.empty(output_shape, dtype=dtype)
            self._truth_buffers[key] = output
        np.take(source_array, indices, axis=axis, out=output)
        return output


def _take_time(source: Any, indices: np.ndarray) -> np.ndarray:
    axis = source.dims.index("time")
    data = source.data
    if hasattr(data, "take") and not isinstance(data, np.ndarray):
        values = data.take(indices, axis=axis)
    else:
        values = np.take(data, indices, axis=axis)
    if axis != 0:
        values = np.moveaxis(values, axis, 0)
    return np.asarray(values)


def _temporal_dataset(
    store,
    variable_names: Iterable[str],
    local_indices: np.ndarray,
    *,
    dt: pd.Timedelta,
    truth_workspace: EndpointFrameWorkspace | None = None,
    truth_workspace_slot: int = 0,
) -> tuple[xr.Dataset, int]:
    local = np.asarray(local_indices, dtype=np.int64)
    if local.ndim != 1 or local.size == 0:
        raise ValueError("Temporal selection requires a non-empty 1D index array")
    if local.min() < 0 or local.max() >= store.sizes["time"]:
        raise IndexError(
            f"Temporal selection {local.min()}..{local.max()} lies outside "
            f"0..{store.sizes['time'] - 1}"
        )
    global_indices = np.asarray(store._time_indices, dtype=np.int64)[local]
    time_ns = int(dt / pd.Timedelta(1, "ns"))
    time_coord = np.arange(local.size, dtype=np.int64) * np.timedelta64(time_ns, "ns")
    variables: dict[str, xr.DataArray] = {}
    bytes_loaded = 0
    for name in variable_names:
        source = store.data_vars[name]
        if "time" not in source.dims:
            continue
        if truth_workspace is None:
            values = _take_time(source, global_indices)
        else:
            axis = source.dims.index("time")
            values = truth_workspace.take_truth(
                slot=truth_workspace_slot,
                name=name,
                source=source.data,
                indices=global_indices,
                axis=axis,
            )
            if axis != 0:
                values = np.moveaxis(values, axis, 0)
        bytes_loaded += int(values.nbytes)
        remaining_dims = tuple(dim for dim in source.dims if dim != "time")
        dims = ("batch", "time", *remaining_dims)
        coords: dict[str, Any] = {
            "batch": np.arange(1),
            "time": time_coord,
        }
        for dim in remaining_dims:
            if dim in source.coords:
                coords[dim] = source.coords[dim]
            elif dim in store.coords:
                coords[dim] = np.asarray(store.coords[dim])
        variables[name] = xr.DataArray(values[None, ...], dims=dims, coords=coords)
    coords = {"batch": np.arange(1), "time": time_coord}
    return xr.Dataset(variables, coords=coords), bytes_loaded


def _static_dataset(
    store,
    variable_names: Iterable[str],
) -> tuple[xr.Dataset, int]:
    variables: dict[str, xr.DataArray] = {}
    bytes_loaded = 0
    for name in variable_names:
        source = store.data_vars[name]
        if "time" in source.dims:
            continue
        values = np.asarray(source.data)
        bytes_loaded += int(values.nbytes)
        dims = ("batch", *source.dims)
        coords: dict[str, Any] = {"batch": np.arange(1)}
        for dim in source.dims:
            if dim in source.coords:
                coords[dim] = source.coords[dim]
            elif dim in store.coords:
                coords[dim] = np.asarray(store.coords[dim])
        variables[name] = xr.DataArray(values[None, ...], dims=dims, coords=coords)
    return xr.Dataset(variables, coords={"batch": np.arange(1)}), bytes_loaded


def _split_steps(dataset: xr.Dataset, dt: pd.Timedelta) -> tuple[xr.Dataset, ...]:
    target_time = np.asarray([np.timedelta64(int(dt / pd.Timedelta(1, "ns")), "ns")])
    return tuple(
        dataset.isel(time=slice(index, index + 1)).assign_coords(time=target_time)
        for index in range(dataset.sizes.get("time", 0))
    )


def load_endpoint_frame_batch(
    *,
    store,
    raw_anchor_indices: np.ndarray,
    input_steps: int,
    truth_prefix_steps: int,
    loss_mode: str,
    supervised_step_indices: tuple[int, ...] | None = None,
    task_config,
    dt: pd.Timedelta,
    truth_workspace: EndpointFrameWorkspace | None = None,
    truth_workspace_slot: int = 0,
) -> EndpointFrameBatch:
    """Read teacher inputs, forcings, and only the truth required by the loss."""

    anchors = np.asarray(raw_anchor_indices, dtype=np.int64)
    if anchors.ndim != 1 or anchors.size == 0:
        raise ValueError("raw_anchor_indices must be a non-empty 1D array")
    if input_steps != 2:
        raise ValueError(
            "v24_Ilya unique-frame BPTT currently requires exactly two input steps"
        )
    if not 1 <= truth_prefix_steps <= anchors.size:
        raise ValueError("truth_prefix_steps must lie within the BPTT chunk")
    if anchors.size > 1 and not np.all(np.diff(anchors) == 1):
        raise ValueError("v24_Ilya BPTT chunks require consecutive anchor indices")
    if loss_mode not in {"last_step", "all_steps", "sparse_steps"}:
        raise ValueError(f"Unsupported loss_mode={loss_mode!r}")
    if loss_mode == "sparse_steps":
        if not supervised_step_indices:
            raise ValueError(
                "sparse_steps requires non-empty supervised_step_indices"
            )
        selected = np.asarray(supervised_step_indices, dtype=np.int64)
        if (
            selected.ndim != 1
            or np.any(selected < 0)
            or np.any(selected >= anchors.size)
        ):
            raise ValueError(
                "supervised_step_indices must lie within the BPTT chunk"
            )
        if selected.size > 1 and np.any(np.diff(selected) <= 0):
            raise ValueError("supervised_step_indices must be ordered and unique")
        if selected[-1] != anchors.size - 1:
            raise ValueError("sparse_steps must supervise the final BPTT endpoint")
    elif supervised_step_indices is not None:
        raise ValueError(
            "supervised_step_indices is only valid for loss_mode='sparse_steps'"
        )

    first = int(anchors[0])
    teacher_frame_indices = np.arange(
        first - (input_steps - 1),
        first + truth_prefix_steps,
        dtype=np.int64,
    )
    input_names = tuple(task_config.input_variables)
    dynamic_input_names = tuple(
        name for name in input_names if "time" in store.data_vars[name].dims
    )
    teacher_data, teacher_bytes = _temporal_dataset(
        store,
        dynamic_input_names,
        teacher_frame_indices,
        dt=dt,
    )
    input_frames = tuple(
        teacher_data.isel(time=slice(index, index + 1))
        for index in range(teacher_data.sizes["time"])
    )
    static_inputs, static_bytes = _static_dataset(store, input_names)

    target_indices = anchors + 1
    forcing_data, forcing_bytes = _temporal_dataset(
        store,
        task_config.forcing_variables,
        target_indices,
        dt=dt,
    )
    forcings = _split_steps(forcing_data, dt)

    if loss_mode == "last_step":
        truth_indices = target_indices[-1:]
    elif loss_mode == "all_steps":
        truth_indices = target_indices
    else:
        truth_indices = target_indices[selected]
    truth_data, truth_bytes = _temporal_dataset(
        store,
        task_config.target_variables,
        truth_indices,
        dt=dt,
        truth_workspace=truth_workspace,
        truth_workspace_slot=truth_workspace_slot,
    )
    truths = _split_steps(truth_data, dt)
    return EndpointFrameBatch(
        input_frames=input_frames,
        static_inputs=static_inputs,
        truths=truths,
        forcings=forcings,
        report=FrameDataReport(
            teacher_input_windows=truth_prefix_steps,
            unique_teacher_frames=input_steps + truth_prefix_steps - 1,
            forcing_steps=int(anchors.size),
            truth_target_steps=len(truths),
            bytes_loaded=teacher_bytes + static_bytes + forcing_bytes + truth_bytes,
        ),
    )
