from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from src.data_operations.loaders.graphcast_dataset import open_graphcast_era5
from src.models.graphcast.training.core.config import DEFAULT_CKPT, GRAPHCAST_VARS
from src.models.graphcast.training.core.dataset import (
    _ensure_datetime_coord,
    prepare_dataset_for_task,
    resolution_tag,
    select_resolution,
)
from src.models.graphcast.training.core.model import load_graphcast_checkpoint
from src.models.graphcast.training.core.prepared_array import PREPARED_ARRAY_FORMAT_VERSION


DEFAULT_PREPARED_STREAM_ROOT = "data/graphcast/graphcast/dataset/prepared_stream"
DEFAULT_PREPARE_DATA_PATH = "data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr"
DEFAULT_RESOLUTIONS = (1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 18.0)


def _jsonable_tuple(values: Iterable) -> list:
    return [item.item() if hasattr(item, "item") else item for item in values]


def _cast_float_vars_to_float32(ds: xr.Dataset) -> xr.Dataset:
    for name in list(ds.data_vars):
        if ds[name].dtype.kind == "f" and ds[name].dtype != np.float32:
            ds[name] = ds[name].astype(np.float32)
    return ds


def _drop_singleton_batch(var: xr.DataArray) -> xr.DataArray:
    if "batch" not in var.dims:
        return var
    if var.sizes["batch"] != 1:
        raise ValueError(f"Expected singleton batch for {var.name}, got {var.sizes['batch']}.")
    return var.isel(batch=0, drop=True)


def _select_pressure_levels(var: xr.DataArray, pressure_levels: Iterable[int]) -> xr.DataArray:
    if "level" not in var.dims:
        return var
    return var.sel(level=list(pressure_levels))


def _finite_time_mask(values: np.ndarray, dims: tuple[str, ...]) -> list[bool] | None:
    if "time" not in dims:
        return None
    time_axis = dims.index("time")
    data = np.moveaxis(values, time_axis, 0)
    return np.isfinite(data.reshape(data.shape[0], -1)).all(axis=1).astype(bool).tolist()


def _write_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(values))


def _prepare_resolution_store(
    source: xr.Dataset,
    *,
    source_data_path: str,
    task_cfg,
    resolution: float,
    out_root: Path,
    overwrite: bool,
) -> Path:
    store_path = out_root / resolution_tag(resolution)
    if store_path.exists():
        if not overwrite:
            raise FileExistsError(f"Prepared streaming store already exists: {store_path}. Use --overwrite.")
        shutil.rmtree(store_path)

    ds_res, base_resolution, stride = select_resolution(source.copy(), resolution)
    del stride
    prepared = prepare_dataset_for_task(ds_res, task_cfg)
    prepared = _cast_float_vars_to_float32(prepared)
    task_vars = tuple(
        dict.fromkeys(
            list(task_cfg.input_variables)
            + list(task_cfg.target_variables)
            + list(task_cfg.forcing_variables)
        )
    )
    missing = [name for name in task_vars if name not in prepared.data_vars]
    if missing:
        raise ValueError(f"Prepared source is missing task variables: {missing}")

    coords_dir = store_path / "coords"
    vars_dir = store_path / "vars"
    coords_dir.mkdir(parents=True, exist_ok=True)
    vars_dir.mkdir(parents=True, exist_ok=True)

    coord_names = ["time", "lat", "lon", "level"]
    written_coords: dict[str, dict] = {}
    for name in coord_names:
        if name in prepared.coords:
            values = (
                np.asarray(task_cfg.pressure_levels)
                if name == "level"
                else np.asarray(prepared.coords[name].values)
            )
            _write_array(coords_dir / f"{name}.npy", values)
            written_coords[name] = {"shape": list(values.shape), "dtype": str(values.dtype)}

    variables: dict[str, dict] = {}
    validity: dict[str, dict] = {
        "static_nonfinite_variables": [],
        "time_finite_by_variable": {},
    }
    for name in task_vars:
        var = _select_pressure_levels(prepared[name], task_cfg.pressure_levels)
        var = _drop_singleton_batch(var)
        values = np.asarray(var.values)
        if values.dtype.kind == "f" and values.dtype != np.float32:
            values = values.astype(np.float32)
        dims = tuple(var.dims)
        _write_array(vars_dir / f"{name}.npy", values)
        variables[name] = {
            "dims": list(dims),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        }
        finite_mask = _finite_time_mask(values, dims)
        if finite_mask is None:
            if not np.isfinite(values).all():
                validity["static_nonfinite_variables"].append(name)
        else:
            validity["time_finite_by_variable"][name] = finite_mask

    time_values = np.asarray(prepared.time.values).astype("datetime64[ns]")
    metadata = {
        "prepared_array_format_version": PREPARED_ARRAY_FORMAT_VERSION,
        "source_data_path": source_data_path,
        "resolution": float(resolution),
        "base_resolution": float(base_resolution),
        "pressure_levels": _jsonable_tuple(task_cfg.pressure_levels),
        "task_input_variables": list(task_cfg.input_variables),
        "task_target_variables": list(task_cfg.target_variables),
        "task_forcing_variables": list(task_cfg.forcing_variables),
        "coords": written_coords,
        "variables": variables,
        "time_start": str(time_values[0]),
        "time_end": str(time_values[-1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (store_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (store_path / "validity.json").write_text(json.dumps(validity, indent=2, sort_keys=True) + "\n")
    return store_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare memmap-backed GraphCast stores for streaming training.")
    parser.add_argument("--data-path", default=DEFAULT_PREPARE_DATA_PATH)
    parser.add_argument("--ckpt-in", default=DEFAULT_CKPT)
    parser.add_argument("--out-root", default=DEFAULT_PREPARED_STREAM_ROOT)
    parser.add_argument("--resolutions", nargs="+", type=float, default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--time-start", default=None, help="Optional inclusive source time bound.")
    parser.add_argument("--time-end", default=None, help="Optional inclusive source time bound.")
    parser.add_argument("--input-duration", default=None, help="Optional task input duration override.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ckpt = load_graphcast_checkpoint(Path(args.ckpt_in))
    task_cfg = ckpt.task_config
    if args.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=args.input_duration)

    print(f"[prepare-stream] opening source {args.data_path}")
    source = open_graphcast_era5(args.data_path)
    source = _ensure_datetime_coord(source)
    if args.time_start is not None or args.time_end is not None:
        source = source.sel(time=slice(args.time_start, args.time_end))
        if source.sizes.get("time", 0) == 0:
            raise ValueError(
                f"Time window is empty for {args.data_path}: {args.time_start!r} to {args.time_end!r}"
            )
    available_vars = [name for name in GRAPHCAST_VARS if name in source.data_vars]
    if not available_vars:
        raise ValueError("None of the required GraphCast variables were found in source dataset.")
    source = source[available_vars]

    out_root = Path(args.out_root)
    for resolution in args.resolutions:
        path = _prepare_resolution_store(
            source,
            source_data_path=args.data_path,
            task_cfg=task_cfg,
            resolution=float(resolution),
            out_root=out_root,
            overwrite=args.overwrite,
        )
        print(f"[prepare-stream] wrote {path}")


if __name__ == "__main__":
    main()
