from __future__ import annotations

import argparse
import dataclasses
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from src.data_operations.loaders.graphcast_dataset import open_graphcast_era5
from src.models.graphcast.training.core.config import (
    DEFAULT_CKPT,
    DEFAULT_PREPARED_DATA_ROOT,
    GRAPHCAST_VARS,
)
from src.models.graphcast.training.core.dataset import (
    PREPARED_FORMAT_VERSION,
    _ensure_datetime_coord,
    prepare_dataset_for_task,
    resolution_tag,
    select_resolution,
)
from src.models.graphcast.training.core.model import load_graphcast_checkpoint

DEFAULT_PREPARE_DATA_PATH = "data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr"
DEFAULT_RESOLUTIONS = (1.0, 2.0, 4.0, 9.0, 15.0)


def _jsonable_tuple(values: Iterable) -> list:
    return [item.item() if hasattr(item, "item") else item for item in values]


def _cast_float_vars_to_float32(ds: xr.Dataset) -> xr.Dataset:
    for name in list(ds.data_vars):
        if ds[name].dtype.kind == "f" and ds[name].dtype != np.float32:
            ds[name] = ds[name].astype(np.float32)
    return ds


def _metadata_attrs(
    prepared: xr.Dataset,
    *,
    source_data_path: str,
    resolution: float,
    base_resolution: float,
    task_cfg,
) -> dict:
    time_values = prepared.time.values
    return {
        "prepared_format_version": PREPARED_FORMAT_VERSION,
        "source_data_path": source_data_path,
        "resolution": float(resolution),
        "base_resolution": float(base_resolution),
        "pressure_levels": _jsonable_tuple(task_cfg.pressure_levels),
        "task_input_variables": list(task_cfg.input_variables),
        "task_target_variables": list(task_cfg.target_variables),
        "task_forcing_variables": list(task_cfg.forcing_variables),
        "time_start": str(np.asarray(time_values[0]).astype("datetime64[ns]")),
        "time_end": str(np.asarray(time_values[-1]).astype("datetime64[ns]")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def prepare_resolution_store(
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
            raise FileExistsError(f"Prepared store already exists: {store_path}. Use --overwrite to replace it.")
        shutil.rmtree(store_path)

    ds_res, base_resolution, stride = select_resolution(source.copy(), resolution)
    del stride
    prepared = prepare_dataset_for_task(ds_res, task_cfg)
    prepared = _cast_float_vars_to_float32(prepared)
    prepared.attrs.update(
        _metadata_attrs(
            prepared,
            source_data_path=source_data_path,
            resolution=resolution,
            base_resolution=base_resolution,
            task_cfg=task_cfg,
        )
    )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] writing {store_path}")
    prepared.to_zarr(store_path, mode="w", consolidated=True, zarr_version=2)
    return store_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GraphCast-compatible Zarr stores for training.")
    parser.add_argument("--data-path", default=DEFAULT_PREPARE_DATA_PATH)
    parser.add_argument("--ckpt-in", default=DEFAULT_CKPT)
    parser.add_argument("--out-root", default=DEFAULT_PREPARED_DATA_ROOT)
    parser.add_argument("--resolutions", nargs="+", type=float, default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--input-duration", default=None, help="Optional task input duration override.")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ckpt = load_graphcast_checkpoint(Path(args.ckpt_in))
    task_cfg = ckpt.task_config
    if args.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=args.input_duration)

    print(f"[prepare] opening source {args.data_path}")
    source = open_graphcast_era5(args.data_path)
    source = _ensure_datetime_coord(source)
    available_vars = [name for name in GRAPHCAST_VARS if name in source.data_vars]
    if not available_vars:
        raise ValueError("None of the required GraphCast variables were found in source dataset.")
    source = source[available_vars]

    out_root = Path(args.out_root)
    for resolution in args.resolutions:
        path = prepare_resolution_store(
            source,
            source_data_path=args.data_path,
            task_cfg=task_cfg,
            resolution=float(resolution),
            out_root=out_root,
            overwrite=args.overwrite,
        )
        print(f"[prepare] wrote {path}")


if __name__ == "__main__":
    main()
