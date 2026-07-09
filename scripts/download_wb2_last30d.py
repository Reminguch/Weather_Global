#!/usr/bin/env python3
"""Download trailing WB2 ERA5 window to local 1.0-degree / 13-level Zarr."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_URI = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
DEFAULT_OUTPUT = Path("data/graphcast/graphcast/dataset/wb2_res1_levels13_last30d.zarr")

GRAPHCAST_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "geopotential_at_surface",
    "land_sea_mask",
    "toa_incident_solar_radiation",
]
OPTIONAL_VARS = {"toa_incident_solar_radiation"}
PRESSURE_LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download trailing WB2 ERA5 window to local Zarr.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-tisr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include toa_incident_solar_radiation if present (default: true).",
    )
    args = parser.parse_args()
    if args.days <= 0:
        raise ValueError("--days must be > 0")
    return args


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    return ds


def _ensure_datetime_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        raise KeyError("Source dataset must contain `time` coordinate.")
    if np.issubdtype(ds.time.dtype, np.datetime64):
        return ds
    decoded = xr.decode_cf(ds)
    if not np.issubdtype(decoded.time.dtype, np.datetime64):
        raise TypeError(f"Unable to decode `time` as datetime64; dtype={decoded.time.dtype}")
    return decoded


def _pick_vars(ds: xr.Dataset, include_tisr: bool) -> list[str]:
    requested = [v for v in GRAPHCAST_VARS if include_tisr or v not in OPTIONAL_VARS]
    available = set(ds.data_vars)
    missing_required = [v for v in requested if v not in available and v not in OPTIONAL_VARS]
    if missing_required:
        raise ValueError(f"Missing required variables in source dataset: {missing_required}")
    picked = [v for v in requested if v in available]
    if not picked:
        raise ValueError("No GraphCast variables found in source dataset.")
    return picked


def main() -> None:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()

    print(f"Opening source: {args.uri}")
    ds = xr.open_zarr(args.uri, consolidated=True, storage_options={"token": "anon"})
    ds = _normalize_coords(ds)
    ds = _ensure_datetime_time(ds)

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("Expected `lat` and `lon` coordinates in source dataset.")

    keep_vars = _pick_vars(ds, include_tisr=args.include_tisr)
    ds = ds[keep_vars]

    if "level" in ds.coords:
        ds = ds.sel(level=PRESSURE_LEVELS_13)

    # WB2 native grid is 0.25deg. For 1.0deg keep every 4th point.
    ds = ds.isel(lat=slice(0, None, 4), lon=slice(0, None, 4))

    for name in list(ds.data_vars):
        if ds[name].dtype.kind == "f" and ds[name].dtype != np.float32:
            ds[name] = ds[name].astype(np.float32)

    steps = args.days * 4  # 6-hour cadence
    if ds.sizes.get("time", 0) < steps:
        raise ValueError(
            f"Not enough source timesteps ({ds.sizes.get('time', 0)}) for --days={args.days} ({steps} steps)."
        )
    out = ds.isel(time=slice(-steps, None))

    out.to_zarr(output, mode="w", consolidated=True)

    t0 = pd.Timestamp(out.time.values[0])
    t1 = pd.Timestamp(out.time.values[-1])
    print("Done.")
    print(f"output: {output}")
    print(f"time_start: {t0}")
    print(f"time_end:   {t1}")
    print(f"dims: {dict(out.sizes)}")
    print(f"vars: {list(out.data_vars)}")


if __name__ == "__main__":
    main()
