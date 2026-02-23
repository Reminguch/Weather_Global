#!/usr/bin/env python3
"""Regrid a lat/lon dataset to a target degree resolution (e.g. 4.0, 6.0)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def _open_dataset(path: Path) -> xr.Dataset:
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _target_axis(values: xr.DataArray, resolution: float, *, is_lon: bool) -> np.ndarray:
    arr = np.asarray(values.values, dtype=float)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    if is_lon and vmin >= 0.0 and vmax > 180.0:
        # Common 0..360 longitude convention.
        target = np.arange(0.0, 360.0, resolution, dtype=float)
    else:
        start = np.ceil(vmin / resolution) * resolution
        stop = np.floor(vmax / resolution) * resolution
        target = np.arange(start, stop + 0.5 * resolution, resolution, dtype=float)

    if target.size == 0:
        raise ValueError("Could not build target grid. Check input coordinates and resolution.")
    return np.round(target, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regrid dataset to a coarser lat/lon resolution.")
    parser.add_argument("--input", type=Path, required=True, help="Input .nc or .zarr dataset path.")
    parser.add_argument("--output", type=Path, required=True, help="Output .nc path.")
    parser.add_argument("--resolution", type=float, required=True, help="Target degree resolution.")
    parser.add_argument("--lat-name", default="lat", help="Latitude coordinate name.")
    parser.add_argument("--lon-name", default="lon", help="Longitude coordinate name.")
    parser.add_argument(
        "--method",
        default="linear",
        choices=["linear", "nearest"],
        help="Interpolation method.",
    )
    args = parser.parse_args()

    if args.resolution <= 0:
        raise ValueError("--resolution must be > 0.")

    ds = _open_dataset(args.input)
    if args.lat_name not in ds.coords or args.lon_name not in ds.coords:
        raise KeyError(f"Missing required coords: {args.lat_name!r}, {args.lon_name!r}")

    lat_desc = bool(ds[args.lat_name].values[0] > ds[args.lat_name].values[-1])
    ds = ds.sortby(args.lat_name).sortby(args.lon_name)

    target_lat = _target_axis(ds[args.lat_name], args.resolution, is_lon=False)
    target_lon = _target_axis(ds[args.lon_name], args.resolution, is_lon=True)

    out = ds.interp({args.lat_name: target_lat, args.lon_name: target_lon}, method=args.method)
    if lat_desc:
        out = out.sortby(args.lat_name, ascending=False)

    out.attrs["regridded_resolution_deg"] = args.resolution
    out.attrs["regrid_method"] = args.method
    out.to_netcdf(args.output)

    print(f"saved: {args.output}")
    print(f"{args.lat_name}: {out.sizes[args.lat_name]}, {args.lon_name}: {out.sizes[args.lon_name]}")


if __name__ == "__main__":
    main()
