"""Load GraphCast-style ERA5 datasets (NetCDF or Zarr) in a unified layout.

The CDS NetCDF layout uses (batch=1, time, ...); the WB13 Zarr uses (time, ...)
with no batch and static (lat, lon) for mask/geopotential_at_surface. This module
opens either format and returns an xarray Dataset in the CDS-compatible layout so
downstream code (notebooks, training) can treat both the same.

See docs/data_formats.md for format comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import xarray as xr


def open_graphcast_era5(
    path: Union[str, Path],
    *,
    time_slice: Optional[Union[slice, int]] = None,
    engine: Optional[str] = None,
) -> xr.Dataset:
    """Open a GraphCast ERA5 dataset (NetCDF or Zarr) in CDS-compatible layout.

    - NetCDF (e.g. CDS rolling window): already has batch dim; only time_slice is applied.
    - Zarr (e.g. WB13 latest-1y): adds batch dim and expands static (lat, lon) vars
      to (batch=1, time, lat, lon), then applies time_slice if given.

    Parameters
    ----------
    path
        Path to a .nc or .zarr dataset.
    time_slice
        Optional slice or single index on the time dimension (after adding batch
        for Zarr). E.g. slice(-117, None) for last 117 steps, or 0 for first step.
    engine
        NetCDF engine when path is .nc (default "netcdf4"). Ignored for Zarr.

    Returns
    -------
    xr.Dataset
        Dataset with dims (batch, time, lat, lon) or (batch, time, level, lat, lon)
        for data vars, and coordinates batch, time, lat, lon, level.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if path.suffix == ".zarr" or path.is_dir() and (path / ".zmetadata").exists():
        ds = xr.open_zarr(str(path), consolidated=True)
        ds = _zarr_to_graphcast_layout(ds)
    else:
        engine = engine or "netcdf4"
        ds = xr.open_dataset(path, engine=engine)
        # CDS NetCDF already has batch; ensure we don't double-expand
        if "batch" not in ds.dims:
            ds = _zarr_to_graphcast_layout(ds)

    if time_slice is not None:
        ds = ds.isel(time=time_slice)
    return ds


def _zarr_to_graphcast_layout(ds: xr.Dataset) -> xr.Dataset:
    """Add batch dimension and expand static 2D vars to (batch=1, time, lat, lon)."""
    out_vars = {}
    time_coord = ds.coords["time"]
    ntime = time_coord.sizes["time"]

    for name in list(ds.data_vars):
        var = ds[name]
        dims = list(var.dims)

        if "batch" in dims:
            out_vars[name] = var
            continue

        if "time" in dims:
            # (time, lat, lon) or (time, level, lat, lon) -> (batch, time, ...)
            out_vars[name] = var.expand_dims(batch=[0]).transpose("batch", *dims)
        else:
            # Static (lat, lon) -> (batch=1, time, lat, lon) by broadcasting
            expanded = var.expand_dims(batch=[0], time=time_coord)
            out_vars[name] = expanded.transpose("batch", "time", "lat", "lon")

    # Ensure batch coordinate exists
    coords = dict(ds.coords)
    if "batch" not in coords:
        coords["batch"] = [0]

    return xr.Dataset(data_vars=out_vars, coords=coords, attrs=ds.attrs)
