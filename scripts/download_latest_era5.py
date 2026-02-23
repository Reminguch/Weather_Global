#!/usr/bin/env python3
"""Download the latest N-year ERA5 slice from WeatherBench2 for GraphCast workflows.

The script reads the public WeatherBench2 ERA5 wb13 archive, keeps GraphCast-relevant
variables, downsamples to a 1.0-degree grid (from 0.25-degree), and writes a local
dataset (Zarr by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


DEFAULT_URI = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

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
    # Optional in source data; included when available.
    "toa_incident_solar_radiation",
]


def _default_output(years: int) -> Path:
    return Path(
        f"data/graphcast/graphcast/dataset/"
        f"source-era5_wb13_latest-{years}y_res-1.0_levels-13_steps-all.zarr"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the latest N years of ERA5 (WeatherBench2 wb13) to local storage."
    )
    parser.add_argument("--years", type=int, default=3, help="Number of trailing years to keep.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Input WeatherBench2 Zarr URI.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output dataset path. Default is under data/graphcast/graphcast/dataset.",
    )
    parser.add_argument(
        "--format",
        choices=("zarr", "netcdf"),
        default="zarr",
        help="Output format. Zarr is recommended for multi-year slices.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=120,
        help="Time chunk size used during write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    return parser.parse_args()


def _prep_dataset(
    ds: xr.Dataset, years: int, time_chunk: int
) -> tuple[xr.Dataset, pd.Timestamp, pd.Timestamp, list[str]]:
    if "latitude" in ds.coords and "longitude" in ds.coords:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    if "time" not in ds.coords:
        raise KeyError("Expected `time` coordinate in source dataset.")
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("Expected `lat` and `lon` coordinates after rename.")

    available_vars = [name for name in GRAPHCAST_VARS if name in ds.data_vars]
    missing_vars = [name for name in GRAPHCAST_VARS if name not in ds.data_vars]
    if not available_vars:
        raise ValueError("None of the requested GraphCast variables were found in the source dataset.")

    # Use the latest available timestamp from the source and roll back by N years.
    end_time = pd.Timestamp(ds.time.values[-1])
    start_time = end_time - pd.DateOffset(years=years)

    subset = ds[available_vars].sel(time=slice(start_time, end_time))

    # 0.25° -> 1.0° via exact stride on aligned grid.
    # This yields 181 lat x 360 lon, matching GraphCast 1.0° layout.
    subset = subset.isel(lat=slice(0, None, 4), lon=slice(0, None, 4))

    # Ensure small, consistent on-disk type.
    for name in subset.data_vars:
        if subset[name].dtype.kind == "f" and subset[name].dtype != "float32":
            subset[name] = subset[name].astype("float32")

    subset = subset.chunk({"time": max(1, time_chunk)})
    return subset, start_time, end_time, missing_vars


def main() -> None:
    args = _parse_args()
    if args.years <= 0:
        raise ValueError("--years must be > 0")

    output = args.output or _default_output(args.years)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        if output.is_dir():
            import shutil

            shutil.rmtree(output)
        else:
            output.unlink()

    print(f"Opening source: {args.uri}")
    ds = xr.open_zarr(args.uri, consolidated=True, storage_options={"token": "anon"})

    prepared, start_time, end_time, missing_vars = _prep_dataset(
        ds, years=args.years, time_chunk=args.time_chunk
    )

    if args.format == "zarr":
        prepared.to_zarr(output, mode="w", consolidated=True)
    else:
        # NetCDF can be very large for multi-year slices; use compression.
        netcdf_encoding = {
            name: {"zlib": True, "complevel": 2}
            for name in prepared.data_vars
            if prepared[name].dtype.kind == "f"
        }
        prepared.to_netcdf(output, engine="netcdf4", encoding=netcdf_encoding)

    print("Done.")
    print(f"output: {output}")
    print(f"time_start: {start_time}")
    print(f"time_end:   {end_time}")
    print(f"dims: {dict(prepared.sizes)}")
    print(f"vars: {list(prepared.data_vars)}")
    if missing_vars:
        print(f"missing_optional_or_unavailable_vars: {missing_vars}")


if __name__ == "__main__":
    main()
