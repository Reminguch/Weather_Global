#!/usr/bin/env python3
"""Stage a 0.25-degree, 37-level WeatherBench2 ERA5 window for GraphCast."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr


DEFAULT_URI = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)
DEFAULT_OUTPUT = Path("data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr")
DEFAULT_START = "2021-12-31 18:00"
DEFAULT_END = "2022-02-10 18:00"

PRESSURE_LEVELS_37 = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

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
DERIVED_PRECIP_SOURCE = "total_precipitation"
EXPECTED_STEP = pd.Timedelta(hours=6)


@dataclass
class RunReport:
    uri: str
    output: str
    start_time: str
    end_time: str
    time_stride_hours: int
    chunk_time: int
    overwrite: bool
    dry_run: bool
    derive_precip_6hr: bool
    variables: list[str]
    pressure_levels: list[int]
    time_count: int
    lat_count: int
    lon_count: int
    started_at_unix: float
    ended_at_unix: float
    elapsed_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-time", default=DEFAULT_START)
    parser.add_argument("--end-time", default=DEFAULT_END)
    parser.add_argument("--time-stride-hours", type=int, default=6)
    parser.add_argument("--chunk-time", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--derive-precip-6hr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive total_precipitation_6hr from total_precipitation when needed.",
    )
    args = parser.parse_args()
    if args.time_stride_hours <= 0:
        raise ValueError("--time-stride-hours must be positive.")
    if args.chunk_time <= 0:
        raise ValueError("--chunk-time must be positive.")
    if pd.Timestamp(args.end_time) <= pd.Timestamp(args.start_time):
        raise ValueError("--end-time must be after --start-time.")
    return args


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    return ds.rename(rename) if rename else ds


def _canonical_report_path(output: Path) -> Path:
    return output.parent / f"{output.name}.stage_report.json"


def _select_6h_window(ds: xr.Dataset, start: str, end: str, stride_hours: int) -> xr.Dataset:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    ds = ds.sel(time=slice(start_ts, end_ts))
    if ds.sizes.get("time", 0) == 0:
        raise ValueError(f"Empty time window: {start} to {end}")
    wanted = pd.date_range(start_ts, end_ts, freq=f"{stride_hours}h")
    return ds.sel(time=wanted)


def _validate_layout(ds: xr.Dataset, pressure_levels: list[int]) -> None:
    if ds.sizes.get("lat") != 721 or ds.sizes.get("lon") != 1440:
        raise ValueError(
            f"Expected 0.25-degree grid lat=721/lon=1440, got "
            f"lat={ds.sizes.get('lat')} lon={ds.sizes.get('lon')}"
        )
    if "level" in ds.coords:
        levels = [int(level) for level in ds["level"].values.tolist()]
        if levels != pressure_levels:
            raise ValueError(f"Unexpected pressure levels: {levels}")
    times = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    if len(times) > 1:
        deltas = pd.Index(times[1:] - times[:-1]).unique()
        if len(deltas) != 1 or pd.Timedelta(deltas[0]) != EXPECTED_STEP:
            raise ValueError(f"Expected strict 6-hour cadence, found {list(deltas)}")


def _pick_vars(source: xr.Dataset, *, derive_precip_6hr: bool) -> list[str]:
    available = set(source.data_vars)
    requested = [name for name in GRAPHCAST_VARS if name in available]
    if "total_precipitation_6hr" not in available and derive_precip_6hr and DERIVED_PRECIP_SOURCE in available:
        requested.append(DERIVED_PRECIP_SOURCE)
    missing = [
        name
        for name in GRAPHCAST_VARS
        if name not in requested
        and name not in OPTIONAL_VARS
        and not (name == "total_precipitation_6hr" and DERIVED_PRECIP_SOURCE in requested)
    ]
    if missing:
        raise ValueError(f"Missing required GraphCast variables in source dataset: {missing}")
    return requested


def _derive_total_precipitation_6hr(ds: xr.Dataset) -> xr.Dataset:
    if "total_precipitation_6hr" in ds:
        return ds
    if DERIVED_PRECIP_SOURCE not in ds:
        return ds
    precip = ds[DERIVED_PRECIP_SOURCE].rolling(time=6, min_periods=6).sum()
    ds = ds.assign(total_precipitation_6hr=precip)
    return ds.drop_vars(DERIVED_PRECIP_SOURCE)


def _prepare_window(args: argparse.Namespace) -> xr.Dataset:
    source = xr.open_zarr(args.uri, consolidated=True, storage_options={"token": "anon"})
    source = _normalize_coords(source)
    vars_to_keep = _pick_vars(source, derive_precip_6hr=args.derive_precip_6hr)
    source = source[vars_to_keep]

    precip_pad_start = pd.Timestamp(args.start_time) - pd.Timedelta(hours=5)
    raw_start = precip_pad_start if DERIVED_PRECIP_SOURCE in source else pd.Timestamp(args.start_time)
    source = source.sel(time=slice(raw_start, pd.Timestamp(args.end_time)))
    source = _derive_total_precipitation_6hr(source)
    window = _select_6h_window(source, args.start_time, args.end_time, args.time_stride_hours)
    if "level" in window.coords:
        window = window.sel(level=PRESSURE_LEVELS_37)

    keep = [name for name in GRAPHCAST_VARS if name in window.data_vars]
    window = window[keep]
    for name in list(window.data_vars):
        if window[name].dtype.kind == "f" and window[name].dtype != np.float32:
            window[name] = window[name].astype(np.float32)
    window = window.chunk({"time": args.chunk_time, "level": -1, "lat": 90, "lon": 180})
    for variable in window.variables:
        window[variable].encoding.pop("chunks", None)
        window[variable].encoding.pop("preferred_chunks", None)
    _validate_layout(window, PRESSURE_LEVELS_37)
    return window


def _save_report(path: Path, report: RunReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output.resolve()
    report_path = _canonical_report_path(output)

    print(f"Opening source WB2 Zarr: {args.uri}")
    window = _prepare_window(args)
    report = RunReport(
        uri=args.uri,
        output=str(output),
        start_time=args.start_time,
        end_time=args.end_time,
        time_stride_hours=args.time_stride_hours,
        chunk_time=args.chunk_time,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        derive_precip_6hr=args.derive_precip_6hr,
        variables=sorted(window.data_vars),
        pressure_levels=PRESSURE_LEVELS_37,
        time_count=int(window.sizes["time"]),
        lat_count=int(window.sizes["lat"]),
        lon_count=int(window.sizes["lon"]),
        started_at_unix=started,
        ended_at_unix=time.time(),
        elapsed_sec=time.time() - started,
    )
    if args.dry_run:
        _save_report(report_path, report)
        print("Dry run only: no writes performed.")
        print(f"report: {report_path}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    window.to_zarr(output, mode="w", consolidated=True, zarr_version=2)
    zarr.consolidate_metadata(str(output))
    report.ended_at_unix = time.time()
    report.elapsed_sec = report.ended_at_unix - started
    _save_report(report_path, report)
    print("Done.")
    print(f"output: {output}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
