#!/usr/bin/env python3
"""Fetch a rolling ERA5 window from CDS and materialize a GraphCast-style dataset.

Requires CDS credentials (typically in ~/.cdsapirc).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import cdsapi
import pandas as pd
import xarray as xr


PRESSURE_LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SIX_HOURLY_TIMES = [f"{h:02d}:00" for h in (0, 6, 12, 18)]
HOURLY_TIMES = [f"{h:02d}:00" for h in range(24)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch a rolling ERA5 window from CDS.")
    parser.add_argument("--days", type=int, default=30, help="Window length in days.")
    parser.add_argument(
        "--lag-days",
        type=int,
        default=5,
        help="Availability lag in days from today (ERA5T is typically delayed).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc"),
        help="Output NetCDF path.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("data/graphcast/graphcast/dataset/.tmp_cds"),
        help="Temporary directory for intermediate CDS files.",
    )
    parser.add_argument("--grid-deg", type=float, default=1.0, help="Output CDS grid resolution in degrees.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument(
        "--reuse-tmp",
        action="store_true",
        help="Reuse previously downloaded intermediate files in --tmp-dir when present.",
    )
    return parser.parse_args()


def _build_date_lists(start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict[str, list[str]]:
    days = pd.date_range(start_date, end_date, freq="D")
    return {
        "year": sorted({d.strftime("%Y") for d in days}),
        "month": sorted({d.strftime("%m") for d in days}),
        "day": sorted({d.strftime("%d") for d in days}),
    }


def _iter_month_ranges(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = pd.Timestamp(start_date.year, start_date.month, 1)
    end_anchor = pd.Timestamp(end_date.year, end_date.month, 1)
    while cur <= end_anchor:
        month_end = (cur + pd.offsets.MonthEnd(0)).normalize()
        a = max(start_date.normalize(), cur.normalize())
        b = min(end_date.normalize(), month_end)
        if a <= b:
            ranges.append((a, b))
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return ranges


def _rename_dims(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "pressure_level" in ds.dims or "pressure_level" in ds.coords:
        rename["pressure_level"] = "level"
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)

    # CDS files may include extra bookkeeping coordinates.
    if "number" in ds.dims:
        ds = ds.isel(number=0, drop=True)
    if "expver" in ds.dims:
        ds = ds.isel(expver=0, drop=True)

    drop_coords = [name for name in ("number", "expver") if name in ds.coords and name not in ds.dims]
    if drop_coords:
        ds = ds.drop_vars(drop_coords)
    return ds


def _pick_var(ds: xr.Dataset, candidates: Iterable[str], target_name: str) -> xr.DataArray:
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(f"Could not find {target_name!r}. Tried: {list(candidates)}")


def _retrieve(client: cdsapi.Client, dataset: str, request: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    # Legacy CDS API key used by most accounts.
    req = dict(request)
    req.setdefault("format", "netcdf")
    client.retrieve(dataset, req, str(target))


def _to_float32(ds: xr.Dataset) -> xr.Dataset:
    for var in list(ds.data_vars):
        if ds[var].dtype.kind == "f":
            ds[var] = ds[var].astype("float32")
    return ds


def main() -> None:
    args = _parse_args()
    if args.days <= 0:
        raise ValueError("--days must be > 0")
    if args.lag_days < 0:
        raise ValueError("--lag-days must be >= 0")

    output = args.output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    utc_today = pd.Timestamp.utcnow().tz_localize(None).floor("D")
    end_date = utc_today - pd.Timedelta(days=args.lag_days)
    start_date = end_date - pd.Timedelta(days=args.days - 1)

    # Need preceding hours for first 6h accumulation.
    tp_start_date = start_date - pd.Timedelta(days=1)

    print(f"window_start={start_date.date()} window_end={end_date.date()} lag_days={args.lag_days}")

    sl_file = args.tmp_dir / "single_levels_6h.nc"
    pl_file = args.tmp_dir / "pressure_levels_6h.nc"
    tp_file = args.tmp_dir / "tp_hourly.nc"

    sl_dates = _build_date_lists(start_date, end_date)
    tp_dates = _build_date_lists(tp_start_date, end_date)
    month_ranges = _iter_month_ranges(start_date, end_date)
    planned_pl_parts = [args.tmp_dir / f"pressure_levels_6h_part{idx:02d}.nc" for idx, _ in enumerate(month_ranges, start=1)]

    need_sl = not (args.reuse_tmp and sl_file.exists())
    need_tp = not (args.reuse_tmp and tp_file.exists())
    need_pl = any(not (args.reuse_tmp and part.exists()) for part in planned_pl_parts)
    need_download = need_sl or need_tp or need_pl

    client: cdsapi.Client | None = None
    if need_download:
        # Allow credentials via env vars as an alternative to ~/.cdsapirc.
        cds_url = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
        cds_key = os.getenv("CDSAPI_KEY")
        try:
            client = cdsapi.Client(url=cds_url, key=cds_key) if cds_key else cdsapi.Client()
        except Exception as exc:
            raise RuntimeError(
                "CDS credentials not configured. Set ~/.cdsapirc or env vars "
                "`CDSAPI_URL` and `CDSAPI_KEY`."
            ) from exc

    sl_request = {
        "product_type": "reanalysis",
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "geopotential",
            "land_sea_mask",
        ],
        "year": sl_dates["year"],
        "month": sl_dates["month"],
        "day": sl_dates["day"],
        "time": SIX_HOURLY_TIMES,
        "grid": [args.grid_deg, args.grid_deg],
    }
    if args.reuse_tmp and sl_file.exists():
        print(f"Reusing existing file: {sl_file}")
    else:
        print("Downloading single-level variables (6-hourly)...")
        assert client is not None
        _retrieve(client, "reanalysis-era5-single-levels", sl_request, sl_file)

    pl_request = {
        "product_type": "reanalysis",
        "variable": [
            "temperature",
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "specific_humidity",
        ],
        "pressure_level": [str(level) for level in PRESSURE_LEVELS_13],
        "time": SIX_HOURLY_TIMES,
        "grid": [args.grid_deg, args.grid_deg],
    }
    print("Downloading pressure-level variables (6-hourly, month-chunked)...")
    pl_parts: list[Path] = []
    for idx, (m_start, m_end) in enumerate(month_ranges, start=1):
        month_dates = _build_date_lists(m_start, m_end)
        part = args.tmp_dir / f"pressure_levels_6h_part{idx:02d}.nc"
        req = dict(pl_request)
        req["year"] = month_dates["year"]
        req["month"] = month_dates["month"]
        req["day"] = month_dates["day"]
        if args.reuse_tmp and part.exists():
            print(f"  reusing pressure chunk {idx}: {part}")
        else:
            print(f"  pressure chunk {idx}: {m_start.date()} -> {m_end.date()}")
            assert client is not None
            _retrieve(client, "reanalysis-era5-pressure-levels", req, part)
        pl_parts.append(part)

    tp_request = {
        "product_type": "reanalysis",
        "variable": ["total_precipitation"],
        "year": tp_dates["year"],
        "month": tp_dates["month"],
        "day": tp_dates["day"],
        "time": HOURLY_TIMES,
        "grid": [args.grid_deg, args.grid_deg],
    }
    if args.reuse_tmp and tp_file.exists():
        print(f"Reusing existing file: {tp_file}")
    else:
        print("Downloading total precipitation (hourly for 6h accumulation)...")
        assert client is not None
        _retrieve(client, "reanalysis-era5-single-levels", tp_request, tp_file)

    print("Building merged evaluation dataset...")
    ds_sl = _rename_dims(xr.open_dataset(sl_file))
    if pl_parts:
        ds_pl = xr.concat([_rename_dims(xr.open_dataset(part)) for part in pl_parts], dim="time")
    else:
        ds_pl = _rename_dims(xr.open_dataset(pl_file))
    ds_tp = _rename_dims(xr.open_dataset(tp_file))

    # Normalize coordinate ordering.
    ds_sl = ds_sl.sortby("time")
    ds_pl = ds_pl.sortby("time")
    ds_tp = ds_tp.sortby("time")

    # CDS accepts separate month/day lists and may return broader combinations.
    # Enforce exact rolling window bounds post-download.
    sl_start = pd.Timestamp(start_date)
    sl_end = pd.Timestamp(end_date) + pd.Timedelta(hours=18)
    tp_start = pd.Timestamp(tp_start_date)
    tp_end = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    ds_sl = ds_sl.sel(time=slice(sl_start, sl_end))
    ds_pl = ds_pl.sel(time=slice(sl_start, sl_end))
    ds_tp = ds_tp.sel(time=slice(tp_start, tp_end))
    if "level" in ds_pl.coords:
        ds_pl["level"] = ds_pl["level"].astype("int32")
        ds_pl = ds_pl.sel(level=PRESSURE_LEVELS_13)

    target_times = ds_sl.time

    tp_hourly = _pick_var(ds_tp, ("total_precipitation", "tp"), "total_precipitation")
    tp_6h = tp_hourly.rolling(time=6, min_periods=6).sum().sel(time=target_times)
    tp_6h = tp_6h.rename("total_precipitation_6hr")

    out = xr.Dataset(
        data_vars={
            "2m_temperature": _pick_var(ds_sl, ("2m_temperature", "t2m"), "2m_temperature"),
            "10m_u_component_of_wind": _pick_var(ds_sl, ("10m_u_component_of_wind", "u10"), "10m_u_component_of_wind"),
            "10m_v_component_of_wind": _pick_var(ds_sl, ("10m_v_component_of_wind", "v10"), "10m_v_component_of_wind"),
            "mean_sea_level_pressure": _pick_var(ds_sl, ("mean_sea_level_pressure", "msl"), "mean_sea_level_pressure"),
            "total_precipitation_6hr": tp_6h,
            "temperature": _pick_var(ds_pl, ("temperature", "t"), "temperature"),
            "geopotential": _pick_var(ds_pl, ("geopotential", "z"), "geopotential"),
            "u_component_of_wind": _pick_var(ds_pl, ("u_component_of_wind", "u"), "u_component_of_wind"),
            "v_component_of_wind": _pick_var(ds_pl, ("v_component_of_wind", "v"), "v_component_of_wind"),
            "vertical_velocity": _pick_var(ds_pl, ("vertical_velocity", "w"), "vertical_velocity"),
            "specific_humidity": _pick_var(ds_pl, ("specific_humidity", "q"), "specific_humidity"),
            "geopotential_at_surface": _pick_var(ds_sl, ("geopotential", "z"), "geopotential_at_surface"),
            "land_sea_mask": _pick_var(ds_sl, ("land_sea_mask", "lsm"), "land_sea_mask"),
        }
    )

    # Match GraphCast-style batch dimension for time-dependent variables.
    for name in list(out.data_vars):
        if "time" in out[name].dims:
            dims = list(out[name].dims)
            out[name] = out[name].expand_dims(batch=[0]).transpose("batch", *dims)

    out = _to_float32(out)
    out.attrs["source"] = "CDS ERA5 / ERA5T rolling window"
    out.attrs["window_start"] = str(start_date.date())
    out.attrs["window_end"] = str(end_date.date())
    out.attrs["lag_days"] = int(args.lag_days)
    out.attrs["grid_deg"] = float(args.grid_deg)

    encoding = {name: {"zlib": True, "complevel": 2} for name in out.data_vars if out[name].dtype.kind == "f"}
    out.to_netcdf(output, engine="netcdf4", encoding=encoding)

    print(f"saved={output}")
    print(f"dims={dict(out.sizes)}")
    print(f"time_start={pd.Timestamp(out.time.values[0])} time_end={pd.Timestamp(out.time.values[-1])}")


if __name__ == "__main__":
    main()
