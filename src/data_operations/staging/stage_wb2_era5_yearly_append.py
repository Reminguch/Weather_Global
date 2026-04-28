#!/usr/bin/env python3
"""Stage WeatherBench2 ERA5 to a local 1.0-degree / 13-level Zarr via yearly append.

Behavior:
- Reads remote WB2 Zarr lazily (default public GCS URI)
- Keeps GraphCast variables, downsampled 0.25 -> 1.0 degree
- Enforces 13 pressure levels
- Appends missing years only
- Default with existing output: append one next year (max_year + 1)
- Default with no output: bootstrap 1979..2022
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr

DEFAULT_URI = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)
DEFAULT_OUTPUT = Path("data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr")
DEFAULT_BOOTSTRAP_START = 1979
DEFAULT_BOOTSTRAP_END = 2022

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
    "toa_incident_solar_radiation",  # optional
]
OPTIONAL_VARS = {"toa_incident_solar_radiation"}
PRESSURE_LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
EXPECTED_STEP = pd.Timedelta(hours=6)


@dataclass
class RunReport:
    uri: str
    output: str
    start_year: int | None
    end_year: int | None
    chunk_time: int
    include_tisr: bool
    overwrite: bool
    dry_run: bool
    source_year_min: int
    source_year_max: int
    existing_years: list[int]
    requested_years: list[int]
    skipped_years: list[int]
    appended_years: list[int]
    rebuild_required: bool
    write_mode: str
    started_at_unix: float
    ended_at_unix: float
    elapsed_sec: float


class LockFile:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "LockFile":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Lock file already exists: {self.path}. Another staging process may be running."
            ) from exc
        payload = f"pid={os.getpid()} time={time.time()}\n".encode("ascii")
        os.write(self.fd, payload)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage WB2 ERA5 locally as yearly-appended Zarr.")
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--chunk-time", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-tisr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include toa_incident_solar_radiation when present (default: true).",
    )
    args = parser.parse_args()

    if args.chunk_time <= 0:
        raise ValueError("--chunk-time must be > 0")

    if (args.start_year is None) ^ (args.end_year is None):
        raise ValueError("Provide both --start-year and --end-year, or neither.")

    if args.start_year is not None and args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

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
        raise KeyError("Dataset must include a `time` coordinate.")

    if np.issubdtype(ds.time.dtype, np.datetime64):
        return ds

    decoded = xr.decode_cf(ds)
    if not np.issubdtype(decoded.time.dtype, np.datetime64):
        raise TypeError(f"Unable to decode `time` as datetime64; dtype={decoded.time.dtype}")
    return decoded


def _expected_year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01 00:00:00",
        f"{year}-12-31 18:00:00",
        freq="6h",
    )


def _validate_monotonic_6h(times: pd.DatetimeIndex) -> None:
    if len(times) <= 1:
        return
    if not np.all(times[1:] > times[:-1]):
        raise ValueError("Local output has non-monotonic or duplicate time values; aborting.")
    diffs = pd.Index(times[1:] - times[:-1]).unique()
    if len(diffs) != 1 or pd.Timedelta(diffs[0]) != EXPECTED_STEP:
        raise ValueError(
            f"Local output time cadence is not strictly 6-hourly; unique diffs found: {list(diffs)}"
        )


def _infer_existing_years(ds: xr.Dataset) -> tuple[list[int], dict[int, str]]:
    times = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
    if len(times) == 0:
        return [], {}

    _validate_monotonic_6h(times)

    years = sorted(set(times.year.astype(int).tolist()))
    partial: dict[int, str] = {}
    for year in years:
        year_times = times[times.year == year]
        expected = _expected_year_index(year)
        if len(year_times) != len(expected):
            partial[year] = f"expected {len(expected)} steps, found {len(year_times)}"
            continue
        if year_times[0] != expected[0] or year_times[-1] != expected[-1]:
            partial[year] = (
                f"expected bounds [{expected[0]}, {expected[-1]}], "
                f"found [{year_times[0]}, {year_times[-1]}]"
            )
            continue
        if not np.all(year_times.values == expected.values):
            partial[year] = "timestamps do not match exact 6-hourly year grid"

    return years, partial


def _validate_local_layout(ds: xr.Dataset) -> None:
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Local output must contain `lat`/`lon` coordinates.")
    if ds.sizes.get("lat") != 181 or ds.sizes.get("lon") != 360:
        raise ValueError(
            f"Local output grid is not 1.0-degree (expected lat=181/lon=360, got "
            f"lat={ds.sizes.get('lat')} lon={ds.sizes.get('lon')})."
        )

    if "level" in ds.coords:
        levels = [int(x) for x in ds["level"].values.tolist()]
        if levels != PRESSURE_LEVELS_13:
            raise ValueError(
                f"Local output levels do not match required 13-level set: {levels}"
            )


def _pick_vars(source_vars: Iterable[str], include_tisr: bool) -> list[str]:
    available = set(source_vars)
    requested = [v for v in GRAPHCAST_VARS if include_tisr or v not in OPTIONAL_VARS]

    missing_required = [v for v in requested if v not in available and v not in OPTIONAL_VARS]
    if missing_required:
        raise ValueError(f"Missing required variables in source dataset: {missing_required}")

    picked = [v for v in requested if v in available]
    if not picked:
        raise ValueError("No GraphCast variables found in source dataset.")
    return picked


def _prepare_year_chunk(ds_source: xr.Dataset, year: int, vars_to_keep: list[str], chunk_time: int) -> xr.Dataset:
    start = pd.Timestamp(year=year, month=1, day=1, hour=0)
    end = pd.Timestamp(year=year, month=12, day=31, hour=18)

    year_ds = ds_source[vars_to_keep].sel(time=slice(start, end))
    if year_ds.sizes.get("time", 0) == 0:
        raise ValueError(f"No data available for year {year} in source dataset.")

    if "level" in year_ds.coords:
        year_ds = year_ds.sel(level=PRESSURE_LEVELS_13)

    if "lat" not in year_ds.coords or "lon" not in year_ds.coords:
        raise KeyError("Expected `lat` and `lon` coordinates in source dataset.")

    year_ds = year_ds.isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
    year_ds = year_ds.sortby("time")

    actual_idx = pd.DatetimeIndex(pd.to_datetime(year_ds.time.values))
    expected_idx = _expected_year_index(year)
    if len(actual_idx) != len(expected_idx) or not np.all(actual_idx.values == expected_idx.values):
        raise ValueError(
            f"Source year {year} is not a complete 6-hourly year after slicing. "
            f"Expected {len(expected_idx)} points, got {len(actual_idx)}."
        )

    for name in list(year_ds.data_vars):
        if year_ds[name].dtype.kind == "f" and year_ds[name].dtype != np.float32:
            year_ds[name] = year_ds[name].astype(np.float32)

    year_ds = year_ds.chunk({"time": chunk_time})
    return year_ds


def _write_year(ds_year: xr.Dataset, output: Path, *, mode: str) -> None:
    if mode == "w":
        # Force Zarr v2 for compatibility with upstream WB2 encodings that
        # include numcodecs compressors (e.g., Blosc), which fail under v3.
        ds_year.to_zarr(output, mode="w", consolidated=True, zarr_format=2)
    else:
        ds_year.to_zarr(output, mode="a", append_dim="time", zarr_format=2)
        zarr.consolidate_metadata(str(output))


def _resolve_requested_years(
    *,
    start_year: int | None,
    end_year: int | None,
    output_exists: bool,
    existing_years: list[int],
) -> list[int]:
    if start_year is not None and end_year is not None:
        return list(range(start_year, end_year + 1))

    if output_exists and existing_years:
        return [max(existing_years) + 1]

    return list(range(DEFAULT_BOOTSTRAP_START, DEFAULT_BOOTSTRAP_END + 1))


def _save_report(path: Path, report: RunReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)


def main() -> None:
    args = parse_args()
    started = time.time()

    output = args.output.resolve()
    lock_path = output.parent / f"{output.name}.lock"
    report_path = output.parent / f"{output.name}.stage_report.json"

    print(f"Opening source WB2 Zarr: {args.uri}")
    source = xr.open_zarr(args.uri, consolidated=True, storage_options={"token": "anon"})
    source = _normalize_coords(source)
    source = _ensure_datetime_time(source)

    source_min_year = int(pd.Timestamp(source.time.values[0]).year)
    source_max_year = int(pd.Timestamp(source.time.values[-1]).year)

    vars_to_keep = _pick_vars(source.data_vars.keys(), include_tisr=args.include_tisr)

    output_exists_physical = output.exists()
    output_exists_effective = output_exists_physical and not args.overwrite

    existing_years: list[int] = []
    existing_ds: xr.Dataset | None = None

    if output_exists_effective:
        existing_ds = xr.open_zarr(output, consolidated=True)
        existing_ds = _normalize_coords(existing_ds)
        existing_ds = _ensure_datetime_time(existing_ds)
        _validate_local_layout(existing_ds)
        existing_years, partial_years = _infer_existing_years(existing_ds)
        if partial_years:
            detail = "; ".join(f"{y}: {msg}" for y, msg in sorted(partial_years.items()))
            raise ValueError(
                "Detected partial/incomplete years in existing output. "
                "Repair by recreating with --overwrite. Details: "
                f"{detail}"
            )

        existing_var_set = set(existing_ds.data_vars)
        incoming_var_set = set(vars_to_keep)
        if existing_var_set != incoming_var_set:
            raise ValueError(
                "Existing local output variable set does not match this run settings. "
                f"existing={sorted(existing_var_set)} incoming={sorted(incoming_var_set)}. "
                "Use matching --include-tisr setting or rebuild with --overwrite."
            )

    requested_years = _resolve_requested_years(
        start_year=args.start_year,
        end_year=args.end_year,
        output_exists=output_exists_effective,
        existing_years=existing_years,
    )

    requested_years = [y for y in requested_years if source_min_year <= y <= source_max_year]
    if args.start_year is not None and args.end_year is not None:
        outside = [y for y in range(args.start_year, args.end_year + 1) if y not in requested_years]
        if outside:
            raise ValueError(
                f"Requested years outside source coverage [{source_min_year}, {source_max_year}]: {outside}"
            )

    missing_years = sorted(y for y in requested_years if y not in set(existing_years))
    skipped_years = sorted(y for y in requested_years if y in set(existing_years))

    rebuild_required = (
        output_exists_effective
        and bool(missing_years)
        and bool(existing_years)
        and min(missing_years) < max(existing_years)
    )

    write_mode = "none"
    if missing_years:
        if args.overwrite or not output_exists_effective:
            write_mode = "create"
        elif rebuild_required:
            write_mode = "rebuild"
        else:
            write_mode = "append"

    print(f"source_year_range: {source_min_year}..{source_max_year}")
    print(f"existing_years: {existing_years}")
    print(f"requested_years: {requested_years}")
    print(f"missing_years: {missing_years}")
    print(f"write_mode: {write_mode}")

    if args.dry_run or not missing_years:
        ended = time.time()
        report = RunReport(
            uri=args.uri,
            output=str(output),
            start_year=args.start_year,
            end_year=args.end_year,
            chunk_time=args.chunk_time,
            include_tisr=args.include_tisr,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            source_year_min=source_min_year,
            source_year_max=source_max_year,
            existing_years=existing_years,
            requested_years=requested_years,
            skipped_years=skipped_years,
            appended_years=[] if args.dry_run else [],
            rebuild_required=rebuild_required,
            write_mode=write_mode,
            started_at_unix=started,
            ended_at_unix=ended,
            elapsed_sec=ended - started,
        )
        _save_report(report_path, report)
        if not missing_years:
            print("Nothing to append: all requested years already exist.")
        else:
            print("Dry run only: no writes performed.")
        print(f"report: {report_path}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)

    with LockFile(lock_path):
        if args.overwrite and output_exists_physical:
            print(f"--overwrite enabled, removing existing output: {output}")
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
            output_exists_physical = False

        appended_years: list[int] = []

        if write_mode in {"create", "append"}:
            mode = "a" if output.exists() else "w"
            total = len(missing_years)
            for i, year in enumerate(missing_years, start=1):
                t0 = time.time()
                print(f"processing year {year} ({i}/{total}) ({'append' if mode == 'a' else 'create'})")
                year_ds = _prepare_year_chunk(source, year, vars_to_keep, args.chunk_time)
                _write_year(year_ds, output, mode=mode)
                appended_years.append(year)
                mode = "a"
                elapsed = time.time() - t0
                print(f"  completed in {elapsed:.0f}s" + (f" (est. {elapsed * (total - i) / 60:.0f} min left)" if i < total else ""))

        elif write_mode == "rebuild":
            if existing_ds is None:
                raise RuntimeError("Internal error: rebuild requested but existing dataset was not loaded.")

            tmp_output = output.parent / f"{output.name}.tmp_rebuild"
            if tmp_output.exists():
                shutil.rmtree(tmp_output)

            years_to_write = sorted(set(existing_years) | set(missing_years))
            total = len(years_to_write)
            mode = "w"
            for i, year in enumerate(years_to_write, start=1):
                t0 = time.time()
                print(f"processing year {year} ({i}/{total}) ({'missing->source' if year in missing_years else 'existing->local'})")
                if year in missing_years:
                    year_ds = _prepare_year_chunk(source, year, vars_to_keep, args.chunk_time)
                    appended_years.append(year)
                else:
                    start = pd.Timestamp(year=year, month=1, day=1, hour=0)
                    end = pd.Timestamp(year=year, month=12, day=31, hour=18)
                    year_ds = existing_ds.sel(time=slice(start, end))

                _write_year(year_ds, tmp_output, mode=mode)
                mode = "a"
                elapsed = time.time() - t0
                print(f"  completed in {elapsed:.0f}s" + (f" (est. {elapsed * (total - i) / 60:.0f} min left)" if i < total else ""))

            if output.exists():
                shutil.rmtree(output)
            tmp_output.rename(output)
            zarr.consolidate_metadata(str(output))

        else:
            raise RuntimeError(f"Unsupported write mode: {write_mode}")

    final_ds = xr.open_zarr(output, consolidated=True)
    final_ds = _ensure_datetime_time(_normalize_coords(final_ds))
    final_years, partial_final = _infer_existing_years(final_ds)
    if partial_final:
        detail = "; ".join(f"{y}: {msg}" for y, msg in sorted(partial_final.items()))
        raise RuntimeError(f"Post-write verification failed (partial years): {detail}")

    ended = time.time()
    report = RunReport(
        uri=args.uri,
        output=str(output),
        start_year=args.start_year,
        end_year=args.end_year,
        chunk_time=args.chunk_time,
        include_tisr=args.include_tisr,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        source_year_min=source_min_year,
        source_year_max=source_max_year,
        existing_years=existing_years,
        requested_years=requested_years,
        skipped_years=skipped_years,
        appended_years=sorted(appended_years),
        rebuild_required=rebuild_required,
        write_mode=write_mode,
        started_at_unix=started,
        ended_at_unix=ended,
        elapsed_sec=ended - started,
    )
    _save_report(report_path, report)

    print("Done.")
    print(f"output: {output}")
    print(f"final_years: {final_years}")
    print(f"appended_years: {sorted(appended_years)}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
