#!/usr/bin/env python3
"""Stage native GraphCast inputs and one shared 1.5-degree ERA5 truth.

The frozen selection is 122 starts at 00 UTC every third day in 2023 (or
--year), each with 40 six-hour leads. Native inputs contain only the two necessary 37-level
states. Truth contains the twelve diagnostics used in the historical figure,
conservatively remapped directly from native 0.25-degree ERA5. No forecasts
are downloaded and no full-year native-resolution truth store is written.

Run with the GraphCast environment on a host with public GCS access. A bounded
first run is --limit-initializations 1 --limit-truth-times 1; remove both limits
and add --resume for production. Each atomically published NetCDF has a
.nc.READY.json receipt; resume verifies its configuration, size and SHA-256.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from scripts.analyze_models.common_grid_metrics import DIAGNOSTICS, regrid_diagnostics
from src.data_operations.staging.stage_arco_era5_res1 import (
    ChunkSource, DEFAULT_URI, PRESSURE_VARIABLES, STATIC_VARIABLES,
    SURFACE_VARIABLES, _json_hash, _write_json_atomic, _writer_lock,
)

SCHEMA_VERSION = 1
NATIVE_LAT = np.arange(-90, 90.25, .25, dtype=np.float32)
NATIVE_LON = np.arange(0, 360, .25, dtype=np.float32)
NATIVE_VARIABLES = (*SURFACE_VARIABLES, *PRESSURE_VARIABLES, "total_precipitation_6hr")
# The installed netCDF/HDF5 library is not thread-safe. Network reads and
# regridding remain parallel; every call into the native I/O library is locked.
NETCDF_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_text(value) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def file_id(value) -> str:
    return pd.Timestamp(utc_text(value)).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initializations(manifest: Path | None = None, year: int = 2023) -> pd.DatetimeIndex:
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="3D")
    if manifest is None:
        return expected
    content = json.loads(manifest.read_text())
    if isinstance(content, dict):
        content = content.get("initialization_times", content.get("initializations"))
    if not isinstance(content, list):
        raise ValueError("Manifest must be a list or contain initialization_times/initializations")
    values = pd.to_datetime(content, utc=True).tz_localize(None)
    if not values.equals(expected):
        raise ValueError(f"Manifest differs from the every-third-day {year} starts")
    return expected


def verification_times(starts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted({
        start + pd.Timedelta(hours=6 * lead)
        for start in starts for lead in range(1, 41)
    }))


def receipt_path(path: Path) -> Path:
    return Path(str(path) + ".READY.json")


def validate_receipt(path: Path, expected: dict) -> dict | None:
    """Only a matching, checksummed receipt authorizes skipping a staged file."""
    sidecar = receipt_path(path)
    if not path.exists() or not sidecar.exists():
        return None
    receipt = json.loads(sidecar.read_text())
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"Receipt {sidecar} has mismatched {key}")
    if receipt.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"Staged file size changed: {path}")
    if receipt.get("sha256") != sha256_file(path):
        raise ValueError(f"Staged file checksum changed: {path}")
    return receipt


def publish_receipt(path: Path, identity: dict) -> dict:
    receipt = {
        **identity, "schema_version": SCHEMA_VERSION, "path": str(path.resolve()),
        "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
        "created_at": utc_now(),
    }
    _write_json_atomic(receipt_path(path), receipt)
    return receipt


@contextmanager
def netcdf_file(path):
    with NETCDF_LOCK:
        handle = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        yield handle
    finally:
        with NETCDF_LOCK:
            handle.close()


def configure_file(handle, dimensions, attrs):
    with NETCDF_LOCK:
        for name, size in dimensions.items():
            handle.createDimension(name, size)
        handle.setncatts(attrs)


def assign(variable, index, values):
    with NETCDF_LOCK:
        variable[index] = values


def coordinate(handle, name, values, attrs=None):
    values = np.asarray(values)
    with NETCDF_LOCK:
        field = handle.createVariable(name, values.dtype, (name,))
        field[:] = values
        if attrs:
            field.setncatts(attrs)


def data_variable(handle, name, dims, attrs=None, *, dtype="f4"):
    with NETCDF_LOCK:
        sizes = [len(handle.dimensions[dim]) for dim in dims]
        chunks = tuple(1 if dim in {"time", "level"} else min(size, 180)
                       for dim, size in zip(dims, sizes))
        field = handle.createVariable(name, dtype, dims, chunksizes=chunks,
                                      zlib=True, complevel=1, shuffle=True,
                                      fill_value=np.nan)
        if attrs:
            field.setncatts(attrs)
    return field


def simple_attrs(source: ChunkSource, name: str) -> dict:
    original = source.attrs(name)
    return {key: original[key] for key in ("units", "long_name", "standard_name")
            if key in original}


def native_field(source: ChunkSource, name: str, hour: int) -> np.ndarray:
    indices = (hour, 0, 0, 0) if name in PRESSURE_VARIABLES else (hour, 0, 0)
    values = source.read_chunk(name, indices)[0]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite native ERA5 field {name} at source hour {hour}")
    # The native ARCO latitude order is north to south. Files use ascending lat.
    return values[..., ::-1, :]


def static_fields(source: ChunkSource, hour: int) -> dict:
    fields = {}
    for name in STATIC_VARIABLES:
        if source.spec(name)["chunks"] != [1, 721, 1440]:
            raise ValueError(f"Unsupported source static-field layout for {name}")
        fields[name] = np.array(native_field(source, name, hour), copy=True)
    return fields


def write_inputs(source: ChunkSource, path: Path, start: pd.Timestamp,
                 hour_lookup: dict, levels: np.ndarray, statics: dict,
                 identity: dict) -> dict:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
    times = pd.DatetimeIndex([start - pd.Timedelta(hours=6), start])
    with netcdf_file(temporary) as handle:
        configure_file(handle, {"time": 2, "level": 37, "lat": 721, "lon": 1440}, {
            "source": source.uri, "configuration_sha256": identity["configuration_sha256"],
            "initialization_time": utc_text(start),
            "precipitation_convention": "Six hourly depths ending at t-5h,...,t; metres",
            "solar_forcing": "Generated downstream by vendored GraphCast one-hour integration",
            "static_source_time": identity["static_source_time"],
        })
        coordinate(handle, "time", times.to_numpy().astype("datetime64[h]").astype(np.int64),
                   {"units": "hours since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"})
        coordinate(handle, "lat", NATIVE_LAT, {"units": "degrees_north"})
        coordinate(handle, "lon", NATIVE_LON, {"units": "degrees_east"})
        coordinate(handle, "level", levels, {"units": "hPa"})
        for name in NATIVE_VARIABLES:
            source_name = "total_precipitation" if name == "total_precipitation_6hr" else name
            dims = (("time", "level", "lat", "lon") if name in PRESSURE_VARIABLES
                    else ("time", "lat", "lon"))
            attrs = simple_attrs(source, source_name)
            if name == "total_precipitation_6hr":
                attrs["cell_methods"] = "time: sum (interval: 1 hour; accumulation: 6 hours ending at time)"
            variable = data_variable(handle, name, dims, attrs)
            for index, stamp in enumerate(times):
                hour = hour_lookup[stamp]
                if name == "total_precipitation_6hr":
                    values = np.zeros((721, 1440), dtype=np.float32)
                    for raw_hour in range(hour - 5, hour + 1):
                        values += native_field(source, "total_precipitation", raw_hour)
                else:
                    values = native_field(source, name, hour)
                assign(variable, index, values)
                del values
        for name, values in statics.items():
            variable = data_variable(handle, name, ("lat", "lon"), simple_attrs(source, name))
            assign(variable, slice(None), values)
    os.replace(temporary, path)
    return publish_receipt(path, {**identity, "input_times": [utc_text(t) for t in times]})


def write_truth(source: ChunkSource, path: Path, stamp: pd.Timestamp,
                hour_lookup: dict, levels: np.ndarray, identity: dict) -> dict:
    fields = {}
    hour = hour_lookup[stamp]
    # Decode each all-level native chunk once, retaining only selected planes.
    for source_name in dict.fromkeys(spec.variable for spec in DIAGNOSTICS.values()):
        native = native_field(source, source_name, hour)
        for key, spec in DIAGNOSTICS.items():
            if spec.variable != source_name:
                continue
            if spec.level is None:
                values = native.copy()
            else:
                matches = np.flatnonzero(levels == spec.level)
                if len(matches) != 1:
                    raise ValueError(f"Missing unique pressure level {spec.level}")
                values = native[int(matches[0])].copy()
            fields[key] = xr.DataArray(values, dims=("lat", "lon"),
                                       coords={"lat": NATIVE_LAT, "lon": NATIVE_LON})
        del native
    truth = regrid_diagnostics(xr.Dataset(fields))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
    with netcdf_file(temporary) as handle:
        configure_file(handle, {"lat": 121, "lon": 240}, {**truth.attrs, "source": source.uri,
                             "valid_time": utc_text(stamp),
                             "configuration_sha256": identity["configuration_sha256"]})
        coordinate(handle, "lat", truth.lat.values, {"units": "degrees_north"})
        coordinate(handle, "lon", truth.lon.values, {"units": "degrees_east"})
        for key in DIAGNOSTICS:
            # Preserve float64 regridding arithmetic for all three forecasts.
            variable = data_variable(handle, key, ("lat", "lon"), truth[key].attrs, dtype="f8")
            assign(variable, slice(None), truth[key].values)
    os.replace(temporary, path)
    return publish_receipt(path, identity)


def stage(args) -> dict:
    if args.workers < 1 or any(value is not None and value < 1 for value in (
            args.limit_initializations, args.limit_truth_times)):
        raise ValueError("Worker and optional limit counts must be positive")
    starts = initializations(args.manifest, args.year)
    all_truth = verification_times(starts)
    requested_starts = starts[:args.limit_initializations] if args.limit_initializations else starts
    requested_truth = verification_times(requested_starts)
    if args.limit_truth_times is not None:
        requested_truth = requested_truth[:args.limit_truth_times]
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _writer_lock(output / ".staging.lock"):
        source = ChunkSource(args.source_uri)
        source.validate_layout()
        levels = source.read_vector("level")
        if not np.array_equal(levels, np.sort(levels)):
            raise ValueError("Native pressure levels must be ascending")
        all_times = pd.DatetimeIndex(sorted(set(all_truth) | set(starts)
                                             | set(starts - pd.Timedelta(hours=6))))
        hours = source.hourly_indices(all_times)
        hour_lookup = dict(zip(all_times, map(int, hours)))
        static_stamp = starts[0] - pd.Timedelta(hours=6)
        config = {
            "schema_version": SCHEMA_VERSION, "source_uri": source.uri,
            "source_structure_sha256": source.fingerprint,
            "source_static_layout_sha256": _json_hash({
                name: {"spec": source.spec(name), "attrs": source.attrs(name)}
                for name in STATIC_VARIABLES
            }),
            "initialization_times": [utc_text(value) for value in starts],
            "lead_hours": list(range(6, 241, 6)),
            "input_grid": "0.25 degrees; 721 ascending lat x 1440 lon; 37 pressure levels",
            "verification_grid": "1.5 degrees; 121 ascending lat x 240 lon",
            "truth_source_grid": "Native 0.25-degree ERA5; first-order conservative spherical remap",
            "diagnostics": {key: dict(spec._asdict()) for key, spec in DIAGNOSTICS.items()},
            "static_source_time": utc_text(static_stamp),
            "precipitation_convention": "Sum six hourly depths ending at t-5h through t; metres",
            "solar_forcing": "Generated downstream with vendored GraphCast one-hour integration",
            "stager_sha256": sha256_file(Path(__file__)),
            "regrid_code_sha256": sha256_file(ROOT / "scripts/analyze_models/common_grid_metrics.py"),
        }
        config_hash = _json_hash(config)
        manifest_path = output / "staging_manifest.json"
        manifest = {"configuration": config, "configuration_sha256": config_hash,
                    "expected_inputs025_count": len(starts),
                    "expected_common_truth_count": len(all_truth),
                    "truth_first": utc_text(all_truth[0]), "truth_last": utc_text(all_truth[-1])}
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old != manifest:
                raise ValueError("Existing staging manifest differs; use a new output directory")
            if not args.resume and not args.metadata_only:
                raise FileExistsError("Staging manifest exists; use --resume")
        else:
            _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(output / "source_metadata.json", source.metadata)
        print(json.dumps(manifest, indent=2), flush=True)
        if args.metadata_only:
            return manifest
        for folder in ("inputs025", "common_truth"):
            (output / folder).mkdir(exist_ok=True)
        jobs = []
        if args.mode in {"both", "inputs"}:
            jobs.extend(("inputs025", stamp) for stamp in requested_starts)
        if args.mode in {"both", "truth"}:
            jobs.extend(("common_truth", stamp) for stamp in requested_truth)
        needed, verified = [], []
        for kind, stamp in jobs:
            path = output / kind / f"{file_id(stamp)}.nc"
            identity = {
                "kind": kind, "timestamp": utc_text(stamp),
                "configuration_sha256": config_hash, "source_uri": source.uri,
                "source_structure_sha256": source.fingerprint,
                "static_source_time": utc_text(static_stamp),
            }
            receipt = validate_receipt(path, identity)
            if receipt is not None:
                verified.append(receipt)
            else:
                if (path.exists() or receipt_path(path).exists()) and not args.resume:
                    raise FileExistsError(f"Unreceipted staging artifact exists: {path}")
                needed.append((kind, stamp, path, identity))
        statics = (static_fields(source, hour_lookup[static_stamp])
                   if any(job[0] == "inputs025" for job in needed) else {})
        report_path = output / "staging_report.json"
        started = time.monotonic()
        report = {
            "schema_version": SCHEMA_VERSION, "configuration_sha256": config_hash,
            "started_at": utc_now(), "host": socket.gethostname(), "workers": args.workers,
            "status": "running", "requested_count": len(jobs),
            "verified_existing_count": len(verified), "newly_completed_count": 0,
            "mode": args.mode, "limit_initializations": args.limit_initializations,
            "limit_truth_times": args.limit_truth_times,
        }
        _write_json_atomic(report_path, report)

        def run(job):
            kind, stamp, path, identity = job
            if kind == "inputs025":
                return write_inputs(source, path, stamp, hour_lookup, levels, statics, identity)
            return write_truth(source, path, stamp, hour_lookup, levels, identity)

        def update_report(status):
            report.update({"status": status, "updated_at": utc_now(),
                           "elapsed_seconds": time.monotonic() - started,
                           "downloaded_bytes": source.downloaded_bytes,
                           "request_count": source.request_count,
                           "request_seconds": source.request_seconds})
            _write_json_atomic(report_path, report)

        iterator, pending = iter(needed), {}
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for _ in range(min(args.workers, len(needed))):
                    job = next(iterator)
                    pending[executor.submit(run, job)] = job
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = pending.pop(future)
                        receipt = future.result()
                        verified.append(receipt)
                        report["newly_completed_count"] += 1
                        update_report("running")
                        print(f"ready {receipt['kind']}/{Path(receipt['path']).name} "
                              f"files={len(verified)}/{len(jobs)} "
                              f"downloaded_GB={source.downloaded_bytes / 1e9:.2f}", flush=True)
                        next_job = next(iterator, None)
                        if next_job is not None:
                            pending[executor.submit(run, next_job)] = next_job
        except BaseException as error:
            report["error"] = f"{type(error).__name__}: {error}"
            update_report("failed")
            raise
        complete = (args.mode == "both" and len(requested_starts) == len(starts)
                    and len(requested_truth) == len(all_truth))
        update_report("complete" if complete else "requested_subset_complete")
        if complete:
            _write_json_atomic(output / ".READY.json", {
                "schema_version": SCHEMA_VERSION, "configuration_sha256": config_hash,
                "input_count": len(starts), "truth_count": len(all_truth),
                "created_at": utc_now(), "manifest_sha256": sha256_file(manifest_path),
                "files": [{key: receipt[key] for key in (
                    "path", "sha256", "size_bytes", "kind", "timestamp")}
                    for receipt in sorted(verified, key=lambda value: value["path"])],
            })
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--source-uri", default=DEFAULT_URI)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=("both", "inputs", "truth"), default="both")
    parser.add_argument("--limit-initializations", type=int)
    parser.add_argument("--limit-truth-times", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    print(json.dumps(stage(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
