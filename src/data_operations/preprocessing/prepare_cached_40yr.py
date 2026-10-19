"""Bounded, resumable annual preparation for the 1982--2022 cached campaign.

Annual workers never publish a trainable root. Only ``finalize_prepared`` can
publish complete format-v2 metadata and READY after all 41 years verify.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Iterable
import uuid

import dask
import numpy as np
import pandas as pd
import xarray as xr

from src.data_operations.loaders.graphcast_dataset import _zarr_to_graphcast_layout
from src.models.graphcast.training.core.dataset import prepare_dataset_for_task
from src.models.graphcast.training.core.model import load_graphcast_checkpoint
from src.models.graphcast.training.core.prepared_array import PreparedArrayStore

YEARS = tuple(range(1982, 2023))
FORMAT = "cached_40yr_prepared_v1"


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _times(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31 18:00", freq="6h")


def _identity(source: Path, checkpoint: Path) -> dict:
    return {"source": str(source.resolve()), "source_metadata_sha256": _sha(source / ".zmetadata"),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha(checkpoint)}


def _check_grid(dataset: xr.Dataset, checkpoint) -> None:
    if not np.isclose(float(checkpoint.model_config.resolution), 1.0):
        raise ValueError("Campaign requires the 1-degree baseline checkpoint")
    levels = list(checkpoint.task_config.pressure_levels)
    if len(levels) != 13 or not np.array_equal(dataset.level.values, levels):
        raise ValueError("Campaign requires the checkpoint's exact 13 pressure levels")
    lat, lon = np.asarray(dataset.lat), np.asarray(dataset.lon)
    if (lat.shape != (181,) or lon.shape != (360,) or not np.isfinite(lat).all()
            or not np.isfinite(lon).all() or not np.allclose(np.abs(np.diff(lat)), 1)
            or not np.allclose(np.diff(lon), 1) or not np.isclose(np.min(lat), -90)
            or not np.isclose(np.max(lat), 90) or not np.isclose(lon[0], 0)):
        raise ValueError("Source must have the complete 181 x 360 one-degree grid")


def _prepared_month(raw: xr.Dataset, task) -> xr.Dataset:
    prepared = prepare_dataset_for_task(_zarr_to_graphcast_layout(raw), task)
    names = tuple(dict.fromkeys((*task.input_variables, *task.target_variables, *task.forcing_variables)))
    result = {}
    for name in names:
        var = prepared[name]
        if "batch" in var.dims:
            if var.sizes["batch"] != 1:
                raise ValueError("Expected one source batch")
            var = var.isel(batch=0, drop=True)
        if "level" in var.dims:
            var = var.sel(level=list(task.pressure_levels))
        if "time" in var.dims:
            var = var.transpose("time", *(dim for dim in var.dims if dim != "time"))
        result[name] = var.astype(np.float32)
    return xr.Dataset(result, coords={name: prepared.coords[name] for name in ("time", "lat", "lon", "level")})


def _slice_digest(path: Path, start: int, stop: int, chunk_time: int) -> str:
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    digest = hashlib.sha256()
    try:
        for offset in range(start, stop, chunk_time):
            block = np.asarray(mapped[offset:min(stop, offset + chunk_time)])
            if not np.isfinite(block).all():
                raise ValueError(f"Nonfinite prepared values: {path}")
            digest.update(np.ascontiguousarray(block).tobytes())
            del block
    finally:
        mapped._mmap.close()
    return digest.hexdigest()


def _verify_month(root: Path, record: dict, variables: dict, chunk_time: int) -> None:
    for name, expected in record["sha256"].items():
        path = root / "vars" / f"{name}.npy"
        info = variables[name]
        if "time" in info["dims"]:
            actual = _slice_digest(path, record["start"], record["stop"], chunk_time)
        else:
            actual = _sha(path)
        if actual != expected:
            raise ValueError(f"Prepared month checksum mismatch: {path}")
    if set(record["sha256"]) != set(variables):
        raise ValueError("Prepared month has incomplete variable coverage")


def _verify_year(root: Path, identity: dict, year: int, *, chunk_time: int = 8) -> dict:
    record = json.loads((root / "COMPLETE.json").read_text())
    if record.get("identity") != identity or record.get("year") != year or record.get("format") != FORMAT:
        raise ValueError(f"Annual preparation provenance mismatch: {root}")
    months = record["months"]
    if set(months) != {f"{year}-{month:02d}" for month in range(1, 13)}:
        raise ValueError(f"Incomplete annual month coverage: {year}")
    np.testing.assert_array_equal(np.load(root / "coords/time.npy"), _times(year).values)
    expected_files = {f"vars/{name}.npy" for name in record["variables"]} | {
        f"coords/{name}.npy" for name in ("time", "lat", "lon", "level")}
    if set(record["files"]) != expected_files:
        raise ValueError(f"Incomplete annual file coverage: {year}")
    for relative, saved in record["files"].items():
        if _sha(root / relative) != saved["sha256"]:
            raise ValueError(f"Annual checksum mismatch: {root / relative}")
    for month in months.values():
        _verify_month(root, month, record["variables"], chunk_time)
    return record


def prepare_year(source: Path, output: Path, year: int, baseline_checkpoint: Path,
                 *, chunk_time: int = 8) -> dict:
    """Write or verify one annual shard; resume only checksum-verified months."""
    source, output, baseline_checkpoint = map(Path, (source, output, baseline_checkpoint))
    if year not in YEARS or chunk_time < 1:
        raise ValueError("Expected year 1982..2022 and positive chunk_time")
    identity = _identity(source, baseline_checkpoint)
    root = output / "years" / str(year)
    started = time.monotonic()
    with _lock(output / f".year_{year}.lock"):
        if (root / "COMPLETE.json").exists():
            return _verify_year(root, identity, year, chunk_time=chunk_time)
        with _lock(output / ".metadata.lock"):
            if (output / "READY.json").exists():
                raise ValueError("Cannot modify a finalized prepared dataset")
            (output / ".incomplete").touch()
        checkpoint = load_graphcast_checkpoint(baseline_checkpoint)
        task = checkpoint.task_config
        with xr.open_zarr(source, consolidated=True) as dataset, dask.config.set(scheduler="threads", num_workers=2):
            if "latitude" in dataset.coords:
                dataset = dataset.rename({"latitude": "lat", "longitude": "lon"})
            _check_grid(dataset, checkpoint)
            year_data = dataset.sel(time=slice(f"{year}-01-01", f"{year}-12-31T18:00:00"))
            times = _times(year)
            np.testing.assert_array_equal(year_data.time.values.astype("datetime64[ns]"), times.values)
            progress_path = root / "progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                if (progress.get("identity") != identity or progress.get("year") != year
                        or progress.get("format") != FORMAT):
                    raise ValueError("Cannot resume preparation against changed source/checkpoint")
            else:
                progress = {"format": FORMAT, "identity": identity, "year": year, "months": {}, "variables": {}}
            for month in range(1, 13):
                key = f"{year}-{month:02d}"
                if key in progress["months"]:
                    _verify_month(root, progress["months"][key], progress["variables"], chunk_time)
                    continue
                positions = np.flatnonzero(times.month == month)
                start, stop = int(positions[0]), int(positions[-1]) + 1
                prepared = _prepared_month(year_data.isel(time=slice(start, stop)), task)
                checksums = {}
                schemas = {}
                for name, var in prepared.data_vars.items():
                    shape = list(var.shape)
                    if "time" in var.dims:
                        shape[0] = len(times)
                    schemas[name] = {"dims": list(var.dims), "shape": shape, "dtype": "float32"}
                    path = root / "vars" / f"{name}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if "time" not in var.dims:
                        values = np.asarray(var.values, dtype=np.float32)
                        if not np.isfinite(values).all():
                            raise ValueError(f"Nonfinite source static variable: {name}")
                        if path.exists():
                            np.testing.assert_array_equal(values, np.load(path))
                        else:
                            _npy(path, values)
                        checksums[name] = _sha(path)
                        continue
                    destination = (np.load(path, mmap_mode="r+", allow_pickle=False) if path.exists()
                                   else np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=tuple(shape)))
                    try:
                        if destination.shape != tuple(shape) or destination.dtype != np.float32:
                            raise ValueError(f"Prepared annual array schema mismatch: {path}")
                        written = hashlib.sha256()
                        for offset in range(0, stop - start, chunk_time):
                            end = min(stop - start, offset + chunk_time)
                            values = np.ascontiguousarray(var.isel(time=slice(offset, end)).values, dtype=np.float32)
                            if not np.isfinite(values).all():
                                raise ValueError(f"Nonfinite source {key}/{name}")
                            destination[start + offset:start + end] = values
                            written.update(values.tobytes())
                        destination.flush()
                    finally:
                        destination._mmap.close()
                    checksums[name] = written.hexdigest()
                if progress["variables"] and progress["variables"] != schemas:
                    raise ValueError("Prepared schema changed between months")
                progress["variables"] = schemas
                for name in ("time", "lat", "lon", "level"):
                    path = root / "coords" / f"{name}.npy"
                    values = times.values if name == "time" else np.asarray(prepared.coords[name])
                    if path.exists():
                        np.testing.assert_array_equal(np.load(path), values)
                    else:
                        _npy(path, values)
                month_record = {"start": start, "stop": stop, "sha256": checksums}
                _verify_month(root, month_record, schemas, chunk_time)
                progress["months"][key] = month_record
                _json(progress_path, progress)
                print(f"prepared {key}: verified {stop - start} frames", flush=True)
            files = {str(path.relative_to(root)): {"sha256": _sha(path), "bytes": path.stat().st_size}
                     for path in sorted(root.rglob("*.npy"))}
            report = {**progress, "files": files, "elapsed_seconds": time.monotonic() - started,
                      "peak_host_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                      "status": "complete", "bytes": sum(value["bytes"] for value in files.values())}
            _json(root / "COMPLETE.json", report)
            return report


def finalize_prepared(source: Path, output: Path, years: Iterable[int], baseline_checkpoint: Path) -> dict:
    """Verify the exact campaign coverage and publish a trainable format-v2 root."""
    source, output, baseline_checkpoint = map(Path, (source, output, baseline_checkpoint))
    if tuple(years) != YEARS:
        raise ValueError("Finalization requires every year 1982..2022 in order")
    identity = _identity(source, baseline_checkpoint)
    with _lock(output / ".metadata.lock"):
        ready_path = output / "READY.json"
        if ready_path.exists():
            ready = json.loads(ready_path.read_text())
            if (ready.get("identity") != identity or ready.get("years") != list(YEARS)
                    or ready.get("metadata_sha256") != _sha(output / "metadata.json")
                    or (output / ".incomplete").exists()):
                raise ValueError("Finalized prepared identity changed")
            for relative, expected in ready["root_file_sha256"].items():
                if _sha(output / relative) != expected:
                    raise ValueError(f"Finalized prepared file changed: {relative}")
            for year in YEARS:
                with _lock(output / f".year_{year}.lock"):
                    _verify_year(output / "years" / str(year), identity, year)
            return ready
        (output / "READY.json").unlink(missing_ok=True)
        (output / ".incomplete").touch()
        records = []
        for year in YEARS:
            with _lock(output / f".year_{year}.lock"):
                records.append(_verify_year(output / "years" / str(year), identity, year))
        checkpoint = load_graphcast_checkpoint(baseline_checkpoint)
        times = pd.DatetimeIndex(np.concatenate([_times(year).values for year in YEARS]))
        variables = json.loads(json.dumps(records[0]["variables"]))
        for name, info in variables.items():
            if "time" in info["dims"]:
                info["shape"][0] = len(times)
            for record in records:
                current = record["variables"].get(name)
                if current is None or current["dims"] != info["dims"] or current["dtype"] != info["dtype"]:
                    raise ValueError(f"Annual schema mismatch for {name}")
                expected_shape = list(info["shape"])
                if "time" in info["dims"]:
                    expected_shape[0] = len(_times(record["year"]))
                if current["shape"] != expected_shape:
                    raise ValueError(f"Annual shape mismatch for {name}")
            if "time" not in info["dims"]:
                relative = f"vars/{name}.npy"
                if len({r["files"][relative]["sha256"] for r in records}) != 1:
                    raise ValueError(f"Static field changes across years: {name}")
                _npy(output / relative, np.load(output / "years" / "1982" / relative))
        _npy(output / "coords/time.npy", times.values)
        for name in ("lat", "lon", "level"):
            relative = f"coords/{name}.npy"
            if len({r["files"][relative]["sha256"] for r in records}) != 1:
                raise ValueError(f"Coordinate changes across years: {name}")
            _npy(output / relative, np.load(output / "years" / "1982" / relative))
        shards, cursor = [], 0
        for year in YEARS:
            year_times = _times(year)
            shards.append({"year": year, "path": f"years/{year}", "start": cursor,
                           "stop": cursor + len(year_times), "time_start": str(year_times[0].to_datetime64()),
                           "time_end": str(year_times[-1].to_datetime64())})
            cursor += len(year_times)
            count = len(year_times)
            _json(output / f"years/{year}/validity.json", {
                "written_time_mask": [True] * count,
                "time_finite_by_variable": {name: [True] * count for name, info in variables.items() if "time" in info["dims"]}})
        task = checkpoint.task_config
        metadata = {"prepared_array_format_version": 2, "source_data_path": str(source.resolve()),
                    "checkpoint_sha256": identity["checkpoint_sha256"], "resolution": 1.0, "base_resolution": 1.0,
                    "pressure_levels": list(task.pressure_levels), "task_input_variables": list(task.input_variables),
                    "task_target_variables": list(task.target_variables), "task_forcing_variables": list(task.forcing_variables),
                    "time_start": str(times[0].to_datetime64()), "time_end": str(times[-1].to_datetime64()),
                    "time_shards": shards, "variables": variables}
        _json(output / "metadata.json", metadata)
        build = {"format": FORMAT, "identity": identity, "status": "complete",
                 "months": {key: {"status": "complete", **value} for r in records for key, value in r["months"].items()}}
        _json(output / "build_manifest.json", build)
        # Validation can open an incomplete root only with explicit complete bounds.
        store = PreparedArrayStore(output, bounded_reads=True, allow_incomplete=True,
                                   time_start=str(times[0]), time_end=str(times[-1]))
        store.validate(resolution=1.0, task_cfg=task)
        for shard in shards[1:]:
            store.build_batch_from_indices(indices=[shard["start"]], input_steps=2, target_steps=1,
                                           task_cfg=task, dt=pd.Timedelta(hours=6))
        ready = {"format": FORMAT, "identity": identity, "years": list(YEARS), "time_steps": len(times),
                 "metadata_sha256": _sha(output / "metadata.json"), "status": "complete",
                 "root_file_sha256": {str(path.relative_to(output)): _sha(path)
                                      for directory in (output / "coords", output / "vars")
                                      for path in sorted(directory.glob("*.npy"))},
                 "bytes": sum(record["bytes"] for record in records)}
        (output / ".incomplete").unlink()
        _json(output / "READY.json", ready)
        return ready
