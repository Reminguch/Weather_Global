#!/usr/bin/env python3
"""Resume a bounded-memory ARCO ERA5 download on the existing res1 grid.

Only one remote pressure-level chunk is decoded at a time per worker. Source
chunks are read directly because their all-level, whole-globe layout would
otherwise make the memory/concurrency of lazy array reads harder to bound.
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
import threading
import time
from urllib.parse import urlparse

import numcodecs
import numpy as np
import pandas as pd
import requests
import zarr

from src.data_operations.staging.stage_wb2_era5_yearly_append import PRESSURE_LEVELS_13


DEFAULT_URI = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_OUTPUT = Path("data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr")
DEFAULT_REFERENCE = Path("data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr")
DEFAULT_START = "2022-12-31T12:00:00"
DEFAULT_END = "2024-01-10T18:00:00"
SURFACE_VARIABLES = (
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
PRESSURE_VARIABLES = (
    "temperature", "geopotential", "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity",
)
STATIC_VARIABLES = ("geopotential_at_surface", "land_sea_mask")
TIME_VARIABLES = SURFACE_VARIABLES + PRESSURE_VARIABLES + ("total_precipitation_6hr",)
SOURCE_VARIABLES = SURFACE_VARIABLES + PRESSURE_VARIABLES + ("total_precipitation",)
SOURCE_COORDS = ("time", "latitude", "longitude", "level")
FORMAT_VERSION = 1
RETRY_ATTEMPTS = 5

# Workers already provide chunk-level parallelism; nested codec threads waste CPU.
numcodecs.blosc.set_nthreads(1)


def expected_time_grid(start_time: str, end_time: str) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("Use timezone-naive UTC timestamps.")
    if end < start:
        raise ValueError("end_time must be at or after start_time")
    for value in (start, end):
        if value != value.floor("6h"):
            raise ValueError("Output timestamps must align to 00/06/12/18 UTC.")
    return pd.date_range(start, end, freq="6h")


def sum_hourly_precipitation(values: np.ndarray) -> np.ndarray:
    """Sum six hourly depths ending at the output timestamp, preserving metres."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != 6 or not np.isfinite(values).all():
        raise ValueError("Precipitation needs exactly six finite hourly fields.")
    return np.sum(values, axis=0, dtype=np.float32)


def _json_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def structural_metadata_fingerprint(metadata: dict) -> str:
    """Fingerprint required array layouts; ignore mutable archive coverage attrs."""
    entries = metadata.get("metadata", metadata)
    selected = {}
    for name in SOURCE_COORDS + SOURCE_VARIABLES:
        array = dict(entries[f"{name}/.zarray"])
        attrs = dict(entries[f"{name}/.zattrs"])
        # An archive may append times without changing historical chunk layout.
        if attrs["_ARRAY_DIMENSIONS"][0] == "time":
            array["shape"] = ["appendable"] + list(array["shape"][1:])
        selected[name] = {"array": array, "attrs": attrs}
    return _json_hash(selected)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _writer_lock(path: Path):
    """Advisory lock automatically releases after a crash, allowing resume."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another staging process holds {path}") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started": _utc_now()}))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class ChunkSource:
    """Read fixed-layout Zarr v2 chunks without an unbounded task graph."""

    def __init__(self, uri: str, metadata_path: Path | None = None):
        self.uri = str(uri).rstrip("/")
        parsed = urlparse(self.uri)
        self.local_path = None
        if parsed.scheme == "gs":
            self.base_url = f"https://storage.googleapis.com/{parsed.netloc}{parsed.path}"
        elif parsed.scheme in {"http", "https"}:
            self.base_url = self.uri
        elif parsed.scheme in {"", "file"}:
            self.local_path = Path(parsed.path if parsed.scheme else self.uri)
            self.base_url = None
        else:
            raise ValueError(f"Unsupported source URI scheme: {parsed.scheme}")
        self._thread_local = threading.local()
        self._counter_lock = threading.Lock()
        self.downloaded_bytes = 0
        self.request_seconds = 0.0
        self.request_count = 0
        self.metadata = json.loads(self.read_bytes(".zmetadata"))
        self.entries = self.metadata["metadata"]
        self.fingerprint = structural_metadata_fingerprint(self.metadata)
        if metadata_path is not None:
            snapshot = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if structural_metadata_fingerprint(snapshot) != self.fingerprint:
                raise ValueError("Live source layout differs from the supplied metadata snapshot.")

    def read_bytes(self, key: str) -> bytes:
        for attempt in range(RETRY_ATTEMPTS):
            started = time.monotonic()
            try:
                if self.local_path is not None:
                    value = (self.local_path / key).read_bytes()
                else:
                    session = getattr(self._thread_local, "session", None)
                    if session is None:
                        session = requests.Session()
                        self._thread_local.session = session
                    with session.get(f"{self.base_url}/{key}", timeout=(30, 180)) as response:
                        response.raise_for_status()
                        value = response.content
                with self._counter_lock:
                    self.downloaded_bytes += len(value)
                    self.request_seconds += time.monotonic() - started
                    self.request_count += 1
                return value
            except (requests.RequestException, OSError) as exc:
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    if exc.response.status_code < 500 and exc.response.status_code not in {408, 429}:
                        raise
                if isinstance(exc, FileNotFoundError) or attempt == RETRY_ATTEMPTS - 1:
                    raise
                delay = min(2 ** attempt, 16)
                print(f"retry source chunk {key}: attempt={attempt + 1}/{RETRY_ATTEMPTS} error={exc}", flush=True)
                time.sleep(delay)
        raise AssertionError("Unreachable retry loop")

    def attrs(self, name: str) -> dict:
        return dict(self.entries[f"{name}/.zattrs"])

    def spec(self, name: str) -> dict:
        return self.entries[f"{name}/.zarray"]

    def read_chunk(self, name: str, chunk_index: tuple[int, ...]) -> np.ndarray:
        spec = self.spec(name)
        separator = spec.get("dimension_separator", ".")
        key = name + "/" + separator.join(map(str, chunk_index))
        payload = self.read_bytes(key)
        codec = spec.get("compressor")
        decoded = numcodecs.get_codec(codec).decode(payload) if codec else payload
        for filter_spec in reversed(spec.get("filters") or []):
            decoded = numcodecs.get_codec(filter_spec).decode(decoded)
        values = np.frombuffer(decoded, dtype=np.dtype(spec["dtype"]))
        expected_count = int(np.prod(spec["chunks"]))
        if values.size != expected_count:
            raise ValueError(f"Decoded chunk {key}: expected {expected_count} values, got {values.size}")
        return values.reshape(spec["chunks"], order=spec.get("order", "C"))

    def read_vector(self, name: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        spec = self.spec(name)
        if len(spec["shape"]) != 1:
            raise ValueError(f"Expected one-dimensional coordinate {name}")
        stop = int(spec["shape"][0]) if stop is None else stop
        if start < 0 or stop > spec["shape"][0] or start >= stop:
            raise ValueError(f"Invalid source coordinate range {name}[{start}:{stop}]")
        width = int(spec["chunks"][0])
        parts = []
        for chunk in range(start // width, (stop - 1) // width + 1):
            values = self.read_chunk(name, (chunk,))
            lo, hi = max(start - chunk * width, 0), min(stop - chunk * width, width)
            parts.append(values[lo:hi].copy())
        return np.concatenate(parts)

    def validate_layout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for name in SOURCE_VARIABLES:
            pressure = name in PRESSURE_VARIABLES
            expected_dims = ["time"] + (["level"] if pressure else []) + ["latitude", "longitude"]
            expected_chunk = [1] + ([37] if pressure else []) + [721, 1440]
            spec = self.spec(name)
            if self.attrs(name)["_ARRAY_DIMENSIONS"] != expected_dims or spec["chunks"] != expected_chunk:
                raise ValueError(f"Unsupported ARCO source layout for {name}: {spec}")
            if spec["shape"][1:] != expected_chunk[1:] or np.dtype(spec["dtype"]) != np.dtype("float32"):
                raise ValueError(f"Unexpected source shape/dtype for {name}")
        lat = self.read_vector("latitude")
        lon = self.read_vector("longitude")
        levels = self.read_vector("level")
        if not np.array_equal(lat, np.arange(90, -90.25, -0.25, dtype=np.float32)):
            raise ValueError("Source latitude must be descending 90..-90 at 0.25 degrees.")
        if not np.array_equal(lon, np.arange(0, 360, 0.25, dtype=np.float32)):
            raise ValueError("Source longitude must be 0..359.75 at 0.25 degrees.")
        if len(levels) != 37 or len(set(levels.tolist())) != 37:
            raise ValueError("Source pressure-level coordinate must contain 37 unique levels.")
        indices = []
        for level in PRESSURE_LEVELS_13:
            match = np.flatnonzero(levels == level)
            if len(match) != 1:
                raise ValueError(f"Missing pressure level {level}")
            indices.append(int(match[0]))
        return lat[::4].copy(), lon[::4].copy(), np.asarray(indices)

    def hourly_indices(self, times: pd.DatetimeIndex) -> np.ndarray:
        attrs = self.attrs("time")
        units = attrs.get("units", "")
        if not units.startswith("hours since ") or attrs.get("calendar") not in {"proleptic_gregorian", "gregorian", "standard"}:
            raise ValueError(f"Unsupported source time encoding: {attrs}")
        epoch = pd.Timestamp(units.removeprefix("hours since "))
        first = int(self.read_vector("time", 0, 1)[0])
        wanted_hours = ((times - epoch) / pd.Timedelta(hours=1)).to_numpy()
        if not np.all(wanted_hours == wanted_hours.astype(np.int64)):
            raise ValueError("Output times are not whole hours in source encoding.")
        indices = wanted_hours.astype(np.int64) - first
        raw_start, raw_stop = int(indices[0]) - 5, int(indices[-1]) + 1
        actual = self.read_vector("time", raw_start, raw_stop)
        expected = np.arange(raw_start + first, raw_stop + first, dtype=np.int64)
        if not np.array_equal(actual, expected):
            raise ValueError("Source time coordinates are not a complete ordered hourly sequence.")
        for name in SOURCE_VARIABLES:
            if self.spec(name)["shape"][0] < raw_stop:
                raise ValueError(f"Source {name} does not cover the requested window.")
        # Shape may be preallocated years past actual final ERA5 data coverage.
        valid_stop = self.entries.get(".zattrs", {}).get("valid_time_stop")
        if valid_stop is not None and times[-1] >= pd.Timestamp(valid_stop) + pd.Timedelta(days=1):
            raise ValueError(f"Requested window exceeds finalized ERA5 coverage {valid_stop}.")
        return indices

    def field(self, name: str, hour_index: int, level_indices: np.ndarray) -> np.ndarray:
        pressure = name in PRESSURE_VARIABLES
        indices = (hour_index, 0, 0, 0) if pressure else (hour_index, 0, 0)
        whole = self.read_chunk(name, indices)
        # Slice space before advanced level indexing to avoid a 54MB intermediate.
        values = whole[0, :, ::4, ::4][level_indices] if pressure else whole[0, ::4, ::4]
        return np.array(values, dtype=np.float32, order="C", copy=True)


def _array_digest(digest, name: str, values: np.ndarray) -> None:
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(np.ascontiguousarray(values).tobytes())


def _reference_static(reference: Path, lat: np.ndarray, lon: np.ndarray) -> tuple[dict, dict]:
    group = zarr.open_group(str(reference), mode="r", zarr_format=2, use_consolidated=True)
    for name, expected in (("lat", lat), ("lon", lon), ("level", np.asarray(PRESSURE_LEVELS_13))):
        if not np.array_equal(group[name][:], expected):
            raise ValueError(f"Reference coordinate {name} differs from the selected source grid.")
    fields = {}
    provenance = {"path": str(reference.resolve()), "variables": {}}
    for name in STATIC_VARIABLES:
        array = group[name]
        dims = array.attrs.get("_ARRAY_DIMENSIONS")
        if dims != ["lat", "lon"]:
            raise ValueError(f"Reference static variable {name} must have lat/lon dimensions.")
        values = np.asarray(array[:], dtype=np.float32)
        if values.shape != (len(lat), len(lon)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid reference static variable {name}")
        attrs = dict(array.attrs)
        fields[name] = (values, attrs)
        provenance["variables"][name] = {
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(), "attrs": attrs,
        }
    return fields, provenance


def _create_array(group, name: str, shape: tuple, chunks: tuple, dims: list[str], attrs: dict, dtype="float32"):
    array = group.create_array(
        name, shape=shape, chunks=chunks, dtype=dtype,
        compressor=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE),
        fill_value=np.nan if np.dtype(dtype).kind == "f" and name not in {"lat", "lon"} else None,
    )
    array.attrs.update({**attrs, "_ARRAY_DIMENSIONS": dims})
    return array


def _initialize_store(partial: Path, source: ChunkSource, times: pd.DatetimeIndex,
                      lat: np.ndarray, lon: np.ndarray, static_fields: dict, config: dict) -> None:
    group = zarr.open_group(str(partial), mode="w", zarr_format=2)
    group.attrs.update({
        "source": source.uri, "stage_format_version": FORMAT_VERSION,
        "stage_config_sha256": _json_hash(config),
        "spatial_selection": "Point subsampling every fourth 0.25-degree coordinate; no averaging.",
        "precipitation_convention": "Sum six hourly ERA5 depths at t-5h through t, in metres.",
        "solar_forcing_convention": "TISR omitted; generated downstream with GraphCast's one-hour integration.",
        "static_reference_store": config["static_reference"]["path"],
    })
    coords = {
        "lat": (lat, {"long_name": "latitude", "units": "degrees_north"}),
        "lon": (lon, {"long_name": "longitude", "units": "degrees_east"}),
        "level": (np.asarray(PRESSURE_LEVELS_13, dtype=np.int64), {"units": "hPa"}),
        "time": (times.to_numpy().astype("datetime64[h]").astype(np.int64), {
            "units": "hours since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian",
        }),
    }
    for name, (values, attrs) in coords.items():
        array = _create_array(group, name, values.shape, values.shape, [name], attrs, dtype=values.dtype)
        array[:] = values
    for name in TIME_VARIABLES:
        pressure = name in PRESSURE_VARIABLES
        source_name = "total_precipitation" if name == "total_precipitation_6hr" else name
        attrs = source.attrs(source_name)
        if name == "total_precipitation_6hr":
            attrs["cell_methods"] = "time: sum (interval: 1 hour; accumulation: 6 hours ending at time)"
        dims = ["time"] + (["level"] if pressure else []) + ["lat", "lon"]
        shape = (len(times),) + ((13,) if pressure else ()) + (len(lat), len(lon))
        _create_array(group, name, shape, (1,) + shape[1:], dims, attrs)
    for name, (values, attrs) in static_fields.items():
        _create_array(group, name, values.shape, values.shape, ["lat", "lon"], attrs)[:] = values
    zarr.consolidate_metadata(str(partial))
    _write_json_atomic(partial / "source_metadata.json", source.metadata)


def _verify_step(group, index: int) -> str:
    digest = hashlib.sha256()
    for name in TIME_VARIABLES:
        values = np.asarray(group[name][index])
        grid_shape = (group["lat"].shape[0], group["lon"].shape[0])
        expected_shape = (13,) + grid_shape if name in PRESSURE_VARIABLES else grid_shape
        if values.shape != expected_shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Staged variable {name}[{index}] has invalid shape/dtype/nonfinite values.")
        _array_digest(digest, name, values)
    return digest.hexdigest()


def _verify_store_layout(group, times: pd.DatetimeIndex, config: dict, static_fields: dict) -> None:
    if group.attrs.get("stage_config_sha256") != _json_hash(config):
        raise ValueError("Partial store configuration fingerprint differs from the requested run.")
    expected_names = set(TIME_VARIABLES + STATIC_VARIABLES + ("lat", "lon", "level", "time"))
    if set(group.array_keys()) != expected_names:
        raise ValueError("Partial store variable set is incomplete or unexpected.")
    for name in ("lat", "lon"):
        actual = np.asarray(group[name][:], dtype=np.float32)
        if hashlib.sha256(actual.tobytes()).hexdigest() != config[f"{name}_sha256"]:
            raise ValueError(f"Partial coordinate {name} differs from the requested run.")
    for name, expected in (
        ("level", np.asarray(PRESSURE_LEVELS_13, dtype=np.int64)),
        ("time", times.to_numpy().astype("datetime64[h]").astype(np.int64)),
    ):
        if not np.array_equal(group[name][:], expected):
            raise ValueError(f"Partial coordinate {name} differs from the requested run.")
    for name in TIME_VARIABLES:
        grid_shape = (group["lat"].shape[0], group["lon"].shape[0])
        expected_shape = (len(times),) + ((13,) if name in PRESSURE_VARIABLES else ()) + grid_shape
        if group[name].shape != expected_shape or group[name].chunks != (1,) + expected_shape[1:]:
            raise ValueError(f"Partial array {name} has unexpected shape/chunk layout.")
    for name, (expected, _) in static_fields.items():
        if not np.array_equal(group[name][:], expected):
            raise ValueError(f"Partial static variable {name} differs from the reference store.")


def _download_step(source: ChunkSource, group, index: int, hour: int, level_indices: np.ndarray) -> dict:
    started = time.monotonic()
    digest = hashlib.sha256()
    for name in TIME_VARIABLES:
        if name == "total_precipitation_6hr":
            hourly = [source.field("total_precipitation", h, level_indices) for h in range(hour - 5, hour + 1)]
            values = sum_hourly_precipitation(np.stack(hourly))
        else:
            values = source.field(name, hour, level_indices)
        if not np.isfinite(values).all():
            raise ValueError(f"Source variable {name} at hourly index {hour} contains nonfinite values.")
        _array_digest(digest, name, values)
        group[name][index] = values
    expected_digest = digest.hexdigest()
    if _verify_step(group, index) != expected_digest:
        raise ValueError(f"Readback checksum mismatch for staged index {index}")
    return {"sha256": expected_digest, "elapsed_seconds": time.monotonic() - started, "completed_at": _utc_now()}


def stage_arco_res1(*, uri: str = DEFAULT_URI, output: Path = DEFAULT_OUTPUT,
                    start_time: str = DEFAULT_START, end_time: str = DEFAULT_END,
                    workers: int = 4, resume: bool = False, max_steps: int | None = None,
                    reference_store: Path = DEFAULT_REFERENCE, source_metadata: Path | None = None) -> dict:
    if workers <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("workers and max_steps must be positive")
    times = expected_time_grid(start_time, end_time)
    output = Path(output).resolve()
    partial = output.with_name(f".{output.name}.partial")
    progress_path = output.with_name(f"{output.name}.stage_progress.json")
    report_path = output.with_name(f"{output.name}.stage_report.json")
    lock_path = output.with_name(f"{output.name}.lock")
    with _writer_lock(lock_path):
        if output.exists():
            raise FileExistsError(f"Completed output already exists: {output}")
        if (partial.exists() or progress_path.exists()) and not resume:
            raise FileExistsError(f"Partial staging exists; use --resume: {partial}")
        source = ChunkSource(uri, source_metadata)
        lat, lon, level_indices = source.validate_layout()
        hours = source.hourly_indices(times)
        static_fields, static_provenance = _reference_static(Path(reference_store), lat, lon)
        config = {
            "format_version": FORMAT_VERSION, "source_uri": source.uri,
            "source_structure_sha256": source.fingerprint,
            "start_time": times[0].isoformat(), "end_time": times[-1].isoformat(),
            "time_count": len(times), "step_hours": 6,
            "pressure_levels": PRESSURE_LEVELS_13, "static_reference": static_provenance,
            "lat_sha256": hashlib.sha256(lat.tobytes()).hexdigest(),
            "lon_sha256": hashlib.sha256(lon.tobytes()).hexdigest(),
            "source_hour_indices_sha256": hashlib.sha256(hours.tobytes()).hexdigest(),
            "precipitation_hours": "t-5h through t inclusive; sum hourly depths",
            "tisr": "omitted; downstream GraphCast one-hour integration",
        }
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("config") != config:
                raise ValueError("Resume configuration/source/static provenance differs from the original run.")
            if not partial.exists():
                raise FileNotFoundError(f"Progress exists but partial store is missing: {partial}")
        else:
            if partial.exists():
                # A crash during initialization precedes any completion records.
                raise RuntimeError(f"Untracked partial store exists: {partial}; inspect before removal.")
            _initialize_store(partial, source, times, lat, lon, static_fields, config)
            progress = {
                "config": config, "created_at": _utc_now(), "updated_at": _utc_now(),
                "status": "partial", "completed": {}, "sessions": [],
            }
            _write_json_atomic(progress_path, progress)
        group = zarr.open_group(str(partial), mode="r+", zarr_format=2, use_consolidated=True)
        _verify_store_layout(group, times, config, static_fields)
        completed = progress["completed"]
        for key, record in completed.items():
            index = int(key)
            if index < 0 or index >= len(times) or key != str(index):
                raise ValueError(f"Invalid completed index {key}")
            if _verify_step(group, index) != record["sha256"]:
                raise ValueError(f"Previously completed timestamp {times[index]} failed checksum validation.")
        remaining = [index for index in range(len(times)) if str(index) not in completed]
        if max_steps is not None:
            remaining = remaining[:max_steps]
        print(f"staging source={source.uri} output={output} partial={partial}", flush=True)
        print(f"window={times[0]}..{times[-1]} steps={len(times)} completed={len(completed)} scheduled={len(remaining)} workers={workers}", flush=True)
        session = {
            "started_at": _utc_now(), "workers": workers, "scheduled_steps": len(remaining),
            "host": socket.gethostname(), "pid": os.getpid(),
        }
        progress["sessions"].append(session)
        session_started = time.monotonic()
        before_count = len(completed)

        def update_counters() -> None:
            session.update({
                "elapsed_seconds": time.monotonic() - session_started,
                "completed_steps": len(completed) - before_count,
                "downloaded_bytes": source.downloaded_bytes, "request_count": source.request_count,
                "request_seconds": source.request_seconds,
            })
            progress["updated_at"] = _utc_now()
            progress["completed_count"] = len(completed)
            for metric in ("downloaded_bytes", "request_count", "request_seconds"):
                progress[metric] = sum(item.get(metric, 0) for item in progress["sessions"])

        update_counters()
        _write_json_atomic(progress_path, progress)
        pending = {}
        iterator = iter(remaining)
        failure = None
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arco-stage") as executor:
                for _ in range(min(workers, len(remaining))):
                    index = next(iterator)
                    pending[executor.submit(_download_step, source, group, index, int(hours[index]), level_indices)] = index
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        index = pending.pop(future)
                        try:
                            record = future.result()
                        except BaseException as exc:
                            failure = exc
                            print(f"ERROR timestamp={times[index]} error={exc}", flush=True)
                            continue
                        record["timestamp"] = times[index].isoformat()
                        completed[str(index)] = record
                        update_counters()
                        _write_json_atomic(progress_path, progress)
                        print(f"completed={len(completed)}/{len(times)} timestamp={times[index]} seconds={record['elapsed_seconds']:.2f}", flush=True)
                        if failure is None:
                            following = next(iterator, None)
                            if following is not None:
                                pending[executor.submit(_download_step, source, group, following, int(hours[following]), level_indices)] = following
        finally:
            session["ended_at"] = _utc_now()
            update_counters()
            _write_json_atomic(progress_path, progress)
        if failure is not None:
            raise RuntimeError("Staging failed; completed timestamps are retained. Retry with --resume.") from failure
        if len(completed) == len(times):
            print("Verifying complete local store and checksums before publication...", flush=True)
            _verify_store_layout(group, times, config, static_fields)
            for index in range(len(times)):
                if _verify_step(group, index) != completed[str(index)]["sha256"]:
                    raise ValueError(f"Final checksum mismatch for timestamp {times[index]}")
            zarr.consolidate_metadata(str(partial))
            progress["status"] = "complete"
            progress["verified_at"] = _utc_now()
            progress["output"] = str(output)
            _write_json_atomic(partial / "stage_report.json", progress)
            partial.replace(output)
            _write_json_atomic(report_path, progress)
            _write_json_atomic(progress_path, progress)
            print(f"COMPLETE output={output} steps={len(times)} report={report_path}", flush=True)
        else:
            print(f"PARTIAL completed={len(completed)}/{len(times)}; use --resume to continue", flush=True)
        return progress


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-store", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--source-metadata", type=Path, help="Optional source metadata snapshot to check against live layout.")
    parser.add_argument("--start-time", default=DEFAULT_START)
    parser.add_argument("--end-time", default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Download at most this many additional timestamps; retain partial output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stage_arco_res1(**vars(args))


if __name__ == "__main__":
    main()
