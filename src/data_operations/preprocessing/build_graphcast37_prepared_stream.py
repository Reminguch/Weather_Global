#!/usr/bin/env python3
"""Build a resumable, year-sharded GraphCast37 prepared-array store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import dask

from src.data_operations.loaders.graphcast_dataset import open_graphcast_era5
from src.data_operations.staging.stage_wb2_graphcast37_window import (
    DEFAULT_URI,
    PRESSURE_LEVELS_37,
    report_path_for,
    stage_graphcast37_window,
    validate_graphcast37_layout,
)
from src.models.graphcast.training.core.dataset import _ensure_datetime_coord, prepare_dataset_for_task
from src.models.graphcast.training.core.model import load_graphcast_checkpoint
from src.models.graphcast.training.core.prepared_array import (
    PREPARED_ARRAY_SHARDED_FORMAT_VERSION,
    PreparedArrayStore,
)


DEFAULT_CHECKPOINT = Path(
    "data/graphcast/graphcast/params/"
    "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - "
    "mesh 2to6 - precipitation input and output.npz"
)
DEFAULT_OUT_ROOT = Path("data/graphcast/graphcast/dataset/prepared_stream_graphcast37")
DEFAULT_TEMP_ROOT = Path("data/graphcast/graphcast/dataset/.tmp_graphcast37_staging")
RESOLUTION = 0.25
RESOLUTION_TAG = "res0p25"
EXPECTED_STEP = pd.Timedelta(hours=6)
BUILDER_VERSION = 2


@dataclass(frozen=True)
class MonthWindow:
    key: str
    year: int
    start: pd.Timestamp
    end: pd.Timestamp
    global_start: int
    global_stop: int
    shard_start: int

    @property
    def count(self) -> int:
        return self.global_stop - self.global_start

    @property
    def local_start(self) -> int:
        return self.global_start - self.shard_start


class BuildLock:
    def __init__(self, path: Path, *, break_lock: bool = False) -> None:
        self.path = path
        self.break_lock = break_lock
        self.fd: int | None = None

    def _remove_stale_lock(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            payload = {}
        same_host = payload.get("host") == socket.gethostname()
        pid = int(payload.get("pid", -1))
        local_process_alive = same_host and pid > 0 and Path(f"/proc/{pid}").exists()
        if local_process_alive:
            raise RuntimeError(f"Build lock is held by live pid {pid} on {socket.gethostname()}: {self.path}")
        if not same_host and not self.break_lock:
            raise RuntimeError(
                f"Build lock was created on {payload.get('host', 'an unknown host')}: {self.path}. "
                "Use --break-lock only after confirming no build is active."
            )
        self.path.unlink()

    def __enter__(self) -> "BuildLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_stale_lock()
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Build lock already exists: {self.path}") from exc
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": _utc_now(),
        }
        os.write(self.fd, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(values))
    tmp.replace(path)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _size_to_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT])iB", value)
    if match is None:
        raise ValueError(f"Unsupported quota size: {value}")
    scale = {"K": 1, "M": 2, "G": 3, "T": 4}[match.group(2)]
    return int(float(match.group(1)) * (1024**scale))


def check_available_space(path: Path, min_free_tib: float) -> dict[str, object]:
    minimum = int(float(min_free_tib) * 1024**4)
    disk_free = int(shutil.disk_usage(path).free)
    if disk_free < minimum:
        raise RuntimeError(
            f"Filesystem free space is {disk_free / 1024**4:.2f} TiB; "
            f"at least {min_free_tib:.2f} TiB is required."
        )

    quota = subprocess.run(["checkquota"], check=True, capture_output=True, text=True)
    project_free = None
    project_line = None
    for line in quota.stdout.splitlines():
        if "Della scratch GPFS fileset" not in line:
            continue
        sizes = re.findall(r"[0-9]+(?:\.[0-9]+)?[KMGT]iB", line)
        if len(sizes) >= 3:
            project_free = _size_to_bytes(sizes[-1]) - _size_to_bytes(sizes[0])
            project_line = line.strip()
            break
    if project_free is None:
        raise RuntimeError("Could not parse the Della scratch fileset row from checkquota output.")
    if project_free < minimum:
        raise RuntimeError(
            f"Project quota headroom is {project_free / 1024**4:.2f} TiB; "
            f"at least {min_free_tib:.2f} TiB is required."
        )
    return {
        "checked_at": _utc_now(),
        "minimum_free_tib": float(min_free_tib),
        "filesystem_free_bytes": disk_free,
        "project_free_bytes": project_free,
        "project_quota_row": project_line,
    }


def _expected_time_index(args: argparse.Namespace) -> pd.DatetimeIndex:
    if args.start_time is not None or args.end_time is not None:
        if args.start_time is None or args.end_time is None:
            raise ValueError("Provide both --start-time and --end-time, or neither.")
        start = pd.Timestamp(args.start_time)
        end = pd.Timestamp(args.end_time)
    else:
        if args.start_year > args.end_year:
            raise ValueError("--start-year must be <= --end-year")
        start = pd.Timestamp(year=args.start_year, month=1, day=1, hour=0)
        end = pd.Timestamp(year=args.end_year, month=12, day=31, hour=18)
    if end < start:
        raise ValueError("Build end time must be on or after start time.")
    if start.minute or start.second or end.minute or end.second or start.hour % 6 or end.hour % 6:
        raise ValueError("Build bounds must lie on the 00/06/12/18 UTC grid.")
    return pd.date_range(start, end, freq="6h")


def _time_shards(times: pd.DatetimeIndex) -> list[dict[str, object]]:
    shards = []
    years = np.asarray(times.year, dtype=np.int64)
    for year in sorted(set(years.tolist())):
        positions = np.flatnonzero(years == year)
        start = int(positions[0])
        stop = int(positions[-1]) + 1
        shards.append(
            {
                "year": int(year),
                "path": f"years/{year}",
                "start": start,
                "stop": stop,
                "time_start": str(times[start].to_datetime64()),
                "time_end": str(times[stop - 1].to_datetime64()),
            }
        )
    return shards


def _month_windows(times: pd.DatetimeIndex, shards: list[dict[str, object]]) -> list[MonthWindow]:
    shard_starts = {int(shard["year"]): int(shard["start"]) for shard in shards}
    periods = times.to_period("M")
    windows = []
    for period in periods.unique():
        positions = np.flatnonzero(periods == period)
        windows.append(
            MonthWindow(
                key=str(period),
                year=int(period.year),
                start=pd.Timestamp(times[int(positions[0])]),
                end=pd.Timestamp(times[int(positions[-1])]),
                global_start=int(positions[0]),
                global_stop=int(positions[-1]) + 1,
                shard_start=shard_starts[int(period.year)],
            )
        )
    return windows


def _drop_singleton_batch(var: xr.DataArray) -> xr.DataArray:
    if "batch" not in var.dims:
        return var
    if var.sizes["batch"] != 1:
        raise ValueError(f"Expected singleton batch for {var.name}, got {var.sizes['batch']}.")
    return var.isel(batch=0, drop=True)


def _task_vars(task_cfg) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            list(task_cfg.input_variables)
            + list(task_cfg.target_variables)
            + list(task_cfg.forcing_variables)
        )
    )


def _variable_for_task(prepared: xr.Dataset, name: str, task_cfg) -> xr.DataArray:
    var = _drop_singleton_batch(prepared[name])
    if "level" in var.dims:
        var = var.sel(level=list(task_cfg.pressure_levels))
    if var.dtype.kind == "f" and var.dtype != np.float32:
        var = var.astype(np.float32)
    return var


def _finite_time_mask(values: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
    axis = dims.index("time")
    moved = np.moveaxis(values, axis, 0)
    return np.isfinite(moved.reshape(moved.shape[0], -1)).all(axis=1)


class PreparedStoreV2Writer:
    def __init__(
        self,
        *,
        root: Path,
        expected_times: pd.DatetimeIndex,
        task_cfg,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        source_uri: str,
        resume: bool,
        extend_existing: bool = False,
    ) -> None:
        self.root = root
        self.expected_times = expected_times
        self.task_cfg = task_cfg
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_uri = source_uri
        self.extend_existing = extend_existing
        self.shards = _time_shards(expected_times)
        self.metadata_path = root / "metadata.json"
        self.incomplete_path = root / ".incomplete"
        self.root.mkdir(parents=True, exist_ok=True)
        existing = [
            path
            for path in self.root.iterdir()
            if path.name not in {".incomplete", "build_manifest.json"}
        ]
        if existing and not resume:
            raise FileExistsError(f"Prepared store already exists: {self.root}. Use --resume.")
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text())
            self._validate_existing_metadata()
        else:
            self.metadata = None
            if existing:
                raise RuntimeError(f"Prepared store has files but no metadata: {self.root}")
        self.incomplete_path.touch(exist_ok=True)

    def _validate_existing_metadata(self) -> None:
        metadata = self.metadata
        if int(metadata.get("prepared_array_format_version", -1)) != PREPARED_ARRAY_SHARDED_FORMAT_VERSION:
            raise ValueError("Existing prepared store is not format v2.")
        if metadata.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise ValueError("Existing prepared store was created for a different checkpoint.")
        if metadata.get("source_data_path") != self.source_uri:
            raise ValueError("Existing prepared store was created from a different source dataset.")
        stored_times = np.asarray(np.load(self.root / "coords" / "time.npy", mmap_mode="r"))
        expected = self.expected_times.values.astype("datetime64[ns]")
        stored_shards = metadata.get("time_shards")
        if np.array_equal(stored_times, expected) and stored_shards == self.shards:
            return
        if not self.extend_existing:
            raise ValueError(
                "Existing prepared store time coordinate does not match this build. "
                "Use --extend-existing only when the requested range is a strict year-sharded superset."
            )

        stored_index = pd.DatetimeIndex(pd.to_datetime(stored_times))
        requested_index = pd.DatetimeIndex(pd.to_datetime(expected))
        positions = requested_index.get_indexer(stored_index)
        if (
            len(stored_index) == 0
            or np.any(positions < 0)
            or not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions)))
            or len(requested_index) <= len(stored_index)
        ):
            raise ValueError(
                "Existing prepared store is not a contiguous strict subset of the requested extension."
            )
        if not isinstance(stored_shards, list) or not stored_shards:
            raise ValueError("Existing prepared store has no year-shard metadata to extend.")

        requested_by_year = {int(shard["year"]): shard for shard in self.shards}
        for stored_shard in stored_shards:
            year = int(stored_shard["year"])
            requested_shard = requested_by_year.get(year)
            identity_keys = ("year", "path", "time_start", "time_end")
            stored_count = int(stored_shard["stop"]) - int(stored_shard["start"])
            requested_count = (
                -1
                if requested_shard is None
                else int(requested_shard["stop"]) - int(requested_shard["start"])
            )
            if (
                requested_shard is None
                or any(stored_shard.get(key) != requested_shard.get(key) for key in identity_keys)
                or stored_count != requested_count
            ):
                raise ValueError(
                    f"Existing year shard {year} would change boundaries during extension; "
                    "only whole, unchanged annual shards may be preserved."
                )

        # Validate every preserved shard before publishing broader global metadata.
        for stored_shard in stored_shards:
            year = int(stored_shard["year"])
            year_count = int(stored_shard["stop"]) - int(stored_shard["start"])
            validity_path = self.root / str(stored_shard["path"]) / "validity.json"
            if not validity_path.exists():
                raise FileNotFoundError(f"Existing year {year} has no validity record: {validity_path}")
            validity = json.loads(validity_path.read_text())
            if len(validity.get("written_time_mask", [])) != year_count or not all(
                validity["written_time_mask"]
            ):
                raise ValueError(f"Existing year {year} is not fully written and cannot be extended.")
            for name, info in metadata.get("variables", {}).items():
                if "time" not in info["dims"]:
                    continue
                path = self.root / str(stored_shard["path"]) / "vars" / f"{name}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Existing year {year} is missing prepared variable: {path}")
                array = np.load(path, mmap_mode="r")
                expected_shape = (year_count, *tuple(int(value) for value in info["shape"])[1:])
                if tuple(array.shape) != expected_shape or np.dtype(array.dtype) != np.dtype(info["dtype"]):
                    raise ValueError(f"Existing year array is incompatible with extension: {path}")

        old_time_start = metadata.get("time_start")
        old_time_end = metadata.get("time_end")
        old_time_count = len(stored_index)
        self.incomplete_path.touch(exist_ok=True)
        _write_npy_atomic(self.root / "coords" / "time.npy", expected)
        metadata["time_shards"] = self.shards
        metadata["time_start"] = str(self.expected_times[0].to_datetime64())
        metadata["time_end"] = str(self.expected_times[-1].to_datetime64())
        metadata["builder_version"] = BUILDER_VERSION
        for info in metadata.get("variables", {}).values():
            if "time" in info["dims"]:
                info["shape"][0] = len(self.expected_times)
        metadata.setdefault("extensions", []).append(
            {
                "extended_at": _utc_now(),
                "previous_time_start": old_time_start,
                "previous_time_end": old_time_end,
                "previous_time_count": old_time_count,
                "requested_time_start": metadata["time_start"],
                "requested_time_end": metadata["time_end"],
                "requested_time_count": len(self.expected_times),
            }
        )
        _write_json_atomic(self.metadata_path, metadata)

    def _initialize_from_window(self, prepared: xr.Dataset) -> None:
        missing = [name for name in _task_vars(self.task_cfg) if name not in prepared.data_vars]
        if missing:
            raise ValueError(f"Prepared window is missing checkpoint variables: {missing}")
        coords_dir = self.root / "coords"
        vars_dir = self.root / "vars"
        coords_dir.mkdir(parents=True, exist_ok=True)
        vars_dir.mkdir(parents=True, exist_ok=True)
        _write_npy_atomic(coords_dir / "time.npy", self.expected_times.values.astype("datetime64[ns]"))
        for name in ("lat", "lon", "level"):
            if name not in prepared.coords:
                continue
            values = (
                np.asarray(self.task_cfg.pressure_levels)
                if name == "level"
                else np.asarray(prepared.coords[name].values)
            )
            _write_npy_atomic(coords_dir / f"{name}.npy", values)

        variables = {}
        for name in _task_vars(self.task_cfg):
            var = _variable_for_task(prepared, name, self.task_cfg)
            dims = tuple(var.dims)
            shape = list(var.shape)
            if "time" in dims:
                if dims.index("time") != 0:
                    raise ValueError(f"Format-v2 time variable must use time axis 0: {name} dims={dims}")
                shape[0] = len(self.expected_times)
            else:
                values = np.asarray(var.values)
                if not np.isfinite(values).all():
                    raise ValueError(f"Static variable contains non-finite values: {name}")
                _write_npy_atomic(vars_dir / f"{name}.npy", values)
            variables[name] = {"dims": list(dims), "shape": shape, "dtype": str(np.dtype(var.dtype))}

        self.metadata = {
            "prepared_array_format_version": PREPARED_ARRAY_SHARDED_FORMAT_VERSION,
            "builder_version": BUILDER_VERSION,
            "source_data_path": self.source_uri,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "resolution": RESOLUTION,
            "base_resolution": RESOLUTION,
            "pressure_levels": [int(level) for level in self.task_cfg.pressure_levels],
            "task_input_variables": list(self.task_cfg.input_variables),
            "task_target_variables": list(self.task_cfg.target_variables),
            "task_forcing_variables": list(self.task_cfg.forcing_variables),
            "time_shards": self.shards,
            "variables": variables,
            "time_start": str(self.expected_times[0].to_datetime64()),
            "time_end": str(self.expected_times[-1].to_datetime64()),
            "created_at": _utc_now(),
        }
        _write_json_atomic(self.metadata_path, self.metadata)

    def _shard_info(self, year: int) -> dict[str, object]:
        matches = [shard for shard in self.shards if int(shard["year"]) == int(year)]
        if len(matches) != 1:
            raise ValueError(f"Expected one time shard for year {year}, found {len(matches)}")
        return matches[0]

    def _open_year_var(self, year: int, name: str) -> np.memmap:
        assert self.metadata is not None
        info = self.metadata["variables"][name]
        shard = self._shard_info(year)
        year_count = int(shard["stop"]) - int(shard["start"])
        shape = (year_count, *tuple(int(value) for value in info["shape"])[1:])
        path = self.root / str(shard["path"]) / "vars" / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(info["dtype"]), shape=shape)
        array = np.lib.format.open_memmap(path, mode="r+")
        if tuple(array.shape) != shape or np.dtype(array.dtype) != np.dtype(info["dtype"]):
            raise ValueError(f"Existing year array has incompatible shape/dtype: {path}")
        return array

    def _validity_path(self, year: int) -> Path:
        return self.root / str(self._shard_info(year)["path"]) / "validity.json"

    def _load_validity(self, year: int) -> dict:
        path = self._validity_path(year)
        if path.exists():
            return json.loads(path.read_text())
        shard = self._shard_info(year)
        count = int(shard["stop"]) - int(shard["start"])
        time_vars = [
            name for name, info in self.metadata["variables"].items() if "time" in info["dims"]
        ]
        return {
            "written_time_mask": [False] * count,
            "time_finite_by_variable": {name: [False] * count for name in time_vars},
        }

    def write_window(self, prepared: xr.Dataset, window: MonthWindow, *, chunk_time: int) -> None:
        if self.metadata is None:
            self._initialize_from_window(prepared)
        assert self.metadata is not None
        actual_times = pd.DatetimeIndex(pd.to_datetime(prepared.time.values))
        expected = self.expected_times[window.global_start : window.global_stop]
        if not actual_times.equals(expected):
            raise ValueError(
                f"Prepared window time mismatch for {window.key}: expected {expected[0]}..{expected[-1]}, "
                f"found {actual_times[0]}..{actual_times[-1]}."
            )

        for name, info in self.metadata["variables"].items():
            if "time" in info["dims"]:
                continue
            expected_values = np.load(self.root / "vars" / f"{name}.npy", mmap_mode="r")
            actual_values = np.asarray(_variable_for_task(prepared, name, self.task_cfg).values)
            np.testing.assert_allclose(actual_values, expected_values, rtol=1e-6, atol=1e-6, equal_nan=True)

        validity = self._load_validity(window.year)
        local_start = window.local_start
        local_stop = local_start + window.count
        for name, info in self.metadata["variables"].items():
            if "time" not in info["dims"]:
                continue
            var = _variable_for_task(prepared, name, self.task_cfg)
            if tuple(var.dims) != tuple(info["dims"]):
                raise ValueError(f"Variable dims changed for {name}: {var.dims} vs {info['dims']}")
            destination = self._open_year_var(window.year, name)
            finite = np.empty(window.count, dtype=bool)
            for offset in range(0, window.count, chunk_time):
                stop = min(window.count, offset + chunk_time)
                values = np.asarray(var.isel(time=slice(offset, stop)).values)
                if values.dtype.kind == "f" and values.dtype != np.float32:
                    values = values.astype(np.float32)
                destination[local_start + offset : local_start + stop] = values
                finite[offset:stop] = _finite_time_mask(values, tuple(var.dims))
            destination.flush()
            samples = sorted(set((0, window.count // 2, window.count - 1)))
            for sample in samples:
                source_value = np.asarray(var.isel(time=sample).values)
                stored_value = np.asarray(destination[local_start + sample])
                np.testing.assert_allclose(source_value, stored_value, rtol=1e-6, atol=1e-6, equal_nan=True)
            validity["time_finite_by_variable"][name][local_start:local_stop] = finite.tolist()
            del destination

        validity["written_time_mask"][local_start:local_stop] = [True] * window.count
        year_coords = self.root / str(self._shard_info(window.year)["path"]) / "coords"
        year_coords.mkdir(parents=True, exist_ok=True)
        year_times = self.expected_times[
            int(self._shard_info(window.year)["start"]) : int(self._shard_info(window.year)["stop"])
        ]
        time_path = year_coords / "time.npy"
        if not time_path.exists():
            _write_npy_atomic(time_path, year_times.values.astype("datetime64[ns]"))
        _write_json_atomic(self._validity_path(window.year), validity)

    def finalize(self) -> PreparedArrayStore:
        if self.metadata is None:
            raise RuntimeError("Cannot finalize an empty prepared store.")
        for shard in self.shards:
            year = int(shard["year"])
            validity = self._load_validity(year)
            if not all(validity["written_time_mask"]):
                raise RuntimeError(f"Year {year} contains unwritten time positions.")
            nonfinite = [
                name
                for name, mask in validity["time_finite_by_variable"].items()
                if not all(mask)
            ]
            if nonfinite:
                raise RuntimeError(f"Year {year} contains non-finite data: {nonfinite}")
        self.metadata["finalized_at"] = _utc_now()
        _write_json_atomic(self.metadata_path, self.metadata)
        self.incomplete_path.unlink(missing_ok=True)
        try:
            store = PreparedArrayStore(self.root, label="graphcast37-res0p25")
            store.validate(resolution=RESOLUTION, task_cfg=self.task_cfg)
            input_steps = int(pd.Timedelta(self.task_cfg.input_duration) / EXPECTED_STEP)
            years = sorted(set(self.expected_times.year.tolist()))
            for year in years:
                positions = np.flatnonzero(np.asarray(self.expected_times.year) == year)
                if positions.size <= input_steps:
                    continue
                store.build_batch_from_indices(
                    indices=[int(positions[0]) + input_steps - 1],
                    input_steps=input_steps,
                    target_steps=1,
                    task_cfg=self.task_cfg,
                    dt=EXPECTED_STEP,
                )
        except BaseException:
            self.incomplete_path.touch()
            raise
        return store


def _new_manifest(
    *,
    args: argparse.Namespace,
    times: pd.DatetimeIndex,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    windows: list[MonthWindow],
    space_report: dict[str, object] | None,
) -> dict:
    return {
        "builder_version": BUILDER_VERSION,
        "status": "incomplete",
        "source_uri": args.source_uri,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "out_root": str(args.out_root),
        "temp_root": str(args.temp_root),
        "time_start": str(times[0].to_datetime64()),
        "time_end": str(times[-1].to_datetime64()),
        "expected_time_count": len(times),
        "chunk_time": args.chunk_time,
        "staging_chunk_time": args.staging_chunk_time,
        "delete_staged": args.delete_staged,
        "created_at": _utc_now(),
        "space_report": space_report,
        "months": {
            window.key: {
                "status": "pending",
                "start": str(window.start.to_datetime64()),
                "end": str(window.end.to_datetime64()),
                "time_count": window.count,
            }
            for window in windows
        },
    }


def _validate_manifest(
    manifest: dict,
    *,
    args: argparse.Namespace,
    times: pd.DatetimeIndex,
    sha256: str,
    windows: list[MonthWindow],
    space_report: dict[str, object] | None,
) -> bool:
    identity = {"source_uri": args.source_uri, "checkpoint_sha256": sha256}
    identity_mismatches = {
        key: (manifest.get(key), value)
        for key, value in identity.items()
        if manifest.get(key) != value
    }
    if identity_mismatches:
        raise ValueError(f"Existing build manifest does not match requested build: {identity_mismatches}")

    requested_start = str(times[0].to_datetime64())
    requested_end = str(times[-1].to_datetime64())
    same_range = (
        manifest.get("time_start") == requested_start
        and manifest.get("time_end") == requested_end
        and manifest.get("expected_time_count") == len(times)
    )
    if same_range:
        return False
    if not getattr(args, "extend_existing", False):
        raise ValueError(
            "Existing build manifest has a different time range. "
            "Use --extend-existing only to add whole years around the completed range."
        )

    old_start = pd.Timestamp(manifest["time_start"])
    old_end = pd.Timestamp(manifest["time_end"])
    old_times = pd.date_range(old_start, old_end, freq="6h")
    positions = times.get_indexer(old_times)
    if (
        len(old_times) == 0
        or manifest.get("expected_time_count") != len(old_times)
        or np.any(positions < 0)
        or not np.array_equal(positions, np.arange(positions[0], positions[0] + len(positions)))
        or len(times) <= len(old_times)
    ):
        raise ValueError("Existing build range is not a contiguous strict subset of the requested extension.")

    old_months = manifest.get("months", {})
    if not old_months or any(state.get("status") != "complete" for state in old_months.values()):
        raise ValueError("Only a fully completed build may be extended to additional years.")
    requested_windows = {window.key: window for window in windows}
    if any(key not in requested_windows for key in old_months):
        raise ValueError("Requested extension does not preserve every completed month.")

    merged_months = {}
    for window in windows:
        expected_state = {
            "status": "pending",
            "start": str(window.start.to_datetime64()),
            "end": str(window.end.to_datetime64()),
            "time_count": window.count,
        }
        if window.key in old_months:
            state = old_months[window.key]
            for key in ("start", "end", "time_count"):
                if state.get(key) != expected_state[key]:
                    raise ValueError(f"Completed month {window.key} changes boundaries during extension.")
            merged_months[window.key] = state
        else:
            merged_months[window.key] = expected_state

    manifest.setdefault("extensions", []).append(
        {
            "extended_at": _utc_now(),
            "previous_time_start": manifest["time_start"],
            "previous_time_end": manifest["time_end"],
            "previous_time_count": manifest["expected_time_count"],
            "requested_time_start": requested_start,
            "requested_time_end": requested_end,
            "requested_time_count": len(times),
        }
    )
    manifest.update(
        {
            "status": "incomplete",
            "time_start": requested_start,
            "time_end": requested_end,
            "expected_time_count": len(times),
            "space_report": space_report,
            "months": merged_months,
        }
    )
    return True


def _prepare_staged_dataset(path: Path, task_cfg) -> tuple[xr.Dataset, xr.Dataset]:
    source = open_graphcast_era5(path)
    source = _ensure_datetime_coord(source)
    prepared = prepare_dataset_for_task(source, task_cfg)
    missing = [name for name in _task_vars(task_cfg) if name not in prepared.data_vars]
    if missing:
        source.close()
        raise ValueError(f"Staged dataset is missing checkpoint variables after preparation: {missing}")
    return prepared, source


def _validate_staged(path: Path, window: MonthWindow) -> None:
    staged = xr.open_zarr(path, consolidated=True)
    try:
        validate_graphcast37_layout(staged)
        actual = pd.DatetimeIndex(pd.to_datetime(staged.time.values))
        expected = pd.date_range(window.start, window.end, freq="6h")
        if not actual.equals(expected):
            raise ValueError(f"Staged time coordinate is wrong for {window.key}")
    finally:
        staged.close()


def _remove_staged(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    report_path_for(path).unlink(missing_ok=True)


def remove_staged_if_verified(path: Path, *, verified: bool, delete_staged: bool) -> None:
    if verified and delete_staged:
        _remove_staged(path)


def _run_build_inner(args: argparse.Namespace) -> Path:
    expected_times = _expected_time_index(args)
    shards = _time_shards(expected_times)
    windows = _month_windows(expected_times, shards)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GraphCast37 checkpoint not found: {checkpoint_path}")
    checkpoint = load_graphcast_checkpoint(checkpoint_path)
    task_cfg = checkpoint.task_config
    if not np.isclose(float(checkpoint.model_config.resolution), RESOLUTION):
        raise ValueError(f"Checkpoint resolution is {checkpoint.model_config.resolution}, expected {RESOLUTION}")
    if [int(value) for value in task_cfg.pressure_levels] != PRESSURE_LEVELS_37:
        raise ValueError("Checkpoint does not use the official GraphCast37 pressure levels.")
    checkpoint_sha256 = _checkpoint_sha256(checkpoint_path)

    resolution_root = args.out_root.resolve() / RESOLUTION_TAG
    args.temp_root = args.temp_root.resolve()
    args.out_root = args.out_root.resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)
    space_report = None if args.skip_space_check else check_available_space(args.out_root, args.min_free_tib)
    manifest_path = resolution_root / "build_manifest.json"
    lock_path = resolution_root.parent / f".{resolution_root.name}.build.lock"

    with BuildLock(lock_path, break_lock=args.break_lock):
        if manifest_path.exists():
            if not args.resume:
                raise FileExistsError(f"Build manifest already exists: {manifest_path}. Use --resume.")
            manifest = json.loads(manifest_path.read_text())
            manifest_changed = _validate_manifest(
                manifest,
                args=args,
                times=expected_times,
                sha256=checkpoint_sha256,
                windows=windows,
                space_report=space_report,
            )
        else:
            manifest = _new_manifest(
                args=args,
                times=expected_times,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
                windows=windows,
                space_report=space_report,
            )
            resolution_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(manifest_path, manifest)
            manifest_changed = False

        writer = PreparedStoreV2Writer(
            root=resolution_root,
            expected_times=expected_times,
            task_cfg=task_cfg,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            source_uri=args.source_uri,
            resume=args.resume,
            extend_existing=args.extend_existing,
        )
        if manifest_changed:
            _write_json_atomic(manifest_path, manifest)
        args.temp_root.mkdir(parents=True, exist_ok=True)
        build_started = time.time()
        for window in windows:
            month_state = manifest["months"][window.key]
            staged_path = args.temp_root / f"{window.key}.zarr"
            if month_state["status"] == "complete":
                remove_staged_if_verified(staged_path, verified=True, delete_staged=args.delete_staged)
                print(f"[build-gc37] {window.key}: already complete")
                continue
            month_started = time.time()
            print(f"[build-gc37] {window.key}: staging {window.start} through {window.end}")
            if staged_path.exists():
                _validate_staged(staged_path, window)
            else:
                report = stage_graphcast37_window(
                    uri=args.source_uri,
                    output=staged_path,
                    start_time=str(window.start),
                    end_time=str(window.end),
                    chunk_time=args.staging_chunk_time,
                )
                month_state["stage_report"] = asdict(report)
                month_state["status"] = "staged"
                _write_json_atomic(manifest_path, manifest)
            _validate_staged(staged_path, window)

            prepared, source = _prepare_staged_dataset(staged_path, task_cfg)
            try:
                writer.write_window(prepared, window, chunk_time=args.chunk_time)
            finally:
                prepared.close()
                if source is not prepared:
                    source.close()
            month_state.update(
                {
                    "status": "complete",
                    "completed_at": _utc_now(),
                    "elapsed_sec": time.time() - month_started,
                    "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                }
            )
            _write_json_atomic(manifest_path, manifest)
            remove_staged_if_verified(staged_path, verified=True, delete_staged=args.delete_staged)
            print(f"[build-gc37] {window.key}: complete")

        writer.finalize()
        manifest.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "elapsed_sec_this_run": time.time() - build_started,
                "final_size_bytes": sum(
                    path.stat().st_size for path in resolution_root.rglob("*") if path.is_file()
                ),
                "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            }
        )
        _write_json_atomic(manifest_path, manifest)
    return resolution_root


def run_build(args: argparse.Namespace) -> Path:
    with dask.config.set(scheduler="threads", num_workers=args.max_workers):
        return _run_build_inner(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-uri", default=DEFAULT_URI)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    parser.add_argument("--chunk-time", type=int, default=4)
    parser.add_argument("--staging-chunk-time", type=int, default=16)
    parser.add_argument("--max-workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--extend-existing",
        action="store_true",
        help="Safely add whole annual shards around an existing completed v2 store.",
    )
    parser.add_argument("--delete-staged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-tib", type=float, default=3.0)
    parser.add_argument("--skip-space-check", action="store_true")
    parser.add_argument("--break-lock", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_time <= 0 or args.staging_chunk_time <= 0:
        raise ValueError("Chunk sizes must be positive.")
    if args.min_free_tib <= 0:
        raise ValueError("--min-free-tib must be positive.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = run_build(args)
    print(f"[build-gc37] finalized {output}")


if __name__ == "__main__":
    main()
