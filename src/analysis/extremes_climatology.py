"""Independent, local-calendar quantiles for the V24 extreme-event evaluation.

Each of twelve CPU workers stages one observable/UTC hour once, then computes
exact empirical quantiles in bounded spatial tiles.  The source is never fitted
to evaluation forecasts.  TP24 includes its antecedent intervals before leap-day
targets are removed.  Completion metadata is published only after the array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

OBSERVABLES = ("T2m", "WS10", "TP24")
HOURS = (0, 6, 12, 18)
QUANTILES = (0.02, 0.05, 0.95, 0.98)
MONTH_START = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def calendar_day_indices(times: np.ndarray) -> np.ndarray:
    """Zero-based fixed 365-day calendar; February 29 is represented by -1."""
    days = np.asarray(times).astype("datetime64[D]")
    if np.any(np.isnat(days)):
        raise ValueError("Climatology dates cannot contain NaT")
    month_dates = days.astype("datetime64[M]")
    months = month_dates.astype(np.int64) % 12
    day_in_month = (days - month_dates).astype(np.int64)
    result = MONTH_START[months] + day_in_month
    return np.where((months == 1) & (day_in_month == 28), -1, result).astype(np.int16)


def calendar_day_index(timestamp) -> int:
    day = int(calendar_day_indices(np.asarray([timestamp], dtype="datetime64[ns]"))[0])
    if day < 0:
        raise ValueError("February 29 is outside this 365-day evaluation protocol")
    return day


def circular_window_mask(days: np.ndarray, target: int, radius: int = 30) -> np.ndarray:
    if not 0 <= target < 365 or not 0 <= radius < 183:
        raise ValueError("Invalid target day or circular calendar radius")
    delta = np.abs(np.asarray(days, dtype=np.int64) - target)
    return (np.asarray(days) >= 0) & (np.minimum(delta, 365 - delta) <= radius)


def tp24_mm(intervals: np.ndarray) -> np.ndarray:
    """Four raw six-hour accumulations, ending at the target time, in mm."""
    values = np.asarray(intervals)
    if values.shape[0] != 4 or not np.all(np.isfinite(values)):
        raise ValueError("TP24 requires four finite consecutive TP6 fields")
    return (np.sum(values, axis=0, dtype=np.float64) * 1000.0).astype(np.float32)


def _source_times(group) -> np.ndarray:
    array = group["time"]
    attrs = dict(array.attrs)
    calendar = attrs.get("calendar", "standard")
    if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"Unsupported source calendar: {calendar}")
    unit, separator, origin = attrs["units"].partition(" since ")
    units = {"hours": "h", "days": "D", "minutes": "m", "seconds": "s"}
    if not separator or unit not in units:
        raise ValueError("Unsupported source time encoding")
    raw = np.asarray(array[:])
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.round(raw)):
        raise ValueError("Expected finite integral source time offsets")
    result = np.datetime64(origin, "ns") + raw.astype(f"timedelta64[{units[unit]}]")
    if result.ndim != 1 or np.any(np.diff(result) != np.timedelta64(6, "h")):
        raise ValueError("Source times must be unique and contiguous every six hours")
    return result


def reference_indices(times: np.ndarray, hour: int, start_year: int, end_year: int) -> np.ndarray:
    times = np.asarray(times).astype("datetime64[ns]")
    begin, end = np.datetime64(f"{start_year}-01-01"), np.datetime64(f"{end_year + 1}-01-01")
    hours = times.astype("datetime64[h]").astype(np.int64) % 24
    keep = (times >= begin) & (times < end) & (hours == hour) & (calendar_day_indices(times) >= 0)
    selected = np.flatnonzero(keep)
    if len(selected) != 365 * (end_year - start_year + 1):
        raise ValueError("Historical reference does not contain every non-leap date at this hour")
    return selected


def _field(group, name: str, index: int, shape: tuple[int, int]) -> np.ndarray:
    field = np.asarray(group[name][index, :, :], dtype=np.float32)
    if field.shape != shape or not np.all(np.isfinite(field)):
        raise ValueError(f"Nonfinite or mismatched historical field {name} at index {index}")
    return field


def stage_reference(source: Path, output: Path, observable: str, hour: int,
                    start_year: int = 1991, end_year: int = 2020) -> tuple[Path, dict]:
    import zarr

    group = zarr.open_consolidated(str(source), mode="r")
    times = _source_times(group)
    selected = reference_indices(times, hour, start_year, end_year)
    lat, lon = np.asarray(group["lat"][:]), np.asarray(group["lon"][:])
    if (lat.shape != (181,) or lon.shape != (360,)
            or not np.array_equal(np.sort(lat), np.arange(-90, 91))
            or not np.array_equal(lon, np.arange(360))):
        raise ValueError("Expected the exact native 181x360 one-degree grid")
    expected_units = {"2m_temperature": "K", "10m_u_component_of_wind": "m s**-1",
                      "10m_v_component_of_wind": "m s**-1", "total_precipitation_6hr": "m"}
    for name, unit in expected_units.items():
        if group[name].attrs.get("units") != unit:
            raise ValueError(f"Unexpected physical units for {name}")
    shape = (len(lat), len(lon))
    output.mkdir(parents=True, exist_ok=True)
    for name, array in (("lat", lat), ("lon", lon)):
        coord_path = output / f"{name}.npy"
        if coord_path.exists() and not np.array_equal(np.load(coord_path), array):
            raise ValueError("Existing climatology grid differs from source")
        _atomic_npy(coord_path, array)
    stem = f"{observable}_h{hour:02d}"
    stage = output / f"{stem}.reference.npy"
    stage_metadata_path = output / f"{stem}.reference.json"
    signature = dict(source=str(source.resolve()), source_metadata_sha256=sha256(source / ".zmetadata"),
                     source_time_sha256=hashlib.sha256(times.tobytes()).hexdigest(),
                     observable=observable, hour=hour, start_year=start_year, end_year=end_year,
                     reference_dates=[str(v) for v in times[selected]],
                     lat_sha256=sha256(output / "lat.npy"), lon_sha256=sha256(output / "lon.npy"),
                     units={"T2m": "K", "WS10": "m s-1", "TP24": "mm"}[observable])
    if stage.exists() and stage_metadata_path.exists():
        metadata = json.loads(stage_metadata_path.read_text())
        if metadata.get("signature") == signature and sha256(stage) == metadata.get("sha256"):
            return stage, metadata
        raise ValueError("Existing reference-stage integrity or identity mismatch")
    temporary = stage.with_name(stage.name + f".tmp.{os.getpid()}")
    data = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                     shape=(len(selected), *shape))
    started = time.monotonic()
    for row, index in enumerate(selected):
        if observable == "T2m":
            field = _field(group, "2m_temperature", index, shape)
        elif observable == "WS10":
            u = _field(group, "10m_u_component_of_wind", index, shape)
            v = _field(group, "10m_v_component_of_wind", index, shape)
            field = np.hypot(u.astype(np.float64), v.astype(np.float64)).astype(np.float32)
        else:
            if index < 3 or times[index] - times[index - 3] != np.timedelta64(18, "h"):
                raise ValueError("Missing precipitation history at reference boundary")
            field = tp24_mm(np.stack([_field(group, "total_precipitation_6hr", i, shape)
                                       for i in range(index - 3, index + 1)]))
        if not np.all(np.isfinite(field)):
            raise ValueError("Derived historical observable is nonfinite")
        data[row] = field
        if (row + 1) % 365 == 0:
            data.flush()
            print(json.dumps(dict(stage=stem, completed=row + 1, total=len(selected),
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
    data.flush()
    del data
    os.replace(temporary, stage)
    metadata = dict(signature=signature, sha256=sha256(stage), shape=[len(selected), *shape],
                    elapsed_seconds=time.monotonic() - started)
    _atomic_json(stage_metadata_path, metadata)
    return stage, metadata


def compute_quantiles(stage: Path, metadata: dict, output: Path,
                      window_days: int = 30, tile_points: int = 1024) -> dict:
    if not 0 <= window_days < 183 or tile_points < 1:
        raise ValueError("Invalid window or tile size")
    signature = metadata["signature"]
    stem = f"{signature['observable']}_h{signature['hour']:02d}"
    path, record_path = output / f"{stem}.npy", output / f"{stem}.json"
    specification = dict(version=1, signature=signature, reference_sha256=metadata["sha256"],
                         quantiles=list(QUANTILES), quantile_method="linear", window_days=window_days,
                         calendar="365_day_excluding_feb29", dtype="float32")
    if path.exists() and record_path.exists():
        existing = json.loads(record_path.read_text())
        if existing.get("specification") == specification and existing.get("sha256") == sha256(path):
            return existing
        raise ValueError("Existing quantile output integrity or identity mismatch")
    values = np.load(stage, mmap_mode="r")
    if list(values.shape) != metadata["shape"] or values.dtype != np.float32:
        raise ValueError("Reference shape/dtype differs from completion metadata")
    flat = values.reshape(values.shape[0], -1)
    days = calendar_day_indices(np.asarray(signature["reference_dates"], dtype="datetime64[ns]"))
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(365, len(QUANTILES), *values.shape[1:]))
    result_flat = result.reshape(365, len(QUANTILES), -1)
    sample_counts = []
    started = time.monotonic()
    for day in range(365):
        indices = np.flatnonzero(circular_window_mask(days, day, window_days))
        expected = (2 * window_days + 1) * (signature["end_year"] - signature["start_year"] + 1)
        if len(indices) != expected:
            raise ValueError("Climatology window lacks its complete historical sample")
        sample_counts.append(len(indices))
        for left in range(0, flat.shape[1], tile_points):
            right = min(left + tile_points, flat.shape[1])
            block = flat[indices, left:right]
            if not np.all(np.isfinite(block)):
                raise ValueError("Nonfinite reference quantile sample")
            result_flat[day, :, left:right] = np.quantile(block, QUANTILES, axis=0, method="linear")
        if (day + 1) % 10 == 0:
            result.flush()
            print(json.dumps(dict(quantiles=stem, completed_days=day + 1,
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
    result.flush()
    del result_flat, result
    os.replace(temporary, path)
    record = dict(path=path.name, sha256=sha256(path), specification=specification,
                  shape=[365, len(QUANTILES), *values.shape[1:]], sample_counts=sample_counts,
                  elapsed_seconds=time.monotonic() - started)
    _atomic_json(record_path, record)
    return record


def finalize(output: str | Path) -> dict:
    output = Path(output)
    records = []
    reference_common = None
    for observable in OBSERVABLES:
        for hour in HOURS:
            stem = f"{observable}_h{hour:02d}"
            record = json.loads((output / f"{stem}.json").read_text())
            specification = record["specification"]
            signature = specification["signature"]
            common = {key: value for key, value in signature.items()
                      if key not in ("observable", "hour", "reference_dates", "units")}
            common.update({key: specification[key] for key in
                           ("quantiles", "quantile_method", "window_days", "calendar", "dtype")})
            if reference_common is None:
                reference_common = common
            if common != reference_common or signature["observable"] != observable or signature["hour"] != hour:
                raise ValueError("Quantile workers used incompatible reference identities")
            path = output / record["path"]
            if path.name != f"{stem}.npy" or sha256(path) != record["sha256"]:
                raise ValueError("Quantile output checksum mismatch")
            values = np.load(path, mmap_mode="r")
            if values.shape != (365, 4, 181, 360) or values.dtype != np.float32:
                raise ValueError("Quantile output shape/dtype mismatch")
            for block in values:
                if not np.all(np.isfinite(block)) or np.any(np.diff(block, axis=0) < 0):
                    raise ValueError("Quantile arrays contain nonfinite or unordered thresholds")
            records.append(dict(path=path.name, sha256=record["sha256"], observable=observable,
                                hour=hour, shape=list(values.shape), units=signature["units"],
                                metadata_path=f"{stem}.json", metadata_sha256=sha256(output / f"{stem}.json")))
    for name in ("lat", "lon"):
        path = output / f"{name}.npy"
        if sha256(path) != reference_common[f"{name}_sha256"]:
            raise ValueError("Climatology coordinate checksum mismatch")
        records.append(dict(path=path.name, sha256=sha256(path), coordinate=name))
    result = dict(version=1, status="complete", reference=reference_common, arrays=records,
                  observables=list(OBSERVABLES), hours=list(HOURS), quantiles=list(QUANTILES),
                  note="Historical percentile occurrence/intensity thresholds; not record extremes or station validation.")
    _atomic_json(output / "metadata.json", result)
    return result


def load_thresholds(output: str | Path, observable: str, timestamp) -> np.ndarray:
    timestamp = np.datetime64(timestamp, "ns")
    hour = int(timestamp.astype("datetime64[h]").astype(np.int64) % 24)
    if observable not in OBSERVABLES or hour not in HOURS:
        raise ValueError("Unknown observable or unsupported valid UTC hour")
    return np.load(Path(output) / f"{observable}_h{hour:02d}.npy", mmap_mode="r")[calendar_day_index(timestamp)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--tile-points", type=int, default=1024)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        result = finalize(args.output_dir)
        print(json.dumps(dict(status=result["status"], arrays=len(result["arrays"]))))
        return
    if args.source is None or args.task_index is None or not 0 <= args.task_index < 12:
        parser.error("workers require --source and --task-index in0..11")
    if args.start_year > args.end_year:
        parser.error("reference years are reversed")
    observable, hour = OBSERVABLES[args.task_index // 4], HOURS[args.task_index % 4]
    stage, metadata = stage_reference(args.source, args.output_dir, observable, hour,
                                      args.start_year, args.end_year)
    result = compute_quantiles(stage, metadata, args.output_dir, args.window_days, args.tile_points)
    print(json.dumps(dict(status="complete", path=result["path"], sha256=result["sha256"])))


if __name__ == "__main__":
    main()
