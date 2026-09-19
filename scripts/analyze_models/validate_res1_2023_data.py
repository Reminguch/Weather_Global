#!/usr/bin/env python3
"""Bounded, read-only preflight for the frozen daily 2023 canonical-loss evaluation.

Read Zarr v2 chunks synchronously: this avoids async local-store clients and never
loads the full year. The staging report certifies a prior full finite/checksum
scan; this preflight verifies all chunk files and rechecks selected boundaries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numcodecs
import numpy as np

SURFACE = (
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
PRESSURE = (
    "temperature", "geopotential", "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity",
)
TARGETS = SURFACE + PRESSURE + ("total_precipitation_6hr",)
STATIC = ("geopotential_at_surface", "land_sea_mask")
LEVELS = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LocalV2:
    def __init__(self, path: Path):
        self.path = path
        self.metadata = json.loads((path / ".zmetadata").read_text())["metadata"]

    def spec(self, name: str) -> dict:
        return self.metadata[f"{name}/.zarray"]

    def attrs(self, name: str) -> dict:
        return self.metadata[f"{name}/.zattrs"]

    def chunk_path(self, name: str, index: tuple[int, ...]) -> Path:
        return self.path / name / self.spec(name).get("dimension_separator", ".").join(map(str, index))

    def chunk(self, name: str, index: tuple[int, ...]) -> np.ndarray:
        spec = self.spec(name)
        payload = self.chunk_path(name, index).read_bytes()
        if spec.get("compressor"):
            payload = numcodecs.get_codec(spec["compressor"]).decode(payload)
        for transform in reversed(spec.get("filters") or []):
            payload = numcodecs.get_codec(transform).decode(payload)
        values = np.frombuffer(payload, dtype=spec["dtype"])
        require(values.size == int(np.prod(spec["chunks"])), f"Invalid decoded chunk for {name} {index}")
        return values.reshape(spec["chunks"], order=spec.get("order", "C"))

    def vector(self, name: str) -> np.ndarray:
        spec = self.spec(name)
        require(len(spec["shape"]) == 1, f"Expected vector {name}")
        length, width = spec["shape"][0], spec["chunks"][0]
        return np.concatenate([self.chunk(name, (i,))[:min(width, length - i * width)]
                               for i in range((length + width - 1) // width)])

    def field(self, name: str, time_index: int | None = None) -> np.ndarray:
        spec = self.spec(name)
        dims = self.attrs(name)["_ARRAY_DIMENSIONS"]
        if time_index is not None:
            require(dims[0] == "time" and spec["chunks"][0] == 1, f"Unsupported temporal chunks: {name}")
            require(all(c >= n for c, n in zip(spec["chunks"][1:], spec["shape"][1:])),
                    f"Unsupported spatial chunks: {name}")
            values = self.chunk(name, (time_index,) + (0,) * (len(dims) - 1))[0]
            shape = spec["shape"][1:]
        else:
            require(all(c >= n for c, n in zip(spec["chunks"], spec["shape"])), f"Unsupported static chunks: {name}")
            values = self.chunk(name, (0,) * len(dims))
            shape = spec["shape"]
        result = np.array(values[tuple(slice(n) for n in shape)], copy=True)
        require(np.isfinite(result).all(), f"Nonfinite {name} at {time_index}")
        return result

    def times(self) -> np.ndarray:
        attrs = self.attrs("time")
        require(attrs["units"].startswith("hours since "), "Expected hourly time encoding")
        epoch = np.datetime64(attrs["units"].removeprefix("hours since "), "h")
        return epoch + self.vector("time").astype("timedelta64[h]")


def validate(args: argparse.Namespace) -> dict:
    numcodecs.blosc.set_nthreads(1)
    store, reference = LocalV2(args.dataset), LocalV2(args.reference)
    stage_path = args.dataset.with_name(args.dataset.name + ".stage_report.json")
    stage = json.loads(stage_path.read_text())
    source_metadata_path = args.dataset / "source_metadata.json"
    source_attrs = json.loads(source_metadata_path.read_text())["metadata"][".zattrs"]
    require(stage == json.loads((args.dataset / "stage_report.json").read_text()), "Published staging reports disagree")
    times, reference_times = store.times(), reference.times()
    require(stage.get("status") == "complete" and bool(stage.get("verified_at")), "Staging is not completely verified")
    require(len(times) == stage["config"]["time_count"] == stage["completed_count"], "Staging/time counts disagree")
    require(np.all(np.diff(times) == np.timedelta64(6, "h")), "Data has a missing, duplicated, or disordered time")
    require(str(times[0]) == str(np.datetime64(stage["config"]["start_time"], "h")) and
            str(times[-1]) == str(np.datetime64(stage["config"]["end_time"], "h")), "Staging bounds disagree")
    require(times[-1].astype("datetime64[D]") <= np.datetime64(source_attrs["valid_time_stop"], "D"),
            "Requested truth extends past finalized ERA5 source coverage")
    config_hash = hashlib.sha256(json.dumps(stage["config"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(store.metadata[".zattrs"]["stage_config_sha256"] == config_hash, "Store/config hash mismatch")
    for name, wanted in (("lat", np.arange(90, -91, -1)), ("lon", np.arange(360)), ("level", LEVELS)):
        actual = store.vector(name)
        require(np.array_equal(actual, wanted), f"Unexpected {name} coordinates")
        require(np.array_equal(actual, reference.vector(name)), f"Reference {name} coordinates disagree")
    inits = np.arange(np.datetime64("2023-01-01", "D"), np.datetime64("2024-01-01", "D")).astype("datetime64[h]")
    lookup = {int(t.astype(np.int64)): i for i, t in enumerate(times)}
    origins = []
    for init in inits:
        wanted = init + np.arange(-1, args.target_steps + 1) * np.timedelta64(6, "h")
        require(all(int(t.astype(np.int64)) in lookup for t in wanted), f"Incomplete inputs/truth for {init}")
        origins.append(lookup[int(init.astype(np.int64))])
    records = stage["completed"]
    require(set(records) == {str(i) for i in range(len(times))}, "Incomplete timestamp validation records")
    for i, stamp in enumerate(times):
        record = records[str(i)]
        require(np.datetime64(record["timestamp"], "h") == stamp and
                bool(re.fullmatch(r"[0-9a-f]{64}", record["sha256"])) and bool(record.get("completed_at")),
                f"Invalid validation record at {i}")
    chunk_count = 0
    for name in TARGETS:
        pressure = name in PRESSURE
        shape = [len(times)] + ([13] if pressure else []) + [181, 360]
        dims = ["time"] + (["level"] if pressure else []) + ["lat", "lon"]
        spec = store.spec(name)
        require(spec["shape"] == shape and spec["chunks"] == [1] + shape[1:], f"Invalid layout for {name}")
        require(store.attrs(name)["_ARRAY_DIMENSIONS"] == dims and np.dtype(spec["dtype"]) == np.dtype("float32"), f"Invalid dimensions/dtype for {name}")
        require(store.attrs(name).get("units") == reference.attrs(name).get("units"), f"Units differ for {name}")
        for i in range(len(times)):
            chunk = store.chunk_path(name, (i,) + (0,) * (len(shape) - 1))
            require(chunk.is_file() and chunk.stat().st_size > 0, f"Missing/empty chunk: {chunk}")
            chunk_count += 1
    # Verify a deterministic sample including both forecast-year boundaries.
    sample_indices = sorted({0, 1, origins[0], origins[0] + args.target_steps,
                             origins[-1] - 1, origins[-1], origins[-1] + args.target_steps, len(times) - 1})
    for i in sample_indices:
        digest = hashlib.sha256()
        for name in TARGETS:
            values = store.field(name, i)
            digest.update(name.encode() + b"\0")
            digest.update(np.ascontiguousarray(values).tobytes())
        require(digest.hexdigest() == records[str(i)]["sha256"], f"Boundary checksum mismatch at {times[i]}")
    common = np.intersect1d(times, reference_times)
    require(len(common) >= 2 and np.all(common < np.datetime64("2023-01-01", "h")), "Expected pre-2023 overlap")
    comparisons = {}
    for name in TARGETS + STATIC:
        comparisons[name] = []
        stamps = common if name in TARGETS else [None]
        for stamp in stamps:
            i = int(np.flatnonzero(times == stamp)[0]) if stamp is not None else None
            j = int(np.flatnonzero(reference_times == stamp)[0]) if stamp is not None else None
            actual = store.field(name, i).astype(np.float64)
            expected = reference.field(name, j).astype(np.float64)
            difference = actual - expected
            rms = float(np.sqrt(np.mean(difference * difference)))
            maximum = float(np.max(np.abs(difference)))
            # Relative to the reference spatial variability, independently at
            # each pressure level, so large geopotential means cannot mask errors.
            axes = (-2, -1)
            reference_std = np.std(expected, axis=axes)
            absolute_floor = 1e-12
            normalized_rms = np.sqrt(np.mean(difference * difference, axis=axes)) / np.maximum(reference_std, absolute_floor)
            normalized_max = np.max(np.abs(difference), axis=axes) / np.maximum(reference_std, absolute_floor)
            passed = bool(np.max(normalized_rms) <= args.max_normalized_rms and np.max(normalized_max) <= args.max_normalized_abs)
            if name in STATIC:
                passed = bool(np.array_equal(actual, expected))
            comparisons[name].append({
                "timestamp": str(stamp) if stamp is not None else None,
                "rms_difference": rms, "max_abs_difference": maximum,
                "max_level_normalized_rms": float(np.max(normalized_rms)),
                "max_level_normalized_abs": float(np.max(normalized_max)),
                "reference_min": float(expected.min()), "reference_max": float(expected.max()),
                "staged_min": float(actual.min()), "staged_max": float(actual.max()),
                "units": store.attrs(name).get("units"), "passed": passed,
            })
            print(f"parity {name} {stamp}: rms={rms:.7g} max={maximum:.7g} normalized_rms={np.max(normalized_rms):.7g} passed={passed}", flush=True)
    passed = all(row["passed"] for rows in comparisons.values() for row in rows)
    return {
        "schema_version": 1, "status": "passed" if passed else "failed", "passed": passed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()), "reference": str(args.reference.resolve()),
        "source_uri": stage["config"]["source_uri"], "staging_verified_at": stage["verified_at"],
        "source_finalized_time_stop": source_attrs["valid_time_stop"],
        "source_metadata_sha256": sha256(source_metadata_path),
        "staging_report_sha256": sha256(stage_path), "dataset_metadata_sha256": sha256(args.dataset / ".zmetadata"),
        "validator_sha256": sha256(Path(__file__)), "time_count": len(times),
        "first_time": str(times[0]), "last_time": str(times[-1]), "step_hours": 6,
        "grid": {"lat": 181, "lon": 360, "levels": LEVELS.tolist()},
        "target_planes_per_timestamp": len(SURFACE) + len(PRESSURE) * len(LEVELS) + 1,
        "validated_timestamp_records": len(records), "existing_target_chunks": chunk_count,
        "checksum_rechecked_times": [str(times[i]) for i in sample_indices],
        "initialization_count": len(inits), "initialization_first": str(inits[0]), "initialization_last": str(inits[-1]),
        "initialization_utc_hour": 0, "input_frames": 2, "warmup_steps": 0, "target_steps": args.target_steps,
        "initialization_indices": origins,
        "source_parity": {
            "times": common.astype(str).tolist(), "comparisons": comparisons,
            "normalization": "Reference spatial standard deviation independently at each pressure level; maximum over levels.",
            "thresholds": {"max_normalized_rms": args.max_normalized_rms, "max_normalized_abs": args.max_normalized_abs,
                           "static_fields": "exact equality"},
            "threshold_rationale": "Reject source/unit/grid/accumulation mismatches exceeding 1% RMS or 10% maximum of reference spatial variability; these are data checks, not forecast skill criteria.",
            "precipitation_convention": stage["config"]["precipitation_hours"],
        },
        "limitations": [
            "Source parity is tested only at the two overlapping 2022-12-31 12/18 UTC timestamps; seasonal and all-four-hour parity is not established.",
            "Full-store finiteness and checksums are certified by the completed staging report; this bounded preflight rechecks only the listed boundaries and existence of all chunks.",
            "Coverage supports cold initialization with two input frames; it does not support a preceding 24-step warmup for every 2023 start.",
            "TISR is generated by the evaluator using GraphCast solar forcing, and is absent from the staged store.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr"))
    parser.add_argument("--reference", type=Path, default=Path("data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluations/v24_res1_2023_daily_20260917/data_validation.json"))
    parser.add_argument("--target-steps", type=int, default=40)
    parser.add_argument("--max-normalized-rms", type=float, default=0.01)
    parser.add_argument("--max-normalized-abs", type=float, default=0.1)
    args = parser.parse_args()
    require(args.target_steps > 0, "target-steps must be positive")
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"data preflight passed={report['passed']} report={args.output}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
