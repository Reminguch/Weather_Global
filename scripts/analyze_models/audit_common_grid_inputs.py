#!/usr/bin/env python3
"""Audit two staged input frames against the principal campaign at shared points.

This is a bounded source-parity check. It does not run forecasts, interpolate
inputs, compare verification scores, or change either prepared dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analyze_models.validate_res1_2023_data import LocalV2, TARGETS, STATIC, PRESSURE, LEVELS


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def comparison(actual, reference, *, levels=None):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    if actual.shape != reference.shape or actual.dtype != np.float32 or reference.dtype != np.float32:
        raise ValueError("Compared arrays must have identical shapes and float32 physical values")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite initial-state fields")
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    spatial_rmse = np.sqrt(np.mean(difference ** 2, axis=(-2, -1)))
    spatial_maximum = np.max(np.abs(difference), axis=(-2, -1))
    spatial_std = np.std(reference.astype(np.float64), axis=(-2, -1))
    # Classify differences bounded by eight float32 relative roundoff units.
    # No tolerance is used to relabel unequal arrays as bitwise identical.
    scale = np.maximum(np.abs(actual.astype(np.float64)), np.abs(reference.astype(np.float64)))
    tolerance = (8 * np.finfo(np.float32).eps * scale
                 + 8 * float(np.nextafter(np.float32(0), np.float32(1))))
    exact = np.array_equal(actual, reference)
    roundoff = bool(np.all(np.abs(difference) <= tolerance))
    detail = []
    for index, (rms, maximum, std) in enumerate(zip(np.atleast_1d(spatial_rmse),
                                                  np.atleast_1d(spatial_maximum),
                                                  np.atleast_1d(spatial_std))):
        detail.append({
            "level_hpa": int(levels[index]) if levels is not None else None,
            "rms_difference": float(rms), "max_abs_difference": float(maximum),
            "reference_spatial_std": float(std),
            "normalized_rms_difference": float(rms / std) if std > 0 else None,
            "normalized_max_abs_difference": float(maximum / std) if std > 0 else None,
        })
    normalized_rms = [v["normalized_rms_difference"] for v in detail if v["normalized_rms_difference"] is not None]
    normalized_max = [v["normalized_max_abs_difference"] for v in detail if v["normalized_max_abs_difference"] is not None]
    return {
        "shape": list(actual.shape), "dtype": str(actual.dtype),
        "classification": "exact" if exact else "float32_roundoff" if roundoff else "different",
        "exactly_equal": bool(exact),
        "candidate_array_sha256": array_hash(actual), "reference_array_sha256": array_hash(reference),
        "changed_value_count": int(np.count_nonzero(actual != reference)),
        "rms_difference": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "max_level_normalized_rms": max(normalized_rms) if normalized_rms else None,
        "max_level_normalized_abs": max(normalized_max) if normalized_max else None,
        "per_level": detail,
        "within_historical_weather_source_check_thresholds": bool(
            normalized_rms and normalized_max
            and max(normalized_rms) <= .01 and max(normalized_max) <= .1),
    }


def aggregate_classification(items):
    classes = {item["classification"] for item in items}
    return "different" if "different" in classes else "float32_roundoff" if "float32_roundoff" in classes else "exact"


def audit(workflow, initialization, input_path=None):
    protocol_path = workflow / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    campaign_path = Path(protocol["original_campaign"]) / "protocol.json"
    old = json.loads(campaign_path.read_text())
    if file_hash(campaign_path) != protocol["original_protocol_sha256"]:
        raise ValueError("Reference campaign protocol identity changed")
    init = pd.Timestamp(initialization)
    if init.tzinfo is None or init.utcoffset() != pd.Timedelta(0):
        raise ValueError("Initialization must have an explicit UTC offset")
    expected_times = pd.date_range(init - pd.Timedelta("6h"), periods=2, freq="6h")
    identifier = init.strftime("%Y%m%dT%H%M%SZ")
    input_path = (input_path or Path(protocol["inputs025"]) / f"{identifier}.nc").resolve()
    receipt_path = Path(str(input_path) + ".READY.json")
    if not input_path.exists() or not receipt_path.exists():
        raise FileNotFoundError(f"Staged input is not ready: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("kind") != "inputs025" or receipt.get("schema_version") != 1
            or Path(receipt["path"]).resolve() != input_path
            or pd.Timestamp(receipt["timestamp"]) != init
            or receipt.get("size_bytes") != input_path.stat().st_size
            or receipt.get("sha256") != file_hash(input_path)):
        raise ValueError("Input readiness receipt or checksum is invalid")
    reference = LocalV2(Path(old["data_path"]))
    metadata_record = old["data_metadata"]
    if file_hash(metadata_record["path"]) != metadata_record["sha256"]:
        raise ValueError("Principal evaluation dataset metadata changed")
    reference_coords = {name: reference.vector(name) for name in ("lat", "lon", "level")}
    if not np.array_equal(reference_coords["level"], LEVELS):
        raise ValueError("Principal evaluation pressure levels changed")
    lookup = {stamp: index for index, stamp in enumerate(reference.times())}
    records = []
    statics = {}
    with xr.open_dataset(input_path) as native:
        expected_naive = expected_times.tz_localize(None).values
        if not np.array_equal(native.time.values.astype("datetime64[ns]"), expected_naive):
            raise ValueError("Two input timestamps do not match the forecast initialization")
        if "batch" in native.dims:
            if native.sizes["batch"] != 1:
                raise ValueError("Expected one staged initialization")
            native = native.isel(batch=0, drop=True)
        # Label selection explicitly reverses ascending native latitude to the
        # descending principal grid. No nearest selection or averaging occurs.
        selected = native.sel(lat=reference_coords["lat"], lon=reference_coords["lon"],
                              level=reference_coords["level"])
        for name in ("lat", "lon", "level"):
            if not np.array_equal(selected[name].values, reference_coords[name]):
                raise ValueError(f"Shared {name} coordinate selection is not exact")
        for stamp in expected_times:
            naive = stamp.tz_localize(None).to_datetime64().astype("datetime64[h]")
            if naive not in lookup:
                raise ValueError(f"Principal dataset is missing input time {stamp}")
            variables = {}
            for name in TARGETS:
                axes = ("level", "lat", "lon") if name in PRESSURE else ("lat", "lon")
                actual = selected[name].sel(time=naive).transpose(*axes).values
                wanted = reference.field(name, lookup[naive])
                result = comparison(actual, wanted, levels=LEVELS if name in PRESSURE else None)
                result.update(candidate_units=native[name].attrs.get("units"),
                              reference_units=reference.attrs(name).get("units"),
                              reference_chunk_sha256=file_hash(reference.chunk_path(
                                  name, (lookup[naive],) + (0,) * len(axes))))
                variables[name] = result
                print(f"[input-parity] {stamp} {name}: {result['classification']} "
                      f"max={result['max_abs_difference']:.9g}", flush=True)
            records.append({"time": stamp.isoformat().replace("+00:00", "Z"), "variables": variables})
        for name in STATIC:
            actual = selected[name].transpose("lat", "lon").values
            result = comparison(actual, reference.field(name))
            result.update(candidate_units=native[name].attrs.get("units"),
                          reference_units=reference.attrs(name).get("units"),
                          reference_chunk_sha256=file_hash(reference.chunk_path(name, (0, 0))))
            statics[name] = result
        native_static_time = native.attrs.get("static_source_time")
    weather_status = aggregate_classification(item for sample in records for item in sample["variables"].values())
    static_status = aggregate_classification(statics.values())
    return {
        "schema_version": 1, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "initialization_time": init.isoformat().replace("+00:00", "Z"),
        "audit_code": {"path": str(Path(__file__).resolve()), "sha256": file_hash(__file__)},
        "local_zarr_reader": {"path": str(Path(sys.modules[LocalV2.__module__].__file__).resolve()),
                              "sha256": file_hash(sys.modules[LocalV2.__module__].__file__)},
        "comparison_protocol": {"path": str(protocol_path), "sha256": file_hash(protocol_path),
                                "protocol_hash": protocol["protocol_hash"]},
        "original_protocol": {"path": str(campaign_path), "sha256": file_hash(campaign_path),
                              "protocol_hash": old["protocol_hash"]},
        "candidate_input": receipt, "candidate_receipt_sha256": file_hash(receipt_path),
        "reference_store": {"path": old["data_path"],
                            "metadata_sha256": file_hash(Path(old["data_path"]) / ".zmetadata")},
        "coordinates": {name: values.tolist() for name, values in reference_coords.items()},
        "selection": "Exact coordinate labels; 1-degree point subset and 13 pressure levels; no remapping",
        "precipitation": "Both sum six hourly depths ending at t-5h,...,t; native sequential float32 sum versus original numpy float32 sum",
        "weather_classification": weather_status, "static_classification": static_status,
        "weather_samples": records, "static_fields": statics,
        "static_sources": {
            "candidate": "ARCO native 0.25-degree fields", "candidate_time": native_static_time,
            "reference": reference.metadata[".zattrs"].get("static_reference_store"),
        },
        "interpretation": (
            "Weather and static initial inputs agree exactly at every shared grid point and level for both audited times."
            if weather_status == static_status == "exact" else
            "Some inputs differ; classifications and per-level discrepancies above distinguish float32 roundoff from source differences. Forecast impact is not measured by this audit."
        ),
        "limitations": [
            "One initialization and two input times; does not establish parity for all 122 starts.",
            "Native model resolutions, pressure-level counts, model architectures and pretraining coverage differ.",
            "Solar/calendar forcings are generated downstream and are excluded from this raw-weather audit.",
            "The 1% RMS/10% maximum normalized thresholds describe the earlier data-source check, not a demonstrated forecast-impact threshold.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--initialization", default="2023-01-01T00:00:00Z")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    workflow = args.workflow.resolve()
    output = args.output or workflow / "audits/input_parity_first.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace completed audit: {output}")
    result = audit(workflow, args.initialization, args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(output), "weather": result["weather_classification"],
                      "statics": result["static_classification"], "interpretation": result["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
