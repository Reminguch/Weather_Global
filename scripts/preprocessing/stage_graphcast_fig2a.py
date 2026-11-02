#!/usr/bin/env python3
"""Stage two native 37-level inputs and native Z500 truth for a frozen manifest."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.preprocessing.stage_common_grid_2023 import (
    NATIVE_LAT, NATIVE_LON, assign, configure_file, coordinate, data_variable,
    file_id, native_field, netcdf_file, publish_receipt, receipt_path,
    sha256_file, simple_attrs, static_fields, utc_text, validate_receipt,
    write_inputs,
)
from src.data_operations.staging.stage_arco_era5_res1 import (
    ChunkSource, _json_hash, _write_json_atomic, _writer_lock,
)


def write_z500(source, path, stamp, hour_lookup, levels, identity):
    values = native_field(source, "geopotential", hour_lookup[stamp])[
        int(np.flatnonzero(levels == 500)[0])].copy()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with netcdf_file(temporary) as handle:
        configure_file(handle, {"lat": 721, "lon": 1440}, {
            "source": source.uri, "valid_time": utc_text(stamp),
            "configuration_sha256": identity["configuration_sha256"],
            "pressure_level_hpa": 500,
        })
        coordinate(handle, "lat", NATIVE_LAT, {"units": "degrees_north"})
        coordinate(handle, "lon", NATIVE_LON, {"units": "degrees_east"})
        field = data_variable(handle, "Z500", ("lat", "lon"),
                              simple_attrs(source, "geopotential"))
        assign(field, slice(None), values)
    temporary.replace(path)
    return publish_receipt(path, identity)


def stage(workflow, workers):
    protocol = json.loads((workflow / "protocol.json").read_text())
    output = Path(protocol["data_root"])
    output.mkdir(parents=True, exist_ok=True)
    starts = pd.DatetimeIndex(pd.to_datetime(protocol["initialization_times"], utc=True)).tz_localize(None)
    leads = protocol["lead_hours"]
    truths = pd.DatetimeIndex(sorted({s + pd.Timedelta(hours=h) for s in starts for h in leads}))
    with _writer_lock(output / ".staging.lock"):
        source = ChunkSource(protocol["era5_source"])
        source.validate_layout()
        levels = source.read_vector("level")
        np.testing.assert_array_equal(levels, protocol["pressure_levels"])
        times = pd.DatetimeIndex(sorted(set(truths) | set(starts) | set(starts - pd.Timedelta("6h"))))
        hour_lookup = dict(zip(times, map(int, source.hourly_indices(times))))
        config = {"protocol_sha256": sha256_file(workflow / "protocol.json"),
                  "source_structure_sha256": source.fingerprint,
                  "stager_sha256": sha256_file(Path(__file__)),
                  "source_uri": source.uri,
                  "static_source_time": utc_text(starts[0] - pd.Timedelta("6h")),
                  "initialization_times": protocol["initialization_times"],
                  "lead_hours": leads, "truth": "native 0.25-degree Z500 in m2/s2"}
        config_hash = _json_hash(config)
        manifest = {"configuration": config, "configuration_sha256": config_hash,
                    "expected_inputs": len(starts), "expected_truth": len(truths)}
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Staging identity changed; use a new campaign")
        _write_json_atomic(manifest_path, manifest)
        _write_json_atomic(output / "source_metadata.json", source.metadata)
        for folder in ("inputs", "truth"):
            (output / folder).mkdir(exist_ok=True)
        jobs, seen = [], set()
        # Finish earlier calendar months first, preserving resumability.
        for stamp in starts:
            for kind, date in [("inputs", stamp)] + [("truth", stamp + pd.Timedelta(hours=h)) for h in leads]:
                if (kind, date) in seen:
                    continue
                seen.add((kind, date))
                path = output / kind / f"{file_id(date)}.nc"
                identity = {"kind": kind, "timestamp": utc_text(date),
                            "configuration_sha256": config_hash, "source_uri": source.uri,
                            "source_structure_sha256": source.fingerprint,
                            "static_source_time": config["static_source_time"]}
                jobs.append((kind, date, path, identity))
        statics = static_fields(source, hour_lookup[starts[0] - pd.Timedelta("6h")])
        started = time.monotonic()
        receipts = []

        def run(job):
            kind, stamp, path, identity = job
            receipt = validate_receipt(path, identity)
            if receipt is not None:
                return receipt
            if kind == "inputs":
                return write_inputs(source, path, stamp, hour_lookup, levels, statics, identity)
            return write_z500(source, path, stamp, hour_lookup, levels, identity)

        report = {"status": "running", "expected_files": len(jobs), "completed_files": 0}
        _write_json_atomic(output / "progress.json", report)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run, job) for job in jobs]
                for future in as_completed(futures):
                    receipt = future.result()
                    receipts.append(receipt)
                    report.update(completed_files=len(receipts), elapsed_seconds=time.monotonic() - started,
                                  downloaded_bytes=source.downloaded_bytes)
                    _write_json_atomic(output / "progress.json", report)
                    print(f"ready {receipt['kind']}/{Path(receipt['path']).name} "
                          f"{len(receipts)}/{len(jobs)} {source.downloaded_bytes / 1e9:.2f} GB", flush=True)
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            _write_json_atomic(output / "progress.json", report)
            raise
        report["status"] = "complete"
        _write_json_atomic(output / "progress.json", report)
        _write_json_atomic(output / "READY.json", {
            **manifest, "files": sorted(receipts, key=lambda r: r["path"]),
            "elapsed_seconds": time.monotonic() - started,
            "downloaded_bytes": source.downloaded_bytes,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    stage(args.workflow.resolve(), args.workers)
