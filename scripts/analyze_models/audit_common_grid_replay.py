#!/usr/bin/env python3
"""Read-only forcing-context and historical-baseline checks for the Jan 1 pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    array = np.asarray(array)
    result = hashlib.sha256()
    result.update(str(array.dtype).encode())
    result.update(str(array.shape).encode())
    result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def stats(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Invalid comparison arrays")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    nonzero = reference != 0
    return {
        "shape": list(candidate.shape), "dtype": str(candidate.dtype),
        "exactly_equal": bool(np.array_equal(candidate, reference)),
        "candidate_sha256": array_digest(candidate), "reference_sha256": array_digest(reference),
        "rms_difference": float(np.sqrt(np.mean(delta ** 2))),
        "max_absolute_difference": float(np.abs(delta).max()),
        "max_relative_difference_nonzero_reference": float(np.max(np.abs(delta[nonzero] / reference[nonzero]))) if nonzero.any() else None,
        "outside_historical_tolerance_count": int(np.count_nonzero(np.abs(delta) > 1e-6 + .001 * np.abs(reference))),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    workflow = args.workflow.resolve()
    output = args.output or workflow / "audits/replay_context_first.json"
    if output.exists():
        raise FileExistsError(output)
    protocol = read(workflow / "protocol.json")
    original_path = Path(protocol["original_campaign"])
    original = read(original_path / "protocol.json")
    source = Path(original["source_root"])
    if digest(original_path / "protocol.json") != protocol["original_protocol_sha256"]:
        raise ValueError("Historical protocol changed")
    sys.path[:0] = [str(source), str(source / "third_party/graphcast")]
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("This bounded diagnostic must explicitly use JAX_PLATFORMS=cpu")
    os.environ["XLA_FLAGS"] = original["numerics"]["xla_flags"]
    for name, value in original["numerics"].get("environment", {}).items():
        os.environ[name] = value
    import jax
    from src.models.graphcast.training.core.dataset import prepare_dataset_for_task
    from src.models.graphcast.training.core.model import load_graphcast_checkpoint
    from scripts.analyze_models.validate_res1_2023_data import LocalV2

    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("Unexpected non-CPU device")
    checkpoint = load_graphcast_checkpoint(Path(original["baseline"]["path"]))
    task = checkpoint.task_config
    del checkpoint
    store = LocalV2(Path(original["data_path"]))
    starts = pd.DatetimeIndex(original["initialization_times"]).tz_convert("UTC").tz_localize(None)
    full_times = pd.date_range(starts[0] - pd.Timedelta("6h"),
                               starts[-1] + pd.Timedelta("240h"), freq="6h")
    smoke_times = full_times[:42]
    generated = {}
    elapsed = {}
    for label, times in (("smoke", smoke_times), ("full", full_times)):
        tick = time.monotonic()
        dataset = xr.Dataset(coords={
            "batch": [0], "time": times.values,
            "datetime": (("batch", "time"), times.values[None, :]),
            "lat": store.vector("lat"), "lon": store.vector("lon"),
        })
        dataset = prepare_dataset_for_task(dataset, task)
        jax.block_until_ready(dataset["toa_incident_solar_radiation"].data)
        generated[label] = {name: np.asarray(dataset[name].isel(time=slice(0, 42)).values).copy()
                            for name in task.forcing_variables}
        elapsed[label] = time.monotonic() - tick
        print(f"[replay-audit] generated {label}: {len(times)} timestamps in {elapsed[label]:.2f}s", flush=True)
        del dataset
    comparisons = {name: stats(generated["smoke"][name], generated["full"][name])
                   for name in task.forcing_variables}
    comparisons["all_generated_forcings_equal"] = all(v["exactly_equal"] for v in comparisons.values())

    new_path = workflow / "pilot/native/m1_00.json"
    new = read(new_path)
    stamp = "2023-01-01T00:00:00Z"
    fresh = next(r for r in new["original_graphcast_loss_per_initialization"] if r["initialization_time"] == stamp)
    baselines = {}
    sources = []
    for model in original["models"]:
        path = original_path / "shards" / (model["model_id"] + "_00.json")
        payload = read(path)
        record = next(r for r in payload["original_graphcast_loss_per_initialization"] if r["initialization_time"] == stamp)
        values = np.asarray(record["baseline_per_step"])
        baselines[model["model_id"]] = {
            "evaluator_family": model["evaluator_family"], "per_lead": values.tolist(),
            "mean": float(values.mean()), "day10": float(values[-1]),
            "gpu_uuid": payload.get("gpu_uuid"), "driver": payload.get("driver_version"),
            "fresh_pilot_difference": stats(fresh["baseline_per_step"], values),
            "fresh_pilot_mean_relative_pct": float(100 * (np.mean(fresh["baseline_per_step"]) / values.mean() - 1)),
        }
        sources.append({"path": str(path), "sha256": digest(path)})
        if model["model_id"] == "M1":
            original_m1 = record
    ranges = {}
    for label, ids in (("v24_same_evaluator", [m for m, v in baselines.items() if v["evaluator_family"] == "v24_Ilya"]),
                       ("all_seven_including_legacy_evaluator", list(baselines))):
        values = np.asarray([baselines[m]["per_lead"] for m in ids])
        fresh_values = np.asarray(fresh["baseline_per_step"])
        means = values.mean(axis=1)
        ranges[label] = {"models": ids, "pooled_baseline_min": float(means.min()),
                         "pooled_baseline_max": float(means.max()),
                         "fresh_pooled_baseline": float(fresh_values.mean()),
                         "fresh_pooled_within_historical_range": bool(means.min() <= fresh_values.mean() <= means.max()),
                         "fresh_leads_within_historical_range": int(np.count_nonzero(
                             (fresh_values >= values.min(axis=0)) & (fresh_values <= values.max(axis=0)))),
                         "historical_max_per_lead_range_pct_of_M1": float(np.max(
                             (values.max(axis=0) - values.min(axis=0)) / np.asarray(original_m1["baseline_per_step"])) * 100)}
    code_paths = [Path(__file__), source / "src/models/graphcast/training/core/dataset.py",
                  source / "third_party/graphcast/graphcast/data_utils.py",
                  source / "third_party/graphcast/graphcast/solar_radiation.py"]
    result = {
        "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol["protocol_hash"], "original_protocol_hash": original["protocol_hash"],
        "code": [{"path": str(path.resolve()), "sha256": digest(path)} for path in code_paths],
        "runtime": {"device": str(jax.devices()), "jax_version": jax.__version__,
                    "platform": "cpu", "elapsed_seconds": elapsed,
                    "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2},
        "forcing_context": {"smoke_timestamp_count": len(smoke_times), "full_timestamp_count": len(full_times),
                            "start": str(full_times[0]), "full_end": str(full_times[-1]),
                            "comparison_end": str(smoke_times[-1]), "comparisons": comparisons},
        "native_replay": {branch: stats(fresh[branch + "_per_step"], original_m1[branch + "_per_step"])
                          for branch in ("baseline", "full")},
        "historical_baselines": baselines, "historical_ranges": ranges,
        "data_sources": sources + [{"path": str(new_path), "sha256": digest(new_path)}],
        "limits": [
            "The forcing context test runs on CPU; it does not reproduce GPU kernels or establish GPU numerical repeatability.",
            "Coordinate-only datasets invoke the unchanged frozen preparation function; meteorological fields are absent because this function only generates forcings and handles static dimensions.",
            "Solar radiation uses one timestamp per scalar-time JIT call and is subsequently stacked; source inspection reveals no mathematical dependence of an individual timestamp on context length.",
            "Historical baselines span different residual policies and GPUs; four use the same V24 evaluator, three use the legacy evaluator. This is existing execution variation, not a new controlled repeatability experiment.",
            "No forecast implementation, checkpoint, input dataset, runtime policy, or production gate was changed. Numerical replay differences remain advisory and their cause is not established.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "forcing_equal": comparisons["all_generated_forcings_equal"],
                      "historical_ranges": ranges}, indent=2))


if __name__ == "__main__":
    main()
