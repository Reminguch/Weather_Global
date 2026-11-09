#!/usr/bin/env python3
"""Frozen, development-fitted output smoothing and annual verification.

The first initialization in each cohort measures resources and is retained.
Only submit/release/submit-sweep stages submit jobs. Finite diagnostic drift is
a recorded warning, never a condition for continuing valid evaluations.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments.frozen_runtime import verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/run_v24_smoothing.py"
WRAPPER = "scripts/experiments/v24_smoothing.slurm"
REFERENCE = "artifacts/evaluations/v24_mgda_power_2023_20261106/shards/00/manifest.json"
ANNUAL = "artifacts/evaluations/v24_res1_2023_stride3_memory_20261004/initializations_2023.json"


def prepare(root, workspace, source, *, checkpoint=None, run_config=None, model_id="M1",
            development_manifest=None, calibration_shards=4, evaluation_shards=8,
            training_campaign=None):
    from src.models.mamba.v24_Ilya.development_manifest import create_development_manifest
    root, workspace, source = map(lambda p: Path(p).resolve(), (root, workspace, source))
    if root.exists():
        raise FileExistsError("Use a new smoothing campaign directory")
    frozen = verify_source(source)
    if DRIVER not in frozen["files"] or "src/analysis/smoothing_observer.py" not in frozen["files"]:
        raise ValueError("Snapshot predates the smoothing implementation")
    if not 2 <= calibration_shards <= 32 or not 2 <= evaluation_shards <= 122:
        raise ValueError("Shard count must include the retained first-start resource measurement")
    reference = read(workspace / REFERENCE)
    model = reference["task_specs"][0]
    checkpoint = Path(checkpoint or model["checkpoint"]).resolve()
    run_config = Path(run_config or model["run_config"]).resolve()
    configuration = read(run_config)
    if model_id == "GC" or not model_id or "/" in model_id:
        raise ValueError("A distinct, path-safe model identifier is required")
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "checks", "manifests", "calibrate", "evaluate"):
        (root / sub).mkdir()
    exact, annual = reference["protocols"]["exact"], reference["protocols"]["spectral"]
    dev_path = root / "manifests/development.json"
    development = create_development_manifest(exact, dev_path)
    if development_manifest and read(development_manifest) != development:
        raise ValueError("Supplied development manifest differs from the fixed exact32 cohort")
    annual_path = workspace / ANNUAL
    cohort = read(annual_path)
    if len(development["initializations"]) != 32 or len(cohort["initializations"]) != 122:
        raise ValueError("Expected the fixed development32 and annual122 cohorts")
    inputs = [pinned(checkpoint), pinned(run_config), pinned(workspace / REFERENCE),
              pinned(dev_path), pinned(annual_path), pinned(exact["baseline"]["path"])]
    inputs.extend(exact["statistics"])
    inputs.extend((exact["data_metadata"], annual["data_metadata"]))
    shards = {}
    for stage, payload, count in (("calibrate", development, calibration_shards),
                                  ("evaluate", cohort, evaluation_shards)):
        entries = payload["initializations"]
        chunks = [[entries[0]], *[list(part) for part in np.array_split(np.asarray(entries[1:], dtype=object), count - 1)]]
        shards[stage] = []
        for index, chunk in enumerate(chunks):
            path = root / f"manifests/{stage}_{index:02d}.json"
            write(path, {**payload, "initializations": chunk})
            record = pinned(path)
            inputs.append(record)
            shards[stage].append({"manifest": record, "count": len(chunk)})
    for item in inputs:
        check(item)
    manifest = seal(root / "manifest.json", {
        "format": "v24_smoothing_campaign_v1", "workspace": str(workspace),
        "source_root": str(source), "source_digest": frozen["source_digest"],
        "model_id": model_id, "model_family": configuration.get("model_family", "auto"),
        "training_campaign": str(Path(training_campaign).resolve()) if training_campaign else None,
        "checkpoint": pinned(checkpoint), "run_config": pinned(run_config),
        "architecture": configuration["architecture"], "inputs": inputs,
        "baseline": exact["baseline"], "stats_dir": exact["stats_dir"],
        "data": {"calibrate": exact["data_path"], "evaluate": annual["data_path"]},
        "cohorts": {"calibrate": pinned(dev_path), "evaluate": pinned(annual_path)},
        "shards": shards, "target_steps": 40,
        "policy": {"eta": .01, "lmax": 179, "knots": [1, 10, 40, 179],
            "gaussian_widths_degrees": [0, .25, .5, 1, 2, 4, 8],
            "lead_group_ends_hours": [24, 72, 168, 240], "uv_tied": True,
            "clipping": "none", "feedback": "raw forecast", "fit_year": 2022,
            "evaluation_year": 2023, "finite_drift": "warning",
            "known_exposure": "2023 has informed previous analysis; no held-out claim",
            "uncertainty_unit": "initialization date; overlapping forecasts are dependent"},
        "resource_measurement": {"first_initialization_retained": True,
            "memory_gib": 32, "minutes": 120, "cpus": 4,
            "basis": "res1 exact32 jobs12530195/12530971:12.3–12.7GiB,38–41min; "
                     "new bounded transform buffers and compilation unmeasured; first start measures actual RSS/runtime"}})
    return manifest


def verify(root, *, frozen=True):
    manifest = unseal(root / "manifest.json")
    for item in manifest["inputs"]:
        check(item)
    source = verify_source(manifest["source_root"])
    if source["source_digest"] != manifest["source_digest"]:
        raise ValueError("Campaign source identity changed")
    if frozen and ROOT.resolve() != Path(manifest["source_root"]).resolve():
        raise ValueError("Execute workers from the prepared source snapshot")
    return manifest


def run_worker(root, manifest, stage, task):
    import jax
    from src.analysis.smoothing_observer import SmoothingObserver
    from src.models.graphcast.training.core.model import load_stats
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Production smoothing forecasts require a GPU allocation")
    shard = manifest["shards"][stage][task]
    directory = root / stage / f"{task:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    filters = unseal(root / "filters.json") if stage == "evaluate" else None
    if filters and filters["campaign_hash"] != manifest["record_hash"]:
        raise ValueError("Filters belong to another campaign")
    identity = {"campaign_hash": manifest["record_hash"], "model_id": manifest["model_id"],
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "protocol_hash": shard["manifest"]["sha256"], "source_digest": manifest["source_digest"]}
    observer = SmoothingObserver(directory / "statistics",
        load_stats(Path(manifest["stats_dir"]))["diffs_stddev_by_level"], identity, mode=stage,
        filters=filters, branches=("GC", manifest["model_id"]))
    allowed = {field.name for field in fields(V24IlyaEvalConfig)}
    architecture = {key: value for key, value in manifest["architecture"].items() if key in allowed}
    config = V24IlyaEvalConfig(**architecture, ckpt=Path(manifest["checkpoint"]["path"]),
        out_json=directory / "exact_metrics.json", eval_mode="cold_full",
        ckpt_in=Path(manifest["baseline"]["path"]), stats_dir=Path(manifest["stats_dir"]),
        data_path=manifest["data"][stage], target_steps=40, warmup_steps=0,
        n_samples=shard["count"], initialization_manifest=Path(shard["manifest"]["path"]),
        val_year=2022 if stage == "calibrate" else 2023, model_id=manifest["model_id"],
        model_family=manifest["model_family"], residual_state_init="zero")
    started = time.monotonic()
    metrics = evaluate_v24_Ilya(config, prediction_observer=observer)
    check(manifest["checkpoint"])
    paths = sorted((directory / "statistics").glob("*.npz"))
    physical_loss = np.zeros(2)
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            if stage == "calibrate":
                physical_loss += np.sum(archive["e0"] * observer.layout.weights[None, None], axis=(1, 2)) / 40
            else:
                physical_loss += archive["exact_gc_loss"][:, :, 0].mean(axis=0)
    physical_loss /= len(paths)
    engine_loss = np.asarray([metrics["original_graphcast_loss"][key]
                              for key in ("baseline_rollout", "full_rollout")])
    drift = physical_loss - engine_loss
    bound = 1e-5 * np.maximum(np.abs(engine_loss), np.finfo(float).tiny)
    warnings = (["Finite FP64 observer/FP32 evaluator MSE difference exceeds1e-5 relative reference; retained"]
                if np.any(np.abs(drift) > bound) else [])
    return seal(root / f"checks/{stage}_{task:02d}.json", {
        "status": "complete", "campaign_hash": manifest["record_hash"], "count": shard["count"],
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "observer_step_seconds": observer.step_seconds,
        "grid_mse_roundoff": {"observer_fp64": physical_loss.tolist(), "evaluator_fp32": engine_loss.tolist(),
            "difference": drift.tolist(), "reference_bound": bound.tolist(), "warnings": warnings},
        "metrics": pinned(config.out_json),
        "statistics": [pinned(path) for path in paths]})


def expected_ids(payload):
    from datetime import datetime
    ids = []
    for entry in payload["initializations"]:
        stamp = entry if isinstance(entry, str) else entry["initialization_time"]
        ids.append(datetime.fromisoformat(stamp.replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ"))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate initialization dates")
    return ids


def statistics_paths(root, manifest, stage):
    paths = []
    for task, shard in enumerate(manifest["shards"][stage]):
        proof = unseal(root / f"checks/{stage}_{task:02d}.json")
        if proof["campaign_hash"] != manifest["record_hash"] or proof["count"] != shard["count"]:
            raise ValueError("Worker proof belongs to a different campaign/cohort")
        for record in proof["statistics"]:
            check(record)
            paths.append(Path(record["path"]))
    expected = expected_ids(read(manifest["cohorts"][stage]["path"]))
    if sorted(path.stem for path in paths) != sorted(expected):
        raise ValueError("Missing, duplicated or unexpected initialization statistics")
    return sorted(paths)


def load_statistics(path, manifest, stage):
    import json
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata["campaign_hash"] != manifest["record_hash"] or metadata["mode"] != stage:
            raise ValueError("Statistic provenance mismatch")
        if metadata["initialization_id"] != path.stem or metadata["status"] != "complete":
            raise ValueError("Incomplete/mismatched statistics")
        arrays = {name: archive[name] for name in archive.files if name != "metadata"}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Nonfinite saved statistics")
    return arrays, metadata


def fit(root, manifest):
    from src.analysis.spherical_smoothing import fit_filters
    paths = statistics_paths(root, manifest, "calibrate")
    aggregate, layout = None, None
    for path in paths:
        arrays, metadata = load_statistics(path, manifest, "calibrate")
        if aggregate is None:
            aggregate = {key: np.zeros_like(value) for key, value in arrays.items()}
            layout = metadata["layout"]
        if metadata["layout"] != layout or metadata["branches"] != ["GC", manifest["model_id"]]:
            raise ValueError("Calibration channel/predictor identity differs")
        for key, value in arrays.items():
            aggregate[key] += value
    filters = fit_filters(aggregate, layout, ("GC", manifest["model_id"]))
    return seal(root / "filters.json", {**filters, "campaign_hash": manifest["record_hash"],
        "development_count": len(paths), "calibration_inputs": [pinned(path) for path in paths]})


def ratio(numerator, denominator):
    """Undefined zero-truth bins are NaN with explicit counts, never floored."""
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def report(root, manifest):
    from src.analysis.smoothing_observer import atomic_npz
    from src.analysis.spherical_smoothing import json_digest
    paths = statistics_paths(root, manifest, "evaluate")
    frozen = unseal(root / "filters.json")
    aggregate, individual, means, closure, field_max = None, [], 0., 0., 0.
    minimum, negative = None, None
    for path in paths:
        arrays, metadata = load_statistics(path, manifest, "evaluate")
        if metadata["filter_digest"] != json_digest(frozen):
            raise ValueError("Annual statistics used a different filter")
        if aggregate is None:
            aggregate = {key: np.zeros_like(value) for key, value in arrays.items()}
            minimum = arrays["positive_field_minimum"].copy()
            negative = np.zeros_like(arrays["negative_fraction"])
        if metadata["layout"] != frozen["layout"]:
            raise ValueError("Annual layout differs from development")
        for key, value in arrays.items():
            aggregate[key] += value
        individual.append(arrays["exact_gc_loss"])
        means = max(means, float(np.max(np.abs(arrays["mean_drift"]))))
        raw_grid = arrays["grid_mse"]
        # A discrete orthogonal projector should close; finite errors are warnings.
        residual = raw_grid - arrays["projected_grid_mse"] - arrays["unresolved_grid_mse"]
        bound = 3e-4 * np.maximum(raw_grid, np.finfo(float).tiny)
        closure = max(closure, float(np.max(np.abs(residual) / bound)))
        field_max = max(field_max, float(np.max(np.abs(residual))))
        minimum = np.minimum(minimum, arrays["positive_field_minimum"])
        negative += arrays["negative_fraction"]
    pooled = {key: value / len(paths) for key, value in aggregate.items()}
    truth = pooled["truth_power"][:, None, None]
    prediction = pooled["prediction_power"]
    power_ratio = ratio(prediction, truth)
    correlation = ratio(pooled["cross_power"], np.sqrt(prediction * truth))
    mismatch = np.abs(power_ratio[..., 1:] - 1)
    valid = np.isfinite(mismatch)
    numerator = np.where(valid, mismatch, 0).sum(axis=(-2, -1))
    count = valid.sum(axis=(-2, -1))
    macro = ratio(numerator, count)
    # Derived undefined ratios may be NaN; raw finite statistics remain intact.
    atomic_npz(root / "annual_pooled.npz", {**pooled, "power_ratio": power_ratio,
        "correlation": correlation, "macro_power_mismatch": macro,
        "per_initialization_exact_gc_loss": np.stack(individual),
        "positive_field_minimum_over_starts": minimum,
        "negative_fraction_over_starts": negative / len(paths)},
        {"campaign_hash": manifest["record_hash"], "filter_digest": json_digest(frozen),
         "initialization_ids": [path.stem for path in paths], "branches": ["GC", manifest["model_id"]],
         "variants": ["raw", "gaussian", "band"], "layout": frozen["layout"],
         "ratio_policy": "pool raw powers across starts, then ratios; zero denominator NaN",
         "degree_macro": "equal weight across83 channels and degrees1..179; undefined bins excluded/count recorded"})
    warnings = []
    if closure > 1:
        warnings.append("Finite projection closure exceeds3e-4 relative reference; evaluation retained")
    if means > 1e-8:
        warnings.append("Finite absolute mean drift exceeds1e-8 in at least one physical channel; evaluation retained")
    if np.any(np.abs(correlation[np.isfinite(correlation)]) > 1 + 3e-4):
        warnings.append("Finite spectral correlation exceeds unit magnitude+3e-4; values retained")
    diagnostics = {"projection_closure_max_bound_ratio": closure,
        "projection_closure_reference_relative": 3e-4,
        "projection_closure_max_absolute": field_max,
        "mean_drift_max_absolute": means, "mean_drift_reference_absolute": 1e-8,
        "zero_truth_spherical_bins": int(np.sum(pooled["truth_power"] == 0)),
        "macro_valid_bin_count": count.tolist(), "warnings": warnings}
    result = seal(root / "report.json", {"status": "complete", "count": len(paths),
        "campaign_hash": manifest["record_hash"], "filter": pinned(root / "filters.json"),
        "pooled_statistics": pinned(root / "annual_pooled.npz"), "numerical_diagnostics": diagnostics,
        "branches": ["GC", manifest["model_id"]], "variants": ["raw", "gaussian", "band"],
        "exact_gc_loss_by_lead": pooled["exact_gc_loss"].tolist(),
        "macro_power_mismatch_by_lead": [[[(float(v) if np.isfinite(v) else None) for v in row]
                                           for row in lead] for lead in macro],
        "scientific_improvement_required": False, "no_clipping": True})
    plot_summary(root, pooled["exact_gc_loss"], macro, manifest["model_id"])
    if manifest.get("training_campaign"):
        from scripts.analyze_models.report_v24_scalar_lambda import build_if_complete
        build_if_complete(Path(manifest["training_campaign"]))
    return result


def plot_summary(root, loss, mismatch, model_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    days = np.arange(1, 41) / 4
    for branch, label in enumerate(("GC", model_id)):
        for variant, name in enumerate(("raw", "Gaussian", "band")):
            for axis, values in zip(axes, (loss, mismatch)):
                axis.plot(days, values[:, branch, variant], color=f"C{branch}",
                          linestyle=("-", "--", ":")[variant], label=f"{label} {name}")
    axes[0].set_ylabel("Exact GC grid MSE")
    axes[1].set_ylabel("Mean |forecast/truth power −1|\n83channels × degrees1–179")
    for axis in axes:
        axis.set_xlabel("Forecast lead (days)")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(root / "annual_summary.png", dpi=180)
    plt.close(fig)


def sbatch(root, manifest, stage, *, memory=8, minutes=20, array=None, dependency=None, task=0):
    gpu = stage in ("calibrate", "evaluate")
    args = ["sbatch", "--parsable", "--account=dabanin", f"--job-name=smth-{manifest['model_id']}-{stage}",
        f"--chdir={manifest['workspace']}", f"--mem={memory}G", f"--time={minutes}",
        f"--cpus-per-task={4 if gpu else 2}",
        f"--output={root}/logs/{stage}-%A_%a.out", f"--error={root}/logs/{stage}-%A_%a.err"]
    if gpu:
        args += ["--gres=gpu:1", "--constraint=gpu80",
                 "--qos=gpu-short" if minutes <= 480 else "--qos=gpu-medium"]
    if array is not None:
        args.append(f"--array={array}")
    if dependency:
        args.append(f"--dependency=afterok:{dependency}")
    return args + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(root),
                   manifest["source_root"], str(task)]


def job_id(record):
    if isinstance(record, dict):
        return str(record.get("job_id", record.get("jobid")))
    return str(record)


def submit(root, manifest):
    budget = manifest["resource_measurement"]
    first = submit_once(root, "calibrate-first", sbatch(root, manifest, "calibrate",
                        memory=budget["memory_gib"], minutes=budget["minutes"]))
    release = submit_once(root, "release-calibration", sbatch(root, manifest, "release-calibration",
                                                            dependency=job_id(first)))
    return {"calibrate_first": first, "release_calibration": release}


def resource_budget(root, manifest, stage):
    first = unseal(root / f"checks/{stage}_00.json")
    longest = max(shard["count"] for shard in manifest["shards"][stage][1:])
    # The first measurement includes compilation; count-scaling is conservative.
    minutes = max(62, math.ceil(first["elapsed_seconds"] * longest * 1.35 / 60 + 10))
    memory = max(16, 4 * math.ceil(first["max_rss_gib"] * 1.5 / 4))
    if minutes > 1440 or memory > 256:
        raise RuntimeError("Measured workload exceeds bounded automatic resource policy; revise shard packing")
    return seal(root / f"checks/{stage}_resources.json", {"minutes": minutes, "memory_gib": memory,
        "measurement": pinned(root / f"checks/{stage}_00.json"), "longest_shard_starts": longest,
        "basis": "retained first start including compile;1.35walltime+10min and1.5RSS rounded4GiB"})


def release(root, manifest, stage):
    budget = resource_budget(root, manifest, stage)
    remaining = submit_once(root, f"{stage}-remaining", sbatch(root, manifest, stage,
        memory=budget["memory_gib"], minutes=budget["minutes"],
        array=f"1-{len(manifest['shards'][stage])-1}"))
    if stage == "calibrate":
        fitted = submit_once(root, "fit", sbatch(root, manifest, "fit", dependency=job_id(remaining)))
        first = submit_once(root, "evaluate-first", sbatch(root, manifest, "evaluate", memory=32,
                            minutes=120, dependency=job_id(fitted)))
        continuation = submit_once(root, "release-evaluation", sbatch(root, manifest, "release-evaluation",
                                   dependency=job_id(first)))
        return {"remaining": remaining, "fit": fitted, "first": first, "release": continuation}
    reported = submit_once(root, "report", sbatch(root, manifest, "report", memory=8,
                          minutes=30, dependency=job_id(remaining)))
    return {"remaining": remaining, "report": reported}


def submit_training_sweep(training_campaign, family=None):
    training_campaign = Path(training_campaign).resolve()
    training = unseal(training_campaign / "manifest.json")
    if family is None:
        available = [name for name in ("full_gc", "gc_mamba")
                     if (training_campaign / f"training_complete_{name}.json").exists()]
        if available:
            return {name: submit_training_sweep(training_campaign, name) for name in available}
    suffix = f"_{family}" if family else ""
    selected = unseal(training_campaign / f"selected_arms{suffix}.json")
    completion = unseal(training_campaign / f"training_complete{suffix}.json")
    if completion["manifest_hash"] != training["record_hash"]:
        raise ValueError("Training completion provenance differs")
    for proof in completion["final_checkpoints"]:
        check(proof["checkpoint"])
    records = {}
    for task in selected["tasks"]:
        model_id = task["logical_id"]
        directory = training_campaign / "runs" / task["id"]
        root = training_campaign / "smoothing" / model_id
        if not (root / "manifest.json").exists():
            prepare(root, Path(training["workspace"]), Path(training["source_root"]),
                checkpoint=directory / "checkpoints/checkpoint_step00002500.pkl",
                run_config=directory / "run_config.json", model_id=model_id,
                development_manifest=training["development_manifest"]["path"],
                training_campaign=training_campaign)
        manifest = verify(root)
        records[model_id] = submit(root, manifest)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "submit", "calibrate", "fit", "evaluate", "report",
        "release-calibration", "release-evaluation", "submit-sweep"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--model-id", default="M1")
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--training-campaign", type=Path)
    parser.add_argument("--family", choices=("full_gc", "gc_mamba"))
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args(argv)
    if args.stage == "submit-sweep":
        if not args.training_campaign:
            parser.error("--training-campaign is required")
        result = submit_training_sweep(args.training_campaign, args.family)
    elif args.root is None:
        parser.error("--root is required")
    elif args.stage == "prepare":
        if args.source is None:
            parser.error("--source is required; freeze integrated source before prepare")
        result = prepare(args.root, args.workspace, args.source, checkpoint=args.checkpoint,
            run_config=args.run_config, model_id=args.model_id, development_manifest=args.development_manifest)
    else:
        root = args.root.resolve()
        manifest = verify(root)
        if args.stage in ("calibrate", "evaluate"):
            result = run_worker(root, manifest, args.stage, args.task)
        elif args.stage in ("release-calibration", "release-evaluation"):
            result = release(root, manifest, "calibrate" if args.stage == "release-calibration" else "evaluate")
        else:
            result = {"submit": submit, "fit": fit, "report": report}[args.stage](root, manifest)
    import json
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
