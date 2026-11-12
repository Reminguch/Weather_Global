#!/usr/bin/env python3
"""Small matched MSE/spherical-spectrum evaluation of saved, unaveraged weights."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments.frozen_runtime import freeze_source, verify_source
from scripts.experiments.run_v24_full_gc_ar import RawSphericalObserver
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

PARENT = "artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107"
M1 = "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba"
TAG = "v24_raw_checkpoint_pilot_20261110"
DEFAULT = "artifacts/evaluations/" + TAG
DRIVER = "scripts/experiments/run_v24_raw_checkpoint_pilot.py"
WRAPPER = "scripts/experiments/v24_raw_checkpoint_pilot.slurm"
POWER_KEYS = ("truth_power", "prediction_power", "error_power", "cross_power")


def pilot_cohort(development):
    """Take the earliest available development start in each quarter, without scores."""
    entries = development["initializations"]
    indices = []
    for quarter in range(4):
        candidates = [(entry["initialization_time"], i) for i, entry in enumerate(entries)
                      if (int(entry["initialization_time"][5:7]) - 1) // 3 == quarter]
        if not candidates:
            raise ValueError("Development cohort must cover every quarter")
        indices.append(min(candidates)[1])
    selected = [entries[i] for i in indices]
    if any(not entry["initialization_time"].startswith("2022-") for entry in selected):
        raise ValueError("Pilot requires 2022 development dates")
    return {**development, "role": "development_pilot", "initializations": selected,
            "source_chosen_idx": [development["source_chosen_idx"][i] for i in indices],
            "pilot_parent_indices": indices,
            "selection_rule": "Earliest available development32 start in each calendar quarter"}


def model_records(workspace):
    run = workspace / M1
    models = []
    for name, label, relative in (
        ("M1_raw", "M1 raw (main step 2000)", "checkpoints/checkpoint_step00002000.pkl"),
        ("M1_swa", "M1 SWA (main steps 500–2000)", "swa/swa_step00500-02000.pkl"),
    ):
        models.append({"id": name, "label": label, "model_family": "gc_mamba", "lambda": None,
                       "step": 2000, "total_training_updates": 2500,
                       "checkpoint": pinned(run / relative), "run_config": pinned(run / "run_config.json"),
                       "selection": "Historical prescribed SWA" if name.endswith("swa") else "Final raw endpoint"})
    for family in ("gc_mamba", "full_gc"):
        tasks = unseal(workspace / PARENT / f"selected_arms_{family}.json")["tasks"]
        for task in tasks:
            run = workspace / PARENT / "runs" / task["id"]
            models.append({"id": task["logical_id"], "label": f"{'GC–Mamba' if family == 'gc_mamba' else 'Full GC'} λ={task['lambda']:g}",
                           "model_family": family, "lambda": task["lambda"], "step": 1250,
                           "learning_rate": task["learning_rate"], "training_run_id": task["id"],
                           "checkpoint": pinned(run / "checkpoints/checkpoint_step00001250.pkl"),
                           "run_config": pinned(run / "run_config.json"),
                           "selection": "Latest common completed raw endpoint; planned training endpoint is 2500"})
    return models


def prepare(root, workspace):
    workspace = workspace.resolve()
    if root.exists():
        raise FileExistsError("Use a new campaign directory; existing evidence is immutable")
    parent = unseal(workspace / PARENT / "manifest.json")
    development_record = parent["development_manifest"]
    check(development_record)
    cohort = pilot_cohort(read(development_record["path"]))
    models = model_records(workspace)
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "evaluation"):
        (root / sub).mkdir()
    cohort_path = root / "cohort.json"
    write(cohort_path, cohort)
    config = read(models[0]["run_config"]["path"])
    baseline = pinned(workspace / config["data"]["baseline_checkpoint"])
    stats_dir = (workspace / config["data"]["stats_dir"]).resolve()
    inputs = [development_record, baseline, pinned(workspace / PARENT / "selected_arms_gc_mamba.json"),
              pinned(workspace / PARENT / "selected_arms_full_gc.json")]
    inputs += [pinned(stats_dir / name) for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")]
    for model in models:
        cfg = read(model["run_config"]["path"])
        if pinned(workspace / cfg["data"]["baseline_checkpoint"])["sha256"] != baseline["sha256"]:
            raise ValueError("Models do not use the same public GC backbone")
        if (workspace / cfg["data"]["stats_dir"]).resolve() != stats_dir:
            raise ValueError("Models do not use the same statistics")
    snapshot = freeze_source(workspace, root / "source")
    return seal(root / "manifest.json", {
        "format": "v24_raw_checkpoint_pilot_v1", "workspace": str(workspace),
        "source_root": str(root / "source"), "source_digest": snapshot["source_digest"],
        "models": models, "cohort": pinned(cohort_path), "target_steps": 40,
        "development_data": parent["development_data"], "baseline": baseline,
        "stats_dir": str(stats_dir), "inputs": inputs,
        "image_root": str(workspace / "plots/analyze_models/images/resolution_eval" / root.name),
        "data_root": str(workspace / "plots/analyze_models/data/resolution_eval" / root.name),
        "protocol": {"eval_mode": "cold_full", "residual_state_init": "zero", "warmup_steps": 0,
                     "samples": 4, "year": 2022, "lmax": 179, "smoothing": "none",
                     "checkpoint_selection": "Predetermined endpoints; no pilot-score selection",
                     "finite_variance_policy": "warning only; record values, bounds and observed baseline spread",
                     "limitations": "Four-date interim diagnostic, not annual skill. GCFT λ0 LR3e-7 differs from positive-λ LR1e-5."},
        "resources": {"gpu": 1, "cpus": 4, "mem_gib": 16, "minutes": 62,
                      "evidence": "17022624: 1 start/40 leads, 9m57s,9.17GiB; 17024230_1:11 starts,33m03s,9.88GiB; both include more expensive smoothing statistics. 62min avoids documented gputest routing/QOS restriction at <=1h; expected execution10–20min."}})


class PilotObserver(RawSphericalObserver):
    def finish_sample(self):
        from src.analysis.smoothing_observer import atomic_npz
        base = self.base
        if base.sample is None or base.steps != base.target_steps or len(base.records) != base.target_steps:
            raise ValueError("Incomplete spherical forecast")
        arrays = {key: np.stack([r[key] for r in base.records]) for key in POWER_KEYS}
        metadata = {**base.sample, "status": "complete", "layout": base.layout.metadata(),
                    "branches": ["public_gc", base.identity["model_id"]], "lmax": base.lmax,
                    "spectrum": "physical spherical fitted coefficients; degree zero retained"}
        atomic_npz(base.path, arrays, metadata)
        base.sample = base.accumulated = None
        base.records = []


def pool_spectra(dest, manifest, model):
    from scripts.experiments.run_v24_smoothing import expected_ids
    expected = sorted(expected_ids(read(manifest["cohort"]["path"])))
    paths = sorted((dest / "spectra").glob("*.npz"))
    if [path.stem for path in paths] != expected:
        raise ValueError("Missing or unexpected spectrum dates")
    total, layout = None, None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if (meta["campaign_hash"] != manifest["record_hash"] or meta["checkpoint_sha256"] != model["checkpoint"]["sha256"]
                    or meta["model_id"] != model["id"] or meta["initialization_id"] != path.stem
                    or meta["status"] != "complete" or meta["lead_hours"] != list(range(6, 241, 6))
                    or meta["branches"] != ["public_gc", model["id"]]
                    or meta["cohort_sha256"] != manifest["cohort"]["sha256"]
                    or meta["source_digest"] != manifest["source_digest"] or meta["lmax"] != 179):
                raise ValueError("Spectrum provenance/coverage mismatch")
            if layout is not None and layout != meta["layout"]:
                raise ValueError("Spectrum channel layout differs")
            layout = meta["layout"]
            arrays = {key: np.asarray(data[key], dtype=np.float64) for key in POWER_KEYS}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Nonfinite spectrum")
            channels = len(layout["channels"])
            for key, value in arrays.items():
                shape = (40, channels, 180) if key == "truth_power" else (40, 2, channels, 180)
                if value.shape != shape:
                    raise ValueError(f"Spectrum shape mismatch: {key}")
            if total is None:
                total = {key: np.zeros_like(value) for key, value in arrays.items()}
            for key in total:
                total[key] += arrays[key] / len(paths)
    pooled = dest / "pooled_spectra.npz"
    np.savez_compressed(pooled, **total, metadata=np.asarray(json.dumps({
        "layout": layout, "count": len(paths), "initialization_ids": expected,
        "lead_hours": list(range(6, 241, 6)), "campaign_hash": manifest["record_hash"],
        "checkpoint_sha256": model["checkpoint"]["sha256"], "branches": ["public_gc", model["id"]],
        "cohort_sha256": manifest["cohort"]["sha256"], "source_digest": manifest["source_digest"], "lmax": 179})))
    return pooled


def evaluate(root, manifest, task):
    if "reuse" in manifest["models"][task]:
        raise ValueError("This model reuses an existing evaluation; do not forecast it again")
    from src.models.graphcast.training.core.model import load_stats
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from scripts.experiments.run_v24_scalar_lambda import gpu_required
    gpu_required()
    model = manifest["models"][task]
    for item in [model["checkpoint"], model["run_config"], manifest["cohort"], *manifest["inputs"]]:
        check(item)
    dest = root / "evaluation" / model["id"]
    if dest.exists():
        raise FileExistsError(f"Use a new output for retries: {dest}")
    dest.mkdir()
    allowed = {field.name for field in fields(V24IlyaEvalConfig)}
    architecture = {k: v for k, v in read(model["run_config"]["path"])["architecture"].items() if k in allowed}
    observer = PilotObserver(dest / "spectra", load_stats(Path(manifest["stats_dir"]))["diffs_stddev_by_level"],
        {"campaign_hash": manifest["record_hash"], "model_id": model["id"],
         "checkpoint_sha256": model["checkpoint"]["sha256"], "cohort_sha256": manifest["cohort"]["sha256"],
         "source_digest": manifest["source_digest"]})
    config = V24IlyaEvalConfig(**architecture, ckpt=Path(model["checkpoint"]["path"]),
        out_json=dest / "evaluation.json", model_id=model["id"], model_family=model["model_family"],
        ckpt_in=Path(manifest["baseline"]["path"]), stats_dir=Path(manifest["stats_dir"]),
        data_path=manifest["development_data"], initialization_manifest=Path(manifest["cohort"]["path"]),
        target_steps=40, n_samples=4, val_year=2022, train_start_year=2015, train_end_year=2021,
        warmup_steps=0, eval_mode="cold_full", residual_state_init="zero", omit_rms_bias=True)
    started = time.monotonic()
    result = evaluate_v24_Ilya(config, prediction_observer=observer)
    expected_dates = [item["initialization_time"] for item in read(manifest["cohort"]["path"])["initializations"]]
    if (result["evaluated_samples"] != 4 or result["evaluation_status"] != "complete"
            or not result["metrics_complete"] or result["remaining_samples"] != 0
            or result["initialization_manifest_sha256"] != manifest["cohort"]["sha256"]
            or result["global_initialization_times"] != expected_dates):
        raise ValueError("Incomplete exact-MSE coverage")
    loss = result["original_graphcast_loss"]
    if not all(np.isfinite(np.asarray(loss[key])).all() for key in
               ("baseline_rollout", "full_rollout", "baseline_per_step", "full_per_step")):
        raise ValueError("Nonfinite exact GC MSE")
    pooled = pool_spectra(dest, manifest, model)
    check(model["checkpoint"])
    return seal(dest / "complete.json", {"status": "complete", "manifest_hash": manifest["record_hash"],
        "checkpoint": model["checkpoint"], "evaluation": pinned(config.out_json), "pooled_spectra": pinned(pooled),
        "count": 4, "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2})


def submission_command(root, manifest, stage, dependency=None):
    args = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={manifest['workspace']}",
            "--kill-on-invalid-dep=yes", f"--job-name=rawpilot-{stage}",
            f"--output={root}/logs/{stage}_%A_%a.out", f"--error={root}/logs/{stage}_%A_%a.err"]
    if stage == "evaluate":
        indices = [str(i) for i, model in enumerate(manifest["models"]) if "reuse" not in model]
        if not indices:
            raise ValueError("No new evaluations requested")
        args += ["--gres=gpu:1", "--constraint=gpu80", "--qos=gpu-short", "--cpus-per-task=4",
                 "--mem=16G", "--time=62", f"--array={','.join(indices)}"]
    else:
        args += ["--cpus-per-task=2", "--mem=8G", "--time=15"]
    if dependency:
        args.append(f"--dependency=afterok:{dependency}")
    return args + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(root), manifest["source_root"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "prepare-legacy", "plan", "submit", "evaluate", "report"))
    parser.add_argument("--root", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    args = parser.parse_args(); root = args.root.resolve()
    if args.stage in ("prepare", "prepare-legacy"):
        if args.stage == "prepare-legacy":
            from scripts.experiments.prepare_v24_legacy_raw_pilot import prepare_legacy
            manifest = prepare_legacy(root, args.workspace, args.reference_root)
        else:
            manifest = prepare(root, args.workspace)
        result = {"manifest": str(root / "manifest.json"), "models": len(manifest["models"]), "cohort": read(manifest["cohort"]["path"])["initializations"]}
    else:
        manifest = unseal(root / "manifest.json")
        if verify_source(manifest["source_root"])["source_digest"] != manifest["source_digest"]:
            raise ValueError("Frozen runtime digest mismatch")
        if args.stage in ("evaluate", "report") and ROOT != Path(manifest["source_root"]):
            raise ValueError("Execute evaluation/report from the frozen source")
        if args.stage == "plan":
            dependency = ":".join(["<evaluation-job>", *manifest.get("report_dependencies", [])])
            result = {"evaluate": submission_command(root, manifest, "evaluate"),
                      "report": submission_command(root, manifest, "report", dependency)}
        elif args.stage == "submit":
            evaluation = submit_once(root, "evaluate", submission_command(root, manifest, "evaluate"))
            job = str(evaluation.get("job_id", evaluation.get("jobid"))) if isinstance(evaluation, dict) else str(evaluation)
            dependency = ":".join([job, *manifest.get("report_dependencies", [])])
            result = {"evaluate": evaluation, "report": submit_once(root, "report", submission_command(root, manifest, "report", dependency))}
        elif args.stage == "evaluate":
            result = evaluate(root, manifest, args.task)
        else:
            from scripts.analyze_models.report_v24_raw_checkpoint_pilot import report
            result = report(root)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
