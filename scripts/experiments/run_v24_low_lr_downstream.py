#!/usr/bin/env python3
"""Join fresh full-GC training and the submitted M1-recipe GC–Mamba sweep."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import prepare_v24_mixed_final_comparison as mixed
from scripts.experiments import run_v24_m1_matched_downstream as previous
from scripts.experiments.frozen_runtime import verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/run_v24_low_lr_downstream.py"
WRAPPER = "scripts/experiments/v24_low_lr_downstream.slurm"
DEFAULT = "artifacts/evaluations/v24_low_lr_gc_lambda_annual_20261112"
CONSTANT_DEFAULT = "artifacts/evaluations/v24_constant_lr_gc_lambda_annual_20261112"
OVERLAYS = tuple(dict.fromkeys((*mixed.OVERLAYS, DRIVER, WRAPPER,
    "scripts/experiments/run_v24_m1_matched_downstream.py",
    "scripts/analyze_models/report_v24_scalar_lambda.py")))
LIMITS = (
    "All five full-GC arms use peakLR3e-7. Lambda0 retains its existing fresh trajectory; "
    "positive lambdas use fresh public-GC starts and new optimizers. No rejected1e-5 "
    "checkpoint is reused. GC–Mamba retains the submitted historical-M1 training recipe. "
    "All models complete2500 total updates, but feedback gradients, loss arithmetic, "
    "optimizer schedules and data order differ across families. Filters are fitted on "
    "the same32 development2022 starts and evaluated on the same122 annual2023 starts. "
    "Prior2023 exposure is disclosed. Finite drift and finite worse scores are warnings."
)
CONSTANT_LIMITS = (
    "All five full-GC lambda arms start fresh from public GC with new optimizers and "
    "constantLR3e-7 from the first update, with no warmup or decay. No prior lambda0 "
    "or cosine-schedule full-GC weights are reused in this comparison. GC–Mamba "
    "retains its submitted historical-M1 training recipe and separate LR schedule. "
    "All models complete2500 total updates; feedback gradients, loss arithmetic, "
    "optimizer schedules and data order differ across families. Filters are fitted "
    "on the same32 development2022 starts and evaluated on the same122 annual2023 "
    "starts. Prior2023 exposure is disclosed. Finite worse scores and drift are warnings."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_tasks(specs, *, constant=False):
    from scripts.analyze_models.comparison_policy import partition_models
    tasks = []
    for spec in specs:
        campaign = Path(spec["final_proof_path"]).parent.parent
        gcm = spec["model_family"] == "gc_mamba"
        tasks.append({"id": spec["training_run_id"], "logical_id": spec["id"],
            "model_family": spec["model_family"], "lambda": spec["lambda"],
            "learning_rate": spec["learning_rate"], "config": spec["training_config"],
            "training_proof_root": str(campaign / "checks"),
            "training_proof_pattern": f"{spec['training_run_id']}_{'total' if gcm else 'step'}*.json",
            "training_elapsed_field": "allocation_elapsed_seconds" if gcm else "elapsed_seconds"})
    require(len(tasks) == len({task["logical_id"] for task in tasks}) == 10, "Expected ten distinct arms")
    require(not partition_models(tasks)[1], "Rejected full-GC lineage entered annual selection")
    for family in ("full_gc", "gc_mamba"):
        require(sorted(task["lambda"] for task in tasks if task["model_family"] == family)
                == [0, .25, .5, .75, 1], "Incomplete family lambda coverage")
    require(all(task["learning_rate"] == 3e-7 for task in tasks if task["model_family"] == "full_gc"),
            "Full-GC annual arms must all use3e-7")
    if constant:
        gc_specs = [spec for spec in specs if spec["model_family"] == "full_gc"]
        require(all(spec.get("constant_learning_rate") is True for spec in gc_specs),
                "Constant comparison cannot include retained or cosine full-GC arms")
        require(len({str(Path(spec["final_proof_path"]).parent.parent) for spec in gc_specs}) == 1,
                "Constant full-GC arms must share the same fresh training campaign")
    return tasks


def prepare(root, workspace, gcm_training, full_gc_training, positive_full_gc_training,
            dependencies, reference_root, constant_full_gc_training=None):
    root, workspace, gcm_training, full_gc_training, reference_root = [Path(p).resolve()
        for p in (root, workspace, gcm_training, full_gc_training, reference_root)]
    positive_full_gc_training = Path(positive_full_gc_training).resolve() if positive_full_gc_training else None
    constant_full_gc_training = Path(constant_full_gc_training).resolve() if constant_full_gc_training else None
    require(not root.exists(), "Use a new annual bridge directory")
    require(bool(positive_full_gc_training) != bool(constant_full_gc_training), "Choose one fresh full-GC recipe")
    expected_dependencies = 2 if constant_full_gc_training else 3
    require(len(dependencies) == expected_dependencies and all(mixed.valid_dependency(job) for job in dependencies),
            "Supply one final training dependency per source")
    gcm_specs, gcm_inputs = mixed.gcm_specs(gcm_training)
    if constant_full_gc_training:
        retained_specs, retained_inputs = [], []
        new_specs, new_inputs = mixed.constant_full_gc_specs(constant_full_gc_training)
        training_sources = (gcm_training, constant_full_gc_training)
    else:
        retained_specs, retained_inputs = mixed.full_gc_specs(full_gc_training, weights=(0,))
        new_specs, new_inputs = mixed.positive_full_gc_specs(positive_full_gc_training)
        training_sources = (gcm_training, full_gc_training, positive_full_gc_training)
    specs = gcm_specs + retained_specs + new_specs
    tasks = selected_tasks(specs, constant=bool(constant_full_gc_training))
    gcm_manifest = unseal(gcm_training / "manifest.json")
    development = gcm_manifest["development_manifest"]
    check(development)
    for training in training_sources[1:]:
        other = unseal(training / "manifest.json")["development_manifest"]
        check(other)
        require(read(other["path"]) == read(development["path"]), "Development cohorts differ")
    reference_record = pinned(full_gc_training / "reference_smoothing.json")
    reference = Path(read(reference_record["path"])["campaign"])
    require(unseal(reference / "report.json")["status"] == "complete", "M1 annual reference incomplete")
    reference_report = pinned(reference / "report.json")
    parent_record = pinned(reference_root / "manifest.json")
    parent = unseal(parent_record["path"])
    inputs = [*gcm_inputs, *retained_inputs, *new_inputs, development,
              reference_record, reference_report, parent_record]
    for item in inputs:
        check(item)
    root.mkdir(parents=True)
    for name in ("logs", "submissions", "checks", "report"):
        (root / name).mkdir()
    source = mixed.freeze_extension(workspace, Path(parent["source_root"]), root / "source", overlays=OVERLAYS)
    limits = CONSTANT_LIMITS if constant_full_gc_training else LIMITS
    selected = seal(root / "selected_arms.json", {"tasks": tasks, "limits": limits,
        "sources": [pinned(path / "manifest.json") for path in training_sources]})
    return seal(root / "manifest.json", {
        "format": "v24_two_source_constant_lr_annual_v1" if constant_full_gc_training else "v24_three_source_low_lr_annual_v1",
        "workspace": str(workspace),
        "source_root": str(root / "source"), "source_digest": source["source_digest"],
        "selected_arms": pinned(root / "selected_arms.json"), "selected_hash": selected["record_hash"],
        "specs": specs, "training_dependencies": list(dependencies), "development_manifest": development,
        "directories": {task["logical_id"]: str(root / "smoothing" / task["logical_id"]) for task in tasks},
        "reference": str(reference), "limits": limits,
        "constant_full_gc_training": str(constant_full_gc_training) if constant_full_gc_training else None,
        "inputs": [*inputs, pinned(root / "selected_arms.json")],
        "resources": {"cpus": 2, "mem_gib": 8, "minutes": 30},
        "completion_policy": "External controller validates all10 endpoints, preserving GC–Mamba initial/final proof evidence without invoking superseded hooks."})


def verify(root):
    manifest = unseal(root / "manifest.json")
    require(verify_source(Path(manifest["source_root"]))["source_digest"] == manifest["source_digest"],
            "Annual bridge source differs")
    for item in manifest["inputs"]:
        check(item)
    require(selected_tasks(manifest["specs"], constant=bool(manifest.get("constant_full_gc_training")))
            == unseal(manifest["selected_arms"]["path"])["tasks"],
            "Annual selected arms differ from their pinned training specs")
    return manifest


def command(root, manifest, stage, dependencies=()):
    require(stage in ("start", "wait-calibration", "wait-evaluation", "wait-reports", "report"), "Unknown stage")
    require(all(mixed.valid_dependency(job) for job in dependencies), "Invalid dependency")
    budget = manifest["resources"]
    args = ["sbatch", "--parsable", "--account=dabanin", "--kill-on-invalid-dep=yes",
        f"--chdir={manifest['workspace']}", f"--job-name=low-lr-{stage}",
        f"--output={root}/logs/{stage}_%j.out", f"--error={root}/logs/{stage}_%j.err",
        f"--cpus-per-task={budget['cpus']}", f"--mem={budget['mem_gib']}G", f"--time={budget['minutes']}"]
    if dependencies:
        args.append("--dependency=afterok:" + ":".join(dict.fromkeys(map(str, dependencies))))
    return args + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(root), manifest["source_root"]]


def validate_endpoints(root, manifest):
    resolved = [mixed.final_sources(spec) for spec in manifest["specs"]]
    gcm_initial = []
    for spec, _, _ in resolved:
        if spec["model_family"] == "gc_mamba":
            final = unseal(spec["final_proof_path"])
            initial = unseal(final["initial_stage_proof"]["path"])
            gcm_initial.append(initial["initial_numerical_identity"]["residual_params"])
    matches = len(set(gcm_initial)) == 1
    record = seal(root / "checks/training_endpoints.json", {
        "status": "complete", "manifest_hash": manifest["record_hash"], "total_training_updates": 2500,
        "models": [{"model_id": spec["id"], "model_family": spec["model_family"],
            "lambda": spec["lambda"], "training_manifest_hash": spec["training_manifest_hash"],
            "final_checkpoint": sources[-1], "run_config": spec["run_config"], "proofs": proofs}
            for spec, sources, proofs in resolved],
        "gcm_initial_parameter_identity_matches": matches, "gcm_initial_parameter_hashes": gcm_initial,
        "warnings": [] if matches else ["Finite fresh GC–Mamba parameter identity differs across arms; inspect preserved initial checkpoints."],
        "scientific_improvement_required": False,
        "superseded_hooks_invoked": False})
    return resolved, record


def start(root, manifest):
    from scripts.experiments import run_v24_smoothing as smoothing
    resolved, audit = validate_endpoints(root, manifest)
    submissions = {}
    for spec, sources, _ in resolved:
        destination = Path(manifest["directories"][spec["id"]])
        if not (destination / "manifest.json").exists():
            smoothing.prepare(destination, Path(manifest["workspace"]), Path(manifest["source_root"]),
                checkpoint=Path(sources[-1]["path"]), run_config=Path(spec["run_config"]["path"]),
                model_id=spec["id"], development_manifest=Path(manifest["development_manifest"]["path"]),
                training_campaign=None)
        prepared = smoothing.verify(destination)
        require(prepared["checkpoint"] == sources[-1] and prepared["run_config"] == spec["run_config"],
                "Smoothing source identity differs")
        submissions[spec["id"]] = smoothing.submit(destination, prepared)
    write(root / "checks/smoothing_submissions.json", submissions)
    next_job = submit_once(root, "wait-calibration", command(root, manifest, "wait-calibration"))
    return {"status": "submitted", "endpoint_audit": audit["record_hash"],
            "smoothing": submissions, "annual_join": next_job}


def advance(root, manifest, stage):
    current, following, next_stage = {
        "wait-calibration": ("release-calibration.json", "release-evaluation.json", "wait-evaluation"),
        "wait-evaluation": ("release-evaluation.json", "report.json", "wait-reports"),
        "wait-reports": ("report.json", None, "report")}[stage]
    dependencies = previous.pending_release_dependencies(manifest, current, following)
    return submit_once(root, next_stage, command(root, manifest, next_stage, dependencies))


def report(root, manifest):
    from scripts.analyze_models import report_v24_scalar_lambda as reporter
    selected = unseal(manifest["selected_arms"]["path"])
    expected = [({"logical_id": "M1", "model_family": "historical_gc_mamba", "lambda": None},
                 Path(manifest["reference"]))]
    expected += [(task, Path(manifest["directories"][task["logical_id"]])) for task in selected["tasks"]]
    for _, directory in expected:
        require(unseal(directory / "report.json")["status"] == "complete", "Annual report incomplete")
    destination = root / "report"
    with (destination / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "complete.json").exists():
            return unseal(destination / "complete.json")
        return reporter._build(root, destination, selected, expected, extra_metadata={
            "comparison_limits": manifest["limits"], "bridge_manifest": pinned(root / "manifest.json"),
            "full_gc_peak_learning_rate": 3e-7,
            "full_gc_constant_learning_rate": bool(manifest.get("constant_full_gc_training")),
            "retained_full_gc_lambda": None if manifest.get("constant_full_gc_training") else 0,
            "fresh_full_gc_lambdas": [0, .25, .5, .75, 1] if manifest.get("constant_full_gc_training") else [.25, .5, .75, 1],
            "gcm_total_training_updates": 2500,
            "training_endpoint_audit": pinned(root / "checks/training_endpoints.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "submit", "start", "wait-calibration", "wait-evaluation", "wait-reports", "report"))
    parser.add_argument("--root", type=Path, default=ROOT / CONSTANT_DEFAULT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--gcm-training", type=Path, default=ROOT / mixed.GCM)
    parser.add_argument("--full-gc-training", type=Path, default=ROOT / mixed.FULL_GC)
    parser.add_argument("--positive-full-gc-training", type=Path)
    parser.add_argument("--constant-full-gc-training", type=Path)
    parser.add_argument("--reference-root", type=Path, default=ROOT / mixed.REFERENCE)
    parser.add_argument("--training-dependencies", nargs='+')
    args = parser.parse_args(); root = args.root.resolve()
    if args.stage == "prepare":
        if not args.training_dependencies:
            parser.error("prepare requires one final training dependency per source")
        if args.positive_full_gc_training is None and args.constant_full_gc_training is None:
            args.constant_full_gc_training = ROOT / mixed.CONSTANT_FULL_GC
        manifest = prepare(root, args.workspace, args.gcm_training, args.full_gc_training,
                           args.positive_full_gc_training, args.training_dependencies, args.reference_root,
                           args.constant_full_gc_training)
        result = {"manifest": str(root / "manifest.json"), "command": command(root, manifest, "start", manifest["training_dependencies"])}
    else:
        manifest = verify(root)
        if args.stage in ("plan", "submit"):
            planned = command(root, manifest, "start", manifest["training_dependencies"])
            result = planned if args.stage == "plan" else submit_once(root, "start", planned)
        else:
            require(ROOT == Path(manifest["source_root"]), "Run controller from its frozen source")
            result = start(root, manifest) if args.stage == "start" else report(root, manifest) if args.stage == "report" else advance(root, manifest, args.stage)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
