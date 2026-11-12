#!/usr/bin/env python3
"""Preserve annual smoothing and combine new GC–Mamba with unchanged full GC."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/run_v24_m1_matched_downstream.py"
WRAPPER = "scripts/experiments/v24_m1_matched_downstream.slurm"
OLD_TRAINING = "artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107"
DEFAULT = "artifacts/evaluations/v24_m1matched_gc_lambda_annual_20261112"
FULL_GC_FINISH = "17049165"
LIMITS = (
    "GC–Mamba uses a fresh historical-M1 training recipe; full GC retains its "
    "existing training trajectory and learning rates. Both have2500 total updates "
    "but differ in feedback gradients, loss arithmetic, optimizer schedule and "
    "training order. The same122 annual2023 starts and independently development-"
    "fitted filters are compared. Prior2023 exposure is disclosed. Full-GC lambda0 "
    "retains peak3e-7 versus1e-5 for positive lambda. Finite numerical differences "
    "remain advisory and do not gate reporting."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def job_id(record):
    return str(record["job_id"])


def prepare_bridge(training_campaign, training):
    workspace = Path(training["workspace"])
    root = workspace / DEFAULT
    old = workspace / OLD_TRAINING
    if (root / "manifest.json").exists():
        existing = verify(root)
        require(existing["gcm_training"] == pinned(training_campaign / "manifest.json"),
                "Existing annual bridge belongs to another GC–Mamba campaign")
        return root, existing
    require(not root.exists(), "Incomplete annual bridge exists; inspect before retrying")
    old_record = pinned(old / "manifest.json")
    old_manifest = unseal(old_record["path"])
    old_selection = pinned(old / "selected_arms_full_gc.json")
    full_tasks = unseal(old_selection["path"])["tasks"]
    tasks, directories = [], {}
    for task in training["tasks"]:
        item = copy.deepcopy(task)
        item.update(training_proof_root=str(training_campaign / "checks"),
            training_proof_pattern=f"{task['id']}_total*.json",
            training_elapsed_field="allocation_elapsed_seconds")
        tasks.append(item)
        directories[item["logical_id"]] = str(training_campaign / "smoothing" / item["logical_id"])
    for task in full_tasks:
        item = copy.deepcopy(task)
        item.update(training_proof_root=str(old / "checks"),
                    training_proof_pattern=f"{task['id']}_step*.json")
        tasks.append(item)
        directories[item["logical_id"]] = str(old / "smoothing" / item["logical_id"])
    require(len(tasks) == len({task["logical_id"] for task in tasks}) == 10,
            "Annual bridge requires five arms per family")
    reference_record = pinned(old / "reference_smoothing.json")
    reference = Path(read(reference_record["path"])["campaign"])
    root.mkdir(parents=True)
    for name in ("logs", "submissions", "checks", "report"):
        (root / name).mkdir()
    selected = seal(root / "selected_arms.json", {"tasks": tasks,
        "sources": [pinned(training_campaign / "manifest.json"), old_selection],
        "limits": LIMITS})
    manifest = seal(root / "manifest.json", {
        "format": "v24_m1matched_annual_bridge_v1", "workspace": str(workspace),
        "source_root": training["source_root"], "source_digest": training["source_digest"],
        "gcm_training": pinned(training_campaign / "manifest.json"),
        "full_gc_training": old_record, "full_gc_finish_job": FULL_GC_FINISH,
        "full_gc_downstream_record": str(old / "downstream_submitted_full_gc.json"),
        "selected_arms": pinned(root / "selected_arms.json"),
        "directories": directories, "reference": str(reference), "limits": LIMITS,
        "inputs": [pinned(training_campaign / "manifest.json"), old_record, old_selection,
                   reference_record, pinned(root / "selected_arms.json")],
        "full_gc_training_manifest_hash": old_manifest["record_hash"],
        "selected_hash": selected["record_hash"]})
    return root, manifest


def verify(root):
    manifest = unseal(root / "manifest.json")
    require(verify_source(Path(manifest["source_root"]))["source_digest"] == manifest["source_digest"],
            "Annual bridge source changed")
    for item in manifest["inputs"]:
        check(item)
    return manifest


def command(root, manifest, stage, dependencies=()):
    require(stage in ("wait-calibration", "wait-evaluation", "wait-reports", "report"),
            "Unknown annual bridge stage")
    args = ["sbatch", "--parsable", "--account=dabanin", "--kill-on-invalid-dep=yes",
        f"--chdir={manifest['workspace']}", f"--job-name=m1matched-{stage}",
        f"--output={root}/logs/{stage}_%j.out", f"--error={root}/logs/{stage}_%j.err",
        "--cpus-per-task=2", "--mem=8G", "--time=30"]
    if dependencies:
        require(all(str(value).isdigit() for value in dependencies), "Invalid report dependency")
        args.append("--dependency=afterok:" + ":".join(dict.fromkeys(map(str, dependencies))))
    return args + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(root), manifest["source_root"]]


def submit_smoothing(training_campaign):
    """Called by the new campaign's finish hook; never changes the old campaign."""
    from scripts.experiments import run_v24_smoothing as smoothing
    training_campaign = Path(training_campaign).resolve()
    training = unseal(training_campaign / "manifest.json")
    require(ROOT == Path(training["source_root"]), "Run finish hook from frozen training source")
    completion = unseal(training_campaign / "training_complete_gc_mamba.json")
    require(completion["manifest_hash"] == training["record_hash"], "Training completion differs")
    root, bridge = prepare_bridge(training_campaign, training)
    records = {}
    for task in training["tasks"]:
        proof = unseal(training_campaign / "checks" / f"{task['id']}_total00002500.json")
        require(proof["status"] == "complete" and proof["completed_steps"] == 2000
                and proof["total_training_updates"] == 2500
                and proof["manifest_hash"] == training["record_hash"], "New model endpoint incomplete")
        check(proof["checkpoint"]); check(proof["run_config"])
        destination = Path(bridge["directories"][task["logical_id"]])
        if not (destination / "manifest.json").exists():
            smoothing.prepare(destination, Path(training["workspace"]), Path(training["source_root"]),
                checkpoint=Path(proof["checkpoint"]["path"]), run_config=Path(proof["run_config"]["path"]),
                model_id=task["logical_id"],
                development_manifest=Path(training["development_manifest"]["path"]),
                training_campaign=None)
        manifest = smoothing.verify(destination)
        require(manifest["checkpoint"] == proof["checkpoint"] and manifest["run_config"] == proof["run_config"],
                "Existing smoothing campaign points at different weights")
        records[task["logical_id"]] = smoothing.submit(destination, manifest)
    write(root / "checks/gcm_smoothing_submissions.json", records)
    old_submitted = Path(bridge["full_gc_downstream_record"])
    dependencies = [] if old_submitted.exists() else [bridge["full_gc_finish_job"]]
    if old_submitted.exists():
        record = unseal(old_submitted)
        require(record["status"] == "submitted" and record["manifest_hash"] == bridge["full_gc_training_manifest_hash"],
                "Full-GC downstream completion identity differs")
    submitted = submit_once(root, "wait-calibration", command(root, bridge, "wait-calibration", dependencies))
    return {"smoothing": records, "annual_join": submitted, "annual_root": str(root)}


def pending_release_dependencies(manifest, current_record, next_record):
    """Advance only through actual submission records; no guessed future IDs."""
    dependencies = []
    for directory in manifest["directories"].values():
        directory = Path(directory)
        if (directory / "report.json").exists():
            require(unseal(directory / "report.json")["status"] == "complete", "Incomplete annual report")
            continue
        if next_record and (directory / "submissions" / next_record).exists():
            continue
        record = read(directory / "submissions" / current_record)
        identity = job_id(record)
        require(identity.isdigit(), "Invalid smoothing submission ID")
        dependencies.append(identity)
    return dependencies


def advance(root, manifest, stage):
    stages = {
        "wait-calibration": ("release-calibration.json", "release-evaluation.json", "wait-evaluation"),
        "wait-evaluation": ("release-evaluation.json", "report.json", "wait-reports"),
        "wait-reports": ("report.json", None, "report"),
    }
    current, subsequent, next_stage = stages[stage]
    dependencies = pending_release_dependencies(manifest, current, subsequent)
    return submit_once(root, next_stage, command(root, manifest, next_stage, dependencies))


def report(root, manifest):
    from scripts.analyze_models import report_v24_scalar_lambda as reporter
    selected = unseal(manifest["selected_arms"]["path"])
    reference = Path(manifest["reference"])
    expected = [({"logical_id": "M1", "model_family": "historical_gc_mamba", "lambda": None}, reference)]
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
            "unchanged_full_gc": True, "gcm_total_training_updates": 2500})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("submit-smoothing", "wait-calibration", "wait-evaluation", "wait-reports", "report"))
    parser.add_argument("--training-campaign", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.stage == "submit-smoothing":
        if args.training_campaign is None:
            parser.error("submit-smoothing requires --training-campaign")
        result = submit_smoothing(args.training_campaign)
    else:
        if args.root is None:
            parser.error("annual join stages require --root")
        root = args.root.resolve(); manifest = verify(root)
        require(ROOT == Path(manifest["source_root"]), "Run annual join from frozen source")
        result = report(root, manifest) if args.stage == "report" else advance(root, manifest, args.stage)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
