#!/usr/bin/env python3
"""Four fresh positive-lambda full-GC runs at the prescribed peak LR 3e-7.

The existing lambda-zero trajectory is a separate preserved run. This driver
reuses the original scalar computation with no calibration or profile stage.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_v24_scalar_lambda as scalar
from scripts.experiments.frozen_runtime import FORMAT as SOURCE_FORMAT, _digest, file_sha256, verify_source
from scripts.experiments.run_v24_mgda import call, check, pinned, read, seal, unseal, write

FORMAT = "v24_full_gc_low_lr_restart_v1"
DEFAULT = "artifacts/checkpoints/v24_Ilya/res1_full_gc_positive_lambda_lr3em7_20261112"
REFERENCE = "artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107"
DRIVER = "scripts/experiments/run_v24_full_gc_low_lr_restart.py"
WRAPPER = "scripts/experiments/v24_full_gc_low_lr_restart.slurm"
OVERLAYS = (DRIVER, WRAPPER)
LAMBDAS = (.25, .5, .75, 1.)
ENDPOINTS = (1250, 2500)
HISTORY_STEPS = (500, 1000, 1500, 2000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_config(template, experiment, task):
    """Change only the prescribed rate and fresh output identity."""
    config = copy.deepcopy(template)
    require(config["model_family"] == "full_gc", "Expected a full-GC template")
    require(config["objective"]["spectral_excess_weight"] == task["lambda"], "Template lambda differs")
    config["optimizer"]["learning_rate"] = 3e-7
    config["optimizer"]["end_learning_rate"] = 3e-8
    config["output"] = {"output_root": str(experiment / "runs"), "run_name": task["id"]}
    return config


def validate_config(config, task):
    optimizer = config["optimizer"]
    sequence = config["sequence"]
    require(config["model_family"] == "full_gc", "Model family differs")
    require(task["lambda"] in LAMBDAS and task["learning_rate"] == 3e-7
            and task["resume_step"] == 0, "Fresh task policy differs")
    require(optimizer["learning_rate"] == 3e-7 and optimizer["end_learning_rate"] == 3e-8,
            "Expected prescribed full-GC learning rates")
    require(optimizer["max_steps"] == 2500 and optimizer["warmup_steps"] == 100
            and optimizer["learning_rate_schedule"] == "cosine" and optimizer["seed"] == 22,
            "Continuous 2500-update schedule differs")
    require(tuple(sequence[key] for key in ("segment_steps", "bptt_steps", "ar_tail_k")) == (96, 24, 20)
            and sequence["feedback_mode"] == "closed_loop", "Full-feedback K20 recipe differs")
    require(config["memory"]["bptt_backend"] == "explicit_weather_reverse_vjp"
            and config["distributed"]["per_device_batch_size"] == 4, "Full-GC execution shape differs")
    require(config["objective"]["loss_kind"] == "gc_mse_amse_excess"
            and config["objective"]["spectral_excess_weight"] == task["lambda"], "Scalar objective differs")
    require(config["data"].get("train_order_manifest"), "Frozen training order is required")
    require(not config.get("init_from") and not config.get("resume"), "Fresh runs cannot inherit trained weights")
    require(config["output"]["run_name"] == task["id"], "Fresh output identity differs")


def freeze_extension(workspace, parent, destination):
    """Copy all original computational source byte-for-byte, adding orchestration."""
    if destination.exists():
        raise FileExistsError(destination)
    previous = verify_source(parent)
    files = {}
    for relative in sorted(set(previous["files"]) | set(OVERLAYS)):
        source = (workspace if relative in OVERLAYS else parent) / relative
        digest = file_sha256(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(file_sha256(target) == file_sha256(source) == digest,
                f"Source changed while freezing: {source}")
        files[relative] = digest
    require(verify_source(parent)["source_digest"] == previous["source_digest"], "Parent source changed")
    write(destination / "snapshot.json", {
        "format": SOURCE_FORMAT, "workspace": str(workspace), "files": files,
        "source_digest": _digest(files), "parent_snapshot": pinned(parent / "snapshot.json"),
        "overlays": list(OVERLAYS)})
    return verify_source(destination)


def prepare(experiment, workspace, reference=None):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    reference = Path(reference).resolve() if reference else workspace / REFERENCE
    if experiment.exists():
        raise FileExistsError("Use a new campaign directory; existing campaigns remain immutable")
    original_record = pinned(reference / "manifest.json")
    original = unseal(original_record["path"])
    selection_record = pinned(reference / "selected_arms_full_gc.json")
    selected = unseal(selection_record["path"])["tasks"]
    require([task["lambda"] for task in selected] == [0., *LAMBDAS], "Original full-GC selection differs")
    require(selected[0]["learning_rate"] == 3e-7, "Existing lambda-zero reference rate differs")
    inputs = [original_record, selection_record, *original["inputs"], selected[0]["config"]]
    for item in inputs:
        check(item)
    lambda_zero = read(selected[0]["config"]["path"])
    order = pinned(lambda_zero["data"]["train_order_manifest"])
    inputs.append(order)
    parent = Path(original["source_root"])
    require(verify_source(parent)["source_digest"] == original["source_digest"], "Parent computational source differs")
    for directory in ("configs", "checks", "logs", "submissions"):
        (experiment / directory).mkdir(parents=True)
    tasks = []
    for prior in selected[1:]:
        check(prior["config"])
        inputs.append(prior["config"])
        weight = prior["lambda"]
        task = {"id": f"gcft_s22_l{round(weight * 100):03d}_lr3em7_fresh_20261112",
                "logical_id": f"gcft_s22_l{round(weight * 100):03d}",
                "model_family": "full_gc", "lambda": weight, "learning_rate": 3e-7,
                "resume_step": 0, "role": "user_prescribed_fresh_low_lr",
                "total_training_updates": 2500, "template_config": prior["config"]}
        config = make_config(read(prior["config"]["path"]), experiment, task)
        validate_config(config, task)
        require(config["data"]["train_order_manifest"] == order["path"], "Training order changed")
        # The existing accepted lambda-zero recipe differs only in lambda and output.
        matched_zero = copy.deepcopy(lambda_zero)
        matched_zero["objective"]["spectral_excess_weight"] = weight
        matched_zero["output"] = config["output"]
        require(config == matched_zero, "Fresh positive-lambda recipe differs from accepted lambda-zero recipe")
        path = experiment / "configs" / f"{task['id']}.json"
        write(path, config)
        load_training_config(path)
        task["config"] = pinned(path)
        inputs.append(task["config"])
        tasks.append(task)
    source = freeze_extension(workspace, parent, experiment / "source")
    manifest = seal(experiment / "manifest.json", {
        "format": FORMAT, "workspace": str(workspace), "experiment": str(experiment),
        "source_root": str(experiment / "source"), "source_digest": source["source_digest"],
        "parent_source_digest": original["source_digest"], "reference_campaign": original_record,
        "inputs": inputs, "tasks": tasks, "retained_lambda_zero": selected[0],
        "development_manifest": original["development_manifest"],
        "development_data": original["development_data"],
        "completion_hook": None,
        "policy": {"lambdas": list(LAMBDAS), "peak_learning_rate": 3e-7, "end_learning_rate": 3e-8,
            "seed": 22, "max_steps": 2500, "warmup_steps": 100, "endpoints": list(ENDPOINTS),
            "fresh_initialization": "Public GraphCast-small; no trained positive-lambda checkpoint reuse",
            "training_order": order, "finite_diagnostics": "warning", "scientific_improvement_required": False,
            "automatic_retries": 0, "profile_or_calibration_updates": 0},
        "resources": {"training": {"memory_gib": 384, "minutes": 1800, "cpus": 8, "gpu": 1},
            "basis": "Completed full-GC phase0 tasks17049162_0–4 took19:57:40–20:07:45 with peak224.534–228.220GiB host RSS; preserve proven1gpu80/8CPU/384GiB/30h per1250updates, without a new qualification job."}})
    selection = {"tasks": tasks, "manifest_hash": manifest["record_hash"],
                 "selection": "User-prescribed fresh positive-lambda3e-7 runs; no score-based selection"}
    seal(experiment / "selected_arms_full_gc.json", selection)
    seal(experiment / "selected_arms.json", selection)
    write(experiment / "submission_plan.json", submission_plan(experiment, manifest))
    return manifest


def verify(experiment, *, frozen=False):
    manifest = unseal(experiment / "manifest.json")
    require(manifest["format"] == FORMAT and Path(manifest["experiment"]) == experiment, "Campaign identity differs")
    if frozen:
        require(ROOT == Path(manifest["source_root"]), "Execute the frozen campaign source")
    require(verify_source(manifest["source_root"])["source_digest"] == manifest["source_digest"], "Frozen source differs")
    for item in manifest["inputs"]:
        check(item)
    selection = unseal(experiment / "selected_arms_full_gc.json")
    require(selection["manifest_hash"] == manifest["record_hash"] and selection["tasks"] == manifest["tasks"],
            "Selected training identities differ")
    for task in manifest["tasks"]:
        validate_config(read(task["config"]["path"]), task)
    return manifest


def job_command(experiment, manifest, stage, *, phase=0, dependency=None, dependency_kind="afterok"):
    gpu = stage in ("train", "history")
    training = stage == "train"
    memory, minutes, cpus = (384, 1800, 8) if training else (24, 360, 4) if gpu else (8, 30, 2)
    command = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={manifest['workspace']}",
        "--kill-on-invalid-dep=yes", f"--job-name=gc3em7-{stage}-{phase}",
        f"--cpus-per-task={cpus}", f"--mem={memory}G", f"--time={minutes}",
        f"--output={experiment}/logs/{stage}_{phase}_%A_%a.out", f"--error={experiment}/logs/{stage}_{phase}_%A_%a.err"]
    if gpu:
        command += ["--gres=gpu:1", "--constraint=gpu80", "--qos=" + ("gpu-medium" if training else "gpu-short"),
                    "--array=0-3"]
    if dependency:
        command.append(f"--dependency={dependency_kind}:{dependency}")
    return command + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(experiment), str(phase)]


def submission_plan(experiment, manifest):
    return [
        {"id": "train-part1", "command": job_command(experiment, manifest, "train", phase=0)},
        {"id": "train-part2", "command": job_command(experiment, manifest, "train", phase=1,
            dependency="<train-part1>", dependency_kind="aftercorr")},
        {"id": "history", "command": job_command(experiment, manifest, "history",
            dependency="<train-part2>", dependency_kind="aftercorr")},
        {"id": "finish", "command": job_command(experiment, manifest, "finish", dependency="<train-part2>")},
    ]


def train(experiment, manifest, task_index, phase):
    require(task_index in range(len(manifest["tasks"])) and phase in (0, 1), "Invalid training task or phase")
    return scalar.run_training_stage(experiment, manifest, manifest["tasks"][task_index],
        stop=ENDPOINTS[phase], previous=1250 if phase else None)


def history(experiment, manifest, task_index, checkpoint_step=None):
    require(task_index in range(len(manifest["tasks"])), "Invalid history task")
    task = manifest["tasks"][task_index]
    if checkpoint_step is None:
        for step in HISTORY_STEPS:
            call([sys.executable, "-u", str(Path(manifest["source_root"]) / DRIVER), "history",
                  "--experiment", str(experiment), "--task", str(task_index), "--checkpoint-step", str(step)])
        return {"status": "complete", "task": task["logical_id"], "steps": list(HISTORY_STEPS)}
    require(checkpoint_step in HISTORY_STEPS, "Invalid history checkpoint")
    return scalar.evaluate_checkpoint(experiment, manifest, task, checkpoint_step,
        experiment / "development_history" / task["logical_id"] / f"step{checkpoint_step:08d}" / "evaluation.json")


def finish(experiment, manifest):
    proofs = []
    for task in manifest["tasks"]:
        proof = unseal(scalar.proof_path(experiment, task, 2500))
        require(proof["status"] == "complete" and proof["completed_steps"] == 2500
                and proof["task_id"] == task["id"] and proof["manifest_hash"] == manifest["record_hash"]
                and proof["source_digest"] == manifest["source_digest"], "Final completion identity differs")
        check(proof["checkpoint"])
        check(proof["metrics"])
        proofs.append(proof)
    return seal(experiment / "training_complete_full_gc.json", {
        "status": "complete", "manifest_hash": manifest["record_hash"],
        "selected_arms": pinned(experiment / "selected_arms_full_gc.json"),
        "final_checkpoints": proofs, "scientific_improvement_required": False,
        "downstream": "External three-source controller combines preserved lambda-zero and GC–Mamba"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "validate", "plan", "submit", "train", "history", "finish"))
    parser.add_argument("--experiment", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--phase", type=int, choices=(0, 1), default=0)
    parser.add_argument("--checkpoint-step", type=int, choices=HISTORY_STEPS)
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    if args.stage == "prepare":
        result = prepare(experiment, ROOT, args.reference)
    else:
        manifest = verify(experiment, frozen=args.stage in ("train", "history", "finish"))
        if args.stage == "validate":
            from src.models.mamba.v24_Ilya.training.config import load_training_config
            for task in manifest["tasks"]:
                load_training_config(Path(task["config"]["path"]))
            result = {"status": "valid", "tasks": len(manifest["tasks"]), "source_digest": manifest["source_digest"]}
        elif args.stage in ("plan", "submit"):
            nodes = submission_plan(experiment, manifest)
            result = nodes if args.stage == "plan" else scalar.submit_nodes(experiment, nodes)
        elif args.stage == "train":
            result = train(experiment, manifest, args.task, args.phase)
        elif args.stage == "history":
            result = history(experiment, manifest, args.task, args.checkpoint_step)
        else:
            result = finish(experiment, manifest)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
