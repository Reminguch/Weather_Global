#!/usr/bin/env python3
"""Fresh five-lambda GC–Mamba sweep using M1's two-stage training recipe.

The first allocation runs initial500 then main750; the second exactly resumes
main750 through2000. Finite deterioration never stops the prescribed updates.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import freeze_source, verify_source
from scripts.experiments.run_v24_mgda import (
    call, check, gpu_required, pinned, read, seal, submit_once, unseal, write,
)
from scripts.experiments.run_v24_scalar_lambda import numeric_initial_identity, read_metrics

FORMAT = "v24_m1_matched_gcm_lambda_v1"
DEFAULT = "artifacts/checkpoints/v24_Ilya/res1_m1_matched_gcm_lambda_20261112"
REFERENCE = "artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107"
M1 = "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908"
TEMPLATES = {"initial": f"{M1}/configs/warmup_di16_fast_mamba.json",
             "main": f"{M1}/configs/di16_fast_mamba.json"}
DRIVER = "scripts/experiments/run_v24_m1_matched_gcm_lambda.py"
WRAPPER = "scripts/experiments/v24_m1_matched_gcm_lambda.slurm"
LAMBDAS = (0., .25, .5, .75, 1.)
HISTORY_TOTAL_UPDATES = (500, 1000, 1500, 2000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run_dir(experiment, task, stage):
    return experiment / ("initial" if stage == "initial" else "runs") / task["id"]


def checkpoint_path(experiment, task, stage, step):
    return run_dir(experiment, task, stage) / "checkpoints" / f"checkpoint_step{step:08d}.pkl"


def proof_path(experiment, task, total):
    tag = "initial" if total == 500 else "total"
    return experiment / "checks" / f"{task['id']}_{tag}{total:08d}.json"


def checkpoint_for_total(experiment, task, total):
    require(total in (500, 1000, 1250, 1500, 2000, 2500), "Unsupported total update")
    stage, step = ("initial", 500) if total == 500 else ("main", total - 500)
    return stage, step, checkpoint_path(experiment, task, stage, step)


def make_config(template, experiment, task, workspace, stage):
    require(stage in TEMPLATES, "Unknown training stage")
    cfg = copy.deepcopy(template)
    cfg["model_family"] = "gc_mamba"
    for name in ("prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir"):
        path = Path(cfg["data"][name])
        cfg["data"][name] = str(path if path.is_absolute() else workspace / path)
    cfg["data"].pop("train_order_manifest", None)
    cfg["architecture"]["residual_initialization"] = "baseline_overlay"
    cfg["sequence"].update(segment_steps=96, bptt_steps=24, ar_tail_k=20,
                           feedback_mode="closed_loop_sg", temporal_state_policy="carry")
    cfg["memory"] = {"weather_tape_precision": "fp32", "bptt_backend": "explicit_reverse_vjp"}
    cfg["objective"] = {"loss_mode": "all_steps", "loss_kind": "mse"}
    if task["lambda"]:
        cfg["objective"].update(loss_kind="native_mse_amse_excess",
            spectral_excess_weight=task["lambda"], amse_lmax=179, amse_power_epsilon=1e-12)
    cfg["optimizer"].update(optimizer_kind="adamw", max_steps=500 if stage == "initial" else 2000,
        checkpoint_every=250, learning_rate=1e-4, warmup_steps=50,
        learning_rate_schedule="constant" if stage == "initial" else "cosine",
        end_learning_rate=None if stage == "initial" else 1e-5,
        mamba_lr_multiplier=1. if stage == "initial" else 3., spatial_lr_multiplier=1.,
        adam_beta1=.9, adam_beta2=.98, weight_decay=1e-4, weight_decay_policy="all",
        grad_clip=1., seed=22, precision="bf16")
    cfg["distributed"] = {"mode": "single", "num_devices": 1,
        "per_device_batch_size": 4, "drop_incomplete_replica_group": True}
    cfg["validation"]["enabled"] = False
    cfg["output"] = {"output_root": str(run_dir(experiment, task, stage).parent), "run_name": task["id"]}
    return cfg


def draft(experiment, workspace):
    """Write reviewable configurations; do not freeze incomplete source or submit."""
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    if experiment.exists():
        raise FileExistsError("Use a new campaign directory")
    reference_record = pinned(workspace / REFERENCE / "manifest.json")
    reference = unseal(reference_record["path"])
    templates = {stage: read(workspace / path) for stage, path in TEMPLATES.items()}
    inputs = [reference_record, *[pinned(workspace / path) for path in TEMPLATES.values()],
              reference["development_manifest"]]
    data = templates["initial"]["data"]
    for name in ("metadata.json", "anchors/anchor_indices.npy", "anchors/anchor_times.npy",
                 "anchors/split_train.npy", "anchors/split_val.npy"):
        inputs.append(pinned(workspace / data["anchor_manifest_root"] / name))
    baseline = pinned(workspace / data["baseline_checkpoint"])
    inputs.append(baseline)
    inputs.append(pinned(workspace / data["prepared_root"] / "metadata.json"))
    for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"):
        inputs.append(pinned(workspace / data["stats_dir"] / name))
    for name in ("logs", "checks", "submissions", "configs"):
        (experiment / name).mkdir(parents=True)
    tasks = []
    for weight in LAMBDAS:
        identity = f"gcm_m1matched_s22_l{round(weight * 100):03d}"
        task = {"id": identity, "logical_id": identity, "model_family": "gc_mamba",
                "lambda": weight, "learning_rate": 1e-4, "total_training_updates": 2500}
        for stage, template in templates.items():
            path = experiment / "configs" / f"{identity}_{stage}.json"
            write(path, make_config(template, experiment, task, workspace, stage))
            task[stage + "_config"] = pinned(path)
            inputs.append(task[stage + "_config"])
        task["config"] = task["main_config"]
        tasks.append(task)
    for item in inputs:
        check(item)
    return seal(experiment / "draft.json", {
        "format": FORMAT, "workspace": str(workspace), "experiment": str(experiment),
        "inputs": inputs, "tasks": tasks, "reference_campaign": reference_record, "baseline": baseline,
        "development_manifest": reference["development_manifest"],
        "development_data": reference["development_data"],
        "completion_hook": [sys.executable,
            "{source}/scripts/experiments/run_v24_m1_matched_downstream.py", "submit-smoothing",
            "--training-campaign", "{experiment}"],
        "policy": {"lambdas": list(LAMBDAS), "seed": 22, "initial_updates": 500,
            "main_updates": 2000, "total_updates_per_arm": 2500,
            "allocation_endpoints_total": [1250, 2500], "main_split_step": 750,
            "feedback_mode": "closed_loop_sg", "training_order": "historical contiguous segment groups",
            "stage_transfer": "parameters only; reset optimizer/RNG/cursor/recurrent state",
            "allocation_resume": "exact optimizer/RNG/cursor/recurrent state continuation",
            "finite_diagnostics": "warning", "scientific_improvement_required": False,
            "automatic_retries": 0},
        "resources": {"training": {"memory_gib": 384, "minutes": 1800, "cpus": 8, "gpu": 1},
            "basis": "Old production p90 near59–60s/update and host RSS234.71GiB; conservative allowance for two initializations in part1. Revised stopped-feedback cost is unmeasured."}})


def prepare(experiment, workspace):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    experiment, workspace = Path(experiment).resolve(), Path(workspace).resolve()
    if (experiment / "manifest.json").exists():
        raise FileExistsError("Campaign already frozen")
    plan = unseal(experiment / "draft.json") if experiment.exists() else draft(experiment, workspace)
    require(plan["workspace"] == str(workspace), "Draft workspace differs")
    for item in plan["inputs"]:
        check(item)
    for task in plan["tasks"]:
        for stage in TEMPLATES:
            cfg = load_training_config(Path(task[stage + "_config"]["path"]))
            require(cfg.train_order_manifest is None and cfg.feedback_mode == "closed_loop_sg",
                    "Training recipe differs from M1")
    source = freeze_source(workspace, experiment / "source")
    manifest = {k: v for k, v in plan.items() if k != "record_hash"}
    manifest.update(source_root=str(experiment / "source"), source_digest=source["source_digest"],
                    draft_record=pinned(experiment / "draft.json"))
    result = seal(experiment / "manifest.json", manifest)
    selection = {"tasks": result["tasks"], "manifest_hash": result["record_hash"],
                 "selection": "Prescribed five lambdas; no score-based selection"}
    seal(experiment / "selected_arms_gc_mamba.json", selection)
    seal(experiment / "selected_arms.json", selection)
    write(experiment / "submission_plan.json", submission_plan(experiment, result))
    return result


def verify(experiment, *, frozen=False):
    manifest = unseal(experiment / "manifest.json")
    require(manifest["format"] == FORMAT and Path(manifest["experiment"]) == experiment,
            "Campaign identity differs")
    if frozen:
        require(ROOT == Path(manifest["source_root"]), "Execute the frozen campaign source")
    require(verify_source(manifest["source_root"])["source_digest"] == manifest["source_digest"],
            "Frozen source identity differs")
    for item in manifest["inputs"]:
        check(item)
    return manifest


def training_command(experiment, manifest, task, stage, stop, *, resume=None, init_from=None):
    require(not (resume and init_from), "Choose exact resume or stage initialization")
    command = [sys.executable, "-u", str(Path(manifest["source_root"]) / "scripts/training/train_v24_Ilya.py"),
               "--config", task[stage + "_config"]["path"], "--stop-after-steps", str(stop)]
    if resume:
        command += ["--resume", str(resume)]
    if init_from:
        command += ["--init-from", str(init_from), "--zero-state-on-init"]
    return command


def verify_training_checkpoint(path, config_path, baseline_sha256, step):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    saved = load_v24_Ilya_training_checkpoint(path)
    cfg = load_training_config(Path(config_path))
    require(saved.completed_step == step and saved.model_family == "gc_mamba", "Checkpoint progress differs")
    require(saved.baseline_checkpoint_fingerprint == baseline_sha256, "Checkpoint baseline identity differs")
    require(saved.resolved_training_config == cfg.to_dict(), "Checkpoint training configuration differs")
    for tree in (saved.residual_params, saved.residual_state, saved.optimizer_state):
        for leaf in jax.tree_util.tree_leaves(tree):
            value = np.asarray(leaf)
            if np.issubdtype(value.dtype, np.inexact):
                require(np.isfinite(value).all(), "Nonfinite saved checkpoint computation")


def run_stage(experiment, manifest, task, stage, stop, total, *, previous=None, initial_proof=None):
    path = proof_path(experiment, task, total)
    if path.exists():
        raise FileExistsError(f"Completed stage must not be repeated: {path}")
    check(task[stage + "_config"])
    run = run_dir(experiment, task, stage)
    resume, init_from, start = None, None, 0
    if previous:
        require(previous["manifest_hash"] == manifest["record_hash"]
                and previous["task_id"] == task["id"]
                and previous["completed_steps"] == 750
                and previous["total_training_updates"] == 1250, "Resume trajectory differs")
        for key in ("checkpoint", "metrics", "run_config"):
            check(previous[key])
        require(pinned(run / "train_metrics.jsonl")["sha256"] == previous["metrics"]["sha256"],
                "Live training history differs from the committed split")
        resume, start = previous["checkpoint"]["path"], 750
    elif run.exists():
        raise FileExistsError(f"Fresh stage output already exists: {run}")
    if initial_proof:
        require(initial_proof["manifest_hash"] == manifest["record_hash"]
                and initial_proof["task_id"] == task["id"]
                and initial_proof["completed_steps"] == 500, "Initial-stage trajectory differs")
        check(initial_proof["checkpoint"])
        if not previous:
            init_from = initial_proof["checkpoint"]["path"]
    started = time.monotonic()
    call(training_command(experiment, manifest, task, stage, stop, resume=resume, init_from=init_from))
    elapsed = time.monotonic() - started
    rows = read_metrics(run / "train_metrics.jsonl")
    require([row["step"] for row in rows] == list(range(1, stop + 1)),
            "Training history must contain every prescribed update exactly once")
    require(all(math.isfinite(float(row[key])) for row in rows for key in ("loss", "gradient_norm")),
            "Nonfinite training metrics")
    checkpoint = checkpoint_path(experiment, task, stage, stop)
    verify_training_checkpoint(checkpoint, task[stage + "_config"]["path"],
                               manifest["baseline"]["sha256"], stop)
    metrics = path.with_suffix(".metrics.jsonl")
    with metrics.open("xb") as stream:
        stream.write((run / "train_metrics.jsonl").read_bytes())
    initial = checkpoint_path(experiment, task, stage, 0)
    durations = [float(row["end_to_end_seconds"]) for row in rows[start:]]
    record = {"status": "complete", "task_id": task["id"], "model_family": "gc_mamba",
        "lambda": task["lambda"], "stage": stage, "completed_steps": stop,
        "total_training_updates": total, "started_from_step": start,
        "manifest_hash": manifest["record_hash"], "source_digest": manifest["source_digest"],
        "checkpoint": pinned(checkpoint), "run_config": pinned(run / "run_config.json"),
        "metrics": pinned(metrics), "live_metrics_path": str(run / "train_metrics.jsonl"),
        "initial_checkpoint": pinned(initial), "initial_numerical_identity": numeric_initial_identity(initial),
        "elapsed_seconds": elapsed, "mean_update_seconds": sum(durations) / len(durations),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
        "scientific_improvement_required": False}
    if initial_proof:
        record["initial_stage_proof"] = pinned(proof_path(experiment, task, 500))
    record["allocation_elapsed_seconds"] = (elapsed + initial_proof["elapsed_seconds"]
        if initial_proof and not previous else elapsed)
    if previous:
        record["previous_allocation_proof"] = pinned(proof_path(experiment, task, 1250))
    return seal(path, record)


def train(experiment, manifest, task_index, part):
    gpu_required()
    task = manifest["tasks"][task_index]
    if part == 1:
        initial_path = proof_path(experiment, task, 500)
        initial = (unseal(initial_path) if initial_path.exists()
                   else run_stage(experiment, manifest, task, "initial", 500, 500))
        return run_stage(experiment, manifest, task, "main", 750, 1250, initial_proof=initial)
    require(part == 2, "Unknown allocation part")
    return run_stage(experiment, manifest, task, "main", 2000, 2500,
        previous=unseal(proof_path(experiment, task, 1250)),
        initial_proof=unseal(proof_path(experiment, task, 500)))


def history(experiment, manifest, task_index, total=None):
    task = manifest["tasks"][task_index]
    if total is None:
        for value in HISTORY_TOTAL_UPDATES:
            call([sys.executable, "-u", str(Path(manifest["source_root"]) / DRIVER), "history",
                  "--experiment", str(experiment), "--task", str(task_index), "--total-step", str(value)])
        return seal(experiment / "checks" / f"{task['id']}_history.json", {
            "status": "complete", "task_id": task["id"], "total_updates": list(HISTORY_TOTAL_UPDATES),
            "manifest_hash": manifest["record_hash"]})
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    gpu_required()
    stage, step, checkpoint = checkpoint_for_total(experiment, task, total)
    proof = unseal(proof_path(experiment, task, 500 if stage == "initial" else 2500))
    require(proof["manifest_hash"] == manifest["record_hash"], "Evaluation trajectory differs")
    checkpoint_record = pinned(checkpoint)
    config = read(task[stage + "_config"]["path"])
    output = experiment / "development_history" / task["id"] / f"total{total:08d}" / "evaluation.json"
    if output.parent.exists():
        raise FileExistsError(f"Do not overwrite an evaluation: {output.parent}")
    output.parent.mkdir(parents=True)
    allowed = {field.name for field in fields(V24IlyaEvalConfig)}
    cfg = V24IlyaEvalConfig(ckpt=checkpoint, out_json=output, model_family="gc_mamba",
        ckpt_in=Path(config["data"]["baseline_checkpoint"]), stats_dir=Path(config["data"]["stats_dir"]),
        data_path=manifest["development_data"], initialization_manifest=Path(manifest["development_manifest"]["path"]),
        model_id=f"{task['id']}_total{total}", val_year=2022, train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=32, warmup_steps=0, eval_mode="cold_full", residual_state_init="zero",
        omit_rms_bias=True, **{k: v for k, v in config["architecture"].items() if k in allowed})
    result = evaluate_v24_Ilya(cfg)
    require(result["evaluation_status"] == "complete" and result["metrics_complete"]
            and result["evaluated_samples"] == 32 and result["remaining_samples"] == 0,
            "Incomplete history evaluation")
    value = float(result["original_graphcast_loss"]["full_rollout"])
    require(math.isfinite(value), "Nonfinite exact development loss")
    check(checkpoint_record)
    return seal(output.parent / "score.json", {"task": task, "exact_gc_rollout_loss": value,
        "checkpoint": checkpoint_record, "evaluation": pinned(output), "step": step,
        "total_training_updates": total, "manifest_hash": manifest["record_hash"], "selection_year": 2022})


def finish(experiment, manifest):
    proofs = []
    for task in manifest["tasks"]:
        proof = unseal(proof_path(experiment, task, 2500))
        require(proof["manifest_hash"] == manifest["record_hash"] and proof["completed_steps"] == 2000
                and proof["total_training_updates"] == 2500, "Final progress differs")
        check(proof["checkpoint"])
        proofs.append(proof)
    initial_proofs = [unseal(proof_path(experiment, task, 500)) for task in manifest["tasks"]]
    identities = [p["initial_numerical_identity"]["residual_params"] for p in initial_proofs]
    initial_match = len(set(identities)) == 1
    record = {"status": "complete", "manifest_hash": manifest["record_hash"],
        "selected_arms": pinned(experiment / "selected_arms_gc_mamba.json"), "final_checkpoints": proofs,
        "initial_parameter_identity_matches": initial_match,
        "initial_parameter_hashes": identities,
        "warnings": [] if initial_match else ["Finite fresh parameter identity differs across arms; inspect recorded initial checkpoints."],
        "scientific_improvement_required": False}
    seal(experiment / "training_complete_gc_mamba.json", record)
    result = seal(experiment / "training_complete.json", record)
    hook = manifest.get("completion_hook")
    destination = experiment / "downstream_submitted_gc_mamba.json"
    if hook and not destination.exists():
        command = [part.replace("{source}", manifest["source_root"]).replace("{experiment}", str(experiment))
                   for part in hook]
        call(command)
        seal(destination, {"status": "submitted", "manifest_hash": manifest["record_hash"], "command": command})
    return result


def sbatch(experiment, manifest, stage, *, part=0, dependency=None, dependency_kind="afterok"):
    training = stage == "train"
    gpu = stage in ("train", "history")
    memory, minutes = (384, 1800) if training else (20, 240) if gpu else (8, 30)
    command = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={manifest['workspace']}",
        "--kill-on-invalid-dep=yes", f"--job-name=m1-gcm-{stage}-{part}",
        f"--cpus-per-task={8 if training else 4 if gpu else 2}", f"--mem={memory}G", f"--time={minutes}",
        f"--output={experiment}/logs/{stage}_{part}_%A_%a.out", f"--error={experiment}/logs/{stage}_{part}_%A_%a.err"]
    if gpu:
        command += ["--gres=gpu:1", "--constraint=gpu80", "--qos=gpu-medium" if training else "--qos=gpu-short",
                    "--array=0-4"]
    if dependency:
        command.append(f"--dependency={dependency_kind}:{dependency}")
    return command + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(experiment), str(part)]


def submission_plan(experiment, manifest):
    return [{"id": "train-part1", "command": sbatch(experiment, manifest, "train", part=1)},
            {"id": "train-part2", "command": sbatch(experiment, manifest, "train", part=2,
                dependency="<train-part1>", dependency_kind="aftercorr")},
            {"id": "history", "command": sbatch(experiment, manifest, "history",
                dependency="<train-part2>", dependency_kind="aftercorr")},
            {"id": "finish", "command": sbatch(experiment, manifest, "finish", dependency="<train-part2>")}]


def submit(experiment, manifest):
    jobs = {}
    for node in submission_plan(experiment, manifest):
        command = list(node["command"])
        for name, job in jobs.items():
            command = [arg.replace(f"<{name}>", str(job)) for arg in command]
        jobs[node["id"]] = submit_once(experiment, node["id"], command)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("draft", "prepare", "plan", "submit", "train", "history", "finish"))
    parser.add_argument("--experiment", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total-step", type=int, choices=HISTORY_TOTAL_UPDATES)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage in ("draft", "prepare"):
        value = globals()[args.stage](experiment, args.workspace)
        result = {"stage": args.stage, "experiment": str(experiment), "record_hash": value["record_hash"]}
    else:
        manifest = verify(experiment, frozen=args.stage in ("train", "history", "finish"))
        require(args.task in range(5), "Invalid task index")
        if args.stage == "plan":
            result = submission_plan(experiment, manifest)
        elif args.stage == "submit":
            result = submit(experiment, manifest)
        elif args.stage == "train":
            result = train(experiment, manifest, args.task, args.part)
        elif args.stage == "history":
            result = history(experiment, manifest, args.task, args.total_step)
        else:
            result = finish(experiment, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
