#!/usr/bin/env python3
"""Fresh scalar GC/Mamba sweep with retained profiling and development LR selection.

Only the explicit submit/release/select stages submit Slurm jobs. Finite losses,
drift and weak development scores never cancel an otherwise valid trajectory.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
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

FORMAT = "v24_scalar_lambda_v1"
DRIVER = "scripts/experiments/run_v24_scalar_lambda.py"
WRAPPER = "scripts/experiments/v24_scalar_lambda.slurm"
TEMPLATE = "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/configs/di16_fast_mamba.json"
REFERENCE_PROTOCOL = "artifacts/evaluations/v24_mgda_power_2023_20261106/shards/00/manifest.json"
DEFAULT_DEV_DATA = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr"
FAMILIES = ("full_gc", "gc_mamba")
LEARNING_RATES = {"full_gc": (1e-5, 3e-5), "gc_mamba": (3e-5, 1e-4)}
LAMBDAS = (0., .25, .5, .75, 1.)
PROFILE_STEPS = 8
CALIBRATION_STEPS = 500
MAX_STEPS = 2500
ENDPOINTS = (1250, 2500)


def logical_id(family, weight):
    prefix = "gcft" if family == "full_gc" else "gcm"
    return f"{prefix}_s22_l{round(weight * 100):03d}"


def initial_tasks():
    tasks = []
    for family in FAMILIES:
        for index, rate in enumerate(LEARNING_RATES[family]):
            tasks.append({"id": f"{logical_id(family, 0.)}_lr{index}",
                          "model_family": family, "lambda": 0., "learning_rate": rate,
                          "role": "lr_candidate"})
    for family in FAMILIES:
        tasks.append({"id": f"{logical_id(family, .5)}_profile", "model_family": family,
                      "lambda": .5, "learning_rate": LEARNING_RATES[family][0],
                      "role": "positive_lambda_profile"})
    return tasks


def make_config(template, experiment, task, workspace):
    config = copy.deepcopy(template)
    config["model_family"] = task["model_family"]
    for name in ("prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir"):
        path = Path(config["data"][name])
        config["data"][name] = str(path if path.is_absolute() else workspace / path)
    config["data"]["train_order_manifest"] = str(experiment / "manifests/train_order_seed22.json")
    config["architecture"]["residual_initialization"] = "baseline_overlay"
    config["sequence"].update(segment_steps=96, bptt_steps=24, ar_tail_k=20,
                              feedback_mode="closed_loop", temporal_state_policy="carry")
    config["memory"] = {"weather_tape_precision": "fp32", "bptt_backend": "explicit_weather_reverse_vjp"}
    config["objective"] = {"loss_mode": "all_steps", "loss_kind": "gc_mse_amse_excess",
                           "spectral_excess_weight": task["lambda"],
                           "amse_lmax": 179, "amse_power_epsilon": 1e-12}
    config["optimizer"].update(optimizer_kind="adamw", max_steps=MAX_STEPS,
        checkpoint_every=250, learning_rate=task["learning_rate"],
        learning_rate_schedule="cosine", end_learning_rate=.1 * task["learning_rate"],
        warmup_steps=100, adam_beta1=.9, adam_beta2=.98, grad_clip=1.,
        weight_decay=1e-4, weight_decay_policy="all", seed=22, precision="bf16",
        spatial_lr_multiplier=1., mamba_lr_multiplier=3. if task["model_family"] == "gc_mamba" else 1.)
    config["distributed"] = {"mode": "single", "num_devices": 1,
        "per_device_batch_size": 4, "drop_incomplete_replica_group": True}
    config["validation"]["enabled"] = False
    config["output"] = {"output_root": str(experiment / "runs"), "run_name": task["id"]}
    return config


def config_path(experiment, task):
    return experiment / "configs" / f"{task['id']}.json"


def checkpoint_path(experiment, task, step):
    return experiment / "runs" / task["id"] / "checkpoints" / f"checkpoint_step{step:08d}.pkl"


def proof_path(experiment, task, stop):
    return experiment / "checks" / f"{task['id']}_step{stop:08d}.json"


def prepare(experiment, workspace, development_manifest, development_data, completion_hook=None):
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.models.mamba.v24_Ilya.training.segment_order import create_train_order_manifest
    from src.models.mamba.v24_Ilya.development_manifest import create_development_manifest
    if experiment.exists():
        raise FileExistsError("Use a new campaign directory; existing campaigns are immutable")
    experiment.mkdir(parents=True)
    for name in ("logs", "checks", "submissions", "configs", "manifests"):
        (experiment / name).mkdir()
    template = read(workspace / TEMPLATE)
    data = template["data"]
    if development_manifest is None:
        development_manifest = experiment / "manifests/development_2022_32.json"
        create_development_manifest(read(workspace / REFERENCE_PROTOCOL)["protocols"]["exact"],
                                    development_manifest)
    create_train_order_manifest(workspace / data["anchor_manifest_root"],
        workspace / data["prepared_root"], experiment / "manifests/train_order_seed22.json",
        seed=22, max_steps=MAX_STEPS, batch_size=4, segment_steps=96, bptt_steps=24,
        train_end="2021-12-31T18:00:00")
    inputs = [pinned(workspace / TEMPLATE), pinned(workspace / REFERENCE_PROTOCOL), pinned(development_manifest),
              pinned(experiment / "manifests/train_order_seed22.json")]
    for name in ("metadata.json", "anchors/anchor_indices.npy", "anchors/anchor_times.npy",
                 "anchors/split_train.npy", "anchors/split_val.npy"):
        inputs.append(pinned(workspace / data["anchor_manifest_root"] / name))
    inputs.append(pinned(workspace / data["baseline_checkpoint"]))
    for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc"):
        inputs.append(pinned(workspace / data["stats_dir"] / name))
    metadata = next((development_data / name for name in ("zarr.json", ".zmetadata")
                     if (development_data / name).exists()), None)
    if metadata is None:
        raise FileNotFoundError("Development dataset metadata is required")
    inputs.append(pinned(metadata))
    tasks = initial_tasks()
    for task in tasks:
        path = config_path(experiment, task)
        write(path, make_config(template, experiment, task, workspace))
        load_training_config(path)
        task["config"] = pinned(path)
        inputs.append(task["config"])
    source = freeze_source(workspace, experiment / "source")
    manifest = seal(experiment / "manifest.json", {
        "format": FORMAT, "workspace": str(workspace), "experiment": str(experiment),
        "source_root": str(experiment / "source"), "source_digest": source["source_digest"],
        "inputs": inputs, "template": template, "initial_tasks": tasks,
        "development_manifest": pinned(development_manifest), "development_data": str(development_data),
        "completion_hook": completion_hook or [sys.executable,
            "{source}/scripts/experiments/run_v24_smoothing.py", "submit-sweep",
            "--training-campaign", "{experiment}", "--family", "{family}"],
        "prepared_loader_policy": "Current format1 mmap loader, WEATHER_PREPARED_BOUNDED_READS=0 explicitly; no historical source substitution.",
        "profile": {"updates": PROFILE_STEPS, "memory_gib": 384, "minutes": 180,
            "basis": "Retained actual full-feedback scalar updates; FT runtime/GPU fit is unmeasured. "
                     "Historical full-feedback host RSS234.71GiB; one80GB GPU and8CPUs."},
        "policy": {"seed": 22, "lambdas": LAMBDAS, "max_steps": MAX_STEPS,
            "calibration_steps": CALIBRATION_STEPS, "endpoints": ENDPOINTS,
            "learning_rates": LEARNING_RATES, "selection": "minimum 2022 pooled exact GC rollout loss; exact tie lower LR",
            "finite_diagnostics": "warning", "automatic_retries": 0,
            "automatic_batch_or_horizon_reduction": False,
            "main_updates": 25000, "losing_lr_candidate_updates": 1000,
            "positive_profile_overhead_max_updates": 16}})
    write(experiment / "submission_plan.json", initial_submission_plan(experiment, manifest))
    return manifest


def verify(experiment, *, frozen=False):
    manifest = unseal(experiment / "manifest.json")
    if manifest["format"] != FORMAT or Path(manifest["experiment"]) != experiment:
        raise ValueError("Campaign identity differs")
    if frozen and ROOT != Path(manifest["source_root"]):
        raise ValueError("Run the campaign's frozen source")
    if verify_source(manifest["source_root"])["source_digest"] != manifest["source_digest"]:
        raise ValueError("Frozen source identity differs")
    for value in manifest["inputs"]:
        check(value)
    return manifest


def read_metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numeric_initial_identity(path):
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import numerical_checkpoint_identity
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    if checkpoint.completed_step != 0:
        raise ValueError("Initial checkpoint is not step zero")
    return {**numerical_checkpoint_identity(checkpoint),
            "training_cursor": dict(checkpoint.training_cursor)}


def run_training_stage(experiment, manifest, task, *, stop, previous=None):
    import numpy as np
    gpu_required()
    path = proof_path(experiment, task, stop)
    if path.exists():
        raise FileExistsError("Completed training stage must not be repeated")
    check(task["config"])
    run = experiment / "runs" / task["id"]
    command = [sys.executable, "-u", str(Path(manifest["source_root"]) / "scripts/training/train_v24_Ilya.py"),
               "--config", task["config"]["path"], "--stop-after-steps", str(stop)]
    start = 0
    if previous is not None:
        proof = unseal(proof_path(experiment, task, previous))
        if proof["manifest_hash"] != manifest["record_hash"] or proof["completed_steps"] != previous:
            raise ValueError("Prior stage does not belong to this trajectory")
        check(proof["checkpoint"])
        check(proof["metrics"])
        if pinned(run / "train_metrics.jsonl")["sha256"] != proof["metrics"]["sha256"]:
            raise ValueError("Live training history differs from the committed stage snapshot")
        command += ["--resume", proof["checkpoint"]["path"]]
        start = previous
    elif run.exists():
        raise FileExistsError("Fresh run output already exists; diagnose before retrying")
    started = time.monotonic()
    call(command)
    elapsed = time.monotonic() - started
    rows = read_metrics(run / "train_metrics.jsonl")
    if [row["step"] for row in rows] != list(range(1, stop + 1)):
        raise ValueError("Training history does not cover each expected optimizer update exactly once")
    if any(not math.isfinite(float(row[key])) for row in rows for key in ("loss", "gradient_norm")):
        raise ValueError("Nonfinite training metrics")
    checkpoint = checkpoint_path(experiment, task, stop)
    initial = checkpoint_path(experiment, task, 0)
    durations = [float(row["end_to_end_seconds"]) for row in rows[start:]]
    steady = durations[2:] or durations
    p90 = float(np.percentile(steady, 90))
    if not math.isfinite(p90) or p90 <= 0:
        raise ValueError("Unusable timing measurements")
    # Subsequent resumes append the live history; keep each stage's evidence immutable.
    metrics_snapshot = path.with_suffix(".metrics.jsonl")
    metrics_snapshot.parent.mkdir(parents=True, exist_ok=True)
    with metrics_snapshot.open("xb") as stream:
        stream.write((run / "train_metrics.jsonl").read_bytes())
    return seal(path, {"status": "complete", "task_id": task["id"],
        "model_family": task["model_family"], "lambda": task["lambda"],
        "completed_steps": stop, "started_from_step": start,
        "manifest_hash": manifest["record_hash"], "source_digest": manifest["source_digest"],
        "checkpoint": pinned(checkpoint), "metrics": pinned(metrics_snapshot),
        "live_metrics_path": str(run / "train_metrics.jsonl"),
        "initial_checkpoint": pinned(initial), "initial_numerical_identity": numeric_initial_identity(initial),
        "elapsed_seconds": elapsed, "steady_p90_seconds": p90,
        "startup_seconds": max(0., elapsed - sum(durations)),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
        "device_memory_samples": [row.get("device_memory") for row in rows[start:] if row.get("device_memory")],
        "scientific_improvement_required": False})


def measured_resources(proofs, updates):
    p90 = max(item["steady_p90_seconds"] for item in proofs)
    peak = max(234.71, *(item["peak_rss_gib"] for item in proofs))
    startup = max(1800., *(item["startup_seconds"] * 1.4 for item in proofs))
    memory = math.ceil(peak * 1.5 / 32) * 32
    minutes = max(62, math.ceil((updates * p90 * 1.4 + startup) / 1800) * 30)
    if memory > 768 or minutes > 8640:
        raise ValueError("Measured resource estimate exceeds supported execution limits; retain checkpoints for diagnosis")
    return {"memory": memory, "minutes": minutes, "steady_p90_seconds": p90,
            "host_peak_basis_gib": peak, "startup_allowance_seconds": startup}


def sbatch(experiment, manifest, stage, *, memory, minutes, gpu=False, array=None,
           dependency=None, dependency_kind="afterok", phase=0, family="all"):
    args = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={manifest['workspace']}",
            "--kill-on-invalid-dep=yes", f"--job-name=scalar-{family}-{stage}-{phase}",
            f"--cpus-per-task={8 if gpu else 2}", f"--mem={memory}G", f"--time={minutes}",
            f"--output={experiment}/logs/{stage}_{phase}_%A_%a.out",
            f"--error={experiment}/logs/{stage}_{phase}_%A_%a.err"]
    if gpu:
        args += ["--gres=gpu:1", "--constraint=gpu80", "--qos=" + (
            "gpu-short" if minutes <= 480 else "gpu-medium" if minutes <= 4320 else "gpu-long")]
    if array is not None:
        if "%" in array:
            raise ValueError("Array concurrency throttles are not part of this campaign")
        args.append(f"--array={array}")
    if dependency:
        args.append(f"--dependency={dependency_kind}:{dependency}")
    return args + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(experiment), str(phase), family]


def initial_submission_plan(experiment, manifest):
    return [{"id": "profile", "command": sbatch(experiment, manifest, "profile", memory=384,
               minutes=180, gpu=True, array="0-5")},
            *[{"id": f"release-calibration-{family}", "command": sbatch(experiment, manifest,
                "release-calibration", memory=16, minutes=30, family=family,
                dependency=":".join(f"<profile>_{i}" for i in indices))}
              for family, indices in (("full_gc", (0, 1, 4)), ("gc_mamba", (2, 3, 5)))]]


def submit_nodes(experiment, nodes):
    jobs = {}
    for node in nodes:
        command = list(node["command"])
        for name, job in jobs.items():
            command = [arg.replace(f"<{name}>", str(job)) for arg in command]
        jobs[node["id"]] = submit_once(experiment, node["id"], command)
    return jobs


def release_calibration(experiment, manifest, family):
    tasks = [task for task in manifest["initial_tasks"] if task["model_family"] == family]
    proofs = [unseal(proof_path(experiment, task, PROFILE_STEPS)) for task in tasks]
    warnings = []
    if any(p["initial_numerical_identity"] != proofs[0]["initial_numerical_identity"] for p in proofs[1:]):
        warnings.append(f"{family}: finite initialization repeat difference; hashes retained")
    resources = measured_resources([p for p in proofs if p["lambda"] == 0],
                                   CALIBRATION_STEPS - PROFILE_STEPS)
    seal(experiment / f"checks/profile_summary_{family}.json", {"resources": resources, "profiles": proofs,
        "warnings": warnings, "finite_diagnostics": "warning", "lora_required": False,
        "lora_note": "Successful full-GC profiling supports this execution shape; no LoRA fallback was substituted."})
    limits = {key: resources[key] for key in ("memory", "minutes")}
    index = FAMILIES.index(family)
    array = f"{index * 2}-{index * 2 + 1}"
    nodes = [
            {"id": f"calibrate-{family}", "command": sbatch(experiment, manifest, "calibrate",
                **limits, gpu=True, array=array, family=family)},
            {"id": f"evaluate-calibration-{family}", "command": sbatch(experiment, manifest,
                "evaluate-calibration", memory=24, minutes=92, gpu=True, array=array,
                family=family, dependency=f"<calibrate-{family}>", dependency_kind="aftercorr")}]
    nodes.append({"id": f"select-{family}", "command": sbatch(experiment, manifest, "select", memory=16,
        minutes=30, family=family, dependency=f"<evaluate-calibration-{family}>")})
    write(experiment / f"calibration_submission_plan_{family}.json", nodes)
    return submit_nodes(experiment, nodes)


def evaluate_checkpoint(experiment, manifest, task, step, output):
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    gpu_required()
    checkpoint = pinned(checkpoint_path(experiment, task, step))
    config = read(task["config"]["path"])
    if output.exists():
        raise FileExistsError("Completed development evaluation must not be repeated")
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = V24IlyaEvalConfig(ckpt=Path(checkpoint["path"]), out_json=output,
        model_family=task["model_family"], ckpt_in=Path(config["data"]["baseline_checkpoint"]),
        stats_dir=Path(config["data"]["stats_dir"]), data_path=manifest["development_data"],
        initialization_manifest=Path(manifest["development_manifest"]["path"]), model_id=task["id"],
        val_year=2022, train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=32, warmup_steps=0, eval_mode="cold_full",
        residual_state_init="zero", omit_rms_bias=True,
        **config["architecture"])
    result = evaluate_v24_Ilya(cfg)
    value = float(result["original_graphcast_loss"]["full_rollout"])
    if not math.isfinite(value):
        raise ValueError("Nonfinite development exact loss")
    return seal(output.parent / "score.json", {"task": task, "exact_gc_rollout_loss": value,
        "checkpoint": checkpoint, "evaluation": pinned(output), "step": step,
        "manifest_hash": manifest["record_hash"], "selection_year": 2022})


def evaluate_calibration(experiment, manifest, task):
    proof = unseal(proof_path(experiment, task, CALIBRATION_STEPS))
    check(proof["checkpoint"])
    return evaluate_checkpoint(experiment, manifest, task, CALIBRATION_STEPS,
        experiment / "calibration_evaluation" / task["id"] / "evaluation.json")


def history_evaluate(experiment, manifest, task_index, family, checkpoint_step=None):
    selected = unseal(experiment / f"selected_arms_{family}.json")
    if task_index not in range(len(selected["tasks"])):
        raise ValueError("Invalid history task index")
    task = selected["tasks"][task_index]
    if checkpoint_step is None:
        # Each forecast executes in a new process; compiled executables cannot accumulate.
        for step in (500, 1000, 1500, 2000):
            call([sys.executable, "-u", str(Path(manifest["source_root"]) / DRIVER),
                  "history-evaluate", "--experiment", str(experiment), "--task", str(task_index),
                  "--checkpoint-step", str(step), "--family", family])
        return {"status": "complete", "task": task["logical_id"], "steps": [500, 1000, 1500, 2000]}
    output = experiment / "development_history" / task["logical_id"] / f"step{checkpoint_step:08d}" / "evaluation.json"
    if checkpoint_step == 500 and task["lambda"] == 0:
        score = unseal(experiment / "calibration_evaluation" / task["id"] / "score.json")
        check(score["checkpoint"])
        check(score["evaluation"])
        return seal(output.parent / "reuse.json", {"source": score,
            "reason": "Identical selected lambda0 checkpoint, source, development dates and evaluation protocol."})
    return evaluate_checkpoint(experiment, manifest, task, checkpoint_step, output)


def select_winners(scores, families=FAMILIES):
    winners = {}
    for family in families:
        candidates = [score for score in scores if score["task"]["model_family"] == family]
        if len(candidates) != 2 or {x["task"]["learning_rate"] for x in candidates} != set(LEARNING_RATES[family]):
            raise ValueError("Both prespecified LR candidates are required")
        if any(not math.isfinite(x["exact_gc_rollout_loss"]) for x in candidates):
            raise ValueError("Nonfinite development loss")
        winners[family] = min(candidates, key=lambda x: (x["exact_gc_rollout_loss"], x["task"]["learning_rate"]))
    return winners


def choose_arms(experiment, manifest, scores, families=FAMILIES):
    winners = select_winners(scores, families)
    tasks, overhead = [], []
    for family in families:
        winner = winners[family]["task"]
        for weight in LAMBDAS:
            candidates = [task for task in manifest["initial_tasks"] if task["model_family"] == family
                          and task["lambda"] == weight and task["learning_rate"] == winner["learning_rate"]]
            if candidates:
                task = copy.deepcopy(candidates[0])
                task["resume_step"] = CALIBRATION_STEPS if weight == 0 else PROFILE_STEPS
            else:
                task = {"id": logical_id(family, weight), "model_family": family,
                    "lambda": weight, "learning_rate": winner["learning_rate"], "role": "primary",
                    "resume_step": 0}
                path = config_path(experiment, task)
                write(path, make_config(manifest["template"], experiment, task, Path(manifest["workspace"])))
                from src.models.mamba.v24_Ilya.training.config import load_training_config
                load_training_config(path)
                task["config"] = pinned(path)
            task["logical_id"] = logical_id(family, weight)
            tasks.append(task)
        overhead.extend(task["id"] for task in manifest["initial_tasks"]
                        if task["model_family"] == family and task["lambda"] > 0
                        and task["learning_rate"] != winner["learning_rate"])
    return {"tasks": tasks, "winners": winners, "scores": scores,
            "unused_positive_profiles": overhead, "profile_overhead_updates": len(overhead) * PROFILE_STEPS,
            "manifest_hash": manifest["record_hash"]}


def select(experiment, manifest, family):
    tasks = [task for task in manifest["initial_tasks"] if task["model_family"] == family]
    scores = [unseal(experiment / "calibration_evaluation" / task["id"] / "score.json")
              for task in tasks if task["lambda"] == 0]
    for score in scores:
        check(score["checkpoint"])
        check(score["evaluation"])
        if score["manifest_hash"] != manifest["record_hash"]:
            raise ValueError("Development score is from another campaign")
    selected = seal(experiment / f"selected_arms_{family}.json", choose_arms(experiment, manifest, scores, (family,)))
    profile_proofs = [unseal(proof_path(experiment, task, PROFILE_STEPS)) for task in tasks]
    calibration_proofs = [unseal(proof_path(experiment, task, CALIBRATION_STEPS))
                          for task in tasks if task["lambda"] == 0]
    resources = measured_resources(profile_proofs + calibration_proofs, 1250)
    nodes = []
    limits = {key: resources[key] for key in ("memory", "minutes")}
    for phase in range(2):
        nodes.append({"id": f"train-{family}-{phase}", "command": sbatch(experiment, manifest, "train",
            **limits, gpu=True, array="0-4", phase=phase, family=family,
            dependency=f"<train-{family}-{phase - 1}>" if phase else None, dependency_kind="aftercorr")})
    nodes.append({"id": f"history-evaluate-{family}", "command": sbatch(experiment, manifest,
        "history-evaluate", memory=24, minutes=360, gpu=True, array="0-4", family=family,
        dependency=f"<train-{family}-1>", dependency_kind="aftercorr")})
    nodes.append({"id": f"finish-{family}", "command": sbatch(experiment, manifest, "finish", memory=16,
        minutes=30, family=family, dependency=f"<train-{family}-1>")})
    seal(experiment / f"checks/production_resources_{family}.json", resources)
    write(experiment / f"production_submission_plan_{family}.json", nodes)
    return {"selected": selected, "jobs": submit_nodes(experiment, nodes)}


def finish(experiment, manifest, family):
    selected = unseal(experiment / f"selected_arms_{family}.json")
    proofs = []
    for task in selected["tasks"]:
        proof = unseal(proof_path(experiment, task, MAX_STEPS))
        check(proof["checkpoint"])
        proofs.append(proof)
    record = seal(experiment / f"training_complete_{family}.json", {"status": "complete",
        "manifest_hash": manifest["record_hash"], "selected_arms": pinned(experiment / f"selected_arms_{family}.json"),
        "final_checkpoints": proofs, "scientific_improvement_required": False})
    if manifest.get("completion_hook") and not (experiment / f"downstream_submitted_{family}.json").exists():
        call([part.replace("{experiment}", str(experiment)).replace("{source}", manifest["source_root"])
                  .replace("{family}", family) for part in manifest["completion_hook"]])
        seal(experiment / f"downstream_submitted_{family}.json", {"status": "submitted",
            "manifest_hash": manifest["record_hash"], "hook": manifest["completion_hook"], "family": family})
    with (experiment / "submissions/finish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all((experiment / f"training_complete_{f}.json").exists() for f in FAMILIES):
            return {**record, "other_method_pending": True}
        if (experiment / "training_complete.json").exists():
            record = unseal(experiment / "training_complete.json")
        else:
            families = [unseal(experiment / f"selected_arms_{f}.json") for f in FAMILIES]
            combined = {"tasks": [task for value in families for task in value["tasks"]],
                        "families": families, "manifest_hash": manifest["record_hash"]}
            seal(experiment / "selected_arms.json", combined)
            completed = [unseal(experiment / f"training_complete_{f}.json") for f in FAMILIES]
            record = seal(experiment / "training_complete.json", {"status": "complete", "families": completed,
                "selected_arms": pinned(experiment / "selected_arms.json"), "manifest_hash": manifest["record_hash"]})
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "dry-run", "submit", "profile", "release-calibration",
        "calibrate", "evaluate-calibration", "history-evaluate", "select", "train", "finish"))
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--development-data", type=Path, default=Path(DEFAULT_DEV_DATA))
    parser.add_argument("--completion-hook-json", type=Path,
                        help="JSON list of command arguments; {experiment}, {source}, {family} placeholders supported")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--phase", type=int, choices=(0, 1), default=0)
    parser.add_argument("--family", choices=(*FAMILIES, "all"), default="all")
    parser.add_argument("--checkpoint-step", type=int, choices=(500, 1000, 1500, 2000))
    args = parser.parse_args(argv)
    experiment = args.experiment.resolve()
    if args.stage == "prepare":
        hook = read(args.completion_hook_json) if args.completion_hook_json else None
        if hook is not None and (not isinstance(hook, list) or not hook or not all(isinstance(x, str) for x in hook)):
            parser.error("completion hook must be a nonempty JSON list of strings")
        result = prepare(experiment, ROOT, args.development_manifest.resolve() if args.development_manifest else None,
                         args.development_data.resolve(), hook)
    else:
        manifest = verify(experiment, frozen=args.stage not in ("dry-run", "submit"))
        if args.stage in ("dry-run", "submit"):
            plan = initial_submission_plan(experiment, manifest)
            result = plan if args.stage == "dry-run" else submit_nodes(experiment, plan)
        elif args.stage in ("profile", "calibrate", "evaluate-calibration"):
            count = 6 if args.stage == "profile" else 4
            if args.task not in range(count):
                raise ValueError("Invalid task index")
            task = manifest["initial_tasks"][args.task]
            if args.stage == "evaluate-calibration":
                result = evaluate_calibration(experiment, manifest, task)
            else:
                result = run_training_stage(experiment, manifest, task,
                    stop=PROFILE_STEPS if args.stage == "profile" else CALIBRATION_STEPS,
                    previous=None if args.stage == "profile" else PROFILE_STEPS)
        elif args.stage == "train":
            selected = unseal(experiment / f"selected_arms_{args.family}.json")
            if args.task not in range(len(selected["tasks"])):
                raise ValueError("Invalid training task index")
            task = selected["tasks"][args.task]
            result = run_training_stage(experiment, manifest, task, stop=ENDPOINTS[args.phase],
                previous=ENDPOINTS[0] if args.phase else task["resume_step"] or None)
        elif args.stage == "history-evaluate":
            result = history_evaluate(experiment, manifest, args.task, args.family, args.checkpoint_step)
        else:
            if args.family not in FAMILIES:
                parser.error(f"{args.stage} requires an individual --family")
            result = globals()[args.stage.replace("-", "_")](experiment, manifest, args.family)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
