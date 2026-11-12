#!/usr/bin/env python3
"""Prepare and run four fresh full-GC horizons on an independently configured cluster.

This adapter never submits jobs. It relocates inputs, preserves the fixed training
recipe, and records two exact-continuation allocations of 1,250 updates each.
"""
from __future__ import annotations

import argparse
import copy
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
from scripts.experiments.run_v24_mgda import call, check, gpu_required, pinned, read, seal, unseal, write
from scripts.experiments.run_v24_scalar_lambda import numeric_initial_identity, read_metrics

FORMAT = "v24_portable_full_gc_ar_v1"
HORIZONS = (8, 12, 16, 20)
INPUT_KEYS = ("prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir")
DRIVER = "scripts/experiments/run_v24_portable_full_gc_ar.py"
WRAPPER = "scripts/experiments/v24_portable_full_gc_ar.sh"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_config(template, experiment, horizon, input_paths, order_path):
    require(horizon in HORIZONS, "Unknown full-GC horizon")
    config = copy.deepcopy(template)
    require(config["model_family"] == "full_gc"
            and config["sequence"]["feedback_mode"] == "closed_loop"
            and config["sequence"]["bptt_steps"] == 24
            and config["sequence"]["segment_steps"] == 96
            and config["distributed"]["per_device_batch_size"] == 4
            and config["optimizer"]["max_steps"] == 2500
            and config["optimizer"]["learning_rate"] == 3e-7
            and config["optimizer"]["end_learning_rate"] == 3e-8
            and config["optimizer"]["learning_rate_schedule"] == "cosine"
            and config["optimizer"]["warmup_steps"] == 100
            and config["optimizer"]["seed"] == 22
            and config["objective"]["loss_kind"] == "gc_mse_amse_excess"
            and config["objective"]["spectral_excess_weight"] == 0,
            "Full-GC template differs from the prescribed standard-loss recipe")
    for key in INPUT_KEYS:
        config["data"][key] = str(Path(input_paths[key]).resolve())
    config["data"]["train_order_manifest"] = str(Path(order_path).resolve())
    config["sequence"]["ar_tail_k"] = horizon
    config["output"] = {"output_root": str(experiment / "runs"),
                        "run_name": f"gcft_s22_k{horizon:02d}_lr3em7"}
    return config


def relocate_order(original, input_paths):
    """Change descriptive source locations without changing any replay entry."""
    order = copy.deepcopy(original)
    local = {"anchor_indices": Path(input_paths["anchor_manifest_root"]) / "anchors/anchor_indices.npy",
             "split_train": Path(input_paths["anchor_manifest_root"]) / "anchors/split_train.npy",
             "time": Path(input_paths["prepared_root"]) / "coords/time.npy"}
    require(set(order["sources"]) == set(local), "Unexpected training-order source set")
    for name, path in local.items():
        record = pinned(path)
        require(record["sha256"] == order["sources"][name]["sha256"],
                f"Training-order input differs: {name}")
        order["sources"][name]["path"] = record["path"]
    return order


def prepare(package_root, experiment, input_paths):
    from scripts.experiments.run_v24_portable_gcm_lambda import validate_inputs
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.models.mamba.v24_Ilya.training.segment_order import load_train_order_manifest
    import numpy as np

    package_root, experiment = Path(package_root).resolve(), Path(experiment).resolve()
    require(package_root == ROOT, "Run preparation from the transferred package's source")
    if experiment.exists():
        raise FileExistsError("Use a new experiment directory for four fresh runs")
    input_paths = {key: Path(input_paths[key]).resolve() for key in INPUT_KEYS}
    reference, inputs = validate_inputs(package_root, input_paths)
    template_path = package_root / "templates/full_gc.json"
    order_source = package_root / "portable/train_order.json"
    template = read(template_path)
    order = relocate_order(read(order_source), input_paths)
    for name in ("inputs", "configs", "checks", "logs", "runs", "submissions"):
        (experiment / name).mkdir(parents=True, exist_ok=True)
    order_path = experiment / "inputs/train_order.json"
    write(order_path, order)
    inputs = list(inputs) + [pinned(template_path), pinned(order_source), pinned(order_path)]
    tasks = []
    arrays = {"anchors": np.load(input_paths["anchor_manifest_root"] / "anchors/anchor_indices.npy"),
              "split": np.load(input_paths["anchor_manifest_root"] / "anchors/split_train.npy"),
              "times": np.load(input_paths["prepared_root"] / "coords/time.npy")}
    for horizon in HORIZONS:
        config = make_config(template, experiment, horizon, input_paths, order_path)
        path = experiment / "configs" / f"k{horizon:02d}.json"
        write(path, config)
        cfg = load_training_config(path)
        load_train_order_manifest(order_path, config=cfg, **arrays)
        task = {"id": config["output"]["run_name"], "logical_id": config["output"]["run_name"],
                "model_family": "full_gc", "horizon": horizon, "lambda": 0.,
                "learning_rate": 3e-7, "total_training_updates": 2500, "config": pinned(path)}
        tasks.append(task)
        inputs.append(task["config"])
    source = freeze_source(package_root, experiment / "source")
    result = seal(experiment / "manifest.json", {
        "format": FORMAT, "workspace": str(package_root), "experiment": str(experiment),
        "source_root": str(experiment / "source"), "source_digest": source["source_digest"],
        "input_reference": reference, "inputs": inputs,
        "baseline": pinned(input_paths["baseline_checkpoint"]), "tasks": tasks,
        "policy": {"fresh_horizons": list(HORIZONS), "seed": 22,
                   "allocation_endpoints_total": [1250, 2500], "total_updates_per_arm": 2500,
                   "initialization": "public GraphCast only; fresh optimizer for each horizon",
                   "allocation_resume": "exact optimizer/RNG/data continuation; no reset",
                   "swa_steps": [1000, 1500, 2000, 2500],
                   "finite_diagnostics": "warning", "scientific_improvement_required": False,
                   "automatic_submission": False}})
    write(experiment / "execution_plan.json", plan(experiment, result))
    return result


def verify(experiment, *, frozen=False):
    manifest = unseal(experiment / "manifest.json")
    require(manifest["format"] == FORMAT and Path(manifest["experiment"]) == experiment,
            "Portable full-GC campaign identity differs")
    if frozen:
        require(ROOT == Path(manifest["source_root"]), "Execute the frozen campaign source")
    require(verify_source(manifest["source_root"])["source_digest"] == manifest["source_digest"],
            "Frozen source changed")
    for item in manifest["inputs"]:
        check(item)
    require([task["horizon"] for task in manifest["tasks"]] == list(HORIZONS),
            "All four fresh horizons are required")
    return manifest


def checkpoint_path(experiment, task, step):
    return experiment / "runs" / task["id"] / "checkpoints" / f"checkpoint_step{step:08d}.pkl"


def proof_path(experiment, task, step):
    return experiment / "checks" / f"{task['id']}_total{step:08d}.json"


def training_command(manifest, task, stop, *, resume=None):
    command = [sys.executable, "-u", str(Path(manifest["source_root"]) / "scripts/training/train_v24_Ilya.py"),
               "--config", task["config"]["path"], "--stop-after-steps", str(stop)]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def verify_training_checkpoint(path, config_path, baseline_sha256, step):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    saved = load_v24_Ilya_training_checkpoint(path)
    cfg = load_training_config(Path(config_path))
    require(saved.completed_step == step and saved.model_family == "full_gc", "Checkpoint progress/family differs")
    require(saved.baseline_checkpoint_fingerprint == baseline_sha256, "Checkpoint baseline differs")
    require(saved.resolved_training_config == cfg.to_dict(), "Checkpoint training configuration differs")
    for tree in (saved.residual_params, saved.residual_state, saved.optimizer_state):
        for leaf in jax.tree_util.tree_leaves(tree):
            value = np.asarray(leaf)
            if np.issubdtype(value.dtype, np.inexact):
                require(np.isfinite(value).all(), "Nonfinite saved checkpoint computation")


def train(experiment, manifest, task_index, part):
    require(task_index in range(4) and part in (1, 2), "Invalid horizon task/allocation part")
    gpu_required()
    task = manifest["tasks"][task_index]
    stop = 1250 if part == 1 else 2500
    proof = proof_path(experiment, task, stop)
    if proof.exists():
        raise FileExistsError(f"Completed allocation must not be repeated: {proof}")
    run = experiment / "runs" / task["id"]
    resume, previous, start = None, None, 0
    if part == 1:
        if run.exists():
            raise FileExistsError(f"Fresh run output already exists: {run}")
    else:
        previous = unseal(proof_path(experiment, task, 1250))
        require(previous["manifest_hash"] == manifest["record_hash"]
                and previous["task_id"] == task["id"] and previous["completed_steps"] == 1250
                and previous["total_training_updates"] == 1250, "Resume trajectory differs")
        for key in ("checkpoint", "metrics", "run_config"):
            check(previous[key])
        require(pinned(run / "train_metrics.jsonl")["sha256"] == previous["metrics"]["sha256"],
                "Live training history differs from the committed split")
        resume, start = previous["checkpoint"]["path"], 1250
    check(task["config"])
    started = time.monotonic()
    call(training_command(manifest, task, stop, resume=resume))
    elapsed = time.monotonic() - started
    rows = read_metrics(run / "train_metrics.jsonl")
    require([row["step"] for row in rows] == list(range(1, stop + 1)),
            "Training history must contain each prescribed update exactly once")
    require(all(math.isfinite(float(row[key])) for row in rows for key in ("loss", "gradient_norm")),
            "Nonfinite training metrics")
    checkpoint = checkpoint_path(experiment, task, stop)
    verify_training_checkpoint(checkpoint, task["config"]["path"], manifest["baseline"]["sha256"], stop)
    metrics = proof.with_suffix(".metrics.jsonl")
    with metrics.open("xb") as stream:
        stream.write((run / "train_metrics.jsonl").read_bytes())
    initial = checkpoint_path(experiment, task, 0)
    record = {"status": "complete", "task_id": task["id"], "horizon": task["horizon"],
              "model_family": "full_gc", "lambda": 0., "completed_steps": stop,
              "total_training_updates": stop, "started_from_step": start,
              "manifest_hash": manifest["record_hash"], "source_digest": manifest["source_digest"],
              "checkpoint": pinned(checkpoint), "run_config": pinned(run / "run_config.json"),
              "metrics": pinned(metrics), "initial_checkpoint": pinned(initial),
              "initial_numerical_identity": numeric_initial_identity(initial),
              "elapsed_seconds": elapsed, "allocation_elapsed_seconds": elapsed,
              "peak_rss_gib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
              "scientific_improvement_required": False}
    if previous:
        record["previous_allocation_proof"] = pinned(proof_path(experiment, task, 1250))
    return seal(proof, record)


def finish(experiment, manifest):
    proofs = []
    for task in manifest["tasks"]:
        record = unseal(proof_path(experiment, task, 2500))
        require(record["manifest_hash"] == manifest["record_hash"] and record["task_id"] == task["id"]
                and record["completed_steps"] == record["total_training_updates"] == 2500,
                "Final progress differs")
        for key in ("checkpoint", "run_config", "metrics", "initial_checkpoint"):
            check(record[key])
        proofs.append(record)
    identities = [record["initial_numerical_identity"]["residual_params"] for record in proofs]
    warnings = ([] if len(set(identities)) == 1 else
                ["Finite initial parameter identity differs across horizons; inspect saved values."])
    result = {"status": "complete", "manifest_hash": manifest["record_hash"],
              "final_checkpoints": proofs, "initial_parameter_hashes": identities,
              "warnings": warnings, "scientific_improvement_required": False,
              "automatic_submission": False}
    seal(experiment / "training_complete_full_gc.json", result)
    return seal(experiment / "training_complete.json", result)


def plan(experiment, manifest):
    wrapper = str(Path(manifest["source_root"]) / WRAPPER)
    return {"training": [{"task": index, "horizon": task["horizon"],
             "part1": ["bash", wrapper, "train", str(experiment), "1", str(index)],
             "part2": ["bash", wrapper, "train", str(experiment), "2", str(index)]}
            for index, task in enumerate(manifest["tasks"])],
            "finish": ["bash", wrapper, "finish", str(experiment)],
            "scheduler": "Run each part2 only after its part1 succeeds; finish after all part2 jobs succeed.",
            "resources_per_training_job": {"gpu": "one 80GB GPU", "cpus": 8, "host_ram_gib": 384,
                                           "walltime_hours": 30},
            "resource_basis": "Existing K20 production p90 about60s/update and host RSS about235GiB; remote runtime unmeasured."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "train", "finish"))
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--experiment", type=Path, required=True)
    for key in INPUT_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--part", type=int, default=1)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage == "prepare":
        inputs = {key: getattr(args, key) for key in INPUT_KEYS}
        if any(value is None for value in inputs.values()):
            parser.error("prepare requires all four local input paths")
        result = prepare(args.package_root, experiment, inputs)
    else:
        manifest = verify(experiment, frozen=args.stage in ("train", "finish"))
        result = (plan(experiment, manifest) if args.stage == "plan" else
                  train(experiment, manifest, args.task, args.part) if args.stage == "train" else
                  finish(experiment, manifest))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
