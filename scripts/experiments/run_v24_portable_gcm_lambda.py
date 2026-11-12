#!/usr/bin/env python3
"""Relocate the submitted M1-recipe GC–Mamba sweep without Della job dependencies.

The scientific configuration and checkpoint execution are imported from the
submitted campaign driver. Only local paths and execution orchestration differ.
This module never submits, cancels, or modifies scheduler jobs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_v24_m1_matched_gcm_lambda as campaign
from scripts.experiments.frozen_runtime import freeze_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, write

DRIVER = "scripts/experiments/run_v24_portable_gcm_lambda.py"
WRAPPER = "scripts/experiments/v24_portable_gcm_lambda.sh"
INPUT_NAMES = ("prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir")
REQUIRED_INPUTS = {
    ("baseline_checkpoint", ""),
    ("prepared_root", "metadata.json"),
    ("anchor_manifest_root", "metadata.json"),
    *[("anchor_manifest_root", f"anchors/{name}.npy")
      for name in ("anchor_indices", "anchor_times", "split_train", "split_val")],
    *[("stats_dir", name) for name in
      ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")],
}


def validate_inputs(package_root, input_paths):
    """Validate transferred input identities before writing a local campaign."""
    package_root = Path(package_root).resolve()
    paths = {name: Path(input_paths[name]).resolve() for name in INPUT_NAMES}
    reference = read(package_root / "portable/input_reference.json")
    records = reference.get("files", [])
    identities = [(entry["input"], entry.get("relative_path", "")) for entry in records]
    campaign.require(len(identities) == len(set(identities)), "Duplicate input reference")
    campaign.require(REQUIRED_INPUTS.issubset(identities), "Required input fingerprints are missing")
    result = []
    for entry in records:
        name, relative = entry["input"], entry.get("relative_path", "")
        campaign.require(name in paths, f"Unknown portable input: {name}")
        campaign.require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                         "Input reference must use a relative path")
        path = paths[name] / relative if relative else paths[name]
        campaign.require(path.is_file(), f"Required input is absent: {path}")
        record = pinned(path)
        campaign.require(record["sha256"] == entry["sha256"],
                         f"Input fingerprint differs: {name}/{relative}")
        result.append(record)
    return reference, result


def local_config(template, experiment, task, package_root, input_paths, stage):
    """Apply only local I/O paths before using the scientific campaign builder."""
    template = copy.deepcopy(template)
    for name in INPUT_NAMES:
        template["data"][name] = str(Path(input_paths[name]).resolve())
    return campaign.make_config(template, experiment, task, package_root, stage)


def execution_plan(experiment, manifest):
    wrapper = str(Path(manifest["source_root"]) / WRAPPER)
    return {
        "scheduler": "site supplied; no jobs submitted by this program",
        "array_tasks": {str(i): {"id": t["id"], "lambda": t["lambda"]}
                        for i, t in enumerate(manifest["tasks"])},
        "part1": ["bash", wrapper, "train", str(experiment), "1", "<task:0-4>"],
        "part2": ["bash", wrapper, "train", str(experiment), "2", "<task:0-4>"],
        "part2_dependency": "corresponding part1 success (Slurm aftercorr)",
        "finish": ["bash", wrapper, "finish", str(experiment)],
        "finish_dependency": "all part2 tasks succeed",
        "training_resource_reference": {"gpu_memory_gib": 80, "gpu_count": 1,
            "host_memory_gib": 384, "cpus": 8, "hours_per_part": 30},
        "resource_limit": "Match local GPU type, account, partition and walltime to the site.",
    }


def prepare(package_root, experiment, input_paths):
    from src.models.mamba.v24_Ilya.training.config import load_training_config

    package_root, experiment = Path(package_root).resolve(), Path(experiment).resolve()
    campaign.require(package_root == ROOT, "Run prepare from this package's driver")
    if experiment.exists():
        raise FileExistsError("Use a new campaign directory; existing campaigns are immutable")
    reference, inputs = validate_inputs(package_root, input_paths)
    templates = {stage: read(package_root / "templates" / f"{stage}.json")
                 for stage in ("initial", "main")}
    for name in ("configs", "checks", "logs", "submissions", "provenance"):
        (experiment / name).mkdir(parents=True)
    reference_copy = experiment / "provenance/input_reference.json"
    write(reference_copy, reference)
    inputs.append(pinned(reference_copy))
    for stage, template in templates.items():
        path = experiment / "provenance" / f"template_{stage}.json"
        write(path, template)
        inputs.append(pinned(path))
    tasks = []
    for weight in campaign.LAMBDAS:
        identity = f"gcm_m1matched_s22_l{round(weight * 100):03d}"
        task = {"id": identity, "logical_id": identity, "model_family": "gc_mamba",
                "lambda": weight, "learning_rate": 1e-4, "total_training_updates": 2500}
        for stage, template in templates.items():
            config_path = experiment / "configs" / f"{identity}_{stage}.json"
            write(config_path, local_config(template, experiment, task, package_root, input_paths, stage))
            cfg = load_training_config(config_path)
            campaign.require(cfg.train_order_manifest is None and cfg.feedback_mode == "closed_loop_sg",
                             "Training recipe differs from M1")
            task[f"{stage}_config"] = pinned(config_path)
            inputs.append(task[f"{stage}_config"])
        task["config"] = task["main_config"]
        tasks.append(task)
    for entry in inputs:
        check(entry)
    source = freeze_source(package_root, experiment / "source")
    baseline = pinned(input_paths["baseline_checkpoint"])
    manifest = seal(experiment / "manifest.json", {
        "format": campaign.FORMAT, "portable_format": "v24_portable_gcm_lambda_v1",
        "workspace": str(package_root), "experiment": str(experiment),
        "source_root": str(experiment / "source"), "source_digest": source["source_digest"],
        "inputs": inputs, "baseline": baseline, "tasks": tasks, "completion_hook": None,
        "input_reference": pinned(reference_copy),
        "policy": {"lambdas": list(campaign.LAMBDAS), "seed": 22,
            "initial_updates": 500, "main_updates": 2000, "total_updates_per_arm": 2500,
            "allocation_endpoints_total": [1250, 2500], "main_split_step": 750,
            "stage_transfer": "parameters only; reset optimizer/RNG/cursor/recurrent state",
            "allocation_resume": "exact optimizer/RNG/cursor/recurrent state continuation",
            "feedback_mode": "closed_loop_sg", "training_order": "historical contiguous segment groups",
            "swa_main_steps": [500, 1000, 1500, 2000],
            "swa_total_updates": [1000, 1500, 2000, 2500],
            "finite_diagnostics": "warning", "scientific_improvement_required": False,
            "automatic_retries": 0},
    })
    selection = {"tasks": tasks, "manifest_hash": manifest["record_hash"],
                 "selection": "Prescribed five lambdas; no score-based selection"}
    seal(experiment / "selected_arms_gc_mamba.json", selection)
    seal(experiment / "selected_arms.json", selection)
    write(experiment / "execution_plan.json", execution_plan(experiment, manifest))
    return manifest


def verify(experiment, *, frozen=False):
    manifest = campaign.verify(experiment, frozen=frozen)
    campaign.require(manifest.get("portable_format") == "v24_portable_gcm_lambda_v1",
                     "This is not a portable GC–Mamba campaign")
    campaign.require(manifest.get("completion_hook") is None, "Portable campaign has an unexpected hook")
    if frozen:
        campaign.require(ROOT == Path(manifest["source_root"]), "Execute the frozen portable driver")
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="stage", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--package-root", type=Path, default=ROOT)
    prep.add_argument("--experiment", type=Path, required=True)
    for name in INPUT_NAMES:
        prep.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("plan", "train", "finish"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--experiment", type=Path, required=True)
        if name == "train":
            sub.add_argument("--task", type=int, choices=range(5), required=True)
            sub.add_argument("--part", type=int, choices=(1, 2), required=True)
    return result


def main():
    args = parser().parse_args()
    experiment = args.experiment.resolve()
    if args.stage == "prepare":
        manifest = prepare(args.package_root, experiment,
                           {name: getattr(args, name) for name in INPUT_NAMES})
        result = {"experiment": str(experiment), "manifest_hash": manifest["record_hash"],
                  "source_digest": manifest["source_digest"]}
    else:
        manifest = verify(experiment, frozen=args.stage in ("train", "finish"))
        if args.stage == "plan":
            result = execution_plan(experiment, manifest)
        elif args.stage == "train":
            result = campaign.train(experiment, manifest, args.task, args.part)
        else:
            result = campaign.finish(experiment, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
