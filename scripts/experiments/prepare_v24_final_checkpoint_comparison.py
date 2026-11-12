#!/usr/bin/env python3
"""Seal and dispatch final 2500-update raw/SWA MSE and spherical comparisons."""
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

from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

TRAINING = "artifacts/checkpoints/v24_Ilya/res1_fresh_gc_mamba_gc_lambda_20261107"
REFERENCE = "artifacts/evaluations/v24_raw_checkpoint_pilot_20261110"
DEFAULT = "artifacts/evaluations/v24_final_checkpoint_comparison_20261111"
DRIVER = "scripts/experiments/prepare_v24_final_checkpoint_comparison.py"
WRAPPER = "scripts/experiments/v24_final_checkpoint_comparison.slurm"
OVERLAYS = (DRIVER, WRAPPER, "scripts/experiments/run_v24_raw_checkpoint_pilot.py",
            "scripts/analyze_models/report_v24_raw_checkpoint_pilot.py")
SWA_STEPS = (1000, 1500, 2000, 2500)
TRAINING_DEPENDENCIES = ("17049163", "17062358")
LIMITS = (
    "Four matched 2022 development starts, not annual skill or checkpoint selection. "
    "All fresh models complete 2500 updates; raw2500 and equal-weight SWA of "
    "updates1000/1500/2000/2500 are prescribed before the final evaluations. "
    "M1 raw main-step2000 follows500 warm-up updates; its historical SWA500–2000 "
    "uses the same total-update1000/1500/2000/2500 window. M1 forecasts are reused. "
    "Full-GC lambda0 uses peakLR3e-7; positive-lambda full-GC uses1e-5, so their "
    "comparison does not isolate lambda. Finite drift and deterioration remain warnings."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_extension(workspace, parent, destination):
    """Preserve the evaluated computation; overlay orchestration/report files only."""
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
    verify_source(parent)
    write(destination / "snapshot.json", {
        "format": FORMAT, "workspace": str(workspace), "files": files,
        "source_digest": _digest(files), "parent_snapshot": pinned(parent / "snapshot.json"),
        "overlays": list(OVERLAYS)})
    return verify_source(destination)


def training_specs(workspace, training):
    """Pin available identity/progress, but never invent hashes for future weights."""
    specs, inputs = [], []
    for family in ("gc_mamba", "full_gc"):
        selection_path = training / f"selected_arms_{family}.json"
        selection = unseal(selection_path)
        tasks = selection["tasks"]
        require(len(tasks) == 5 and [task["lambda"] for task in tasks] == [0, .25, .5, .75, 1],
                f"Unexpected lambda selection for {family}")
        inputs.append(pinned(selection_path))
        for task in tasks:
            require(task["model_family"] == family, "Selection model family differs")
            check(task["config"])
            run = training / "runs" / task["id"]
            config = pinned(run / "run_config.json")
            cfg = read(config["path"])
            require(cfg["model_family"] == family and cfg["optimizer"]["max_steps"] == 2500,
                    "Run is not the expected family/2500-update schedule")
            proof_path = training / "checks" / f"{task['id']}_step00001250.json"
            proof = unseal(proof_path)
            require(proof["status"] == "complete" and proof["completed_steps"] == 1250
                    and proof["task_id"] == task["id"] and proof["model_family"] == family,
                    "Missing matching completed midpoint")
            check(proof["checkpoint"])
            paths = {str(step): str(run / "checkpoints" / f"checkpoint_step{step:08d}.pkl")
                     for step in SWA_STEPS}
            available = pinned(paths["1000"])
            inputs += [task["config"], config, pinned(proof_path), proof["checkpoint"], available]
            specs.append({
                "id": task["logical_id"], "training_run_id": task["id"],
                "model_family": family, "lambda": task["lambda"],
                "learning_rate": task["learning_rate"], "run_config": config,
                "training_config": task["config"], "midpoint_proof": pinned(proof_path),
                "available_sources": [available], "checkpoint_paths": paths,
                "final_proof_path": str(training / "checks" / f"{task['id']}_step00002500.json"),
                "label": f"{'GC–Mamba' if family == 'gc_mamba' else 'Full GC'} λ={task['lambda']:g}"})
    require(len({s["id"] for s in specs}) == 10, "Duplicate selected arm identity")
    return specs, inputs


def reference_models(reference_root, parent_record, parent):
    models, inputs = [], []
    for identity, kind in (("M1_raw", "raw"), ("M1_swa", "swa")):
        matches = [m for m in parent["models"] if m["id"] == identity]
        require(len(matches) == 1, "Missing/duplicate completed M1 reference")
        model = copy.deepcopy(matches[0])
        directory = reference_root / "evaluation" / identity
        record_path = directory / "complete.json"
        proof = unseal(record_path)
        require(proof["status"] == "complete" and proof["manifest_hash"] == parent["record_hash"]
                and proof["checkpoint"] == model["checkpoint"] and proof["count"] == 4,
                "M1 reference completion does not match the parent campaign")
        for key in ("checkpoint", "evaluation", "pooled_spectra"):
            check(proof[key]); inputs.append(proof[key])
        check(model["run_config"])
        inputs += [pinned(record_path), model["run_config"]]
        model.update(checkpoint_kind=kind, comparison_pair="M1", training_updates=2500,
                     total_training_updates=2500)
        if kind == "swa":
            model["swa_steps"] = [500, 1000, 1500, 2000]
            model["swa_total_updates"] = list(SWA_STEPS)
        model["reuse"] = {"campaign": parent_record, "directory": str(directory),
                          "source_digest": parent["source_digest"], "manifest_hash": parent["record_hash"]}
        models.append(model)
    return models, inputs


def prepare(root, workspace, reference_root, training_dependencies=TRAINING_DEPENDENCIES):
    root, workspace, reference_root = [Path(p).resolve() for p in (root, workspace, reference_root)]
    if root.exists():
        raise FileExistsError("Use a new final-comparison directory; preserve existing evidence")
    training = workspace / TRAINING
    parent_record = pinned(reference_root / "manifest.json")
    parent = unseal(parent_record["path"])
    require(parent["target_steps"] == 40, "Reference must have40 leads")
    check(parent["cohort"])
    cohort = read(parent["cohort"]["path"])
    require(len(cohort["initializations"]) == 4, "Reference must have4 matched starts")
    require(verify_source(Path(parent["source_root"]))["source_digest"] == parent["source_digest"],
            "Reference runtime mismatch")
    training_record = pinned(training / "manifest.json")
    training_manifest = unseal(training_record["path"])
    specs, inputs = training_specs(workspace, training)
    models, reference_inputs = reference_models(reference_root, parent_record, parent)
    inputs += [parent_record, training_record, *parent["inputs"], *reference_inputs]
    for item in inputs:
        check(item)
    for spec in specs:
        cfg = read(spec["run_config"]["path"])
        require(pinned(workspace / cfg["data"]["baseline_checkpoint"])["sha256"]
                == parent["baseline"]["sha256"], "Training baseline differs from M1 reference")
        require((workspace / cfg["data"]["stats_dir"]).resolve() == Path(parent["stats_dir"]).resolve(),
                "Training normalization differs from reference")
    require(training_dependencies and all(str(job).isdigit() for job in training_dependencies),
            "Expected whole-array numeric Slurm dependencies")
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "evaluation", "checks", "swa"):
        (root / sub).mkdir()
    snapshot = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    template = copy.deepcopy(parent)
    template.pop("record_hash")
    template.update(format="v24_final_checkpoint_comparison_v1", models=models,
        source_root=str(root / "source"), source_digest=snapshot["source_digest"],
        inputs=inputs, parent_campaign=parent_record, training_campaign=training_record,
        image_root=str(workspace / "plots/analyze_models/images/resolution_eval" / root.name),
        data_root=str(workspace / "plots/analyze_models/data/resolution_eval" / root.name),
        report_title="Final 2500-update checkpoints: raw versus SWA MSE and power spectra",
        figure_title="2500-update raw/SWA comparison", report_limits=LIMITS)
    template["protocol"].update(limitations=LIMITS, checkpoint_selection="Prescribed final raw2500 and SWA1000/1500/2000/2500",
                                swa_steps=list(SWA_STEPS), swa_weights=[.25] * 4)
    return seal(root / "plan.json", {
        "format": "v24_final_checkpoint_comparison_plan_v1", "workspace": str(workspace),
        "source_root": template["source_root"], "source_digest": snapshot["source_digest"],
        "training_dependencies": list(training_dependencies), "training_manifest_hash": training_manifest["record_hash"],
        "training_source_digest": training_manifest["source_digest"],
        "swa_steps": list(SWA_STEPS), "inputs": inputs, "specs": specs, "manifest_template": template,
        "resources": {"materialize": {"cpus": 2, "mem_gib": 8, "minutes": 30},
                      "evaluate": {"cpus": 4, "mem_gib": 16, "minutes": 62, "gpu": 1},
                      "report": {"cpus": 2, "mem_gib": 4, "minutes": 15}},
        "resource_evidence": "Nov10 four-date pilots:14m44s–15m46s/task, maximum8.75GiB RSS. CPU SWA loads4 checkpoints/arm (about1.2GiB GCM/1.6GiB fullGC) sequentially across10 arms;8GiB allocation."})


def verify_plan(root):
    plan = unseal(root / "plan.json")
    require(verify_source(Path(plan["source_root"]))["source_digest"] == plan["source_digest"],
            "Final comparison source snapshot changed")
    for item in plan["inputs"]:
        check(item)
    return plan


def final_sources(plan, spec):
    """Require valid final training completion; numerical skill is not a condition."""
    proof_record = pinned(spec["final_proof_path"])
    proof = unseal(proof_record["path"])
    require(proof["status"] == "complete" and proof["completed_steps"] == 2500
            and proof["task_id"] == spec["training_run_id"]
            and proof["model_family"] == spec["model_family"]
            and proof["lambda"] == spec["lambda"]
            and proof["manifest_hash"] == plan["training_manifest_hash"]
            and proof["source_digest"] == plan["training_source_digest"],
            "Final training completion does not match the sealed arm")
    sources = [pinned(spec["checkpoint_paths"][str(step)]) for step in SWA_STEPS]
    require(proof["checkpoint"] == sources[-1], "Final training checkpoint differs")
    check(proof["metrics"])
    return sources, [proof_record, proof["metrics"]]


def build_swa(root, plan, spec, sources):
    import jax
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import build_v24_Ilya_swa, load_v24_Ilya_checkpoint
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    destination = root / "swa" / f"{spec['id']}_steps01000-02500.pkl"
    proof_path = root / "checks" / f"swa_{spec['id']}.json"
    if proof_path.exists():
        proof = unseal(proof_path)
        require(proof["plan_hash"] == plan["record_hash"] and proof["sources"] == sources,
                "Previously built SWA has different inputs")
        check(proof["checkpoint"])
        return proof["checkpoint"], pinned(proof_path)
    if destination.exists():
        raise FileExistsError(f"Unverified SWA output exists; preserve it and use a new campaign: {destination}")
    expected_config = read(spec["run_config"]["path"])
    stored_steps = tuple(spec.get("swa_steps", SWA_STEPS))
    require(len(stored_steps) == 4, "SWA requires four prescribed checkpoints")
    for step, record in zip(stored_steps, sources, strict=True):
        checkpoint = load_v24_Ilya_training_checkpoint(Path(record["path"]))
        require(checkpoint.completed_step == step and checkpoint.model_family == spec["model_family"],
                "SWA input step/model family differs")
        require(checkpoint.baseline_checkpoint_fingerprint == plan["manifest_template"]["baseline"]["sha256"],
                "SWA input baseline identity differs")
        require(all(key in expected_config and expected_config[key] == value
                    for key, value in checkpoint.resolved_training_config.items()),
                "SWA input resolved training configuration differs from the pinned run")
        require(all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(checkpoint.residual_params)),
                "Nonfinite SWA input parameters")
        del checkpoint
    payload = build_v24_Ilya_swa([Path(s["path"]) for s in sources], list(stored_steps), destination,
                                residual_state_policy="zero")
    require(all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(payload["residual_params"])),
            "Nonfinite SWA output parameters")
    loaded = load_v24_Ilya_checkpoint(destination)
    require(loaded.model_family == spec["model_family"] and not loaded.has_residual_state,
            "SWA model family/state policy differs")
    for record in sources:
        check(record)
    proof = seal(proof_path, {"status": "complete", "plan_hash": plan["record_hash"],
        "comparison_pair": spec["id"], "source_steps": list(stored_steps), "sources": sources,
        "weights": [.25] * 4, "residual_state_policy": "zero", "checkpoint": pinned(destination)})
    return proof["checkpoint"], pinned(proof_path)


def materialize(root):
    plan = verify_plan(root)
    if (root / "manifest.json").exists():
        manifest = unseal(root / "manifest.json")
        require(manifest["plan_hash"] == plan["record_hash"], "Existing final manifest uses a different plan")
        for item in [*manifest["inputs"], *(m["checkpoint"] for m in manifest["models"])]:
            check(item)
        return manifest
    # Check every endpoint before building any average; never substitute a midpoint.
    final = [(spec, *final_sources(plan, spec)) for spec in plan["specs"]]
    manifest = copy.deepcopy(plan["manifest_template"])
    manifest["plan_hash"] = plan["record_hash"]
    manifest["inputs"].append(pinned(root / "plan.json"))
    for spec, sources, proofs in final:
        averaged, average_proof = build_swa(root, plan, spec, sources)
        common = {key: spec[key] for key in ("model_family", "lambda", "learning_rate", "run_config", "training_run_id")}
        common.update(step=2500, training_updates=2500, total_training_updates=2500,
                      comparison_pair=spec["id"], swa_steps=list(SWA_STEPS),
                      selection="Predetermined final raw2500/SWA1000–2500; no score selection")
        for kind, checkpoint, suffix in (("raw", sources[-1], "raw update2500"),
                                         ("swa", averaged, "SWA updates1000–2500")):
            manifest["models"].append({**common, "id": f"{spec['id']}_{kind}",
                "label": f"{spec['label']} — {suffix}", "checkpoint_kind": kind, "checkpoint": checkpoint})
        manifest["inputs"] += sources + proofs + [average_proof]
    require(len(manifest["models"]) == 22, "Incomplete raw/SWA model set")
    for item in manifest["inputs"]:
        check(item)
    return seal(root / "manifest.json", manifest)


def submission_command(root, plan, stage, dependency):
    require(stage in ("materialize", "evaluate", "report"), "Unknown submission stage")
    budget = plan["resources"][stage]
    args = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={plan['workspace']}",
            "--kill-on-invalid-dep=yes", f"--job-name=final2500-{stage}",
            f"--output={root}/logs/{stage}_%A_%a.out", f"--error={root}/logs/{stage}_%A_%a.err",
            f"--cpus-per-task={budget['cpus']}", f"--mem={budget['mem_gib']}G", f"--time={budget['minutes']}"]
    if stage == "evaluate":
        args += ["--gres=gpu:1", "--constraint=gpu80", "--qos=gpu-short", "--array=2-21"]
    if dependency:
        args.append(f"--dependency=afterok:{dependency}")
    return args + [str(Path(plan["source_root"]) / WRAPPER), stage, str(root), plan["source_root"]]


def submission_plan(root, plan):
    return {stage: submission_command(root, plan, stage, dependency) for stage, dependency in (
        ("materialize", ":".join(plan["training_dependencies"])),
        ("evaluate", "<materialize-job>"), ("report", "<evaluate-job>"))}


def submit(root, plan):
    results = {}
    dependency = ":".join(plan["training_dependencies"])
    for stage in ("materialize", "evaluate", "report"):
        command = submission_command(root, plan, stage, dependency)
        print(json.dumps({"stage": stage, "command": command}), flush=True)
        result = submit_once(root, stage, command)
        results[stage] = result
        dependency = str(result.get("job_id", result.get("jobid"))) if isinstance(result, dict) else str(result)
        require(dependency.isdigit(), "Submission returned an invalid job identity")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "submit", "materialize"))
    parser.add_argument("--root", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=ROOT / REFERENCE)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        plan = prepare(root, args.workspace, args.reference_root)
        result = {"plan": str(root / "plan.json"), "plan_hash": plan["record_hash"],
                  "models": 22, "new_evaluations": 20, "commands": submission_plan(root, plan)}
    else:
        plan = verify_plan(root)
        if args.stage == "materialize":
            require(ROOT == Path(plan["source_root"]), "Materialize from the frozen source")
            manifest = materialize(root)
            result = {"manifest": str(root / "manifest.json"), "models": len(manifest["models"]),
                      "manifest_hash": manifest["record_hash"]}
        elif args.stage == "plan":
            result = submission_plan(root, plan)
        else:
            result = submit(root, plan)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
