#!/usr/bin/env python3
"""Compare the fresh M1-recipe GC–Mamba sweep with the unchanged full-GC sweep."""
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

from scripts.experiments import prepare_v24_final_checkpoint_comparison as original
from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/prepare_v24_mixed_final_comparison.py"
WRAPPER = "scripts/experiments/v24_mixed_final_comparison.slurm"
DEFAULT = "artifacts/evaluations/v24_mixed_final_comparison_20261112"
GCM = "artifacts/checkpoints/v24_Ilya/res1_m1_matched_gcm_lambda_20261112"
FULL_GC = original.TRAINING
REFERENCE = original.REFERENCE
TOTAL_SWA_STEPS = (1000, 1500, 2000, 2500)
GCM_SWA_STEPS = (500, 1000, 1500, 2000)
OVERLAYS = (DRIVER, WRAPPER, "scripts/experiments/prepare_v24_final_checkpoint_comparison.py",
            "scripts/experiments/run_v24_raw_checkpoint_pilot.py",
            "scripts/analyze_models/report_v24_raw_checkpoint_pilot.py")
LIMITS = (
    "Four matched 2022 development starts and 40 six-hour leads, not annual skill. "
    "All models have 2500 total updates. Fresh GC–Mamba uses the historical M1 "
    "500+2000-update recipe, native BF16 MSE and stopped weather gradients; full GC "
    "retains its existing 2500-update continuous schedule, FP32 grid MSE and full "
    "weather gradients. This is not an identical-training-protocol architecture test. "
    "Full-GC lambda0 retains peakLR3e-7, positive lambda retains1e-5. "
    "Raw final checkpoints and equal-weight SWAs at total1000/1500/2000/2500 are "
    "prescribed. GC–Mamba stored main steps500/1000/1500/2000 include500 prior "
    "updates. M1 raw/SWA forecasts are reused. Finite drift and deterioration "
    "remain advisory; no numerical improvement is required for completion."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_extension(workspace, parent, destination):
    """Keep all forecasting/metric source byte-identical to the completed pilot."""
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


def full_gc_specs(training):
    manifest_record = pinned(training / "manifest.json")
    manifest = unseal(manifest_record["path"])
    selection_record = pinned(training / "selected_arms_full_gc.json")
    tasks = unseal(selection_record["path"])["tasks"]
    require([task["lambda"] for task in tasks] == [0, .25, .5, .75, 1],
            "Expected unchanged five full-GC arms")
    specs, inputs = [], [manifest_record, selection_record]
    for task in tasks:
        require(task["model_family"] == "full_gc", "Full-GC selection family differs")
        check(task["config"])
        run = training / "runs" / task["id"]
        config = pinned(run / "run_config.json")
        require(read(config["path"])["optimizer"]["max_steps"] == 2500,
                "Full-GC schedule changed")
        midpoint_record = pinned(training / "checks" / f"{task['id']}_step00001250.json")
        midpoint = unseal(midpoint_record["path"])
        require(midpoint["status"] == "complete" and midpoint["completed_steps"] == 1250
                and midpoint["task_id"] == task["id"], "Missing full-GC midpoint evidence")
        check(midpoint["checkpoint"])
        paths = {str(step): str(run / "checkpoints" / f"checkpoint_step{step:08d}.pkl")
                 for step in TOTAL_SWA_STEPS}
        available = pinned(paths["1000"])
        inputs += [task["config"], config, midpoint_record, midpoint["checkpoint"], available]
        specs.append({"id": task["logical_id"], "training_run_id": task["id"],
            "model_family": "full_gc", "lambda": task["lambda"],
            "learning_rate": task["learning_rate"], "run_config": config,
            "training_config": task["config"], "run_config_path": config["path"],
            "checkpoint_paths": paths, "swa_steps": list(TOTAL_SWA_STEPS), "final_stored_step": 2500,
            "training_manifest_hash": manifest["record_hash"],
            "training_source_digest": manifest["source_digest"],
            "final_proof_path": str(training / "checks" / f"{task['id']}_step00002500.json"),
            "label": f"Full GC λ={task['lambda']:g} (existing recipe)"})
    return specs, inputs


def gcm_specs(training):
    """Pin declared stage configs now; actual run configs/weights do not yet exist."""
    record = pinned(training / "manifest.json")
    manifest = unseal(record["path"])
    require([task["lambda"] for task in manifest["tasks"]] == [0, .25, .5, .75, 1],
            "Expected five fresh GC–Mamba arms")
    specs, inputs = [], [record]
    for task in manifest["tasks"]:
        require(task["model_family"] == "gc_mamba", "New campaign must contain GC–Mamba only")
        for key in ("initial_config", "main_config"):
            check(task[key]); inputs.append(task[key])
        cfg = read(task["main_config"]["path"])
        require(cfg["optimizer"]["max_steps"] == 2000 and cfg["sequence"]["feedback_mode"] == "closed_loop_sg",
                "New GC–Mamba main schedule/feedback differs")
        run = training / "runs" / task["id"]
        specs.append({"id": task["logical_id"], "training_run_id": task["id"],
            "model_family": "gc_mamba", "lambda": task["lambda"],
            "learning_rate": task["learning_rate"], "training_config": task["main_config"],
            "initial_config": task["initial_config"], "run_config_path": str(run / "run_config.json"),
            "checkpoint_paths": {str(step): str(run / "checkpoints" / f"checkpoint_step{step:08d}.pkl")
                                 for step in GCM_SWA_STEPS},
            "swa_steps": list(GCM_SWA_STEPS), "final_stored_step": 2000,
            "training_manifest_hash": manifest["record_hash"],
            "training_source_digest": manifest["source_digest"],
            "final_proof_path": str(training / "checks" / f"{task['id']}_total00002500.json"),
            "label": f"GC–Mamba λ={task['lambda']:g} (M1 recipe)"})
    return specs, inputs


def prepare(root, workspace, reference_root, gcm_training, full_gc_training, training_dependencies):
    from scripts.analyze_models.report_v24_raw_checkpoint_pilot import verify_computational_sources
    root, workspace, reference_root, gcm_training, full_gc_training = [Path(p).resolve()
        for p in (root, workspace, reference_root, gcm_training, full_gc_training)]
    require(not root.exists(), "Use a new comparison directory; preserve completed evidence")
    require(len(training_dependencies) == 2 and all(str(job).isdigit() for job in training_dependencies),
            "Supply unchanged full-GC and new GC–Mamba second-part array IDs")
    parent_record = pinned(reference_root / "manifest.json")
    parent = unseal(parent_record["path"])
    check(parent["cohort"])
    require(parent["target_steps"] == 40 and len(read(parent["cohort"]["path"])["initializations"]) == 4,
            "Expected completed four-start/40-lead pilot")
    require(verify_source(Path(parent["source_root"]))["source_digest"] == parent["source_digest"],
            "Reference source differs")
    gc_specs, gc_inputs = full_gc_specs(full_gc_training)
    new_specs, new_inputs = gcm_specs(gcm_training)
    specs = new_specs + gc_specs
    models, reference_inputs = original.reference_models(reference_root, parent_record, parent)
    inputs = [parent_record, *parent["inputs"], *gc_inputs, *new_inputs, *reference_inputs]
    for spec in specs:
        cfg = read(spec["training_config"]["path"])
        require(pinned(workspace / cfg["data"]["baseline_checkpoint"])["sha256"] == parent["baseline"]["sha256"],
                "Training public backbone differs")
        require((workspace / cfg["data"]["stats_dir"]).resolve() == Path(parent["stats_dir"]).resolve(),
                "Training statistics differ")
    for item in inputs:
        check(item)
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "evaluation", "checks", "swa"):
        (root / sub).mkdir()
    snapshot = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    template = copy.deepcopy(parent)
    template.pop("record_hash")
    template.update(models=models, source_root=str(root / "source"), source_digest=snapshot["source_digest"],
        format="v24_mixed_final_comparison_v1", inputs=inputs, parent_campaign=parent_record,
        image_root=str(workspace / "plots/analyze_models/images/resolution_eval" / root.name),
        data_root=str(workspace / "plots/analyze_models/data/resolution_eval" / root.name),
        report_title="2500 total updates: M1-recipe GC–Mamba and unchanged full GC, raw versus SWA",
        figure_title="2500 total updates: two training recipes", report_limits=LIMITS)
    template["protocol"].update(limitations=LIMITS, swa_total_updates=list(TOTAL_SWA_STEPS),
                                swa_weights=[.25]*4, checkpoint_selection="Prescribed final raw and SWA endpoints")
    verify_computational_sources(parent, template)
    return seal(root / "plan.json", {
        "format": "v24_mixed_final_comparison_plan_v1", "workspace": str(workspace),
        "source_root": template["source_root"], "source_digest": snapshot["source_digest"],
        "training_dependencies": list(training_dependencies), "specs": specs,
        "inputs": inputs, "manifest_template": template,
        "resources": {"materialize": {"cpus": 2, "mem_gib": 8, "minutes": 30},
                      "evaluate": {"cpus": 4, "mem_gib": 16, "minutes": 62},
                      "report": {"cpus": 2, "mem_gib": 4, "minutes": 15}},
        "resource_evidence": "Completed four-date pilots:14m44s–15m46s/model, maximum8.75GiB RSS; sequential four-checkpoint SWA fits8GiB."})


def verify_plan(root):
    plan = unseal(root / "plan.json")
    require(verify_source(Path(plan["source_root"]))["source_digest"] == plan["source_digest"],
            "Mixed comparison frozen source differs")
    for item in plan["inputs"]:
        check(item)
    return plan


def final_sources(spec):
    """Resolve real completion records without inventing future checkpoint hashes."""
    resolved = copy.deepcopy(spec)
    record = pinned(spec["final_proof_path"])
    proof = unseal(record["path"])
    require(proof["status"] == "complete" and proof["completed_steps"] == spec["final_stored_step"]
            and proof["task_id"] == spec["training_run_id"] and proof["model_family"] == spec["model_family"]
            and proof["lambda"] == spec["lambda"] and proof["manifest_hash"] == spec["training_manifest_hash"]
            and proof["source_digest"] == spec["training_source_digest"], "Final training completion differs")
    proofs = [record, proof["metrics"]]
    if spec["model_family"] == "gc_mamba":
        require(proof["total_training_updates"] == 2500, "New GC–Mamba did not complete2500 total updates")
        check(proof["initial_stage_proof"])
        initial = unseal(proof["initial_stage_proof"]["path"])
        require(initial["status"] == "complete" and initial["completed_steps"] == 500
                and initial["total_training_updates"] == 500
                and initial["manifest_hash"] == spec["training_manifest_hash"], "Initial500 evidence differs")
        resolved["run_config"] = pinned(spec["run_config_path"])
        require(proof["run_config"] == resolved["run_config"], "Actual main run configuration differs")
        cfg = read(resolved["run_config"]["path"])
        require(Path(cfg["config_source"]).resolve() == Path(spec["training_config"]["path"]).resolve(),
                "Main run used a different declared configuration")
        proofs += [proof["initial_stage_proof"], initial["metrics"], resolved["run_config"]]
    sources = [pinned(spec["checkpoint_paths"][str(step)]) for step in spec["swa_steps"]]
    require(proof["checkpoint"] == sources[-1], "Final checkpoint differs")
    for item in proofs:
        check(item)
    return resolved, sources, proofs


def materialize(root):
    plan = verify_plan(root)
    if (root / "manifest.json").exists():
        manifest = unseal(root / "manifest.json")
        require(manifest["plan_hash"] == plan["record_hash"], "Existing manifest uses a different plan")
        for item in [*manifest["inputs"], *(model["checkpoint"] for model in manifest["models"])]:
            check(item)
        return manifest
    final = [final_sources(spec) for spec in plan["specs"]]
    manifest = copy.deepcopy(plan["manifest_template"])
    manifest["plan_hash"] = plan["record_hash"]
    manifest["inputs"].append(pinned(root / "plan.json"))
    for spec, sources, proofs in final:
        average, average_proof = original.build_swa(root, plan, spec, sources)
        common = {key: spec[key] for key in ("model_family", "lambda", "learning_rate", "run_config", "training_run_id")}
        common.update(step=spec["final_stored_step"], training_updates=2500, total_training_updates=2500,
            comparison_pair=spec["id"], swa_steps=spec["swa_steps"], swa_total_updates=list(TOTAL_SWA_STEPS),
            selection="Prescribed raw total2500 and SWA at total1000/1500/2000/2500")
        for kind, checkpoint, label in (("raw", sources[-1], "raw total2500"),
                                       ("swa", average, "SWA total1000–2500")):
            manifest["models"].append({**common, "id": f"{spec['id']}_{kind}",
                "label": f"{spec['label']} — {label}", "checkpoint_kind": kind, "checkpoint": checkpoint})
        manifest["inputs"] += sources + proofs + [average_proof]
    require(len(manifest["models"]) == 22, "Expected ten raw/SWA pairs and two M1 references")
    for item in manifest["inputs"]:
        check(item)
    return seal(root / "manifest.json", manifest)


def submission_command(root, plan, stage, dependency):
    budget = plan["resources"][stage]
    args = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={plan['workspace']}",
        "--kill-on-invalid-dep=yes", f"--job-name=mixed2500-{stage}",
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
    records, dependency = {}, ":".join(plan["training_dependencies"])
    for stage in ("materialize", "evaluate", "report"):
        command = submission_command(root, plan, stage, dependency)
        print(json.dumps({"stage": stage, "command": command}), flush=True)
        result = submit_once(root, stage, command)
        records[stage] = result
        dependency = str(result.get("job_id", result.get("jobid"))) if isinstance(result, dict) else str(result)
        require(dependency.isdigit(), "Invalid submitted job identity")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "submit", "materialize"))
    parser.add_argument("--root", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=ROOT / REFERENCE)
    parser.add_argument("--gcm-training", type=Path, default=ROOT / GCM)
    parser.add_argument("--full-gc-training", type=Path, default=ROOT / FULL_GC)
    parser.add_argument("--training-dependencies", nargs=2, metavar=("FULL_GC_ARRAY", "GCM_ARRAY"))
    args = parser.parse_args(); root = args.root.resolve()
    if args.stage == "prepare":
        if not args.training_dependencies:
            parser.error("prepare requires both final training array IDs")
        plan = prepare(root, args.workspace, args.reference_root, args.gcm_training,
                       args.full_gc_training, args.training_dependencies)
        result = {"plan": str(root / "plan.json"), "plan_hash": plan["record_hash"],
                  "commands": submission_plan(root, plan)}
    else:
        plan = verify_plan(root)
        if args.stage == "materialize":
            require(ROOT == Path(plan["source_root"]), "Materialize from frozen source")
            manifest = materialize(root)
            result = {"manifest": str(root / "manifest.json"), "models": len(manifest["models"])}
        elif args.stage == "plan":
            result = submission_plan(root, plan)
        else:
            result = submit(root, plan)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
