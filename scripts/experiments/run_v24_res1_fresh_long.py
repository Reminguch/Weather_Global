#!/usr/bin/env python3
"""Run the frozen-source fresh-width512 experiment and its matched evaluation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import run_v24_res1_width128 as shared


def validate_result(result, reference):
    for key in ("chosen_idx", "evaluated_samples", "evaluation_status", "target_steps",
                "anchor_history_steps", "anchor_index_semantics", "baseline_branch",
                "residual_state_init_resolved", "eval_mode", "sample_total_steps"):
        if result[key] != reference[key]:
            raise ValueError(f"Evaluation differs from reference: {key}")
    return shared.validate_baseline_reference(result, reference)


def archive_directory(experiment, label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = experiment / "checks/recovery_history" / f"{label}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def trim_training_records(directory, completed, experiment):
    """Preserve unsaved trailing updates before replaying from an exact checkpoint."""
    path = directory / "train_metrics.jsonl"
    original = path.read_text()
    lines = original.splitlines()
    records = []
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"Malformed training record at line {i + 1}")
            break
        if int(record["step"]) <= completed:
            records.append(record)
    if [r["step"] for r in records] != list(range(1, completed + 1)):
        raise ValueError("Training history before the resume checkpoint is incomplete")
    retained = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    if len(lines) != len(records) or not original.endswith("\n"):
        backup = archive_directory(experiment, f"resume_step{completed:08d}")
        shutil.copy2(path, backup / path.name)
        temporary = path.with_suffix(".jsonl.recovery_tmp")
        temporary.write_text(retained)
        temporary.replace(path)


def train(experiment, manifest, entry):
    directory = experiment / "runs" / entry["name"]
    config = shared.read(entry["config"])
    target = config["optimizer"]["max_steps"]
    preflight = experiment / "checks/preflight.json"
    if not preflight.exists():
        smoke_directory = experiment / "smoke" / entry["name"]
        smoke_checkpoint = smoke_directory / "checkpoints/checkpoint_step00000003.pkl"
        smoke_report = experiment / "checks" / f"{entry['name']}_training.json"
        smoke_eval = experiment / "checks/smoke_roundtrip.json"
        smoke_complete = (smoke_checkpoint.is_file() and smoke_report.is_file()
                          and shared.read(smoke_report).get("passed") is True)
        if not smoke_complete:
            leftovers = [p for p in (smoke_directory, smoke_report, smoke_eval) if p.exists()]
            if leftovers:
                backup = archive_directory(experiment, "incomplete_smoke")
                for path in leftovers:
                    path.rename(backup / path.name)
            shared.command([sys.executable, "-u", __file__, "smoke", "--experiment", experiment])
        if not smoke_eval.exists():
            shared.command(shared.eval_command(experiment, entry, smoke_checkpoint, smoke_eval, smoke=True))
        result = shared.read(smoke_eval)
        if result["evaluation_status"] != "complete" or result["evaluated_samples"] != 1:
            raise ValueError("Smoke evaluation is incomplete")
        if not math.isfinite(result["original_graphcast_loss"]["full_rollout"]):
            raise ValueError("Smoke evaluation loss is not finite")
        shared.write(preflight, {"passed": True, "checkpoint_roundtrip": True})
    elif shared.read(preflight).get("passed") is not True:
        raise ValueError("Preflight did not pass")

    # The native checkpoint contains Adam moments, RNG, recurrent state and cursor.
    args = [sys.executable, "-u", ROOT / "scripts/experiments/train_v24_res1_fresh_long.py",
            "--config", entry["config"]]
    latest_path = directory / "latest_checkpoint.json"
    completed = 0
    if latest_path.exists():
        latest = shared.read(latest_path)
        checkpoint = Path(latest["checkpoint"])
        if checkpoint.parent.resolve() != (directory / "checkpoints").resolve():
            raise ValueError("Latest checkpoint is outside this run")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        completed = int(latest["completed_step"])
        args += ["--resume", checkpoint]
    elif directory.exists():
        raise RuntimeError(f"Existing run has no resumable checkpoint: {directory}")
    if completed > target:
        raise ValueError("Run has already passed the requested target")
    if completed < target:
        if completed:
            trim_training_records(directory, completed, experiment)
        shared.command(args)

    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    final_path = directory / f"checkpoints/checkpoint_step{target:08d}.pkl"
    final = load_v24_Ilya_training_checkpoint(final_path)
    if final.completed_step != target or final.resolved_training_config != load_training_config(Path(entry["config"])).to_dict():
        raise ValueError("Completed checkpoint does not match the planned run")
    del final
    records = [json.loads(line) for line in (directory / "train_metrics.jsonl").read_text().splitlines()]
    if [r["step"] for r in records] != list(range(1, target + 1)):
        raise ValueError("Training records are not complete and consecutive")
    if not all(math.isfinite(r[k]) for r in records for k in ("loss", "gradient_norm")):
        raise ValueError("Training contains non-finite loss or gradient norms")
    steps = manifest["primary_swa_steps"]
    swa = directory / "swa" / f"swa_step{steps[0]:05d}-{steps[-1]:05d}.pkl"
    if not swa.exists():
        shared.command([sys.executable, "-u", ROOT / "scripts/training/build_v24_Ilya_swa.py",
                        "--inputs", *[directory / f"checkpoints/checkpoint_step{s:08d}.pkl" for s in steps],
                        "--source-steps", *steps, "--output", swa])
    shared.write(experiment / "checks/completed.json", {
        "passed": True, "completed_step": target, "final_checkpoint": str(final_path),
        "swa": str(swa), "swa_steps": steps,
    })


def evaluate(experiment, manifest, entry):
    directory = experiment / "runs" / entry["name"]
    reference = shared.read(experiment / "reference_evaluation.json")
    rows = []
    checks = {}
    for artifact in manifest["evaluation_artifacts"]:
        output = directory / "eval/cold_full_zero_exact_gc/matched32" / f"{artifact['tag']}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            shared.command(shared.eval_command(experiment, entry, directory / artifact["checkpoint"], output))
        result = shared.read(output)
        checks[artifact["tag"]] = validate_result(result, reference)
        loss = result["original_graphcast_loss"]
        rows.append({"artifact": artifact["tag"], "evaluated_samples": result["evaluated_samples"],
                     "graphcast_loss_reduction_pct": loss["improvement_pct_rollout"],
                     "day10_loss_reduction_pct": loss["improvement_pct_per_step"][39],
                     "baseline_loss": loss["baseline_rollout"], "model_loss": loss["full_rollout"],
                     "source": str(output)})
    shared.write(experiment / "checks/evaluation.json", {"passed": True, "baseline_comparisons": checks})
    shared.write(experiment / "results/summary.json", {"rows": rows, "comparison_note": manifest["comparison_note"]})
    with (experiment / "results/summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "smoke", "train", "eval"))
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    shared.verify(experiment)
    manifest = shared.read(experiment / "manifest.json")
    entry = manifest["entries"][0]
    if args.stage == "smoke":
        shared.smoke_train(experiment, entry)
    elif args.stage == "train":
        train(experiment, manifest, entry)
    elif args.stage == "eval":
        evaluate(experiment, manifest, entry)
    else:
        print(f"Verified frozen source and configuration: {experiment}")


if __name__ == "__main__":
    main()
