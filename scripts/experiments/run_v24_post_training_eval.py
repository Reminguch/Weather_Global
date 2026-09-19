#!/usr/bin/env python3
"""Discover retained checkpoints after training, build SWA, and submit exact eval."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SHARDS = 4


def validate_graphcast_runtime():
    """Reject incomplete snapshots before loading data or submitting an array."""
    expected = ROOT / "third_party/graphcast/graphcast/graphcast.py"
    if not expected.is_file():
        raise RuntimeError(
            f"Missing vendored GraphCast: {expected}. "
            "Include third_party/graphcast in the evaluation source snapshot; "
            "the installed GraphCast package does not implement the Mamba model."
        )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.models.graphcast.training.core import bootstrap  # noqa: F401
    from graphcast import graphcast as gc

    if Path(gc.__file__).resolve() != expected.resolve():
        raise RuntimeError(
            f"Evaluation loaded GraphCast from {gc.__file__}; expected {expected}. "
            "Start a fresh process with the snapshot and its third_party/graphcast "
            "first on PYTHONPATH."
        )
    if (
        "is_training" not in inspect.signature(gc.GraphCast._run_mesh_gnn).parameters
        or not callable(getattr(gc.GraphCast, "_run_mesh_gnn_interleaved", None))
    ):
        raise RuntimeError(f"GraphCast at {expected} lacks the required Mamba extensions")


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def discover(run):
    return sorted((int(p.stem.removeprefix("checkpoint_step")), p.resolve())
                  for p in (run / "checkpoints").glob("checkpoint_step*.pkl")
                  if p.stem.removeprefix("checkpoint_step").isdigit())


def swa_windows(steps):
    if len(steps) < 2:
        return []
    windows = [("swa_all", steps)]
    if len(steps) > 3:
        windows.append(("swa_last3", steps[-3:]))
    return windows


def command(args):
    print(shlex.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def eval_args(config, checkpoint, output, shard):
    args = ["--ckpt", str(checkpoint), "--ckpt-in", config["data"]["baseline_checkpoint"],
            "--stats-dir", config["data"]["stats_dir"], "--data-path",
            "data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr",
            "--val-year", "2022", "--train-start-year", "2019", "--train-end-year", "2021",
            "--input-duration", "12h", "--eval-mode", "cold_full",
            "--residual-state-init", "zero", "--residual-alpha", "1",
            "--target-steps", "40", "--warmup-steps", "24", "--n-samples", "32",
            "--seed", "0", "--omit-rms-bias", "--anchor-shard-count", str(SHARDS),
            "--anchor-shard-index", str(shard), "--merge-state-out",
            str(output.with_suffix(".merge_state.pkl")), "--out-json", str(output)]
    for key, value in config["architecture"].items():
        if value is None:
            continue
        if isinstance(value, bool):
            flags = {"temporal_conv_bias": (False, "--no-temporal-conv-bias"),
                     "temporal_stateful": (True, "--temporal-stateful"),
                     "temporal_zero_init_out": (False, "--no-zero-init-out"),
                     "temporal_bias": (True, "--temporal-bias")}
            expected, flag = flags[key]
            if value == expected:
                args.append(flag)
        else:
            args.extend(["--" + key.replace("_", "-"), str(value)])
    return args


def submit(workflow, phase, options):
    args = ["sbatch", "--parsable", *options,
            str(ROOT / "scripts/experiments/v24_post_training_eval.slurm"),
            phase, str(workflow)]
    print(shlex.join(args), flush=True)
    result = subprocess.check_output(args, text=True).strip()
    job = result.split(";")[0]
    if not job.isdigit():
        raise ValueError(f"Unexpected sbatch result: {result}")
    write_json(workflow / f"submission_{phase}.json", {"job_id": job, "command": args})
    return job


def prepare(workflow):
    validate_graphcast_runtime()
    settings = json.loads((workflow / "settings.json").read_text())
    run = Path(settings["run_dir"])
    found = discover(run)
    if not found:
        raise ValueError("No saved checkpoints to evaluate")
    if (workflow / "manifest.json").exists():
        raise FileExistsError("Preparation already completed; inspect submission records before retrying")
    config = json.loads((run / "run_config.json").read_text())
    variants = [{"tag": f"step{step:08d}", "checkpoint": str(path),
                 "source_steps": [step], "sha256": sha256(path)} for step, path in found]
    paths = dict(found)
    for label, steps in swa_windows(list(paths)):
        tag = f"{label}_step{steps[0]:08d}-{steps[-1]:08d}"
        output = workflow / "swa" / f"{tag}.pkl"
        command([sys.executable, ROOT / "scripts/training/build_v24_Ilya_swa.py",
                 "--inputs", *(paths[s] for s in steps), "--source-steps", *steps,
                 "--output", output])
        variants.append({"tag": tag, "checkpoint": str(output), "source_steps": steps,
                         "sha256": sha256(output)})
    write_json(workflow / "manifest.json", {"training_job": settings["training_job"],
               "config": config, "variants": variants, "shards": SHARDS,
               "evaluation": "32 cold starts, zero state, 40 leads, exact GraphCast loss",
               "last_three_alias_all": len(found) <= 3})
    dependency = os.environ["SLURM_JOB_ID"]
    eval_job = submit(workflow, "evaluate", [f"--dependency=afterok:{dependency}",
        "--job-name=v24-r025-ckpts-eval", f"--array=0-{SHARDS * len(variants) - 1}",
        "--qos=gpu", "--gres=gpu:1", "--constraint=gpu80", "--cpus-per-task=8",
        "--mem=128G", "--time=05:00:00"])
    submit(workflow, "merge", [f"--dependency=afterok:{eval_job}",
        "--job-name=v24-r025-ckpts-merge", "--qos=short", "--cpus-per-task=2",
        "--mem=16G", "--time=01:00:00"])


def evaluate(workflow, manifest):
    validate_graphcast_runtime()
    index, shard = divmod(int(os.environ["SLURM_ARRAY_TASK_ID"]), SHARDS)
    variant = manifest["variants"][index]
    checkpoint = Path(variant["checkpoint"])
    if sha256(checkpoint) != variant["sha256"]:
        raise ValueError("Checkpoint changed after preparation")
    output = workflow / "shards" / f"{variant['tag']}_shard{shard}.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    command([sys.executable, "-u", ROOT / "scripts/analyze_models/eval_v24_Ilya.py",
             *eval_args(manifest["config"], checkpoint, output, shard)])


def merge(workflow, manifest):
    validate_graphcast_runtime()
    rows = []
    for variant in manifest["variants"]:
        output = workflow / f"{variant['tag']}.json"
        args = [sys.executable, ROOT / "scripts/analyze_models/merge_v24_Ilya_eval_shards.py"]
        for shard in range(SHARDS):
            path = workflow / "shards" / f"{variant['tag']}_shard{shard}.json"
            args.extend(["--shard-json", path, "--merge-state", path.with_suffix(".merge_state.pkl")])
        command([*args, "--out-json", output, "--delete-merge-states"])
        metric = json.loads(output.read_text())["original_graphcast_loss"]
        rows.append({"tag": variant["tag"], "source_steps": variant["source_steps"],
                     "baseline_loss": metric["baseline_rollout"],
                     "model_loss": metric["full_rollout"],
                     "improvement_pct": metric["improvement_pct_rollout"],
                     "day10_improvement_pct": metric["improvement_pct_per_step"][39]})
    write_json(workflow / "comparison.json", rows)
    lines = ["# Checkpoint and SWA exact GraphCast evaluation", "",
             "32 matched cold starts; zero recurrent state; 40 six-hour leads.", "",
             "| Checkpoint | Source steps | Exact loss | Loss reduction (%) | Day 10 reduction (%) |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['tag']} | {row['source_steps']} | {row['model_loss']:.6f} | "
                     f"{row['improvement_pct']:.4f} | {row['day10_improvement_pct']:.4f} |")
    (workflow / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "merge"))
    parser.add_argument("workflow", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.workflow)
    else:
        manifest = json.loads((args.workflow / "manifest.json").read_text())
        {"evaluate": evaluate, "merge": merge}[args.phase](args.workflow, manifest)


if __name__ == "__main__":
    main()
