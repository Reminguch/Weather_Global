#!/usr/bin/env python3
"""Prepare and submit the three authorized fresh width128 cached experiments."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import run_v24_cached_pair as pair
from src.models.mamba.v24_Ilya.checkpoint import atomic_json_dump, file_sha256
from src.models.mamba.v24_Ilya.training.cached_config import load_cached_training_config

REFERENCE = ROOT / "artifacts/checkpoints/v24_Ilya/res1_fresh_w128_constlr_20260914/configs/w128_fresh_mamba1_di16_bcg1_lr1em4_const20k_seed22.json"
CACHE = ROOT / "data/graphcast/graphcast/dataset/precomputed_baselines/v24_res1_gcsmall_bptt24_k20"
ARMS = (("di32_carry", 32, True), ("di64_carry", 64, True), ("di16_reset_each_prediction", 16, False))


def arm_config(reference: dict, *, di: int, stateful: bool) -> dict:
    value = copy.deepcopy(reference)
    if value["architecture"]["residual_width"] != 128 or value["architecture"]["residual_initialization"] != "fresh":
        raise ValueError("The reference must use fresh width128 residual initialization")
    value["architecture"].update(temporal_d_inner=di, temporal_stateful=stateful)
    value["sequence"].update(feedback_mode="baseline", temporal_state_policy="carry")
    # The cached evaluator always streams every held-out segment.
    value["validation"].update(num_segments=None, final_num_segments=None)
    return value


def prepare(root: Path, *, reference_path: Path = REFERENCE, cache_root: Path = CACHE) -> dict:
    root, reference_path, cache_root = (Path(path).resolve() for path in (root, reference_path, cache_root))
    reference = json.loads(reference_path.read_text())
    entries = []
    for name, di, stateful in ARMS:
        config_path = root / "configs" / f"{name}.json"
        pair.write_once_json(config_path, arm_config(reference, di=di, stateful=stateful))
        load_cached_training_config(config_path, cache_root=cache_root)
        experiment = root / name
        manifest = pair.prepare_experiment(
            experiment, cache_root=cache_root, workspace=ROOT, training_config=config_path,
            cached_only=True, smoke_time_minutes=360, smoke_memory_gib=64,
            job_name=f"v24-w128-{name}",
        )
        entries.append({"name": name, "di": di, "stateful": stateful,
                        "experiment_root": str(experiment), "source_digest": manifest["source_digest"],
                        "config_sha256": file_sha256(Path(manifest["configs"]["cached"]))})
    sweep = {
        "format": "v24_cached_width128_sweep_v1", "experiment_root": str(root),
        "reference_config": str(reference_path), "reference_sha256": file_sha256(reference_path),
        "cache_root": str(cache_root), "entries": entries,
        "initialization": "fresh; no weights loaded from the reference trained checkpoint",
        "reset_ablation": "both SSM and convolution state reset before every prediction",
        "feedback": "fixed cached GraphCast baseline trajectories",
        "max_steps": reference["optimizer"]["max_steps"],
        "validation": "all 15 complete held-out 2022 segments every 2000 updates; exact original GraphCast loss",
        "production": "cached only; online executions are limited to correctness checks",
    }
    pair.write_once_json(root / "manifest.json", sweep)
    return sweep


def submit(root: Path, *, dry_run: bool = False) -> dict:
    root = Path(root).resolve()
    sweep = json.loads((root / "manifest.json").read_text())
    if sweep["format"] != "v24_cached_width128_sweep_v1":
        raise ValueError("Unsupported width128 sweep")
    jobs = {}
    for entry in sweep["entries"]:
        manifest = pair.load_experiment(Path(entry["experiment_root"]))
        if manifest["source_digest"] != entry["source_digest"] or file_sha256(Path(manifest["configs"]["cached"])) != entry["config_sha256"]:
            raise ValueError("Sweep source or cached configuration changed")
        if not dry_run:
            pair.enable_automation(manifest)
        jobs[entry["name"]] = pair.submit_stage(manifest, "smoke", dry_run=dry_run)
        if not dry_run:
            atomic_json_dump(jobs, root / "jobs.json")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "submit"))
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE)
    parser.add_argument("--cache-root", type=Path, default=CACHE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = (prepare(args.experiment_root, reference_path=args.reference_config, cache_root=args.cache_root)
              if args.stage == "prepare" else submit(args.experiment_root, dry_run=args.dry_run))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
