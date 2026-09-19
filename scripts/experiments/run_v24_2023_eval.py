#!/usr/bin/env python3
"""Freeze and execute the daily 2023 V24 exact-loss evaluation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SHARDS = 23
MODEL_PATHS = (
    ("M1", "Batch-4 di16, faster Mamba", "res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba", "swa_step00500-02000.pkl"),
    ("M2", "Batch-4 di32, joint learning rate", "res1_batch4_temporal_lr_20260908/runs/di32_joint", "swa_step00500-02000.pkl"),
    ("M3", "Optimizer study di16 Mamba1", "res1_optimizer_matrix_20260904/mamba1_di16_bcg1_lr1em4_b1p9_b2p98_cos10k", "swa_step02000-08000.pkl"),
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def protocol_digest(protocol):
    body = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen_file(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def check_file(record):
    if sha256(record["path"]) != record["sha256"]:
        raise ValueError(f"Frozen input changed: {record['path']}")


def make_initializations(year):
    first, stop = dt.date(year, 1, 1), dt.date(year + 1, 1, 1)
    return [(first + dt.timedelta(days=i)).isoformat() + "T00:00:00Z" for i in range((stop - first).days)]


def snapshot_source(workflow):
    destination = workflow / "source"
    if destination.exists():
        raise FileExistsError(destination)
    files = []
    for directory in ("src", "scripts", "third_party/graphcast"):
        files.extend(p for p in (ROOT / directory).rglob("*.py") if "__pycache__" not in p.parts)
    files.extend(ROOT / p for p in ("scripts/experiments/v24_2023_eval_gpu.slurm", "scripts/experiments/v24_2023_eval_cpu.slurm"))
    hashes = {}
    for source in sorted(set(files)):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256(target)
    write_json(workflow / "source_hashes.json", hashes)
    (workflow / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
    (workflow / "working_tree_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    return frozen_file(workflow / "source_hashes.json")


def prepare(workflow, data_path):
    if (workflow / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen; use a new workflow directory for a new experiment")
    workflow.mkdir(parents=True, exist_ok=True)
    data_validation = workflow / "data_validation.json"
    if not data_validation.is_file():
        raise ValueError("Run validate_res1_2023_data.py before freezing this experiment")
    validation = read_json(data_validation)
    if validation.get("status") != "passed":
        raise ValueError("Data validation has not passed")
    if Path(validation["dataset"]).resolve() != data_path.resolve():
        raise ValueError("Data validation belongs to a different dataset")
    if validation["staging_report_sha256"] != sha256(data_path / "stage_report.json"):
        raise ValueError("Staging report changed after data validation")
    if validation["dataset_metadata_sha256"] != sha256(data_path / ".zmetadata"):
        raise ValueError("Dataset metadata changed after data validation")
    initializations = make_initializations(2023)
    init_path = workflow / "initializations_2023.json"
    write_json(init_path, {"schema_version": 1, "initializations": [{"initialization_time": value} for value in initializations]})
    pilot_times = [f"2022-{month:02d}-15T00:00:00Z" for month in (1, 4, 7, 10)]
    pilot_manifest = workflow / "initializations_pilot_2022.json"
    write_json(pilot_manifest, {"schema_version": 1, "initializations": [{"initialization_time": value} for value in pilot_times]})
    models = []
    baselines, stats_dirs = set(), set()
    for model_id, label, run_path, checkpoint_name in MODEL_PATHS:
        run = ROOT / "artifacts/checkpoints/v24_Ilya" / run_path
        config_file = frozen_file(run / "run_config.json")
        config = read_json(config_file["path"])
        checkpoint = frozen_file(run / "swa" / checkpoint_name)
        baselines.add(str((ROOT / config["data"]["baseline_checkpoint"]).resolve()))
        stats_dirs.add(str((ROOT / config["data"]["stats_dir"]).resolve()))
        selection = config["derived"]["prepared_store_selection"]
        if selection["time_end"][:4] > "2022":
            raise ValueError(f"Unexpected training/validation coverage for {model_id}")
        models.append({"model_id": model_id, "label": label, "checkpoint": checkpoint["path"],
                       "checkpoint_sha256": checkpoint["sha256"], "run_config": config_file,
                       "architecture": config["architecture"], "training_data_end": selection["time_end"]})
    if len(baselines) != 1 or len(stats_dirs) != 1:
        raise ValueError("Models do not share the same baseline and normalization statistics")
    stats_dir = Path(next(iter(stats_dirs)))
    baseline = frozen_file(next(iter(baselines)))
    stats = [frozen_file(stats_dir / name) for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")]
    packages = {}
    for name in ("jax", "jaxlib", "numpy", "xarray", "pandas", "dm-haiku", "zarr"):
        packages[name] = importlib.metadata.version(name)
    source = snapshot_source(workflow)
    protocol = {
        "schema_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": "v24_res1_2023_daily_exact_graphcast_loss", "project_root": str(ROOT),
        "scope": "Exact original GraphCast loss; no extreme-event or spatial-preservation claims",
        "initialization_times": initializations, "initialization_manifest": frozen_file(init_path),
        "initialization_manifest_sha256": sha256(init_path), "target_steps": 40,
        "lead_hours": list(range(6, 241, 6)), "models": models,
        "baseline": {"path": baseline["path"], "checkpoint_sha256": baseline["sha256"],
                     "canonical_model_id": "M1", "repeatability_rtol": 1e-3, "repeatability_atol": 1e-6},
        "stats_dir": str(stats_dir), "statistics": stats,
        "data_path": str(data_path.resolve(strict=True)),
        "data_report": frozen_file(data_path / "stage_report.json"), "data_validation": frozen_file(data_validation),
        "data_metadata": frozen_file(data_path / ".zmetadata"),
        "source_metadata": frozen_file(data_path / "source_metadata.json"),
        "training_manifest": frozen_file(ROOT / "data/graphcast/graphcast/dataset/anchor_manifests/v24_Ilya_res1/metadata.json"),
        "source_hashes": source, "source_root": str(workflow / "source"), "packages": packages,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "forecast": {"eval_mode": "cold_full", "residual_state_init": "zero", "residual_alpha": 1.0,
                     "warmup_steps": 0, "input_duration": "12h", "seed": 0},
        "bootstrap": {"replicates": 10000, "seed": 20260916, "block_days": 14,
                      "sensitivity_block_days": [10, 21], "segmentation": "five_seasons"},
        "anchor_shard_count": SHARDS,
        "pilot": {"manifest": frozen_file(pilot_manifest), "initialization_times": pilot_times,
                  "data_path": str((ROOT / "data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr").resolve(strict=True))},
        "numerics": {"xla_flags": "--xla_gpu_enable_triton_gemm=false --xla_gpu_deterministic_ops=true",
                     "physical_fields": "float32", "state_boundaries": "float32"},
        "resources": {"gpu": {"gpus": 1, "cpus": 8, "mem_gib": 16, "walltime_seconds": 3720},
                      "evidence": {"job_ids": ["13617441_3", "13617441_4", "13740086_6"],
                                   "max_rss_gib": 9.80, "steady_sample_seconds": 104,
                                   "first_sample_seconds": 523},
                      "cpu": {"cpus": 2, "mem_gib": 4, "walltime_seconds": 900}},
    }
    protocol["protocol_hash"] = protocol_digest(protocol)
    write_json(workflow / "protocol.json", protocol)
    print(f"Frozen {len(models)} models, {len(initializations)} dates, {len(models)*SHARDS} shards at {workflow}")


def verify_protocol(workflow):
    protocol = read_json(workflow / "protocol.json")
    if protocol_digest(protocol) != protocol["protocol_hash"]:
        raise ValueError("Protocol hash mismatch")
    for record in [protocol["source_hashes"], protocol["initialization_manifest"], protocol["pilot"]["manifest"],
                   protocol["data_report"], protocol["data_validation"], protocol["data_metadata"],
                   protocol["source_metadata"], protocol["training_manifest"], *protocol["statistics"]]:
        check_file(record)
    for path, digest in read_json(protocol["source_hashes"]["path"]).items():
        check_file({"path": str(Path(protocol["source_root"]) / path), "sha256": digest})
    return protocol


def validate_runtime(protocol):
    import inspect
    from src.models.graphcast.training.core import bootstrap  # noqa: F401
    from graphcast import graphcast
    expected = Path(protocol["source_root"]) / "third_party/graphcast/graphcast/graphcast.py"
    if Path(graphcast.__file__).resolve() != expected.resolve():
        raise RuntimeError(f"Wrong GraphCast imported: {graphcast.__file__}")
    if "is_training" not in inspect.signature(graphcast.GraphCast._run_mesh_gnn).parameters:
        raise RuntimeError("Vendored GraphCast lacks required Mamba extensions")
    for name, version in protocol["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Package version changed: {name}")
    if os.environ.get("XLA_FLAGS") != protocol["numerics"]["xla_flags"]:
        raise RuntimeError("XLA numerical settings differ from frozen protocol")


def evaluate(workflow, task, pilot=False):
    protocol = verify_protocol(workflow)
    validate_runtime(protocol)
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    count = protocol["anchor_shard_count"]
    model_index, shard = (task, 0) if pilot else divmod(task, count)
    if not 0 <= model_index < len(protocol["models"]):
        raise ValueError(f"Invalid task {task}")
    model = protocol["models"][model_index]
    check_file({"path": model["checkpoint"], "sha256": model["checkpoint_sha256"]})
    check_file(model["run_config"])
    check_file({"path": protocol["baseline"]["path"], "sha256": protocol["baseline"]["checkpoint_sha256"]})
    if not pilot:
        gate = read_json(workflow / "pilot_validation.json")
        if gate["status"] != "passed" or gate["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("The 2022 pilot has not passed for this protocol")
    directory = workflow / ("pilot" if pilot else "shards")
    output = directory / f"{model['model_id']}_{shard:02d}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation {output}")
    directory.mkdir(parents=True, exist_ok=True)
    unpublished = output.with_name(output.stem + ".unpublished.json")
    manifest = protocol["pilot"]["manifest"] if pilot else protocol["initialization_manifest"]
    config = V24IlyaEvalConfig(
        ckpt=Path(model["checkpoint"]), out_json=unpublished,
        ckpt_in=Path(protocol["baseline"]["path"]), stats_dir=Path(protocol["stats_dir"]),
        data_path=protocol["pilot"]["data_path"] if pilot else protocol["data_path"],
        val_year=2022 if pilot else 2023, train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=4 if pilot else 365, omit_rms_bias=True,
        initialization_manifest=Path(manifest["path"]), model_id=model["model_id"],
        anchor_shard_count=1 if pilot else count, anchor_shard_index=shard,
        **protocol["forecast"], **model["architecture"],
    )
    started = time.monotonic()
    result = evaluate_v24_Ilya(config)
    import jax
    result.update(protocol_hash=protocol["protocol_hash"], checkpoint_sha256=model["checkpoint_sha256"],
                  baseline_checkpoint_sha256=protocol["baseline"]["checkpoint_sha256"],
                  elapsed_seconds=time.monotonic() - started,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  devices=[{"platform": device.platform, "kind": device.device_kind} for device in jax.devices()],
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    write_json(output, result)
    unpublished.unlink()


def validate_pilot(workflow):
    import numpy as np
    protocol = verify_protocol(workflow)
    results = [read_json(workflow / "pilot" / f"{m['model_id']}_00.json") for m in protocol["models"]]
    canonical = None
    details = []
    for model, result in zip(protocol["models"], results, strict=True):
        if result.get("evaluation_status") != "complete" or result["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Incomplete or mismatched pilot")
        identities = {"model_id": model["model_id"], "checkpoint_sha256": model["checkpoint_sha256"],
                      "baseline_checkpoint_sha256": protocol["baseline"]["checkpoint_sha256"],
                      "initialization_manifest_sha256": protocol["pilot"]["manifest"]["sha256"]}
        if any(result.get(key) != value for key, value in identities.items()):
            raise ValueError("Pilot model or data identity mismatch")
        records = result["original_graphcast_loss_per_initialization"]
        if [r["initialization_time"] for r in records] != protocol["pilot"]["initialization_times"]:
            raise ValueError("Pilot timestamps differ from frozen dates")
        baseline = np.asarray([r["baseline_per_step"] for r in records])
        full = np.asarray([r["full_per_step"] for r in records])
        if baseline.shape != (4, 40) or full.shape != (4, 40) or not np.isfinite(full).all():
            raise ValueError("Invalid per-initialization pilot losses")
        if not np.isfinite(baseline).all() or np.any(baseline <= 0) or np.any(full < 0):
            raise ValueError("Nonpositive baseline or invalid model losses in pilot")
        for key, values in (("baseline_per_step", baseline), ("full_per_step", full)):
            np.testing.assert_allclose(values.mean(axis=0), result["original_graphcast_loss"][key], rtol=1e-6, atol=1e-7)
        if canonical is None:
            canonical = baseline
        np.testing.assert_allclose(baseline, canonical, rtol=protocol["baseline"]["repeatability_rtol"],
                                   atol=protocol["baseline"]["repeatability_atol"])
        if result["peak_rss_gib"] > 13.0:
            raise ValueError("Pilot RAM exceeds the production budget's safety margin")
        sample_seconds = np.asarray(result["sample_elapsed_seconds"], dtype=float)
        if sample_seconds.shape != (4,) or not np.isfinite(sample_seconds).all() or np.any(sample_seconds <= 0):
            raise ValueError("Missing or invalid pilot runtime measurements")
        projected = result["elapsed_seconds"] + 12 * float(sample_seconds[1:].max())
        if projected * 1.25 > protocol["resources"]["gpu"]["walltime_seconds"]:
            raise ValueError(f"Projected 16-start elapsed {projected:.0f}s exceeds production safety margin")
        details.append({"model_id": model["model_id"], "elapsed_seconds": result["elapsed_seconds"],
                        "peak_rss_gib": result["peak_rss_gib"],
                        "projected_16_start_seconds": projected,
                        "max_relative_baseline_difference": float(np.max(np.abs(baseline-canonical)/canonical))})
    write_json(workflow / "pilot_validation.json", {"status": "passed", "protocol_hash": protocol["protocol_hash"], "models": details})
    print("2022 pilot passed; full 2023 evaluation may proceed.")


def merge(workflow):
    protocol = verify_protocol(workflow)
    paths = [workflow / "shards" / f"{model['model_id']}_{shard:02d}.json"
             for model in protocol["models"] for shard in range(protocol["anchor_shard_count"])]
    if not all(p.is_file() for p in paths):
        raise ValueError("Missing production shards")
    args = [sys.executable, str(Path(protocol["source_root"]) / "scripts/analyze_models/merge_v24_2023_eval.py"),
            "--protocol", str(workflow / "protocol.json"), "--output-dir", str(workflow / "report"),
            "--output-image-dir", str(workflow / "report")]
    for path in paths:
        args.extend(["--shard-json", str(path)])
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "validate-pilot", "evaluate", "merge", "verify"))
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/graphcast/graphcast/dataset/arco_res1_levels13_2023_eval.zarr")
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    if args.phase == "prepare":
        prepare(workflow, args.data_path)
    elif args.phase in ("pilot", "evaluate"):
        evaluate(workflow, args.task, pilot=args.phase == "pilot")
    elif args.phase == "validate-pilot":
        validate_pilot(workflow)
    elif args.phase == "verify":
        protocol = verify_protocol(workflow)
        print(f"Verified protocol {protocol['protocol_hash']}")
    else:
        merge(workflow)


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
