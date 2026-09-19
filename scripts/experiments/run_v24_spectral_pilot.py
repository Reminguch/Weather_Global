#!/usr/bin/env python3
"""Freeze, run, and report an eight-start V24 spatial-spectrum pilot."""
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


def frozen_file(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def check_file(record):
    if sha256(record["path"]) != record["sha256"]:
        raise ValueError(f"Frozen input changed: {record['path']}")


def protocol_digest(protocol):
    body = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pilot_initializations(year=2023):
    if year not in (2022, 2023):
        raise ValueError("Supported pilot years are 2022 and 2023")
    return [f"{year}-{month:02d}-15T00:00:00Z" for month in (1, 2, 4, 5, 7, 8, 10, 11)]


def default_data_path(year):
    pilot_initializations(year)  # Reject unsupported years before selecting a store.
    name = "arco_res1_levels13_2023_eval.zarr" if year == 2023 else "wb2_res1_levels13_train.zarr"
    return ROOT / "data/graphcast/graphcast/dataset" / name


def snapshot_source(workflow):
    destination = workflow / "source"
    if destination.exists():
        raise FileExistsError(destination)
    files = []
    for directory in ("src", "scripts", "third_party/graphcast"):
        files.extend(path for path in (ROOT / directory).rglob("*.py") if "__pycache__" not in path.parts)
    files.extend(ROOT / path for path in (
        "scripts/graphcast_env.sh", "scripts/experiments/v24_spectral_pilot_gpu.slurm",
        "scripts/experiments/v24_spectral_pilot_cpu.slurm",
    ))
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


def validate_data(data_path, initializations):
    """Check coordinates and coverage without loading the meteorological fields."""
    import numpy as np
    import xarray as xr

    required = {
        "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
        "mean_sea_level_pressure", "total_precipitation_6hr", "temperature",
        "specific_humidity", "u_component_of_wind", "v_component_of_wind",
    }
    with xr.open_zarr(data_path, consolidated=True) as dataset:
        absent = required - set(dataset.data_vars)
        if absent:
            raise ValueError(f"Pilot data is missing variables: {sorted(absent)}")
        coordinates = {name: np.asarray(dataset[name].values) for name in ("time", "lat", "lon", "level")}
        times = coordinates["time"].astype("datetime64[ns]")
        if times.ndim != 1 or not np.all(np.diff(times) > np.timedelta64(0, "ns")):
            raise ValueError("Dataset times must be unique and increasing")
        for stamp in initializations:
            origin = np.datetime64(stamp.removesuffix("Z"), "ns")
            required_times = origin + np.arange(-1, 41) * np.timedelta64(6, "h")
            if not np.isin(required_times, times).all():
                raise ValueError(f"Pilot initialization lacks inputs or targets: {stamp}")
        latitudes, longitudes = coordinates["lat"], coordinates["lon"]
        if latitudes.shape != (181,) or longitudes.shape != (360,):
            raise ValueError("Pilot requires the native 181 by 360 one-degree grid")
        if not np.allclose(np.sort(latitudes), np.arange(-90, 91)):
            raise ValueError("Unexpected latitude coordinates")
        if not np.allclose(np.diff(longitudes), 1.0):
            raise ValueError("Longitude must be a complete, increasing, uniform periodic grid")
        if not {700, 850}.issubset(set(coordinates["level"].tolist())):
            raise ValueError("Pilot requires pressure levels 700 and 850 hPa")
        return {
            "status": "passed", "dataset": str(data_path), "scope": "Coordinate/coverage validation; field chunks are not hashed",
            "sizes": dict(dataset.sizes), "time_start": str(times[0]), "time_end": str(times[-1]),
            "coordinates_sha256": {name: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                                   for name, values in coordinates.items()},
            "initialization_times": initializations,
        }


def prepare(workflow, data_path=None, *, year=2023):
    if (workflow / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen; use a new workflow directory for a new experiment")
    from src.models.mamba.v24_Ilya.spectral_observer import SPECTRAL_VARIABLES
    from src.models.mamba.v24_Ilya.spectral_diagnostics import zonal_spectral_metadata

    workflow.mkdir(parents=True, exist_ok=True)
    data_path = (data_path if data_path is not None else default_data_path(year)).resolve(strict=True)
    initializations = pilot_initializations(year)
    write_json(workflow / "data_validation.json", validate_data(data_path, initializations))
    manifest = workflow / f"initializations_pilot_{year}.json"
    write_json(manifest, {"schema_version": 1, "initializations": [{"initialization_time": value} for value in initializations]})
    models, baselines, statistics_dirs = [], set(), set()
    for model_id, label, relative, checkpoint_name in MODEL_PATHS:
        run = ROOT / "artifacts/checkpoints/v24_Ilya" / relative
        configuration = frozen_file(run / "run_config.json")
        config = read_json(configuration["path"])
        checkpoint = frozen_file(run / "swa" / checkpoint_name)
        baselines.add(str((ROOT / config["data"]["baseline_checkpoint"]).resolve()))
        statistics_dirs.add(str((ROOT / config["data"]["stats_dir"]).resolve()))
        models.append({"model_id": model_id, "label": label, "checkpoint": checkpoint["path"],
                       "checkpoint_sha256": checkpoint["sha256"], "run_config": configuration,
                       "architecture": config["architecture"]})
    if len(baselines) != 1 or len(statistics_dirs) != 1:
        raise ValueError("Pilot models must share the baseline and normalization statistics")
    baseline = frozen_file(next(iter(baselines)))
    stats_dir = Path(next(iter(statistics_dirs)))
    statistics = [frozen_file(stats_dir / name) for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")]
    packages = {name: importlib.metadata.version(name) for name in
                ("jax", "jaxlib", "numpy", "scipy", "xarray", "pandas", "dm-haiku", "zarr", "numcodecs", "matplotlib")}
    source = snapshot_source(workflow)
    protocol = {
        "schema_version": 1, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": f"v24_res1_spatial_spectra_{year}_pilot", "project_root": str(ROOT),
        "evaluation_year": year, "evaluation_period": f"{year} pilot",
        "scope": f"Exploratory {year} spatial Fourier pilot on eight fixed starts; no full-year or preservation claim",
        "initialization_times": initializations, "initialization_manifest": frozen_file(manifest),
        "initialization_manifest_sha256": sha256(manifest), "target_steps": 40,
        "lead_hours": list(range(6, 241, 6)), "models": models,
        "spectral_variables": list(SPECTRAL_VARIABLES), "spectral_metadata": zonal_spectral_metadata(),
        "baseline": {"path": baseline["path"], "checkpoint_sha256": baseline["sha256"],
                     "canonical_model_id": "M1", "repeatability_rtol": 1e-3, "repeatability_atol": 1e-6,
                     "spectral_repeatability_rtol": 1e-3, "spectral_repeatability_atol": 1e-10},
        "stats_dir": str(stats_dir), "statistics": statistics,
        "data_path": str(data_path), "data_metadata": frozen_file(data_path / ".zmetadata"),
        "data_provenance": [frozen_file(data_path / name) for name in ("stage_report.json", "source_metadata.json")
                            if (data_path / name).is_file()],
        "data_validation": frozen_file(workflow / "data_validation.json"),
        "source_root": str(workflow / "source"), "source_hashes": source, "packages": packages,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "forecast": {"eval_mode": "cold_full", "residual_state_init": "zero", "residual_alpha": 1.0,
                     "warmup_steps": 0, "input_duration": "12h", "seed": 0},
        "numerics": {"xla_flags": "--xla_gpu_enable_triton_gemm=false",
                     "determinism": "Historical evaluation settings; paired baseline repeatability is checked before reporting",
                     "physical_fields": "float32", "state_boundaries": "float32", "spectral_accumulation": "float64"},
        "resources": {"gpu": {"gpus": 1, "cpus": 8, "mem_gib": 16, "walltime_seconds": 3720},
                      "cpu": {"cpus": 2, "mem_gib": 4, "walltime_seconds": 900},
                      "evidence": {"job_ids": ["13740086_6", "13617441_3"], "max_rss_gib": 9.80,
                                   "first_sample_seconds": 523, "steady_sample_seconds": 104}},
        "output_data_dir": str(workflow / "report"),
        "output_image_dir": str(ROOT / "plots/analyze_models/images/resolution_eval" / workflow.name),
    }
    protocol["protocol_hash"] = protocol_digest(protocol)
    write_json(workflow / "protocol.json", protocol)
    print(f"Frozen spectral pilot: {len(models)} models, {len(initializations)} starts, 40 leads at {workflow}")


def verify_protocol(workflow):
    protocol = read_json(workflow / "protocol.json")
    if protocol_digest(protocol) != protocol["protocol_hash"]:
        raise ValueError("Protocol hash mismatch")
    for record in (protocol["source_hashes"], protocol["initialization_manifest"], protocol["data_metadata"],
                   protocol["data_validation"], *protocol["statistics"], *protocol.get("data_provenance", [])):
        check_file(record)
    for path, digest in read_json(protocol["source_hashes"]["path"]).items():
        check_file({"path": str(Path(protocol["source_root"]) / path), "sha256": digest})
    return protocol


def validate_runtime(protocol, *, gpu=False):
    expected = Path(protocol["source_root"]).resolve()
    if ROOT.resolve() != expected:
        raise RuntimeError("Evaluate/report must run the frozen workflow/source script")
    for name, version in protocol["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Package version changed: {name}")
    if gpu:
        import inspect
        from src.models.mamba.v24_Ilya import evaluation
        from graphcast import graphcast
        if Path(evaluation.__file__).resolve() != expected / "src/models/mamba/v24_Ilya/evaluation.py":
            raise RuntimeError("Evaluator imported from outside the frozen source")
        if Path(graphcast.__file__).resolve() != expected / "third_party/graphcast/graphcast/graphcast.py":
            raise RuntimeError(f"Wrong GraphCast imported: {graphcast.__file__}")
        if "is_training" not in inspect.signature(graphcast.GraphCast._run_mesh_gnn).parameters:
            raise RuntimeError("Vendored GraphCast lacks required Mamba extensions")
        if os.environ.get("XLA_FLAGS") != protocol["numerics"]["xla_flags"]:
            raise RuntimeError("XLA numerical settings differ from frozen protocol")


def evaluate(workflow, task):
    protocol = verify_protocol(workflow)
    if not 0 <= task < len(protocol["models"]):
        raise ValueError(f"Invalid model task {task}")
    validate_runtime(protocol, gpu=True)
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from src.models.mamba.v24_Ilya.spectral_observer import SpectralObserver
    import jax

    devices = [{"platform": device.platform, "kind": device.device_kind} for device in jax.devices()]
    if not any(device["platform"] == "gpu" for device in devices):
        raise RuntimeError("The spectral inference pilot requires a GPU allocation")
    model = protocol["models"][task]
    check_file({"path": model["checkpoint"], "sha256": model["checkpoint_sha256"]})
    check_file(model["run_config"])
    check_file({"path": protocol["baseline"]["path"], "sha256": protocol["baseline"]["checkpoint_sha256"]})
    output_dir = workflow / "models" / model["model_id"]
    output = output_dir / "evaluation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {output}")
    output_dir.mkdir(parents=True, exist_ok=True)
    unpublished = output_dir / "evaluation.unpublished.json"
    identity = {"protocol_hash": protocol["protocol_hash"], "model_id": model["model_id"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "evaluation_year": protocol.get("evaluation_year", 2022),
                "baseline_checkpoint_sha256": protocol["baseline"]["checkpoint_sha256"]}
    observer = SpectralObserver(output_dir=output_dir, metadata={**identity, "spectral_metadata": protocol["spectral_metadata"]},
                                target_steps=40, export_fields=True)
    config = V24IlyaEvalConfig(
        ckpt=Path(model["checkpoint"]), out_json=unpublished,
        ckpt_in=Path(protocol["baseline"]["path"]), stats_dir=Path(protocol["stats_dir"]),
        data_path=protocol["data_path"], val_year=protocol.get("evaluation_year", 2022), train_start_year=2015, train_end_year=2021,
        target_steps=40, n_samples=len(protocol["initialization_times"]), omit_rms_bias=True,
        initialization_manifest=Path(protocol["initialization_manifest"]["path"]),
        model_id=model["model_id"], anchor_shard_count=1, anchor_shard_index=0,
        **protocol["forecast"], **model["architecture"],
    )
    started, status = time.monotonic(), "failed"
    try:
        result = evaluate_v24_Ilya(config, prediction_observer=observer)
        resources = {"elapsed_seconds": time.monotonic() - started,
                     "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                     "devices": devices, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                     "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
        result.update(identity, **resources)
        write_json(output, result)
        unpublished.unlink(missing_ok=True)
        status = "complete"
    finally:
        write_json(output_dir / "resources.json", {
            **identity, "status": status, "elapsed_seconds": time.monotonic() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "devices": devices, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        })


def validate_exact_losses(workflow, protocol):
    """Require paired complete results and use only M1's baseline for comparisons."""
    import numpy as np

    canonical = None
    summaries, sources = [], []
    ordered = sorted(protocol["models"], key=lambda model: model["model_id"] != protocol["baseline"]["canonical_model_id"])
    for model in ordered:
        path = workflow / "models" / model["model_id"] / "evaluation.json"
        result = read_json(path)
        identities = {"protocol_hash": protocol["protocol_hash"], "model_id": model["model_id"],
                      "checkpoint_sha256": model["checkpoint_sha256"],
                      "baseline_checkpoint_sha256": protocol["baseline"]["checkpoint_sha256"],
                      "initialization_manifest_sha256": protocol["initialization_manifest_sha256"]}
        if result.get("evaluation_status") != "complete" or any(result.get(key) != value for key, value in identities.items()):
            raise ValueError(f"Incomplete or mismatched evaluation for {model['model_id']}")
        records = result["original_graphcast_loss_per_initialization"]
        if [record["initialization_time"] for record in records] != protocol["initialization_times"]:
            raise ValueError("Evaluation dates differ from frozen pilot dates")
        baseline = np.asarray([record["baseline_per_step"] for record in records], dtype=np.float64)
        full = np.asarray([record["full_per_step"] for record in records], dtype=np.float64)
        shape = (len(protocol["initialization_times"]), protocol["target_steps"])
        if baseline.shape != shape or full.shape != shape:
            raise ValueError("Incomplete lead coverage in exact losses")
        if not np.isfinite(baseline).all() or not np.isfinite(full).all() or np.any(baseline <= 0) or np.any(full < 0):
            raise ValueError("Invalid exact losses")
        for branch, values in (("baseline", baseline), ("full", full)):
            np.testing.assert_allclose(values.mean(axis=0), result["original_graphcast_loss"][f"{branch}_per_step"], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(values.mean(), result["original_graphcast_loss"][f"{branch}_rollout"], rtol=1e-6, atol=1e-7)
        if canonical is None:
            if model["model_id"] != protocol["baseline"]["canonical_model_id"]:
                raise ValueError("Missing canonical baseline model")
            canonical = baseline
        np.testing.assert_allclose(baseline, canonical, rtol=protocol["baseline"]["repeatability_rtol"],
                                   atol=protocol["baseline"]["repeatability_atol"])
        summaries.append({"model_id": model["model_id"], "baseline_rollout": float(canonical.mean()),
                          "full_rollout": float(full.mean()),
                          "improvement_pct_rollout": float(100 * (1 - full.mean() / canonical.mean())),
                          "improvement_pct_per_step": (100 * (1 - full.mean(axis=0) / canonical.mean(axis=0))).tolist(),
                          "max_relative_baseline_difference": float(np.max(np.abs(baseline - canonical) / canonical)),
                          "elapsed_seconds": result["elapsed_seconds"], "peak_rss_gib": result["peak_rss_gib"]})
        sources.append(frozen_file(path))
    validation = {"status": "passed", "protocol_hash": protocol["protocol_hash"], "models": summaries,
                  "evaluation_sources": sources, "canonical_baseline_model_id": protocol["baseline"]["canonical_model_id"]}
    write_json(workflow / "exact_loss_validation.json", validation)
    return validation


def report(workflow):
    protocol = verify_protocol(workflow)
    validate_runtime(protocol)
    validate_exact_losses(workflow, protocol)
    from scripts.analyze_models.report_v24_spectral_pilot import report as make_report
    make_report(workflow)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate", "report", "verify"))
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--year", type=int, choices=(2022, 2023), default=2023,
                        help="Preparation year; evaluation always uses the frozen protocol (default: 2023)")
    parser.add_argument("--data-path", type=Path, help="Preparation data store; defaults to the local store for --year")
    parser.add_argument("--task", type=int, default=0, help="Model index: 0=M1, 1=M2, 2=M3")
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    if args.phase == "prepare":
        prepare(workflow, args.data_path, year=args.year)
    elif args.phase == "evaluate":
        evaluate(workflow, args.task)
    elif args.phase == "report":
        report(workflow)
    else:
        print(f"Verified protocol {verify_protocol(workflow)['protocol_hash']}")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
