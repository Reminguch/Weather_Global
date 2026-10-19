#!/usr/bin/env python3
"""Freeze and execute fixed-date 2023 V24 exact-loss evaluation campaigns."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_operations.loaders.zarr_metadata import (  # noqa: E402
    frozen_zarr_metadata, verify_frozen_zarr_metadata,
)
from scripts.experiments.numerical_policy import is_advisory, numerical_comparison

SHARDS = 23
MAX_STARTS_PER_SHARD = 16
DEFAULT_REFERENCE_PROTOCOL = ROOT / "artifacts/evaluations/v24_res1_2023_daily_memory_20260925/protocol.json"
PRIOR_EXPOSURE_TIMES = [f"2023-{month:02d}-15T00:00:00Z" for month in (1, 2, 4, 5, 7, 8, 10, 11)]
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


def make_initializations(year, stride_days=1):
    if type(stride_days) is not int or stride_days <= 0:
        raise ValueError("Initialization stride must be a positive integer number of days")
    first, stop = dt.date(year, 1, 1), dt.date(year + 1, 1, 1)
    return [(first + dt.timedelta(days=i)).isoformat() + "T00:00:00Z"
            for i in range(0, (stop - first).days, stride_days)]


def shard_count(initializations):
    if not initializations:
        raise ValueError("An evaluation requires at least one initialization")
    return math.ceil(len(initializations) / MAX_STARTS_PER_SHARD)


def pilot_repeat_count(protocol):
    return 2 if protocol.get("schema_version", 1) >= 2 else 1


def task_assignment(protocol, task, *, pilot=False):
    count = pilot_repeat_count(protocol) if pilot else protocol["anchor_shard_count"]
    if type(task) is not int or not 0 <= task < len(protocol["models"]) * count:
        raise ValueError(f"Invalid task {task}")
    return divmod(task, count)


def validate_reference_identities(protocol, reference):
    """Retain every prespecified September checkpoint, policy and normalization."""
    if protocol_digest(reference) != reference.get("protocol_hash"):
        raise ValueError("Reference protocol hash mismatch")
    expected_ids = ["M1", "M2", "M3", "M1_reset", "C_carry", "C_reset", "R_reset"]
    if [m["model_id"] for m in protocol["models"]] != expected_ids:
        raise ValueError("The three-day campaign requires all seven frozen policies")
    old = {model["model_id"]: model for model in reference["models"]}
    for model in protocol["models"]:
        previous = old.get(model["model_id"], {})
        for key in ("checkpoint_sha256", "architecture", "evaluator_family", "reset_state_every_step", "run_config"):
            if model.get(key) != previous.get(key):
                raise ValueError(f"Reference model identity changed: {model['model_id']} {key}")
    if protocol["baseline"]["checkpoint_sha256"] != reference["baseline"]["checkpoint_sha256"]:
        raise ValueError("Reference baseline checkpoint changed")
    statistics = lambda value: {Path(record["path"]).name: record["sha256"] for record in value["statistics"]}
    if statistics(protocol) != statistics(reference):
        raise ValueError("Reference normalization statistics changed")


def require_numerical_qualification(workflow, protocol):
    if is_advisory(protocol) or protocol.get("schema_version", 1) < 2:
        return None
    from scripts.experiments.v24_2023_qualification import validate_numerical_qualification
    validate_numerical_qualification(workflow, protocol)
    return frozen_file(protocol["numerical_qualification"]["path"])


def qualification_runtime(record):
    runtime = read_json(record["path"]).get("runtime", {})
    if any(not isinstance(runtime.get(key), str) or not runtime[key]
           for key in ("gpu_kind", "driver_version")):
        raise ValueError("Numerical qualification lacks the qualified GPU kind and driver")
    return {key: runtime[key] for key in ("gpu_kind", "driver_version")}


def capture_runtime_hardware(protocol):
    from scripts.analyze_models.diagnose_v24_2023_numerics import gpu_identity
    return gpu_identity(protocol)


def require_production_gate(workflow, protocol):
    gate = read_json(workflow / "pilot_validation.json")
    accepted = ("passed", "complete_with_warnings") if is_advisory(protocol) else ("passed",)
    if gate["status"] not in accepted or gate["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("The 2022 pilot has not passed for this protocol")
    if is_advisory(protocol) and gate.get("integrity_passed") is not True:
        raise ValueError("Pilot integrity and resource checks have not passed")
    qualification = require_numerical_qualification(workflow, protocol)
    if qualification is not None:
        if gate.get("numerical_qualification") != qualification:
            raise ValueError("Numerical qualification changed after the 2022 pilot gate")
        if any(gate.get(key) != value for key, value in qualification_runtime(qualification).items()):
            raise ValueError("Pilot gate GPU kind or driver differs from numerical qualification")
    if protocol.get("schema_version", 1) >= 2:
        for model in gate["models"]:
            check_file(model["pilot_file"])
    return gate


def generated_input_variables(protocol, *, pilot=True):
    """Only absent source variables generated by GraphCast may be advisory."""
    path = protocol["pilot"]["data_path"] if pilot else protocol["data_path"]
    metadata = read_json(Path(path) / ".zmetadata")["metadata"]
    candidates = ("day_progress_cos", "day_progress_sin", "year_progress_cos",
                  "year_progress_sin", "toa_incident_solar_radiation")
    return sorted(name for name in candidates if f"{name}/.zarray" not in metadata)


def pilot_observation_identity(result, times, *, advisory_derived_variables=()):
    observation = result.get("numerical_observation")
    if not isinstance(observation, dict) or not isinstance(observation.get("setup"), dict):
        raise ValueError("Pilot lacks effective numerical fingerprints")
    setup = observation["setup"]
    shared_keys = ("baseline_params", "baseline_state", "statistics", "baseline_model_config", "task_config")
    if any(key not in setup for key in (*shared_keys, "residual_params", "residual_state")):
        raise ValueError("Pilot numerical setup fingerprints are incomplete")
    initializations = observation.get("initializations", [])
    if [row.get("initialization_time") for row in initializations] != times:
        raise ValueError("Pilot numerical input timestamps differ from frozen dates")
    if any(not row.get("inputs") or not row.get("rng") for row in initializations):
        raise ValueError("Pilot numerical input or RNG fingerprints are incomplete")
    shared = {key: setup[key] for key in shared_keys}
    if advisory_derived_variables:
        from scripts.experiments.v24_2023_qualification import _require_tree_fingerprint
        shared["initializations"] = []
        expected_variables = sorted(advisory_derived_variables)
        for row in initializations:
            if row.get("derived_input_variables") != expected_variables or row.get("derived_inputs_finite") is not True:
                raise ValueError("Missing or mismatched generated-input evidence")
            for field in ("source_inputs", "derived_inputs"):
                _require_tree_fingerprint(row.get(field), field)
            # Preserve exact stored weather/coordinate and RNG identity. Only
            # recomputed features get the same advisory treatment as forecasts.
            shared["initializations"].append({
                "initialization_time": row["initialization_time"], "inputs": row["source_inputs"],
                "rng": row["rng"], "derived_schema": row["derived_inputs"]["leaves"],
            })
    else:
        shared["initializations"] = initializations
    predictions = observation.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Pilot numerical truth coverage is missing")
    expected = [(stamp, lead) for stamp in times for lead in range(40)]
    if [(row.get("initialization_time"), row.get("step_index")) for row in predictions] != expected:
        raise ValueError("Pilot numerical truth coverage is incomplete")
    if any(not row.get(field) for row in predictions for field in ("truth", "baseline", "full")):
        raise ValueError("Pilot numerical truth or prediction fingerprints are incomplete")
    shared["truth"] = [row["truth"] for row in predictions]
    return shared, {key: setup[key] for key in ("residual_params", "residual_state")}


def snapshot_source(workflow):
    destination = workflow / "source"
    if destination.exists():
        raise FileExistsError(destination)
    files = []
    for directory in ("src", "scripts", "third_party/graphcast"):
        files.extend(p for p in (ROOT / directory).rglob("*.py") if "__pycache__" not in p.parts)
    files.extend(ROOT / p for p in ("scripts/experiments/v24_2023_eval_gpu.slurm", "scripts/experiments/v24_2023_eval_cpu.slurm",
                                    "scripts/experiments/v24_report_recovery.slurm"))
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


def prepare(workflow, data_path, *, include_memory_controls=False, stride_days=1,
            reference_protocol=DEFAULT_REFERENCE_PROTOCOL, xla_flags="--xla_gpu_enable_triton_gemm=false",
            numerical_policy="strict"):
    if (workflow / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen; use a new workflow directory for a new experiment")
    if stride_days not in (1, 3):
        raise ValueError("Supported campaign strides are one or three days")
    schema_version = 2 if stride_days == 3 else 1
    include_memory_controls = include_memory_controls or schema_version >= 2
    data_metadata = frozen_zarr_metadata(data_path)
    # validate_res1_2023_data reads v2 chunks directly. Supporting v3 metadata
    # here must not imply that its chunk/coverage audit also supports v3.
    if Path(data_metadata["path"]).name != ".zmetadata":
        raise ValueError("Daily 2023 evaluation requires Zarr v2 chunk validation via validate_res1_2023_data.py")
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
    if validation["dataset_metadata_sha256"] != data_metadata["sha256"]:
        raise ValueError("Dataset metadata changed after data validation")
    pilot_data = (ROOT / "data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr").resolve(strict=True)
    pilot_data_metadata = frozen_zarr_metadata(pilot_data)
    initializations = make_initializations(2023, stride_days)
    shards = shard_count(initializations)
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
    if include_memory_controls:
        models.append(dict(models[0], model_id="M1_reset", label="V24 selected checkpoint: carry train / reset eval",
                           reset_state_every_step=True))
        legacy_runs = (
            ("C_carry", "Memory control: carry train / carry eval", "res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k", False),
            ("C_reset", "Memory control: carry train / reset eval", "res1_dm_7yr_k20_di_bcg_20k_20260813/di16_bcg1_closed_sg_stateful_20k", True),
            ("R_reset", "Memory control: reset train / reset eval", "res1_dm_7yr_k20_di_bcg_reset_every_anchor_20k_20260815/di16_bcg1_closed_sg_reset_every_anchor_20k", True),
        )
        for model_id, label, relative, reset in legacy_runs:
            run = ROOT / "artifacts/checkpoints/v22_final" / relative
            config_file = frozen_file(run / "run_config.json")
            config = read_json(config_file["path"])
            checkpoint = frozen_file(run / "swa/swa_step02000-08000.pkl")
            baselines.add(str((ROOT / config["data"]["baseline_checkpoint"]).resolve()))
            stats_dirs.add(str((ROOT / config["data"]["stats_dir"]).resolve()))
            models.append({"model_id": model_id, "label": label, "checkpoint": checkpoint["path"],
                           "checkpoint_sha256": checkpoint["sha256"], "run_config": config_file,
                           "architecture": config["architecture"], "evaluator_family": "v22_final",
                           "training_state_policy": config["sequence"]["temporal_state_policy"],
                           "reset_state_every_step": reset})
    for model in models:
        model.setdefault("reset_state_every_step", False)
        model.setdefault("evaluator_family", "v24_Ilya")
    if len(baselines) != 1 or len(stats_dirs) != 1:
        raise ValueError("Models do not share the same baseline and normalization statistics")
    stats_dir = Path(next(iter(stats_dirs)))
    baseline = frozen_file(next(iter(baselines)))
    stats = [frozen_file(stats_dir / name) for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")]
    packages = {}
    for name in ("jax", "jaxlib", "numpy", "xarray", "pandas", "dm-haiku", "zarr"):
        packages[name] = importlib.metadata.version(name)
    check_file(data_metadata)
    check_file(pilot_data_metadata)
    protocol = {
        "numerical_policy": numerical_policy,
        "schema_version": schema_version, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": f"v24_res1_2023_{'daily' if stride_days == 1 else 'stride3'}_exact_graphcast_loss", "project_root": str(ROOT),
        "scope": "Exact original GraphCast loss; no extreme-event or spatial-preservation claims",
        "initialization_times": initializations, "initialization_manifest": frozen_file(init_path),
        "initialization_manifest_sha256": sha256(init_path), "target_steps": 40,
        "lead_hours": list(range(6, 241, 6)), "models": models,
        "baseline": {"path": baseline["path"], "checkpoint_sha256": baseline["sha256"],
                     "canonical_model_id": "M1", "repeatability_rtol": 1e-3, "repeatability_atol": 1e-6},
        "stats_dir": str(stats_dir), "statistics": stats,
        "data_path": str(data_path.resolve(strict=True)),
        "data_report": frozen_file(data_path / "stage_report.json"), "data_validation": frozen_file(data_validation),
        "data_metadata": data_metadata,
        "source_metadata": frozen_file(data_path / "source_metadata.json"),
        "training_manifest": frozen_file(ROOT / "data/graphcast/graphcast/dataset/anchor_manifests/v24_Ilya_res1/metadata.json"),
        "source_root": str(workflow / "source"), "packages": packages,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "forecast": {"eval_mode": "cold_full", "residual_state_init": "zero", "residual_alpha": 1.0,
                     "warmup_steps": 0, "input_duration": "12h", "seed": 0},
        "bootstrap": {"replicates": 10000, "seed": 20260916, "block_days": 14,
                      "sensitivity_block_days": [10, 21], "segmentation": "five_seasons"},
        "anchor_shard_count": shards,
        "pilot": {"manifest": frozen_file(pilot_manifest), "initialization_times": pilot_times,
                  "data_path": str(pilot_data), "data_metadata": pilot_data_metadata},
        "numerics": {"xla_flags": xla_flags, "internal_predictors": "bfloat16",
                     "physical_fields": "float32", "state_boundaries": "float32"},
        "resources": {"gpu": {"gpus": 1, "gres": "gpu:a100:1", "constraint": "gpu80",
                              "cpus": 8, "mem_gib": 16, "walltime_seconds": 3720},
                      "evidence": {"job_ids": ["14080456_0", "14080456_1", "14080456_2", "12530971_3"],
                                   "max_rss_gib": 12.1, "steady_sample_seconds": 104,
                                   "first_sample_seconds": 523,
                                   "runtime_correction": "Remove forced deterministic GPU ops, matching successful spectral pilot; gate actual 16-start projection"},
                      "cpu": {"cpus": 2, "mem_gib": 4, "walltime_seconds": 900}},
    }
    if schema_version >= 2:
        reference_record = frozen_file(reference_protocol)
        validate_reference_identities(protocol, read_json(reference_record["path"]))
        protocol.update(
            reference_protocol=reference_record, initialization_stride_days=stride_days,
            prior_exposure={"initialization_times": PRIOR_EXPOSURE_TIMES,
                            "sampled_initialization_times": [t for t in initializations if t in PRIOR_EXPOSURE_TIMES],
                            "selection_influence": "none_reported"},
            analysis={"primary_endpoint": "M1_vs_GC_rollout", "secondary_endpoints": [
                "M1_vs_GC_day10", "M1_vs_M1_reset_rollout", "C_carry_vs_C_reset_rollout", "C_carry_vs_R_reset_rollout"]},
            bootstrap={"replicates": 10000, "seed": 20260916, "block_starts": 5,
                       "sensitivity_block_starts": [3, 7], "segmentation": "annual", "quarter_sensitivity": True},
            numerical_qualification={"path": str(workflow / "numerical_qualification.json"),
                                     "required": numerical_policy == "strict", "schema_version": 1},
        )
        protocol["pilot"]["repeats"] = 2
        protocol["resources"]["gpu"]["ram_fraction_limit"] = 0.8
        protocol["resources"]["evidence"].pop("runtime_correction", None)
        protocol["resources"]["evidence"]["interpretation"] = (
            "Historical resource measurements only; production requires fresh 2022 numerical and resource qualification"
        )
        for model in protocol["models"]:
            if "training_data_end" in model:
                model["prepared_store_end"] = model.pop("training_data_end")
        from scripts.experiments.v24_2023_qualification import audit_training_provenance
        protocol["training_provenance"] = audit_training_provenance(protocol)
    if is_advisory(protocol):
        protocol["validation_policy"] = {
            "integrity": "blocking", "resources": "blocking", "numerical_repeatability": "advisory",
            "diagnostic_dependency": "none", "runtime": "ordinary GPU execution; numerical flags frozen above",
        }
        protocol["resources"]["evidence"]["interpretation"] = (
            "Historical resource measurements; fresh 2022 integrity/resource pilots release production. "
            "Repeated-computation differences are recorded and do not block release.")
    protocol["source_hashes"] = snapshot_source(workflow)
    protocol["protocol_hash"] = protocol_digest(protocol)
    write_json(workflow / "protocol.json", protocol)
    print(f"Frozen {len(models)} models, {len(initializations)} dates, {len(models)*shards} shards at {workflow}")


def verify_protocol(workflow):
    protocol = read_json(workflow / "protocol.json")
    is_advisory(protocol)
    if protocol_digest(protocol) != protocol["protocol_hash"]:
        raise ValueError("Protocol hash mismatch")
    if "recovery_of" in protocol:
        check_file(protocol["recovery_of"])
    if protocol.get("schema_version", 1) >= 2:
        if protocol.get("initialization_stride_days") != 3 or protocol["pilot"].get("repeats") != 2:
            raise ValueError("Schema 2 requires three-day initializations and two pilot repeats")
        if (protocol["baseline"]["repeatability_rtol"], protocol["baseline"]["repeatability_atol"]) != (1e-3, 1e-6):
            raise ValueError("Schema 2 numerical repeatability tolerances must not change")
        expected = make_initializations(2023, protocol["initialization_stride_days"])
        if protocol["initialization_times"] != expected or protocol["anchor_shard_count"] != shard_count(expected):
            raise ValueError("Initialization calendar or shard count differs from frozen stride")
        check_file(protocol["reference_protocol"])
        validate_reference_identities(protocol, read_json(protocol["reference_protocol"]["path"]))
        from scripts.experiments.v24_2023_qualification import verify_records
        verify_records(protocol["training_provenance"])
    verify_frozen_zarr_metadata(protocol["data_path"], protocol["data_metadata"])
    for record in [protocol["source_hashes"], protocol["initialization_manifest"], protocol["pilot"]["manifest"],
                   protocol["data_report"], protocol["data_validation"], protocol["data_metadata"],
                   protocol["source_metadata"], protocol["training_manifest"], *protocol["statistics"]]:
        check_file(record)
    # Older frozen protocols predate the explicit pilot-data fingerprint.
    if "data_metadata" in protocol["pilot"]:
        verify_frozen_zarr_metadata(protocol["pilot"]["data_path"], protocol["pilot"]["data_metadata"])
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
    for key, expected_value in protocol["numerics"].get("environment", {}).items():
        if os.environ.get(key) != expected_value:
            raise RuntimeError(f"Numerical runtime environment differs: {key}")


def evaluate(workflow, task, pilot=False):
    protocol = verify_protocol(workflow)
    validate_runtime(protocol)
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    count = protocol["anchor_shard_count"]
    model_index, shard = task_assignment(protocol, task, pilot=pilot)
    model = protocol["models"][model_index]
    check_file({"path": model["checkpoint"], "sha256": model["checkpoint_sha256"]})
    check_file(model["run_config"])
    check_file({"path": protocol["baseline"]["path"], "sha256": protocol["baseline"]["checkpoint_sha256"]})
    hardware = None
    if not pilot:
        gate = require_production_gate(workflow, protocol)
    if protocol.get("schema_version", 1) >= 2:
        hardware = capture_runtime_hardware(protocol)
        if not pilot and any(hardware[key] != gate[key] for key in ("gpu_kind", "driver_version")):
            raise ValueError("Production GPU kind or driver differs from the qualified pilots")
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
        target_steps=protocol["target_steps"],
        n_samples=len(protocol["pilot"]["initialization_times"] if pilot else protocol["initialization_times"]),
        omit_rms_bias=True,
        initialization_manifest=Path(manifest["path"]), model_id=model["model_id"],
        anchor_shard_count=1 if pilot else count, anchor_shard_index=0 if pilot else shard,
        reset_state_every_step=model.get("reset_state_every_step", False),
        **protocol["forecast"], **model["architecture"],
    )
    numerical_observer = None
    if pilot and protocol.get("schema_version", 1) >= 2:
        from scripts.experiments.v24_2023_qualification import NumericalObserver
        numerical_observer = NumericalObserver(
            derived_input_variables=generated_input_variables(protocol)
            if protocol.get("generated_input_identity") == "separate_v1" else ())
    observer_kwargs = {"numerical_observer": numerical_observer} if numerical_observer is not None else {}
    started = time.monotonic()
    if model.get("evaluator_family", "v24_Ilya") == "v22_final":
        from scripts.analyze_models.eval_v22_2023_manifest import evaluate_legacy_manifest
        result = evaluate_legacy_manifest(config, model["architecture"], **observer_kwargs)
    else:
        result = evaluate_v24_Ilya(config, **observer_kwargs)
    import jax
    result.update(protocol_hash=protocol["protocol_hash"], checkpoint_sha256=model["checkpoint_sha256"],
                  baseline_checkpoint_sha256=protocol["baseline"]["checkpoint_sha256"],
                  elapsed_seconds=time.monotonic() - started,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  devices=[{"platform": device.platform, "kind": device.device_kind} for device in jax.devices()],
                  evaluator_family=model.get("evaluator_family", "v24_Ilya"),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    if hardware is not None:
        result.update(hardware)
    if pilot and protocol.get("schema_version", 1) >= 2:
        result["pilot_repeat_index"] = shard
        result["numerical_observation"] = numerical_observer.data
    write_json(output, result)
    unpublished.unlink()


def validate_pilot(workflow):
    import numpy as np
    protocol = verify_protocol(workflow)
    advisory = is_advisory(protocol)
    output = workflow / "pilot_validation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed pilot validation {output}")
    qualification = require_numerical_qualification(workflow, protocol)
    qualified_runtime = qualification_runtime(qualification) if qualification is not None else None
    repeats = pilot_repeat_count(protocol)
    sample_count = len(protocol["pilot"]["initialization_times"])
    if any(not stamp.startswith("2022-") for stamp in protocol["pilot"]["initialization_times"]):
        raise ValueError("Numerical pilots must use 2022 initializations only")
    max_shard_starts = (math.ceil(len(protocol["initialization_times"]) / protocol["anchor_shard_count"])
                        if "initialization_times" in protocol else MAX_STARTS_PER_SHARD)
    comparisons = {"rtol": protocol["baseline"]["repeatability_rtol"],
                   "atol": protocol["baseline"]["repeatability_atol"]}
    canonical = None
    canonical_observation = None
    first_repeat = {}
    first_residual_identity = {}
    gpu_kind = None
    driver_version = None
    numerical_diagnostics = []
    derived_input_diagnostics = []
    canonical_derived = None
    derived_variables = (generated_input_variables(protocol)
                         if advisory and protocol.get("generated_input_identity") == "separate_v1" else ())
    details = []
    for model, repeat in ((m, r) for m in protocol["models"] for r in range(repeats)):
        path = workflow / "pilot" / f"{model['model_id']}_{repeat:02d}.json"
        result = read_json(path)
        if result.get("evaluation_status") != "complete" or result["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Incomplete or mismatched pilot")
        identities = {"model_id": model["model_id"], "checkpoint_sha256": model["checkpoint_sha256"],
                      "baseline_checkpoint_sha256": protocol["baseline"]["checkpoint_sha256"],
                      "initialization_manifest_sha256": protocol["pilot"]["manifest"]["sha256"]}
        if any(result.get(key) != value for key, value in identities.items()):
            raise ValueError("Pilot model or data identity mismatch")
        if repeats > 1:
            if result.get("pilot_repeat_index") != repeat:
                raise ValueError("Pilot repeat identity mismatch")
            if result.get("metrics_complete") is not True or result.get("evaluated_samples") != sample_count:
                raise ValueError("Pilot metrics or sample coverage is incomplete")
            devices = result.get("devices", [])
            if len(devices) != 1 or devices[0].get("platform") != "gpu" or not devices[0].get("kind"):
                raise ValueError("Each pilot must record exactly one GPU")
            if not result.get("gpu_uuid") or result.get("gpu_kind") != devices[0]["kind"]:
                raise ValueError("Pilot GPU identity is incomplete or inconsistent")
            if qualified_runtime is not None and any(result.get(key) != value for key, value in qualified_runtime.items()):
                raise ValueError("Pilot GPU kind or driver differs from numerical qualification")
            if gpu_kind is None:
                gpu_kind = devices[0]["kind"]
                driver_version = result.get("driver_version")
            if devices[0]["kind"] != gpu_kind:
                raise ValueError("Pilot GPU types differ")
            if not driver_version or result.get("driver_version") != driver_version:
                raise ValueError("Pilot GPU drivers differ or are missing")
            shared, residual_identity = pilot_observation_identity(
                result, protocol["pilot"]["initialization_times"],
                advisory_derived_variables=derived_variables)
            if derived_variables:
                derived = [row["derived_inputs"]["sha256"]
                           for row in result["numerical_observation"]["initializations"]]
                if canonical_derived is None:
                    canonical_derived = derived
                derived_input_diagnostics.append({
                    "model_id": model["model_id"], "repeat": repeat,
                    "variables": derived_variables, "sha256_per_start": derived,
                    "passed": derived == canonical_derived,
                    "interpretation": "Finite regenerated-feature differences are advisory; stored inputs and RNG match exactly",
                })
            if canonical_observation is None:
                canonical_observation = shared
            if shared != canonical_observation:
                raise ValueError("Effective pilot baseline parameters, state, statistics, configuration or inputs differ")
            if repeat == 0:
                first_residual_identity[model["model_id"]] = residual_identity
            elif residual_identity != first_residual_identity[model["model_id"]]:
                raise ValueError("Effective pilot residual parameters or initial state differ across repeats")
        if "reset_state_every_step" in model and result.get("rs_reset_every_step") != model["reset_state_every_step"]:
            raise ValueError("Pilot memory reset policy mismatch")
        records = result["original_graphcast_loss_per_initialization"]
        if [r["initialization_time"] for r in records] != protocol["pilot"]["initialization_times"]:
            raise ValueError("Pilot timestamps differ from frozen dates")
        baseline = np.asarray([r["baseline_per_step"] for r in records])
        full = np.asarray([r["full_per_step"] for r in records])
        if baseline.shape != (sample_count, 40) or full.shape != (sample_count, 40) or not np.isfinite(full).all():
            raise ValueError("Invalid per-initialization pilot losses")
        if not np.isfinite(baseline).all() or np.any(baseline <= 0) or np.any(full < 0):
            raise ValueError("Nonpositive baseline or invalid model losses in pilot")
        for key, values in (("baseline_per_step", baseline), ("full_per_step", full)):
            np.testing.assert_allclose(values.mean(axis=0), result["original_graphcast_loss"][key], rtol=1e-6, atol=1e-7)
        if canonical is None:
            canonical = baseline
        numerical_diagnostics.append(numerical_comparison(
            baseline, canonical, **comparisons, label=f"{model['model_id']}/{repeat}: baseline vs M1",
            advisory=advisory))
        if repeat == 0:
            first_repeat[model["model_id"]] = (baseline, full)
        else:
            first_baseline, first_full = first_repeat[model["model_id"]]
            for label, candidate, reference in (("baseline", baseline, first_baseline), ("full", full, first_full)):
                numerical_diagnostics.append(numerical_comparison(
                    candidate, reference, **comparisons,
                    label=f"{model['model_id']}/{repeat}: {label} repeat", advisory=advisory))
        memory_limit = (0.8 if repeats > 1 else 0.8125) * protocol["resources"]["gpu"].get("mem_gib", 16)
        if not np.isfinite(result["peak_rss_gib"]) or result["peak_rss_gib"] <= 0 or result["peak_rss_gib"] > memory_limit:
            raise ValueError("Pilot RAM exceeds the production budget's safety margin")
        sample_seconds = np.asarray(result["sample_elapsed_seconds"], dtype=float)
        if sample_seconds.shape != (sample_count,) or not np.isfinite(sample_seconds).all() or np.any(sample_seconds <= 0):
            raise ValueError("Missing or invalid pilot runtime measurements")
        if not np.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"] <= 0:
            raise ValueError("Missing or invalid pilot elapsed time")
        projected = result["elapsed_seconds"] + max(0, max_shard_starts - sample_count) * float(sample_seconds[1:].max())
        if projected * 1.25 > protocol["resources"]["gpu"]["walltime_seconds"]:
            raise ValueError(f"Projected {max_shard_starts}-start elapsed {projected:.0f}s exceeds production safety margin")
        details.append({"model_id": model["model_id"], "pilot_repeat_index": repeat,
                        "pilot_file": frozen_file(path), "elapsed_seconds": result["elapsed_seconds"],
                        "peak_rss_gib": result["peak_rss_gib"],
                        "projected_shard_starts": max_shard_starts, "projected_shard_seconds": projected,
                        "max_relative_baseline_difference": float(np.max(np.abs(baseline-canonical)/canonical))})
    warnings = any(not row["passed"] for row in numerical_diagnostics + derived_input_diagnostics)
    gate = {"status": "complete_with_warnings" if warnings else "passed",
            "integrity_passed": True, "resources_passed": True,
            "numerical_status": "warning" if warnings else "passed",
            "numerical_diagnostics": numerical_diagnostics,
            "derived_input_diagnostics": derived_input_diagnostics,
            "protocol_hash": protocol["protocol_hash"], "models": details}
    if repeats > 1:
        gate.update(gpu_kind=gpu_kind, driver_version=driver_version)
    if qualification is not None:
        gate.update(numerical_qualification=qualification, **qualified_runtime)
    write_json(output, gate)
    print(f"2022 integrity/resource pilot passed ({gate['numerical_status']} numerical diagnostics); full 2023 evaluation may proceed.")


def merge(workflow):
    protocol = verify_protocol(workflow)
    if protocol.get("schema_version", 1) >= 2:
        require_production_gate(workflow, protocol)
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
    parser.add_argument("--include-memory-controls", action="store_true")
    parser.add_argument("--initialization-stride-days", "--stride-days", dest="stride_days", type=int, choices=(1, 3), default=1)
    parser.add_argument("--reference-protocol", type=Path, default=DEFAULT_REFERENCE_PROTOCOL)
    parser.add_argument("--xla-flags", default="--xla_gpu_enable_triton_gemm=false")
    parser.add_argument("--numerical-policy", choices=("strict", "advisory"), default="strict")
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    if args.phase == "prepare":
        prepare(workflow, args.data_path, include_memory_controls=args.include_memory_controls,
                stride_days=args.stride_days, reference_protocol=args.reference_protocol,
                xla_flags=args.xla_flags, numerical_policy=args.numerical_policy)
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
