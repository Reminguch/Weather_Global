#!/usr/bin/env python3
"""Frozen arbitrary-checkpoint evaluation and joint RMSE/spectrum selection.

This module does not submit jobs. Inference requires a GPU allocation; report
and preparation read saved data on CPU. Each inference runs in a new process.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.run_v24_spectral_pilot import (  # noqa: E402
    check_file, frozen_file, protocol_digest, read_json, write_json,
)
from scripts.analyze_models.report_v24_spectral_pilot import (  # noqa: E402
    SPECTRAL_KEYS, _load_record, _time,
)
from scripts.analyze_models.amse_feature_report_data import (  # noqa: E402
    FEATURE_SPECS, spectral_summary,
)
from src.data_operations.loaders.zarr_metadata import verify_frozen_zarr_metadata  # noqa: E402

REFERENCE = ROOT/"artifacts/checkpoints/v24_Ilya/res1_amse_pair_20260919"
MSE_EXPERIMENTS = (REFERENCE, ROOT/"artifacts/checkpoints/v24_Ilya/res1_amse_blend_lambda0p1_20260924")
TAGS = ("step2000", "swa_step00500-02000")
DAYS = (1, 3, 7, 10)
FEATURES = tuple(row[0] for row in FEATURE_SPECS if row[2] is not None)
CHANNELS = {code: f"{variable}_level{level}" if level else variable
            for code, _, variable, level in FEATURE_SPECS if variable is not None}
SAFETY_FEATURES = FEATURES
POLICY = {"qualification": "All three aggregate conditions use the same spectral initializations; exploratory selection.",
          "selected_days": list(DAYS), "features": list(FEATURES),
          "safety_features": list(SAFETY_FEATURES), "maximum_safety_rmse_harm_pct": 2.,
          "coherence_drop_tolerance": .001, "max_finalists": 2,
          "bootstrap_replicates": 2000, "bootstrap_seed": 20261004,
          "bootstrap_calendar_block_days": 14}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def subset_equal(actual, expected, label="configuration"):
    for key, value in expected.items():
        require(key in actual, f"Missing {label}.{key}")
        if isinstance(value, dict):
            subset_equal(actual[key], value, f"{label}.{key}")
        else:
            require(actual[key] == value, f"Changed {label}.{key}")


def validate_planned_config(actual, path):
    """Compare serialized semantics, including intentionally omitted defaults."""
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    expected = load_training_config(Path(path)).to_dict()
    for key in ("architecture", "data", "objective", "optimizer", "sequence", "memory",
                "distributed", "validation", "output"):
        subset_equal(actual[key], expected[key], key)


def select_initialization(experiments=MSE_EXPERIMENTS, out=None):
    """Choose only from complete 2022 exact32 scores; never read 2023 scores."""
    candidates = []
    reference_origins, reference_spec = None, None
    for experiment in map(Path, experiments):
        for tag in TAGS:
            directory = experiment/"eval/exact/mse_fp32"/tag
            spec = read_json(directory/"specification.json")
            result = read_json(directory/"evaluation.json")
            require(spec["evaluation_year"] == 2022 and spec["n_samples"] == 32, "Initialization selection needs 2022 exact32")
            verify_specification(spec)
            config = read_json(Path(spec["run_config"]["path"]))
            require(config["objective"]["loss_kind"] == "mse_fp32", "Initialization candidate is not FP32 MSE")
            require(config["architecture"] == spec["architecture"], "Initialization architecture/config differ")
            if reference_spec is None:
                reference_spec = spec
            for field in ("architecture", "baseline", "statistics", "data_path", "data_metadata", "forecast", "lead_hours"):
                require(spec[field] == reference_spec[field], f"Initialization candidate {field} differs")
            validate_result(result, spec)
            if reference_origins is None:
                reference_origins = result["chosen_idx"]
            require(result["chosen_idx"] == reference_origins, "MSE selection origins differ")
            check_file({"path": spec["checkpoint"], "sha256": spec["checkpoint_sha256"]})
            candidates.append({"experiment": str(experiment.resolve()), "tag": tag,
                               "checkpoint": spec["checkpoint"], "checkpoint_sha256": spec["checkpoint_sha256"],
                               "run_config": spec["run_config"], "source_run_dir": str(Path(spec["run_config"]["path"]).parent),
                               "exact32_rollout_loss": result["original_graphcast_loss"]["full_rollout"],
                               "evaluation": frozen_file(directory/"evaluation.json"),
                               "specification": frozen_file(directory/"specification.json")})
    selected = min(candidates, key=lambda row: (row["exact32_rollout_loss"], row["checkpoint"]))
    output = {"schema_version": 1, "selection_data": "32 matched 2022 development starts only",
              "selection_metric": "minimum exact original GraphCast rollout loss",
              "selection_caveat": "Small differences between repeated MSE runs are not statistical evidence of superiority.",
              "candidate_set": "Both existing FP32-MSE campaigns, native step2000 and prespecified SWA only",
              "candidates": candidates, "selected": selected,
              "warm_start": "Load residual parameters from selected SWA/native checkpoint; reset optimizer, RNG, cursor and recurrent state."}
    if out is not None:
        write_json(Path(out), output)
    return output


def bootstrap_identity(stamp):
    date = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    season = ("DJF", "MAM", "JJA", "SON")[(date.month % 12)//3]
    return f"{date.year}-{season}-{(date.timetuple().tm_yday-1)//14:02d}", f"{date.year}-{season}"


def prepare_evaluation(tasks, out_dir, mse_control_id, reference_experiment=REFERENCE,
                       spectral_initializations=None, observer_factory=None, policy=None,
                       role="development", workspace=None):
    """Freeze protocols and task paths, allowing checkpoints that do not yet exist."""
    out_dir = Path(out_dir).resolve()
    workspace = Path(workspace or os.environ.get("WEATHER_WORKSPACE", ROOT)).resolve()
    path = out_dir/"manifest.json"
    require(not path.exists(), f"Refusing to replace evaluation manifest: {path}")
    task_specs = copy.deepcopy(tasks["task_specs"] if isinstance(tasks, dict) else tasks)
    ids = [task["id"] for task in task_specs]
    require(ids and len(set(ids)) == len(ids) and mse_control_id in ids, "Unique task IDs and canonical MSE task required")
    for task in task_specs:
        require(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", task["id"]), "Unsafe task ID")
        require(task.get("role", "candidate") in ("candidate", "mse_control", "reference"), "Unknown task role")
        for key in ("checkpoint", "source_run_dir", "run_config", "planned_config"):
            if key in task:
                task[key] = str((workspace/Path(task[key])).resolve())
        require("checkpoint" in task and "source_run_dir" in task, "Task needs checkpoint and source_run_dir")
        task.setdefault("run_config", str(Path(task["source_run_dir"])/"run_config.json"))
        task.setdefault("label", task["id"])
        task.setdefault("role", "candidate")
        if task.get("planned_config"):
            task["planned_config_record"] = frozen_file(Path(task["planned_config"]))
        if Path(task["checkpoint"]).exists():
            task["checkpoint_record"] = frozen_file(Path(task["checkpoint"]))
        if Path(task["run_config"]).exists():
            task["run_config_record"] = frozen_file(Path(task["run_config"]))
        task["reuse"] = {stage: str((workspace/Path(directory)).resolve())
                         for stage, directory in task.get("reuse", {}).items()}
    protocols, inputs = {}, []
    for stage in ("exact", "spectral"):
        prototype_path = Path(reference_experiment)/"eval"/stage/"mse_fp32"/TAGS[1]/"specification.json"
        prototype = read_json(prototype_path)
        inputs.append(frozen_file(prototype_path))
        excluded = {"protocol_hash", "model_id", "checkpoint", "checkpoint_sha256", "run_config",
                    "experiment_manifest", "arm", "tag"}
        protocols[stage] = {key: value for key, value in prototype.items() if key not in excluded}
        verify_frozen_zarr_metadata(prototype["data_path"], prototype["data_metadata"])
    if spectral_initializations is not None:
        path_dates = Path(spectral_initializations).resolve()
        dates = read_json(path_dates)
        entries = dates.get("initializations", dates.get("initialization_times")) if isinstance(dates, dict) else dates
        require(isinstance(entries, list) and entries, "Custom initializations must be a nonempty list")
        stamps = [entry["initialization_time"] if isinstance(entry, dict) else entry for entry in entries]
        parsed = [datetime.fromisoformat(stamp.replace("Z", "+00:00")) for stamp in stamps]
        require(all(value.tzinfo is not None and value.utcoffset().total_seconds() == 0 for value in parsed),
                "Custom dates must use explicit UTC")
        require(all(value.hour % 6 == 0 and value.minute == value.second == value.microsecond == 0 for value in parsed),
                "Custom dates must align with six-hour predictions")
        stamps = [_time(stamp) for stamp in stamps]
        require(len(set(stamps)) == len(stamps), "Duplicate custom dates")
        require(parsed == sorted(parsed), "Custom dates must be chronological")
        require(all(value.year == 2023 for value in parsed),
                "Current confirmation data protocol supports 2023 dates")
        # Preserve optional grouping metadata while normalizing the evaluator input.
        normalized = [{**(entry if isinstance(entry, dict) else {}), "initialization_time": stamp}
                      for entry, stamp in zip(entries, stamps, strict=True)]
        write_json(out_dir/"initializations.json", {"schema_version": 1, "initializations": normalized})
        record = frozen_file(out_dir/"initializations.json")
        protocols["spectral"].update(n_samples=len(stamps), initialization_times=stamps,
                                     initialization_manifest=record, initialization_manifest_sha256=record["sha256"])
        inputs.append(frozen_file(path_dates))
    else:
        normalized = [{"initialization_time": stamp} for stamp in protocols["spectral"]["initialization_times"]]
    grouping = []
    for entry in normalized:
        stamp = entry["initialization_time"]
        group, stratum = bootstrap_identity(stamp)
        grouping.append({"initialization_time": stamp, "group": str(entry.get("bootstrap_group", group)),
                         "stratum": str(entry.get("bootstrap_stratum", stratum))})
    merged_policy = {**POLICY, **(policy or {})}
    require(role in ("development", "confirmation"), "Unknown evaluation role")
    if role == "confirmation":
        old = set(read_json(Path(reference_experiment)/"eval/spectral/mse_fp32"/TAGS[1]/"specification.json")["initialization_times"])
        require(not old.intersection(protocols["spectral"]["initialization_times"]),
                "Confirmation dates overlap the explored eight-start pilot")
    manifest = {"schema_version": 1, "format": "v24_amse_regime_evaluation_v1", "workspace": str(workspace),
                "output_dir": str(out_dir), "task_specs": task_specs, "canonical_mse_id": mse_control_id,
                "protocols": protocols, "preparation_sources": inputs, "selection_policy": merged_policy,
                "evaluation_role": role, "bootstrap_grouping": grouping,
                "observer_factory": observer_factory,
                "scope": "Same spectral starts for exact GC loss, global primitive-field RMSE, zonal power and coherence. Exact32 remains a separate 2022 diagnostic.",
                "limitations": "Development selection is exploratory. Confirmation is evaluated separately; repeated baseline drift is retained and not silently merged."}
    manifest["manifest_hash"] = protocol_digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    write_json(path, manifest)
    return manifest


def load_manifest(path):
    manifest = read_json(Path(path))
    require(manifest.get("format") == "v24_amse_regime_evaluation_v1", "Unknown evaluation manifest")
    require(manifest["manifest_hash"] == protocol_digest({k: v for k, v in manifest.items() if k != "manifest_hash"}),
            "Evaluation manifest changed")
    for record in manifest["preparation_sources"]:
        check_file(record)
    return manifest


def task_architecture(task, reference, actual):
    """Honor a manifest-pinned temporal width while retaining the shared model."""
    expected = task.get("architecture", reference)
    require(isinstance(expected, dict), "Task architecture must be a mapping")
    if "architecture" in task:
        require(set(expected) == set(reference), "Task architecture fields differ from reference")
        changed = {key for key in reference if expected[key] != reference[key]}
        require(changed <= {"temporal_d_inner"},
                "Task architecture may override only temporal_d_inner")
        width = expected.get("temporal_d_inner")
        require(type(width) is int and width > 0, "Invalid task temporal_d_inner")
    require(actual == expected, "Run architecture differs from pinned evaluation task")
    return copy.deepcopy(expected)


def make_specification(manifest_path, task_index, stage):
    manifest = load_manifest(manifest_path)
    require(stage in ("exact", "spectral") and 0 <= task_index < len(manifest["task_specs"]), "Invalid task/stage")
    task = manifest["task_specs"][task_index]
    for key in ("checkpoint_record", "run_config_record", "planned_config_record"):
        if key in task:
            check_file(task[key])
    config_record = frozen_file(Path(task["run_config"]))
    config = read_json(Path(task["run_config"]))
    if "seed" in task:
        require(config["optimizer"]["seed"] == task["seed"], "Evaluation task/training seed differ")
    if task.get("planned_config"):
        validate_planned_config(config, task["planned_config"])
    spec = copy.deepcopy(manifest["protocols"][stage])
    spec["architecture"] = task_architecture(task, spec["architecture"], config["architecture"])
    baseline = Path(config["data"]["baseline_checkpoint"])
    if not baseline.is_absolute():
        baseline = Path(manifest["workspace"])/baseline
    baseline_record = frozen_file(baseline)
    require(baseline_record["sha256"] == spec["baseline"]["checkpoint_sha256"], "Run uses another original GC")
    stats_dir = Path(config["data"]["stats_dir"])
    if not stats_dir.is_absolute():
        stats_dir = Path(manifest["workspace"])/stats_dir
    for record in spec["statistics"]:
        actual = frozen_file(stats_dir/Path(record["path"]).name)
        require(actual["sha256"] == record["sha256"], "Run normalization statistics differ")
    checkpoint = frozen_file(Path(task["checkpoint"]))
    spec.update(model_id=task["id"], arm=task.get("family", task["id"]), tag=task.get("tag", str(task.get("step", "checkpoint"))),
                checkpoint=checkpoint["path"], checkpoint_sha256=checkpoint["sha256"], run_config=config_record,
                experiment_manifest=frozen_file(Path(manifest_path)), evaluation_role=manifest["evaluation_role"])
    spec["protocol_hash"] = protocol_digest(spec)
    verify_specification(spec)
    return spec


def verify_specification(spec):
    require(protocol_digest(spec) == spec["protocol_hash"], "Specification digest changed")
    for record in (spec["run_config"], spec["experiment_manifest"], spec["data_metadata"], *spec["statistics"],
                   {"path": spec["checkpoint"], "sha256": spec["checkpoint_sha256"]},
                   {"path": spec["baseline"]["path"], "sha256": spec["baseline"]["checkpoint_sha256"]}):
        check_file(record)
    for key in ("evaluation_reference", "initialization_manifest", "source_spectral_protocol"):
        if key in spec:
            check_file(spec[key])
    verify_frozen_zarr_metadata(spec["data_path"], spec["data_metadata"])


def validate_result(result, spec):
    require(result.get("evaluation_status") == "complete" and result.get("metrics_complete", True) and
            result.get("remaining_samples", 0) == 0 and result["evaluated_samples"] == spec["n_samples"] and
            result["target_steps"] == spec["target_steps"], "Incomplete evaluation")
    for key in ("model_id", "protocol_hash", "checkpoint_sha256"):
        require(result[key] == spec[key], f"Result {key} mismatch")
    require(result["baseline_checkpoint_sha256"] == spec["baseline"]["checkpoint_sha256"], "Baseline identity mismatch")
    arrays = {}
    loss = result["original_graphcast_loss"]
    for branch in ("baseline", "full"):
        values = np.asarray(loss[f"{branch}_per_step"], dtype=float)
        require(values.shape == (spec["target_steps"],) and np.isfinite(values).all() and np.all(values > 0),
                f"Invalid {branch} losses")
        np.testing.assert_allclose(values.mean(), loss[f"{branch}_rollout"], rtol=1e-7, atol=1e-8)
        arrays[branch] = values
    if spec["stage"] == "exact":
        require(result["chosen_idx"] == spec["chosen_idx"], "Exact32 origins differ")
    else:
        records = result["original_graphcast_loss_per_initialization"]
        require([_time(r["initialization_time"]) for r in records] == [_time(v) for v in spec["initialization_times"]],
                "Spectral initialization dates differ")
        require(result["initialization_manifest_sha256"] == spec["initialization_manifest_sha256"], "Initialization manifest differs")
        for branch in ("baseline", "full"):
            np.testing.assert_allclose(np.mean([r[f"{branch}_per_step"] for r in records], axis=0), arrays[branch], rtol=1e-7)
        for channel in CHANNELS.values():
            values = result["per_channel_per_step"][channel]
            for branch in ("baseline", "full"):
                vector = np.asarray(values[f"rmse_{branch}"])
                require(vector.shape == (spec["target_steps"],) and np.isfinite(vector).all() and np.all(vector > 0),
                        f"Missing/nonfinite channel RMSE: {channel}")
    return arrays


def compatible_reuse(old_spec, requested):
    fields = ("stage", "checkpoint_sha256", "architecture", "baseline", "statistics", "forecast",
              "data_path", "data_metadata", "target_steps", "lead_hours", "n_samples", "evaluation_year", "warmup_steps")
    for field in fields:
        require(old_spec[field] == requested[field], f"Reuse protocol differs: {field}")
    require(old_spec["run_config"]["sha256"] == requested["run_config"]["sha256"], "Reuse run configuration differs")
    fields = ("chosen_idx",) if requested["stage"] == "exact" else (
        "initialization_times", "initialization_manifest_sha256", "spectral_variables", "spectral_metadata")
    for field in fields:
        require(old_spec[field] == requested[field], f"Reuse protocol differs: {field}")


def evaluation_directory(manifest, task, stage):
    return Path(manifest["output_dir"])/"results"/task["id"]/stage


def resolve_result(manifest_path, index, stage):
    manifest = load_manifest(manifest_path)
    task = manifest["task_specs"][index]
    requested = make_specification(manifest_path, index, stage)
    destination = evaluation_directory(manifest, task, stage)
    reused = False
    if not (destination/"evaluation.json").exists() and task.get("reuse", {}).get(stage):
        destination = Path(task["reuse"][stage])
        reused = True
    spec = read_json(destination/"specification.json")
    verify_specification(spec)
    if reused:
        compatible_reuse(spec, requested)
    else:
        require(spec == requested, "Existing result uses another requested specification")
    result = read_json(destination/"evaluation.json")
    validate_result(result, spec)
    return destination, spec, result, reused


class GlobalFieldObserver:
    """Small per-case physical MSE files enable paired confirmation intervals."""
    def __init__(self, destination, identity, target_steps=40):
        self.destination = Path(destination)/"global_metrics"
        self.identity, self.target_steps = identity, target_steps
        self.sample = None

    def begin_sample(self, metadata):
        require(self.sample is None, "Unfinished global sample")
        self.sample = metadata
        self.values = {branch: [] for branch in ("baseline", "full")}

    def update_step(self, step_index, truth, baseline, full):
        from src.models.mamba.v24_Ilya.spectral_observer import extract_fields
        require(step_index == len(self.values["full"]), "Global metric leads out of order")
        target = extract_fields(truth).astype(np.float64)
        latitude = np.asarray(truth.lat, dtype=np.float64)
        weights = np.cos(np.deg2rad(latitude))
        weights /= weights.sum()
        for name, fields in (("baseline", baseline), ("full", full)):
            error = extract_fields(fields).astype(np.float64)-target
            self.values[name].append(np.einsum("vjl,j->v", error**2, weights)/error.shape[-1])

    def finish_sample(self):
        require(self.sample is not None and len(self.values["full"]) == self.target_steps, "Incomplete global sample")
        path = self.destination/f"{self.sample['initialization_id']}.json"
        require(not path.exists(), f"Refusing to overwrite {path}")
        payload = {**self.sample, **self.identity, "status": "complete", "features": list(FEATURES),
                   "scope": "Global cosine-latitude-weighted physical MSE; includes zonal means.",
                   **{f"{branch}_mse": np.stack(values).tolist() for branch, values in self.values.items()}}
        write_json(path, payload)
        self.sample = None


class CompositeObserver:
    def __init__(self, observers):
        self.observers = observers

    def begin_sample(self, metadata):
        for observer in self.observers:
            observer.begin_sample(metadata)

    def update_step(self, step_index, truth, baseline, full):
        for observer in self.observers:
            observer.update_step(step_index, truth, baseline, full)

    def finish_sample(self):
        for observer in self.observers:
            observer.finish_sample()


def evaluate_worker(manifest_path, index, stage):
    import jax
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from src.models.mamba.v24_Ilya.spectral_observer import SpectralObserver

    require(any(device.platform == "gpu" for device in jax.devices()), "Evaluation requires an allocated GPU")
    manifest = load_manifest(manifest_path)
    task = manifest["task_specs"][index]
    spec = make_specification(manifest_path, index, stage)
    destination = evaluation_directory(manifest, task, stage)
    require(not (destination/"evaluation.json").exists(), "Refusing to overwrite completed evaluation")
    if (destination/"specification.json").exists():
        require(read_json(destination/"specification.json") == spec, "Existing specification differs")
    else:
        write_json(destination/"specification.json", spec)
    identity = {key: spec[key] for key in ("protocol_hash", "model_id", "checkpoint_sha256")}
    identity["baseline_checkpoint_sha256"] = spec["baseline"]["checkpoint_sha256"]
    observer = None
    if stage == "spectral":
        observers = [SpectralObserver(destination, identity, target_steps=spec["target_steps"], export_fields=False),
                     GlobalFieldObserver(destination, identity, target_steps=spec["target_steps"])]
        if manifest.get("observer_factory"):
            module, function = manifest["observer_factory"].split(":", 1)
            observers.append(getattr(importlib.import_module(module), function)(spec, destination, identity))
        observer = CompositeObserver(observers)
    cfg = V24IlyaEvalConfig(ckpt=Path(spec["checkpoint"]), out_json=destination/"evaluation.unpublished.json",
                           ckpt_in=Path(spec["baseline"]["path"]), stats_dir=Path(spec["stats_dir"]),
                           data_path=spec["data_path"], val_year=spec["evaluation_year"],
                           train_start_year=2015, train_end_year=2021, target_steps=spec["target_steps"],
                           n_samples=spec["n_samples"], warmup_steps=spec["warmup_steps"], omit_rms_bias=True,
                           initialization_manifest=(Path(spec["initialization_manifest"]["path"]) if stage == "spectral" else None),
                           model_id=spec["model_id"], **spec["forecast"], **spec["architecture"])
    started, status = time.monotonic(), "failed"
    try:
        result = evaluate_v24_Ilya(cfg, prediction_observer=observer)
        result.update(identity)
        validate_result(result, spec)
        write_json(destination/"evaluation.json", result)
        (destination/"evaluation.unpublished.json").unlink(missing_ok=True)
        status = "complete"
    finally:
        write_json(destination/"resources.json", {**identity, "status": status,
                   "elapsed_seconds": time.monotonic()-started,
                   "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID")})


def evaluate(manifest_path, index, stage="both"):
    for selected in (("exact", "spectral") if stage == "both" else (stage,)):
        try:
            destination, _, _, reused = resolve_result(manifest_path, index, selected)
        except FileNotFoundError:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--manifest", str(manifest_path),
                            "--task", str(index), "--stage", selected], check=True)
        else:
            print(f"{'Reusing' if reused else 'Verified'} complete evaluation: {destination}", flush=True)


def load_spectral_result(directory, spec, result):
    stamps = [_time(v) for v in spec["initialization_times"]]
    records, sources, k = {}, [], None
    for path in sorted(Path(directory).glob("*.npz")):
        stamp, k, arrays, source = _load_record(path, spec, spec, stamps, k)
        require(stamp not in records, "Duplicate spectral start")
        records[stamp] = arrays
        sources.append(source)
    require(set(records) == set(stamps), "Incomplete spectral records")
    stacked = {key: np.stack([records[stamp][key] for stamp in stamps]) for key in SPECTRAL_KEYS}
    pooled = {key: value.mean(axis=0) for key, value in stacked.items()}
    global_records = {}
    for path in sorted((Path(directory)/"global_metrics").glob("*.json")):
        record = read_json(path)
        require(record["status"] == "complete" and record["features"] == list(FEATURES), "Global metric feature/status mismatch")
        for key in ("protocol_hash", "model_id", "checkpoint_sha256"):
            require(record[key] == spec[key], f"Global metric {key} differs")
        require(record["baseline_checkpoint_sha256"] == spec["baseline"]["checkpoint_sha256"],
                "Global metric original GC identity differs")
        stamp = _time(record["initialization_time"])
        require(stamp in stamps and stamp not in global_records, "Unexpected/duplicate global metric date")
        global_records[stamp] = record
        sources.append(frozen_file(path))
    global_mse = None
    if global_records:
        require(set(global_records) == set(stamps), "Incomplete per-initialization global metrics")
        global_mse = {branch: np.asarray([global_records[stamp][f"{branch}_mse"] for stamp in stamps])
                      for branch in ("baseline", "full")}
        for branch, values in global_mse.items():
            require(values.shape == (len(stamps), spec["target_steps"], len(FEATURES)) and
                    np.isfinite(values).all() and np.all(values >= 0), "Invalid per-start physical MSE")
            for vi, feature in enumerate(FEATURES):
                np.testing.assert_allclose(np.sqrt(values[:, :, vi].mean(axis=0)),
                                           result["per_channel_per_step"][CHANNELS[feature]][f"rmse_{branch}"],
                                           rtol=2e-6, atol=1e-10)
    return {"pooled": pooled, "stacked": stacked, "global_mse": global_mse, "sources": sources,
            "times": stamps, "variables": spec["spectral_variables"], "wavenumber": k, "result": result}


def feature_rows(record, branch="full"):
    rows = []
    for feature in FEATURES:
        vi = record["variables"].index(feature)
        for day in DAYS:
            li = day*4-1
            pooled = record["pooled"]
            rows.append({"feature": feature, "day": day,
                         "rmse": record["result"]["per_channel_per_step"][CHANNELS[feature]][f"rmse_{branch}"][li],
                         **spectral_summary(pooled[f"{branch}_power"][li, vi], pooled["truth_power"][li, vi],
                                            pooled[f"{branch}_cross_power"][li, vi], pooled[f"{branch}_error_power"][li, vi])})
    return rows


def score_candidate(candidate_rows, gc_rows, mse_rows, loss_ratio, policy=POLICY):
    comparisons = []
    for candidate, gc, mse in zip(candidate_rows, gc_rows, mse_rows, strict=True):
        require((candidate["feature"], candidate["day"]) == (gc["feature"], gc["day"]) ==
                (mse["feature"], mse["day"]), "Feature/lead ordering differs")
        row = dict(candidate)
        for label, reference in (("gc", gc), ("mse", mse)):
            row[f"rmse_ratio_vs_{label}"] = candidate["rmse"]/reference["rmse"]
            row[f"power_error_ratio_vs_{label}"] = candidate["mean_relative_power_error_pct"]/reference["mean_relative_power_error_pct"]
            row[f"coherence_change_vs_{label}"] = candidate["mean_coherence"]-reference["mean_coherence"]
            row[f"joint_win_vs_{label}"] = bool(row[f"rmse_ratio_vs_{label}"] < 1 and row[f"power_error_ratio_vs_{label}"] < 1)
        comparisons.append(row)
    mean_rmse = float(np.mean([r["rmse_ratio_vs_gc"] for r in comparisons]))
    mean_power = float(np.mean([r["mean_relative_power_error_pct"] for r in comparisons]))
    gc_power = float(np.mean([r["mean_relative_power_error_pct"] for r in gc_rows]))
    harms = [r for r in comparisons if r["feature"] in policy["safety_features"] and
             100*(r["rmse_ratio_vs_gc"]-1) > policy["maximum_safety_rmse_harm_pct"]]
    qualified = bool(loss_ratio < 1 and mean_rmse < 1 and mean_power < gc_power)
    return {"exact_gc_loss_ratio_vs_gc": float(loss_ratio), "mean_rmse_ratio_vs_gc": mean_rmse,
            "mean_relative_power_error_pct": mean_power, "mean_power_error_ratio_vs_gc": mean_power/gc_power,
            "aggregate_qualified": qualified, "safety_guard_passed": not harms,
            "eligible": qualified and not harms, "safety_guard_failures": harms,
            "joint_wins_vs_gc": sum(r["joint_win_vs_gc"] for r in comparisons),
            "joint_wins_vs_mse": sum(r["joint_win_vs_mse"] for r in comparisons), "feature_lead_count": len(comparisons),
            "mean_coherence_change_vs_mse": float(np.mean([r["coherence_change_vs_mse"] for r in comparisons])),
            "precipitation_humidity_guard": [r for r in comparisons if r["feature"] in ("TP6", "Q700")],
            "per_feature_lead": comparisons,
            "qualification_caveat": "Aggregate qualification does not imply every field/lead improves; joint wins and safety exceptions are explicit."}


def rank_finalists(candidates, policy=POLICY):
    eligible = [row for row in candidates if row["score"]["eligible"]]
    objectives = ("exact_gc_loss_ratio_vs_gc", "mean_rmse_ratio_vs_gc", "mean_relative_power_error_pct")
    for row in eligible:
        point = np.array([row["score"][key] for key in objectives])
        row["pareto_front"] = not any(np.all(np.array([other["score"][k] for k in objectives]) <= point) and
                                     np.any(np.array([other["score"][k] for k in objectives]) < point)
                                     for other in eligible if other is not row)
    ordered = sorted(eligible, key=lambda row: (not row["pareto_front"], -row["score"]["joint_wins_vs_gc"],
                                               row["score"]["mean_rmse_ratio_vs_gc"],
                                               row["score"]["mean_relative_power_error_pct"], row["id"]))
    selected, groups = [], set()
    for row in ordered:
        family = row.get("source_run_dir", row.get("family", row["id"]))
        if family not in groups:
            selected.append(row)
            groups.add(family)
        if len(selected) == policy["max_finalists"]:
            break
    for row in ordered:
        if len(selected) >= policy["max_finalists"]:
            break
        if row not in selected:
            selected.append(row)
    feedback = not selected or any(row["score"]["mean_coherence_change_vs_mse"] <
                                   -policy["coherence_drop_tolerance"] for row in selected)
    return selected, feedback


def grouped_weights(grouping, replicates, seed):
    rng = np.random.default_rng(seed)
    strata = sorted({row["stratum"] for row in grouping})
    groups = {}
    for i, row in enumerate(grouping):
        key = (row["stratum"], row["group"])
        groups.setdefault(key, []).append(i)
    # A named block cannot span strata; that would break the paired block identity.
    require(all(len({row["stratum"] for row in grouping if row["group"] == group}) == 1
                for group in {row["group"] for row in grouping}), "Bootstrap group spans strata")
    weights = np.zeros((replicates, len(grouping)))
    for stratum in strata:
        blocks = [indices for (s, _), indices in groups.items() if s == stratum]
        n_stratum = sum(map(len, blocks))
        for b in range(replicates):
            for index in rng.integers(0, len(blocks), size=len(blocks)):
                weights[b, blocks[index]] += 1
            selected = [i for block in blocks for i in block]
            weights[b, selected] *= (n_stratum/len(grouping))/weights[b, selected].sum()
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    return weights


def bootstrap_joint(candidate, reference, reference_branch, grouping, policy=POLICY, return_draws=False):
    """Paired seasonal calendar-block intervals; unavailable for reused old RMSE."""
    if candidate["global_mse"] is None or reference["global_mse"] is None:
        return {"status": "unavailable", "reason": "Saved evaluation lacks per-initialization primitive-channel MSE; aggregate RMSE cannot reconstruct paired intervals."}
    require(candidate["times"] == reference["times"] == [_time(r["initialization_time"]) for r in grouping],
            "Bootstrap paired dates differ")
    weights = grouped_weights(grouping, policy["bootstrap_replicates"], policy["bootstrap_seed"])
    leads = [day*4-1 for day in DAYS]
    vi = [candidate["variables"].index(feature) for feature in FEATURES]
    select = lambda a: a[:, leads][:, :, vi, 1:]
    cm = candidate["global_mse"]["full"][:, leads]
    rm = reference["global_mse"][reference_branch][:, leads]
    cp = select(candidate["stacked"]["full_power"])
    rp = select(reference["stacked"][f"{reference_branch}_power"])
    cc = select(candidate["stacked"]["full_cross_power"])
    rc = select(reference["stacked"][f"{reference_branch}_cross_power"])
    truth = select(candidate["stacked"]["truth_power"])
    np.testing.assert_array_equal(truth, select(reference["stacked"]["truth_power"]))
    metrics = {"mean_rmse_ratio": [], "mean_power_error_ratio": [], "exact_gc_loss_ratio": []}
    field_metrics = {"rmse_ratio": [], "power_error_ratio": [], "coherence_change": []}
    cl = np.mean([r["full_per_step"] for r in candidate["result"]["original_graphcast_loss_per_initialization"]], axis=1)
    rl = np.mean([r[f"{reference_branch}_per_step"] for r in reference["result"]["original_graphcast_loss_per_initialization"]], axis=1)
    for start in range(0, len(weights), 32):
        w = weights[start:start+32]
        rmse_ratio = np.sqrt(np.einsum("bn,nlv->blv", w, cm))/np.sqrt(np.einsum("bn,nlv->blv", w, rm))
        metrics["mean_rmse_ratio"].extend(np.mean(rmse_ratio, axis=(1, 2)))
        t = np.tensordot(w, truth, axes=1)
        c = np.tensordot(w, cp, axes=1)
        r = np.tensordot(w, rp, axes=1)
        c_error, r_error = np.mean(np.abs(c-t)/t, axis=3), np.mean(np.abs(r-t)/t, axis=3)
        metrics["mean_power_error_ratio"].extend(np.mean(c_error, axis=(1, 2))/np.mean(r_error, axis=(1, 2)))
        metrics["exact_gc_loss_ratio"].extend((w@cl)/(w@rl))
        field_metrics["rmse_ratio"].extend(rmse_ratio)
        field_metrics["power_error_ratio"].extend(c_error/r_error)
        field_metrics["coherence_change"].extend(np.mean(np.tensordot(w, cc, axes=1)/np.sqrt(c*t)-
                                                        np.tensordot(w, rc, axes=1)/np.sqrt(r*t), axis=3))
    arrays = {key: np.asarray(values) for key, values in metrics.items()}
    joint = np.logical_and.reduce([values < 1 for values in arrays.values()])
    field_arrays = {key: np.asarray(values) for key, values in field_metrics.items()}
    output = {"status": "complete", "replicates": len(weights), "seed": policy["bootstrap_seed"],
            "grouping": "Paired calendar blocks within seasons, fixed original season weights; groups supplied in manifest.",
            "group_count": len({(r["stratum"], r["group"]) for r in grouping}),
            "percentile95": {key: np.quantile(values, [.025, .975]).tolist() for key, values in arrays.items()},
            "per_feature_lead95": [{"feature": feature, "day": day,
                                     **{key: np.quantile(values[:, di, vi], [.025, .975]).tolist()
                                        for key, values in field_arrays.items()}}
                                    for vi, feature in enumerate(FEATURES) for di, day in enumerate(DAYS)],
            "joint_better_bootstrap_fraction": float(joint.mean()),
            "caveat": "Exploratory bootstrap fraction is not a posterior probability or correction for selecting candidates on these cases."}
    if return_draws:
        output["_draws"] = {"aggregate": arrays, "per_feature": field_arrays}
    return output


def aggregate_recipe(rows, draw_cache, policy):
    """Average scores of separately trained forecasts; never ensemble their fields."""
    first = rows[0]
    require(len({row.get("seed") for row in rows}) == len(rows), "Recipe contains duplicate/missing seed identities")
    scores = [row["score"] for row in rows]
    score = {key: float(np.mean([row[key] for row in scores])) for key in
             ("exact_gc_loss_ratio_vs_gc", "mean_rmse_ratio_vs_gc", "mean_relative_power_error_pct",
              "mean_power_error_ratio_vs_gc", "mean_coherence_change_vs_mse")}
    combined = []
    for index, template in enumerate(scores[0]["per_feature_lead"]):
        row = {"feature": template["feature"], "day": template["day"]}
        for key, value in template.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key != "day":
                row[key] = float(np.mean([s["per_feature_lead"][index][key] for s in scores]))
        for reference in ("gc", "mse"):
            row[f"joint_win_vs_{reference}"] = bool(row[f"rmse_ratio_vs_{reference}"] < 1 and
                                                    row[f"power_error_ratio_vs_{reference}"] < 1)
        combined.append(row)
    score["per_feature_lead"] = combined
    score["aggregate_qualified"] = all(score[key] < 1 for key in
                                       ("exact_gc_loss_ratio_vs_gc", "mean_rmse_ratio_vs_gc", "mean_power_error_ratio_vs_gc"))
    score["safety_guard_failures"] = [row for row in combined if row["feature"] in policy["safety_features"] and
                                      100*(row["rmse_ratio_vs_gc"]-1) > policy["maximum_safety_rmse_harm_pct"]]
    score["safety_guard_passed"] = not score["safety_guard_failures"]
    score["eligible"] = score["aggregate_qualified"] and score["safety_guard_passed"]
    score["feature_lead_count"] = len(combined)
    for reference in ("gc", "mse"):
        score[f"joint_wins_vs_{reference}"] = sum(row[f"joint_win_vs_{reference}"] for row in combined)
    score["precipitation_humidity_guard"] = [row for row in combined if row["feature"] in ("TP6", "Q700")]
    output = {"id": first["recipe_id"], "label": first.get("recipe_label", first["recipe_id"]),
              "seed_ids": [row.get("seed") for row in rows], "n_seeds": len(rows),
              "task_ids": [row["id"] for row in rows], "score": score,
              "all_seed_safety_guards_passed": all(s["safety_guard_passed"] for s in scores),
              "aggregation": "Mean of per-seed forecast metrics/ratios across independently trained models; not an ensemble mean field."}
    for reference in ("gc", "mse"):
        key = f"bootstrap_vs_{reference}"
        if not all((row["id"], reference) in draw_cache for row in rows):
            output[key] = {"status": "unavailable", "reason": "At least one seed lacks per-start paired metrics."}
            continue
        draws = [draw_cache[(row["id"], reference)] for row in rows]
        B = len(next(iter(draws[0]["aggregate"].values())))
        rng = np.random.default_rng(policy["bootstrap_seed"]+177)
        seed_indices = rng.integers(0, len(rows), size=(B, len(rows)))
        def combine(section, name):
            arrays = np.stack([draw[section][name] for draw in draws])
            # Every seed uses the same date-block draw b, preserving paired weather.
            return np.stack([arrays[seed_indices[b], b].mean(axis=0) for b in range(B)])
        aggregate = {name: combine("aggregate", name) for name in draws[0]["aggregate"]}
        fields = {name: combine("per_feature", name) for name in draws[0]["per_feature"]}
        output[key] = {"status": "complete", "replicates": B,
                       "percentile95": {name: np.quantile(values, [.025, .975]).tolist() for name, values in aggregate.items()},
                       "per_feature_lead95": [{"feature": feature, "day": day,
                                                **{name: np.quantile(values[:, di, vi], [.025, .975]).tolist()
                                                   for name, values in fields.items()}}
                                               for vi, feature in enumerate(FEATURES) for di, day in enumerate(DAYS)],
                       "joint_better_bootstrap_fraction": float(np.logical_and.reduce([v < 1 for v in aggregate.values()]).mean()),
                       "grouping": "Hierarchical seed resampling and shared paired seasonal date-block draws; candidate/control seed identity remains paired.",
                       "caveat": "Only three seeds give limited information about training variability. Percentile intervals and joint bootstrap fractions are descriptive, not a probability that a recipe is best."}
    return output


def report(manifest_path, allow_incomplete=False):
    manifest = load_manifest(manifest_path)
    complete, missing, sources = {}, [], []
    required_stages = ("spectral",) if manifest["evaluation_role"] == "confirmation" else ("exact", "spectral")
    for index, task in enumerate(manifest["task_specs"]):
        try:
            stages = {stage: resolve_result(manifest_path, index, stage) for stage in required_stages}
        except FileNotFoundError as error:
            if not allow_incomplete:
                raise
            missing.append({"id": task["id"], "reason": str(error)})
            continue
        spectral_dir, spectral_spec, spectral_result, reused = stages["spectral"]
        spectra = load_spectral_result(spectral_dir, spectral_spec, spectral_result)
        for directory, _, _, _ in stages.values():
            sources.extend(frozen_file(directory/name) for name in ("evaluation.json", "specification.json"))
        sources.extend(spectra["sources"])
        complete[task["id"]] = {"task": task, "spectra": spectra, "stages": stages, "spectral_reused": reused}
    require(manifest["canonical_mse_id"] in complete, "Canonical MSE control is incomplete")
    control = complete[manifest["canonical_mse_id"]]
    gc_rows, mse_rows = feature_rows(control["spectra"], "baseline"), feature_rows(control["spectra"], "full")
    gc_loss = control["spectra"]["result"]["original_graphcast_loss"]["baseline_rollout"]
    rows, draw_cache = [], {}
    for identity, item in complete.items():
        record = item["spectra"]
        np.testing.assert_array_equal(record["stacked"]["truth_power"], control["spectra"]["stacked"]["truth_power"])
        loss = record["result"]["original_graphcast_loss"]
        paired_id = item["task"].get("paired_mse_id", manifest["canonical_mse_id"])
        require(paired_id in complete, f"Paired MSE control is incomplete: {paired_id}")
        paired_control = complete[paired_id]
        paired_rows = feature_rows(paired_control["spectra"], "full")
        score = score_candidate(feature_rows(record), gc_rows, paired_rows, loss["full_rollout"]/gc_loss,
                                manifest["selection_policy"])
        intervals = {}
        for reference, comparison, branch in (("gc", control, "baseline"), ("mse", paired_control, "full")):
            interval = bootstrap_joint(record, comparison["spectra"], branch, manifest["bootstrap_grouping"],
                                       manifest["selection_policy"], return_draws=True)
            if "_draws" in interval:
                draw_cache[(identity, reference)] = interval.pop("_draws")
            intervals[f"bootstrap_vs_{reference}"] = interval
        spherical_paths = sorted((item["stages"]["spectral"][0]/"spherical").glob("*.npz"))
        expected_spherical = len(manifest["protocols"]["spectral"]["initialization_times"])
        if manifest.get("observer_factory") and not item["spectral_reused"]:
            require(len(spherical_paths) == expected_spherical, "New evaluation lacks required spherical observer outputs")
        sources.extend(frozen_file(path) for path in spherical_paths)
        row = {**item["task"], "score": score, "spectral_reused": item["spectral_reused"],
               "paired_mse_id": paired_id,
               "spherical_status": "complete" if len(spherical_paths) == expected_spherical else "not_saved_in_reused_evaluation",
               "paired_gc_rollout_loss": loss["baseline_rollout"], "canonical_gc_rollout_loss": gc_loss,
               "paired_gc_rollout_drift_pct": 100*(loss["baseline_rollout"]/gc_loss-1),
               "exact32_loss": (item["stages"]["exact"][2]["original_graphcast_loss"] if "exact" in item["stages"] else None),
               **intervals}
        rows.append(row)
    candidates = [row for row in rows if row["id"] != manifest["canonical_mse_id"] and row["role"] == "candidate"]
    if manifest["evaluation_role"] == "confirmation":
        finalists, feedback_needed = [], False
    else:
        finalists, feedback_needed = rank_finalists(candidates, manifest["selection_policy"])
    recipes = {}
    for row in rows:
        if row.get("recipe_id"):
            recipes.setdefault(row["recipe_id"], []).append(row)
    recipe_summaries = [aggregate_recipe(value, draw_cache, manifest["selection_policy"]) for value in recipes.values()]
    output = {"schema_version": 1, "status": "partial" if missing else "complete", "manifest": frozen_file(Path(manifest_path)),
              "evaluation_role": manifest["evaluation_role"], "scope": manifest["scope"], "selection_policy": manifest["selection_policy"],
              "canonical_mse_id": manifest["canonical_mse_id"], "records": rows, "missing": missing,
              "recipe_summaries": recipe_summaries,
              "confirmation_policy": "Confirmation evaluates prespecified recipes/seeds; it never selects new finalists or requests adaptive changes.",
              "finalists": [{"id": row["id"], "checkpoint": row["checkpoint"], "score": row["score"],
                              "pareto_front": row["pareto_front"]} for row in finalists],
              "feedback_needed": feedback_needed, "sources": sources,
              "interpretation": "Aggregate qualification is not an all-feature joint win. Small margins can reflect repeatability drift; intervals require per-start physical MSE. Confirmation must be evaluated separately."}
    destination = Path(manifest["output_dir"])/"report"
    write_json(destination/"joint_selection.json", output)
    lines = ["# AMSE regime evaluation", "", manifest["scope"], "", output["interpretation"], "",
             "| Candidate | GC loss ratio | Mean RMSE ratio | Power-error ratio | Joint wins / 36 | Safety guard | Qualified |",
             "|---|---:|---:|---:|---:|---|---|"]
    for row in rows:
        score = row["score"]
        lines.append(f"| {row['label']} | {score['exact_gc_loss_ratio_vs_gc']:.5f} | {score['mean_rmse_ratio_vs_gc']:.5f} | "
                     f"{score['mean_power_error_ratio_vs_gc']:.5f} | {score['joint_wins_vs_gc']} | {score['safety_guard_passed']} | {score['eligible']} |")
    lines += ["", "Finalists: " + (", ".join(row["id"] for row in finalists) or "none"),
              f"", f"Feedback investigation needed: {feedback_needed}.", "", f"Incomplete tasks: {len(missing)}."]
    (destination/"README.md").write_text("\n".join(lines)+"\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select-init", "prepare", "evaluate", "worker", "report"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task", type=int)
    parser.add_argument("--stage", choices=("exact", "spectral", "both"), default="both")
    parser.add_argument("--tasks-json", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mse-control-id")
    parser.add_argument("--reference-experiment", type=Path, default=REFERENCE)
    parser.add_argument("--spectral-initializations", type=Path)
    parser.add_argument("--observer-factory")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--role", choices=("development", "confirmation"), default="development")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if args.phase == "select-init":
        print(json.dumps(select_initialization(out=args.out), indent=2))
    elif args.phase == "prepare":
        if not args.tasks_json or not args.out_dir or not args.mse_control_id:
            parser.error("prepare requires --tasks-json --out-dir --mse-control-id")
        prepare_evaluation(read_json(args.tasks_json), args.out_dir, args.mse_control_id,
                           args.reference_experiment, args.spectral_initializations, args.observer_factory,
                           role=args.role, workspace=args.workspace)
        print(args.out_dir/"manifest.json")
    else:
        if not args.manifest:
            parser.error("--manifest is required")
        if args.phase == "report":
            report(args.manifest, args.allow_incomplete)
        elif args.task is None:
            parser.error("--task is required")
        elif args.phase == "worker":
            if args.stage == "both":
                parser.error("worker requires --stage exact|spectral")
            evaluate_worker(args.manifest, args.task, args.stage)
        else:
            evaluate(args.manifest, args.task, args.stage)


if __name__ == "__main__":
    main()
