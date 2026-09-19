"""Finalize the paired 4k cached-stepwise/online experiment with real evidence.

The default CLI runs six isolated GPU replay workers, then writes JSON,
Markdown, PNG and PDF artifacts. A missing checkpoint, validation, or replay
cannot produce a passing completion report. No training or job submission is
performed here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


FINAL_STEP = 4000
VALIDATION_STEPS = tuple(range(500, FINAL_STEP + 1, 500))
REPLAY_STEPS = (500, 2000, 4000)
REPLAY_PROTOCOL = "same_checkpoint_first_two_chunks_zero_then_carry_v1"
VALIDATION_PROTOCOL = "cached_stepwise_all_segments_v1"


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    from ..checkpoint import atomic_json_dump
    atomic_json_dump(value, Path(path))


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _source(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": _sha256(path)} if path.is_file() else {"path": str(path.resolve()), "missing": True}


def _records(path):
    if not Path(path).is_file():
        return []
    records = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected a record at {path}:{number}")
        records.append(record)
    return records


def _experiment(root):
    root = Path(root).resolve()
    manifest = _json(root / "manifest.json")
    if manifest.get("format") != "v24_cached_pair_v1":
        raise ValueError("Unsupported paired experiment manifest")
    configs = {name: _json(manifest["configs"][name]) for name in ("online", "cached")}
    run_dirs = {
        name: Path(config["output"]["output_root"]) / config["output"]["run_name"]
        for name, config in configs.items()
    }
    return manifest, configs, run_dirs


def checkpoint_path(run_dir, step):
    return Path(run_dir) / "checkpoints" / f"checkpoint_step{step:08d}.pkl"


def _replay_path(root, name, step):
    return Path(root) / "reports/replays" / f"{name}_step{step:08d}.json"


def _replay_identity(manifest, checkpoint, name, step):
    return {
        "protocol": REPLAY_PROTOCOL, "run": name, "step": int(step),
        "checkpoint_sha256": _sha256(checkpoint),
        "cache_manifest_sha256": manifest["execution"]["expected_manifest_sha256"],
        "source_digest": manifest["source_digest"],
    }


def replay_checkpoint(root, name, step):
    """Compare both backends at fixed weights, with independent weather data."""
    import jax
    import jax.numpy as jnp
    from ..checkpoint import load_v24_Ilya_training_checkpoint
    from .cached_diagnostics import memory_snapshot, tree_comparison
    from .cached_checkpoint import digest_json
    from .cached_parity import _context, _data_signature, _keys, _online, _trajectory
    from .data import TrainingCursor

    started = time.monotonic()
    manifest, _configs, run_dirs = _experiment(root)
    path = checkpoint_path(run_dirs[name], step)
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    if checkpoint.completed_step != step:
        raise ValueError("Replay checkpoint step differs from its requested filename")
    base_context = _context(root)
    # _context loads step zero. Replace that element explicitly and pass the
    # requested checkpoint parameters to every residual forward below.
    context = base_context[:3] + (checkpoint,) + base_context[4:]
    _, resolved, cached, _, _, _, cached_step = context
    if resolved.common.temporal_state_policy != "carry":
        raise ValueError("The paired replay protocol requires temporal state carry")
    _online_train_step, baseline, online_forward = _online(context)
    params = jax.device_put(checkpoint.residual_params)
    online_state = jax.tree_util.tree_map(jnp.zeros_like, checkpoint.residual_state)
    cached_state = jax.tree_util.tree_map(jnp.zeros_like, checkpoint.residual_state)
    rng = base_context[3].rng_key
    chunks = []
    for chunk_index in range(2):
        cursor = TrainingCursor(segment_index=0, segment_offset=chunk_index * resolved.common.bptt_steps)
        rng, keys = _keys(rng, resolved.common.bptt_steps)
        live_inputs, live_targets, live_forcings = _trajectory(context, cursor, keys, baseline)
        cache_batch = cached.build_chunk(cursor)
        live_signature = _data_signature((live_inputs, live_targets, live_forcings), resolved.common)
        cache_signature = _data_signature((cache_batch.inputs, cache_batch.targets, cache_batch.forcings), resolved.common)
        checks = []
        online_losses, cached_losses = [], []
        for index in range(resolved.common.bptt_steps):
            online_loss, online_prediction, online_state = online_forward(
                params, online_state, keys[index], live_inputs[index], live_targets[index], live_forcings[index]
            )
            cached_loss, cached_state, cached_prediction = cached_step.evaluate_step(
                params, cached_state, keys[index], cache_batch.inputs[index],
                cache_batch.targets[index], cache_batch.forcings[index],
            )
            online_loss, cached_loss, online_prediction, cached_prediction = jax.device_get(
                (online_loss, cached_loss, online_prediction, cached_prediction)
            )
            online_losses.append(float(online_loss))
            cached_losses.append(float(cached_loss))
            checks.append({
                "step_index": index,
                "prediction": tree_comparison(cached_prediction, online_prediction),
                "state": tree_comparison(jax.device_get(cached_state), jax.device_get(online_state)),
                "loss": tree_comparison(np.asarray(cached_loss), np.asarray(online_loss), relative=1e-4),
            })
            del online_prediction, cached_prediction
        chunk_loss = tree_comparison(np.asarray(cached_losses), np.asarray(online_losses), relative=1e-4)
        chunks.append({
            "cursor": cursor.to_dict(), "state_reset": chunk_index == 0,
            "data_identity": live_signature == cache_signature,
            "online_data_sha256": digest_json(live_signature),
            "cached_data_sha256": digest_json(cache_signature),
            "loss_components": chunk_loss, "per_step": checks,
            "online_training_losses": online_losses, "cached_training_losses": cached_losses,
            "passed": live_signature == cache_signature and chunk_loss["passed"]
            and all(item[key]["passed"] for item in checks for key in ("prediction", "state", "loss")),
        })
        del live_inputs, live_targets, live_forcings, cache_batch
    result = {
        **_replay_identity(manifest, path, name, step),
        "checkpoint": str(path), "initial_state": "zero", "chunks": chunks,
        "passed": len(chunks) == 2 and all(chunk["passed"] for chunk in chunks),
        "duration_seconds": time.monotonic() - started, "memory": memory_snapshot(),
        "scope": "Forward predictions, losses and recurrent state at fixed checkpoint weights; no optimizer updates",
    }
    _write_json(_replay_path(root, name, step), result)
    return result


def run_replays(root):
    """Use one process per checkpoint to release all compiled GPU allocations."""
    manifest, _configs, run_dirs = _experiment(root)
    directory = Path(root) / "reports/replays"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("online", "cached"):
        for step in REPLAY_STEPS:
            checkpoint = checkpoint_path(run_dirs[name], step)
            output = _replay_path(root, name, step)
            if not checkpoint.is_file():
                continue  # Completion assessment records the missing checkpoint/replay.
            identity = _replay_identity(manifest, checkpoint, name, step)
            if output.is_file():
                previous = _json(output)
                if assess_replay(previous, identity)["passed"]:
                    continue
            command = [sys.executable, "-m", "src.models.mamba.v24_Ilya.training.cached_report",
                       "--experiment-root", str(root), "--replay-run", name, "--replay-step", str(step)]
            print(f"[cached report] replay {name} step={step}", flush=True)
            with output.with_suffix(".log").open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=os.environ.copy())
            if completed.returncode:
                _write_json(output, {**identity, "passed": False, "worker_exit_code": completed.returncode,
                                     "log": str(output.with_suffix(".log"))})


def final_quality(online, cached):
    """Compare exact aggregate losses, not averages of lead improvements."""
    a = online["original_graphcast_loss"]
    b = cached["original_graphcast_loss"]
    baseline_a, baseline_b = float(a["baseline_rollout"]), float(b["baseline_rollout"])
    loss_a, loss_b = float(a["full_rollout"]), float(b["full_rollout"])
    values = (baseline_a, baseline_b, loss_a, loss_b)
    if not all(math.isfinite(value) and value >= 0 for value in values) or baseline_a <= 0 or baseline_b <= 0 or loss_a <= 0:
        return {"passed": False, "reason": "Nonfinite or invalid exact GraphCast loss"}
    improvement_a = 100 * (1 - loss_a / baseline_a)
    improvement_b = 100 * (1 - loss_b / baseline_b)
    relative_difference = abs(loss_b - loss_a) / loss_a
    improvement_difference = abs(improvement_b - improvement_a)
    same_baseline = math.isclose(baseline_a, baseline_b, rel_tol=1e-12, abs_tol=1e-12)
    return {
        "passed": bool(same_baseline and relative_difference <= 0.01 and improvement_difference <= 0.5),
        "same_baseline": same_baseline,
        "online_exact_loss": loss_a, "cached_exact_loss": loss_b,
        "relative_exact_loss_difference": relative_difference,
        "relative_exact_loss_limit": 0.01,
        "online_improvement_pct": improvement_a, "cached_improvement_pct": improvement_b,
        "improvement_difference_pp": improvement_difference, "improvement_limit_pp": 0.5,
    }


def assess_replay(record, expected_identity):
    """Require complete step evidence as well as a current checkpoint hash."""
    current = bool(expected_identity) and all(record.get(key) == value for key, value in expected_identity.items())
    chunks = record.get("chunks", [])
    complete = len(chunks) == 2 and record.get("initial_state") == "zero"
    for index, chunk in enumerate(chunks):
        per_step = chunk.get("per_step", [])
        complete = complete and (
            chunk.get("passed") is True and chunk.get("data_identity") is True
            and chunk.get("state_reset") is (index == 0)
            and chunk.get("cursor", {}).get("segment_index") == 0
            and chunk.get("cursor", {}).get("segment_offset") == index * 24
            and chunk.get("loss_components", {}).get("passed") is True
            and [row.get("step_index") for row in per_step] == list(range(24))
            and all(row.get(key, {}).get("passed") is True for row in per_step for key in ("prediction", "state", "loss"))
        )
    return {"passed": record.get("passed") is True and bool(current) and bool(complete),
            "identity_current": bool(current), "complete_step_evidence": bool(complete)}


def assess_run(training, validation, checkpoints):
    """Pure evidence gate; available files alone do not imply completed work."""
    reasons = []
    train_by_step = {int(row["step"]): row for row in training}
    missing_updates = sorted(set(range(1, FINAL_STEP + 1)) - train_by_step.keys())
    if missing_updates:
        reasons.append(f"Missing {len(missing_updates)} training updates")
    if max(train_by_step, default=0) != FINAL_STEP:
        reasons.append("Final logged update is not 4,000")
    if any(not math.isfinite(float(row.get("loss", float("nan")))) or not math.isfinite(float(row.get("gradient_norm", float("nan"))))
           for step, row in train_by_step.items() if 1 <= step <= FINAL_STEP):
        reasons.append("Nonfinite training losses or gradients")
    fixed = [row for row in validation if row.get("role") == "fixed_checkpoint"]
    validation_by_step = {int(row["step"]): row for row in fixed}
    if len(fixed) != len(validation_by_step):
        reasons.append("Duplicate fixed-validation records")
    missing_validation = [step for step in VALIDATION_STEPS if step not in validation_by_step]
    if missing_validation:
        reasons.append(f"Missing validation steps {missing_validation}")
    for step in VALIDATION_STEPS:
        checkpoint = checkpoints.get(str(step), {})
        if checkpoint.get("valid") is not True or checkpoint.get("completed_step") != step:
            reasons.append(f"Missing or invalid native checkpoint at {step}")
        row = validation_by_step.get(step)
        if row is None:
            continue
        if (row.get("validation_protocol") != VALIDATION_PROTOCOL or row.get("eval_feedback") != "baseline"
                or row.get("selected_segments") != 15 or row.get("available_segments") != 15
                or row.get("num_chunks") != 60 or row.get("num_anchors") != 1440
                or row.get("subset_policy") != "all_complete_segments"
                or row.get("physical_metric_precision") != "fp32"):
            reasons.append(f"Incomplete or different validation protocol at {step}")
        exact = row.get("original_graphcast_loss", {})
        needed = (row.get("loss"), exact.get("baseline_rollout"), exact.get("full_rollout"), exact.get("improvement_pct_rollout"))
        if any(value is None or not math.isfinite(float(value)) for value in needed):
            reasons.append(f"Missing/nonfinite exact validation metrics at {step}")
    return {"passed": not reasons, "reasons": reasons,
            "completed_training_updates": len(set(range(1, FINAL_STEP + 1)) & train_by_step.keys()),
            "final_logged_step": max(train_by_step, default=0),
            "fixed_validation_steps": sorted(step for step in validation_by_step if step in VALIDATION_STEPS)}


def _summary(values):
    values = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {"count": int(values.size), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "sum": float(values.sum()), "maximum": float(values.max())}


def _accounting(root):
    submission = Path(root) / "submission_production.json"
    if not submission.is_file():
        return {"available": False, "reason": "No production submission record"}
    jobs = _json(submission).get("jobs", {})
    ids = [str(jobs[name]) for name in ("online", "cached") if name in jobs]
    if not ids or any(not value.isdigit() for value in ids):
        return {"available": False, "reason": "No valid production job IDs"}
    fields = ["JobIDRaw", "State", "ElapsedRaw", "MaxRSS", "AllocCPUS", "ReqMem", "NodeList"]
    command = ["sacct", "-j", ",".join(ids), "--parsable2", "--noheader", "--format", ",".join(fields)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error), "command": command}
    path = Path(root) / "reports/slurm_accounting.txt"
    path.write_text(completed.stdout)
    records = [dict(zip(fields, line.split("|"), strict=True)) for line in completed.stdout.splitlines() if line]
    return {"available": bool(records), "jobs": jobs, "records": records, "source": _source(path), "command": command}


def _production_performance(root, run_dirs, training):
    result = {}
    submission_path = Path(root) / "submission_production.json"
    jobs = _json(submission_path).get("jobs", {}) if submission_path.is_file() else {}
    for name in ("online", "cached"):
        records = {int(row["step"]): row for row in training[name]}
        steady = [record for step, record in sorted(records.items()) if 20 < step <= FINAL_STEP]
        gpu_path = Path(root) / "logs" / f"{name}_{jobs.get(name, 'unknown')}_gpu.csv"
        gpu_peak = None
        if gpu_path.is_file():
            samples = []
            with gpu_path.open() as handle:
                for row in csv.reader(handle):
                    try:
                        samples.append(float(row[3].strip()))
                    except (IndexError, ValueError):
                        continue
            gpu_peak = max(samples) if samples else None
        device_peaks = [row.get("device_memory", {}).get("max_peak_bytes_in_use") for row in steady]
        device_peaks = [float(value) / 1024**2 for value in device_peaks if value is not None]
        startup_path = run_dirs[name] / "startup_metrics.jsonl"
        startup = _records(startup_path)
        config_path = run_dirs[name] / "run_config.json"
        if not startup and config_path.is_file():
            derived = _json(config_path).get("derived", {})
            if derived.get("startup_seconds") is not None:
                startup = [{"startup_seconds": derived["startup_seconds"]}]
        result[name] = {
            "compute_seconds": _summary(row.get("step_seconds") for row in steady),
            "end_to_end_seconds": _summary(row.get("end_to_end_seconds") for row in steady),
            "data_wait_seconds": _summary(row.get("data_wait_seconds") for row in steady),
            "sampled_gpu_peak_mib": gpu_peak,
            "jax_gpu_peak_mib": max(device_peaks) if device_peaks else None,
            "gpu_samples_source": _source(gpu_path),
            "checkpoint_seconds": _summary(row.get("checkpoint_seconds") for row in _records(run_dirs[name] / "checkpoint_metrics.jsonl")),
            "startup_seconds": _summary(row.get("startup_seconds") for row in startup),
            "definition": "Updates 21–4000; compute excludes loading, end-to-end includes loading. Checkpointing and validation are separate. Missing end-to-end fields in older logs remain unavailable.",
        }
    return result


def cache_generation_cost(cache_root):
    """Sum recorded worker phases; writes are already included in chunk time."""
    root = Path(cache_root)
    if not (root / "manifest.json").is_file():
        return {"available": False, "reason": "Cache manifest unavailable"}
    manifest = _json(root / "manifest.json")
    paths = sorted(root.glob("shard_*/COMPLETE.json"))
    records = []
    observed = []
    for path in paths:
        marker = _json(path)
        if marker.get("manifest_sha256") != manifest.get("manifest_sha256"):
            return {"available": False, "reason": f"Shard identity differs: {path}"}
        timings = marker.get("timings", [])
        initial = float(marker["initialization_seconds"])
        seconds = sum(float(row["seconds"]) for row in timings)
        writes = sum(float(row["write_seconds"]) for row in timings)
        if any(not math.isfinite(value) or value < 0 for value in (initial, seconds, writes)):
            return {"available": False, "reason": f"Invalid timing: {path}"}
        observed.extend(int(row["chunk_id"]) for row in timings)
        records.append({"source": _source(path), "initialization_seconds": initial,
                        "chunk_seconds_including_writes": seconds, "write_seconds_already_included": writes,
                        "recorded_worker_seconds": initial + seconds,
                        "payload_bytes": sum(int(row["bytes"]) for row in marker.get("files", {}).values())})
    expected = sorted(int(row["id"]) for row in manifest.get("chunks", []))
    complete = len(paths) == manifest.get("num_shards") and sorted(observed) == expected and bool(expected)
    total = sum(row["recorded_worker_seconds"] for row in records)
    return {"available": complete, "complete_chunk_coverage": complete, "recorded_chunks": len(observed),
            "summed_recorded_worker_seconds": total, "summed_recorded_worker_hours": total / 3600,
            "payload_gib": sum(row["payload_bytes"] for row in records) / 1024**3,
            "shards": records,
            "definition": "Sum of predictor initialization and all chunk seconds across GPU workers, not concurrent elapsed time or pure GPU compute. Chunk seconds already include write/flush/hash time. Excludes prepared-context setup, final shard verification, pilot/full verification, and scheduler overhead; those costs are unavailable here."}


def _plot(path, training, validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), constrained_layout=True)
    colors = {"online": "#16697a", "cached": "#d35f27"}
    for name in ("online", "cached"):
        by_step = {int(row["step"]): row for row in training[name] if 1 <= int(row["step"]) <= FINAL_STEP}
        rows = [value for _, value in sorted(by_step.items())]
        if rows:
            steps = np.asarray([row["step"] for row in rows])
            losses = np.asarray([row["loss"] for row in rows], dtype=float)
            window = min(100, len(rows))
            smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
            axes[0, 0].plot(steps[window - 1:], smooth, label=name, color=colors[name], linestyle="--" if name == "cached" else "-")
        values = sorted((row for row in validation[name] if row.get("role") == "fixed_checkpoint"), key=lambda row: row["step"])
        fields = ((axes[0, 1], lambda row: row["loss"]),
                  (axes[1, 0], lambda row: row["original_graphcast_loss"]["full_rollout"]),
                  (axes[1, 1], lambda row: 100 * (1 - row["original_graphcast_loss"]["full_rollout"] / row["original_graphcast_loss"]["baseline_rollout"])))
        for axis, getter in fields:
            usable = []
            for row in values:
                try:
                    value = float(getter(row))
                except (KeyError, TypeError, ZeroDivisionError):
                    continue
                if math.isfinite(value):
                    usable.append((row["step"], value))
            if usable:
                axis.plot(*zip(*usable), marker="o", markersize=3, color=colors[name],
                          linestyle="--" if name == "cached" else "-", label=name)
    titles = ("Training objective · trailing 100 updates", "Held-out training objective",
              "Exact GraphCast loss · held-out 2022", "Exact GraphCast loss reduction")
    labels = ("Legacy BF16 objective", "Legacy BF16 objective", "Original GC weighted loss", "Improvement (%)")
    for axis, title, label in zip(axes.flat, titles, labels, strict=True):
        axis.set(title=title, xlabel="Optimizer update", ylabel=label, xlim=(0, FINAL_STEP))
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(frameon=False)
    fig.suptitle("Online versus cached stepwise residual GC + Mamba · identical open-loop setup")
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def build_report(root, *, collect_accounting=True, make_plots=True):
    from ..checkpoint import load_v24_Ilya_training_checkpoint
    from .cached_checkpoint import build_provenance, scientific_config
    from .config import load_training_config

    root = Path(root).resolve()
    manifest, configs, run_dirs = _experiment(root)
    training = {name: _records(directory / "train_metrics.jsonl") for name, directory in run_dirs.items()}
    validation = {name: _records(directory / "validation_metrics.jsonl") for name, directory in run_dirs.items()}
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    shared_init_sha256 = _sha256(manifest["shared_init"]) if Path(manifest["shared_init"]).is_file() else None
    runs = {}
    for name, directory in run_dirs.items():
        checkpoints = {}
        for step in VALIDATION_STEPS:
            path = checkpoint_path(directory, step)
            evidence = _source(path)
            try:
                checkpoint = load_v24_Ilya_training_checkpoint(path)
                compatible = scientific_config(checkpoint.resolved_training_config) == scientific_config(configs[name])
                evidence.update(completed_step=checkpoint.completed_step, valid=checkpoint.completed_step == step and compatible)
                del checkpoint
            except (FileNotFoundError, ValueError) as error:
                evidence.update(valid=False, error=str(error))
            checkpoints[str(step)] = evidence
        initialization_path = directory / "shared_initialization.json"
        initialization = _json(initialization_path) if initialization_path.is_file() else {}
        initial_checkpoint_path = checkpoint_path(directory, 0)
        initialization_valid = (
            shared_init_sha256 is not None
            and initialization.get("shared_init_sha256") == shared_init_sha256
            and initial_checkpoint_path.is_file()
            and initialization.get("projected_checkpoint_sha256") == _sha256(initial_checkpoint_path)
        )
        runs[name] = {
            "run_dir": str(directory), "evidence": assess_run(training[name], validation[name], checkpoints),
            "shared_initialization": {"passed": bool(initialization_valid), "source": _source(initialization_path)},
            "checkpoints": checkpoints, "validation": validation[name],
            "sources": {"config": _source(manifest["configs"][name]),
                        "training": _source(directory / "train_metrics.jsonl"),
                        "validation": _source(directory / "validation_metrics.jsonl")},
        }
    replay_records = {}
    for name in ("online", "cached"):
        for step in REPLAY_STEPS:
            path = _replay_path(root, name, step)
            record = _json(path) if path.is_file() else {"passed": False, "missing": True}
            checkpoint = checkpoint_path(run_dirs[name], step)
            expected = _replay_identity(manifest, checkpoint, name, step) if checkpoint.is_file() else {}
            replay_records[f"{name}:{step}"] = {**assess_replay(record, expected), "source": _source(path), "result": record}
    final = {name: next((row for row in validation[name] if row.get("role") == "fixed_checkpoint" and row.get("step") == FINAL_STEP), None)
             for name in ("online", "cached")}
    try:
        quality = final_quality(final["online"], final["cached"]) if all(final.values()) else {"passed": False, "reason": "Missing final validation"}
    except (KeyError, TypeError, ValueError) as error:
        quality = {"passed": False, "reason": f"Invalid final validation: {error}"}
    identity_keys = ("validation_protocol", "subset_fingerprint", "cache_manifest_sha256", "segment_ids",
                     "training_loss_precision", "physical_metric_precision", "eval_feedback")
    validation_identity = bool(all(final.values()))
    for step in VALIDATION_STEPS:
        paired_rows = {
            name: next((row for row in validation[name] if row.get("role") == "fixed_checkpoint" and row.get("step") == step), None)
            for name in ("online", "cached")
        }
        if not all(paired_rows.values()):
            validation_identity = False
            continue
        for key in identity_keys:
            expected = final["online"].get(key) if final["online"] is not None else None
            validation_identity = validation_identity and expected is not None and all(row.get(key) == expected for row in paired_rows.values())
        validation_identity = validation_identity and all(
            row.get("cache_manifest_sha256") == manifest["execution"]["expected_manifest_sha256"]
            for row in paired_rows.values()
        )
    same_science = scientific_config(configs["online"]) == scientific_config(configs["cached"])
    gates = {}
    for name in ("full_parity", "paired20", "benchmark"):
        path = Path(manifest["reports"][name])
        value = _json(path) if path.is_file() else {}
        gates[name] = {"passed": value.get("passed") is True, "source": _source(path), "report": value}
    gate_identities = [value["report"].get("provenance") for value in gates.values()]
    try:
        expected_provenance = build_provenance(
            load_training_config(Path(manifest["configs"]["cached"])),
            Path(manifest["shared_init"]), Path(manifest["cache_root"]),
            Path(manifest["source_root"]), execution=manifest["execution"],
        )
        same_gate_provenance = all(value == expected_provenance for value in gate_identities)
        provenance_error = None
    except (OSError, ValueError, KeyError) as error:
        same_gate_provenance = False
        provenance_error = str(error)
    checks = {
        "online_complete": runs["online"]["evidence"]["passed"], "cached_complete": runs["cached"]["evidence"]["passed"],
        "same_scientific_configuration": same_science, "same_validation_protocol_and_examples": bool(validation_identity),
        "same_shared_initialization": all(run["shared_initialization"]["passed"] for run in runs.values()),
        "preproduction_gates": all(value["passed"] for value in gates.values()) and same_gate_provenance,
        "all_six_same_checkpoint_replays": all(value["passed"] for value in replay_records.values()),
        "final_quality_within_limits": quality["passed"],
    }
    performance = {
        "matched_benchmark": gates["benchmark"]["report"],
        "production": _production_performance(root, run_dirs, training),
        "cache_generation": cache_generation_cost(manifest["cache_root"]),
        "slurm_accounting": _accounting(root) if collect_accounting else {"available": False, "reason": "Not requested"},
    }
    result = {
        "format": "v24_cached_stepwise_pair_report_v1", "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": all(checks.values()), "status": "passed" if all(checks.values()) else "incomplete_or_failed",
        "checks": checks, "final_quality": quality, "runs": runs, "replays": replay_records,
        "performance": performance, "preproduction_gates": gates,
        "preproduction_provenance_error": provenance_error,
        "manifest_source": _source(root / "manifest.json"),
        "notes": ["Same-checkpoint replays compare each learned checkpoint through both execution backends; they do not require the two training runs to have identical parameters.",
                  "Validation uses all 60 cached 2022 chunks and preserves the cache's observed prefix and four-chunk segment-state policy.",
                  "Production end-to-end fields include loading when present; older missing fields are not reconstructed. The separate benchmark holds resource settings matched."],
    }
    if make_plots:
        _plot(reports / "paired_learning_curves", training, validation)
    result["artifacts"] = {"json": str(reports / "final_report.json"), "markdown": str(reports / "final_report.md"),
                           "plot_png": str(reports / "paired_learning_curves.png"), "plot_pdf": str(reports / "paired_learning_curves.pdf")}
    _write_json(reports / "final_report.json", result)
    _write_markdown(reports / "final_report.md", result)
    return result


def _write_markdown(path, report):
    lines = ["# Paired cached-stepwise training report", "", f"Status: **{report['status']}**.", "",
             "Both runs target 4,000 updates with identical residual GC+Mamba initialization, open-loop trajectories and optimizer settings.",
             "", "## Acceptance", "", "| Check | Result |", "|---|---|"]
    lines += [f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL / MISSING'} |" for name, passed in report["checks"].items()]
    quality = report["final_quality"]
    if "online_exact_loss" in quality:
        lines += ["", "## Final exact GraphCast loss", "", "| Run | Exact loss | Improvement over GC |", "|---|---:|---:|",
                  f"| Online | {quality['online_exact_loss']:.9g} | {quality['online_improvement_pct']:.4f}% |",
                  f"| Cached | {quality['cached_exact_loss']:.9g} | {quality['cached_improvement_pct']:.4f}% |", "",
                  f"Relative loss difference: **{100*quality['relative_exact_loss_difference']:.4f}%** (limit 1%). Improvement difference: **{quality['improvement_difference_pp']:.4f} percentage points** (limit 0.5)."]
    lines += ["", "## Completion evidence", ""]
    for name, run in report["runs"].items():
        evidence = run["evidence"]
        lines.append(f"- {name}: {evidence['completed_training_updates']}/4,000 logged updates; {len(evidence['fixed_validation_steps'])}/8 fixed validations. Run: `{run['run_dir']}`.")
        lines.extend(f"  - {reason}" for reason in evidence["reasons"])
    benchmark = report["performance"]["matched_benchmark"]
    lines += ["", "## Runtime and memory", ""]
    if benchmark.get("matched_speedup") is not None:
        lines.append(f"Measured matched end-to-end median speedup: **{float(benchmark['matched_speedup']):.3f}×**. This comes from the separate benchmark after warmup.")
    lines += ["", "| Production run | Median compute, loading excluded | Median end-to-end | P90 end-to-end | Sampled GPU peak |", "|---|---:|---:|---:|---:|"]
    for name, values in report["performance"]["production"].items():
        timing = values["compute_seconds"]
        median = f"{timing['median']:.3f}s" if timing else "unavailable"
        elapsed = values["end_to_end_seconds"]
        end_to_end = f"{elapsed['median']:.3f}s" if elapsed else "unavailable"
        p90 = f"{elapsed['p90']:.3f}s" if elapsed else "unavailable"
        gpu = f"{values['sampled_gpu_peak_mib']:.0f}MiB" if values["sampled_gpu_peak_mib"] is not None else "unavailable"
        lines.append(f"| {name} | {median} | {end_to_end} | {p90} | {gpu} |")
    generation = report["performance"]["cache_generation"]
    if generation.get("available"):
        lines += ["", f"One-time cache generation recorded **{generation['summed_recorded_worker_hours']:.3f} summed worker-hours** across {len(generation['shards'])} shards, producing {generation['payload_gib']:.2f} GiB. Writes are already included. This excludes setup outside the recorded phases and pilot/final verification; their additional costs are unavailable."]
    else:
        lines += ["", "One-time cache-generation accounting is incomplete or unavailable; see JSON evidence."]
    lines += ["", "End-to-end update timing includes chunk loading when logged; older absent fields remain unavailable. Checkpointing/validation are separate. GPU telemetry is sampled and may miss short peaks. Slurm accounting and source hashes are preserved in the JSON report.",
              "", "## Same-checkpoint replays", "", "Each run's checkpoints 500, 2,000 and 4,000 are replayed through both backends on the first two fixed training chunks: zero initial memory, then carry. Parameters remain fixed; predictions, losses and recurrent state are compared.", ""]
    for label, record in report["replays"].items():
        lines.append(f"- {label}: {'PASS' if record['passed'] else 'FAIL / MISSING'} ([details]({record['source']['path']})).")
    lines += ["", "## Curves", "", "![Paired training and validation curves](paired_learning_curves.png)", "",
              "[PDF curves](paired_learning_curves.pdf) · [Machine-readable report and source hashes](final_report.json)", ""]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--replay-run", choices=("online", "cached"))
    parser.add_argument("--replay-step", type=int, choices=REPLAY_STEPS)
    parser.add_argument("--summarize-only", action="store_true", help="Read existing replay evidence; never treat missing replays as passing")
    args = parser.parse_args(argv)
    if bool(args.replay_run) != bool(args.replay_step):
        parser.error("--replay-run and --replay-step must be supplied together")
    if args.replay_run:
        result = replay_checkpoint(args.experiment_root, args.replay_run, args.replay_step)
    else:
        if not args.summarize_only:
            run_replays(args.experiment_root)
        result = build_report(args.experiment_root)
        print(f"[cached report] {result['status']}: {result['artifacts']['markdown']}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
