"""Fresh, batched one-step residual training on immutable teacher caches.

The cache supplies all weather inputs and frozen predictions. Only the residual
network is initialized on the accelerator. Native checkpoints retain parameter
compatibility with live training; an additional backend record owns the exact
batched sampler cursor, recurrent state, RNG and immutable provenance.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import os
import pickle
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ..checkpoint import (
    atomic_json_dump, atomic_pickle_dump, file_sha256,
    load_v24_Ilya_training_checkpoint, training_checkpoint_payload,
)
from .cached_checkpoint import (
    code_fingerprints, digest_json, numerical_tree_sha256, reconcile_run_metrics,
)
from .config import load_training_config
from .endpoint_step import build_optimizer, cast_state_boundary_fp32
from .validation import append_validation_record, load_validation_records, update_best_validation, validation_step_keys


BACKEND = "cached_k1_batched_v1"
CHECKPOINT_KEY = "cached_k1_backend"


def require_k1_runtime() -> dict:
    """Pin this backend's runtime without inheriting the old parity kernel flags."""
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("Cached k1 requires exactly one allocated GPU")
    return {"device": devices[0].device_kind,
            "jax": importlib.metadata.version("jax"),
            "jaxlib": importlib.metadata.version("jaxlib"),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "xla_python_client_allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")}


def _append(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def scientific_config(config) -> dict:
    """Keep the entire cosine schedule fixed, including in bounded preflights."""
    value = json.loads(json.dumps(config.to_dict()))
    value.pop("output", None)
    value.pop("validation", None)
    return value


def build_k1_provenance(config, producer_config_path: Path, cached, *, epochs: int, shuffle: bool,
                        runtime: Mapping[str, Any] | None = None) -> dict:
    return {
        "format": BACKEND,
        "scientific_config_sha256": digest_json(scientific_config(config)),
        "producer_config_sha256": file_sha256(producer_config_path),
        "cache_manifest_sha256": cached.manifest["manifest_sha256"],
        "cache_compatibility_sha256": cached.manifest["compatibility_sha256"],
        "code_sha256": code_fingerprints(Path(__file__).resolve().parents[5]),
        "environment": {name: importlib.metadata.version(name)
                        for name in ("jax", "jaxlib", "dm-haiku", "optax")},
        "epochs": epochs, "shuffle": shuffle,
        "padding": "retain_all_segments_mask_invalid_lanes",
        "initialization": "fresh_residual_only",
        "runtime": dict(runtime or {}),
    }


def validate_k1_config(config, cached, data, *, epochs: int) -> None:
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if config.distributed.mode != "single":
        raise ValueError("Cached k1 requires one device with a batched residual")
    if config.architecture.residual_initialization != "fresh":
        raise ValueError("Cached k1 requires fresh residual initialization")
    if (config.feedback_mode, config.loss_mode, config.loss_kind, config.ar_tail_k) != (
            "baseline", "all_steps", "mse", 0):
        raise ValueError("Cached k1 requires baseline feedback, all_steps MSE and ar_tail_k=0")
    if config.temporal_state_policy != "carry" or (config.segment_steps, config.bptt_steps) != (96, 24):
        raise ValueError("Cached k1 requires state carry over 96-step segments and 24-step BPTT")
    producer = cached.common
    for name in ("baseline_checkpoint", "stats_dir", "prepared_root", "anchor_manifest_root",
                 "input_duration", "target_steps", "segment_steps", "bptt_steps", "ar_tail_k",
                 "feedback_mode", "time_start", "time_end"):
        if getattr(config, name) != getattr(producer, name):
            raise ValueError(f"Cached consumer {name} differs from the verified producer")
    for name in ("resolution", "width", "mesh_size", "baseline_msg_steps"):
        if getattr(config.architecture, name) != getattr(producer.architecture, name):
            raise ValueError(f"Cached consumer baseline architecture {name} differs from the verified producer")
    expected = epochs * data.steps_per_epoch
    if config.max_steps != expected:
        raise ValueError(f"Full learning-rate schedule requires max_steps={expected}; got {config.max_steps}")
    if config.learning_rate_schedule != "cosine":
        raise ValueError("Cached k1 production and preflight must use the full cosine schedule")


def write_k1_checkpoint(path: Path, *, config, completed_step: int, params, state,
                        optimizer_state, rng, cursor, samples_seen: int,
                        baseline_fingerprint: str, anchor_fingerprint: str,
                        provenance: Mapping[str, Any]) -> Path:
    payload = training_checkpoint_payload(
        completed_step=completed_step, residual_params=params, residual_state=state,
        optimizer_state=optimizer_state, rng_key=rng,
        # Deliberately retain the backend's distinct cursor names: native init-
        # from uses parameters, while accidental live exact-resume must fail.
        training_cursor=cursor.to_dict(), resolved_training_config=config.to_dict(),
        baseline_checkpoint_path=str(config.baseline_checkpoint),
        baseline_checkpoint_fingerprint=baseline_fingerprint,
        anchor_manifest_fingerprint=anchor_fingerprint,
        parameter_overlay_metadata={"residual_initialization": "fresh", "residual_copied": 0,
                                    "baseline_initializations": 0, "baseline_apply_calls": 0},
    )
    payload[CHECKPOINT_KEY] = {
        "format": BACKEND, "cursor": cursor.to_dict(), "samples_seen": int(samples_seen),
        "provenance": dict(provenance),
    }
    atomic_pickle_dump(payload, path)
    atomic_json_dump({"format": BACKEND, "checkpoint_sha256": file_sha256(path),
                      "completed_step": completed_step, "provenance": dict(provenance)},
                     path.with_suffix(".cached_k1.json"))
    atomic_json_dump({"completed_step": completed_step, "checkpoint": str(path)},
                     config.run_dir / "latest_checkpoint.json")
    return path


def load_k1_checkpoint(path: Path, provenance: Mapping[str, Any], *, data=None):
    """Validate identity before returning the native checkpoint and backend state."""
    sidecar = json.loads(path.with_suffix(".cached_k1.json").read_text())
    if sidecar.get("format") != BACKEND or sidecar.get("checkpoint_sha256") != file_sha256(path):
        raise ValueError("Cached k1 checkpoint format or checksum differs")
    if sidecar.get("provenance") != dict(provenance):
        raise ValueError("Cached k1 resume provenance differs")
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    with path.open("rb") as handle:
        backend = pickle.load(handle).get(CHECKPOINT_KEY)
    if not isinstance(backend, dict) or backend.get("format") != BACKEND:
        raise ValueError("Missing versioned cached k1 backend state")
    if backend.get("provenance") != dict(provenance):
        raise ValueError("Cached k1 payload provenance differs")
    if checkpoint.completed_step != sidecar.get("completed_step"):
        raise ValueError("Cached k1 completed step differs from sidecar")
    if checkpoint.training_cursor != backend.get("cursor"):
        raise ValueError("Cached k1 cursor differs between native and backend records")
    if type(backend.get("samples_seen")) is not int or backend["samples_seen"] < 0:
        raise ValueError("Invalid cached k1 sample count")
    if data is not None:
        from .cached_k1_data import CachedK1Cursor
        cursor = CachedK1Cursor.from_mapping(backend["cursor"])
        if data.completed_updates(cursor) != checkpoint.completed_step:
            raise ValueError("Cached k1 cursor does not match completed updates")
        expected_samples = samples_before_cursor(data, cursor)
        if expected_samples != backend["samples_seen"]:
            raise ValueError("Cached k1 sample count includes missing or padded lanes")
    return checkpoint, backend


def samples_before_cursor(data, cursor) -> int:
    """Count real weather targets, including an incomplete final lane group."""
    batch_size = data.batch_size
    completed_segments = min(cursor.group_index * batch_size, data.num_segments)
    lanes = min(batch_size, max(0, data.num_segments - completed_segments))
    return int((cursor.epoch * data.num_segments + completed_segments) * 96
               + cursor.chunk_index * 24 * lanes)


def checkpoint_numerical_identity(checkpoint, backend) -> dict:
    return {**{name: numerical_tree_sha256(getattr(checkpoint, name))
               for name in ("residual_params", "residual_state", "optimizer_state", "rng_key")},
            "cursor": dict(backend["cursor"]), "samples_seen": backend["samples_seen"],
            "completed_step": checkpoint.completed_step}


def compare_k1_checkpoints(left: Path, right: Path, provenance: Mapping[str, Any]) -> dict:
    """Exact numerical restart gate; paths/output directories need not match."""
    actual = checkpoint_numerical_identity(*load_k1_checkpoint(left, provenance))
    expected = checkpoint_numerical_identity(*load_k1_checkpoint(right, provenance))
    return {"passed": actual == expected, "actual": actual, "expected": expected,
            "kind": "cached_k1_exact_resume", "provenance": dict(provenance)}


def parameter_update_report(before: Path, after: Path) -> dict:
    """Require finite learning in each spatial block, Mamba and correction head.

    Only group totals must change: zero-initialized output projections can delay
    updates to individual leaves. Compare bounded FP64 pieces without copying
    the complete parameter tree or retaining either checkpoint's optimizer.
    """
    before, after = Path(before), Path(after)
    expected = load_v24_Ilya_training_checkpoint(before).residual_params
    actual = load_v24_Ilya_training_checkpoint(after).residual_params
    report = {"kind": "cached_k1_parameter_updates", "passed": False,
              "before": str(before), "after": str(after),
              "before_sha256": file_sha256(before), "after_sha256": file_sha256(after)}
    if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(expected):
        return {**report, "reason": "parameter tree structure differs"}
    groups = {name: {"elements": 0, "changed_elements": 0, "max_abs": 0.0, "squared_l2": 0.0}
              for name in ("grid2mesh", "mesh_processor", "mesh2grid", "temporal", "correction_head")}
    spatial = {"grid2mesh_gnn": "grid2mesh", "mesh_gnn": "mesh_processor", "mesh2grid_gnn": "mesh2grid"}
    for module, values in expected.items():
        root = module.split("/")[0]
        if root == "temporal_residual_head":
            group = "correction_head"
        elif root.startswith("mesh_interleaved_temporal_") or "temporal_mesh_mamba" in module:
            group = "temporal"
        elif root in spatial:
            group = spatial[root]
        else:
            return {**report, "reason": f"Unclassified parameter module: {module}"}
        item = groups[group]
        for left, right in zip(jax.tree_util.tree_leaves(actual[module]),
                               jax.tree_util.tree_leaves(values), strict=True):
            left, right = np.asarray(left), np.asarray(right)
            if left.shape != right.shape or left.dtype != right.dtype:
                return {**report, "reason": f"Parameter shape or dtype differs: {module}"}
            left, right = left.reshape(-1), right.reshape(-1)
            for offset in range(0, left.size, 262144):
                a = left[offset:offset + 262144].astype(np.float64)
                b = right[offset:offset + 262144].astype(np.float64)
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    return {**report, "reason": f"Nonfinite parameters: {module}"}
                delta = a - b
                item["elements"] += int(delta.size)
                item["changed_elements"] += int(np.count_nonzero(delta))
                item["max_abs"] = max(item["max_abs"], float(np.max(np.abs(delta), initial=0)))
                item["squared_l2"] += float(np.sum(delta * delta))
    for item in groups.values():
        squared_l2 = item.pop("squared_l2")
        finite = math.isfinite(squared_l2) and math.isfinite(item["max_abs"])
        item["absolute_l2"] = math.sqrt(squared_l2) if finite else None
        item["finite"] = finite
        item["passed"] = finite and item["elements"] > 0 and item["changed_elements"] > 0
    return {**report, "groups": groups, "passed": all(item["passed"] for item in groups.values())}


def prediction_roundtrip_check(*, train_step, checkpoint, params, state, rng, batch,
                               relative: float = 1e-3, absolute: float = 1e-5) -> dict:
    """Separate repeated-forward variability from native-reload differences.

    The PRNG is derived without advancing the training stream. Parameter/state
    checkpoint bytes are checked separately and exactly; this check permits
    normal kernel roundoff for the explicitly recorded GPU runtime. Components
    are gated separately so a large recurrent state cannot hide prediction or
    loss errors. Keep only the reference and one candidate output on the host.
    """
    from .cached_diagnostics import tree_comparison
    device = jax.devices()[0]
    # Explicit placement makes both calls use committed device arrays. Passing
    # NumPy checkpoint leaves directly can select a different JIT executable.
    params, state, rng = jax.device_put((params, state, rng), device)
    key = jax.random.fold_in(rng, 0x4B31)
    args = (key, batch.inputs[0], batch.targets[0], batch.forcings[0], batch.valid_mask)
    expected = jax.device_get(train_step.evaluate_step(params, state, *args))

    def compare(actual):
        integrity = True
        if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(expected):
            raise ValueError('Prediction repeat tree structure differs')
        for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
            left, right = np.asarray(left), np.asarray(right)
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError('Prediction repeat shape or dtype differs')
            if not (np.isfinite(left).all() and np.isfinite(right).all()):
                integrity = False
        aggregate = tree_comparison(actual, expected, relative=relative, absolute=absolute)
        components = {name: tree_comparison(left, right, relative=relative, absolute=absolute)
                      for name, left, right in zip(
                          ("loss", "state", "prediction"), actual, expected, strict=True)}
        return {"passed": aggregate["passed"] and all(item["passed"] for item in components.values()),
                "integrity_passed": integrity, "aggregate": aggregate, "components": components}

    repeated = jax.device_get(train_step.evaluate_step(params, state, *args))
    repeat = compare(repeated)
    del repeated
    reloaded_params, reloaded_state = jax.device_put(
        (checkpoint.residual_params, checkpoint.residual_state), device)
    actual = jax.device_get(train_step.evaluate_step(
        reloaded_params, reloaded_state, *args))
    reload = compare(actual)
    # Preserve legacy aggregate metrics at the top level; passed now also
    # requires each physical component and the unchanged-input repeat gate.
    return {"kind": "cached_k1_prediction_roundtrip", **reload["aggregate"],
            "integrity_passed": repeat['integrity_passed'] and reload['integrity_passed'],
            "passed": repeat["passed"] and reload["passed"],
            "repeat": repeat, "reload": reload,
            "relative_tolerance": relative, "absolute_tolerance": absolute}


def roundtrip_result(fields, prediction, *, numerical_policy='strict', **metadata):
    """Serialization is exact; post-load computation can be advisory."""
    if numerical_policy not in ('strict', 'advisory'):
        raise ValueError('Unknown numerical policy')
    integrity = all(fields.values()) and prediction.get('integrity_passed') is True
    numerical = prediction['passed']
    return {'kind': 'cached_k1_checkpoint_roundtrip', **metadata, 'fields': fields,
            'serialization_passed': all(fields.values()), 'integrity_passed': integrity,
            'numerical_policy': numerical_policy, 'prediction_roundtrip': prediction,
            'numerical_status': 'pass' if numerical else 'warning',
            'status': ('complete' if numerical else 'complete_with_warnings') if integrity else 'failed_integrity',
            'passed': integrity and (numerical or numerical_policy == 'advisory')}


def exact_saved_state(checkpoint, expected):
    """Reject nonfinite or altered serialized arrays, including signed zeros."""
    fields = {}
    for name, value in expected.items():
        saved = getattr(checkpoint, name)
        for tree in (value, saved):
            for leaf in jax.tree_util.tree_leaves(tree):
                array = np.asarray(jax.device_get(leaf))
                if array.dtype.hasobject or not np.isfinite(array).all():
                    raise ValueError(f'Nonfinite or unsupported checkpoint field: {name}')
        fields[name] = numerical_tree_sha256(value) == numerical_tree_sha256(saved)
    if not all(fields.values()):
        raise ValueError('Cached k1 exact checkpoint serialization failed')
    return fields


def run_k1_validation(*, train_step, data, cached, config, params, zero_state,
                      step: int, max_segments: int | None = None) -> dict:
    """Fixed cached validation; padding never contributes to either loss."""
    from ..metrics import V24IlyaMetricAccumulator
    available = len(cached.training_data.validation_segments)
    selected = available if max_segments is None else min(available, max_segments)
    if selected < 1:
        raise ValueError("Validation requires at least one segment")
    started = time.monotonic()
    exact = V24IlyaMetricAccumulator(
        config.bptt_steps, np.asarray(cached.manifest["coordinates"]["lat"]),
        diffs_stddev_by_level=cached.context[3]["diffs_stddev_by_level"], store_spatial_bias=False,
    )
    total_loss = 0.0
    components_total = np.zeros(config.bptt_steps, np.float64)
    count = 0
    chunks = 0
    for group in range(math.ceil(selected / data.batch_size)):
        state = jax.tree_util.tree_map(jnp.zeros_like, zero_state)
        for chunk_index in range(data.chunks_per_segment):
            batch = data.build_validation_batch(group, chunk_index, include_validation_fields=True)
            valid = np.asarray(batch.valid_mask, bool) & (np.asarray(batch.segment_ids) < selected)
            lanes = np.flatnonzero(valid)
            keys = validation_step_keys(seed=config.seed, segment_id=group,
                                        chunk_index=chunk_index, bptt_steps=config.bptt_steps)

            def score(index, prediction):
                from ..baseline_cache import host_dataset
                truth = batch.truths[index].isel(batch=lanes).astype(np.float32)
                baseline = batch.baselines[index].isel(batch=lanes).astype(np.float32)
                residual = host_dataset(prediction).isel(batch=lanes).astype(np.float32)
                exact._update_original_graphcast_loss_step(index, truth, baseline, baseline + residual)

            loss, state, components, _ = train_step.evaluate(
                params, state, keys, batch.inputs, batch.targets, batch.forcings, valid,
                prediction_callback=score,
            )
            loss, components = jax.device_get((loss, components))
            if not np.isfinite(loss) or not np.all(np.isfinite(components)):
                raise FloatingPointError("Nonfinite cached k1 validation")
            total_loss += float(loss) * len(lanes)
            components_total += np.asarray(components, np.float64) * len(lanes)
            count += len(lanes)
            chunks += len(lanes)
    segment_ids = np.arange(selected, dtype=np.int64)
    return {
        "step": int(step), "loss": total_loss / count,
        "loss_by_horizon": {str(i + 1): float(value / count) for i, value in enumerate(components_total)},
        "original_graphcast_loss": exact._finalize_original_graphcast_loss(),
        "duration_seconds": time.monotonic() - started, "role": "fixed_checkpoint",
        "validation_protocol": BACKEND + "_fixed_teacher_inputs",
        "training_loss_precision": config.precision, "physical_metric_precision": "fp32",
        "component_axis": "sequence_step", "forecast_horizon_steps": 1,
        "available_segments": available, "selected_segments": selected,
        "segment_ids": segment_ids.tolist(), "num_chunks": chunks,
        "num_anchors": count * config.bptt_steps, "bptt_steps": config.bptt_steps,
        "ar_tail_k": 0, "eval_feedback": "baseline", "temporal_state_policy": "carry",
        "subset_policy": "all_complete_segments" if selected == available else "preflight_first_segments",
        "subset_fingerprint": cached.training_data.fingerprint_validation_subset(segment_ids),
        "cache_manifest_sha256": cached.manifest["manifest_sha256"],
    }


def run_cached_k1_training(config_path: Path, producer_config_path: Path, *, cache_root: Path,
                           expected_manifest_sha256: str, epochs: int = 25,
                           max_steps: int | None = None, resume: Path | None = None,
                           shuffle: bool = True, validate_at_stop: bool = False,
                           validation_segments: int | None = None,
                           verify_roundtrip: bool = False, numerical_policy: str = 'strict') -> Path:
    from .cached_config import load_cached_training_config
    from .cached_data import open_cached_training_data
    from .cached_k1_data import CachedK1Cursor, CachedK1Data
    from .cached_k1_step import make_cached_k1_step

    started = time.monotonic()
    if numerical_policy not in ('strict', 'advisory'):
        raise ValueError('Unknown numerical policy')
    runtime = require_k1_runtime()
    config = load_training_config(Path(config_path))
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive bounded stopping point")
    if validation_segments is not None and (type(validation_segments) is not int or validation_segments < 1):
        raise ValueError("validation_segments must be a positive preflight subset")
    stop = min(config.max_steps, max_steps) if max_steps is not None else config.max_steps
    if validation_segments is not None and stop == config.max_steps:
        raise ValueError("Production validation must cover all complete segments")
    if resume is None and config.run_dir.exists() and any(config.run_dir.iterdir()):
        raise FileExistsError(f"Existing cached k1 run requires explicit resume: {config.run_dir}")
    if resume is not None and Path(resume).resolve().parent.parent != config.run_dir.resolve():
        raise ValueError("Resume checkpoint must belong to the configured run directory")
    resolved = load_cached_training_config(Path(producer_config_path), cache_root=Path(cache_root),
                                          expected_manifest_sha256=expected_manifest_sha256)
    # Baseline metadata/parameters may be read on host by producer verification;
    # they are never placed on GPU or bound to a live predictor.
    with jax.default_device(jax.devices("cpu")[0]):
        cached = open_cached_training_data(resolved)
    data = CachedK1Data(cached, batch_size=config.distributed.global_batch_size,
                        seed=config.seed, shuffle=shuffle)
    validate_k1_config(config, cached, data, epochs=epochs)
    provenance = build_k1_provenance(config, Path(producer_config_path), cached, epochs=epochs,
                                      shuffle=shuffle, runtime=runtime)
    if numerical_policy != 'strict':
        provenance['numerical_policy'] = numerical_policy
    _, _, task_config, stats, _, baseline_model_config = cached.context
    model_config = dataclasses.replace(baseline_model_config,
        latent_size=config.architecture.residual_width or config.architecture.width,
        gnn_msg_steps=config.architecture.residual_msg_steps)
    optimizer, learning_rate = build_optimizer(config)
    train_step = make_cached_k1_step(config=config, model_config=model_config, task_config=task_config,
                                     stats=stats, optimizer=optimizer)
    baseline_fingerprint = cached.manifest["compatibility"]["baseline_sha256"]
    anchor_fingerprint = cached.training_data.manifest_fingerprint
    checkpoint_dir = config.run_dir / "checkpoints"
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if resume is None:
        cursor = CachedK1Cursor()
        sample = data.build_batch(cursor)
        rng, init_key = jax.random.split(jax.random.PRNGKey(config.seed))
        params, state = train_step.initialize(init_key, sample.inputs[0], sample.targets[0], sample.forcings[0])
        state = jax.tree_util.tree_map(jnp.zeros_like, cast_state_boundary_fp32(state))
        opt_state = optimizer.init(params)
        completed_step = samples_seen = 0
        del sample
        atomic_json_dump(config.to_dict(), config.run_dir / "run_config.json")
    else:
        checkpoint, backend = load_k1_checkpoint(Path(resume), provenance, data=data)
        cursor = CachedK1Cursor.from_mapping(backend["cursor"])
        params, state, opt_state, rng = (checkpoint.residual_params, checkpoint.residual_state,
                                      checkpoint.optimizer_state, checkpoint.rng_key)
        completed_step, samples_seen = checkpoint.completed_step, backend["samples_seen"]
        if checkpoint.baseline_checkpoint_fingerprint != baseline_fingerprint or checkpoint.anchor_manifest_fingerprint != anchor_fingerprint:
            raise ValueError("Cached k1 checkpoint baseline/anchor identity differs")
        reconcile_run_metrics(config.run_dir, completed_step)
        retained_validation = load_validation_records(config.run_dir / "validation_metrics.jsonl")
        best_path = config.run_dir / "best_validation.json"
        if retained_validation:
            update_best_validation(records=retained_validation,
                checkpoint_for_step=lambda n: checkpoint_dir / f"checkpoint_step{n:08d}.pkl",
                output_path=best_path, subset_fingerprint=retained_validation[-1]["subset_fingerprint"])
        elif best_path.exists():
            # This generated pointer must not survive rollback past its record.
            best_path.unlink()
    # Normalize fresh and resumed trees alike, including optimizer counters and
    # RNG, before any compiled training or validation call sees them.
    params, state, opt_state, rng = jax.device_put((params, state, opt_state, rng), jax.devices()[0])
    zero_state = jax.tree_util.tree_map(jnp.zeros_like, state)
    metrics_path = config.run_dir / "train_metrics.jsonl"
    validation_path = config.run_dir / "validation_metrics.jsonl"
    execution = {"provenance": provenance, "runtime": runtime, "cache_identity": cached.identity,
                     "numerical_policy": numerical_policy,
                     "epochs": epochs, "steps_per_epoch": data.steps_per_epoch,
                     "train_segments": data.num_segments, "batch_size": data.batch_size,
                     "baseline_initializations": 0, "baseline_apply_calls": 0,
                     "stop_step": stop, "startup_seconds": time.monotonic() - started}
    atomic_json_dump(execution, config.run_dir / "cached_k1_execution.json")

    def save(step):
        path = write_k1_checkpoint(checkpoint_dir / f"checkpoint_step{step:08d}.pkl", config=config,
            completed_step=step, params=params, state=state, optimizer_state=opt_state, rng=rng,
            cursor=cursor, samples_seen=samples_seen, baseline_fingerprint=baseline_fingerprint,
            anchor_fingerprint=anchor_fingerprint, provenance=provenance)
        if verify_roundtrip:
            loaded, backend = load_k1_checkpoint(path, provenance, data=data)
            fields = exact_saved_state(loaded, {'residual_params': params, 'residual_state': state,
                'optimizer_state': opt_state, 'rng_key': rng})
            if (loaded.resolved_training_config != config.to_dict()
                    or loaded.training_cursor != cursor.to_dict() or backend['samples_seen'] != samples_seen
                    or loaded.baseline_checkpoint_fingerprint != baseline_fingerprint
                    or loaded.anchor_manifest_fingerprint != anchor_fingerprint):
                raise ValueError('Cached k1 exact checkpoint metadata differs')
            atomic_json_dump({'integrity_passed': True, 'passed': True, 'fields': fields,
                              'checkpoint': str(path), 'completed_step': step,
                              'kind': 'exact_save_load'}, path.with_suffix('.serialization.json'))
        return path

    def validate(step):
        if any(int(row["step"]) == step for row in load_validation_records(validation_path)):
            return
        record = run_k1_validation(train_step=train_step, data=data, cached=cached, config=config,
            params=params, zero_state=zero_state, step=step, max_segments=validation_segments)
        append_validation_record(validation_path, record)
        update_best_validation(records=load_validation_records(validation_path),
            checkpoint_for_step=lambda n: checkpoint_dir / f"checkpoint_step{n:08d}.pkl",
            output_path=config.run_dir / "best_validation.json", subset_fingerprint=record["subset_fingerprint"])
        print(f"[cached_k1] validation step={step} loss={record['loss']:.8f} segments={record['selected_segments']}", flush=True)

    last_path = Path(resume) if resume is not None else save(0)
    if completed_step and (completed_step % data.steps_per_epoch == 0 or (validate_at_stop and completed_step == stop)):
        validate(completed_step)
    for step in range(completed_step + 1, stop + 1):
        update_started = time.monotonic()
        batch = data.build_batch(cursor)
        loaded = time.monotonic()
        if batch.reset_state:
            state = jax.tree_util.tree_map(jnp.zeros_like, zero_state)
        consumed_cursor = cursor
        rng, *keys = jax.random.split(rng, config.bptt_steps + 1)
        params, state, opt_state, loss, grad_norm, components = train_step(
            params, state, opt_state, jnp.stack(keys), batch.inputs, batch.targets, batch.forcings, batch.valid_mask)
        loss, grad_norm, components = jax.device_get((loss, grad_norm, components))
        ended = time.monotonic()
        if not np.isfinite(loss) or not np.isfinite(grad_norm) or not np.all(np.isfinite(components)):
            raise FloatingPointError(f"Nonfinite cached k1 update at step {step}")
        cursor = batch.next_cursor
        samples_seen += batch.valid_weather_steps
        if samples_seen != samples_before_cursor(data, cursor):
            raise ValueError("Sampler and true sample accounting diverged")
        _append(metrics_path, {"step": step, "loss": float(loss), "gradient_norm": float(grad_norm),
            "learning_rate": float(jax.device_get(learning_rate(step - 1))) if callable(learning_rate) else float(learning_rate),
            "step_seconds": ended - loaded, "end_to_end_seconds": ended - update_started,
            "data_wait_seconds": loaded - update_started, "batch_size": data.batch_size,
            "valid_lanes": int(np.asarray(batch.valid_mask).sum()), "samples_seen": samples_seen,
            "anchors_seen": samples_seen, "valid_weather_steps": batch.valid_weather_steps,
            "component_axis": "sequence_step", "forecast_horizon_steps": 1,
            "segment_ids": np.asarray(batch.segment_ids).tolist(), **consumed_cursor.to_dict(),
            "loss_by_horizon": {str(i + 1): float(value) for i, value in enumerate(components)}})
        if step <= 5 or step % 10 == 0:
            print(f"[cached_k1] step {step}/{config.max_steps} loss={float(loss):.8f} samples={samples_seen} step_seconds={ended-loaded:.3f}", flush=True)
        epoch_end = cursor.epoch != consumed_cursor.epoch
        if epoch_end or step == stop:
            checkpoint_started = time.monotonic()
            last_path = save(step)
            _append(config.run_dir / "checkpoint_metrics.jsonl", {"step": step,
                "checkpoint_seconds": time.monotonic() - checkpoint_started, "checkpoint": str(last_path)})
        if epoch_end or (step == stop and validate_at_stop):
            validate(step)
        del batch
    if verify_roundtrip:
        loaded_checkpoint, backend = load_k1_checkpoint(last_path, provenance, data=data)
        expected = {"residual_params": params, "residual_state": state, "optimizer_state": opt_state, "rng_key": rng}
        fields = exact_saved_state(loaded_checkpoint, expected)
        probe_batch = data.build_batch(cursor)
        prediction_check = prediction_roundtrip_check(train_step=train_step, checkpoint=loaded_checkpoint,
            params=params, state=state, rng=rng, batch=probe_batch)
        report = roundtrip_result(fields, prediction_check, numerical_policy=numerical_policy,
            checkpoint=str(last_path), provenance=provenance, cursor=backend['cursor'], samples_seen=backend['samples_seen'])
        del probe_batch
        atomic_json_dump(report, last_path.with_suffix('.roundtrip.json'))
        atomic_json_dump(report, config.run_dir / "checkpoint_roundtrip.json")
        if not report["passed"]:
            raise ValueError("Cached k1 numerical checkpoint roundtrip failed")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()] if metrics_path.exists() else []
    steady = rows[1:] or rows
    timings = [row["end_to_end_seconds"] for row in steady]
    validations = load_validation_records(validation_path)
    execution.update({"completed_steps": max(completed_step, stop), "samples_seen": samples_seen,
        "checkpoint": str(last_path), "peak_host_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
        "elapsed_seconds": time.monotonic() - started,
        "p50_end_to_end_seconds": float(np.median(timings)) if timings else None,
        "p90_end_to_end_seconds": float(np.percentile(timings, 90)) if timings else None,
        "validation_seconds": max((float(row.get("duration_seconds", 0)) for row in validations), default=None)})
    atomic_json_dump(execution, config.run_dir / "cached_k1_execution.json")
    return last_path
