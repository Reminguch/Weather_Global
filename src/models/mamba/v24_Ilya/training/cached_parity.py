"""Separate-process correctness gates and measurements for cached stepwise training.

Run on an allocated GPU with ``--experiment-root``. No production job is
authorized by this module unless every recorded numerical comparison passes.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

from .cached_diagnostics import (
    capture_gradients, gradient_comparison, memory_snapshot,
    require_matched_gpu_environment, timing_summary, tree_comparison, tree_digest,
)


def _json(path, value):
    from ..checkpoint import atomic_json_dump
    atomic_json_dump(value, Path(path))


def _save(path, value):
    import jax
    from ..checkpoint import atomic_pickle_dump
    atomic_pickle_dump(jax.device_get(value), Path(path))


def _read(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _manifest(root):
    return json.loads((Path(root) / "manifest.json").read_text())


def _context(root, backend="cached"):
    import jax
    from ..checkpoint import load_v24_Ilya_training_checkpoint
    from ..model import build_model_configs
    from .cached_config import load_cached_training_config
    from .cached_data import open_cached_training_data
    from .cached_stepwise import make_cached_stepwise_train_step
    from .endpoint_step import build_optimizer
    from .config import load_training_config
    require_matched_gpu_environment()
    manifest = _manifest(root)
    resolved = load_cached_training_config(
        Path(manifest["configs"]["cached"]), cache_root=Path(manifest["cache_root"]),
        prefetch_batches=int(manifest["execution"].get("prefetch_batches", 0)),
        expected_manifest_sha256=manifest["execution"].get("expected_manifest_sha256"),
    )
    cached = open_cached_training_data(resolved)
    config = resolved.common
    local_config = load_training_config(Path(manifest["configs"]["smoke_" + backend]))
    local_initial = local_config.run_dir / "checkpoints/checkpoint_step00000000.pkl"
    checkpoint = load_v24_Ilya_training_checkpoint(
        local_initial if local_initial.exists() else Path(manifest["shared_init"]))
    _, baseline_checkpoint, task, stats, data, _ = cached.context
    models = build_model_configs(baseline_checkpoint.model_config, config.architecture)
    optimizer, _ = build_optimizer(config)
    step = make_cached_stepwise_train_step(
        config=config, model_config=models.residual, task_config=task, stats=stats,
        optimizer=optimizer,
    )
    return manifest, resolved, cached, checkpoint, models, optimizer, step


def _online(context, *, record_gradients=False):
    import jax
    import jax.numpy as jnp
    from ..checkpoint import overlay_frozen_baseline_params
    from .data import TrainingCursor
    from .endpoint_step import build_training_transforms, make_train_step
    manifest, resolved, cached, checkpoint, models, optimizer, _ = context
    config, official, task, stats, data, _ = cached.context
    transforms = build_training_transforms(models, task, stats, resolved.common)
    chunk = data.build_chunk(TrainingCursor(), resolved.common, task)
    from .endpoint_step import input_window_from_frames
    inputs = input_window_from_frames(chunk.input_frames[0], chunk.input_frames[1], chunk.static_inputs,
                                      step_index=0, truth_prefix_steps=config.truth_prefix_steps,
                                      time_step=data.time_step)
    baseline_params, baseline_state = transforms.baseline_predict.init(
        jax.random.PRNGKey(config.seed), inputs, chunk.truths[0], chunk.forcings[0])
    baseline_params, _ = overlay_frozen_baseline_params(baseline_params, official.params)
    if record_gradients:
        optimizer = capture_gradients(optimizer)
    step = make_train_step(transforms=transforms, optimizer=optimizer,
                           baseline_params=baseline_params, baseline_state=baseline_state,
                           config=resolved.common, time_step=data.time_step, input_steps=data.input_steps)

    @jax.jit
    def baseline(key, inputs, truth, forcing):
        prediction, _ = transforms.baseline_predict.apply(
            baseline_params, baseline_state, key,
            jax.tree_util.tree_map(jax.lax.stop_gradient, inputs), truth, forcing)
        return jax.tree_util.tree_map(jax.lax.stop_gradient, prediction)

    return step, baseline, _diagnostic_forward(transforms, resolved.common)


def _diagnostic_forward(transforms, config):
    """Independent one-step forward with the maintained state-boundary policy."""
    import jax
    import jax.numpy as jnp
    from .endpoint_step import cast_state_boundary_fp32, scalarize_loss
    reset = config.temporal_state_policy == "reset_every_anchor"

    @jax.jit
    def forward(params, state, key, inputs, target, forcing):
        state = cast_state_boundary_fp32(state)
        if reset:
            state = jax.tree_util.tree_map(jnp.zeros_like, state)
        ((loss, _), prediction), next_state = transforms.residual_loss_and_predictions.apply(
            params, state, key, inputs, target, forcing)
        if reset:
            next_state = state
        return scalarize_loss(loss), prediction, cast_state_boundary_fp32(next_state)

    return forward


def _keys(rng, count):
    import jax
    import jax.numpy as jnp
    retained, *keys = jax.random.split(rng, count + 1)
    return retained, jnp.stack(keys)


def _nonzero(params, state):
    import jax
    rng = np.random.default_rng(220914)
    params = jax.tree_util.tree_map(lambda value: np.array(value, copy=True), params)
    changed = []
    for module, leaves in params.items():
        if "temporal_residual_head" in module or ("temporal" in module and "out_proj" in module):
            for name, value in leaves.items():
                scale = 0.02 / np.sqrt(value.shape[0]) if value.ndim > 1 else 0.002
                leaves[name] = rng.normal(0, scale, value.shape).astype(np.float32)
                changed.append(module + "/" + name)
    if not any("temporal_residual_head" in name for name in changed) or len(changed) < 3:
        raise AssertionError("Nontrivial gate failed to find residual and temporal output projections")
    def perturb(value):
        import jax.numpy as jnp
        result = rng.normal(0, 0.01, value.shape).astype(np.float32)
        # Incoming memory represents the legacy rounded recurrent boundary.
        return np.asarray(jnp.asarray(result).astype(jnp.bfloat16).astype(jnp.float32))
    return params, jax.tree_util.tree_map(perturb, state), changed


def _forbid_gc():
    from unittest.mock import patch
    stack = contextlib.ExitStack()
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached worker attempted to construct/execute frozen GraphCast")
    for location in ("src.models.mamba.v24_Ilya.model.make_baseline_predictor",
                     "src.models.mamba.v24_Ilya.training.endpoint_step.make_baseline_predictor",
                     "src.models.mamba.v24_Ilya.baseline_cache.make_predictor"):
        stack.enter_context(patch(location, forbidden))
    return stack


def _trajectory(context, cursor, keys, baseline, *, split="train"):
    """Independent live-GC trajectory using the maintained physical helpers."""
    import jax
    from ..baseline_cache import host_dataset
    from .endpoint_step import input_window_from_frames, next_dynamic_frame, residual_target
    _, resolved, cached, _, _, _, _ = context
    config = resolved.common
    _, _, task, _, data, _ = cached.context
    chunk = (data.build_chunk(cursor, config, task) if split == "train" else
             data.load_segment_chunk(data.validation_segments[cursor.segment_index], cursor.segment_offset, config, task))
    frames = list(chunk.input_frames)
    windows, targets, forcings = [], [], []
    for index in range(config.bptt_steps):
        inputs = input_window_from_frames(frames[index], frames[index + 1], chunk.static_inputs,
                                          step_index=index, truth_prefix_steps=config.truth_prefix_steps,
                                          time_step=data.time_step)
        forcing = chunk.forcings[index]
        prediction = baseline(keys[index], inputs, chunk.truths[index], forcing)
        windows.append(host_dataset(inputs))
        targets.append(host_dataset(residual_target(chunk.truths[index], prediction)))
        forcings.append(host_dataset(forcing))
        if config.truth_prefix_steps - 1 <= index < config.bptt_steps - 1:
            frames.append(host_dataset(next_dynamic_frame(inputs, prediction, forcing, time_step=data.time_step)))
    return tuple(windows), tuple(targets), tuple(forcings)


def _data_signature(values, config):
    return {"windows": [tree_digest(item) for item in values[0]],
            "targets": [tree_digest(item) for item in values[1]],
            "forcings": [tree_digest(item) for item in values[2]],
            "weights": tree_digest(np.asarray(config.normalized_supervised_weights, np.float32))}


def _full_worker(root, backend):
    import jax
    import jax.numpy as jnp
    import optax
    from .data import TrainingCursor
    guard = contextlib.nullcontext() if backend == "online" else _forbid_gc()
    context = _context(root, backend)
    manifest, resolved, cached, checkpoint, _, optimizer, cached_step = context
    config = resolved.common
    output = Path(root) / "reports/workers" / ("full_" + backend)
    output.mkdir(parents=True, exist_ok=True)
    online = _online(context, record_gradients=True) if backend == "online" else None
    _, keys = _keys(checkpoint.rng_key, config.bptt_steps)
    with guard:
        # Cover a complete first segment, its reset, and the last train/val chunks.
        signatures = {}
        offsets = range(0, config.segment_steps, config.bptt_steps)
        last_offset = offsets[-1]
        cursors = [TrainingCursor(segment_offset=offset) for offset in offsets]
        if len(cached.training_data.segments) > 1:
            cursors += [TrainingCursor(segment_index=1),
                        TrainingCursor(segment_index=len(cached.training_data.segments) - 1,
                                       segment_offset=last_offset)]
        first_values = None
        for index, cursor in enumerate(cursors):
            if online:
                values = _trajectory(context, cursor, keys, online[1])
            else:
                batch = cached.build_chunk(cursor)
                values = batch.inputs, batch.targets, batch.forcings
            signatures["train:" + str(cursor.to_dict())] = _data_signature(values, config)
            if index == 0:
                first_values = values
            else:
                del values
        for segment_id, offset in ((0, 0), (len(cached.training_data.validation_segments) - 1, last_offset)):
            cursor = TrainingCursor(segment_index=segment_id, segment_offset=offset)
            if online:
                values = _trajectory(context, cursor, keys, online[1], split="val")
            else:
                batch = cached.load_validation_chunk(segment_id, offset, include_validation_fields=False)
                values = batch.inputs, batch.targets, batch.forcings
            signatures["val:" + str(cursor.to_dict())] = _data_signature(values, config)
            del values
        _json(output / "data_identity.json", signatures)
        for case in ("initial", "nonzero"):
            params, state = checkpoint.residual_params, checkpoint.residual_state
            changed = []
            if case == "nonzero":
                params, state, changed = _nonzero(params, state)
            assert first_values is not None
            inputs, targets, forcings = first_values
            if online:
                predictions, components = [], []
                evaluation_state = state
                for i in range(config.bptt_steps):
                    step_loss, prediction, evaluation_state = online[2](params, evaluation_state, keys[i],
                                                                       inputs[i], targets[i], forcings[i])
                    predictions.append(jax.device_get(prediction))
                    components.append(float(step_loss))
                chunk = cached.training_data.build_chunk(TrainingCursor(), config, cached.context[2])
                captured_state = (checkpoint.optimizer_state, jax.tree_util.tree_map(jnp.zeros_like, params))
                new_params, new_state, new_opt, loss, grad_norm, step_losses = online[0](
                    params, state, captured_state, keys, chunk.input_frames, chunk.static_inputs,
                    chunk.truths, chunk.forcings)
                opt_state, gradients = new_opt
                forward_check = tree_comparison(evaluation_state, new_state)
                if not forward_check["passed"]:
                    raise AssertionError("Online diagnostic forward differs from maintained training state")
            else:
                loss, new_state, step_losses, predictions, gradients = cached_step.loss_and_grad(
                    params, state, keys, inputs, targets, forcings)
                updates, opt_state = optimizer.update(gradients, checkpoint.optimizer_state, params)
                new_params = optax.apply_updates(params, updates)
                grad_norm = optax.global_norm(gradients)
            payload = {"loss": np.asarray(loss), "step_losses": np.asarray(step_losses),
                       "predictions": tuple(predictions), "state": new_state, "gradients": gradients,
                       "params": new_params, "optimizer": opt_state, "grad_norm": np.asarray(grad_norm)}
            _save(output / (case + ".pkl"), payload)
            _json(output / (case + ".json"), {"changed_projections": changed, "loss": float(loss),
                                              "memory": memory_snapshot()})
            print(f"[cached parity] full {backend} {case} loss={float(loss):.9f}", flush=True)
            del payload, gradients, predictions, new_state, new_params, opt_state
    _json(output / "completed.json", {"backend": backend, "cases": ["initial", "nonzero"],
                                      "memory": memory_snapshot()})


def _native_checkpoint(context, output, step, params, state, opt_state, rng, cursor):
    from ..checkpoint import training_checkpoint_payload, atomic_pickle_dump
    _, resolved, _, source, _, _, _ = context
    payload = training_checkpoint_payload(
        completed_step=step, residual_params=params, residual_state=state,
        optimizer_state=opt_state, rng_key=rng, training_cursor=cursor.to_dict(),
        resolved_training_config=resolved.common.to_dict(),
        baseline_checkpoint_path=str(resolved.common.baseline_checkpoint),
        baseline_checkpoint_fingerprint=source.baseline_checkpoint_fingerprint,
        anchor_manifest_fingerprint=source.anchor_manifest_fingerprint,
        parameter_overlay_metadata=source.parameter_overlay_metadata)
    path = output / f"checkpoint_step{step:08d}.pkl"
    atomic_pickle_dump(payload, path)
    return path


def _continuation_worker(root, backend, *, resume=False, cpus=None, prefetch=0,
                         preflight=False, resource_check=False):
    import jax
    import jax.numpy as jnp
    from ..checkpoint import load_v24_Ilya_training_checkpoint
    from .data import TrainingCursor
    from .cached_validation import run_cached_validation
    started = time.monotonic()
    guard = contextlib.nullcontext() if backend == "online" else _forbid_gc()
    context = _context(root, backend)
    manifest, resolved, cached, checkpoint, _, optimizer, cached_step = context
    config = resolved.common
    label = backend + ("_resume" if resume else "") + (f"_{cpus}cpu" if cpus else "") + ("_prefetch" if prefetch else "")
    label += "_preflight" if preflight else "_resource" if resource_check else ""
    output = Path(root) / "reports/workers" / label
    output.mkdir(parents=True, exist_ok=True)
    online = _online(context) if backend == "online" else None
    zero_state = jax.tree_util.tree_map(jnp.zeros_like, checkpoint.residual_state)
    if resume:
        checkpoint = load_v24_Ilya_training_checkpoint(
            Path(root) / "reports/workers/cached/checkpoint_step00000002.pkl")
    params, state, opt_state, rng = (checkpoint.residual_params, checkpoint.residual_state,
                                   checkpoint.optimizer_state, checkpoint.rng_key)
    cursor = TrainingCursor.from_mapping(checkpoint.training_cursor)
    records = []
    initial_identity = {name: tree_digest(value) for name, value in
                        (("params", params), ("state", state), ("optimizer", opt_state), ("rng", rng))}
    initial_identity["cursor"] = cursor.to_dict()
    _json(output / "initial_identity.json", initial_identity)
    setup_seconds = time.monotonic() - started
    limit = 3 if preflight else 20 if resume else 70
    iterator = None if online else cached.iter_chunks(cursor, prefetch_batches=prefetch,
                                                      max_chunks=limit - checkpoint.completed_step)
    checkpoint_seconds = []
    with guard:
        try:
            for step in range(checkpoint.completed_step + 1, limit + 1):
                start = time.monotonic()
                reset = cursor.segment_offset == 0
                if reset:
                    state = zero_state
                if online:
                    batch = cached.training_data.build_chunk(cursor, config, cached.context[2])
                else:
                    batch = next(iterator)
                rng, keys = _keys(rng, config.bptt_steps)
                loaded = time.monotonic()
                if online:
                    result = online[0](params, state, opt_state, keys, batch.input_frames,
                                       batch.static_inputs, batch.truths, batch.forcings)
                else:
                    result = cached_step(params, state, opt_state, keys, batch.inputs, batch.targets, batch.forcings)
                params, state, opt_state, loss, grad_norm, components = result
                jax.block_until_ready(result)
                ended = time.monotonic()
                consumed = cursor
                cursor = batch.next_cursor
                record = {"step": step, "loss": float(loss), "grad_norm": float(grad_norm),
                          "components": np.asarray(components).tolist(), "cursor": consumed.to_dict(),
                          "next_cursor": cursor.to_dict(), "reset_state": reset,
                          "compute_seconds": ended - loaded, "end_to_end_seconds": ended - start,
                          "load_seconds": loaded - start, "memory": memory_snapshot()}
                records.append(record)
                _json(output / "updates.json", records)
                if step in (2, 20):
                    writing = time.monotonic()
                    _native_checkpoint(context, output, step, params, state, opt_state, rng, cursor)
                    checkpoint_seconds.append(time.monotonic() - writing)
                print(f"[cached parity] {label} step={step} loss={float(loss):.8f} seconds={ended-start:.3f}", flush=True)
                del batch, result
        finally:
            if iterator is not None:
                iterator.close()
        if preflight:
            summary = {"backend": backend, "completed_steps": limit,
                       "startup_seconds": setup_seconds + records[0]["end_to_end_seconds"],
                       "p90_end_to_end_seconds": float(np.percentile([r["end_to_end_seconds"] for r in records[1:]], 90)),
                       "peak_host_gib": memory_snapshot()["host_peak_gib"],
                       "memory": memory_snapshot(), "total_seconds": time.monotonic() - started}
            _json(output / "preflight.json", summary)
        elif not resume:
            validation = run_cached_validation(train_step=cached_step, cached_data=cached, config=config,
                                                params=params, zero_state=zero_state, step=limit)
            _json(output / "validation.json", validation)
            summary = timing_summary(records, weather_steps=config.bptt_steps)
            summary.update(setup_seconds=setup_seconds, first_update_seconds=records[0]["end_to_end_seconds"],
                           checkpoint_seconds=checkpoint_seconds, validation_seconds=validation["duration_seconds"],
                           memory=memory_snapshot(), backend=backend, cpu_limit=cpus,
                           prefetch_batches=prefetch, total_seconds=time.monotonic() - started)
            _json(output / "benchmark.json", summary)
    _json(output / "completed.json", {"backend": backend, "completed_steps": limit,
                                      "cursor": cursor.to_dict(), "memory": memory_snapshot()})


def _worker_command(root, *, worker, backend, cpus=None, prefetch=0):
    command = [sys.executable, "-m", "src.models.mamba.v24_Ilya.training.cached_parity", "--experiment-root", str(root),
               "--worker", worker, "--backend", backend]
    if cpus:
        command += ["--cpus", str(cpus)]
        available = sorted(os.sched_getaffinity(0))
        if len(available) < cpus or shutil.which("taskset") is None:
            raise RuntimeError("CPU-limited workers require taskset and sufficient allocated cores")
        command = ["taskset", "--cpu-list", ",".join(map(str, available[:cpus])), *command]
    if prefetch:
        command += ["--prefetch", str(prefetch)]
    return command


def _run_worker(root, **kwargs):
    label = "_".join(str(value) for value in kwargs.values())
    log = Path(root) / "reports" / ("worker_" + label + ".log")
    command = _worker_command(root, **kwargs)
    print("[cached parity] starting " + " ".join(command), flush=True)
    environment = os.environ.copy()
    if kwargs.get("cpus"):
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = str(kwargs["cpus"])
    with log.open("w") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True, env=environment)


def _provenance(root):
    from .cached_checkpoint import build_provenance
    from .cached_config import load_cached_training_config
    from .cached_runner import execution_metadata
    manifest = _manifest(root)
    resolved = load_cached_training_config(Path(manifest["configs"]["cached"]),
                                            cache_root=Path(manifest["cache_root"]), prefetch_batches=0,
                                            expected_manifest_sha256=manifest["execution"].get("expected_manifest_sha256"))
    return build_provenance(resolved.common, Path(manifest["shared_init"]),
                            Path(manifest["cache_root"]), execution=execution_metadata(resolved.execution))


def run_preflight(root):
    root = Path(root)
    (root / "reports").mkdir(exist_ok=True)
    for backend in ("online", "cached"):
        _run_worker(root, worker="preflight", backend=backend)
    summaries = {backend: json.loads((root / "reports/workers" / (backend + "_preflight") / "preflight.json").read_text())
                 for backend in ("online", "cached")}
    online, cached = summaries["online"], summaries["cached"]
    # Two online compilations, four cached compilations; bound all forward-only
    # validation chunks by a complete cached forward+backward update initially.
    estimate = 1.35 * (2 * online["startup_seconds"] + 4 * cached["startup_seconds"]
                       + 90 * online["p90_end_to_end_seconds"] + 370 * cached["p90_end_to_end_seconds"])
    budget = {"memory_gib": max(128, math.ceil(1.4 * max(v["peak_host_gib"] for v in summaries.values()) / 16) * 16),
              "time_minutes": max(120, math.ceil(estimate / 1800) * 30),
              "cpus": 8, "gpu": "gpu40&nomig"}
    _json(root / "reports/preflight.json", {"passed": True, "provenance": _provenance(root),
                                            "backends": summaries, "gates_budget": budget})


def run_resource_check(root):
    root = Path(root)
    _run_worker(root, worker="resource", backend="cached", cpus=4)
    workers = root / "reports/workers"
    result = _checkpoint_compare(workers / "cached_4cpu_resource/checkpoint_step00000020.pkl",
                                  workers / "cached/checkpoint_step00000020.pkl", exact=True)
    benchmark = json.loads((workers / "cached_4cpu_resource/benchmark.json").read_text())
    validation = json.loads((workers / "cached_4cpu_resource/validation.json").read_text())
    coverage = _validation_coverage(root, validation)
    full_validation = coverage["passed"]
    report = {"passed": result["passed"] and full_validation, "provenance": _provenance(root), "checks": result,
              "allocated_memory_mib": int(os.environ.get("SLURM_MEM_PER_NODE", 0)),
              "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", 0)),
              "completed_steps": 70,
              "full_validation_completed": full_validation,
              "validation_coverage": coverage,
              "allocation": {"memory_gib": int(os.environ.get("SLURM_MEM_PER_NODE", 0)) / 1024,
                             "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", 0))},
              "benchmark": benchmark}
    _json(root / "reports/resource_check.json", report)
    if not report["passed"]:
        raise RuntimeError("Reduced-resource replay differs from the accepted cached trajectory")


def _validation_coverage(root, validation):
    """Require every cached validation segment, without a fixed dataset size."""
    from .config import load_training_config
    manifest = _manifest(root)
    config = load_training_config(Path(manifest["configs"]["cached"]))
    cache_manifest = json.loads((Path(manifest["cache_root"]) / "manifest.json").read_text())
    chunks = [item for item in cache_manifest["chunks"] if item["split"] == "val"]
    segment_ids = sorted({item["segment_id"] for item in chunks})
    expected = {(segment_id, offset) for segment_id in segment_ids
                for offset in range(0, config.segment_steps, config.bptt_steps)}
    actual = {(item["segment_id"], item["segment_offset"]) for item in chunks}
    complete_cache = bool(segment_ids) and actual == expected and len(chunks) == len(expected)
    checks = {
        "cache_complete": complete_cache,
        "segments": validation.get("selected_segments") == len(segment_ids),
        "available_segments": validation.get("available_segments") == len(segment_ids),
        "segment_ids": validation.get("segment_ids") == segment_ids,
        "chunks": validation.get("num_chunks") == len(expected),
        "anchors": validation.get("num_anchors") == len(expected) * config.bptt_steps,
        "chunk_length": validation.get("bptt_steps") == config.bptt_steps,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "expected_segments": len(segment_ids), "expected_chunks": len(expected)}


def compare_full_outputs(root):
    reports = {}
    directory = Path(root) / "reports/workers"
    data_a = json.loads((directory / "full_online/data_identity.json").read_text())
    data_b = json.loads((directory / "full_cached/data_identity.json").read_text())
    reports["data_identity"] = {"passed": data_a == data_b, "checked_chunks": len(data_a)}
    for case in ("initial", "nonzero"):
        expected = _read(directory / "full_online" / (case + ".pkl"))
        actual = _read(directory / "full_cached" / (case + ".pkl"))
        checks = {key: tree_comparison(actual[key], expected[key], relative=1e-4 if key in ("loss", "step_losses") else 1e-3)
                  for key in ("loss", "step_losses", "predictions", "state", "params", "optimizer", "grad_norm")}
        checks["gradients"] = gradient_comparison(actual["gradients"], expected["gradients"])
        reports[case] = {"passed": all(check["passed"] for check in checks.values()), "checks": checks}
        del expected, actual
    return {"passed": all(report["passed"] for report in reports.values()), "checks": reports}


def _checkpoint_compare(left, right, *, exact=False):
    from ..checkpoint import load_v24_Ilya_training_checkpoint
    a = load_v24_Ilya_training_checkpoint(left)
    b = load_v24_Ilya_training_checkpoint(right)
    checks = {name: tree_comparison(getattr(a, name), getattr(b, name))
              for name in ("residual_params", "residual_state", "optimizer_state")}
    if exact:
        for name in ("residual_params", "residual_state", "optimizer_state"):
            checks[name]["exact_hash_match"] = tree_digest(getattr(a, name)) == tree_digest(getattr(b, name))
            checks[name]["passed"] = checks[name]["passed"] and checks[name]["exact_hash_match"]
    checks["rng"] = {"passed": tree_digest(a.rng_key) == tree_digest(b.rng_key)}
    checks["cursor"] = {"passed": a.training_cursor == b.training_cursor}
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def _initialization_checks(source, config):
    """Verify the requested spatial initialization and zero-output contract."""
    import jax
    metadata = source.parameter_overlay_metadata
    policy = config.architecture.residual_initialization
    copied, fresh = metadata.get("residual_copied"), metadata.get("residual_fresh")
    parameter_leaves = len(jax.tree_util.tree_leaves(source.residual_params))
    counts_match = (type(copied) is int and type(fresh) is int
                    and copied >= 0 and fresh >= 0 and copied + fresh <= parameter_leaves)
    policy_matches = metadata.get("residual_initialization") == policy and counts_match
    if policy == "fresh":
        policy_matches = policy_matches and copied == 0 and fresh == parameter_leaves and fresh > 0
    elif policy == "baseline_overlay":
        policy_matches = policy_matches and copied > 0
    else:
        policy_matches = False
    output_heads = [value for module, values in source.residual_params.items()
                    if "temporal_residual_head" in module or ("temporal" in module and "out_proj" in module)
                    for value in values.values()]
    checks = {
        "zero_recurrent_state": all(np.count_nonzero(value) == 0
                                    for value in jax.tree_util.tree_leaves(source.residual_state)),
        "zero_output_heads": bool(output_heads) and all(np.count_nonzero(value) == 0 for value in output_heads),
        "spatial_initialization": bool(policy_matches),
        "frozen_baseline_overlay": (metadata.get("baseline_copied", 0) > 0
                                    and metadata.get("baseline_fresh") == 0),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "residual_initialization": policy, "residual_copied": copied,
            "residual_fresh": fresh, "parameter_leaves": parameter_leaves}


def run_gates(root):
    from .cached_checkpoint import build_provenance, project_shared_initialization
    from .cached_config import load_cached_training_config
    from .cached_runner import execution_metadata
    from .config import load_training_config
    from ..checkpoint import load_v24_Ilya_training_checkpoint
    root = Path(root).resolve()
    manifest = _manifest(root)
    (root / "reports").mkdir(exist_ok=True)
    resolved = load_cached_training_config(Path(manifest["configs"]["cached"]),
                                            cache_root=Path(manifest["cache_root"]),
                                            prefetch_batches=0,
                                            expected_manifest_sha256=manifest["execution"].get("expected_manifest_sha256"))
    provenance = build_provenance(resolved.common, Path(manifest["shared_init"]),
                                  Path(manifest["cache_root"]),
                                  execution=execution_metadata(resolved.execution))
    source = load_v24_Ilya_training_checkpoint(Path(manifest["shared_init"]))
    state_fields = ("residual_params", "residual_state", "optimizer_state", "rng_key")
    source_hashes = {name: tree_digest(getattr(source, name)) for name in state_fields}
    source_hashes["cursor"] = source.training_cursor
    projected = {}
    for backend in ("online", "cached"):
        path = project_shared_initialization(Path(manifest["shared_init"]),
                                             load_training_config(Path(manifest["configs"]["smoke_" + backend])),
                                             provenance=provenance, execution=execution_metadata(resolved.execution))
        saved = load_v24_Ilya_training_checkpoint(path)
        hashes = {name: tree_digest(getattr(saved, name)) for name in state_fields}
        hashes["cursor"] = saved.training_cursor
        projected[backend] = {"passed": hashes == source_hashes, "hashes": hashes, "path": str(path)}
    initial_report = {"passed": all(value["passed"] for value in projected.values()),
                      "canonical": source_hashes, "projections": projected}
    initialization = _initialization_checks(source, resolved.common)
    initial_report.update(initialization["checks"])
    initial_report["initialization_policy"] = initialization
    initial_report["passed"] = initial_report["passed"] and initialization["passed"]
    _json(root / "reports/initialization_identity.json", initial_report)
    if not initial_report["passed"]:
        raise RuntimeError("Shared initialization projection differs")
    del saved, source
    for backend in ("online", "cached"):
        _run_worker(root, worker="full", backend=backend)
    full = compare_full_outputs(root)
    full["checks"]["initialization_identity"] = initial_report
    full.update(kind="full_parity", provenance=provenance)
    _json(root / "reports/full_parity.json", full)
    if not full["passed"]:
        raise RuntimeError("Full parity gate failed; production remains disabled")
    for backend in ("online", "cached"):
        _run_worker(root, worker="continue", backend=backend)
    _run_worker(root, worker="resume", backend="cached")
    workers = root / "reports/workers"
    online_records = json.loads((workers / "online/updates.json").read_text())[:20]
    cached_records = json.loads((workers / "cached/updates.json").read_text())[:20]
    initial_a = json.loads((workers / "online/initial_identity.json").read_text())
    initial_b = json.loads((workers / "cached/initial_identity.json").read_text())
    paired = {"initialization": {"passed": initial_a == initial_b, "online": initial_a, "cached": initial_b},
              "losses": tree_comparison(np.array([r["loss"] for r in cached_records]),
                                         np.array([r["loss"] for r in online_records]), relative=1e-4),
              "step20_state": _checkpoint_compare(workers / "cached/checkpoint_step00000020.pkl",
                                                   workers / "online/checkpoint_step00000020.pkl"),
              "resume": _checkpoint_compare(workers / "cached_resume/checkpoint_step00000020.pkl",
                                              workers / "cached/checkpoint_step00000020.pkl", exact=True),
              "boundaries": {"passed": all(a["cursor"] == b["cursor"] and a["reset_state"] == b["reset_state"]
                                               for a, b in zip(online_records, cached_records, strict=True))}}
    report = {"kind": "paired20", "completed_steps": 20, "provenance": provenance,
              "passed": all(value["passed"] for value in paired.values()), "checks": paired}
    _json(root / "reports/paired20.json", report)
    if not report["passed"]:
        raise RuntimeError("Paired continuation/resume gate failed; production remains disabled")
    # Four-core verification uses the same GPU, numerical policy and examples.
    _run_worker(root, worker="continue", backend="cached", cpus=4)
    benchmarks = {name: json.loads((workers / name / "benchmark.json").read_text())
                  for name in ("online", "cached", "cached_4cpu")}
    for summary in benchmarks.values():
        summary.update(peak_host_gib=summary["memory"]["host_peak_gib"],
                       p90_end_to_end_seconds=summary["end_to_end_p90_seconds"],
                       startup_seconds=summary["setup_seconds"] + summary["first_update_seconds"],
                       checkpoint_samples_seconds=summary["checkpoint_seconds"])
        summary["checkpoint_seconds"] = max(summary["checkpoint_seconds"], default=0.0)
    online_time = benchmarks["online"]["end_to_end_median_seconds"]
    cached_time = benchmarks["cached"]["end_to_end_median_seconds"]
    report = {"passed": True, "provenance": provenance, "backends": benchmarks,
              "matched_speedup": online_time / cached_time, "prefetch_batches": 0,
              "validation_seconds": max(value["validation_seconds"] for value in benchmarks.values())}
    _json(root / "reports/benchmark.json", report)
    print(f"[cached parity] all gates passed; matched median speedup={online_time/cached_time:.3f}x", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--worker", choices=("full", "continue", "resume", "preflight", "resource"))
    parser.add_argument("--backend", choices=("online", "cached"), default="cached")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--prefetch", type=int, choices=(0, 1), default=0)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--preflight", action="store_true")
    stages.add_argument("--resource-check", action="store_true")
    args = parser.parse_args(argv)
    if args.cpus:
        available = sorted(os.sched_getaffinity(0))
        if len(available) < args.cpus:
            raise ValueError("Requested CPU affinity exceeds allocation")
        os.sched_setaffinity(0, available[:args.cpus])
    if args.worker == "full":
        _full_worker(args.experiment_root, args.backend)
    elif args.worker in ("continue", "resume", "preflight", "resource"):
        _continuation_worker(args.experiment_root, args.backend, resume=args.worker == "resume",
                             cpus=args.cpus, prefetch=args.prefetch, preflight=args.worker == "preflight",
                             resource_check=args.worker == "resource")
    elif args.preflight:
        run_preflight(args.experiment_root)
    elif args.resource_check:
        run_resource_check(args.experiment_root)
    else:
        run_gates(args.experiment_root)


if __name__ == "__main__":
    main()
