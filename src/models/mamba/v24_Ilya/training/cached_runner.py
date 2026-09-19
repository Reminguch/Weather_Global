"""Residual-only cached training and a canonical maintained step-zero initializer."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from ..checkpoint import atomic_json_dump, atomic_pickle_dump, file_sha256, training_checkpoint_payload
from .cached_checkpoint import (
    build_provenance, code_fingerprints, digest_json, latest_checkpoint_for_run,
    project_shared_initialization, reconcile_run_metrics, require_production_gates,
    validate_cached_checkpoint, write_cached_checkpoint,
)
from .config import load_training_config, validate_resume_config
from .data import TrainingCursor
from .endpoint_step import build_optimizer, cast_physical_boundary_fp32, cast_state_boundary_fp32
from .validation import append_validation_record, load_validation_records, update_best_validation


def prepare_shared_initialization(config_path: Path, output_dir: Path) -> Path:
    """Reproduce the maintained initializer, including its baseline RNG split.

    This is the sole preparation stage that initializes the frozen GraphCast.
    Cached optimization loads the resulting residual-only native checkpoint.
    """
    from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats, validate_stats_coverage
    from src.models.mamba.training.param_utils import overlay_matching_params
    from ..checkpoint import manifest_sha256, overlay_frozen_baseline_params, validate_param_tree_compatible
    from ..model import build_model_configs
    from .data import open_training_data
    from .endpoint_step import build_training_transforms
    from .runner import _validate_tree_shapes, validate_input_paths
    from .cached_diagnostics import require_matched_gpu_environment

    runtime = require_matched_gpu_environment()
    config = load_training_config(Path(config_path))
    if config.distributed.mode != "single" or config.distributed.per_device_batch_size != 1:
        raise ValueError("Canonical paired initialization requires a single-device batch of one")
    output_dir = Path(output_dir)
    output_path = output_dir / "checkpoint_step00000000.pkl"
    marker = output_dir / "shared_init.json"
    identity = {"config_sha256": digest_json(config.to_dict()),
                "baseline_sha256": file_sha256(config.baseline_checkpoint),
                "anchor_manifest_sha256": manifest_sha256(config.anchor_manifest_root),
                "prepared_metadata_sha256": file_sha256(config.prepared_root / "metadata.json"),
                "stats_sha256": {path.name: file_sha256(path) for path in sorted(config.stats_dir.glob("*.nc"))},
                "code_sha256": code_fingerprints(Path(__file__).resolve().parents[5])}
    if marker.is_file():
        existing = json.loads(marker.read_text())
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing shared initialization belongs to another configuration")
        if existing.get("checkpoint_sha256") != file_sha256(output_path):
            raise ValueError("Shared initialization checkpoint was modified")
        return output_path
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Incomplete or conflicting shared initialization: {output_dir}")
    validate_input_paths(config)
    baseline_checkpoint = load_graphcast_checkpoint(config.baseline_checkpoint)
    task_config = dataclasses.replace(baseline_checkpoint.task_config, input_duration=config.input_duration)
    stats = load_stats(config.stats_dir)
    validate_stats_coverage(task_config, stats)
    data = open_training_data(config, task_config)
    transforms = build_training_transforms(build_model_configs(baseline_checkpoint.model_config, config.architecture),
                                           task_config, stats, config)
    anchor = int(data.anchor_indices[int(data.train_split[0])])
    sample = data.store.build_batch_from_indices(indices=[anchor], input_steps=data.input_steps,
                                                 target_steps=config.target_steps, task_cfg=task_config, dt=data.time_step)
    sample = tuple(cast_physical_boundary_fp32(value) for value in sample)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, residual_key = jax.random.split(rng_key)
    params, state = transforms.residual_loss.init(residual_key, *sample)
    prediction_params, prediction_state = transforms.residual_predict.init(residual_key, *sample)
    validate_param_tree_compatible(params, prediction_params)
    state = cast_state_boundary_fp32(state)
    _validate_tree_shapes(state, cast_state_boundary_fp32(prediction_state), "residual state")
    params, residual_overlay = overlay_matching_params(
        params, baseline_checkpoint.params if config.architecture.residual_initialization == "baseline_overlay" else {},
        strict=False)
    rng_key, baseline_key = jax.random.split(rng_key)
    baseline_params, baseline_state = transforms.baseline_predict.init(baseline_key, *sample)
    _, baseline_overlay = overlay_frozen_baseline_params(baseline_params, baseline_checkpoint.params)
    if jax.tree_util.tree_leaves(baseline_state):
        raise ValueError("Frozen GraphCast unexpectedly has recurrent state")
    optimizer, _ = build_optimizer(config)
    overlay = {"residual_initialization": config.architecture.residual_initialization,
               "residual_copied": residual_overlay.copied, "residual_fresh": residual_overlay.initialized,
               "baseline_copied": baseline_overlay.copied, "baseline_fresh": 0,
               "baseline_ignored_source_count": len(baseline_overlay.ignored_source),
               "baseline_ignored_source": list(baseline_overlay.ignored_source)}
    payload = training_checkpoint_payload(
        completed_step=0, residual_params=params, residual_state=state,
        optimizer_state=optimizer.init(params), rng_key=rng_key, training_cursor=TrainingCursor().to_dict(),
        resolved_training_config=config.to_dict(), baseline_checkpoint_path=str(config.baseline_checkpoint),
        baseline_checkpoint_fingerprint=identity["baseline_sha256"], anchor_manifest_fingerprint=data.manifest_fingerprint,
        parameter_overlay_metadata=overlay)
    atomic_pickle_dump(payload, output_path)
    atomic_json_dump({**identity, "format": "v24_shared_native_init_v1", "checkpoint": str(output_path),
                      "checkpoint_sha256": file_sha256(output_path), "runtime": runtime,
                      "rng_order": "residual_loss,residual_predict_same_key,baseline"}, marker)
    return output_path


def execution_metadata(execution) -> dict:
    value = dataclasses.asdict(execution) if dataclasses.is_dataclass(execution) else dict(execution)
    value["backend"] = "cached_stepwise"
    return json.loads(json.dumps(value, default=str))


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def run_cached_validation(*, train_step, cached_data, config, params, zero_state, step: int) -> dict:
    """Delegate both backends to the shared streaming physical-field evaluator."""
    from .cached_validation import run_cached_validation as evaluate
    return evaluate(train_step=train_step, cached_data=cached_data, config=config, params=params,
                    zero_state=zero_state, step=step)


def run_cached_training(config_path: Path, shared_init_path: Path, *, cache_root: Path,
                        prefetch_batches: int = 0, expected_manifest_sha256: str | None = None,
                        resume: Path | None = None, max_steps: int | None = None,
                        backend: str = "cached_stepwise",
                        parity_report: Path | None = None, paired_report: Path | None = None) -> Path:
    from .cached_config import load_cached_training_config
    from .cached_data import open_cached_training_data
    from .cached_stepwise import make_cached_stepwise_train_step
    from .cached_diagnostics import require_matched_gpu_environment

    startup_started = time.monotonic()
    runtime = require_matched_gpu_environment()
    if backend != "cached_stepwise":
        raise ValueError("The cached runner supports only backend='cached_stepwise'")
    resolved = load_cached_training_config(Path(config_path), cache_root=Path(cache_root),
                                           prefetch_batches=prefetch_batches,
                                           expected_manifest_sha256=expected_manifest_sha256)
    config = resolved.common
    stop = min(config.max_steps, max_steps) if max_steps is not None else config.max_steps
    if stop < 1:
        raise ValueError("Training requires a positive stopping step")
    execution = execution_metadata(resolved.execution)
    provenance = build_provenance(config, Path(shared_init_path), Path(cache_root), execution=execution)
    gates = None
    if stop > 20:
        if parity_report is None or paired_report is None:
            raise ValueError("Production training requires full-parity and paired-20 reports")
        gates = require_production_gates(parity_report, paired_report, provenance)
    if resume is None:
        from ..checkpoint import load_v24_Ilya_training_checkpoint
        existing = latest_checkpoint_for_run(config)
        if existing is not None and load_v24_Ilya_training_checkpoint(existing).completed_step > 0:
            raise ValueError(f"Existing training run requires explicit --resume {existing}")
        resume = project_shared_initialization(Path(shared_init_path), config, provenance=provenance, execution=execution)
    if Path(resume).resolve().parent.parent != config.run_dir.resolve():
        raise ValueError("Cached resume checkpoint must belong to this run directory")
    checkpoint = validate_cached_checkpoint(Path(resume), provenance, execution)
    reconcile_run_metrics(config.run_dir, checkpoint.completed_step)
    validation_path = config.run_dir / "validation_metrics.jsonl"
    validation_due = (config.validation.enabled and checkpoint.completed_step > 0
                      and (checkpoint.completed_step % config.validation.every_steps == 0
                           or checkpoint.completed_step == config.max_steps))
    validation_pending = validation_due and not any(
        int(row["step"]) == checkpoint.completed_step and row["role"] == "fixed_checkpoint"
        for row in load_validation_records(validation_path))
    if checkpoint.completed_step >= stop and not validation_pending:
        return Path(resume)
    if checkpoint.completed_step < stop:
        validate_resume_config(config, checkpoint.resolved_training_config, completed_step=checkpoint.completed_step)
    cached = open_cached_training_data(resolved)
    if checkpoint.anchor_manifest_fingerprint != cached.training_data.manifest_fingerprint:
        raise ValueError("Shared initialization and cache use different anchor manifests")
    if checkpoint.baseline_checkpoint_fingerprint != cached.manifest["compatibility"]["baseline_sha256"]:
        raise ValueError("Shared initialization and cache use different frozen GraphCast checkpoints")
    _, _, task_config, stats, data, baseline_model_config = cached.context
    model_config = dataclasses.replace(baseline_model_config,
                                       latent_size=config.architecture.residual_width or config.architecture.width,
                                       gnn_msg_steps=config.architecture.residual_msg_steps)
    cursor = TrainingCursor.from_mapping(checkpoint.training_cursor)
    data.validate_cursor(cursor, config)
    optimizer, learning_rate = build_optimizer(config)
    train_step = make_cached_stepwise_train_step(config=config, model_config=model_config, task_config=task_config,
                                                stats=stats, optimizer=optimizer)
    params, state, opt_state, rng = (checkpoint.residual_params, checkpoint.residual_state,
                                    checkpoint.optimizer_state, checkpoint.rng_key)
    zero_state = jax.tree_util.tree_map(jnp.zeros_like, state)
    metadata_path = config.run_dir / "cached_execution.json"
    atomic_json_dump({"provenance": provenance, "execution": execution, "gates": gates,
                      "cache_identity": cached.identity, "baseline_initializations": 0, "baseline_apply_calls": 0,
                      "runtime": runtime,
                      "startup_seconds": time.monotonic() - startup_started}, metadata_path)
    _append(config.run_dir / "startup_metrics.jsonl", {"resume_step": checkpoint.completed_step,
            "resume_checkpoint": str(resume), "startup_seconds": time.monotonic() - startup_started})
    metrics_path = config.run_dir / "train_metrics.jsonl"
    validation_path = config.run_dir / "validation_metrics.jsonl"
    last_path = Path(resume)

    def validate(step):
        if any(int(row["step"]) == step and row["role"] == "fixed_checkpoint"
               for row in load_validation_records(validation_path)):
            return
        record = run_cached_validation(train_step=train_step, cached_data=cached, config=config,
                                       params=params, zero_state=zero_state, step=step)
        append_validation_record(validation_path, record)
        update_best_validation(records=load_validation_records(validation_path),
                               checkpoint_for_step=lambda n: config.run_dir / "checkpoints" / f"checkpoint_step{n:08d}.pkl",
                               output_path=config.run_dir / "best_validation.json",
                               subset_fingerprint=record["subset_fingerprint"])
        print(f"[v24_cached] validation step={step} loss={record['loss']:.8f} segments={record['selected_segments']}", flush=True)

    if validation_pending:
        validate(checkpoint.completed_step)
    if checkpoint.completed_step >= stop:
        return last_path
    # Validation and training share the prepared store. The data adapter owns
    # any optional one-batch prefetch and serializes source materialization.
    iterator = cached.iter_chunks(cursor, prefetch_batches=prefetch_batches,
                                  max_chunks=stop - checkpoint.completed_step)
    try:
        for step in range(checkpoint.completed_step + 1, stop + 1):
            started = time.monotonic()
            batch = next(iterator)
            loaded = time.monotonic()
            if cursor.segment_offset == 0 and step > 1:
                state = jax.tree_util.tree_map(jnp.zeros_like, state)
            consumed_cursor = cursor
            rng, *keys = jax.random.split(rng, config.bptt_steps + 1)
            params, state, opt_state, loss, grad_norm, components = train_step(
                params, state, opt_state, jnp.stack(keys), batch.inputs, batch.targets, batch.forcings)
            loss, grad_norm, components = jax.device_get((loss, grad_norm, components))
            completed = time.monotonic()
            if not math.isfinite(float(loss)) or not math.isfinite(float(grad_norm)):
                raise FloatingPointError(f"Nonfinite cached update at step {step}")
            cursor = batch.next_cursor
            record = {"step": step, "loss": float(loss), "gradient_norm": float(grad_norm),
                      "step_seconds": completed - loaded, "end_to_end_seconds": completed - started,
                      "data_wait_seconds": loaded - started, "learning_rate": float(jax.device_get(learning_rate(step - 1))) if callable(learning_rate) else float(learning_rate),
                      "batch_size": 1, "anchors_seen": step * config.bptt_steps,
                      **consumed_cursor.to_dict(), "chunk_id": batch.chunk_id,
                      "loss_by_horizon": {str(h): float(components[i]) for i, h in enumerate(config.supervised_horizon_labels)}}
            _append(metrics_path, record)
            if step <= 5 or step % 10 == 0:
                print(f"step {step}/{stop} loss {float(loss):.8f} grad_norm {float(grad_norm):.6f} step_time {completed-loaded:.3f}s end_to_end {completed-started:.3f}s", flush=True)
            if step % config.checkpoint_every == 0 or step == stop:
                checkpoint_started = time.monotonic()
                last_path = write_cached_checkpoint(config.run_dir / "checkpoints" / f"checkpoint_step{step:08d}.pkl",
                    config=config, completed_step=step, residual_params=params, residual_state=state,
                    optimizer_state=opt_state, rng_key=rng, cursor=cursor,
                    baseline_fingerprint=checkpoint.baseline_checkpoint_fingerprint,
                    anchor_fingerprint=checkpoint.anchor_manifest_fingerprint,
                    overlay_metadata=checkpoint.parameter_overlay_metadata, provenance=provenance, execution=execution)
                _append(config.run_dir / "checkpoint_metrics.jsonl", {"step": step,
                        "checkpoint_seconds": time.monotonic() - checkpoint_started,
                        "checkpoint": str(last_path)})
            if config.validation.enabled and (step % config.validation.every_steps == 0 or step == config.max_steps):
                validate(step)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    return last_path
