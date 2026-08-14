"""Initialization, exact resume, logging, and loop orchestration for v23_Ilya."""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from src.models.graphcast.training.core.model import (
    load_graphcast_checkpoint,
    load_stats,
    validate_stats_coverage,
)
from src.models.mamba.training.param_utils import overlay_matching_params

from ..checkpoint import (
    atomic_json_dump,
    atomic_pickle_dump,
    file_sha256,
    load_v23_Ilya_checkpoint,
    load_v23_Ilya_training_checkpoint,
    training_checkpoint_payload,
    validate_param_tree_compatible,
)
from ..model import build_model_configs
from .config import (
    V23IlyaTrainConfig,
    V23IlyaTrainInvocation,
    validate_resume_config,
)
from .data import TrainingCursor, open_training_data
from .step import (
    build_optimizer,
    build_training_transforms,
    make_train_step,
    make_validation_step,
    memory_contract,
)
from .validation import (
    append_validation_record,
    find_validation_record,
    load_validation_records,
    run_fixed_validation,
    select_validation_segment_ids,
    update_best_validation,
    update_train_validation_plot,
)


def validate_input_paths(config: V23IlyaTrainConfig) -> None:
    required_files = [
        config.baseline_checkpoint,
        config.prepared_root / "metadata.json",
        config.anchor_manifest_root / "metadata.json",
        config.anchor_manifest_root / "anchors/anchor_indices.npy",
        config.anchor_manifest_root / "anchors/split_train.npy",
        config.anchor_manifest_root / "anchors/split_val.npy",
        config.stats_dir / "stddev_by_level.nc",
        config.stats_dir / "mean_by_level.nc",
        config.stats_dir / "diffs_stddev_by_level.nc",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing v23_Ilya training inputs: {missing}")


def _validate_tree_shapes(reference, candidate, label: str) -> None:
    if jax.tree_util.tree_structure(reference) != jax.tree_util.tree_structure(candidate):
        raise ValueError(f"{label} tree structure is incompatible")
    for index, (expected, actual) in enumerate(
        zip(
            jax.tree_util.tree_leaves(reference),
            jax.tree_util.tree_leaves(candidate),
            strict=True,
        )
    ):
        if getattr(expected, "shape", None) != getattr(actual, "shape", None):
            raise ValueError(
                f"{label} leaf {index} shape mismatch: "
                f"expected={getattr(expected, 'shape', None)} "
                f"actual={getattr(actual, 'shape', None)}"
            )


def _learning_rate_value(schedule, update_index: int) -> float:
    if callable(schedule):
        return float(jax.device_get(schedule(update_index)))
    return float(schedule)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _prepare_run_directory(invocation: V23IlyaTrainInvocation) -> None:
    run_dir = invocation.config.run_dir
    if invocation.resume is not None:
        expected = invocation.resume.resolve().parent.parent
        if run_dir.resolve() != expected:
            raise ValueError(
                f"Resume checkpoint belongs to {expected}, but config resolves to {run_dir.resolve()}"
            )
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to start in non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def _device_memory_snapshot() -> dict[str, Any] | None:
    devices = []
    for device in jax.local_devices():
        try:
            stats = device.memory_stats() or {}
        except Exception:
            stats = {}
        numeric = {
            str(name): int(value)
            for name, value in stats.items()
            if isinstance(value, (int, np.integer))
        }
        if numeric:
            devices.append(
                {"device": str(device), "platform": device.platform, **numeric}
            )
    if not devices:
        return None
    peak = max(int(item.get("peak_bytes_in_use", 0)) for item in devices)
    reference_limit = 27 * 1024**3
    return {
        "devices": devices,
        "max_peak_bytes_in_use": peak,
        "reference_peak_limit_bytes": reference_limit,
        "within_reference_peak_limit": peak <= reference_limit,
    }


def _save_checkpoint(
    *,
    config: V23IlyaTrainConfig,
    completed_step: int,
    residual_params,
    residual_state,
    optimizer_state,
    rng_key,
    cursor: TrainingCursor,
    baseline_fingerprint: str,
    manifest_fingerprint: str,
    overlay_metadata: Mapping[str, Any],
) -> Path:
    path = config.run_dir / "checkpoints" / f"checkpoint_step{completed_step:08d}.pkl"
    payload = training_checkpoint_payload(
        completed_step=completed_step,
        residual_params=residual_params,
        residual_state=residual_state,
        optimizer_state=optimizer_state,
        rng_key=rng_key,
        training_cursor=cursor.to_dict(),
        resolved_training_config=config.to_dict(),
        baseline_checkpoint_path=str(config.baseline_checkpoint),
        baseline_checkpoint_fingerprint=baseline_fingerprint,
        anchor_manifest_fingerprint=manifest_fingerprint,
        parameter_overlay_metadata=overlay_metadata,
    )
    atomic_pickle_dump(payload, path)
    atomic_json_dump(
        {"completed_step": completed_step, "checkpoint": str(path)},
        config.run_dir / "latest_checkpoint.json",
    )
    return path


def run_training(invocation: V23IlyaTrainInvocation) -> Path | None:
    config = invocation.config
    validate_input_paths(config)
    if invocation.dry_run:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        print(f"run_dir={config.run_dir}")
        print(f"resume={invocation.resume} init_from={invocation.init_from}")
        return None

    _prepare_run_directory(invocation)
    baseline_fingerprint = file_sha256(config.baseline_checkpoint)
    baseline_checkpoint = load_graphcast_checkpoint(config.baseline_checkpoint)
    task_config = baseline_checkpoint.task_config
    if config.input_duration is not None:
        task_config = dataclasses.replace(
            task_config,
            input_duration=config.input_duration,
        )
    stats = load_stats(config.stats_dir)
    validate_stats_coverage(task_config, stats)
    training_data = open_training_data(config, task_config)
    model_configs = build_model_configs(
        baseline_checkpoint.model_config,
        config.architecture,
    )
    transforms = build_training_transforms(
        model_configs,
        task_config,
        stats,
        config,
    )

    sample_anchor = int(
        training_data.anchor_indices[int(training_data.train_split[0])]
    )
    sample_inputs, sample_targets, sample_forcings = (
        training_data.store.build_batch_from_indices(
            indices=[sample_anchor],
            input_steps=training_data.input_steps,
            target_steps=config.target_steps,
            task_cfg=task_config,
            dt=training_data.time_step,
        )
    )

    # Match v20 RNG order exactly: residual loss init, residual prediction
    # compatibility init with the same key, then frozen baseline init.
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, residual_init_key = jax.random.split(rng_key)
    residual_params, zero_residual_state = transforms.residual_loss.init(
        residual_init_key,
        sample_inputs,
        sample_targets,
        sample_forcings,
    )
    prediction_params, prediction_state = transforms.residual_predict.init(
        residual_init_key,
        sample_inputs,
        sample_targets,
        sample_forcings,
    )
    validate_param_tree_compatible(residual_params, prediction_params)
    _validate_tree_shapes(zero_residual_state, prediction_state, "residual state")
    residual_params, residual_overlay = overlay_matching_params(
        residual_params,
        baseline_checkpoint.params,
        strict=False,
    )

    rng_key, baseline_init_key = jax.random.split(rng_key)
    baseline_params, baseline_state = transforms.baseline_predict.init(
        baseline_init_key,
        sample_inputs,
        sample_targets,
        sample_forcings,
    )
    baseline_params, baseline_overlay = overlay_matching_params(
        baseline_params,
        baseline_checkpoint.params,
        strict=True,
    )
    optimizer, learning_rate = build_optimizer(config)
    initial_optimizer_state = optimizer.init(residual_params)
    residual_state = zero_residual_state
    optimizer_state = initial_optimizer_state
    cursor = TrainingCursor()
    completed_step = 0
    resume_checkpoint_was_final = False
    overlay_metadata = {
        "residual_copied": residual_overlay.copied,
        "residual_fresh": residual_overlay.initialized,
        "baseline_copied": baseline_overlay.copied,
        "baseline_fresh": baseline_overlay.initialized,
    }

    if invocation.resume is not None:
        checkpoint = load_v23_Ilya_training_checkpoint(invocation.resume)
        validate_resume_config(
            config,
            checkpoint.resolved_training_config,
            completed_step=checkpoint.completed_step,
        )
        if checkpoint.baseline_checkpoint_fingerprint != baseline_fingerprint:
            raise ValueError("Baseline checkpoint fingerprint differs from resume checkpoint")
        if checkpoint.anchor_manifest_fingerprint != training_data.manifest_fingerprint:
            raise ValueError("Anchor manifest fingerprint differs from resume checkpoint")
        validate_param_tree_compatible(residual_params, checkpoint.residual_params)
        _validate_tree_shapes(zero_residual_state, checkpoint.residual_state, "residual state")
        _validate_tree_shapes(
            initial_optimizer_state,
            checkpoint.optimizer_state,
            "optimizer state",
        )
        residual_params = checkpoint.residual_params
        residual_state = checkpoint.residual_state
        optimizer_state = checkpoint.optimizer_state
        rng_key = checkpoint.rng_key
        cursor = TrainingCursor.from_mapping(checkpoint.training_cursor)
        training_data.validate_cursor(cursor, config)
        completed_step = checkpoint.completed_step
        try:
            saved_max_steps = int(
                checkpoint.resolved_training_config["optimizer"]["max_steps"]
            )
        except (KeyError, TypeError, ValueError):
            saved_max_steps = -1
        resume_checkpoint_was_final = saved_max_steps == completed_step
        overlay_metadata = dict(checkpoint.parameter_overlay_metadata)
        print(f"[v23_Ilya] exact resume from {invocation.resume} at step {completed_step}")
    elif invocation.init_from is not None:
        checkpoint = load_v23_Ilya_checkpoint(invocation.init_from)
        if checkpoint.checkpoint_kind == "swa":
            # SWA is allowed as a warm start, never as exact resume.
            print(f"[v23_Ilya] warm-starting from SWA checkpoint {invocation.init_from}")
        validate_param_tree_compatible(residual_params, checkpoint.residual_params)
        residual_params = checkpoint.residual_params
        if checkpoint.has_residual_state:
            assert checkpoint.residual_state is not None
            _validate_tree_shapes(
                zero_residual_state,
                checkpoint.residual_state,
                "legacy residual state",
            )
            residual_state = checkpoint.residual_state
        print(f"[v23_Ilya] warm start from {invocation.init_from}; optimizer reset")

    n_residual_parameters = sum(
        int(leaf.size) for leaf in jax.tree_util.tree_leaves(residual_params)
    )
    n_baseline_parameters = sum(
        int(leaf.size) for leaf in jax.tree_util.tree_leaves(baseline_params)
    )
    memory_sample = training_data.load_segment_chunk(
        training_data.segments[0],
        0,
        config,
        task_config,
    )
    memory_metadata = memory_contract(
        input_frames=memory_sample.input_frames,
        static_inputs=memory_sample.static_inputs,
        truths=memory_sample.truths,
        residual_state=zero_residual_state,
        config=config,
    )
    memory_metadata["selective_data"] = dataclasses.asdict(memory_sample.data_report)

    run_metadata = {
        **config.to_dict(),
        "config_source": str(invocation.config_path),
        "derived": {
            "input_steps": training_data.input_steps,
            "train_anchors": int(training_data.train_split.size),
            "validation_anchors": int(training_data.val_split.size),
            "segments": len(training_data.segments),
            "validation_complete_segments": len(training_data.validation_segments),
            "validation_fixed_subset": (
                {
                    "policy": training_data.validation_subset_policy,
                    "fingerprint": training_data.validation_subset_fingerprint,
                    "segments": training_data.validation_segment_metadata(
                        training_data.fixed_validation_segment_ids
                    ),
                }
                if config.validation.enabled
                else None
            ),
            "chunks_per_segment": config.segment_steps // config.bptt_steps,
            "residual_parameters": n_residual_parameters,
            "baseline_parameters_frozen": n_baseline_parameters,
            "baseline_checkpoint_fingerprint": baseline_fingerprint,
            "anchor_manifest_fingerprint": training_data.manifest_fingerprint,
            "parameter_overlay": overlay_metadata,
            "residual_target": "truth_minus_live_frozen_baseline",
            "loss_mode": config.loss_mode,
            "memory_contract": memory_metadata,
            "batch_size": 1,
        },
    }
    run_config_path = config.run_dir / "run_config.json"
    if completed_step == 0:
        atomic_json_dump(run_metadata, run_config_path)
    elif not run_config_path.is_file():
        raise FileNotFoundError(f"Resume run is missing {run_config_path}")

    train_step = make_train_step(
        transforms=transforms,
        optimizer=optimizer,
        baseline_params=baseline_params,
        baseline_state=baseline_state,
        config=config,
        time_step=training_data.time_step,
        input_steps=training_data.input_steps,
    )
    metrics_path = config.run_dir / "train_metrics.jsonl"
    validation_metrics_path = config.run_dir / "validation_metrics.jsonl"
    validation_step = None
    if config.validation.enabled:
        validation_step = make_validation_step(
            transforms=transforms,
            baseline_params=baseline_params,
            baseline_state=baseline_state,
            config=config,
            time_step=training_data.time_step,
            input_steps=training_data.input_steps,
        )

    def checkpoint_for_step(step: int) -> Path:
        return config.run_dir / "checkpoints" / f"checkpoint_step{step:08d}.pkl"

    def perform_validation(
        *, step: int, segment_ids: np.ndarray, role: str, subset_policy: str
    ) -> Mapping[str, Any]:
        if validation_step is None:
            raise RuntimeError("Validation requested while disabled")
        expected_fingerprint = training_data.fingerprint_validation_subset(
            segment_ids
        )
        records = load_validation_records(validation_metrics_path)
        existing = find_validation_record(
            records,
            step=step,
            role=role,
            subset_fingerprint=expected_fingerprint,
        )
        if existing is not None:
            return existing
        checkpoint_path = checkpoint_for_step(step)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Validation requires the completed checkpoint {checkpoint_path}"
            )
        record = run_fixed_validation(
            validation_step=validation_step,
            residual_params=residual_params,
            zero_residual_state=zero_residual_state,
            training_data=training_data,
            task_config=task_config,
            config=config,
            segment_ids=segment_ids,
            step=step,
            role=role,
            subset_policy=subset_policy,
        )
        append_validation_record(validation_metrics_path, record)
        records = load_validation_records(validation_metrics_path)
        if role == "fixed_checkpoint":
            update_best_validation(
                records=records,
                checkpoint_for_step=checkpoint_for_step,
                output_path=config.run_dir / "best_validation.json",
                subset_fingerprint=expected_fingerprint,
            )
        try:
            update_train_validation_plot(
                train_metrics_path=metrics_path,
                validation_metrics_path=validation_metrics_path,
                output_path=config.run_dir / "train_validation_loss.png",
                subset_fingerprint=expected_fingerprint,
            )
        except Exception as exc:  # plotting must never invalidate training
            warnings.warn(f"Could not update validation loss plot: {exc}")
        print(
            f"[v23_Ilya] validation step={step} role={role} "
            f"loss={float(record['loss']):.6f} "
            f"segments={int(record['selected_segments'])} "
            f"time={float(record['duration_seconds']):.1f}s",
            flush=True,
        )
        return record

    print(
        f"[v23_Ilya] residual_params={n_residual_parameters:,} "
        f"segments={len(training_data.segments)} truth_prefix={config.truth_prefix_steps} "
        f"ar_tail={config.ar_tail_k} feedback={config.feedback_mode} "
        f"state_policy={config.temporal_state_policy}",
        flush=True,
    )
    memory_profile_pending = completed_step == 0

    last_checkpoint: Path | None = invocation.resume
    has_consumed_update = completed_step > 0
    if (
        config.validation.enabled
        and completed_step > 0
        and (
            completed_step % config.validation.every_steps == 0
            or resume_checkpoint_was_final
        )
    ):
        # Recover a checkpoint that was saved immediately before an interrupted
        # validation without consuming any training state.
        perform_validation(
            step=completed_step,
            segment_ids=training_data.fixed_validation_segment_ids,
            role="fixed_checkpoint",
            subset_policy=training_data.validation_subset_policy,
        )

    for step in range(completed_step + 1, config.max_steps + 1):
        if cursor.segment_offset == 0 and has_consumed_update:
            residual_state = jax.tree_util.tree_map(
                jnp.zeros_like,
                residual_state,
            )
        consumed_cursor = cursor
        chunk = training_data.build_chunk(cursor, config, task_config)
        rng_key, *step_keys = jax.random.split(rng_key, config.bptt_steps + 1)
        keys = jnp.stack(step_keys)
        started = time.monotonic()
        (
            residual_params,
            residual_state,
            optimizer_state,
            loss,
            gradient_norm,
        ) = train_step(
            residual_params,
            residual_state,
            optimizer_state,
            keys,
            chunk.input_frames,
            chunk.static_inputs,
            chunk.truths,
            chunk.forcings,
        )
        loss = jax.block_until_ready(loss)
        gradient_norm = jax.block_until_ready(gradient_norm)
        step_seconds = time.monotonic() - started
        cursor = chunk.next_cursor
        has_consumed_update = True
        loss_value = float(jax.device_get(loss))
        gradient_norm_value = float(jax.device_get(gradient_norm))
        device_memory = _device_memory_snapshot()
        record = {
            "step": step,
            "loss": loss_value,
            "gradient_norm": gradient_norm_value,
            "learning_rate": _learning_rate_value(learning_rate, step - 1),
            "step_seconds": step_seconds,
            "epoch": consumed_cursor.epoch,
            "segment_index": consumed_cursor.segment_index,
            "segment_offset": consumed_cursor.segment_offset,
        }
        if device_memory is not None:
            record["device_memory"] = device_memory
            if memory_profile_pending:
                run_metadata["derived"]["memory_contract"][
                    "observed_device_profile"
                ] = device_memory
                atomic_json_dump(run_metadata, run_config_path)
                memory_profile_pending = False
        _append_jsonl(metrics_path, record)
        if step <= 5 or step % 10 == 0:
            print(
                f"step {step}/{config.max_steps} loss {loss_value:.5f} "
                f"grad_norm {gradient_norm_value:.4f} step_time {step_seconds:.2f}s",
                flush=True,
            )
        should_validate = config.validation.enabled and (
            step % config.validation.every_steps == 0 or step == config.max_steps
        )
        should_checkpoint = (
            step % config.checkpoint_every == 0
            or step == config.max_steps
            or should_validate
        )
        if should_checkpoint:
            last_checkpoint = _save_checkpoint(
                config=config,
                completed_step=step,
                residual_params=residual_params,
                residual_state=residual_state,
                optimizer_state=optimizer_state,
                rng_key=rng_key,
                cursor=cursor,
                baseline_fingerprint=baseline_fingerprint,
                manifest_fingerprint=training_data.manifest_fingerprint,
                overlay_metadata=overlay_metadata,
            )
            print(f"[v23_Ilya] saved {last_checkpoint}", flush=True)
        if should_validate:
            fixed_record = perform_validation(
                step=step,
                segment_ids=training_data.fixed_validation_segment_ids,
                role="fixed_checkpoint",
                subset_policy=training_data.validation_subset_policy,
            )
            if step == config.max_steps:
                final_ids, final_policy = select_validation_segment_ids(
                    training_data, config.validation.final_num_segments
                )
                if np.array_equal(
                    final_ids, training_data.fixed_validation_segment_ids
                ):
                    final_record = dict(fixed_record)
                    final_record["role"] = "final_full"
                    append_validation_record(validation_metrics_path, final_record)
                else:
                    perform_validation(
                        step=step,
                        segment_ids=final_ids,
                        role="final_full",
                        subset_policy=final_policy,
                    )

    if last_checkpoint is None:
        raise RuntimeError("Training completed without writing a checkpoint")
    print(f"[v23_Ilya] training complete at step {config.max_steps}")
    return last_checkpoint
