"""Initialization, exact resume, logging, and loop orchestration for v22_final."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp

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
    load_v22_final_checkpoint,
    load_v22_final_training_checkpoint,
    training_checkpoint_payload,
    validate_param_tree_compatible,
)
from ..model import build_model_configs
from .config import (
    V22FinalTrainConfig,
    V22FinalTrainInvocation,
    validate_resume_config,
)
from .data import TrainingCursor, open_training_data
from .step import build_optimizer, build_training_transforms, make_train_step


def validate_input_paths(config: V22FinalTrainConfig) -> None:
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
        raise FileNotFoundError(f"Missing v22_final training inputs: {missing}")


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


def _prepare_run_directory(invocation: V22FinalTrainInvocation) -> None:
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


def _save_checkpoint(
    *,
    config: V22FinalTrainConfig,
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


def run_training(invocation: V22FinalTrainInvocation) -> Path | None:
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
    overlay_metadata = {
        "residual_copied": residual_overlay.copied,
        "residual_fresh": residual_overlay.initialized,
        "baseline_copied": baseline_overlay.copied,
        "baseline_fresh": baseline_overlay.initialized,
    }

    if invocation.resume is not None:
        checkpoint = load_v22_final_training_checkpoint(invocation.resume)
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
        overlay_metadata = dict(checkpoint.parameter_overlay_metadata)
        print(f"[v22_final] exact resume from {invocation.resume} at step {completed_step}")
    elif invocation.init_from is not None:
        checkpoint = load_v22_final_checkpoint(invocation.init_from)
        if checkpoint.checkpoint_kind == "swa":
            # SWA is allowed as a warm start, never as exact resume.
            print(f"[v22_final] warm-starting from SWA checkpoint {invocation.init_from}")
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
        print(f"[v22_final] warm start from {invocation.init_from}; optimizer reset")

    n_residual_parameters = sum(
        int(leaf.size) for leaf in jax.tree_util.tree_leaves(residual_params)
    )
    n_baseline_parameters = sum(
        int(leaf.size) for leaf in jax.tree_util.tree_leaves(baseline_params)
    )
    run_metadata = {
        **config.to_dict(),
        "config_source": str(invocation.config_path),
        "derived": {
            "input_steps": training_data.input_steps,
            "train_anchors": int(training_data.train_split.size),
            "validation_anchors": int(training_data.val_split.size),
            "segments": len(training_data.segments),
            "chunks_per_segment": config.segment_steps // config.bptt_steps,
            "residual_parameters": n_residual_parameters,
            "baseline_parameters_frozen": n_baseline_parameters,
            "baseline_checkpoint_fingerprint": baseline_fingerprint,
            "anchor_manifest_fingerprint": training_data.manifest_fingerprint,
            "parameter_overlay": overlay_metadata,
            "residual_target": "truth_minus_live_frozen_baseline",
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
    print(
        f"[v22_final] residual_params={n_residual_parameters:,} "
        f"segments={len(training_data.segments)} truth_prefix={config.truth_prefix_steps} "
        f"ar_tail={config.ar_tail_k} feedback={config.feedback_mode}",
        flush=True,
    )

    last_checkpoint: Path | None = None
    has_consumed_update = completed_step > 0
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
            chunk.truth_inputs,
            chunk.targets,
            chunk.forcings,
        )
        loss = jax.block_until_ready(loss)
        gradient_norm = jax.block_until_ready(gradient_norm)
        step_seconds = time.monotonic() - started
        cursor = chunk.next_cursor
        has_consumed_update = True
        loss_value = float(jax.device_get(loss))
        gradient_norm_value = float(jax.device_get(gradient_norm))
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
        _append_jsonl(metrics_path, record)
        if step <= 5 or step % 10 == 0:
            print(
                f"step {step}/{config.max_steps} loss {loss_value:.5f} "
                f"grad_norm {gradient_norm_value:.4f} step_time {step_seconds:.2f}s",
                flush=True,
            )
        if step % config.checkpoint_every == 0 or step == config.max_steps:
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
            print(f"[v22_final] saved {last_checkpoint}", flush=True)

    if last_checkpoint is None:
        raise RuntimeError("Training completed without writing a checkpoint")
    print(f"[v22_final] training complete at step {config.max_steps}")
    return last_checkpoint
