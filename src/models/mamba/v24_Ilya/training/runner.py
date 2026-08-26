"""Initialization, exact resume, logging, and loop orchestration for v24_Ilya."""

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
    data_parallel_training_checkpoint_payload,
    file_sha256,
    load_v24_Ilya_checkpoint,
    load_v24_Ilya_data_parallel_training_checkpoint,
    load_v24_Ilya_training_checkpoint,
    overlay_frozen_baseline_params,
    training_checkpoint_payload,
    validate_param_tree_compatible,
)
from ..model import build_model_configs
from .config import (
    V24IlyaTrainConfig,
    V24IlyaTrainInvocation,
    validate_resume_config,
)
from .data import ReplicaGroupCursor, TrainingCursor, open_training_data
from .data_parallel import (
    derive_replica_step_keys,
    replica_max_abs_difference,
    replica_tree_to_host,
    replicate_tree,
    shard_replica_tree,
    unreplicate_tree,
    validate_data_parallel_runtime,
)
from .step import (
    build_optimizer,
    build_training_transforms,
    cast_physical_boundary_fp32,
    cast_state_boundary_fp32,
    make_data_parallel_train_step,
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


def validate_input_paths(config: V24IlyaTrainConfig) -> None:
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
        raise FileNotFoundError(f"Missing v24_Ilya training inputs: {missing}")


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


def _validate_replica_tree_shapes(reference, candidate, replicas: int, label: str) -> None:
    if jax.tree_util.tree_structure(reference) != jax.tree_util.tree_structure(candidate):
        raise ValueError(f"{label} tree structure is incompatible")
    for index, (expected, actual) in enumerate(
        zip(
            jax.tree_util.tree_leaves(reference),
            jax.tree_util.tree_leaves(candidate),
            strict=True,
        )
    ):
        expected_shape = getattr(expected, "shape", None)
        actual_shape = getattr(actual, "shape", None)
        replica_shape = (
            (replicas, *expected_shape) if expected_shape is not None else None
        )
        if actual_shape != replica_shape:
            raise ValueError(
                f"{label} leaf {index} shape mismatch: "
                f"expected={replica_shape} actual={actual_shape}"
            )


def _learning_rate_value(schedule, update_index: int) -> float:
    if callable(schedule):
        return float(jax.device_get(schedule(update_index)))
    return float(schedule)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _prepare_run_directory(invocation: V24IlyaTrainInvocation) -> None:
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


def _device_memory_snapshot(
    reference_peak_limit_bytes: int = 27 * 1024**3,
) -> dict[str, Any] | None:
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
    return {
        "devices": devices,
        "max_peak_bytes_in_use": peak,
        "reference_peak_limit_bytes": reference_peak_limit_bytes,
        "within_reference_peak_limit": peak <= reference_peak_limit_bytes,
    }


def _save_checkpoint(
    *,
    config: V24IlyaTrainConfig,
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


def _save_data_parallel_checkpoint(
    *,
    config: V24IlyaTrainConfig,
    completed_step: int,
    residual_params,
    replica_states,
    optimizer_state,
    rng_key,
    cursor: ReplicaGroupCursor,
    active_segment_ids: tuple[int, ...],
    baseline_fingerprint: str,
    manifest_fingerprint: str,
    overlay_metadata: Mapping[str, Any],
) -> Path:
    path = config.run_dir / "checkpoints" / f"checkpoint_step{completed_step:08d}.pkl"
    payload = data_parallel_training_checkpoint_payload(
        completed_step=completed_step,
        residual_params=residual_params,
        replica_states=replica_states,
        optimizer_state=optimizer_state,
        rng_key=rng_key,
        replica_group_cursor=cursor.to_dict(),
        active_segment_ids=active_segment_ids,
        num_devices=config.distributed.num_devices,
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


def run_training(invocation: V24IlyaTrainInvocation) -> Path | None:
    config = invocation.config
    if invocation.baseline_validation_only and invocation.validation_compare is not None:
        raise ValueError(
            "--baseline-validation-only and --validation-compare are mutually exclusive"
        )
    if invocation.validation_compare is not None and (
        invocation.resume is not None or invocation.init_from is not None
    ):
        raise ValueError("--validation-compare cannot be combined with resume or init-from")
    if invocation.validation_compare is not None:
        if not config.validation.enabled:
            raise ValueError("--validation-compare requires validation.enabled=true")
        if not config.architecture.temporal_zero_init_out:
            raise ValueError(
                "--validation-compare requires temporal_zero_init_out=true"
            )
    if invocation.baseline_validation_only:
        if invocation.resume is not None or invocation.init_from is not None:
            raise ValueError(
                "--baseline-validation-only requires fresh initialization"
            )
        if not config.validation.enabled:
            raise ValueError(
                "--baseline-validation-only requires validation.enabled=true"
            )
        if not config.architecture.temporal_zero_init_out:
            raise ValueError(
                "--baseline-validation-only requires temporal_zero_init_out=true"
            )
    validate_input_paths(config)
    if invocation.dry_run:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        print(f"run_dir={config.run_dir}")
        print(f"resume={invocation.resume} init_from={invocation.init_from}")
        return None

    data_parallel = config.distributed.mode == "data_parallel"
    devices = None
    if data_parallel:
        devices = validate_data_parallel_runtime(config.distributed.num_devices)
        print(f"[v24_Ilya] data_parallel devices={[str(device) for device in devices]}")

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
    sample_inputs = cast_physical_boundary_fp32(sample_inputs)
    sample_targets = cast_physical_boundary_fp32(sample_targets)
    sample_forcings = cast_physical_boundary_fp32(sample_forcings)

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
    zero_residual_state = cast_state_boundary_fp32(zero_residual_state)
    prediction_state = cast_state_boundary_fp32(prediction_state)
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
    baseline_params, baseline_overlay = overlay_frozen_baseline_params(
        baseline_params,
        baseline_checkpoint.params,
    )
    del sample_inputs, sample_targets, sample_forcings
    optimizer, learning_rate = build_optimizer(config)
    initial_optimizer_state = optimizer.init(residual_params)
    residual_state = zero_residual_state
    optimizer_state = initial_optimizer_state
    cursor = TrainingCursor()
    replica_cursor = ReplicaGroupCursor()
    checkpoint_replica_states = None
    completed_step = 0
    resume_checkpoint_was_final = False
    overlay_metadata = {
        "residual_copied": residual_overlay.copied,
        "residual_fresh": residual_overlay.initialized,
        "baseline_copied": baseline_overlay.copied,
        "baseline_fresh": 0,
        "baseline_ignored_source_count": len(baseline_overlay.ignored_source),
        "baseline_ignored_source": list(baseline_overlay.ignored_source),
    }

    if invocation.resume is not None and data_parallel:
        checkpoint = load_v24_Ilya_data_parallel_training_checkpoint(invocation.resume)
        if checkpoint.num_devices != config.distributed.num_devices:
            raise ValueError(
                f"Resume checkpoint has {checkpoint.num_devices} devices, expected "
                f"{config.distributed.num_devices}"
            )
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
        _validate_replica_tree_shapes(
            zero_residual_state,
            checkpoint.replica_states,
            config.distributed.num_devices,
            "replica states",
        )
        _validate_tree_shapes(initial_optimizer_state, checkpoint.optimizer_state, "optimizer state")
        residual_params = checkpoint.residual_params
        checkpoint_replica_states = cast_state_boundary_fp32(checkpoint.replica_states)
        optimizer_state = checkpoint.optimizer_state
        rng_key = checkpoint.rng_key
        replica_cursor = ReplicaGroupCursor.from_mapping(checkpoint.replica_group_cursor)
        training_data.validate_replica_cursor(replica_cursor, config)
        expected_ids = training_data.active_replica_segment_ids(
            replica_cursor, config.distributed.num_devices
        )
        if checkpoint.active_segment_ids != expected_ids:
            raise ValueError(
                "Checkpoint active segment IDs do not match its replica cursor: "
                f"saved={checkpoint.active_segment_ids} expected={expected_ids}"
            )
        completed_step = checkpoint.completed_step
        try:
            saved_max_steps = int(
                checkpoint.resolved_training_config["optimizer"]["max_steps"]
            )
        except (KeyError, TypeError, ValueError):
            saved_max_steps = -1
        resume_checkpoint_was_final = saved_max_steps == completed_step
        overlay_metadata = dict(checkpoint.parameter_overlay_metadata)
        print(f"[v24_Ilya] exact resume from {invocation.resume} at step {completed_step}")
    elif invocation.resume is not None:
        checkpoint = load_v24_Ilya_training_checkpoint(invocation.resume)
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
        _validate_tree_shapes(initial_optimizer_state, checkpoint.optimizer_state, "optimizer state")
        residual_params = checkpoint.residual_params
        residual_state = cast_state_boundary_fp32(checkpoint.residual_state)
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
        print(f"[v24_Ilya] exact resume from {invocation.resume} at step {completed_step}")
    elif invocation.init_from is not None:
        checkpoint = load_v24_Ilya_checkpoint(invocation.init_from)
        if checkpoint.checkpoint_kind == "swa":
            # SWA is allowed as a warm start, never as exact resume.
            print(f"[v24_Ilya] warm-starting from SWA checkpoint {invocation.init_from}")
        validate_param_tree_compatible(residual_params, checkpoint.residual_params)
        residual_params = checkpoint.residual_params
        if checkpoint.has_residual_state and not data_parallel:
            assert checkpoint.residual_state is not None
            _validate_tree_shapes(
                zero_residual_state,
                checkpoint.residual_state,
                "legacy residual state",
            )
            residual_state = cast_state_boundary_fp32(checkpoint.residual_state)
        elif checkpoint.has_residual_state:
            print(
                "[v24_Ilya] ignoring warm-start lane state in data-parallel mode"
            )
        print(f"[v24_Ilya] warm start from {invocation.init_from}; optimizer reset")

    comparison_params = None
    if invocation.validation_compare is not None:
        comparison_checkpoint = load_v24_Ilya_checkpoint(
            invocation.validation_compare
        )
        validate_param_tree_compatible(
            residual_params,
            comparison_checkpoint.residual_params,
        )
        comparison_params = comparison_checkpoint.residual_params

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
    del memory_sample
    anchors_per_update = config.bptt_steps * config.distributed.global_batch_size
    if data_parallel:
        replica_group_count = training_data.replica_group_count(
            config.distributed.num_devices
        )
        if replica_group_count == 0:
            raise ValueError(
                "Data parallelism has no complete replica group: "
                f"segments={len(training_data.segments)} "
                f"devices={config.distributed.num_devices}"
            )
    else:
        replica_group_count = None

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
            "prepared_store_selection": training_data.store.selection_metadata,
            "residual_parameters": n_residual_parameters,
            "baseline_parameters_frozen": n_baseline_parameters,
            "baseline_checkpoint_fingerprint": baseline_fingerprint,
            "anchor_manifest_fingerprint": training_data.manifest_fingerprint,
            "parameter_overlay": overlay_metadata,
            "residual_target": "truth_minus_live_frozen_baseline",
            "loss_mode": config.loss_mode,
            "memory_contract": memory_metadata,
            "batch_size": config.distributed.global_batch_size,
            "global_batch_size": config.distributed.global_batch_size,
            "per_device_batch_size": config.distributed.per_device_batch_size,
            "anchors_per_update": anchors_per_update,
            "replica_group_count": replica_group_count,
            "dropped_replica_segments": (
                training_data.dropped_replica_segments(config.distributed.num_devices)
                if data_parallel
                else 0
            ),
        },
    }
    run_config_path = config.run_dir / "run_config.json"
    if completed_step == 0:
        atomic_json_dump(run_metadata, run_config_path)
    elif not run_config_path.is_file():
        raise FileNotFoundError(f"Resume run is missing {run_config_path}")

    if data_parallel:
        assert devices is not None
        train_step = make_data_parallel_train_step(
            transforms=transforms,
            optimizer=optimizer,
            baseline_params=baseline_params,
            baseline_state=baseline_state,
            config=config,
            time_step=training_data.time_step,
            input_steps=training_data.input_steps,
            devices=devices,
        )
    else:
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

    if invocation.validation_compare is not None:
        assert validation_step is not None
        assert comparison_params is not None
        common = {
            "validation_step": validation_step,
            "zero_residual_state": zero_residual_state,
            "training_data": training_data,
            "task_config": task_config,
            "config": config,
            "segment_ids": training_data.fixed_validation_segment_ids,
            "step": 0,
            "subset_policy": training_data.validation_subset_policy,
        }
        baseline_record = run_fixed_validation(
            residual_params=residual_params,
            role="zero_residual_baseline",
            **common,
        )
        checkpoint_record = run_fixed_validation(
            residual_params=comparison_params,
            role="checkpoint",
            **common,
        )
        baseline_by_horizon = baseline_record["loss_by_horizon"]
        checkpoint_by_horizon = checkpoint_record["loss_by_horizon"]
        improvement_by_horizon = {
            horizon: 100.0
            * (
                1.0
                - float(checkpoint_by_horizon[horizon])
                / float(baseline_by_horizon[horizon])
            )
            for horizon in baseline_by_horizon
        }
        baseline_loss = float(baseline_record["loss"])
        checkpoint_loss = float(checkpoint_record["loss"])
        payload = {
            "definition": (
                "Exact v24_Ilya fixed-validation sparse loss comparison using "
                "identical anchors, RNG keys, recurrent-state protocol, and "
                "horizon weights"
            ),
            "source_checkpoint": str(invocation.validation_compare),
            "baseline": baseline_record,
            "checkpoint": checkpoint_record,
            "improvement_pct": 100.0 * (1.0 - checkpoint_loss / baseline_loss),
            "improvement_pct_by_horizon": improvement_by_horizon,
        }
        output_path = config.run_dir / "validation_comparison.json"
        atomic_json_dump(payload, output_path)
        print(
            f"[v24_Ilya] fixed-validation comparison "
            f"baseline={baseline_loss:.6f} checkpoint={checkpoint_loss:.6f} "
            f"improvement={payload['improvement_pct']:.3f}% "
            f"by_horizon={improvement_by_horizon} output={output_path}",
            flush=True,
        )
        return output_path

    if invocation.baseline_validation_only:
        assert validation_step is not None
        record = run_fixed_validation(
            validation_step=validation_step,
            residual_params=residual_params,
            zero_residual_state=zero_residual_state,
            training_data=training_data,
            task_config=task_config,
            config=config,
            segment_ids=training_data.fixed_validation_segment_ids,
            step=0,
            role="zero_residual_baseline",
            subset_policy=training_data.validation_subset_policy,
        )
        record = {
            **record,
            "definition": (
                "Frozen GraphCast loss under the exact v24_Ilya fixed-validation "
                "protocol, produced by the freshly initialized zero-output "
                "residual model"
            ),
        }
        output_path = config.run_dir / "baseline_validation.json"
        atomic_json_dump(record, output_path)
        print(
            f"[v24_Ilya] zero-residual baseline validation "
            f"loss={float(record['loss']):.6f} "
            f"segments={int(record['selected_segments'])} "
            f"time={float(record['duration_seconds']):.1f}s "
            f"output={output_path}",
            flush=True,
        )
        return output_path

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
            f"[v24_Ilya] validation step={step} role={role} "
            f"loss={float(record['loss']):.6f} "
            f"segments={int(record['selected_segments'])} "
            f"time={float(record['duration_seconds']):.1f}s",
            flush=True,
        )
        return record

    print(
        f"[v24_Ilya] residual_params={n_residual_parameters:,} "
        f"segments={len(training_data.segments)} truth_prefix={config.truth_prefix_steps} "
        f"ar_tail={config.ar_tail_k} feedback={config.feedback_mode} "
        f"state_policy={config.temporal_state_policy}",
        flush=True,
    )
    memory_profile_pending = completed_step == 0

    if data_parallel:
        assert devices is not None
        replicated_params = replicate_tree(residual_params, devices)
        replicated_optimizer_state = replicate_tree(optimizer_state, devices)
        if checkpoint_replica_states is None:
            replica_states = replicate_tree(zero_residual_state, devices)
        else:
            replica_states = shard_replica_tree(checkpoint_replica_states, devices)

        print(
            f"[v24_Ilya] global_batch_size={config.distributed.global_batch_size} "
            f"anchors_per_update={anchors_per_update} "
            f"replica_groups={replica_group_count} "
            f"dropped_segments={training_data.dropped_replica_segments(config.distributed.num_devices)}",
            flush=True,
        )
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
            residual_params = unreplicate_tree(replicated_params)
            perform_validation(
                step=completed_step,
                segment_ids=training_data.fixed_validation_segment_ids,
                role="fixed_checkpoint",
                subset_policy=training_data.validation_subset_policy,
            )

        for step in range(completed_step + 1, config.max_steps + 1):
            if replica_cursor.segment_offset == 0 and has_consumed_update:
                replica_states = replicate_tree(zero_residual_state, devices)
            consumed_cursor = replica_cursor
            chunk = training_data.build_replica_chunk(
                replica_cursor,
                config,
                task_config,
            )
            rng_key, keys = derive_replica_step_keys(
                rng_key,
                num_replicas=config.distributed.num_devices,
                bptt_steps=config.bptt_steps,
            )
            started = time.monotonic()
            (
                replicated_params,
                replica_states,
                replicated_optimizer_state,
                mean_loss,
                averaged_gradient_norm,
                lane_losses,
                lane_loss_components,
                phase_seconds,
            ) = train_step(
                replicated_params,
                replica_states,
                replicated_optimizer_state,
                keys,
                chunk.input_frames,
                chunk.static_inputs,
                chunk.truths,
                chunk.forcings,
            )
            mean_loss = jax.block_until_ready(mean_loss)
            averaged_gradient_norm = jax.block_until_ready(averaged_gradient_norm)
            step_seconds = time.monotonic() - started
            replica_cursor = chunk.next_cursor
            has_consumed_update = True

            lane_losses = np.asarray(lane_losses, dtype=np.float64)
            lane_loss_components = np.asarray(
                lane_loss_components, dtype=np.float64
            )
            loss_value = float(jax.device_get(mean_loss))
            gradient_norm_value = float(jax.device_get(averaged_gradient_norm))
            should_validate = config.validation.enabled and (
                step % config.validation.every_steps == 0
                or step == config.max_steps
            )
            should_checkpoint = (
                step % config.checkpoint_every == 0
                or step == config.max_steps
                or should_validate
            )
            replica_divergence_checked = step == 1 or should_checkpoint
            diagnostics_started = time.monotonic()
            if replica_divergence_checked:
                parameter_divergence = replica_max_abs_difference(replicated_params)
                optimizer_divergence = replica_max_abs_difference(
                    replicated_optimizer_state
                )
            else:
                parameter_divergence = None
                optimizer_divergence = None
            phase_seconds = {
                **phase_seconds,
                "diagnostics": time.monotonic() - diagnostics_started,
            }
            device_memory = _device_memory_snapshot(72 * 1024**3)
            record = {
                "step": step,
                "loss": loss_value,
                "mean_loss": loss_value,
                "lane_losses": lane_losses.tolist(),
                "lane_loss_min": float(np.min(lane_losses)),
                "lane_loss_max": float(np.max(lane_losses)),
                "lane_loss_std": float(np.std(lane_losses)),
                "gradient_norm": gradient_norm_value,
                "averaged_gradient_norm": gradient_norm_value,
                "learning_rate": _learning_rate_value(learning_rate, step - 1),
                "step_seconds": step_seconds,
                "seconds_per_anchor": step_seconds / anchors_per_update,
                "global_batch_size": config.distributed.global_batch_size,
                "anchors_per_update": anchors_per_update,
                "cumulative_anchors_seen": step * anchors_per_update,
                "epoch": consumed_cursor.epoch,
                "group_index": consumed_cursor.group_index,
                "chunk_offset": consumed_cursor.segment_offset,
                "segment_ids": list(chunk.segment_ids),
                "raw_anchor_indices": [
                    values.tolist() for values in chunk.raw_anchor_indices
                ],
                "max_parameter_replica_divergence": parameter_divergence,
                "max_optimizer_replica_divergence": optimizer_divergence,
                "replica_divergence_checked": replica_divergence_checked,
                "phase_seconds": phase_seconds,
            }
            if config.loss_mode == "sparse_steps":
                record["loss_by_horizon"] = {
                    str(horizon): float(np.mean(lane_loss_components[position]))
                    for position, horizon in enumerate(
                        config.supervised_horizon_labels
                    )
                }
                record["lane_loss_by_horizon"] = {
                    str(horizon): lane_loss_components[position].tolist()
                    for position, horizon in enumerate(
                        config.supervised_horizon_labels
                    )
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
            horizon_text = (
                " loss_by_horizon="
                + ",".join(
                    f"{horizon}:{value:.5f}"
                    for horizon, value in record["loss_by_horizon"].items()
                )
                if config.loss_mode == "sparse_steps"
                else ""
            )
            divergence_text = (
                f"{parameter_divergence:.3e}"
                if parameter_divergence is not None
                else "not_checked"
            )
            print(
                f"step {step}/{config.max_steps} mean_loss {loss_value:.5f} "
                f"lane_loss_min/max/std {np.min(lane_losses):.5f}/"
                f"{np.max(lane_losses):.5f}/{np.std(lane_losses):.5f} "
                f"avg_grad_norm {gradient_norm_value:.4f} "
                f"segments={list(chunk.segment_ids)} offset={consumed_cursor.segment_offset} "
                f"anchors_seen={step * anchors_per_update} "
                f"replica_divergence={divergence_text} "
                f"step_time={step_seconds:.2f}s{horizon_text}",
                flush=True,
            )
            if should_checkpoint:
                residual_params = unreplicate_tree(replicated_params)
                optimizer_state = unreplicate_tree(replicated_optimizer_state)
                last_checkpoint = _save_data_parallel_checkpoint(
                    config=config,
                    completed_step=step,
                    residual_params=residual_params,
                    replica_states=replica_tree_to_host(replica_states),
                    optimizer_state=optimizer_state,
                    rng_key=rng_key,
                    cursor=replica_cursor,
                    active_segment_ids=training_data.active_replica_segment_ids(
                        replica_cursor, config.distributed.num_devices
                    ),
                    baseline_fingerprint=baseline_fingerprint,
                    manifest_fingerprint=training_data.manifest_fingerprint,
                    overlay_metadata=overlay_metadata,
                )
                print(f"[v24_Ilya] saved {last_checkpoint}", flush=True)
            if should_validate:
                perform_validation(
                    step=step,
                    segment_ids=training_data.fixed_validation_segment_ids,
                    role="fixed_checkpoint",
                    subset_policy=training_data.validation_subset_policy,
                )

        if last_checkpoint is None:
            raise RuntimeError("Data-parallel training completed without a checkpoint")
        print(f"[v24_Ilya] training complete at step {config.max_steps}")
        return last_checkpoint

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
            loss_components,
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
        loss_components = np.asarray(loss_components, dtype=np.float64)
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
        if config.loss_mode == "sparse_steps":
            record["loss_by_horizon"] = {
                str(horizon): float(loss_components[position])
                for position, horizon in enumerate(config.supervised_horizon_labels)
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
            horizon_text = (
                " loss_by_horizon="
                + ",".join(
                    f"{horizon}:{value:.5f}"
                    for horizon, value in record["loss_by_horizon"].items()
                )
                if config.loss_mode == "sparse_steps"
                else ""
            )
            print(
                f"step {step}/{config.max_steps} loss {loss_value:.5f} "
                f"grad_norm {gradient_norm_value:.4f} "
                f"step_time {step_seconds:.2f}s{horizon_text}",
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
            print(f"[v24_Ilya] saved {last_checkpoint}", flush=True)
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
    print(f"[v24_Ilya] training complete at step {config.max_steps}")
    return last_checkpoint
