from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from src.models.graphcast.training.core.batching import BatchBuilder, build_batch_from_indices_vectorized
from src.models.graphcast.training.core.eval_selection import (
    EVAL_SUBSET_STRATIFIED_FIXED,
    select_eval_subset,
)
from src.models.graphcast.training.core.logging import _write_run_config
from src.models.graphcast.training.core.model import (
    advance_residual_inputs,
    build_zero_residual_inputs,
    build_predictor,
    build_residual_correction_predictor,
    gc,
    reset_residual_input_lanes,
    xarray_jax,
)
from src.models.graphcast.training.core.segments import (
    _advance_autoregressive_inputs,
    _build_chunk_batches,
    _loss_by_lane,
    _reset_dataset_lanes,
    _reset_temporal_state_lanes,
    _stop_gradient_dataset,
    build_full_segments,
    iter_eval_segment_chunks,
    segment_midpoint_times,
)

def _use_zero_init_temporal_out(cfg, temporal_backbone: str | None = None) -> bool:
    backbone = cfg.temporal_backbone if temporal_backbone is None else temporal_backbone
    return backbone == "mamba"


def build_loss_transform(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    norm_stats: dict[str, xr.Dataset],
    cfg,
) -> hk.TransformedWithState:
    def forward_fn(inputs, targets, forcings, is_training):
        predictor = build_residual_correction_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=bool(is_training),
            temporal_backbone=cfg.temporal_backbone,
            temporal_location=cfg.temporal_location,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank=cfg.temporal_dt_rank,
            temporal_bias=cfg.temporal_bias,
            temporal_conv_bias=cfg.temporal_conv_bias,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=cfg.temporal_dropout,
            temporal_stateful=cfg.temporal_stateful,
            temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=_use_zero_init_temporal_out(cfg, cfg.temporal_backbone),
        )
        return predictor.loss(inputs, targets, forcings)

    return hk.transform_with_state(forward_fn)


def build_loss_prediction_transform(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    norm_stats: dict[str, xr.Dataset],
    cfg,
    *,
    gradient_checkpointing: bool,
) -> hk.TransformedWithState:
    def forward_fn(inputs, targets, forcings, is_training):
        predictor = build_residual_correction_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=gradient_checkpointing and bool(is_training),
            temporal_backbone=cfg.temporal_backbone,
            temporal_location=cfg.temporal_location,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank=cfg.temporal_dt_rank,
            temporal_bias=cfg.temporal_bias,
            temporal_conv_bias=cfg.temporal_conv_bias,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=cfg.temporal_dropout,
            temporal_stateful=cfg.temporal_stateful,
            temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=_use_zero_init_temporal_out(cfg, cfg.temporal_backbone),
            autoregressive_loss_mode="none",
        )
        return predictor.loss_and_predictions(inputs, targets, forcings)

    return hk.transform_with_state(forward_fn)


def build_predict_transform(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    norm_stats: dict[str, xr.Dataset],
    cfg,
    *,
    temporal_backbone: str,
    temporal_location: str,
    temporal_d_inner: int | None,
    temporal_d_state: int,
    temporal_d_conv: int,
    temporal_dt_rank: str,
    temporal_bias: bool,
    temporal_conv_bias: bool,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool,
) -> hk.TransformedWithState:
    def predict_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone=temporal_backbone,
            temporal_location=temporal_location,
            temporal_d_inner=temporal_d_inner,
            temporal_d_state=temporal_d_state,
            temporal_d_conv=temporal_d_conv,
            temporal_dt_rank=temporal_dt_rank,
            temporal_bias=temporal_bias,
            temporal_conv_bias=temporal_conv_bias,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            temporal_stateful=temporal_stateful,
            temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=_use_zero_init_temporal_out(cfg, temporal_backbone),
        )
        return predictor(inputs, targets_template=targets, forcings=forcings)

    return hk.transform_with_state(predict_fn)


def build_eval_loss_transform(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    norm_stats: dict[str, xr.Dataset],
    cfg,
    *,
    temporal_backbone: str,
    temporal_location: str,
    temporal_d_inner: int | None,
    temporal_d_state: int,
    temporal_d_conv: int,
    temporal_dt_rank: str,
    temporal_bias: bool,
    temporal_conv_bias: bool,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool,
) -> hk.TransformedWithState:
    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_residual_correction_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone=temporal_backbone,
            temporal_location=temporal_location,
            temporal_d_inner=temporal_d_inner,
            temporal_d_state=temporal_d_state,
            temporal_d_conv=temporal_d_conv,
            temporal_dt_rank=temporal_dt_rank,
            temporal_bias=temporal_bias,
            temporal_conv_bias=temporal_conv_bias,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            temporal_stateful=temporal_stateful,
            temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=_use_zero_init_temporal_out(cfg, temporal_backbone),
        )
        return predictor.loss(inputs, targets, forcings)

    return hk.transform_with_state(forward_fn)


def compute_residual_targets(targets: xr.Dataset, baseline_predictions: xr.Dataset) -> xr.Dataset:
    return targets - baseline_predictions


def reconstruct_full_predictions(
    baseline_predictions: xr.Dataset,
    residual_predictions: xr.Dataset,
) -> xr.Dataset:
    return baseline_predictions + residual_predictions


def _constant_inputs(inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset) -> xr.Dataset:
    constant_inputs = inputs.drop_vars(targets_template.keys(), errors="ignore")
    constant_inputs = constant_inputs.drop_vars(forcings.keys(), errors="ignore")
    for name, var in constant_inputs.items():
        if "time" in var.dims:
            raise ValueError(
                f"Time-dependent input variable {name} must either be a forcing variable or target variable."
            )
    return constant_inputs


def _update_inputs(inputs: xr.Dataset, next_frame: xr.Dataset) -> xr.Dataset:
    num_inputs = inputs.sizes["time"]
    predicted_or_forced_inputs = next_frame[list(inputs.keys())]
    return (
        xr.concat([inputs, predicted_or_forced_inputs], dim="time")
        .tail(time=num_inputs)
        .assign_coords(time=inputs.coords["time"])
    )


def residual_autoregressive_final_horizon(
    residual_loss_prediction_transform,
    baseline_predict_transform,
    *,
    residual_params: hk.Params,
    baseline_params: hk.Params,
    residual_state: hk.State,
    baseline_state: hk.State,
    rng_key: jax.Array,
    inputs: xr.Dataset,
    targets: xr.Dataset,
    forcings: xr.Dataset,
    residual_inputs: xr.Dataset,
    is_training: bool,
) -> tuple[jax.Array, hk.State, xr.Dataset]:
    """Run residual AR training loss and return one-step teacher-forced carry state."""
    constant_inputs = _constant_inputs(inputs, targets, forcings)
    rolling_inputs = inputs.drop_vars(constant_inputs.keys())
    current_residual_inputs = residual_inputs
    current_residual_state = residual_state
    current_baseline_state = baseline_state
    step_keys = jax.random.split(rng_key, targets.sizes["time"] * 2)
    final_loss_by_lane = None
    carry_state = None
    teacher_carry_residual_inputs = None

    for step_i in range(targets.sizes["time"]):
        target_step = targets.isel(time=slice(step_i, step_i + 1))
        forcings_step = forcings.isel(time=slice(step_i, step_i + 1))
        full_inputs = xr.merge([constant_inputs, rolling_inputs])
        baseline_preds, current_baseline_state = baseline_predict_transform.apply(
            baseline_params,
            current_baseline_state,
            step_keys[2 * step_i],
            full_inputs,
            target_step,
            forcings_step,
            False,
        )
        residual_targets = compute_residual_targets(target_step, baseline_preds)
        (loss_and_diag, residual_preds), current_residual_state = residual_loss_prediction_transform.apply(
            residual_params,
            current_residual_state,
            step_keys[2 * step_i + 1],
            current_residual_inputs,
            residual_targets,
            forcings_step,
            is_training,
        )
        if step_i == 0:
            carry_state = current_residual_state
            teacher_carry_residual_inputs = advance_residual_inputs(residual_inputs, residual_targets)
        if step_i == targets.sizes["time"] - 1:
            final_loss_by_lane = xarray_jax.unwrap_data(loss_and_diag[0])

        current_residual_inputs = advance_residual_inputs(current_residual_inputs, residual_preds)
        full_preds = reconstruct_full_predictions(baseline_preds, residual_preds)
        rolling_inputs = _update_inputs(rolling_inputs, xr.merge([full_preds, forcings_step]))

    if final_loss_by_lane is None or carry_state is None or teacher_carry_residual_inputs is None:
        raise ValueError("Residual autoregressive rollout requires at least one target step.")
    return final_loss_by_lane, carry_state, teacher_carry_residual_inputs


def augment_run_config(
    out_dir: Path,
    *,
    segment_cfg: Any,
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    numpy_cache_active: bool,
    train_cache_estimate_gib: float | None,
    effective_train_batch_builder: str,
    effective_eval_batch_builder: str,
) -> None:
    _write_run_config(
        out_dir,
        segment_cfg.base_cfg,
        model_cfg,
        task_cfg,
        numpy_cache_active=numpy_cache_active,
        train_cache_estimate_gib=train_cache_estimate_gib,
        effective_train_batch_builder=effective_train_batch_builder,
        effective_eval_batch_builder=effective_eval_batch_builder,
    )
    path = out_dir / "run_config.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["segment_training"] = {
        "len_segment": segment_cfg.len_segment,
        "bptt_steps": segment_cfg.bptt_steps,
        "chunk_load_workers": segment_cfg.chunk_load_workers,
        "prefetch_chunks": 1,
        "eval_num_segments": segment_cfg.eval_num_segments,
        "final_eval_num_segments": segment_cfg.final_eval_num_segments,
        "eval_subset_policy": segment_cfg.eval_subset_policy,
        "eval_rotating_diagnostics": segment_cfg.eval_rotating_diagnostics,
        "shuffle_segments": True,
        "drop_short_tail_segments": True,
        "max_steps_unit": "optimizer_updates",
    }
    if segment_cfg.base_cfg.target_steps > 1:
        payload["autoregressive_training"] = {
            "enabled": True,
            "loss_mode": "rolling_delayed_horizon",
            "target_steps": int(segment_cfg.base_cfg.target_steps),
            "state_carry_steps": "rolling_stream_stop_gradient_chunks",
        }
    payload["residual_training"] = {
        "enabled": True,
        "training_target": segment_cfg.training_target,
        "residual_definition": "target_minus_frozen_baseline",
        "baseline_checkpoint": segment_cfg.baseline_ckpt,
        "baseline_rollout_mode": (
            "rolling_full_feedback_autoregressive" if segment_cfg.base_cfg.target_steps > 1 else "per_window_one_step"
        ),
        "autoregressive_feedback": "baseline_plus_predicted_residual",
        "bptt_residual_carry": (
            "predicted_residual_stream_stop_gradient_chunks"
            if segment_cfg.base_cfg.target_steps > 1
            else "teacher_forced_truth_minus_baseline"
        ),
        "eval_metric": "full_forecast_against_truth",
        "eval_loss_equivalence": "loss(residual_pred, target-baseline) == loss(baseline+residual_pred, target)",
        "trainable_init": "fresh" if segment_cfg.resume_ckpt is None else "resume_checkpoint",
        "temporal_zero_init_out": _use_zero_init_temporal_out(segment_cfg.base_cfg),
        "resume_checkpoint": segment_cfg.resume_ckpt,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_residual_eval(
    residual_eval_transform,
    baseline_predict_transform,
    params: hk.Params,
    baseline_params: hk.Params,
    rng: jax.Array,
    eval_ds: xr.Dataset,
    eval_indices: np.ndarray,
    *,
    eval_batch_size: int,
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
    len_segment: int,
    bptt_steps: int,
    progress_label: str,
    batch_builder: BatchBuilder = build_batch_from_indices_vectorized,
    chunk_load_workers: int = 1,
    load_executor=None,
    max_segments: int | None = None,
    subset_policy: str = EVAL_SUBSET_STRATIFIED_FIXED,
    subset_role: str = "fixed_checkpoint",
    subset_fold: int | None = None,
) -> dict[str, float]:
    rolling_ar = target_steps > 1
    target_load_steps = 1 if rolling_ar else target_steps
    eval_segments = build_full_segments(eval_indices, len_segment)
    if not eval_segments:
        raise ValueError(
            "No full eval segments after timestamp-contiguous filtering. "
            f"len_segment={len_segment}, valid_windows={len(eval_indices)}"
        )
    available_segments = len(eval_segments)
    selection = select_eval_subset(
        np.arange(available_segments, dtype=np.int64),
        max_segments,
        times=segment_midpoint_times(eval_ds, eval_segments),
        policy=subset_policy,
        role=subset_role,
        fold=subset_fold,
    )
    eval_segments = [eval_segments[int(position)] for position in selection.positions.tolist()]
    if not eval_segments:
        raise ValueError("No eval segments selected.")
    if selection.capped:
        readable_policy = selection.policy.replace("_", " ")
        print(
            f"[{progress_label}] using {readable_policy} "
            f"{len(eval_segments)}/{available_segments} validation segments"
        )
    residual_state_by_batch_size: dict[int, hk.State] = {}
    baseline_state_by_batch_size: dict[int, hk.State] = {}
    residual_inputs_by_batch_size: dict[int, xr.Dataset] = {}
    rolling_inputs_by_batch_size: dict[int, xr.Dataset] = {}
    rollout_age_by_batch_size: dict[int, jax.Array] = {}
    eval_fn_by_batch_size: dict[int, callable] = {}
    total_weighted_loss = 0.0
    total_windows = 0
    segment_len = len(eval_segments[0])
    n_lane_groups = (len(eval_segments) + eval_batch_size - 1) // eval_batch_size
    n_chunks_per_group = segment_len // bptt_steps
    n_chunks = n_lane_groups * n_chunks_per_group
    t_eval0 = time.time()

    for chunk_i, (chunk_indices, reset_mask_np) in enumerate(
        iter_eval_segment_chunks(
            eval_segments,
            batch_size=eval_batch_size,
            bptt_steps=bptt_steps,
            segment_ids=selection.item_ids,
        ),
        start=1,
    ):
        chunk_inputs, chunk_targets, chunk_forcings = _build_chunk_batches(
            eval_ds,
            chunk_indices,
            input_steps=input_steps,
            target_steps=target_load_steps,
            task_cfg=task_cfg,
            dt=dt,
            batch_builder=batch_builder,
            chunk_load_workers=chunk_load_workers,
            load_executor=load_executor,
        )
        batch_size = int(len(reset_mask_np))

        if batch_size not in residual_state_by_batch_size:
            rng, base_init_key, residual_init_key = jax.random.split(rng, 3)
            init_targets = chunk_targets[0].isel(time=slice(0, 1))
            init_forcings = chunk_forcings[0].isel(time=slice(0, 1))
            _, baseline_state_by_batch_size[batch_size] = baseline_predict_transform.init(
                base_init_key,
                chunk_inputs[0],
                init_targets,
                init_forcings,
                False,
            )
            residual_inputs_by_batch_size[batch_size] = build_zero_residual_inputs(
                chunk_inputs[0],
                init_targets,
            )
            _, residual_state_by_batch_size[batch_size] = residual_eval_transform.init(
                residual_init_key,
                residual_inputs_by_batch_size[batch_size],
                init_targets,
                init_forcings,
                False,
            )
            if rolling_ar:
                rolling_inputs_by_batch_size[batch_size] = chunk_inputs[0]
                rollout_age_by_batch_size[batch_size] = jnp.zeros((batch_size,), dtype=jnp.int32)
        if batch_size not in eval_fn_by_batch_size:
            residual_eval_state = residual_state_by_batch_size[batch_size]
            baseline_eval_state = baseline_state_by_batch_size[batch_size]

            if rolling_ar:
                @jax.jit
                def eval_chunk(
                    params,
                    baseline_params,
                    residual_state,
                    base_key,
                    eval_key,
                    chunk_inputs,
                    chunk_targets,
                    chunk_forcings,
                    residual_inputs,
                    rolling_inputs,
                    rollout_age,
                    reset_mask,
                ):
                    del base_key
                    current_state = _reset_temporal_state_lanes(residual_state, reset_mask)
                    current_residual_inputs = reset_residual_input_lanes(
                        residual_inputs,
                        chunk_targets[0],
                        reset_mask,
                    )
                    current_rolling_inputs = _reset_dataset_lanes(rolling_inputs, chunk_inputs[0], reset_mask)
                    current_age = jnp.where(reset_mask, jnp.zeros_like(rollout_age), rollout_age)
                    current_baseline_state = baseline_eval_state
                    weighted_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                    valid_count = jnp.asarray(0.0, dtype=jnp.float32)
                    eval_keys = jax.random.split(eval_key, len(chunk_inputs) * 2)
                    for bptt_i in range(len(chunk_inputs)):
                        baseline_preds, current_baseline_state = baseline_predict_transform.apply(
                            baseline_params,
                            current_baseline_state,
                            eval_keys[2 * bptt_i],
                            current_rolling_inputs,
                            chunk_targets[bptt_i],
                            chunk_forcings[bptt_i],
                            False,
                        )
                        baseline_preds = _stop_gradient_dataset(baseline_preds)
                        residual_targets = chunk_targets[bptt_i] - baseline_preds
                        (loss_and_diag, residual_preds), current_state = residual_eval_transform.apply(
                            params,
                            current_state,
                            eval_keys[2 * bptt_i + 1],
                            current_residual_inputs,
                            residual_targets,
                            chunk_forcings[bptt_i],
                            False,
                        )
                        next_age = current_age + 1
                        loss_by_lane = _loss_by_lane(loss_and_diag[0])
                        valid_weight = (next_age >= target_steps).astype(loss_by_lane.dtype)
                        weighted_loss_sum = weighted_loss_sum + jnp.sum(loss_by_lane * valid_weight)
                        valid_count = valid_count + jnp.sum(valid_weight)
                        current_residual_inputs = advance_residual_inputs(current_residual_inputs, residual_preds)
                        current_rolling_inputs = _advance_autoregressive_inputs(
                            current_rolling_inputs,
                            baseline_preds + residual_preds,
                            chunk_forcings[bptt_i],
                        )
                        current_age = next_age
                    return (
                        current_state,
                        _stop_gradient_dataset(current_residual_inputs),
                        _stop_gradient_dataset(current_rolling_inputs),
                        jax.lax.stop_gradient(current_age),
                        weighted_loss_sum,
                        valid_count,
                    )
            else:
                @jax.jit
                def eval_chunk(
                    params,
                    baseline_params,
                    residual_state,
                    base_key,
                    eval_key,
                    chunk_inputs,
                    chunk_targets,
                    chunk_forcings,
                    residual_inputs,
                    reset_mask,
                ):
                    current_state = _reset_temporal_state_lanes(residual_state, reset_mask)
                    current_residual_inputs = reset_residual_input_lanes(
                        residual_inputs,
                        chunk_targets[0],
                        reset_mask,
                    )
                    del base_key
                    eval_keys = jax.random.split(eval_key, len(chunk_inputs))
                    losses = []
                    for bptt_i in range(len(chunk_inputs)):
                        loss_by_lane, current_state, current_residual_inputs = residual_autoregressive_final_horizon(
                            residual_eval_transform,
                            baseline_predict_transform,
                            residual_params=params,
                            baseline_params=baseline_params,
                            residual_state=current_state,
                            baseline_state=baseline_eval_state,
                            rng_key=eval_keys[bptt_i],
                            inputs=chunk_inputs[bptt_i],
                            targets=chunk_targets[bptt_i],
                            forcings=chunk_forcings[bptt_i],
                            residual_inputs=current_residual_inputs,
                            is_training=False,
                        )
                        valid_lanes = ~reset_mask if bptt_i == 0 else jnp.ones_like(reset_mask, dtype=bool)
                        valid_weight = valid_lanes.astype(loss_by_lane.dtype)
                        losses.append((jnp.sum(loss_by_lane * valid_weight), jnp.sum(valid_weight)))
                    loss_sums, valid_counts = zip(*losses, strict=True)
                    return (
                        current_state,
                        current_residual_inputs,
                        jnp.sum(jnp.stack(loss_sums)),
                        jnp.sum(jnp.stack(valid_counts)),
                    )

            eval_fn_by_batch_size[batch_size] = eval_chunk

        rng, base_key, eval_key = jax.random.split(rng, 3)
        if rolling_ar:
            (
                next_state,
                next_residual_inputs,
                next_rolling_inputs,
                next_rollout_age,
                chunk_loss_sum,
                chunk_valid_count,
            ) = eval_fn_by_batch_size[batch_size](
                params,
                baseline_params,
                residual_state_by_batch_size[batch_size],
                base_key,
                eval_key,
                chunk_inputs,
                chunk_targets,
                chunk_forcings,
                residual_inputs_by_batch_size[batch_size],
                rolling_inputs_by_batch_size[batch_size],
                rollout_age_by_batch_size[batch_size],
                jnp.asarray(reset_mask_np),
            )
            residual_state_by_batch_size[batch_size] = next_state
            residual_inputs_by_batch_size[batch_size] = next_residual_inputs
            rolling_inputs_by_batch_size[batch_size] = next_rolling_inputs
            rollout_age_by_batch_size[batch_size] = next_rollout_age
        else:
            next_state, next_residual_inputs, chunk_loss_sum, chunk_valid_count = eval_fn_by_batch_size[batch_size](
                params,
                baseline_params,
                residual_state_by_batch_size[batch_size],
                base_key,
                eval_key,
                chunk_inputs,
                chunk_targets,
                chunk_forcings,
                residual_inputs_by_batch_size[batch_size],
                jnp.asarray(reset_mask_np),
            )
            residual_state_by_batch_size[batch_size] = next_state
            residual_inputs_by_batch_size[batch_size] = next_residual_inputs

        chunk_loss_sum_f = float(jax.device_get(chunk_loss_sum))
        chunk_valid_count_f = float(jax.device_get(chunk_valid_count))
        total_weighted_loss += chunk_loss_sum_f
        total_windows += int(chunk_valid_count_f)

        if chunk_i == 1 or chunk_i % 10 == 0 or chunk_i == n_chunks:
            elapsed = time.time() - t_eval0
            current_loss = chunk_loss_sum_f / max(chunk_valid_count_f, 1.0)
            print(
                f"[{progress_label}] chunk {chunk_i}/{n_chunks} "
                f"elapsed {elapsed:.1f}s current_loss {current_loss:.6f}"
            )

    if total_windows <= 0:
        raise ValueError("No residual eval windows remained after excluding zero-history reset steps.")

    return {
        "total": float(total_weighted_loss / total_windows),
        "segments": float(len(eval_segments)),
        "chunks": float(n_chunks),
        **selection.metadata(item_name="segments"),
    }
