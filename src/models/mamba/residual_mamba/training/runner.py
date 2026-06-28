#!/usr/bin/env python3
"""Train GraphCast/Mamba on residual targets over shuffled chronological segments."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import json
import time
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr

from src.models.graphcast.training.core.batching import (
    infer_time_step,
    input_steps_from_duration,
    select_batch_builders,
)
from src.models.graphcast.training.core.dataset import (
    _training_cache_decision,
    maybe_cache_training_data,
    open_training_splits,
)
from src.models.graphcast.training.core.logging import (
    _filter_pairs_upto_step,
    _load_dict_series_upto_step,
    _load_json_list,
    _load_step_value_pairs,
    _load_train_losses,
    build_batch_builder_metadata,
    plot_loss_curves,
    sample_actual_usage,
    save_checkpoint,
    save_logs,
    save_usage_logs,
)
from src.models.graphcast.training.core.eval_selection import EVAL_SUBSET_STRATIFIED_ROTATING
from src.models.graphcast.training.core.model import (
    advance_residual_inputs,
    build_zero_residual_inputs,
    derive_model_config_from_checkpoint,
    load_graphcast_checkpoint,
    load_stats,
    reset_residual_input_lanes,
    validate_stats_coverage,
)
from src.models.graphcast.training.core.segments import (
    AR_LOSS_MODE_TAIL_UNIFORM,
    SegmentBatchScheduler,
    SegmentBlockBatchLoader,
    SegmentChunk,
    SegmentLoadStats,
    _advance_autoregressive_inputs,
    _build_chunk_batches,
    _chunk_ar_truth_prefix,
    _loss_by_lane,
    _reset_temporal_state_lanes,
    _save_chunk_timing_logs,
    _stop_gradient_dataset,
    _stop_gradient_temporal_state,
    build_full_segments,
    include_bptt_loss_step,
    valid_contiguous_final_input_indices,
)
from src.models.mamba.training.param_utils import overlay_matching_params
from src.models.mamba.residual_mamba.feedback import (
    RESIDUAL_AR_FEEDBACK_BASELINE,
    residual_physical_feedback,
)
from .config import ResidualSegmentRunConfig, parse_args
from .model import (
    augment_run_config,
    build_loss_prediction_transform,
    build_predict_transform,
    prepare_baseline_only_residual_targets,
    residual_autoregressive_final_horizon,
    run_residual_eval,
    should_checkpoint_residual_ar_step,
)


def _read_existing_output_head_enabled(out_dir: Path) -> bool:
    run_config_path = out_dir / "run_config.json"
    if not run_config_path.exists():
        return False
    with run_config_path.open("r", encoding="utf-8") as f:
        run_config = json.load(f)
    output_head = run_config.get("residual_training", {}).get("output_head", {})
    return bool(output_head.get("enabled", False))


def _resolve_residual_output_head(segment_cfg: ResidualSegmentRunConfig, out_dir: Path) -> bool:
    mode = segment_cfg.residual_output_head_mode
    if mode == "enabled":
        return True
    if mode == "disabled":
        return False
    if mode != "auto":
        raise ValueError(f"Unknown residual output head mode: {mode!r}")
    if segment_cfg.resume_ckpt:
        return _read_existing_output_head_enabled(out_dir)
    return True


def run_training(
    segment_cfg: ResidualSegmentRunConfig | None = None,
    *,
    argv: list[str] | None = None,
) -> None:
    if segment_cfg is None:
        segment_cfg = parse_args(argv)
    cfg = segment_cfg.base_cfg
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.residual_output_head = _resolve_residual_output_head(segment_cfg, out_dir)
    print(
        "Residual output head "
        f"{'enabled' if cfg.residual_output_head else 'disabled'} "
        f"(mode={segment_cfg.residual_output_head_mode})"
    )

    baseline_ckpt = load_graphcast_checkpoint(Path(segment_cfg.baseline_ckpt))
    resume_ckpt = load_graphcast_checkpoint(Path(segment_cfg.resume_ckpt)) if segment_cfg.resume_ckpt else None

    base_model_cfg = baseline_ckpt.model_config
    task_cfg = baseline_ckpt.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)

    model_cfg = derive_model_config_from_checkpoint(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.processor_msg_steps,
        hidden_layers=1,
    )
    norm_stats = load_stats(Path(cfg.stats_dir))
    validate_stats_coverage(task_cfg, norm_stats)
    train_ds, eval_ds = open_training_splits(cfg, task_cfg)

    dt_train = infer_time_step(train_ds)
    dt_eval = infer_time_step(eval_ds)
    if dt_train != dt_eval:
        raise ValueError(f"Train/eval time step mismatch: train={dt_train}, eval={dt_eval}")

    input_steps = input_steps_from_duration(task_cfg.input_duration, dt_train)
    if input_steps < 2:
        raise ValueError("Residual segment training expects at least two input frames.")
    target_steps = cfg.target_steps
    rolling_ar = target_steps > 1
    target_load_steps = 1 if rolling_ar else target_steps

    train_final_indices = valid_contiguous_final_input_indices(
        train_ds,
        input_steps=input_steps,
        target_steps=target_steps,
        dt=dt_train,
    )
    eval_final_indices = valid_contiguous_final_input_indices(
        eval_ds,
        input_steps=input_steps,
        target_steps=target_steps,
        dt=dt_train,
    )
    segments = build_full_segments(train_final_indices, segment_cfg.len_segment)
    eval_segments = build_full_segments(eval_final_indices, segment_cfg.len_segment)
    if not segments:
        raise ValueError(
            "No full training segments after timestamp-contiguous filtering. "
            f"len_segment={segment_cfg.len_segment}, valid_windows={len(train_final_indices)}"
        )
    if not eval_segments:
        raise ValueError(
            "No full eval segments after timestamp-contiguous filtering. "
            f"len_segment={segment_cfg.len_segment}, valid_windows={len(eval_final_indices)}"
        )
    print(
        "Prepared residual segment windows: "
        f"train_windows={len(train_final_indices)}, eval_windows={len(eval_final_indices)}, "
        f"train_segments={len(segments)}, eval_segments={len(eval_segments)}, "
        f"len_segment={segment_cfg.len_segment}, "
        f"bptt_steps={segment_cfg.bptt_steps}, input_steps={input_steps}, "
        f"target_steps={target_steps}, target_load_steps={target_load_steps}"
    )

    should_cache_train, train_cache_estimate_gib = _training_cache_decision(train_ds, cfg, task_cfg)
    train_ds, eval_ds = maybe_cache_training_data(train_ds, eval_ds, cfg, task_cfg)
    requested_batch_builder = cfg.batch_builder
    use_segment_block_loader = cfg.data_source == "prepared_array" and not should_cache_train
    if use_segment_block_loader and requested_batch_builder == "numpy":
        print(
            "[residual-segment-block] batch_builder=numpy requires a full train cache; "
            "using segment block loader with direct fallback for streaming data."
        )
        requested_batch_builder = "direct"

    builder_selection = select_batch_builders(
        train_ds,
        eval_ds,
        requested=requested_batch_builder,
        should_cache_train=should_cache_train,
        task_cfg=task_cfg,
        train_label="residual-segment-train",
        eval_label="residual-segment-eval",
    )
    train_batch_builder = builder_selection.train_builder
    eval_batch_builder = builder_selection.eval_builder
    numpy_cache_active = builder_selection.numpy_cache_active
    effective_train_batch_builder = builder_selection.effective_train_batch_builder
    effective_eval_batch_builder = builder_selection.effective_eval_batch_builder
    if use_segment_block_loader:
        effective_train_batch_builder = "segment_block"
        effective_eval_batch_builder = "segment_block"

    residual_loss_transform = build_loss_prediction_transform(
        model_cfg,
        task_cfg,
        norm_stats,
        cfg,
        gradient_checkpointing=True,
    )
    residual_eval_transform = build_loss_prediction_transform(
        model_cfg,
        task_cfg,
        norm_stats,
        cfg,
        gradient_checkpointing=False,
    )
    baseline_predict_transform = build_predict_transform(
        base_model_cfg,
        task_cfg,
        norm_stats,
        cfg,
        temporal_backbone="none",
        temporal_location="mesh_post_encoder",
        temporal_d_inner=None,
        temporal_d_state=cfg.temporal_d_state,
        temporal_d_conv=cfg.temporal_d_conv,
        temporal_dt_rank="auto",
        temporal_bias=False,
        temporal_conv_bias=True,
        temporal_layers=1,
        temporal_dropout=0.0,
        temporal_stateful=False,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    init_indices = [int(segments[lane % len(segments)][0]) for lane in range(cfg.batch_size)]
    sample_inputs, sample_targets, sample_forcings = train_batch_builder(
        train_ds,
        indices=init_indices,
        input_steps=input_steps,
        target_steps=target_load_steps,
        task_cfg=task_cfg,
        dt=dt_train,
    )
    sample_targets_step = sample_targets.isel(time=slice(0, 1))
    sample_forcings_step = sample_forcings.isel(time=slice(0, 1))
    residual_inputs_state = build_zero_residual_inputs(sample_inputs, sample_targets_step)
    params, state = residual_loss_transform.init(
        rng,
        residual_inputs_state,
        sample_targets_step,
        sample_forcings_step,
        True,
    )
    if cfg.resume_step is not None:
        assert resume_ckpt is not None
        params, overlay_stats = overlay_matching_params(params, resume_ckpt.params)
        print(f"Resuming residual model from step {cfg.resume_step} ({segment_cfg.resume_ckpt})")
        if overlay_stats.initialized:
            print(
                "Initialized "
                f"{overlay_stats.initialized} new residual output head parameter(s) "
                "while overlaying resume checkpoint."
            )
    else:
        print("Residual model uses fresh initialization; frozen baseline is used only for residual targets.")
        if cfg.temporal_backbone == "mamba":
            print("Residual Mamba fresh init uses zero-initialized temporal out_proj, so inserted Mamba starts as a no-op.")
        if cfg.residual_output_head:
            print("Residual output head is zero-initialized, so step-0 full forecast equals the frozen baseline.")

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)
    augment_run_config(
        out_dir,
        segment_cfg=segment_cfg,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        numpy_cache_active=numpy_cache_active,
        train_cache_estimate_gib=train_cache_estimate_gib,
        effective_train_batch_builder=effective_train_batch_builder,
        effective_eval_batch_builder=effective_eval_batch_builder,
    )
    batch_builder_metadata = build_batch_builder_metadata(
        requested_batch_builder=cfg.batch_builder,
        effective_train_batch_builder=effective_train_batch_builder,
        effective_eval_batch_builder=effective_eval_batch_builder,
        numpy_cache_active=numpy_cache_active,
    )

    _, baseline_train_state = baseline_predict_transform.init(
        rng,
        sample_inputs,
        sample_targets_step,
        sample_forcings_step,
        False,
    )
    memory_mode = getattr(cfg, "memory_mode", "standard")
    stop_baseline_gradient = memory_mode in ("conservative", "optimal")
    checkpoint_residual_ar_step = should_checkpoint_residual_ar_step(memory_mode)
    split_baseline_outside_bptt = rolling_ar and segment_cfg.residual_ar_feedback == RESIDUAL_AR_FEEDBACK_BASELINE
    if stop_baseline_gradient:
        print(
            "Stopping gradients through online frozen-baseline predictions "
            f"(memory_mode={memory_mode})."
        )
    if checkpoint_residual_ar_step:
        print(f"Checkpointing residual AR step body (memory_mode={memory_mode}).")
    if split_baseline_outside_bptt:
        print("Using split-JIT frozen baseline residual targets outside residual BPTT.")

    if rolling_ar:
        truth_prefix_steps = _chunk_ar_truth_prefix(target_steps, segment_cfg.bptt_steps)
        ar_loss_mode = getattr(segment_cfg, "autoregressive_loss_mode", AR_LOSS_MODE_TAIL_UNIFORM)
    else:
        ar_loss_mode = AR_LOSS_MODE_TAIL_UNIFORM

    if rolling_ar:
        if split_baseline_outside_bptt:
            @functools.partial(jax.jit)
            def prepare_train_residual_targets(
                baseline_params: hk.Params,
                rng_key: jax.Array,
                chunk_inputs: tuple[xr.Dataset, ...],
                chunk_targets: tuple[xr.Dataset, ...],
                chunk_forcings: tuple[xr.Dataset, ...],
            ) -> tuple[xr.Dataset, ...]:
                return prepare_baseline_only_residual_targets(
                    baseline_predict_transform,
                    baseline_params=baseline_params,
                    baseline_state=baseline_train_state,
                    rng_key=rng_key,
                    chunk_inputs=chunk_inputs,
                    chunk_targets=chunk_targets,
                    chunk_forcings=chunk_forcings,
                    truth_prefix_steps=truth_prefix_steps,
                )

            @functools.partial(jax.jit)
            def train_chunk(
                params: hk.Params,
                state: hk.State,
                opt_state: optax.OptState,
                rng_key: jax.Array,
                residual_targets: tuple[xr.Dataset, ...],
                chunk_forcings: tuple[xr.Dataset, ...],
                residual_inputs: xr.Dataset,
                reset_mask: jax.Array,
            ):
                state = _reset_temporal_state_lanes(state, reset_mask)
                residual_inputs = reset_residual_input_lanes(
                    residual_inputs,
                    residual_targets[0],
                    jnp.ones_like(reset_mask, dtype=bool),
                )

                def loss_fn(p, s, key):
                    current_state = s
                    current_residual_inputs = residual_inputs
                    weighted_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                    valid_count = jnp.asarray(0.0, dtype=jnp.float32)
                    keys = jax.random.split(key, segment_cfg.bptt_steps * 2)

                    def one_bptt_step(
                        p_step,
                        current_state_step,
                        current_residual_inputs_step,
                        residual_key,
                        residual_target_step,
                        forcing_step,
                        use_truth_residual_feedback: bool,
                    ):
                        (loss_and_diag, residual_preds), next_state = residual_loss_transform.apply(
                            p_step,
                            current_state_step,
                            residual_key,
                            current_residual_inputs_step,
                            residual_target_step,
                            forcing_step,
                            True,
                        )
                        residual_feedback = (
                            residual_target_step if use_truth_residual_feedback else residual_preds
                        )
                        next_residual_inputs = advance_residual_inputs(
                            current_residual_inputs_step,
                            residual_feedback,
                        )
                        return next_state, next_residual_inputs, _loss_by_lane(loss_and_diag[0])

                    one_bptt_step_fn = (
                        jax.checkpoint(one_bptt_step, static_argnums=(6,))
                        if checkpoint_residual_ar_step
                        else one_bptt_step
                    )

                    for bptt_i in range(segment_cfg.bptt_steps):
                        (
                            current_state,
                            current_residual_inputs,
                            loss_by_lane,
                        ) = one_bptt_step_fn(
                            p,
                            current_state,
                            current_residual_inputs,
                            keys[2 * bptt_i + 1],
                            residual_targets[bptt_i],
                            chunk_forcings[bptt_i],
                            bptt_i < truth_prefix_steps,
                        )
                        if include_bptt_loss_step(
                            ar_loss_mode,
                            bptt_i,
                            truth_prefix_steps,
                        ):
                            weighted_loss_sum = weighted_loss_sum + jnp.sum(loss_by_lane)
                            valid_count = valid_count + jnp.asarray(loss_by_lane.size, dtype=loss_by_lane.dtype)
                    loss = weighted_loss_sum / jnp.maximum(valid_count, 1.0)
                    return loss, current_state

                (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params,
                    state,
                    rng_key,
                )
                updates, new_opt_state = opt.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (
                    new_params,
                    _stop_gradient_temporal_state(new_state),
                    new_opt_state,
                    loss,
                )
        else:
            @functools.partial(jax.jit)
            def train_chunk(
                params: hk.Params,
                state: hk.State,
                opt_state: optax.OptState,
                rng_key: jax.Array,
                chunk_inputs: tuple[xr.Dataset, ...],
                chunk_targets: tuple[xr.Dataset, ...],
                chunk_forcings: tuple[xr.Dataset, ...],
                residual_inputs: xr.Dataset,
                reset_mask: jax.Array,
            ):
                state = _reset_temporal_state_lanes(state, reset_mask)
                residual_inputs = reset_residual_input_lanes(
                    residual_inputs,
                    chunk_targets[0],
                    jnp.ones_like(reset_mask, dtype=bool),
                )

                def loss_fn(p, s, key):
                    current_state = s
                    current_residual_inputs = residual_inputs
                    current_rolling_inputs = chunk_inputs[0]
                    current_baseline_state = baseline_train_state
                    weighted_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                    valid_count = jnp.asarray(0.0, dtype=jnp.float32)
                    keys = jax.random.split(key, segment_cfg.bptt_steps * 2)

                    def one_bptt_step(
                        p_step,
                        current_state_step,
                        current_residual_inputs_step,
                        current_baseline_state_step,
                        current_rolling_inputs_step,
                        baseline_key,
                        residual_key,
                        target_step,
                        forcing_step,
                        use_truth_residual_feedback: bool,
                        update_tail_rolling_inputs: bool,
                    ):
                        baseline_preds, next_baseline_state = baseline_predict_transform.apply(
                            baseline_ckpt.params,
                            current_baseline_state_step,
                            baseline_key,
                            current_rolling_inputs_step,
                            target_step,
                            forcing_step,
                            False,
                        )
                        if stop_baseline_gradient:
                            baseline_preds = _stop_gradient_dataset(baseline_preds)
                            next_baseline_state = jax.tree_util.tree_map(
                                jax.lax.stop_gradient,
                                next_baseline_state,
                            )
                        residual_targets = target_step - baseline_preds
                        (loss_and_diag, residual_preds), next_state = residual_loss_transform.apply(
                            p_step,
                            current_state_step,
                            residual_key,
                            current_residual_inputs_step,
                            residual_targets,
                            forcing_step,
                            True,
                        )
                        residual_feedback = residual_targets if use_truth_residual_feedback else residual_preds
                        next_residual_inputs = advance_residual_inputs(
                            current_residual_inputs_step,
                            residual_feedback,
                        )
                        if update_tail_rolling_inputs:
                            full_preds = baseline_preds + residual_preds
                            feedback_preds = residual_physical_feedback(
                                baseline_pred=baseline_preds,
                                full_pred=full_preds,
                                mode=segment_cfg.residual_ar_feedback,
                            )
                            next_rolling_inputs = _advance_autoregressive_inputs(
                                current_rolling_inputs_step,
                                feedback_preds,
                                forcing_step,
                            )
                        else:
                            next_rolling_inputs = current_rolling_inputs_step
                        return (
                            next_state,
                            next_baseline_state,
                            next_residual_inputs,
                            next_rolling_inputs,
                            _loss_by_lane(loss_and_diag[0]),
                        )

                    one_bptt_step_fn = (
                        jax.checkpoint(one_bptt_step, static_argnums=(9, 10))
                        if checkpoint_residual_ar_step
                        else one_bptt_step
                    )

                    for bptt_i in range(segment_cfg.bptt_steps):
                        if bptt_i < truth_prefix_steps:
                            current_rolling_inputs = chunk_inputs[bptt_i]
                        (
                            current_state,
                            current_baseline_state,
                            current_residual_inputs,
                            next_rolling_inputs,
                            loss_by_lane,
                        ) = one_bptt_step_fn(
                            p,
                            current_state,
                            current_residual_inputs,
                            current_baseline_state,
                            current_rolling_inputs,
                            keys[2 * bptt_i],
                            keys[2 * bptt_i + 1],
                            chunk_targets[bptt_i],
                            chunk_forcings[bptt_i],
                            bptt_i < truth_prefix_steps,
                            bptt_i < segment_cfg.bptt_steps - 1 and bptt_i + 1 >= truth_prefix_steps,
                        )
                        if include_bptt_loss_step(
                            ar_loss_mode,
                            bptt_i,
                            truth_prefix_steps,
                        ):
                            weighted_loss_sum = weighted_loss_sum + jnp.sum(loss_by_lane)
                            valid_count = valid_count + jnp.asarray(loss_by_lane.size, dtype=loss_by_lane.dtype)
                        if bptt_i < segment_cfg.bptt_steps - 1:
                            if bptt_i + 1 < truth_prefix_steps:
                                current_rolling_inputs = chunk_inputs[bptt_i + 1]
                            else:
                                current_rolling_inputs = next_rolling_inputs
                    loss = weighted_loss_sum / jnp.maximum(valid_count, 1.0)
                    return loss, current_state

                (loss, new_state), grads = (
                    jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key)
                )
                updates, new_opt_state = opt.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (
                    new_params,
                    _stop_gradient_temporal_state(new_state),
                    new_opt_state,
                    loss,
                )
    else:
        @functools.partial(jax.jit)
        def train_chunk(
            params: hk.Params,
            state: hk.State,
            opt_state: optax.OptState,
            rng_key: jax.Array,
            chunk_inputs: tuple[xr.Dataset, ...],
            chunk_targets: tuple[xr.Dataset, ...],
            chunk_forcings: tuple[xr.Dataset, ...],
            residual_inputs: xr.Dataset,
            reset_mask: jax.Array,
        ):
            state = _reset_temporal_state_lanes(state, reset_mask)
            residual_inputs = reset_residual_input_lanes(residual_inputs, chunk_targets[0], reset_mask)

            def loss_fn(p, s, key):
                current_state = s
                current_residual_inputs = residual_inputs
                weighted_loss_sum = jnp.asarray(0.0, dtype=jnp.float32)
                valid_count = jnp.asarray(0.0, dtype=jnp.float32)
                keys = jax.random.split(key, segment_cfg.bptt_steps)
                for bptt_i in range(segment_cfg.bptt_steps):
                    loss_by_lane, current_state, current_residual_inputs = residual_autoregressive_final_horizon(
                        residual_loss_transform,
                        baseline_predict_transform,
                        residual_params=p,
                        baseline_params=baseline_ckpt.params,
                        residual_state=current_state,
                        baseline_state=baseline_train_state,
                        rng_key=keys[bptt_i],
                        inputs=chunk_inputs[bptt_i],
                        targets=chunk_targets[bptt_i],
                        forcings=chunk_forcings[bptt_i],
                        residual_inputs=current_residual_inputs,
                        is_training=True,
                        residual_ar_feedback=segment_cfg.residual_ar_feedback,
                        stop_baseline_gradient=stop_baseline_gradient,
                        checkpoint_step=checkpoint_residual_ar_step,
                    )
                    valid_lanes = ~reset_mask if bptt_i == 0 else jnp.ones_like(reset_mask, dtype=bool)
                    valid_weight = valid_lanes.astype(loss_by_lane.dtype)
                    weighted_loss_sum = weighted_loss_sum + jnp.sum(loss_by_lane * valid_weight)
                    valid_count = valid_count + jnp.sum(valid_weight)
                loss = weighted_loss_sum / jnp.maximum(valid_count, 1.0)
                return loss, (current_state, current_residual_inputs)

            (loss, (new_state, new_residual_inputs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params,
                state,
                rng_key,
            )
            updates, new_opt_state = opt.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (
                new_params,
                _stop_gradient_temporal_state(new_state),
                new_opt_state,
                _stop_gradient_dataset(new_residual_inputs),
                loss,
            )

    step = cfg.resume_step if cfg.resume_step is not None else 0
    train_losses: list[tuple[int, float]] = []
    eval_losses: list[tuple[int, float]] = []
    eval_details: list[dict[str, Any]] = []
    step_times: list[tuple[int, float]] = []
    mem_usage: list[tuple[int, float]] = []
    actual_usage: list[dict[str, Any]] = []
    epoch_summaries: list[dict[str, Any]] = []
    chunk_timing: list[dict[str, Any]] = []

    if cfg.resume_step is not None:
        train_losses = _filter_pairs_upto_step(_load_train_losses(out_dir / "train_loss.json"), cfg.resume_step)
        eval_losses = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "eval_loss.json"), cfg.resume_step)
        step_times = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "step_times.json"), cfg.resume_step)
        mem_usage = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "memory_gib.json"), cfg.resume_step)
        eval_details = _load_dict_series_upto_step(out_dir / "eval_details.json", cfg.resume_step)
        actual_usage = _load_dict_series_upto_step(out_dir / "actual_usage.json", cfg.resume_step)
        epoch_summaries = _load_json_list(out_dir / "epoch_summary.json")
        chunk_timing = _load_dict_series_upto_step(out_dir / "chunk_timing.json", cfg.resume_step)

    best_eval_step: int | None = None
    best_eval_loss = float("inf")
    if eval_losses:
        best_eval_step, best_eval_loss = min(eval_losses, key=lambda x: (x[1], x[0]))
        print(f"[best:init] step {best_eval_step} val {best_eval_loss:.6f}")

    def maybe_save_best_checkpoint(eval_step: int, eval_total: float) -> None:
        nonlocal best_eval_step, best_eval_loss
        if eval_total >= best_eval_loss:
            return
        best_eval_step = int(eval_step)
        best_eval_loss = float(eval_total)
        save_checkpoint(
            out_dir,
            params=params,
            step=eval_step,
            model_cfg=model_cfg,
            task_cfg=task_cfg,
            description=baseline_ckpt.description,
            license_text=baseline_ckpt.license,
            filename="ckpt_best.npz",
        )
        with (out_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_eval_step": best_eval_step,
                    "best_eval_loss": best_eval_loss,
                    "best_checkpoint": "ckpt_best.npz",
                    "match_type": "full_forecast_equivalent",
                },
                f,
                indent=2,
            )
        print(f"[best] updated step {best_eval_step} val {best_eval_loss:.6f}")

    scheduler = SegmentBatchScheduler(
        segments,
        batch_size=cfg.batch_size,
        bptt_steps=segment_cfg.bptt_steps,
        seed=cfg.seed,
    )
    pass_start_step = step + 1
    pass_loss_accum: list[float] = []
    observed_epoch: int | None = None

    def save_all_logs() -> None:
        save_logs(
            out_dir,
            train_losses,
            eval_losses,
            eval_details,
            step_times,
            [],
            mem_usage,
            actual_usage,
            epoch_summaries,
        )
        _save_chunk_timing_logs(out_dir, chunk_timing)

    usage_log_flush_every = 50

    def maybe_flush_usage_logs() -> None:
        if actual_usage and (len(actual_usage) == 1 or len(actual_usage) % usage_log_flush_every == 0):
            save_usage_logs(out_dir, actual_usage)

    load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=segment_cfg.chunk_load_workers)
    prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    train_segment_loader = (
        SegmentBlockBatchLoader(
            train_ds,
            segments,
            input_steps=input_steps,
            target_steps=target_load_steps,
            task_cfg=task_cfg,
            dt=dt_train,
            load_executor=load_executor,
            max_workers=segment_cfg.chunk_load_workers,
            label="residual-segment-block-train",
        )
        if use_segment_block_loader
        else None
    )

    def load_chunk_payload(
        chunk: SegmentChunk,
    ) -> tuple[tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], np.ndarray, int, SegmentLoadStats]:
        t_load = time.time()
        if train_segment_loader is not None:
            chunk_inputs, chunk_targets, chunk_forcings, load_info = train_segment_loader.load_chunk(chunk)
        else:
            chunk_inputs, chunk_targets, chunk_forcings = _build_chunk_batches(
                train_ds,
                chunk.chunk_indices,
                input_steps=input_steps,
                target_steps=target_load_steps,
                task_cfg=task_cfg,
                dt=dt_train,
                batch_builder=train_batch_builder,
                chunk_load_workers=segment_cfg.chunk_load_workers,
                load_executor=load_executor,
            )
            load_info = SegmentLoadStats(load_s=time.time() - t_load)
        return chunk_inputs, chunk_targets, chunk_forcings, chunk.reset_mask, chunk.epoch, load_info

    def submit_next_chunk() -> concurrent.futures.Future:
        return prefetch_executor.submit(load_chunk_payload, scheduler.next_chunk())

    pending_chunk = submit_next_chunk()

    try:
        while step < cfg.max_steps:
            iteration_t0 = time.time()
            t_data = time.time()
            chunk_inputs, chunk_targets, chunk_forcings, reset_mask_np, chunk_epoch, load_info = pending_chunk.result()
            data_wait_s = time.time() - t_data

            if observed_epoch is None:
                observed_epoch = chunk_epoch
            elif chunk_epoch != observed_epoch and pass_loss_accum:
                epoch_summaries.append(
                    {
                        "pass": observed_epoch,
                        "steps": step - pass_start_step + 1,
                        "train_loss_mean": float(np.mean(pass_loss_accum)),
                        "time_per_step_mean": float(
                            np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                        ),
                        "mem_gib_max": float(
                            np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])
                        ),
                    }
                )
                observed_epoch = chunk_epoch
                pass_start_step = step + 1
                pass_loss_accum = []

            rng, step_key = jax.random.split(rng)
            next_chunk = submit_next_chunk() if step + 1 < cfg.max_steps else None
            t0 = time.time()
            if rolling_ar:
                if split_baseline_outside_bptt:
                    residual_targets = prepare_train_residual_targets(
                        baseline_ckpt.params,
                        step_key,
                        chunk_inputs,
                        chunk_targets,
                        chunk_forcings,
                    )
                    chunk_inputs = ()
                    chunk_targets = ()
                    params, state, opt_state, loss = train_chunk(
                        params,
                        state,
                        opt_state,
                        step_key,
                        residual_targets,
                        chunk_forcings,
                        residual_inputs_state,
                        jnp.asarray(reset_mask_np),
                    )
                else:
                    params, state, opt_state, loss = train_chunk(
                        params,
                        state,
                        opt_state,
                        step_key,
                        chunk_inputs,
                        chunk_targets,
                        chunk_forcings,
                        residual_inputs_state,
                        jnp.asarray(reset_mask_np),
                    )
            else:
                params, state, opt_state, residual_inputs_state, loss = train_chunk(
                    params,
                    state,
                    opt_state,
                    step_key,
                    chunk_inputs,
                    chunk_targets,
                    chunk_forcings,
                    residual_inputs_state,
                    jnp.asarray(reset_mask_np),
                )

            loss_f = float(loss)
            gpu_train_s = time.time() - t0
            iteration_wall_s = time.time() - iteration_t0

            step += 1
            train_losses.append((step, loss_f))
            pass_loss_accum.append(loss_f)
            step_times.append((step, gpu_train_s))
            chunk_timing.append(
                {
                    "step": step,
                    "data_wait_s": data_wait_s,
                    "gpu_train_s": gpu_train_s,
                    "iteration_wall_s": iteration_wall_s,
                    "batch_size": cfg.batch_size,
                    "bptt_steps": segment_cfg.bptt_steps,
                    "chunk_load_workers": segment_cfg.chunk_load_workers,
                    "loader": load_info.loader,
                    "load_s": load_info.load_s,
                    "cache_hits": load_info.cache_hits,
                    "cache_misses": load_info.cache_misses,
                    "loaded_gib": load_info.loaded_gib,
                }
            )

            usage = sample_actual_usage(step=step)
            actual_usage.append(usage)
            if usage.get("gpu_mem_gib") is not None:
                mem_usage.append((step, float(usage["gpu_mem_gib"])))
            maybe_flush_usage_logs()

            if step % 200 == 0:
                print(
                    f"step {step}/{cfg.max_steps} loss {loss_f:.6f} "
                    f"segment_epoch {chunk_epoch} reset_lanes {int(reset_mask_np.sum())} "
                    f"load={load_info.load_s:.3f}s data_wait={data_wait_s:.3f}s "
                    f"gpu={gpu_train_s:.3f}s iter={iteration_wall_s:.3f}s "
                    f"cache={load_info.cache_hits}/{load_info.cache_misses}"
                )

            if step % cfg.eval_every == 0:
                eval_metrics = run_residual_eval(
                    residual_eval_transform,
                    baseline_predict_transform,
                    params,
                    baseline_ckpt.params,
                    rng,
                    eval_ds,
                    eval_final_indices,
                    eval_batch_size=cfg.eval_batch_size,
                    input_steps=input_steps,
                    target_steps=target_steps,
                    task_cfg=task_cfg,
                    dt=dt_train,
                    len_segment=segment_cfg.len_segment,
                    bptt_steps=segment_cfg.bptt_steps,
                    progress_label=f"eval@step{step}",
                    batch_builder=eval_batch_builder,
                    chunk_load_workers=segment_cfg.chunk_load_workers,
                    load_executor=load_executor,
                    max_segments=segment_cfg.eval_num_segments,
                    subset_policy=segment_cfg.eval_subset_policy,
                    subset_role="fixed_checkpoint",
                    subset_fold=0,
                    residual_ar_feedback=segment_cfg.residual_ar_feedback,
                    rolling_loss_mode=ar_loss_mode,
                )
                eval_losses.append((step, eval_metrics["total"]))
                maybe_save_best_checkpoint(step, float(eval_metrics["total"]))
                eval_details.append({"step": step, **eval_metrics, **batch_builder_metadata})
                print(f"[eval] step {step} total {eval_metrics['total']:.6f}")
                if segment_cfg.eval_rotating_diagnostics and segment_cfg.eval_num_segments is not None:
                    rotating_eval = run_residual_eval(
                        residual_eval_transform,
                        baseline_predict_transform,
                        params,
                        baseline_ckpt.params,
                        rng,
                        eval_ds,
                        eval_final_indices,
                        eval_batch_size=cfg.eval_batch_size,
                        input_steps=input_steps,
                        target_steps=target_steps,
                        task_cfg=task_cfg,
                        dt=dt_train,
                        len_segment=segment_cfg.len_segment,
                        bptt_steps=segment_cfg.bptt_steps,
                        progress_label=f"eval_rotating@step{step}",
                        batch_builder=eval_batch_builder,
                        chunk_load_workers=segment_cfg.chunk_load_workers,
                        load_executor=load_executor,
                        max_segments=segment_cfg.eval_num_segments,
                        subset_policy=EVAL_SUBSET_STRATIFIED_ROTATING,
                        subset_role="rotating_diagnostic",
                        subset_fold=step // cfg.eval_every,
                        residual_ar_feedback=segment_cfg.residual_ar_feedback,
                        rolling_loss_mode=ar_loss_mode,
                    )
                    eval_details.append({"step": step, **rotating_eval, **batch_builder_metadata})
                    print(f"[eval_rotating] step {step} total {rotating_eval['total']:.6f}")
                plot_loss_curves(out_dir, train_losses, eval_losses)
                save_all_logs()

            if step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    out_dir,
                    params=params,
                    step=step,
                    model_cfg=model_cfg,
                    task_cfg=task_cfg,
                    description=baseline_ckpt.description,
                    license_text=baseline_ckpt.license,
                )

            if next_chunk is not None:
                pending_chunk = next_chunk
    finally:
        prefetch_executor.shutdown(wait=False, cancel_futures=True)
        load_executor.shutdown(wait=False, cancel_futures=True)

    if pass_loss_accum:
        epoch_summaries.append(
            {
                "pass": observed_epoch,
                "steps": step - pass_start_step + 1,
                "train_loss_mean": float(np.mean(pass_loss_accum)),
                "time_per_step_mean": float(
                    np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                ),
                "mem_gib_max": float(np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])),
            }
        )

    final_eval = run_residual_eval(
        residual_eval_transform,
        baseline_predict_transform,
        params,
        baseline_ckpt.params,
        rng,
        eval_ds,
        eval_final_indices,
        eval_batch_size=cfg.eval_batch_size,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
        len_segment=segment_cfg.len_segment,
        bptt_steps=segment_cfg.bptt_steps,
        progress_label="eval@final",
        batch_builder=eval_batch_builder,
        chunk_load_workers=segment_cfg.chunk_load_workers,
        max_segments=segment_cfg.final_eval_num_segments,
        subset_policy=segment_cfg.eval_subset_policy,
        subset_role="final",
        subset_fold=None,
        residual_ar_feedback=segment_cfg.residual_ar_feedback,
        rolling_loss_mode=ar_loss_mode,
    )
    eval_losses.append((step, final_eval["total"]))
    maybe_save_best_checkpoint(step, float(final_eval["total"]))
    eval_details.append({"step": step, "final": True, **final_eval, **batch_builder_metadata})

    save_checkpoint(
        out_dir,
        params=params,
        step=step,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        description=baseline_ckpt.description,
        license_text=baseline_ckpt.license,
    )
    save_all_logs()
    plot_loss_curves(out_dir, train_losses, eval_losses)
    print(f"Done. Final eval total {final_eval['total']:.6f}. Outputs in {out_dir}")


def main(argv: list[str] | None = None) -> None:
    run_training(argv=argv)


def train() -> None:
    main()


if __name__ == "__main__":
    main()
