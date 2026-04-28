from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import numpy as np
import pandas as pd
import xarray as xr

from src.models.graphcast.training.core.batching import BatchBuilder, build_batch_from_indices_vectorized
from src.models.graphcast.training.core.logging import _write_run_config
from src.models.graphcast.training.core.model import (
    build_predictor,
    build_residual_correction_predictor,
    gc,
    scalarize_loss,
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
            temporal_hidden_size=cfg.temporal_hidden_size,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank=cfg.temporal_dt_rank,
            temporal_bias=cfg.temporal_bias,
            temporal_conv_bias=cfg.temporal_conv_bias,
            temporal_layers=cfg.temporal_layers,
            temporal_dropout=cfg.temporal_dropout,
            temporal_stateful=cfg.temporal_stateful,
            zero_init_temporal_out=_use_zero_init_temporal_out(cfg, cfg.temporal_backbone),
        )
        return predictor.loss(inputs, targets, forcings)

    return hk.transform_with_state(forward_fn)


def build_predict_transform(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    norm_stats: dict[str, xr.Dataset],
    cfg,
    *,
    temporal_backbone: str,
    temporal_location: str,
    temporal_hidden_size: int,
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
            temporal_hidden_size=temporal_hidden_size,
            temporal_d_inner=temporal_d_inner,
            temporal_d_state=temporal_d_state,
            temporal_d_conv=temporal_d_conv,
            temporal_dt_rank=temporal_dt_rank,
            temporal_bias=temporal_bias,
            temporal_conv_bias=temporal_conv_bias,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            temporal_stateful=temporal_stateful,
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
    temporal_hidden_size: int,
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
            temporal_hidden_size=temporal_hidden_size,
            temporal_d_inner=temporal_d_inner,
            temporal_d_state=temporal_d_state,
            temporal_d_conv=temporal_d_conv,
            temporal_dt_rank=temporal_dt_rank,
            temporal_bias=temporal_bias,
            temporal_conv_bias=temporal_conv_bias,
            temporal_layers=temporal_layers,
            temporal_dropout=temporal_dropout,
            temporal_stateful=temporal_stateful,
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
        "shuffle_segments": True,
        "drop_short_tail_segments": True,
        "max_steps_unit": "optimizer_updates",
    }
    payload["residual_training"] = {
        "enabled": True,
        "training_target": segment_cfg.training_target,
        "residual_definition": "target_minus_frozen_baseline",
        "baseline_checkpoint": segment_cfg.baseline_ckpt,
        "baseline_rollout_mode": "per_window_one_step",
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
    progress_label: str,
    batch_builder: BatchBuilder = build_batch_from_indices_vectorized,
) -> dict[str, float]:
    losses: list[float] = []

    residual_state_by_batch_size: dict[int, hk.State] = {}
    baseline_state_by_batch_size: dict[int, hk.State] = {}
    eval_fn_by_batch_size: dict[int, callable] = {}

    n_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    t_eval0 = time.time()

    for batch_i, i in enumerate(range(0, len(eval_indices), eval_batch_size), start=1):
        idx = eval_indices[i : i + eval_batch_size]
        inputs, targets, forcings = batch_builder(
            eval_ds,
            indices=idx,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )
        batch_size = len(idx)

        if batch_size not in residual_state_by_batch_size:
            rng, base_init_key, residual_init_key = jax.random.split(rng, 3)
            _, baseline_state_by_batch_size[batch_size] = baseline_predict_transform.init(
                base_init_key,
                inputs,
                targets,
                forcings,
                False,
            )
            _, residual_state_by_batch_size[batch_size] = residual_eval_transform.init(
                residual_init_key,
                inputs,
                targets,
                forcings,
                False,
            )
        if batch_size not in eval_fn_by_batch_size:
            residual_eval_state = residual_state_by_batch_size[batch_size]
            baseline_eval_state = baseline_state_by_batch_size[batch_size]

            @jax.jit
            def eval_batch(params, baseline_params, base_key, eval_key, inputs, targets, forcings):
                baseline_preds, _ = baseline_predict_transform.apply(
                    baseline_params,
                    baseline_eval_state,
                    base_key,
                    inputs,
                    targets,
                    forcings,
                    False,
                )
                residual_targets = compute_residual_targets(targets, baseline_preds)
                loss_and_diag, _ = residual_eval_transform.apply(
                    params,
                    residual_eval_state,
                    eval_key,
                    inputs,
                    residual_targets,
                    forcings,
                    False,
                )
                return scalarize_loss(loss_and_diag[0])

            eval_fn_by_batch_size[batch_size] = eval_batch

        rng, base_key, eval_key = jax.random.split(rng, 3)
        loss_value = eval_fn_by_batch_size[batch_size](
            params,
            baseline_params,
            base_key,
            eval_key,
            inputs,
            targets,
            forcings,
        )
        loss = float(loss_value)
        losses.append(loss)

        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_loss {loss:.6f}"
            )

    return {"total": float(np.mean(losses))}
