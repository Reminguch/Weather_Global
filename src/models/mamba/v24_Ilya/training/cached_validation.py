"""Common streamed validation for online and cached stepwise checkpoints.

Both branches use the same fixed cached inputs, legacy per-step training loss,
and exact FP32 physical-field GraphCast metric. Only one prediction is copied
back at a time; no full prediction or spatial-diagnostic tape is retained.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from ..baseline_cache import host_dataset
from ..metrics import V24IlyaMetricAccumulator
from .validation import validation_step_keys


VALIDATION_PROTOCOL = "cached_stepwise_all_segments_v1"


def _corrected_prediction(truth, baseline, residual):
    residual = host_dataset(residual)
    if set(residual.data_vars) != set(truth.data_vars):
        raise ValueError("Residual prediction variables differ from validation truth")
    truth, baseline, residual = xr.align(truth, baseline, residual, join="exact", copy=False)
    return baseline.astype(np.float32, copy=False) + residual


def run_cached_validation(
    *, train_step, cached_data, config, params, zero_state, step: int,
) -> dict[str, Any]:
    """Evaluate every complete validation segment without touching train state.

    ``train_step.evaluate_step`` has the maintained supervised-forward contract:
    (params, state, key, inputs, residual_target, forcing) ->
    (legacy_step_loss, next_state, physical_residual_prediction).
    """
    if config.feedback_mode != "baseline" or config.loss_mode != "all_steps":
        raise ValueError("Common cached validation requires baseline feedback and all_steps loss")
    if config.distributed.global_batch_size != 1:
        raise ValueError("Common cached validation currently requires batch size one")
    segment_ids = np.arange(len(cached_data.training_data.validation_segments), dtype=np.int64)
    if segment_ids.size == 0:
        raise ValueError("Cached validation has no complete validation segments")
    started = time.monotonic()
    weights = np.asarray(config.normalized_supervised_weights, dtype=np.float32)
    total_loss = 0.0
    total_components = np.zeros(config.bptt_steps, dtype=np.float64)
    total_anchors = 0
    num_chunks = 0
    exact = V24IlyaMetricAccumulator(
        config.bptt_steps, np.asarray(cached_data.manifest["coordinates"]["lat"]),
        diffs_stddev_by_level=cached_data.context[3]["diffs_stddev_by_level"],
        store_spatial_bias=False,
    )
    for segment_id in segment_ids.tolist():
        state = jax.tree_util.tree_map(jnp.zeros_like, zero_state)
        for chunk_index, offset in enumerate(range(0, config.segment_steps, config.bptt_steps)):
            keys = validation_step_keys(
                seed=config.seed, segment_id=segment_id, chunk_index=chunk_index,
                bptt_steps=config.bptt_steps,
            )
            components = np.empty(config.bptt_steps, dtype=np.float32)
            consumed = 0
            with contextlib.closing(cached_data.iter_validation_steps(segment_id, offset)) as rows:
                for row in rows:
                    if row.step_index != consumed or consumed >= config.bptt_steps:
                        raise ValueError("Validation cache steps are not complete and ordered")
                    loss, state, residual = train_step.evaluate_step(
                        params, state, keys[consumed], row.inputs, row.target, row.forcing,
                    )
                    loss = float(jax.device_get(loss))
                    if not np.isfinite(loss):
                        raise FloatingPointError("Nonfinite cached validation training loss")
                    components[consumed] = loss
                    corrected = _corrected_prediction(row.truth, row.baseline, residual)
                    # Reuse exactly the maintained FP32 physical-field metric
                    # implementation, without its optional per-variable/bias
                    # diagnostics or their spatial accumulators.
                    exact._update_original_graphcast_loss_step(
                        consumed, row.truth.astype(np.float32, copy=False),
                        row.baseline.astype(np.float32, copy=False), corrected,
                    )
                    del residual, corrected
                    consumed += 1
            if consumed != config.bptt_steps:
                raise ValueError("Validation cache chunk is incomplete")
            # Match the maintained trainer's FP32 weighted time reduction, then
            # aggregate equal-size chunks in host FP64 like fixed validation.
            chunk_loss = float(np.sum(components * weights, dtype=np.float32))
            total_loss += chunk_loss * config.bptt_steps
            total_components += components.astype(np.float64) * config.bptt_steps
            total_anchors += config.bptt_steps
            num_chunks += 1
    exact_result = exact._finalize_original_graphcast_loss()
    first_ar_index = config.truth_prefix_steps - 1
    ar_baseline = np.asarray(exact_result["baseline_per_step"])[first_ar_index:]
    ar_full = np.asarray(exact_result["full_per_step"])[first_ar_index:]
    ar_result = {
        "first_chunk_step_index": first_ar_index,
        "lead_steps": list(range(1, len(ar_baseline) + 1)),
        "baseline_rollout": float(ar_baseline.mean()),
        "full_rollout": float(ar_full.mean()),
        "improvement_pct_rollout": float(100.0 * (1.0 - ar_full.sum() / ar_baseline.sum())),
    }
    return {
        "step": int(step), "loss": total_loss / total_anchors,
        "loss_by_horizon": {
            str(horizon): float(total_components[index] / total_anchors)
            for index, horizon in enumerate(config.supervised_horizon_labels)
        },
        "original_graphcast_loss": exact_result,
        "original_graphcast_loss_ar": ar_result,
        "duration_seconds": time.monotonic() - started,
        "role": "fixed_checkpoint", "validation_protocol": VALIDATION_PROTOCOL,
        "training_loss_precision": config.precision,
        "physical_metric_precision": "fp32", "eval_feedback": "baseline",
        "available_segments": int(segment_ids.size), "selected_segments": int(segment_ids.size),
        "segment_ids": segment_ids.tolist(), "num_chunks": num_chunks,
        "num_anchors": total_anchors, "bptt_steps": config.bptt_steps,
        "ar_tail_k": config.ar_tail_k, "feedback_mode": config.feedback_mode,
        "temporal_state_policy": config.temporal_state_policy,
        "subset_policy": "all_complete_segments",
        "subset_fingerprint": cached_data.training_data.fingerprint_validation_subset(segment_ids),
        "cache_manifest_sha256": cached_data.manifest["manifest_sha256"],
    }


def make_online_validation_callback(*, train_step, cached_data):
    """Adapt the same evaluator to the maintained online runner's callback."""
    expected_ids = np.arange(len(cached_data.training_data.validation_segments), dtype=np.int64)

    def callback(
        *, residual_params, zero_residual_state, config, segment_ids, step, role,
        training_data, **_unused,
    ):
        selected = np.asarray(segment_ids, dtype=np.int64)
        if not np.array_equal(selected, expected_ids):
            raise ValueError("Common online validation requires all cached validation segments")
        source_fingerprint = training_data.fingerprint_validation_subset(selected)
        cached_fingerprint = cached_data.training_data.fingerprint_validation_subset(expected_ids)
        if source_fingerprint != cached_fingerprint:
            raise ValueError("Online and cached validation subsets differ")
        for name in (
            "bptt_steps", "segment_steps", "ar_tail_k", "precision", "feedback_mode",
            "loss_mode", "temporal_state_policy", "seed", "architecture", "input_duration",
            "baseline_checkpoint", "stats_dir",
        ):
            if getattr(config, name) != getattr(cached_data.common, name):
                raise ValueError(f"Online and cached validation {name} differ")
        result = run_cached_validation(
            train_step=train_step, cached_data=cached_data, config=config,
            params=residual_params, zero_state=zero_residual_state, step=step,
        )
        result["role"] = role
        return result

    return callback
