"""Evaluate archived memory controls on explicit dates without converting checkpoints.

V22 predictors, checkpoint validation and rollout arithmetic remain native.
The shared date adapter includes cross-year truth; streamed host metrics avoid
retaining spatial bias arrays. V24 checkpoint/precision rules are not bypassed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import time

import jax
import numpy as np

from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats
from src.models.mamba.training.param_utils import overlay_matching_params
from src.models.mamba.v22_final.checkpoint import (
    load_v22_final_checkpoint, resolve_residual_state, validate_param_tree_compatible,
)
from src.models.mamba.v22_final.config import V22FinalEvalConfig
from src.models.mamba.v22_final.model import build_predictors
from src.models.mamba.v22_final.rollout import run_v22_final_rollout
from src.models.mamba.v24_Ilya.data import (
    build_eval_batch, open_eval_dataset, resolve_initialization_manifest,
)
from src.models.mamba.v24_Ilya.evaluation import (
    _dataset_to_host, _release_host_memory, build_initialization_loss_record,
    collect_initialization_variable_mse,
    initialization_rng_key, partial_result_path, write_evaluation_snapshot,
)
from src.models.mamba.v24_Ilya.metrics import V24IlyaMetricAccumulator


def evaluate_legacy_manifest(config, architecture, *, numerical_observer=None):
    if config.eval_mode != "cold_full" or config.effective_warmup_steps != 0:
        raise ValueError("Legacy manifest adapter supports cold full-feedback evaluation only")
    legacy = V22FinalEvalConfig(
        ckpt=config.ckpt, out_json=config.out_json, ckpt_in=config.ckpt_in,
        data_path=config.data_path, stats_dir=config.stats_dir, input_duration=config.input_duration,
        target_steps=config.target_steps, warmup_steps=0, eval_mode="cold_full",
        residual_state_init="zero", reset_state_every_step=config.reset_state_every_step,
        residual_alpha=config.residual_alpha, seed=config.seed, **architecture,
    )
    started = time.monotonic()
    baseline_checkpoint = load_graphcast_checkpoint(config.ckpt_in)
    task = baseline_checkpoint.task_config
    if config.input_duration is not None:
        task = dataclasses.replace(task, input_duration=config.input_duration)
    stats = load_stats(config.stats_dir)
    data = open_eval_dataset(config, task)
    records = resolve_initialization_manifest(data, config.initialization_manifest,
                                              target_steps=config.target_steps)
    selected = records[config.anchor_shard_index::config.anchor_shard_count]
    if not selected:
        raise ValueError("Empty legacy initialization shard")
    predictors = build_predictors(baseline_checkpoint.model_config, task, stats, legacy)
    inputs, targets, forcings = build_eval_batch(
        data, indices=[records[0]["dataset_index"]], target_steps=1, task_config=task,
    )
    _, bk, rk = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    bp, bs = predictors.baseline.init(bk, inputs, targets, forcings)
    bp, _ = overlay_matching_params(bp, baseline_checkpoint.params, strict=True)
    rp_init, zero_state = predictors.residual.init(rk, inputs, targets, forcings)
    checkpoint = load_v22_final_checkpoint(config.ckpt)
    validate_param_tree_compatible(rp_init, checkpoint.residual_params)
    state, resolved = resolve_residual_state("zero", zero_state, checkpoint)
    if numerical_observer is not None:
        numerical_observer.setup(
            config=config,
            baseline_model_config=baseline_checkpoint.model_config,
            task_config=task,
            baseline_params=bp,
            residual_params=checkpoint.residual_params,
            baseline_state=bs,
            residual_state=state,
            stats=stats,
        )
    del inputs, targets, forcings, rp_init

    baseline_step = jax.jit(predictors.baseline.apply)
    residual_step = jax.jit(predictors.residual.apply)
    metrics = V24IlyaMetricAccumulator(config.target_steps, data.dataset.lat.values,
                                     diffs_stddev_by_level=stats["diffs_stddev_by_level"],
                                     store_spatial_bias=False)
    losses, runtimes = [], []
    base = dict(
        architecture_id="v22_final", checkpoint_format=checkpoint.checkpoint_format,
        model_id=config.model_id, ckpt=str(config.ckpt), ckpt_in=str(config.ckpt_in),
        eval_mode="cold_full", eval_feedback="full", warmup_steps=0,
        residual_state_init=resolved, rs_reset_every_step=config.reset_state_every_step,
        target_steps=config.target_steps, n_samples=len(records),
        anchor_shard_index=config.anchor_shard_index, anchor_shard_count=config.anchor_shard_count,
        initialization_manifest_sha256=hashlib.sha256(config.initialization_manifest.read_bytes()).hexdigest(),
        global_initialization_times=[r["initialization_time"] for r in records],
        original_graphcast_loss_per_initialization=losses, sample_elapsed_seconds=runtimes,
        variable_mse_metadata=metrics.variable_mse_metadata,
        rng_scheme="sha256_utc_stream_fold_in_v1", setup_elapsed_seconds=time.monotonic()-started,
        numerical_contract="native_v22_predictors_and_rollout_with_shared_exact_host_metrics",
    )
    indices = [r["dataset_index"] for r in selected]
    for number, record in enumerate(selected, 1):
        tick = time.monotonic()
        inputs, targets, forcings = build_eval_batch(
            data, indices=[record["dataset_index"]], target_steps=config.target_steps, task_config=task,
        )
        rollout_key = initialization_rng_key(config.seed, record["initialization_time"], stream="rollout")
        if numerical_observer is not None:
            numerical_observer.initialization(record, inputs, rollout_key)
        rollout = run_v22_final_rollout(
            rng=rollout_key,
            inputs=inputs, all_targets=targets, all_forcings=forcings,
            baseline_step=baseline_step, residual_step=residual_step,
            baseline_params=bp, residual_params=checkpoint.residual_params,
            baseline_state_init=bs, residual_state_init=state,
            time_step=data.time_step, input_steps=data.input_steps, warmup_steps=0,
            target_steps=config.target_steps, full_feedback=True, reset_state_after_warmup=False,
            residual_alpha=config.residual_alpha, reset_state_every_step=config.reset_state_every_step,
        )
        sample = {"baseline": [], "full": []}
        sample_variable_mse = {}
        for lead in range(config.target_steps):
            fields = [_dataset_to_host(ds.isel(time=slice(lead, lead+1)))
                      for ds in (targets, rollout.baseline_prediction, rollout.full_prediction)]
            exact = metrics.update_step(lead, *fields)
            if exact is None:
                raise ValueError("Manifest evaluation requires exact original GraphCast losses")
            for branch in sample:
                values = np.asarray(exact[branch])
                if values.size != 1:
                    raise ValueError("Expected one initialization per loss contribution")
                sample[branch].append(float(values.reshape(-1)[0]))
            collect_initialization_variable_mse(
                sample_variable_mse, step_index=lead, target_steps=config.target_steps,
                contributions=metrics.last_step_variable_mse,
            )
            if numerical_observer is not None:
                numerical_observer.prediction(record, lead, *fields)
            del fields
        losses.append(build_initialization_loss_record(record, sample, variable_mse=sample_variable_mse))
        del inputs, targets, forcings, rollout
        _release_host_memory()
        runtimes.append(time.monotonic()-tick)
        complete = number == len(selected)
        result = write_evaluation_snapshot(
            config.out_json if complete else partial_result_path(config.out_json),
            base_output=base, metric_output=metrics.finalize(include_rms_bias=False),
            chosen_indices=indices, completed_samples=number,
            status="complete" if complete else "partial",
        )
        print(f"[v22_manifest] {config.model_id} {number}/{len(selected)} "
              f"{record['initialization_time']} {runtimes[-1]:.1f}s", flush=True)
    return result
