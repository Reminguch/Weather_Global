"""Top-level v23_Ilya evaluation orchestration."""

from __future__ import annotations

import ctypes
import dataclasses
import gc
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats

from .checkpoint import (
    atomic_json_dump,
    atomic_pickle_dump,
    load_v23_Ilya_checkpoint,
    overlay_frozen_baseline_params,
    resolve_residual_state,
    validate_param_tree_compatible,
)
from .config import ARCHITECTURE_ID, SCHEMA_VERSION, V23IlyaEvalConfig
from .data import (
    build_eval_batch,
    iter_eval_steps,
    open_eval_dataset,
    valid_scored_eval_indices,
)
from .metrics import V23IlyaDeviceMetricReducer, V23IlyaMetricAccumulator
from .model import build_predictors
from .rollout import run_v23_Ilya_rollout


_LIBC = ctypes.CDLL(None)

_INTERMEDIATE_OMITTED_METRICS = (
    "rmsb_baseline",
    "rmsb_full",
    "improvement_pct_rmsb",
)


def _release_host_memory() -> None:
    gc.collect()
    malloc_trim = getattr(_LIBC, "malloc_trim", None)
    if malloc_trim is not None:
        malloc_trim(0)


def _current_rss_gib() -> float:
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    return float("nan")


def _dataset_to_host(dataset):
    return dataset.map(
        lambda data_array: data_array.copy(data=jax.device_get(data_array.data)),
        keep_attrs=True,
    )


def select_eval_indices(
    valid_indices: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    force_idx: int | None,
) -> list[int]:
    candidates = np.asarray(valid_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise ValueError("No valid evaluation anchors are available for the requested horizon")
    if force_idx is not None:
        if force_idx not in candidates:
            raise ValueError(
                f"force_idx={force_idx} is outside the valid anchor range "
                f"{int(candidates.min())}..{int(candidates.max())}"
            )
        return [int(force_idx)]
    if n_samples > candidates.size:
        raise ValueError(
            f"n_samples={n_samples} exceeds the {candidates.size} valid evaluation anchors"
        )
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(candidates, size=n_samples, replace=False))


def partial_result_path(out_json: Path) -> Path:
    """Return the sibling path used for an in-progress evaluation snapshot."""

    suffix = out_json.suffix or ".json"
    return out_json.with_name(f"{out_json.stem}.partial{suffix}")


def build_evaluation_result(
    *,
    base_output: Mapping[str, Any],
    metric_output: Mapping[str, Any],
    chosen_indices: Sequence[int],
    completed_samples: int,
    status: str,
) -> dict[str, Any]:
    """Combine evaluation metadata, progress, and accumulated metrics."""

    if status not in ("partial", "complete"):
        raise ValueError(f"Unsupported evaluation status {status!r}")
    total_samples = len(chosen_indices)
    if completed_samples < 0 or completed_samples > total_samples:
        raise ValueError(
            f"completed_samples={completed_samples} is outside 0..{total_samples}"
        )
    if status == "complete" and completed_samples != total_samples:
        raise ValueError(
            "A complete evaluation result must include every selected sample"
        )

    output = dict(base_output)
    output.update(
        evaluation_status=status,
        evaluated_samples=completed_samples,
        remaining_samples=total_samples - completed_samples,
        chosen_idx=[int(index) for index in chosen_indices],
        completed_chosen_idx=[
            int(index) for index in chosen_indices[:completed_samples]
        ],
        metrics_available=bool(metric_output),
        metrics_complete=status == "complete",
        intermediate_metrics_omitted=(
            [] if status == "complete" else list(_INTERMEDIATE_OMITTED_METRICS)
        ),
    )
    output.update(metric_output)
    return output


def write_evaluation_snapshot(
    path: Path,
    *,
    base_output: Mapping[str, Any],
    metric_output: Mapping[str, Any],
    chosen_indices: Sequence[int],
    completed_samples: int,
    status: str,
) -> dict[str, Any]:
    """Atomically publish a partial or complete evaluation result."""

    output = build_evaluation_result(
        base_output=base_output,
        metric_output=metric_output,
        chosen_indices=chosen_indices,
        completed_samples=completed_samples,
        status=status,
    )
    atomic_json_dump(output, path)
    return output


def evaluate_v23_Ilya(config: V23IlyaEvalConfig) -> dict:
    """Evaluate one v23_Ilya or warm-start-compatible checkpoint."""

    warmup_steps = config.effective_warmup_steps
    metric_backend = config.resolved_metric_backend
    print(
        f"[v23_Ilya] checkpoint={config.ckpt} mode={config.eval_mode} "
        f"warmup={warmup_steps} target_steps={config.target_steps} "
        f"feedback={'full' if config.is_full_feedback else 'baseline'} "
        f"reset_state={config.reset_state_after_warmup} "
        f"reset_state_every_step={config.reset_state_every_step} "
        f"metric_backend={metric_backend}"
    )

    baseline_checkpoint = load_graphcast_checkpoint(config.ckpt_in)
    task_config = baseline_checkpoint.task_config
    if config.input_duration is not None:
        task_config = dataclasses.replace(task_config, input_duration=config.input_duration)
    stats = load_stats(config.stats_dir)
    eval_data = open_eval_dataset(config, task_config)
    predictors = build_predictors(
        baseline_checkpoint.model_config,
        task_config,
        stats,
        config,
    )

    scored_candidates = valid_scored_eval_indices(
        eval_data,
        history_steps=config.effective_anchor_history_steps,
        target_steps=config.target_steps,
    )
    all_chosen_indices = select_eval_indices(
        scored_candidates,
        n_samples=config.n_samples,
        seed=config.seed,
        force_idx=config.force_idx,
    )
    chosen_indices = all_chosen_indices[
        config.anchor_shard_index :: config.anchor_shard_count
    ]
    if not chosen_indices:
        raise ValueError(
            f"Anchor shard {config.anchor_shard_index}/{config.anchor_shard_count} "
            "contains no selected anchors"
        )
    print(
        f"[v23_Ilya] valid scored anchors={len(scored_candidates)} "
        f"selected={len(all_chosen_indices)} shard="
        f"{config.anchor_shard_index}/{config.anchor_shard_count} "
        f"shard_samples={len(chosen_indices)} history="
        f"{config.effective_anchor_history_steps} sample_horizon="
        f"{config.sample_total_steps}"
    )

    sample_start_index = int(scored_candidates[0]) - warmup_steps
    sample_inputs, sample_targets, sample_forcings = build_eval_batch(
        eval_data,
        indices=[sample_start_index],
        target_steps=1,
        task_config=task_config,
    )
    one_step_targets = sample_targets.isel(time=slice(0, 1))
    one_step_forcings = sample_forcings.isel(time=slice(0, 1))

    rng = jax.random.PRNGKey(config.seed)
    rng, baseline_key, residual_key = jax.random.split(rng, 3)
    baseline_params, baseline_state_init = predictors.baseline.init(
        baseline_key,
        sample_inputs,
        one_step_targets,
        one_step_forcings,
    )
    baseline_params, _overlay_stats = overlay_frozen_baseline_params(
        baseline_params,
        baseline_checkpoint.params,
    )
    residual_params_init, zero_residual_state = predictors.residual.init(
        residual_key,
        sample_inputs,
        one_step_targets,
        one_step_forcings,
    )
    if warmup_steps > 0 and jax.tree_util.tree_leaves(baseline_state_init):
        raise ValueError(
            "Truth-streamed warmup skipping requires an empty frozen baseline state"
        )
    del sample_inputs, sample_targets, sample_forcings

    checkpoint = load_v23_Ilya_checkpoint(config.ckpt)
    validate_param_tree_compatible(residual_params_init, checkpoint.residual_params)
    residual_params = checkpoint.residual_params
    residual_state_init, resolved_state_init = resolve_residual_state(
        config.residual_state_init,
        zero_residual_state,
        checkpoint,
    )
    parameter_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(residual_params))
    print(
        f"[v23_Ilya] loaded residual parameters={parameter_count:,} "
        f"state_init={resolved_state_init}"
    )

    rollout_keys = {}
    for selected_index in all_chosen_indices:
        rng, rollout_key = jax.random.split(rng)
        rollout_keys[selected_index] = rollout_key

    @jax.jit
    def baseline_step(params, state, key, inputs, targets, forcings):
        return predictors.baseline.apply(params, state, key, inputs, targets, forcings)

    @jax.jit
    def residual_step(params, state, key, inputs, targets, forcings):
        return predictors.residual.apply(params, state, key, inputs, targets, forcings)

    metrics = V23IlyaMetricAccumulator(
        config.target_steps,
        eval_data.dataset["lat"].values,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        store_spatial_bias=not config.omit_rms_bias,
    )
    device_metrics = (
        V23IlyaDeviceMetricReducer(metrics)
        if metric_backend == "device"
        else None
    )
    base_output = {
        "architecture_id": ARCHITECTURE_ID,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format": checkpoint.checkpoint_format,
        "resolution": config.resolution,
        "seed": config.seed,
        "eval_mode": config.eval_mode,
        "warmup_steps": warmup_steps,
        "warmup_feedback": "truth" if warmup_steps > 0 else "none",
        "eval_feedback": "full" if config.is_full_feedback else "baseline",
        "rs_reset_after_warmup": config.reset_state_after_warmup,
        "rs_reset_every_step": config.reset_state_every_step,
        "baseline_branch": "pure_baseline_self_rollout",
        "residual_state_init": resolved_state_init,
        "residual_state_init_requested": config.residual_state_init,
        "residual_state_init_resolved": resolved_state_init,
        "target_steps": config.target_steps,
        "sample_total_steps": config.sample_total_steps,
        "anchor_history_steps": config.effective_anchor_history_steps,
        "anchor_index_semantics": "last_truth_input_before_scored_h1",
        "stream_block_steps": config.stream_block_steps,
        "metric_profile": (
            "exact_gc_rmse_mae_no_rms_bias"
            if config.omit_rms_bias
            else "full_legacy"
        ),
        "metric_backend_requested": config.metric_backend,
        "metric_backend": metric_backend,
        "metrics_omitted": (
            list(_INTERMEDIATE_OMITTED_METRICS) if config.omit_rms_bias else []
        ),
        "n_samples": config.n_samples,
        "anchor_shard_count": config.anchor_shard_count,
        "anchor_shard_index": config.anchor_shard_index,
        "global_chosen_idx": all_chosen_indices,
        "merge_state_path": (
            str(config.merge_state_out) if config.merge_state_out is not None else None
        ),
        "ckpt": str(config.ckpt),
        "ckpt_in": str(config.ckpt_in),
    }
    partial_json = partial_result_path(config.out_json)
    write_evaluation_snapshot(
        partial_json,
        base_output=base_output,
        metric_output={},
        chosen_indices=chosen_indices,
        completed_samples=0,
        status="partial",
    )
    print(f"[v23_Ilya] partial results={partial_json}", flush=True)

    for sample_number, index in enumerate(chosen_indices, start=1):
        sample_started = time.monotonic()
        rollout_start_index = index - warmup_steps
        inputs, _first_target, _first_forcing = build_eval_batch(
            eval_data,
            indices=[rollout_start_index],
            target_steps=1,
            task_config=task_config,
        )
        del _first_target, _first_forcing
        step_data = iter_eval_steps(
            eval_data,
            final_input_idx=rollout_start_index,
            total_steps=config.total_rollout_steps,
            block_steps=config.stream_block_steps,
            task_config=task_config,
        )
        rollout_key = rollout_keys[index]
        if device_metrics is not None:
            device_metrics.begin_sample()

        def consume_prediction(
            step_index,
            truth,
            baseline_prediction,
            full_prediction,
        ):
            if device_metrics is not None:
                device_metrics.update_step(
                    step_index,
                    truth,
                    baseline_prediction,
                    full_prediction,
                )
                return
            truth_host = _dataset_to_host(truth)
            baseline_host = _dataset_to_host(baseline_prediction)
            full_host = _dataset_to_host(full_prediction)
            metrics.update_step(
                step_index,
                truth_host,
                baseline_host,
                full_host,
            )
            del truth_host, baseline_host, full_host
            _release_host_memory()

        run_v23_Ilya_rollout(
            rng=rollout_key,
            inputs=inputs,
            all_targets=None,
            all_forcings=None,
            baseline_step=baseline_step,
            residual_step=residual_step,
            baseline_params=baseline_params,
            residual_params=residual_params,
            baseline_state_init=baseline_state_init,
            residual_state_init=residual_state_init,
            time_step=eval_data.time_step,
            input_steps=eval_data.input_steps,
            warmup_steps=warmup_steps,
            target_steps=config.target_steps,
            full_feedback=config.is_full_feedback,
            reset_state_after_warmup=config.reset_state_after_warmup,
            residual_alpha=config.residual_alpha,
            reset_state_every_step=config.reset_state_every_step,
            prediction_consumer=consume_prediction,
            retain_predictions=False,
            step_data=step_data,
            skip_baseline_warmup=True,
        )
        if device_metrics is not None:
            device_metrics.finish_sample()
        del inputs, step_data
        _release_host_memory()

        if sample_number < len(chosen_indices):
            partial_output = write_evaluation_snapshot(
                partial_json,
                base_output=base_output,
                metric_output=metrics.finalize(include_rms_bias=False),
                chosen_indices=chosen_indices,
                completed_samples=sample_number,
                status="partial",
            )
            graphcast_improvement = partial_output.get(
                "original_graphcast_loss",
                {},
            ).get("improvement_pct_rollout")
            print(
                f"[v23_Ilya] updated partial results samples="
                f"{sample_number}/{len(chosen_indices)} "
                f"graphcast_improvement_pct={graphcast_improvement}",
                flush=True,
            )
            del partial_output

        if sample_number <= 3 or sample_number % 4 == 0:
            peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            print(
                f"[v23_Ilya] sample {sample_number}/{len(chosen_indices)} index={index} "
                f"time={time.monotonic() - sample_started:.1f}s "
                f"current_rss={_current_rss_gib():.2f}GiB "
                f"peak_rss={peak_rss_gib:.2f}GiB",
                flush=True,
            )

    if config.merge_state_out is not None:
        print(
            f"[v23_Ilya] writing metric merge state={config.merge_state_out}",
            flush=True,
        )
        atomic_pickle_dump(metrics.export_merge_state(), config.merge_state_out)
        print(
            f"[v23_Ilya] wrote metric merge state={config.merge_state_out}",
            flush=True,
        )

    output = write_evaluation_snapshot(
        config.out_json,
        base_output=base_output,
        metric_output=metrics.finalize(),
        chosen_indices=chosen_indices,
        completed_samples=len(chosen_indices),
        status="complete",
    )
    partial_json.unlink(missing_ok=True)
    print(f"[v23_Ilya] wrote {config.out_json}")
    return output


def compare_metric_outputs(
    legacy: dict,
    candidate: dict,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Raise AssertionError when legacy and v23_Ilya metric sections differ."""

    np.testing.assert_array_equal(candidate["chosen_idx"], legacy["chosen_idx"])
    for section in (
        "original_graphcast_loss",
        "per_variable_per_step",
        "per_channel_per_step",
        "residual_diagnostics_per_variable",
    ):
        if section not in legacy and section not in candidate:
            continue
        if section not in legacy or section not in candidate:
            raise AssertionError(
                f"{section}: present in only one metric output"
            )
        _compare_nested(legacy[section], candidate[section], section, rtol=rtol, atol=atol)


def _compare_nested(legacy, candidate, path: str, *, rtol: float, atol: float) -> None:
    if isinstance(legacy, dict):
        if not isinstance(candidate, dict):
            raise AssertionError(f"{path}: expected mapping, got {type(candidate).__name__}")
        if set(legacy) != set(candidate):
            raise AssertionError(
                f"{path}: keys differ; legacy_only={sorted(set(legacy) - set(candidate))}, "
                f"candidate_only={sorted(set(candidate) - set(legacy))}"
            )
        for key in legacy:
            _compare_nested(
                legacy[key],
                candidate[key],
                f"{path}.{key}",
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(legacy, (list, tuple)):
        np.testing.assert_allclose(candidate, legacy, rtol=rtol, atol=atol, err_msg=path)
        return
    if isinstance(legacy, (int, float, np.number)):
        np.testing.assert_allclose(candidate, legacy, rtol=rtol, atol=atol, err_msg=path)
        return
    if candidate != legacy:
        raise AssertionError(f"{path}: candidate={candidate!r}, legacy={legacy!r}")
