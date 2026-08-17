"""Top-level v22_final evaluation orchestration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np

from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats
from src.models.mamba.training.param_utils import overlay_matching_params

from .checkpoint import (
    load_v22_final_checkpoint,
    resolve_residual_state,
    validate_param_tree_compatible,
)
from .config import ARCHITECTURE_ID, SCHEMA_VERSION, V22FinalEvalConfig
from .data import build_eval_batch, open_eval_dataset, valid_eval_indices
from .metrics import V22FinalMetricAccumulator
from .model import build_predictors
from .rollout import run_v22_final_rollout


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


def evaluate_v22_final(config: V22FinalEvalConfig) -> dict:
    """Evaluate one legacy v22 checkpoint and write its JSON result."""

    warmup_steps = config.effective_warmup_steps
    print(
        f"[v22_final] checkpoint={config.ckpt} mode={config.eval_mode} "
        f"warmup={warmup_steps} target_steps={config.target_steps} "
        f"feedback={'full' if config.is_full_feedback else 'baseline'} "
        f"reset_state={config.reset_state_after_warmup} "
        f"reset_state_every_step={config.reset_state_every_step}"
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

    candidates = valid_eval_indices(eval_data, config.sample_total_steps)
    chosen_indices = select_eval_indices(
        candidates,
        n_samples=config.n_samples,
        seed=config.seed,
        force_idx=config.force_idx,
    )
    print(
        f"[v22_final] valid anchors={len(candidates)} selected={len(chosen_indices)} "
        f"sample_horizon={config.sample_total_steps}"
    )

    sample_inputs, sample_targets, sample_forcings = build_eval_batch(
        eval_data,
        indices=[int(candidates[0])],
        target_steps=config.total_rollout_steps,
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
    baseline_params, _overlay_stats = overlay_matching_params(
        baseline_params,
        baseline_checkpoint.params,
        strict=True,
    )
    residual_params_init, zero_residual_state = predictors.residual.init(
        residual_key,
        sample_inputs,
        one_step_targets,
        one_step_forcings,
    )

    checkpoint = load_v22_final_checkpoint(config.ckpt)
    validate_param_tree_compatible(residual_params_init, checkpoint.residual_params)
    residual_params = checkpoint.residual_params
    residual_state_init, resolved_state_init = resolve_residual_state(
        config.residual_state_init,
        zero_residual_state,
        checkpoint,
    )
    parameter_count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(residual_params))
    print(
        f"[v22_final] loaded residual parameters={parameter_count:,} "
        f"state_init={resolved_state_init}"
    )

    @jax.jit
    def baseline_step(params, state, key, inputs, targets, forcings):
        return predictors.baseline.apply(params, state, key, inputs, targets, forcings)

    @jax.jit
    def residual_step(params, state, key, inputs, targets, forcings):
        return predictors.residual.apply(params, state, key, inputs, targets, forcings)

    metrics = V22FinalMetricAccumulator(
        config.target_steps,
        eval_data.dataset["lat"].values,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    for sample_number, index in enumerate(chosen_indices, start=1):
        inputs, targets, forcings = build_eval_batch(
            eval_data,
            indices=[index],
            target_steps=config.total_rollout_steps,
            task_config=task_config,
        )
        rng, rollout_key = jax.random.split(rng)
        rollout = run_v22_final_rollout(
            rng=rollout_key,
            inputs=inputs,
            all_targets=targets,
            all_forcings=forcings,
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
        )
        metric_targets = targets.isel(
            time=slice(warmup_steps, warmup_steps + config.target_steps)
        )
        metrics.update(
            metric_targets,
            rollout.baseline_prediction,
            rollout.full_prediction,
        )
        if sample_number <= 3 or sample_number % 4 == 0:
            print(
                f"[v22_final] sample {sample_number}/{len(chosen_indices)} index={index}",
                flush=True,
            )

    output = {
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
        "n_samples": config.n_samples,
        "evaluated_samples": len(chosen_indices),
        "chosen_idx": chosen_indices,
        "ckpt": str(config.ckpt),
        "ckpt_in": str(config.ckpt_in),
    }
    output.update(metrics.finalize())
    config.out_json.parent.mkdir(parents=True, exist_ok=True)
    config.out_json.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(f"[v22_final] wrote {config.out_json}")
    return output


def compare_metric_outputs(
    legacy: dict,
    candidate: dict,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Raise AssertionError when legacy and v22_final metric sections differ."""

    np.testing.assert_array_equal(candidate["chosen_idx"], legacy["chosen_idx"])
    for section in (
        "per_variable_per_step",
        "per_channel_per_step",
        "residual_diagnostics_per_variable",
    ):
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
