"""Residual Mamba runtime and rollout construction."""

from __future__ import annotations

from pathlib import Path

import haiku as hk
import jax
import xarray

from src.models.graphcast.runtime import (
    ROOT,
    _clone_state,
    _constant_inputs,
    _dataset_to_numpy,
    _update_inputs,
    infer_family,
    load_checkpoint_and_stats,
    load_run_config,
)

from graphcast import casting, checkpoint, graphcast as gc, normalization  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    DirectResidualNormalizer,
    advance_residual_inputs,
    build_zero_residual_inputs,
)

MODEL_NAME = "residual_mamba"


def _residual_output_head_enabled(run_cfg: dict) -> bool:
    output_head = run_cfg.get("residual_training", {}).get("output_head", {})
    return bool(output_head.get("enabled", False))


def _temporal_kwargs(model_cfg, run_cfg: dict) -> dict:
    del model_cfg
    temporal_cfg = run_cfg.get("temporal_config", {})
    residual_cfg = run_cfg.get("residual_training", {})
    temporal_backbone = temporal_cfg.get("backbone", "none")
    temporal_d_inner = temporal_cfg.get("d_inner", None)
    if temporal_backbone == "mamba" and temporal_d_inner is None:
        raise ValueError(
            "Residual Mamba checkpoint is missing temporal_config.d_inner. "
            "Older checkpoints that only stored hidden_size are no longer supported."
        )
    temporal_location = temporal_cfg.get("location")
    if temporal_location is None:
        temporal_location = (
            "mesh_processor_interleaved" if temporal_backbone == "mamba" else "mesh_post_encoder"
        )
    return {
        "temporal_backbone": temporal_backbone,
        "temporal_location": temporal_location,
        "temporal_d_inner": temporal_d_inner,
        "temporal_d_state": temporal_cfg.get("d_state", 16),
        "temporal_d_conv": temporal_cfg.get("d_conv", 4),
        "temporal_dt_rank": temporal_cfg.get("dt_rank", "auto"),
        "temporal_bias": temporal_cfg.get("bias", False),
        "temporal_conv_bias": temporal_cfg.get("conv_bias", True),
        "temporal_layers": temporal_cfg.get("layers", 1),
        "temporal_dropout": temporal_cfg.get("dropout", 0.0),
        "temporal_stateful": bool(temporal_cfg.get("stateful", False)),
        "temporal_insert_count": temporal_cfg.get("insert_count", None),
        "zero_init_temporal_out": bool(residual_cfg.get("temporal_zero_init_out", False)),
        "residual_output_head": _residual_output_head_enabled(run_cfg),
    }


def _apply_temporal_config(predictor: gc.GraphCast, temporal_kwargs: dict) -> gc.GraphCast:
    if hasattr(predictor, "_temporal_backbone"):
        predictor._temporal_backbone = temporal_kwargs["temporal_backbone"]
        predictor._temporal_location = temporal_kwargs["temporal_location"]
        predictor._temporal_d_inner = temporal_kwargs["temporal_d_inner"]
        predictor._temporal_d_state = temporal_kwargs["temporal_d_state"]
        predictor._temporal_d_conv = temporal_kwargs["temporal_d_conv"]
        predictor._temporal_dt_rank = temporal_kwargs["temporal_dt_rank"]
        predictor._temporal_bias = temporal_kwargs["temporal_bias"]
        predictor._temporal_conv_bias = temporal_kwargs["temporal_conv_bias"]
        predictor._temporal_layers = temporal_kwargs["temporal_layers"]
        predictor._temporal_dropout = temporal_kwargs["temporal_dropout"]
        predictor._temporal_stateful = temporal_kwargs["temporal_stateful"]
        predictor._temporal_insert_count = temporal_kwargs["temporal_insert_count"]
        predictor._temporal_zero_init_out = temporal_kwargs["zero_init_temporal_out"]
    predictor._residual_output_head_enabled = temporal_kwargs["residual_output_head"]
    return predictor


def _build_one_step_bundle(ckpt_obj, stats, ckpt_path: Path, *, residual_mode: bool):
    model_cfg = ckpt_obj.model_config
    task_cfg = ckpt_obj.task_config
    params = ckpt_obj.params

    run_cfg = load_run_config(ckpt_path)
    temporal_kwargs = _temporal_kwargs(model_cfg, run_cfg)
    use_bf16 = run_cfg.get("precision", "bf16") == "bf16"

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = gc.GraphCast(model_cfg, task_cfg)
        predictor = _apply_temporal_config(predictor, temporal_kwargs)
        if use_bf16:
            predictor = casting.Bfloat16Cast(predictor)
        if residual_mode:
            predictor = DirectResidualNormalizer(
                predictor,
                stddev_by_level=stats["stddev_by_level"],
                mean_by_level=stats["mean_by_level"],
                diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            )
        else:
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=stats["diffs_stddev_by_level"],
                mean_by_level=stats["mean_by_level"],
                stddev_by_level=stats["stddev_by_level"],
            )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    apply_fn = jax.jit(run_forward.apply)

    def step(*, rng, inputs, targets_template, forcings, state=None):
        if state is None:
            _init_params, active_state = run_forward.init(rng, inputs, targets_template, forcings)
        else:
            active_state = state
        predictions, next_state = apply_fn(
            params=params,
            state=active_state,
            rng=rng,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )
        return _dataset_to_numpy(predictions), next_state

    return {
        "model_cfg": model_cfg,
        "task_cfg": task_cfg,
        "run_cfg": run_cfg,
        "params": params,
        "transformed": run_forward,
        "step": step,
    }


def _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path: Path):
    residual_bundle = _build_one_step_bundle(ckpt_obj, stats, ckpt_path, residual_mode=True)
    run_cfg = residual_bundle["run_cfg"]
    residual_cfg = run_cfg.get("residual_training", {})
    baseline_ckpt_path = residual_cfg.get("baseline_checkpoint")
    if not baseline_ckpt_path:
        raise ValueError(f"Residual checkpoint missing baseline_checkpoint in run_config: {ckpt_path}")
    baseline_ckpt = Path(baseline_ckpt_path)
    if not baseline_ckpt.is_absolute():
        baseline_ckpt = ROOT / baseline_ckpt
    with baseline_ckpt.open("rb") as f:
        baseline_ckpt_obj = checkpoint.load(f, gc.CheckPoint)
    baseline_bundle = _build_one_step_bundle(baseline_ckpt_obj, stats, baseline_ckpt, residual_mode=False)
    return residual_bundle, baseline_bundle


def build_run_jitted(ckpt_obj, stats, ckpt_path: Path):
    run_cfg = load_run_config(ckpt_path)
    family = infer_family(run_cfg)
    if family != MODEL_NAME:
        raise ValueError(f"Checkpoint family mismatch: expected {MODEL_NAME}, found {family}")

    residual_bundle, baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)
    task_cfg = residual_bundle["task_cfg"]
    model_cfg = residual_bundle["model_cfg"]

    def run_jitted(**kw):
        rng = kw["rng"]
        inputs = kw["inputs"]
        targets_template = kw["targets_template"]
        forcings = kw["forcings"]
        constant_inputs = _constant_inputs(inputs, targets_template, forcings)
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        target_times = targets_template.coords["time"]
        residual_state = None
        baseline_state = None
        preds = []
        residual_inputs = build_zero_residual_inputs(inputs, targets_template.isel(time=slice(0, 1)))
        step_keys = jax.random.split(rng, targets_template.sizes["time"] * 2)
        for step_i in range(targets_template.sizes["time"]):
            target_step = targets_template.isel(time=slice(step_i, step_i + 1))
            forcings_step = forcings.isel(time=slice(step_i, step_i + 1))
            all_inputs = xarray.merge([constant_inputs, rolling_inputs])
            baseline_pred, baseline_state = baseline_bundle["step"](
                rng=step_keys[2 * step_i],
                inputs=all_inputs,
                targets_template=target_step,
                forcings=forcings_step,
                state=baseline_state,
            )
            residual_pred, residual_state = residual_bundle["step"](
                rng=step_keys[2 * step_i + 1],
                inputs=residual_inputs,
                targets_template=target_step,
                forcings=forcings_step,
                state=residual_state,
            )
            full_pred = baseline_pred + residual_pred
            preds.append(full_pred.assign_coords(time=target_step.coords["time"]))
            residual_inputs = advance_residual_inputs(residual_inputs, residual_pred)
            rolling_inputs = _update_inputs(rolling_inputs, xarray.merge([full_pred, forcings_step]))
        return xarray.concat(preds, dim=target_times)

    return run_jitted, task_cfg, model_cfg, run_cfg


def build_training_equivalent_run_jitted(ckpt_obj, stats, ckpt_path: Path):
    """Build a residual rollout that matches residual segment training/eval semantics.

    The residual model is scored as a full forecast correction, but its input
    history is teacher-forced with target-minus-baseline residuals.
    """
    run_cfg = load_run_config(ckpt_path)
    family = infer_family(run_cfg)
    if family != MODEL_NAME:
        raise ValueError(f"Checkpoint family mismatch: expected {MODEL_NAME}, found {family}")

    residual_bundle, baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)
    task_cfg = residual_bundle["task_cfg"]
    model_cfg = residual_bundle["model_cfg"]

    def run_jitted(**kw):
        rng = kw["rng"]
        inputs = kw["inputs"]
        targets = kw["targets_template"]
        forcings = kw["forcings"]
        constant_inputs = _constant_inputs(inputs, targets, forcings)
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        target_times = targets.coords["time"]
        residual_state = None
        baseline_state = None
        preds = []
        residual_inputs = build_zero_residual_inputs(inputs, targets.isel(time=slice(0, 1)))
        step_keys = jax.random.split(rng, targets.sizes["time"] * 2)
        for step_i in range(targets.sizes["time"]):
            target_step = targets.isel(time=slice(step_i, step_i + 1))
            forcings_step = forcings.isel(time=slice(step_i, step_i + 1))
            all_inputs = xarray.merge([constant_inputs, rolling_inputs])
            baseline_pred, baseline_state = baseline_bundle["step"](
                rng=step_keys[2 * step_i],
                inputs=all_inputs,
                targets_template=target_step,
                forcings=forcings_step,
                state=baseline_state,
            )
            residual_pred, residual_state = residual_bundle["step"](
                rng=step_keys[2 * step_i + 1],
                inputs=residual_inputs,
                targets_template=target_step,
                forcings=forcings_step,
                state=residual_state,
            )
            full_pred = baseline_pred + residual_pred
            preds.append(full_pred.assign_coords(time=target_step.coords["time"]))
            residual_inputs = advance_residual_inputs(residual_inputs, target_step - baseline_pred)
            rolling_inputs = _update_inputs(rolling_inputs, xarray.merge([target_step, forcings_step]))
        return xarray.concat(preds, dim=target_times)

    return run_jitted, task_cfg, model_cfg, run_cfg


def build_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path: Path):
    run_cfg = load_run_config(ckpt_path)
    family = infer_family(run_cfg)
    if family != MODEL_NAME:
        raise ValueError(f"Warm-start truth-anchored runner requires residual Mamba checkpoint: {ckpt_path}")

    residual_bundle, baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)
    task_cfg = residual_bundle["task_cfg"]
    model_cfg = residual_bundle["model_cfg"]

    def initialize_context(*, inputs, targets_template, forcings):
        constant_inputs = _constant_inputs(inputs, targets_template, forcings)
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        return {
            "constant_inputs": constant_inputs,
            "rolling_inputs": rolling_inputs,
            "residual_inputs": build_zero_residual_inputs(inputs, targets_template.isel(time=slice(0, 1))),
            "baseline_state": None,
            "residual_state": None,
        }

    def _model_step(*, rng, context, target_step, forcings_step):
        all_inputs = xarray.merge([context["constant_inputs"], context["rolling_inputs"]])
        baseline_pred, baseline_state = baseline_bundle["step"](
            rng=rng[0],
            inputs=all_inputs,
            targets_template=target_step,
            forcings=forcings_step,
            state=context["baseline_state"],
        )
        residual_pred, residual_state = residual_bundle["step"](
            rng=rng[1],
            inputs=context["residual_inputs"],
            targets_template=target_step,
            forcings=forcings_step,
            state=context["residual_state"],
        )
        full_pred = baseline_pred + residual_pred
        next_context = {
            "constant_inputs": context["constant_inputs"],
            "rolling_inputs": context["rolling_inputs"],
            "residual_inputs": advance_residual_inputs(context["residual_inputs"], residual_pred),
            "baseline_state": baseline_state,
            "residual_state": residual_state,
        }
        return full_pred, next_context

    def truth_step(*, rng, context, target_step, forcings_step):
        full_pred, next_context = _model_step(
            rng=rng,
            context=context,
            target_step=target_step,
            forcings_step=forcings_step,
        )
        next_context["rolling_inputs"] = _update_inputs(
            next_context["rolling_inputs"],
            xarray.merge([target_step, forcings_step]),
        )
        return full_pred, next_context

    def branch_rollout(*, rng, context, targets_template, forcings):
        branch_context = {
            "constant_inputs": context["constant_inputs"],
            "rolling_inputs": context["rolling_inputs"].copy(deep=False),
            "residual_inputs": context["residual_inputs"].copy(deep=False),
            "baseline_state": _clone_state(context["baseline_state"]),
            "residual_state": _clone_state(context["residual_state"]),
        }
        preds = []
        step_keys = jax.random.split(rng, targets_template.sizes["time"] * 2)
        for step_i in range(targets_template.sizes["time"]):
            target_step = targets_template.isel(time=slice(step_i, step_i + 1))
            forcings_step = forcings.isel(time=slice(step_i, step_i + 1))
            full_pred, branch_context = _model_step(
                rng=(step_keys[2 * step_i], step_keys[2 * step_i + 1]),
                context=branch_context,
                target_step=target_step,
                forcings_step=forcings_step,
            )
            preds.append(full_pred.assign_coords(time=target_step.coords["time"]))
            branch_context["rolling_inputs"] = _update_inputs(
                branch_context["rolling_inputs"],
                xarray.merge([full_pred, forcings_step]),
            )
        return xarray.concat(preds, dim=targets_template.coords["time"])

    return {
        "task_cfg": task_cfg,
        "model_cfg": model_cfg,
        "run_cfg": run_cfg,
        "initialize_context": initialize_context,
        "truth_step": truth_step,
        "branch_rollout": branch_rollout,
    }


def build_training_equivalent_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path: Path):
    """Build a truth-anchored residual runner that matches residual training eval."""
    run_cfg = load_run_config(ckpt_path)
    family = infer_family(run_cfg)
    if family != MODEL_NAME:
        raise ValueError(f"Warm-start truth-anchored runner requires residual Mamba checkpoint: {ckpt_path}")

    residual_bundle, baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)
    task_cfg = residual_bundle["task_cfg"]
    model_cfg = residual_bundle["model_cfg"]

    def initialize_context(*, inputs, targets_template, forcings):
        constant_inputs = _constant_inputs(inputs, targets_template, forcings)
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        return {
            "constant_inputs": constant_inputs,
            "rolling_inputs": rolling_inputs,
            "residual_inputs": build_zero_residual_inputs(inputs, targets_template.isel(time=slice(0, 1))),
            "baseline_state": None,
            "residual_state": None,
        }

    def _model_step(*, rng, context, target_step, forcings_step):
        all_inputs = xarray.merge([context["constant_inputs"], context["rolling_inputs"]])
        baseline_pred, baseline_state = baseline_bundle["step"](
            rng=rng[0],
            inputs=all_inputs,
            targets_template=target_step,
            forcings=forcings_step,
            state=context["baseline_state"],
        )
        residual_pred, residual_state = residual_bundle["step"](
            rng=rng[1],
            inputs=context["residual_inputs"],
            targets_template=target_step,
            forcings=forcings_step,
            state=context["residual_state"],
        )
        full_pred = baseline_pred + residual_pred
        next_context = {
            "constant_inputs": context["constant_inputs"],
            "rolling_inputs": context["rolling_inputs"],
            "residual_inputs": advance_residual_inputs(context["residual_inputs"], target_step - baseline_pred),
            "baseline_state": baseline_state,
            "residual_state": residual_state,
        }
        return full_pred, next_context

    def truth_step(*, rng, context, target_step, forcings_step):
        full_pred, next_context = _model_step(
            rng=rng,
            context=context,
            target_step=target_step,
            forcings_step=forcings_step,
        )
        next_context["rolling_inputs"] = _update_inputs(
            next_context["rolling_inputs"],
            xarray.merge([target_step, forcings_step]),
        )
        return full_pred, next_context

    def branch_rollout(*, rng, context, targets_template, forcings):
        branch_context = {
            "constant_inputs": context["constant_inputs"],
            "rolling_inputs": context["rolling_inputs"].copy(deep=False),
            "residual_inputs": context["residual_inputs"].copy(deep=False),
            "baseline_state": _clone_state(context["baseline_state"]),
            "residual_state": _clone_state(context["residual_state"]),
        }
        preds = []
        step_keys = jax.random.split(rng, targets_template.sizes["time"] * 2)
        for step_i in range(targets_template.sizes["time"]):
            target_step = targets_template.isel(time=slice(step_i, step_i + 1))
            forcings_step = forcings.isel(time=slice(step_i, step_i + 1))
            full_pred, branch_context = _model_step(
                rng=(step_keys[2 * step_i], step_keys[2 * step_i + 1]),
                context=branch_context,
                target_step=target_step,
                forcings_step=forcings_step,
            )
            preds.append(full_pred.assign_coords(time=target_step.coords["time"]))
            branch_context["rolling_inputs"] = _update_inputs(
                branch_context["rolling_inputs"],
                xarray.merge([target_step, forcings_step]),
            )
        return xarray.concat(preds, dim=targets_template.coords["time"])

    return {
        "task_cfg": task_cfg,
        "model_cfg": model_cfg,
        "run_cfg": run_cfg,
        "initialize_context": initialize_context,
        "truth_step": truth_step,
        "branch_rollout": branch_rollout,
    }
