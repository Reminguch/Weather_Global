"""Legacy shared rollout helpers for analysis scripts."""

from __future__ import annotations

from pathlib import Path

from src.models.graphcast.runtime import (
    _build_one_step_bundle as build_graphcast_one_step_bundle,
    _clone_state,
    build_run_jitted as build_graphcast_run_jitted,
    _constant_inputs,
    infer_family,
    load_run_config,
    suppress_graphcast_future_warnings,
    _update_inputs,
)
from src.models.mamba.gc_mamba.runtime import (
    _build_one_step_bundle as build_gc_mamba_one_step_bundle,
    build_run_jitted as build_gc_mamba_run_jitted,
)
from src.models.mamba.gc_mamba.legacy_runtime import (
    _build_one_step_bundle as build_legacy_gc_mamba_one_step_bundle,
    build_run_jitted as build_legacy_gc_mamba_run_jitted,
    is_legacy_gc_mamba_checkpoint,
)
from src.models.mamba.residual_mamba.runtime import (
    build_run_jitted as build_residual_mamba_run_jitted,
    build_truth_anchored_residual_runner,
)
import jax
import xarray


def build_run_jitted(ckpt_obj, stats, ckpt_path: Path):
    family = infer_family(load_run_config(ckpt_path))
    if family == "graphcast":
        return build_graphcast_run_jitted(ckpt_obj, stats, ckpt_path)
    if family == "gc_mamba":
        if is_legacy_gc_mamba_checkpoint(ckpt_obj):
            return build_legacy_gc_mamba_run_jitted(ckpt_obj, stats, ckpt_path)
        return build_gc_mamba_run_jitted(ckpt_obj, stats, ckpt_path)
    if family == "residual_mamba":
        return build_residual_mamba_run_jitted(ckpt_obj, stats, ckpt_path)
    raise ValueError(f"Unknown inference family: {family}")


def _build_truth_anchored_single_model_runner(bundle):
    task_cfg = bundle["task_cfg"]
    model_cfg = bundle["model_cfg"]

    def initialize_context(*, inputs, targets_template, forcings):
        constant_inputs = _constant_inputs(inputs, targets_template, forcings)
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        return {
            "constant_inputs": constant_inputs,
            "rolling_inputs": rolling_inputs,
            "state": None,
        }

    def _model_step(*, rng, context, target_step, forcings_step):
        if isinstance(rng, tuple):
            rng = rng[0]
        all_inputs = xarray.merge([context["constant_inputs"], context["rolling_inputs"]])
        pred_step, next_state = bundle["step"](
            rng=rng,
            inputs=all_inputs,
            targets_template=target_step,
            forcings=forcings_step,
            state=context["state"],
        )
        next_context = {
            "constant_inputs": context["constant_inputs"],
            "rolling_inputs": context["rolling_inputs"],
            "state": next_state,
        }
        return pred_step, next_context

    def truth_step(*, rng, context, target_step, forcings_step):
        pred_step, next_context = _model_step(
            rng=rng,
            context=context,
            target_step=target_step,
            forcings_step=forcings_step,
        )
        next_context["rolling_inputs"] = _update_inputs(
            next_context["rolling_inputs"],
            xarray.merge([target_step, forcings_step]),
        )
        return pred_step, next_context

    def branch_rollout(*, rng, context, targets_template, forcings):
        branch_context = {
            "constant_inputs": context["constant_inputs"],
            "rolling_inputs": context["rolling_inputs"].copy(deep=False),
            "state": _clone_state(context["state"]),
        }
        preds = []
        step_keys = jax.random.split(rng, targets_template.sizes["time"])
        for step_i in range(targets_template.sizes["time"]):
            target_step = targets_template.isel(time=slice(step_i, step_i + 1))
            forcings_step = forcings.isel(time=slice(step_i, step_i + 1))
            pred_step, branch_context = _model_step(
                rng=step_keys[step_i],
                context=branch_context,
                target_step=target_step,
                forcings_step=forcings_step,
            )
            preds.append(pred_step.assign_coords(time=target_step.coords["time"]))
            branch_context["rolling_inputs"] = _update_inputs(
                branch_context["rolling_inputs"],
                xarray.merge([pred_step, forcings_step]),
            )
        return xarray.concat(preds, dim=targets_template.coords["time"])

    return {
        "task_cfg": task_cfg,
        "model_cfg": model_cfg,
        "run_cfg": bundle["run_cfg"],
        "initialize_context": initialize_context,
        "truth_step": truth_step,
        "branch_rollout": branch_rollout,
    }


def build_truth_anchored_runner(ckpt_obj, stats, ckpt_path: Path):
    family = infer_family(load_run_config(ckpt_path))
    if family == "graphcast":
        return _build_truth_anchored_single_model_runner(
            build_graphcast_one_step_bundle(ckpt_obj, stats, ckpt_path)
        )
    if family == "gc_mamba":
        if is_legacy_gc_mamba_checkpoint(ckpt_obj):
            return _build_truth_anchored_single_model_runner(
                build_legacy_gc_mamba_one_step_bundle(ckpt_obj, stats, ckpt_path)
            )
        return _build_truth_anchored_single_model_runner(
            build_gc_mamba_one_step_bundle(ckpt_obj, stats, ckpt_path)
        )
    if family == "residual_mamba":
        return build_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path)
    raise ValueError(f"Unknown inference family: {family}")

__all__ = [
    "build_run_jitted",
    "build_truth_anchored_runner",
    "build_truth_anchored_residual_runner",
    "suppress_graphcast_future_warnings",
]
