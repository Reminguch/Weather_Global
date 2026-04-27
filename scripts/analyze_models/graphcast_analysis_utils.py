from __future__ import annotations

import json
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import numpy as np
import xarray


ROOT = Path(__file__).resolve().parents[2]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
TRAINING_DIR = ROOT / "scripts" / "training"

for path in (ROOT, GRAPHCAST_LOCAL, TRAINING_DIR):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from graphcast import casting, checkpoint, graphcast as gc, normalization
from graphcast_train.model import DirectResidualNormalizer


def _load_run_config(ckpt_path: Path) -> dict[str, Any]:
    run_cfg_path = ckpt_path.parent / "run_config.json"
    if not run_cfg_path.exists():
        return {}
    with run_cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _temporal_kwargs(model_cfg, run_cfg: dict[str, Any]) -> dict[str, Any]:
    temporal_cfg = run_cfg.get("temporal_config", {})
    residual_cfg = run_cfg.get("residual_training", {})
    return {
        "temporal_backbone": temporal_cfg.get("backbone", "none"),
        "temporal_location": temporal_cfg.get("location", "mesh_post_encoder"),
        "temporal_hidden_size": temporal_cfg.get("hidden_size", model_cfg.latent_size),
        "temporal_d_inner": temporal_cfg.get("d_inner", None),
        "temporal_d_state": temporal_cfg.get("d_state", 16),
        "temporal_d_conv": temporal_cfg.get("d_conv", 4),
        "temporal_dt_rank": temporal_cfg.get("dt_rank", "auto"),
        "temporal_bias": temporal_cfg.get("bias", False),
        "temporal_conv_bias": temporal_cfg.get("conv_bias", True),
        "temporal_layers": temporal_cfg.get("layers", 1),
        "temporal_dropout": temporal_cfg.get("dropout", 0.0),
        "temporal_stateful": bool(temporal_cfg.get("stateful", False)),
        "zero_init_temporal_out": bool(residual_cfg.get("temporal_zero_init_out", False)),
    }


def _apply_temporal_config(predictor: gc.GraphCast, temporal_kwargs: dict[str, Any]) -> gc.GraphCast:
    if hasattr(predictor, "_temporal_backbone"):
        predictor._temporal_backbone = temporal_kwargs["temporal_backbone"]
        predictor._temporal_location = temporal_kwargs["temporal_location"]
        predictor._temporal_hidden_size = temporal_kwargs["temporal_hidden_size"]
        predictor._temporal_d_inner = temporal_kwargs["temporal_d_inner"]
        predictor._temporal_d_state = temporal_kwargs["temporal_d_state"]
        predictor._temporal_d_conv = temporal_kwargs["temporal_d_conv"]
        predictor._temporal_dt_rank = temporal_kwargs["temporal_dt_rank"]
        predictor._temporal_bias = temporal_kwargs["temporal_bias"]
        predictor._temporal_conv_bias = temporal_kwargs["temporal_conv_bias"]
        predictor._temporal_layers = temporal_kwargs["temporal_layers"]
        predictor._temporal_dropout = temporal_kwargs["temporal_dropout"]
        predictor._temporal_stateful = temporal_kwargs["temporal_stateful"]
        predictor._temporal_zero_init_out = temporal_kwargs["zero_init_temporal_out"]
    return predictor


def _dataset_to_numpy(ds: xarray.Dataset) -> xarray.Dataset:
    if isinstance(ds, xarray.Dataset):
        data_vars = {}
        for name, var in ds.data_vars.items():
            data_vars[name] = xarray.DataArray(
                np.asarray(jax.device_get(var.data)),
                coords=var.coords,
                dims=var.dims,
                attrs=var.attrs,
                name=var.name,
            )
        return xarray.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
    if isinstance(ds, xarray.DataArray):
        return xarray.DataArray(
            np.asarray(jax.device_get(ds.data)),
            coords=ds.coords,
            dims=ds.dims,
            attrs=ds.attrs,
            name=ds.name,
        )
    return ds


def _build_one_step_bundle(ckpt_obj, stats, ckpt_path: Path, *, residual_mode: bool):
    model_cfg = ckpt_obj.model_config
    task_cfg = ckpt_obj.task_config
    params = ckpt_obj.params

    run_cfg = _load_run_config(ckpt_path)
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
        "step": step,
    }


def _constant_inputs(inputs: xarray.Dataset, targets_template: xarray.Dataset, forcings: xarray.Dataset) -> xarray.Dataset:
    constant_inputs = inputs.drop_vars(targets_template.keys(), errors="ignore")
    constant_inputs = constant_inputs.drop_vars(forcings.keys(), errors="ignore")
    for name, var in constant_inputs.items():
        if "time" in var.dims:
            raise ValueError(
                f"Time-dependent input variable {name} must either be a forcing variable or target variable."
            )
    return constant_inputs


def _update_inputs(inputs: xarray.Dataset, next_frame: xarray.Dataset) -> xarray.Dataset:
    num_inputs = inputs.sizes["time"]
    predicted_or_forced_inputs = next_frame[list(inputs.keys())]
    return (
        xarray.concat([inputs, predicted_or_forced_inputs], dim="time")
        .tail(time=num_inputs)
        .assign_coords(time=inputs.coords["time"])
    )


def _clone_state(state):
    if state is None:
        return None
    return jax.tree_util.tree_map(lambda x: x, state)


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
    run_cfg = _load_run_config(ckpt_path)
    residual_mode = bool(run_cfg.get("residual_training", {}).get("enabled", False))

    if residual_mode:
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
                    inputs=all_inputs,
                    targets_template=target_step,
                    forcings=forcings_step,
                    state=residual_state,
                )
                full_pred = baseline_pred + residual_pred
                preds.append(full_pred.assign_coords(time=target_step.coords["time"]))
                rolling_inputs = _update_inputs(rolling_inputs, xarray.merge([full_pred, forcings_step]))
            return xarray.concat(preds, dim=target_times)

        return run_jitted, task_cfg, model_cfg, run_cfg

    bundle = _build_one_step_bundle(ckpt_obj, stats, ckpt_path, residual_mode=False)
    task_cfg = bundle["task_cfg"]
    model_cfg = bundle["model_cfg"]

    def run_jitted(**kw):
        rng = kw["rng"]
        inputs = kw["inputs"]
        targets_template = kw["targets_template"]
        forcings = kw["forcings"]
        constant_inputs = _constant_inputs(inputs, targets_template, forcings)
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        target_times = targets_template.coords["time"]
        state = None
        preds = []
        step_keys = jax.random.split(rng, targets_template.sizes["time"])
        for step_i in range(targets_template.sizes["time"]):
            target_step = targets_template.isel(time=slice(step_i, step_i + 1))
            forcings_step = forcings.isel(time=slice(step_i, step_i + 1))
            all_inputs = xarray.merge([constant_inputs, rolling_inputs])
            pred_step, state = bundle["step"](
                rng=step_keys[step_i],
                inputs=all_inputs,
                targets_template=target_step,
                forcings=forcings_step,
                state=state,
            )
            preds.append(pred_step.assign_coords(time=target_step.coords["time"]))
            rolling_inputs = _update_inputs(rolling_inputs, xarray.merge([pred_step, forcings_step]))
        return xarray.concat(preds, dim=target_times)

    return run_jitted, task_cfg, model_cfg, run_cfg


def build_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path: Path):
    run_cfg = _load_run_config(ckpt_path)
    residual_mode = bool(run_cfg.get("residual_training", {}).get("enabled", False))
    if not residual_mode:
        raise ValueError(f"Warm-start truth-anchored runner requires residual checkpoint: {ckpt_path}")

    residual_bundle, baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)
    task_cfg = residual_bundle["task_cfg"]
    model_cfg = residual_bundle["model_cfg"]

    def initialize_context(*, inputs, targets_template, forcings):
        constant_inputs = _constant_inputs(inputs, targets_template, forcings)
        rolling_inputs = inputs.drop_vars(constant_inputs.keys())
        return {
            "constant_inputs": constant_inputs,
            "rolling_inputs": rolling_inputs,
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
            inputs=all_inputs,
            targets_template=target_step,
            forcings=forcings_step,
            state=context["residual_state"],
        )
        full_pred = baseline_pred + residual_pred
        next_context = {
            "constant_inputs": context["constant_inputs"],
            "rolling_inputs": context["rolling_inputs"],
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


@contextmanager
def suppress_graphcast_future_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=re.escape(
                "The return type of `Dataset.dims` will be changed to return a set of dimension names in future, "
                "in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names "
                "to lengths, please use `Dataset.sizes`."
            ),
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=re.escape(
                "In a future version of xarray the default value for compat will change from compat='equals' "
                "to compat='override'. This change will result in the following ValueError: Cannot specify both "
                "data_vars='different' and compat='override'. The recommendation is to set compat explicitly "
                "for this case."
            ),
        )
        yield
