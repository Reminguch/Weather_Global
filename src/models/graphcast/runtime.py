"""Baseline GraphCast runtime and rollout construction."""

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

ROOT = Path(__file__).resolve().parents[3]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"

for path in (ROOT, GRAPHCAST_LOCAL):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from graphcast import casting, graphcast as gc, normalization
from src.models.graphcast.training.core.model import load_graphcast_checkpoint, load_stats

FAMILY_NAME = "graphcast"


def load_checkpoint_and_stats(ckpt_path: Path, stats_dir: Path):
    ckpt_obj = load_graphcast_checkpoint(ckpt_path)
    stats = load_stats(stats_dir)
    return ckpt_obj, stats


def load_run_config(ckpt_path: Path) -> dict[str, Any]:
    run_cfg_path = ckpt_path.parent / "run_config.json"
    if not run_cfg_path.exists():
        return {}
    with run_cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_family(run_cfg: dict[str, Any]) -> str:
    if bool(run_cfg.get("residual_training", {}).get("enabled", False)):
        return "residual_mamba"
    if run_cfg.get("temporal_config", {}).get("backbone", "none") == "mamba":
        return "gc_mamba"
    return FAMILY_NAME


def _dataset_to_numpy(ds):
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


def _build_one_step_bundle(ckpt_obj, stats, ckpt_path: Path):
    model_cfg = ckpt_obj.model_config
    task_cfg = ckpt_obj.task_config
    params = ckpt_obj.params

    run_cfg = load_run_config(ckpt_path)
    use_bf16 = run_cfg.get("precision", "bf16") == "bf16"

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = gc.GraphCast(model_cfg, task_cfg)
        if use_bf16:
            predictor = casting.Bfloat16Cast(predictor)
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


def build_run_jitted(ckpt_obj, stats, ckpt_path: Path):
    run_cfg = load_run_config(ckpt_path)
    family = infer_family(run_cfg)
    if family != FAMILY_NAME:
        raise ValueError(f"Checkpoint family mismatch: expected {FAMILY_NAME}, found {family}")

    bundle = _build_one_step_bundle(ckpt_obj, stats, ckpt_path)
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
