"""Runtime for legacy 98-channel GraphCast+Mamba checkpoints."""

from __future__ import annotations

from pathlib import Path

import haiku as hk
import jax
import xarray
from graphcast import casting, normalization

from src.models.graphcast.runtime import (
    _constant_inputs,
    _dataset_to_numpy,
    _update_inputs,
    infer_family,
    load_checkpoint_and_stats,
    load_run_config,
)
from src.models.mamba.gc_mamba.legacy_graphcast import LegacyGraphCastMamba
from src.models.mamba.gc_mamba.runtime import _apply_temporal_config, _temporal_kwargs

MODEL_NAME = "legacy_gc_mamba"
LEGACY_GRID_ENCODER_INPUT_DIM = 98


def _iter_param_leaves(tree, prefix: str = ""):
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _iter_param_leaves(value, f"{prefix}/{key}")
        return
    yield prefix, tree


def grid_encoder_input_dim(params) -> int | None:
    for name, value in _iter_param_leaves(params):
        if "encoder_nodes_grid_nodes_mlp" in name and name.endswith("linear_0/w"):
            return int(value.shape[0])
    return None


def is_legacy_gc_mamba_checkpoint(ckpt_obj) -> bool:
    return grid_encoder_input_dim(ckpt_obj.params) == LEGACY_GRID_ENCODER_INPUT_DIM


def _build_one_step_bundle(ckpt_obj, stats, ckpt_path: Path):
    model_cfg = ckpt_obj.model_config
    task_cfg = ckpt_obj.task_config
    params = ckpt_obj.params

    run_cfg = load_run_config(ckpt_path)
    family = infer_family(run_cfg)
    if family != "gc_mamba":
        raise ValueError(f"Checkpoint family mismatch: expected gc_mamba-compatible config, found {family}")
    if not is_legacy_gc_mamba_checkpoint(ckpt_obj):
        input_dim = grid_encoder_input_dim(params)
        raise ValueError(f"Checkpoint is not legacy GC-Mamba: grid encoder input dim is {input_dim}")

    temporal_kwargs = _temporal_kwargs(model_cfg, run_cfg)
    use_bf16 = run_cfg.get("precision", "bf16") == "bf16"

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = LegacyGraphCastMamba(model_cfg, task_cfg)
        predictor = _apply_temporal_config(predictor, temporal_kwargs)
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
        "step": step,
    }


def build_run_jitted(ckpt_obj, stats, ckpt_path: Path):
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

    return run_jitted, task_cfg, model_cfg, bundle["run_cfg"]


__all__ = [
    "LEGACY_GRID_ENCODER_INPUT_DIM",
    "MODEL_NAME",
    "_build_one_step_bundle",
    "build_run_jitted",
    "grid_encoder_input_dim",
    "is_legacy_gc_mamba_checkpoint",
    "load_checkpoint_and_stats",
]
