"""GraphCast+Mamba runtime and rollout construction."""

from __future__ import annotations

from pathlib import Path

import haiku as hk
import jax
import xarray

from src.models.graphcast.runtime import (
    _constant_inputs,
    _dataset_to_numpy,
    _update_inputs,
    infer_family,
    load_checkpoint_and_stats,
    load_run_config,
)

ROOT = Path(__file__).resolve().parents[4]
from graphcast import casting, graphcast as gc, normalization  # noqa: E402

MODEL_NAME = "gc_mamba"


def _temporal_kwargs(model_cfg, run_cfg: dict) -> dict:
    del model_cfg
    temporal_cfg = run_cfg.get("temporal_config", {})
    temporal_backbone = temporal_cfg.get("backbone", "none")
    temporal_d_inner = temporal_cfg.get("d_inner", None)
    if temporal_backbone == "mamba" and temporal_d_inner is None:
        raise ValueError(
            "GC-Mamba checkpoint is missing temporal_config.d_inner. "
            "Older checkpoints that only stored hidden_size are no longer supported."
        )
    return {
        "temporal_backbone": temporal_backbone,
        "temporal_location": temporal_cfg.get("location", "mesh_post_encoder"),
        "temporal_d_inner": temporal_d_inner,
        "temporal_d_state": temporal_cfg.get("d_state", 16),
        "temporal_d_conv": temporal_cfg.get("d_conv", 4),
        "temporal_dt_rank": temporal_cfg.get("dt_rank", "auto"),
        "temporal_bias": temporal_cfg.get("bias", False),
        "temporal_conv_bias": temporal_cfg.get("conv_bias", True),
        "temporal_layers": temporal_cfg.get("layers", 1),
        "temporal_dropout": temporal_cfg.get("dropout", 0.0),
        "temporal_stateful": bool(temporal_cfg.get("stateful", False)),
        "zero_init_temporal_out": bool(temporal_cfg.get("zero_init_output", False)),
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
        predictor._temporal_zero_init_out = temporal_kwargs["zero_init_temporal_out"]
    return predictor


def _build_one_step_bundle(ckpt_obj, stats, ckpt_path: Path):
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
    if family != MODEL_NAME:
        raise ValueError(f"Checkpoint family mismatch: expected {MODEL_NAME}, found {family}")

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
