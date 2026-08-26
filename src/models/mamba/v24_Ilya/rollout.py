"""Shared autoregressive rollout semantics for v24_Ilya."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr


PredictStep = Callable[
    [Mapping[str, Any], Mapping[str, Any], jax.Array, xr.Dataset, xr.Dataset, xr.Dataset],
    tuple[xr.Dataset, Mapping[str, Any]],
]
PredictionConsumer = Callable[[int, xr.Dataset, xr.Dataset, xr.Dataset], None]
RolloutStep = tuple[xr.Dataset, xr.Dataset]


@dataclass(frozen=True)
class RolloutResult:
    baseline_prediction: xr.Dataset | None
    full_prediction: xr.Dataset | None
    baseline_state: Mapping[str, Any]
    residual_state: Mapping[str, Any]


def physical_fields_fp32(dataset: xr.Dataset) -> xr.Dataset:
    """Cast floating physical fields to FP32 without changing coordinates/integers."""

    replacements = {
        name: value.astype(jnp.float32)
        for name, value in dataset.data_vars.items()
        if jnp.issubdtype(value.dtype, jnp.inexact)
        and value.dtype != jnp.float32
    }
    return dataset.assign(replacements) if replacements else dataset


def state_boundary_fp32(tree):
    """Keep recurrent state crossing rollout steps/checkpoints in FP32."""

    def cast(value):
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            return value.astype(jnp.float32)
        return value

    return jax.tree_util.tree_map(cast, tree)


def shift_inputs_with_field(
    previous_inputs: xr.Dataset,
    new_field: xr.Dataset,
    forcings_next: xr.Dataset,
    *,
    time_step: pd.Timedelta,
    input_steps: int,
) -> xr.Dataset:
    """Append one predicted/truth field and retain the configured input window."""

    target_time = previous_inputs.time.values[-1:] + time_step
    next_state = physical_fields_fp32(new_field).assign_coords(time=target_time)
    next_forcings = physical_fields_fp32(forcings_next).assign_coords(time=target_time)
    next_frame = xr.merge([next_state, next_forcings])
    if "datetime" in next_frame.coords:
        next_frame = next_frame.drop_vars("datetime")
    input_variables = [name for name in next_frame.data_vars if name in previous_inputs.data_vars]
    next_inputs = next_frame[input_variables]
    merged = xr.concat(
        [previous_inputs, next_inputs],
        dim="time",
        data_vars="different",
        compat="equals",
    )
    return merged.tail(time=input_steps)


def combine_predictions(
    baseline_prediction: xr.Dataset,
    residual_prediction: xr.Dataset,
    residual_alpha: float,
) -> xr.Dataset:
    combined = jax.tree_util.tree_map(
        lambda baseline, residual: baseline + residual_alpha * residual,
        baseline_prediction,
        residual_prediction,
    )
    return physical_fields_fp32(combined)


def run_v24_Ilya_rollout(
    *,
    rng: jax.Array,
    inputs: xr.Dataset,
    all_targets: xr.Dataset | None,
    all_forcings: xr.Dataset | None,
    baseline_step: PredictStep,
    residual_step: PredictStep,
    baseline_params: Mapping[str, Any],
    residual_params: Mapping[str, Any],
    baseline_state_init: Mapping[str, Any],
    residual_state_init: Mapping[str, Any],
    time_step: pd.Timedelta,
    input_steps: int,
    warmup_steps: int,
    target_steps: int,
    full_feedback: bool,
    reset_state_after_warmup: bool,
    residual_alpha: float,
    reset_state_every_step: bool = False,
    prediction_consumer: PredictionConsumer | None = None,
    retain_predictions: bool = True,
    step_data: Iterable[RolloutStep] | None = None,
    skip_baseline_warmup: bool = False,
) -> RolloutResult:
    """Run truth warmup followed by baseline- or full-feedback evaluation."""

    required_steps = warmup_steps + target_steps
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    if step_data is None:
        if all_targets is None or all_forcings is None:
            raise ValueError(
                "all_targets and all_forcings are required without step_data"
            )
        if all_targets.sizes.get("time", 0) < required_steps:
            raise ValueError(
                f"Targets provide {all_targets.sizes.get('time', 0)} steps, "
                f"but rollout requires {required_steps}"
            )
        if all_forcings.sizes.get("time", 0) < required_steps:
            raise ValueError(
                f"Forcings provide {all_forcings.sizes.get('time', 0)} steps, "
                f"but rollout requires {required_steps}"
            )

        def dataset_steps():
            for index in range(required_steps):
                selection = {"time": slice(index, index + 1)}
                yield all_targets.isel(**selection), all_forcings.isel(**selection)

        step_iterator = iter(dataset_steps())
    else:
        if all_targets is not None or all_forcings is not None:
            raise ValueError(
                "step_data cannot be combined with all_targets or all_forcings"
            )
        step_iterator = iter(step_data)

    def next_step(step_index: int) -> RolloutStep:
        try:
            target, forcing = next(step_iterator)
        except StopIteration as exc:
            raise ValueError(
                f"step_data ended after {step_index} of {required_steps} steps"
            ) from exc
        return physical_fields_fp32(target), physical_fields_fp32(forcing)

    current_inputs = physical_fields_fp32(inputs)
    baseline_state = state_boundary_fp32(baseline_state_init)
    residual_state_initial_fp32 = state_boundary_fp32(residual_state_init)
    residual_state = residual_state_initial_fp32

    for step_index in range(warmup_steps):
        target, forcing = next_step(step_index)
        rng, baseline_key, residual_key = jax.random.split(rng, 3)
        if not skip_baseline_warmup:
            _baseline_prediction, baseline_state = baseline_step(
                baseline_params,
                baseline_state,
                baseline_key,
                current_inputs,
                target,
                forcing,
            )
            baseline_state = state_boundary_fp32(baseline_state)
        if reset_state_every_step:
            residual_state = residual_state_initial_fp32
        _residual_prediction, residual_state = residual_step(
            residual_params,
            residual_state,
            residual_key,
            current_inputs,
            target,
            forcing,
        )
        residual_state = state_boundary_fp32(residual_state)
        current_inputs = shift_inputs_with_field(
            current_inputs,
            target,
            forcing,
            time_step=time_step,
            input_steps=input_steps,
        )

    if reset_state_after_warmup:
        residual_state = residual_state_initial_fp32

    baseline_chunks: list[xr.Dataset] = []
    full_chunks: list[xr.Dataset] = []

    if full_feedback:
        baseline_inputs = current_inputs
        baseline_branch_state = baseline_state
        full_inputs = current_inputs
        full_branch_baseline_state = baseline_state
        full_branch_residual_state = residual_state

        for metric_index in range(target_steps):
            step_index = warmup_steps + metric_index
            target, forcing = next_step(step_index)
            rng, baseline_key, residual_key = jax.random.split(rng, 3)
            baseline_prediction, baseline_branch_state = baseline_step(
                baseline_params,
                baseline_branch_state,
                baseline_key,
                baseline_inputs,
                target,
                forcing,
            )
            baseline_prediction = physical_fields_fp32(baseline_prediction)
            baseline_branch_state = state_boundary_fp32(baseline_branch_state)
            full_baseline_prediction, full_branch_baseline_state = baseline_step(
                baseline_params,
                full_branch_baseline_state,
                baseline_key,
                full_inputs,
                target,
                forcing,
            )
            full_baseline_prediction = physical_fields_fp32(full_baseline_prediction)
            full_branch_baseline_state = state_boundary_fp32(
                full_branch_baseline_state
            )
            if reset_state_every_step:
                full_branch_residual_state = residual_state_initial_fp32
            residual_prediction, full_branch_residual_state = residual_step(
                residual_params,
                full_branch_residual_state,
                residual_key,
                full_inputs,
                target,
                forcing,
            )
            residual_prediction = physical_fields_fp32(residual_prediction)
            full_branch_residual_state = state_boundary_fp32(
                full_branch_residual_state
            )
            full_prediction = combine_predictions(
                full_baseline_prediction,
                residual_prediction,
                residual_alpha,
            )
            if prediction_consumer is not None:
                prediction_consumer(
                    metric_index,
                    target,
                    baseline_prediction,
                    full_prediction,
                )
            if retain_predictions:
                baseline_chunks.append(baseline_prediction)
                full_chunks.append(full_prediction)
            if metric_index < target_steps - 1:
                baseline_inputs = shift_inputs_with_field(
                    baseline_inputs,
                    baseline_prediction,
                    forcing,
                    time_step=time_step,
                    input_steps=input_steps,
                )
                full_inputs = shift_inputs_with_field(
                    full_inputs,
                    full_prediction,
                    forcing,
                    time_step=time_step,
                    input_steps=input_steps,
                )

        baseline_state = baseline_branch_state
        residual_state = full_branch_residual_state
    else:
        for metric_index in range(target_steps):
            step_index = warmup_steps + metric_index
            target, forcing = next_step(step_index)
            rng, baseline_key, residual_key = jax.random.split(rng, 3)
            baseline_prediction, baseline_state = baseline_step(
                baseline_params,
                baseline_state,
                baseline_key,
                current_inputs,
                target,
                forcing,
            )
            baseline_prediction = physical_fields_fp32(baseline_prediction)
            baseline_state = state_boundary_fp32(baseline_state)
            if reset_state_every_step:
                residual_state = residual_state_initial_fp32
            residual_prediction, residual_state = residual_step(
                residual_params,
                residual_state,
                residual_key,
                current_inputs,
                target,
                forcing,
            )
            residual_prediction = physical_fields_fp32(residual_prediction)
            residual_state = state_boundary_fp32(residual_state)
            full_prediction = combine_predictions(
                baseline_prediction,
                residual_prediction,
                residual_alpha,
            )
            if prediction_consumer is not None:
                prediction_consumer(
                    metric_index,
                    target,
                    baseline_prediction,
                    full_prediction,
                )
            if retain_predictions:
                baseline_chunks.append(baseline_prediction)
                full_chunks.append(full_prediction)
            if metric_index < target_steps - 1:
                current_inputs = shift_inputs_with_field(
                    current_inputs,
                    baseline_prediction,
                    forcing,
                    time_step=time_step,
                    input_steps=input_steps,
                )

    return RolloutResult(
        baseline_prediction=(
            xr.concat(baseline_chunks, dim="time") if retain_predictions else None
        ),
        full_prediction=(
            xr.concat(full_chunks, dim="time") if retain_predictions else None
        ),
        baseline_state=baseline_state,
        residual_state=residual_state,
    )
