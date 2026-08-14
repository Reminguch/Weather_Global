"""Shared autoregressive rollout semantics for v23_Ilya."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import pandas as pd
import xarray as xr


PredictStep = Callable[
    [Mapping[str, Any], Mapping[str, Any], jax.Array, xr.Dataset, xr.Dataset, xr.Dataset],
    tuple[xr.Dataset, Mapping[str, Any]],
]


@dataclass(frozen=True)
class RolloutResult:
    baseline_prediction: xr.Dataset
    full_prediction: xr.Dataset
    baseline_state: Mapping[str, Any]
    residual_state: Mapping[str, Any]


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
    next_state = new_field.assign_coords(time=target_time)
    next_forcings = forcings_next.assign_coords(time=target_time)
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
    return jax.tree_util.tree_map(
        lambda baseline, residual: baseline + residual_alpha * residual,
        baseline_prediction,
        residual_prediction,
    )


def run_v23_Ilya_rollout(
    *,
    rng: jax.Array,
    inputs: xr.Dataset,
    all_targets: xr.Dataset,
    all_forcings: xr.Dataset,
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
) -> RolloutResult:
    """Run truth warmup followed by baseline- or full-feedback evaluation."""

    required_steps = warmup_steps + target_steps
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
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

    current_inputs = inputs
    baseline_state = baseline_state_init
    residual_state = residual_state_init

    for step_index in range(warmup_steps):
        target = all_targets.isel(time=slice(step_index, step_index + 1))
        forcing = all_forcings.isel(time=slice(step_index, step_index + 1))
        rng, baseline_key, residual_key = jax.random.split(rng, 3)
        _baseline_prediction, baseline_state = baseline_step(
            baseline_params,
            baseline_state,
            baseline_key,
            current_inputs,
            target,
            forcing,
        )
        if reset_state_every_step:
            residual_state = residual_state_init
        _residual_prediction, residual_state = residual_step(
            residual_params,
            residual_state,
            residual_key,
            current_inputs,
            target,
            forcing,
        )
        current_inputs = shift_inputs_with_field(
            current_inputs,
            target,
            forcing,
            time_step=time_step,
            input_steps=input_steps,
        )

    if reset_state_after_warmup:
        residual_state = residual_state_init

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
            target = all_targets.isel(time=slice(step_index, step_index + 1))
            forcing = all_forcings.isel(time=slice(step_index, step_index + 1))
            rng, baseline_key, residual_key = jax.random.split(rng, 3)
            baseline_prediction, baseline_branch_state = baseline_step(
                baseline_params,
                baseline_branch_state,
                baseline_key,
                baseline_inputs,
                target,
                forcing,
            )
            full_baseline_prediction, full_branch_baseline_state = baseline_step(
                baseline_params,
                full_branch_baseline_state,
                baseline_key,
                full_inputs,
                target,
                forcing,
            )
            if reset_state_every_step:
                full_branch_residual_state = residual_state_init
            residual_prediction, full_branch_residual_state = residual_step(
                residual_params,
                full_branch_residual_state,
                residual_key,
                full_inputs,
                target,
                forcing,
            )
            full_prediction = combine_predictions(
                full_baseline_prediction,
                residual_prediction,
                residual_alpha,
            )
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
            target = all_targets.isel(time=slice(step_index, step_index + 1))
            forcing = all_forcings.isel(time=slice(step_index, step_index + 1))
            rng, baseline_key, residual_key = jax.random.split(rng, 3)
            baseline_prediction, baseline_state = baseline_step(
                baseline_params,
                baseline_state,
                baseline_key,
                current_inputs,
                target,
                forcing,
            )
            if reset_state_every_step:
                residual_state = residual_state_init
            residual_prediction, residual_state = residual_step(
                residual_params,
                residual_state,
                residual_key,
                current_inputs,
                target,
                forcing,
            )
            full_prediction = combine_predictions(
                baseline_prediction,
                residual_prediction,
                residual_alpha,
            )
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
        baseline_prediction=xr.concat(baseline_chunks, dim="time"),
        full_prediction=xr.concat(full_chunks, dim="time"),
        baseline_state=baseline_state,
        residual_state=residual_state,
    )
