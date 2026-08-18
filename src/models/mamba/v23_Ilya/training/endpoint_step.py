"""Endpoint-only BPTT with an explicit residual-state reverse pass."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xarray as xr

from src.models.graphcast.training.core.model import scalarize_loss

from ..model import (
    V23IlyaModelConfigs,
    make_baseline_predictor,
    make_residual_predictor,
)
from .config import BPTT_BACKEND, V23IlyaTrainConfig
from .data_parallel import (
    pack_replica_trees,
    replicate_tree,
    shard_replica_tree,
    start_tree_to_host,
    unpack_device_replica_trees,
)


@dataclass(frozen=True)
class V23IlyaTrainingTransforms:
    baseline_predict: hk.TransformedWithState
    residual_predict: hk.TransformedWithState
    residual_loss: hk.TransformedWithState
    residual_loss_and_predictions: hk.TransformedWithState


def build_training_transforms(
    model_configs: V23IlyaModelConfigs,
    task_config,
    stats,
    config: V23IlyaTrainConfig,
) -> V23IlyaTrainingTransforms:
    use_bf16 = config.precision == "bf16"

    def baseline_prediction(inputs, targets, forcings):
        return make_baseline_predictor(
            model_configs.baseline,
            task_config,
            stats,
            use_bf16=use_bf16,
        )(inputs, targets_template=targets, forcings=forcings)

    def residual_prediction(inputs, targets, forcings):
        return make_residual_predictor(
            model_configs.residual,
            task_config,
            stats,
            config.architecture,
            use_bf16=use_bf16,
        )(inputs, targets_template=targets, forcings=forcings)

    def residual_loss_and_predictions(inputs, residual_targets, forcings):
        return make_residual_predictor(
            model_configs.residual,
            task_config,
            stats,
            config.architecture,
            use_bf16=use_bf16,
        ).loss_and_predictions(inputs, residual_targets, forcings)

    combined = hk.transform_with_state(residual_loss_and_predictions)
    return V23IlyaTrainingTransforms(
        baseline_predict=hk.transform_with_state(baseline_prediction),
        residual_predict=hk.transform_with_state(residual_prediction),
        # Kept as an initialization-compatible alias; training applies only
        # residual_loss_and_predictions on supervised steps.
        residual_loss=combined,
        residual_loss_and_predictions=combined,
    )


def build_optimizer(config: V23IlyaTrainConfig):
    if config.warmup_steps > 0:
        learning_rate = optax.warmup_constant_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=config.warmup_steps,
        )
    else:
        learning_rate = config.learning_rate
    transforms = []
    if config.grad_clip > 0:
        transforms.append(optax.clip_by_global_norm(config.grad_clip))
    transforms.append(optax.adamw(learning_rate, weight_decay=config.weight_decay))
    optimizer = optax.chain(*transforms) if len(transforms) > 1 else transforms[0]
    return optimizer, learning_rate


def _tree_stop(tree):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def _tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda x, y: x + y, left, right)


def _cast_floating(tree, dtype):
    def cast(value):
        value = jnp.asarray(value)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            return value.astype(dtype)
        return value

    return jax.tree_util.tree_map(cast, tree)


def cast_state_boundary_fp32(tree):
    """Keep recurrent carry/checkpoint boundaries in FP32."""
    return _cast_floating(tree, jnp.float32)


def _template_like(truth, dtype):
    def zeros(value):
        value = jnp.asarray(value)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            return jnp.zeros(value.shape, dtype=dtype)
        return jnp.zeros_like(value)

    return jax.tree_util.tree_map(zeros, truth)


def residual_target(truth, baseline_prediction):
    return jax.tree_util.tree_map(
        lambda target, baseline: target - jax.lax.stop_gradient(baseline),
        truth,
        baseline_prediction,
    )


def feedback_field(mode: str, baseline_prediction, residual_prediction):
    baseline_prediction = _tree_stop(baseline_prediction)
    if mode == "baseline":
        return baseline_prediction
    if mode == "closed_loop_sg":
        return jax.tree_util.tree_map(
            lambda baseline, residual: baseline + jax.lax.stop_gradient(residual),
            baseline_prediction,
            residual_prediction,
        )
    raise ValueError(f"Unsupported feedback mode: {mode!r}")


def _window_times(
    step_index: int,
    truth_prefix_steps: int,
    time_step: pd.Timedelta,
) -> np.ndarray:
    step_ns = int(time_step / pd.Timedelta(1, "ns"))
    if step_index < truth_prefix_steps:
        offsets = np.asarray([-step_ns, 0], dtype="timedelta64[ns]")
    else:
        start = (step_index - truth_prefix_steps) * step_ns
        offsets = np.asarray([start, start + step_ns], dtype="timedelta64[ns]")
    return offsets


def input_window_from_frames(
    previous_frame: xr.Dataset,
    current_frame: xr.Dataset,
    static_inputs: xr.Dataset,
    *,
    step_index: int,
    truth_prefix_steps: int,
    time_step: pd.Timedelta,
) -> xr.Dataset:
    times = _window_times(step_index, truth_prefix_steps, time_step)
    previous = previous_frame.assign_coords(time=times[:1])
    current = current_frame.assign_coords(time=times[1:])
    dynamic = xr.concat(
        [previous, current],
        dim="time",
        data_vars="all",
        coords="minimal",
        compat="override",
    )
    return xr.merge([dynamic, static_inputs], compat="override")


def next_dynamic_frame(
    current_inputs: xr.Dataset,
    field: xr.Dataset,
    forcing: xr.Dataset,
    *,
    time_step: pd.Timedelta,
) -> xr.Dataset:
    target_time = current_inputs.time.values[-1:] + time_step
    state = field.assign_coords(time=target_time)
    forcing = forcing.assign_coords(time=target_time)
    merged = xr.merge([state, forcing])
    if "datetime" in merged.coords:
        merged = merged.drop_vars("datetime")
    variables = [
        name
        for name, value in merged.data_vars.items()
        if name in current_inputs.data_vars and "time" in value.dims
    ]
    return merged[variables]


def tree_nbytes(tree) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is None or dtype is None:
            continue
        total += int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)
    return total


def memory_contract(
    *,
    input_frames,
    static_inputs,
    truths,
    residual_state,
    config: V23IlyaTrainConfig,
) -> dict[str, Any]:
    tape_dtype = jnp.bfloat16 if config.weather_tape_precision == "bf16" else jnp.float32
    sample_frame = _cast_floating(input_frames[0], tape_dtype)
    return {
        "bptt_backend": BPTT_BACKEND,
        "teacher_input_windows": config.truth_prefix_steps,
        "unique_weather_tape_frames": config.bptt_steps + 1,
        "weather_tape_precision": config.weather_tape_precision,
        "weather_tape_bytes": tree_nbytes(sample_frame) * (config.bptt_steps + 1),
        "state_tape_boundaries": config.bptt_steps,
        "state_tape_bytes": tree_nbytes(residual_state) * config.bptt_steps,
        "truth_targets_loaded": len(truths),
        "truth_target_bytes": sum(tree_nbytes(value) for value in truths),
        "static_input_bytes": tree_nbytes(static_inputs),
        "graphcast_backward_calls": 0,
    }


def make_bptt_objective(
    *,
    transforms: V23IlyaTrainingTransforms,
    baseline_params,
    baseline_state,
    config: V23IlyaTrainConfig,
    time_step,
    input_steps: int,
) -> Callable[..., tuple[Any, Any]]:
    """Build an objective whose custom VJP rematerializes residual steps only."""

    if input_steps != 2:
        raise ValueError("v23_Ilya explicit reverse BPTT requires input_steps=2")
    if jax.tree_util.tree_leaves(baseline_state):
        raise ValueError(
            "Endpoint GraphCast warmup skipping requires an empty frozen baseline state"
        )
    tape_dtype = (
        jnp.bfloat16
        if config.weather_tape_precision == "bf16"
        else jnp.float32
    )
    final_index = config.bptt_steps - 1
    supervised_indices = config.supervised_step_indices
    supervised_positions = {
        index: position for position, index in enumerate(supervised_indices)
    }
    supervised_weights = config.normalized_supervised_weights
    expected_truths = len(supervised_indices)

    def baseline_step(key, inputs, template, forcing):
        prediction, _ = transforms.baseline_predict.apply(
            baseline_params,
            baseline_state,
            key,
            _tree_stop(inputs),
            template,
            forcing,
        )
        return _tree_stop(prediction)

    def residual_predict_step(params, state, key, inputs, template, forcing):
        prediction, next_state = transforms.residual_predict.apply(
            params,
            state,
            key,
            inputs,
            template,
            forcing,
        )
        return prediction, cast_state_boundary_fp32(next_state)

    def residual_supervised_step(
        params,
        state,
        key,
        inputs,
        target,
        forcing,
    ):
        output, next_state = transforms.residual_loss_and_predictions.apply(
            params,
            state,
            key,
            inputs,
            target,
            forcing,
        )
        (loss_array, _diagnostics), prediction = output
        return (
            scalarize_loss(loss_array),
            prediction,
            cast_state_boundary_fp32(next_state),
        )

    def run_forward(
        residual_params,
        initial_state,
        keys,
        input_frames,
        static_inputs,
        truths,
        forcings,
        *,
        save_tape: bool,
    ):
        if len(input_frames) != config.truth_prefix_steps + 1:
            raise ValueError(
                f"Expected {config.truth_prefix_steps + 1} unique teacher frames, "
                f"got {len(input_frames)}"
            )
        if len(truths) != expected_truths:
            raise ValueError(
                f"loss_mode={config.loss_mode!r} requires {expected_truths} truths, "
                f"got {len(truths)}"
            )
        if len(forcings) != config.bptt_steps:
            raise ValueError(
                f"Expected {config.bptt_steps} forcings, got {len(forcings)}"
            )

        initial_state = cast_state_boundary_fp32(initial_state)
        zero_state = _tree_zeros_like(initial_state)
        reset_state = config.temporal_state_policy == "reset_every_anchor"
        state = zero_state if reset_state else initial_state
        weather_frames = [
            _tree_stop(_cast_floating(frame, tape_dtype))
            for frame in input_frames
        ]
        state_tape = []
        residual_targets = []
        losses = []
        template_truth = truths[-1]

        for index in range(config.bptt_steps):
            state_in = zero_state if reset_state else state
            if save_tape:
                state_tape.append(state_in)
            current_inputs = input_window_from_frames(
                weather_frames[index],
                weather_frames[index + 1],
                static_inputs,
                step_index=index,
                truth_prefix_steps=config.truth_prefix_steps,
                time_step=time_step,
            )
            key = keys[index]
            forcing = forcings[index]
            supervised_position = supervised_positions.get(index)
            supervised = supervised_position is not None
            needs_feedback = (
                index >= config.truth_prefix_steps - 1 and index < final_index
            )
            template = _template_like(
                template_truth,
                tape_dtype if index < final_index else jnp.float32,
            )

            if supervised:
                truth = truths[supervised_position]
                baseline_prediction = baseline_step(
                    key,
                    current_inputs,
                    truth,
                    forcing,
                )
                target = residual_target(truth, baseline_prediction)
                loss, residual_prediction, next_state = residual_supervised_step(
                    residual_params,
                    state_in,
                    key,
                    current_inputs,
                    target,
                    forcing,
                )
                losses.append(loss)
                if save_tape:
                    residual_targets.append(_tree_stop(target))
            else:
                if needs_feedback:
                    baseline_prediction = baseline_step(
                        key,
                        current_inputs,
                        template,
                        forcing,
                    )
                else:
                    baseline_prediction = None
                residual_prediction, next_state = residual_predict_step(
                    residual_params,
                    state_in,
                    key,
                    current_inputs,
                    template,
                    forcing,
                )

            state = zero_state if reset_state else next_state
            if needs_feedback:
                assert baseline_prediction is not None
                field = feedback_field(
                    config.feedback_mode,
                    baseline_prediction,
                    residual_prediction,
                )
                frame = next_dynamic_frame(
                    current_inputs,
                    field,
                    forcing,
                    time_step=time_step,
                )
                weather_frames.append(
                    _tree_stop(_cast_floating(frame, tape_dtype))
                )

        loss_components = jnp.stack(losses)
        loss = jnp.sum(
            loss_components
            * jnp.asarray(supervised_weights, dtype=loss_components.dtype)
        )
        final_state = zero_state if reset_state else state
        tape = (
            residual_params,
            tuple(weather_frames),
            tuple(state_tape),
            tuple(residual_targets),
            keys,
            static_inputs,
            forcings,
        )
        return loss, _tree_stop(final_state), tape

    def objective(
        residual_params,
        initial_state,
        keys,
        input_frames,
        static_inputs,
        truths,
        forcings,
    ):
        # GraphCast's xarray PyTree node cannot be an argument to custom_vjp:
        # JAX probes the input tree with shape-less dummy objects. Keep the
        # custom boundary array-only and rebuild datasets from captured static
        # tree definitions inside the primal and reverse rules.
        frame_treedef = jax.tree_util.tree_structure(input_frames[0])
        static_treedef = jax.tree_util.tree_structure(static_inputs)
        truth_treedef = jax.tree_util.tree_structure(truths[0])
        forcing_treedef = jax.tree_util.tree_structure(forcings[0])

        def pack(tree):
            return tuple(jax.tree_util.tree_leaves(tree))

        def unpack(treedef, leaves):
            return jax.tree_util.tree_unflatten(treedef, leaves)

        packed_input_frames = tuple(pack(frame) for frame in input_frames)
        packed_static_inputs = pack(static_inputs)
        packed_truths = tuple(pack(truth) for truth in truths)
        packed_forcings = tuple(pack(forcing) for forcing in forcings)

        def unpack_data(
            frame_leaves,
            static_leaves,
            truth_leaves,
            forcing_leaves,
        ):
            return (
                tuple(unpack(frame_treedef, leaves) for leaves in frame_leaves),
                unpack(static_treedef, static_leaves),
                tuple(unpack(truth_treedef, leaves) for leaves in truth_leaves),
                tuple(unpack(forcing_treedef, leaves) for leaves in forcing_leaves),
            )

        @jax.custom_vjp
        def flat_objective(
            params,
            state,
            step_keys,
            frame_leaves,
            static_leaves,
            truth_leaves,
            forcing_leaves,
        ):
            frames, static, truth_values, forcing_values = unpack_data(
                frame_leaves,
                static_leaves,
                truth_leaves,
                forcing_leaves,
            )
            loss, final_state, _ = run_forward(
                params,
                state,
                step_keys,
                frames,
                static,
                truth_values,
                forcing_values,
                save_tape=False,
            )
            return loss, final_state

        def flat_objective_fwd(
            params,
            state,
            step_keys,
            frame_leaves,
            static_leaves,
            truth_leaves,
            forcing_leaves,
        ):
            frames, static, truth_values, forcing_values = unpack_data(
                frame_leaves,
                static_leaves,
                truth_leaves,
                forcing_leaves,
            )
            loss, final_state, tape = run_forward(
                params,
                state,
                step_keys,
                frames,
                static,
                truth_values,
                forcing_values,
                save_tape=True,
            )
            (
                tape_params,
                weather_frames,
                state_tape,
                residual_targets,
                tape_keys,
                tape_static,
                tape_forcings,
            ) = tape
            packed_tape = (
                tape_params,
                tuple(pack(frame) for frame in weather_frames),
                state_tape,
                tuple(pack(target) for target in residual_targets),
                tape_keys,
                pack(tape_static),
                tuple(pack(forcing) for forcing in tape_forcings),
            )
            return (loss, final_state), packed_tape

        def flat_objective_bwd(packed_tape, output_cotangents):
            (
                tape_params,
                weather_leaves,
                state_tape,
                residual_target_leaves,
                tape_keys,
                static_leaves,
                forcing_leaves,
            ) = packed_tape
            weather_frames = tuple(
                unpack(frame_treedef, leaves) for leaves in weather_leaves
            )
            residual_targets = tuple(
                unpack(truth_treedef, leaves) for leaves in residual_target_leaves
            )
            static = unpack(static_treedef, static_leaves)
            forcing_values = tuple(
                unpack(forcing_treedef, leaves) for leaves in forcing_leaves
            )
            tape_initial_state = state_tape[0]
            loss_cotangent, _final_state_cotangent = output_cotangents
            reset_state = config.temporal_state_policy == "reset_every_anchor"
            parameter_cotangent = _tree_zeros_like(tape_params)
            state_cotangent = _tree_zeros_like(tape_initial_state)

            for index in range(final_index, -1, -1):
                current_inputs = input_window_from_frames(
                    weather_frames[index],
                    weather_frames[index + 1],
                    static,
                    step_index=index,
                    truth_prefix_steps=config.truth_prefix_steps,
                    time_step=time_step,
                )
                key = tape_keys[index]
                forcing = forcing_values[index]
                state_in = state_tape[index]
                supervised_position = supervised_positions.get(index)
                supervised = supervised_position is not None

                if supervised:
                    target = residual_targets[supervised_position]

                    def supervised_step(params, state):
                        loss, _prediction, next_state = residual_supervised_step(
                            params,
                            state,
                            key,
                            current_inputs,
                            target,
                            forcing,
                        )
                        return loss, next_state

                    _, pullback = jax.vjp(
                        jax.checkpoint(supervised_step),
                        tape_params,
                        state_in,
                    )
                    weight = jnp.asarray(
                        supervised_weights[supervised_position],
                        dtype=loss_cotangent.dtype,
                    )
                    step_param_cotangent, step_state_cotangent = pullback(
                        (loss_cotangent * weight, state_cotangent)
                    )
                else:
                    template = _template_like(residual_targets[-1], tape_dtype)

                    def state_step(params, state):
                        _prediction, next_state = residual_predict_step(
                            params,
                            state,
                            key,
                            current_inputs,
                            template,
                            forcing,
                        )
                        return next_state

                    _, pullback = jax.vjp(
                        jax.checkpoint(state_step),
                        tape_params,
                        state_in,
                    )
                    step_param_cotangent, step_state_cotangent = pullback(
                        state_cotangent
                    )

                parameter_cotangent = _tree_add(
                    parameter_cotangent,
                    step_param_cotangent,
                )
                state_cotangent = (
                    _tree_zeros_like(tape_initial_state)
                    if reset_state
                    else step_state_cotangent
                )

            zero_frame_leaves = tuple(
                tuple(jnp.zeros_like(value) for value in leaves)
                for leaves in packed_input_frames
            )
            zero_static_leaves = tuple(
                jnp.zeros_like(value) for value in packed_static_inputs
            )
            zero_truth_leaves = tuple(
                tuple(jnp.zeros_like(value) for value in leaves)
                for leaves in packed_truths
            )
            zero_forcing_leaves = tuple(
                tuple(jnp.zeros_like(value) for value in leaves)
                for leaves in packed_forcings
            )
            return (
                parameter_cotangent,
                state_cotangent,
                None,
                zero_frame_leaves,
                zero_static_leaves,
                zero_truth_leaves,
                zero_forcing_leaves,
            )

        flat_objective.defvjp(flat_objective_fwd, flat_objective_bwd)
        return flat_objective(
            residual_params,
            initial_state,
            keys,
            packed_input_frames,
            packed_static_inputs,
            packed_truths,
            packed_forcings,
        )

    return objective


def _make_monolithic_train_step(
    *,
    transforms: V23IlyaTrainingTransforms,
    optimizer,
    baseline_params,
    baseline_state,
    config: V23IlyaTrainConfig,
    time_step,
    input_steps: int,
) -> Callable[..., tuple[Any, Any, Any, Any, Any]]:
    bptt_objective = make_bptt_objective(
        transforms=transforms,
        baseline_params=baseline_params,
        baseline_state=baseline_state,
        config=config,
        time_step=time_step,
        input_steps=input_steps,
    )

    def train_step(
        residual_params,
        residual_state,
        optimizer_state,
        keys,
        input_frames,
        static_inputs,
        truths,
        forcings,
    ):
        def train_objective(params):
            return bptt_objective(
                params,
                residual_state,
                keys,
                input_frames,
                static_inputs,
                truths,
                forcings,
            )

        (loss, next_residual_state), gradients = jax.value_and_grad(
            train_objective,
            has_aux=True,
        )(residual_params)
        gradient_norm = optax.global_norm(gradients)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            residual_params,
        )
        next_residual_params = optax.apply_updates(residual_params, updates)
        return (
            next_residual_params,
            _tree_stop(next_residual_state),
            next_optimizer_state,
            loss,
            gradient_norm,
        )

    return train_step


def make_train_step(
    *,
    transforms: V23IlyaTrainingTransforms,
    optimizer,
    baseline_params,
    baseline_state,
    config: V23IlyaTrainConfig,
    time_step,
    input_steps: int,
) -> Callable[..., tuple[Any, Any, Any, Any, Any, Any]]:
    """Build host-orchestrated BPTT with device-resident one-step kernels."""

    if input_steps != 2:
        raise ValueError("v23_Ilya streaming BPTT requires input_steps=2")
    if jax.tree_util.tree_leaves(baseline_state):
        raise ValueError("Frozen GraphCast must have empty recurrent state")

    tape_dtype = (
        jnp.bfloat16
        if config.weather_tape_precision == "bf16"
        else jnp.float32
    )
    final_index = config.bptt_steps - 1
    reset_state = config.temporal_state_policy == "reset_every_anchor"
    supervised_indices = config.supervised_step_indices
    supervised_positions = {
        index: position for position, index in enumerate(supervised_indices)
    }
    supervised_weights = np.asarray(
        config.normalized_supervised_weights, dtype=np.float32
    )
    expected_truths = len(supervised_indices)

    def to_host(tree):
        return jax.device_get(_tree_stop(tree))

    @jax.jit
    def baseline_forward(key, inputs, template, forcing):
        prediction, _ = transforms.baseline_predict.apply(
            baseline_params,
            baseline_state,
            key,
            _tree_stop(inputs),
            template,
            forcing,
        )
        return _tree_stop(prediction)

    @jax.jit
    def residual_forward(params, state, key, inputs, template, forcing):
        prediction, next_state = transforms.residual_predict.apply(
            params,
            state,
            key,
            inputs,
            template,
            forcing,
        )
        return prediction, cast_state_boundary_fp32(next_state)

    @jax.jit
    def supervised_forward(params, state, key, inputs, target, forcing):
        output, next_state = transforms.residual_loss_and_predictions.apply(
            params,
            state,
            key,
            inputs,
            target,
            forcing,
        )
        (loss_array, _diagnostics), prediction = output
        return (
            scalarize_loss(loss_array),
            prediction,
            cast_state_boundary_fp32(next_state),
        )

    @jax.jit
    def state_pullback(
        params,
        state,
        key,
        inputs,
        template,
        forcing,
        state_cotangent,
    ):
        def rematerialized_step(candidate_params, candidate_state):
            _prediction, next_state = residual_forward(
                candidate_params,
                candidate_state,
                key,
                inputs,
                template,
                forcing,
            )
            return next_state

        _, pullback = jax.vjp(
            jax.checkpoint(rematerialized_step),
            params,
            state,
        )
        return pullback(state_cotangent)

    @jax.jit
    def supervised_pullback(
        params,
        state,
        key,
        inputs,
        target,
        forcing,
        loss_cotangent,
        state_cotangent,
    ):
        def rematerialized_step(candidate_params, candidate_state):
            loss, _prediction, next_state = supervised_forward(
                candidate_params,
                candidate_state,
                key,
                inputs,
                target,
                forcing,
            )
            return loss, next_state

        _, pullback = jax.vjp(
            jax.checkpoint(rematerialized_step),
            params,
            state,
        )
        return pullback((loss_cotangent, state_cotangent))

    @jax.jit
    def accumulate(left, right):
        return _tree_add(left, right)

    @jax.jit
    def apply_optimizer(params, optimizer_state, gradients):
        gradient_norm = optax.global_norm(gradients)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            params,
        )
        next_params = optax.apply_updates(params, updates)
        return next_params, next_optimizer_state, gradient_norm

    def train_step(
        residual_params,
        residual_state,
        optimizer_state,
        keys,
        input_frames,
        static_inputs,
        truths,
        forcings,
    ):
        if len(input_frames) != config.truth_prefix_steps + 1:
            raise ValueError(
                f"Expected {config.truth_prefix_steps + 1} teacher frames, "
                f"got {len(input_frames)}"
            )
        if len(truths) != expected_truths:
            raise ValueError(
                f"loss_mode={config.loss_mode!r} requires {expected_truths} truths, "
                f"got {len(truths)}"
            )
        if len(forcings) != config.bptt_steps:
            raise ValueError(
                f"Expected {config.bptt_steps} forcings, got {len(forcings)}"
            )

        host_keys = to_host(keys)
        host_static = to_host(static_inputs)
        host_truths = tuple(to_host(value) for value in truths)
        host_forcings = tuple(to_host(value) for value in forcings)
        weather_frames = [
            to_host(_cast_floating(frame, tape_dtype))
            for frame in input_frames
        ]
        state_host = to_host(cast_state_boundary_fp32(residual_state))
        zero_state_host = to_host(_tree_zeros_like(state_host))
        if reset_state:
            state_host = zero_state_host

        state_tape = []
        residual_targets = []
        host_losses = []
        template_truth = host_truths[-1]

        for index in range(config.bptt_steps):
            state_in_host = zero_state_host if reset_state else state_host
            state_tape.append(state_in_host)
            current_inputs = input_window_from_frames(
                weather_frames[index],
                weather_frames[index + 1],
                host_static,
                step_index=index,
                truth_prefix_steps=config.truth_prefix_steps,
                time_step=time_step,
            )
            key = host_keys[index]
            forcing = host_forcings[index]
            supervised_position = supervised_positions.get(index)
            supervised = supervised_position is not None
            needs_feedback = (
                index >= config.truth_prefix_steps - 1 and index < final_index
            )
            template = _template_like(
                template_truth,
                tape_dtype if index < final_index else jnp.float32,
            )

            if supervised:
                truth = host_truths[supervised_position]
                baseline_prediction = baseline_forward(
                    key,
                    current_inputs,
                    truth,
                    forcing,
                )
                target = residual_target(truth, baseline_prediction)
                loss, residual_prediction, next_state = supervised_forward(
                    residual_params,
                    state_in_host,
                    key,
                    current_inputs,
                    target,
                    forcing,
                )
                host_losses.append(np.asarray(jax.device_get(loss)))
                residual_targets.append(to_host(target))
            else:
                if needs_feedback:
                    baseline_prediction = baseline_forward(
                        key,
                        current_inputs,
                        template,
                        forcing,
                    )
                else:
                    baseline_prediction = None
                residual_prediction, next_state = residual_forward(
                    residual_params,
                    state_in_host,
                    key,
                    current_inputs,
                    template,
                    forcing,
                )

            state_host = zero_state_host if reset_state else to_host(next_state)
            if needs_feedback:
                assert baseline_prediction is not None
                frame = next_dynamic_frame(
                    current_inputs,
                    feedback_field(
                        config.feedback_mode,
                        baseline_prediction,
                        residual_prediction,
                    ),
                    forcing,
                    time_step=time_step,
                )
                weather_frames.append(
                    to_host(_cast_floating(frame, tape_dtype))
                )

            del current_inputs, residual_prediction, next_state
            if baseline_prediction is not None:
                del baseline_prediction

        if len(weather_frames) != config.bptt_steps + 1:
            raise RuntimeError(
                f"Streaming weather tape has {len(weather_frames)} frames, "
                f"expected {config.bptt_steps + 1}"
            )
        if len(state_tape) != config.bptt_steps:
            raise RuntimeError("Streaming state tape length is incorrect")

        parameter_cotangent = _tree_zeros_like(residual_params)
        state_cotangent = _tree_zeros_like(residual_state)
        for index in range(final_index, -1, -1):
            current_inputs = input_window_from_frames(
                weather_frames[index],
                weather_frames[index + 1],
                host_static,
                step_index=index,
                truth_prefix_steps=config.truth_prefix_steps,
                time_step=time_step,
            )
            key = host_keys[index]
            forcing = host_forcings[index]
            state_in_host = state_tape[index]
            supervised_position = supervised_positions.get(index)
            supervised = supervised_position is not None

            if supervised:
                step_parameter_cotangent, step_state_cotangent = supervised_pullback(
                    residual_params,
                    state_in_host,
                    key,
                    current_inputs,
                    residual_targets[supervised_position],
                    forcing,
                    jnp.asarray(
                        supervised_weights[supervised_position], dtype=jnp.float32
                    ),
                    state_cotangent,
                )
            else:
                template = _template_like(residual_targets[-1], tape_dtype)
                step_parameter_cotangent, step_state_cotangent = state_pullback(
                    residual_params,
                    state_in_host,
                    key,
                    current_inputs,
                    template,
                    forcing,
                    state_cotangent,
                )

            parameter_cotangent = accumulate(
                parameter_cotangent,
                step_parameter_cotangent,
            )
            state_cotangent = (
                _tree_zeros_like(residual_state)
                if reset_state
                else step_state_cotangent
            )
            jax.block_until_ready(parameter_cotangent)
            del current_inputs, step_parameter_cotangent, step_state_cotangent

        next_params, next_optimizer_state, gradient_norm = apply_optimizer(
            residual_params,
            optimizer_state,
            parameter_cotangent,
        )
        jax.block_until_ready(next_params)
        loss_components = np.asarray(host_losses, dtype=np.float32)
        loss_value = np.sum(loss_components * supervised_weights)
        return (
            next_params,
            jax.tree_util.tree_map(jnp.asarray, state_host),
            next_optimizer_state,
            jnp.asarray(loss_value, dtype=jnp.float32),
            gradient_norm,
            loss_components,
        )

    return train_step


def make_data_parallel_train_step(
    *,
    transforms: V23IlyaTrainingTransforms,
    optimizer,
    baseline_params,
    baseline_state,
    config: V23IlyaTrainConfig,
    time_step,
    input_steps: int,
    devices,
) -> Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any, dict[str, float]]]:
    """Build hybrid DP BPTT with device-live rollout and bounded host tape."""

    if input_steps != 2:
        raise ValueError("v23_Ilya streaming BPTT requires input_steps=2")
    if jax.tree_util.tree_leaves(baseline_state):
        raise ValueError("Frozen GraphCast must have empty recurrent state")
    num_replicas = config.distributed.num_devices
    devices = tuple(devices)
    if len(devices) != num_replicas:
        raise ValueError(
            f"Expected {num_replicas} data-parallel devices, got {len(devices)}"
        )

    tape_dtype = (
        jnp.bfloat16
        if config.weather_tape_precision == "bf16"
        else jnp.float32
    )
    final_index = config.bptt_steps - 1
    reset_state = config.temporal_state_policy == "reset_every_anchor"
    supervised_indices = config.supervised_step_indices
    supervised_positions = {
        index: position for position, index in enumerate(supervised_indices)
    }
    supervised_weights = np.asarray(
        config.normalized_supervised_weights, dtype=np.float32
    )
    expected_truths = len(supervised_indices)
    replicated_baseline_params = replicate_tree(baseline_params, devices)
    kernel_caches = {}

    def host_cast(tree, dtype):
        host_tree = jax.device_get(tree)
        return jax.tree_util.tree_map(
            lambda value: np.asarray(value).astype(dtype),
            host_tree,
        )

    def host_template_like(tree, dtype):
        def zeros(value):
            value = np.asarray(value)
            if np.issubdtype(value.dtype, np.inexact):
                return np.zeros(value.shape, dtype=np.dtype(dtype))
            return np.zeros_like(value)

        return jax.tree_util.tree_map(zeros, tree)

    def unpack_host_replicas(treedef, packed_leaves):
        return tuple(
            jax.tree_util.tree_unflatten(
                treedef,
                [np.asarray(value[replica]) for value in packed_leaves],
            )
            for replica in range(num_replicas)
        )

    def build_kernel_cache(
        previous_frame_tree,
        current_frame_tree,
        static_tree,
        target_tree,
        forcing_tree,
        step_index,
    ):
        previous_frame_treedef = jax.tree_util.tree_structure(previous_frame_tree)
        current_frame_treedef = jax.tree_util.tree_structure(current_frame_tree)
        static_treedef = jax.tree_util.tree_structure(static_tree)
        target_treedef = jax.tree_util.tree_structure(target_tree)
        forcing_treedef = jax.tree_util.tree_structure(forcing_tree)
        input_tree = input_window_from_frames(
            previous_frame_tree,
            current_frame_tree,
            static_tree,
            step_index=step_index,
            truth_prefix_steps=config.truth_prefix_steps,
            time_step=time_step,
        )
        frame_treedef = jax.tree_util.tree_structure(
            next_dynamic_frame(
                input_tree,
                target_tree,
                forcing_tree,
                time_step=time_step,
            )
        )

        def unpack(treedef, leaves):
            return jax.tree_util.tree_unflatten(treedef, leaves)

        def pack(tree):
            return tuple(jax.tree_util.tree_leaves(tree))

        def build_inputs(
            previous_frame_leaves,
            current_frame_leaves,
            static_leaves,
        ):
            return input_window_from_frames(
                unpack(previous_frame_treedef, previous_frame_leaves),
                unpack(current_frame_treedef, current_frame_leaves),
                unpack(static_treedef, static_leaves),
                step_index=step_index,
                truth_prefix_steps=config.truth_prefix_steps,
                time_step=time_step,
            )

        def supervised_rollout_local(
            local_baseline_params,
            params,
            state,
            key,
            previous_frame_leaves,
            current_frame_leaves,
            static_leaves,
            truth_leaves,
            forcing_leaves,
        ):
            inputs = build_inputs(
                previous_frame_leaves,
                current_frame_leaves,
                static_leaves,
            )
            truth = unpack(target_treedef, truth_leaves)
            forcing = unpack(forcing_treedef, forcing_leaves)
            baseline_prediction, _ = transforms.baseline_predict.apply(
                local_baseline_params,
                baseline_state,
                key,
                _tree_stop(inputs),
                truth,
                forcing,
            )
            target = residual_target(truth, baseline_prediction)
            output, next_state = transforms.residual_loss_and_predictions.apply(
                params,
                state,
                key,
                inputs,
                target,
                forcing,
            )
            (loss_array, _diagnostics), prediction = output
            frame = next_dynamic_frame(
                inputs,
                feedback_field(
                    config.feedback_mode,
                    baseline_prediction,
                    prediction,
                ),
                forcing,
                time_step=time_step,
            )
            return (
                scalarize_loss(loss_array),
                pack(_tree_stop(_cast_floating(frame, tape_dtype))),
                cast_state_boundary_fp32(next_state),
                pack(_tree_stop(target)),
            )

        def supervised_no_feedback_local(
            local_baseline_params,
            params,
            state,
            key,
            previous_frame_leaves,
            current_frame_leaves,
            static_leaves,
            truth_leaves,
            forcing_leaves,
        ):
            inputs = build_inputs(
                previous_frame_leaves,
                current_frame_leaves,
                static_leaves,
            )
            truth = unpack(target_treedef, truth_leaves)
            forcing = unpack(forcing_treedef, forcing_leaves)
            baseline_prediction, _ = transforms.baseline_predict.apply(
                local_baseline_params,
                baseline_state,
                key,
                _tree_stop(inputs),
                truth,
                forcing,
            )
            target = residual_target(truth, baseline_prediction)
            output, next_state = transforms.residual_loss_and_predictions.apply(
                params,
                state,
                key,
                inputs,
                target,
                forcing,
            )
            (loss_array, _diagnostics), _prediction = output
            return (
                scalarize_loss(loss_array),
                cast_state_boundary_fp32(next_state),
                pack(_tree_stop(target)),
            )

        def residual_rollout_local(
            local_baseline_params,
            params,
            state,
            key,
            previous_frame_leaves,
            current_frame_leaves,
            static_leaves,
            template_leaves,
            forcing_leaves,
        ):
            inputs = build_inputs(
                previous_frame_leaves,
                current_frame_leaves,
                static_leaves,
            )
            template = unpack(target_treedef, template_leaves)
            forcing = unpack(forcing_treedef, forcing_leaves)
            baseline_prediction, _ = transforms.baseline_predict.apply(
                local_baseline_params,
                baseline_state,
                key,
                _tree_stop(inputs),
                template,
                forcing,
            )
            prediction, next_state = transforms.residual_predict.apply(
                params,
                state,
                key,
                inputs,
                template,
                forcing,
            )
            frame = next_dynamic_frame(
                inputs,
                feedback_field(
                    config.feedback_mode,
                    baseline_prediction,
                    prediction,
                ),
                forcing,
                time_step=time_step,
            )
            return (
                pack(_tree_stop(_cast_floating(frame, tape_dtype))),
                cast_state_boundary_fp32(next_state),
            )

        def residual_no_feedback_local(
            params,
            state,
            key,
            previous_frame_leaves,
            current_frame_leaves,
            static_leaves,
            template_leaves,
            forcing_leaves,
        ):
            inputs = build_inputs(
                previous_frame_leaves,
                current_frame_leaves,
                static_leaves,
            )
            template = unpack(target_treedef, template_leaves)
            forcing = unpack(forcing_treedef, forcing_leaves)
            _prediction, next_state = transforms.residual_predict.apply(
                params,
                state,
                key,
                inputs,
                template,
                forcing,
            )
            return cast_state_boundary_fp32(next_state)

        def state_pullback_local(
            params,
            state,
            key,
            previous_frame_leaves,
            current_frame_leaves,
            static_leaves,
            target_leaves,
            forcing_leaves,
            state_cotangent,
        ):
            inputs = build_inputs(
                previous_frame_leaves,
                current_frame_leaves,
                static_leaves,
            )
            target = unpack(target_treedef, target_leaves)
            forcing = unpack(forcing_treedef, forcing_leaves)

            def rematerialized_step(candidate_params, candidate_state):
                _prediction, next_state = transforms.residual_predict.apply(
                    candidate_params,
                    candidate_state,
                    key,
                    inputs,
                    target,
                    forcing,
                )
                return cast_state_boundary_fp32(next_state)

            _, pullback = jax.vjp(
                jax.checkpoint(rematerialized_step),
                params,
                state,
            )
            return pullback(state_cotangent)

        def supervised_pullback_local(
            params,
            state,
            key,
            previous_frame_leaves,
            current_frame_leaves,
            static_leaves,
            target_leaves,
            forcing_leaves,
            loss_cotangent,
            state_cotangent,
        ):
            inputs = build_inputs(
                previous_frame_leaves,
                current_frame_leaves,
                static_leaves,
            )
            target = unpack(target_treedef, target_leaves)
            forcing = unpack(forcing_treedef, forcing_leaves)

            def rematerialized_step(candidate_params, candidate_state):
                output, next_state = transforms.residual_loss_and_predictions.apply(
                    candidate_params,
                    candidate_state,
                    key,
                    inputs,
                    target,
                    forcing,
                )
                (loss_array, _diagnostics), _prediction = output
                return (
                    scalarize_loss(loss_array),
                    cast_state_boundary_fp32(next_state),
                )

            _, pullback = jax.vjp(
                jax.checkpoint(rematerialized_step),
                params,
                state,
            )
            return pullback((loss_cotangent, state_cotangent))

        def accumulate_local(left, right):
            return _tree_add(left, right)

        def optimizer_local(params, optimizer_state, gradients, local_loss):
            mean_gradients = jax.tree_util.tree_map(
                lambda value: jax.lax.pmean(value, "data"),
                gradients,
            )
            gradient_norm = optax.global_norm(mean_gradients)
            updates, next_optimizer_state = optimizer.update(
                mean_gradients,
                optimizer_state,
                params,
            )
            next_params = optax.apply_updates(params, updates)
            mean_loss = jax.lax.pmean(local_loss, "data")
            return next_params, next_optimizer_state, gradient_norm, mean_loss

        pmap_kwargs = {"axis_name": "data", "devices": list(devices)}
        return {
            "previous_frame_treedef": previous_frame_treedef,
            "current_frame_treedef": current_frame_treedef,
            "static_treedef": static_treedef,
            "target_treedef": target_treedef,
            "forcing_treedef": forcing_treedef,
            "frame_treedef": frame_treedef,
            "supervised_rollout": jax.pmap(
                supervised_rollout_local,
                **pmap_kwargs,
            ),
            "supervised_no_feedback": jax.pmap(
                supervised_no_feedback_local,
                **pmap_kwargs,
            ),
            "residual_rollout": jax.pmap(
                residual_rollout_local,
                **pmap_kwargs,
            ),
            "residual_no_feedback": jax.pmap(
                residual_no_feedback_local,
                **pmap_kwargs,
            ),
            "state_pullback": jax.pmap(state_pullback_local, **pmap_kwargs),
            "supervised_pullback": jax.pmap(
                supervised_pullback_local,
                **pmap_kwargs,
            ),
            "accumulate": jax.pmap(accumulate_local, **pmap_kwargs),
            "optimizer": jax.pmap(optimizer_local, **pmap_kwargs),
        }

    def get_kernel_cache(
        previous_frame_tree,
        current_frame_tree,
        static_tree,
        target_tree,
        forcing_tree,
        step_index,
    ):
        cache_key = (
            jax.tree_util.tree_structure(previous_frame_tree),
            jax.tree_util.tree_structure(current_frame_tree),
            jax.tree_util.tree_structure(static_tree),
            jax.tree_util.tree_structure(target_tree),
            jax.tree_util.tree_structure(forcing_tree),
            step_index,
        )
        if cache_key not in kernel_caches:
            kernel_caches[cache_key] = build_kernel_cache(
                previous_frame_tree,
                current_frame_tree,
                static_tree,
                target_tree,
                forcing_tree,
                step_index,
            )
        return kernel_caches[cache_key]

    def pack_expected(trees, expected_treedef):
        actual_treedef, leaves = pack_replica_trees(trees, devices)
        if actual_treedef != expected_treedef:
            raise ValueError("Data-parallel xarray PyTree structure changed")
        return leaves

    def train_step(
        residual_params,
        residual_state,
        optimizer_state,
        keys,
        input_frames,
        static_inputs,
        truths,
        forcings,
    ):
        if len(input_frames) != num_replicas:
            raise ValueError(
                f"Expected {num_replicas} input-frame lanes, got {len(input_frames)}"
            )
        for replica in range(num_replicas):
            if len(input_frames[replica]) != config.truth_prefix_steps + 1:
                raise ValueError("Replica teacher-frame count is incorrect")
            if len(truths[replica]) != expected_truths:
                raise ValueError("Replica truth count is incorrect")
            if len(forcings[replica]) != config.bptt_steps:
                raise ValueError("Replica forcing count is incorrect")

        setup_started = time.monotonic()
        host_keys = np.asarray(jax.device_get(keys))
        host_static = tuple(host_cast(value, jnp.float32) for value in static_inputs)
        host_truths = tuple(
            tuple(host_cast(value, jnp.float32) for value in replica_truths)
            for replica_truths in truths
        )
        host_forcings = tuple(
            tuple(host_cast(value, jnp.float32) for value in replica_forcings)
            for replica_forcings in forcings
        )
        weather_frames = [
            [host_cast(frame, tape_dtype) for frame in replica_frames]
            for replica_frames in input_frames
        ]
        host_templates_bf16 = tuple(
            host_template_like(host_truths[replica][-1], tape_dtype)
            for replica in range(num_replicas)
        )
        host_templates_fp32 = tuple(
            host_template_like(host_truths[replica][-1], jnp.float32)
            for replica in range(num_replicas)
        )

        static_treedef, static_leaves = pack_replica_trees(host_static, devices)
        device_static = unpack_device_replica_trees(
            static_treedef,
            static_leaves,
            num_replicas,
        )
        template_treedef, template_bf16_leaves = pack_replica_trees(
            host_templates_bf16,
            devices,
        )
        template_fp32_treedef, template_fp32_leaves = pack_replica_trees(
            host_templates_fp32,
            devices,
        )
        if template_treedef != template_fp32_treedef:
            raise ValueError("FP32 and BF16 target templates have different structures")

        previous_treedef, previous_leaves = pack_replica_trees(
            tuple(weather_frames[replica][0] for replica in range(num_replicas)),
            devices,
        )
        current_treedef, current_leaves = pack_replica_trees(
            tuple(weather_frames[replica][1] for replica in range(num_replicas)),
            devices,
        )
        device_previous_frames = unpack_device_replica_trees(
            previous_treedef,
            previous_leaves,
            num_replicas,
        )
        device_current_frames = unpack_device_replica_trees(
            current_treedef,
            current_leaves,
            num_replicas,
        )

        device_state = cast_state_boundary_fp32(residual_state)
        zero_state = jax.tree_util.tree_map(jnp.zeros_like, device_state)
        state_tape = []
        residual_targets = []
        loss_arrays = []
        pending_boundary = None
        forward_offload_wait_seconds = 0.0
        host_setup_seconds = time.monotonic() - setup_started

        def consume_pending(boundary):
            nonlocal forward_offload_wait_seconds
            if boundary is None:
                return
            wait_started = time.monotonic()
            snapshot = boundary["copy"].materialize()
            forward_offload_wait_seconds += time.monotonic() - wait_started
            state_tape.append(snapshot["state"])
            if boundary["frame_treedef"] is not None:
                replica_frames = unpack_host_replicas(
                    boundary["frame_treedef"],
                    snapshot["frame_leaves"],
                )
                for replica, frame in enumerate(replica_frames):
                    weather_frames[replica].append(frame)
            if boundary["target_treedef"] is not None:
                residual_targets.append(
                    unpack_host_replicas(
                        boundary["target_treedef"],
                        snapshot["target_leaves"],
                    )
                )

        forward_started = time.monotonic()
        for index in range(config.bptt_steps):
            state_in = zero_state if reset_state else device_state
            forcing = tuple(
                host_forcings[replica][index] for replica in range(num_replicas)
            )
            supervised_position = supervised_positions.get(index)
            supervised = supervised_position is not None
            needs_feedback = (
                index >= config.truth_prefix_steps - 1 and index < final_index
            )
            templates = (
                host_templates_bf16
                if index < final_index
                else host_templates_fp32
            )
            kernels = get_kernel_cache(
                device_previous_frames[0],
                device_current_frames[0],
                device_static[0],
                templates[0],
                forcing[0],
                index,
            )
            forcing_leaves = pack_expected(
                forcing,
                kernels["forcing_treedef"],
            )
            key = host_keys[:, index]

            next_frame_leaves = ()
            target_leaves = ()
            if supervised:
                truth = tuple(
                    host_truths[replica][supervised_position]
                    for replica in range(num_replicas)
                )
                truth_leaves = pack_expected(truth, kernels["target_treedef"])
                if needs_feedback:
                    (
                        loss,
                        next_frame_leaves,
                        next_state,
                        target_leaves,
                    ) = kernels["supervised_rollout"](
                        replicated_baseline_params,
                        residual_params,
                        state_in,
                        key,
                        previous_leaves,
                        current_leaves,
                        static_leaves,
                        truth_leaves,
                        forcing_leaves,
                    )
                else:
                    loss, next_state, target_leaves = kernels[
                        "supervised_no_feedback"
                    ](
                        replicated_baseline_params,
                        residual_params,
                        state_in,
                        key,
                        previous_leaves,
                        current_leaves,
                        static_leaves,
                        truth_leaves,
                        forcing_leaves,
                    )
                loss_arrays.append(loss)
            else:
                template_leaves = (
                    template_bf16_leaves
                    if index < final_index
                    else template_fp32_leaves
                )
                if needs_feedback:
                    next_frame_leaves, next_state = kernels["residual_rollout"](
                        replicated_baseline_params,
                        residual_params,
                        state_in,
                        key,
                        previous_leaves,
                        current_leaves,
                        static_leaves,
                        template_leaves,
                        forcing_leaves,
                    )
                else:
                    next_state = kernels["residual_no_feedback"](
                        residual_params,
                        state_in,
                        key,
                        previous_leaves,
                        current_leaves,
                        static_leaves,
                        template_leaves,
                        forcing_leaves,
                    )

            # Launch the next kernel before materializing the previous boundary.
            # This overlaps one boundary's D2H copy with useful device work while
            # retaining at most one pending device snapshot.
            consume_pending(pending_boundary)
            pending_boundary = {
                "copy": start_tree_to_host(
                    {
                        "state": state_in,
                        "frame_leaves": tuple(next_frame_leaves),
                        "target_leaves": tuple(target_leaves),
                    }
                ),
                "frame_treedef": kernels["frame_treedef"] if needs_feedback else None,
                "target_treedef": (
                    kernels["target_treedef"] if supervised else None
                ),
            }

            device_state = zero_state if reset_state else next_state
            if index < config.truth_prefix_steps - 1:
                next_teacher_treedef, next_teacher_leaves = pack_replica_trees(
                    tuple(
                        weather_frames[replica][index + 2]
                        for replica in range(num_replicas)
                    ),
                    devices,
                )
                device_previous_frames = device_current_frames
                previous_leaves = current_leaves
                current_leaves = next_teacher_leaves
                device_current_frames = unpack_device_replica_trees(
                    next_teacher_treedef,
                    current_leaves,
                    num_replicas,
                )
            elif needs_feedback:
                device_previous_frames = device_current_frames
                previous_leaves = current_leaves
                current_leaves = next_frame_leaves
                device_current_frames = unpack_device_replica_trees(
                    kernels["frame_treedef"],
                    current_leaves,
                    num_replicas,
                )

        consume_pending(pending_boundary)
        forward_seconds = time.monotonic() - forward_started

        def prepare_backward(index):
            prepare_started = time.monotonic()
            forcing = tuple(
                host_forcings[replica][index] for replica in range(num_replicas)
            )
            kernels = get_kernel_cache(
                weather_frames[0][index],
                weather_frames[0][index + 1],
                host_static[0],
                host_templates_bf16[0],
                forcing[0],
                index,
            )
            prepared = {
                "index": index,
                "kernels": kernels,
                "previous_frame_leaves": pack_expected(
                    tuple(
                        weather_frames[replica][index]
                        for replica in range(num_replicas)
                    ),
                    kernels["previous_frame_treedef"],
                ),
                "current_frame_leaves": pack_expected(
                    tuple(
                        weather_frames[replica][index + 1]
                        for replica in range(num_replicas)
                    ),
                    kernels["current_frame_treedef"],
                ),
                "static_leaves": static_leaves,
                "forcing_leaves": pack_expected(
                    forcing,
                    kernels["forcing_treedef"],
                ),
                "state": shard_replica_tree(state_tape[index], devices),
                "key": host_keys[:, index],
            }
            supervised_position = supervised_positions.get(index)
            if supervised_position is not None:
                prepared["target_leaves"] = pack_expected(
                    residual_targets[supervised_position],
                    kernels["target_treedef"],
                )
            else:
                prepared["template_leaves"] = template_bf16_leaves
            return prepared, time.monotonic() - prepare_started

        parameter_cotangent = jax.tree_util.tree_map(jnp.zeros_like, residual_params)
        state_cotangent = jax.tree_util.tree_map(jnp.zeros_like, residual_state)
        prepared, backward_prepare_seconds = prepare_backward(final_index)
        backward_started = time.monotonic()
        for index in range(final_index, -1, -1):
            kernels = prepared["kernels"]
            supervised_position = supervised_positions.get(index)
            supervised = supervised_position is not None

            if supervised:
                step_parameter_cotangent, step_state_cotangent = (
                    kernels["supervised_pullback"](
                        residual_params,
                        prepared["state"],
                        prepared["key"],
                        prepared["previous_frame_leaves"],
                        prepared["current_frame_leaves"],
                        prepared["static_leaves"],
                        prepared["target_leaves"],
                        prepared["forcing_leaves"],
                        np.full(
                            (num_replicas,),
                            supervised_weights[supervised_position],
                            dtype=np.float32,
                        ),
                        state_cotangent,
                    )
                )
            else:
                step_parameter_cotangent, step_state_cotangent = (
                    kernels["state_pullback"](
                        residual_params,
                        prepared["state"],
                        prepared["key"],
                        prepared["previous_frame_leaves"],
                        prepared["current_frame_leaves"],
                        prepared["static_leaves"],
                        prepared["template_leaves"],
                        prepared["forcing_leaves"],
                        state_cotangent,
                    )
                )

            next_prepared = None
            if index > 0:
                next_prepared, prepare_seconds = prepare_backward(index - 1)
                backward_prepare_seconds += prepare_seconds
            parameter_cotangent = kernels["accumulate"](
                parameter_cotangent,
                step_parameter_cotangent,
            )
            state_cotangent = (
                jax.tree_util.tree_map(jnp.zeros_like, residual_state)
                if reset_state
                else step_state_cotangent
            )
            jax.block_until_ready(parameter_cotangent)
            prepared = next_prepared

        backward_seconds = time.monotonic() - backward_started

        loss_started = time.monotonic()
        lane_loss_components = np.asarray(
            jax.device_get(tuple(loss_arrays)),
            dtype=np.float32,
        )
        lane_losses = np.sum(
            lane_loss_components * supervised_weights[:, None], axis=0
        )
        loss_materialize_seconds = time.monotonic() - loss_started
        optimizer_started = time.monotonic()
        (
            next_params,
            next_optimizer_state,
            gradient_norm,
            mean_loss,
        ) = kernels["optimizer"](
            residual_params,
            optimizer_state,
            parameter_cotangent,
            lane_losses,
        )
        jax.block_until_ready(next_params)
        optimizer_seconds = time.monotonic() - optimizer_started
        phase_seconds = {
            "host_setup": host_setup_seconds,
            "forward": forward_seconds,
            "forward_offload_wait": forward_offload_wait_seconds,
            "backward": backward_seconds,
            "backward_prepare": backward_prepare_seconds,
            "loss_materialize": loss_materialize_seconds,
            "optimizer": optimizer_seconds,
        }
        return (
            next_params,
            device_state,
            next_optimizer_state,
            mean_loss[0],
            gradient_norm[0],
            np.asarray(lane_losses, dtype=np.float32),
            lane_loss_components,
            phase_seconds,
        )

    return train_step


def make_validation_step(
    *,
    transforms: V23IlyaTrainingTransforms,
    baseline_params,
    baseline_state,
    config: V23IlyaTrainConfig,
    time_step,
    input_steps: int,
) -> Callable[..., tuple[Any, Any]]:
    bptt_objective = make_bptt_objective(
        transforms=transforms,
        baseline_params=baseline_params,
        baseline_state=baseline_state,
        config=config,
        time_step=time_step,
        input_steps=input_steps,
    )

    @jax.jit
    def validation_step(
        residual_params,
        residual_state,
        keys,
        input_frames,
        static_inputs,
        truths,
        forcings,
    ):
        loss, next_residual_state = bptt_objective(
            residual_params,
            residual_state,
            keys,
            input_frames,
            static_inputs,
            truths,
            forcings,
        )
        return loss, _tree_stop(next_residual_state)

    return validation_step
