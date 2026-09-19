"""Cached residual training using the unchanged one-weather-step model.

Only the frozen baseline computation is removed. Model calls, BF16 loss/state
boundaries, the host state tape, reverse-time VJPs, and FP32 gradient
accumulation follow ``endpoint_step.make_train_step``. Inputs and targets are
already reconstructed by the cache reader; no baseline parameters are needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.models.graphcast.training.core.model import scalarize_loss
from ..model import make_residual_predictor
from .endpoint_step import cast_state_boundary_fp32


@dataclass(frozen=True)
class CachedResidualTransforms:
    """Only residual transforms; constructing this never initializes GC."""

    residual_predict: Any
    residual_loss_and_predictions: Any


def _stop(tree):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def _zeros(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _host(tree):
    return jax.device_get(_stop(tree))


def _template(target):
    return jax.tree_util.tree_map(lambda value: jnp.zeros_like(value, dtype=jnp.float32), target)


class CachedStepwiseTrainer:
    """Host-orchestrated residual BPTT with reusable jitted one-step kernels.

``inputs`` and ``forcings`` contain all chunk steps. ``targets`` contains either
all steps or the supervised steps in ``config.supervised_step_indices`` order.
Targets are physical FP32 residual fields, not normalized targets or ERA5 truth.
The caller owns the source buffers for the duration of each method call.
"""

    def __init__(self, *, config, optimizer, transforms: CachedResidualTransforms):
        if config.feedback_mode != "baseline":
            raise ValueError("Cached stepwise training requires feedback_mode='baseline'")
        self.config = config
        self.optimizer = optimizer
        self.transforms = transforms
        self.steps = int(config.bptt_steps)
        self.reset_state = config.temporal_state_policy == "reset_every_anchor"
        self.supervised_indices = tuple(config.supervised_step_indices)
        self.supervised_positions = {
            index: position for position, index in enumerate(self.supervised_indices)
        }
        self.weights = np.asarray(config.normalized_supervised_weights, dtype=np.float32)
        if not self.supervised_indices or len(self.weights) != len(self.supervised_indices):
            raise ValueError("Supervised steps and weights must be nonempty and aligned")
        if self.supervised_indices != tuple(sorted(set(self.supervised_indices))):
            raise ValueError("Supervised step indices must be unique and increasing")
        if self.supervised_indices[0] < 0 or self.supervised_indices[-1] >= self.steps:
            raise ValueError("Supervised step indices must lie inside the chunk")

        @jax.jit
        def residual_forward(params, state, key, inputs, template, forcing):
            prediction, next_state = transforms.residual_predict.apply(
                params, state, key, inputs, template, forcing
            )
            return prediction, cast_state_boundary_fp32(next_state)

        @jax.jit
        def supervised_forward(params, state, key, inputs, target, forcing):
            output, next_state = transforms.residual_loss_and_predictions.apply(
                params, state, key, inputs, target, forcing
            )
            (loss_array, _diagnostics), prediction = output
            return scalarize_loss(loss_array), prediction, cast_state_boundary_fp32(next_state)

        @jax.jit
        def state_pullback(params, state, key, inputs, template, forcing, state_cotangent):
            def rematerialized_step(candidate_params, candidate_state):
                _prediction, next_state = residual_forward(
                    candidate_params, candidate_state, key, inputs, template, forcing
                )
                return next_state

            _, pullback = jax.vjp(jax.checkpoint(rematerialized_step), params, state)
            return pullback(state_cotangent)

        @jax.jit
        def supervised_pullback(
            params, state, key, inputs, target, forcing, loss_cotangent, state_cotangent
        ):
            def rematerialized_step(candidate_params, candidate_state):
                loss, _prediction, next_state = supervised_forward(
                    candidate_params, candidate_state, key, inputs, target, forcing
                )
                return loss, next_state

            _, pullback = jax.vjp(jax.checkpoint(rematerialized_step), params, state)
            return pullback((loss_cotangent, state_cotangent))

        @jax.jit
        def accumulate(left, right):
            return jax.tree_util.tree_map(lambda a, b: a + b, left, right)

        @jax.jit
        def apply_optimizer(params, optimizer_state, gradients):
            gradient_norm = optax.global_norm(gradients)
            updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, params)
            return optax.apply_updates(params, updates), next_optimizer_state, gradient_norm

        self._residual_forward = residual_forward
        self._supervised_forward = supervised_forward
        self._state_pullback = state_pullback
        self._supervised_pullback = supervised_pullback
        self._accumulate = accumulate
        self._apply_optimizer = apply_optimizer

    def _prepare(self, keys, inputs, targets, forcings):
        for name, values in (("inputs", inputs), ("forcings", forcings), ("keys", keys)):
            if len(values) != self.steps:
                raise ValueError(f"Expected {self.steps} {name}, got {len(values)}")
        if len(targets) == self.steps:
            selected_targets = tuple(targets[index] for index in self.supervised_indices)
        elif len(targets) == len(self.supervised_indices):
            selected_targets = tuple(targets)
        else:
            raise ValueError(
                f"Expected {self.steps} full-step or {len(self.supervised_indices)} "
                f"supervised targets, got {len(targets)}"
            )
        template = (
            _template(selected_targets[-1]) if len(self.supervised_indices) < self.steps else None
        )
        return _host(keys), selected_targets, template

    def evaluate_step(self, params, state, key, inputs, residual_target, forcing):
        """Return ``(loss, next_state, prediction)`` without retaining a tape."""
        state = cast_state_boundary_fp32(state)
        state_in = _zeros(state) if self.reset_state else state
        loss, prediction, next_state = self._supervised_forward(
            params, state_in, key, inputs, residual_target, forcing
        )
        if self.reset_state:
            next_state = _zeros(state)
        return loss, next_state, prediction

    def predict_step(self, params, state, key, inputs, template, forcing):
        """Return ``(prediction, next_state)`` for an unsupervised step."""
        state = cast_state_boundary_fp32(state)
        state_in = _zeros(state) if self.reset_state else state
        prediction, next_state = self._residual_forward(
            params, state_in, key, inputs, template, forcing
        )
        return prediction, _zeros(state) if self.reset_state else next_state

    def _forward(
        self, params, state, keys, inputs, targets, forcings, *, save_tape,
        retain_predictions, prediction_callback,
    ):
        host_keys, selected_targets, template = self._prepare(keys, inputs, targets, forcings)
        state_host = _host(cast_state_boundary_fp32(state))
        zero_state_host = _host(_zeros(state_host))
        if self.reset_state:
            state_host = zero_state_host
        state_tape = []
        host_losses = []
        predictions = []
        for index in range(self.steps):
            state_in = zero_state_host if self.reset_state else state_host
            if save_tape:
                state_tape.append(state_in)
            position = self.supervised_positions.get(index)
            if position is not None:
                loss, prediction, next_state = self._supervised_forward(
                    params, state_in, host_keys[index], inputs[index],
                    selected_targets[position], forcings[index],
                )
                host_losses.append(np.asarray(jax.device_get(loss)))
            else:
                prediction, next_state = self._residual_forward(
                    params, state_in, host_keys[index], inputs[index], template, forcings[index]
                )
            state_host = zero_state_host if self.reset_state else _host(next_state)
            if prediction_callback is not None:
                prediction_callback(index, prediction)
            if retain_predictions:
                predictions.append(_host(prediction))
            del prediction, next_state
        components = np.asarray(host_losses, dtype=np.float32)
        loss_value = np.sum(components * self.weights)
        return (
            jnp.asarray(loss_value, dtype=jnp.float32),
            jax.tree_util.tree_map(jnp.asarray, state_host),
            components,
            tuple(predictions),
            (host_keys, selected_targets, template, tuple(state_tape)),
        )

    def evaluate(
        self, params, state, keys, inputs, targets, forcings, *,
        retain_predictions=True, prediction_callback: Callable | None = None,
    ):
        """Return ``(loss, next_state, components, predictions)``.

        Use ``retain_predictions=False`` with the callback, or ``evaluate_step``
        with the streaming reader, to avoid a full physical prediction tape.
        """
        return self._forward(
            params, state, keys, inputs, targets, forcings, save_tape=False,
            retain_predictions=retain_predictions, prediction_callback=prediction_callback,
        )[:4]

    def loss_and_grad(
        self, params, state, keys, inputs, targets, forcings, *,
        retain_predictions=True, prediction_callback: Callable | None = None,
    ):
        """Return loss, next state, components, predictions and FP32 gradients."""
        loss, next_state, components, predictions, tape = self._forward(
            params, state, keys, inputs, targets, forcings, save_tape=True,
            retain_predictions=retain_predictions, prediction_callback=prediction_callback,
        )
        host_keys, selected_targets, template, state_tape = tape
        parameter_cotangent = _zeros(params)
        state_cotangent = _zeros(cast_state_boundary_fp32(state))
        for index in range(self.steps - 1, -1, -1):
            position = self.supervised_positions.get(index)
            if position is not None:
                step_parameter_cotangent, step_state_cotangent = self._supervised_pullback(
                    params, state_tape[index], host_keys[index], inputs[index],
                    selected_targets[position], forcings[index],
                    jnp.asarray(self.weights[position], dtype=jnp.float32), state_cotangent,
                )
            else:
                step_parameter_cotangent, step_state_cotangent = self._state_pullback(
                    params, state_tape[index], host_keys[index], inputs[index],
                    template, forcings[index], state_cotangent,
                )
            parameter_cotangent = self._accumulate(parameter_cotangent, step_parameter_cotangent)
            state_cotangent = _zeros(state_cotangent) if self.reset_state else step_state_cotangent
            # Match the maintained host driver and bound queued device work.
            jax.block_until_ready(parameter_cotangent)
            del step_parameter_cotangent, step_state_cotangent
        return loss, next_state, components, predictions, parameter_cotangent

    def __call__(self, params, state, optimizer_state, keys, inputs, targets, forcings):
        loss, next_state, components, _predictions, gradients = self.loss_and_grad(
            params, state, keys, inputs, targets, forcings, retain_predictions=False
        )
        next_params, next_optimizer_state, gradient_norm = self._apply_optimizer(
            params, optimizer_state, gradients
        )
        jax.block_until_ready(next_params)
        return next_params, next_state, next_optimizer_state, loss, gradient_norm, components


def make_cached_stepwise_train_step(
    *, config, model_config, task_config, stats, optimizer,
    sample_inputs=None, sample_targets=None,
) -> CachedStepwiseTrainer:
    """Build only the original residual predictor using its residual ModelConfig.

    The optional samples are accepted for factory compatibility; construction
    does not initialize either residual or baseline parameters. Initialization
    and the GC-overlay parameter merge remain the caller's responsibility.
    """
    del sample_inputs, sample_targets
    use_bf16 = config.precision == "bf16"

    def residual_prediction(inputs, targets, forcings):
        return make_residual_predictor(
            model_config, task_config, stats, config.architecture, use_bf16=use_bf16
        )(inputs, targets_template=targets, forcings=forcings)

    def residual_loss_and_predictions(inputs, residual_targets, forcings):
        return make_residual_predictor(
            model_config, task_config, stats, config.architecture, use_bf16=use_bf16
        ).loss_and_predictions(inputs, residual_targets, forcings)

    transforms = CachedResidualTransforms(
        residual_predict=hk.transform_with_state(residual_prediction),
        residual_loss_and_predictions=hk.transform_with_state(residual_loss_and_predictions),
    )
    return CachedStepwiseTrainer(config=config, optimizer=optimizer, transforms=transforms)
