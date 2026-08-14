"""Legacy-compatible v22_final BPTT loss and optimizer update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from src.models.graphcast.training.core.model import scalarize_loss

from ..model import (
    V22FinalModelConfigs,
    make_baseline_predictor,
    make_residual_predictor,
)
from ..rollout import shift_inputs_with_field
from .config import V22FinalTrainConfig


@dataclass(frozen=True)
class V22FinalTrainingTransforms:
    baseline_predict: hk.TransformedWithState
    residual_predict: hk.TransformedWithState
    residual_loss: hk.TransformedWithState


def build_training_transforms(
    model_configs: V22FinalModelConfigs,
    task_config,
    stats,
    config: V22FinalTrainConfig,
) -> V22FinalTrainingTransforms:
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

    def residual_loss(inputs, residual_targets, forcings):
        return make_residual_predictor(
            model_configs.residual,
            task_config,
            stats,
            config.architecture,
            use_bf16=use_bf16,
        ).loss(inputs, residual_targets, forcings)

    return V22FinalTrainingTransforms(
        baseline_predict=hk.transform_with_state(baseline_prediction),
        residual_predict=hk.transform_with_state(residual_prediction),
        residual_loss=hk.transform_with_state(residual_loss),
    )


def build_optimizer(config: V22FinalTrainConfig):
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


def residual_target(truth, baseline_prediction):
    return jax.tree_util.tree_map(
        lambda target, baseline: target - jax.lax.stop_gradient(baseline),
        truth,
        baseline_prediction,
    )


def feedback_field(mode: str, baseline_prediction, residual_prediction):
    if mode == "baseline":
        return baseline_prediction
    if mode == "closed_loop_sg":
        return jax.tree_util.tree_map(
            lambda baseline, residual: baseline + jax.lax.stop_gradient(residual),
            baseline_prediction,
            residual_prediction,
        )
    raise ValueError(f"Unsupported feedback mode: {mode!r}")


def make_bptt_objective(
    *,
    transforms: V22FinalTrainingTransforms,
    baseline_params,
    baseline_state,
    config: V22FinalTrainConfig,
    time_step,
    input_steps: int,
) -> Callable[..., tuple[Any, Any]]:
    """Build the shared legacy-compatible BPTT forward objective."""

    def one_ar_step(
        residual_params,
        residual_state,
        key,
        current_inputs,
        truth,
        forcings,
    ):
        baseline_prediction, _ = transforms.baseline_predict.apply(
            baseline_params,
            baseline_state,
            key,
            current_inputs,
            truth,
            forcings,
        )
        residual_prediction, next_residual_state = transforms.residual_predict.apply(
            residual_params,
            residual_state,
            key,
            current_inputs,
            truth,
            forcings,
        )
        target = residual_target(truth, baseline_prediction)
        (loss_array, _diagnostics), _unused_loss_state = transforms.residual_loss.apply(
            residual_params,
            residual_state,
            key,
            current_inputs,
            target,
            forcings,
        )
        return (
            baseline_prediction,
            residual_prediction,
            scalarize_loss(loss_array),
            next_residual_state,
        )

    checkpointed_ar_step = jax.checkpoint(one_ar_step, static_argnums=())

    def bptt_objective(
        residual_params,
        residual_state,
        keys,
        truth_inputs,
        truths,
        forcings,
    ):
        reset_every_anchor = config.temporal_state_policy == "reset_every_anchor"
        zero_state = jax.tree_util.tree_map(jnp.zeros_like, residual_state)
        state = zero_state if reset_every_anchor else residual_state
        losses = []
        current_inputs = truth_inputs[0]
        for index in range(config.bptt_steps):
            baseline_prediction, residual_prediction, loss, next_state = checkpointed_ar_step(
                residual_params,
                state,
                keys[index],
                current_inputs,
                truths[index],
                forcings[index],
            )
            state = zero_state if reset_every_anchor else next_state
            losses.append(loss)
            if index >= config.bptt_steps - 1:
                continue
            next_index = index + 1
            if next_index < config.truth_prefix_steps:
                current_inputs = truth_inputs[next_index]
            else:
                next_field = feedback_field(
                    config.feedback_mode,
                    baseline_prediction,
                    residual_prediction,
                )
                # The just-used forcing belongs to the frame being inserted.
                current_inputs = shift_inputs_with_field(
                    current_inputs,
                    next_field,
                    forcings[index],
                    time_step=time_step,
                    input_steps=input_steps,
                )
        return jnp.stack(losses).mean(), state

    return bptt_objective


def make_train_step(
    *,
    transforms: V22FinalTrainingTransforms,
    optimizer,
    baseline_params,
    baseline_state,
    config: V22FinalTrainConfig,
    time_step,
    input_steps: int,
) -> Callable[..., tuple[Any, Any, Any, Any, Any]]:
    """Build the static-shape BPTT update used by the v20 reference."""

    bptt_objective = make_bptt_objective(
        transforms=transforms,
        baseline_params=baseline_params,
        baseline_state=baseline_state,
        config=config,
        time_step=time_step,
        input_steps=input_steps,
    )

    @jax.jit
    def train_step(
        residual_params,
        residual_state,
        optimizer_state,
        keys,
        truth_inputs,
        truths,
        forcings,
    ):
        def objective(params):
            return bptt_objective(
                params,
                residual_state,
                keys,
                truth_inputs,
                truths,
                forcings,
            )

        (loss, next_residual_state), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(residual_params)
        gradient_norm = optax.global_norm(gradients)
        updates, next_optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            residual_params,
        )
        next_residual_params = optax.apply_updates(residual_params, updates)
        next_residual_state = jax.tree_util.tree_map(
            jax.lax.stop_gradient, next_residual_state
        )
        return (
            next_residual_params,
            next_residual_state,
            next_optimizer_state,
            loss,
            gradient_norm,
        )

    return train_step


def make_validation_step(
    *,
    transforms: V22FinalTrainingTransforms,
    baseline_params,
    baseline_state,
    config: V22FinalTrainConfig,
    time_step,
    input_steps: int,
) -> Callable[..., tuple[Any, Any]]:
    """Build a no-gradient evaluation of the exact training objective."""

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
        truth_inputs,
        truths,
        forcings,
    ):
        loss, next_residual_state = bptt_objective(
            residual_params,
            residual_state,
            keys,
            truth_inputs,
            truths,
            forcings,
        )
        next_residual_state = jax.tree_util.tree_map(
            jax.lax.stop_gradient, next_residual_state
        )
        return loss, next_residual_state

    return validation_step
