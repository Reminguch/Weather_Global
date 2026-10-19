"""Shared RMS metric for MGDA and its matched single-objective control.

No first moment, weight decay, Adam transform, or post-solve group scaling is
applied. The parameter-group ratio is part of the metric used by the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class RMSState(NamedTuple):
    count: jax.Array
    second_moment: object


def _map(function, *trees):
    return jax.tree_util.tree_map(function, *trees)


def tree_dot(left, right):
    products = jax.tree_util.tree_leaves(_map(
        lambda a, b: jnp.sum(jnp.asarray(a, jnp.float32) * jnp.asarray(b, jnp.float32)), left, right))
    return sum(products, jnp.asarray(0, jnp.float32))


def tree_norm(tree):
    return jnp.sqrt(jnp.maximum(tree_dot(tree, tree), 0))


def _finite(tree):
    checks = [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def warmup_cosine_constant_schedule(config):
    """Optax-compatible zero-based schedule, independent of ``max_steps``."""
    def schedule(completed_updates):
        t = jnp.asarray(completed_updates, jnp.float32) + 1
        fraction = jnp.clip((t - 1) / max(config.warmup_steps - 1, 1), 0, 1)
        warmup = config.warmup_start_learning_rate + fraction * (
            config.learning_rate - config.warmup_start_learning_rate)
        phase = jnp.clip((t - config.warmup_steps) /
                         (config.decay_end_step - config.warmup_steps), 0, 1)
        decay = config.end_learning_rate + 0.5 * (config.learning_rate - config.end_learning_rate) * (
            1 + jnp.cos(jnp.pi * phase))
        return jnp.where(t <= config.warmup_steps, warmup,
                         jnp.where(t <= config.decay_end_step, decay, config.end_learning_rate))
    return schedule


def _group_multipliers(params, spatial, mamba):
    def multiplier(path, _value):
        module = str(getattr(path[0], "key", "")) if path else ""
        name = module.lower()
        temporal = name.startswith("mesh_interleaved_temporal_") or "temporal_mesh_mamba" in name
        return jnp.asarray(mamba if temporal else spatial, jnp.float32)
    return jax.tree_util.tree_map_with_path(multiplier, params)


def metric_mgda_weight(first, second, metric):
    """Solve the two-objective segment projection in a positive diagonal metric."""
    weighted_first = _map(lambda d, g: d * g, metric, first)
    weighted_second = _map(lambda d, g: d * g, metric, second)
    a, b, c = tree_dot(first, weighted_first), tree_dot(first, weighted_second), tree_dot(second, weighted_second)
    difference = _map(lambda x, y: x - y, first, second)
    denominator = tree_dot(difference, _map(lambda d, g: d * g, metric, difference))
    # The protected denominator avoids inactive-branch 0/0 and makes identical
    # and both-zero gradients deterministic, including under jax.jit.
    alpha = jnp.where(denominator > 0,
                      jnp.clip((c - b) / jnp.where(denominator > 0, denominator, 1), 0, 1), 0.5)
    return alpha, {"metric_gc_gc": a, "metric_gc_power": b, "metric_power_power": c,
                   "metric_difference_squared": denominator}


@dataclass(frozen=True)
class RMSOptimizer:
    schedule: object
    decay: float = 0.98
    epsilon: float = 1e-8
    clip: float = 1.0
    spatial_multiplier: float = 1.0
    mamba_multiplier: float = 3.0
    direction_mode: str = "mgda"

    def init(self, params):
        return RMSState(jnp.asarray(0, jnp.int32), _map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params))

    def update_pair(self, g_gc, g_power, state, params):
        return self._step(g_gc, g_power, state, params)

    def update_single(self, gradients, state, params):
        return self._step(gradients, None, state, params)

    def update(self, gradients, state, params=None):
        if params is None:
            raise ValueError("RMS optimizer requires parameters for actual-displacement checks")
        updates, next_state, _ = self.update_single(gradients, state, params)
        return updates, next_state

    def _step(self, first, second, state, params):
        first = _map(lambda x: jnp.asarray(x, jnp.float32), first)
        if second is not None:
            second = _map(lambda x: jnp.asarray(x, jnp.float32), second)
        squared = (_map(lambda x: x ** 2, first) if second is None else
                   _map(lambda x, y: 0.5 * (x ** 2 + y ** 2), first, second))
        count = state.count + 1
        moment = _map(lambda previous, current: self.decay * previous + (1 - self.decay) * current,
                      state.second_moment, squared)
        correction = 1 - jnp.asarray(self.decay, jnp.float32) ** count
        group = _group_multipliers(params, self.spatial_multiplier, self.mamba_multiplier)
        metric = _map(lambda v, mult: mult / (jnp.sqrt(v / correction) + self.epsilon), moment, group)
        if second is None:
            alpha, diagnostics, combined = jnp.asarray(1, jnp.float32), {}, first
        else:
            alpha, diagnostics = metric_mgda_weight(first, second, metric)
            if self.direction_mode == "mgda":
                combined = _map(lambda x, y: alpha * x + (1 - alpha) * y, first, second)
            elif self.direction_mode == "mse_control":
                alpha, combined = jnp.asarray(1., jnp.float32), first
            elif self.direction_mode == "fixed_sum":
                alpha = jnp.asarray(.5, jnp.float32)
                combined = _map(lambda x, y: x + y, first, second)
            else:
                raise ValueError("Unknown pair-RMS direction mode")
        norm = tree_norm(combined)
        clip_factor = (jnp.minimum(1.0, self.clip / jnp.maximum(norm, jnp.finfo(jnp.float32).tiny))
                       if self.clip > 0 else jnp.asarray(1, jnp.float32))
        direction = _map(lambda d, g: d * (g * clip_factor), metric, combined)
        learning_rate = self.schedule(state.count)
        updates = _map(lambda d: -learning_rate * d, direction)
        # Mirrors optax.apply_updates, including the destination parameter dtype.
        displacement = _map(lambda p, u: jnp.asarray(p, jnp.float32) -
                             jnp.asarray((p + u).astype(p.dtype), jnp.float32), params, updates)
        direction_norm, displacement_norm = tree_norm(direction), tree_norm(displacement)
        first_norm = tree_norm(first)
        first_direction, first_displacement = tree_dot(first, direction), tree_dot(first, displacement)
        first_valid = ((first_direction >= -1e-6 * first_norm * direction_norm) &
                       (first_displacement >= -1e-6 * first_norm * displacement_norm))
        diagnostics.update(
            mgda_alpha=alpha, gc_gradient_norm=first_norm, combined_gradient_norm=norm,
            clip_factor=clip_factor, direction_norm=direction_norm,
            update_norm=displacement_norm, learning_rate=learning_rate,
            gc_direction_dot=first_direction, gc_displacement_dot=first_displacement,
            stalled=(direction_norm == 0) | (displacement_norm == 0),
        )
        valid = first_valid
        if second is not None:
            second_norm = tree_norm(second)
            second_direction, second_displacement = tree_dot(second, direction), tree_dot(second, displacement)
            diagnostics.update(
                power_gradient_norm=second_norm,
                gradient_cosine=tree_dot(first, second) / jnp.maximum(first_norm * second_norm, 1e-30),
                power_direction_dot=second_direction, power_displacement_dot=second_displacement,
            )
            second_valid = ((second_direction >= -1e-6 * second_norm * direction_norm) &
                            (second_displacement >= -1e-6 * second_norm * displacement_norm))
            if self.direction_mode == "mgda":
                valid &= second_valid
            elif self.direction_mode == "fixed_sum":
                sum_norm = tree_norm(_map(lambda x, y: x + y, first, second))
                valid = ((first_direction + second_direction >= -1e-6 * sum_norm * direction_norm) &
                         (first_displacement + second_displacement >= -1e-6 * sum_norm * displacement_norm))
        finite = _finite((first, second, metric, moment, updates, displacement, diagnostics))
        diagnostics["finite"] = finite
        diagnostics["descent_valid"] = valid & finite
        return updates, RMSState(count, moment), diagnostics


def validate_optimizer_diagnostics(diagnostics):
    """Host-side guard: call before accepting an update returned from a JIT."""
    values = jax.device_get(diagnostics)
    if not bool(np.asarray(values["finite"])):
        raise FloatingPointError("Nonfinite RMS/MGDA gradients, moments, updates or diagnostics")
    if not bool(np.asarray(values["descent_valid"])):
        raise FloatingPointError("RMS/MGDA proposed or applied update violates common descent beyond FP32 tolerance")


def build_rms_optimizer(config):
    if config.learning_rate_schedule != "warmup_cosine_constant":
        raise ValueError("RMS pilot requires immutable warmup_cosine_constant schedule")
    if config.weight_decay != 0:
        raise ValueError("RMS pilot cannot apply weight decay")
    schedule = warmup_cosine_constant_schedule(config)
    return RMSOptimizer(schedule, config.rms_decay, config.rms_epsilon, config.grad_clip,
                        config.spatial_lr_multiplier, config.mamba_lr_multiplier,
                        getattr(config, "mgda_direction", "mgda")), schedule
