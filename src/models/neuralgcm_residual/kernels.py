"""Decoded-loss recurrent BPTT with a host tape and no solver differentiation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from .native_state import stop


def make_optimizer(config):
    def schedule(step):
        warm = jnp.minimum((step + 1) / max(1, config.warmup_updates), 1.0)
        return jnp.float32(config.learning_rate) * warm
    return optax.chain(optax.clip_by_global_norm(config.clip_norm), optax.adamw(
        schedule, b1=config.beta1, b2=config.beta2, weight_decay=config.weight_decay))


class DecodedTrainer:
    def __init__(self, branch, adapter, normalization, backbone, weather_loss, optimizer):
        self.branch, self.adapter, self.normalization = branch, adapter, normalization
        self.backbone, self.weather_loss, self.optimizer = backbone, weather_loss, optimizer

        def step(params, memory, key, native, known, baseline, forcing, target):
            # All physical inputs are SG. Memory alone carries temporal derivatives.
            delta, next_memory = branch.apply(params, memory, key, stop(native), stop(known))
            corrected = adapter.apply_increment(stop(baseline), normalization.increment(delta))
            prediction = backbone.decode(corrected, stop(forcing))
            return weather_loss(prediction, target), next_memory, corrected

        self.forward = jax.jit(step)

        @jax.jit
        def backward(params, memory, key, record, loss_cotangent, memory_cotangent):
            def differentiable(p, h):
                loss, next_h, _ = step(p, h, key, *record)
                return loss, next_h
            _, pullback = jax.vjp(jax.checkpoint(differentiable), params, memory)
            return pullback((loss_cotangent, memory_cotangent))
        self.backward = backward

        @jax.jit
        def update(params, opt_state, gradient):
            updates, next_opt = optimizer.update(gradient, opt_state, params)
            return optax.apply_updates(params, updates), next_opt, optax.global_norm(gradient)
        self.apply_update = update

    def record(self, origin_state, baseline_state, forcing, target, known):
        return (self.normalization.inputs(self.adapter.features(stop(origin_state))), known,
                stop(baseline_state), stop(forcing), target)

    def reverse(self, params, tape, *, weights=None):
        if not tape:
            raise ValueError("No scored timesteps")
        weights = np.full(len(tape), 1 / len(tape), np.float32) if weights is None else np.asarray(weights, np.float32)
        if len(weights) != len(tape) or not np.isclose(weights.sum(), 1):
            raise ValueError("Loss weights must sum to one")
        gradient = jax.tree_util.tree_map(jnp.zeros_like, params)
        memory_grad = jax.tree_util.tree_map(jnp.zeros_like, tape[-1][0])
        for weight, (memory, key, record) in zip(weights[::-1], reversed(tape), strict=True):
            dp, memory_grad = self.backward(params, memory, key, record, jnp.float32(weight), memory_grad)
            gradient = jax.tree_util.tree_map(lambda a, b: a + b, gradient, dp)
        return gradient

    def cached_gradients(self, params, memory, rng, records):
        tape, losses = [], []
        for record in records:
            rng, key = jax.random.split(rng)
            tape.append((jax.device_get(memory), jax.device_get(key), jax.device_get(record)))
            value, memory, _ = self.forward(params, memory, key, *record)
            losses.append(float(value))
        return np.mean(losses), stop(memory), rng, self.reverse(params, tape)

    def checked_update(self, params, optimizer, gradients):
        if any(not np.isfinite(x).all() for x in jax.device_get(jax.tree_util.tree_leaves(gradients))):
            raise FloatingPointError("Nonfinite gradients; no update was applied")
        return self.apply_update(params, optimizer, gradients)
