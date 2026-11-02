"""Differentiable weather-feedback training with shared host-resident tapes.

Unlike the legacy state-only reverse pass, this backend propagates cotangents
through both previous weather frames, the residual recurrent state, and the
frozen GraphCast's *inputs*. GraphCast parameters are explicitly stopped and
never passed to the optimizer. One-step kernels are rematerialized in reverse;
the full weather rollout is not compiled into a single training executable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax


BACKEND = "explicit_weather_reverse_vjp"


def _helpers():
    # Delay this import: endpoint_step owns the public factories and dispatches
    # into this module only for the opt-in feedback mode.
    from . import endpoint_step
    return endpoint_step


def _check(transforms, config, baseline_state, input_steps):
    if config.feedback_mode != "closed_loop":
        raise ValueError("Differentiable weather backend requires feedback_mode='closed_loop'")
    if config.bptt_steps not in (2, 4, 8, 24) or not 0 <= config.ar_tail_k < config.bptt_steps:
        raise ValueError("Differentiable weather feedback requires 2/4/8/24 steps and a valid teacher prefix")
    if input_steps != 2:
        raise ValueError("Differentiable weather feedback requires two-frame input windows")
    if config.distributed.mode != "single":
        raise ValueError("Differentiable weather feedback initially supports one device")
    if jax.tree_util.tree_leaves(baseline_state):
        raise ValueError("Frozen GraphCast must have empty recurrent state")
    if config.loss_kind not in ("mse_fp32", "amse") or transforms.full_field_loss is None:
        raise ValueError("Differentiable feedback requires a full physical-field mse_fp32 or amse loss")
    pair = getattr(config, "objective_pair", None)
    if pair not in (None, "mse_amse_excess"):
        raise ValueError("Differentiable weather feedback supports only scalar or MSE/AMSE-excess objectives")
    if pair is not None and transforms.full_field_objectives is None:
        raise ValueError("Differentiable MSE/AMSE-excess feedback requires both full-field objectives")


def _pack(tree):
    return tuple(jax.tree_util.tree_leaves(tree))


def _unpack(treedef, leaves):
    return jax.tree_util.tree_unflatten(treedef, leaves)


class _WeatherSteps:
    def __init__(self, *, transforms, baseline_params, baseline_state, config, time_step, input_steps):
        _check(transforms, config, baseline_state, input_steps)
        self.transforms = transforms
        self.baseline_params = baseline_params
        self.baseline_state = baseline_state
        self.config = config
        self.time_step = time_step
        self.positions = {step: pos for pos, step in enumerate(config.supervised_step_indices)}
        self.weights = np.asarray(config.normalized_supervised_weights, np.float32)
        self.paired = getattr(config, "objective_pair", None) == "mse_amse_excess"
        self.cache = {}

    def feeds_weather(self, index):
        """Only the last teacher-window prediction starts generated weather."""
        return self.config.truth_prefix_steps - 1 <= index < self.config.bptt_steps - 1

    def validate(self, keys, input_frames, truths, forcings, optimizer_step):
        if len(input_frames) != self.config.truth_prefix_steps + 1:
            raise ValueError("closed_loop teacher frames must match truth_prefix_steps + 1")
        if len(truths) != len(self.positions):
            raise ValueError("Truth count does not match the configured supervised steps")
        if len(forcings) != self.config.bptt_steps or len(keys) != self.config.bptt_steps:
            raise ValueError("Keys and forcings must cover every feedback step")
        if getattr(self.config, "amse_lambda_ramp_steps", 0) and optimizer_step is None:
            raise ValueError("AMSE lambda ramp requires the completed optimizer_step index")
        return jnp.asarray(0 if optimizer_step is None else optimizer_step, dtype=jnp.int32)

    def kernel(self, previous, current, static, truth, forcing, index):
        ep = _helpers()
        trees = (previous, current, static, truth, forcing)
        definitions = tuple(jax.tree_util.tree_structure(tree) for tree in trees)
        cache_key = (index, *definitions)
        if cache_key in self.cache:
            return self.cache[cache_key]
        supervised = index in self.positions
        reset = self.config.temporal_state_policy == "reset_every_anchor"
        loss_fn = (self.transforms.full_field_objectives if self.paired
                   else self.transforms.full_field_loss)
        predictor = ((self.transforms.residual_supervised_predict or self.transforms.residual_predict)
                     if supervised else self.transforms.residual_predict)

        def frame_shape(prev, curr, fixed, target, force):
            inputs = ep.input_window_from_frames(prev, curr, fixed, step_index=index,
                        truth_prefix_steps=self.config.truth_prefix_steps, time_step=self.time_step)
            return ep.next_dynamic_frame(inputs, target, force, time_step=self.time_step)

        frame_definition = jax.tree_util.tree_structure(jax.eval_shape(frame_shape, *trees))

        def step(params, state, prev, curr, baseline_params, key, fixed, target, force, optimizer_step):
            previous, current, static, truth, forcing = (
                _unpack(definition, leaves)
                for definition, leaves in zip(definitions, (prev, curr, fixed, target, force), strict=True)
            )
            inputs = ep.input_window_from_frames(previous, current, ep._tree_stop(static),
                        step_index=index, truth_prefix_steps=self.config.truth_prefix_steps,
                        time_step=self.time_step)
            inputs = ep.cast_physical_boundary_fp32(inputs)
            truth = ep._tree_stop(ep.cast_physical_boundary_fp32(truth))
            forcing = ep._tree_stop(ep.cast_physical_boundary_fp32(forcing))
            template = ep._template_like(truth, jnp.float32)
            baseline, _ = self.transforms.baseline_predict.apply(
                ep._tree_stop(baseline_params), self.baseline_state, key, inputs, template, forcing)
            state_in = ep._tree_zeros_like(state) if reset else state
            residual, next_state = predictor.apply(params, state_in, key, inputs, template, forcing)
            # Do not stop either forecast here: both depend on earlier weather.
            full = ep._tree_add(ep.cast_physical_boundary_fp32(baseline),
                                ep.cast_physical_boundary_fp32(residual))
            if supervised:
                kwargs = ({"optimizer_step": optimizer_step}
                          if getattr(self.config, "amse_lambda_ramp_steps", 0) else {})
                loss_values, _ = loss_fn(full, truth, **kwargs)
                loss_values = jnp.asarray(loss_values, jnp.float32)
                loss = jnp.mean(loss_values, axis=1) if self.paired else jnp.mean(loss_values)
            else:
                loss = jnp.zeros((2,), jnp.float32) if self.paired else jnp.asarray(0., jnp.float32)
            next_state = ep.cast_state_boundary_fp32(next_state)
            if reset:
                next_state = ep._tree_zeros_like(next_state)
            next_frame = ep.next_dynamic_frame(inputs, full, forcing, time_step=self.time_step)
            return loss, next_state, _pack(next_frame)

        @jax.jit
        def pullback(params, state, prev, curr, baseline_params, key, fixed, target, force,
                     optimizer_step, loss_cotangent, state_cotangent, weather_cotangent):
            def differentiable(p, s, previous, current):
                return step(p, s, previous, current, baseline_params, key, fixed, target, force, optimizer_step)
            _, reverse = jax.vjp(jax.checkpoint(differentiable), params, state, prev, curr)
            return reverse((loss_cotangent, state_cotangent, weather_cotangent))

        result = (step, jax.jit(step), pullback, frame_definition)
        self.cache[cache_key] = result
        return result

    def forward(self, params, state, keys, input_frames, static, truths, forcings, *, optimizer_step=None):
        """Stage one step at a time and retain only host weather/state boundaries."""
        ep = _helpers()
        step_index = self.validate(keys, input_frames, truths, forcings, optimizer_step)
        to_host = lambda value: jax.device_get(ep._tree_stop(value))
        frames = [ep._host_cast_floating(frame, np.float32) for frame in input_frames]
        state = to_host(ep.cast_state_boundary_fp32(state))
        static = ep._host_cast_floating(static, np.float32)
        truths = tuple(ep._host_cast_floating(value, np.float32) for value in truths)
        forcings = tuple(ep._host_cast_floating(value, np.float32) for value in forcings)
        keys = to_host(keys)
        tape, components = [], []
        for index in range(self.config.bptt_steps):
            truth = truths[self.positions[index]] if index in self.positions else truths[-1]
            raw, forward, reverse, frame_definition = self.kernel(
                frames[index], frames[index + 1], static, truth, forcings[index], index)
            del raw
            arguments = (params, state, _pack(frames[index]), _pack(frames[index + 1]),
                         self.baseline_params, keys[index], _pack(static), _pack(truth),
                         _pack(forcings[index]), step_index)
            loss, next_state, next_frame = forward(*arguments)
            # Params stay on device; boundaries and staged data are host arrays.
            tape.append((reverse, arguments[1:], frame_definition))
            if index in self.positions:
                components.append(np.asarray(jax.device_get(loss), np.float32))
            state = to_host(next_state)
            if self.feeds_weather(index):
                frames.append(_unpack(frame_definition, to_host(next_frame)))
        if len(frames) != self.config.bptt_steps + 1:
            raise RuntimeError("Differentiable weather tape has incorrect length")
        components = np.asarray(components, np.float32)
        return np.tensordot(self.weights, components, axes=1), state, components, frames, tape

    def reverse(self, params, initial_state, frames, tape, *, objective=None, scale=1.):
        """Differentiate one objective through the same retained forward tape.

        Each invocation owns its cotangents. Two objective sweeps therefore
        share predictions, keys and state boundaries without advancing them or
        retaining two weather-cotangent tapes simultaneously.
        """
        ep = _helpers()
        if self.paired and objective not in (0, 1):
            raise ValueError("Paired weather reverse requires objective 0 or 1")
        if not self.paired and objective is not None:
            raise ValueError("Scalar weather reverse does not accept an objective index")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Objective gradient scale must be finite and positive")
        parameter_cotangent = ep._tree_zeros_like(params)
        state_cotangent = ep._tree_zeros_like(initial_state)
        weather_cotangents = [jax.tree_util.tree_map(np.zeros_like, _pack(frame)) for frame in frames]
        for index in range(self.config.bptt_steps - 1, -1, -1):
            reverse, arguments, _ = tape[index]
            # Earlier teacher forecasts and the final prediction are scored,
            # but their outputs do not become weather inputs. In particular,
            # an observed frame at index+2 is not that teacher prediction.
            weather_cotangent = (weather_cotangents[index + 2] if self.feeds_weather(index)
                                 else ep._tree_zeros_like(_pack(frames[index + 1])))
            weight = self.weights[self.positions[index]] / scale if index in self.positions else 0.
            seed = (jnp.zeros((2,), jnp.float32).at[objective].set(weight) if self.paired
                    else jnp.asarray(weight, jnp.float32))
            pg, sg, previous_grad, current_grad = reverse(
                params, *arguments, seed, state_cotangent, weather_cotangent)
            parameter_cotangent = ep._tree_add(parameter_cotangent, pg)
            state_cotangent = sg
            # Both overlapping input windows contribute to a generated frame.
            weather_cotangents[index] = jax.device_get(ep._tree_add(weather_cotangents[index], previous_grad))
            weather_cotangents[index + 1] = jax.device_get(ep._tree_add(weather_cotangents[index + 1], current_grad))
            jax.block_until_ready(parameter_cotangent)
        return parameter_cotangent

    def primary(self, value, components):
        """Public scalar interfaces retain exact GC-MSE for paired training."""
        return (value[0], components[:, 0]) if self.paired else (value, components)


def make_weather_feedback_objective(*, return_loss_components=False, **kwargs):
    """Reference/diagnostic AD objective; production training uses streamed VJPs.

    This rematerialized unroll is useful for small derivative checks. The runner
    never JIT-compiles or differentiates the entire rollout through this helper.
    """
    engine = _WeatherSteps(**kwargs)
    ep = _helpers()

    def objective(params, state, keys, input_frames, static, truths, forcings, optimizer_step=None):
        step_index = engine.validate(keys, input_frames, truths, forcings, optimizer_step)
        state = ep._tree_stop(ep.cast_state_boundary_fp32(state))
        frames = [ep._tree_stop(ep.cast_physical_boundary_fp32(frame)) for frame in input_frames]
        losses = []
        for index in range(engine.config.bptt_steps):
            truth = truths[engine.positions[index]] if index in engine.positions else truths[-1]
            raw, _, _, frame_definition = engine.kernel(
                frames[index], frames[index + 1], static, truth, forcings[index], index)
            loss, state, next_frame = jax.checkpoint(raw)(
                params, state, _pack(frames[index]), _pack(frames[index + 1]), engine.baseline_params,
                keys[index], _pack(static), _pack(truth), _pack(forcings[index]), step_index)
            if index in engine.positions:
                losses.append(loss)
            if engine.feeds_weather(index):
                frames.append(_unpack(frame_definition, next_frame))
        components = jnp.stack(losses)
        loss = jnp.tensordot(jnp.asarray(engine.weights), components, axes=1)
        loss, components = engine.primary(loss, components)
        output = (loss, ep._tree_stop(state))
        return (*output, ep._tree_stop(components)) if return_loss_components else output

    return objective


def make_weather_feedback_train_step(*, optimizer, **kwargs):
    engine = _WeatherSteps(**kwargs)
    if engine.paired:
        raise ValueError("Paired weather objectives must use the multiobjective optimizer factory")

    @jax.jit
    def update(params, optimizer_state, gradients):
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, params)
        return optax.apply_updates(params, updates), next_optimizer_state, optax.global_norm(gradients)

    def train_step(params, state, optimizer_state, keys, input_frames, static, truths, forcings,
                   optimizer_step=None):
        value, final_state, components, frames, tape = engine.forward(
            params, state, keys, input_frames, static, truths, forcings, optimizer_step=optimizer_step)
        parameter_cotangent = engine.reverse(params, state, frames, tape)
        next_params, next_optimizer_state, gradient_norm = update(params, optimizer_state, parameter_cotangent)
        return (next_params, jax.tree_util.tree_map(jnp.asarray, final_state), next_optimizer_state,
                jnp.asarray(value, jnp.float32), gradient_norm, components)

    return train_step


def make_weather_feedback_validation_step(**kwargs):
    engine = _WeatherSteps(**kwargs)

    def validation_step(params, state, keys, input_frames, static, truths, forcings, optimizer_step=None):
        if optimizer_step is None and getattr(engine.config, "amse_lambda_ramp_steps", 0):
            # Match validation's fixed target-lambda convention, independent of
            # the training ramp's current position.
            optimizer_step = engine.config.amse_lambda_ramp_steps
        value, final_state, components, _, _ = engine.forward(
            params, state, keys, input_frames, static, truths, forcings, optimizer_step=optimizer_step)
        value, components = engine.primary(value, components)
        return jnp.asarray(value, jnp.float32), jax.tree_util.tree_map(jnp.asarray, final_state), components

    return validation_step
