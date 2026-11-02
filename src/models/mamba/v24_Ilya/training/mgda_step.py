"""Two objectives on one trajectory and a shared explicit reverse tape."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .endpoint_step import (
    _cast_floating, _host_cast_floating, _reusable_fp32_host_tree,
    _template_like, _tree_add, _tree_stop, _tree_zeros_like,
    cast_state_boundary_fp32, feedback_field, input_window_from_frames,
    next_dynamic_frame, supervision_target,
)


class MultiObjectiveResult(NamedTuple):
    params: Any
    state: Any
    optimizer_state: Any
    loss: Any  # Raw exact GC objective; never a changing weighted scalar.
    gradient_norm: Any
    loss_components: Any  # GC components, compatible with existing logging.
    diagnostics: dict[str, Any]


class JointDescentFailure(RuntimeError):
    """A rejected batch: no parameters, optimizer state or memory are committed."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("No accepted finite-step decrease: " + str(diagnostics.get("reason")))


def make_multiobjective_train_step(*, transforms, optimizer, baseline_params,
                                  baseline_state, config, time_step, input_steps):
    """Build a callable with an attached ``forward_only`` calibration method.

    Both gradients use the exact same forward predictions and recurrent state.
    The second reverse pass never advances state, RNG or the data cursor.
    """
    if input_steps != 2 or config.feedback_mode not in ("baseline", "closed_loop_sg", "closed_loop"):
        raise ValueError("Multiobjective training requires two-frame inputs and a supported feedback mode")
    if config.distributed.mode != "single":
        raise ValueError("Multiobjective training currently requires a single device")
    if jax.tree_util.tree_leaves(baseline_state):
        raise ValueError("Frozen GraphCast must have empty recurrent state")
    if transforms.full_field_loss is None:
        raise ValueError("Multiobjective training requires FP32 full-field GC loss")
    positions = {index: pos for pos, index in enumerate(config.supervised_step_indices)}
    amse_split = getattr(config, "objective_pair", None) == "amse_split"
    amse_excess = getattr(config, "objective_pair", None) == "mse_amse_excess"
    weather_engine = None
    if config.feedback_mode == "closed_loop":
        if not amse_excess:
            raise ValueError("Differentiable multiobjective weather feedback requires MSE/AMSE excess")
        from .weather_feedback_step import _WeatherSteps

        weather_engine = _WeatherSteps(
            transforms=transforms, baseline_params=baseline_params,
            baseline_state=baseline_state, config=config, time_step=time_step,
            input_steps=input_steps)
    weights = np.asarray(config.normalized_supervised_weights, np.float32)
    final_index = config.bptt_steps - 1
    reset_state = config.temporal_state_policy == "reset_every_anchor"

    def host(tree):
        return jax.device_get(_tree_stop(tree))

    @jax.jit
    def baseline_forward(key, inputs, target, forcing):
        prediction, _ = transforms.baseline_predict.apply(
            baseline_params, baseline_state, key, _tree_stop(inputs), target, forcing)
        return _tree_stop(_cast_floating(prediction, jnp.float32))

    @jax.jit
    def residual_forward(params, state, key, inputs, target, forcing):
        prediction, next_state = transforms.residual_predict.apply(
            params, state, key, inputs, target, forcing)
        return prediction, cast_state_boundary_fp32(next_state)

    @jax.jit
    def supervised_forward(params, state, key, inputs, target, forcing):
        truth, baseline = target
        predictor = transforms.residual_supervised_predict or transforms.residual_predict
        prediction, next_state = predictor.apply(
            params, state, key, inputs, _template_like(truth, jnp.float32), forcing)
        full = jax.tree_util.tree_map(
            lambda base, residual: jax.lax.stop_gradient(base) + residual,
            _cast_floating(baseline, jnp.float32), _cast_floating(prediction, jnp.float32))
        if transforms.full_field_objectives is None:
            gc, _ = transforms.full_field_loss(full, truth)
            raw = jnp.stack((gc, jnp.zeros_like(gc)))
            floor_fraction = jnp.asarray(0., jnp.float32)
        else:
            raw, diagnostics = transforms.full_field_objectives(full, truth)
            floor_fraction = jnp.mean(diagnostics.get("floor_fraction", jnp.asarray(0.)))
            gc = diagnostics["gc_loss"] if amse_split else raw[0]
        losses = jnp.mean(jnp.asarray(raw, jnp.float32), axis=1)
        return losses, prediction, cast_state_boundary_fp32(next_state), floor_fraction, jnp.mean(gc)

    @jax.jit
    def supervised_reverse(params, state, key, inputs, target, forcing,
                           loss_cotangent, state_cotangent):
        def rematerialize(candidate_params, candidate_state):
            losses, _, next_state, _, _ = supervised_forward(
                candidate_params, candidate_state, key, inputs, target, forcing)
            return losses, next_state
        _, pullback = jax.vjp(jax.checkpoint(rematerialize), params, state)
        return pullback((loss_cotangent, state_cotangent))

    @jax.jit
    def state_reverse(params, state, key, inputs, target, forcing, cotangent):
        def rematerialize(candidate_params, candidate_state):
            _, next_state = residual_forward(
                candidate_params, candidate_state, key, inputs, target, forcing)
            return next_state
        _, pullback = jax.vjp(jax.checkpoint(rematerialize), params, state)
        return pullback(cotangent)

    accumulate = jax.jit(_tree_add)

    def forward(params, initial_state, keys, input_frames, static_inputs, truths, forcings):
        if weather_engine is not None:
            raw, state, components, frames, steps = weather_engine.forward(
                params, initial_state, keys, input_frames, static_inputs, truths, forcings)
            # The excess pair's first component is exact grid GC-MSE. It has
            # no truth-power flooring or independent objective normalization.
            return (frames, steps), state, raw, components, 0., components[:, 0]
        if len(input_frames) != config.truth_prefix_steps + 1:
            raise ValueError("Incorrect teacher-frame count for multiobjective step")
        if len(truths) != len(positions) or len(forcings) != config.bptt_steps:
            raise ValueError("Incorrect truth/forcing count for multiobjective step")
        keys = host(keys)
        static = _host_cast_floating(static_inputs, np.float32)
        truths = tuple(_reusable_fp32_host_tree(value) for value in truths)
        forcings = tuple(_host_cast_floating(value, np.float32) for value in forcings)
        frames = [_host_cast_floating(value, np.float32) for value in input_frames]
        state = host(cast_state_boundary_fp32(initial_state))
        zero = host(_tree_zeros_like(state))
        if reset_state:
            state = zero
        states, targets, losses, floor_fractions, gc_losses = [], list(truths), [], [], []
        for index in range(config.bptt_steps):
            state_in = zero if reset_state else state
            states.append(state_in)
            inputs = input_window_from_frames(
                frames[index], frames[index + 1], static, step_index=index,
                truth_prefix_steps=config.truth_prefix_steps, time_step=time_step)
            pos = positions.get(index)
            needs_feedback = config.truth_prefix_steps - 1 <= index < final_index
            baseline = None
            if pos is not None:
                baseline = baseline_forward(keys[index], inputs, truths[pos], forcings[index])
                target = supervision_target(transforms, truths[pos], baseline)
                values, prediction, next_state, floor_fraction, gc = supervised_forward(
                    params, state_in, keys[index], inputs, target, forcings[index])
                losses.append(np.asarray(jax.device_get(values), np.float32))
                floor_fractions.append(float(jax.device_get(floor_fraction)))
                gc_losses.append(float(jax.device_get(gc)))
                targets[pos] = (truths[pos], host(target[1]))
            else:
                template = _template_like(truths[-1], jnp.float32)
                if needs_feedback:
                    baseline = baseline_forward(keys[index], inputs, template, forcings[index])
                prediction, next_state = residual_forward(
                    params, state_in, keys[index], inputs, template, forcings[index])
            state = zero if reset_state else host(next_state)
            if needs_feedback:
                frame = next_dynamic_frame(inputs, feedback_field(
                    config.feedback_mode, baseline, prediction), forcings[index], time_step=time_step)
                frames.append(host(_cast_floating(frame, jnp.float32)))
            del inputs, prediction, next_state, baseline
        if len(frames) != config.bptt_steps + 1:
            raise RuntimeError("Multiobjective weather tape has incorrect length")
        tape = (keys, static, truths, forcings, frames, states, targets)
        components = np.asarray(losses, np.float32)
        raw = np.sum(components * weights[:, None], axis=0)
        return (tape, state, raw, components, float(np.dot(weights, floor_fractions)),
                np.asarray(gc_losses, np.float32))

    def reverse(params, initial_state, tape, objective, scale):
        if weather_engine is not None:
            frames, steps = tape
            return weather_engine.reverse(
                params, initial_state, frames, steps, objective=objective, scale=scale)
        keys, static, truths, forcings, frames, states, targets = tape
        gradient, cotangent = _tree_zeros_like(params), _tree_zeros_like(initial_state)
        for index in range(final_index, -1, -1):
            inputs = input_window_from_frames(
                frames[index], frames[index + 1], static, step_index=index,
                truth_prefix_steps=config.truth_prefix_steps, time_step=time_step)
            pos = positions.get(index)
            if pos is None:
                step_grad, next_cotangent = state_reverse(
                    params, states[index], keys[index], inputs,
                    _template_like(truths[-1], jnp.float32), forcings[index], cotangent)
            else:
                loss_cotangent = jnp.zeros((2,), jnp.float32).at[objective].set(weights[pos] / scale)
                step_grad, next_cotangent = supervised_reverse(
                    params, states[index], keys[index], inputs, targets[pos], forcings[index],
                    loss_cotangent, cotangent)
            gradient = accumulate(gradient, step_grad)
            cotangent = _tree_zeros_like(initial_state) if reset_state else next_cotangent
            jax.block_until_ready(gradient)
        return gradient

    @jax.jit
    def apply_pair(params, optimizer_state, gc_gradient, power_gradient):
        updates, next_optimizer_state, diagnostics = optimizer.update_pair(
            gc_gradient, power_gradient, optimizer_state, params)
        return optax.apply_updates(params, updates), next_optimizer_state, diagnostics

    @jax.jit
    def apply_single(params, optimizer_state, gc_gradient):
        updates, next_optimizer_state, diagnostics = optimizer.update_single(
            gc_gradient, optimizer_state, params)
        return optax.apply_updates(params, updates), next_optimizer_state, diagnostics

    def forward_only(params, state, keys, input_frames, static_inputs, truths, forcings):
        _, next_state, raw, components, fraction, _ = forward(
            params, state, keys, input_frames, static_inputs, truths, forcings)
        return raw, jax.tree_util.tree_map(jnp.asarray, next_state), components, fraction

    @jax.jit
    def propose_pair(params, optimizer_state, first, second):
        return optimizer.update_pair(first, second, optimizer_state, params)

    @jax.jit
    def scaled_candidate(params, updates, scale):
        scaled = jax.tree_util.tree_map(lambda x: x * scale, updates)
        return optax.apply_updates(params, scaled)

    @jax.jit
    def tree_is_finite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves])) if leaves else jnp.asarray(True)

    def train_excess_step(params, state, optimizer_state, keys, input_frames,
                          static_inputs, truths, forcings):
        from .mgda_acceptance import backtrack_losses, calibrate_loss_tolerances
        from .mgda_optimizer import tree_dot, tree_norm, validate_optimizer_diagnostics

        policy = config.mgda_backtracking
        decrease_policy = policy.get("decrease_policy", "warn") if policy is not None else "warn"
        train_step.last_acceptance_record = None
        if optimizer is None or policy is None or transforms.full_field_objectives is None:
            raise ValueError("MSE/AMSE-excess training requires pair RMS and objective diagnostics")
        args = (keys, input_frames, static_inputs, truths, forcings)
        tape, _, raw, _, _, gc_before = forward(params, state, *args)
        repeated = [np.asarray(raw, np.float64)]
        for _ in range(policy["repeatability_repeats"] - 1):
            repeated.append(np.asarray(forward_only(params, state, *args)[0], np.float64))
        try:
            tolerances = calibrate_loss_tolerances(
                repeated, absolute_floor=policy["absolute_floor"],
                relative_floor=policy["relative_floor"],
                max_relative_noise=policy["max_relative_noise"],
                repeatability_policy=policy.get("repeatability_policy", "warn"))
        except ValueError as error:
            raise JointDescentFailure({"reason": "repeatability_failed", "error": str(error),
                                       "repeated_losses": [x.tolist() for x in repeated]}) from error
        before = np.mean(repeated, axis=0)
        repeatability = {
            **tolerances.to_dict(), "repeated_losses": [x.tolist() for x in repeated],
            "noise_bounds": list(tolerances.noise_bounds),
            "noise_exceeded": list(tolerances.noise_exceeded),
            "warning": any(tolerances.noise_exceeded),
            "noise_thresholds": list(tolerances.noise_thresholds(before))}
        train_step.last_acceptance_record = {"repeatability": repeatability}
        if np.any(before < 0):
            raise JointDescentFailure({"reason": "negative_objective", "losses": before.tolist()})
        first = reverse(params, state, tape, 0, 1.)
        second = reverse(params, state, tape, 1, 1.)
        del tape
        updates, proposed_optimizer_state, diagnostics = propose_pair(
            params, optimizer_state, first, second)
        jax.block_until_ready(updates)
        if decrease_policy == "error" or not bool(diagnostics["finite"]):
            validate_optimizer_diagnostics(diagnostics)
        diagnostics = dict(jax.device_get(diagnostics))
        base_lr = float(diagnostics["learning_rate"])
        mode = config.mgda_direction
        candidate_cache = {}

        def evaluate_candidate(learning_rate):
            candidate = scaled_candidate(params, updates, jnp.asarray(learning_rate / base_lr, jnp.float32))
            displacement = jax.tree_util.tree_map(
                lambda p, q: jnp.asarray(p, jnp.float32) - jnp.asarray(q, jnp.float32), params, candidate)
            norm = float(tree_norm(displacement))
            dots = (float(tree_dot(first, displacement)), float(tree_dot(second, displacement)))
            gradient_norms = (float(tree_norm(first)), float(tree_norm(second)))
            if mode == "mgda":
                valid = all(dot >= -1e-6 * size * norm
                            for dot, size in zip(dots, gradient_norms, strict=True))
            elif mode == "mse_control":
                valid = dots[0] >= -1e-6 * gradient_norms[0] * norm
            else:
                summed = jax.tree_util.tree_map(lambda x, y: x + y, first, second)
                valid = sum(dots) >= -1e-6 * float(tree_norm(summed)) * norm
            if not valid and decrease_policy == "error":
                candidate_cache.clear()
                return (np.nan, np.nan)
            if not bool(tree_is_finite(candidate)):
                raise JointDescentFailure({"reason": "nonfinite_candidate_parameters"})
            _, next_state, values, components, fraction, gc_components = forward(candidate, state, *args)
            candidate_cache.clear()
            candidate_cache.update(params=candidate, state=next_state, values=values,
                                   components=components, fraction=fraction, gc_components=gc_components,
                                   update_norm=norm, dots=dots, descent_valid=bool(valid))
            return values

        acceptance = backtrack_losses(
            before, evaluate_candidate, mode=mode, tolerances=tolerances,
            initial_learning_rate=base_lr, max_halvings=policy["max_halvings"], shrink=policy["shrink"],
            decrease_policy=decrease_policy)
        train_step.last_acceptance_record = {
            "acceptance": acceptance.to_dict(), "repeatability": repeatability}
        if not acceptance.accepted:
            raise JointDescentFailure({"reason": acceptance.reason, "acceptance": acceptance.to_dict(),
                                       "repeatability": repeatability,
                                       "gradient_norms": [float(tree_norm(first)), float(tree_norm(second))],
                                       "mgda_alpha": float(diagnostics["mgda_alpha"])})
        after = np.asarray(acceptance.losses, np.float64)
        if not bool(tree_is_finite(candidate_cache["state"])):
            raise JointDescentFailure({"reason": "nonfinite_candidate_state",
                                       **train_step.last_acceptance_record})
        noise_floors = tolerances.noise_thresholds(before)
        decreases = before - after
        above_noise = (all(delta > floor for delta, floor in zip(decreases, noise_floors, strict=True))
                       if mode == "mgda" else decreases[0] > noise_floors[0] if mode == "mse_control"
                       else float(np.sum(decreases)) > sum(noise_floors))
        repeatability["accepted_decrease_above_noise"] = bool(above_noise)
        diagnostics.update(
            finite=True, descent_valid=bool(diagnostics["descent_valid"]) and candidate_cache["descent_valid"],
            stalled=candidate_cache["update_norm"] == 0,
            measured_decrease=acceptance.measured_decrease,
            decrease_warning=not acceptance.measured_decrease,
            learning_rate=float(acceptance.learning_rate),
            update_norm=candidate_cache["update_norm"],
            gc_displacement_dot=candidate_cache["dots"][0],
            power_displacement_dot=candidate_cache["dots"][1],
            gc_loss=float(raw[0]), amse_excess_loss=float(raw[1]),
            objective_1_before=float(before[0]), objective_2_before=float(before[1]),
            objective_1_after=float(after[0]), objective_2_after=float(after[1]),
            acceptance_threshold_1=acceptance.thresholds[0], acceptance_threshold_2=acceptance.thresholds[1],
            repeatability_span_1=tolerances.repeatability_span[0],
            repeatability_span_2=tolerances.repeatability_span[1],
            repeatability_warning=any(tolerances.noise_exceeded),
            noise_threshold_1=noise_floors[0], noise_threshold_2=noise_floors[1],
            accepted_decrease_above_noise=bool(above_noise),
            backtracking_accepted=True, backtracking_trials=len(acceptance.trials),
            backtracking_halvings=len(acceptance.trials) - 1,
            backtracking_baseline_repeats=len(repeated))
        if not np.all(np.isfinite([np.asarray(value).item() for value in diagnostics.values()])):
            raise JointDescentFailure({"reason": "nonfinite_accepted_diagnostics"})
        diagnostics["objective_pair"] = "mse_amse_excess"
        diagnostics["mgda_direction"] = mode
        diagnostics["decrease_policy"] = decrease_policy
        # Commit exactly the accepted candidate's regenerated end memory. The
        # old-parameter forward memory and all rejected candidates are discarded.
        return MultiObjectiveResult(
            candidate_cache["params"], jax.tree_util.tree_map(jnp.asarray, candidate_cache["state"]),
            proposed_optimizer_state, jnp.asarray(raw[0]), optax.global_norm(first),
            np.asarray(gc_before, np.float32), diagnostics)

    def train_step(params, state, optimizer_state, keys, input_frames, static_inputs,
                   truths, forcings, *, optimizer_step=None):
        del optimizer_step  # RMS count is checkpointed and owns the schedule.
        if amse_excess:
            return train_excess_step(params, state, optimizer_state, keys, input_frames,
                                     static_inputs, truths, forcings)
        calibration = (getattr(config, "amse_split_calibration", None) if amse_split
                       else config.power_calibration)
        if transforms.full_field_objectives is None or not calibration:
            raise ValueError("Training requires frozen shared AMSE calibration and both loss hooks" if amse_split
                             else "Training requires frozen power calibration and both loss hooks")
        tape, next_state, raw, components, fraction, gc_components = forward(
            params, state, keys, input_frames, static_inputs, truths, forcings)
        scales = ((float(calibration["common_scale"]),) * 2 if amse_split else
                  (float(calibration["gc_scale"]), float(calibration["power_scale"])))
        gc_gradient = reverse(params, state, tape, 0, scales[0])
        if config.optimizer_kind == "mgda_rms":
            power_gradient = reverse(params, state, tape, 1, scales[1])
            next_params, next_optimizer_state, diagnostics = apply_pair(
                params, optimizer_state, gc_gradient, power_gradient)
        else:
            next_params, next_optimizer_state, diagnostics = apply_single(
                params, optimizer_state, gc_gradient)
        jax.block_until_ready(next_params)
        from .mgda_optimizer import validate_optimizer_diagnostics

        validate_optimizer_diagnostics(diagnostics)
        diagnostics = dict(jax.device_get(diagnostics))
        gc_loss = float(np.dot(weights, gc_components))
        if amse_split:
            rename = {"gc_gradient_norm": "amplitude_gradient_norm", "power_gradient_norm": "pattern_gradient_norm",
                      "gc_direction_dot": "amplitude_direction_dot", "power_direction_dot": "pattern_direction_dot",
                      "gc_displacement_dot": "amplitude_displacement_dot", "power_displacement_dot": "pattern_displacement_dot",
                      "metric_gc_gc": "metric_amplitude_amplitude", "metric_gc_power": "metric_amplitude_pattern",
                      "metric_power_power": "metric_pattern_pattern"}
            diagnostics = {rename.get(key, key): value for key, value in diagnostics.items()}
            diagnostics.update(gc_loss=gc_loss, amplitude_loss=float(raw[0]), pattern_loss=float(raw[1]),
                               amplitude_loss_normalized=float(raw[0] / scales[0]),
                               pattern_loss_normalized=float(raw[1] / scales[1]),
                               amse_loss=float(raw.sum()), amse_loss_normalized=float(raw.sum() / scales[0]),
                               common_scale=scales[0])
            diagnostics["relative_combined_gradient_norm"] = float(diagnostics["combined_gradient_norm"]) / max(
                float(diagnostics["amplitude_gradient_norm"]) + float(diagnostics["pattern_gradient_norm"]), 1e-30)
        else:
            diagnostics.update(gc_loss=gc_loss, power_loss=float(raw[1]),
                               gc_loss_normalized=float(raw[0] / scales[0]),
                               power_loss_normalized=float(raw[1] / scales[1]),
                               power_floor_fraction=fraction)
        numeric = np.asarray([np.asarray(value).item() for value in diagnostics.values()], float)
        if not np.all(np.isfinite(numeric)):
            raise FloatingPointError("Nonfinite multiobjective loss/update diagnostics; update not accepted")
        # The optimizer checks the actual rounded parameter displacement, not
        # just the ideal preconditioned direction. A failed check aborts before
        # the runner accepts parameters or advances its cursor.
        if "descent_valid" in diagnostics and not bool(diagnostics["descent_valid"]):
            raise FloatingPointError("Multiobjective update violates the numerical descent tolerance")
        if amse_split:
            diagnostics["objective_pair"] = "amse_split"
        return MultiObjectiveResult(
            next_params, jax.tree_util.tree_map(jnp.asarray, next_state), next_optimizer_state,
            jnp.asarray(gc_loss), optax.global_norm(gc_gradient), gc_components, diagnostics)

    train_step.forward_only = forward_only
    train_step.last_acceptance_record = None
    def objective_gradients(params, state, args):
        tape = forward(params, state, *args)[0]
        return tuple(reverse(params, state, tape, index, 1.) for index in range(2))

    train_step.objective_gradients = objective_gradients
    return train_step
