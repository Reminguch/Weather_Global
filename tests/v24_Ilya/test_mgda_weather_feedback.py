"""Both MGDA objectives differentiate the same finite weather/state rollout."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.training.endpoint_step import (
    make_bptt_objective, make_validation_step,
)
from src.models.mamba.v24_Ilya.training.mgda_optimizer import RMSOptimizer
from src.models.mamba.v24_Ilya.training.mgda_step import make_multiobjective_train_step

from test_weather_feedback_step import (
    arguments, assert_close, config, fields, transforms, values,
)
from test_mse_amse_excess_step import assert_tree_identical, snapshot


class StochasticResidual:
    """Distinct previous/current dependencies expose dropped weather edges."""

    def apply(self, params, state, key, inputs, template, forcing):
        previous, current = fields(inputs)
        next_state = {"s": .6 * state["s"] + params["a"] * (.4 * previous + .6 * current)
                           + .01 * jax.random.normal(key, current.shape)}
        prediction = params["b"] * next_state["s"] + params["c"] * current
        return jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(prediction[:, None, :], value.shape), template), next_state


def pair_loss(prediction, truth):
    pred, target = (xarray_jax.unwrap_data(item["x"])[:, 0, :] for item in (prediction, truth))
    # Non-proportional smooth objectives exercise separate reverse seeds. The
    # actual spherical excess formula has independent loss-level tests.
    first = jnp.mean((pred - target) ** 2, axis=1)
    second = jnp.mean((.5 * pred ** 2 + .2 * pred - .7 * target) ** 2, axis=1)
    return jnp.stack((first, second)), {}


def setup(*, steps=8, prefix=4, reset=False, sparse=False, mode="mgda", learning_rate=1e-4):
    cfg = replace(config(steps, "sparse_steps" if sparse else "all_steps", batch=2,
                         reset=reset, prefix=prefix),
                  segment_steps=4 * steps, loss_kind="amse", objective_pair="mse_amse_excess",
                  optimizer_kind="mgda_rms", mgda_direction=mode,
                  learning_rate_schedule="warmup_cosine_constant", end_learning_rate=learning_rate,
                  learning_rate=learning_rate, weight_decay=0.,
                  mgda_backtracking={"decrease_policy": "warn", "repeatability_policy": "warn"})
    params, state, baseline = values(cfg)
    residual = StochasticResidual()
    trans = replace(transforms(), residual_predict=residual, residual_supervised_predict=residual,
                    full_field_objectives=pair_loss)
    optimizer = RMSOptimizer(lambda count: jnp.asarray(learning_rate), clip=0.,
                             spatial_multiplier=1., mamba_multiplier=1., direction_mode=mode)
    kwargs = dict(transforms=trans, baseline_params=baseline, baseline_state={}, config=cfg,
                  time_step=pd.Timedelta("6h"), input_steps=2)
    step = make_multiobjective_train_step(optimizer=optimizer, **kwargs)
    return cfg, params, state, baseline, arguments(cfg), optimizer, step, kwargs


def reference(params, state, baseline, cfg, args, *, stop_previous=False, stop_current=False,
              stop_baseline_inputs=False):
    """Independent ordinary AD unroll; no production kernels or VJP helpers."""
    keys, initial_frames, _, truths, _ = args
    frames = [xarray_jax.unwrap_data(frame["x"])[:, 0, :] for frame in initial_frames]
    positions = {index: pos for pos, index in enumerate(cfg.supervised_step_indices)}
    memory = state["s"]
    components = []
    for index in range(cfg.bptt_steps):
        previous, current = frames[index:index + 2]
        if stop_previous:
            previous = jax.lax.stop_gradient(previous)
        if stop_current:
            current = jax.lax.stop_gradient(current)
        base_inputs = .3 * previous + .7 * current
        if stop_baseline_inputs:
            base_inputs = jax.lax.stop_gradient(base_inputs)
        base = baseline["gain"] * base_inputs + baseline["offset"]
        if cfg.temporal_state_policy == "reset_every_anchor":
            memory = jnp.zeros_like(memory)
        memory = (.6 * memory + params["a"] * (.4 * previous + .6 * current)
                  + .01 * jax.random.normal(keys[index], current.shape))
        prediction = base + params["b"] * memory + params["c"] * current
        if index in positions:
            truth = xarray_jax.unwrap_data(truths[positions[index]]["x"])[:, 0, :]
            components.append(jnp.stack((jnp.mean((prediction - truth) ** 2),
                jnp.mean((.5 * prediction ** 2 + .2 * prediction - .7 * truth) ** 2))))
        if cfg.temporal_state_policy == "reset_every_anchor":
            memory = jnp.zeros_like(memory)
        if cfg.truth_prefix_steps - 1 <= index < cfg.bptt_steps - 1:
            frames.append(prediction)
    components = jnp.stack(components)
    losses = jnp.sum(components * jnp.asarray(cfg.normalized_supervised_weights)[:, None], axis=0)
    return losses, {"s": memory}, components


@pytest.mark.parametrize("steps,reset,sparse", [(8, False, False), (24, False, False), (8, True, True)])
def test_pair_weather_gradients_match_independent_ad_on_one_trajectory(steps, reset, sparse):
    cfg, params, state, baseline, args, _, step, kwargs = setup(steps=steps, reset=reset, sparse=sparse)
    frozen = snapshot((params, state, baseline, args[0]))
    expected = reference(params, state, baseline, cfg, args)
    gradients = tuple(jax.grad(lambda p: reference(p, state, baseline, cfg, args)[0][i])(params)
                      for i in range(2))
    actual = step.objective_gradients(params, state, args)
    assert_close(actual, gradients)
    raw, final_state, components, _ = step.forward_only(params, state, *args)
    assert_close((raw, final_state, components), expected)
    assert_tree_identical((params, state, baseline, args[0]), frozen)
    # Validation and the scalar diagnostic objective still report exact GC-MSE.
    assert_close(make_validation_step(**kwargs)(params, state, *args),
                 (expected[0][0], expected[1], expected[2][:, 0]))
    objective = make_bptt_objective(**kwargs)
    assert_close(jax.grad(lambda p: objective(p, state, *args)[0])(params), gradients[0])
    # A pair loss must not inadvertently make GraphCast's weights trainable.
    frozen_gradient = jax.grad(lambda base: make_bptt_objective(
        **{**kwargs, "baseline_params": base})(params, state, *args)[0])(baseline)
    assert_close(frozen_gradient, jax.tree_util.tree_map(jnp.zeros_like, baseline), rtol=0, atol=0)


def test_both_weather_input_edges_and_frozen_graphcast_input_jacobian_are_required():
    cfg, params, state, baseline, args, _, step, _ = setup(sparse=True)
    gradients = step.objective_gradients(params, state, args)
    for missing_path in ("stop_previous", "stop_current", "stop_baseline_inputs"):
        incomplete = tuple(jax.grad(lambda p: reference(
            p, state, baseline, cfg, args, **{missing_path: True})[0][i])(params) for i in range(2))
        for complete, missing in zip(gradients, incomplete, strict=True):
            assert abs(float(complete["c"] - missing["c"])) > .01


def test_chunk_boundary_carries_memory_value_but_stops_its_history_and_reseeds_weather():
    cfg, params, state, baseline, args, _, step, _ = setup()
    first_memory = step.forward_only(params, state, *args)[1]
    # A new chunk has fresh truth frames and the next RNG keys. Its old memory
    # value is carried, but its parameter history is outside this BPTT window.
    next_keys = jax.random.split(jax.random.PRNGKey(9), cfg.bptt_steps)
    next_args = (next_keys, *args[1:])
    detached_memory = jax.tree_util.tree_map(jax.lax.stop_gradient, first_memory)
    actual = step.objective_gradients(params, detached_memory, next_args)
    expected = tuple(jax.grad(lambda p: reference(p, detached_memory, baseline, cfg, next_args)[0][i])(params)
                     for i in range(2))
    assert_close(actual, expected)
    crossing_boundary = tuple(jax.grad(lambda p: reference(
        p, reference(p, state, baseline, cfg, args)[1], baseline, cfg, next_args)[0][i])(params)
                              for i in range(2))
    assert abs(float(actual[0]["a"] - crossing_boundary[0]["a"])) > .01
    zero = jax.tree_util.tree_map(jnp.zeros_like, state)
    reset_losses, reset_memory, _, _ = step.forward_only(params, zero, *next_args)
    expected_reset = reference(params, zero, baseline, cfg, next_args)
    assert_close((reset_losses, reset_memory), expected_reset[:2])
    assert not np.allclose(first_memory["s"], zero["s"])


@pytest.mark.parametrize("mode,learning_rate", [("mgda", 8.), ("mse_control", 8.), ("fixed_sum", 2.)])
def test_pair_commits_scheduled_candidate_replayed_state_and_optimizer_once(mode, learning_rate):
    _, params, state, baseline, args, optimizer, step, _ = setup(mode=mode, learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    frozen = snapshot((params, state, opt_state, baseline, args[0]))
    first, second = step.objective_gradients(params, state, args)
    updates, expected_optimizer, _ = optimizer.update_pair(first, second, opt_state, params)
    expected_params = jax.tree_util.tree_map(lambda p, update: p + update, params, updates)
    expected_losses, expected_state, _, _ = step.forward_only(expected_params, state, *args)
    original_losses = step.forward_only(params, state, *args)[0]
    assert np.all(expected_losses > original_losses)
    # This deliberately oversized finite update must complete even if its
    # losses increase. It must never search for a smaller passing candidate.
    with pytest.warns(RuntimeWarning, match="committing finite scheduled update"):
        result = step(params, state, opt_state, *args)
    assert_close((result.params, result.state, result.optimizer_state),
                 (expected_params, expected_state, expected_optimizer))
    assert int(result.optimizer_state.count) == 1
    assert result.diagnostics["decrease_warning"]
    assert result.diagnostics["backtracking_trials"] == 1
    assert result.diagnostics["learning_rate"] == pytest.approx(learning_rate)
    assert_close(jnp.asarray([result.diagnostics["objective_1_after"],
                             result.diagnostics["objective_2_after"]]), expected_losses)
    assert_tree_identical((params, state, opt_state, baseline, args[0]), frozen)


def test_pair_nonfinite_optimizer_still_blocks_without_mutating_inputs():
    _, params, state, baseline, args, optimizer, step, _ = setup()
    opt_state = optimizer.init(params)
    invalid = opt_state._replace(second_moment=jax.tree_util.tree_map(
        lambda value: jnp.full_like(value, jnp.nan), opt_state.second_moment))
    frozen = snapshot((params, state, baseline, args[0]))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        step(params, state, invalid, *args)
    assert_tree_identical((params, state, baseline, args[0]), frozen)


def test_pair_retains_large_repeatability_warning_and_finishes_update(monkeypatch):
    from src.models.mamba.v24_Ilya.training import mgda_acceptance

    calibrate = mgda_acceptance.calibrate_loss_tolerances

    def noisy_calibration(repeated, **kwargs):
        repeated[1] = repeated[0] * 1.1
        repeated[2] = repeated[0] * .9
        return calibrate(repeated, **kwargs)

    monkeypatch.setattr(mgda_acceptance, "calibrate_loss_tolerances", noisy_calibration)
    _, params, state, _, args, optimizer, step, _ = setup()
    with pytest.warns(RuntimeWarning, match="advisory only"):
        result = step(params, state, optimizer.init(params), *args)
    assert int(result.optimizer_state.count) == 1
    assert result.diagnostics["repeatability_warning"]
    record = step.last_acceptance_record["repeatability"]
    assert record["warning"] and record["noise_exceeded"] == [True, True]
    assert len(record["repeated_losses"]) == 3
    assert not record["accepted_decrease_above_noise"]
