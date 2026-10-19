"""The short feedback backend differentiates weather, not frozen parameters."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.config import V24IlyaDistributedConfig, V24IlyaTrainConfig
from src.models.mamba.v24_Ilya.training.endpoint_step import (
    V24IlyaTrainingTransforms, make_bptt_objective, make_train_step,
    make_validation_step, memory_contract,
)


def config(steps=4, loss_mode="all_steps", batch=1, reset=False):
    sparse = {"supervised_horizons": (1, steps), "supervised_weights": (1., 3.)} if loss_mode == "sparse_steps" else {}
    return V24IlyaTrainConfig(
        prepared_root=Path("prepared"), anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline"), output_root=Path("output"), run_name="feedback-test",
        architecture=V24IlyaArchitectureConfig(width=4), segment_steps=steps,
        bptt_steps=steps, ar_tail_k=steps - 1, feedback_mode="closed_loop",
        loss_kind="mse_fp32", loss_mode=loss_mode, precision="fp32",
        temporal_state_policy="reset_every_anchor" if reset else "carry",
        distributed=V24IlyaDistributedConfig(per_device_batch_size=batch), **sparse,
    )


def dataset(value, batch=1, name="x", hour=0):
    values = jnp.broadcast_to(jnp.asarray(value, jnp.float32), (batch, 1, 2))
    return xr.Dataset({name: (("batch", "time", "feature"), values)},
        coords={"batch": np.arange(batch), "time": [np.timedelta64(hour, "h")], "feature": [0, 1]})


def arguments(cfg):
    batch = cfg.distributed.per_device_batch_size
    frames = tuple(xr.merge([dataset([v, -.4 * v], batch), dataset([.1, -.2], batch, "forcing")]) for v in (1., 2.))
    truths = tuple(dataset([1.2 + i * .1, -.7 + i * .02], batch, hour=6) for i in cfg.supervised_step_indices)
    return (jax.random.split(jax.random.PRNGKey(1), cfg.bptt_steps), frames, xr.Dataset(), truths,
            tuple(dataset([.1, -.2], batch, "forcing", 6) for _ in range(cfg.bptt_steps)))


def fields(inputs):
    values = xarray_jax.unwrap_data(inputs["x"])
    return values[:, 0, :], values[:, 1, :]


class InputDependentBaseline:
    def apply(self, params, state, key, inputs, template, forcing):
        previous, current = fields(inputs)
        prediction = params["gain"] * (.3 * previous + .7 * current) + params["offset"]
        return jax.tree_util.tree_map(lambda value: jnp.broadcast_to(prediction[:, None, :], value.shape), template), state


class Residual:
    def apply(self, params, state, key, inputs, template, forcing):
        previous, current = fields(inputs)
        next_state = {"s": .6 * state["s"] + params["a"] * (.4 * previous + .6 * current)}
        prediction = params["b"] * next_state["s"] + params["c"] * current
        return jax.tree_util.tree_map(lambda value: jnp.broadcast_to(prediction[:, None, :], value.shape), template), next_state


class ForbiddenResidualLoss:
    def apply(self, *args, **kwargs):
        raise AssertionError("Differentiable feedback must score full physical fields")


def full_loss(prediction, truth, *, optimizer_step=None):
    pred, target = (xarray_jax.unwrap_data(value["x"]) for value in (prediction, truth))
    assert pred.dtype == target.dtype == jnp.float32
    scale = 1. if optimizer_step is None else 1. + .1 * optimizer_step
    return scale * jnp.mean((pred - target) ** 2, axis=(1, 2)), {}


def transforms():
    residual, forbidden = Residual(), ForbiddenResidualLoss()
    return V24IlyaTrainingTransforms(InputDependentBaseline(), residual, forbidden, forbidden,
                                      full_loss, residual)


def reference(params, initial_state, baseline, cfg, args, *, stop_baseline_inputs=False, scale=1.):
    _, frames, _, truths, _ = args
    weather = [xarray_jax.unwrap_data(frame["x"])[:, 0, :] for frame in frames]
    state = initial_state["s"]
    losses = []
    positions = {step: pos for pos, step in enumerate(cfg.supervised_step_indices)}
    reset = cfg.temporal_state_policy == "reset_every_anchor"
    for index in range(cfg.bptt_steps):
        previous, current = weather[index:index + 2]
        baseline_inputs = .3 * previous + .7 * current
        if stop_baseline_inputs:
            baseline_inputs = jax.lax.stop_gradient(baseline_inputs)
        base = baseline["gain"] * baseline_inputs + baseline["offset"]
        state = .6 * (jnp.zeros_like(state) if reset else state) + params["a"] * (.4 * previous + .6 * current)
        prediction = base + params["b"] * state + params["c"] * current
        if index in positions:
            truth = xarray_jax.unwrap_data(truths[positions[index]]["x"])[:, 0, :]
            losses.append(scale * jnp.mean((prediction - truth) ** 2))
        if reset:
            state = jnp.zeros_like(state)
        if index < cfg.bptt_steps - 1:
            weather.append(prediction)
    components = jnp.stack(losses)
    return jnp.dot(components, jnp.asarray(cfg.normalized_supervised_weights)), ({"s": state}, components)


def values(cfg):
    return ({"a": jnp.asarray(.12), "b": jnp.asarray(.23), "c": jnp.asarray(.07)},
            {"s": jnp.full((cfg.distributed.per_device_batch_size, 2), .2)},
            {"gain": jnp.asarray(.78), "offset": jnp.asarray(.04)})


def factories(cfg, baseline):
    return dict(transforms=transforms(), baseline_params=baseline, baseline_state={}, config=cfg,
                time_step=pd.Timedelta("6h"), input_steps=2)


def assert_close(actual, expected, *, rtol=3e-5, atol=2e-6):
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=rtol, atol=atol)


@pytest.mark.parametrize("steps,mode,batch,reset", [
    (2, "last_step", 1, False), (4, "all_steps", 4, False),
    (8, "sparse_steps", 1, False), (4, "all_steps", 1, True),
])
def test_streaming_feedback_matches_direct_ad_and_validation(steps, mode, batch, reset):
    cfg = config(steps, mode, batch, reset)
    params, state, baseline = values(cfg)
    baseline_copy = jax.tree_util.tree_map(lambda value: np.array(value, copy=True), baseline)
    args, kwargs = arguments(cfg), factories(cfg, baseline)
    (loss, (final_state, components)), gradient = jax.value_and_grad(
        lambda p: reference(p, state, baseline, cfg, args), has_aux=True)(params)
    objective = make_bptt_objective(**kwargs, return_loss_components=True)
    assert_close(objective(params, state, *args), (loss, final_state, components))
    objective_gradient = jax.grad(lambda p: objective(p, state, *args)[0])(params)
    assert_close(objective_gradient, gradient)
    optimizer = optax.sgd(.01)
    train_step = make_train_step(**kwargs, optimizer=optimizer)
    result = train_step(params, state, optimizer.init(params), *args)
    expected_params = jax.tree_util.tree_map(lambda p, g: p - .01 * g, params, gradient)
    assert_close((result[0], result[1], result[3], result[4], result[5]),
                 (expected_params, final_state, loss, optax.global_norm(gradient), components))
    assert_close(make_validation_step(**kwargs)(params, state, *args), (loss, final_state, components))
    assert_close(baseline, baseline_copy, rtol=0, atol=0)


def test_feedback_gradient_passes_finite_differences_and_includes_baseline_inputs():
    cfg = config(4, "last_step")
    params, state, baseline = values(cfg)
    args = arguments(cfg)
    objective = make_bptt_objective(**factories(cfg, baseline))
    gradient = jax.grad(lambda p: objective(p, state, *args)[0])(params)
    epsilon = .002
    for name in params:
        plus, minus = dict(params), dict(params)
        plus[name], minus[name] = params[name] + epsilon, params[name] - epsilon
        finite_difference = (objective(plus, state, *args)[0] - objective(minus, state, *args)[0]) / (2 * epsilon)
        np.testing.assert_allclose(gradient[name], finite_difference, rtol=.003, atol=2e-4)
    missing_baseline_gradient = jax.grad(lambda p: reference(
        p, state, baseline, cfg, args, stop_baseline_inputs=True)[0])(params)
    assert abs(float(gradient["c"] - missing_baseline_gradient["c"])) > .01
    # Freezing the baseline parameters must not accidentally unfreeze them via
    # a differentiable outer closure or the explicit weather-input pullback.
    frozen_gradient = jax.grad(lambda base: make_bptt_objective(
        **factories(cfg, base))(params, state, *args)[0])(baseline)
    assert_close(frozen_gradient, jax.tree_util.tree_map(jnp.zeros_like, baseline), rtol=0, atol=0)


def test_differentiable_feedback_changes_gradient_not_forward_weather():
    cfg = config(4, "last_step")
    params, state, baseline = values(cfg)
    args = arguments(cfg)
    differentiable = make_bptt_objective(**factories(cfg, baseline))
    stopped = make_bptt_objective(**factories(replace(cfg, feedback_mode="closed_loop_sg"), baseline))
    assert_close(differentiable(params, state, *args), stopped(params, state, *args))
    grad = jax.grad(lambda p: differentiable(p, state, *args)[0])(params)
    stop_grad = jax.grad(lambda p: stopped(p, state, *args)[0])(params)
    assert abs(float(grad["c"] - stop_grad["c"])) > .01


def test_feedback_memory_contract_discloses_graphcast_reverse_and_weather_tape():
    cfg = config()
    _, state, _ = values(cfg)
    _, frames, static, truths, _ = arguments(cfg)
    contract = memory_contract(input_frames=frames, static_inputs=static, truths=truths,
                               residual_state=state, config=cfg)
    assert contract["bptt_backend"] == "explicit_weather_reverse_vjp"
    assert contract["graphcast_backward_calls"] == 4
    assert contract["graphcast_input_gradients"] is True
    assert contract["graphcast_parameters_trainable"] is False
    assert contract["weather_cotangent_tape_bytes"] == contract["weather_tape_bytes"]
    assert contract["frozen_baseline_target_bytes"] == 0


def test_dynamic_ramp_step_reaches_every_feedback_loss_without_rebuilding():
    cfg = replace(config(2), loss_kind="amse", amse_lambda_ramp_steps=10)
    params, state, baseline = values(cfg)
    args = arguments(cfg)
    kwargs = factories(cfg, baseline)
    objective = make_bptt_objective(**kwargs)
    with pytest.raises(ValueError, match="optimizer_step"):
        objective(params, state, *args)
    zero = objective(params, state, *args, optimizer_step=jnp.asarray(0))[0]
    later = objective(params, state, *args, optimizer_step=jnp.asarray(5))[0]
    np.testing.assert_allclose(later, 1.5 * zero, rtol=2e-6)
    validation_loss = make_validation_step(**kwargs)(params, state, *args)[0]
    np.testing.assert_allclose(validation_loss, 2. * zero, rtol=2e-6)
    optimizer = optax.sgd(.01)
    step = make_train_step(**kwargs, optimizer=optimizer)
    result = step(params, state, optimizer.init(params), *args, optimizer_step=jnp.asarray(5))
    expected = jax.grad(lambda p: reference(p, state, baseline, cfg, args, scale=1.5)[0])(params)
    assert_close(result[0], jax.tree_util.tree_map(lambda p, g: p - .01 * g, params, expected))


def test_actual_graphcast_input_vjp_and_frozen_weights_on_tiny_grid():
    """Use the real GNN and physical normalizer, not a mocked GC derivative.

    This CPU-sized graph verifies correctness; production 1-degree memory still
    requires the separately budgeted GPU smoke before a long training run.
    """
    import haiku as hk
    from graphcast import graphcast as gc
    from src.models.mamba.v24_Ilya.model import make_baseline_predictor

    variable = "2m_temperature"
    forcing_variable = "toa_incident_solar_radiation"
    grid = dict(lat=np.arange(-90, 91, 30, dtype=np.float32), lon=np.arange(0, 360, 30, dtype=np.float32))
    base_values = 280 + np.cos(np.deg2rad(grid["lat"]))[:, None] + .5 * np.sin(np.deg2rad(grid["lon"]))[None, :]

    def weather(offset, hour=0):
        return xr.Dataset({variable: (("batch", "time", "lat", "lon"),
                           jnp.asarray(base_values[None, None] + offset, jnp.float32))},
                          coords={"batch": [0], "time": [np.timedelta64(hour, "h")], **grid})

    def forcing(hour=6):
        return jax.tree_util.tree_map(lambda value: jnp.ones_like(value) * .5, weather(0., hour)).rename(
            {variable: forcing_variable})

    def frame(offset, hour=0):
        return xr.merge([weather(offset, hour), forcing(hour)])

    model = gc.ModelConfig(resolution=30., mesh_size=0, latent_size=4, gnn_msg_steps=1,
                           hidden_layers=1, radius_query_fraction_edge_length=.8)
    task = gc.TaskConfig(input_variables=(variable, forcing_variable), target_variables=(variable,),
                         forcing_variables=(forcing_variable,), pressure_levels=(), input_duration="12h")
    stats = {name: xr.Dataset({variable: xr.DataArray(np.asarray(value, np.float32)),
                               forcing_variable: xr.DataArray(np.asarray(0. if name == "mean_by_level" else 1., np.float32))})
             for name, value in (("mean_by_level", 280.), ("stddev_by_level", 10.),
                                 ("diffs_stddev_by_level", 1.))}

    def baseline_call(inputs, target, forcing):
        return make_baseline_predictor(model, task, stats, use_bf16=False)(inputs, target, forcing)

    class ConstantResidual:
        def apply(self, params, state, key, inputs, template, forcing):
            return jax.tree_util.tree_map(lambda value: jnp.ones_like(value) * params["bias"], template), state

    def temperature_loss(prediction, truth):
        difference = xarray_jax.unwrap_data(prediction[variable]) - xarray_jax.unwrap_data(truth[variable])
        return jnp.mean(difference ** 2, axis=(1, 2, 3)), {}

    baseline_transform = hk.transform_with_state(baseline_call)
    initial_inputs = xr.concat([frame(0., -6), frame(.1)], dim="time")
    key = jax.random.PRNGKey(5)
    initialization_data = jax.tree_util.tree_map(jnp.asarray, (initial_inputs, weather(.2, 6), forcing()))
    baseline_params, baseline_state = baseline_transform.init(key, *initialization_data)
    frozen_copies = jax.tree_util.tree_map(lambda value: np.array(value, copy=True), baseline_params)
    residual = ConstantResidual()
    trans = V24IlyaTrainingTransforms(baseline_transform, residual, ForbiddenResidualLoss(),
                                       ForbiddenResidualLoss(), temperature_loss, residual)
    cfg = config(2, "last_step")
    kwargs = dict(transforms=trans, baseline_params=baseline_params, baseline_state=baseline_state,
                  config=cfg, time_step=pd.Timedelta("6h"), input_steps=2)
    args = (jax.random.split(key, 2), (frame(0.), frame(.1)), xr.Dataset(),
            (weather(.4, 6),), (forcing(), forcing()))
    params = {"bias": jnp.asarray(.05)}
    objective = make_bptt_objective(**kwargs)
    value, derivative = jax.value_and_grad(lambda bias: objective({"bias": bias}, {}, *args)[0])(params["bias"])
    epsilon = .02
    finite_difference = (objective({"bias": params["bias"] + epsilon}, {}, *args)[0]
                         - objective({"bias": params["bias"] - epsilon}, {}, *args)[0]) / (2 * epsilon)
    assert np.isfinite(value) and np.isfinite(derivative)
    assert abs(float(derivative)) > 1e-5
    np.testing.assert_allclose(derivative, finite_difference, rtol=.01, atol=2e-3)
    optimizer = optax.sgd(.01)
    update = make_train_step(**kwargs, optimizer=optimizer)(params, {}, optimizer.init(params), *args)
    np.testing.assert_allclose(update[0]["bias"], params["bias"] - .01 * derivative, rtol=3e-5, atol=2e-6)
    assert_close(baseline_params, frozen_copies, rtol=0, atol=0)
