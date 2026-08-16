from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import optax
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v23_Ilya.config import V23IlyaArchitectureConfig
from src.models.mamba.v23_Ilya.training.config import V23IlyaTrainConfig
from src.models.mamba.v23_Ilya.training.endpoint_step import (
    V23IlyaTrainingTransforms,
    make_bptt_objective,
    make_train_step,
    memory_contract,
)


class FrozenBaseline:
    def __init__(self) -> None:
        self.calls = 0

    def apply(self, params, state, key, inputs, template, forcing):
        del params, key, inputs, forcing
        self.calls += 1
        return jax.tree_util.tree_map(jnp.zeros_like, template), state


class StatefulResidual:
    def apply(self, params, state, key, inputs, template, forcing):
        del key, forcing
        value = jnp.mean(xarray_jax.unwrap_data(inputs["x"]))
        next_state = {"s": 0.7 * state["s"] + params["a"] * value}
        prediction = jax.tree_util.tree_map(
            lambda target: jnp.ones_like(target) * params["b"] * next_state["s"],
            template,
        )
        return prediction, next_state


class StatefulResidualLoss:
    def apply(self, params, state, key, inputs, target, forcing):
        prediction, next_state = StatefulResidual().apply(
            params,
            state,
            key,
            inputs,
            target,
            forcing,
        )
        error = (
            xarray_jax.unwrap_data(prediction["x"])
            - xarray_jax.unwrap_data(target["x"])
        )
        value = jnp.mean(error**2)
        loss = xr.DataArray(
            value[None],
            dims=("batch",),
            coords={"batch": [0]},
        )
        return ((loss, {}), prediction), next_state


def _config(loss_mode: str, tape_precision: str = "fp32") -> V23IlyaTrainConfig:
    sparse_objective = (
        {
            "supervised_horizons": (1, 3),
            "supervised_weights": (1, 3),
        }
        if loss_mode == "sparse_steps"
        else {}
    )
    return V23IlyaTrainConfig(
        prepared_root=Path("prepared"),
        anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline"),
        output_root=Path("output"),
        run_name=f"test-{loss_mode}",
        architecture=V23IlyaArchitectureConfig(width=4),
        segment_steps=4,
        bptt_steps=4,
        ar_tail_k=2,
        feedback_mode="closed_loop_sg",
        loss_mode=loss_mode,
        weather_tape_precision=tape_precision,
        **sparse_objective,
    )


def _dataset(value: float, name: str = "x", hour: int = 0) -> xr.Dataset:
    return xr.Dataset(
        {
            name: (
                ("batch", "time"),
                jnp.asarray([[value]], dtype=jnp.float32),
            )
        },
        coords={"batch": [0], "time": [np.timedelta64(hour, "h")]},
    )


def _input_frame(value: float) -> xr.Dataset:
    return xr.merge([_dataset(value), _dataset(0.0, "forcing")])


def _arguments(loss_mode: str):
    if loss_mode == "last_step":
        truth_values = (5.0,)
    elif loss_mode == "all_steps":
        truth_values = (3.0, 4.0, 5.0, 6.0)
    else:
        truth_values = (4.0, 6.0)
    truths = tuple(_dataset(value, hour=6) for value in truth_values)
    return (
        jnp.stack([jax.random.PRNGKey(index) for index in range(4)]),
        (_input_frame(1.0), _input_frame(2.0), _input_frame(3.0)),
        xr.Dataset(),
        truths,
        tuple(_dataset(0.0, "forcing", 6) for _ in range(4)),
    )


def _naive_reference(params, initial_state, loss_mode: str):
    frames = [jnp.asarray(value, jnp.float32) for value in (1.0, 2.0, 3.0)]
    if loss_mode == "last_step":
        supervised_positions = {3: 0}
        targets = (5.0,)
        weights = (1.0,)
    elif loss_mode == "all_steps":
        supervised_positions = {index: index for index in range(4)}
        targets = (3.0, 4.0, 5.0, 6.0)
        weights = (0.25,) * 4
    else:
        supervised_positions = {1: 0, 3: 1}
        targets = (4.0, 6.0)
        weights = (0.25, 0.75)
    state = initial_state
    losses = []
    for index in range(4):
        value = 0.5 * (frames[index] + frames[index + 1])
        state = 0.7 * state + params["a"] * value
        prediction = params["b"] * state
        if index in supervised_positions:
            target = targets[supervised_positions[index]]
            losses.append((prediction - target) ** 2)
        if index >= 1 and index < 3:
            frames.append(jax.lax.stop_gradient(prediction))
    loss = jnp.sum(jnp.stack(losses) * jnp.asarray(weights))
    return loss, state


@pytest.mark.parametrize(
    ("loss_mode", "expected_baseline_calls"),
    [("last_step", 3), ("all_steps", 4), ("sparse_steps", 3)],
)
def test_explicit_reverse_matches_naive_unroll(
    loss_mode: str,
    expected_baseline_calls: int,
) -> None:
    baseline = FrozenBaseline()
    residual = StatefulResidual()
    residual_loss = StatefulResidualLoss()
    transforms = V23IlyaTrainingTransforms(
        baseline_predict=baseline,
        residual_predict=residual,
        residual_loss=residual_loss,
        residual_loss_and_predictions=residual_loss,
    )
    config = _config(loss_mode)
    objective = make_bptt_objective(
        transforms=transforms,
        baseline_params={},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
    )
    args = _arguments(loss_mode)
    params = {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)}
    state = {"s": jnp.asarray(0.0)}

    (actual_loss, actual_state), actual_gradient = jax.value_and_grad(
        lambda value: objective(value, state, *args),
        has_aux=True,
    )(params)
    (expected_loss, expected_state), expected_gradient = jax.value_and_grad(
        lambda value: _naive_reference(value, state["s"], loss_mode),
        has_aux=True,
    )(params)

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(actual_state["s"], expected_state, rtol=1e-6)
    for name in params:
        np.testing.assert_allclose(
            actual_gradient[name],
            expected_gradient[name],
            rtol=1e-6,
            atol=1e-6,
        )
    assert float(actual_gradient["a"]) != 0.0
    # GraphCast appears only in the custom forward rule.
    assert baseline.calls == expected_baseline_calls



def test_host_tape_train_step_matches_naive_parameter_update() -> None:
    baseline = FrozenBaseline()
    residual = StatefulResidual()
    residual_loss = StatefulResidualLoss()
    config = _config("sparse_steps")
    transforms = V23IlyaTrainingTransforms(
        baseline,
        residual,
        residual_loss,
        residual_loss,
    )
    optimizer = optax.sgd(0.01)
    train_step = make_train_step(
        transforms=transforms,
        optimizer=optimizer,
        baseline_params={},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
    )
    params = {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)}
    state = {"s": jnp.asarray(0.0)}
    keys, frames, static, truths, forcings = _arguments("sparse_steps")

    (
        next_params,
        next_state,
        _optimizer_state,
        loss,
        gradient_norm,
        loss_components,
    ) = train_step(
        params,
        state,
        optimizer.init(params),
        keys,
        frames,
        static,
        truths,
        forcings,
    )
    (expected_loss, expected_state), expected_gradient = jax.value_and_grad(
        lambda value: _naive_reference(value, state["s"], "sparse_steps"),
        has_aux=True,
    )(params)
    expected_params = jax.tree_util.tree_map(
        lambda value, gradient: value - 0.01 * gradient,
        params,
        expected_gradient,
    )

    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(
        loss,
        np.sum(np.asarray(loss_components) * np.asarray([0.25, 0.75])),
        rtol=1e-6,
    )
    assert np.asarray(loss_components).shape == (2,)
    np.testing.assert_allclose(next_state["s"], expected_state, rtol=1e-6)
    for name in params:
        np.testing.assert_allclose(
            next_params[name],
            expected_params[name],
            rtol=1e-6,
            atol=1e-6,
        )
    assert float(gradient_norm) > 0.0

def test_explicit_reverse_jits() -> None:
    baseline = FrozenBaseline()
    residual = StatefulResidual()
    residual_loss = StatefulResidualLoss()
    config = _config("last_step")
    objective = make_bptt_objective(
        transforms=V23IlyaTrainingTransforms(
            baseline,
            residual,
            residual_loss,
            residual_loss,
        ),
        baseline_params={},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
    )
    keys, frames, static, truths, forcings = _arguments("last_step")
    params = {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)}
    state = {"s": jnp.asarray(0.0)}

    @jax.jit
    def compiled_step(value, keys, frames, static, truths, forcings):
        return jax.value_and_grad(
            lambda candidate: objective(
                candidate, state, keys, frames, static, truths, forcings
            ),
            has_aux=True,
        )(value)
    compiled = compiled_step(params, keys, frames, static, truths, forcings)


    assert jnp.isfinite(compiled[0][0])
    assert all(jnp.isfinite(value) for value in compiled[1].values())


def test_memory_contract_is_unique_frame_bf16_tape() -> None:
    config = _config("last_step", tape_precision="bf16")
    arguments = _arguments("last_step")
    report = memory_contract(
        input_frames=arguments[1],
        static_inputs=arguments[2],
        truths=arguments[3],
        residual_state={"s": jnp.asarray(0.0, jnp.float32)},
        config=config,
    )
    assert report["unique_weather_tape_frames"] == 5
    assert report["state_tape_boundaries"] == 4
    assert report["truth_targets_loaded"] == 1
    assert report["weather_tape_precision"] == "bf16"
    assert report["graphcast_backward_calls"] == 0


class Bf16StatefulResidual(StatefulResidual):
    def apply(self, params, state, key, inputs, template, forcing):
        prediction, next_state = super().apply(
            params,
            state,
            key,
            inputs,
            template,
            forcing,
        )
        return prediction, jax.tree_util.tree_map(
            lambda value: value.astype(jnp.bfloat16),
            next_state,
        )


class Bf16StatefulResidualLoss:
    def apply(self, params, state, key, inputs, target, forcing):
        prediction, next_state = Bf16StatefulResidual().apply(
            params,
            state,
            key,
            inputs,
            target,
            forcing,
        )
        error = (
            xarray_jax.unwrap_data(prediction["x"])
            - xarray_jax.unwrap_data(target["x"])
        )
        loss = xr.DataArray(
            jnp.mean(error**2)[None],
            dims=("batch",),
            coords={"batch": [0]},
        )
        return ((loss, {}), prediction), next_state


def test_actual_recurrent_state_boundaries_are_fp32() -> None:
    baseline = FrozenBaseline()
    residual = Bf16StatefulResidual()
    residual_loss = Bf16StatefulResidualLoss()
    config = _config("last_step", tape_precision="bf16")
    objective = make_bptt_objective(
        transforms=V23IlyaTrainingTransforms(
            baseline,
            residual,
            residual_loss,
            residual_loss,
        ),
        baseline_params={},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
    )
    keys, frames, static, truths, forcings = _arguments("last_step")
    _, final_state = objective(
        {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)},
        {"s": jnp.asarray(0.0, dtype=jnp.bfloat16)},
        keys,
        frames,
        static,
        truths,
        forcings,
    )

    assert all(
        leaf.dtype == jnp.float32
        for leaf in jax.tree_util.tree_leaves(final_state)
    )
