from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v22_final.training.config import load_training_config
from src.models.mamba.v22_final.training.step import (
    feedback_field,
    make_train_step,
    make_validation_step,
    residual_target,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = ROOT / "configs/experiments/v22_final/res2_gc500k_k12_open.json"


def test_residual_target_stops_baseline_gradient() -> None:
    target = jnp.asarray([5.0])

    def objective(baseline):
        return jnp.sum(residual_target(target, baseline))

    np.testing.assert_array_equal(jax.grad(objective)(jnp.asarray([2.0])), 0.0)
    np.testing.assert_array_equal(residual_target(target, jnp.asarray([2.0])), 3.0)


def test_closed_loop_feedback_stops_residual_gradient() -> None:
    baseline = jnp.asarray([2.0])

    def objective(residual):
        return jnp.sum(feedback_field("closed_loop_sg", baseline, residual))

    np.testing.assert_array_equal(jax.grad(objective)(jnp.asarray([3.0])), 0.0)
    np.testing.assert_array_equal(objective(jnp.asarray([3.0])), 5.0)
    np.testing.assert_array_equal(
        feedback_field("baseline", baseline, jnp.asarray([99.0])), baseline
    )


def test_feedback_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported feedback"):
        feedback_field("unknown", jnp.asarray([1.0]), jnp.asarray([1.0]))


def _jax_data_array(value: float, *, name: str) -> xr.DataArray:
    return xarray_jax.DataArray(
        jnp.asarray([[[[value]]]], dtype=jnp.float32),
        dims=("batch", "time", "lat", "lon"),
        coords={
            "batch": [0],
            "time": [np.timedelta64(0, "h")],
            "lat": [0.0],
            "lon": [0.0],
        },
        name=name,
    )


def _toy_inputs(state: float, forcing: float) -> xr.Dataset:
    return xr.Dataset(
        {
            "x": _jax_data_array(state, name="x"),
            "forcing": _jax_data_array(forcing, name="forcing"),
        }
    )


def _toy_target(value: float = 0.0) -> xr.Dataset:
    return xr.Dataset({"x": _jax_data_array(value, name="x")})


def _wrap_like(values, template: xr.DataArray) -> xr.DataArray:
    return xarray_jax.DataArray(
        values,
        dims=template.dims,
        coords={name: template.coords[name] for name in template.dims},
        name=template.name,
    )


class _ToyBaselineTransform:
    def apply(self, params, state, key, inputs, truth, forcings):
        del key, forcings
        latest = xarray_jax.unwrap_data(inputs["x"].isel(time=-1, drop=True))
        template = truth["x"]
        prediction = jnp.broadcast_to(
            latest[:, None, :, :] + params["increment"],
            xarray_jax.unwrap_data(template).shape,
        )
        return xr.Dataset({"x": _wrap_like(prediction, template)}), state


class _ToyResidualTransform:
    def apply(self, params, state, key, inputs, truth, forcings):
        del key, inputs, forcings
        template = truth["x"]
        prediction = jnp.zeros_like(xarray_jax.unwrap_data(template)) + params["value"]
        return xr.Dataset({"x": _wrap_like(prediction, template)}), state + 1


class _ToyLossTransform:
    def apply(self, params, state, key, inputs, target, forcings):
        del params, key, target, forcings
        latest_state = xarray_jax.unwrap_data(inputs["x"].isel(time=-1, drop=True))
        latest_forcing = xarray_jax.unwrap_data(
            inputs["forcing"].isel(time=-1, drop=True)
        )
        loss = latest_state + 1000.0 * latest_forcing
        loss_array = xarray_jax.DataArray(
            loss,
            dims=("batch", "lat", "lon"),
            coords={"batch": [0], "lat": [0.0], "lon": [0.0]},
            name="loss",
        )
        # The maintained step must ignore this returned state and carry the
        # residual prediction application's +1 transition instead.
        return (loss_array, {}), state + 100


@pytest.mark.parametrize(
    ("feedback_mode", "expected_loss"),
    [
        ("baseline", np.mean([10_000.0, 11_020.0, 101_021.0, 102_022.0])),
        ("closed_loop_sg", np.mean([10_000.0, 11_020.0, 101_023.0, 102_026.0])),
    ],
)
def test_bptt_uses_truth_prefix_correct_forcing_and_prediction_state(
    feedback_mode: str,
    expected_loss: float,
) -> None:
    config = dataclasses.replace(
        load_training_config(REFERENCE_CONFIG),
        segment_steps=4,
        bptt_steps=4,
        ar_tail_k=2,
        feedback_mode=feedback_mode,
        learning_rate=0.0,
        warmup_steps=0,
        grad_clip=0.0,
    )
    transforms = SimpleNamespace(
        baseline_predict=_ToyBaselineTransform(),
        residual_predict=_ToyResidualTransform(),
        residual_loss=_ToyLossTransform(),
    )
    residual_params = {"value": jnp.asarray(2.0, dtype=jnp.float32)}
    optimizer = optax.sgd(0.0)
    step = make_train_step(
        transforms=transforms,
        optimizer=optimizer,
        baseline_params={"increment": jnp.asarray(1.0, dtype=jnp.float32)},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=1,
    )
    result = step(
        residual_params,
        jnp.asarray(0),
        optimizer.init(residual_params),
        jax.random.split(jax.random.PRNGKey(0), 4),
        (_toy_inputs(0.0, 10.0), _toy_inputs(20.0, 11.0)),
        tuple(_toy_target() for _ in range(4)),
        tuple(
            xr.Dataset({"forcing": _jax_data_array(100.0 + index, name="forcing")})
            for index in range(4)
        ),
    )
    np.testing.assert_allclose(result[3], expected_loss)
    assert int(result[1]) == 4
    # The closed-loop residual is stop-gradient, so this diagnostic loss has
    # no differentiable dependence on residual parameters in either mode.
    np.testing.assert_array_equal(result[4], 0.0)

    validation = make_validation_step(
        transforms=transforms,
        baseline_params={"increment": jnp.asarray(1.0, dtype=jnp.float32)},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=1,
    )
    validation_loss, validation_state = validation(
        residual_params,
        jnp.asarray(0),
        jax.random.split(jax.random.PRNGKey(0), 4),
        (_toy_inputs(0.0, 10.0), _toy_inputs(20.0, 11.0)),
        tuple(_toy_target() for _ in range(4)),
        tuple(
            xr.Dataset({"forcing": _jax_data_array(100.0 + index, name="forcing")})
            for index in range(4)
        ),
    )
    np.testing.assert_allclose(validation_loss, result[3])
    assert int(validation_state) == 4
    np.testing.assert_array_equal(residual_params["value"], 2.0)


class _StateSensitiveLossTransform:
    def apply(self, params, state, key, inputs, target, forcings):
        del params, key, inputs, target, forcings
        loss_array = xarray_jax.DataArray(
            jnp.reshape(jnp.asarray(state, dtype=jnp.float32), (1, 1, 1)),
            dims=("batch", "lat", "lon"),
            coords={"batch": [0], "lat": [0.0], "lon": [0.0]},
            name="loss",
        )
        return (loss_array, {}), state + 100


@pytest.mark.parametrize(
    ("state_policy", "expected_loss", "expected_state"),
    [("carry", 1.5, 4), ("reset_every_anchor", 0.0, 0)],
)
def test_temporal_state_policy_isolates_anchor_state_carry(
    state_policy: str,
    expected_loss: float,
    expected_state: int,
) -> None:
    config = dataclasses.replace(
        load_training_config(REFERENCE_CONFIG),
        segment_steps=4,
        bptt_steps=4,
        ar_tail_k=0,
        temporal_state_policy=state_policy,
    )
    transforms = SimpleNamespace(
        baseline_predict=_ToyBaselineTransform(),
        residual_predict=_ToyResidualTransform(),
        residual_loss=_StateSensitiveLossTransform(),
    )
    residual_params = {"value": jnp.asarray(2.0, dtype=jnp.float32)}
    validation = make_validation_step(
        transforms=transforms,
        baseline_params={"increment": jnp.asarray(1.0, dtype=jnp.float32)},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=1,
    )
    loss, next_state = validation(
        residual_params,
        jnp.asarray(0),
        jax.random.split(jax.random.PRNGKey(0), 4),
        tuple(_toy_inputs(float(i), float(i)) for i in range(4)),
        tuple(_toy_target() for _ in range(4)),
        tuple(
            xr.Dataset({"forcing": _jax_data_array(float(i), name="forcing")})
            for i in range(4)
        ),
    )
    np.testing.assert_allclose(loss, expected_loss)
    assert int(next_state) == expected_state
