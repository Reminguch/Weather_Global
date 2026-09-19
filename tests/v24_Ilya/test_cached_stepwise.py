from __future__ import annotations

import dataclasses
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "third_party" / "graphcast"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr
from graphcast import graphcast as gc, xarray_jax

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.model import attach_temporal
from src.models.mamba.v24_Ilya.training import cached_stepwise
from src.models.mamba.v24_Ilya.training.cached_stepwise import (
    CachedResidualTransforms,
    CachedStepwiseTrainer,
    make_cached_stepwise_train_step,
)
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig
from src.models.mamba.v24_Ilya.training.endpoint_step import (
    V24IlyaTrainingTransforms,
    input_window_from_frames,
    make_train_step,
)


def _dataset(value, name="x", hour=0):
    return xr.Dataset(
        {name: (("batch", "time"), jnp.asarray([[value]], dtype=jnp.float32))},
        coords={"batch": [0], "time": [np.timedelta64(hour, "h")]},
    )


def _frame(value):
    return xr.merge([_dataset(value), _dataset(0.0, "forcing")])


def _config(loss_mode="all_steps", reset=False, precision="bf16"):
    sparse = {"supervised_horizons": (1, 3), "supervised_weights": (1, 3)} if loss_mode == "sparse_steps" else {}
    return V24IlyaTrainConfig(
        prepared_root=Path("prepared"), anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline"), output_root=Path("output"),
        run_name="test-cached-stepwise", architecture=V24IlyaArchitectureConfig(width=4),
        segment_steps=4, bptt_steps=4, ar_tail_k=2, feedback_mode="baseline",
        loss_mode=loss_mode, precision=precision,
        temporal_state_policy="reset_every_anchor" if reset else "carry", **sparse,
    )


class _Baseline:
    def __init__(self):
        self.calls = 0

    def apply(self, params, state, key, inputs, template, forcing):
        del params, key, inputs, forcing
        self.calls += 1
        return jax.tree_util.tree_map(jnp.zeros_like, template), state


class _Residual:
    def __init__(self, dtype):
        self.dtype = dtype

    def apply(self, params, state, key, inputs, template, forcing):
        del key, forcing
        dtype = self.dtype
        value = jnp.mean(xarray_jax.unwrap_data(inputs["x"])).astype(dtype).astype(jnp.float32)
        a = params["a"].astype(dtype).astype(jnp.float32)
        b = params["b"].astype(dtype).astype(jnp.float32)
        previous = state["s"].astype(dtype).astype(jnp.float32)
        raw = 0.7 * previous + a * value
        output = (b * raw).astype(dtype).astype(jnp.float32)
        next_state = {"s": raw.astype(dtype).astype(jnp.float32)}
        prediction = jax.tree_util.tree_map(lambda target: jnp.ones_like(target) * output, template)
        return prediction, next_state


class _ResidualLoss:
    def __init__(self, residual):
        self.residual = residual

    def apply(self, params, state, key, inputs, target, forcing):
        prediction, next_state = self.residual.apply(params, state, key, inputs, target, forcing)
        dtype = self.residual.dtype
        error = xarray_jax.unwrap_data(prediction["x"]).astype(dtype) - xarray_jax.unwrap_data(target["x"]).astype(dtype)
        loss = jnp.mean(error**2).astype(jnp.float32)
        loss = xr.DataArray(loss[None], dims=("batch",), coords={"batch": [0]})
        return ((loss, {}), prediction), next_state


def _setup(loss_mode="all_steps", reset=False, precision="bf16"):
    config = _config(loss_mode, reset, precision)
    dtype = jnp.bfloat16 if precision == "bf16" else jnp.float32
    baseline = _Baseline()
    residual = _Residual(dtype)
    loss = _ResidualLoss(residual)
    optimizer = optax.adam(0.01)
    cached = CachedStepwiseTrainer(
        config=config, optimizer=optimizer,
        transforms=CachedResidualTransforms(residual, loss),
    )
    online = make_train_step(
        transforms=V24IlyaTrainingTransforms(baseline, residual, loss, loss),
        optimizer=optimizer, baseline_params={}, baseline_state={}, config=config,
        time_step=pd.Timedelta("6h"), input_steps=2,
    )
    keys = jnp.stack([jax.random.PRNGKey(index) for index in range(4)])
    frames = tuple(_frame(value) for value in (1.0, 2.0, 3.0, 0.0, 0.0))
    static = xr.Dataset()
    inputs = tuple(input_window_from_frames(
        frames[index], frames[index + 1], static, step_index=index,
        truth_prefix_steps=2, time_step=pd.Timedelta("6h"),
    ) for index in range(4))
    targets = tuple(_dataset(value, hour=6) for value in (3.0, 4.0, 5.0, 6.0))
    forcings = tuple(_dataset(0.0, "forcing", hour=6) for _ in range(4))
    params = {"a": jnp.asarray(0.113, jnp.float32), "b": jnp.asarray(0.241, jnp.float32)}
    state = {"s": jnp.asarray(0.087, jnp.float32)}
    return SimpleNamespace(
        config=config, cached=cached, online=online, baseline=baseline, optimizer=optimizer,
        params=params, state=state, keys=keys, frames=frames[:3], static=static,
        inputs=inputs, targets=targets, forcings=forcings,
    )


def _assert_equal_tree(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("loss_mode", ["all_steps", "last_step", "sparse_steps"])
@pytest.mark.parametrize("reset", [False, True])
def test_cached_update_matches_original_online_host_bptt(precision, loss_mode, reset):
    fixture = _setup(loss_mode, reset, precision)
    optimizer_state = fixture.optimizer.init(fixture.params)
    selected = tuple(fixture.targets[index] for index in fixture.config.supervised_step_indices)
    expected = fixture.online(
        fixture.params, fixture.state, optimizer_state, fixture.keys, fixture.frames,
        fixture.static, selected, fixture.forcings,
    )
    online_calls = fixture.baseline.calls
    assert online_calls > 0
    actual = fixture.cached(
        fixture.params, fixture.state, optimizer_state, fixture.keys,
        fixture.inputs, fixture.targets, fixture.forcings,
    )
    _assert_equal_tree(actual, expected)
    assert fixture.baseline.calls == online_calls
    if reset and loss_mode == "last_step":
        assert float(actual[4]) == 0.0  # Reset memory plus zero last input removes temporal credit.
    else:
        assert float(actual[4]) > 0.0


def test_explicit_gradient_retains_earlier_temporal_contributions():
    fixture = _setup("last_step", precision="fp32")
    result = fixture.cached.loss_and_grad(
        fixture.params, fixture.state, fixture.keys, fixture.inputs,
        fixture.targets, fixture.forcings,
    )
    loss, next_state, components, predictions, gradients = result

    def reference(parameters):
        memory = fixture.state["s"]
        for value in (1.5, 2.5, 1.5, 0.0):
            memory = 0.7 * memory + parameters["a"] * value
        return (parameters["b"] * memory - 6.0) ** 2

    expected_loss, expected_gradients = jax.value_and_grad(reference)(fixture.params)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    for name in gradients:
        np.testing.assert_allclose(gradients[name], expected_gradients[name], rtol=1e-6)
    assert float(gradients["a"]) != 0.0  # Final step input is zero: this is temporal credit.
    assert next_state["s"].dtype == jnp.float32
    assert components.shape == (1,)
    assert len(predictions) == fixture.config.bptt_steps


def test_evaluation_streams_predictions_without_mutating_state_or_inputs():
    fixture = _setup()
    before_state = jax.tree.map(np.array, fixture.state)
    before_targets = tuple(jax.tree.map(np.array, target) for target in fixture.targets)
    seen = []
    result = fixture.cached.evaluate(
        fixture.params, fixture.state, fixture.keys, fixture.inputs, fixture.targets,
        fixture.forcings, retain_predictions=False,
        prediction_callback=lambda index, prediction: seen.append((index, float(xarray_jax.unwrap_data(prediction["x"])[0, 0]))),
    )
    assert result[3] == ()
    assert [index for index, _ in seen] == list(range(4))
    _assert_equal_tree(fixture.state, before_state)
    _assert_equal_tree(fixture.targets, before_targets)
    state = fixture.state
    losses = []
    for index in range(4):
        loss, state, prediction = fixture.cached.evaluate_step(
            fixture.params, state, fixture.keys[index], fixture.inputs[index],
            fixture.targets[index], fixture.forcings[index],
        )
        losses.append(float(loss))
        assert float(xarray_jax.unwrap_data(prediction["x"])[0, 0]) == seen[index][1]
    _assert_equal_tree(state, result[1])
    np.testing.assert_array_equal(losses, result[2])


def test_targets_can_be_supervised_only_and_lengths_are_checked():
    fixture = _setup("sparse_steps")
    selected = tuple(fixture.targets[index] for index in fixture.config.supervised_step_indices)
    full = fixture.cached.evaluate(fixture.params, fixture.state, fixture.keys, fixture.inputs, fixture.targets, fixture.forcings)
    sparse = fixture.cached.evaluate(fixture.params, fixture.state, fixture.keys, fixture.inputs, selected, fixture.forcings)
    _assert_equal_tree(full, sparse)
    with pytest.raises(ValueError, match="Expected 4 inputs"):
        fixture.cached.evaluate(fixture.params, fixture.state, fixture.keys, fixture.inputs[:-1], selected, fixture.forcings)
    with pytest.raises(ValueError, match="supervised targets"):
        fixture.cached.evaluate(fixture.params, fixture.state, fixture.keys, fixture.inputs, selected[:1], fixture.forcings)


def test_factory_builds_only_original_residual_predictor(monkeypatch):
    fixture = _setup()
    constructed = []

    class Predictor:
        def __call__(self, inputs, targets_template, forcings):
            del forcings
            head = hk.get_parameter("head", (), jnp.float32, init=hk.initializers.Constant(0.3))
            memory = hk.get_state("memory", (), jnp.float32, init=jnp.zeros)
            updated = memory + jnp.mean(xarray_jax.unwrap_data(inputs["x"]))
            hk.set_state("memory", updated)
            return jax.tree.map(lambda target: jnp.ones_like(target) * head * updated, targets_template)

        def loss_and_predictions(self, inputs, targets, forcings):
            prediction = self(inputs, targets, forcings)
            error = xarray_jax.unwrap_data(prediction["x"]) - xarray_jax.unwrap_data(targets["x"])
            value = jnp.mean(error**2)
            return ((xr.DataArray(value[None], dims=("batch",)), {}), prediction)

    model_config = object()

    def build_residual(model, task, stats, architecture, *, use_bf16):
        assert model is model_config
        assert use_bf16
        constructed.append(True)
        return Predictor()

    monkeypatch.setattr(cached_stepwise, "make_residual_predictor", build_residual)
    trainer = make_cached_stepwise_train_step(
        config=fixture.config, model_config=model_config, task_config=None, stats=None,
        optimizer=fixture.optimizer,
    )
    assert not constructed  # Factory does not initialize any neural model.
    assert not hasattr(trainer.transforms, "baseline_predict")
    params, state = trainer.transforms.residual_loss_and_predictions.init(
        fixture.keys[0], fixture.inputs[0], fixture.targets[0], fixture.forcings[0]
    )
    loss, next_state, _prediction = trainer.evaluate_step(
        params, state, fixture.keys[0], fixture.inputs[0], fixture.targets[0], fixture.forcings[0]
    )
    assert constructed
    assert np.isfinite(loss)
    assert any(float(leaf) != 0.0 for leaf in jax.tree.leaves(next_state))


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_stateless_full_mamba_matches_reset_ssm_and_conv_cached_training(precision):
    """The memory ablation must preserve Mamba weights and one-step gradients."""
    fixture = _setup(precision=precision)
    architecture = V24IlyaArchitectureConfig(
        width=4, temporal_d_inner=4, temporal_d_state=3, temporal_d_conv=4,
        temporal_dt_rank=2, temporal_init_scheme="mamba1",
        temporal_zero_init_out=False,
    )
    dtype = jnp.bfloat16 if precision == "bf16" else jnp.float32

    def transforms(stateful):
        def prediction(inputs, targets, forcings):
            del forcings
            value = jnp.mean(xarray_jax.unwrap_data(inputs["x"]))
            latent = jnp.stack((value, .2 + .7 * value, 1.0 - .3 * value, -.4 * value))
            predictor = object.__new__(gc.GraphCast)
            attach_temporal(predictor, dataclasses.replace(architecture, temporal_stateful=stateful))
            output = predictor._run_temporal_mesh_block(latent.astype(dtype)[None, None])
            result = output.astype(jnp.float32).mean()
            return jax.tree.map(lambda target: jnp.ones_like(target) * result, targets)

        def supervised(inputs, targets, forcings):
            predicted = prediction(inputs, targets, forcings)
            error = (xarray_jax.unwrap_data(predicted["x"]).astype(dtype)
                     - xarray_jax.unwrap_data(targets["x"]).astype(dtype))
            loss = jnp.mean(error**2).astype(jnp.float32)
            return ((xr.DataArray(loss[None], dims=("batch",)), {}), predicted)

        return CachedResidualTransforms(hk.transform_with_state(prediction), hk.transform_with_state(supervised))

    persistent, stateless = transforms(True), transforms(False)
    init_args = (fixture.keys[0], fixture.inputs[0], fixture.targets[0], fixture.forcings[0])
    params, state = persistent.residual_loss_and_predictions.init(*init_args)
    stateless_params, stateless_state = stateless.residual_loss_and_predictions.init(*init_args)
    _assert_equal_tree(stateless_params, params)
    assert stateless_state == {}
    state_names = {name for module in state.values() for name in module}
    assert any(name.endswith("_ssm_state") for name in state_names)
    assert any(name.endswith("_conv_cache") for name in state_names)

    reset_config = dataclasses.replace(fixture.config, architecture=architecture,
                                      temporal_state_policy="reset_every_anchor")
    stateless_config = dataclasses.replace(
        fixture.config, architecture=dataclasses.replace(architecture, temporal_stateful=False),
        temporal_state_policy="carry",
    )
    reset_trainer = CachedStepwiseTrainer(config=reset_config, optimizer=fixture.optimizer, transforms=persistent)
    stateless_trainer = CachedStepwiseTrainer(config=stateless_config, optimizer=fixture.optimizer, transforms=stateless)
    # Nonzero incoming SSM and convolution state must both be ignored.
    dirty_state = jax.tree.map(lambda value: jnp.ones_like(value) * .25, state)
    args = (fixture.keys, fixture.inputs, fixture.targets, fixture.forcings)
    reference = reset_trainer.loss_and_grad(params, dirty_state, *args)
    actual = stateless_trainer.loss_and_grad(stateless_params, stateless_state, *args)
    for index in (0, 2, 3, 4):
        _assert_equal_tree(actual[index], reference[index])
    assert actual[1] == {}
    optimizer_state = fixture.optimizer.init(params)
    actual_update = stateless_trainer(params, {}, optimizer_state, *args)
    reference_update = reset_trainer(params, dirty_state, optimizer_state, *args)
    for index in (0, 2, 3, 4, 5):
        _assert_equal_tree(actual_update[index], reference_update[index])
