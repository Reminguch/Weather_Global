from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import optax
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig
from src.models.mamba.v24_Ilya.training.endpoint_step import (
    V24IlyaTrainingTransforms,
    _overwrite_host_tree,
    _reusable_fp32_host_tree,
    make_bptt_objective,
    make_train_step,
    make_validation_step,
    memory_contract,
)
from src.models.mamba.v24_Ilya.training.validation import (
    run_fixed_validation,
    validation_step_keys,
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


def _config(loss_mode: str, tape_precision: str = "fp32") -> V24IlyaTrainConfig:
    sparse_objective = (
        {
            "supervised_horizons": (1, 3),
            "supervised_weights": (1, 3),
        }
        if loss_mode == "sparse_steps"
        else {}
    )
    return V24IlyaTrainConfig(
        prepared_root=Path("prepared"),
        anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline"),
        output_root=Path("output"),
        run_name=f"test-{loss_mode}",
        architecture=V24IlyaArchitectureConfig(width=4),
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
    components = jnp.stack(losses)
    loss = jnp.sum(components * jnp.asarray(weights))
    return loss, (state, components)


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
    transforms = V24IlyaTrainingTransforms(
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
    (expected_loss, (expected_state, _expected_components)), expected_gradient = (
        jax.value_and_grad(
            lambda value: _naive_reference(value, state["s"], loss_mode),
            has_aux=True,
        )(params)
    )

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


def test_validation_objective_returns_sparse_horizon_components() -> None:
    config = _config("sparse_steps")
    residual = StatefulResidual()
    residual_loss = StatefulResidualLoss()
    objective = make_bptt_objective(
        transforms=V24IlyaTrainingTransforms(
            FrozenBaseline(),
            residual,
            residual_loss,
            residual_loss,
        ),
        baseline_params={},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
        return_loss_components=True,
    )
    loss, _state, components = objective(
        {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)},
        {"s": jnp.asarray(0.0)},
        *_arguments("sparse_steps"),
    )
    assert np.asarray(components).shape == (2,)
    np.testing.assert_allclose(
        loss,
        np.sum(np.asarray(components) * np.asarray([0.25, 0.75])),
        rtol=1e-6,
    )


@pytest.mark.parametrize("loss_mode", ["last_step", "sparse_steps", "all_steps"])
def test_host_tape_train_step_matches_naive_parameter_update(loss_mode: str) -> None:
    baseline = FrozenBaseline()
    residual = StatefulResidual()
    residual_loss = StatefulResidualLoss()
    config = _config(loss_mode)
    transforms = V24IlyaTrainingTransforms(
        baseline,
        residual,
        residual_loss,
        residual_loss,
    )
    optimizer = optax.adam(0.01)
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
    keys, frames, static, truths, forcings = _arguments(loss_mode)

    initial_optimizer_state = optimizer.init(params)
    (
        next_params,
        next_state,
        next_optimizer_state,
        loss,
        gradient_norm,
        loss_components,
    ) = train_step(
        params,
        state,
        initial_optimizer_state,
        keys,
        frames,
        static,
        truths,
        forcings,
    )
    (
        expected_loss,
        (expected_state, expected_loss_components),
    ), expected_gradient = jax.value_and_grad(
        lambda value: _naive_reference(value, state["s"], loss_mode),
        has_aux=True,
    )(params)
    expected_updates, expected_optimizer_state = optimizer.update(
        expected_gradient,
        initial_optimizer_state,
        params,
    )
    expected_params = optax.apply_updates(params, expected_updates)

    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(
        loss,
        np.sum(
            np.asarray(loss_components)
            * np.asarray(config.normalized_supervised_weights)
        ),
        rtol=1e-6,
    )
    assert np.asarray(loss_components).shape == (
        len(config.supervised_step_indices),
    )
    np.testing.assert_allclose(
        loss_components,
        expected_loss_components,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(next_state["s"], expected_state, rtol=1e-6)
    for name in params:
        np.testing.assert_allclose(
            next_params[name],
            expected_params[name],
            rtol=1e-6,
            atol=1e-6,
        )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(next_optimizer_state),
        jax.tree_util.tree_leaves(expected_optimizer_state),
        strict=True,
    ):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-6, atol=1e-6)
    assert float(gradient_norm) > 0.0


def test_reusable_target_uses_writable_fp32_storage() -> None:
    values = np.asarray([[3.0]], dtype=np.float32)
    truth = xr.Dataset(
        {"x": (("batch", "time"), values)},
        coords={"batch": [0], "time": [np.timedelta64(6, "h")]},
    )
    reusable = _reusable_fp32_host_tree(truth)
    assert np.shares_memory(reusable["x"].values, truth["x"].values)

    target = truth.assign(x=(("batch", "time"), np.asarray([[1.5]], np.float32)))
    _overwrite_host_tree(reusable, target)
    np.testing.assert_array_equal(truth["x"].values, [[1.5]])


def test_reusable_target_copies_read_only_storage() -> None:
    values = np.asarray([[3.0]], dtype=np.float32)
    values.flags.writeable = False
    truth = xr.Dataset(
        {"x": (("batch", "time"), values)},
        coords={"batch": [0], "time": [np.timedelta64(6, "h")]},
    )
    reusable = _reusable_fp32_host_tree(truth)
    assert not np.shares_memory(reusable["x"].values, truth["x"].values)
    assert reusable["x"].values.flags.writeable

    target = truth.assign(x=(("batch", "time"), np.asarray([[1.5]], np.float32)))
    _overwrite_host_tree(reusable, target)
    np.testing.assert_array_equal(truth["x"].values, [[3.0]])
    np.testing.assert_array_equal(reusable["x"].values, [[1.5]])


def test_reusable_target_never_mutates_memmap(tmp_path: Path) -> None:
    path = tmp_path / "truth.bin"
    source = np.memmap(path, mode="w+", dtype=np.float32, shape=(1, 1))
    source[:] = 3.0
    source.flush()
    checksum_before = hashlib.sha256(path.read_bytes()).hexdigest()
    truth = xr.Dataset(
        {"x": (("batch", "time"), source)},
        coords={"batch": [0], "time": [np.timedelta64(6, "h")]},
    )
    reusable = _reusable_fp32_host_tree(truth)
    assert not np.shares_memory(reusable["x"].values, source)

    target = truth.assign(x=(("batch", "time"), np.asarray([[1.5]], np.float32)))
    _overwrite_host_tree(reusable, target)
    source.flush()
    checksum_after = hashlib.sha256(path.read_bytes()).hexdigest()
    reloaded = np.memmap(path, mode="r", dtype=np.float32, shape=(1, 1))
    assert checksum_after == checksum_before
    np.testing.assert_array_equal(reloaded, [[3.0]])
    np.testing.assert_array_equal(reusable["x"].values, [[1.5]])

def test_explicit_reverse_jits() -> None:
    baseline = FrozenBaseline()
    residual = StatefulResidual()
    residual_loss = StatefulResidualLoss()
    config = _config("last_step")
    objective = make_bptt_objective(
        transforms=V24IlyaTrainingTransforms(
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


def test_memory_contract_is_unique_frame_fp32_tape() -> None:
    config = _config("last_step", tape_precision="fp32")
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
    assert report["truth_target_storage"] == "reuse_owned_fp32_truth_buffer"
    assert report["truth_target_retained_copies"] == 1
    assert report["truth_target_staging_frames"] == 1
    assert report["truth_target_logical_bytes"] == report["truth_target_bytes"]
    assert report["truth_target_staging_bytes"] == report["truth_target_bytes"]
    assert report["weather_tape_precision"] == "fp32"
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
    config = _config("last_step", tape_precision="fp32")
    objective = make_bptt_objective(
        transforms=V24IlyaTrainingTransforms(
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


class ValidationBaseline:
    """Nonzero, input-dependent baseline exercising live forecast feedback."""

    def __init__(self, dtype):
        self.dtype = dtype
        self.calls = 0

    def apply(self, params, state, key, inputs, template, forcing):
        self.calls += 1
        value = jnp.mean(xarray_jax.unwrap_data(inputs["x"]).astype(self.dtype))
        prediction = params["scale"].astype(self.dtype) * value
        return jax.tree_util.tree_map(
            lambda target: jnp.ones(target.shape, self.dtype) * prediction, template,
        ), state


class ValidationResidual:
    def __init__(self, dtype):
        self.dtype = dtype

    def apply(self, params, state, key, inputs, template, forcing):
        dtype = self.dtype
        value = jnp.mean(xarray_jax.unwrap_data(inputs["x"]).astype(dtype))
        forcing_value = jnp.mean(xarray_jax.unwrap_data(forcing["forcing"]).astype(dtype))
        next_value = (
            jnp.asarray(0.7, dtype) * state["s"].astype(dtype)
            + params["a"].astype(dtype) * value + forcing_value
            + jax.random.uniform(key, (), dtype=dtype) * jnp.asarray(0.01, dtype)
        )
        prediction = jax.tree_util.tree_map(
            lambda target: jnp.ones(target.shape, dtype) * params["b"].astype(dtype) * next_value,
            template,
        )
        return prediction, {"s": next_value}


class ValidationResidualLoss(ValidationResidual):
    def apply(self, params, state, key, inputs, target, forcing):
        prediction, next_state = super().apply(params, state, key, inputs, target, forcing)
        error = (
            xarray_jax.unwrap_data(prediction["x"]).astype(jnp.float32)
            - xarray_jax.unwrap_data(target["x"]).astype(self.dtype).astype(jnp.float32)
        )
        loss = xr.DataArray(jnp.mean(error**2)[None], dims=("batch",), coords={"batch": [0]})
        return ((loss, {}), prediction), next_state


def _validation_builders(config, dtype=jnp.float32):
    baseline = ValidationBaseline(dtype)
    residual = ValidationResidual(dtype)
    loss = ValidationResidualLoss(dtype)
    kwargs = dict(
        transforms=V24IlyaTrainingTransforms(baseline, residual, loss, loss),
        baseline_params={"scale": jnp.asarray(0.3)},
        baseline_state={}, config=config, time_step=pd.Timedelta("6h"), input_steps=2,
    )
    return kwargs, baseline


@pytest.mark.parametrize("loss_mode", ["last_step", "sparse_steps", "all_steps"])
@pytest.mark.parametrize("feedback_mode", ["closed_loop_sg", "baseline"])
@pytest.mark.parametrize("state_policy", ["carry", "reset_every_anchor"])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_streaming_validation_matches_objective(loss_mode, feedback_mode, state_policy, precision):
    config = replace(_config(loss_mode), feedback_mode=feedback_mode,
                     temporal_state_policy=state_policy, precision=precision)
    dtype = jnp.bfloat16 if precision == "bf16" else jnp.float32
    kwargs, baseline = _validation_builders(config, dtype)
    reference = jax.jit(make_bptt_objective(**kwargs, return_loss_components=True))
    validation = make_validation_step(**kwargs)
    args = jax.device_get(_arguments(loss_mode))
    before = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), args)
    params = {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)}
    actual_state = expected_state = {"s": jnp.asarray(0.4, jnp.float32)}
    rtol, atol = (1e-2, 1e-3) if precision == "bf16" else (1e-5, 1e-6)
    for chunk in range(2):
        # A second chunk checks state carry and runtime (not captured) parameters.
        params = {**params, "a": jnp.asarray(0.1 + chunk * 0.03)}
        actual_loss, actual_state, actual_components = validation(params, actual_state, *args)
        expected_loss, expected_state, expected_components = reference(params, expected_state, *args)
        for actual, expected in zip(
            jax.tree_util.tree_leaves((actual_loss, actual_state, actual_components)),
            jax.tree_util.tree_leaves((expected_loss, expected_state, expected_components)), strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        assert actual_state["s"].dtype == jnp.float32
        if chunk == 0:
            traces = baseline.calls
        else:
            assert baseline.calls == traces, "identical chunk structures should reuse compiled kernels"
    for actual, expected in zip(jax.tree_util.tree_leaves(args), jax.tree_util.tree_leaves(before), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_streaming_validation_full24_fixed_segments_and_state_isolation():
    config = replace(_config("all_steps"), segment_steps=120, bptt_steps=24, ar_tail_k=19)
    kwargs, _baseline = _validation_builders(config)
    validation = make_validation_step(**kwargs)
    reference = jax.jit(make_bptt_objective(**kwargs, return_loss_components=True))
    chunk = SimpleNamespace(
        input_frames=tuple(jax.device_get(_input_frame(float(i + 1))) for i in range(6)),
        static_inputs=xr.Dataset(),
        truths=tuple(jax.device_get(_dataset(float(i + 3), hour=6)) for i in range(24)),
        forcings=tuple(jax.device_get(_dataset(0.01 * i, "forcing", 6)) for i in range(24)),
    )
    data = SimpleNamespace(
        validation_segments=[np.arange(120) for _ in range(8)],
        load_segment_chunk=lambda *args: chunk,
        fingerprint_validation_subset=lambda ids: "fixed-test-subset",
    )
    params = {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)}
    zero_state = {"s": jnp.asarray(0.0)}
    observed_states = []

    def observe(*args):
        observed_states.append(float(args[1]["s"]))
        return validation(*args)

    record = run_fixed_validation(
        validation_step=observe, residual_params=params, zero_residual_state=zero_state,
        training_data=data, task_config=None, config=config, segment_ids=np.arange(8),
        step=100, role="fixed_checkpoint", subset_policy="stratified_fixed",
    )
    expected_losses, expected_components = [], []
    for segment in range(8):
        state = zero_state
        for index in range(5):
            keys = validation_step_keys(seed=config.seed, segment_id=segment, chunk_index=index, bptt_steps=24)
            loss, state, components = reference(
                params, state, keys, chunk.input_frames, chunk.static_inputs, chunk.truths, chunk.forcings,
            )
            expected_losses.append(float(loss))
            expected_components.append(np.asarray(components))
    np.testing.assert_allclose(record["loss"], np.mean(expected_losses), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(list(record["loss_by_horizon"].values()),
                               np.mean(expected_components, axis=0), rtol=1e-5, atol=1e-6)
    assert record["num_anchors"] == 960
    assert record["num_chunks"] == 40
    assert record["subset_fingerprint"] == "fixed-test-subset"
    assert observed_states[::5] == [0.0] * 8
    assert all(value != 0.0 for i, value in enumerate(observed_states) if i % 5)
    assert float(zero_state["s"]) == 0.0


def test_streaming_validation_matches_training_forward_loss_and_state():
    config = _config("all_steps")
    kwargs, _baseline = _validation_builders(config)
    validation = make_validation_step(**kwargs)
    optimizer = optax.adam(0.01)
    training = make_train_step(**kwargs, optimizer=optimizer)
    params = {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)}
    state = {"s": jnp.asarray(0.4)}
    args = jax.device_get(_arguments("all_steps"))
    loss, final_state, components = validation(params, state, *args)
    _, train_state, _, train_loss, _, train_components = training(
        params, state, optimizer.init(params), *args,
    )
    np.testing.assert_allclose(loss, train_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(final_state["s"], train_state["s"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(components, train_components, rtol=1e-5, atol=1e-6)
