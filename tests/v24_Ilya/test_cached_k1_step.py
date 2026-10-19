"""Numerical checks of native batching, masking and reverse-time credit."""

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
import pytest
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training import cached_stepwise
from src.models.mamba.v24_Ilya.training.cached_k1_step import (
    CachedK1Trainer, make_cached_k1_step,
)
from src.models.mamba.v24_Ilya.training.cached_stepwise import CachedResidualTransforms
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig


def _config(precision="fp32", **kwargs):
    return V24IlyaTrainConfig(
        prepared_root=Path("prepared"), anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline"), output_root=Path("output"),
        run_name="cached-k1-test", architecture=V24IlyaArchitectureConfig(width=4),
        segment_steps=4, bptt_steps=4, ar_tail_k=0, feedback_mode="baseline",
        loss_mode="all_steps", precision=precision, **kwargs,
    )


def _dataset(values, *, transpose=False):
    values = jnp.asarray(values, dtype=jnp.float32)
    data = values[None, :] if transpose else values[:, None]
    dims = ("time", "batch") if transpose else ("batch", "time")
    return xr.Dataset({"x": (dims, data)}, coords={"time": [np.timedelta64(6, "h")]})


def _values(dataset):
    return xarray_jax.unwrap_data(dataset["x"].transpose("batch", "time"))[:, 0]


def _recurrence(params, state, values, dtype):
    cast = lambda x: x.astype(dtype).astype(jnp.float32)
    memory = cast(cast(params["decay"]) * cast(state) + cast(params["input"]) * cast(values))
    output = cast(cast(params["head"]) * memory + cast(params["bias"]))
    return output, memory


class _Prediction:
    def __init__(self, dtype):
        self.dtype = dtype

    def apply(self, params, state, key, inputs, target, forcing):
        del key, target
        prediction, memory = _recurrence(
            params, state["memory"], _values(inputs) + _values(forcing), self.dtype,
        )
        return _dataset(prediction), {"memory": memory}


class _Loss:
    def __init__(self, prediction):
        self.prediction = prediction

    def apply(self, params, state, key, inputs, target, forcing):
        prediction, memory = self.prediction.apply(params, state, key, inputs, target, forcing)
        errors = _values(prediction).astype(self.prediction.dtype) - _values(target).astype(self.prediction.dtype)
        losses = (errors ** 2).astype(jnp.float32)
        return ((xr.DataArray(xarray_jax.wrap(losses), dims=("batch",)), {}), prediction), memory


def _fixture(batch=4, precision="fp32"):
    dtype = jnp.bfloat16 if precision == "bf16" else jnp.float32
    prediction = _Prediction(dtype)
    optimizer = optax.adam(0.003)
    trainer = CachedK1Trainer(
        config=_config(precision), optimizer=optimizer,
        transforms=CachedResidualTransforms(prediction, _Loss(prediction)),
    )
    inputs = np.arange(4 * batch, dtype=np.float32).reshape(4, batch) / 10 + 0.2
    inputs[-1] = 0.0  # Last loss still needs early-step recurrent credit.
    targets = np.linspace(-.7, 1.1, 4 * batch, dtype=np.float32).reshape(4, batch)
    forcings = np.full_like(inputs, .03)
    params = {name: jnp.asarray(value, jnp.float32) for name, value in
              {"input": .31, "decay": .72, "head": .41, "bias": .17}.items()}
    state = {"memory": jnp.linspace(.1, .6, batch)}
    return SimpleNamespace(
        trainer=trainer, optimizer=optimizer, dtype=dtype, params=params, state=state,
        keys=jax.random.split(jax.random.PRNGKey(11), 4),
        inputs=tuple(_dataset(x, transpose=True) for x in inputs),
        targets=tuple(_dataset(x) for x in targets),
        forcings=tuple(_dataset(x) for x in forcings),
        input_values=inputs, target_values=targets, forcing_values=forcings,
    )


def _reference(fixture, params, lane, *, detach=False):
    memory = fixture.state["memory"][lane]
    losses = []
    for index in range(4):
        if detach:
            memory = jax.lax.stop_gradient(memory)
        predicted, memory = _recurrence(
            params, memory,
            jnp.asarray(fixture.input_values[index, lane] + fixture.forcing_values[index, lane]),
            fixture.dtype,
        )
        error = predicted.astype(fixture.dtype) - jnp.asarray(fixture.target_values[index, lane], fixture.dtype)
        losses.append((error ** 2).astype(jnp.float32))
    return jnp.mean(jnp.stack(losses))


def _tree_close(actual, expected, **kwargs):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, **kwargs)


@pytest.mark.parametrize("batch,valid", [(8, 8), (8, 3), (4, 4), (4, 1)])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_batched_gradients_and_optimizer_equal_independent_lanes(batch, valid, precision):
    f = _fixture(batch, precision)
    mask = np.arange(batch) < valid
    loss, state, components, predictions, gradient = f.trainer.loss_and_grad(
        f.params, f.state, f.keys, f.inputs, f.targets, f.forcings, mask,
        retain_predictions=True,
    )
    independent = [jax.value_and_grad(lambda p: _reference(f, p, lane))(f.params)
                   for lane in range(valid)]
    expected_loss = jnp.mean(jnp.stack([x[0] for x in independent]))
    expected_grad = jax.tree.map(lambda *xs: sum(xs) / valid, *[x[1] for x in independent])
    # BF16 rounds shared-parameter reductions at different sites from B=1 calls.
    tolerance = 0.025 if precision == "bf16" else 2e-6
    np.testing.assert_allclose(loss, expected_loss, rtol=tolerance, atol=1e-7)
    _tree_close(gradient, expected_grad, rtol=tolerance,
                atol=2e-4 if precision == "bf16" else 1e-6)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(gradient))
    assert components.shape == (4,)
    assert len(predictions) == 4
    np.testing.assert_array_equal(np.asarray(state["memory"])[~mask], 0.0)
    initial_opt = f.optimizer.init(f.params)
    actual = f.trainer(f.params, f.state, initial_opt, f.keys,
                       f.inputs, f.targets, f.forcings, mask)
    updates, expected_opt = f.optimizer.update(gradient, initial_opt, f.params)
    expected = optax.apply_updates(f.params, updates)
    _tree_close(actual[:3], (expected, state, expected_opt), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual[4], optax.global_norm(gradient), rtol=2e-6)


def test_all_step_gradient_contains_reverse_time_credit():
    f = _fixture()
    result = f.trainer.loss_and_grad(f.params, f.state, f.keys, f.inputs, f.targets,
                                     f.forcings, [True] * 4)
    full = jax.grad(lambda p: sum(_reference(f, p, i) for i in range(4)) / 4)(f.params)
    detached = jax.grad(lambda p: sum(_reference(f, p, i, detach=True) for i in range(4)) / 4)(f.params)
    _tree_close(result[4], full, rtol=2e-6, atol=1e-7)
    assert abs(float(full["input"] - detached["input"])) > .01


def test_nan_padding_is_ignored_and_changing_lane_does_not_change_other_lanes():
    f = _fixture()
    mask = np.array([True, False, True, False])
    expected = f.trainer.loss_and_grad(f.params, f.state, f.keys, f.inputs, f.targets,
                                      f.forcings, mask, retain_predictions=True)
    def corrupt(sequence):
        return tuple(_dataset(np.where(mask, np.asarray(_values(ds)), np.nan)) for ds in sequence)
    corrupted_state = {"memory": jnp.where(mask, f.state["memory"], jnp.nan)}
    actual = f.trainer.loss_and_grad(f.params, corrupted_state, f.keys, corrupt(f.inputs),
                                    corrupt(f.targets), corrupt(f.forcings), mask,
                                    retain_predictions=True)
    _tree_close(actual, expected, rtol=2e-6, atol=1e-7)
    changed_state = {"memory": f.state["memory"].at[0].set(7.)}
    changed_inputs = tuple(_dataset(_values(ds).at[0].set(12.)) for ds in f.inputs)
    other = f.trainer.evaluate(f.params, changed_state, f.keys, changed_inputs,
                               f.targets, f.forcings, mask, retain_predictions=True)
    np.testing.assert_array_equal(other[1]["memory"][2:], expected[1]["memory"][2:])
    for prediction, original in zip(other[3], expected[3], strict=True):
        np.testing.assert_array_equal(_values(prediction)[2:], _values(original)[2:])


def test_native_mamba_ssm_and_convolution_carry_match_independent_lanes():
    """Exercise real native Mamba state layouts, not only scalar recurrence."""
    from graphcast import graphcast as gc
    from src.models.mamba.v24_Ilya.model import attach_temporal

    f = _fixture(3)
    architecture = V24IlyaArchitectureConfig(
        width=4, temporal_d_inner=4, temporal_d_state=3, temporal_d_conv=3,
        temporal_dt_rank=2, temporal_init_scheme="mamba1",
        temporal_zero_init_out=False,
    )

    def prediction(inputs, targets, forcings):
        del targets, forcings
        value = _values(inputs)
        features = jnp.stack([value, value * .3 + .2, 1. - value * .1, -.4 * value], -1)
        predictor = object.__new__(gc.GraphCast)
        attach_temporal(predictor, architecture)
        latent = predictor._run_temporal_mesh_block(features[None, :, :])
        result = hk.Linear(1, name="test_head")(latent)[0, :, 0]
        return _dataset(result)

    def supervised(inputs, targets, forcings):
        predicted = prediction(inputs, targets, forcings)
        loss = (_values(predicted) - _values(targets)) ** 2
        return ((xr.DataArray(xarray_jax.wrap(loss), dims=("batch",)), {}), predicted)

    transforms = CachedResidualTransforms(hk.transform_with_state(prediction),
                                           hk.transform_with_state(supervised))
    batched = CachedK1Trainer(config=_config(), optimizer=f.optimizer, transforms=transforms)
    params, state = batched.initialize(f.keys[0], f.inputs[0], f.targets[0], f.forcings[0])
    # Nonzero incoming SSM and convolution state catches accidental resets.
    state = jax.tree.map(lambda x: jnp.ones_like(x) * .13, state)
    assert state and all(x.shape[0] == 3 for x in jax.tree.leaves(state))
    actual = batched.loss_and_grad(params, state, f.keys, f.inputs, f.targets,
                                   f.forcings, [True, False, True], retain_predictions=True)
    single = cached_stepwise.CachedStepwiseTrainer(config=_config(), optimizer=f.optimizer,
                                                   transforms=transforms)
    independent = []
    for lane in (0, 2):
        select = lambda sequence: tuple(ds.isel(batch=slice(lane, lane + 1)) for ds in sequence)
        independent.append(single.loss_and_grad(
            params, jax.tree.map(lambda x: x[lane:lane + 1], state), f.keys,
            select(f.inputs), select(f.targets), select(f.forcings), retain_predictions=True,
        ))
    expected_grad = jax.tree.map(lambda a, b: (a + b) / 2,
                                  independent[0][4], independent[1][4])
    np.testing.assert_allclose(actual[0], (independent[0][0] + independent[1][0]) / 2,
                               rtol=2e-5, atol=1e-6)
    _tree_close(actual[4], expected_grad, rtol=2e-4, atol=2e-6)
    for result, lane in zip(independent, (0, 2), strict=True):
        _tree_close(jax.tree.map(lambda x: x[lane:lane + 1], actual[1]), result[1],
                    rtol=2e-5, atol=1e-6)
    for leaf in jax.tree.leaves(actual[1]):
        np.testing.assert_array_equal(leaf[1], 0.0)


def test_streaming_predictions_and_invalid_masks():
    f = _fixture()
    seen = []
    result = f.trainer.evaluate(f.params, f.state, f.keys, f.inputs, f.targets,
                                f.forcings, [1, 0, 1, 0],
                                prediction_callback=lambda index, prediction: seen.append(index))
    assert seen == [0, 1, 2, 3]
    assert result[3] == ()
    for bad in ([0, 0, 0, 0], [1, np.nan, 0, 0], [[1, 1, 1, 1]], [1, 1]):
        with pytest.raises(ValueError, match="valid_mask|batch dimension"):
            f.trainer.evaluate(f.params, f.state, f.keys, f.inputs, f.targets, f.forcings, bad)
    with pytest.raises(ValueError, match="Expected 4 inputs"):
        f.trainer.evaluate(f.params, f.state, f.keys, f.inputs[:3], f.targets,
                           f.forcings, [1, 1, 1, 1])


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_single_step_evaluation_matches_chunk_and_checkpoint_roundtrip(precision, tmp_path):
    import pickle

    f = _fixture(4, precision)
    mask = [1, 0, 1, 0]
    expected = f.trainer.evaluate(f.params, f.state, f.keys, f.inputs, f.targets,
                                 f.forcings, mask, retain_predictions=True)
    memory = f.state
    losses = []
    for index in range(4):
        loss, memory, predicted = f.trainer.evaluate_step(
            f.params, memory, f.keys[index], f.inputs[index], f.targets[index],
            f.forcings[index], mask,
        )
        losses.append(float(loss))
        _tree_close(predicted, expected[3][index], rtol=0, atol=0)
    _tree_close(memory, expected[1], rtol=0, atol=0)
    np.testing.assert_array_equal(losses, expected[2])
    # Pickled host parameters/state load without any renaming or model init.
    path = tmp_path / "roundtrip.pkl"
    with path.open("wb") as handle:
        pickle.dump(jax.device_get((f.params, memory)), handle)
    with path.open("rb") as handle:
        restored_params, restored_state = pickle.load(handle)
    args = (f.keys[0], f.inputs[0], f.targets[0], f.forcings[0], mask)
    before = f.trainer.evaluate_step(f.params, memory, *args)
    after = f.trainer.evaluate_step(restored_params, restored_state, *args)
    _tree_close(after, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="valid lane"):
        f.trainer.evaluate_step(f.params, memory, *args[:-1], [0, 0, 0, 0])


def test_factory_initializes_native_residual_only_and_preserves_parameter_tree(monkeypatch):
    f = _fixture(4)
    constructed = []

    class Predictor:
        def __call__(self, inputs, targets_template, forcings):
            del targets_template, forcings
            head = hk.get_parameter("native_head", (), init=hk.initializers.Constant(.3))
            memory = hk.get_state("native_memory", (inputs.sizes["batch"],), init=jnp.zeros)
            next_memory = .7 * memory + _values(inputs)
            hk.set_state("native_memory", next_memory)
            return _dataset(head * next_memory)

        def loss_and_predictions(self, inputs, targets, forcings):
            prediction = self(inputs, targets, forcings)
            losses = (_values(prediction) - _values(targets)) ** 2
            return ((xr.DataArray(xarray_jax.wrap(losses), dims=("batch",)), {}), prediction)

    def build(*args, **kwargs):
        constructed.append(kwargs["use_bf16"])
        return Predictor()

    monkeypatch.setattr(cached_stepwise, "make_residual_predictor", build)
    step = make_cached_k1_step(config=_config(), model_config=object(),
                               task_config=None, stats=None, optimizer=f.optimizer)
    assert constructed == []
    params, state = step.initialize(f.keys[0], f.inputs[0], f.targets[0], f.forcings[0])
    assert constructed and not any(constructed)
    assert set(params["~"]) == {"native_head"}
    native = cached_stepwise.CachedStepwiseTrainer(
        config=_config(), optimizer=f.optimizer, transforms=step.transforms,
    )
    expected = native.loss_and_grad(params, state, f.keys, f.inputs, f.targets, f.forcings,
                                    retain_predictions=False)
    actual = step.loss_and_grad(params, state, f.keys, f.inputs, f.targets, f.forcings,
                                [True] * 4)
    _tree_close(actual, expected, rtol=2e-6, atol=1e-7)
