"""Dual recurrent gradients and forward-only calibration share one trajectory."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.training.endpoint_step import V24IlyaTrainingTransforms
from src.models.mamba.v24_Ilya.training.mgda_optimizer import RMSOptimizer
from src.models.mamba.v24_Ilya.training.mgda_step import make_multiobjective_train_step


def dataset(value, batch=4, name="x", hour=0):
    values = jnp.broadcast_to(jnp.asarray(value, jnp.float32), (batch, 1, 2))
    return xr.Dataset({name: (("batch", "time", "mode"), values)}, coords={
        "batch": np.arange(batch), "time": [np.timedelta64(hour, "h")], "mode": [0, 1]})


class Baseline:
    def apply(self, params, state, key, inputs, target, forcing):
        del params, key, inputs, forcing
        return jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * .5, target), state


class Residual:
    def apply(self, params, state, key, inputs, target, forcing):
        del forcing
        value = jnp.mean(xarray_jax.unwrap_data(inputs["x"]), axis=(1, 2))
        # A key-dependent term exposes accidental independent objective tapes.
        next_state = {"s": .7 * state["s"] + params["a"] * value + .01 * jax.random.normal(key, value.shape)}
        prediction = jax.tree_util.tree_map(
            lambda x: jnp.ones_like(x) * params["b"] * next_state["s"][:, None, None], target)
        return prediction, next_state


def gc_loss(prediction, truth):
    f, y = (xarray_jax.unwrap_data(x["x"])[:, 0] for x in (prediction, truth))
    return jnp.mean((f - y) ** 2, axis=-1), {}


def pair_loss(prediction, truth):
    f, y = (xarray_jax.unwrap_data(x["x"])[:, 0] for x in (prediction, truth))
    return jnp.stack((jnp.mean((f - y) ** 2, axis=-1),
                      jnp.mean((f ** 2 - y ** 2) ** 2, axis=-1))), {"floor_fraction": jnp.zeros(f.shape[0])}


def setup(kind="mgda_rms", reset=False):
    cfg = SimpleNamespace(bptt_steps=4, truth_prefix_steps=2,
        supervised_step_indices=(0, 1, 2, 3), normalized_supervised_weights=(.25,) * 4,
        feedback_mode="closed_loop_sg", temporal_state_policy="reset_every_anchor" if reset else "carry",
        distributed=SimpleNamespace(mode="single"), optimizer_kind=kind,
        power_calibration={"gc_scale": 2., "power_scale": 7.}, seed=22, loss_kind="mse_fp32")
    residual = Residual()
    transforms = V24IlyaTrainingTransforms(Baseline(), residual, None, None, gc_loss,
                                         residual, pair_loss)
    frames = tuple(xr.merge((dataset([i, .5 * i]), dataset([0., 0.], name="forcing"))) for i in (1, 2, 3))
    truths = tuple(dataset([i + 3., .5 * (i + 3.)], hour=6) for i in range(4))
    args = (jax.random.split(jax.random.PRNGKey(2), 4), frames, xr.Dataset(), truths,
            tuple(dataset([0., 0.], name="forcing", hour=6) for _ in range(4)))
    optimizer = RMSOptimizer(lambda count: jnp.asarray(1e-4), spatial_multiplier=1., mamba_multiplier=1.)
    params, state = {"a": jnp.asarray(.11), "b": jnp.asarray(.23)}, {"s": jnp.arange(4, dtype=jnp.float32) / 10}
    step = make_multiobjective_train_step(transforms=transforms, optimizer=optimizer,
        baseline_params={}, baseline_state={}, config=cfg, time_step=pd.Timedelta("6h"), input_steps=2)
    return cfg, transforms, optimizer, params, state, args, step


def reference(params, state, cfg, args):
    keys, initial_frames, _, truths, _ = args
    frames = [xarray_jax.unwrap_data(frame["x"])[:, 0] for frame in initial_frames]
    values = []
    positions = {index: pos for pos, index in enumerate(cfg.supervised_step_indices)}
    for i in range(4):
        old = jnp.zeros_like(state["s"]) if cfg.temporal_state_policy == "reset_every_anchor" else state["s"]
        mean = jnp.mean(jnp.stack((frames[i], frames[i + 1])), axis=(0, 2))
        state = {"s": .7 * old + params["a"] * mean + .01 * jax.random.normal(keys[i], mean.shape)}
        f = .5 + params["b"] * state["s"][:, None] * jnp.ones((4, 2))
        if i in positions:
            y = xarray_jax.unwrap_data(truths[positions[i]]["x"])[:, 0]
            values.append(jnp.stack((jnp.mean((f - y) ** 2), jnp.mean((f ** 2 - y ** 2) ** 2))))
        if i >= 1 and i < 3:
            frames.append(jax.lax.stop_gradient(f))
        if cfg.temporal_state_policy == "reset_every_anchor":
            state = {"s": jnp.zeros_like(state["s"])}
    return jnp.sum(jnp.stack(values) * jnp.asarray(cfg.normalized_supervised_weights)[:, None], axis=0), state


def assert_tree_close(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=4e-5, atol=3e-6)


@pytest.mark.parametrize("kind", ["mgda_rms", "rms_mse"])
@pytest.mark.parametrize("reset", [False, True])
def test_dual_reverse_matches_direct_ad_and_single_state_advance(kind, reset):
    cfg, _, optimizer, params, state, args, step = setup(kind, reset)
    values, next_state = reference(params, state, cfg, args)
    expected_gradients = tuple(jax.grad(lambda p: reference(p, state, cfg, args)[0][i])(params) for i in range(2))
    actual_gradients = step.objective_gradients(params, state, args)
    assert_tree_close(actual_gradients, expected_gradients)
    raw, forward_state, _, _ = step.forward_only(params, state, *args)
    assert_tree_close((raw, forward_state), (values, next_state))
    normalized = tuple(jax.tree_util.tree_map(lambda g: g / scale, gradient)
                       for scale, gradient in zip((2., 7.), expected_gradients))
    opt_state = optimizer.init(params)
    updates, expected_opt_state, _ = (optimizer.update_pair(*normalized, opt_state, params)
        if kind == "mgda_rms" else optimizer.update_single(normalized[0], opt_state, params))
    result = step(params, state, opt_state, *args)
    assert_tree_close((result.params, result.state, result.optimizer_state),
                      (optax.apply_updates(params, updates), next_state, expected_opt_state))
    assert float(result.loss) == pytest.approx(float(values[0]))
    assert result.diagnostics["power_loss"] == pytest.approx(float(values[1]))
    assert result.diagnostics["finite"]


def test_forward_only_does_not_need_calibration_but_training_does():
    cfg, transforms, optimizer, params, state, args, _ = setup()
    step = make_multiobjective_train_step(transforms=replace(transforms, full_field_objectives=None),
        optimizer=optimizer, baseline_params={}, baseline_state={}, config=cfg,
        time_step=pd.Timedelta("6h"), input_steps=2)
    raw, _, _, _ = step.forward_only(params, state, *args)
    assert raw[0] > 0 and raw[1] == 0
    with pytest.raises(ValueError, match="frozen power calibration"):
        step(params, state, optimizer.init(params), *args)


def test_dual_reverse_sparse_supervision_propagates_unsupervised_state():
    cfg, transforms, optimizer, params, state, args, _ = setup()
    cfg.supervised_step_indices = (1, 3)
    cfg.normalized_supervised_weights = (.25, .75)
    args = (*args[:3], (args[3][1], args[3][3]), args[4])
    step = make_multiobjective_train_step(transforms=transforms, optimizer=optimizer,
        baseline_params={}, baseline_state={}, config=cfg, time_step=pd.Timedelta("6h"), input_steps=2)
    expected = tuple(jax.grad(lambda p: reference(p, state, cfg, args)[0][i])(params) for i in range(2))
    assert_tree_close(step.objective_gradients(params, state, args), expected)


def test_calibration_replays_batches_rng_and_segment_state_without_updates(monkeypatch):
    from src.models.mamba.v24_Ilya.training import power_loss, runner
    from src.models.mamba.v24_Ilya.training.data import TrainingCursor

    cfg, transforms, _, params, state, args, _ = setup()
    calls = []
    def chunk(cursor, config, task):
        calls.append(cursor)
        return SimpleNamespace(input_frames=args[1], static_inputs=args[2], truths=args[3],
            forcings=args[4], raw_anchor_indices=np.array([cursor.segment_index * 8 + cursor.segment_offset]),
            next_cursor=(TrainingCursor(segment_index=cursor.segment_index + 1) if cursor.segment_offset else
                         TrainingCursor(segment_index=cursor.segment_index, segment_offset=4)))
    data = SimpleNamespace(build_chunk=chunk, input_steps=2, time_step=pd.Timedelta("6h"))
    monkeypatch.setattr(power_loss, "truth_powers", lambda truth: jnp.ones((4, 1, 9, 2)) *
                        (jnp.mean(xarray_jax.unwrap_data(truth["x"])) + 1))
    monkeypatch.setattr(power_loss, "build_power_loss", lambda calibration: lambda f, y:
                        (pair_loss(f, y)[0][1], {"floor_fraction": jnp.ones(4) * .125}))
    original_params = jax.tree_util.tree_map(np.asarray, params)
    original_state = jax.tree_util.tree_map(np.asarray, state)
    rng = jax.random.PRNGKey(19)
    result = runner._calibrate_initialized_training(config=cfg,
        transforms=replace(transforms, full_field_objectives=None), training_data=data, task_config=None,
        params=params, zero_state=state, rng_key=rng, baseline_params={}, baseline_state={},
        batches=4, provenance={"test_provenance": "fixed"})
    assert calls[:4] == calls[4:]
    assert [x.segment_offset for x in calls[:4]] == [0, 4, 0, 4]
    assert result["gc_scale"] > 0 and result["power_scale"] > 0
    assert result["floor_activation_fraction"] == pytest.approx(.125)
    assert result["test_provenance"] == "fixed"
    assert len(result["batch_manifest"]) == 4
    assert_tree_close(params, original_params)
    assert_tree_close(state, original_state)
    np.testing.assert_array_equal(rng, jax.random.PRNGKey(19))
