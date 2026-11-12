"""M1-native loss, detached weather feedback and additive spectral scoring."""
from dataclasses import replace
import json

from graphcast import xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.training import spectral_loss
from src.models.mamba.v24_Ilya.training.config import load_training_config, validate_resume_config
from src.models.mamba.v24_Ilya.training.endpoint_step import (
    V24IlyaTrainingTransforms, make_bptt_objective, make_train_step,
    make_validation_step, supervised_step, supervision_target,
)
from src.models.mamba.v24_Ilya.training.scalar_excess_loss import build_native_mse_excess_penalty
from test_amse_split_mgda_core import fields83
from test_endpoint_step import (
    FrozenBaseline, StatefulResidual, StatefulResidualLoss, _arguments, _config,
)


def native_config(mode="all_steps", weight=.5):
    return replace(_config(mode), loss_kind="native_mse_amse_excess",
                   spectral_excess_weight=weight, amse_lmax=5)


class NonzeroBaseline(FrozenBaseline):
    def apply(self, params, state, key, inputs, template, forcing):
        self.calls += 1
        return jax.tree_util.tree_map(lambda v: jnp.full_like(v, 1.25), template), state


class QuantizedNativeLoss(StatefulResidualLoss):
    def apply(self, params, state, key, inputs, target, forcing):
        prediction, next_state = StatefulResidual().apply(params, state, key, inputs, target, forcing)
        pred = xarray_jax.unwrap_data(prediction["x"]).astype(jnp.bfloat16)
        truth = xarray_jax.unwrap_data(target["x"]).astype(jnp.bfloat16)
        value = jnp.mean((pred-truth)**2).astype(jnp.float32)
        loss = xr.DataArray(value[None], dims=("batch",), coords={"batch": [0]})
        return ((loss, {}), prediction), next_state


def toy_penalty(weight):
    def penalty(prediction, truth):
        pred, true = [xarray_jax.unwrap_data(x["x"]) for x in (prediction, truth)]
        excess = jnp.mean((pred+true)**2, axis=1)
        weighted = weight*excess
        # A deliberately unrelated grid diagnostic catches substitution of the
        # FP32 grid diagnostic for the native predictor's quantized MSE.
        return weighted, {"amse_excess": excess, "weighted_amse_excess": weighted,
                          "grid_mse_fp32": jnp.full_like(excess, 999.)}
    return penalty


def transforms(weight=.5):
    native = QuantizedNativeLoss()
    return V24IlyaTrainingTransforms(NonzeroBaseline(), StatefulResidual(), native, native,
                                    full_field_loss=toy_penalty(weight),
                                    native_grid_with_excess=True)


def reference(params, state, cfg):
    args = _arguments(cfg.loss_mode)
    weather = [xarray_jax.unwrap_data(x["x"])[0, 0] for x in args[1]]
    positions = {step: i for i, step in enumerate(cfg.supervised_step_indices)}
    terms = []
    for index in range(cfg.bptt_steps):
        state = .7*state + params["a"]*.5*(weather[index]+weather[index+1])
        prediction = params["b"]*state
        if index in positions:
            truth = xarray_jax.unwrap_data(args[3][positions[index]]["x"])[0, 0]
            residual_target = truth-jnp.float32(1.25)
            native = ((prediction.astype(jnp.bfloat16)-residual_target.astype(jnp.bfloat16))**2).astype(jnp.float32)
            excess = (jnp.float32(1.25)+prediction+truth)**2
            terms.append(jnp.stack((native, excess)))
        if cfg.truth_prefix_steps-1 <= index < cfg.bptt_steps-1:
            weather.append(jax.lax.stop_gradient(jnp.float32(1.25)+prediction))
    native, excess = jnp.sum(jnp.stack(terms)*jnp.asarray(cfg.normalized_supervised_weights)[:, None], axis=0)
    return native+cfg.spectral_excess_weight*excess, (state, native, excess)


@pytest.mark.parametrize("mode", ["all_steps", "sparse_steps", "last_step"])
def test_native_loss_detached_weather_gradients_streamed_update_and_validation(mode):
    cfg = native_config(mode)
    kwargs = dict(transforms=transforms(), baseline_params={}, baseline_state={}, config=cfg,
                  time_step=pd.Timedelta("6h"), input_steps=2)
    params = {"a": jnp.array(.131), "b": jnp.array(.217)}
    state = {"s": jnp.array(.019)}
    args = _arguments(mode)
    (expected, (expected_state, native, excess)), gradient = jax.value_and_grad(
        lambda p: reference(p, state["s"], cfg), has_aux=True)(params)
    objective = make_bptt_objective(**kwargs)
    (actual, actual_state), actual_grad = jax.value_and_grad(
        lambda p: objective(p, state, *args), has_aux=True)(params)
    np.testing.assert_allclose(actual, expected, rtol=3e-6)
    np.testing.assert_allclose(actual_state["s"], expected_state, rtol=3e-6)
    for key in params:
        np.testing.assert_allclose(actual_grad[key], gradient[key], rtol=3e-6, atol=2e-6)
    optimizer = optax.sgd(.01)
    result = make_train_step(**kwargs, optimizer=optimizer)(params, state, optimizer.init(params), *args)
    np.testing.assert_allclose(result.loss, expected, rtol=3e-6)
    np.testing.assert_allclose(result.diagnostics["native_grid_mse"], native, rtol=3e-6)
    np.testing.assert_allclose(result.diagnostics["amse_excess"], excess, rtol=3e-6)
    np.testing.assert_allclose(result.diagnostics["weighted_amse_excess"], .5*excess, rtol=3e-6)
    assert result.diagnostics["spectral_diagnostics_computed"] is True
    for key in params:
        np.testing.assert_allclose(result.params[key], params[key]-.01*gradient[key], rtol=3e-6, atol=2e-6)
    validation = make_validation_step(**kwargs)(params, state, *args)
    np.testing.assert_allclose(validation[0], expected, rtol=3e-6)


@pytest.mark.parametrize("weight", [0., .25, .5, .75, 1.])
def test_real_penalty_matches_existing_spectral_excess_and_gradient(weight):
    truth, stats = fields83()
    amse = spectral_loss.build_full_field_loss(stats, "amse", lmax=5, amse_lambda=1.)
    penalty = build_native_mse_excess_penalty(stats, spectral_excess_weight=weight, lmax=5)
    actual, diagnostics = penalty(truth*.73+.1, truth)
    expected = lambda scale: weight*amse(truth*scale+.1, truth)[1]["amse_excess"]
    np.testing.assert_allclose(actual, expected(.73), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(jax.grad(lambda scale: penalty(truth*scale+.1, truth)[0].sum())(.73),
                               jax.grad(lambda scale: expected(scale).sum())(.73), rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(actual, diagnostics["weighted_amse_excess"])


def test_zero_penalty_does_not_construct_or_evaluate_spherical_transform(monkeypatch):
    truth, stats = fields83()
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero lambda must not create/evaluate any full-field loss")
    from src.models.mamba.v24_Ilya.training import scalar_excess_loss
    monkeypatch.setattr(scalar_excess_loss, "build_full_field_loss", forbidden)
    penalty = build_native_mse_excess_penalty(stats, spectral_excess_weight=0, lmax=5)
    values, diagnostics = jax.jit(penalty)(truth*.7, truth)
    np.testing.assert_array_equal(values, 0.)
    assert "grid_mse_fp32" not in diagnostics


def test_native_config_roundtrip_rejects_other_backends_and_schedule_changes(tmp_path):
    cfg = native_config()
    path = tmp_path/"config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert load_training_config(path) == cfg
    assert cfg.to_dict()["objective"]["spectral_excess_weight"] == .5
    for change in ({"model_family": "full_gc"}, {"feedback_mode": "closed_loop"},
                   {"feedback_mode": "baseline"}, {"amse_lambda": .5},
                   {"spectral_excess_weight": float("nan")}, {"amse_lambda_ramp_steps": 1}):
        with pytest.raises(ValueError):
            replace(cfg, **change)
    validate_resume_config(cfg, cfg.to_dict(), completed_step=2)
    with pytest.raises(ValueError, match="max_steps fixed"):
        validate_resume_config(replace(cfg, max_steps=cfg.max_steps+1), cfg.to_dict(), completed_step=2)


def test_historical_native_zero_loss_and_gradient_are_unchanged():
    native = QuantizedNativeLoss()
    unchanged = V24IlyaTrainingTransforms(NonzeroBaseline(), StatefulResidual(), native, native)
    args = _arguments("all_steps")
    inputs, truth, forcing = args[1][0], args[3][0], args[4][0]
    baseline = jax.tree_util.tree_map(lambda v: jnp.full_like(v, 1.25), truth)
    target = supervision_target(unchanged, truth, baseline)
    params = {"a": jnp.array(.131), "b": jnp.array(.217)}
    state = {"s": jnp.array(.019)}
    expected = lambda p: xarray_jax.unwrap_data(native.apply(p, state, args[0][0], inputs, target, forcing)[0][0][0]).mean()
    actual = lambda p: supervised_step(unchanged, p, state, args[0][0], inputs, target, forcing)[0]
    np.testing.assert_array_equal(actual(params), expected(params))
    for a, e in zip(jax.tree_util.tree_leaves(jax.grad(actual)(params)),
                    jax.tree_util.tree_leaves(jax.grad(expected)(params))):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("weight", [0., .5])
def test_runner_stage_reset_and_exact_midsegment_resume(tmp_path, monkeypatch, weight):
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    from src.models.mamba.v24_Ilya.training import runner
    from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainInvocation
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer
    from src.models.mamba.v24_Ilya.training.weather_feedback_step import ScalarTrainResult
    from test_mgda_runner import mock_run

    original = mock_run(tmp_path, monkeypatch)
    # The fixture has one parameter module; treat it as the temporal group so
    # the main-stage 3x optimizer schedule is exercised without a real GNN.
    from src.models.mamba.v24_Ilya.training import endpoint_step
    monkeypatch.setattr(endpoint_step, "is_mamba_parameter", lambda name: name == "module")
    monkeypatch.setattr(runner, "is_mamba_parameter", lambda name: name == "module")
    cfg = replace(original, run_name="initial", loss_kind="native_mse_amse_excess" if weight else "mse",
                  spectral_excess_weight=weight, optimizer_kind="adamw", power_calibration=None,
                  max_steps=3, checkpoint_every=1, learning_rate_schedule="constant",
                  learning_rate=1e-4, end_learning_rate=None, warmup_steps=1)
    def make_step(optimizer, **kwargs):
        def step(params, state, optimizer_state, keys, *args):
            gradients = jax.tree_util.tree_map(
                lambda p: 2*p+jnp.mean(state["s"])+.01*jax.random.normal(keys[0], p.shape), params)
            updates, next_optimizer = optimizer.update(gradients, optimizer_state, params)
            next_params = optax.apply_updates(params, updates)
            next_state = {"s": .7*state["s"]+params["module"]["w"][None]}
            return ScalarTrainResult(next_params, next_state, next_optimizer, jnp.array(1.),
                optax.global_norm(gradients), jnp.ones(2),
                {"native_grid_mse": .8 if weight else 1., "weighted_amse_excess": .2 if weight else 0.,
                 "scalar_total": 1., "spectral_diagnostics_computed": bool(weight)})
        return step
    monkeypatch.setattr(runner, "make_train_step", make_step)
    initial_path = runner.run_training(V24IlyaTrainInvocation(cfg, tmp_path/"initial.json"))
    initial = load_v24_Ilya_training_checkpoint(initial_path)
    assert initial.completed_step == 3 and np.any(initial.residual_state["s"])
    main_cfg = replace(cfg, run_name="main", max_steps=6, learning_rate_schedule="cosine",
                       end_learning_rate=1e-5, mamba_lr_multiplier=3.)
    invocation = V24IlyaTrainInvocation(main_cfg, tmp_path/"main.json", init_from=initial_path,
                                       zero_state_on_init=True, stop_after_steps=3)
    partial_path = runner.run_training(invocation)
    start = load_v24_Ilya_training_checkpoint(main_cfg.run_dir/"checkpoints/checkpoint_step00000000.pkl")
    initial_zero = load_v24_Ilya_training_checkpoint(cfg.run_dir/"checkpoints/checkpoint_step00000000.pkl")
    for a, b in zip(jax.tree_util.tree_leaves(start.residual_params),
                    jax.tree_util.tree_leaves(initial.residual_params)):
        np.testing.assert_array_equal(a, b)
    assert not np.any(start.residual_state["s"])
    assert start.training_cursor == initial_zero.training_cursor
    np.testing.assert_array_equal(start.rng_key, initial_zero.rng_key)
    expected_optimizer = build_optimizer(main_cfg)[0].init(start.residual_params)
    for a, b in zip(jax.tree_util.tree_leaves(start.optimizer_state),
                    jax.tree_util.tree_leaves(expected_optimizer)):
        np.testing.assert_array_equal(a, b)
    partial = load_v24_Ilya_training_checkpoint(partial_path)
    assert partial.training_cursor["segment_offset"] != 0
    resumed_path = runner.run_training(replace(invocation, init_from=None, resume=partial_path,
                                               stop_after_steps=None, zero_state_on_init=False))
    resumed = load_v24_Ilya_training_checkpoint(resumed_path)
    uninterrupted_path = runner.run_training(replace(invocation, config=replace(main_cfg, run_name="uninterrupted"),
                                                     stop_after_steps=None))
    uninterrupted = load_v24_Ilya_training_checkpoint(uninterrupted_path)
    for field in ("residual_params", "residual_state", "optimizer_state", "rng_key"):
        for a, b in zip(jax.tree_util.tree_leaves(getattr(resumed, field)),
                        jax.tree_util.tree_leaves(getattr(uninterrupted, field))):
            np.testing.assert_array_equal(a, b)
    assert resumed.training_cursor == uninterrupted.training_cursor
