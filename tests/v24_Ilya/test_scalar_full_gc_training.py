"""Scalar objective, full-GC gradient routing, and native continuation checks."""
from dataclasses import replace
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

from graphcast import losses as gc_losses, xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

from src.models.mamba.v24_Ilya.checkpoint import (
    load_v24_Ilya_checkpoint, load_v24_Ilya_training_checkpoint,
    training_checkpoint_payload,
)
from src.models.mamba.v24_Ilya.metrics import ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS
from src.models.mamba.v24_Ilya.training import runner, spectral_loss
from src.models.mamba.v24_Ilya.training.config import (
    load_training_config, parse_cli, validate_resume_config, V24IlyaTrainInvocation,
)
from src.models.mamba.v24_Ilya.training.endpoint_step import (
    V24IlyaTrainingTransforms, build_optimizer, make_bptt_objective, make_train_step,
)
from src.models.mamba.v24_Ilya.training.scalar_excess_loss import build_gc_mse_amse_excess_loss
from src.models.mamba.v24_Ilya.training.weather_feedback_step import ScalarTrainResult
from test_amse_split_mgda_core import fields83
from test_weather_feedback_step import (
    arguments, config as weather_config, fields, assert_close, ForbiddenResidualLoss,
)


def scalar_config(**changes):
    return replace(weather_config(4, "all_steps", batch=4, prefix=2),
                   loss_kind="gc_mse_amse_excess", spectral_excess_weight=.5,
                   **changes)


@pytest.mark.parametrize("coefficient", [0., .25, .5, .75, 1.])
def test_scalar_83_channel_loss_and_gradient_match_exact_grid_plus_excess(coefficient):
    truth, stats = fields83()
    amse = spectral_loss.build_full_field_loss(stats, "amse", lmax=5, amse_lambda=1.)
    objective = build_gc_mse_amse_excess_loss(stats, spectral_excess_weight=coefficient, lmax=5)

    def expected(scale):
        prediction = truth * scale + .1
        exact, _ = gc_losses.weighted_mse_per_level(
            prediction / stats, truth / stats, ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS)
        _, diagnostic = amse(prediction, truth)
        return xarray_jax.unwrap_data(exact) + coefficient * diagnostic["amse_excess"]

    actual, diagnostic = objective(truth * .73 + .1, truth)
    np.testing.assert_allclose(actual, expected(.73), rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(
        jax.grad(lambda s: objective(truth * s + .1, truth)[0].sum())(.73),
        jax.grad(lambda s: expected(s).sum())(.73), rtol=3e-6, atol=1e-6)
    np.testing.assert_allclose(diagnostic["scalar_total"], actual)
    if coefficient == 1.:
        total_amse, terms = amse(truth * .73 + .1, truth)
        np.testing.assert_allclose(actual - total_amse,
                                   terms["grid_mse_fp32"] - terms["spectral_mse"], atol=2e-6)
        assert np.max(np.abs(np.asarray(actual - total_amse))) > .01


def test_lambda_zero_never_constructs_spherical_transform(monkeypatch):
    truth, stats = fields83()
    def forbidden(*args, **kwargs):
        raise AssertionError("Zero-lambda grid objective must not construct SHT")
    monkeypatch.setattr(spectral_loss, "build_spherical_transform", forbidden)
    objective = build_gc_mse_amse_excess_loss(stats, spectral_excess_weight=0, lmax=5)
    value, diagnostic = jax.jit(objective)(truth * .7, truth)
    assert "spectral_mse" not in diagnostic and "amse_excess" not in diagnostic
    np.testing.assert_array_equal(value, diagnostic["grid_mse_fp32"])
    np.testing.assert_array_equal(diagnostic["weighted_amse_excess"], 0.)


def test_scalar_config_and_cli_roundtrip_reject_ambiguous_objectives(tmp_path):
    cfg = scalar_config(model_family="full_gc", train_order_manifest=tmp_path / "order.json")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert load_training_config(path) == cfg
    invocation = parse_cli(["--config", str(path), "--train-order-manifest", str(tmp_path / "new.json"),
                            "--stop-after-steps", "2"])
    assert invocation.config.train_order_manifest == tmp_path / "new.json"
    assert invocation.config.max_steps == cfg.max_steps and invocation.stop_after_steps == 2
    for change in ({"amse_lambda": .5}, {"amse_lambda_ramp_steps": 3},
                   {"spectral_excess_weight": float("nan")}, {"spectral_excess_weight": 1.1},
                   {"objective_pair": "mse_amse_excess"}, {"optimizer_kind": "mgda_rms"},
                   {"feedback_mode": "closed_loop_sg"}, {"model_family": "lora"},
                   {"mamba_lr_multiplier": 3.}):
        with pytest.raises(ValueError):
            replace(cfg, **change)
    validate_resume_config(cfg, cfg.to_dict(), completed_step=2)
    with pytest.raises(ValueError, match="max_steps fixed"):
        validate_resume_config(replace(cfg, max_steps=cfg.max_steps + 1), cfg.to_dict(), completed_step=2)
    with pytest.raises(ValueError, match="differs"):
        validate_resume_config(replace(cfg, model_family="gc_mamba"), cfg.to_dict(), completed_step=2)


class FullForecast:
    def apply(self, params, state, key, inputs, template, forcing):
        previous, current = fields(inputs)
        prediction = params["gain"] * (.3 * previous + .7 * current) + params["offset"]
        return jax.tree_util.tree_map(lambda v: jnp.broadcast_to(prediction[:, None, :], v.shape), template), state


class ForbiddenBaseline:
    def apply(self, *args):
        raise AssertionError("Full GC must not add or execute a second baseline")


def toy_scalar_loss(weight):
    def loss(prediction, truth):
        pred, true = (xarray_jax.unwrap_data(x["x"]) for x in (prediction, truth))
        grid = jnp.mean((pred - true) ** 2, axis=(1, 2))
        excess = jnp.mean((pred + true) ** 2, axis=(1, 2))
        total = grid + weight * excess
        diagnostics = dict(grid_mse_fp32=grid, weighted_amse_excess=weight * excess, scalar_total=total)
        if weight:
            diagnostics.update(spectral_mse=grid * .9, amse_excess=excess)
        return total, diagnostics
    return loss


@pytest.mark.parametrize("weight", [0., .5])
def test_full_gc_streamed_parameter_gradients_and_diagnostics(weight):
    cfg = replace(scalar_config(model_family="full_gc"), spectral_excess_weight=weight)
    predictor = FullForecast()
    forbidden = ForbiddenResidualLoss()
    transforms = V24IlyaTrainingTransforms(ForbiddenBaseline(), predictor, forbidden, forbidden,
                                          toy_scalar_loss(weight), predictor)
    params = {"gain": jnp.array(.78), "offset": jnp.array(.04)}
    args = arguments(cfg)
    kwargs = dict(transforms=transforms, baseline_params={"unused": jnp.array(1000.)}, baseline_state={},
                  config=cfg, time_step=pd.Timedelta("6h"), input_steps=2)
    objective = make_bptt_objective(**kwargs)

    # Independent explicit recurrence: the predictor is the whole forecast.
    def reference(p):
        weather = [xarray_jax.unwrap_data(frame["x"])[:, 0] for frame in args[1]]
        values = []
        for index in range(cfg.bptt_steps):
            prediction = p["gain"] * (.3 * weather[index] + .7 * weather[index + 1]) + p["offset"]
            target = xarray_jax.unwrap_data(args[3][index]["x"])[:, 0]
            values.append(jnp.mean((prediction-target)**2 + weight*(prediction+target)**2))
            if cfg.truth_prefix_steps - 1 <= index < cfg.bptt_steps - 1:
                weather.append(prediction)
        return jnp.dot(jnp.stack(values), jnp.asarray(cfg.normalized_supervised_weights))

    expected_value, expected_gradient = jax.value_and_grad(reference)(params)
    assert_close(objective(params, {}, *args)[0], expected_value)
    assert_close(jax.grad(lambda p: objective(p, {}, *args)[0])(params), expected_gradient)
    assert all(abs(float(value)) > .01 for value in expected_gradient.values())
    optimizer = optax.sgd(.01)
    result = make_train_step(**kwargs, optimizer=optimizer)(params, {}, optimizer.init(params), *args)
    assert isinstance(result, ScalarTrainResult) and result.state == {}
    assert_close(result.params, jax.tree_util.tree_map(lambda p, g: p - .01 * g, params, expected_gradient))
    assert_close(result.loss, expected_value)
    assert_close(result.diagnostics["scalar_total"], expected_value)
    assert_close(result.diagnostics["update_norm"], .01 * optax.global_norm(expected_gradient))
    if not weight:
        assert result.diagnostics["spectral_mse"] is None
        assert result.diagnostics["amse_excess"] is None
        assert result.diagnostics["spectral_diagnostics_computed"] is False


@pytest.mark.parametrize("family", ["gc_mamba", "full_gc"])
def test_native_family_roundtrip_optimizer_resume_and_mismatch(tmp_path, family):
    cfg = scalar_config(model_family=family)
    params = {"graphcast": {"w": jnp.array([.2, -.1], jnp.float32)}}
    optimizer, _ = build_optimizer(cfg)
    gradients = jax.tree_util.tree_map(lambda p: 2*p, params)
    updates, state = optimizer.update(gradients, optimizer.init(params), params)
    params = optax.apply_updates(params, updates)
    payload = training_checkpoint_payload(completed_step=1, residual_params=params,
        residual_state={}, optimizer_state=state, rng_key=jax.random.PRNGKey(22),
        training_cursor={"epoch": 0, "segment_index": 0, "segment_offset": 4},
        resolved_training_config=cfg.to_dict(), baseline_checkpoint_path="public.npz",
        baseline_checkpoint_fingerprint="baseline", anchor_manifest_fingerprint="anchors",
        parameter_overlay_metadata={})
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(pickle.dumps(payload))
    assert load_v24_Ilya_checkpoint(path).model_family == family
    saved = load_v24_Ilya_training_checkpoint(path)
    assert saved.model_family == family and saved.residual_state == {}
    direct = optimizer.update(gradients, state, params)
    resumed = optimizer.update(gradients, saved.optimizer_state, saved.residual_params)
    assert_close(direct, resumed, rtol=0, atol=0)
    payload["model_family"] = "full_gc" if family == "gc_mamba" else "gc_mamba"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="differs"):
        load_v24_Ilya_checkpoint(path)
    with pytest.raises(ValueError, match="differs"):
        load_v24_Ilya_training_checkpoint(path)


@pytest.mark.parametrize("family", ["gc_mamba", "full_gc"])
def test_scalar_runner_step_zero_and_operational_resume_preserve_trajectory(tmp_path, monkeypatch, family):
    from test_mgda_runner import mock_run
    original = mock_run(tmp_path, monkeypatch)
    if family == "full_gc":
        params = {"module": {"w": jnp.array([.2, .4], jnp.float32)}}
        transform = SimpleNamespace(init=lambda *args: (params, {}))
        monkeypatch.setattr(runner, "build_training_transforms", lambda *args: SimpleNamespace(
            residual_loss=transform, residual_predict=transform, baseline_predict=transform))
    manifest = tmp_path / "shared_order.json"
    manifest.write_text('{"order": "fixture"}')
    cfg = replace(original, model_family=family, train_order_manifest=manifest,
                  loss_kind="gc_mse_amse_excess", feedback_mode="closed_loop",
                  optimizer_kind="adamw", power_calibration=None, learning_rate_schedule="cosine",
                  warmup_steps=1, adam_beta2=.98, spectral_excess_weight=0.)
    def make_step(optimizer, **kwargs):
        def step(params, state, optimizer_state, keys, *args):
            gradients = jax.tree_util.tree_map(lambda p: 2*p + .01*jax.random.normal(keys[0], p.shape), params)
            updates, next_optimizer = optimizer.update(gradients, optimizer_state, params)
            return ScalarTrainResult(optax.apply_updates(params, updates), state, next_optimizer,
                jnp.array(1.), optax.global_norm(gradients), jnp.ones(2),
                dict(grid_mse_fp32=1., spectral_mse=None, amse_excess=None,
                     weighted_amse_excess=0., scalar_total=1., update_norm=optax.global_norm(updates)))
        return step
    monkeypatch.setattr(runner, "make_train_step", make_step)
    invocation = V24IlyaTrainInvocation(cfg, tmp_path / "config.json", stop_after_steps=3)
    partial = runner.run_training(invocation)
    initial = load_v24_Ilya_training_checkpoint(cfg.run_dir / "checkpoints/checkpoint_step00000000.pkl")
    assert initial.completed_step == 0 and initial.model_family == family
    if family == "full_gc":
        assert initial.residual_state == {}
        assert initial.parameter_overlay_metadata["residual_initialization"] == "complete_public_graphcast"
    assert initial.resolved_training_config["optimizer"]["max_steps"] == 6
    final = runner.run_training(replace(invocation, resume=partial, stop_after_steps=None))
    resumed = load_v24_Ilya_training_checkpoint(final)
    # Restart exactly from saved zero rather than inferring a new warm-start.
    final = runner.run_training(replace(invocation, resume=cfg.run_dir / "checkpoints/checkpoint_step00000000.pkl",
                                        stop_after_steps=None))
    repeated = load_v24_Ilya_training_checkpoint(final)
    assert_close((resumed.residual_params, resumed.optimizer_state, resumed.rng_key),
                 (repeated.residual_params, repeated.optimizer_state, repeated.rng_key), rtol=0, atol=0)
    assert repeated.training_cursor == resumed.training_cursor
    records = [json.loads(line) for line in (cfg.run_dir / "train_metrics.jsonl").read_text().splitlines()]
    assert records[-1]["step"] == 6 and records[-1]["amse_excess"] is None
    manifest.write_text('{"order": "modified"}')
    with pytest.raises(ValueError, match="order manifest fingerprint"):
        runner.run_training(replace(invocation, resume=partial))


def test_real_full_gc_transform_uses_public_tree_and_differentiates_spatial_weights():
    """Exercise the actual GNN/normalizer with trainable full-GC parameters."""
    import xarray as xr
    from graphcast import graphcast as gc
    from src.models.mamba.v24_Ilya.model import V24IlyaModelConfigs
    from src.models.mamba.v24_Ilya.training.endpoint_step import build_training_transforms
    from src.models.mamba.v24_Ilya.checkpoint import overlay_frozen_baseline_params

    variable, force_name = "2m_temperature", "toa_incident_solar_radiation"
    variables = (variable, "10m_u_component_of_wind", "10m_v_component_of_wind",
                 "mean_sea_level_pressure", "total_precipitation_6hr")
    grid = dict(lat=np.arange(-90, 91, 30, dtype=np.float32), lon=np.arange(0, 360, 30, dtype=np.float32))
    spatial = np.cos(np.deg2rad(grid["lat"]))[:, None] + .5*np.sin(np.deg2rad(grid["lon"]))[None, :]
    def weather(offset, hour):
        return xr.Dataset({name: (("batch", "time", "lat", "lon"),
                                  jnp.asarray((280 + spatial + offset)[None, None], jnp.float32))
                           for name in variables},
                          coords={"batch": [0], "time": [np.timedelta64(hour, "h")], **grid})
    def forcing(hour=6):
        return jax.tree_util.tree_map(lambda x: jnp.ones_like(x)*.5, weather(0, hour)[[variable]]).rename({variable: force_name})
    def frame(offset, hour=0):
        return xr.merge((weather(offset, hour), forcing(hour)))
    model = gc.ModelConfig(resolution=30., mesh_size=0, latent_size=4, gnn_msg_steps=1,
                           hidden_layers=1, radius_query_fraction_edge_length=.8)
    task = gc.TaskConfig(input_variables=(*variables, force_name), target_variables=variables,
                         forcing_variables=(force_name,), pressure_levels=(), input_duration="12h")
    stats = {name: xr.Dataset({**{field: xr.DataArray(np.float32(value)) for field in variables},
                               force_name: xr.DataArray(np.float32(0. if name == "mean_by_level" else 1.))})
             for name, value in (("mean_by_level", 280.), ("stddev_by_level", 10.), ("diffs_stddev_by_level", 1.))}
    cfg = replace(weather_config(2, "last_step"), model_family="full_gc",
                  loss_kind="gc_mse_amse_excess", spectral_excess_weight=0.)
    transforms = build_training_transforms(V24IlyaModelConfigs(model, model), task, stats, cfg)
    inputs = xr.concat([frame(0., -6), frame(.1)], dim="time")
    target, force = weather(.3, 6), forcing()
    inputs, target, force = jax.tree_util.tree_map(jnp.asarray, (inputs, target, force))
    key = jax.random.PRNGKey(5)
    baseline_params, baseline_state = transforms.baseline_predict.init(key, inputs, target, force)
    params, state = transforms.residual_loss.init(key, inputs, target, force)
    params, audit = overlay_frozen_baseline_params(params, baseline_params)
    assert not state and not baseline_state and audit.copied == len(jax.tree_util.tree_leaves(params))
    full, _ = transforms.residual_supervised_predict.apply(params, {}, key, inputs, target, force)
    public, _ = transforms.baseline_predict.apply(baseline_params, {}, key, inputs, target, force)
    assert_close(full, public, rtol=0, atol=0)
    args = (jax.random.split(key, 2), (frame(0.), frame(.1)), xr.Dataset(),
            (weather(.4, 6),), (forcing(), forcing()))
    kwargs = dict(transforms=transforms, baseline_params=baseline_params, baseline_state={},
                  config=cfg, time_step=pd.Timedelta("6h"), input_steps=2)
    objective = make_bptt_objective(**kwargs)
    value, gradient = jax.value_and_grad(lambda p: objective(p, {}, *args)[0])(params)
    assert np.isfinite(value)
    assert jax.tree_util.tree_structure(gradient) == jax.tree_util.tree_structure(params)
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(gradient))
    # Gradients reach the spatial backbone as well as the output projection.
    for block in ("grid2mesh", "mesh_gnn", "mesh2grid"):
        leaves = [value for module, values in gradient.items() if block in module for value in values.values()]
        assert leaves and float(optax.global_norm(leaves)) > 1e-8, block
    optimizer = optax.sgd(.001)
    result = make_train_step(**kwargs, optimizer=optimizer)(params, {}, optimizer.init(params), *args)
    assert_close(result.params, jax.tree_util.tree_map(lambda p, g: p - .001*g, params, gradient),
                 rtol=5e-5, atol=3e-6)
