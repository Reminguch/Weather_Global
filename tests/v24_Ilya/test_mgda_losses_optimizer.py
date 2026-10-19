"""Scientific and configuration checks for the shared-metric MGDA pilot."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from graphcast import xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.spectral_diagnostics import zonal_spectral_statistics
from src.models.mamba.v24_Ilya.spectral_observer import FIELD_SPECS
from src.models.mamba.v24_Ilya.training.config import (
    V24IlyaTrainConfig, load_training_config, parse_cli, validate_resume_config,
)
from src.models.mamba.v24_Ilya.training.mgda_optimizer import (
    RMSOptimizer, metric_mgda_weight, validate_optimizer_diagnostics,
    warmup_cosine_constant_schedule,
)
from src.models.mamba.v24_Ilya.training.power_loss import (
    POWER_FIELDS, build_power_loss, calibration_fingerprint, calibration_floors,
    make_power_calibration, power_surrogate, truth_powers, zonal_power,
)
from src.models.mamba.v24_Ilya.training.spectral_loss import build_full_field_loss


def _calibration(nk=12):
    return make_power_calibration(np.ones((9, nk)), 2.0, 3.0)


def _config(**changes):
    values = dict(prepared_root=Path("prepared"), anchor_manifest_root=Path("anchors"),
                  baseline_checkpoint=Path("baseline"), output_root=Path("output"),
                  run_name="test", architecture=V24IlyaArchitectureConfig(),
                  feedback_mode="closed_loop_sg", loss_kind="mse_fp32", loss_mode="all_steps",
                  optimizer_kind="mgda_rms", power_calibration=_calibration(),
                  learning_rate=1e-5, end_learning_rate=3e-6, warmup_steps=100,
                  warmup_start_learning_rate=1e-6, decay_end_step=1000,
                  learning_rate_schedule="warmup_cosine_constant", max_steps=2000,
                  weight_decay=0, mamba_lr_multiplier=3)
    values.update(changes)
    return V24IlyaTrainConfig(**values)


@pytest.mark.parametrize("nlon", [23, 24, 360])
@pytest.mark.parametrize("descending", [False, True])
def test_zonal_power_matches_numpy_diagnostic_and_latitude_boundaries(nlon, descending):
    lat = np.array([-90, -60, -45, -30, -29, 0, 29, 30, 45, 60, 90.])
    if descending:
        lat = lat[::-1]
    lon = np.arange(nlon) * 360.0 / nlon
    rng = np.random.default_rng(42)
    fields = rng.normal(size=(3, len(lat), nlon)).astype(np.float32)
    # Huge values outside the selected rows must not affect the objective.
    fields[:, np.abs(lat) < 30] += 500
    actual = np.asarray(zonal_power(fields, lat, lon))
    expected = zonal_spectral_statistics(fields, fields, lat)["power_truth"][:, 1:]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    grad = jax.grad(lambda x: zonal_power(x, lat, lon).sum())(jnp.asarray(fields))
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(np.asarray(grad)[:, np.abs(lat) < 30], 0)


def test_known_modes_phase_invariance_nyquist_and_physical_units():
    lat = np.array([-60, -45, -30, 0, 30, 45, 60.])
    lon = np.arange(24) * 15.
    phase = 2 * np.pi * np.arange(24) / 24
    fields = np.broadcast_to(np.stack([3 * np.cos(2 * phase),
                                      3 * np.cos(2 * phase + 0.7),
                                      3 * (-1.) ** np.arange(24)])[:, None], (3, len(lat), 24))
    power = np.asarray(zonal_power(fields, lat, lon))
    np.testing.assert_allclose(power[:2, 1], 4.5, atol=1e-6)
    np.testing.assert_allclose(power[2, -1], 9, atol=1e-6)
    np.testing.assert_allclose(zonal_power(fields * 1000, lat, lon), power * 1e6, rtol=1e-6, atol=1)


def _fields(batch=2):
    lat = np.linspace(-90, 90, 13)
    lon = np.arange(24) * 15.
    levels = [500, 700, 850]
    rng = np.random.default_rng(22)
    variables = {}
    for _, name, level, _ in FIELD_SPECS:
        dims = ("batch", "time", "lat", "lon") if level is None else ("batch", "time", "level", "lat", "lon")
        shape = (batch, 1, 13, 24) if level is None else (batch, 1, 3, 13, 24)
        variables[name] = (dims, jnp.asarray(rng.normal(size=shape), dtype=jnp.float32))
    return xr.Dataset(variables, coords={"batch": np.arange(batch), "time": [0],
                                        "level": levels, "lat": lat, "lon": lon})


def test_nine_exact_pressure_fields_per_batch_loss_and_gradient():
    observed = _fields()
    powers = truth_powers(observed)
    assert powers.shape == (2, 1, 9, 12)
    calibration = make_power_calibration(np.mean(powers, axis=(0, 1)), 2, 3)
    loss = build_power_loss(calibration)
    value, diagnostics = loss(observed, observed)
    np.testing.assert_allclose(value, 0, atol=1e-9)
    assert diagnostics["floor_fraction"].shape == (2,)
    assert np.isfinite(jax.grad(lambda scale: loss(observed * scale, observed)[0].sum())(0.8))
    # Changing the unused 500-hPa plane cannot affect the nine-field objective.
    modified = observed.copy()
    arr = xarray_jax.unwrap_data(modified["temperature"])
    modified["temperature"] = modified["temperature"].copy(data=arr.at[:, :, 0].set(1e6))
    np.testing.assert_allclose(truth_powers(modified), powers)
    with pytest.raises(KeyError):
        truth_powers(observed.sel(level=[500, 850]))


def test_per_example_surrogate_is_not_error_after_pooled_powers():
    truth = jnp.ones((2, 1, 9, 12))
    prediction = truth.at[0].set(0).at[1].set(2)
    floors = np.ones((9, 12)) * 1e-3
    assert float(power_surrogate(prediction, truth, floors).mean()) > 0.99
    assert float(power_surrogate(prediction.mean(axis=0), truth.mean(axis=0), floors).mean()) == 0


def test_calibration_is_frozen_deterministic_and_rejects_bad_values():
    means = np.arange(1, 109).reshape(9, 12)
    first = make_power_calibration(means, 2, 3, batch_ids=list(range(32)))
    np.testing.assert_allclose(first["floors"], means * 1e-3)
    second = json.loads(json.dumps(first))
    assert calibration_fingerprint(first) == calibration_fingerprint(second)
    second["gc_scale"] = 4
    assert calibration_fingerprint(first) != calibration_fingerprint(second)
    for value in (0, -1, np.nan, np.inf):
        bad = np.ones((9, 12)); bad[0, 0] = value
        with pytest.raises(ValueError):
            calibration_floors(bad)
        with pytest.raises(ValueError):
            make_power_calibration(means, value, 3)


@pytest.mark.parametrize("first,second,expected", [
    ([1., 0], [2., 0], 1), ([1., 0], [1., 0], .5),
    ([1., 0], [-1., 0], .5), ([0., 0], [1., 0], 1),
    ([1., 0], [0., 0], 0), ([0., 0], [0., 0], .5),
    ([1., 1], [-1., 1], .5),
])
def test_mgda_closed_form_cases(first, second, expected):
    g1, g2 = jnp.array(first), jnp.array(second)
    alpha, _ = metric_mgda_weight(g1, g2, jnp.ones_like(g1))
    assert float(alpha) == pytest.approx(expected)
    direction = alpha * g1 + (1 - alpha) * g2
    assert float(g1 @ direction) >= -1e-7
    assert float(g2 @ direction) >= -1e-7


def test_anisotropic_metric_solution_matches_brute_force_search():
    first, second, metric = jnp.array([1., -2]), jnp.array([-2., 1]), jnp.array([1., 9])
    alpha, _ = metric_mgda_weight(first, second, metric)
    grid = np.linspace(0, 1, 10001)
    combination = grid[:, None] * first + (1 - grid[:, None]) * second
    optimum = grid[np.argmin(np.sum(combination ** 2 * metric, axis=1))]
    assert float(alpha) == pytest.approx(optimum, abs=1e-4)


def test_rms_bias_correction_group_multiplier_clip_and_no_hidden_decay():
    params = {"spatial": {"w": jnp.array([10.])}, "temporal_mesh_mamba_0": {"w": jnp.array([10.])}}
    gradient = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 2, params)
    optimizer = RMSOptimizer(lambda count: jnp.array(.1), clip=1)
    updates, state, diagnostics = jax.jit(optimizer.update_single)(gradient, optimizer.init(params), params)
    validate_optimizer_diagnostics(diagnostics)
    np.testing.assert_allclose(state.second_moment["spatial"]["w"], [.08], rtol=1e-6)
    expected = -.1 / np.sqrt(8)
    np.testing.assert_allclose(updates["spatial"]["w"], [expected], rtol=1e-6)
    np.testing.assert_allclose(updates["temporal_mesh_mamba_0"]["w"], [expected * 3], rtol=1e-6)
    zero = jax.tree_util.tree_map(jnp.zeros_like, gradient)
    update_zero, _, _ = optimizer.update_single(zero, state, params)
    for value in jax.tree_util.tree_leaves(update_zero):
        np.testing.assert_array_equal(value, 0)


def test_shared_squared_gradients_opposition_stall_and_nonfinite_guard():
    optimizer = RMSOptimizer(lambda count: jnp.array(.01))
    params = {"x": jnp.array([0., 0.])}
    gradient = {"x": jnp.array([1., 2.])}
    negative = {"x": -gradient["x"]}
    updates, state, diagnostics = jax.jit(optimizer.update_pair)(gradient, negative, optimizer.init(params), params)
    validate_optimizer_diagnostics(diagnostics)
    np.testing.assert_allclose(state.second_moment["x"], [.02, .08])
    assert bool(diagnostics["stalled"])
    np.testing.assert_array_equal(updates["x"], 0)
    _, _, bad = optimizer.update_single({"x": jnp.array([np.nan, 1])}, state, params)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        validate_optimizer_diagnostics(bad)


def test_actual_parameter_rounding_is_checked_and_resume_replays_exactly():
    optimizer = RMSOptimizer(lambda count: jnp.array(.0001))
    parameters = {"x": jnp.array([1., 2.])}
    gradients = {"x": jnp.array([.3, .5])}
    state = optimizer.init(parameters)
    def step(params, current):
        updates, following, diagnostics = optimizer.update_single(gradients, current, params)
        validate_optimizer_diagnostics(diagnostics)
        next_params = optax.apply_updates(params, updates)
        actual_dot = np.sum(np.asarray(gradients["x"]) * np.asarray(params["x"] - next_params["x"]))
        assert float(diagnostics["gc_displacement_dot"]) == pytest.approx(actual_dot)
        return next_params, following
    parameters, state = step(parameters, state)
    saved = jax.tree_util.tree_map(lambda x: np.array(x), (parameters, state))
    expected = step(parameters, state)
    resumed = step(*jax.tree_util.tree_map(jnp.asarray, saved))
    for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(a, b)


def test_rounding_can_break_common_descent_and_guard_rejects_it():
    # The shared-metric direction improves both, but at 1000 the first FP32
    # coordinate cannot represent its tiny displacement. The other coordinate
    # then harms objective 2, demonstrating why checking only the direction is
    # insufficient.
    optimizer = RMSOptimizer(lambda count: jnp.array(.001))
    params = {"x": jnp.array([1000., 0.])}
    first, second = {"x": jnp.array([1., 1.])}, {"x": jnp.array([-1., -.99])}
    _, _, diagnostics = optimizer.update_pair(first, second, optimizer.init(params), params)
    assert float(diagnostics["gc_direction_dot"]) > 0
    assert float(diagnostics["power_direction_dot"]) > 0
    assert float(diagnostics["power_displacement_dot"]) < 0
    with pytest.raises(FloatingPointError, match="violates common descent"):
        validate_optimizer_diagnostics(diagnostics)


def test_immutable_schedule_boundaries_smoke_and_config_roundtrip(tmp_path):
    config = _config()
    schedule = warmup_cosine_constant_schedule(config)
    expected = {1: 1e-6, 100: 1e-5, 1000: 3e-6, 1001: 3e-6, 2000: 3e-6}
    for update, value in expected.items():
        assert float(schedule(update - 1)) == pytest.approx(value, rel=1e-6)
    assert 3e-6 < float(schedule(100)) < 1e-5
    smoke_schedule = warmup_cosine_constant_schedule(replace(config, max_steps=100))
    np.testing.assert_array_equal(jax.vmap(schedule)(jnp.arange(2000)),
                                  jax.vmap(smoke_schedule)(jnp.arange(2000)))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    loaded = load_training_config(path)
    assert loaded.to_dict() == config.to_dict()
    parsed = parse_cli(["--config", str(path), "--stop-after-steps", "50"])
    assert parsed.stop_after_steps == 50 and parsed.config.max_steps == 2000
    validate_resume_config(config, config.to_dict(), completed_step=100)
    for change in (dict(decay_end_step=1200), dict(rms_decay=.95), dict(optimizer_kind="rms_mse"),
                   dict(power_calibration=make_power_calibration(np.ones((9, 12)), 3, 3))):
        with pytest.raises(ValueError, match="differs"):
            validate_resume_config(replace(config, **change), config.to_dict(), completed_step=100)


@pytest.mark.parametrize("change", [dict(power_calibration=None), dict(weight_decay=.01),
                                    dict(feedback_mode="baseline"), dict(loss_kind="mse"),
                                    dict(spatial_lr_multiplier=0)])
def test_unsupported_rms_config_rejected(change):
    with pytest.raises(ValueError):
        _config(**change)


def test_legacy_serialization_unchanged_by_new_defaults():
    config = _config(optimizer_kind="adamw", power_calibration=None,
                     learning_rate_schedule="constant", end_learning_rate=None)
    assert "optimizer_kind" not in config.to_dict()["optimizer"]
    assert "power_calibration" not in config.to_dict()["objective"]
    assert "rms_decay" not in config.to_dict()["optimizer"]


def test_real_gc_and_power_pair_batch_shapes_and_single_fixed_normalization():
    observed = _fields(batch=4)
    stats = xr.Dataset({name: ("level", np.ones(3)) if "level" in value.dims else 1.
                        for name, value in observed.data_vars.items()}, coords={"level": observed.level})
    gc = build_full_field_loss(stats, "mse_fp32")
    calibration = make_power_calibration(np.asarray(truth_powers(observed)).mean(axis=(0, 1)), 2., 7.)
    power = build_power_loss(calibration)
    def pair(parameters):
        prediction = observed * parameters[0] + observed.roll(lon=1) * parameters[1]
        gc_values, _ = gc(prediction, observed)
        power_values, _ = power(prediction, observed)
        assert gc_values.shape == power_values.shape == (4,)
        return jnp.stack((gc_values, power_values)).mean(axis=1)
    parameters = jnp.asarray([.8, .05])
    _, pullback = jax.vjp(pair, parameters)
    for objective, scale in enumerate((2., 7.)):
        cotangent = jnp.zeros(2).at[objective].set(1 / scale)
        expected = jax.grad(lambda p: pair(p)[objective] / scale)(parameters)
        np.testing.assert_allclose(pullback(cotangent)[0], expected, rtol=2e-6)


def test_native_checkpoint_preserves_rms_state_count_and_next_update(tmp_path):
    from src.models.mamba.v24_Ilya.checkpoint import (
        atomic_pickle_dump, load_v24_Ilya_training_checkpoint, training_checkpoint_payload,
    )
    config = _config()
    optimizer = RMSOptimizer(warmup_cosine_constant_schedule(config))
    params = {"spatial": {"w": jnp.asarray([.1, .2])}}
    gradient = {"spatial": {"w": jnp.asarray([.3, .4])}}
    state = optimizer.init(params)
    updates, state, diagnostics = optimizer.update_single(gradient, state, params)
    validate_optimizer_diagnostics(diagnostics)
    params = optax.apply_updates(params, updates)
    payload = training_checkpoint_payload(
        completed_step=1, residual_params=params,
        residual_state={"temporal": {"s": jnp.asarray([0., 0.])}},
        optimizer_state=state, rng_key=jax.random.PRNGKey(22), training_cursor={"segment_offset": 24},
        resolved_training_config=config.to_dict(), baseline_checkpoint_path="baseline",
        baseline_checkpoint_fingerprint="baseline_hash", anchor_manifest_fingerprint="anchor_hash",
        parameter_overlay_metadata={"init_from_sha256": "origin_hash"})
    path = tmp_path / "native.pkl"
    atomic_pickle_dump(jax.device_get(payload), path)
    loaded = load_v24_Ilya_training_checkpoint(path)
    assert loaded.optimizer_state.count == 1
    assert loaded.resolved_training_config["objective"]["power_calibration"] == config.power_calibration
    expected = optimizer.update_single(gradient, state, params)
    actual = optimizer.update_single(gradient, loaded.optimizer_state, loaded.residual_params)
    for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)
