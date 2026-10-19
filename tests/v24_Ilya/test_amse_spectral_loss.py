"""Mathematical checks for the full-forecast spherical AMSE experiment."""

from __future__ import annotations

from graphcast import losses as gc_losses
from graphcast import xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.training.spectral_loss import (
    _legendre_basis,
    amse_per_degree,
    build_full_field_loss,
    build_spherical_transform,
    latitude_area_weights,
    scheduled_amse_lambda,
)


def _grid(nlat=13, nlon=24):
    return np.linspace(-90, 90, nlat), np.arange(nlon) * 360.0 / nlon


def _power(coefficients):
    factor = np.full(coefficients.shape[-1], 2.0)
    factor[0] = 1
    return np.sum(np.abs(coefficients)**2 * factor, axis=-1)


@pytest.mark.parametrize("descending", [False, True])
def test_analytic_spherical_modes_and_continuous_area_normalization(descending):
    lat, lon = _grid()
    if descending:
        lat = lat[::-1]
    x = np.sin(np.deg2rad(lat))[:, None]
    phi = np.deg2rad(lon)[None, :]
    # Constant, Y10, and real Y11 each have unit spherical mean-square.
    fields = np.stack([
        np.broadcast_to(np.ones_like(x), (len(lat), len(lon))),
        np.broadcast_to(np.sqrt(3) * x, (len(lat), len(lon))),
        np.sqrt(3) * np.sqrt(1 - x*x) * np.cos(phi),
    ])
    transform = build_spherical_transform(lat, lon)
    coefficients = np.asarray(transform.analysis(fields))
    expected = np.zeros_like(coefficients)
    expected[0, 0, 0] = 1
    expected[1, 1, 0] = 1
    expected[2, 1, 1] = -1 / np.sqrt(2)
    np.testing.assert_allclose(coefficients, expected, atol=3e-7)
    np.testing.assert_allclose(_power(coefficients).sum(axis=-1), 1, atol=5e-7)


def test_res1_high_degree_recovery_conditioning_and_unresolved_nyquist():
    lat, lon = _grid(181, 360)
    transform = build_spherical_transform(lat, lon)
    assert transform.lmax == 179
    assert transform.maximum_condition_number < 2.5
    # Independent Legendre-polynomial evaluation at the highest included degree.
    weights = np.zeros(180)
    weights[179] = 1
    latitude_field = np.sqrt(359) * np.polynomial.legendre.legval(
        np.sin(np.deg2rad(lat)), weights)
    field = np.broadcast_to(latitude_field[:, None], (181, 360))
    coefficients = np.asarray(transform.analysis(field)).copy()
    np.testing.assert_allclose(coefficients[179, 0], 1, atol=1e-6)
    coefficients[179, 0] = 0
    assert np.max(np.abs(coefficients)) < 2e-6
    # The objective explicitly excludes this longitude Nyquist component.
    nyquist = np.broadcast_to((-1.0)**np.arange(360), (181, 360))
    np.testing.assert_allclose(transform.analysis(nyquist), 0, atol=1e-6)


def test_bandlimited_reconstruction_parseval_against_gauss_quadrature():
    lat, lon = _grid(21, 40)
    transform = build_spherical_transform(lat, lon, lmax=14)
    rng = np.random.default_rng(124)
    coefficient = (rng.normal(size=(15, 15)) + 1j*rng.normal(size=(15, 15))) / 30
    coefficient[np.triu_indices(15, k=1)] = 0
    coefficient[:, 0] = coefficient[:, 0].real

    def synthesize(latitude, longitude):
        basis = _legendre_basis(latitude, 14)
        phase = np.exp(1j*np.deg2rad(longitude[:, None])*np.arange(15)[None, :])
        factor = np.ones(15) * 2
        factor[0] = 1
        return np.einsum("lm,lmj,im,m->ji", coefficient, basis, phase, factor).real

    field = synthesize(lat, lon)
    recovered = np.asarray(transform.analysis(field))
    np.testing.assert_allclose(recovered, coefficient, atol=1e-7)
    # Independent high-order integration of the reconstructed field, not the
    # discrete cell quadrature used to choose the least-squares projection.
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(40)
    gauss_field = synthesize(np.rad2deg(np.arcsin(nodes)), lon)
    sphere_mean_square = np.sum(gauss_field**2 * quadrature_weights[:, None]) / (2*len(lon))
    np.testing.assert_allclose(_power(recovered).sum(), sphere_mean_square, rtol=1e-6)


def test_amse_matches_equation_six_and_discourages_amplitude_shrinking():
    truth = jnp.asarray([[1.0, 0.0]], dtype=jnp.complex64)
    forecast = jnp.asarray([[0.5, np.sqrt(0.375)]], dtype=jnp.complex64)
    sharp, _ = amse_per_degree(forecast, truth)
    weak, diagnostics = amse_per_degree(forecast * 0.5, truth)
    np.testing.assert_allclose(sharp, [1], atol=2e-6)
    np.testing.assert_allclose(weak, [1.25], atol=2e-6)
    np.testing.assert_allclose(diagnostics["spectral_mse"], [0.75], atol=2e-6)

    def amplitude_loss(scale):
        return amse_per_degree(scale*forecast, truth)[0].sum()

    assert float(jax.grad(amplitude_loss)(0.5)) < 0
    # max(Pf, Py) has a kink at equal powers for imperfect coherence;
    # its one-sided derivatives point toward the minimum from either side.
    assert float(jax.grad(amplitude_loss)(1.01)) > 0


@pytest.mark.parametrize("amplitude", [0.0, 1e-10, 1.0, 10000.0])
def test_identical_and_zero_coefficients_have_finite_exact_zero_loss_and_gradient(amplitude):
    truth = jnp.asarray([[amplitude, amplitude*0.25j]], dtype=jnp.complex64)

    def objective(value):
        return amse_per_degree(value, truth)[0].sum()

    value, gradient = jax.value_and_grad(objective)(truth)
    assert float(value) == 0
    np.testing.assert_array_equal(gradient, 0)
    zero_value, zero_gradient = jax.value_and_grad(objective)(jnp.zeros_like(truth))
    assert np.isfinite(zero_value)
    assert np.all(np.isfinite(zero_gradient))


def _dataset(value, name="temperature", levels=None):
    lat, lon = _grid()
    dims = ("batch", "time", "lat", "lon")
    coords = {"batch": [0, 1], "time": [0], "lat": lat, "lon": lon}
    shape = (2, 1, len(lat), len(lon))
    if levels is not None:
        dims = ("batch", "time", "level", "lat", "lon")
        coords["level"] = levels
        shape = (2, 1, len(levels), len(lat), len(lon))
    values = jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float32), shape)
    return xarray_jax.Dataset({name: (dims, values)}, coords=coords)


@pytest.mark.parametrize("kind", ["mse_fp32", "amse"])
def test_pressure_std_variable_weights_and_batch_preserved(kind):
    levels = [500, 1000]
    truth = _dataset(0, levels=levels)
    prediction = _dataset(np.asarray([2.0, 6.0])[None, None, :, None, None], levels=levels)
    stats = xr.Dataset({"temperature": (("level",), [2.0, 3.0])}, coords={"level": levels})
    score = build_full_field_loss({"diffs_stddev_by_level": stats}, kind,
                                 per_variable_weights={"temperature": 0.25})
    # MSE normalized errors 1,2 with pressure weights 2/3,4/3 gives 3.
    # AMSE vs zero truth tends to 3*MSE because zero-field coherence tends to 0.
    expected = 0.75 if kind == "mse_fp32" else 2.25
    loss, diagnostic = score(prediction, truth)
    assert loss.shape == (2,)
    assert loss.dtype == jnp.float32
    np.testing.assert_allclose(loss, expected, atol=1e-5)
    np.testing.assert_allclose(diagnostic["grid_mse_fp32"], 0.75, atol=1e-6)


def test_fp32_grid_mse_matches_original_graphcast_cell_level_and_variable_weights():
    levels = [500, 1000]
    truth = _dataset(0, name="10m_u_component_of_wind")
    truth = truth.assign(temperature=_dataset(0, levels=levels).temperature)
    rng = np.random.default_rng(37)
    prediction = truth.copy()
    for name in prediction:
        prediction[name] = prediction[name].copy(data=rng.normal(size=prediction[name].shape).astype(np.float32))
    stats = xr.Dataset({"temperature": (("level",), [2.0, 3.0]), "10m_u_component_of_wind": 4.0},
                       coords={"level": levels})
    actual, _ = build_full_field_loss(stats, "mse_fp32")(prediction, truth)
    expected, _ = gc_losses.weighted_mse_per_level(prediction/stats, truth/stats,
                                                 {"10m_u_component_of_wind": 0.1})
    np.testing.assert_allclose(actual, xarray_jax.unwrap_data(expected), rtol=2e-6)
    np.testing.assert_allclose(latitude_area_weights(truth.lat),
                               np.asarray(gc_losses.normalized_latitude_weights(truth))/truth.sizes["lat"],
                               rtol=1e-12)


def test_full_field_jit_gradient_matches_finite_difference_and_senses_phase():
    lat, lon = _grid()
    truth_values = 280 + 2*np.cos(np.deg2rad(lat))[:, None]*np.cos(np.deg2rad(lon))[None, :]
    truth = _dataset(truth_values)
    stats = xr.Dataset({"temperature": 1.0})
    loss = build_full_field_loss(stats, "amse")
    pattern = jnp.asarray(np.cos(np.deg2rad(lat))[:, None]*np.sin(np.deg2rad(lon))[None, :])

    def objective(amount):
        prediction = _dataset(jnp.asarray(truth_values) + amount * pattern)
        return jnp.mean(loss(prediction, truth)[0])

    function = jax.jit(jax.value_and_grad(objective))
    value, gradient = function(jnp.float32(0.5))
    assert np.isfinite(value) and value > 0
    assert np.isfinite(gradient) and gradient > 0
    finite_difference = (objective(0.51) - objective(0.49))/0.02
    np.testing.assert_allclose(gradient, finite_difference, rtol=5e-3)
    equal_value, equal_gradient = function(jnp.float32(0))
    assert float(equal_value) == 0
    np.testing.assert_allclose(equal_gradient, 0, atol=1e-7)


def test_invalid_grid_and_normalization_fail_explicitly():
    lat, lon = _grid()
    with pytest.raises(ValueError, match="lmax"):
        build_spherical_transform(lat, lon, lmax=12)
    with pytest.raises(ValueError, match="complete circle"):
        build_spherical_transform(lat, np.linspace(0, 360, 24))
    with pytest.raises(ValueError, match="both poles"):
        build_spherical_transform(np.linspace(-80, 80, 13), lon)
    with pytest.raises(ValueError, match="standard deviation"):
        build_full_field_loss(xr.Dataset({"temperature": 0}), "amse")(_dataset(1), _dataset(0))


def test_lambda_ramp_is_dynamic_has_explicit_update_semantics_and_resumes():
    ramp = jax.jit(lambda step: scheduled_amse_lambda(.5, 100, step, start=.1))
    indices = [0, 25, 50, 100, 101, 500]
    actual = [float(ramp(jnp.int32(step))) for step in indices]
    np.testing.assert_allclose(actual, [.1, .2, .3, .5, .5, .5], atol=2e-8)
    # A resumed run uses the saved completed-update count, not a new local zero.
    resumed = jax.jit(lambda step: scheduled_amse_lambda(.5, 100, step, start=.1))
    np.testing.assert_array_equal(ramp(jnp.int32(73)), resumed(jnp.int32(73)))
    assert float(scheduled_amse_lambda(.5, 0, None)) == .5
    with pytest.raises(ValueError, match="explicit optimizer_step"):
        scheduled_amse_lambda(.5, 100, None)
    with pytest.raises(ValueError, match="scalar integer"):
        ramp(jnp.float32(2.5))
    with pytest.raises(ValueError, match="non-negative"):
        scheduled_amse_lambda(.5, 100, -1)


def test_jitted_dynamic_lambda_zero_matches_spectral_mse_value_and_gradient():
    truth = jnp.asarray([[1.0, .2j], [.3, .1-.2j]], dtype=jnp.complex64)
    prediction = jnp.asarray([[.45, .15+.5j], [.1, -.2j]], dtype=jnp.complex64)

    @jax.jit
    def value_and_gradient(pred, coefficient):
        return jax.value_and_grad(
            lambda p: amse_per_degree(p, truth, amse_lambda=coefficient)[0].sum()
        )(pred)

    zero_value, zero_gradient = value_and_gradient(prediction, jnp.float32(0))
    full_value, full_gradient = value_and_gradient(prediction, jnp.float32(1))
    mid_value, mid_gradient = value_and_gradient(prediction, jnp.float32(.25))
    expected_value, expected_gradient = jax.value_and_grad(
        lambda p: jnp.sum(jnp.abs(p-truth)**2 * jnp.asarray([1., 2.]))
    )(prediction)
    np.testing.assert_allclose(zero_value, expected_value, atol=3e-7)
    np.testing.assert_allclose(zero_gradient, expected_gradient, atol=5e-7)
    np.testing.assert_allclose(mid_value, .75*zero_value+.25*full_value, atol=3e-7)
    np.testing.assert_allclose(mid_gradient, .75*zero_gradient+.25*full_gradient, atol=5e-7)
    for coefficient in (0., .25, 1.):
        value, gradient = value_and_gradient(jnp.zeros_like(prediction), jnp.float32(coefficient))
        assert np.isfinite(value) and np.isfinite(gradient).all()


def test_variable_lambda_multipliers_preserve_loss_and_normalization_for_zero_weight():
    lat, lon = _grid()
    envelope = np.cos(np.deg2rad(lat))[:, None]
    truth_field = envelope * np.cos(np.deg2rad(lon))[None, :]
    forecast_field = .5 * envelope * np.sin(np.deg2rad(lon))[None, :]
    truth = xr.merge([_dataset(truth_field), _dataset(2*truth_field, "10m_u_component_of_wind")])
    prediction = xr.merge([_dataset(forecast_field), _dataset(2*forecast_field, "10m_u_component_of_wind")])
    stats = xr.Dataset({"temperature": 2., "10m_u_component_of_wind": 3.})
    score = build_full_field_loss(stats, "amse", amse_lambda=.8,
                                 amse_variable_lambda_multipliers={"temperature": 0., "10m_u_component_of_wind": .5})
    actual, diagnostic = jax.jit(score)(prediction, truth)
    expected = sum(build_full_field_loss(stats, "amse", amse_lambda=coefficient)(
        prediction[[name]], truth[[name]])[0]
        for name, coefficient in (("temperature", 0.), ("10m_u_component_of_wind", .4)))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    grid, _ = build_full_field_loss(stats, "mse_fp32")(prediction, truth)
    np.testing.assert_allclose(diagnostic["grid_mse_fp32"], grid, rtol=2e-6)
    assert np.all(diagnostic["temperature"] > 0)  # Zero multiplier keeps spectral MSE supervision.


def test_degree_multipliers_select_spherical_degrees_without_removing_mse():
    lat, lon = _grid()
    envelope = np.cos(np.deg2rad(lat))[:, None]
    truth = _dataset(envelope * np.cos(np.deg2rad(lon))[None, :])
    prediction = _dataset(.5 * envelope * np.sin(np.deg2rad(lon))[None, :])
    stats = xr.Dataset({"temperature": 1.})
    # Both fields are pure total degree 1, verified independently by the analytic tests above.
    weights = np.ones(12)
    weights[1] = 0
    adjusted, _ = build_full_field_loss(stats, "amse", amse_degree_lambda_multipliers=tuple(weights))(
        prediction, truth)
    mse, _ = build_full_field_loss(stats, "amse", amse_lambda=0)(prediction, truth)
    np.testing.assert_allclose(adjusted, mse, atol=3e-7)
    assert np.all(adjusted > 0)
    weights[1], weights[2] = 1, 0
    unchanged, _ = build_full_field_loss(stats, "amse", amse_degree_lambda_multipliers=tuple(weights))(
        prediction, truth)
    full, _ = build_full_field_loss(stats, "amse")(prediction, truth)
    np.testing.assert_allclose(unchanged, full, atol=3e-7)


def test_full_field_lambda_schedule_dynamic_gradient_and_missing_counter():
    lat, lon = _grid()
    envelope = np.cos(np.deg2rad(lat))[:, None]
    truth = _dataset(envelope * np.cos(np.deg2rad(lon))[None, :])
    pattern = envelope * np.sin(np.deg2rad(lon))[None, :]
    score = build_full_field_loss(xr.Dataset({"temperature": 1.}), "amse",
                                 amse_lambda=.5, amse_lambda_ramp_steps=100)

    @jax.jit
    def evaluate(amplitude, step):
        return jax.value_and_grad(lambda a: score(_dataset(a*pattern), truth,
                                                optimizer_step=step)[0].mean())(amplitude)

    for step, coefficient in ((0, 0.), (50, .25), (100, .5), (125, .5)):
        value, gradient = evaluate(jnp.float32(.5), jnp.int32(step))
        expected = build_full_field_loss(xr.Dataset({"temperature": 1.}), "amse", amse_lambda=coefficient)
        reference_value, reference_gradient = jax.value_and_grad(
            lambda a: expected(_dataset(a*pattern), truth)[0].mean())(jnp.float32(.5))
        np.testing.assert_allclose(value, reference_value, atol=5e-7)
        np.testing.assert_allclose(gradient, reference_gradient, atol=5e-7)
        assert np.isfinite(gradient)
    with pytest.raises(ValueError, match="explicit optimizer_step"):
        score(truth, truth)


def test_lambda_multiplier_names_and_degree_count_fail_explicitly():
    stats = xr.Dataset({"temperature": 1.})
    with pytest.raises(ValueError, match="unknown output variables"):
        build_full_field_loss(stats, "amse", amse_variable_lambda_multipliers={"typo": .5})(
            _dataset(1), _dataset(0))
    with pytest.raises(ValueError, match="lmax\\+1"):
        build_full_field_loss(stats, "amse", amse_degree_lambda_multipliers=(1., 1.))(
            _dataset(1), _dataset(0))
    for value in (-1., 1.1, np.nan):
        with pytest.raises(ValueError, match="multipliers"):
            build_full_field_loss(stats, "amse", amse_variable_lambda_multipliers={"temperature": value})
