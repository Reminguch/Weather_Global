"""Mathematical checks for the GraphCast-style optimal filtering reference."""
import numpy as np
import pytest

from src.analysis.optimal_spherical_filter import (
    OptimalSphericalFilter, degree_power, fit_degree_gains, spectral_moments,
)


@pytest.fixture(scope="module")
def operator():
    return OptimalSphericalFilter(np.linspace(-90, 90, 9), np.arange(16) * 22.5,
                                  lmax=5)


def test_real_optimum_uses_complex_cross_power_and_negative_order_multiplicity():
    forecast = np.zeros((2, 3, 3), dtype=np.complex128)
    truth = np.zeros_like(forecast)
    forecast[:, 1, 0] = [2, 1]
    forecast[:, 1, 1] = [1 + 3j, 2 - 1j]
    truth[:, 1, 0] = [3, -2]
    truth[:, 1, 1] = [-1 + 2j, 1 + 4j]
    stats = spectral_moments(forecast, truth)
    # Both the imaginary components and the m>0 doubling affect this optimum.
    expected_power = 2**2 + 1**2 + 2 * (1**2 + 3**2 + 2**2 + 1**2)
    expected_cross = 2 * 3 - 2 + 2 * (-1 + 6 + 2 - 4)
    assert stats['prediction_power'][:, 1].sum() == expected_power
    assert stats['cross_power'][:, 1].sum() == expected_cross
    gains, _ = fit_degree_gains(stats['prediction_power'].sum(axis=0),
                               stats['cross_power'].sum(axis=0))
    assert gains[1] == pytest.approx(expected_cross / expected_power)
    for offset in (-1., -.03, .03, 1.):
        trial = gains.copy(); trial[1] += offset
        optimum_error = degree_power(forecast * gains[None, :, None] - truth).sum()
        trial_error = degree_power(forecast * trial[None, :, None] - truth).sum()
        assert trial_error > optimum_error


def test_fits_every_prefix_independently_and_does_not_clip():
    expected = np.array([[[[.5, 1.4, -2.], [3., .2, 20.]],
                          [[1., 4., .7], [-.1, 2., .9]]]])
    powers = np.arange(1, expected.size + 1, dtype=np.float64).reshape(expected.shape)
    gains, info = fit_degree_gains(powers, powers * expected)
    np.testing.assert_allclose(gains, expected)
    assert info['shape'] == list(expected.shape)
    assert info['negative_gain_count'] == 2
    assert info['gain_above_one_count'] == 5
    assert info['absolute_gain_above_ten_count'] == 1
    assert info['zero_power_count'] == 0


def test_zero_power_and_tiny_nonzero_power_are_distinct():
    gains, info = fit_degree_gains([0., 1e-200, 2.], [0., 3e-200, -1.])
    np.testing.assert_allclose(gains, [1., 3., -.5])
    assert info['zero_power_count'] == 1
    assert info['warnings']
    with pytest.raises(ValueError, match="zero forecast power"):
        fit_degree_gains([0.], [1.])
    with pytest.raises(ValueError, match="Negative forecast power"):
        fit_degree_gains([-1.], [0.])
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_degree_gains([1.], [np.nan])


def test_mean_mode_is_fitted_and_projection_discards_unresolved_grid(operator):
    latitude = np.deg2rad(operator.latitude)
    harmonic = np.sin(latitude)[:, None] * np.ones((1, 16))
    residual = .3 * np.cos(latitude)[:, None] * (-1.)**np.arange(16)[None]
    forecast = (100000.001 + 20 * harmonic + residual)[None]
    projected = operator.apply(forecast, np.ones((1, 6)))
    np.testing.assert_allclose(projected, (100000.001 + 20 * harmonic)[None],
                               atol=1e-5, rtol=0)
    assert np.max(np.abs(projected - forecast)) > .2
    gains = np.full((1, 6), .4)
    gains[:, 0] = .9
    filtered = operator.apply(forecast, gains)
    np.testing.assert_allclose(filtered, (90000.0009 + 8 * harmonic)[None],
                               atol=1e-5, rtol=0)
    # FP32 higher-mode transforms are not repaired to force an exact grid mean.
    np.testing.assert_allclose(operator.mean(filtered), .9 * operator.mean(forecast),
                               atol=5e-7, rtol=0)


def test_projection_and_filter_round_trip_resolved_modes(operator):
    coefficients = np.zeros((2, 6, 6), dtype=np.complex128)
    coefficients[:, 0, 0] = [2., 100000.001]
    coefficients[:, 1, 1] = [.7 - .3j, 2 + 1j]
    coefficients[:, 4, 2] = [-.8 + .4j, -.1 + 3j]
    fields = operator.synthesize(coefficients)
    recovered = operator.analyze(fields)
    np.testing.assert_allclose(recovered, coefficients, atol=1e-6, rtol=2e-6)
    gains = np.array([[1.2, -.5, .1, 2, .3, 1], [1., 1.5, 2, -.5, .8, 1]])
    expected = operator.synthesize(coefficients * gains[..., None])
    actual = operator.apply(fields, gains)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-6)


def test_direct_error_power_preserves_small_offset_difference(operator):
    shape = (1, len(operator.latitude), len(operator.longitude))
    truth = operator.analyze(np.full(shape, 100000.))
    forecast = operator.analyze(np.full(shape, 100000.001))
    stats = spectral_moments(forecast, truth)
    assert stats['error_power'][0, 0] == pytest.approx(1e-6, rel=2e-5)
    np.testing.assert_allclose(stats['prediction_power'], degree_power(forecast))


def test_discrete_grid_score_preserves_pole_cell_weights(operator):
    forecast = np.zeros((1, 9, 16))
    forecast[:, [0, -1], :] = 2.
    expected = 4 * operator.area[[0, -1]].sum()
    assert operator.grid_mse(forecast, np.zeros_like(forecast))[0] == pytest.approx(expected)
    assert expected > 0


def test_invalid_fields_and_coefficients_are_errors(operator):
    with pytest.raises(ValueError, match="Nonfinite"):
        operator.analyze(np.full((9, 16), np.nan))
    coefficients = np.zeros((6, 6), dtype=np.complex128)
    coefficients[1, 2] = 1
    with pytest.raises(ValueError, match="order greater than degree"):
        operator.synthesize(coefficients)
    coefficients[1, 2] = 0
    coefficients[1, 0] = 1j
    with pytest.raises(ValueError, match="must be real"):
        operator.synthesize(coefficients)

