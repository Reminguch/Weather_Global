from __future__ import annotations

import json

import numpy as np
import pytest

from src.models.mamba.v24_Ilya.spectral_diagnostics import (
    zonal_latitude_weights,
    zonal_spectral_metadata,
    zonal_spectral_statistics,
)


LATITUDES = np.asarray([-90.0, -60.0, -45.0, -30.0, 0.0, 30.0, 45.0, 60.0, 90.0])


def _wave(*, n_lon: int = 32, mode: int = 3, phase: float = 0.0) -> np.ndarray:
    longitude = np.arange(n_lon) * (2.0 * np.pi / n_lon)
    return np.broadcast_to(np.cos(mode * longitude + phase), (1, LATITUDES.size, n_lon)).copy()


def test_single_mode_has_analytic_power_and_removes_latitude_dependent_mean() -> None:
    signal = np.concatenate([3.0 * _wave(), 5.0 * _wave()], axis=0)
    signal += 270.0 + np.arange(LATITUDES.size)[None, :, None]
    result = zonal_spectral_statistics(signal, signal, LATITUDES)
    expected = np.zeros((2, 17))
    expected[:, 3] = [9.0 / 2, 25.0 / 2]
    np.testing.assert_allclose(result["power_truth"], expected, atol=1e-25)
    np.testing.assert_allclose(result["power_forecast"], expected, atol=1e-25)
    np.testing.assert_allclose(result["cross_power"], expected, atol=1e-25)
    np.testing.assert_array_equal(result["error_power"], np.zeros_like(expected))
    np.testing.assert_array_equal(result["wavenumber"], np.arange(17))


def test_attenuation_has_reduced_power_and_nonzero_error_despite_perfect_correlation() -> None:
    truth = _wave()
    result = zonal_spectral_statistics(truth, 0.4 * truth, LATITUDES)
    observed = result["power_truth"][0, 3]
    predicted = result["power_forecast"][0, 3]
    cross = result["cross_power"][0, 3]
    assert predicted / observed == pytest.approx(0.16)
    assert cross / np.sqrt(observed * predicted) == pytest.approx(1.0)
    assert result["error_power"][0, 3] / observed == pytest.approx(0.36)


@pytest.mark.parametrize("phase", [np.pi / 3, np.pi / 2, np.pi])
def test_phase_error_is_visible_with_identical_power(phase: float) -> None:
    result = zonal_spectral_statistics(_wave(), _wave(phase=phase), LATITUDES)
    observed = result["power_truth"][0, 3]
    predicted = result["power_forecast"][0, 3]
    assert predicted == pytest.approx(observed)
    correlation = result["cross_power"][0, 3] / np.sqrt(observed * predicted)
    assert correlation == pytest.approx(np.cos(phase), abs=1e-14)
    assert result["error_power"][0, 3] == pytest.approx(2 * observed * (1 - np.cos(phase)))


@pytest.mark.parametrize("n_lon", [31, 32])
def test_parseval_and_spectral_error_identity_for_noisy_multivariable_fields(n_lon: int) -> None:
    rng = np.random.default_rng(492)
    truth = rng.normal(size=(3, LATITUDES.size, n_lon)) + 280.0
    forecast = 0.7 * truth + rng.normal(size=truth.shape)
    result = zonal_spectral_statistics(truth, forecast, LATITUDES)
    weights = zonal_latitude_weights(LATITUDES)
    for name, field in (("power_truth", truth), ("power_forecast", forecast), ("error_power", forecast - truth)):
        anomaly = field - field.mean(axis=-1, keepdims=True)
        direct_mean_square = np.einsum("vl,l->v", np.mean(anomaly**2, axis=-1), weights)
        np.testing.assert_allclose(result[name].sum(axis=-1), direct_mean_square, rtol=1e-13)
        np.testing.assert_array_equal(result[name][:, 0], np.zeros(3))
    np.testing.assert_allclose(
        result["error_power"],
        result["power_truth"] + result["power_forecast"] - 2 * result["cross_power"],
        rtol=1e-13,
        atol=1e-15,
    )


@pytest.mark.parametrize("n_lon,expected", [(7, 0.5), (8, 1.0)])
def test_highest_mode_normalization_includes_even_grid_nyquist(n_lon: int, expected: float) -> None:
    truth = _wave(n_lon=n_lon, mode=n_lon // 2)
    result = zonal_spectral_statistics(truth, truth, LATITUDES)
    assert result["power_truth"][0, -1] == pytest.approx(expected)
    assert result["power_truth"].sum() == pytest.approx(expected)


def test_latitude_weighting_selects_both_inclusive_bands_and_ignores_other_rows() -> None:
    weights = zonal_latitude_weights(LATITUDES)
    expected = np.zeros(LATITUDES.size)
    expected[[1, 2, 3, 5, 6, 7]] = np.cos(np.deg2rad(LATITUDES[[1, 2, 3, 5, 6, 7]]))
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)
    truth = _wave() * np.arange(1.0, LATITUDES.size + 1)[None, :, None]
    truth[:, weights == 0, :] = 1e8 * _wave()[:, weights == 0, :]
    result = zonal_spectral_statistics(truth, truth, LATITUDES)
    expected_power = np.sum(weights * np.arange(1.0, LATITUDES.size + 1) ** 2) / 2
    assert result["power_truth"][0, 3] == pytest.approx(expected_power)
    reversed_result = zonal_spectral_statistics(truth[:, ::-1], truth[:, ::-1], LATITUDES[::-1])
    np.testing.assert_allclose(reversed_result["power_truth"], result["power_truth"])


def test_raw_powers_must_be_pooled_before_forming_ratios() -> None:
    weak = zonal_spectral_statistics(_wave(), 2 * _wave(), LATITUDES)
    strong = zonal_spectral_statistics(3 * _wave(), 3 * _wave(), LATITUDES)
    ratio = (weak["power_forecast"][0, 3] + strong["power_forecast"][0, 3]) / (
        weak["power_truth"][0, 3] + strong["power_truth"][0, 3]
    )
    assert ratio == pytest.approx(1.3)
    assert ratio != pytest.approx((4.0 + 1.0) / 2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("which", ["truth", "forecast"])
def test_nonfinite_fields_are_rejected_even_outside_selected_band(bad_value: float, which: str) -> None:
    fields = {"truth": _wave(), "forecast": _wave()}
    fields[which][0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        zonal_spectral_statistics(**fields, latitudes=LATITUDES)


@pytest.mark.parametrize(
    "bad_latitudes",
    [
        [30.0, 45.0, 60.0],
        [-60.0, -45.0, -30.0],
        [-90.0, 0.0, 90.0],
        [-100.0, -45.0, 45.0],
        [-45.0, np.nan, 45.0],
        [-45.0, -45.0, 45.0],
        [-45.0, 45.0, 0.0],
        [[-45.0, 45.0]],
    ],
)
def test_invalid_latitude_coordinates_are_rejected(bad_latitudes: list) -> None:
    with pytest.raises(ValueError):
        zonal_latitude_weights(np.asarray(bad_latitudes))


def test_invalid_field_shapes_and_nonreal_values_are_rejected() -> None:
    truth = _wave()
    with pytest.raises(ValueError, match="equal"):
        zonal_spectral_statistics(truth, truth[:, :, :-1], LATITUDES)
    with pytest.raises(ValueError, match="equal"):
        zonal_spectral_statistics(truth[0], truth[0], LATITUDES)
    with pytest.raises(ValueError, match="latitude coordinate length"):
        zonal_spectral_statistics(truth, truth, LATITUDES[1:-1])
    with pytest.raises(ValueError, match="real numeric"):
        zonal_spectral_statistics(truth.astype(complex), truth, LATITUDES)
    with pytest.raises(ValueError, match="two longitudes"):
        zonal_spectral_statistics(truth[:, :, :1], truth[:, :, :1], LATITUDES)


def test_metadata_is_json_safe_and_records_scope() -> None:
    metadata = json.loads(json.dumps(zonal_spectral_metadata()))
    assert metadata["latitude_band_degrees_absolute"] == [30.0, 60.0]
    assert metadata["smoothing"] == "none"
    assert "zonal" in metadata["zonal_mean"] or "each latitude" in metadata["zonal_mean"]
