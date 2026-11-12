"""Scientific identities and observer integrity for output-only sweeps."""
import json

import numpy as np
import pytest
import xarray as xr

from src.analysis.spherical_smoothing import ChannelLayout, SphericalAttenuator
from src.analysis.smoothing_sweep_observer import (
    BASE_NAMES, FIXED_WIDTHS_DEGREES, STATISTICS_SCHEMA, SmoothingSweepObserver,
    correction_statistics, reconstruct_sweep, sweep_gains,
)


@pytest.fixture(scope="module")
def operator():
    return SphericalAttenuator(np.linspace(-90, 90, 9), np.arange(16) * 22.5,
                              lmax=5, knots=(1, 2, 3, 5), chunk_channels=2)


def example_fields(operator):
    harmonic = np.sin(np.deg2rad(operator.latitude))[:, None] * np.ones((1, 16))
    unresolved = .3 * (-1.)**np.arange(16)[None] * np.cos(np.deg2rad(operator.latitude))[:, None]
    forecast = np.stack((3 + 2*harmonic + unresolved, 10 - 3*harmonic - unresolved))
    truth = np.stack((3 + harmonic, 11 - harmonic))
    return forecast, truth


def test_native_grid_and_spherical_strength_polynomials(operator):
    forecast, truth = example_fields(operator)
    scales = np.asarray([2., .3])[:, None, None]
    gains = sweep_gains(operator, np.array([3, 5]), np.array([[.8,.4,.3,.7], [.1,.6,.9,.3]]))
    deltas = operator.corrections(forecast / scales, gains - 1) * scales[None]
    moments = correction_statistics(operator, forecast, truth, deltas)
    for base in (0, 8, 9, 10):
        for alpha in (0., .1, .5, 1.):
            result = reconstruct_sweep(moments, base, alpha)
            combined = forecast + alpha * deltas[base]
            direct = operator.spectral_chunk(combined, truth)
            np.testing.assert_allclose(result["grid_mse"], operator.mean((combined-truth)**2), rtol=1e-13, atol=1e-13)
            # Reanalysis rounds FP32 transforms independently; this tolerance
            # verifies the formula, not a production acceptance threshold.
            for key in ("prediction_power", "cross_power", "error_power"):
                np.testing.assert_allclose(result[key], direct[key], rtol=3e-6, atol=3e-7)
    # The native residual is retained, so a continuous spherical norm is not
    # an interchangeable reconstruction of the native-grid error.
    assert not np.allclose(moments["grid_mse"],
        reconstruct_sweep(moments, 0, 0.)["error_power"].sum(axis=-1), rtol=.001)


def test_sweep_endpoints_match_original_filters_and_preserve_mean(operator):
    forecast, _ = example_fields(operator)
    forecast += 100000.
    scales = np.array([2., .3])
    indices = np.array([0, 5])
    bands = np.array([[1., 1., 1., 1.], [.1,.6,.9,.3]])
    gains = sweep_gains(operator, indices, bands)
    assert gains.shape == (11, 2, 6)
    np.testing.assert_array_equal(gains[..., 0], 1)
    assert len(BASE_NAMES) == 11 and len(FIXED_WIDTHS_DEGREES) == 9
    deltas = operator.corrections(forecast / scales[:, None, None], gains-1) * scales[None, :, None, None]
    original_pair = operator.filtered_pair(forecast, scales, indices, bands)
    np.testing.assert_allclose(forecast[None] + deltas[-2:], original_pair, rtol=0, atol=2e-7)
    np.testing.assert_array_equal(deltas[-2:, 0], 0)
    np.testing.assert_allclose(operator.mean(deltas), 0, atol=1e-14, rtol=0)
    np.testing.assert_allclose(operator.mean(forecast[None] + deltas),
                               np.broadcast_to(operator.mean(forecast), (11, 2)), atol=1e-9, rtol=0)
    with pytest.raises(ValueError, match="index"):
        sweep_gains(operator, np.array([7, 0]), bands)
    with pytest.raises(ValueError, match="four-band"):
        sweep_gains(operator, indices, np.full((2,4), np.nan))


def test_reconstruction_lead_model_axes_and_finite_warning_policy(operator):
    forecast, truth = example_fields(operator)
    gains = sweep_gains(operator, np.array([3, 5]), np.full((2,4), .5))
    delta = operator.corrections(forecast, gains-1)
    chunk = correction_statistics(operator, forecast, truth, delta)
    leading = {key: np.tile(value, (3, 2) + (1,)*value.ndim) for key, value in chunk.items()}
    leading["truth_power"] = np.tile(chunk["truth_power"], (3,1,1))
    result = reconstruct_sweep(leading, 10, .5)
    assert result["prediction_power"].shape == (3,2,2,6)
    np.testing.assert_allclose(result["grid_mse"][2,1], reconstruct_sweep(chunk, 10, .5)["grid_mse"])
    # Finite negative cancellation is retained, not clipped or rejected.
    chunk["grid_mse"][:] = -1e-16
    np.testing.assert_array_equal(reconstruct_sweep(chunk, 0, 0.)["grid_mse"], -1e-16)
    with pytest.raises(ValueError, match="strength"):
        reconstruct_sweep(chunk, 0, float("nan"))
    delta[0,0,0,0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        correction_statistics(operator, forecast, truth, delta)


def test_observer_contract_immutability_and_saved_schema(tmp_path, operator):
    forecast, target = example_fields(operator)
    coords = {"lat": operator.latitude, "lon": operator.longitude}
    truth = xr.Dataset({"2m_temperature": (("lat","lon"), target[0])}, coords=coords)
    prediction = xr.Dataset({"2m_temperature": (("lat","lon"), forecast[0])}, coords=coords)
    unchanged = prediction.copy(deep=True)
    diffs = xr.Dataset({"2m_temperature": xr.DataArray(2.)})
    layout = ChannelLayout.from_dataset(truth, diffs, expected_channels=1)
    filters = {"layout": layout.metadata(), "branches": {
        branch: {"gaussian_indices": [[3]]*4, "band_gains": [[[.5]*4]]*4}
        for branch in ("GC", "M1")}}
    observer = SmoothingSweepObserver(tmp_path, diffs, {"checkpoint_sha256":"test"},
        filters=filters, target_steps=1, lmax=5, knots=(1,2,3,5), expected_channels=1)
    metadata = {"initialization_id":"20230101T000000Z", "initialization_time":"2023-01-01T00:00:00Z", "lead_hours":[6]}
    observer.begin_sample(metadata)
    observer.update_step(0, truth, prediction, prediction)
    observer.finish_sample()
    xr.testing.assert_identical(prediction, unchanged)
    with np.load(tmp_path/"20230101T000000Z.npz") as archive:
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["schema"] == STATISTICS_SCHEMA
        assert metadata["base_names"] == list(BASE_NAMES)
        assert metadata["base_width_degrees"] == list(FIXED_WIDTHS_DEGREES) + [None,None]
        assert archive["exact_gc_loss"].shape == (1,2,1)
        assert archive["delta_power"].shape == (1,2,11,1,6)
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
        raw_loss = np.sum(arrays["grid_mse"] * (layout.weights/layout.scales**2), axis=-1)
        np.testing.assert_array_equal(raw_loss, archive["exact_gc_loss"][...,0])
        assert reconstruct_sweep(arrays, 10, .5)["error_power"].shape == (1,2,1,6)
    with pytest.raises(FileExistsError):
        observer.begin_sample({"initialization_id":"20230101T000000Z", "lead_hours":[6]})
    with pytest.raises(ValueError, match="only evaluates"):
        SmoothingSweepObserver(tmp_path, diffs, {}, mode="calibrate")
