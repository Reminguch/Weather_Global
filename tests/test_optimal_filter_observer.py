"""Scientific identities for independent optimal filters and output-only scoring."""
import json

import numpy as np
import pytest
import xarray as xr

from src.analysis.optimal_filter_observer import OptimalFilterObserver, STATISTICS_SCHEMA, VARIANTS
from src.analysis.optimal_spherical_filter import fit_degree_gains
from src.analysis.spherical_smoothing import ChannelLayout


LATITUDE = np.linspace(-90, 90, 9)
LONGITUDE = np.arange(16) * 22.5
HARMONIC = np.sin(np.deg2rad(LATITUDE))[:, None] * np.ones((1, len(LONGITUDE)))
UNRESOLVED = .3 * np.cos(np.deg2rad(LATITUDE))[:, None] * (-1.)**np.arange(len(LONGITUDE))[None]


def fields(mapping):
    return xr.Dataset({name: (("lat", "lon"), values) for name, values in mapping.items()},
                      coords={"lat": LATITUDE, "lon": LONGITUDE})


def metadata(year=2022, steps=2):
    return {"initialization_id": f"{year}0101T000000Z",
            "initialization_time": f"{year}-01-01T00:00:00Z",
            "lead_hours": list(range(6, 6*steps+1, 6))}


def read_statistics(path):
    with np.load(path, allow_pickle=False) as archive:
        return ({key: archive[key] for key in archive.files if key != "metadata"},
                json.loads(str(archive["metadata"])))


def test_calibration_keeps_winds_and_leads_independent(tmp_path):
    names = ("10m_u_component_of_wind", "10m_v_component_of_wind")
    target = fields({names[0]: 3 + HARMONIC, names[1]: 5 - 2*HARMONIC})
    diffs = xr.Dataset({names[0]: xr.DataArray(2.), names[1]: xr.DataArray(3.)})
    observer = OptimalFilterObserver(tmp_path, diffs, {"campaign_hash": "fixture"},
        mode="calibrate", target_steps=2, lmax=5, expected_channels=2, chunk_channels=1)
    observer.begin_sample(metadata())
    # Each branch, lead, and wind component requires a different gain.
    factors = np.asarray([[[2., 4.], [.5, -2.]], [[1.25, .25], [3., -.5]]])
    unchanged = target.copy(deep=True)
    for lead in range(2):
        predictions = [fields({name: factors[branch, lead, channel] * target[name].values
                               for channel, name in enumerate(names)}) for branch in range(2)]
        copies = [prediction.copy(deep=True) for prediction in predictions]
        observer.update_step(lead, target, *predictions)
        for prediction, copy in zip(predictions, copies):
            xr.testing.assert_identical(prediction, copy)
    observer.finish_sample()
    xr.testing.assert_identical(target, unchanged)
    arrays, saved = read_statistics(tmp_path / "20220101T000000Z.npz")
    assert saved["schema"] == STATISTICS_SCHEMA
    assert arrays["prediction_power"].shape == (2, 2, 2, 6)
    assert arrays["truth_power"].shape == (2, 2, 6)
    assert arrays["e0"].shape == (2, 2, 2)
    # The established layout ties these channels, but this fitter does not.
    assert saved["layout"]["group_index"] == [0, 0]
    gains, _ = fit_degree_gains(arrays["prediction_power"], arrays["cross_power"])
    np.testing.assert_allclose(gains[..., :2], np.repeat((1/factors)[..., None], 2, axis=-1),
                               rtol=2e-6, atol=2e-6)
    for branch in range(2):
        for lead in range(2):
            expected = observer.operator.mean(np.stack([
                ((factors[branch, lead, channel]-1)*target[name].values / float(diffs[name]))**2
                for channel, name in enumerate(names)]))
            np.testing.assert_allclose(arrays["e0"][branch, lead], expected, rtol=1e-13)


def test_evaluation_projects_residual_and_allows_mean_change_and_negative_gains(tmp_path):
    name = "total_precipitation_6hr"
    smooth = 3 + HARMONIC
    target = fields({name: smooth})
    baseline = fields({name: 2*smooth + UNRESOLVED})
    full = fields({name: 4*smooth + UNRESOLVED})
    original = [dataset.copy(deep=True) for dataset in (target, baseline, full)]
    diffs = xr.Dataset({name: xr.DataArray(2.)})
    layout = ChannelLayout.from_dataset(target, diffs, expected_channels=1)
    filters = {"layout": layout.metadata(), "lmax": 5, "lead_hours": [6, 12], "branches": {
        "GC": {"gains": np.asarray([[[.5]*6], [[1.]*6]]).tolist()},
        "M1": {"gains": np.asarray([[[.25]*6], [[-.25]*6]]).tolist()}}}
    observer = OptimalFilterObserver(tmp_path, diffs, {"campaign_hash": "fixture"},
        mode="evaluate", filters=filters, target_steps=2, lmax=5, expected_channels=1)
    observer.begin_sample(metadata(2023))
    observer.update_step(0, target, baseline, full)
    observer.update_step(1, target, baseline, full)
    observer.finish_sample()
    for dataset, copy in zip((target, baseline, full), original):
        xr.testing.assert_identical(dataset, copy)
    arrays, saved = read_statistics(tmp_path / "20230101T000000Z.npz")
    assert saved["variants"] == list(VARIANTS)
    assert arrays["exact_gc_loss"].shape == (2, 2, 3)
    assert arrays["grid_mse"].shape == (2, 2, 3, 1)
    assert arrays["error_power"].shape == (2, 2, 3, 1, 6)
    # Identity in coefficient space is a projection, not native-grid identity.
    assert np.all(arrays["forecast_projection_residual_mse"] > .01)
    np.testing.assert_allclose(arrays["grid_mse"][0, :, 2, 0], 0, atol=2e-11)
    np.testing.assert_allclose(arrays["prediction_mean"][0, :, 2, 0], 3., atol=2e-6)
    assert np.all(np.abs(arrays["mean_drift"][0, :, 2, 0]) > 2.9)
    np.testing.assert_array_equal(arrays["prediction_power"][:, :, 0], arrays["prediction_power"][:, :, 1])
    np.testing.assert_array_equal(arrays["error_power"][:, :, 0], arrays["error_power"][:, :, 1])
    # A finite worsening and negative precipitation remain visible, without clipping.
    expected_bad = observer.operator.mean((2*smooth)**2)
    np.testing.assert_allclose(arrays["grid_mse"][1, 1, 2, 0], expected_bad, rtol=2e-6)
    assert arrays["negative_fraction"][1, 1, 2, 0] == 1.
    assert arrays["positive_field_minimum"][1, 1, 2, 0] < -3.9
    weights = layout.weights / layout.scales**2
    np.testing.assert_allclose(arrays["exact_gc_loss"],
                               np.sum(arrays["grid_mse"] * weights, axis=-1), rtol=1e-14)
    np.testing.assert_allclose(arrays["spectral_weighted_gc_loss"],
        np.sum(arrays["error_power"].sum(axis=-1) * weights, axis=-1), rtol=1e-14)
    with pytest.raises(FileExistsError):
        observer.begin_sample(metadata(2023))


def test_incomplete_and_invalid_filter_inputs_are_errors(tmp_path):
    name = "2m_temperature"
    target = fields({name: 3 + HARMONIC})
    diffs = xr.Dataset({name: xr.DataArray(1.)})
    layout = ChannelLayout.from_dataset(target, diffs, expected_channels=1)
    with pytest.raises(ValueError, match="frozen"):
        OptimalFilterObserver(tmp_path, diffs, {}, mode="evaluate")
    observer = OptimalFilterObserver(tmp_path / "cal", diffs, {}, mode="calibrate",
                                      target_steps=2, lmax=5, expected_channels=1)
    with pytest.raises(ValueError, match="schedule"):
        observer.begin_sample(metadata(steps=1))
    observer.begin_sample(metadata())
    observer.update_step(0, target, target, target)
    with pytest.raises(ValueError, match="Incomplete"):
        observer.finish_sample()
    with pytest.raises(ValueError, match="out-of-order"):
        observer.update_step(0, target, target, target)
    assert not list((tmp_path / "cal").glob("*.npz"))
    gains = np.ones((2, 1, 6))
    gains[0, 0, 0] = np.nan
    filters = {"layout": layout.metadata(), "branches": {
        name: {"gains": gains.tolist()} for name in ("GC", "M1")}}
    observer = OptimalFilterObserver(tmp_path / "eval", diffs, {}, mode="evaluate", filters=filters,
        target_steps=2, lmax=5, expected_channels=1)
    observer.begin_sample(metadata(2023))
    with pytest.raises(ValueError, match="gains"):
        observer.update_step(0, target, target, target)
