from __future__ import annotations

import numpy as np
import xarray as xr
from graphcast import losses as graphcast_losses
from graphcast import normalization as graphcast_normalization

from src.models.mamba.v22_final.metrics import (
    ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS,
    V22FinalMetricAccumulator,
)


def _dataset(values: list[float]) -> xr.Dataset:
    array = np.asarray(values, dtype=np.float32)[None, :, None, None]
    return xr.Dataset(
        {"x": (("batch", "time", "lat", "lon"), array)},
        coords={"batch": [0], "time": np.arange(len(values)), "lat": [0.0], "lon": [0.0]},
    )


def test_metrics_match_simple_closed_form_values() -> None:
    accumulator = V22FinalMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
    accumulator.update(
        _dataset([0.0, 0.0]),
        _dataset([1.0, 2.0]),
        _dataset([0.0, 1.0]),
    )
    output = accumulator.finalize()
    entry = output["per_variable_per_step"]["x"]
    np.testing.assert_allclose(entry["rmse_baseline"], [1.0, 2.0])
    np.testing.assert_allclose(entry["rmse_full"], [0.0, 1.0])
    np.testing.assert_allclose(entry["mae_baseline"], [1.0, 2.0])
    np.testing.assert_allclose(entry["mae_full"], [0.0, 1.0])
    np.testing.assert_allclose(entry["improvement_pct_rmse"], [100.0, 50.0])
    np.testing.assert_allclose(entry["rmsb_baseline"], [1.0, 2.0])
    np.testing.assert_allclose(entry["rmsb_full"], [0.0, 1.0])

    diagnostics = output["residual_diagnostics_per_variable"]["x"]
    np.testing.assert_allclose(diagnostics["residual_cosine_by_lead"], [1.0, 1.0])
    np.testing.assert_allclose(diagnostics["residual_gain_by_lead"], [1.0, 0.5])
    np.testing.assert_allclose(diagnostics["delta_mse_by_lead"], [1.0, 3.0])
    np.testing.assert_allclose(diagnostics["E_post_mse_by_lead"], [0.0, 1.0])


def test_pressure_level_metrics_keep_legacy_channel_keys() -> None:
    values = np.asarray([1.0, 2.0], dtype=np.float32)[None, :, None, None, None]
    baseline = xr.Dataset(
        {"temperature": (("batch", "time", "level", "lat", "lon"), values)},
        coords={"batch": [0], "time": [0, 1], "level": [850], "lat": [0.0], "lon": [0.0]},
    )
    truth = baseline * 0
    full = baseline * 0.5
    accumulator = V22FinalMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
    accumulator.update(truth, baseline, full)
    output = accumulator.finalize()
    assert "temperature_level850" in output["per_channel_per_step"]
    np.testing.assert_allclose(
        output["per_channel_per_step"]["temperature_level850"]["rmse_full"],
        [0.5, 1.0],
    )


def test_original_graphcast_loss_matches_graphcast_implementation() -> None:
    coordinates = {
        "batch": [0],
        "time": [0, 1],
        "level": [500, 1000],
        "lat": [-90.0, 0.0, 90.0],
        "lon": [0.0, 180.0],
    }
    surface_shape = (1, 2, 3, 2)
    atmospheric_shape = (1, 2, 2, 3, 2)
    truth = xr.Dataset(
        {
            "2m_temperature": (
                ("batch", "time", "lat", "lon"),
                np.zeros(surface_shape, dtype=np.float32),
            ),
            "temperature": (
                ("batch", "time", "level", "lat", "lon"),
                np.zeros(atmospheric_shape, dtype=np.float32),
            ),
        },
        coords=coordinates,
    )
    baseline = xr.Dataset(
        {
            "2m_temperature": xr.ones_like(truth["2m_temperature"]) * 4.0,
            "temperature": xr.ones_like(truth["temperature"]) * 8.0,
        }
    )
    baseline = baseline * xr.DataArray([1.0, 2.0], dims="time", coords={"time": [0, 1]})
    full = baseline * xr.DataArray([0.5, 1.0], dims="time", coords={"time": [0, 1]})
    difference_scales = xr.Dataset(
        {
            "2m_temperature": xr.DataArray(2.0),
            "temperature": (("level",), np.asarray([4.0, 8.0], dtype=np.float32)),
        },
        coords={"level": coordinates["level"]},
    )

    accumulator = V22FinalMetricAccumulator(
        target_steps=2,
        latitudes=np.asarray(coordinates["lat"]),
        diffs_stddev_by_level=difference_scales,
    )
    accumulator.update(truth, baseline, full)
    result = accumulator.finalize()["original_graphcast_loss"]

    expected_baseline = []
    expected_full = []
    for step_index in range(2):
        for prediction, destination in (
            (baseline, expected_baseline),
            (full, expected_full),
        ):
            normalized_error = graphcast_normalization.normalize(
                prediction - truth,
                difference_scales,
                None,
            ).isel(time=slice(step_index, step_index + 1))
            loss, _ = graphcast_losses.weighted_mse_per_level(
                normalized_error,
                xr.zeros_like(normalized_error),
                per_variable_weights={
                    variable: weight
                    for variable, weight in ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS.items()
                    if variable in normalized_error
                },
            )
            destination.append(float(loss.values[0]))

    np.testing.assert_allclose(result["baseline_per_step"], expected_baseline)
    np.testing.assert_allclose(result["full_per_step"], expected_full)
    np.testing.assert_allclose(result["improvement_pct_per_step"], [75.0, 0.0])
    expected_rollout_improvement = 100.0 * (
        1.0 - np.mean(expected_full) / np.mean(expected_baseline)
    )
    np.testing.assert_allclose(
        result["improvement_pct_rollout"],
        expected_rollout_improvement,
    )
    assert not np.isclose(
        result["improvement_pct_rollout"],
        np.mean(result["improvement_pct_per_step"]),
    )
