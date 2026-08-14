from __future__ import annotations

import numpy as np
import xarray as xr

from src.models.mamba.v23_Ilya.metrics import V23IlyaMetricAccumulator


def _dataset(values: list[float]) -> xr.Dataset:
    array = np.asarray(values, dtype=np.float32)[None, :, None, None]
    return xr.Dataset(
        {"x": (("batch", "time", "lat", "lon"), array)},
        coords={"batch": [0], "time": np.arange(len(values)), "lat": [0.0], "lon": [0.0]},
    )


def test_metrics_match_simple_closed_form_values() -> None:
    accumulator = V23IlyaMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
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
    accumulator = V23IlyaMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
    accumulator.update(truth, baseline, full)
    output = accumulator.finalize()
    assert "temperature_level850" in output["per_channel_per_step"]
    np.testing.assert_allclose(
        output["per_channel_per_step"]["temperature_level850"]["rmse_full"],
        [0.5, 1.0],
    )
