from __future__ import annotations

import numpy as np
import xarray as xr
from graphcast import losses as graphcast_losses
from graphcast import normalization as graphcast_normalization

from src.models.mamba.v24_Ilya.metrics import (
    ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS,
    V24IlyaMetricAccumulator,
)


def _dataset(values: list[float]) -> xr.Dataset:
    array = np.asarray(values, dtype=np.float32)[None, :, None, None]
    return xr.Dataset(
        {"x": (("batch", "time", "lat", "lon"), array)},
        coords={"batch": [0], "time": np.arange(len(values)), "lat": [0.0], "lon": [0.0]},
    )


def test_metrics_match_simple_closed_form_values() -> None:
    accumulator = V24IlyaMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
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
    accumulator = V24IlyaMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
    accumulator.update(truth, baseline, full)
    intermediate = accumulator.finalize(include_rms_bias=False)
    intermediate_entry = intermediate["per_channel_per_step"]["temperature_level850"]
    np.testing.assert_allclose(intermediate_entry["rmse_full"], [0.5, 1.0])
    assert "rmsb_baseline" not in intermediate_entry
    assert "rmsb_full" not in intermediate_entry
    assert "improvement_pct_rmsb" not in intermediate_entry

    output = accumulator.finalize()
    assert "temperature_level850" in output["per_channel_per_step"]
    np.testing.assert_allclose(
        output["per_channel_per_step"]["temperature_level850"]["rmse_full"],
        [0.5, 1.0],
    )
    assert "rmsb_baseline" in output["per_channel_per_step"]["temperature_level850"]


def test_leadwise_metrics_match_batched_updates_across_samples() -> None:
    def sample(offset: float) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        surface = np.asarray([offset, offset + 1], dtype=np.float32)[None, :, None, None]
        pressure = np.asarray(
            [offset + 2, offset + 3, offset + 4, offset + 5],
            dtype=np.float32,
        ).reshape(1, 2, 2, 1, 1)
        truth = xr.Dataset(
            {
                "surface": (("batch", "time", "lat", "lon"), surface),
                "temperature": (
                    ("batch", "time", "level", "lat", "lon"),
                    pressure,
                ),
            },
            coords={
                "batch": [0],
                "time": [0, 1],
                "level": [500, 850],
                "lat": [0.0],
                "lon": [0.0],
            },
        )
        baseline = truth + 2.0
        full = truth + 0.5
        return truth, baseline, full

    batched = V24IlyaMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
    streamed = V24IlyaMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
    for offset in (0.0, 10.0):
        truth, baseline, full = sample(offset)
        batched.update(truth, baseline, full)
        for step_index in range(2):
            selection = {"time": slice(step_index, step_index + 1)}
            streamed.update_step(
                step_index,
                truth.isel(**selection),
                baseline.isel(**selection),
                full.isel(**selection),
            )

    assert streamed.finalize() == batched.finalize()


def test_streamed_original_graphcast_loss_matches_graphcast_implementation() -> None:
    coordinates = {
        "batch": [0],
        "time": [0, 1],
        "level": [500, 1000],
        "lat": [-90.0, 0.0, 90.0],
        "lon": [0.0, 180.0],
    }
    truth = xr.Dataset(
        {
            "2m_temperature": (
                ("batch", "time", "lat", "lon"),
                np.zeros((1, 2, 3, 2), dtype=np.float32),
            ),
            "temperature": (
                ("batch", "time", "level", "lat", "lon"),
                np.zeros((1, 2, 2, 3, 2), dtype=np.float32),
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
    baseline = baseline * xr.DataArray(
        [1.0, 2.0],
        dims="time",
        coords={"time": [0, 1]},
    )
    full = baseline * xr.DataArray(
        [0.5, 1.0],
        dims="time",
        coords={"time": [0, 1]},
    )
    difference_scales = xr.Dataset(
        {
            "2m_temperature": xr.DataArray(2.0),
            "temperature": (
                ("level",),
                np.asarray([4.0, 8.0], dtype=np.float32),
            ),
        },
        coords={"level": coordinates["level"]},
    )

    accumulator = V24IlyaMetricAccumulator(
        target_steps=2,
        latitudes=np.asarray(coordinates["lat"]),
        diffs_stddev_by_level=difference_scales,
    )
    for step_index in range(2):
        selection = {"time": slice(step_index, step_index + 1)}
        accumulator.update_step(
            step_index,
            truth.isel(**selection),
            baseline.isel(**selection),
            full.isel(**selection),
        )
    intermediate = accumulator.finalize(include_rms_bias=False)
    result = intermediate["original_graphcast_loss"]
    assert "2m_temperature" in intermediate["per_variable_per_step"]
    assert "rmsb_baseline" not in intermediate["per_variable_per_step"]["2m_temperature"]

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


def test_batched_and_leadwise_original_graphcast_loss_match() -> None:
    def fields(values: list[float]) -> xr.Dataset:
        data = np.broadcast_to(
            np.asarray(values, dtype=np.float32)[None, :, None, None],
            (1, 2, 2, 1),
        ).copy()
        return xr.Dataset(
            {"x": (("batch", "time", "lat", "lon"), data)},
            coords={
                "batch": [0],
                "time": [0, 1],
                "lat": [-45.0, 45.0],
                "lon": [0.0],
            },
        )

    truth = fields([0.0, 0.0])
    baseline = fields([1.0, 2.0])
    full = fields([0.5, 1.0])
    scales = xr.Dataset({"x": xr.DataArray(2.0)})
    common = {
        "target_steps": 2,
        "latitudes": np.asarray([-45.0, 45.0]),
        "diffs_stddev_by_level": scales,
    }
    batched = V24IlyaMetricAccumulator(**common)
    streamed = V24IlyaMetricAccumulator(**common)
    batched.update(truth, baseline, full)
    for step_index in range(2):
        selection = {"time": slice(step_index, step_index + 1)}
        streamed.update_step(
            step_index,
            truth.isel(**selection),
            baseline.isel(**selection),
            full.isel(**selection),
        )
    assert streamed.finalize() == batched.finalize()


def test_exported_metric_states_merge_to_monolithic_result() -> None:
    coordinates = {
        "batch": [0],
        "time": [0, 1],
        "level": [500, 850],
        "lat": [-45.0, 45.0],
        "lon": [0.0, 180.0],
    }
    scales = xr.Dataset(
        {
            "2m_temperature": xr.DataArray(2.0),
            "temperature": (
                ("level",),
                np.asarray([3.0, 4.0], dtype=np.float32),
            ),
        },
        coords={"level": coordinates["level"]},
    )
    common = {
        "target_steps": 2,
        "latitudes": np.asarray(coordinates["lat"]),
        "diffs_stddev_by_level": scales,
    }
    monolithic = V24IlyaMetricAccumulator(**common)
    shards = [V24IlyaMetricAccumulator(**common) for _ in range(4)]

    for sample_index, shard in enumerate(shards):
        truth = xr.Dataset(
            {
                "2m_temperature": (
                    ("batch", "time", "lat", "lon"),
                    np.full((1, 2, 2, 2), sample_index, dtype=np.float32),
                ),
                "temperature": (
                    ("batch", "time", "level", "lat", "lon"),
                    np.full((1, 2, 2, 2, 2), sample_index, dtype=np.float32),
                ),
            },
            coords=coordinates,
        )
        surface_error = np.asarray(
            [
                1.0 + sample_index,
                -0.5 - sample_index,
                2.0 - sample_index,
                -1.0 + sample_index,
            ],
            dtype=np.float32,
        ).reshape(1, 1, 2, 2)
        surface_error = np.repeat(surface_error, 2, axis=1)
        pressure_error = np.repeat(
            surface_error[:, :, None, :, :],
            2,
            axis=2,
        )
        baseline = truth + xr.Dataset(
            {
                "2m_temperature": (
                    ("batch", "time", "lat", "lon"),
                    surface_error,
                ),
                "temperature": (
                    ("batch", "time", "level", "lat", "lon"),
                    pressure_error,
                ),
            },
            coords=coordinates,
        )
        full = truth + (baseline - truth) * (0.2 + 0.1 * sample_index)
        monolithic.update(truth, baseline, full)
        shard.update(truth, baseline, full)

    merged = V24IlyaMetricAccumulator.from_merge_state(shards[0].export_merge_state())
    for shard in shards[1:]:
        merged.merge_exported_state(shard.export_merge_state())

    expected = monolithic.finalize()
    actual = merged.finalize()

    def assert_nested_close(left, right) -> None:
        if isinstance(left, dict):
            assert set(left) == set(right)
            for key in left:
                assert_nested_close(left[key], right[key])
        elif isinstance(left, list):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)
        elif isinstance(left, (int, float)):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)
        else:
            assert left == right

    assert_nested_close(expected, actual)


def test_metric_merge_rejects_different_latitudes() -> None:
    first = V24IlyaMetricAccumulator(target_steps=1, latitudes=np.asarray([0.0]))
    second = V24IlyaMetricAccumulator(target_steps=1, latitudes=np.asarray([10.0]))
    truth = _dataset([0.0])
    baseline = _dataset([1.0])
    full = _dataset([0.5])
    first.update(truth, baseline, full)
    second.update(truth.assign_coords(lat=[10.0]), baseline.assign_coords(lat=[10.0]), full.assign_coords(lat=[10.0]))

    with np.testing.assert_raises_regex(ValueError, "latitude coordinates differ"):
        first.merge_exported_state(second.export_merge_state())


def test_lightweight_metrics_skip_spatial_bias_and_merge_exactly() -> None:
    truth = _dataset([0.0, 0.0])
    baseline = _dataset([2.0, 4.0])
    full = _dataset([1.0, 2.0])
    first = V24IlyaMetricAccumulator(
        target_steps=2,
        latitudes=np.asarray([0.0]),
        store_spatial_bias=False,
    )
    second = V24IlyaMetricAccumulator(
        target_steps=2,
        latitudes=np.asarray([0.0]),
        store_spatial_bias=False,
    )
    first.update(truth, baseline, full)
    second.update(truth, baseline, full)

    assert first.sum_err_cell_b == {}
    assert all(not levels for levels in first.sum_err_cell_b_pl.values())
    output = first.finalize()
    entry = output["per_variable_per_step"]["x"]
    assert "rmsb_baseline" not in entry
    np.testing.assert_allclose(entry["rmse_full"], [1.0, 2.0])

    merged = V24IlyaMetricAccumulator.from_merge_state(first.export_merge_state())
    merged.merge_exported_state(second.export_merge_state())
    assert merged.stores_spatial_bias is False
    np.testing.assert_allclose(
        merged.finalize()["per_variable_per_step"]["x"]["rmse_full"],
        [1.0, 2.0],
    )
    with np.testing.assert_raises_regex(ValueError, "without spatial bias"):
        merged.finalize(include_rms_bias=True)
