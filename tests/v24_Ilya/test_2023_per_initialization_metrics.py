"""Physical-error exports preserve the legacy metric and pool before sqrt."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.evaluation import (
    build_initialization_loss_record,
    collect_initialization_variable_mse,
)
from src.models.mamba.v24_Ilya.metrics import (
    VARIABLE_PHYSICAL_UNITS,
    V24IlyaMetricAccumulator,
)


SURFACE_VARIABLES = {
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr",
}


def _fields(error: float) -> xr.Dataset:
    # Unequal latitude errors and unequal pressure-level errors distinguish
    # cosine weighting and equal-level averaging from competing reductions.
    surface = error * np.asarray([1.0, 2.0, 4.0], dtype=np.float32).reshape(1, 1, 3, 1)
    pressure = surface[:, :, None] * np.asarray([1., 3.], np.float32).reshape(1, 1, 2, 1, 1)
    return xr.Dataset(
        {
            variable: (
                ("batch", "time", "lat", "lon"), surface.copy(),
            ) if variable in SURFACE_VARIABLES else (
                ("batch", "time", "level", "lat", "lon"), pressure.copy(),
            )
            for variable in VARIABLE_PHYSICAL_UNITS
        },
        coords={
            "batch": [0], "time": [np.timedelta64(6, "h")],
            "lat": [-60., 0., 60.], "lon": [0.], "level": [100, 1000],
        },
    )


def _accumulator(steps: int = 2) -> V24IlyaMetricAccumulator:
    return V24IlyaMetricAccumulator(
        steps, np.asarray([-60., 0., 60.]), store_spatial_bias=False,
        diffs_stddev_by_level=xr.Dataset({v: xr.DataArray(2.) for v in VARIABLE_PHYSICAL_UNITS}),
    )


def test_per_initialization_mse_reproduces_all_eleven_aggregate_rmse_groups():
    accumulator = _accumulator()
    records = []
    for error in (1., 5., 2.):
        losses = {"baseline": [], "full": []}
        variable_mse = {}
        for lead in range(2):
            exact = accumulator.update_step(
                lead, _fields(0), _fields(error * (lead + 1)), _fields(error / 2),
            )
            assert set(exact) == {"baseline", "full"}
            for branch in losses:
                losses[branch].append(float(exact[branch].item()))
            collect_initialization_variable_mse(
                variable_mse, step_index=lead, target_steps=2,
                contributions=accumulator.last_step_variable_mse,
            )
        records.append(build_initialization_loss_record(
            {"valid_times": ["lead1", "lead2"]}, losses, variable_mse=variable_mse,
        ))

    output = accumulator.finalize()
    assert len(accumulator.variable_mse_metadata) == 11
    for variable in VARIABLE_PHYSICAL_UNITS:
        metadata = accumulator.variable_mse_metadata[variable]
        assert metadata == {
            "units": VARIABLE_PHYSICAL_UNITS[variable],
            "spatial_weighting": "normalized_cosine_latitude",
            "pressure_pooling": "surface" if variable in SURFACE_VARIABLES else "equal_levels",
        }
        for branch in ("baseline", "full"):
            per_start = np.asarray([r["variable_mse"][variable][f"{branch}_per_step"] for r in records])
            expected = np.sqrt(per_start.mean(axis=0))
            np.testing.assert_allclose(
                output["per_variable_per_step"][variable][f"rmse_{branch}"], expected,
            )
            assert not np.allclose(expected, np.sqrt(per_start).mean(axis=0))
    assert accumulator.sum_err_cell_b == {}


def test_per_call_mse_is_direct_even_when_accumulated_sum_loses_small_increment():
    accumulator = _accumulator(1)
    accumulator.update_step(0, _fields(0), _fields(1.e10), _fields(0))
    first_sum = accumulator.sum_sq_b["2m_temperature"].copy()
    accumulator.update_step(0, _fields(0), _fields(1), _fields(0))
    # The direct contribution survives even though subtraction of accumulated
    # floating-point sums would return zero.
    np.testing.assert_array_equal(accumulator.sum_sq_b["2m_temperature"], first_sum)
    surface_mse = (.5 * 1. + 1. * 4. + .5 * 16.) / 2.
    assert accumulator.last_step_variable_mse["2m_temperature"]["baseline"] == pytest.approx(surface_mse)
    assert accumulator.last_step_variable_mse["temperature"]["baseline"] == pytest.approx(surface_mse * 5)
    assert all(v["full"] == 0 for v in accumulator.last_step_variable_mse.values())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.])
def test_record_cannot_publish_nonfinite_negative_or_missing_leads(invalid):
    with pytest.raises(ValueError, match="Incomplete or invalid physical"):
        build_initialization_loss_record(
            {"valid_times": ["lead1", "lead2"]}, {"baseline": [1, 2], "full": [0, 1]},
            variable_mse={"temperature": {"baseline_per_step": [1, invalid], "full_per_step": [0, 1]}},
        )
    with pytest.raises(ValueError, match="Incomplete or invalid physical"):
        build_initialization_loss_record(
            {"valid_times": ["lead1", "lead2"]}, {"baseline": [1, 2], "full": [0, 1]},
            variable_mse={"temperature": {"baseline_per_step": [1], "full_per_step": [0, 1]}},
        )


def test_collection_rejects_duplicate_leads_and_changing_variable_sets():
    destination = {}
    kwargs = dict(step_index=0, target_steps=2, contributions={"temperature": {"baseline": 1., "full": 0.}})
    collect_initialization_variable_mse(destination, **kwargs)
    with pytest.raises(ValueError, match="Duplicate"):
        collect_initialization_variable_mse(destination, **kwargs)
    with pytest.raises(ValueError, match="variables changed"):
        collect_initialization_variable_mse(destination, step_index=1, target_steps=2,
                                            contributions={"geopotential": {"baseline": 1., "full": 0.}})
    with pytest.raises(ValueError, match="Incomplete or invalid"):
        build_initialization_loss_record(
            {"valid_times": ["lead1", "lead2"]}, {"baseline": [1, 2], "full": [0, 1]},
            variable_mse=destination,
        )


def test_physical_mse_rejects_missing_grid_cells_without_exact_loss_scales():
    accumulator = V24IlyaMetricAccumulator(1, np.asarray([-60., 0., 60.]), store_spatial_bias=False)
    baseline = _fields(1)
    baseline["2m_temperature"].values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid physical variable MSE"):
        accumulator.update_step(0, _fields(0), baseline, _fields(0))


def test_legacy_loss_record_shape_is_unchanged_without_optional_mse():
    assert build_initialization_loss_record(
        {"valid_times": ["lead1"]}, {"baseline": [2.], "full": [1.]},
    ) == {"valid_times": ["lead1"], "baseline_per_step": [2.], "full_per_step": [1.]}
