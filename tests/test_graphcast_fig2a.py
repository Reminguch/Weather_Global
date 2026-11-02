"""Scientific invariants specific to the paper's native-grid RMSE protocol."""
import numpy as np
import pytest

from scripts.analyze_models.eval_graphcast_fig2a import paper_rmse, spatial_mse
from scripts.analyze_models.report_graphcast_fig2a import independent_area_weights
from scripts.analyze_models.graphcast_fig2a_data import array_sha


def test_rmse_is_averaged_after_each_forecast_root():
    values = np.array([[1., 4.], [9., 16.]], dtype=np.float32)
    np.testing.assert_array_equal(paper_rmse(values), [2., 3.])
    assert not np.allclose(paper_rmse(values), np.sqrt(values.mean(axis=0)))


def test_global_constant_bias_keeps_physical_units():
    lat = np.arange(-90, 90.25, .25)
    weights = independent_area_weights(lat)
    truth = np.full((721, 8), 50000., dtype=np.float32)
    assert np.isclose(spatial_mse(truth + 10, truth, weights), 100, rtol=1e-6)
    assert weights[0] > 0 and np.isclose(weights.mean(), 1)
    assert np.isclose(weights[0] / weights[1], .125, rtol=1e-5)


def test_polar_errors_have_finite_weight_and_latitude_matters():
    lat = np.arange(-90, 90.25, .25)
    weights = independent_area_weights(lat)
    truth = np.zeros((721, 4), dtype=np.float32)
    polar = truth.copy(); polar[0] = 1
    equatorial = truth.copy(); equatorial[360] = 1
    assert 0 < spatial_mse(polar, truth, weights) < spatial_mse(equatorial, truth, weights)


def test_nonfinite_or_misaligned_fields_fail():
    with pytest.raises(ValueError):
        spatial_mse([[np.nan]], [[0]], [1])
    with pytest.raises(ValueError):
        spatial_mse(np.zeros((2, 3)), np.zeros((3, 2)), [1, 1])
    with pytest.raises(ValueError):
        paper_rmse([[1, -1]])


def test_array_identity_covers_dates_shape_and_orientation():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert array_sha(a) == array_sha(a.copy())
    assert array_sha(a) != array_sha(a[::-1])
    assert array_sha(a) != array_sha(a.reshape(3, 2))
    dates = np.array(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]")
    assert array_sha(dates) != array_sha(dates + np.timedelta64(1, "h"))
