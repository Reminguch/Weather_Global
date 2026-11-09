import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from src.analysis.spherical_smoothing import (
    ChannelLayout, SphericalAttenuator, VARIABLES, band_hats, fit_band, fit_filters,
    gaussian_gains, lead_group,
)
from src.analysis.smoothing_observer import SmoothingObserver


@pytest.fixture(scope="module")
def operator():
    return SphericalAttenuator(np.linspace(-90, 90, 9), np.arange(16) * 22.5,
                              lmax=5, knots=(1, 2, 3, 5))


def fields(operator):
    harmonic = np.sin(np.deg2rad(operator.latitude))[:, None] * np.ones((1, 16))
    residual = .3 * (-1.)**np.arange(16)[None] * np.cos(np.deg2rad(operator.latitude))[:, None]
    return harmonic, residual


def test_hats_boundaries_and_lead_groups():
    hats = band_hats()
    np.testing.assert_array_equal(hats[:, 0], 0)
    np.testing.assert_allclose(hats[:, 1:].sum(axis=0), 1, atol=1e-15)
    np.testing.assert_array_equal(hats[:, [1, 10, 40, 179]], np.eye(4))
    assert [lead_group(h) for h in (6, 24, 30, 72, 78, 168, 174, 240)] == [0,0,1,1,2,2,3,3]
    np.testing.assert_array_equal(gaussian_gains()[0], 1)


def test_mean_identity_and_unresolved_residual(operator):
    harmonic, unresolved = fields(operator)
    forecast = (100000 + 20 * harmonic + unresolved)[None]
    identity = operator.apply(forecast, np.ones((1, 6)))
    np.testing.assert_array_equal(identity, forecast)
    gains = np.full((1, 6), .4); gains[:, 0] = 1
    filtered = operator.apply(forecast, gains)
    np.testing.assert_allclose(operator.mean(filtered), operator.mean(forecast), atol=2e-10, rtol=0)
    np.testing.assert_allclose(filtered, (100000 + 8 * harmonic + unresolved)[None], atol=2e-6, rtol=0)
    np.testing.assert_allclose(operator.apply(np.ones_like(forecast)*100000, gains), 100000, atol=1e-10)


def test_quadratic_statistics_equal_direct_grid_error(operator):
    harmonic, residual = fields(operator)
    forecast = np.stack((3 + 2*harmonic + residual, 10 - 3*harmonic - residual))
    truth = np.stack((3 + harmonic, 11 - harmonic))
    scales = np.array([2., .3])
    stats = operator.calibration_chunk(forecast, truth, scales)
    gains = np.array([[.8,.4,.3,.7], [.1,.6,.9,.3]])
    d = gains - 1
    quadratic = stats['e0'] + 2*np.einsum('ci,ci->c', stats['cross'], d) + np.einsum('ci,cij,cj->c', d, stats['gram'], d)
    pair = operator.filtered_pair(forecast, scales, [3,5], gains)
    direct = operator.mean(((pair[1]-truth)/scales[:,None,None])**2)
    np.testing.assert_allclose(quadratic, direct, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(stats['gaussian'][np.arange(2), [3,5]],
                               operator.mean(((pair[0]-truth)/scales[:,None,None])**2), rtol=2e-6)


def test_band_fit_and_finite_nonconvergence_warning(monkeypatch):
    # Independent unit-variance bases; truth is .4 times each basis.
    cross = np.full(4, .6)
    gains, info = fit_band(4*.6**2, cross, np.eye(4))
    expected = 1 - .6 / (1 + .01 * (4*.6**2) / 4)
    np.testing.assert_allclose(gains, expected, atol=1e-8)
    assert info['objective'] < 1
    np.testing.assert_array_equal(fit_band(0, np.zeros(4), np.zeros((4,4)))[0], 1)
    monkeypatch.setattr('src.analysis.spherical_smoothing.minimize',
        lambda *a, **kw: SimpleNamespace(x=np.full(4,.5), fun=.1, success=False, message='finite limit', nit=1000))
    gains, info = fit_band(1.44, cross, np.eye(4))
    assert info['status'] == 'convergence_warning'
    assert np.all(gains == .5)
    with pytest.raises(ValueError, match='Nonfinite|nonfinite'):
        fit_band(float('nan'), cross, np.eye(4))


def all_channels_dataset():
    levels = np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000])
    upper = {'temperature','geopotential','u_component_of_wind','v_component_of_wind','vertical_velocity','specific_humidity'}
    coords = {'batch':[0], 'time':[np.timedelta64(6,'h')], 'lat':np.linspace(-90,90,9),
              'lon':np.arange(16)*22.5, 'level':levels}
    data, std = {}, {}
    rng = np.random.default_rng(8)
    for name in VARIABLES:
        dims = ('batch','time','level','lat','lon') if name in upper else ('batch','time','lat','lon')
        shape = tuple(len(coords[d]) for d in dims)
        data[name] = (dims, rng.normal(size=shape).astype('float32'))
        std[name] = (('level',), np.linspace(.5,2,len(levels))) if name in upper else ((), 1.7)
    return xr.Dataset(data,coords=coords), xr.Dataset(std,coords={'level':levels})


def test_all83_channels_tied_groups_and_exact_graphcast_metric(operator):
    from src.models.mamba.v24_Ilya.metrics import V24IlyaMetricAccumulator
    truth, scales = all_channels_dataset()
    layout = ChannelLayout.from_dataset(truth, scales)
    assert len(layout.channels) == 83 and len(layout.groups) == 69
    for level in truth.level.values:
        u = layout.channels.index(('u_component_of_wind',level))
        v = layout.channels.index(('v_component_of_wind',level))
        assert layout.group_index[u] == layout.group_index[v]
    forecast = truth*1.3
    error = layout.pack(forecast) - layout.pack(truth)
    own = np.sum(operator.mean(error**2) * layout.weights/layout.scales**2)
    accumulator = V24IlyaMetricAccumulator(1, truth.lat.values, diffs_stddev_by_level=scales,
                                         store_spatial_bias=False)
    values = accumulator.update_step(0, truth, forecast, forecast)
    np.testing.assert_allclose(own, np.asarray(values['full']).mean(), rtol=2e-6)


def test_spectral_physical_mean_precision_and_closure(operator):
    harmonic, residual = fields(operator)
    truth = (100000 + harmonic)[None]
    forecast = (100000.001 + .8*harmonic + residual)[None]
    values = operator.spectral_chunk(forecast, truth)
    assert values['error_power'][0,0] == pytest.approx(1e-6, rel=2e-5)
    np.testing.assert_allclose(values['error_power'], values['prediction_power']+values['truth_power']-2*values['cross_power'], atol=4e-6)
    np.testing.assert_allclose(values['grid_mse'], values['projected_grid_mse']+values['unresolved_grid_mse'], rtol=2e-6)


def test_observer_calibration_evaluation_and_no_mutation(tmp_path, operator):
    harmonic, _ = fields(operator)
    truth = xr.Dataset({'2m_temperature':(('lat','lon'), harmonic)},
                       coords={'lat':operator.latitude,'lon':operator.longitude})
    forecast = truth * 1.4
    original = forecast.copy(deep=True)
    scales = xr.Dataset({'2m_temperature':xr.DataArray(1.)})
    metadata = {'initialization_id':'20220101T000000Z', 'initialization_time':'2022-01-01T00:00:00Z', 'lead_hours':[6]}
    arguments = dict(target_steps=1,lmax=5,knots=(1,2,3,5),expected_channels=1,compute_zonal=False)
    calibration = SmoothingObserver(tmp_path/'cal', scales, {}, mode='calibrate',**arguments)
    calibration.begin_sample(metadata)
    calibration.update_step(0, truth, forecast, forecast)
    calibration.finish_sample()
    with np.load(tmp_path/'cal/20220101T000000Z.npz') as archive:
        layout = json.loads(str(archive['metadata']))['layout']
        stats = {k:archive[k] for k in ('e0','cross','gram','gaussian')}
    filters = fit_filters(stats,layout)
    observer = SmoothingObserver(tmp_path/'eval', scales, {}, mode='evaluate',filters=filters,**arguments)
    observer.begin_sample(metadata)
    observer.update_step(0, truth, forecast, forecast)
    observer.finish_sample()
    xr.testing.assert_identical(forecast,original)
    with np.load(tmp_path/'eval/20220101T000000Z.npz') as archive:
        assert archive['exact_gc_loss'].shape == (1,2,3)
        assert archive['exact_gc_loss'][0,0,2] < archive['exact_gc_loss'][0,0,0]
    with pytest.raises(FileExistsError):
        observer.begin_sample(metadata)
