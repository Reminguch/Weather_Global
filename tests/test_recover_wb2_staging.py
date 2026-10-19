import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.preprocessing.recover_wb2_staging import repair_year, write_year_bounded


def test_repair_rewrites_complete_year_without_changing_prefix(tmp_path):
    times = pd.date_range('1987-12-31', '1988-12-31T18:00', freq='6h')
    expected = xr.Dataset(
        {'temperature': (('time', 'lat', 'lon'),
                         np.arange(len(times) * 2, dtype=np.float32).reshape(-1, 1, 2)),
         'land_sea_mask': (('lat', 'lon'), np.ones((1, 2), np.float32))},
        coords={'time': times, 'lat': [0.], 'lon': [0., 1.]})
    output = tmp_path / 'source.zarr'
    damaged = expected.copy(deep=True)
    damaged.temperature.values[10:] = np.nan
    damaged.chunk({'time': 1}).to_zarr(output, zarr_format=2, consolidated=True)
    result = repair_year(expected.sel(time=slice('1988', '1988-12-31T18:00')).chunk({'time': 120}), output, 1988)
    assert result['start'] == 4 and result['stop'] == len(times)
    assert set(result['weather_sha256']) == {'temperature'}
    with xr.open_zarr(output, consolidated=True, chunks=None) as actual:
        xr.testing.assert_equal(actual, expected)


def test_repair_rejects_partial_year_and_multistep_storage(tmp_path):
    times = pd.date_range('1988-01-01', '1988-12-31T18:00', freq='6h')
    data = xr.Dataset({'weather': ('time', np.ones(len(times), np.float32))}, coords={'time': times})
    output = tmp_path / 'source.zarr'
    data.chunk({'time': 120}).to_zarr(output, zarr_format=2, consolidated=True)
    with pytest.raises(ValueError, match='one-timestep'):
        repair_year(data, output, 1988)
    with pytest.raises(AssertionError):
        repair_year(data.isel(time=slice(1, None)), output, 1988)


def test_repair_partial_final_year_and_bounded_progress(tmp_path):
    times = pd.date_range('1988-01-01', '1988-12-31T18:00', freq='6h')
    data = xr.Dataset({'weather': ('time', np.arange(len(times), dtype=np.float32))}, coords={'time': times})
    output = tmp_path / 'partial.zarr'
    data.isel(time=slice(0, 10)).chunk({'time': 1}).to_zarr(output, zarr_format=2, consolidated=True)
    progress = []
    result = repair_year(data.chunk({'time': 8}), output, 1988, block_steps=128,
                         progress=lambda done, total: progress.append((done, total)))
    assert result['stop'] == len(times) and progress[-1] == (len(times), len(times))
    assert all(b[0] - a[0] <= 128 for a, b in zip(progress, progress[1:]))
    with xr.open_zarr(output, consolidated=True, chunks=None) as actual:
        xr.testing.assert_equal(actual, data)


def test_bounded_append_rejects_nonfinite_before_write(tmp_path):
    data = xr.Dataset({'weather': ('time', np.array([1., 2., np.nan], dtype=np.float32))},
                      coords={'time': pd.date_range('2000-01-01', periods=3, freq='6h')})
    output = tmp_path / 'append.zarr'
    with pytest.raises(ValueError, match='Nonfinite'):
        write_year_bounded(data, output, mode='w', block_steps=2)
    with xr.open_zarr(output, consolidated=False, chunks=None) as actual:
        assert actual.sizes['time'] == 2
        np.testing.assert_array_equal(actual.weather.values, [1., 2.])
