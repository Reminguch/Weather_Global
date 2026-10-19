"""Coordinate compatibility must distinguish platform roundoff from wrong grids."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.models.neuralgcm_residual.runtime import validate_data_grid


def grid():
    longitude = np.arange(128) * (2*np.pi/128)
    latitude = np.arcsin(np.polynomial.legendre.leggauss(64)[0])
    levels = np.array([1, 50, 100, 250, 500, 850, 1000])
    coordinates = SimpleNamespace(horizontal=SimpleNamespace(longitudes=longitude, latitudes=latitude),
                                  vertical=SimpleNamespace(centers=levels))
    prepared = dict(longitude=longitude.tolist(), latitude=latitude.tolist(), levels=levels.tolist())
    return prepared, coordinates


def test_matching_grid_and_float64_platform_roundoff():
    prepared, coordinates = grid()
    validate_data_grid(prepared, coordinates)
    prepared['latitude'][30] = np.nextafter(prepared['latitude'][30], np.inf)
    prepared['longitude'][90] = np.nextafter(prepared['longitude'][90], np.inf)
    validate_data_grid(prepared, coordinates)


@pytest.mark.parametrize('fault', ['reversed', 'degrees', 'shifted', 'shape', 'nonfinite',
                                 'pressure', 'pressure_roundoff', 'missing', 'extra', 'nested'])
def test_rejects_real_mismatches(fault):
    prepared, coordinates = grid()
    if fault == 'reversed': prepared['latitude'].reverse()
    elif fault == 'degrees': prepared['latitude'] = np.rad2deg(prepared['latitude']).tolist()
    elif fault == 'shifted': prepared['longitude'][17] += 1e-10
    elif fault == 'shape': prepared['latitude'].pop()
    elif fault == 'nonfinite': prepared['latitude'][17] = np.nan
    elif fault == 'pressure': prepared['levels'][2] += 1
    elif fault == 'pressure_roundoff': prepared['levels'][2] += 1e-10
    elif fault == 'missing': del prepared['levels']
    elif fault == 'extra': prepared['other'] = []
    elif fault == 'nested': prepared['longitude'] = [prepared['longitude']]
    with pytest.raises(ValueError, match='Prepared data grid'):
        validate_data_grid(prepared, coordinates)
