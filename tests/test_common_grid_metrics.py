"""Scientific invariants for conservative common-grid verification."""
import numpy as np
import pytest
import xarray as xr

from scripts.analyze_models.common_grid_metrics import (
    DEFAULT_LAT,
    DEFAULT_LON,
    DIAGNOSTICS,
    regrid_diagnostics,
    score_step,
    select_diagnostics,
)


def _fields(lat, lon, values):
    values = np.broadcast_to(values, (len(lat), len(lon)))
    return xr.Dataset({key: (("lat", "lon"), values) for key in DIAGNOSTICS},
                      coords={"lat": lat, "lon": lon})


def _integral(ds, key="Z500"):
    # Independent finite-volume integral on a unit sphere, including polar caps.
    lat = ds.lat.values
    edges = np.r_[-90, (lat[:-1] + lat[1:]) / 2, 90]
    area = np.diff(np.sin(np.deg2rad(edges))) * (2 * np.pi / ds.sizes["lon"])
    return np.sum(ds[key].values * area[:, None])


@pytest.mark.parametrize("resolution", [0.25, 1.0])
def test_constant_is_preserved_at_actual_forecast_resolutions(resolution):
    lat = np.arange(-90.0, 90.0 + resolution / 2, resolution)
    lon = np.arange(0.0, 360.0, resolution)
    output = regrid_diagnostics(_fields(lat, lon, 3.25))
    assert dict(output.sizes) == {"lat": 121, "lon": 240}
    for key in DIAGNOSTICS:
        np.testing.assert_allclose(output[key], 3.25, rtol=0, atol=1e-12)


def test_spherical_integral_is_conserved_for_nonconstant_fields_and_polar_caps():
    lat, lon = np.arange(-90, 91, 1.0), np.arange(0, 360, 1.0)
    values = np.cos(np.deg2rad(lat[:, None])) ** 4 + np.cos(np.deg2rad(3 * lon[None, :]))
    values[0, :] += 1000  # A finite polar cell must retain its small, nonzero area.
    source = _fields(lat, lon, values)
    output = regrid_diagnostics(source)
    assert _integral(output) == pytest.approx(_integral(source), rel=2e-13, abs=1e-13)


def test_periodic_seam_averages_cells_on_both_sides_of_zero():
    lat, lon = np.arange(-90, 91, 1.0), np.arange(0, 360, 1.0)
    values = np.zeros((len(lat), len(lon)))
    values[:, 359], values[:, 0], values[:, 1] = 6, 3, 12
    output = regrid_diagnostics(_fields(lat, lon, values))
    # Target [-0.75,+0.75]: 0.25*6 + 1*3 + 0.25*12, divided by width 1.5.
    np.testing.assert_allclose(output.Z500.sel(lon=0), 5.0, atol=1e-12)


def test_latitude_order_and_longitude_convention_do_not_change_pairing():
    lat, lon = np.arange(-90, 91, 1.0), np.arange(0, 360, 1.0)
    values = lat[:, None] / 90 + np.sin(np.deg2rad(lon[None, :]))
    source = _fields(lat, lon, values)
    reversed_source = source.isel(lat=slice(None, None, -1), lon=slice(None, None, -1))
    reversed_source = reversed_source.assign_coords(lon=(reversed_source.lon + 180) % 360 - 180)
    expected = regrid_diagnostics(source)
    xr.testing.assert_allclose(regrid_diagnostics(reversed_source), expected)
    reversed_truth = expected.isel(lat=slice(None, None, -1), lon=slice(None, None, -1))
    assert all(value == pytest.approx(0, abs=1e-25)
               for value in score_step(source, reversed_truth).values())


def test_paired_mse_uses_truth_grid_and_cosine_weights_before_pooling():
    lat, lon = np.arange(-90, 91, 1.0), np.arange(0, 360, 1.0)
    source = _fields(lat, lon, 280.0)
    truth = regrid_diagnostics(source)
    error = np.broadcast_to(2 + np.sin(np.deg2rad(DEFAULT_LAT[:, None])), (121, 240))
    for key in DIAGNOSTICS:
        truth[key] = truth[key] - error
    weights = np.cos(np.deg2rad(DEFAULT_LAT))
    expected = np.sum(error[:, 0] ** 2 * weights) / weights.sum()
    result = score_step(source, truth)
    assert set(result) == set(DIAGNOSTICS)
    for mse in result.values():
        assert mse == pytest.approx(expected, rel=1e-13)
    assert score_step(source, regrid_diagnostics(source))["T2m"] == pytest.approx(0, abs=1e-25)


def test_native_fields_select_correct_levels_and_preserve_si_values():
    lat, lon, levels = [-90., 0., 90.], [0., 120., 240.], [50000., 70000., 85000.]
    pressure_values = np.broadcast_to(np.asarray(levels)[None, None, :, None, None], (1, 1, 3, 3, 3))
    variables = {}
    for spec in DIAGNOSTICS.values():
        variables[spec.variable] = (("batch", "time", "level", "latitude", "longitude"), pressure_values) if spec.level else (
            ("batch", "time", "latitude", "longitude"), np.full((1, 1, 3, 3), 17.))
    source = xr.Dataset(variables, coords={"latitude": lat, "longitude": lon, "level": levels,
                                         "batch": [0], "time": [np.timedelta64(6, "h")]})
    source.level.attrs["units"] = "Pa"
    output = select_diagnostics(source)
    for key, spec in DIAGNOSTICS.items():
        np.testing.assert_array_equal(output[key], spec.level * 100 if spec.level else 17)
        assert output[key].attrs["units"] == spec.units
    assert set(output.dims) == {"lat", "lon"}


@pytest.mark.parametrize("kind", ["irregular_lat", "partial_lat", "duplicate_lon", "partial_lon", "nonfinite"])
def test_incomplete_or_ambiguous_grids_and_fields_are_rejected(kind):
    source = _fields(np.arange(-90., 91.), np.arange(0., 360.), 0.).copy(deep=True)
    if kind == "irregular_lat":
        changed = source.lat.values.copy()
        changed[1] += 0.1
        source = source.assign_coords(lat=changed)
    elif kind == "partial_lat":
        source = source.isel(lat=slice(1, None))
    elif kind == "duplicate_lon":
        changed = source.lon.values.copy()
        changed[-1] = 0
        source = source.assign_coords(lon=changed)
    elif kind == "partial_lon":
        source = source.isel(lon=slice(1, None))
    else:
        source["Z500"].values[1, 1] = np.nan
    with pytest.raises(ValueError):
        regrid_diagnostics(source)


def test_multiple_forecast_steps_and_missing_diagnostics_are_rejected():
    source = _fields(DEFAULT_LAT, DEFAULT_LON, 0.)
    with pytest.raises(ValueError, match="one forecast step"):
        select_diagnostics(source.expand_dims(time=[0, 1]))
    with pytest.raises(ValueError, match="Missing diagnostic Q700"):
        select_diagnostics(source.drop_vars("Q700"))


def test_incompatible_physical_units_cannot_be_silently_relabeled():
    source = _fields(DEFAULT_LAT, DEFAULT_LON, 0.)
    source.MSL.attrs["units"] = "hPa"
    with pytest.raises(ValueError, match="expected SI units"):
        select_diagnostics(source)
    source.MSL.attrs["units"] = "Pa"
    source.Z500.attrs["units"] = "m**2 s**-2"
    assert select_diagnostics(source).Z500.attrs["units"] == "m2 s-2"
