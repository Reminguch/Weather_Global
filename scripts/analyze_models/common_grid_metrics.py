"""Conservative global remapping and physical MSE for the 2023 comparison.

Remapping treats grid values as cell averages, with midpoint cell boundaries
clipped at the poles and periodic longitude. Spherical cell area is used for
the remap. Verification then uses normalized cosine-of-center-latitude weights,
as in the paper, rather than the remapper's finite polar-cell areas. Returned
scores are MSE in SI units: pool over initializations before taking a square
root. No pressure, humidity, or geopotential unit conversions are applied.
"""
from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import numpy as np
import xarray as xr
from scipy import sparse


class Diagnostic(NamedTuple):
    variable: str
    level: int | None
    units: str


DIAGNOSTICS = {
    "Z500": Diagnostic("geopotential", 500, "m2 s-2"),
    "Z850": Diagnostic("geopotential", 850, "m2 s-2"),
    "T500": Diagnostic("temperature", 500, "K"),
    "T850": Diagnostic("temperature", 850, "K"),
    "T2m": Diagnostic("2m_temperature", None, "K"),
    "MSL": Diagnostic("mean_sea_level_pressure", None, "Pa"),
    "Q700": Diagnostic("specific_humidity", 700, "kg kg-1"),
    "Q850": Diagnostic("specific_humidity", 850, "kg kg-1"),
    "U850": Diagnostic("u_component_of_wind", 850, "m s-1"),
    "V850": Diagnostic("v_component_of_wind", 850, "m s-1"),
    "U10": Diagnostic("10m_u_component_of_wind", None, "m s-1"),
    "V10": Diagnostic("10m_v_component_of_wind", None, "m s-1"),
}
DEFAULT_LAT = np.arange(-90.0, 90.0 + 1.5, 1.5)
DEFAULT_LON = np.arange(0.0, 360.0, 1.5)


def _validate_units(field: xr.DataArray, key: str, expected: str) -> None:
    """Do not silently label Celsius, hPa, g/kg, or height as SI fields."""
    units = str(field.attrs.get("units", "")).strip()
    if not units:
        # GraphCast predictions commonly carry no unit attributes; their
        # physical variable names establish the GraphCast SI convention.
        return
    normalized = units.lower().replace(" ", "").replace("**", "").replace("^", "")
    accepted = {
        "m2 s-2": {"m2s-2", "m2/s2"},
        "K": {"k", "kelvin", "kelvins"},
        "Pa": {"pa", "pascal", "pascals"},
        "kg kg-1": {"kgkg-1", "kg/kg", "1", "dimensionless"},
        "m s-1": {"ms-1", "m/s"},
    }
    if normalized not in accepted[expected]:
        raise ValueError(f"{key}: expected SI units {expected!r}, got {units!r}; convert explicitly")


def _coordinate_names(ds: xr.Dataset) -> xr.Dataset:
    rename = {old: new for old, new in (("latitude", "lat"), ("longitude", "lon"))
              if old in ds.dims and new not in ds.dims}
    return ds.rename(rename)


def select_diagnostics(ds: xr.Dataset) -> xr.Dataset:
    """Extract all twelve fields, retaining SI values and selecting exact levels.

    Accept GraphCast/WeatherBench variable names or already extracted diagnostic
    names. Latitude/longitude aliases and singleton batch/time axes are accepted.
    Pressure coordinates use hPa unless explicitly labelled Pa. Missing fields,
    ambiguous non-singleton axes, and missing pressure levels fail explicitly.
    """
    ds = _coordinate_names(ds)
    fields = {}
    for key, spec in DIAGNOSTICS.items():
        if key in ds:
            field = ds[key]
        elif spec.variable in ds:
            field = ds[spec.variable]
            if spec.level is not None:
                level_names = [name for name in ("level", "pressure_level", "isobaricInhPa")
                               if name in field.dims]
                if len(level_names) != 1:
                    raise ValueError(f"{key}: expected one pressure-level dimension")
                level_name = level_names[0]
                coord = field[level_name]
                units = str(coord.attrs.get("units", "hPa")).lower()
                if units not in ("pa", "hpa", "millibar", "millibars", "mbar"):
                    raise ValueError(f"{key}: unsupported pressure-coordinate units {units!r}")
                target = spec.level * (100 if units == "pa" else 1)
                values = np.asarray(coord.values, dtype=np.float64)
                matches = np.flatnonzero(np.isclose(values, target, rtol=0, atol=1e-6))
                if matches.size != 1:
                    raise ValueError(f"{key}: missing or duplicate level {spec.level} hPa")
                field = field.isel({level_name: int(matches[0])}, drop=True)
        else:
            raise ValueError(f"Missing diagnostic {key} ({spec.variable})")
        extra = [dim for dim in field.dims if dim not in ("lat", "lon")]
        if any(field.sizes[dim] != 1 for dim in extra):
            raise ValueError(f"{key}: select one forecast step before scoring; dimensions={field.dims}")
        if extra:
            field = field.isel({dim: 0 for dim in extra}, drop=True)
        if set(field.dims) != {"lat", "lon"}:
            raise ValueError(f"{key}: expected latitude and longitude dimensions")
        _validate_units(field, key, spec.units)
        # Drop unrelated scalar time/level coordinates, which could otherwise
        # cause xarray to align fields to different forecast times silently.
        field = field.transpose("lat", "lon").reset_coords(drop=True)
        fields[key] = field.assign_attrs({
            **field.attrs, "units": spec.units, "source_variable": spec.variable,
            "pressure_level_hpa": "surface" if spec.level is None else spec.level,
        })
    return xr.Dataset(fields)


def _latitude_geometry(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Latitude must be a finite one-dimensional global regular grid")
    order = np.argsort(values)
    centers = values[order]
    spacing = np.diff(centers)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=0, atol=1e-6):
        raise ValueError("Latitude must be regular and unique")
    half = spacing[0] / 2
    if (centers[0] < -90 - 1e-6 or centers[-1] > 90 + 1e-6
            or centers[0] - half > -90 + 1e-6 or centers[-1] + half < 90 - 1e-6):
        raise ValueError("Latitude cells must cover the full sphere without exceeding the poles")
    edges = np.concatenate(([-90.0], (centers[:-1] + centers[1:]) / 2, [90.0]))
    return centers, np.sin(np.deg2rad(edges)), order


def _longitude_geometry(values: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Longitude must be a finite one-dimensional global periodic grid")
    normalized = np.mod(values, 360.0)
    order = np.argsort(normalized)
    centers = normalized[order]
    spacing = 360.0 / len(centers)
    gaps = np.diff(np.concatenate((centers, [centers[0] + 360.0])))
    if not np.allclose(gaps, spacing, rtol=0, atol=1e-6):
        raise ValueError("Longitude must be regular, unique, and cover one complete period")
    return centers, spacing, order


@lru_cache(maxsize=16)
def _operators(source_lat: tuple, source_lon: tuple, target_lat: tuple, target_lon: tuple):
    _, source_edges, source_lat_order = _latitude_geometry(np.asarray(source_lat))
    _, target_edges, target_lat_order = _latitude_geometry(np.asarray(target_lat))
    source_centers, source_width, source_lon_order = _longitude_geometry(np.asarray(source_lon))
    target_centers, target_width, target_lon_order = _longitude_geometry(np.asarray(target_lon))
    lat_overlap = np.maximum(
        0.0, np.minimum(target_edges[1:, None], source_edges[None, 1:])
        - np.maximum(target_edges[:-1, None], source_edges[None, :-1]),
    )
    lat_weights = sparse.csr_matrix(lat_overlap / np.diff(target_edges)[:, None])
    lon_overlap = np.zeros((len(target_centers), len(source_centers)), dtype=np.float64)
    for shift in (-360.0, 0.0, 360.0):
        lon_overlap += np.maximum(
            0.0,
            np.minimum(target_centers[:, None] + target_width / 2,
                       source_centers[None, :] + shift + source_width / 2)
            - np.maximum(target_centers[:, None] - target_width / 2,
                         source_centers[None, :] + shift - source_width / 2),
        )
    lon_weights = sparse.csr_matrix(lon_overlap / target_width)
    if (not np.allclose(np.asarray(lat_weights.sum(axis=1)).ravel(), 1, rtol=0, atol=1e-10)
            or not np.allclose(np.asarray(lon_weights.sum(axis=1)).ravel(), 1, rtol=0, atol=1e-10)):
        raise ValueError("Remapping weights do not cover every target cell")
    return (lat_weights, lon_weights, source_lat_order, source_lon_order,
            np.argsort(target_lat_order), np.argsort(target_lon_order))


def regrid_diagnostics(
    ds: xr.Dataset, target_lat: np.ndarray | None = None, target_lon: np.ndarray | None = None,
) -> xr.Dataset:
    """Conservatively remap twelve diagnostics to a complete regular global grid.

    Default output is 121 x 240, with latitude -90..90 and longitude 0..358.5.
    Custom target coordinates retain their supplied ordering. Source longitudes
    may use either convention and latitudes may ascend or descend.
    """
    fields = select_diagnostics(ds)
    target_lat = DEFAULT_LAT if target_lat is None else np.asarray(target_lat, dtype=np.float64)
    target_lon = DEFAULT_LON if target_lon is None else np.asarray(target_lon, dtype=np.float64)
    operators = _operators(tuple(fields.lat.values), tuple(fields.lon.values),
                           tuple(target_lat), tuple(target_lon))
    lat_weights, lon_weights, source_lat_order, source_lon_order, undo_lat, undo_lon = operators
    output = {}
    for key in DIAGNOSTICS:
        values = np.asarray(fields[key].values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{key}: nonfinite field values cannot be conservatively remapped")
        ordered = values[np.ix_(source_lat_order, source_lon_order)]
        remapped = lon_weights.dot(lat_weights.dot(ordered).T).T
        remapped = remapped[np.ix_(undo_lat, undo_lon)]
        output[key] = xr.DataArray(remapped, dims=("lat", "lon"),
                                   coords={"lat": target_lat, "lon": target_lon},
                                   attrs=fields[key].attrs)
    return xr.Dataset(output, attrs={
        "remapping": "first_order_conservative_spherical_cell_overlap",
        "cell_boundaries": "coordinate midpoints, clipped at poles; periodic longitude",
        "spatial_weighting": "normalized_cosine_latitude_for_MSE",
    })


def score_step(prediction: xr.Dataset, common_truth: xr.Dataset) -> dict[str, float]:
    """Return twelve paired common-grid, cosine-latitude-weighted physical MSEs.

    The caller must select matching valid times. Truth must already be on the
    final common grid; prediction is remapped to those exact coordinates. Both
    operands are converted to float64 before subtraction and squaring.
    """
    truth = select_diagnostics(common_truth)
    prediction = regrid_diagnostics(prediction, truth.lat.values, truth.lon.values)
    weights = np.maximum(np.cos(np.deg2rad(truth.lat.values.astype(np.float64))), 0.0)
    weights /= weights.sum()
    scores = {}
    for key in DIAGNOSTICS:
        actual = np.asarray(truth[key].values, dtype=np.float64)
        if not np.isfinite(actual).all():
            raise ValueError(f"{key}: nonfinite verification truth")
        error = np.asarray(prediction[key].values, dtype=np.float64) - actual
        mse = float(np.dot(weights, np.mean(np.square(error), axis=1, dtype=np.float64)))
        if not np.isfinite(mse):
            raise ValueError(f"{key}: nonfinite verification MSE")
        scores[key] = mse
    return scores
