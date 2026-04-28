"""Legacy shared normalized analysis metrics for analysis scripts."""

from __future__ import annotations

import numpy as np
import xarray

try:
    from graphcast import losses as gc_losses
except Exception:  # pragma: no cover - imported only inside the GraphCast env.
    gc_losses = None


GRAPHCAST_PER_VARIABLE_WEIGHTS: dict[str, float] = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


def latitude_weights_with_fallback(data: xarray.DataArray) -> xarray.DataArray:
    if gc_losses is not None:
        try:
            return gc_losses.normalized_latitude_weights(data)
        except Exception:
            pass

    latitude = data.coords["lat"]
    lat_vals = np.asarray(latitude.values, dtype=np.float64)
    if lat_vals.ndim != 1 or lat_vals.size == 0:
        raise ValueError("Latitude coordinate must be one-dimensional and non-empty.")
    if lat_vals.size == 1:
        return xarray.DataArray(np.ones(1, dtype=np.float32), coords=latitude.coords, dims=latitude.dims)

    edges = np.empty(lat_vals.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (lat_vals[:-1] + lat_vals[1:])
    edges[0] = lat_vals[0] - (lat_vals[1] - lat_vals[0]) / 2.0
    edges[-1] = lat_vals[-1] + (lat_vals[-1] - lat_vals[-2]) / 2.0
    edges = np.clip(edges, -90.0, 90.0)
    weights_np = np.abs(np.sin(np.deg2rad(edges[:-1])) - np.sin(np.deg2rad(edges[1:])))
    weights = xarray.DataArray(weights_np, coords=latitude.coords, dims=latitude.dims).astype(np.float32)
    return weights / weights.mean(skipna=False)


def normalized_per_variable_mse(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    *,
    use_latitude_weights: bool,
    diffs_stddev_by_level: xarray.Dataset | None = None,
) -> dict[str, xarray.DataArray]:
    """Return one normalized loss tensor per variable, reduced to batch-only."""
    losses: dict[str, xarray.DataArray] = {}
    for name, target in targets.data_vars.items():
        if name not in predictions:
            continue
        prediction = predictions[name]
        loss = (prediction - target) ** 2
        if diffs_stddev_by_level is not None and name in diffs_stddev_by_level:
            scale = diffs_stddev_by_level[name].astype(loss.dtype)
            loss = loss / (scale ** 2)
        if use_latitude_weights and "lat" in loss.dims:
            lat_w = latitude_weights_with_fallback(target).astype(loss.dtype)
            loss = loss.weighted(lat_w).mean("lat", skipna=False)
        if "lon" in loss.dims:
            loss = loss.mean("lon", skipna=False)
        if "level" in loss.dims:
            loss = loss.mean("level", skipna=False)
        reduce_dims = [dim for dim in loss.dims if dim not in ("batch",)]
        if reduce_dims:
            loss = loss.mean(reduce_dims, skipna=False)
        losses[name] = loss

    if not losses:
        raise ValueError("No overlapping prediction/target variables found.")
    return losses


def normalized_weighted_mse_allvars(
    predictions: xarray.Dataset,
    targets: xarray.Dataset,
    *,
    per_variable_weights: dict[str, float],
    use_latitude_weights: bool,
    diffs_stddev_by_level: xarray.Dataset | None = None,
) -> xarray.DataArray:
    per_var_losses = normalized_per_variable_mse(
        predictions,
        targets,
        use_latitude_weights=use_latitude_weights,
        diffs_stddev_by_level=diffs_stddev_by_level,
    )
    per_var_weights = {
        name: float(per_variable_weights.get(name, 1.0))
        for name in per_var_losses
    }
    total_var_weight = float(np.sum(list(per_var_weights.values())))
    if total_var_weight <= 0.0:
        raise ValueError("Total variable weight must be positive.")

    weighted_losses = [loss * per_var_weights[name] for name, loss in per_var_losses.items()]
    return xarray.concat(weighted_losses, dim="variable", join="exact").sum("variable", skipna=False) / total_var_weight
