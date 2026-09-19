"""FP32 full-field objectives for the matched MSE/AMSE residual experiment.

AMSE is equation (6) of https://arxiv.org/html/2501.19374v2. Each variable,
pressure level, batch member and time is scored separately; only afterwards
are time/level losses averaged and variables summed, with GraphCast weights.
Inputs are physical FULL predictions and truth, never the correction alone.

The spherical transform is a longitude FFT followed by a normalized associated
Legendre analysis. On the regular, pole-inclusive 181 x 360 grid the default is
triangular T179: total degrees 0..179, including the mean, excluding longitude
Nyquist. Latitude analysis is an area-weighted least-squares fit, computed once
in float64 with QR, then applied in FP32. This recovers bandlimited harmonics
through degree 179; naive latitude quadrature at this resolution only integrates
products exactly through about degree 90. This is a spherical transform, not a
zonal power diagnostic. Its normalized coefficients obey continuous spherical
Parseval for the fitted field. They do NOT obey exact discrete GraphCast-cell
Parseval for arbitrary grid values. Unresolved components (including longitude
Nyquist and longitude-varying pole values) are outside this spectral objective;
the exact grid MSE is always reported separately. Changing lmax changes the
objective and must be recorded in experiment metadata.

AMSE uses a tiny fixed power floor implemented as an extra shared coefficient
sqrt(epsilon) in every degree. This resolves undefined coherence at zero power,
keeps gradients finite and gives exactly zero loss for identical coefficients.
It converges to the paper's formula as epsilon tends to zero. There is no
auxiliary spectral mixing coefficient. Computation uses float32 / complex64.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from graphcast import xarray_jax
import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg
import xarray as xr

from ..metrics import ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS


def latitude_area_weights(latitudes: np.ndarray) -> np.ndarray:
    """GraphCast's exact pole-cell areas, normalized to sum one."""
    latitudes = np.asarray(latitudes, dtype=np.float64)
    if latitudes.ndim != 1 or len(latitudes) < 3:
        raise ValueError("Spherical loss needs at least three latitude rows")
    differences = np.diff(latitudes)
    if not np.allclose(differences, differences[0], atol=1e-6):
        raise ValueError("Spherical loss requires uniformly spaced latitudes")
    if not np.allclose(np.sort(latitudes[[0, -1]]), [-90, 90], atol=1e-6):
        raise ValueError("Spherical loss requires both poles in the latitude grid")
    delta = abs(float(differences[0]))
    weights = np.cos(np.deg2rad(latitudes)) * np.sin(np.deg2rad(delta / 2))
    weights[[0, -1]] = np.sin(np.deg2rad(delta / 4)) ** 2
    return weights / weights.sum()


def _legendre_basis(latitudes: np.ndarray, lmax: int) -> np.ndarray:
    """Return p[l,m,j] with integral p_lm**2 dx / 2 = 1."""
    x = np.sin(np.deg2rad(np.asarray(latitudes, dtype=np.float64)))
    p = np.zeros((lmax + 1, lmax + 1, len(x)), dtype=np.float64)
    p[0, 0] = 1
    sin_colatitude = np.sqrt(np.maximum(0, 1 - x * x))
    for m in range(lmax + 1):
        if m:
            p[m, m] = -np.sqrt((2 * m + 1) / (2 * m)) * sin_colatitude * p[m - 1, m - 1]
        if m < lmax:
            p[m + 1, m] = np.sqrt(2 * m + 3) * x * p[m, m]
        for ell in range(m + 2, lmax + 1):
            a = np.sqrt((4 * ell**2 - 1) / (ell**2 - m**2))
            b = np.sqrt(((2 * ell + 1) * ((ell - 1)**2 - m**2)) /
                        ((2 * ell - 3) * (ell**2 - m**2)))
            p[ell, m] = a * x * p[ell - 1, m] - b * p[ell - 2, m]
    return p


@dataclass(frozen=True)
class SphericalTransform:
    """Static geometry; arrays supplied to analysis have shape (..., lat, lon)."""

    analysis_weights: np.ndarray  # (m, ell, latitude)
    nlon: int
    lmax: int
    maximum_condition_number: float

    def analysis(self, field: jax.Array) -> jax.Array:
        if field.shape[-2:] != (self.analysis_weights.shape[-1], self.nlon):
            raise ValueError("Field shape does not match spherical transform grid")
        fourier = jnp.fft.rfft(jnp.asarray(field, dtype=jnp.float32), axis=-1) / self.nlon
        fourier = fourier[..., :self.lmax + 1]
        weights = jnp.asarray(self.analysis_weights, dtype=jnp.float32)
        # Explicit real GEMMs avoid silently lowering precision of complex dots.
        real = jnp.einsum("mlj,...jm->...lm", weights, fourier.real,
                          precision=jax.lax.Precision.HIGHEST)
        imag = jnp.einsum("mlj,...jm->...lm", weights, fourier.imag,
                          precision=jax.lax.Precision.HIGHEST)
        return jax.lax.complex(real, imag)


@lru_cache(maxsize=8)
def _cached_transform(latitudes: tuple[float, ...], nlon: int,
                      lmax: int) -> SphericalTransform:
    latitude = np.asarray(latitudes, dtype=np.float64)
    area = latitude_area_weights(latitude)
    limit = min(len(latitude) - 2, (nlon - 1) // 2)
    if not 0 <= lmax <= limit:
        raise ValueError(f"lmax must be between 0 and {limit} for this grid; got {lmax}")
    basis = _legendre_basis(latitude, lmax)
    analysis = np.zeros_like(basis)
    sqrt_area = np.sqrt(area)
    max_condition = 0.0
    for m in range(lmax + 1):
        design = basis[m:, m].T * sqrt_area[:, None]
        # Thin QR is stable here; no normal-equation inverse is formed.
        q, r = linalg.qr(design, mode="economic", check_finite=False)
        singular_values = linalg.svdvals(r, check_finite=False)
        condition = float(singular_values[0] / singular_values[-1])
        max_condition = max(max_condition, condition)
        if condition > 100:
            raise ValueError(f"Ill-conditioned spherical latitude fit: m={m}, cond={condition}")
        analysis[m, m:] = linalg.solve_triangular(
            r, q.T * sqrt_area[None, :], check_finite=False)
    return SphericalTransform(analysis.astype(np.float32), nlon, lmax, max_condition)


def build_spherical_transform(latitudes: np.ndarray, longitudes: np.ndarray,
                              lmax: int | None = None) -> SphericalTransform:
    """Build/cache a full-sphere T(lmax) transform for the supplied regular grid."""
    longitudes = np.asarray(longitudes, dtype=np.float64)
    if longitudes.ndim != 1 or len(longitudes) < 4:
        raise ValueError("Spherical loss needs at least four longitude columns")
    delta = np.diff(longitudes)
    if not np.allclose(delta, 360.0 / len(longitudes), atol=1e-6):
        raise ValueError("Longitudes must increase uniformly around a complete circle without duplicate endpoint")
    # A longitude-origin change rotates each m equally in both fields, leaving
    # degree powers and cross coherence unchanged.
    if lmax is None:
        lmax = min(len(latitudes) - 2, (len(longitudes) - 1) // 2)
    return _cached_transform(tuple(float(x) for x in latitudes), len(longitudes), int(lmax))


def amse_per_degree(prediction: jax.Array, truth: jax.Array,
                    power_epsilon: float = 1e-12) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Score (..., degree, nonnegative-order) complex coefficients.

    The omitted negative orders have identical power, hence multiplicity two
    except for m=0. The tiny shared regularization coefficient makes the
    normalized-vector distance a stable evaluation of 2*(1-coherence).
    """
    if power_epsilon <= 0 or not np.isfinite(power_epsilon):
        raise ValueError("power_epsilon must be finite and positive")
    prediction = jnp.asarray(prediction, dtype=jnp.complex64)
    truth = jnp.asarray(truth, dtype=jnp.complex64)
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and truth coefficients must have matching shapes")
    multiplicity = jnp.where(jnp.arange(prediction.shape[-1]) == 0, 1.0, 2.0)

    def power(value):
        return jnp.sum(multiplicity * (value.real**2 + value.imag**2), axis=-1)

    p_prediction = power(prediction)
    p_truth = power(truth)
    a_prediction = jnp.sqrt(p_prediction + power_epsilon)
    a_truth = jnp.sqrt(p_truth + power_epsilon)
    amplitude = (a_prediction - a_truth)**2
    normalized_difference = prediction / a_prediction[..., None] - truth / a_truth[..., None]
    twice_decorrelation = power(normalized_difference) + power_epsilon * (
        1 / a_prediction - 1 / a_truth)**2
    pattern = jnp.maximum(p_prediction + power_epsilon, p_truth + power_epsilon) * twice_decorrelation
    return amplitude + pattern, {
        "amplitude": amplitude,
        "pattern": pattern,
        "spectral_mse": power(prediction - truth),
    }


def _normalization_scales(stats: Mapping[str, xr.Dataset] | xr.Dataset) -> xr.Dataset:
    if isinstance(stats, xr.Dataset):
        return stats
    return stats["diffs_stddev_by_level"]


def build_full_field_loss(stats: Mapping[str, xr.Dataset] | xr.Dataset,
                          loss_kind: str, *, lmax: int | None = None,
                          power_epsilon: float = 1e-12,
                          per_variable_weights: Mapping[str, float] | None = None):
    """Return (full_prediction, truth) -> (FP32 batch loss, batch diagnostics).

    ``mse_fp32`` is GraphCast's grid MSE in FP32. ``amse`` directly computes the
    spherical objective. Stats may be the core loader's dict or its difference
    standard-deviation Dataset. Diagnostics also include the matched grid MSE.
    """
    if loss_kind not in ("mse_fp32", "amse"):
        raise ValueError(f"Unknown full-field loss kind {loss_kind!r}")
    if power_epsilon <= 0 or not np.isfinite(power_epsilon):
        raise ValueError("power_epsilon must be finite and positive")
    scales = _normalization_scales(stats)
    variable_weights = dict(ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS if per_variable_weights is None
                            else per_variable_weights)

    def loss(predictions: xr.Dataset, targets: xr.Dataset):
        if set(predictions.data_vars) != set(targets.data_vars):
            raise ValueError("Full prediction and truth variable sets must match")
        latitude = np.asarray(targets.lat)
        longitude = np.asarray(targets.lon)
        for coord in ("lat", "lon"):
            if not np.array_equal(np.asarray(predictions[coord]), np.asarray(targets[coord])):
                raise ValueError(f"Prediction and truth {coord} coordinates differ")
        area = jnp.asarray(latitude_area_weights(latitude), dtype=jnp.float32)
        transform = (build_spherical_transform(latitude, longitude, lmax)
                     if loss_kind == "amse" else None)
        packed_predictions, packed_truths = [], []
        variable_layout = []
        diagnostics = {}
        grid_total = jnp.zeros(targets.sizes["batch"], dtype=jnp.float32)
        offset = 0
        for name, target in targets.data_vars.items():
            prediction = predictions[name]
            if set(prediction.dims) != set(target.dims):
                raise ValueError(f"Prediction/truth dimensions differ for {name}")
            if name not in scales:
                raise ValueError(f"Missing difference standard deviation for {name}")
            extra_dims = tuple(d for d in target.dims if d not in ("batch", "lat", "lon"))
            order = ("batch", *extra_dims, "lat", "lon")
            if any(prediction.sizes[d] != target.sizes[d] for d in order):
                raise ValueError(f"Prediction/truth sizes differ for {name}")
            for dim in extra_dims:
                if dim in target.coords and dim in prediction.coords:
                    if not np.array_equal(np.asarray(target[dim]), np.asarray(prediction[dim])):
                        raise ValueError(f"Prediction/truth {dim} coordinates differ for {name}")
            scale_da = scales[name]
            if any(d != "level" for d in scale_da.dims):
                raise ValueError(f"Difference standard deviation for {name} must be scalar or level-only")
            if "level" in scale_da.dims:
                scale_da = scale_da.sel(level=target.level)
            scale = np.asarray(scale_da, dtype=np.float32)
            if not np.all(np.isfinite(scale) & (scale > 0)):
                raise ValueError(f"Invalid difference standard deviation for {name}")
            scale = scale.reshape(tuple(target.sizes[d] if d in scale_da.dims else 1 for d in order))
            pred = jnp.asarray(xarray_jax.unwrap_data(prediction.transpose(*order)), dtype=jnp.float32)
            true = jnp.asarray(xarray_jax.unwrap_data(target.transpose(*order)), dtype=jnp.float32)
            error = (pred - true) / scale
            grid_channels = jnp.sum(jnp.mean(error**2, axis=-1) * area, axis=-1)
            channel_shape = tuple(target.sizes[d] for d in extra_dims)
            channel_weights = np.ones(channel_shape or (), dtype=np.float32)
            if "level" in extra_dims:
                level = np.asarray(target.level, dtype=np.float32)
                if not np.all(np.isfinite(level) & (level > 0)):
                    raise ValueError("Pressure levels must be finite and positive")
                shape = tuple(len(level) if d == "level" else 1 for d in extra_dims)
                channel_weights = channel_weights * (level / level.mean()).reshape(shape)
            channel_weights = jnp.asarray(channel_weights.reshape(-1), dtype=jnp.float32)
            channels = channel_weights.shape[0]
            variable_weight = variable_weights.get(name, 1.0)
            grid_loss = jnp.mean(grid_channels.reshape((-1, channels)) * channel_weights, axis=-1)
            diagnostics[f"{name}_mse_fp32"] = grid_loss
            grid_total = grid_total + variable_weight * grid_loss
            if transform is not None:
                packed_shape = (target.sizes["batch"], channels, len(latitude), len(longitude))
                packed_predictions.append((pred / scale).reshape(packed_shape))
                packed_truths.append((true / scale).reshape(packed_shape))
                variable_layout.append((name, offset, offset + channels, channel_weights, variable_weight))
                offset += channels
            else:
                diagnostics[name] = grid_loss
        diagnostics["grid_mse_fp32"] = grid_total
        if transform is None:
            return grid_total, diagnostics
        # Transform all variables/levels together so there is one batched latitude
        # GEMM per field, rather than a separate large constant for every variable.
        pred_coefficients = transform.analysis(jnp.concatenate(packed_predictions, axis=1))
        true_coefficients = transform.analysis(jnp.concatenate(packed_truths, axis=1))
        per_degree, components = amse_per_degree(pred_coefficients, true_coefficients, power_epsilon)
        channels_loss = jnp.sum(per_degree, axis=-1)
        total = jnp.zeros_like(grid_total)
        component_totals = {key: jnp.zeros_like(grid_total) for key in components}
        for name, start, end, channel_weights, variable_weight in variable_layout:
            variable_loss = jnp.mean(channels_loss[:, start:end] * channel_weights, axis=-1)
            diagnostics[name] = variable_loss
            total = total + variable_weight * variable_loss
            for key, values in components.items():
                value = jnp.mean(jnp.sum(values[:, start:end], axis=-1) * channel_weights, axis=-1)
                component_totals[key] = component_totals[key] + variable_weight * value
        diagnostics.update(component_totals)
        return total, diagnostics

    return loss
