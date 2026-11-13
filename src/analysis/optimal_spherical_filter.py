"""GraphCast-style, degree-wise least-squares spherical output filtering.

This implements the filter class in GraphCast Supplement 7.4.2, adapted to
the repository's normalized QR spherical transform and resolution.  Each
forecast lead and physical variable/pressure channel is fitted independently;
there are no gain bounds, degree/lead pooling, wind-component ties or ridge.

The transform stores nonnegative orders only.  Its basis has continuous
spherical mean-square norm one, so m>0 receives multiplicity two and the sum
of degree powers is the fitted field's continuous spherical mean square.
This norm is distinct from the exact GraphCast discrete-cell norm.

Application is strictly S[b A f], including degree zero.  Even unit gains
return the resolved projection S[A f]; unresolved grid components are not
retained and the discrete mean is not forced to match the input.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.models.mamba.v24_Ilya.training.spectral_loss import (
    _legendre_basis,
    build_spherical_transform,
    latitude_area_weights,
)


def _coefficients(value):
    value = np.asarray(value, dtype=np.complex128)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2] or not value.shape[-1]:
        raise ValueError("Expected (..., degree, nonnegative order) square coefficients")
    if not np.all(np.isfinite(value)):
        raise ValueError("Nonfinite spherical coefficients")
    invalid_degree, invalid_order = np.triu_indices(value.shape[-1], 1)
    if np.any(value[..., invalid_degree, invalid_order] != 0):
        raise ValueError("Nonzero coefficients with order greater than degree")
    return value


def degree_power(coefficients):
    """Physical continuous mean-square power by degree, with m>0 doubled."""
    coefficients = _coefficients(coefficients)
    multiplicity = np.ones(coefficients.shape[-1], dtype=np.float64)
    multiplicity[1:] = 2
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.sum((coefficients.real**2 + coefficients.imag**2) * multiplicity,
                       axis=-1)
    if not np.all(np.isfinite(value)):
        raise ValueError("Nonfinite spherical degree power")
    return value


def spectral_moments(prediction, truth):
    """Degree statistics sufficient for fitting one real gain per degree.

    No prefix dimension is reduced here: callers pool forecast initializations
    while retaining predictor, lead, channel and degree axes.  Computing the
    error power directly avoids cancellation for nearly identical fields with
    large degree-zero offsets.
    """
    prediction, truth = _coefficients(prediction), _coefficients(truth)
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and truth coefficient shapes differ")
    multiplicity = np.ones(prediction.shape[-1], dtype=np.float64)
    multiplicity[1:] = 2
    with np.errstate(over="ignore", invalid="ignore"):
        cross = np.sum(np.real(prediction * truth.conj()) * multiplicity, axis=-1)
    if not np.all(np.isfinite(cross)):
        raise ValueError("Nonfinite spherical cross power")
    return {
        "prediction_power": degree_power(prediction),
        "truth_power": degree_power(truth),
        "cross_power": cross,
        "error_power": degree_power(prediction - truth),
    }


def fit_degree_gains(prediction_power, cross_power):
    """Fit the unconstrained real least-squares optimum C/P elementwise.

    Inputs are date-pooled sufficient statistics with arbitrary matching prefix
    axes, normally (predictor, lead, channel, degree).  No denominator floor is
    used.  Exactly zero forecast power makes the gain unidentifiable; identity
    is selected deterministically and its count recorded.  Negative gains and
    amplification are retained and reported, never silently clipped.
    """
    prediction_power = np.asarray(prediction_power, dtype=np.float64)
    cross_power = np.asarray(cross_power, dtype=np.float64)
    if prediction_power.shape != cross_power.shape or not prediction_power.size:
        raise ValueError("Prediction and cross power must have matching nonempty shapes")
    if (not np.all(np.isfinite(prediction_power))
            or not np.all(np.isfinite(cross_power))):
        raise ValueError("Nonfinite filter fitting statistics")
    if np.any(prediction_power < 0):
        raise ValueError("Negative forecast power in filter fitting statistics")
    zero = prediction_power == 0
    if np.any(cross_power[zero] != 0):
        raise ValueError("Nonzero cross power with exactly zero forecast power")
    gains = np.ones_like(prediction_power)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(cross_power, prediction_power, out=gains, where=~zero)
    if not np.all(np.isfinite(gains)):
        raise ValueError("Nonfinite fitted spherical gain")
    negative_count = int(np.count_nonzero(gains < 0))
    amplified_count = int(np.count_nonzero(gains > 1))
    large_count = int(np.count_nonzero(np.abs(gains) > 10))
    zero_count = int(np.count_nonzero(zero))
    warnings = []
    if zero_count:
        warnings.append("Exactly zero forecast power: unidentified gains set to identity")
    if negative_count:
        warnings.append("Negative least-squares gains retained")
    if amplified_count:
        warnings.append("Gains greater than one retained")
    if large_count:
        warnings.append("Absolute gain greater than ten retained; inspect low-power degrees")
    return gains, {
        "method": "unconstrained_real_least_squares_cross_power_over_prediction_power",
        "shape": list(gains.shape),
        "zero_power_count": zero_count,
        "zero_power_policy": "identity; no denominator floor",
        "negative_gain_count": negative_count,
        "gain_above_one_count": amplified_count,
        "absolute_gain_above_ten_count": large_count,
        "large_gain_warning_threshold": 10.0,
        "gain_min": float(gains.min()),
        "gain_max": float(gains.max()),
        "warnings": warnings,
    }


class OptimalSphericalFilter:
    """Analysis and pure spectral reconstruction on a native regular grid.

    The geometry and main transforms use the existing FP32 implementation.
    Means are separated and restored in FP64 to avoid losing small pressure
    anomalies beside a large offset.  This algebraically changes neither the
    fitted constant mode nor the operator; it does not preserve the input mean
    after arbitrary gains or add back the unresolved residual.
    """

    def __init__(self, latitude, longitude, *, lmax=179):
        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.longitude = np.asarray(longitude, dtype=np.float64)
        self.area = latitude_area_weights(self.latitude)
        self.transform = build_spherical_transform(self.latitude, self.longitude, lmax)
        self.lmax = self.transform.lmax
        self._analysis = jax.jit(self.transform.analysis)
        basis = jnp.asarray(_legendre_basis(self.latitude, self.lmax), dtype=jnp.float32)
        nlon = len(self.longitude)

        @jax.jit
        def synthesis(coefficients):
            real = jnp.einsum("lmj,...lm->...jm", basis, coefficients.real,
                              precision=jax.lax.Precision.HIGHEST)
            imag = jnp.einsum("lmj,...lm->...jm", basis, coefficients.imag,
                              precision=jax.lax.Precision.HIGHEST)
            rings = jax.lax.complex(real, imag)
            padding = [(0, 0)] * (rings.ndim - 1)
            padding.append((0, nlon // 2 + 1 - rings.shape[-1]))
            rings = jnp.pad(rings, padding)
            return jnp.fft.irfft(rings * nlon, n=nlon, axis=-1)

        self._synthesis = synthesis

    def _fields(self, fields):
        fields = np.asarray(fields, dtype=np.float64)
        if fields.ndim < 2 or fields.shape[-2:] != (len(self.latitude), len(self.longitude)):
            raise ValueError("Field shape does not match spherical filter grid")
        if not np.all(np.isfinite(fields)):
            raise ValueError("Nonfinite spherical filter input field")
        return fields

    def mean(self, fields):
        """Exact discrete GraphCast area mean, preserving all prefix axes."""
        fields = self._fields(fields)
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.einsum("...ij,i->...", fields, self.area) / len(self.longitude)
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite spherical filter grid mean")
        return result

    def analyze(self, fields):
        """Return normalized complex128 coefficients with FP64 offset handling."""
        fields = self._fields(fields)
        mean = self.mean(fields)
        centered = fields - mean[..., None, None]
        coefficients = np.asarray(jax.device_get(self._analysis(
            jnp.asarray(centered, dtype=jnp.float32))), dtype=np.complex128)
        # np.asarray may return a read-only JAX-host buffer when the dtype matches.
        coefficients = coefficients.copy()
        coefficients[..., 0, 0] += mean
        return _coefficients(coefficients)

    def synthesize(self, coefficients):
        """Reconstruct the represented field, without mean or residual repair."""
        coefficients = _coefficients(coefficients)
        if coefficients.shape[-2:] != (self.lmax + 1, self.lmax + 1):
            raise ValueError("Coefficient shape does not match spherical filter degree")
        if np.any(coefficients[..., :, 0].imag != 0):
            raise ValueError("Order-zero coefficients of a real field must be real")
        constant = coefficients[..., 0, 0].real
        centered = coefficients.copy()
        centered[..., 0, 0] = 0
        field = np.asarray(jax.device_get(self._synthesis(
            jnp.asarray(centered, dtype=jnp.complex64))), dtype=np.float64)
        field = field + constant[..., None, None]
        return self._fields(field)

    def apply(self, fields, degree_gains):
        """Return S[b A f], allowing any finite real gain, including degree zero."""
        coefficients = self.analyze(fields)
        gains = np.asarray(degree_gains)
        if np.iscomplexobj(gains):
            raise ValueError("Spherical filter gains must be real")
        gains = gains.astype(np.float64)
        if gains.shape != coefficients.shape[:-1] or not np.all(np.isfinite(gains)):
            raise ValueError("Invalid per-degree spherical gains")
        return self.synthesize(coefficients * gains[..., None])

    def grid_mse(self, forecast, truth):
        """Physical-unit native-grid MSE, including exact GraphCast pole cells."""
        forecast, truth = self._fields(forecast), self._fields(truth)
        if forecast.shape != truth.shape:
            raise ValueError("Forecast and truth grid shapes differ")
        with np.errstate(over="ignore", invalid="ignore"):
            return self.mean((forecast - truth)**2)

    spectral_moments = staticmethod(spectral_moments)

