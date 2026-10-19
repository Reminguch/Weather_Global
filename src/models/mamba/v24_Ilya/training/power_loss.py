"""Differentiable per-example power surrogate for the MGDA continuation.

Reporting deliberately continues to pool raw powers across forecast starts.
This module instead scores each example before reduction and never changes the
historical reporting statistic. All FFT and loss arithmetic is FP32.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from graphcast import xarray_jax
import jax.numpy as jnp
import numpy as np

from ..spectral_diagnostics import zonal_latitude_weights
from ..spectral_observer import FIELD_SPECS

POWER_FIELDS = tuple(spec[0] for spec in FIELD_SPECS)
POWER_SMOOTHING = 1e-3


def zonal_power(fields, latitudes, longitudes):
    """Return physical power at positive bins for ``[..., lat, lon]`` fields."""
    longitude = np.asarray(longitudes, dtype=np.float64)
    if (longitude.ndim != 1 or len(longitude) < 4
            or not np.all(np.isfinite(longitude))
            or not np.allclose(np.diff(longitude), 360 / len(longitude), atol=1e-6)):
        raise ValueError("Power loss needs increasing periodic longitude without duplicate endpoint")
    weight = jnp.asarray(zonal_latitude_weights(latitudes), dtype=jnp.float32)
    array = jnp.asarray(fields, dtype=jnp.float32)
    if array.shape[-2:] != (len(weight), len(longitude)):
        raise ValueError("Power field shape does not match latitude/longitude coordinates")
    anomaly = array - jnp.mean(array, axis=-1, keepdims=True)
    coefficient = jnp.fft.rfft(anomaly, axis=-1, norm="forward")
    factor = np.full(len(longitude) // 2 + 1, 2.0, dtype=np.float32)
    factor[0] = 1
    if len(longitude) % 2 == 0:
        factor[-1] = 1
    power = (coefficient.real ** 2 + coefficient.imag ** 2) * jnp.asarray(factor)
    return jnp.sum(power * weight[:, None], axis=-2)[..., 1:]


def truth_powers(dataset):
    """Extract nine physical fields as powers ``[batch,time,field,positive_k]``.

    Pressure fields select the exact named level; no nearest-level fallback is
    permitted. Missing singleton batch/time dimensions are added explicitly.
    """
    fields = []
    for _, name, level, _ in FIELD_SPECS:
        value = dataset[name]
        if level is not None:
            value = value.sel(level=level, drop=True)
        for dim in ("batch", "time"):
            if dim not in value.dims:
                value = value.expand_dims({dim: [0]})
        if set(value.dims) != {"batch", "time", "lat", "lon"}:
            raise ValueError(f"Unexpected power field dimensions in {name}: {value.dims}")
        fields.append(jnp.asarray(xarray_jax.unwrap_data(value.transpose("batch", "time", "lat", "lon")),
                                  dtype=jnp.float32))
    return zonal_power(jnp.stack(fields, axis=2), np.asarray(dataset.lat), np.asarray(dataset.lon))


def calibration_floors(mean_truth_power):
    means = np.asarray(mean_truth_power, dtype=np.float64)
    if means.ndim != 2 or means.shape[0] != len(POWER_FIELDS) or means.shape[1] < 1:
        raise ValueError("Calibration means must have shape [9, positive_wavenumbers]")
    if not np.all(np.isfinite(means) & (means > 0)):
        raise ValueError("Calibration truth-power means must be positive and finite")
    floors = np.asarray(1e-3 * means, dtype=np.float32)
    if not np.all(np.isfinite(floors) & (floors > 0)):
        raise ValueError("Calibration floors are not representable as positive FP32 values")
    return floors


def validate_power_calibration(calibration):
    if not isinstance(calibration, Mapping) or calibration.get("version") != 1:
        raise ValueError("Power calibration requires version=1")
    if list(calibration.get("fields", ())) != list(POWER_FIELDS):
        raise ValueError("Power calibration field ordering differs from the nine primitive fields")
    floors = np.asarray(calibration.get("floors"), dtype=np.float32)
    if floors.ndim != 2 or floors.shape[0] != len(POWER_FIELDS) or floors.shape[1] < 1:
        raise ValueError("Power calibration floors must have shape [9, positive_wavenumbers]")
    if list(calibration.get("wavenumbers", ())) != list(range(1, floors.shape[1] + 1)):
        raise ValueError("Power calibration wavenumbers must be consecutive positive bins")
    if not np.all(np.isfinite(floors) & (floors > 0)):
        raise ValueError("Power calibration floors must be positive and finite")
    for key in ("gc_scale", "power_scale"):
        value = calibration.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Power calibration {key} must be positive and finite")
    # Reject mutable non-JSON values and NaN metadata before checkpointing.
    return json.loads(json.dumps(dict(calibration), sort_keys=True, allow_nan=False))


def make_power_calibration(mean_truth_power, gc_scale, power_scale, **metadata):
    floors = calibration_floors(mean_truth_power)
    reserved = {"version", "fields", "wavenumbers", "floors", "gc_scale", "power_scale"}
    if reserved & metadata.keys():
        raise ValueError("Calibration metadata cannot replace numerical calibration values")
    return validate_power_calibration(dict(
        **metadata, version=1, fields=list(POWER_FIELDS),
        wavenumbers=list(range(1, floors.shape[1] + 1)), floors=floors.tolist(),
        gc_scale=float(gc_scale), power_scale=float(power_scale),
    ))


def calibration_fingerprint(calibration):
    payload = validate_power_calibration(calibration)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def power_surrogate(predicted_power, observed_power, floors):
    """Return unreduced per-example, per-field/bin smooth relative errors."""
    predicted = jnp.asarray(predicted_power, dtype=jnp.float32)
    observed = jnp.asarray(observed_power, dtype=jnp.float32)
    if predicted.shape != observed.shape or predicted.shape[-2:] != np.shape(floors):
        raise ValueError("Predicted, observed and calibrated powers have incompatible shapes")
    ratio = (predicted - observed) / jnp.maximum(observed, jnp.asarray(floors, dtype=jnp.float32))
    return jnp.sqrt(ratio ** 2 + POWER_SMOOTHING ** 2) - POWER_SMOOTHING


def build_power_loss(calibration):
    """Build ``(physical_prediction, truth) -> (raw_batch_loss, diagnostics)``.

    Scales are metadata here: the dual-objective caller divides the returned
    raw power loss and exact GC loss by their respective frozen scales.
    Time is reduced equally within a single supervised call; outer BPTT uses
    the existing normalized supervision weights across scored calls.
    """
    values = validate_power_calibration(calibration)
    floors = np.asarray(values["floors"], dtype=np.float32)

    def loss(predictions, targets):
        for coordinate in ("lat", "lon", "time", "batch"):
            if coordinate in predictions.coords and coordinate in targets.coords:
                if not np.array_equal(np.asarray(predictions[coordinate]), np.asarray(targets[coordinate])):
                    raise ValueError(f"Prediction and truth {coordinate} coordinates differ")
        predicted = truth_powers(predictions)
        observed = truth_powers(targets)
        errors = power_surrogate(predicted, observed, floors)
        value = jnp.mean(errors, axis=(1, 2, 3))
        floor_fraction = jnp.mean(observed < jnp.asarray(floors), axis=(1, 2, 3))
        return value, {"power_loss": value, "floor_fraction": floor_fraction}

    return loss
