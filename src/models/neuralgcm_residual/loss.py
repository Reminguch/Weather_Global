"""NeuralGCM-field normalized MSE on Gaussian verification grids."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from .config import FIELDS, LOSS_NAME, BALANCED_LOSS_NAME

# Amplitude factors, applied before squaring. These are not MSE weights.
BALANCED_AMPLITUDES = {k: (0.05 if k.startswith("specific_cloud_") else
                           2.0 if k == "geopotential" else
                           0.66 if k == "specific_humidity" else 1.0) for k in FIELDS}


def gaussian_weights(latitudes_radians):
    lat = np.asarray(latitudes_radians)
    nodes, weights = np.polynomial.legendre.leggauss(len(lat))
    if not np.allclose(np.sin(lat), nodes, atol=2e-6):
        raise ValueError("Expected ascending Gaussian latitude centers")
    return (weights / weights.sum()).astype(np.float32)


class WeatherLoss:
    name = LOSS_NAME

    def __init__(self, latitude, pressure_levels, scales):
        self.area = jnp.asarray(gaussian_weights(latitude))
        levels = np.asarray(pressure_levels, dtype=np.float32)
        if np.any(levels <= 0):
            raise ValueError("Pressure levels must be positive")
        self.level_weights = jnp.asarray(levels / levels.sum())
        if set(scales) != set(FIELDS):
            raise ValueError("Loss requires all seven decoded fields")
        self.scales = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in scales.items()}
        for k, v in self.scales.items():
            if v.shape != levels.shape or not np.all(np.isfinite(v)) or np.any(np.asarray(v) <= 0):
                raise ValueError(f"Invalid loss scales: {k}")

    def field_scores(self, prediction, target):
        return {k: jnp.sum(self.level_weights * jnp.mean(jnp.sum(
            jnp.square((prediction[k].astype(jnp.float32) - target[k].astype(jnp.float32)) /
                       self.scales[k][:, None, None]) * self.area, axis=-1), axis=-1)) for k in FIELDS}

    def __call__(self, prediction, target):
        return sum(self.field_scores(prediction, target).values()) / len(FIELDS)


def balanced_scales(scales):
    """Pool existing train-only six-hour change variances across levels.

    This is a versioned objective, not the original NGCM training loss. It uses
    RMS per-level standard deviations, excluding between-level mean changes.
    Humidity retains its level-dependent scale. Cloud upper-level floors can
    no longer create enormous isolated weights.
    """
    result = {}
    for name in FIELDS:
        values = np.asarray(scales[name], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Invalid training scales for {name}")
        if name != "specific_humidity":
            values = np.full_like(values, np.sqrt(np.mean(values ** 2)))
        result[name] = values / BALANCED_AMPLITUDES[name]
    return result


def make_weather_loss(name, latitude, pressure_levels, scales):
    if name not in (LOSS_NAME, BALANCED_LOSS_NAME):
        raise ValueError(f"Unknown loss: {name}")
    loss = WeatherLoss(latitude, pressure_levels,
                       balanced_scales(scales) if name == BALANCED_LOSS_NAME else scales)
    loss.name = name
    return loss


def aggregate_reduction(corrected, baseline):
    b = float(np.sum(baseline, dtype=np.float64))
    c = float(np.sum(corrected, dtype=np.float64))
    if not np.isfinite(b + c) or b <= 0:
        raise ValueError("Reduction requires finite paired scores and a positive baseline")
    return 100 * (1 - c / b)
