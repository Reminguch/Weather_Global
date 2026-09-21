"""NeuralGCM-field normalized MSE on Gaussian verification grids."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from .config import FIELDS, LOSS_NAME


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


def aggregate_reduction(corrected, baseline):
    b = float(np.sum(baseline, dtype=np.float64))
    c = float(np.sum(corrected, dtype=np.float64))
    if not np.isfinite(b + c) or b <= 0:
        raise ValueError("Reduction requires finite paired scores and a positive baseline")
    return 100 * (1 - c / b)
