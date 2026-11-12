"""Exact-grid and spherical moments for output-only smoothing-strength curves.

Each forecast is generated once.  The native-grid loss of ``f + alpha * d``
is recovered from E0, <(f-t)d>, and <d*d>; spherical powers use the analyzed
physical correction d.  These are distinct metrics: spherical coefficient
powers must never substitute for the exact GraphCast grid objective.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .smoothing_observer import SmoothingObserver, atomic_npz
from .spherical_smoothing import gaussian_gains, json_digest


FIXED_WIDTHS_DEGREES = (.25, .5, .75, 1., 1.5, 2., 3., 4., 8.)
BASE_NAMES = tuple(f"fixed_gaussian_{width:g}" for width in FIXED_WIDTHS_DEGREES) + (
    "fitted_gaussian", "fitted_band")
STATISTICS_SCHEMA = "v24_smoothing_sweep_statistics_v1"


def sweep_gains(operator, gaussian_indices, band_gains):
    """Nine global Gaussian widths, then the two frozen fitted profiles."""
    indices = np.asarray(gaussian_indices)
    bands = np.asarray(band_gains, dtype=np.float64)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Gaussian profile indices must be an integer channel vector")
    channels = len(indices)
    if np.any(indices < 0) or np.any(indices >= len(operator.gaussians)):
        raise ValueError("Gaussian profile index outside the frozen candidate set")
    if bands.shape != (channels, 4) or not np.all(np.isfinite(bands)):
        raise ValueError("Invalid frozen four-band profile")
    if np.any(bands < 0) or np.any(bands > 1):
        raise ValueError("Frozen four-band gains must be in [0,1]")
    fixed = np.broadcast_to(gaussian_gains(operator.lmax, FIXED_WIDTHS_DEGREES)[:, None],
                            (len(FIXED_WIDTHS_DEGREES), channels, operator.lmax + 1))
    band = bands @ operator.hats
    band[:, 0] = 1.
    gains = np.concatenate((fixed, operator.gaussians[indices][None], band[None]), axis=0)
    # Interpolated hats can exceed one by roundoff.  Retain finite values as
    # constructed instead of clipping the forecast or changing the fitted map.
    if not np.all(np.isfinite(gains)) or np.any(gains[..., 0] != 1):
        raise ValueError("Nonfinite smoothing transfer or changed global mean")
    return gains


def physical_coefficients(operator, fields):
    """Analyze centered FP32 fields; restore the physical mean in FP64."""
    values = np.asarray(fields, dtype=np.float64)
    means = operator.mean(values)
    centered = values - means[..., None, None]
    coefficients = np.asarray(jax.device_get(operator._analysis(
        jnp.asarray(centered, dtype=jnp.float32))), dtype=np.complex128)
    coefficients[..., 0, 0] += means
    return coefficients


def degree_cross(left, right):
    """Real spherical cross-power, including both signs of nonzero order."""
    multiplicity = np.ones(left.shape[-1], dtype=np.float64)
    multiplicity[1:] = 2.
    return np.sum(np.real(left * right.conj()) * multiplicity, axis=-1)


def correction_statistics(operator, forecast, truth, deltas):
    """Physical moments for a channel chunk and any number of corrections.

    The supplied deltas are the actual synthesized, discrete-mean-centered
    corrections, not an idealized degree transfer.  Spectral reconstruction is
    linear in their analyzed coefficients, up to the documented FP32 transform
    roundoff from reanalyzing an explicitly combined field.
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    if forecast.ndim != 3 or truth.shape != forecast.shape or deltas.ndim != 4 or deltas.shape[1:] != forecast.shape:
        raise ValueError("Expected matching channel grids and (base,channel,lat,lon) corrections")
    if not all(np.all(np.isfinite(v)) for v in (forecast, truth, deltas)):
        raise ValueError("Nonfinite forecast, truth or smoothing correction")
    cf = physical_coefficients(operator, forecast)
    ct = physical_coefficients(operator, truth)
    cd = physical_coefficients(operator, deltas)
    error = forecast - truth
    values = {
        "grid_mse": operator.mean(error * error),
        "prediction_power": degree_cross(cf, cf),
        "truth_power": degree_cross(ct, ct),
        "cross_power": degree_cross(cf, ct),
        "delta_grid_cross": operator.mean(error[None] * deltas),
        "delta_grid_norm": operator.mean(deltas * deltas),
        "delta_power": degree_cross(cd, cd),
        "forecast_delta_cross": degree_cross(cf[None], cd),
        "delta_truth_cross": degree_cross(cd, ct[None]),
        "mean_drift": operator.mean(deltas),
    }
    if not all(np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("Nonfinite smoothing sweep moments")
    return values


def reconstruct_sweep(statistics, base_index, alpha):
    """Reconstruct one point from moments with optional leading batch axes.

    ``alpha`` must be a finite scalar in [0,1].  No nonnegative clipping or
    numerical acceptance gate is applied to finite reconstructed diagnostics.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("Smoothing strength must be finite and in [0,1]")
    if not isinstance(base_index, (int, np.integer)) or not 0 <= base_index < statistics["delta_grid_cross"].shape[-2]:
        raise ValueError("Invalid smoothing base index")
    scalar = lambda key: np.asarray(statistics[key])[..., base_index, :]
    spectral = lambda key: np.asarray(statistics[key])[..., base_index, :, :]
    grid = statistics["grid_mse"] + 2 * alpha * scalar("delta_grid_cross") + alpha**2 * scalar("delta_grid_norm")
    power = statistics["prediction_power"] + 2 * alpha * spectral("forecast_delta_cross") + alpha**2 * spectral("delta_power")
    cross = statistics["cross_power"] + alpha * spectral("delta_truth_cross")
    truth = np.asarray(statistics["truth_power"])
    # Per-lead truth lacks the model axis.  A chunk has no model axis either.
    if truth.ndim == power.ndim - 1:
        truth = np.expand_dims(truth, -3)
    return {"grid_mse": grid, "prediction_power": power, "cross_power": cross,
            "error_power": power + truth - 2 * cross}


class SmoothingSweepObserver(SmoothingObserver):
    """Streaming, output-only sweep using the original observer lifecycle."""

    def __init__(self, *args, mode="evaluate", **kwargs):
        if mode != "evaluate":
            raise ValueError("Smoothing sweep only evaluates frozen development filters")
        kwargs["compute_zonal"] = False
        super().__init__(*args, mode=mode, **kwargs)
        if len(self.branches) != 2:
            raise ValueError("Expected the public GC and M1 forecast branches")

    def _evaluate(self, predictions, target, group):
        channels = len(self.layout.channels)
        nbases, ndegree = len(BASE_NAMES), self.lmax + 1
        record = {key: np.empty((2, channels, ndegree), dtype=np.float64)
                  for key in ("prediction_power", "cross_power")}
        record["grid_mse"] = np.empty((2, channels), dtype=np.float64)
        record["truth_power"] = np.empty((channels, ndegree), dtype=np.float64)
        for key in ("delta_grid_cross", "delta_grid_norm", "mean_drift"):
            record[key] = np.empty((2, nbases, channels), dtype=np.float64)
        for key in ("delta_power", "forecast_delta_cross", "delta_truth_cross"):
            record[key] = np.empty((2, nbases, channels, ndegree), dtype=np.float64)
        for branch_index, (branch, prediction) in enumerate(zip(self.branches, predictions)):
            recipe = self.filters["branches"][branch]
            gaussian = np.asarray(recipe["gaussian_indices"])[group]
            bands = np.asarray(recipe["band_gains"], dtype=np.float64)[group]
            for start in range(0, channels, self.chunk_channels):
                selection = slice(start, min(start + self.chunk_channels, channels))
                scales = self.layout.scales[selection, None, None]
                gains = sweep_gains(self.operator, gaussian[selection], bands[selection])
                delta = self.operator.corrections(prediction[selection] / scales, gains - 1) * scales[None]
                values = correction_statistics(self.operator, prediction[selection], target[selection], delta)
                for key in ("grid_mse", "prediction_power", "cross_power"):
                    record[key][branch_index, selection] = values[key]
                for key in ("delta_grid_cross", "delta_grid_norm", "delta_power", "forecast_delta_cross", "delta_truth_cross", "mean_drift"):
                    record[key][branch_index, :, selection] = values[key]
                if branch_index == 0:
                    record["truth_power"][selection] = values["truth_power"]
        record["exact_gc_loss"] = np.sum(record["grid_mse"] *
            (self.layout.weights / self.layout.scales**2)[None], axis=-1)[:, None]
        if not all(np.all(np.isfinite(v)) for v in record.values()):
            raise ValueError("Nonfinite smoothing sweep evaluation statistic")
        self.records.append(record)

    def finish_sample(self):
        if self.sample is None or self.steps != self.target_steps:
            raise ValueError("Incomplete smoothing sweep initialization")
        arrays = {key: np.stack([row[key] for row in self.records]) for key in self.records[0]}
        metadata = {
            **self.sample, "schema": STATISTICS_SCHEMA, "status": "complete", "mode": self.mode,
            "layout": self.layout.metadata(), "branches": list(self.branches), "variants": ["raw"],
            "base_names": list(BASE_NAMES), "base_width_degrees": list(FIXED_WIDTHS_DEGREES) + [None, None],
            "lmax": self.lmax, "knots": list(self.knots), "filter_digest": json_digest(self.filters),
            "operator": "f + alpha * (S[(g-1)Af] - mean_GC(S[(g-1)Af])); unrepresented residual retained",
            "feedback": "raw forecast; no filtered output is fed back",
            "grid_mse": "FP64 physical error and cross-moments, exact pole-cell areas; canonical GC weights and difference scales applied after reconstruction",
            "spherical": "Physical continuous fitted norm; mean-centered FP32 QR analysis, FP64 moments; distinct from exact discrete grid MSE",
            "reconstruction": {"grid_mse": "E0 + 2 alpha q + alpha^2 Q",
                "prediction_power": "Pf + 2 alpha C_f_delta + alpha^2 P_delta",
                "cross_power": "C_ft + alpha C_delta_t",
                "roundoff": "Moment reconstruction uses linear analyzed coefficients; reanalysis of the combined field can differ by finite FP32 transform roundoff (warning only)"},
            "clipping": "none", "step_seconds": self.step_seconds[-self.target_steps:],
        }
        atomic_npz(self.path, arrays, metadata)
        self.sample = self.accumulated = None
        self.records = []
