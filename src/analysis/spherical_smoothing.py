"""Development-fitted output attenuation with exact discrete-mean preservation.

No predictions are fed back after filtering. The T179 projection residual is
retained. Statistics and calibration use the exact GraphCast grid objective,
not the continuous spherical coefficient norm.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from src.models.mamba.v24_Ilya.metrics import ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS
from src.models.mamba.v24_Ilya.training.spectral_loss import (
    _legendre_basis, build_spherical_transform, latitude_area_weights,
)

GAUSSIAN_DEGREES = np.asarray([0., .25, .5, 1., 2., 4., 8.])
LEAD_GROUP_ENDS = np.asarray([24, 72, 168, 240])
BAND_KNOTS = (1, 10, 40, 179)
REGULARIZATION = .01
VARIABLES = (
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr", "temperature",
    "geopotential", "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity",
)
WIND_GROUP = {
    "10m_u_component_of_wind": "wind10", "10m_v_component_of_wind": "wind10",
    "u_component_of_wind": "wind", "v_component_of_wind": "wind",
}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def lead_group(lead_hours):
    if lead_hours < 6 or lead_hours > 240 or lead_hours % 6:
        raise ValueError("Expected a six-hour lead in 6..240")
    return int(np.searchsorted(LEAD_GROUP_ENDS, lead_hours))


def band_hats(lmax=179, knots=BAND_KNOTS):
    knots = np.asarray(knots, dtype=np.float64)
    if len(knots) != 4 or knots[0] != 1 or knots[-1] != lmax or np.any(np.diff(knots) <= 0):
        raise ValueError("Four increasing degree knots must span 1..lmax")
    hats = np.zeros((4, lmax + 1), dtype=np.float64)
    degrees = np.arange(1, lmax + 1)
    for i in range(3):
        mask = (degrees >= knots[i]) & (degrees <= knots[i + 1])
        x = degrees[mask]
        t = (np.log(x) - np.log(knots[i])) / (np.log(knots[i + 1]) - np.log(knots[i]))
        hats[i, x] = 1 - t
        hats[i + 1, x] = t
    return hats


def gaussian_gains(lmax=179, widths=GAUSSIAN_DEGREES):
    degrees = np.arange(lmax + 1, dtype=np.float64)
    sigma = np.deg2rad(np.asarray(widths, dtype=np.float64))
    if np.any(sigma < 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("Gaussian widths must be finite and nonnegative")
    return np.exp(-.5 * sigma[:, None]**2 * degrees[None, :] * (degrees[None, :] + 1))


@dataclass(frozen=True)
class ChannelLayout:
    """Stable physical channel order, normalization and canonical weights."""
    channels: tuple
    scales: np.ndarray
    weights: np.ndarray
    group_index: np.ndarray
    groups: tuple

    @classmethod
    def from_dataset(cls, dataset, diffs_stddev, *, expected_channels=83):
        if set(dataset.data_vars) - set(VARIABLES):
            raise ValueError("Unexpected forecast fields in smoothing observer")
        channels, scales, weights, group_keys = [], [], [], []
        for name in sorted(dataset.data_vars):
            field = dataset[name]
            levels = np.asarray(field.level, dtype=np.float64).tolist() if "level" in field.dims else [None]
            level_sum = sum(levels) if levels != [None] else 1.
            for level in levels:
                scale = diffs_stddev[name]
                if level is not None:
                    scale = scale.sel(level=level)
                scale = float(np.asarray(scale).squeeze())
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Invalid difference normalization: {name}/{level}")
                channels.append((name, level))
                scales.append(scale)
                weights.append(ORIGINAL_GRAPHCAST_VARIABLE_WEIGHTS.get(name, 1.) *
                               (level / level_sum if level is not None else 1.))
                group_keys.append((WIND_GROUP.get(name, name), level))
        if expected_channels is not None and len(channels) != expected_channels:
            raise ValueError(f"Expected {expected_channels} physical channels, got {len(channels)}")
        groups = tuple(dict.fromkeys(group_keys))
        return cls(tuple(channels), np.asarray(scales), np.asarray(weights),
                   np.asarray([groups.index(key) for key in group_keys]), groups)

    def pack(self, dataset):
        if set(dataset.data_vars) != {c[0] for c in self.channels}:
            raise ValueError("Physical variable set changed")
        fields = []
        for name, level in self.channels:
            field = dataset[name].sel(level=level) if level is not None else dataset[name]
            for dim in tuple(field.dims):
                if dim not in ("lat", "lon"):
                    if field.sizes[dim] != 1:
                        raise ValueError(f"Expected one field per forecast step: {name}/{dim}")
                    field = field.isel({dim: 0}, drop=True)
            fields.append(np.asarray(field.transpose("lat", "lon"), dtype=np.float64))
        result = np.stack(fields)
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite physical forecast")
        return result

    def metadata(self):
        return {"channels": [list(v) for v in self.channels], "scales": self.scales.tolist(),
                "weights": self.weights.tolist(), "groups": [list(v) for v in self.groups],
                "group_index": self.group_index.tolist()}


class SphericalAttenuator:
    """FP32 QR analysis/synthesis, FP64 correction centering and grid scoring.

    Centering the input before analysis is algebraically equivalent for every
    positive-degree transfer and avoids large pressure offsets. Neither truth
    nor forecasts are projected onto a different verification grid.
    """
    def __init__(self, latitude, longitude, *, lmax=179, knots=BAND_KNOTS, chunk_channels=8):
        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.longitude = np.asarray(longitude, dtype=np.float64)
        if not np.allclose(np.diff(self.longitude), 360 / len(self.longitude), rtol=0, atol=1e-6):
            raise ValueError("A complete periodic longitude grid is required")
        self.area = latitude_area_weights(self.latitude)
        self.lmax, self.chunk_channels = lmax, chunk_channels
        self.hats = band_hats(lmax, knots)
        self.gaussians = gaussian_gains(lmax)
        self.transform = build_spherical_transform(self.latitude, self.longitude, lmax)
        basis = jnp.asarray(_legendre_basis(self.latitude, lmax), dtype=jnp.float32)
        nlon = len(self.longitude)

        def synthesis(coefficients):
            real = jnp.einsum("lmj,...lm->...jm", basis, coefficients.real,
                              precision=jax.lax.Precision.HIGHEST)
            imag = jnp.einsum("lmj,...lm->...jm", basis, coefficients.imag,
                              precision=jax.lax.Precision.HIGHEST)
            rings = jax.lax.complex(real, imag)
            rings = jnp.pad(rings, [(0, 0)] * (rings.ndim - 1) + [(0, nlon // 2 + 1 - rings.shape[-1])])
            return jnp.fft.irfft(rings * nlon, n=nlon, axis=-1)

        self._synthesis = jax.jit(synthesis)

        @jax.jit
        def corrections(centered, transfers):
            coefficients = self.transform.analysis(centered)
            return synthesis(coefficients[None] * transfers[..., None])

        self._corrections = corrections
        self._analysis = jax.jit(self.transform.analysis)

    def mean(self, values):
        return np.einsum("...ij,i->...", np.asarray(values, dtype=np.float64), self.area) / len(self.longitude)

    def corrections(self, fields, transfers):
        """Return centered S[transfer A field], preserving arbitrary batch axes."""
        fields = np.asarray(fields, dtype=np.float64)
        transfers = np.asarray(transfers, dtype=np.float32)
        if fields.ndim != 3 or transfers.ndim != 3 or transfers.shape[1:] != (len(fields), self.lmax + 1):
            raise ValueError("Expected channel grids and (filter,channel,degree) transfers")
        if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(transfers)) or np.any(transfers[..., 0] != 0):
            raise ValueError("Nonfinite filter input or a changed degree-zero transfer")
        centered = fields - self.mean(fields)[..., None, None]
        values = np.asarray(jax.device_get(self._corrections(
            jnp.asarray(centered, dtype=jnp.float32), jnp.asarray(transfers))), dtype=np.float64)
        values -= self.mean(values)[..., None, None]
        if not np.all(np.isfinite(values)):
            raise ValueError("Nonfinite spherical filter computation")
        return values

    def apply(self, fields, degree_gains):
        fields = np.asarray(fields)
        gains = np.asarray(degree_gains)
        if gains.shape != (len(fields), self.lmax + 1) or not np.all(np.isfinite(gains)):
            raise ValueError("Invalid per-channel filter gains")
        if np.any(gains < 0) or np.any(gains > 1) or np.any(gains[:, 0] != 1):
            raise ValueError("Attenuation gains must be in [0,1], with degree zero unchanged")
        if np.all(gains == 1):
            return fields.copy()
        return fields.astype(np.float64) + self.corrections(fields, gains[None] - 1)[0]

    def calibration_chunk(self, forecast, truth, scales):
        """Exact-grid sufficient statistics, before canonical channel weighting."""
        forecast, truth = np.asarray(forecast, dtype=np.float64), np.asarray(truth, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)[:, None, None]
        error = (forecast - truth) / scales
        transfers = np.concatenate((self.hats, self.gaussians - 1), axis=0)
        transfers = np.broadcast_to(transfers[:, None], (len(transfers), len(forecast), self.lmax + 1))
        corrections = self.corrections(forecast / scales, transfers)
        bases = corrections[:4]
        gaussian_error = error[None] + corrections[4:]
        nlon = len(self.longitude)
        e0 = self.mean(error * error)
        cross = np.einsum("cxy,jcxy,x->cj", error, bases, self.area, optimize=True) / nlon
        gram = np.einsum("icxy,jcxy,x->cij", bases, bases, self.area, optimize=True) / nlon
        gaussian = self.mean(gaussian_error * gaussian_error).T
        return {"e0": e0, "cross": cross, "gram": gram, "gaussian": gaussian}

    def filtered_pair(self, forecast, scales, gaussian_indices, band_gains):
        """Apply the two frozen filters together; returns Gaussian and band fields."""
        forecast = np.asarray(forecast, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)[:, None, None]
        gaussian = self.gaussians[np.asarray(gaussian_indices, dtype=int)]
        band = np.asarray(band_gains) @ self.hats
        band[:, 0] = 1
        transfers = np.stack((gaussian - 1, band - 1))
        delta = self.corrections(forecast / scales, transfers) * scales[None]
        values = forecast[None] + delta
        # Avoid roundoff in channels explicitly assigned identity.
        for fi, gains in enumerate((gaussian, band)):
            identity = np.all(gains == 1, axis=-1)
            values[fi, identity] = forecast[identity]
        return values

    def spectral_chunk(self, forecast, truth):
        """Unnormalized physical spherical statistics; degree zero retained."""
        f = np.asarray(forecast, dtype=np.float64)
        t = np.asarray(truth, dtype=np.float64)
        # Analyze each mean separately to avoid losing pressure anomaly precision.
        mf, mt = self.mean(f), self.mean(t)
        cf = np.asarray(jax.device_get(self._analysis(jnp.asarray(f - mf[:, None, None], dtype=jnp.float32))), dtype=np.complex128)
        ct = np.asarray(jax.device_get(self._analysis(jnp.asarray(t - mt[:, None, None], dtype=jnp.float32))), dtype=np.complex128)
        cf[:, 0, 0] += mf
        ct[:, 0, 0] += mt
        multiplicity = np.ones(self.lmax + 1, dtype=np.float64)
        multiplicity[1:] = 2
        cf, ct = cf.astype(np.complex128), ct.astype(np.complex128)
        delta = cf - ct
        power = lambda c: np.sum(np.abs(c)**2 * multiplicity, axis=-1)
        projected_error = np.asarray(jax.device_get(self._synthesis(jnp.asarray(delta, dtype=jnp.complex64))), dtype=np.float64)
        error = f - t
        return {"prediction_power": power(cf), "truth_power": power(ct),
                "cross_power": np.sum(np.real(cf * ct.conj()) * multiplicity, axis=-1),
                "error_power": power(delta), "grid_mse": self.mean(error**2),
                "projected_grid_mse": self.mean(projected_error**2),
                "unresolved_grid_mse": self.mean((error - projected_error)**2)}


def fit_band(e0, cross, gram, *, eta=REGULARIZATION):
    """Fit the fixed regularized quadratic, retaining finite optimizer warnings."""
    e0 = float(e0)
    cross, gram = np.asarray(cross, dtype=np.float64), np.asarray(gram, dtype=np.float64)
    if e0 < 0 or not np.isfinite(e0) or cross.shape != (4,) or gram.shape != (4, 4) or not np.all(np.isfinite(cross)) or not np.all(np.isfinite(gram)):
        raise ValueError("Corrupt/nonfinite calibration statistics")
    if e0 == 0:
        return np.ones(4), {"status": "zero_error_identity", "objective": 1., "e0": e0}
    gram = (gram + gram.T) / 2
    difference = np.diff(np.eye(4), axis=0)
    penalty = difference.T @ difference / 3 + np.eye(4) / 4

    def objective(a):
        d = a - 1
        return float(1 + (2 * cross @ d + d @ gram @ d) / e0 + eta * (d @ penalty @ d))

    def gradient(a):
        d = a - 1
        return 2 * (cross + gram @ d) / e0 + 2 * eta * (penalty @ d)

    result = minimize(objective, np.ones(4), jac=gradient, bounds=[(0., 1.)] * 4,
                      method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000})
    if not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun):
        raise ValueError("Nonfinite filter optimizer output")
    if np.any(result.x < 0) or np.any(result.x > 1):
        raise ValueError("Optimizer returned infeasible attenuation gains")
    gains = result.x if objective(result.x) <= 1 else np.ones(4)
    grad = gradient(gains)
    projected = gains - np.clip(gains - grad, 0, 1)
    return gains, {"status": "complete" if result.success else "convergence_warning",
                   "message": str(result.message), "iterations": int(result.nit),
                   "objective": objective(gains), "e0": e0,
                   "unpenalized_loss": float(e0 + 2 * cross @ (gains - 1) + (gains - 1) @ gram @ (gains - 1)),
                   "projected_gradient_max": float(np.max(np.abs(projected))),
                   "identity_retained": bool(np.all(gains == 1))}


def fit_filters(statistics, layout, branches=("GC", "M1")):
    """Fit independently per predictor, lead group and tied channel group."""
    weights = np.asarray(layout["weights"], dtype=np.float64)
    group_index = np.asarray(layout["group_index"], dtype=int)
    groups = layout["groups"]
    result = {}
    for bi, branch in enumerate(branches):
        gaussian = np.zeros((4, len(weights)), dtype=int)
        band = np.ones((4, len(weights), 4), dtype=np.float64)
        diagnostics = []
        for hg in range(4):
            for gi, group in enumerate(groups):
                mask = group_index == gi
                e0 = np.sum(statistics["e0"][bi, hg, mask] * weights[mask])
                cross = np.einsum("c,cj->j", weights[mask], statistics["cross"][bi, hg, mask])
                gram = np.einsum("c,cij->ij", weights[mask], statistics["gram"][bi, hg, mask])
                errors = np.einsum("c,cf->f", weights[mask], statistics["gaussian"][bi, hg, mask])
                if not np.all(np.isfinite(errors)) or np.any(errors < 0):
                    raise ValueError("Invalid Gaussian candidate errors")
                selected = int(np.argmin(errors))  # Candidate order resolves exact ties.
                gains, fit = fit_band(e0, cross, gram)
                gaussian[hg, mask] = selected
                band[hg, mask] = gains
                diagnostics.append({"lead_group": hg, "channel_group": group, **fit,
                                    "gaussian_width_degrees": float(GAUSSIAN_DEGREES[selected]),
                                    "gaussian_candidate_losses": errors.tolist()})
        result[branch] = {"gaussian_indices": gaussian.tolist(), "band_gains": band.tolist(),
                          "diagnostics": diagnostics}
    return {"format": "v24_smoothing_filters_v1", "layout": layout,
            "gaussian_widths_degrees": GAUSSIAN_DEGREES.tolist(), "knots": list(BAND_KNOTS),
            "lead_group_ends_hours": LEAD_GROUP_ENDS.tolist(), "eta": REGULARIZATION,
            "branches": result, "operator": "f + S[(g-1)Af] - mean_GC(S[(g-1)Af])",
            "precision": "FP32 QR transforms; FP64 grid corrections, sufficient statistics and optimization"}

