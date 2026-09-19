"""CPU diagnostics for unsmoothed, midlatitude zonal Fourier spectra.

These are spectra of scalar fields along longitude, not spherical-harmonic
spectra or kinetic-energy spectra.  Their integer wavenumber counts cycles
around a latitude circle and therefore has no latitude-independent wavelength.
"""

from __future__ import annotations

import numpy as np


MID_LATITUDE_BOUNDS_DEGREES = (30.0, 60.0)


def zonal_spectral_metadata() -> dict:
    """Return JSON-safe definitions shared by the exporter and plot report."""
    return {
        "diagnostic": "zonal_fourier_spectrum",
        "version": 1,
        "latitude_band_degrees_absolute": list(MID_LATITUDE_BOUNDS_DEGREES),
        "latitude_band_endpoints": "inclusive; both hemispheres required",
        "latitude_weighting": "cos(latitude), normalized over selected rows",
        "longitude_transform": "real FFT with forward normalization (1/n_lon)",
        "one_sided_power": "positive modes doubled except the even-grid Nyquist mode",
        "zonal_mean": "removed separately at each latitude; k=0 retained as exact zero",
        "wavenumber": "integer cycles around a latitude circle",
        "power_units": "squared units of the original scalar field",
        "cross_power": "real part of forecast FFT times conjugate truth FFT",
        "parseval": "sum over k equals latitude-weighted zonal-anomaly mean square",
        "pooling": "average raw powers across starts before computing ratios or correlation",
        "smoothing": "none",
    }


def zonal_latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    """Return normalized cosine weights for 30--60 degrees in both hemispheres.

    Latitude coordinates may increase or decrease, but must be unique and
    monotonic.  Rows outside the selected band receive weight zero.  The
    evaluator supplies its regular latitude grid; this function defines a
    discrete cosine-weighted average rather than a general quadrature rule.
    """
    raw = np.asarray(latitudes)
    if raw.ndim != 1 or raw.size < 2 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError("latitudes must be a one-dimensional numeric coordinate")
    if np.iscomplexobj(raw):
        raise ValueError("latitudes must be real")
    lat = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(lat)) or np.any(np.abs(lat) > 90.0):
        raise ValueError("latitudes must be finite and within [-90, 90] degrees")
    differences = np.diff(lat)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("latitudes must be unique and strictly monotonic")
    lower, upper = MID_LATITUDE_BOUNDS_DEGREES
    selected = (np.abs(lat) >= lower) & (np.abs(lat) <= upper)
    if not np.any(selected & (lat > 0)) or not np.any(selected & (lat < 0)):
        raise ValueError("latitudes must cover the 30--60 degree band in both hemispheres")
    weights = np.where(selected, np.cos(np.deg2rad(lat)), 0.0)
    return weights / weights.sum()


def zonal_spectral_statistics(
    truth: np.ndarray,
    forecast: np.ndarray,
    latitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate raw per-variable powers for one forecast initialization/lead.

    ``truth`` and ``forecast`` have shape ``(variable, latitude, longitude)``
    and contain finite, real physical values.  Longitude must be a complete,
    uniform periodic circle without a duplicated endpoint; the caller checks
    its coordinates.  The calculation uses float64 on CPU and does not smooth
    the fields or spectra.

    The four power arrays have shape ``(variable, n_lon // 2 + 1)``.  The
    ``wavenumber`` coordinate includes k=0, whose power is exactly zero after
    removing each row's zonal mean.  ``latitude_weights`` has length n_lat.
    ``cross_power`` is a real co-spectrum; it retains phase errors rather than
    taking an absolute value.  Thus error power equals truth power plus
    forecast power minus twice cross power, up to floating-point roundoff.

    Average these raw powers across initializations *before* forming power
    ratios, normalized error, or correlation.  Spectral correlation is
    cross_power / sqrt(power_truth * power_forecast); zero-power bins are
    undefined and should remain NaN in derived products.
    """
    observed = np.asarray(truth)
    predicted = np.asarray(forecast)
    if observed.ndim != 3 or predicted.shape != observed.shape:
        raise ValueError("truth and forecast must have equal (variable, latitude, longitude) shapes")
    if observed.shape[0] < 1 or observed.shape[1] < 2 or observed.shape[2] < 2:
        raise ValueError("fields require at least one variable, two latitudes, and two longitudes")
    for name, field in (("truth", observed), ("forecast", predicted)):
        if not np.issubdtype(field.dtype, np.number) or np.iscomplexobj(field):
            raise ValueError(f"{name} must contain real numeric values")
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must contain only finite values")

    weights = zonal_latitude_weights(latitudes)
    if weights.size != observed.shape[1]:
        raise ValueError("latitude coordinate length must match the field latitude axis")
    selected = weights > 0
    observed = np.asarray(observed[:, selected, :], dtype=np.float64)
    predicted = np.asarray(predicted[:, selected, :], dtype=np.float64)
    observed = observed - observed.mean(axis=-1, keepdims=True)
    predicted = predicted - predicted.mean(axis=-1, keepdims=True)
    observed_fft = np.fft.rfft(observed, axis=-1, norm="forward")
    predicted_fft = np.fft.rfft(predicted, axis=-1, norm="forward")
    observed_fft[..., 0] = 0.0
    predicted_fft[..., 0] = 0.0

    n_lon = observed.shape[-1]
    n_modes = observed_fft.shape[-1]
    one_sided_factor = np.full(n_modes, 2.0)
    one_sided_factor[0] = 1.0
    if n_lon % 2 == 0:
        one_sided_factor[-1] = 1.0

    def average_rows(values: np.ndarray) -> np.ndarray:
        return np.einsum("vlk,l->vk", values, weights[selected]) * one_sided_factor

    return {
        "power_truth": average_rows(np.abs(observed_fft) ** 2),
        "power_forecast": average_rows(np.abs(predicted_fft) ** 2),
        "cross_power": average_rows(np.real(predicted_fft * observed_fft.conj())),
        "error_power": average_rows(np.abs(predicted_fft - observed_fft) ** 2),
        "wavenumber": np.arange(n_modes, dtype=np.int64),
        "latitude_weights": weights,
    }
