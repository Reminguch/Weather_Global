#!/usr/bin/env python
"""WeatherBench-5 style zonal energy spectrum analysis.

For each model's dense forecast .nc (anchors × leads × lat × lon),
compute the zonal energy spectrum S_k along constant-latitude circles
(30° ≤ |lat| ≤ 60°), averaged over latitudes and anchors.

S_0 = C |F_0|^2,   S_k = 2C |F_k|^2 for k=1..L/2
where F_k = (1/L) Σ_l f_l exp(-i 2π k l / L)  is the discrete FFT and
C is the latitude circle circumference.

Diagnostic intent: over-smoothed forecasts (low RMSE but unphysical)
show power deficit at large wavenumbers. Comparing v22 (d_conv=4) and
v23 (d_conv=8) at long leads directly tests the "d_conv=4 wins MSE
via over-smoothing" hypothesis.

Usage:
  python spectral_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

R_EARTH = 6371_000.0  # m
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-05-30-v23/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Models: label → path. Use what's available now; v23 fresh added later.
MODELS = {
    "v22 K=22 (d_conv=4)": "/scratch/gpfs/DABANIN/lm8598/Weather_Global/"
                            "results/0522_v22_extreme/"
                            "v22_K22_extreme_records_K40_2022.nc",
    "v23 K=22 20k (d_conv=8, partial)": "/scratch/gpfs/DABANIN/lm8598/"
                                         "Weather_Global/results/"
                                         "0526_v23_extreme/"
                                         "v23_K22_20k_extreme_records_K40_2022.nc",
    "v23 K=22 FRESH (d_conv=8, clean)": "/scratch/gpfs/DABANIN/lm8598/"
                                         "Weather_Global/results/"
                                         "0526_v23_extreme/"
                                         "v23_K22_fresh_extreme_records_K40_2022.nc",
}

# Variables and their save_extreme_records_data suffixes.
VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
TRUTH_SUFFIX = "_truth"
BASELINE_SUFFIX = "_baseline"
MODEL_SUFFIX = "_v18"  # legacy name; this is the residual model prediction

LEADS_TO_PLOT_H = [24, 72, 168, 240]  # 1d, 3d, 7d, 10d
LAT_BAND_DEG = (30.0, 60.0)


def zonal_spectrum(field_3d: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Compute WB-5 zonal energy spectrum per latitude.

    field_3d: shape (anchors, lat, lon), one lead.
    lat_deg: shape (lat,), latitudes in degrees.

    Returns:
        S: shape (anchors, lat, n_k) where n_k = lon//2 + 1.
    """
    n_anchor, n_lat, n_lon = field_3d.shape
    # DFT along longitude: F_k = (1/L) sum f_l e^{-i 2pi k l / L}
    F = np.fft.rfft(field_3d, axis=-1) / n_lon
    power = np.abs(F) ** 2

    # Circle circumference at each latitude: C(lat) = 2 pi R cos(lat)
    C = 2 * np.pi * R_EARTH * np.cos(np.deg2rad(lat_deg))   # (lat,)
    C = C[None, :, None]  # broadcast over (anchor, k)

    S = 2 * C * power     # S_k = 2 C |F_k|^2 (default, k>0)
    S[:, :, 0] /= 2       # k=0 special case: S_0 = C |F_0|^2
    # For real DFT, k=n_lon//2 is also "unique" (Nyquist); leave 2x for consistency
    return S


def average_band(S: np.ndarray, lat_deg: np.ndarray,
                 lat_band: tuple[float, float]) -> np.ndarray:
    """Average S over latitudes in band and over anchors. Returns (n_k,)."""
    lo, hi = lat_band
    mask = (np.abs(lat_deg) >= lo) & (np.abs(lat_deg) <= hi)
    S_band = S[:, mask, :]                  # (anchor, n_lat_band, k)
    return S_band.mean(axis=(0, 1))         # (k,)


def compute_for_lead(ds: xr.Dataset, var: str, lead_idx: int,
                     lat_deg: np.ndarray) -> dict[str, np.ndarray]:
    """Return {'truth': S_k, 'baseline': S_k, 'model': S_k} at this lead."""
    out = {}
    for tag, suffix in [("truth", TRUTH_SUFFIX),
                        ("baseline", BASELINE_SUFFIX),
                        ("model", MODEL_SUFFIX)]:
        key = var + suffix
        # save_extreme_records writes (anchor_time, lead_h, lat, lon)
        field = (ds[key].isel(lead_h=lead_idx)
                 .transpose("anchor_time", "lat", "lon").values)
        S = zonal_spectrum(field, lat_deg)
        out[tag] = average_band(S, lat_deg, LAT_BAND_DEG)
    return out


def plot_var(var: str, model_spectra: dict[str, dict[int, dict]],
             leads_h: list[int], out_path: Path) -> None:
    """One PNG per variable, 4 panels (leads), each shows S_k ratio."""
    fig, axes = plt.subplots(1, len(leads_h), figsize=(5 * len(leads_h), 4.5),
                              sharey=True)
    colors = {"v22 K=22 (d_conv=4)": "C3",
              "v23 K=22 20k (d_conv=8, partial)": "C0",
              "v23 K=22 FRESH (d_conv=8, clean)": "C2"}

    for col, lead_h in enumerate(leads_h):
        ax = axes[col]
        # Use the first available model to get truth + baseline
        first_label = next(iter(model_spectra))
        first_data = model_spectra[first_label].get(lead_h)
        if first_data is None:
            ax.set_title(f"lead {lead_h}h\n(no data)")
            continue
        S_truth = first_data["truth"]
        k = np.arange(len(S_truth))
        # Avoid k=0 in plots (DC component)
        k_plot = k[1:]
        ax.loglog(k_plot, S_truth[1:], "k-", lw=2.0, label="ERA5 truth")
        ax.loglog(k_plot, first_data["baseline"][1:], color="grey",
                  lw=1.4, ls="--", label="GraphCast baseline")
        for label, lead_dict in model_spectra.items():
            d = lead_dict.get(lead_h)
            if d is None:
                continue
            ax.loglog(k_plot, d["model"][1:],
                      color=colors.get(label, "k"), lw=1.6, label=label)

        ax.set_title(f"lead = {lead_h}h ({lead_h // 24}d)")
        ax.set_xlabel("zonal wavenumber k")
        if col == 0:
            ax.set_ylabel(f"S_k(zonal energy)")
            ax.legend(fontsize=8, loc="lower left")
        ax.grid(which="both", alpha=0.3)

    fig.suptitle(
        f"WB-5 zonal energy spectrum ({var}); avg over |lat| ∈ "
        f"[{LAT_BAND_DEG[0]:.0f}°, {LAT_BAND_DEG[1]:.0f}°], 2022 anchors",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def plot_var_ratio(var: str, model_spectra: dict[str, dict[int, dict]],
                   leads_h: list[int], out_path: Path) -> None:
    """Same layout, but plot S_k(forecast) / S_k(truth). 1.0 = matches truth."""
    fig, axes = plt.subplots(1, len(leads_h), figsize=(5 * len(leads_h), 4.5),
                              sharey=True)
    colors = {"v22 K=22 (d_conv=4)": "C3",
              "v23 K=22 20k (d_conv=8, partial)": "C0",
              "v23 K=22 FRESH (d_conv=8, clean)": "C2"}

    for col, lead_h in enumerate(leads_h):
        ax = axes[col]
        first_label = next(iter(model_spectra))
        first_data = model_spectra[first_label].get(lead_h)
        if first_data is None:
            ax.set_title(f"lead {lead_h}h\n(no data)")
            continue
        S_truth = first_data["truth"]
        k = np.arange(len(S_truth))
        k_plot = k[1:]
        ax.axhline(1.0, color="k", lw=1.5, label="truth (ratio = 1)")
        ax.semilogx(k_plot, first_data["baseline"][1:] / S_truth[1:],
                    color="grey", lw=1.4, ls="--", label="GraphCast baseline")
        for label, lead_dict in model_spectra.items():
            d = lead_dict.get(lead_h)
            if d is None:
                continue
            ax.semilogx(k_plot, d["model"][1:] / S_truth[1:],
                        color=colors.get(label, "k"), lw=1.6, label=label)

        ax.set_title(f"lead = {lead_h}h ({lead_h // 24}d)")
        ax.set_xlabel("zonal wavenumber k")
        if col == 0:
            ax.set_ylabel("S_k(forecast) / S_k(truth)")
            ax.legend(fontsize=8, loc="lower left")
        ax.set_ylim(0, 1.4)
        ax.grid(which="both", alpha=0.3)

    fig.suptitle(
        f"WB-5 spectrum ratio vs truth ({var}); avg over |lat| ∈ "
        f"[{LAT_BAND_DEG[0]:.0f}°, {LAT_BAND_DEG[1]:.0f}°], 2022 anchors. "
        f"<1 at high k = over-smoothing.", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def main() -> None:
    # Load each model's nc; compute spectra at all selected leads, all vars.
    # spectra[label][var][lead_h] = {'truth': arr, 'baseline': arr, 'model': arr}
    spectra: dict[str, dict[str, dict[int, dict]]] = {}
    for label, path in MODELS.items():
        p = Path(path)
        if not p.exists():
            print(f"SKIP (missing): {label}  ({p})")
            continue
        print(f"Loading {label}: {p}")
        ds = xr.open_dataset(p)
        n_lead = ds.sizes["lead_h"]
        lat_deg = ds["lat"].values.astype("float32")
        # save_extreme_records writes lead_h at indices 0..K-1 == steps 1..K (k_idx = h/6 - 1)
        spectra[label] = {var: {} for var in VARS}
        for lead_h in LEADS_TO_PLOT_H:
            k_idx = lead_h // 6 - 1
            if k_idx < 0 or k_idx >= n_lead:
                continue
            for var in VARS:
                spectra[label][var][lead_h] = compute_for_lead(
                    ds, var, k_idx, lat_deg)
        ds.close()

    if not spectra:
        print("No spectra computed. Check MODELS paths.")
        return

    # Reorganize for plotting: per var, dict[label][lead_h] = {'truth', 'baseline', 'model'}
    for var in VARS:
        per_label: dict[str, dict[int, dict]] = {
            label: spectra[label][var] for label in spectra
        }
        plot_var(var, per_label, LEADS_TO_PLOT_H,
                 OUT_DIR / f"spectrum_{var}_loglog.png")
        plot_var_ratio(var, per_label, LEADS_TO_PLOT_H,
                       OUT_DIR / f"spectrum_{var}_ratio.png")


if __name__ == "__main__":
    main()
