#!/usr/bin/env python
"""WB-5 zonal energy spectrum across v22's K-scan (K=1 to K=22).

Tests the hypothesis: large K (long AR-tail training) forces the model
to learn over-smoothing because long AR MSE rewards regression to mean.
Small K should preserve more high-wavenumber power at long leads.

Outputs per-variable ratio plots: S_k(K-variant) / S_k(truth), with
12 K curves colored by K (viridis), at leads 24h / 72h / 168h / 240h.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

R_EARTH = 6371_000.0  # m
EXTREME_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/"
                    "results/0522_v22_extreme")
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-05-30-v23/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
LEADS_TO_PLOT_H = [24, 72, 168, 240]
LAT_BAND_DEG = (30.0, 60.0)

TRUTH_SUFFIX = "_truth"
BASELINE_SUFFIX = "_baseline"
MODEL_SUFFIX = "_v18"


def zonal_spectrum(field_3d: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    n_lon = field_3d.shape[-1]
    F = np.fft.rfft(field_3d, axis=-1) / n_lon
    power = np.abs(F) ** 2
    C = 2 * np.pi * R_EARTH * np.cos(np.deg2rad(lat_deg))
    C = C[None, :, None]
    S = 2 * C * power
    S[:, :, 0] /= 2
    return S


def average_band(S: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lo, hi = LAT_BAND_DEG
    mask = (np.abs(lat_deg) >= lo) & (np.abs(lat_deg) <= hi)
    return S[:, mask, :].mean(axis=(0, 1))


def spectra_for_var_lead(ds: xr.Dataset, var: str, lead_idx: int,
                          lat_deg: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for tag, suffix in [("truth", TRUTH_SUFFIX),
                        ("baseline", BASELINE_SUFFIX),
                        ("model", MODEL_SUFFIX)]:
        field = (ds[var + suffix].isel(lead_h=lead_idx)
                 .transpose("anchor_time", "lat", "lon").values)
        S = zonal_spectrum(field, lat_deg)
        out[tag] = average_band(S, lat_deg)
    return out


def main() -> None:
    # spectra[K][var][lead_h] = {'truth', 'baseline', 'model'}
    spectra: dict[int, dict[str, dict[int, dict]]] = {}
    for K in KS:
        p = EXTREME_DIR / f"v22_K{K}_extreme_records_K40_2022.nc"
        if not p.exists():
            print(f"SKIP missing: {p}")
            continue
        print(f"Loading v22 K={K}: {p}")
        ds = xr.open_dataset(p)
        lat_deg = ds["lat"].values.astype("float32")
        n_lead = ds.sizes["lead_h"]
        spectra[K] = {var: {} for var in VARS}
        for lead_h in LEADS_TO_PLOT_H:
            k_idx = lead_h // 6 - 1
            if k_idx < 0 or k_idx >= n_lead:
                continue
            for var in VARS:
                spectra[K][var][lead_h] = spectra_for_var_lead(
                    ds, var, k_idx, lat_deg)
        ds.close()

    if not spectra:
        print("No data loaded.")
        return

    # One PNG per variable: 4 panels (leads), 12 K curves + truth + baseline.
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(spectra), max(spectra))

    for var in VARS:
        fig, axes = plt.subplots(1, len(LEADS_TO_PLOT_H),
                                  figsize=(5.5 * len(LEADS_TO_PLOT_H), 5),
                                  sharey=True)
        for col, lead_h in enumerate(LEADS_TO_PLOT_H):
            ax = axes[col]
            # truth + baseline from K=1 entry (or any K, they should match)
            ref_K = next(iter(spectra))
            ref_data = spectra[ref_K][var].get(lead_h)
            if ref_data is None:
                ax.set_title(f"lead {lead_h}h\n(no data)")
                continue
            S_truth = ref_data["truth"]
            S_base = ref_data["baseline"]
            k = np.arange(len(S_truth))
            k_plot = k[1:]

            ax.axhline(1.0, color="k", lw=1.5, label="truth (ratio = 1)")
            ax.semilogx(k_plot, S_base[1:] / S_truth[1:],
                        color="grey", lw=1.4, ls="--",
                        label="GraphCast baseline")

            for K in sorted(spectra):
                d = spectra[K][var].get(lead_h)
                if d is None:
                    continue
                ax.semilogx(k_plot, d["model"][1:] / S_truth[1:],
                            color=cmap(norm(K)), lw=1.4, alpha=0.9,
                            label=f"K={K}")

            ax.set_title(f"lead = {lead_h}h ({lead_h // 24}d)")
            ax.set_xlabel("zonal wavenumber k")
            if col == 0:
                ax.set_ylabel("S_k(forecast) / S_k(truth)")
            ax.set_ylim(0, 1.3)
            ax.grid(which="both", alpha=0.3)
            if col == len(LEADS_TO_PLOT_H) - 1:
                ax.legend(fontsize=6.5, loc="lower left", ncol=2)

        fig.suptitle(
            f"v22 K-scan WB-5 spectrum ratio ({var}); avg |lat| ∈ "
            f"[{LAT_BAND_DEG[0]:.0f}°, {LAT_BAND_DEG[1]:.0f}°], 2022 anchors. "
            f"<1 at high k = over-smoothing. Hypothesis: large K → more deficit at long lead.",
            fontsize=11)
        plt.tight_layout()
        out_path = OUT_DIR / f"spectrum_v22_kscan_{var}_ratio.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
