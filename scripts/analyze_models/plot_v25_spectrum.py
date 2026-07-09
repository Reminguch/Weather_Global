"""Zonal energy spectrum eval for v25 gated extreme vs v22 K=18 baseline at res=2.

Computes S_k = 2*C * |F_k|^2 along constant-latitude circles, averaged over
30°-60° lat band and over all 2022 anchor times. Plots ratios vs truth at
multiple leads, plus raw spectra.

Inputs:
  v25_step3000_res2_extreme_records_K40_2022.nc  (gated extreme)
  v22_K18_step20000_res2_extreme_records_K40_2022.nc  (open-loop baseline)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_EARTH = 6371_000.0
DATA_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/2026-6-27-v25-extreme")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-v25-extreme/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
LEADS_TO_PLOT_H = [24, 72, 168, 240]
LAT_BAND_DEG = (30.0, 60.0)

TAGS = [
    ("v22 K=18 step20k",       "v22_K18_step20000_res2_extreme_records_K40_2022.nc",     "C0"),
    ("v25 phase 1 step3k",     "v25_step3000_res2_extreme_records_K40_2022.nc",          "C1"),
    ("v25 phase 1.5 step5k",   "v25_phase1p5_step5000_res2_extreme_records_K40_2022.nc", "C2"),
]


def zonal_spectrum(field_3d, lat_deg):
    n_lon = field_3d.shape[-1]
    F = np.fft.rfft(field_3d, axis=-1) / n_lon
    power = np.abs(F) ** 2
    C = 2 * np.pi * R_EARTH * np.cos(np.deg2rad(lat_deg))
    C = C[None, :, None]
    S = 2 * C * power
    S[:, :, 0] /= 2
    return S


def average_band(S, lat_deg):
    lo, hi = LAT_BAND_DEG
    mask = (np.abs(lat_deg) >= lo) & (np.abs(lat_deg) <= hi)
    return S[:, mask, :].mean(axis=(0, 1))


def spectra_for_var_lead(ds, var, lead_idx, lat_deg):
    out = {}
    for tag, suffix in [("truth", "_truth"), ("baseline", "_baseline"), ("model", "_v18")]:
        field = (ds[var + suffix].isel(lead_h=lead_idx)
                 .transpose("anchor_time", "lat", "lon").values)
        S = zonal_spectrum(field, lat_deg)
        out[tag] = average_band(S, lat_deg)
    return out


def main():
    fig_raw, ax_raw = plt.subplots(len(VARS), len(LEADS_TO_PLOT_H),
                                    figsize=(6 * len(LEADS_TO_PLOT_H), 4.5 * len(VARS)),
                                    sharex=True)
    fig_ratio, ax_ratio = plt.subplots(len(VARS), len(LEADS_TO_PLOT_H),
                                        figsize=(6 * len(LEADS_TO_PLOT_H), 4.5 * len(VARS)),
                                        sharex=True)
    # Make 2D index work
    if len(VARS) == 1:
        ax_raw = np.array([ax_raw]); ax_ratio = np.array([ax_ratio])

    truth_plotted = False

    for tag, fname, color in TAGS:
        p = DATA_DIR / fname
        if not p.exists():
            print(f"[skip] {p}")
            continue
        ds = xr.open_dataset(p)
        lat_deg = ds.lat.values.astype("float32")
        for i, var in enumerate(VARS):
            for j, lead_h in enumerate(LEADS_TO_PLOT_H):
                k_idx = lead_h // 6 - 1
                if k_idx < 0 or k_idx >= ds.sizes["lead_h"]:
                    continue
                S = spectra_for_var_lead(ds, var, k_idx, lat_deg)
                ks = np.arange(len(S["truth"]))
                # raw spectra: plot truth (once) + baseline (once) + model per tag
                if not truth_plotted:
                    ax_raw[i, j].loglog(ks[1:], S["truth"][1:],     "k-",  lw=1.5, label="truth")
                    ax_raw[i, j].loglog(ks[1:], S["baseline"][1:], color="gray", lw=1.5, ls="--", label="GC base")
                ax_raw[i, j].loglog(ks[1:], S["model"][1:], color=color, lw=1.8, label=tag)

                ratio_model = S["model"] / np.maximum(S["truth"], 1e-30)
                ratio_base  = S["baseline"] / np.maximum(S["truth"], 1e-30)
                if not truth_plotted:
                    ax_ratio[i, j].axhline(1.0, color="k", lw=0.8, ls=":")
                    ax_ratio[i, j].semilogx(ks[1:], ratio_base[1:], color="gray", lw=1.5, ls="--", label="GC base / truth")
                ax_ratio[i, j].semilogx(ks[1:], ratio_model[1:], color=color, lw=1.8, label=f"{tag} / truth")

                if i == 0:
                    ax_raw[i, j].set_title(f"lead {lead_h}h", fontsize=11)
                    ax_ratio[i, j].set_title(f"lead {lead_h}h", fontsize=11)
                if j == 0:
                    ax_raw[i, j].set_ylabel(f"{var}\n$S_k$  (m$^4$/s$^2$)", fontsize=10)
                    ax_ratio[i, j].set_ylabel(f"{var}\n$S_k$ / $S_k^{{truth}}$", fontsize=10)
                if i == len(VARS) - 1:
                    ax_raw[i, j].set_xlabel("zonal wavenumber k", fontsize=10)
                    ax_ratio[i, j].set_xlabel("zonal wavenumber k", fontsize=10)
                ax_raw[i, j].grid(alpha=0.3, which="both")
                ax_ratio[i, j].grid(alpha=0.3, which="both")
                ax_ratio[i, j].set_ylim(0, 1.6)
        truth_plotted = True
        ds.close()

    for axarr in (ax_raw, ax_ratio):
        axarr[0, 0].legend(loc="lower left", fontsize=8)
    fig_raw.suptitle(f"Zonal energy spectrum (lat {LAT_BAND_DEG[0]}-{LAT_BAND_DEG[1]}°), 2022, res=2", fontsize=13)
    fig_ratio.suptitle(f"Spectrum ratio (model/truth), lat {LAT_BAND_DEG[0]}-{LAT_BAND_DEG[1]}°", fontsize=13)
    fig_raw.tight_layout(); fig_ratio.tight_layout()
    fig_raw.savefig(OUT_DIR / "spectrum_raw_v25_vs_v22.png", dpi=130, bbox_inches="tight")
    fig_ratio.savefig(OUT_DIR / "spectrum_ratio_v25_vs_v22.png", dpi=130, bbox_inches="tight")
    print(f"saved {OUT_DIR / 'spectrum_raw_v25_vs_v22.png'}")
    print(f"saved {OUT_DIR / 'spectrum_ratio_v25_vs_v22.png'}")


if __name__ == "__main__":
    main()
