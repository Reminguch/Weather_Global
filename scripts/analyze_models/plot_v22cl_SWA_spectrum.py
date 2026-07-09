"""Zonal energy spectrum analysis for v22cl SWA cold_full across K=2..22.

Reads the extreme-records NC files (which contain full global grids for
2m_T + 10m_u + 10m_v at 40 leads × 118 anchors) and produces:

  1. Per-K raw log-log spectra (03_spectra style) — 3 vars, multi-lead panels
     -> 11 figures showing v22cl SWA vs GC baseline vs ERA5 truth power at
        each wavenumber
  2. K-scan spectrum RATIO plot (model / truth) — 1 figure per variable, all
     K as color gradient, 4-lead panels
     -> shows over-smoothing behaviour as K increases

Output dir: /home/lm8598/Weather_Global_experiments/results/2026-07-04-v22cl-SWA-spectrum/
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

R_EARTH = 6371_000.0

NC_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/2026-07-03-v22cl-SWA-extreme")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-04-v22cl-SWA-spectrum")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
VAR_LABEL = {"2m_temperature": "2m_T",
             "10m_u_component_of_wind": "10m_u",
             "10m_v_component_of_wind": "10m_v"}
LEADS_TO_PLOT_H = [24, 72, 168, 240]
LAT_BAND_DEG = (30.0, 60.0)

TRUTH_SUFFIX = "_truth"
BASELINE_SUFFIX = "_baseline"
MODEL_SUFFIX = "_v18"


def zonal_spectrum(field_3d: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Return power spectrum of shape (n_anchor, n_lat, n_wavenumber)."""
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


def plot_per_K_raw(spectra):
    """One figure per K: 3 vars × 4 leads = 12 panels.
    Log-log axes; 3 lines per panel (truth, baseline, v22cl SWA)."""
    for K in sorted(spectra):
        fig, axes = plt.subplots(len(VARS), len(LEADS_TO_PLOT_H),
                                 figsize=(5 * len(LEADS_TO_PLOT_H), 4 * len(VARS)))
        for row, var in enumerate(VARS):
            for col, lead_h in enumerate(LEADS_TO_PLOT_H):
                ax = axes[row, col]
                d = spectra[K][var].get(lead_h)
                if d is None:
                    ax.set_title(f"{VAR_LABEL[var]} lead {lead_h}h\n(no data)")
                    continue
                k = np.arange(len(d["truth"]))
                k_plot = k[1:]
                ax.loglog(k_plot, d["truth"][1:], "k-", lw=2.0, label="ERA5 truth")
                ax.loglog(k_plot, d["baseline"][1:], "-", color="gray",
                          lw=1.4, alpha=0.85, label="GC baseline")
                ax.loglog(k_plot, d["model"][1:], "-", color="C0",
                          lw=1.4, label=f"v22cl K={K} SWA")
                if row == 0:
                    ax.set_title(f"lead {lead_h}h", fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"{VAR_LABEL[var]}\nS_k", fontsize=11)
                if row == len(VARS) - 1:
                    ax.set_xlabel("zonal wavenumber k")
                ax.grid(which="both", alpha=0.3)
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"v22cl K={K} SWA (cold_full) zonal energy spectrum\n"
                     f"lat band |lat|∈[{LAT_BAND_DEG[0]:.0f}°,{LAT_BAND_DEG[1]:.0f}°], "
                     "2022 anchors averaged",
                     fontsize=12)
        plt.tight_layout()
        out = OUT_DIR / f"spectra_K{K}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")


def plot_Kscan_ratio(spectra):
    """One figure per variable: 4 lead panels; all K curves as viridis gradient.
    y = S_k(model) / S_k(truth); <1 = missing power at high k (over-smoothing)."""
    cmap = cm.get_cmap("viridis")
    Ks_valid = sorted(spectra)
    norm = plt.Normalize(min(Ks_valid), max(Ks_valid))
    for var in VARS:
        fig, axes = plt.subplots(1, len(LEADS_TO_PLOT_H),
                                 figsize=(5.5 * len(LEADS_TO_PLOT_H), 5),
                                 sharey=True)
        for col, lead_h in enumerate(LEADS_TO_PLOT_H):
            ax = axes[col]
            ref_K = Ks_valid[0]
            ref = spectra[ref_K][var].get(lead_h)
            if ref is None:
                ax.set_title(f"lead {lead_h}h\n(no data)"); continue
            S_truth = ref["truth"]; S_base = ref["baseline"]
            k = np.arange(len(S_truth))
            k_plot = k[1:]
            ax.axhline(1.0, color="k", lw=1.4, label="truth (=1)")
            ax.semilogx(k_plot, S_base[1:] / S_truth[1:],
                        color="grey", lw=1.5, ls="--", label="GC baseline")
            for K in Ks_valid:
                d = spectra[K][var].get(lead_h)
                if d is None: continue
                ax.semilogx(k_plot, d["model"][1:] / S_truth[1:],
                            color=cmap(norm(K)), lw=1.4, alpha=0.9,
                            label=f"K={K}")
            ax.set_title(f"lead = {lead_h}h", fontsize=11)
            ax.set_xlabel("zonal wavenumber k")
            if col == 0:
                ax.set_ylabel("S_k(forecast) / S_k(truth)")
            ax.set_ylim(0, 1.3)
            ax.grid(which="both", alpha=0.3)
            if col == len(LEADS_TO_PLOT_H) - 1:
                ax.legend(fontsize=7, loc="lower left", ncol=2)
        fig.suptitle(f"v22cl SWA cold_full K-scan spectrum ratio — {VAR_LABEL[var]}\n"
                     f"lat |{LAT_BAND_DEG[0]:.0f}°|-{LAT_BAND_DEG[1]:.0f}°, "
                     "<1 at high k → over-smoothing",
                     fontsize=11)
        plt.tight_layout()
        out = OUT_DIR / f"spectrum_ratio_Kscan_{var}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")


def main():
    spectra = {}
    for K in KS:
        p = NC_DIR / f"v22cl_K{K}_SWA_extreme_records_K40_2022.nc"
        if not p.exists():
            print(f"SKIP K={K}: missing {p}"); continue
        print(f"Loading K={K}: {p}")
        ds = xr.open_dataset(p)
        lat_deg = ds["lat"].values.astype("float32")
        n_lead = ds.sizes["lead_h"]
        spectra[K] = {var: {} for var in VARS}
        for lead_h in LEADS_TO_PLOT_H:
            k_idx = lead_h // 6 - 1
            if k_idx < 0 or k_idx >= n_lead: continue
            for var in VARS:
                spectra[K][var][lead_h] = spectra_for_var_lead(ds, var, k_idx, lat_deg)
        ds.close()

    if not spectra:
        print("No data loaded."); return
    plot_per_K_raw(spectra)
    plot_Kscan_ratio(spectra)
    print(f"\nDone. All figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
