#!/usr/bin/env python
"""K-scan comparison: v22 (d_conv=4) vs v23-fresh (d_conv=8 from scratch).

Outputs 3 plots:
1. MSE-improvement vs K, at 4 lead times (4 panels)
2. WB-5 spectrum ratio @ 240h vs K (overlay all K)
3. Same plot for short lead (24h) showing the reverse story
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

R_EARTH = 6371_000.0
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/"
               "results/2026-05-30-v23/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W_VAR = {
    "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
    "2m_temperature": 1.0, "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1, "geopotential": 1.0,
    "temperature": 1.0, "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0, "vertical_velocity": 1.0, "specific_humidity": 1.0,
}
sigma_ds = xr.open_dataset(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/"
    "stats/diffs_stddev_by_level.nc")
SIGMA = {v: float(sigma_ds[v].values if sigma_ds[v].values.ndim == 0
                  else sigma_ds[v].values.mean())
         for v in W_VAR if v in sigma_ds}


def mse_imp(ev: dict, k_idx: int) -> float:
    tb = tf = 0.0
    for v in ev["per_variable_per_step"]:
        if v not in SIGMA: continue
        w = W_VAR.get(v, 1.0); s = SIGMA[v]
        rb = ev["per_variable_per_step"][v]["rmse_baseline"][k_idx]
        rf = ev["per_variable_per_step"][v]["rmse_full"][k_idx]
        tb += w * rb ** 2 / s ** 2
        tf += w * rf ** 2 / s ** 2
    return (1 - tf / tb) * 100 if tb > 0 else 0


# ---------- 1. MSE-improvement vs K, 4 panels (leads) ----------
def plot_kscan_mse() -> None:
    KS_v22 = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    KS_v23 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]  # no fresh K=1 eval
    leads = [(24, 3), (72, 11), (168, 27), (240, 39)]
    V22 = "/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons"
    V23F = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_eval_K40"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    for col, (lead_h, k_idx) in enumerate(leads):
        ax = axes[col]
        v22_imp = []
        for K in KS_v22:
            p = f"{V22}/v22_K{K}_K40.json"
            try:
                v22_imp.append(mse_imp(json.load(open(p)), k_idx))
            except Exception:
                v22_imp.append(np.nan)
        v23_imp = []
        for K in KS_v23:
            p = f"{V23F}/v23_K{K}_fresh_K40.json"
            try:
                v23_imp.append(mse_imp(json.load(open(p)), k_idx))
            except Exception:
                v23_imp.append(np.nan)

        ax.axhline(0, color="k", lw=0.5)
        ax.plot(KS_v22, v22_imp, "o-", color="C3", lw=2, ms=7,
                label="v22 (d_conv=4)")
        ax.plot(KS_v23, v23_imp, "s-", color="C2", lw=2, ms=7,
                label="v23 fresh (d_conv=8)")
        ax.set_xlabel("K (training AR-tail length)")
        ax.set_ylabel("MSE-improvement vs baseline (%)" if col == 0 else "")
        ax.set_title(f"lead = {lead_h}h ({lead_h // 24}d)")
        ax.set_xticks(KS_v22)
        ax.grid(alpha=0.3)
        if col == 3:
            ax.legend(loc="upper left", fontsize=10)

    fig.suptitle(
        "K-scan: v22 (d_conv=4) vs v23 fresh (d_conv=8). "
        "At long lead, v22 ≥ v23 (over-smoothing reward). "
        "At 24h, large K hurts both (long-AR training sacrifices short lead).",
        fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "kscan_v22_vs_v23fresh_mse_4leads.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


# ---------- 2. WB-5 spectrum ratio @ 240h vs K (v22 vs v23-fresh overlay) ----------
def zonal_spectrum(field_3d: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    n_lon = field_3d.shape[-1]
    F = np.fft.rfft(field_3d, axis=-1) / n_lon
    power = np.abs(F) ** 2
    C = 2 * np.pi * R_EARTH * np.cos(np.deg2rad(lat_deg))
    C = C[None, :, None]
    S = 2 * C * power
    S[:, :, 0] /= 2
    return S


def avg_band(S: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    mask = (np.abs(lat_deg) >= 30) & (np.abs(lat_deg) <= 60)
    return S[:, mask, :].mean(axis=(0, 1))


def load_var_spectrum(nc_path: str, var: str, lead_idx: int) -> dict:
    ds = xr.open_dataset(nc_path)
    lat_deg = ds["lat"].values.astype("float32")
    out = {}
    for tag, suf in [("truth", "_truth"), ("baseline", "_baseline"),
                      ("model", "_v18")]:
        f = (ds[var + suf].isel(lead_h=lead_idx)
             .transpose("anchor_time", "lat", "lon").values)
        out[tag] = avg_band(zonal_spectrum(f, lat_deg), lat_deg)
    ds.close()
    return out


def plot_kscan_spectrum(lead_h: int, var: str) -> None:
    k_idx = lead_h // 6 - 1
    V22_DIR = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0522_v22_extreme"
    V23F_DIR = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0526_v23_extreme"

    KS_v22 = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    KS_v23 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = cm.get_cmap("viridis")
    for ax, (title, ks, dir_, fname_fmt) in zip(
        axes,
        [("v22 K-scan (d_conv=4)", KS_v22, V22_DIR, "v22_K{K}_extreme_records_K40_2022.nc"),
         ("v23 FRESH K-scan (d_conv=8)", KS_v23, V23F_DIR, "v23_K{K}_fresh_extreme_records_K40_2022.nc")]):
        norm = plt.Normalize(min(ks), max(ks))
        S_truth = None
        S_base = None
        for K in ks:
            p = f"{dir_}/{fname_fmt.format(K=K)}"
            try:
                d = load_var_spectrum(p, var, k_idx)
            except Exception as e:
                print(f"skip {p}: {e}")
                continue
            if S_truth is None:
                S_truth = d["truth"]; S_base = d["baseline"]
                k_arr = np.arange(len(S_truth))
                ax.axhline(1.0, color="k", lw=1.2, label="truth (ratio=1)")
                ax.semilogx(k_arr[1:], S_base[1:] / S_truth[1:],
                            color="grey", lw=1.4, ls="--", label="GC baseline")
            ax.semilogx(k_arr[1:], d["model"][1:] / S_truth[1:],
                        color=cmap(norm(K)), lw=1.4, alpha=0.85, label=f"K={K}")
        ax.set_title(title)
        ax.set_xlabel("zonal wavenumber k")
        ax.set_ylim(0, 1.3)
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    axes[0].set_ylabel(f"S_k(forecast) / S_k(truth)")
    fig.suptitle(
        f"K-scan WB-5 spectrum ratio @ lead {lead_h}h ({var}); |lat| ∈ [30°, 60°]. "
        f"Lower curves = more over-smoothing.", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / f"kscan_spectrum_{var}_lead{lead_h}h_v22_vs_v23fresh.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    plot_kscan_mse()
    for lead_h in [24, 240]:
        for var in ["10m_u_component_of_wind", "2m_temperature"]:
            plot_kscan_spectrum(lead_h, var)
