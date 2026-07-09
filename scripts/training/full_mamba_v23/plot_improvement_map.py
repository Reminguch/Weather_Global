"""Map per-cell RMSE improvement of v22 vs baseline at lead 240h.

Reads bias map NC (output of compute_per_cell_bias_map.py) and computes:
  per-cell improvement_pct = (RMSE_v22 - RMSE_baseline) / RMSE_baseline * 100
Shows where the +5.85% global improvement actually comes from.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-nc", required=True)
    p.add_argument("--out-png", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    ds = xr.open_dataset(cfg.in_nc)
    lat = ds["lat"].values
    lon = ds["lon"].values

    bias_b = ds["bias_baseline"].values     # (lat, lon)
    bias_f = ds["bias_v22"].values
    var_b = ds["variance_baseline"].values
    var_f = ds["variance_v22"].values

    mse_b = bias_b**2 + var_b
    mse_f = bias_f**2 + var_f
    rmse_b = np.sqrt(mse_b)
    rmse_f = np.sqrt(mse_f)

    # Per-cell improvement_pct
    improvement_pct = (rmse_f - rmse_b) / np.maximum(rmse_b, 1e-12) * 100  # negative = v22 better
    delta_mse = mse_f - mse_b
    delta_bias_sq = bias_f**2 - bias_b**2
    delta_var = var_f - var_b

    # Lat weights
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_w = cos_lat / cos_lat.mean()

    # Global aggregates (for the title — should match earlier 5.8% report)
    glob_mse_b = float((mse_b * cos_lat_w[:, None]).mean())
    glob_mse_f = float((mse_f * cos_lat_w[:, None]).mean())
    glob_rmse_b = np.sqrt(glob_mse_b)
    glob_rmse_f = np.sqrt(glob_mse_f)
    glob_imp = (glob_rmse_f - glob_rmse_b) / glob_rmse_b * 100

    # Largest contributors to RMSE² improvement (negative delta_mse helps)
    flat_weighted = (delta_mse * cos_lat_w[:, None])  # contribution sign
    # Top 10 cells where v22 hurts most (positive)
    flat_idx_sorted_hurt = np.argsort(flat_weighted.ravel())[::-1]
    # Top 10 cells where v22 helps most (negative)
    flat_idx_sorted_help = np.argsort(flat_weighted.ravel())

    # Find center longitude range to make map readable
    # convert lon 0..360 to -180..180 for visual
    lon_shifted = ((lon + 180) % 360) - 180

    # 4 panels figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))

    # 1. baseline RMSE
    ax = axes[0, 0]
    im = ax.pcolormesh(lon, lat, rmse_b, cmap="viridis", shading="auto", vmin=0, vmax=8)
    ax.set_title(f"(a) baseline RMSE map [K]\nglobal lat-w RMSE = {glob_rmse_b:.3f} K")
    plt.colorbar(im, ax=ax, label="RMSE (K)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    # 2. v22 RMSE
    ax = axes[0, 1]
    im = ax.pcolormesh(lon, lat, rmse_f, cmap="viridis", shading="auto", vmin=0, vmax=8)
    ax.set_title(f"(b) v22 RMSE map [K]\nglobal lat-w RMSE = {glob_rmse_f:.3f} K")
    plt.colorbar(im, ax=ax, label="RMSE (K)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    # 3. Per-cell improvement (negative = v22 better)
    ax = axes[1, 0]
    vmax = max(abs(np.nanpercentile(improvement_pct, 2)),
               abs(np.nanpercentile(improvement_pct, 98)))
    im = ax.pcolormesh(lon, lat, improvement_pct, cmap="RdBu_r",
                       shading="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(f"(c) per-cell ΔRMSE % (v22 - baseline) / baseline\n"
                 f"BLUE = v22 better, RED = v22 worse;  global = {glob_imp:+.2f}%")
    plt.colorbar(im, ax=ax, label="% change in RMSE")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    # 4. Histogram of per-cell improvement
    ax = axes[1, 1]
    weights = np.broadcast_to(cos_lat_w[:, None], improvement_pct.shape).ravel()
    ax.hist(improvement_pct.ravel(), bins=60, weights=weights / weights.sum(),
            color="C0", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.axvline(glob_imp, color="red", lw=2, label=f"global = {glob_imp:+.2f}%")
    pct_better = (improvement_pct.ravel() < 0).mean() * 100
    ax.set_title(f"(d) lat-weighted histogram of per-cell ΔRMSE %\n"
                 f"{pct_better:.1f}% of cells: v22 better")
    ax.set_xlabel("ΔRMSE (%)")
    ax.set_ylabel("lat-weighted density")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"v22 K=22 vs baseline (2m_T, lead 240h, n_anchors={ds.attrs.get('n_anchors')}, "
        f"warm_bp eval)\n"
        f"Per-cell MSE breakdown: Δbias² = {(delta_bias_sq * cos_lat_w[:, None]).mean():+.4f} K²  |  "
        f"Δvariance = {(delta_var * cos_lat_w[:, None]).mean():+.4f} K²",
        fontsize=12)

    plt.tight_layout()
    Path(cfg.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {cfg.out_png}")

    # Print text summary: regional aggregates
    print("\n=== REGIONAL BREAKDOWN (lat bands) ===")
    print(f"{'band':<22}  {'cells':>8}  {'base RMSE':>10}  {'v22 RMSE':>10}  {'Δ%':>8}")
    bands = [
        ("Arctic (lat > 70°)", lat > 70),
        ("NH temperate (30..70°)", (lat > 30) & (lat <= 70)),
        ("Tropics (-30..30°)", (lat >= -30) & (lat <= 30)),
        ("SH temperate (-70..-30°)", (lat < -30) & (lat >= -70)),
        ("Antarctic (lat < -70°)", lat < -70),
    ]
    for name, lat_mask in bands:
        idx = np.where(lat_mask)[0]
        if len(idx) == 0:
            continue
        w_band = cos_lat_w[idx][:, None]
        n_cells = lat_mask.sum() * len(lon)
        mse_b_band = (mse_b[idx] * w_band).mean()
        mse_f_band = (mse_f[idx] * w_band).mean()
        rb = np.sqrt(mse_b_band); rf = np.sqrt(mse_f_band)
        imp = (rf - rb) / rb * 100
        print(f"{name:<22}  {n_cells:>8}  {rb:>10.3f}  {rf:>10.3f}  {imp:>+7.2f}%")

    # Land/ocean split via lat patterns (no land mask available; use rough heuristic)
    print(f"\n=== TOP cells where v22 HELPS most ===")
    for rank, flat_i in enumerate(flat_idx_sorted_help[:8]):
        i, j = np.unravel_index(flat_i, mse_b.shape)
        lon_v = lon_shifted[j] if abs(lon_shifted[j]) <= 180 else lon[j]
        print(f"  #{rank+1}: lat={lat[i]:+.1f}, lon={lon[j]:.1f}  "
              f"base_RMSE={np.sqrt(mse_b[i,j]):.2f}K  v22_RMSE={np.sqrt(mse_f[i,j]):.2f}K  "
              f"Δ={improvement_pct[i,j]:+.1f}%")
    print(f"\n=== TOP cells where v22 HURTS most ===")
    for rank, flat_i in enumerate(flat_idx_sorted_hurt[:8]):
        i, j = np.unravel_index(flat_i, mse_b.shape)
        print(f"  #{rank+1}: lat={lat[i]:+.1f}, lon={lon[j]:.1f}  "
              f"base_RMSE={np.sqrt(mse_b[i,j]):.2f}K  v22_RMSE={np.sqrt(mse_f[i,j]):.2f}K  "
              f"Δ={improvement_pct[i,j]:+.1f}%")


if __name__ == "__main__":
    main()
