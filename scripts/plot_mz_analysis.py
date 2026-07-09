#!/usr/bin/env python3
"""Generate the canonical set of MZ analysis plots from a saved infer tensor
bundle. Used by both the r=4 (step 400) and r=6 (step 200) analyses so they
share identical layouts.

Usage:
    python scripts/plot_mz_analysis.py \
        --tensors /path/to/tensors_stepNNN_firstsegs.npz \
        --out-dir /path/to/plots \
        --title-suffix "r=6 step 10000, MZ step 200"

Produces:
    timeseries_truth_vs_baseline_vs_mz.png
    timeseries_errors.png
    spatial_error_map_z500.png
    error_distribution.png
    zonal_rmse.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tensors", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--title-suffix", default="")
    p.add_argument("--location-lat", type=float, default=48.0)
    p.add_argument("--location-lon", type=float, default=320.0)
    args = p.parse_args()

    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    d = np.load(args.tensors, allow_pickle=True)
    # Squeeze batch dim [seg, T, B=1, lat, lon, F] -> [seg*T, lat, lon, F]
    baseline = d["baseline"].squeeze(2).reshape(-1, *d["baseline"].shape[3:])
    corrected = d["corrected"].squeeze(2).reshape(-1, *d["corrected"].shape[3:])
    truth = d["truth"].squeeze(2).reshape(-1, *d["truth"].shape[3:])
    tarr = d["times"].reshape(-1)
    lat = d["lat"]
    lon = d["lon"]
    levels = d["pressure_levels"]
    feat_slices = d["feature_slices"]
    z_start = int(feat_slices[1][0])
    z500_idx = int(np.where(levels == 500)[0][0])
    z500 = z_start + z500_idx
    msl_idx = 0
    nseg = int(d["baseline"].shape[0])
    seg_len = int(d["baseline"].shape[1])

    # Area weighting
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat = np.where(np.abs(lat) >= 89.99, 0, cos_lat)
    w_lat = cos_lat / cos_lat.sum()

    # Picked location nearest (user-provided lat/lon)
    lat_idx = int(np.argmin(np.abs(lat - args.location_lat)))
    lon_idx = int(np.argmin(np.abs(lon - args.location_lon)))
    loc_str = f"(lat={lat[lat_idx]:.0f}°N, lon={lon[lon_idx]-360 if lon[lon_idx]>180 else lon[lon_idx]:.0f}°W)"

    tarr_pd = pd.to_datetime(tarr)
    title_suf = args.title_suffix

    # ---------- FIG 1: time series truth vs baseline vs MZ ----------
    tru_msl = truth[:, lat_idx, lon_idx, msl_idx] / 100.0
    bsl_msl = baseline[:, lat_idx, lon_idx, msl_idx] / 100.0
    cor_msl = corrected[:, lat_idx, lon_idx, msl_idx] / 100.0
    tru_z500 = truth[:, lat_idx, lon_idx, z500] / 9.81
    bsl_z500 = baseline[:, lat_idx, lon_idx, z500] / 9.81
    cor_z500 = corrected[:, lat_idx, lon_idx, z500] / 9.81

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax = axes[0]
    ax.plot(tarr_pd, tru_msl, "k-", lw=2.5, label="truth (ERA5)")
    ax.plot(tarr_pd, bsl_msl, "-", color="steelblue", lw=1.8, label="baseline")
    ax.plot(tarr_pd, cor_msl, "-", color="crimson", lw=1.8, label="MZ corrected")
    ax.set_ylabel("MSL pressure (hPa)")
    ax.set_title(f"Mean Sea Level Pressure {loc_str}  [{title_suf}]")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    for k in range(1, nseg):
        ax.axvline(tarr_pd[k * seg_len], color="gray", ls=":", lw=0.8)
    ax = axes[1]
    ax.plot(tarr_pd, tru_z500, "k-", lw=2.5, label="truth")
    ax.plot(tarr_pd, bsl_z500, "-", color="steelblue", lw=1.8, label="baseline")
    ax.plot(tarr_pd, cor_z500, "-", color="crimson", lw=1.8, label="MZ corrected")
    ax.set_ylabel("500 hPa geopotential height (m)")
    ax.set_xlabel("datetime (UTC)")
    ax.set_title(f"Geopotential @ 500 hPa {loc_str}")
    ax.grid(alpha=0.3); ax.legend()
    for k in range(1, nseg):
        ax.axvline(tarr_pd[k * seg_len], color="gray", ls=":", lw=0.8)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / "timeseries_truth_vs_baseline_vs_mz.png", dpi=140, bbox_inches="tight")
    print("saved:", OUT / "timeseries_truth_vs_baseline_vs_mz.png")

    # ---------- FIG 2: forecast error over time ----------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax = axes[0]
    bsl_e = bsl_msl - tru_msl
    cor_e = cor_msl - tru_msl
    ax.plot(tarr_pd, bsl_e, "-", color="steelblue", lw=1.5, label="baseline − truth")
    ax.plot(tarr_pd, cor_e, "-", color="crimson", lw=1.5, label="MZ − truth")
    ax.fill_between(tarr_pd, bsl_e, cor_e, color="gold", alpha=0.3, label="|improvement|")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("MSLP error (hPa)")
    ax.set_title(f"Forecast error: MSLP {loc_str}  [{title_suf}]")
    ax.grid(alpha=0.3); ax.legend()
    for k in range(1, nseg):
        ax.axvline(tarr_pd[k * seg_len], color="gray", ls=":", lw=0.8)
    ax = axes[1]
    bsl_ez = bsl_z500 - tru_z500
    cor_ez = cor_z500 - tru_z500
    ax.plot(tarr_pd, bsl_ez, "-", color="steelblue", lw=1.5, label="baseline − truth")
    ax.plot(tarr_pd, cor_ez, "-", color="crimson", lw=1.5, label="MZ − truth")
    ax.fill_between(tarr_pd, bsl_ez, cor_ez, color="gold", alpha=0.3, label="|improvement|")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("z500 error (m)")
    ax.set_xlabel("datetime (UTC)")
    ax.set_title(f"Forecast error: z500 {loc_str}")
    ax.grid(alpha=0.3); ax.legend()
    for k in range(1, nseg):
        ax.axvline(tarr_pd[k * seg_len], color="gray", ls=":", lw=0.8)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / "timeseries_errors.png", dpi=140, bbox_inches="tight")
    print("saved:", OUT / "timeseries_errors.png")

    # ---------- FIG 3: spatial error map at worst-baseline z500 snapshot ----------
    z500_err = baseline[..., z500] - truth[..., z500]
    zonal_err = np.sqrt((z500_err ** 2 * w_lat[None, :, None]).sum(axis=1).mean(axis=-1))
    t_bad = int(np.argmax(zonal_err))
    bsl_e_map = (baseline[t_bad, :, :, z500] - truth[t_bad, :, :, z500]) / 9.81
    cor_e_map = (corrected[t_bad, :, :, z500] - truth[t_bad, :, :, z500]) / 9.81
    improvement = np.abs(bsl_e_map) - np.abs(cor_e_map)
    vmax = max(abs(bsl_e_map).max(), abs(cor_e_map).max())
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    LON, LAT = np.meshgrid(lon, lat)
    im = axes[0].pcolormesh(LON, LAT, bsl_e_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    axes[0].set_title(f"baseline error (m)\nt={t_bad} zonal-RMS={zonal_err[t_bad]:.1f} m")
    fig.colorbar(im, ax=axes[0], orientation="horizontal", pad=0.15, shrink=0.8)
    im = axes[1].pcolormesh(LON, LAT, cor_e_map, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    axes[1].set_title("MZ-corrected error (m)")
    fig.colorbar(im, ax=axes[1], orientation="horizontal", pad=0.15, shrink=0.8)
    imax = np.abs(improvement).max()
    im = axes[2].pcolormesh(LON, LAT, improvement, cmap="RdYlGn", vmin=-imax, vmax=imax, shading="auto")
    axes[2].set_title(f"|baseline|−|MZ|  (green=MZ better)\nmean = {improvement.mean():+.2f} m")
    fig.colorbar(im, ax=axes[2], orientation="horizontal", pad=0.15, shrink=0.8)
    for ax in axes:
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.contour(LON, LAT, truth[t_bad, :, :, z500] / 9.81, levels=10, colors="black", linewidths=0.5, alpha=0.4)
    fig.suptitle(f"z500 forecast error at worst-baseline snapshot  [{title_suf}]", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "spatial_error_map_z500.png", dpi=140, bbox_inches="tight")
    print("saved:", OUT / "spatial_error_map_z500.png")

    # ---------- FIG 4: error distribution ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    w = np.broadcast_to(w_lat[None, :, None], baseline.shape[:3])
    w_flat = w.ravel()
    for ax_i, (feat_idx, name, unit, scale) in enumerate([
        (msl_idx, "MSL pressure", "hPa", 100.0),
        (z500, "z500 height", "m", 9.81),
    ]):
        ax = axes[ax_i]
        bsl_errs = ((baseline[..., feat_idx] - truth[..., feat_idx]) / scale).ravel()
        cor_errs = ((corrected[..., feat_idx] - truth[..., feat_idx]) / scale).ravel()
        lo, hi = np.quantile(bsl_errs, [0.001, 0.999])
        bins = np.linspace(lo, hi, 80)
        bsl_rmse = np.sqrt((bsl_errs ** 2 * w_flat).sum() / w_flat.sum())
        cor_rmse = np.sqrt((cor_errs ** 2 * w_flat).sum() / w_flat.sum())
        ax.hist(bsl_errs, bins=bins, weights=w_flat, alpha=0.55,
                label=f"baseline  RMSE={bsl_rmse:.2f}", color="steelblue", density=True)
        ax.hist(cor_errs, bins=bins, weights=w_flat, alpha=0.55,
                label=f"MZ         RMSE={cor_rmse:.2f}", color="crimson", density=True)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel(f"forecast error ({unit})")
        ax.set_ylabel("area-weighted density")
        ax.set_title(f"{name}: error distribution  [{title_suf}]")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "error_distribution.png", dpi=140, bbox_inches="tight")
    print("saved:", OUT / "error_distribution.png")

    # ---------- FIG 5: zonal RMSE vs latitude ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax_i, (feat_idx, name, unit, scale) in enumerate([
        (msl_idx, "MSL pressure", "hPa", 100.0),
        (z500, "z500 height", "m", 9.81),
    ]):
        ax = axes[ax_i]
        bsl_rmse_zonal = np.sqrt(np.mean(((baseline[..., feat_idx] - truth[..., feat_idx]) / scale) ** 2, axis=(0, 2)))
        cor_rmse_zonal = np.sqrt(np.mean(((corrected[..., feat_idx] - truth[..., feat_idx]) / scale) ** 2, axis=(0, 2)))
        ax.plot(bsl_rmse_zonal, lat, "-", color="steelblue", lw=2.2, label="baseline")
        ax.plot(cor_rmse_zonal, lat, "-", color="crimson", lw=2.2, label="MZ-corrected")
        ax.fill_betweenx(lat, cor_rmse_zonal, bsl_rmse_zonal,
                         where=(bsl_rmse_zonal > cor_rmse_zonal),
                         color="green", alpha=0.2, label="MZ better")
        ax.set_xlabel(f"RMSE ({unit})"); ax.set_ylabel("latitude")
        ax.set_title(f"Zonal RMSE: {name}")
        ax.grid(alpha=0.3); ax.legend(); ax.set_ylim(-90, 90)
    fig.suptitle(f"Zonally averaged RMSE  [{title_suf}]", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "zonal_rmse.png", dpi=140, bbox_inches="tight")
    print("saved:", OUT / "zonal_rmse.png")


if __name__ == "__main__":
    main()
