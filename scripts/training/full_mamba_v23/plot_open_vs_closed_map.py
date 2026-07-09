"""Per-cell ΔRMSE map: closed (warm_full) vs open (warm_bp).

ΔRMSE(x) = RMSE_closed(x) - RMSE_open(x)
- negative → closed better at that cell
- positive → open  better at that cell

Also plots an area-weighted (cos(lat)) histogram of ΔRMSE so you can see
the global distribution: is closed strictly worse globally (mass on +)?
or does closed have heavy negative tail at hotspots (mass on -) but
hurt everywhere else?
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
    p.add_argument("--nc-open",   required=True, help="K22 NC (bias_v22, variance_v22)")
    p.add_argument("--nc-closed", required=True, help="closed NC (bias_closed, variance_closed)")
    p.add_argument("--out-png",   required=True)
    return p.parse_args()


def rmse_from(bias, var):
    return np.sqrt(bias**2 + var)


def main():
    cfg = parse_args()
    op = xr.open_dataset(cfg.nc_open)
    cl = xr.open_dataset(cfg.nc_closed)

    lat = op.lat.values
    lon = op.lon.values
    n_lat, n_lon = len(lat), len(lon)
    cos_lat = np.cos(np.deg2rad(lat))
    # area_w(i,j) sums to 1 across the whole grid (cos(lat) replicated over lon).
    area_w = np.broadcast_to(cos_lat[:, None] / (cos_lat.sum() * n_lon),
                              (n_lat, n_lon))

    mse_b_open   = op.bias_baseline.values**2 + op.variance_baseline.values
    mse_open     = op.bias_v22.values**2      + op.variance_v22.values
    mse_b_close  = cl.bias_baseline.values**2 + cl.variance_baseline.values
    mse_close    = cl.bias_closed.values**2   + cl.variance_closed.values

    rmse_b_open  = np.sqrt(mse_b_open)
    rmse_open    = np.sqrt(mse_open)
    rmse_b_close = np.sqrt(mse_b_close)
    rmse_close   = np.sqrt(mse_close)

    dRMSE = rmse_close - rmse_open  # negative = closed better, positive = open better

    def area_sum(arr):
        return float(np.sum(arr * area_w))

    # Proper lat-weighted RMSE: sqrt(lat-weighted MSE)
    rmse_open_global  = np.sqrt(area_sum(mse_open))
    rmse_close_global = np.sqrt(area_sum(mse_close))
    rmse_b_global     = np.sqrt(area_sum(mse_b_open))

    print(f"Lead = {op.attrs.get('lead_hours','?')}h")
    print(f"baseline RMSE (lat-w)   = {rmse_b_global:.3f} K")
    print(f"v22 open RMSE (lat-w)   = {rmse_open_global:.3f} K  (warm_bp,  {op.attrs.get('n_anchors')} anchors)")
    print(f"v22 closed RMSE (lat-w) = {rmse_close_global:.3f} K  (warm_full,{cl.attrs.get('n_anchors')} anchors)")
    print(f"  open imp vs baseline  = {100*(1-rmse_open_global/rmse_b_global):+.2f}%")
    print(f"  closed imp vs baseline= {100*(1-rmse_close_global/rmse_b_global):+.2f}%")

    closed_better_frac = area_sum((dRMSE < 0).astype(float))
    print(f"\nClosed-better area fraction (lat-w) = {closed_better_frac*100:.1f}%")
    print(f"Open-better   area fraction (lat-w) = {(1-closed_better_frac)*100:.1f}%")
    print(f"mean ΔRMSE (lat-w)                  = {area_sum(dRMSE):+.3f} K")

    # Quantiles of dRMSE (area-weighted)
    flat_d = dRMSE.ravel()
    flat_w = area_w.ravel()
    order = np.argsort(flat_d)
    flat_d_s = flat_d[order]; flat_w_s = flat_w[order]
    cum_w = np.cumsum(flat_w_s)
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    print(f"\nArea-weighted quantiles of ΔRMSE = RMSE_closed - RMSE_open:")
    for q in qs:
        v = flat_d_s[np.searchsorted(cum_w, q)]
        print(f"  q{int(q*100):>2} = {v:+.3f} K")

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])

    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    vmax_rmse = float(max(np.percentile(rmse_open, 98), np.percentile(rmse_close, 98)))

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(rmse_open, origin="upper", extent=extent, cmap="viridis",
                    vmin=0, vmax=vmax_rmse, aspect="auto")
    ax1.set_title(f"RMSE  v22 open-loop (warm_bp)\nlat-w global = {rmse_open_global:.3f} K")
    plt.colorbar(im, ax=ax1, fraction=0.04, label="RMSE [K]")

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(rmse_close, origin="upper", extent=extent, cmap="viridis",
                    vmin=0, vmax=vmax_rmse, aspect="auto")
    ax2.set_title(f"RMSE  v22closed (warm_full self-rollout)\nlat-w global = {rmse_close_global:.3f} K")
    plt.colorbar(im, ax=ax2, fraction=0.04, label="RMSE [K]")

    # Diff map
    ax3 = fig.add_subplot(gs[1, :])
    vlim = float(max(abs(np.percentile(dRMSE, 2)), abs(np.percentile(dRMSE, 98))))
    im = ax3.imshow(dRMSE, origin="upper", extent=extent, cmap="RdBu_r",
                    vmin=-vlim, vmax=vlim, aspect="auto")
    ax3.set_title("ΔRMSE = RMSE(closed warm_full) − RMSE(open warm_bp)\n"
                  "blue = closed better, red = open better")
    plt.colorbar(im, ax=ax3, fraction=0.025, label="ΔRMSE [K]")

    # Histogram (area-weighted)
    ax4 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(-vlim*1.2, vlim*1.2, 60)
    weights = area_w.ravel()
    n, edges, _ = ax4.hist(dRMSE.ravel(), bins=bins, weights=weights,
                            color="C0", alpha=0.75, edgecolor="k", linewidth=0.3)
    ax4.axvline(0, color="k", lw=1)
    ax4.axvline(rmse_close_global - rmse_open_global, color="r", lw=2,
                label=f"global mean ΔRMSE = {rmse_close_global - rmse_open_global:+.3f} K")
    ax4.set_xlabel("ΔRMSE [K]")
    ax4.set_ylabel("area-weighted fraction (cos(lat))")
    ax4.set_title(f"Area-weighted histogram of ΔRMSE\n"
                  f"closed-better area = {closed_better_frac*100:.1f}%, "
                  f"open-better area = {(1-closed_better_frac)*100:.1f}%")
    ax4.grid(alpha=0.3); ax4.legend()

    # Cumulative
    ax5 = fig.add_subplot(gs[2, 1])
    sort_idx = np.argsort(dRMSE.ravel())
    cdf_x = dRMSE.ravel()[sort_idx]
    cdf_w = weights[sort_idx]
    cdf_y = np.cumsum(cdf_w)
    ax5.plot(cdf_x, cdf_y, lw=1.5)
    ax5.axvline(0, color="k", lw=1)
    ax5.axhline(closed_better_frac, color="r", lw=1, ls="--",
                label=f"area where ΔRMSE < 0 = {closed_better_frac*100:.1f}%")
    ax5.set_xlabel("ΔRMSE [K]")
    ax5.set_ylabel("cumulative area-weighted fraction")
    ax5.set_title("CDF of ΔRMSE (area-weighted)")
    ax5.grid(alpha=0.3); ax5.legend()

    fig.suptitle(
        f"v22closed vs v22 open, 2m_T at lead {op.attrs.get('lead_hours','?')}h "
        f"(open: warm_bp / closed: warm_full)\n"
        f"Each model evaluated in its training-matched mode.",
        fontsize=12)
    plt.tight_layout()
    Path(cfg.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {cfg.out_png}")


if __name__ == "__main__":
    main()
