"""v22cl SWA extreme-records plotting (adapted from plot_extreme_records_v23.py).

6 rows (24h, 48h, 72h, 120h, 168h, 240h) × 6 cols (Heat/Cold/Wind × RMSE/bias).
Compares SWA closed-loop residual vs GC baseline for pointwise record-exceedance
events (heat, cold, wind extremes).

Usage:
  python plot_v22cl_SWA_extreme_records.py [K]
    K in {2,4,6,8,10,12,14,16,18,20,22} (default: all)
"""
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LEAD_K = [4, 8, 12, 20, 28, 40]  # 24h, 48h, 72h, 120h, 168h, 240h
T_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
W_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

DATA_DIR = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/2026-07-03-v22cl-SWA-extreme"
CLIMATOLOGY = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0522_v22_extreme/records_climatology_2015_2021.nc"
OUT_BASE = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-03-v22cl-SWA-extreme/plots")
OUT_BASE.mkdir(parents=True, exist_ok=True)

KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]


def bin_stats(exc, err, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = len(centers)
    rmse = np.full(n, np.nan); bias = np.full(n, np.nan); count = np.zeros(n, int)
    for i in range(n):
        m = (exc >= edges[i]) & (exc < edges[i + 1])
        if m.sum() > 0:
            e = err[m]
            rmse[i] = np.sqrt(np.mean(e ** 2)); bias[i] = np.mean(e); count[i] = int(m.sum())
    return count, rmse, bias, centers


def plot_one(K: int, preds: xr.Dataset, recs: xr.Dataset, out_path: Path) -> None:
    panels_template = ["Heat RMSE", "Cold RMSE", "Wind RMSE", "Heat bias", "Cold bias", "Wind bias"]
    fig, axes = plt.subplots(len(LEAD_K), 6, figsize=(22, 3.0 * len(LEAD_K)))

    def _av(name, k_idx):
        return preds[name].isel(lead_h=k_idx).transpose("anchor_time", "lat", "lon").values

    BASE = "_baseline"
    FULL = "_v18"  # nc convention (residual model output stored under _v18 key)

    for row, K_lead in enumerate(LEAD_K):
        k_idx = K_lead - 1
        lead_h = K_lead * 6
        t_tru = _av("2m_temperature_truth", k_idx)
        u_tru = _av("10m_u_component_of_wind_truth", k_idx)
        v_tru = _av("10m_v_component_of_wind_truth", k_idx)
        w_tru = np.sqrt(u_tru ** 2 + v_tru ** 2)
        t_b = _av("2m_temperature" + BASE, k_idx); t_f = _av("2m_temperature" + FULL, k_idx)
        u_b = _av("10m_u_component_of_wind" + BASE, k_idx); v_b = _av("10m_v_component_of_wind" + BASE, k_idx)
        u_f = _av("10m_u_component_of_wind" + FULL, k_idx); v_f = _av("10m_v_component_of_wind" + FULL, k_idx)
        w_b = np.sqrt(u_b ** 2 + v_b ** 2); w_f = np.sqrt(u_f ** 2 + v_f ** 2)

        anc_times = preds.anchor_time.values
        valid_times = anc_times + np.timedelta64(lead_h, "h")
        doys = pd.DatetimeIndex(valid_times).dayofyear.values
        rmx = recs["t2m_record_max"].sel(doy=xr.DataArray(doys, dims="anchor_time")).transpose("anchor_time", "lat", "lon").values
        rmn = recs["t2m_record_min"].sel(doy=xr.DataArray(doys, dims="anchor_time")).transpose("anchor_time", "lat", "lon").values
        rwmx = recs["wind10m_record_max"].sel(doy=xr.DataArray(doys, dims="anchor_time")).transpose("anchor_time", "lat", "lon").values

        exc_h = (t_tru - rmx).ravel(); err_b_t = (t_b - t_tru).ravel(); err_f_t = (t_f - t_tru).ravel()
        exc_c = (rmn - t_tru).ravel()
        exc_w = (w_tru - rwmx).ravel(); err_b_w = (w_b - w_tru).ravel(); err_f_w = (w_f - w_tru).ravel()
        mh = exc_h > 0; mc = exc_c > 0; mw = exc_w > 0
        data_panels = [
            ("heat", exc_h[mh], err_b_t[mh], err_f_t[mh], T_EDGES),
            ("cold", exc_c[mc], err_b_t[mc], err_f_t[mc], T_EDGES),
            ("wind", exc_w[mw], err_b_w[mw], err_f_w[mw], W_EDGES),
        ]
        for col_offset in [0, 3]:
            metric = "rmse" if col_offset == 0 else "bias"
            for j, (name, exc, eb, ef, edges) in enumerate(data_panels):
                ax = axes[row, col_offset + j]
                cb, rb, bib, ctr = bin_stats(exc, eb, edges)
                cf, rf, bif, _ = bin_stats(exc, ef, edges)
                yb = rb if metric == "rmse" else bib
                yf = rf if metric == "rmse" else bif
                ax.plot(ctr, yb, "o-", color="gray", lw=1.8, ms=5, label="GC base")
                ax.plot(ctr, yf, "D-", color="C0", lw=1.8, ms=4, label=f"v22cl K{K} SWA")
                if metric == "bias":
                    ax.axhline(0, color="k", lw=0.5)
                if row == 0:
                    ax.set_title(panels_template[col_offset + j], fontsize=9)
                if col_offset + j == 0:
                    ax.set_ylabel(f"lead K={K_lead} ({lead_h}h)", fontsize=10)
                ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
                if row == 0 and col_offset + j == 0:
                    ax.legend(fontsize=7)
    fig.suptitle(
        f"v22cl K={K} SWA (cold_full closed-loop) extreme-records: "
        f"2022 events vs 2015-2021 record (rows = forecast lead, max 240h)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    ks_to_run = [int(sys.argv[1])] if len(sys.argv) > 1 else KS
    recs = xr.open_dataset(CLIMATOLOGY)
    for K in ks_to_run:
        preds_path = f"{DATA_DIR}/v22cl_K{K}_SWA_extreme_records_K40_2022.nc"
        if not Path(preds_path).exists():
            print(f"SKIP K={K}: missing {preds_path}")
            continue
        preds = xr.open_dataset(preds_path)
        out_dir = OUT_BASE / f"K{K}"; out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "extreme_records_K40.png"
        print(f"Plotting K={K} → {out_path}")
        plot_one(K, preds, recs, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
