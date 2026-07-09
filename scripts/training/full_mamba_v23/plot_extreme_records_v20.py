"""v20 extreme-records plotting — adapted from plot_extreme_records_v22.py.

v20 has 20 leads (max 120h), so only 4 rows (24h, 48h, 72h, 120h) — not the
v22-style 6 rows that extend to 240h.
"""
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

V20_KS = [1, 4, 6, 8, 10, 12, 14]
LEAD_K = [4, 8, 12, 20]   # 24h, 48h, 72h, 120h (v20 max = 120h)
T_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
W_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
DATA_DIR = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0520_v20_extreme"
OUT_BASE = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-20-v20/plots")
RECS = f"{DATA_DIR}/records_climatology_2015_2021.nc"


def bin_stats(exc, err, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = len(centers); rmse = np.full(n, np.nan); bias = np.full(n, np.nan); count = np.zeros(n, int)
    for i in range(n):
        m = (exc >= edges[i]) & (exc < edges[i + 1])
        if m.sum() > 0:
            e = err[m]
            rmse[i] = np.sqrt(np.mean(e ** 2)); bias[i] = np.mean(e); count[i] = int(m.sum())
    return count, rmse, bias, centers


def plot_one_Ktrain(K_train, preds, recs, out_path):
    panels_template = ["Heat RMSE", "Cold RMSE", "Wind RMSE", "Heat bias", "Cold bias", "Wind bias"]
    fig, axes = plt.subplots(len(LEAD_K), 6, figsize=(22, 3.0 * len(LEAD_K)))

    def _av(name, k_idx):
        return preds[name].isel(lead_h=k_idx).transpose("anchor_time", "lat", "lon").values

    for row, K_lead in enumerate(LEAD_K):
        k_idx = K_lead - 1
        lead_h = K_lead * 6
        t_tru = _av("2m_temperature_truth", k_idx)
        u_tru = _av("10m_u_component_of_wind_truth", k_idx)
        v_tru = _av("10m_v_component_of_wind_truth", k_idx)
        w_tru = np.sqrt(u_tru ** 2 + v_tru ** 2)
        t_b = _av("2m_temperature_baseline", k_idx)
        t_f = _av("2m_temperature_v18", k_idx)
        u_b = _av("10m_u_component_of_wind_baseline", k_idx)
        v_b = _av("10m_v_component_of_wind_baseline", k_idx)
        u_f = _av("10m_u_component_of_wind_v18", k_idx)
        v_f = _av("10m_v_component_of_wind_v18", k_idx)
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
                ax.plot(ctr, yf, "D-", color="C0", lw=1.8, ms=4, label=f"v20 K{K_train}")
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
        f"v20 K_train={K_train} chunk=24 d_conv=4 extreme-records: 2022 events vs 2015-2021 record "
        f"(rows = forecast lead, max 120h)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Loading climatology {RECS}")
    recs = xr.open_dataset(RECS)
    for K_train in V20_KS:
        preds_path = f"{DATA_DIR}/v20_K{K_train}_extreme_records_K20_2022.nc"
        if not Path(preds_path).exists():
            print(f"SKIP K={K_train}: missing {preds_path}")
            continue
        preds = xr.open_dataset(preds_path)
        out_dir = OUT_BASE / f"K{K_train}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "extreme_records_K20.png"
        print(f"Plotting K_train={K_train} → {out_path}")
        plot_one_Ktrain(K_train, preds, recs, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
