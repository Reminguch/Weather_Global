"""3-way extreme records comparison (res=2):
  - GC small baseline (frozen)
  - v22 K=18 step20000 (residual mamba, NO extreme head)
  - v25 phase 1.5 step 5000 (residual mamba + GATED extreme head, gate_bias=-1)

Same layout as 2026-05-23-v22/plots/K{K}/extreme_records_K40.png:
  6 rows (leads 24, 48, 72, 120, 168, 240h) × 6 cols (Heat/Cold/Wind × RMSE/bias).
Each panel: 3 curves (GC base, v22 K=18, v25 phase1.5) vs amount truth event
exceeds the 2015-2021 record.
"""
from __future__ import annotations
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

LEAD_K   = [4, 8, 12, 20, 28, 40]  # 24, 48, 72, 120, 168, 240h
T_EDGES  = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
W_EDGES  = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

DATA_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/2026-6-27-v25-extreme")
RECS_PATH = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/0522_v22_extreme/records_climatology_2015_2021.nc"
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-28-v25-extreme/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bin_stats(exc, err, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = len(centers); rmse = np.full(n, np.nan); bias = np.full(n, np.nan); count = np.zeros(n, int)
    for i in range(n):
        m = (exc >= edges[i]) & (exc < edges[i+1])
        if m.sum() > 0:
            e = err[m]
            rmse[i] = np.sqrt(np.mean(e**2)); bias[i] = np.mean(e); count[i] = int(m.sum())
    return count, rmse, bias, centers


def _av(ds, name, k_idx):
    return ds[name].isel(lead_h=k_idx).transpose("anchor_time", "lat", "lon").values


def plot_3way(ds_v22, ds_v25p15, recs, out_path: Path, title_suffix=""):
    panels_template = ["Heat RMSE", "Cold RMSE", "Wind RMSE", "Heat bias", "Cold bias", "Wind bias"]
    fig, axes = plt.subplots(len(LEAD_K), 6, figsize=(22, 3.0*len(LEAD_K)))

    for row, K_lead in enumerate(LEAD_K):
        k_idx = K_lead - 1
        lead_h = K_lead * 6

        # Use v22's anchors (assume v22 and v25p15 share same anchor schedule)
        t_tru = _av(ds_v22, "2m_temperature_truth", k_idx)
        u_tru = _av(ds_v22, "10m_u_component_of_wind_truth", k_idx)
        v_tru = _av(ds_v22, "10m_v_component_of_wind_truth", k_idx)
        w_tru = np.sqrt(u_tru**2 + v_tru**2)

        # Baseline (GC small, frozen — same across both files; take from v22)
        t_b   = _av(ds_v22, "2m_temperature_baseline", k_idx)
        u_b   = _av(ds_v22, "10m_u_component_of_wind_baseline", k_idx)
        v_b   = _av(ds_v22, "10m_v_component_of_wind_baseline", k_idx)
        w_b   = np.sqrt(u_b**2 + v_b**2)

        # v22 K=18 (no extreme head)
        t_v22 = _av(ds_v22, "2m_temperature_v18", k_idx)
        u_v22 = _av(ds_v22, "10m_u_component_of_wind_v18", k_idx)
        v_v22 = _av(ds_v22, "10m_v_component_of_wind_v18", k_idx)
        w_v22 = np.sqrt(u_v22**2 + v_v22**2)

        # v25 phase 1.5 (with extreme head)
        t_v25 = _av(ds_v25p15, "2m_temperature_v18", k_idx)
        u_v25 = _av(ds_v25p15, "10m_u_component_of_wind_v18", k_idx)
        v_v25 = _av(ds_v25p15, "10m_v_component_of_wind_v18", k_idx)
        w_v25 = np.sqrt(u_v25**2 + v_v25**2)

        anc_times = ds_v22.anchor_time.values
        valid_times = anc_times + np.timedelta64(lead_h, "h")
        doys = pd.DatetimeIndex(valid_times).dayofyear.values
        rmx  = recs["t2m_record_max"].sel(doy=xr.DataArray(doys, dims="anchor_time")).transpose("anchor_time", "lat", "lon").values
        rmn  = recs["t2m_record_min"].sel(doy=xr.DataArray(doys, dims="anchor_time")).transpose("anchor_time", "lat", "lon").values
        rwmx = recs["wind10m_record_max"].sel(doy=xr.DataArray(doys, dims="anchor_time")).transpose("anchor_time", "lat", "lon").values

        exc_h = (t_tru - rmx).ravel()
        err_b_t   = (t_b   - t_tru).ravel()
        err_v22_t = (t_v22 - t_tru).ravel()
        err_v25_t = (t_v25 - t_tru).ravel()
        exc_c = (rmn - t_tru).ravel()
        exc_w = (w_tru - rwmx).ravel()
        err_b_w   = (w_b   - w_tru).ravel()
        err_v22_w = (w_v22 - w_tru).ravel()
        err_v25_w = (w_v25 - w_tru).ravel()
        mh = exc_h > 0; mc = exc_c > 0; mw = exc_w > 0
        data_panels = [
            ("heat", exc_h[mh], err_b_t[mh],   err_v22_t[mh], err_v25_t[mh], T_EDGES),
            ("cold", exc_c[mc], err_b_t[mc],   err_v22_t[mc], err_v25_t[mc], T_EDGES),
            ("wind", exc_w[mw], err_b_w[mw],   err_v22_w[mw], err_v25_w[mw], W_EDGES),
        ]
        for col_offset in [0, 3]:
            metric = "rmse" if col_offset == 0 else "bias"
            for j, (name, exc, eb, e22, e25, edges) in enumerate(data_panels):
                ax = axes[row, col_offset + j]
                _, rb, bib, ctr = bin_stats(exc, eb,  edges)
                _, r22, b22, _  = bin_stats(exc, e22, edges)
                _, r25, b25, _  = bin_stats(exc, e25, edges)
                yb  = rb  if metric == "rmse" else bib
                y22 = r22 if metric == "rmse" else b22
                y25 = r25 if metric == "rmse" else b25
                ax.plot(ctr, yb,  "o-", color="0.4",  lw=1.8, ms=5, label="GC base")
                ax.plot(ctr, y22, "s-", color="C0",   lw=1.6, ms=4, label="v22 K=18 (no ext head)")
                ax.plot(ctr, y25, "D-", color="C2",   lw=1.8, ms=4, label="v25 phase1.5 (gated ext)")
                if metric == "bias":
                    ax.axhline(0, color="k", lw=0.5)
                if row == 0:
                    ax.set_title(panels_template[col_offset + j], fontsize=9)
                if col_offset + j == 0:
                    ax.set_ylabel(f"lead K={K_lead} ({lead_h}h)", fontsize=10)
                ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
                if row == 0 and col_offset + j == 0:
                    ax.legend(fontsize=7)
    fig.suptitle(f"Extreme-records 2022 vs 2015-2021 record (3-way comparison){title_suffix}",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out_path}")


def main():
    p_v22 = DATA_DIR / "v22_K18_step20000_res2_extreme_records_K40_2022.nc"
    p_v25 = DATA_DIR / "v25_phase1p5_step5000_res2_extreme_records_K40_2022.nc"
    p_v25_phase1 = DATA_DIR / "v25_step3000_res2_extreme_records_K40_2022.nc"

    recs = xr.open_dataset(RECS_PATH)

    if not p_v22.exists():
        print(f"[skip] missing {p_v22}"); return
    ds_v22 = xr.open_dataset(p_v22)

    # Phase 1.5 (target) — also fall back to phase 1 if phase 1.5 nc not yet ready
    if p_v25.exists():
        ds_v25p15 = xr.open_dataset(p_v25)
        plot_3way(ds_v22, ds_v25p15, recs,
                  OUT_DIR / "extreme_records_3way_v25_phase1p5.png",
                  title_suffix=" — GC base vs v22 K=18 vs v25 phase 1.5")
        ds_v25p15.close()
    else:
        print(f"[skip] {p_v25} not ready yet — phase 1.5 save job still running")

    # Also do phase 1 (the previous null result) for reference
    if p_v25_phase1.exists():
        ds_v25p1 = xr.open_dataset(p_v25_phase1)
        plot_3way(ds_v22, ds_v25p1, recs,
                  OUT_DIR / "extreme_records_3way_v25_phase1.png",
                  title_suffix=" — GC base vs v22 K=18 vs v25 phase 1 (null)")
        ds_v25p1.close()

    ds_v22.close()


if __name__ == "__main__":
    main()
