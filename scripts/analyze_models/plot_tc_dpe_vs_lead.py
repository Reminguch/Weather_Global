"""DPE (Direct Positional Error) vs lead time — standard TC-track verification plot.

Compare IBTrACS best-track vs GC baseline / v22cl SWA / truth-tracker
for smoke storms under the locked warmcore_relaxed tracker config.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0088
SMOKE_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-04-tcbench-smoke")
OUT_DIR = SMOKE_DIR / "plots"
IBTRACS = "/home/lm8598/Weather_Global_experiments/external/TCBench/data/ibtracs/ibtracs.ALL.list.v04r01.csv"


def gcd_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = phi2 - phi1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def per_lead_dpe(csv_path, init_time, ibtracs_df, sid, leads_h):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {L: np.nan for L in leads_h}
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        return {L: np.nan for L in leads_h}
    if df.empty:
        return {L: np.nan for L in leads_h}
    df["Valid Time"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]))
    df["lon_norm"] = np.where(df["lon"] > 180, df["lon"] - 360, df["lon"])
    ib_this = ibtracs_df[ibtracs_df["SID"] == sid]
    ib_init = ib_this[ib_this["ISO_TIME"] == init_time]
    if len(ib_init) == 0:
        cand = ib_this[np.abs((ib_this["ISO_TIME"] - init_time).dt.total_seconds()) <= 3*3600]
        if len(cand) == 0:
            return {L: np.nan for L in leads_h}
        ib_init = cand.iloc[[0]]
    ib_lat, ib_lon = float(ib_init["LAT"].iloc[0]), float(ib_init["LON"].iloc[0])
    best_id, best_d = None, float("inf")
    for tid in df["track_id"].unique():
        first = df[df["track_id"] == tid].sort_values("Valid Time").iloc[0]
        d = gcd_km(first["lat"], first["lon_norm"], ib_lat, ib_lon)
        if d < best_d:
            best_d, best_id = d, tid
    if best_d > 500:
        return {L: np.nan for L in leads_h}
    matched = df[df["track_id"] == best_id]
    out = {}
    for L in leads_h:
        vt = init_time + pd.Timedelta(hours=int(L))
        pred = matched[matched["Valid Time"] == vt]
        obs = ib_this[ib_this["ISO_TIME"] == vt]
        if len(pred) == 0 or len(obs) == 0:
            out[L] = np.nan; continue
        out[L] = gcd_km(pred["lat"].iloc[0], pred["lon_norm"].iloc[0],
                        obs["LAT"].iloc[0], obs["LON"].iloc[0])
    return out


def main():
    ib = pd.read_csv(IBTRACS, skiprows=[1], low_memory=False)
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    ib["LAT"] = pd.to_numeric(ib["LAT"], errors="coerce")
    ib["LON"] = pd.to_numeric(ib["LON"], errors="coerce")
    ib = ib.dropna(subset=["ISO_TIME", "LAT", "LON"])

    storms = [
        ("2023201N13134", "2023-07-24 12:00", "Doksuri (WP major)"),
        ("2023234N18128", "2023-08-30 00:00", "Saola (WP major)"),
    ]
    LEADS = list(range(6, 121, 6))
    CONFIG = "warmcore_relaxed"
    tags = {
        "GC baseline":            ("sweep_baseline", "baseline", "C0", "o", "-"),
        "v22cl SWA (cold_full)":  ("sweep_v18",       "v18",      "C3", "s", "--"),
        "truth-tracker (ERA5)":   ("sweep_truth",     "truth",    "C2", "^", ":"),
    }

    fig, axes = plt.subplots(1, len(storms), figsize=(7*len(storms), 5), sharey=True)
    if len(storms) == 1: axes = [axes]
    for ax, (sid, init_str, name) in zip(axes, storms):
        init_time = pd.Timestamp(init_str)
        for label, (subdir, tag, color, marker, ls) in tags.items():
            csv = SMOKE_DIR / subdir / f"init_{init_time.strftime('%Y%m%dT%H')}__{tag}.preproc__{CONFIG}.tracks.csv"
            dpe = per_lead_dpe(csv, init_time, ib, sid, LEADS)
            xs, ys = [], []
            for L in LEADS:
                if not np.isnan(dpe[L]):
                    xs.append(L); ys.append(dpe[L])
            if xs:
                ax.plot(xs, ys, marker=marker, ls=ls, color=color, lw=1.7, ms=6,
                        label=f"{label}  ({len(xs)} pts)")
        ax.set_xlabel("Forecast lead time (h)")
        if ax is axes[0]:
            ax.set_ylabel("DPE (km)")
        ax.set_title(f"{name}\ninit {init_time.strftime('%Y-%m-%d %HZ')}", fontsize=10)
        ax.set_xticks([6, 24, 48, 72, 96, 120])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"TC track error (DPE) vs forecast lead — tracker='{CONFIG}', 1° GraphCast",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"tc_dpe_vs_lead_{CONFIG}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
