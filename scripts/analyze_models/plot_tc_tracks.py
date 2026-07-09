"""Plot TC tracks: IBTrACS best-track + GC baseline + v22cl SWA forecast tracks,
plus truth-tracker calibration reference. Uses the warmcore_relaxed config CSVs
from the tracker sweep.

Two-storm smoke plot: Doksuri (WP major) and Saola (WP major).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0088
SMOKE_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-04-tcbench-smoke")
OUT_DIR = SMOKE_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IBTRACS = "/home/lm8598/Weather_Global_experiments/external/TCBench/data/ibtracs/ibtracs.ALL.list.v04r01.csv"


def gcd_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = phi2 - phi1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))


def load_predicted_track(csv_path, init_time, ibtracs_init_lat, ibtracs_init_lon):
    """Load a StitchNodes CSV; return the ONE track closest to IBTrACS init position."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    df["Valid Time"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]))
    df["lon_norm"] = np.where(df["lon"] > 180, df["lon"] - 360, df["lon"])
    # Pick the track_id whose earliest point is closest to IBTrACS init
    best_id, best_d = None, float("inf")
    for tid in df["track_id"].unique():
        sub = df[df["track_id"] == tid].sort_values("Valid Time")
        d = gcd_km(sub.iloc[0]["lat"], sub.iloc[0]["lon_norm"],
                   ibtracs_init_lat, ibtracs_init_lon)
        if d < best_d:
            best_d, best_id = d, tid
    if best_d > 500:
        return pd.DataFrame()
    matched = df[df["track_id"] == best_id].sort_values("Valid Time").reset_index(drop=True)
    matched["lead_h"] = ((matched["Valid Time"] - init_time).dt.total_seconds() / 3600).astype(int)
    return matched


def plot_storm(ax, sid, init_str, storm_name, config="warmcore_relaxed"):
    init_time = pd.Timestamp(init_str)
    # IBTrACS best-track
    ib = pd.read_csv(IBTRACS, skiprows=[1], low_memory=False)
    ib["ISO_TIME"] = pd.to_datetime(ib["ISO_TIME"], errors="coerce")
    ib["LAT"] = pd.to_numeric(ib["LAT"], errors="coerce")
    ib["LON"] = pd.to_numeric(ib["LON"], errors="coerce")
    ib["USA_WIND"] = pd.to_numeric(ib["USA_WIND"], errors="coerce")
    obs = (ib[(ib["SID"] == sid) &
              (ib["ISO_TIME"] >= init_time) &
              (ib["ISO_TIME"] <= init_time + pd.Timedelta(hours=120))]
           [["ISO_TIME", "LAT", "LON", "USA_WIND"]].dropna(subset=["LAT", "LON"]))
    obs["lead_h"] = ((obs["ISO_TIME"] - init_time).dt.total_seconds() / 3600).astype(int)
    if len(obs) == 0:
        ax.set_title(f"{storm_name}\n(no IBTrACS data)")
        return
    ib_init_lat = float(obs.iloc[0]["LAT"])
    ib_init_lon = float(obs.iloc[0]["LON"])

    # Predicted tracks
    tag_files = {
        "GC baseline": (SMOKE_DIR / "sweep_baseline"
                        / f"init_{init_time.strftime('%Y%m%dT%H')}__baseline.preproc__{config}.tracks.csv"),
        "v22cl SWA (cold_full)": (SMOKE_DIR / "sweep_v18"
                                  / f"init_{init_time.strftime('%Y%m%dT%H')}__v18.preproc__{config}.tracks.csv"),
        "truth-tracker (ERA5)": (SMOKE_DIR / "sweep_truth"
                                 / f"init_{init_time.strftime('%Y%m%dT%H')}__truth.preproc__{config}.tracks.csv"),
    }
    tag_style = {
        "GC baseline": dict(color="C0", marker="o", ls="-", ms=6, lw=1.7),
        "v22cl SWA (cold_full)": dict(color="C3", marker="s", ls="--", ms=6, lw=1.7),
        "truth-tracker (ERA5)": dict(color="C2", marker="^", ls=":", ms=5, lw=1.4, alpha=0.75),
    }

    # Plot IBTrACS best-track
    ax.plot(obs["LON"], obs["LAT"], "k-", lw=2.5, label="IBTrACS best-track", zorder=5)
    ax.plot(obs.iloc[0]["LON"], obs.iloc[0]["LAT"], "k*", ms=14, zorder=6, label="_init_")
    # Every-12h markers for IBTrACS
    obs12 = obs[obs["ISO_TIME"].dt.hour.isin([0, 12])]
    ax.scatter(obs12["LON"], obs12["LAT"], c="k", s=25, zorder=6)

    # Predicted tracks
    for label, path in tag_files.items():
        if not path.exists():
            continue
        pred = load_predicted_track(path, init_time, ib_init_lat, ib_init_lon)
        if pred.empty:
            continue
        ax.plot(pred["lon_norm"], pred["lat"], label=f"{label}  ({len(pred)} pts)",
                zorder=4, **tag_style.get(label, {}))
        # Label each point with its lead time (h)
        for _, row in pred.iterrows():
            ax.annotate(f"{int(row['lead_h'])}", (row["lon_norm"], row["lat"]),
                        fontsize=7, color=tag_style[label]["color"], alpha=0.8,
                        xytext=(5, 3), textcoords="offset points")

    # Axes
    all_lat = list(obs["LAT"])
    all_lon = list(obs["LON"])
    for label, path in tag_files.items():
        if path.exists():
            pred = load_predicted_track(path, init_time, ib_init_lat, ib_init_lon)
            if not pred.empty:
                all_lat.extend(pred["lat"].tolist())
                all_lon.extend(pred["lon_norm"].tolist())
    lat_min, lat_max = min(all_lat) - 3, max(all_lat) + 3
    lon_min, lon_max = min(all_lon) - 3, max(all_lon) + 3
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_title(f"{storm_name}  (SID {sid})\n"
                 f"init {init_time.strftime('%Y-%m-%d %HZ')}   tracker={config}",
                 fontsize=10)
    ax.legend(loc="best", fontsize=8)


def main():
    storms = [
        ("2023201N13134", "2023-07-24 12:00", "Doksuri (WP major typhoon)"),
        ("2023234N18128", "2023-08-30 00:00", "Saola (WP major typhoon)"),
    ]
    for config in ["warmcore_relaxed", "permissive"]:
        fig, axes = plt.subplots(1, len(storms), figsize=(9 * len(storms), 7))
        if len(storms) == 1: axes = [axes]
        for ax, (sid, init, name) in zip(axes, storms):
            plot_storm(ax, sid, init, name, config=config)
        fig.suptitle(f"TC track forecast vs IBTrACS — 5-day rollout, 1° GraphCast, "
                     f"tracker config = '{config}'", fontsize=12)
        plt.tight_layout()
        out = OUT_DIR / f"tc_tracks_{config}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
