"""Per-city 4-line comparison plot for arbitrary K (v22 paper open vs v22cl SWA closed).

For each of 10 cities, produce a 4-row × 2-col figure:
  rows = anchors (Jan/Apr/Jul/Oct 15 2022)
  cols = 2m_temperature, total_precipitation_6hr
  lines: ERA5 truth (black), GC baseline (blue dashed), v22 paper K=K open cold_bp (gray),
         v22cl K=K SWA cold_full (red)

Both v22 paper and v22cl SWA use v22clean-tree JSONs so all curves are trustworthy.

Usage: python plot_v22cl_SWA_city_K_comparison.py K
       where K in {14, 18, 20, 22}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

V22_PAPER_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-03-v22paper-city-fixed-v22cleantree")
V22CL_SWA_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-03-city-traces-fixed-v22cleantree")
OUT_DIR       = V22CL_SWA_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_TAGS = ["20220115", "20220415", "20220715", "20221015"]
SEASON_LABEL = {"20220115": "Jan 15", "20220415": "Apr 15",
                "20220715": "Jul 15", "20221015": "Oct 15"}
CITIES = ["NYC", "LA", "Chicago", "Tokyo", "Beijing", "Shanghai",
          "London", "Paris", "Mumbai", "Sydney"]
VAR_INFO = {
    "2m_temperature":         ("2m T",     "K"),
    "total_precipitation_6hr": ("Precip 6h", "mm"),
}


def load_json(path):
    if not path.exists(): return None
    d = json.load(open(path))
    anchor = pd.Timestamp(d["anchor_actual_time"])
    def _to_abs(s):
        ns = int(s.split()[0])
        return anchor + pd.Timedelta(nanoseconds=ns)
    lead_times = [_to_abs(t) for t in d["lead_times"]]
    out = {}
    for city, info in d["cities"].items():
        out[city] = {}
        for var, vd in info["vars"].items():
            t = np.asarray(vd["truth"])
            b = np.asarray(vd["baseline"])
            f = np.asarray(vd["full"])
            if var == "total_precipitation_6hr":
                t, b, f = t*1000, b*1000, f*1000
            out[city][var] = (lead_times, t, b, f)
    return out


def plot_city_variant_A(city, K):
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex="row")
    for row, tag in enumerate(ANCHOR_TAGS):
        v22 = load_json(V22_PAPER_DIR / f"v22_K{K}_cities_{tag}.json")
        cl  = load_json(V22CL_SWA_DIR / f"v22cl_K{K}_SWA_cf_cities_{tag}.json")
        for col, var in enumerate(["2m_temperature", "total_precipitation_6hr"]):
            ax = axes[row, col]
            label, unit = VAR_INFO[var]
            if v22 is not None:
                t, tru, base, full = v22[city][var]
                ax.plot(t, tru,  "k-",  lw=2.4, label="ERA5 truth")
                ax.plot(t, base, "C0--",lw=1.4, alpha=0.7, label="GC baseline")
                ax.plot(t, full, "0.55",lw=1.4, label=f"v22 K={K} open (cold_bp)")
            if cl is not None:
                t, _, _, full_swa = cl[city][var]
                ax.plot(t, full_swa, "C3-", lw=2.0, label=f"v22cl K={K} SWA (cold_full)")
            ax.set_ylabel(f"{label} ({unit})")
            ax.set_title(f"{city} — {SEASON_LABEL[tag]} 2022")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"{city}: 10-day forecast — v22 K={K} open (gray) vs "
                 f"v22cl K={K} SWA cold_full (red) + ERA5 + baseline",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"city_{city}_K{K}_comparison.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_v22cl_SWA_city_K_comparison.py K")
        sys.exit(1)
    K = int(sys.argv[1])
    for city in CITIES:
        plot_city_variant_A(city, K)
    print(f"\nDone. 10 plots for K={K} in {OUT_DIR}")


if __name__ == "__main__":
    main()
