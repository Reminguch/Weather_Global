"""Per-city 4-line comparison plot: ERA5 truth / GC baseline / v22 K=22 open (paper) /
v22cl K=22 SWA (our best closed-loop).

Same structure as 2026-06-05-city-traces/plots/city_*_aligned.png but with our SWA ckpt.

Two variants:
  A. Best-model comparison: truth + baseline + v22 K=22 open + v22cl K=22 SWA
  B. K-scan variant: truth + baseline + v22cl K=8/16/22 SWA (see K effect)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

V22_PAPER_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-06-05-city-traces")
V22CL_SWA_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-03-city-traces-fixed-v22cleantree")
OUT_DIR       = V22CL_SWA_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_TAGS = ["20220115", "20220415", "20220715", "20221015"]
SEASON_LABEL = {"20220115": "Jan 15", "20220415": "Apr 15",
                "20220715": "Jul 15", "20221015": "Oct 15"}
CITIES = ["NYC", "LA", "Chicago", "Tokyo", "Beijing", "Shanghai",
          "London", "Paris", "Mumbai", "Sydney"]
VAR_INFO = {
    "2m_temperature":         ("2m T", "K"),
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
                t, b, f = t*1000, b*1000, f*1000  # m → mm
            out[city][var] = (lead_times, t, b, f)
    return out


def plot_city_variant_A(city):
    """K=22 comparison: truth + baseline + v22 open K=22 (cold_bp) + v22cl K=22 SWA (cold_full)."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex="row")
    for row, tag in enumerate(ANCHOR_TAGS):
        v22 = load_json(V22_PAPER_DIR / f"v22_K22_cities_{tag}.json")
        cl22_swa = load_json(V22CL_SWA_DIR / f"v22cl_K22_SWA_cf_cities_{tag}.json")

        for col, var in enumerate(["2m_temperature", "total_precipitation_6hr"]):
            ax = axes[row, col]
            label, unit = VAR_INFO[var]

            if v22 is not None:
                t, tru, base, full = v22[city][var]
                ax.plot(t, tru,  "k-",  lw=2.4, label="ERA5 truth")
                ax.plot(t, base, "C0--",lw=1.4, alpha=0.7, label="baseline (GraphCast)")
                ax.plot(t, full, "0.55",lw=1.4, label="v22 K=22 open (cold_bp, paper)")
            if cl22_swa is not None:
                t, _, _, full_swa = cl22_swa[city][var]
                ax.plot(t, full_swa, "C3-", lw=2.0, label="v22cl K=22 SWA (cold_full)")

            ax.set_ylabel(f"{label} ({unit})")
            ax.set_title(f"{city} — {SEASON_LABEL[tag]} 2022")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{city}: 10-day forecast — v22 K=22 open (gray) vs v22cl K=22 SWA cold_full (red) + ERA5 + baseline",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"city_{city}_K22_comparison.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def plot_city_variant_B(city):
    """K-scan: truth + baseline + v22cl SWA K=14/18/20/22 cold_full."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex="row")
    KS = [14, 18, 20, 22]
    colors = {14: "#ff9999", 18: "#ff5555", 20: "#dd2222", 22: "#990000"}

    for row, tag in enumerate(ANCHOR_TAGS):
        per_k = {}
        for K in KS:
            per_k[K] = load_json(V22CL_SWA_DIR / f"v22cl_K{K}_SWA_cf_cities_{tag}.json")

        for col, var in enumerate(["2m_temperature", "total_precipitation_6hr"]):
            ax = axes[row, col]
            label, unit = VAR_INFO[var]

            ref = per_k[22]
            if ref is not None:
                t, tru, base, _ = ref[city][var]
                ax.plot(t, tru,  "k-",   lw=2.2, label="ERA5 truth")
                ax.plot(t, base, "C0--", lw=1.3, alpha=0.7, label="GC baseline")

            for K in KS:
                dd = per_k[K]
                if dd is None: continue
                t, _, _, full = dd[city][var]
                ax.plot(t, full, "-", lw=1.6, color=colors[K],
                        label=f"v22cl K={K} SWA")

            ax.set_ylabel(f"{label} ({unit})")
            ax.set_title(f"{city} — {SEASON_LABEL[tag]} 2022")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{city}: 10-day forecast (cold_full) — v22cl SWA K-scan K=14/18/20/22", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"city_{city}_v22cl_SWA_Kscan.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def main():
    for city in CITIES:
        plot_city_variant_A(city)
        plot_city_variant_B(city)
    print(f"\nDone. {len(CITIES)*2} plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
