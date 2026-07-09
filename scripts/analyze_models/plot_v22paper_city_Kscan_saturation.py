"""Two figures from the v22 paper cold_bp city traces (44 JSONs, K=2..22 × 4 anchors):

  1. Per-anchor K-scan overview
     One PNG per anchor. 10 cities × 2 variables. In each subplot: truth (black),
     GC baseline (gray), and v22 paper at K=2/6/10/14/18/22 (color gradient).
     → visualize how much longer K-curriculum extends useful skill.

  2. Saturation summary
     One PNG. For each anchor × variable (4 × 2 = 8 subplots), plot MAE at
     lead 240h vs K, one line per city.
     → identify where K adds skill vs plateaus per (city, anchor, variable).
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
import numpy as np
import pandas as pd

SRC = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-03-v22paper-city-fixed-v22cleantree")
OUT = SRC / "plots"
OUT.mkdir(parents=True, exist_ok=True)

KS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
ANCHOR_TAGS = ["20220115", "20220415", "20220715", "20221015"]
SEASON_LABEL = {"20220115": "Jan 15", "20220415": "Apr 15",
                "20220715": "Jul 15", "20221015": "Oct 15"}
CITIES = ["NYC", "LA", "Chicago", "Tokyo", "Beijing", "Shanghai",
          "London", "Paris", "Mumbai", "Sydney"]
VAR_INFO = {
    "2m_temperature":         ("2m T",     "K",  1.0),
    "total_precipitation_6hr": ("Precip 6h", "mm", 1000.0),  # m → mm
}


def load(K, tag):
    p = SRC / f"v22_K{K}_cities_{tag}.json"
    if not p.exists(): return None
    d = json.load(open(p))
    anchor = pd.Timestamp(d["anchor_actual_time"])
    def _to_abs(s):
        ns = int(s.split()[0])
        return anchor + pd.Timedelta(nanoseconds=ns)
    lead_times = [_to_abs(t) for t in d["lead_times"]]
    return d, lead_times


def figure_1_per_anchor_overview():
    K_show = [2, 6, 10, 14, 18, 22]
    cmap = cm.get_cmap("viridis")
    norm = plt.Normalize(min(K_show), max(K_show))
    for tag in ANCHOR_TAGS:
        # Load reference (any K, take K=22 for truth & baseline reference)
        ref = load(22, tag)
        if ref is None:
            print(f"skip {tag}"); continue
        d_ref, lead_times = ref
        fig, axes = plt.subplots(len(CITIES), 2, figsize=(14, 3 * len(CITIES)),
                                 sharex="col")
        for row, city in enumerate(CITIES):
            for col, (var, (label, unit, scale)) in enumerate(VAR_INFO.items()):
                ax = axes[row, col]
                # Truth and baseline (from K=22 JSON — same across K in cold_bp)
                t_arr = np.asarray(d_ref["cities"][city]["vars"][var]["truth"]) * scale
                b_arr = np.asarray(d_ref["cities"][city]["vars"][var]["baseline"]) * scale
                ax.plot(lead_times, t_arr, "k-", lw=2.4, label="ERA5 truth")
                ax.plot(lead_times, b_arr, "--", color="0.55", lw=1.5,
                        label="GC baseline")
                for K in K_show:
                    dd = load(K, tag)
                    if dd is None: continue
                    d_k, _ = dd
                    f_arr = np.asarray(d_k["cities"][city]["vars"][var]["full"]) * scale
                    ax.plot(lead_times, f_arr, "-", lw=1.5,
                            color=cmap(norm(K)), label=f"K={K}")
                ax.set_ylabel(f"{city}\n{label} ({unit})", fontsize=8)
                if row == 0:
                    ax.set_title(f"{SEASON_LABEL[tag]} 2022 — {label}", fontsize=10)
                ax.grid(alpha=0.3)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                if row == 0 and col == 0:
                    ax.legend(loc="best", fontsize=7, ncol=2)
        fig.suptitle(f"v22 paper cold_bp K-scan — {SEASON_LABEL[tag]} 2022",
                     fontsize=13)
        plt.tight_layout()
        out = OUT / f"Kscan_overview_{tag}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"saved {out}")


def figure_2_saturation():
    """Per (anchor × variable), MAE at lead 240h vs K, one line per city."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    cmap = cm.get_cmap("tab10")
    city_colors = {c: cmap(i) for i, c in enumerate(CITIES)}
    for col, tag in enumerate(ANCHOR_TAGS):
        for row, (var, (label, unit, scale)) in enumerate(VAR_INFO.items()):
            ax = axes[row, col]
            for city in CITIES:
                ys = []
                for K in KS:
                    dd = load(K, tag)
                    if dd is None: ys.append(np.nan); continue
                    d, _ = dd
                    vd = d["cities"][city]["vars"][var]
                    f = np.asarray(vd["full"])[39] * scale   # lead 240h
                    t = np.asarray(vd["truth"])[39] * scale
                    ys.append(abs(f - t))
                ax.plot(KS, ys, "-o", ms=4, lw=1.4, color=city_colors[city],
                        label=city)
                # baseline reference
            # Also plot mean baseline error across cities as a horizontal line
            base_errs = []
            for city in CITIES:
                dd = load(22, tag)
                if dd is None: continue
                d, _ = dd
                vd = d["cities"][city]["vars"][var]
                b = np.asarray(vd["baseline"])[39] * scale
                t = np.asarray(vd["truth"])[39] * scale
                base_errs.append(abs(b - t))
            ax.axhline(np.mean(base_errs), color="k", lw=1.0, ls="--",
                       label=f"GC baseline mean ({np.mean(base_errs):.2f})")
            ax.set_title(f"{SEASON_LABEL[tag]} 2022 — {label} @ 240h",
                         fontsize=10)
            ax.set_xlabel("K")
            ax.set_ylabel(f"|full - truth| ({unit})")
            ax.set_xticks(KS)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("v22 paper cold_bp — saturation: |Δ| at lead 240h vs K, "
                 "per city (one line each)", fontsize=13)
    plt.tight_layout()
    out = OUT / "Kscan_saturation_lead240.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def figure_3_improvement_vs_baseline():
    """K improvement over baseline at select leads (48h, 120h, 240h), aggregated
       over 10 cities. One panel per (anchor × variable)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    lead_show = [7, 19, 39]  # 48h, 120h, 240h  (0-indexed)
    lead_lbl = ["48h", "120h", "240h"]
    lead_colors = ["C0", "C1", "C3"]
    for col, tag in enumerate(ANCHOR_TAGS):
        for row, (var, (label, unit, scale)) in enumerate(VAR_INFO.items()):
            ax = axes[row, col]
            for li, k_lead in enumerate(lead_show):
                mean_imp = []
                for K in KS:
                    dd = load(K, tag)
                    if dd is None: mean_imp.append(np.nan); continue
                    d, _ = dd
                    # Improvement% (RMSE reduction) across 10 cities at this lead
                    imps = []
                    for city in CITIES:
                        vd = d["cities"][city]["vars"][var]
                        f = abs(np.asarray(vd["full"])[k_lead] - np.asarray(vd["truth"])[k_lead])
                        b = abs(np.asarray(vd["baseline"])[k_lead] - np.asarray(vd["truth"])[k_lead])
                        if b > 1e-6:
                            imps.append(1 - f / b)
                    mean_imp.append(100 * np.mean(imps) if imps else np.nan)
                ax.plot(KS, mean_imp, "-o", ms=5, lw=1.6, color=lead_colors[li],
                        label=f"lead {lead_lbl[li]}")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{SEASON_LABEL[tag]} 2022 — {label}", fontsize=10)
            ax.set_xlabel("K")
            ax.set_ylabel("mean |Δ| improvement% over baseline\n(10 cities avg)")
            ax.set_xticks(KS)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle("v22 paper cold_bp — K-scan mean improvement% vs GC baseline "
                 "(avg over 10 cities, per anchor × variable)", fontsize=13)
    plt.tight_layout()
    out = OUT / "Kscan_improvement_over_baseline.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    figure_1_per_anchor_overview()
    figure_2_saturation()
    figure_3_improvement_vs_baseline()
    print("\nDone.")
