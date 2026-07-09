#!/usr/bin/env python
"""Plot city traces: per-city, 4 anchors (seasons) × 2 vars (temp + precip) grid.
4 lines: ERA5 truth, baseline GraphCast, Mamba v22 (open-loop), Mamba v22closed.

X-axis: calendar datetime (from JSON lead_times). v22 and v22closed have
different metric windows (v22closed's W=24 warmup shifts forecast +6 days);
both plotted at their actual forecast times.
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

DATA_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-05-city-traces")
OUT_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-06-05-city-traces/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_TAGS = ["20220115", "20220415", "20220715", "20221015"]
SEASON_LABEL = {
    "20220115": "Jan 15", "20220415": "Apr 15",
    "20220715": "Jul 15", "20221015": "Oct 15",
}

CITIES = ["NYC", "LA", "Chicago", "Tokyo", "Beijing", "Shanghai",
          "London", "Paris", "Mumbai", "Sydney"]

VAR_INFO = {
    "2m_temperature":         ("2m T", "K"),
    "total_precipitation_6hr": ("Precip 6h", "mm"),  # convert m → mm
}

def load_traces(tag, ckpt_tag):
    """Load a JSON file → dict[city][var] = (datetimes, truth, baseline, full).

    lead_times in JSON are stored as 'N nanoseconds' (timedelta from anchor),
    so we add to anchor_actual_time to get absolute datetimes."""
    path = DATA_DIR / f"{ckpt_tag}_cities_{tag}.json"
    d = json.load(open(path))
    anchor = pd.Timestamp(d["anchor_actual_time"])
    def _to_abs(s):
        # "21600000000000 nanoseconds" → pd.Timedelta(nanoseconds=int)
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
                # convert m → mm for readability
                t = t * 1000.0; b = b * 1000.0; f = f * 1000.0
            out[city][var] = (lead_times, t, b, f)
    return out


def plot_city(city: str):
    """One figure per city: 4 rows (seasons) × 2 cols (temp + precip).
    All 4 lines share SAME calendar window (v22closed uses aligned anchor)."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex="row")

    for row, tag in enumerate(ANCHOR_TAGS):
        v22 = load_traces(tag, "v22_K22")
        # ALIGNED v22closed: metric phase matches v22 forecast window
        v22cl = load_traces(tag, "v22closed_K8_aligned")

        for col, var in enumerate(["2m_temperature", "total_precipitation_6hr"]):
            ax = axes[row, col]
            label, unit = VAR_INFO[var]

            t_v22, truth_v22, base_v22, full_v22 = v22[city][var]
            t_v22cl, truth_v22cl, base_v22cl, full_v22cl = v22cl[city][var]

            # Single truth (both runs see same truth in aligned setup)
            ax.plot(t_v22, truth_v22, "k-", lw=2.2, label="ERA5 truth")
            # Single baseline reference (from v22 — could also use v22closed, very similar)
            ax.plot(t_v22, base_v22, "C0--", lw=1.4, alpha=0.7,
                    label="baseline (GraphCast)")

            # 2 Mamba predictions — the comparison
            ax.plot(t_v22, full_v22, "C3-", lw=1.6,
                    label="Mamba v22 (open-loop train)")
            ax.plot(t_v22cl, full_v22cl, "C2-", lw=1.6,
                    label="Mamba v22closed K=8 (closed train)")

            ax.set_ylabel(f"{label} ({unit})")
            ax.set_title(f"{city} — {SEASON_LABEL[tag]} 2022")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{city}: 10-day forecast — v22 open-loop (red) vs v22closed K=8 closed (green) vs ERA5 (black) and baseline (blue dashed)",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"city_{city}_aligned.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    for city in CITIES:
        plot_city(city)
    print(f"\nDone. {len(CITIES)} city plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
