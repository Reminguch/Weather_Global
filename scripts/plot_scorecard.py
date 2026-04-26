#!/usr/bin/env python3
"""Plot per-target scorecard heatmaps and win-rate diagrams from scorecard
JSONs produced by eval_per_level.py with --per-lead-time output.

Three figures per (K, mode) combination:
  1. Heatmap: y = (variable, level), x = lead_time, color = ΔRMSE_latw
  2. Lead-time win-rate bar: % of 83 channels improved at each lead time
  3. Per-variable line plot: ΔRMSE_latw vs lead time for headline variables
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

SCORECARD_DIR = Path(
    "/home/lm8598/Weather_Global_experiments/results/2026-04-26_scorecard"
)
OUT_DIR = SCORECARD_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper variable ordering: 5 surface first, then 6 atmospheric
SURFACE_ORDER = [
    "2m_temperature", "mean_sea_level_pressure",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "total_precipitation_6hr",
]
ATM_ORDER = [
    "geopotential", "temperature", "specific_humidity",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
]
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Short names for figures
SHORT = {
    "2m_temperature": "T2m",
    "mean_sea_level_pressure": "MSLP",
    "10m_u_component_of_wind": "U10m",
    "10m_v_component_of_wind": "V10m",
    "total_precipitation_6hr": "precip",
    "geopotential": "Z",
    "temperature": "T",
    "specific_humidity": "q",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "vertical_velocity": "W",
}


def load_scorecard(path):
    d = json.load(open(path))
    targets = d.get("per_channel_per_leadtime", [])
    return d, targets


def channel_label(t):
    v = SHORT.get(t["variable"], t["variable"])
    if t["level"] is not None:
        return f"{v}{t['level']}"
    return v


def channel_sort_key(t):
    """Order: 5 surface, then atmospheric grouped (Z all levels, T all levels, ...)."""
    v = t["variable"]
    if v in SURFACE_ORDER:
        return (0, SURFACE_ORDER.index(v), 0)
    return (1, ATM_ORDER.index(v), t["level"] if t["level"] else 0)


def heatmap(targets, title, savepath, metric="delta_RMSE_pct_latw"):
    by_chan = defaultdict(list)
    for t in targets:
        key = (t["variable"], t["level"])
        by_chan[key].append(t)

    # Build (n_chan, n_lead) matrix
    chan_keys = sorted(
        by_chan.keys(),
        key=lambda k: channel_sort_key({"variable": k[0], "level": k[1]}),
    )
    leads = sorted({t["lead_time_h"] for t in targets})
    n_chan = len(chan_keys)
    n_lead = len(leads)
    M = np.zeros((n_chan, n_lead))
    for i, key in enumerate(chan_keys):
        chs = sorted(by_chan[key], key=lambda t: t["lead_time_h"])
        for j, t in enumerate(chs):
            M[i, j] = t[metric]

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n_lead + 3), max(8, 0.13 * n_chan + 2)))
    vmax = max(3.0, np.percentile(np.abs(M), 95))
    # RdBu (NOT _r) -> negative=Red, positive=Blue. We want positive (Δ% > 0,
    # improvement) to show as BLUE, negative (degradation) as RED -- this
    # matches most weather-paper conventions and the verbal description.
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(M, aspect="auto", cmap="RdBu", norm=norm,
                   interpolation="nearest")

    # Annotate every cell with the value
    for i in range(n_chan):
        for j in range(n_lead):
            v = M[i, j]
            color = "white" if abs(v) > vmax * 0.5 else "black"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=6, color=color)

    # X axis: lead times in hours
    ax.set_xticks(range(n_lead))
    ax.set_xticklabels([f"+{l}h" for l in leads])
    # Y axis: channel labels
    labels = [
        channel_label({"variable": k[0], "level": k[1]}) for k in chan_keys
    ]
    ax.set_yticks(range(n_chan))
    ax.set_yticklabels(labels, fontsize=7)

    # Variable group separators
    last_var = None
    for i, key in enumerate(chan_keys):
        v = key[0]
        if last_var is not None and v != last_var:
            ax.axhline(i - 0.5, color="black", linewidth=0.8, alpha=0.6)
        last_var = v

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ RMSE (lat-weighted, %)\n+ = our model better", fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Lead time")
    ax.set_ylabel("Variable / Pressure level")
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return M, chan_keys, leads


def winrate_plot(scorecards_dict, savepath):
    """Per-lead-time win rate (% channels improved on RMSE)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"K=4 TF": "C0", "K=4 AR": "C1", "K=6 TF": "C2", "K=6 AR": "C3"}
    markers = {"K=4 TF": "o", "K=4 AR": "s", "K=6 TF": "^", "K=6 AR": "D"}
    for label, targets in scorecards_dict.items():
        by_lead = defaultdict(list)
        for t in targets:
            by_lead[t["lead_time_h"]].append(t)
        leads = sorted(by_lead.keys())
        wr = [sum(1 for x in by_lead[l] if x["delta_RMSE_pct_latw"] > 0) / len(by_lead[l]) * 100
              for l in leads]
        ax.plot(leads, wr, marker=markers[label], color=colors[label],
                label=label, linewidth=2, markersize=8)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="50% (random)")
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("Win rate (% of 83 channels improved on RMSE)")
    ax.set_title("Per-lead-time win rate (lat-weighted RMSE)\n"
                 "All Δ% relative to frozen GraphCast small baseline")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_var_trajectory(scorecards_dict, savepath):
    """Mean ΔRMSE_pct_latw vs lead time, per variable."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=False)
    axes = axes.flatten()
    headline = ["mean_sea_level_pressure", "geopotential", "temperature",
                "specific_humidity", "10m_u_component_of_wind", "total_precipitation_6hr",
                "2m_temperature", "vertical_velocity"]
    colors = {"K=4 TF": "C0", "K=4 AR": "C1", "K=6 TF": "C2", "K=6 AR": "C3"}
    markers = {"K=4 TF": "o", "K=4 AR": "s", "K=6 TF": "^", "K=6 AR": "D"}
    for ax, var in zip(axes, headline):
        for label, targets in scorecards_dict.items():
            by_lead = defaultdict(list)
            for t in targets:
                if t["variable"] == var:
                    by_lead[t["lead_time_h"]].append(t["delta_RMSE_pct_latw"])
            leads = sorted(by_lead.keys())
            means = [np.mean(by_lead[l]) for l in leads]
            ax.plot(leads, means, marker=markers[label], color=colors[label],
                    label=label, linewidth=1.5, markersize=6)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.set_title(SHORT.get(var, var), fontsize=11)
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("Δ RMSE (%, lat-weighted)")
        ax.grid(alpha=0.3)
        if var == headline[0]:
            ax.legend(loc="best", fontsize=8)
    plt.suptitle("Per-variable mean Δ RMSE vs lead time (lat-weighted)\n"
                 "+ = our model better; averaged over levels for atm vars",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Load all 4 JSONs
    files = {
        "K=4 TF": "scorecard_K4_step6000_h4_tf.json",
        "K=4 AR": "scorecard_K4_step6000_h4_ar.json",
        "K=6 TF": "scorecard_K6_step8000_h6_tf.json",
        "K=6 AR": "scorecard_K6_step8000_h6_ar.json",
    }
    scorecards = {}
    for label, fname in files.items():
        d, targets = load_scorecard(SCORECARD_DIR / fname)
        scorecards[label] = targets
        print(f"{label}: {len(targets)} targets, "
              f"win rate (RMSE) {d['n_targets_better_RMSE_latw']}/{d['n_targets_total']} "
              f"= {100*d['n_targets_better_RMSE_latw']/d['n_targets_total']:.1f}%")

    # 1. Heatmaps (one per K × mode = 4 figures)
    for label, targets in scorecards.items():
        savepath = OUT_DIR / f"heatmap_{label.replace(' ', '_').replace('=', '')}.png"
        title = f"Scorecard heatmap: {label}\nΔ RMSE (lat-weighted, %) per channel × lead time"
        heatmap(targets, title, savepath)
        print(f"  saved heatmap: {savepath}")

    # 2. Win rate plot (single figure with all 4)
    winrate_plot(scorecards, OUT_DIR / "winrate_per_lead.png")
    print(f"  saved winrate: {OUT_DIR / 'winrate_per_lead.png'}")

    # 3. Per-variable trajectory plot
    per_var_trajectory(scorecards, OUT_DIR / "per_variable_trajectory.png")
    print(f"  saved per-variable: {OUT_DIR / 'per_variable_trajectory.png'}")

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
