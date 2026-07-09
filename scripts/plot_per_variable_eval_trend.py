#!/usr/bin/env python3
"""Per-variable eval RMSE Δ% trajectory over training step.

The K=8 run contains the full curriculum chain (K=1 → K=2 → K=4 → K=6 → K=8)
in a single train_log/eval_log. This script plots, for each of the 11 GraphCast
target variables, the relative improvement Δ% (= (baseline - corrected) /
baseline) of TF and AR forecasts vs the frozen baseline, over the entire
training trajectory. K-phase boundaries are marked with vertical dashed lines.

Note: per-variable TRAIN loss is not logged (train_log has only the aggregate
loss). Only EVAL is per-variable. Train aggregate loss is plotted in the
existing plot_train_eval_curves.py.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory/"
           "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K8_10k")
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT.mkdir(parents=True, exist_ok=True)

VARS = [
    ("2m_temperature",            "2m T",       "surface"),
    ("mean_sea_level_pressure",   "MSLP",       "surface"),
    ("10m_u_component_of_wind",   "10m u",      "surface"),
    ("10m_v_component_of_wind",   "10m v",      "surface"),
    ("total_precipitation_6hr",   "precip 6h",  "surface"),
    ("geopotential",              "Z",          "atmos"),
    ("temperature",               "T",          "atmos"),
    ("specific_humidity",         "q",          "atmos"),
    ("u_component_of_wind",       "u wind",     "atmos"),
    ("v_component_of_wind",       "v wind",     "atmos"),
    ("vertical_velocity",         "w",          "atmos"),
]

# K-phase boundaries (steps where horizon changes; detected from baseline_MAE jumps)
PHASES = [
    (0,    2000,  "K=1", "h=1"),
    (2200, 5000,  "K=2", "h=2"),
    (5200, 6000,  "K=4", "h=4"),
    (6200, 8000,  "K=6", "h=6"),
    (8200, 10000, "K=8", "h=8"),
]
BOUNDARIES = [2100, 5100, 6100, 8100]


def load():
    ev = json.load(open(RUN / "eval_log.json"))
    ev.sort(key=lambda e: e["step"])
    return ev


def delta_pct(ev, base_key, corr_key):
    out = []
    for e in ev:
        b = e.get(base_key)
        c = e.get(corr_key)
        if b is None or c is None or b == 0:
            continue
        out.append((e["step"], 100 * (b - c) / b))
    return out


def main():
    ev = load()

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    axes = axes.flatten()

    for ax, (var, label, _) in zip(axes, VARS):
        tf_pts = delta_pct(ev, f"baseline_{var}_RMSE", f"corrected_{var}_RMSE")
        ar_pts = delta_pct(ev, f"baseline_{var}_RMSE", f"corrected_ar_{var}_RMSE")

        if tf_pts:
            xs, ys = zip(*tf_pts)
            ax.plot(xs, ys, color="C3", linewidth=1.8, marker="o",
                    markersize=4, label="TF")
        if ar_pts:
            xs, ys = zip(*ar_pts)
            ax.plot(xs, ys, color="C0", linewidth=1.5, marker="s",
                    markersize=3, linestyle="--", alpha=0.7, label="AR")

        for b in BOUNDARIES:
            ax.axvline(b, color="black", linestyle=":", linewidth=0.7, alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("RMSE Δ% (TF=red, AR=blue)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    # Annotate K-phase labels on top of last row
    for ax in axes[-4:]:
        ax.set_xlabel("Training step")
        for (s0, s1, k_lbl, h_lbl) in PHASES:
            mid = (s0 + s1) / 2
            ax.annotate(f"{k_lbl}\n({h_lbl})", xy=(mid, ax.get_ylim()[1]),
                        xytext=(mid, ax.get_ylim()[1] * 0.95), ha="center",
                        fontsize=7, color="black", alpha=0.5)

    # Hide unused panel
    if len(axes) > len(VARS):
        for ax in axes[len(VARS):]:
            ax.set_visible(False)

    fig.suptitle(
        "Per-variable RMSE Δ% (vs frozen GraphCast baseline) over training step\n"
        f"Run: K=8_10k (full chain K=1→2→4→6→8). Vertical dotted = K-phase boundaries.\n"
        "Each panel: TF (red ●) and AR (blue ■, dashed). +ve = corrected better than baseline.",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    out = OUT / "per_variable_eval_trend.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
