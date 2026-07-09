#!/usr/bin/env python3
"""Per-variable % improvement vs K (matched baseline), plus baseline MAE
growth with K to show the headroom structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MATCHED = {
    1: "mz_r4_m3_i32_seg32_h16_teacher_step10000",
    2: "mz_r4_m3_i32_seg32_h16_targetK2_matched",
    4: "mz_r4_m3_i32_seg32_h16_targetK4_matched",
}
VARIABLES = [
    ("mean_sea_level_pressure", "MSLP (surface mass)", "#1f77b4"),
    ("geopotential",            "Z (upper-level mass)", "#2ca02c"),
    ("u_component_of_wind",     "U (zonal wind)",       "#d62728"),
    ("v_component_of_wind",     "V (meridional wind)",  "#9467bd"),
]


def load_last(d: Path, step_target=400):
    with open(d / "eval_log.json") as f:
        log = json.load(f)
    return max(log, key=lambda e: e["step"] if e["step"] <= step_target else -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path,
                   default=Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory"))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    Ks = sorted(MATCHED.keys())
    data = {k: load_last(args.results_root / MATCHED[k]) for k in Ks}

    # ----- plot 1: per-variable % improvement vs K -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for vkey, vlabel, color in VARIABLES:
        imp = [100 * (data[k][f"baseline_{vkey}_MAE"] - data[k][f"corrected_{vkey}_MAE"])
               / data[k][f"baseline_{vkey}_MAE"] for k in Ks]
        ax1.plot(Ks, imp, "o-", color=color, lw=2, markersize=9, label=vlabel)
        for k, v in zip(Ks, imp):
            ax1.annotate(f"{v:+.2f}%", (k, v), textcoords="offset points",
                         xytext=(6, 6), fontsize=8, color=color)
    # overall
    imp_all = [100 * (data[k]["baseline_overall_MAE"] - data[k]["corrected_overall_MAE"])
               / data[k]["baseline_overall_MAE"] for k in Ks]
    ax1.plot(Ks, imp_all, "ks--", lw=1.4, alpha=0.6, markersize=8, label="overall (weighted)")

    ax1.axhline(0, color="k", lw=0.6)
    ax1.set_xlabel("target_steps K  (rollout horizon in Δt)")
    ax1.set_ylabel("MZ improvement over matched baseline (%)")
    ax1.set_title("Per-variable MZ improvement vs K  (step 400, r=4)")
    ax1.set_xticks([1, 2, 4])
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="best")

    # ----- plot 2: baseline MAE growth vs K (shows headroom) -----
    for vkey, vlabel, color in VARIABLES:
        base = [data[k][f"baseline_{vkey}_MAE"] for k in Ks]
        # normalize so K=1 → 1.0
        base_norm = [b / base[0] for b in base]
        ax2.plot(Ks, base_norm, "o-", color=color, lw=2, markersize=9, label=vlabel)
        for k, v in zip(Ks, base_norm):
            ax2.annotate(f"{v:.2f}×", (k, v), textcoords="offset points",
                         xytext=(6, 6), fontsize=8, color=color)
    ax2.axhline(1, color="k", lw=0.6)
    ax2.set_xlabel("target_steps K")
    ax2.set_ylabel("baseline MAE  /  baseline MAE at K=1")
    ax2.set_title("Baseline error growth with K  (headroom for MZ)")
    ax2.set_xticks([1, 2, 4])
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="best")

    fig.tight_layout()
    out = args.out_dir / "per_variable_improvement_vs_K.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")

    # Also dump the numbers as a table
    lines = ["K,variable,base_MAE,MZ_MAE,improvement_%"]
    for k in Ks:
        for vkey, vlabel, _ in VARIABLES:
            b = data[k][f"baseline_{vkey}_MAE"]
            m = data[k][f"corrected_{vkey}_MAE"]
            lines.append(f"{k},{vkey},{b:.4f},{m:.4f},{100*(b-m)/b:.4f}")
        b = data[k]["baseline_overall_MAE"]; m = data[k]["corrected_overall_MAE"]
        lines.append(f"{k},overall,{b:.4f},{m:.4f},{100*(b-m)/b:.4f}")
    (args.out_dir / "per_variable_K_table.csv").write_text("\n".join(lines) + "\n")
    print(f"saved: {args.out_dir / 'per_variable_K_table.csv'}")


if __name__ == "__main__":
    main()
