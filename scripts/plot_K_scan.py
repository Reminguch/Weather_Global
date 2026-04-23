#!/usr/bin/env python3
"""K-scan comparison: MZ improvement over matched (apples-to-apples) vs weak baseline
across target_steps K=1, 2, 4. Also per-variable bars at step 400.

Reads eval_log.json from:
  K=1 matched:   mz_r4_m3_i32_seg32_h16_teacher_step10000  (baseline trained at K=1)
  K=2 matched:   mz_r4_m3_i32_seg32_h16_targetK2_matched   (FT baseline at K=2, step28000)
  K=4 matched:   mz_r4_m3_i32_seg32_h16_targetK4_matched   (FT baseline at K=4, step28000)
  K=2 weak:      mz_r4_m3_i32_seg32_h16_targetK2_step4000  (K=1-trained baseline at step4000)
  K=4 weak:      mz_r4_m3_i32_seg32_h16_targetK4_step4000  (K=1-trained baseline at step4000)
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
WEAK = {
    2: "mz_r4_m3_i32_seg32_h16_targetK2_step4000",
    4: "mz_r4_m3_i32_seg32_h16_targetK4_step4000",
}

VARIABLES = [
    ("mean_sea_level_pressure", "MSLP"),
    ("geopotential", "Z"),
    ("u_component_of_wind", "U"),
    ("v_component_of_wind", "V"),
]


def load_last(run_dir: Path, step_target: int = 400):
    with open(run_dir / "eval_log.json") as f:
        log = json.load(f)
    # pick entry with largest step <= step_target
    return max(log, key=lambda e: e["step"] if e["step"] <= step_target else -1)


def overall_improvement(e):
    return 100.0 * (e["baseline_overall_MAE"] - e["corrected_overall_MAE"]) / e["baseline_overall_MAE"]


def plot_K_vs_improvement(matched, weak, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    Ks_m = sorted(matched.keys())
    imp_m = [overall_improvement(matched[k]) for k in Ks_m]
    ax.plot(Ks_m, imp_m, "o-", lw=2, color="#1f77b4", markersize=10,
            label="matched baseline (apples-to-apples)")
    for k, v in zip(Ks_m, imp_m):
        ax.annotate(f"{v:+.2f}%", (k, v), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color="#1f77b4")

    Ks_w = sorted(weak.keys())
    imp_w = [overall_improvement(weak[k]) for k in Ks_w]
    ax.plot(Ks_w, imp_w, "s--", lw=1.4, color="#ff7f0e", alpha=0.8, markersize=9,
            label="weak baseline (K=1-trained, step4000)")
    for k, v in zip(Ks_w, imp_w):
        ax.annotate(f"{v:+.2f}%", (k, v), textcoords="offset points",
                    xytext=(8, -12), fontsize=9, color="#ff7f0e")

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("target_steps K")
    ax.set_ylabel("overall MAE improvement (%)")
    ax.set_title("MZ % improvement vs K — matched vs weak baseline (r=4, step 400)")
    ax.set_xticks([1, 2, 4])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def plot_K_base_vs_MZ(matched, out: Path):
    """Absolute base vs MZ MAE (not %) — shows baseline degrades as K grows."""
    fig, ax = plt.subplots(figsize=(8, 5))
    Ks = sorted(matched.keys())
    base = [matched[k]["baseline_overall_MAE"] for k in Ks]
    mz = [matched[k]["corrected_overall_MAE"] for k in Ks]
    width = 0.35
    x = np.arange(len(Ks))
    ax.bar(x - width / 2, base, width, color="#888", label="baseline")
    ax.bar(x + width / 2, mz, width, color="#2ca02c", label="MZ")
    for xi, (b, m) in zip(x, zip(base, mz)):
        ax.text(xi - width / 2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(xi + width / 2, m + 0.5, f"{m:.1f}", ha="center", fontsize=8)
        imp = 100 * (b - m) / b
        ax.text(xi, max(b, m) * 1.06, f"Δ{imp:+.2f}%", ha="center",
                fontsize=9, color="darkgreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel("overall MAE (physical units)")
    ax.set_title("Matched baseline vs MZ — overall MAE at step 400 (r=4)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def plot_K_per_variable(matched, out: Path):
    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(4 * len(VARIABLES), 4.5))
    Ks = sorted(matched.keys())
    x = np.arange(len(Ks))
    width = 0.38

    for ax, (vkey, vlabel) in zip(axes, VARIABLES):
        base = [matched[k][f"baseline_{vkey}_MAE"] for k in Ks]
        mz = [matched[k][f"corrected_{vkey}_MAE"] for k in Ks]
        base = np.array(base); mz = np.array(mz)
        ax.bar(x - width / 2, base, width, color="#888", label="baseline")
        ax.bar(x + width / 2, mz, width, color="#2ca02c", label="MZ")
        for xi, (b, m) in enumerate(zip(base, mz)):
            imp = 100 * (b - m) / b if b > 0 else 0.0
            ax.text(xi, max(b, m) * 1.04, f"{imp:+.2f}%",
                    ha="center", va="bottom", fontsize=8, color="darkgreen")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in Ks])
        ax.set_title(vlabel)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("MAE (physical units)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Per-variable MAE at step 400 — matched baseline vs MZ (r=4)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path,
                   default=Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--step-target", type=int, default=400)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    matched = {k: load_last(args.results_root / d, args.step_target)
               for k, d in MATCHED.items()}
    weak = {k: load_last(args.results_root / d, args.step_target)
            for k, d in WEAK.items()}

    plot_K_vs_improvement(matched, weak, args.out_dir / "K_vs_improvement.png")
    plot_K_base_vs_MZ(matched, args.out_dir / "K_base_vs_MZ_overall.png")
    plot_K_per_variable(matched, args.out_dir / "K_per_variable_step400.png")


if __name__ == "__main__":
    main()
