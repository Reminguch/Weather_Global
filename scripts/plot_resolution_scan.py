#!/usr/bin/env python3
"""Resolution-scan comparison plots for MZ residual runs at r=2,4,6,8.

Reads train_log.json + eval_log.json from each
results/mz_residual_memory/mz_r{R}_m3_i32_seg32_h16_teacher_step10000
and produces four figures in --out-dir:
  - loss_curves.png            (train loss vs step, one color per resolution)
  - eval_mae.png               (baseline vs MZ overall MAE vs step)
  - eval_improvement.png       (per-step % improvement vs step)
  - per_variable_mae_step400.png  (bars: base vs MZ per variable, per res)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RES_COLORS = {2: "#1f77b4", 4: "#ff7f0e", 6: "#2ca02c", 8: "#d62728"}
VARIABLES = [
    ("mean_sea_level_pressure", "MSLP"),
    ("geopotential", "Z"),
    ("u_component_of_wind", "U"),
    ("v_component_of_wind", "V"),
]


def load_logs(results_root: Path, resolutions):
    runs = {}
    for r in resolutions:
        d = results_root / f"mz_r{r}_m3_i32_seg32_h16_teacher_step10000"
        if not d.exists():
            print(f"[warn] missing run dir for r={r}: {d}")
            continue
        with open(d / "train_log.json") as f:
            train = json.load(f)
        with open(d / "eval_log.json") as f:
            evals = json.load(f)
        runs[r] = {"train": train, "eval": evals, "dir": d}
    return runs


def plot_loss_curves(runs, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, bundle in runs.items():
        tr = bundle["train"]
        steps = [e["step"] for e in tr]
        loss = [e["loss"] for e in tr]
        ax.plot(steps, loss, color=RES_COLORS[r], lw=1.2, alpha=0.9, label=f"r={r}")
    ax.set_xlabel("training step")
    ax.set_ylabel("total loss (normalized)")
    ax.set_title("MZ training loss — resolution scan")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def plot_eval_mae(runs, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, bundle in runs.items():
        ev = bundle["eval"]
        steps = [e["step"] for e in ev]
        base = [e["baseline_overall_MAE"] for e in ev]
        mz = [e["corrected_overall_MAE"] for e in ev]
        c = RES_COLORS[r]
        ax.plot(steps, base, color=c, lw=1.0, ls="--", alpha=0.6,
                label=f"r={r} baseline")
        ax.plot(steps, mz, color=c, lw=1.8, marker="o", label=f"r={r} MZ")
    ax.set_xlabel("training step")
    ax.set_ylabel("overall MAE")
    ax.set_title("Eval overall MAE: baseline vs MZ-corrected")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def plot_improvement(runs, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, bundle in runs.items():
        ev = bundle["eval"]
        steps = [e["step"] for e in ev]
        imp = [
            100.0 * (e["baseline_overall_MAE"] - e["corrected_overall_MAE"])
            / e["baseline_overall_MAE"]
            for e in ev
        ]
        ax.plot(steps, imp, color=RES_COLORS[r], lw=1.8, marker="o", label=f"r={r}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("training step")
    ax.set_ylabel("MAE improvement over baseline (%)")
    ax.set_title("MZ improvement vs training step — resolution scan")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def plot_per_variable(runs, out: Path, step_target: int = 400):
    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(4 * len(VARIABLES), 4.5),
                             sharey=False)
    res_list = list(runs.keys())
    x = np.arange(len(res_list))
    width = 0.38

    for ax, (vkey, vlabel) in zip(axes, VARIABLES):
        base_vals, mz_vals = [], []
        for r in res_list:
            ev = runs[r]["eval"]
            # pick entry with largest step <= step_target
            entry = max(ev, key=lambda e: e["step"] if e["step"] <= step_target else -1)
            base_vals.append(entry[f"baseline_{vkey}_MAE"])
            mz_vals.append(entry[f"corrected_{vkey}_MAE"])
        base_vals = np.array(base_vals)
        mz_vals = np.array(mz_vals)
        ax.bar(x - width / 2, base_vals, width, color="#888", label="baseline")
        ax.bar(x + width / 2, mz_vals, width,
               color=[RES_COLORS[r] for r in res_list], label="MZ")
        ax.set_xticks(x)
        ax.set_xticklabels([f"r={r}" for r in res_list])
        ax.set_title(vlabel)
        ax.grid(axis="y", alpha=0.3)
        for xi, (b, m) in enumerate(zip(base_vals, mz_vals)):
            imp = 100 * (b - m) / b if b > 0 else 0.0
            ax.text(xi, max(b, m) * 1.03, f"{imp:+.1f}%",
                    ha="center", va="bottom", fontsize=8, color="darkgreen")
    axes[0].set_ylabel("MAE (physical units)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Per-variable MAE at step {step_target}  (baseline vs MZ)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path,
                   default=Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--resolutions", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--step-target", type=int, default=400)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_logs(args.results_root, args.resolutions)
    if not runs:
        raise SystemExit("no runs found")

    plot_loss_curves(runs, args.out_dir / "loss_curves.png")
    plot_eval_mae(runs, args.out_dir / "eval_mae.png")
    plot_improvement(runs, args.out_dir / "eval_improvement.png")
    plot_per_variable(runs, args.out_dir / f"per_variable_mae_step{args.step_target}.png",
                      step_target=args.step_target)


if __name__ == "__main__":
    main()
