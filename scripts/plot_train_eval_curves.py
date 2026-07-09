#!/usr/bin/env python3
"""Plot training loss and evaluation loss curves side-by-side, to verify
they trend together (or diverge -> overfitting).

For each run, plots:
  - Training loss (raw + MA(50) smoothed) on left y-axis
  - Eval corrected_overall_MAE (TF and AR if available) on right y-axis
  - Eval baseline_overall_MAE as constant reference line
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/home/lm8598/Weather_Global_experiments/results/mz_residual_memory")
OUT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-04-27_train_eval_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Runs to plot — each entry: (label, run_dir_name, phase_start_step)
# phase_start_step = first step of the K-specific phase to keep on the x-axis.
# K=4_7k and K=6_8k contain the full chain up to that K, so we filter aggressively.
RUNS = [
    ("K=1 best (segment=16)", "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_4k_cont", 0),
    # phase_start strictly AFTER previous phase's last eval entry, so the eval
    # line for each panel sees only its own horizon (avoids the horizon-shift
    # discontinuity at K-boundary that looks like an MAE jump).
    ("K=2 fresh from K=1",    "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K2_8k_fresh", 4001),
    ("K=4 phase (resumed from K=2 step 5000)", "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K4_7k", 5001),
    ("K=6 phase (resumed from K=4 step 6000)", "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K6_8k", 6001),
    ("K=8 phase (resumed from K=6 step 8000)", "mz_fullmamba_paperckpt_r1_in2_seg16_meshed_m5_h128_ds16_fullvars_K8_10k", 8001),
    ("slide K=1 (segment=1)", "mz_fullmamba_slide_paperckpt_r1_in2_seg1_meshed_m5_h128_ds16_fullvars_K1_4k", 0),
]


def smooth(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_run(label, run_dir, ax_train, ax_eval, color, phase_start=0):
    train_path = run_dir / "train_log.json"
    eval_path = run_dir / "eval_log.json"

    if not train_path.exists():
        print(f"  {label}: train_log missing, skip")
        return

    train = json.load(open(train_path))
    train.sort(key=lambda e: e["step"])
    train = [e for e in train if e["step"] >= phase_start]

    steps = np.array([e["step"] for e in train])
    losses = np.array([e.get("loss", e.get("total_loss", 0)) for e in train])

    # Pick a smoothing window that actually fits in the phase — for short phases
    # (K=4 has only ~800 steps) MA-200 would consume half the data
    win = 200 if len(steps) >= 400 else max(50, len(steps) // 4)
    if len(steps) > win:
        smoothed = smooth(losses, win)
        sm_steps = steps[win - 1:]
        ax_train.plot(sm_steps, smoothed, color=color, linewidth=2.0,
                      label=f"Train loss (MA-{win})")

    # Eval: only the corrected TF curve + the constant baseline as a faint reference
    if eval_path.exists():
        ev = json.load(open(eval_path))
        ev.sort(key=lambda e: e["step"])
        ev = [e for e in ev if e["step"] >= phase_start]

        ev_steps = [e["step"] for e in ev]
        bm = [e.get("baseline_overall_MAE") for e in ev]
        cm_tf = [e.get("corrected_overall_MAE") for e in ev]

        if any(b is not None for b in bm):
            valid = [(s, b) for s, b in zip(ev_steps, bm) if b is not None]
            if valid:
                _, b0 = valid[0]
                ax_eval.axhline(b0, color="black", linestyle=":", linewidth=1.0,
                                alpha=0.4, label=f"baseline MAE = {b0:.3f}")

        if any(c is not None for c in cm_tf):
            valid = [(s, c) for s, c in zip(ev_steps, cm_tf) if c is not None]
            if valid:
                steps_v, vals_v = zip(*valid)
                ax_eval.plot(steps_v, vals_v, color="C3", linestyle="-",
                             linewidth=2.5, marker="o", markersize=6,
                             label="Eval MAE (TF, validation 2022)")


def main():
    # Plot 1: train + eval overall_MAE on same x-axis (per run)
    n_runs = len(RUNS)
    fig, axes = plt.subplots(n_runs, 1, figsize=(11, 3.5 * n_runs), sharex=False)
    if n_runs == 1:
        axes = [axes]

    colors = ["C0", "C1", "C4", "C5", "C6", "C2"]
    for ax_main, (label, run_name, phase_start), color in zip(axes, RUNS, colors):
        run_dir = RESULTS_DIR / run_name
        if not run_dir.exists():
            print(f"  {label}: dir missing, skip")
            continue

        ax_train = ax_main
        ax_eval = ax_main.twinx()

        plot_run(label, run_dir, ax_train, ax_eval, color, phase_start=phase_start)

        ax_train.set_xlabel("Training step")
        ax_train.set_ylabel("Train loss (MA-200, normalized MSE)", color=color)
        ax_train.tick_params(axis="y", labelcolor=color)
        ax_train.grid(alpha=0.25)
        ax_train.set_title(label, fontsize=12, fontweight="bold")

        ax_eval.set_ylabel("Eval overall_MAE", color="C3")
        ax_eval.tick_params(axis="y", labelcolor="C3")

        lines_t, labels_t = ax_train.get_legend_handles_labels()
        lines_e, labels_e = ax_eval.get_legend_handles_labels()
        ax_main.legend(lines_t + lines_e, labels_t + labels_e,
                       loc="upper right", fontsize=9, framealpha=0.9)

    plt.suptitle(
        "Training loss vs Eval MAE — train (left axis, blue/orange/green smoothed) "
        "vs eval (right axis, red ●)",
        fontsize=11, y=1.005,
    )
    plt.tight_layout()
    out = OUT_DIR / "train_vs_eval_per_run.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)

    # Plot 2: All runs train loss together (normalized to start of own phase)
    fig, ax = plt.subplots(figsize=(12, 6))
    for (label, run_name, phase_start), color in zip(RUNS, colors):
        run_dir = RESULTS_DIR / run_name
        if not run_dir.exists():
            continue
        train = json.load(open(run_dir / "train_log.json"))
        train.sort(key=lambda e: e["step"])
        train = [e for e in train if e["step"] >= phase_start]

        steps = np.array([e["step"] for e in train])
        losses = np.array([e.get("loss", e.get("total_loss", 0)) for e in train])

        win = 200 if len(steps) >= 400 else max(50, len(steps) // 4)
        if len(steps) > win:
            sm = smooth(losses, win)
            sm_steps = steps[win - 1:]
            ax.plot(sm_steps, sm, color=color, linewidth=2, label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Train loss (smoothed MA(50))")
    ax.set_title("Training loss across runs")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "train_loss_compare.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)

    # Plot 3: All eval Z RMSE TF Δ% trajectory together
    fig, ax = plt.subplots(figsize=(12, 6))
    for (label, run_name, phase_start), color in zip(RUNS, colors):
        run_dir = RESULTS_DIR / run_name
        if not run_dir.exists():
            continue
        ev_path = run_dir / "eval_log.json"
        if not ev_path.exists():
            continue
        ev = json.load(open(ev_path))
        ev.sort(key=lambda e: e["step"])
        ev = [e for e in ev if e["step"] >= phase_start]

        # Z RMSE TF Δ%
        valid = []
        for e in ev:
            bz = e.get("baseline_geopotential_RMSE")
            tz = e.get("corrected_geopotential_RMSE")
            if bz is not None and tz is not None:
                pct = 100 * (bz - tz) / bz
                valid.append((e["step"], pct))
        if valid:
            steps_v, vals_v = zip(*valid)
            ax.plot(steps_v, vals_v, color=color, linewidth=2, marker="o", markersize=5,
                    label=f"{label} (TF)")

        # Z RMSE AR Δ%
        valid_ar = []
        for e in ev:
            bz = e.get("baseline_geopotential_RMSE")
            az = e.get("corrected_ar_geopotential_RMSE")
            if bz is not None and az is not None:
                pct = 100 * (bz - az) / bz
                valid_ar.append((e["step"], pct))
        if valid_ar:
            steps_v, vals_v = zip(*valid_ar)
            ax.plot(steps_v, vals_v, color=color, linewidth=1.5, marker="s", markersize=3,
                    linestyle="--", label=f"{label} (AR)", alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Z (geopotential 13L) RMSE Δ% (uniform)")
    ax.set_title("Z RMSE improvement vs baseline — trajectory per run\n"
                 "Solid=TF, Dashed=AR; +ve = corrected better than baseline")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "eval_Z_RMSE_compare.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
