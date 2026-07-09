#!/usr/bin/env python
"""Plot v23 K=1 training loss vs step, overlay v15 (the d_conv=4 fresh-init baseline).

Parses stdout logs for lines like:
    step N/M loss X grad_norm Y step_time Z

v23 chain has multiple log files (one per slurm seg); we concatenate by step.
v15 has a single log (it never crashed/chained).
"""
from __future__ import annotations
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"step (\d+)/\d+ loss ([\d.]+)")


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (steps, losses) from a single stdout log file."""
    steps, losses = [], []
    with path.open() as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.asarray(steps), np.asarray(losses)


def parse_logs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Parse multiple logs, sort by step, dedupe."""
    all_steps, all_losses = [], []
    for p in paths:
        s, l = parse_log(p)
        all_steps.append(s)
        all_losses.append(l)
    steps = np.concatenate(all_steps) if all_steps else np.array([])
    losses = np.concatenate(all_losses) if all_losses else np.array([])
    order = np.argsort(steps)
    steps = steps[order]
    losses = losses[order]
    # dedupe (same step might appear in multiple log files because chain restarts produce duplicate first lines)
    if len(steps):
        _, unique_idx = np.unique(steps, return_index=True)
        steps = steps[unique_idx]
        losses = losses[unique_idx]
    return steps, losses


def smooth(y: np.ndarray, window: int = 50) -> np.ndarray:
    """Centered moving average that handles edges by reflecting (no spurious zeros)."""
    if window <= 1 or len(y) < window:
        return y
    half = window // 2
    # Pad with reflection to avoid edge artifacts
    padded = np.pad(y, half, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[half:half + len(y)]


def main() -> int:
    LOGS_DIR = Path("/home/lm8598/Weather_Global_experiments/logs")
    OUT_PLOT = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-26-v23/plots/train_loss_v23_vs_v15.png")
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    # v23 chain log files
    v23_logs = sorted(LOGS_DIR.glob("v23_chain_*.out"))
    # Drop the OOM-only logs (they have no `step N/M loss X` lines, but parse_log handles that)
    v23_steps, v23_losses = parse_logs(v23_logs)
    print(f"v23: parsed {len(v23_logs)} log files, {len(v23_steps)} (step, loss) pairs")
    print(f"     step range {v23_steps.min() if len(v23_steps) else 'NA'} .. {v23_steps.max() if len(v23_steps) else 'NA'}")

    # v15 — was also chained (2 logs). Combine.
    v15_logs = [
        LOGS_DIR / "v15_20k_v2_8018520.out",       # step 1 - ~17940
        LOGS_DIR / "v15_20k_v2_resume_8074257.out", # step 17001 - 20000
    ]
    v15_logs = [p for p in v15_logs if p.exists()]
    v15_steps, v15_losses = parse_logs(v15_logs)
    print(f"v15: parsed {len(v15_logs)} log files, {len(v15_steps)} (step, loss) pairs")
    print(f"     step range {v15_steps.min() if len(v15_steps) else 'NA'} .. {v15_steps.max() if len(v15_steps) else 'NA'}")

    if not len(v23_steps) or not len(v15_steps):
        print("ERROR: no data to plot")
        return 1

    # Bin into "epochs" using each version's actual steps_per_epoch (different because
    # of segment_steps/bptt_steps differences). v23: 424 steps/epoch, v15: 1276 steps/epoch.
    V23_STEPS_PER_EPOCH = 424
    V15_STEPS_PER_EPOCH = 1276

    def epoch_bin(steps: np.ndarray, losses: np.ndarray, steps_per_epoch: int,
                  drop_incomplete: bool = True):
        """Return arrays of (epoch_idx, mean_loss, p25_loss, p75_loss).

        If drop_incomplete: drop the last epoch if its sample count is < 50%
        of the typical epoch's sample count (logs are sampled every 10 steps,
        so typical = ~steps_per_epoch / 10).
        """
        epoch_idx = (steps // steps_per_epoch).astype(int)
        unique = np.unique(epoch_idx)
        means = np.zeros_like(unique, dtype=float)
        p25 = np.zeros_like(unique, dtype=float)
        p75 = np.zeros_like(unique, dtype=float)
        counts = np.zeros_like(unique, dtype=int)
        for i, e in enumerate(unique):
            mask = epoch_idx == e
            counts[i] = mask.sum()
            if counts[i]:
                means[i] = np.mean(losses[mask])
                p25[i] = np.percentile(losses[mask], 25)
                p75[i] = np.percentile(losses[mask], 75)
        if drop_incomplete and len(unique) >= 2:
            typical_count = int(np.median(counts[:-1]))  # exclude potentially-partial last
            if counts[-1] < typical_count * 0.5:
                unique = unique[:-1]
                means = means[:-1]
                p25 = p25[:-1]
                p75 = p75[:-1]
                counts = counts[:-1]
        return unique, means, p25, p75

    v23_ep, v23_mean, v23_p25, v23_p75 = epoch_bin(v23_steps, v23_losses, V23_STEPS_PER_EPOCH)
    v15_ep, v15_mean, v15_p25, v15_p75 = epoch_bin(v15_steps, v15_losses, V15_STEPS_PER_EPOCH)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Left: per-epoch (each point = 1 epoch's mean loss, shaded = IQR)
    ax = axes[0]
    ax.fill_between(v23_ep, v23_p25, v23_p75, color="C0", alpha=0.20, label="_v23 IQR")
    ax.plot(v23_ep, v23_mean, color="C0", lw=2.2, marker="o", markersize=3,
            label=f"v23 d_conv=8 (fresh, {V23_STEPS_PER_EPOCH} steps/epoch)")
    ax.fill_between(v15_ep, v15_p25, v15_p75, color="C3", alpha=0.20, label="_v15 IQR")
    ax.plot(v15_ep, v15_mean, color="C3", lw=2.2, marker="o", markersize=3,
            label=f"v15 d_conv=4 (fresh, {V15_STEPS_PER_EPOCH} steps/epoch)")
    ax.set_xlabel("epoch (one pass through training data)")
    ax.set_ylabel("training loss (mean per epoch ± IQR)")
    ax.set_title("v23 vs v15 K=0 training loss — per-epoch\n(both fresh init; v23 epoch=424 steps, v15 epoch=1276 steps)")
    ax.set_ylim(0.40, 0.56)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Right: per-step, heavily smoothed (window=500)
    ax = axes[1]
    ax.plot(v23_steps, v23_losses, color="C0", alpha=0.10, lw=0.4, label="_v23 raw")
    ax.plot(v23_steps, smooth(v23_losses, 500), color="C0", lw=2.0,
            label="v23 d_conv=8 (smoothed-500)")
    ax.plot(v15_steps, v15_losses, color="C3", alpha=0.10, lw=0.4, label="_v15 raw")
    ax.plot(v15_steps, smooth(v15_losses, 500), color="C3", lw=2.0,
            label="v15 d_conv=4 (smoothed-500)")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_title("v23 vs v15 K=0 training loss — per-step\n(smoothed over 500 steps)")
    ax.set_ylim(0.40, 0.56)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=120, bbox_inches="tight")
    print(f"saved: {OUT_PLOT}")

    # also print summary stats at key steps
    print()
    print("=== loss snapshots (smoothed-100) ===")
    print(f"{'step':>6}  {'v23':>8}  {'v15':>8}  {'Δ (v23-v15)':>12}")
    v23_smooth = smooth(v23_losses, 100)
    v15_smooth = smooth(v15_losses, 100)
    for step_target in [100, 500, 1000, 2000, 5000, 10000, 15000, 20000]:
        v23_idx = np.argmin(np.abs(v23_steps - step_target)) if len(v23_steps) else None
        v15_idx = np.argmin(np.abs(v15_steps - step_target)) if len(v15_steps) else None
        v23_val = v23_smooth[v23_idx] if v23_idx is not None and abs(v23_steps[v23_idx] - step_target) < 50 else None
        v15_val = v15_smooth[v15_idx] if v15_idx is not None and abs(v15_steps[v15_idx] - step_target) < 50 else None
        if v23_val is not None and v15_val is not None:
            print(f"{step_target:>6}  {v23_val:>8.4f}  {v15_val:>8.4f}  {v23_val - v15_val:>+12.4f}")
        else:
            print(f"{step_target:>6}  {v23_val or 'n/a':>8}  {v15_val or 'n/a':>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
