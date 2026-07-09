#!/usr/bin/env python
"""Zoom into v23 K=1 trailing loss to verify plateau before K=22 launch."""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"step (\d+)/\d+ loss ([\d.]+)")


def parse_log(p: Path):
    steps, losses = [], []
    with p.open() as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1))); losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def load_combined(paths):
    s_all, l_all = [], []
    for p in paths:
        s, l = parse_log(p)
        s_all.append(s); l_all.append(l)
    s = np.concatenate(s_all) if s_all else np.array([])
    l = np.concatenate(l_all) if l_all else np.array([])
    order = np.argsort(s); s, l = s[order], l[order]
    _, idx = np.unique(s, return_index=True)
    return s[idx], l[idx]


def block_stats(s, l, block_size=500):
    """Return arrays of (block_center, mean, p25, p75)."""
    if not len(s):
        return np.array([]), np.array([]), np.array([]), np.array([])
    lo, hi = int(s.min()), int(s.max())
    starts = np.arange(lo, hi + 1, block_size)
    centers = starts + block_size // 2
    means, p25, p75 = [], [], []
    for st in starts:
        mask = (s >= st) & (s < st + block_size)
        if mask.sum():
            means.append(l[mask].mean())
            p25.append(np.percentile(l[mask], 25))
            p75.append(np.percentile(l[mask], 75))
        else:
            means.append(np.nan); p25.append(np.nan); p75.append(np.nan)
    return centers, np.array(means), np.array(p25), np.array(p75)


def main():
    LOGS = Path("/home/lm8598/Weather_Global_experiments/logs")
    OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-26-v23/plots/K1_trailing_plateau.png")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    v23 = load_combined(sorted(LOGS.glob("v23_chain_*.out")))
    v15 = load_combined([
        LOGS / "v15_20k_v2_8018520.out",
        LOGS / "v15_20k_v2_resume_8074257.out",
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left: zoom into steps 10000-20000 with 500-step blocks
    ax = axes[0]
    for steps, losses, color, label in [
        (v23[0], v23[1], "C0", "v23 d_conv=8"),
        (v15[0], v15[1], "C3", "v15 d_conv=4"),
    ]:
        c, m, p25, p75 = block_stats(steps, losses, block_size=500)
        mask = (c >= 10000) & (c <= 20000)
        ax.fill_between(c[mask], p25[mask], p75[mask], color=color, alpha=0.18)
        ax.plot(c[mask], m[mask], color=color, lw=2.2, marker="o", markersize=4, label=label)

    ax.set_xlabel("training step")
    ax.set_ylabel("training loss (500-step block mean ± IQR)")
    ax.set_title("K=1 trailing loss zoom (steps 10000-20000)\nverifying plateau before K=22 launch")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.43, 0.50)
    ax.legend(loc="upper right")
    ax.axhline(0.47, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(10200, 0.4705, "y=0.47 (guide)", fontsize=8, color="gray")

    # Right: per-1000-step rolling stats with slope estimate
    ax = axes[1]
    bin_size = 1000
    for steps, losses, color, label in [
        (v23[0], v23[1], "C0", "v23 d_conv=8"),
        (v15[0], v15[1], "C3", "v15 d_conv=4"),
    ]:
        c, m, p25, p75 = block_stats(steps, losses, block_size=bin_size)
        # only window 10000+ for slope calc
        late_mask = (c >= 10000) & ~np.isnan(m)
        c_late, m_late = c[late_mask], m[late_mask]
        if len(c_late) >= 2:
            # linear fit on the late portion to quantify slope
            slope, intercept = np.polyfit(c_late, m_late, 1)
            slope_per_1k = slope * 1000  # loss change per 1000 steps
            ax.plot(c_late, m_late, color=color, lw=2.2, marker="o", markersize=5,
                    label=f"{label}: slope={slope_per_1k:+.4f}/1k-steps")
            ax.plot(c_late, slope * c_late + intercept, color=color, lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("training step (late portion only)")
    ax.set_ylabel(f"training loss ({bin_size}-step block mean)")
    ax.set_title("Late-K=1 linear slope\n(slope ~0 = plateau confirmed)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"saved: {OUT}")

    # numerical summary
    print()
    print("=== late-K=1 block means (500-step windows) ===")
    for steps, losses, name in [(v23[0], v23[1], "v23"), (v15[0], v15[1], "v15")]:
        c, m, _, _ = block_stats(steps, losses, block_size=500)
        print(f"\n{name}:")
        for i, (cc, mm) in enumerate(zip(c, m)):
            if cc >= 10000 and not np.isnan(mm):
                print(f"  step [{cc-250:>5}, {cc+249:>5}] mean = {mm:.4f}")


if __name__ == "__main__":
    main()
