"""Train loss comparison: v12 vs v12 bigmamba.
Both ran 20K steps, same data, same seed. Different Mamba capacity:
  v12:        d_inner=128, dt_rank=8,   ~117K Mamba params
  bigmamba:   d_inner=256, dt_rank=256, ~486K Mamba params
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

V12 = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_fast_20k/train_log.json"
BIG = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_bigmamba_20k/train_log.json"
OUT = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_vs_bigmamba_train_loss.png"

EPOCH_LEN = 364


def per_epoch(log):
    steps = np.asarray([e["step"] for e in log])
    losses = np.asarray([e["loss"] for e in log])
    mid, m, s = [], [], []
    for i in range(int(steps.max()) // EPOCH_LEN + 1):
        s_lo, s_hi = i * EPOCH_LEN + 1, (i + 1) * EPOCH_LEN
        mask = (steps >= s_lo) & (steps <= s_hi)
        if mask.sum() < 50:
            continue
        mid.append((s_lo + s_hi) / 2)
        m.append(losses[mask].mean())
        s.append(losses[mask].std())
    return steps, losses, np.asarray(mid), np.asarray(m), np.asarray(s)


def main():
    v12 = json.loads(Path(V12).read_text())
    big = json.loads(Path(BIG).read_text())
    s_v, l_v, mid_v, ep_v, std_v = per_epoch(v12)
    s_b, l_b, mid_b, ep_b, std_b = per_epoch(big)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-step loss with smoothing
    win = 50
    sm_v = np.convolve(l_v, np.ones(win) / win, mode="valid")
    sm_b = np.convolve(l_b, np.ones(win) / win, mode="valid")
    axes[0].plot(s_v[:len(sm_v)] + win // 2, sm_v, color="tab:green",
                 label=f"v12 (151K trainable, ~117K Mamba)", lw=1.5)
    axes[0].plot(s_b[:len(sm_b)] + win // 2, sm_b, color="tab:purple",
                 label=f"v12 bigmamba (520K trainable, ~486K Mamba)", lw=1.5)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("training loss (50-step moving avg)")
    axes[0].set_title("Per-step train loss (50-step moving avg)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0.78, 0.86)

    # Right: per-epoch mean
    axes[1].errorbar(mid_v, ep_v, yerr=std_v, fmt="o-", color="tab:green",
                     label=f"v12 (epoch 1={ep_v[0]:.4f}, epoch {len(ep_v)}={ep_v[-1]:.4f})",
                     markersize=4, lw=1.6, capsize=2, alpha=0.85)
    axes[1].errorbar(mid_b, ep_b, yerr=std_b, fmt="o-", color="tab:purple",
                     label=f"bigmamba (epoch 1={ep_b[0]:.4f}, epoch {len(ep_b)}={ep_b[-1]:.4f})",
                     markersize=4, lw=1.6, capsize=2, alpha=0.85)
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("per-epoch mean train loss\n(epoch=364 steps; error bars = within-epoch std)")
    axes[1].set_title("Per-epoch mean train loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].set_ylim(0.815, 0.83)

    fig.suptitle("v12 vs v12 bigmamba: 4× Mamba capacity gives essentially identical train loss",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")
    print(f"v12        epoch 1->{len(ep_v)}: {ep_v[0]:.5f} -> {ep_v[-1]:.5f}  drop {ep_v[0]-ep_v[-1]:.5f}")
    print(f"bigmamba   epoch 1->{len(ep_b)}: {ep_b[0]:.5f} -> {ep_b[-1]:.5f}  drop {ep_b[0]-ep_b[-1]:.5f}")
    diffs = np.abs(ep_v[:min(len(ep_v), len(ep_b))] - ep_b[:min(len(ep_v), len(ep_b))])
    print(f"max abs per-epoch diff between v12 and bigmamba: {diffs.max():.6f}")


if __name__ == "__main__":
    main()
