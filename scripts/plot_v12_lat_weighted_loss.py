"""Plot v12 training loss (already lat-weighted by DeepMind weighted_mse_per_level)
versus training step. Shows raw + per-epoch mean to disambiguate noise from trend.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_fast_20k/train_log.json")
EVAL_DIR = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_fast_20k/eval_results")
OUT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_lat_weighted_loss.png")

EPOCH_LEN = 364  # steps per pass through training data


def main():
    log = json.loads(LOG.read_text())
    steps = np.array([e["step"] for e in log])
    losses = np.array([e["loss"] for e in log])
    n = steps.max()

    # per-epoch mean
    ep_mid, ep_mean, ep_std = [], [], []
    for i in range(n // EPOCH_LEN + 1):
        s_lo = i * EPOCH_LEN + 1
        s_hi = (i + 1) * EPOCH_LEN
        mask = (steps >= s_lo) & (steps <= s_hi)
        if mask.sum() < 50:
            continue
        ep_mid.append((s_lo + s_hi) / 2)
        ep_mean.append(losses[mask].mean())
        ep_std.append(losses[mask].std())
    ep_mid, ep_mean, ep_std = map(np.asarray, (ep_mid, ep_mean, ep_std))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: raw + per-epoch mean
    ax = axes[0]
    ax.plot(steps, losses, color="tab:blue", lw=0.4, alpha=0.25, label="single step")
    # running mean window 50 for visualisation
    win = 50
    sm = np.convolve(losses, np.ones(win) / win, mode="valid")
    ax.plot(steps[: len(sm)] + win // 2, sm, color="tab:blue", lw=1.2, alpha=0.9,
            label="running mean (50 step)")
    ax.errorbar(ep_mid, ep_mean, yerr=ep_std, color="tab:red", fmt="o-",
                markersize=5, lw=1.6, capsize=3, alpha=0.95,
                label=f"per-epoch mean ± std (epoch={EPOCH_LEN} steps)")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss\n(DeepMind weighted_mse_per_level, includes lat weights)")
    ax.set_title("v12 training loss (lat-weighted) — per-epoch mean reveals slow real decline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # right: zoomed per-epoch mean only
    ax = axes[1]
    ax.errorbar(np.arange(1, len(ep_mean) + 1), ep_mean, yerr=ep_std, color="tab:red",
                fmt="o-", markersize=6, lw=1.8, capsize=3, alpha=0.95)
    drop = ep_mean[0] - ep_mean[-1]
    drop_pct = drop / ep_mean[0] * 100
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-epoch mean lat-weighted training loss")
    ax.set_title(f"per-epoch view: epoch 1={ep_mean[0]:.4f} → "
                 f"epoch {len(ep_mean)}={ep_mean[-1]:.4f}\n"
                 f"total drop {drop:.4f} ({drop_pct:.2f}%)")
    ax.grid(alpha=0.3)
    ax.set_ylim(ep_mean[-1] - 0.005, ep_mean[0] + 0.005)

    fig.suptitle("v12 fast 20K — lat-weighted training loss", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")
    print(f"  total epochs: {len(ep_mean)}")
    print(f"  drop per epoch (mean): {(ep_mean[0]-ep_mean[-1])/(len(ep_mean)-1):.5f}")


if __name__ == "__main__":
    main()
