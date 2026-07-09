"""Per-epoch mean loss for v9 / v10 / v11 (removes seasonal/data noise)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results")
EPOCH = 368  # 92 segments × 4 chunks/segment

RUNS = {
    "v9 frozen (Mamba only, 11M)":
        ("tab:red",
         RESULTS / "v9_asym_residual_mamba/v9_arm_10k/train_log.json"),
    "v9 corrected (trainable=all, 11M)":
        ("tab:orange",
         RESULTS / "v9_asym_residual_mamba/v9_full_train_10k/train_log.json"),
    "v10 (no processor, 6.7M)":
        ("tab:blue",
         RESULTS / "v10_grid_mesh_mamba/v10_gmm_smoke/train_log.json"),
    "v11 (fixed geom, 151K)":
        ("tab:green",
         RESULTS / "v11_fixed_geom_mamba/v11_fix_smoke/train_log.json"),
    "v11 (fixed geom, 151K) - 10K run":
        ("tab:cyan",
         RESULTS / "v11_fixed_geom_mamba/v11_fix_10k/train_log.json"),
}


def per_epoch_mean(steps, losses, epoch_len=EPOCH):
    means, std, mid = [], [], []
    n_ep = int(steps.max()) // epoch_len + 1
    for i in range(n_ep):
        s_lo = i * epoch_len + 1
        s_hi = (i + 1) * epoch_len
        mask = (steps >= s_lo) & (steps <= s_hi)
        if mask.sum() < 50:
            continue
        means.append(losses[mask].mean())
        std.append(losses[mask].std())
        mid.append((s_lo + s_hi) / 2)
    return np.array(mid), np.array(means), np.array(std)


def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (color, path) in RUNS.items():
        if not path.exists():
            print(f"missing {path}, skipping")
            continue
        with open(path) as f:
            log = json.load(f)
        steps = np.array([e["step"] for e in log])
        losses = np.array([e["loss"] for e in log])
        mid, means, std = per_epoch_mean(steps, losses)
        if len(means) == 0:
            continue
        ax.errorbar(mid, means, yerr=std, color=color, fmt="o-",
                    markersize=4, lw=1.4, capsize=3, alpha=0.85,
                    label=f"{label} (n_epochs={len(means)})")

    ax.set_xlabel("training step (mid-epoch)")
    ax.set_ylabel("per-epoch mean loss (368 steps/epoch)\nerror bars = within-epoch std")
    ax.set_title("Per-epoch mean loss strips out 0.04 within-epoch noise.\n"
                 "Real trend: ~0.001-0.005 drop per epoch.")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0.78, 0.86)
    fig.tight_layout()

    out_path = RESULTS / "v9_v10_v11_per_epoch_loss.png"
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
