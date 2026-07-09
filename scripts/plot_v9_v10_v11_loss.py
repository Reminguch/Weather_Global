"""Compare v9 / v10 / v11 training loss curves on one plot."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results")

RUNS = {
    "v9 frozen (Mamba only, 11M params)":
        RESULTS / "v9_asym_residual_mamba/v9_arm_10k/train_log.json",
    "v9 corrected (trainable=all, 11M)":
        RESULTS / "v9_asym_residual_mamba/v9_full_train_10k/train_log.json",
    "v10 (no processor, 6.7M)":
        RESULTS / "v10_grid_mesh_mamba/v10_gmm_smoke/train_log.json",
    "v11 (fixed geom + Mamba, 151K)":
        RESULTS / "v11_fixed_geom_mamba/v11_fix_smoke/train_log.json",
}

COLORS = {
    "v9 frozen (Mamba only, 11M params)": "tab:red",
    "v9 corrected (trainable=all, 11M)": "tab:orange",
    "v10 (no processor, 6.7M)": "tab:blue",
    "v11 (fixed geom + Mamba, 151K)": "tab:green",
}


def running_mean(x, w=20):
    if len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: zoom 0-500 (full overlap)
    # Right: full extent 0-9500
    for label, path in RUNS.items():
        if not path.exists():
            print(f"missing {path}, skipping")
            continue
        with open(path) as f:
            log = json.load(f)
        steps = np.array([e["step"] for e in log])
        losses = np.array([e["loss"] for e in log])
        color = COLORS[label]

        for ax, xmax in [(axes[0], 500), (axes[1], steps.max())]:
            mask = steps <= xmax
            ax.plot(steps[mask], losses[mask], alpha=0.2, color=color, lw=0.6)
            sm = running_mean(losses[mask], w=20)
            x_sm = steps[mask][:len(sm)] + 9
            ax.plot(x_sm, sm, color=color, lw=1.6, label=label)

    axes[0].set_xlim(0, 500)
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title("Zoom: steps 1-500 (smoke comparison)")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training loss (DeepMind weighted_mse_per_level)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].axhline(0.84, color="gray", lw=0.5, ls="--", alpha=0.5)
    axes[0].text(20, 0.84, "init ~0.84", fontsize=7, color="gray", va="bottom")

    axes[1].set_title("Full extent (10K steps, includes long runs)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("training loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].axhline(0.74, color="gray", lw=0.5, ls=":", alpha=0.5)
    axes[1].text(100, 0.74, "first trough ~0.74", fontsize=7, color="gray", va="bottom")
    axes[1].set_ylim(0.65, 1.0)

    fig.suptitle("v9 / v10 / v11 training loss: same loss space, different architectures",
                 fontsize=11)
    fig.tight_layout()

    out_path = Path(
        "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_v10_v11_loss_comparison.png"
    )
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
