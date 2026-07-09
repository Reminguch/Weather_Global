"""Per-architecture train loss (per-epoch) and eval mean improvement curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_LEN = 364

RUNS = {
    "v9 frozen (Mamba only, 11M base)":
        ("tab:red",
         "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_asym_residual_mamba/v9_arm_10k/train_log.json",
         None),
    "v9 corrected (trainable=all, 11M)":
        ("tab:orange",
         "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_asym_residual_mamba/v9_full_train_10k/train_log.json",
         "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_asym_residual_mamba/v9_full_train_10k/eval_results"),
    "v12 (fixed geom, 151K)":
        ("tab:green",
         "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_fast_20k/train_log.json",
         "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_fast_20k/eval_results"),
    "v12 bigmamba (fixed geom + 4x Mamba, 520K)":
        ("tab:purple",
         "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_bigmamba_20k/train_log.json",
         "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_bigmamba_20k/eval_results"),
}


def per_epoch_mean(steps, losses, ep=EPOCH_LEN):
    mid, m, s = [], [], []
    for i in range(int(steps.max()) // ep + 1):
        s_lo, s_hi = i * ep + 1, (i + 1) * ep
        mask = (steps >= s_lo) & (steps <= s_hi)
        if mask.sum() < 50:
            continue
        mid.append((s_lo + s_hi) / 2)
        m.append(losses[mask].mean())
        s.append(losses[mask].std())
    return np.asarray(mid), np.asarray(m), np.asarray(s)


def load_eval_curve(eval_dir: str | None):
    if eval_dir is None:
        return [], []
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        return [], []
    points = []
    for f in eval_dir.glob("eval_step*.json"):
        try:
            step = int(f.stem.replace("eval_step", ""))
            d = json.loads(f.read_text())
            rels = [v["rel_change_pct"] for v in d["per_channel"].values()]
            mean_improve = -sum(rels) / len(rels)
            points.append((step, mean_improve, d.get("channels_improved", 0)))
        except Exception:
            continue
    points.sort()
    if not points:
        return [], []
    steps = [p[0] for p in points]
    means = [p[1] for p in points]
    return steps, means


def main():
    fig, (ax_tr, ax_ev) = plt.subplots(1, 2, figsize=(14, 5))

    for label, (color, log_path, eval_dir) in RUNS.items():
        if not Path(log_path).exists():
            print(f"missing {log_path}")
            continue
        log = json.loads(Path(log_path).read_text())
        steps = np.asarray([e["step"] for e in log])
        losses = np.asarray([e["loss"] for e in log])
        ep_mid, ep_mean, ep_std = per_epoch_mean(steps, losses)
        if len(ep_mean):
            ax_tr.errorbar(ep_mid, ep_mean, yerr=ep_std, color=color, fmt="o-",
                           markersize=4, lw=1.4, capsize=2, alpha=0.85, label=label)

        es, ems = load_eval_curve(eval_dir)
        if es:
            ax_ev.plot(es, ems, "o-", color=color, lw=1.6, markersize=6,
                       alpha=0.9, label=label)

    # v3 reference (single point)
    ax_ev.plot(4000, 0.643, "*", markersize=20, color="black",
               label="v3 step 4000 (reference)")

    ax_tr.set_xlabel("training step")
    ax_tr.set_ylabel("per-epoch mean train loss\n(DeepMind weighted_mse_per_level, lat-weighted)")
    ax_tr.set_title("Train loss (per-epoch mean ± std)")
    ax_tr.grid(alpha=0.3)
    ax_tr.legend(fontsize=8, loc="upper right")
    ax_tr.set_ylim(0.79, 0.83)

    ax_ev.set_xlabel("training step")
    ax_ev.set_ylabel("mean improvement % over baseline GC1\n(83 channels, lat-weighted RMSE, positive=better)")
    ax_ev.set_title("Eval mean improvement vs training step")
    ax_ev.grid(alpha=0.3)
    ax_ev.legend(fontsize=8, loc="upper right")
    ax_ev.axhline(0, color="black", lw=0.5, alpha=0.3)

    fig.suptitle("Train loss vs eval improvement: v9 corrected (11M, DeepMind init) is best",
                 fontsize=11)
    fig.tight_layout()

    out = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v9_v12_train_eval_curves.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
