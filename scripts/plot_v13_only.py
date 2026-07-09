"""v13 only: train loss (per-epoch + raw) and eval mean improvement curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_LEN = 364
RES = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results"
TRAIN_LOG = f"{RES}/v13_v9arch_fast/v13_20k/train_log.json"
EVAL_DIR = f"{RES}/v13_v9arch_fast/v13_20k/eval_results"
OUT = f"{RES}/v13_only_curves.png"


def main():
    log = json.loads(Path(TRAIN_LOG).read_text())
    steps = np.asarray([e["step"] for e in log])
    losses = np.asarray([e["loss"] for e in log])

    # per-epoch
    mid, m, s = [], [], []
    for i in range(int(steps.max()) // EPOCH_LEN + 1):
        s_lo, s_hi = i * EPOCH_LEN + 1, (i + 1) * EPOCH_LEN
        mask = (steps >= s_lo) & (steps <= s_hi)
        if mask.sum() < 50:
            continue
        mid.append((s_lo + s_hi) / 2)
        m.append(losses[mask].mean())
        s.append(losses[mask].std())
    mid, m, s = np.asarray(mid), np.asarray(m), np.asarray(s)

    # eval
    eval_pts = []
    for f in Path(EVAL_DIR).glob("eval_step*.json"):
        try:
            step = int(f.stem.replace("eval_step", ""))
            d = json.loads(f.read_text())
            rels = [v["rel_change_pct"] for v in d["per_channel"].values()]
            mean_imp = -sum(rels) / len(rels)
            n_chan = d.get("channels_improved", 0)
            n_total = d.get("channels_total", 83)
            eval_pts.append((step, mean_imp, n_chan, n_total))
        except Exception:
            continue
    eval_pts.sort()
    es = [p[0] for p in eval_pts]
    ems = [p[1] for p in eval_pts]
    ens = [p[2] for p in eval_pts]
    en_total = eval_pts[0][3] if eval_pts else 83

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: train loss
    ax = axes[0]
    win = 50
    sm = np.convolve(losses, np.ones(win) / win, mode="valid")
    ax.plot(steps[:len(sm)] + win // 2, sm, color="tab:blue", lw=0.8, alpha=0.4,
            label=f"50-step moving avg")
    ax.errorbar(mid, m, yerr=s, color="tab:blue", fmt="o-", markersize=4,
                lw=1.8, capsize=2, alpha=0.95,
                label=f"per-epoch mean ± std (epoch={EPOCH_LEN} steps)")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss\n(DeepMind weighted_mse_per_level, lat-weighted)")
    ax.set_title(f"v13 train loss: epoch 1={m[0]:.4f} → epoch {len(m)}={m[-1]:.4f}\n"
                 f"drop {m[0]-m[-1]:.4f} ({(m[0]-m[-1])/m[0]*100:.1f}%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0.7, 0.9)

    # Right: eval
    ax = axes[1]
    if es:
        ax.plot(es, ems, "o-", color="tab:blue", lw=2, markersize=8, alpha=0.95,
                label="v13 mean improvement")
        for x, y, n in zip(es, ems, ens):
            ax.annotate(f"{n}/{en_total}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="tab:blue")

    # Reference baselines
    ax.axhline(1.157, color="tab:orange", lw=1, ls="--", alpha=0.6,
               label="v9 corrected step 1000 (+1.157%)")
    ax.axhline(1.235, color="tab:orange", lw=1, ls=":", alpha=0.6,
               label="v9 corrected step 2000 (+1.235%)")
    ax.axhline(0.643, color="black", lw=1, ls="--", alpha=0.6,
               label="v3 step 4000 (+0.643%)")
    ax.axhline(0.463, color="tab:green", lw=1, ls="--", alpha=0.6,
               label="v12 best step 3000 (+0.463%)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean improvement % over baseline GC1\n(83 channels lat-weighted RMSE; positive=better)")
    title_n = f", best={max(ems):.3f}%" if ems else ""
    ax.set_title(f"v13 eval ({len(es)} ckpts evaluated{title_n})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(f"v13 = v9 architecture (11M, DeepMind init) + v12 fast pipeline",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")
    print(f"v13 train: epoch 1={m[0]:.4f} → epoch {len(m)}={m[-1]:.4f}")
    if es:
        print(f"v13 eval points: {list(zip(es, [round(e,3) for e in ems]))}")
        best_idx = int(np.argmax(ems))
        print(f"v13 BEST: step {es[best_idx]}: {ems[best_idx]:+.3f}% ({ens[best_idx]}/{en_total} channels)")


if __name__ == "__main__":
    main()
