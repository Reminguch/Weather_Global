"""v13 train loss + eval improvement on the SAME plot (twin y-axes)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_LEN = 364
RES = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results"
TRAIN_LOG = f"{RES}/v13_v9arch_fast/v13_20k/train_log.json"
EVAL_DIR = f"{RES}/v13_v9arch_fast/v13_20k/eval_results"
OUT = f"{RES}/v13_combined_curves.png"


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

    fig, ax_tr = plt.subplots(figsize=(12, 6))

    # Train loss on left y
    win = 50
    sm = np.convolve(losses, np.ones(win) / win, mode="valid")
    ax_tr.plot(steps[:len(sm)] + win // 2, sm, color="tab:blue", lw=0.6, alpha=0.3)
    ax_tr.errorbar(mid, m, yerr=s, color="tab:blue", fmt="o-", markersize=4,
                   lw=1.8, capsize=2, alpha=0.95,
                   label="train loss per-epoch ± std (left axis)")
    ax_tr.set_xlabel("training step")
    ax_tr.set_ylabel("train loss (DeepMind weighted_mse_per_level)", color="tab:blue")
    ax_tr.tick_params(axis="y", labelcolor="tab:blue")
    ax_tr.grid(alpha=0.3)
    ax_tr.set_ylim(0.74, 0.86)

    # Eval improvement on right y
    ax_ev = ax_tr.twinx()
    ax_ev.plot(es, ems, "o-", color="tab:red", lw=2, markersize=8, alpha=0.95,
               label="eval mean improvement % (right axis)")
    for x, y, n in zip(es, ems, ens):
        ax_ev.annotate(f"{n}/{en_total}", (x, y), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=7, color="tab:red")
    ax_ev.set_ylabel("eval mean improvement % over baseline GC1\n"
                     "(83 channels lat-weighted RMSE; positive=better)",
                     color="tab:red")
    ax_ev.tick_params(axis="y", labelcolor="tab:red")
    ax_ev.set_ylim(0, 1.5)

    # Combine legends
    lines1, labels1 = ax_tr.get_legend_handles_labels()
    lines2, labels2 = ax_ev.get_legend_handles_labels()
    ax_tr.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    # Mark the peak eval point
    if ems:
        best_i = int(np.argmax(ems))
        ax_ev.axvline(es[best_i], color="tab:red", lw=0.8, ls="--", alpha=0.4)
        ax_ev.annotate(f"peak eval +{ems[best_i]:.3f}% at step {es[best_i]}",
                       xy=(es[best_i], ems[best_i]),
                       xytext=(es[best_i] + 1500, ems[best_i] - 0.05),
                       fontsize=9, color="tab:red",
                       arrowprops=dict(color="tab:red", lw=0.5, arrowstyle="->"))

    ax_tr.set_title(f"v13 = v9 arch (11M, DeepMind init) + fast pipeline\n"
                    f"Train loss keeps dropping {m[0]:.3f}→{m[-1]:.3f}, "
                    f"but eval peaks at step {es[int(np.argmax(ems))] if ems else '?'} "
                    f"and overfits later (classic signature)",
                    fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
