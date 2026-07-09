"""v13 train + eval curves vs all reference baselines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_LEN = 364
RES = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results"

RUNS = {
    "v9 frozen (Mamba only, 11M base, slow pipeline)":
        ("tab:red", f"{RES}/v9_asym_residual_mamba/v9_arm_10k/train_log.json", None),
    "v9 corrected (trainable=all, 11M, slow pipeline)":
        ("tab:orange", f"{RES}/v9_asym_residual_mamba/v9_full_train_10k/train_log.json",
         f"{RES}/v9_asym_residual_mamba/v9_full_train_10k/eval_results"),
    "v12 (fixed geom, 151K)":
        ("tab:green", f"{RES}/v12_fast/v12_fast_20k/train_log.json",
         f"{RES}/v12_fast/v12_fast_20k/eval_results"),
    "v13 (v9 arch + fast pipeline, 11M)":
        ("tab:blue", f"{RES}/v13_v9arch_fast/v13_20k/train_log.json",
         f"{RES}/v13_v9arch_fast/v13_20k/eval_results"),
}


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
    return np.asarray(mid), np.asarray(m), np.asarray(s)


def load_eval(eval_dir):
    if eval_dir is None or not Path(eval_dir).exists():
        return [], []
    pts = []
    for f in Path(eval_dir).glob("eval_step*.json"):
        try:
            step = int(f.stem.replace("eval_step", ""))
            d = json.loads(f.read_text())
            rels = [v["rel_change_pct"] for v in d["per_channel"].values()]
            mean_imp = -sum(rels) / len(rels)
            pts.append((step, mean_imp))
        except Exception:
            continue
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    fig, (ax_tr, ax_ev) = plt.subplots(1, 2, figsize=(15, 5))
    for label, (color, log_path, eval_dir) in RUNS.items():
        if not Path(log_path).exists():
            continue
        log = json.loads(Path(log_path).read_text())
        mid, m, s = per_epoch(log)
        if len(m):
            ax_tr.errorbar(mid, m, yerr=s, color=color, fmt="o-",
                           markersize=4, lw=1.6, capsize=2, alpha=0.85, label=label)
        es, ems = load_eval(eval_dir)
        if es:
            ax_ev.plot(es, ems, "o-", color=color, lw=1.8, markersize=7,
                       alpha=0.9, label=label)

    # v3 reference
    ax_ev.plot(4000, 0.643, "*", markersize=22, color="black",
               label="v3 step 4000 (reference)")

    ax_tr.set_xlabel("training step")
    ax_tr.set_ylabel("per-epoch mean train loss\n(DeepMind weighted_mse_per_level)")
    ax_tr.set_title("Train loss (per-epoch mean ± std)")
    ax_tr.grid(alpha=0.3)
    ax_tr.legend(fontsize=8, loc="upper right")
    ax_tr.set_ylim(0.77, 0.83)

    ax_ev.set_xlabel("training step")
    ax_ev.set_ylabel("mean improvement % over baseline GC1\n(83 channels lat-weighted, positive=better)")
    ax_ev.set_title("Eval mean improvement vs training step")
    ax_ev.grid(alpha=0.3)
    ax_ev.legend(fontsize=8, loc="upper right")
    ax_ev.axhline(0, color="black", lw=0.5, alpha=0.3)

    fig.suptitle("v13 = v9 arch + fast pipeline: train loss drops faster, eval expected to dominate",
                 fontsize=11)
    fig.tight_layout()
    out = Path(f"{RES}/v13_train_eval_curves.png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
