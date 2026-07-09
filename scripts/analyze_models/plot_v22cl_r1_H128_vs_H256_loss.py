"""Per-epoch mean training loss: res=1 v22cl H=128 (old) vs H=256 (new)
across the shared K's {4, 8, 12, 16, 20, 22}. One subplot per K.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS_PER_EPOCH = 424
KS = [4, 8, 12, 16, 20, 22]
ROOT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed")
OUT = Path("/home/lm8598/Weather_Global_experiments/results/2026-07-04-v22cl-r1-H128-vs-H256/plots")
OUT.mkdir(parents=True, exist_ok=True)


def load_train_log(path: Path) -> list[dict]:
    if not path.exists(): return []
    try:
        return json.load(open(path))
    except Exception:
        return []


def epoch_avg(entries: list[dict]) -> tuple[list[int], list[float]]:
    if not entries: return [], []
    buckets: dict[int, list[float]] = {}
    for e in entries:
        ep = (e["step"] - 1) // STEPS_PER_EPOCH + 1
        buckets.setdefault(ep, []).append(float(e["loss"]))
    xs = sorted(buckets)
    ys = [float(np.mean(buckets[e])) for e in xs]
    return xs, ys


fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True)
axes = axes.flatten()

def try_paths(root: Path, candidates: list[str]) -> list[dict]:
    for c in candidates:
        p = root / c / "train_log.json"
        if p.exists():
            entries = load_train_log(p)
            if entries: return entries
    return []


for i, K in enumerate(KS):
    ax = axes[i]
    for tag, candidates, color, marker in [
        ("H=128 (baseline)",
         [f"K{K}_sg_fresh_20k",
          f"K{K}_sg_fresh_20k/v22closed_sg_K{K}_fresh_20k"],
         "C0", "o"),
        ("H=256 (new)",
         [f"K{K}_H256_fresh_20k/v22closed_H256_K{K}_fresh_20k",
          f"K{K}_H256_fresh_20k"],
         "C3", "s"),
    ]:
        entries = try_paths(ROOT, candidates)
        xs, ys = epoch_avg(entries)
        if not xs:
            print(f"K={K}: {tag} → no data at {subdir}")
            continue
        ax.plot(xs, ys, "-", color=color, marker=marker, ms=4, lw=1.6,
                label=f"{tag} (last ep={xs[-1]}, loss={ys[-1]:.3f})")
        print(f"K={K}  {tag:<18s} epochs 1..{max(xs):3d}  final={ys[-1]:.3f}  min={min(ys):.3f}")
    ax.set_title(f"K={K}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    if i >= 3: ax.set_xlabel("epoch (= 424 steps)")
    if i % 3 == 0: ax.set_ylabel("mean training loss / epoch")

fig.suptitle("v22cl res=1 K-scan: H=128 (baseline, complete 20k) vs H=256 (new, in-progress)",
             fontsize=13)
plt.tight_layout()
out = OUT / "H128_vs_H256_loss_per_epoch.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nsaved {out}")
