"""All eval curves overlaid: v9-corrected / v11 / v12 / v12-bigmamba / v13."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RES = "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results"

RUNS = [
    ("v9 corrected (11M, DeepMind init, slow pipeline)", "tab:orange",
     f"{RES}/v9_asym_residual_mamba/v9_full_train_10k/eval_results"),
    ("v11 (151K, fixed geom)", "tab:olive",
     f"{RES}/v11_fixed_geom_mamba/v11_fix_10k/eval_results"),
    ("v12 (151K, fast pipeline)", "tab:green",
     f"{RES}/v12_fast/v12_fast_20k/eval_results"),
    ("v12 bigmamba (520K, 4× Mamba)", "tab:purple",
     f"{RES}/v12_fast/v12_bigmamba_20k/eval_results"),
    ("v13 (11M, v9 arch + fast pipeline)", "tab:blue",
     f"{RES}/v13_v9arch_fast/v13_20k/eval_results"),
]


def load_curve(eval_dir):
    pts = []
    for f in Path(eval_dir).glob("eval_step*.json"):
        try:
            step = int(f.stem.replace("eval_step", ""))
            d = json.loads(f.read_text())
            rels = [v["rel_change_pct"] for v in d["per_channel"].values()]
            mean_imp = -sum(rels) / len(rels)
            pts.append((step, mean_imp, d.get("channels_improved", 0),
                        d.get("channels_total", 83)))
        except Exception:
            continue
    pts.sort()
    return pts


def main():
    fig, ax = plt.subplots(figsize=(13, 6))

    for label, color, eval_dir in RUNS:
        if not Path(eval_dir).exists():
            print(f"missing {eval_dir}"); continue
        pts = load_curve(eval_dir)
        if not pts:
            print(f"empty {eval_dir}"); continue
        es = [p[0] for p in pts]
        ems = [p[1] for p in pts]
        ax.plot(es, ems, "o-", color=color, lw=1.8, markersize=6, alpha=0.9, label=label)
        # Mark peak
        bi = int(np.argmax(ems))
        ax.plot(es[bi], ems[bi], "*", markersize=18, color=color,
                markeredgecolor="black", markeredgewidth=0.8, zorder=10)
        print(f"{label[:50]:<50}  best step={es[bi]:>5}  +{ems[bi]:.3f}%  ({pts[bi][2]}/{pts[bi][3]})")

    # v3 reference
    ax.plot(4000, 0.643, "*", markersize=22, color="black", zorder=10,
            label="v3 step 4000 (reference)")
    ax.text(4400, 0.643, "v3 +0.643%", fontsize=9, va="center")

    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean improvement % over baseline GC1\n(83 channels, lat-weighted RMSE; positive=better)")
    ax.set_title("Eval mean improvement across all v9–v13 architectures\n"
                 "Same val set (32 samples, val_year=2022, fixed seed). Stars = peak per run.",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(left=0)

    fig.tight_layout()
    out = Path(f"{RES}/all_eval_curves.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
