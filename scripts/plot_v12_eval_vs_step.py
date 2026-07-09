"""Plot v12 eval RMSE improvement % per variable vs training step."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_fast/v12_fast_20k/eval_results"
)
OUT_PATH = Path(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v12_eval_vs_step.png"
)


def load_eval(step: int) -> dict:
    path = EVAL_DIR / f"eval_step{step}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    steps = list(range(1000, 21000, 1000))
    evals = {}
    for s in steps:
        d = load_eval(s)
        if d is not None:
            evals[s] = d
    if not evals:
        print("no eval results"); return

    # Sign convention: improvement = -rel_change_pct (so positive = better)
    # collect per-variable series
    var_names = sorted(set().union(*[set(e["per_variable"].keys()) for e in evals.values()]))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-variable improvement %
    for var in var_names:
        ys = [-evals[s]["per_variable"][var]["rel_change_pct"] for s in steps if var in evals[s]["per_variable"]]
        xs = [s for s in steps if var in evals[s]["per_variable"]]
        ax_left.plot(xs, ys, "o-", lw=1.5, markersize=6, label=var, alpha=0.85)

    ax_left.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax_left.set_xlabel("training step")
    ax_left.set_ylabel("eval RMSE improvement % (positive = better)")
    ax_left.set_title("v12 lead-1 eval per-variable improvement vs training step")
    ax_left.grid(alpha=0.3)
    ax_left.legend(fontsize=7, loc="best", ncol=2)

    # Right: channels improved fraction
    chan_xs = list(evals.keys())
    chan_ys = [evals[s]["channels_improved_pct"] for s in chan_xs]
    chan_n = [evals[s]["channels_improved"] for s in chan_xs]
    chan_total = evals[chan_xs[0]]["channels_total"]
    bars = ax_right.bar([str(s) for s in chan_xs], chan_ys, color="tab:blue", alpha=0.7)
    for b, n in zip(bars, chan_n):
        ax_right.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                      f"{n}/{chan_total}", ha="center", fontsize=10, fontweight="bold")
    ax_right.set_ylabel("channels improved %")
    ax_right.set_xlabel("training step")
    ax_right.set_title(f"v12: channels with improved RMSE / {chan_total}")
    ax_right.set_ylim(80, 100)
    ax_right.grid(alpha=0.3, axis="y")
    ax_right.axhline(50, color="black", lw=0.5, alpha=0.3, ls="--")

    fig.suptitle("v12 eval (32 val samples, 2022) — generalization peaks ~step 5000 then degrades",
                 fontsize=11)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
