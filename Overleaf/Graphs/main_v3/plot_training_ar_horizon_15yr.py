#!/usr/bin/env python3
"""Rebuild the closed-loop horizon figure from its frozen plotted values.

Run from the repository root after sourcing scripts/graphcast_env.sh.
The CSV preserves the original seven curves, including the K=1 legend label
for the ar_tail_K=0 run. The adjacent sources JSON records their provenance.
No checkpoint evaluation is performed.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/main_v3_horizon_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STEM = HERE / "training_ar_horizon_15yr_closedloop"
sys.path.insert(0, str(HERE.parent))
from paper_style import ANNOTATION_SIZE, apply_style


def main():
    provenance = json.loads(STEM.with_suffix(".sources.json").read_text())
    csv_path = STEM.with_suffix(".csv")
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == provenance["plotted_csv_sha256"]
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    ks = np.unique(data["ar_tail_K"]).astype(int)
    assert ks.tolist() == [0, 4, 8, 10, 12, 16, 20]
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    fig.subplots_adjust(left=.15, right=.975, top=.90, bottom=.18)
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(ks):
        rows = data[data["ar_tail_K"] == k]
        np.testing.assert_array_equal(rows["lead_hours"], np.arange(1, 41) * 6)
        np.testing.assert_array_equal(rows["legend_K"], 1 if k == 0 else k)
        assert np.isfinite(rows["reduction_pct"]).all()
        ax.plot(rows["lead_hours"] / 24, rows["reduction_pct"],
                color=cmap(i / (len(ks) - 1) * .92),
                lw=1.5, marker="o", ms=2.2,
                label=f"$K={int(rows['legend_K'][0])}$")
    ax.axhline(0, color="#7f7f7f", lw=.9, zorder=0)
    training_tail_days = 20 * 6 / 24
    ax.axvline(training_tail_days, color="0.25", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(training_tail_days, .97, r"$K=20$: 120 h (5 days)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=ANNOTATION_SIZE, color="0.25",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.5})
    ax.grid(color="#ececec", lw=.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.set_ylim(-12, 28)
    ax.set_yticks(np.arange(-10, 26, 5))
    ax.set_xlabel("Forecast lead time (days)")
    ax.set_ylabel("GraphCast loss reduction (%)")
    ax.legend(ncol=2, loc="upper left", frameon=True, framealpha=.9,
              edgecolor="#dddddd", handlelength=2.2)
    fig.savefig(STEM.with_suffix(".pdf"), metadata={
        "Title": "Fifteen-year closed-loop training-horizon sweep",
        "Subject": "Original seven curves; approximate paper-formula weighted MSE reduction. Vertical dashed line: K=20 training tail, 120 h = 5 days.",
        "Creator": Path(__file__).name + "; frozen saved evaluation curves only",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(STEM.with_suffix(".png"), dpi=190)
    plt.close(fig)
    print("Wrote", STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
