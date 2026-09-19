"""Plot saved exact GraphCast loss reduction by forecast lead."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.eval_json.read_bytes()
    result = json.loads(raw)
    if result["evaluation_status"] != "complete":
        raise ValueError("A complete evaluation is required")
    exact = result["original_graphcast_loss"]
    baseline = np.asarray(exact["baseline_per_step"], dtype=float)
    model = np.asarray(exact["full_per_step"], dtype=float)
    if baseline.shape != model.shape or baseline.shape != (result["target_steps"],):
        raise ValueError("Incomplete per-lead exact losses")
    if not np.isfinite([baseline, model]).all() or np.any(baseline <= 0):
        raise ValueError("Invalid exact losses")
    improvement = 100 * (1 - model / baseline)
    aggregate = 100 * (1 - model.sum() / baseline.sum())
    if not np.allclose(improvement, exact["improvement_pct_per_step"], rtol=1e-10, atol=1e-10):
        raise ValueError("Saved per-lead percentages disagree with exact loss ratios")
    if not np.isclose(aggregate, exact["improvement_pct_rollout"], rtol=1e-10, atol=1e-10):
        raise ValueError("Saved rollout percentage disagrees with aggregated exact losses")
    days = np.arange(1, len(model) + 1) / 4
    for directory in (args.data_dir, args.image_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows = [{"lead_hours": 6 * (index + 1), "lead_days": float(day),
             "baseline_exact_graphcast_loss": float(baseline[index]),
             "model_exact_graphcast_loss": float(model[index]),
             "exact_graphcast_loss_reduction_pct": float(improvement[index])}
            for index, day in enumerate(days)]
    with (args.data_dir / "rollout_improvement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": str(args.eval_json.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint": result["ckpt"], "label": args.label,
        "evaluated_samples": result["evaluated_samples"], "resolution": result["resolution"],
        "eval_mode": result["eval_mode"], "residual_state_init": result["residual_state_init"],
        "metric": exact["definition"], "per_lead_formula": "100 * (1 - model_loss / baseline_loss)",
        "rollout_formula": "100 * (1 - sum(model_loss) / sum(baseline_loss))",
        "rollout_improvement_pct": float(aggregate),
        "selected_leads": [row for row in rows if row["lead_days"] in (1, 2, 4, 5, 7, 10)],
        "minimum": rows[int(np.argmin(improvement))], "maximum": rows[int(np.argmax(improvement))],
    }
    (args.data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ink, blue, red = "#243849", "#16799C", "#BE5A52"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink,
                         "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 5.9))
    fig.subplots_adjust(left=.105, right=.96, top=.77, bottom=.20)
    fig.text(.105, .95, "Width-128 Mamba: improvement over the rollout", fontsize=19, weight="bold")
    fig.text(.105, .895, args.label, fontsize=12)
    protocol = (f"{result['resolution']:g}° resolution  ·  {result['evaluated_samples']} cold starts  ·  "
                "zero initial memory  ·  corrected-feedback rollout")
    fig.text(.105, .85, protocol, fontsize=10, color="#596C79")
    if result["eval_mode"] != "cold_full" or result["residual_state_init"] != "zero":
        raise ValueError("This figure's protocol label requires cold_full and zero state")
    ax.axhline(0, color="#8998A1", lw=1, linestyle="--")
    ax.fill_between(days, 0, improvement, where=improvement >= 0, interpolate=True, color=blue, alpha=.10)
    ax.fill_between(days, 0, improvement, where=improvement < 0, interpolate=True, color=red, alpha=.20)
    ax.plot(days, improvement, color=blue, lw=2.4)
    ax.scatter(days, improvement, color=blue, s=12, zorder=3)
    for day, offset, align in ((2, (2, -25), "center"), (5, (-8, 16), "right"),
                               (7, (-8, 16), "right"), (10, (-6, 14), "right")):
        index = int(day * 4) - 1
        if index >= len(days):
            continue
        value = improvement[index]
        ax.scatter([day], [value], s=35, color=blue, zorder=4)
        ax.annotate(f"{value:+.2f}%", (day, value), xytext=offset, textcoords="offset points",
                    ha=align, fontsize=11, weight="bold", color=red if value < 0 else blue)
    ax.text(.025, .92, f"Whole-rollout loss reduction: {aggregate:.2f}%",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F0F4F6", "edgecolor": "none"})
    ax.set(xlim=(0, float(days[-1]) + .2), ylim=(min(-3.2, improvement.min() - 2), improvement.max() + 3),
           xlabel="Forecast lead (days)", ylabel="Exact GraphCast loss reduction (%)")
    ax.set_xticks(np.arange(0, int(days[-1]) + 1))
    ax.grid(axis="y", color="#DFE6EA", lw=.7)
    ax.set_axisbelow(True)
    fig.text(.105, .095, "Positive = lower error than frozen GraphCast; negative = higher error.", fontsize=10)
    fig.text(.105, .052, "Exact normalized, latitude- and pressure-weighted MSE. Whole-rollout value uses aggregated losses.",
             fontsize=9, color="#596C79")
    for suffix in ("png", "pdf"):
        fig.savefig(args.image_dir / f"rollout_improvement.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
