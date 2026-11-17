#!/usr/bin/env python3
"""Summarize saved same-cohort MSE and power-mismatch reductions versus raw GC."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "v24_literature_spectral_matched36_20261114"
WINDOWS = ("rollout", "day5", "day10")
MODELS = (("GC_filtered", "GraphCast + optimal filter", "#238B8E"),
          ("M1", "Carry/carry (M1)", "#D97720"),
          ("R_matched", "Reset/reset", "#9467BD"))
FIELDS = (("loss_reduction_vs_raw_gc_pct", "exact_gc_loss", "Forecast accuracy", "GraphCast MSE reduction (%)"),
          ("power_mismatch_reduction_vs_raw_gc_pct", "power_mismatch_pct", "Spectral power fidelity", "Power-mismatch reduction (%)"))


def pin(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be one directory name")
    data = ROOT / "plots/analyze_models/data/resolution_eval" / args.name
    images = ROOT / "plots/analyze_models/images/resolution_eval" / args.name
    source = data / "comparison_scores.csv"
    summary_path = data / "summary.json"
    summary = json.loads(summary_path.read_text())
    rows = list(csv.DictReader(source.open(newline="")))
    lookup = {(row["window"], row["model"]): row for row in rows}
    expected = {(window, model) for window in WINDOWS for model in ("GC", *[item[0] for item in MODELS])}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Expected exactly four predictors in each of three windows")
    count = summary["count"]
    if count != len(set(summary["initialization_ids"])):
        raise ValueError("Invalid summary cohort")
    for window in WINDOWS:
        baseline = lookup[window, "GC"]
        for model in ("GC", *[item[0] for item in MODELS]):
            row = lookup[window, model]
            if int(row["forecast_count"]) != count or int(row["lead_count"]) != (40 if window == "rollout" else 1):
                raise ValueError("Mixed cohorts or leads")
            for reduction, score, _, _ in FIELDS:
                value, denominator = float(row[score]), float(baseline[score])
                if not np.isfinite([value, denominator, float(row[reduction])]).all() or denominator <= 0:
                    raise ValueError("Invalid summary score")
                np.testing.assert_allclose(float(row[reduction]), 100 * (1 - value / denominator), rtol=1e-12, atol=1e-12)
    sources = [pin(source), pin(summary_path), pin(Path(__file__).resolve())]
    panels = {}
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
        x, width = np.arange(3), .23
        for axis, (field, _, title, ylabel) in zip(axes, FIELDS):
            values = np.asarray([[float(lookup[window, model][field]) for window in WINDOWS]
                                 for model, _, _ in MODELS])
            panels[field] = {model: values[i].tolist() for i, (model, _, _) in enumerate(MODELS)}
            low, high = min(0., float(values.min())), max(0., float(values.max()))
            span = max(high - low, 1.)
            for mi, (_, label, color) in enumerate(MODELS):
                bars = axis.bar(x + (mi - 1) * width, values[mi], width=.205,
                                color=color, edgecolor="none", label=label, zorder=3)
                for wi, (bar, value) in enumerate(zip(bars, values[mi])):
                    offset = 5 if value >= 0 else -5
                    if value < 0 and mi == 1 and abs(value - values[2, wi]) < .06 * span:
                        offset = -18  # Separate adjacent labels at nearly equal endpoints.
                    axis.annotate(f"{value:+.2f}", (bar.get_x() + bar.get_width() / 2, value),
                                  xytext=(0, offset), textcoords="offset points",
                                  ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
            axis.axhline(0, color="#2864A5", lw=1.2, ls=(0, (4, 3)), zorder=4)
            axis.set(xticks=x, xticklabels=["Rollout\n6 h–10 days", "Day 5", "Day 10"],
                     ylabel=ylabel, ylim=(low - .13 * span, high + .17 * span), xlim=(-.55, 2.55))
            axis.set_title(title, fontsize=12, pad=12)
            axis.grid(axis="y", color=".90", lw=.7, zorder=0)
            axis.set_axisbelow(True)
        handles = [Line2D([], [], color="#2864A5", lw=1.2, ls=(0, (4, 3)), label="Raw GraphCast (0%)")]
        handles += [Patch(facecolor=color, label=label) for _, label, color in MODELS]
        fig.suptitle("Improvements relative to raw GraphCast", fontsize=15, y=.98)
        fig.text(.5, .915, "Higher is better within each panel", ha="center", fontsize=10, color=".3")
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .875), ncol=4,
                   frameon=False, fontsize=9.5, handlelength=2.3, columnspacing=1.5)
        dates = [datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").strftime("%d %b %Y").lstrip("0")
                 for stamp in (summary["initialization_ids"][0], summary["initialization_ids"][-1])]
        fig.text(.5, .095, f"{count} matched forecast dates · {dates[0]}–{dates[1]} · fixed-checkpoint point estimates",
                 ha="center", fontsize=9, color=".25")
        fig.text(.5, .055, "Reduction = 100 × (1 − model score / raw-GraphCast score). Negative values indicate deterioration.",
                 ha="center", fontsize=8.5, color=".3")
        fig.text(.5, .02, "The panels normalize different metrics; their percentages are not interchangeable.",
                 ha="center", fontsize=8.5, color=".3")
        fig.subplots_adjust(left=.07, right=.99, bottom=.235, top=.725, wspace=.24)
        outputs = []
        for suffix in ("pdf", "png"):
            path = images / f"improvement_summary_rollout_day5_day10.{suffix}"
            fig.savefig(path, dpi=190, bbox_inches="tight")
            outputs.append(pin(path))
        plt.close(fig)
    record = {"count": count, "initialization_ids": summary["initialization_ids"], "windows": WINDOWS,
              "reference": "GC raw, zero in both panels", "models": ["GC", *[item[0] for item in MODELS]],
              "formula": "100*(1-model_score/raw_GC_score), applied independently to each metric",
              "rollout": "40 saved lead losses pooled before the ratio; not mean per-lead improvement",
              "sign": "positive improvement; negative deterioration", "panels": panels,
              "source_and_formula_checks": "passed", "sources": sources, "figures": outputs,
              "uncertainty": "fixed-checkpoint point estimates; no added uncertainty calculations"}
    (data / "improvement_summary_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"count": count, "figures": [item["path"] for item in outputs]}))


if __name__ == "__main__":
    main()
