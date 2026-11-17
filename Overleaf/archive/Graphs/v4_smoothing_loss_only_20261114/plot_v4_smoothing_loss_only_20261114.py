#!/usr/bin/env python3
"""Plot the frozen 36-date exact losses on one axis, without the reduction panel.

Portable reproduction: python plot_v4_smoothing_loss_only_20261114.py
    --data loss_by_lead.csv --output .
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "v4_smoothing_loss_only_20261114"
STEM = "fig12_smoothing_loss_v4_matched36"
SOURCE = ROOT / "plots/analyze_models/data/resolution_eval/v4_appendix_e_matched36_20261114/loss_by_lead.csv"
SOURCE_SHA256 = "f40d308d4f7d23a1658f3f4ffc348d344f6592fef5cc79344f64401bdc6b5e58"
MODELS = [("GC", "GraphCast", "#2864A5"),
          ("M1", "Carry/carry (M1)", "#D97720"),
          ("GC_filter", "GC + optimal filter", "#238B8E"),
          ("reset", "Reset/reset", "#9467BD")]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SOURCE, help="Frozen loss_by_lead.csv")
    parser.add_argument("--output", type=Path, help="Portable directory for figures, data and source; default uses project data/image folders")
    args = parser.parse_args()
    source = args.data.resolve()
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Input differs from the frozen 36-date exact-loss CSV")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["model"], int(row["lead_hours"])): row for row in rows}
    leads = np.arange(6, 241, 6)
    expected = {(model, int(lead)) for model, _, _ in MODELS for lead in leads}
    if len(rows) != 160 or set(lookup) != expected or any(int(r["forecast_count"]) != 36 for r in rows):
        raise ValueError("Expected four models and forty leads on the same 36 dates")
    values = np.asarray([[float(lookup[model, int(lead)]["exact_gc_loss"])
                          for model, _, _ in MODELS] for lead in leads])
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid exact loss")
    images = args.output or ROOT / "plots/analyze_models/images/resolution_eval" / NAME
    data = args.output or ROOT / "plots/analyze_models/data/resolution_eval" / NAME
    images, data = images.resolve(), data.resolve()
    images.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axis = plt.subplots(figsize=(6.25, 2.65))
        for index, (_, label, color) in enumerate(MODELS):
            line, = axis.plot(leads / 24., values[:, index], color=color, lw=1.35, label=label)
            np.testing.assert_array_equal(line.get_ydata(), values[:, index])
        axis.axvline(5, color=".5", ls=":", lw=.7)
        axis.set(xlim=(.25, 10), xlabel="Forecast lead (days)", xticks=[1, 3, 5, 7, 10],
                 ylabel="Exact GraphCast loss")
        axis.grid(alpha=.18, linewidth=.5)
        fig.text(.5, .995, "36 matched forecast dates", ha="center", va="top", fontsize=8)
        handles = [Line2D([], [], color=color, lw=1.5, label=label) for _, label, color in MODELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95),
                   ncol=2, frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.3)
        fig.subplots_adjust(left=.095, right=.965, bottom=.19, top=.745)
        outputs = []
        for suffix in ("pdf", "png"):
            path = images / f"{STEM}.{suffix}"
            fig.savefig(path, dpi=230)
            outputs.append({"path": str(path), "sha256": sha256(path)})
        plt.close(fig)
    for original in (source, Path(__file__).resolve()):
        target = data / original.name
        if original != target:
            shutil.copy2(original, target)
    manifest = {
        "description": "Single-panel exact-loss figure; the relative-reduction panel is omitted",
        "cohort": {"count": 36, "start": "2023-01-01", "end": "2023-04-16", "cadence_days": 3},
        "models": [{"id": model, "label": label, "color": color} for model, label, color in MODELS],
        "lead_hours": leads.tolist(), "source_csv": source.name, "source_sha256": SOURCE_SHA256,
        "source_script": Path(__file__).name, "script_sha256": sha256(Path(__file__).resolve()),
        "figure_size_inches": [6.25, 2.65], "axes": 1, "forecast_curves": 4,
        "exact_loss_values_preserved": int(values.size), "five_day_marker": True,
        "new_forecasts_or_calculations": False, "figures": outputs,
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": outputs, "exact_loss_values_preserved": int(values.size)}, indent=2))


if __name__ == "__main__":
    main()
