#!/usr/bin/env python3
"""Two-lead spectra from the frozen 36-date CSV; no reanalysis or forecasts.

Portable reproduction: python plot_v4_spectra_day5_day10_20261114.py
    --data weighted_spectral_curves.csv --output .
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
NAME = "v4_spectra_day5_day10_20261114"
STEM = "fig7_smoothing_day5_day10_v4_matched36"
SOURCE = ROOT / "plots/analyze_models/data/resolution_eval/v4_appendix_e_matched36_20261114/weighted_spectral_curves.csv"
SOURCE_SHA256 = "7dd434bc0e09bb1f52dcd4e1e711af35c4101276df9c991969b7b93060721894"
MODELS = [("GC", "GraphCast", "#2864A5"),
          ("M1", "Carry/carry (M1)", "#D97720"),
          ("GC_filter", "GC + optimal filter", "#238B8E"),
          ("reset", "Reset/reset", "#9467BD")]
LEADS = (120, 240)
METRICS = (("weighted_amplitude_ratio", "Amplitude ratio"),
           ("weighted_coherence", "Coherence"))
DEGREES = np.arange(1, 180)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reciprocal(value):
    value = np.asarray(value, dtype=float)
    return np.divide(40030., value, out=np.full_like(value, np.inf), where=value != 0)


def spatial_axis(axis, *, top):
    axis.set_xscale("log")
    axis.set_xlim(1, 179)
    axis.set_xticks([1, 10, 100, 179], ["1", "10", "100", "179"])
    axis.set_xlabel("Total wavenumber ℓ")
    axis.grid(alpha=.18, linewidth=.5)
    if top:
        secondary = axis.secondary_xaxis("top", functions=(reciprocal, reciprocal))
        secondary.set_xticks([10000, 1000, 224], ["10,000", "1,000", "224"])
        secondary.set_xlabel("Wavelength (km)", labelpad=3)
        secondary.tick_params(labelsize=7, length=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SOURCE, help="Frozen weighted_spectral_curves.csv")
    parser.add_argument("--output", type=Path, help="Portable directory for figures, CSV and source; default uses project data/image folders")
    args = parser.parse_args()
    source = args.data.resolve()
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Input differs from the frozen 36-date weighted spectral CSV")
    with source.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if any(int(row["forecast_count"]) != 36 for row in all_rows):
        raise ValueError("Expected the same 36-date cohort for all curves")
    rows = [row for row in all_rows if int(row["lead_hours"]) in LEADS]
    lookup = {(int(row["lead_hours"]), row["model"], int(row["degree"])): row for row in rows}
    expected = {(lead, model, int(degree)) for lead in LEADS for model, _, _ in MODELS for degree in DEGREES}
    if len(rows) != 1432 or set(lookup) != expected:
        raise ValueError("Expected two leads, four models and 179 degrees without duplicates")
    values = {metric: np.asarray([[[float(lookup[lead, model, int(degree)][metric])
                                   for degree in DEGREES] for model, _, _ in MODELS] for lead in LEADS])
              for metric, _ in METRICS}
    if any(not np.isfinite(array).all() for array in values.values()):
        raise ValueError("Nonfinite spectral curve")
    images = (args.output or ROOT / "plots/analyze_models/images/resolution_eval" / NAME).resolve()
    data = (args.output or ROOT / "plots/analyze_models/data/resolution_eval" / NAME).resolve()
    images.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    limits = {}
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 4.9), sharey="row")
        for row, (metric, ylabel) in enumerate(METRICS):
            curves = values[metric]
            lower = min(-.025, float(curves.min()) - .025) if row == 1 else 0.
            upper = max(1.05, float(curves.max()) * 1.04)
            limits[metric] = [lower, upper]
            for col, lead in enumerate(LEADS):
                axis = axes[row, col]
                for mi, (_, label, color) in enumerate(MODELS):
                    style = {"color": color, "lw": 1.35, "label": label}
                    if row == 1:
                        style.update(ls=(5 if mi == 2 else 0, (5, 5)), dash_capstyle="butt")
                    line, = axis.plot(DEGREES, curves[col, mi], **style)
                    np.testing.assert_array_equal(line.get_ydata(), curves[col, mi])
                axis.axhline(1., color=".4", ls=":", lw=.6)
                axis.set_ylim(lower, upper)
                axis.set_ylabel(ylabel if col == 0 else "")
                spatial_axis(axis, top=row == 0)
                if row == 0:
                    axis.set_title(f"Day {lead // 24}", fontsize=8.5, pad=39)
        fig.text(.5, .995, "36 matched forecast dates", ha="center", va="top", fontsize=8)
        handles = [Line2D([], [], color=color, lw=1.5, label=label) for _, label, color in MODELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95),
                   ncol=2, frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.3)
        fig.subplots_adjust(left=.09, right=.965, bottom=.095, top=.74, hspace=.62, wspace=.25)
        outputs = []
        for suffix in ("pdf", "png"):
            path = images / f"{STEM}.{suffix}"
            fig.savefig(path, dpi=230)
            outputs.append({"path": str(path), "sha256": sha256(path)})
        plt.close(fig)
    for original in (source, Path(__file__).resolve()):
        if original != data / original.name:
            shutil.copy2(original, data / original.name)
    manifest = {
        "description": "Frozen day-5/day-10 amplitude and coherence; columns=lead, rows=metric",
        "cohort": {"count": 36, "start": "2023-01-01", "end": "2023-04-16", "cadence_days": 3},
        "models": [{"id": model, "label": label, "color": color} for model, label, color in MODELS],
        "lead_hours": list(LEADS), "degrees": [1, 179], "source_csv": source.name,
        "source_sha256": SOURCE_SHA256, "source_script": Path(__file__).name,
        "script_sha256": sha256(Path(__file__).resolve()), "figure_size_inches": [6.25, 4.9],
        "axes": 4, "curves": 16, "source_rows_selected": len(rows),
        "plotted_scalar_values_preserved": sum(array.size for array in values.values()),
        "shared_y_limits_per_metric": limits,
        "coherence_dashes": "GC/filter complementary (5,5) dashes with phases 0/5; butt caps; no markers or arrows",
        "metric_definition": "Saved variable/pressure-weighted per-channel amplitude ratios and coherence; no reanalysis",
        "new_forecasts_or_metric_computation": False, "figures": outputs,
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": outputs, "source_rows_selected": len(rows),
                      "plotted_scalar_values_preserved": manifest["plotted_scalar_values_preserved"]}, indent=2))


if __name__ == "__main__":
    main()
