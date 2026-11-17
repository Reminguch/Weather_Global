#!/usr/bin/env python3
"""Relabel v5 figures from frozen exports, retaining every plotted value.

Reuses the existing v4 layouts. No evaluation, metric calculation, bootstrap
resampling, or historical output writes are performed. Original-label and
M1-label renders are checked against each other before v5 PDFs are published.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pandas as pd

import plot_v4_memory_two_policies_20261114 as memory
import plot_v4_spectra_day5_day10_20261114 as spectra
import plot_v4_smoothing_loss_only_20261114 as loss
import plot_v4_appendix_e_matched36_20261114 as appendix

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "plots/analyze_models/data/resolution_eval/v4_appendix_e_matched36_20261114"
MODEL_IDS = ("GC", "M1", "GC_filter", "reset")
NAMES = {
    "fig6_memory_policies_v4_two_policies": "fig5_memory_policies_v5_matched_only",
    "fig7_smoothing_day5_day10_v4_matched36": "fig6_smoothing_day5_day10_v5_matched36",
    "fig12_smoothing_loss_v4_matched36": "fig10_smoothing_loss_v5_m1",
    "fig13_smoothing_feature_amplitude_matched36": "fig11_smoothing_feature_amplitude_v5_m1",
    "fig14_smoothing_feature_coherence_matched36": "fig12_smoothing_feature_coherence_v5_m1",
    "fig15_smoothing_lead_spectra_matched36": "fig13_smoothing_lead_spectra_v5_m1",
    "fig17_memory_power_day5_matched36": "fig14_memory_power_day5_v5_m1",
    "fig18_memory_power_day10_matched36": "fig15_memory_power_day10_v5_m1",
}
LABELS = {
    "Carry/carry": "M1",
    "Carry/carry (M1)": "M1",
    "Carry/carry (5 seeds)": "M1 retrainings (5 seeds)",
    "Band: ±1 SD over carry seeds": "Band: ±1 SD over M1 retrainings",
}
SOURCES: set[Path] = set()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(name):
    path = FROZEN / name
    SOURCES.add(path)
    return pd.read_csv(path, float_precision="round_trip")


def frozen_values():
    """Arrange saved values for old layout functions, without recomputation."""
    path = FROZEN / "all_channels_selected_leads_curves.npz"
    SOURCES.add(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        assert metadata["forecast_count"] == 36
        assert metadata["models"] == list(MODEL_IDS)
        assert metadata["lead_hours"] == [6, 120, 240]
        channels = metadata["channels"]
        lead_indices = np.array([0, 19, 39])
        values = {}
        for key, source_key in (("amplitude_by_channel", "amplitude_ratio"),
                                ("coherence_by_channel", "coherence")):
            values[key] = np.full((40, 4, len(channels), 179), np.nan)
            values[key][lead_indices] = data[source_key]
            np.testing.assert_array_equal(values[key][lead_indices], data[source_key])
        values["prediction_power"] = np.full((40, 4, len(channels), 180), np.nan)
        values["prediction_power"][lead_indices, :, :, 1:] = data["prediction_power"]
        truth = np.full((40, len(channels), 180), np.nan)
        truth[lead_indices, :, 1:] = data["truth_power"]
        np.testing.assert_array_equal(values["prediction_power"][lead_indices, :, :, 1:], data["prediction_power"])
        np.testing.assert_array_equal(truth[lead_indices, :, 1:], data["truth_power"])
    frame = read_csv("loss_by_lead.csv")
    values["loss"] = frame.pivot(index="lead_hours", columns="model", values="exact_gc_loss").loc[:, MODEL_IDS].to_numpy()
    assert values["loss"].shape == (40, 4)
    frame = read_csv("weighted_spectral_curves.csv")
    for key, column in (("amplitude", "weighted_amplitude_ratio"), ("coherence", "weighted_coherence")):
        values[key] = np.full((40, 4, 179), np.nan)
        for hours, li in zip(metadata["lead_hours"], lead_indices):
            for mi, model in enumerate(MODEL_IDS):
                rows = frame.loc[(frame.lead_hours == hours) & (frame.model == model)].sort_values("degree")
                np.testing.assert_array_equal(rows.degree, np.arange(1, 180))
                values[key][li, mi] = rows[column].to_numpy()
    frame = read_csv("all_channels_selected_leads_summary.csv")
    values["power_error_by_channel"] = np.full((40, 4, len(channels)), np.nan)
    for row in frame.itertuples(index=False):
        channel = [row.variable, None if pd.isna(row.pressure_hpa) else row.pressure_hpa]
        values["power_error_by_channel"][int(row.lead_hours) // 6 - 1, MODEL_IDS.index(row.model), channels.index(channel)] = row.power_mismatch_equal_degree_pct
    return values, truth, channels


def numerical_signature(fig):
    """Hash coordinates of every line and uncertainty-band path, not labels."""
    h = hashlib.sha256()
    lines = bands = scalars = 0
    def add(array):
        nonlocal scalars
        array = np.asarray(array, dtype="<f8")
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
        scalars += array.size
    for axis in fig.axes:
        add(axis.get_xlim())
        add(axis.get_ylim())
        for line in axis.lines:
            add(line.get_xdata())
            add(line.get_ydata())
            lines += 1
        for collection in axis.collections:
            for path in collection.get_paths():
                add(path.vertices)
                bands += 1
    return {"sha256": h.hexdigest(), "lines": lines, "band_paths": bands, "coordinate_scalars": scalars}


@contextmanager
def render_hooks(signatures, relabel):
    original_text = Text.set_text
    original_save = Figure.savefig
    def set_text(self, value):
        return original_text(self, LABELS.get(value, value) if relabel and isinstance(value, str) else value)
    def save(self, filename, *args, **kwargs):
        path = Path(filename)
        if path.stem in NAMES:
            sig = numerical_signature(self)
            if path.stem in signatures:
                assert signatures[path.stem] == sig
            signatures[path.stem] = sig
        return original_save(self, filename, *args, **kwargs)
    Text.set_text, Figure.savefig = set_text, save
    try:
        yield
    finally:
        Text.set_text, Figure.savefig = original_text, original_save


def render(folder, relabel, values, truth, channels):
    folder.mkdir(parents=True, exist_ok=True)
    signatures = {}
    original_export, original_summary = appendix.base.export, appendix.make_summary
    def selected_export(fig, images, stem):
        if stem not in NAMES:
            plt.close(fig)
            return []
        return original_export(fig, images, stem)
    with render_hooks(signatures, relabel), redirect_stdout(io.StringIO()):
        memory.DATA = folder / "memory-data"
        memory.IMAGES = folder
        memory.main()
        old_argv = sys.argv
        try:
            for module in (spectra, loss):
                sys.argv = [str(module.__file__), "--output", str(folder)]
                module.main()
        finally:
            sys.argv = old_argv
        appendix.base.export = selected_export
        appendix.make_summary = lambda *args, **kwargs: []
        try:
            with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                                 "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
                                 "xtick.major.width": .6, "ytick.major.width": .6,
                                 "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                                 "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
                appendix.make_figures(values, truth, channels, folder)
        finally:
            appendix.base.export, appendix.make_summary = original_export, original_summary
    assert set(signatures) == set(NAMES)
    return signatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manuscripts/v5_m1_labels_20261114")
    parser.add_argument("--publish", action="store_true", help="Copy validated PDFs and source manifest into Overleaf")
    args = parser.parse_args()
    output = args.output.resolve()
    values, truth, channels = frozen_values()
    for module in (memory, spectra, loss, appendix, appendix.base, appendix.layout, appendix.physical):
        SOURCES.add(Path(module.__file__).resolve())
    SOURCES.add(Path(__file__).resolve())
    SOURCES.update(memory.SOURCE.glob("*.csv"))
    SOURCES.add(memory.SOURCE / "summary.json")
    before = {str(path.relative_to(ROOT)): digest(path) for path in sorted(SOURCES)}
    original = render(output / "original-labels", False, values, truth, channels)
    revised = render(output / "m1-labels", True, values, truth, channels)
    assert original == revised, "Relabeling changed plotted coordinates"
    assert before == {str(path.relative_to(ROOT)): digest(path) for path in sorted(SOURCES)}
    products = []
    for old, new in NAMES.items():
        for suffix in ("pdf", "png"):
            source = output / "m1-labels" / f"{old}.{suffix}"
            target = output / f"{new}.{suffix}"
            shutil.copy2(source, target)
            if args.publish and suffix == "pdf":
                shutil.copy2(target, ROOT / "Overleaf/arxiv_pics" / target.name)
            products.append({"file": target.name, "sha256": digest(target), "bytes": target.stat().st_size})
    manifest = {"purpose": "v5 model naming and figure numbering only", "sources": before,
                "label_substitutions": LABELS, "figure_names": NAMES,
                "plotted_coordinates_identical": True, "coordinate_signatures": revised,
                "source_exports_unchanged": True, "new_evaluations_or_metric_calculations": False,
                "figures": products}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        archive = ROOT / "Overleaf/archive/Graphs/v5_m1_labels_20261114"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "manifest.json", archive / "manifest.json")
        shutil.copy2(Path(__file__), archive / Path(__file__).name)
        (archive / "README.md").write_text("# v5 model labels and reordered appendix figures\n\n"
            "These eight PDFs reuse the frozen v4 curve exports and layout functions. "
            "M1 identifies the original selected checkpoint; the independent five-seed panel "
            "is labeled M1 retrainings. No forecasts or metric estimates were recomputed.\n\n"
            "The generator compares every line and uncertainty-band coordinate before and "
            "after relabeling, then publishes only after exact agreement. Source hashes and "
            "figure hashes are in manifest.json. Historical scripts, exports, and v4 PDFs "
            "remain unchanged. Run the generator from the parent Weather_global workspace "
            "using the GraphCast environment with --publish.\n")
    print(json.dumps({"output": str(output), "pdfs": len(NAMES), "coordinates_identical": original == revised,
                      "sources_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
