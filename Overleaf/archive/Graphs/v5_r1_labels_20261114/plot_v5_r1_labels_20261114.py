#!/usr/bin/env python3
"""Rename the reset control R1 in the six retained v5 comparison figures.

Reuse frozen exports and existing layouts. The original matched SWA checkpoint
is R1; the independent seed-22 raw control is explicitly an R1 retraining.
Deleted physical-power figures are never exported or published by this wrapper.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt

import plot_v5_m1_labels_20261114 as previous
from plot_v5_spectral_pending122_20261114 import HEADINGS

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "fig6_memory_policies_v4_two_policies": "fig5_memory_policies_v5_matched_only",
    "fig7_smoothing_day5_day10_v4_matched36": "fig6_smoothing_day5_day10_v5_pending122",
    "fig12_smoothing_loss_v4_matched36": "fig10_smoothing_loss_v5_pending122",
    "fig13_smoothing_feature_amplitude_matched36": "fig11_smoothing_feature_amplitude_v5_pending122",
    "fig14_smoothing_feature_coherence_matched36": "fig12_smoothing_feature_coherence_v5_pending122",
    "fig15_smoothing_lead_spectra_matched36": "fig13_smoothing_lead_spectra_v5_pending122",
}
R1_LABELS = {
    "Reset/reset": "R1",
    "Reset/reset (1 seed)": "R1 retraining (seed 22)",
}


@contextmanager
def inspect_figures(records, require_r1):
    original_save = Figure.savefig

    def save(self, filename, *args, **kwargs):
        stem = Path(filename).stem
        if stem in NAMES:
            self.canvas.draw()
            labels = sorted({artist.get_text() for artist in self.findobj(Text)
                             if artist.get_text()})
            if require_r1:
                assert "R1" in labels, (stem, labels)
                assert not any("reset/reset" in label.lower() for label in labels), stem
                if stem == "fig6_memory_policies_v4_two_policies":
                    assert "R1 retraining (seed 22)" in labels
            records[stem] = {
                "size_inches": self.get_size_inches().tolist(),
                "axes_positions": [list(axis.get_position().bounds) for axis in self.axes],
                "labels": labels,
            }
        return original_save(self, filename, *args, **kwargs)

    Figure.savefig = save
    try:
        yield
    finally:
        Figure.savefig = original_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/manuscripts/v5_r1_labels_20261114")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    values, truth, channels = previous.frozen_values()
    for module in (previous, previous.memory, previous.spectra, previous.loss,
                   previous.appendix, previous.appendix.base, previous.appendix.layout,
                   previous.appendix.physical):
        previous.SOURCES.add(Path(module.__file__).resolve())
    previous.SOURCES.add(Path(__file__).resolve())
    previous.SOURCES.add(ROOT / "scripts/analyze_models/plot_v5_spectral_pending122_20261114.py")
    previous.SOURCES.update(previous.memory.SOURCE.glob("*.csv"))
    previous.SOURCES.add(previous.memory.SOURCE / "summary.json")
    source_hashes = {str(path.relative_to(ROOT)): previous.digest(path)
                     for path in sorted(previous.SOURCES)}

    figure_dir = ROOT / "Overleaf/arxiv_pics"
    target_names = {f"{name}.pdf" for name in NAMES.values()}
    unchanged = {str(path.relative_to(ROOT)): previous.digest(path)
                 for path in sorted(figure_dir.glob("*.pdf"))
                 if path.name not in target_names}
    before_figures = output / "before_figures"
    before_figures.mkdir(exist_ok=True)
    for name in target_names:
        shutil.copy2(figure_dir / name, before_figures / name)

    prior_paths = [ROOT / "Overleaf/archive/Graphs/v5_m1_labels_20261114/manifest.json",
                   ROOT / "Overleaf/archive/Graphs/v5_spectral_pending122_20261114/manifest.json"]
    prior = [json.loads(path.read_text()) for path in prior_paths]
    expected = {**prior[0]["coordinate_signatures"], **prior[1]["coordinate_signatures"]}
    old_names, old_labels, old_memory = previous.NAMES, previous.LABELS, previous.memory.main

    def memory_in_context():
        # The current spectral PDFs were generated in a fresh matplotlib context;
        # memory.main's global style must not leak into their labels or layout.
        with plt.rc_context():
            old_memory()

    previous.NAMES = NAMES
    previous.memory.main = memory_in_context
    original_layout, revised_layout = {}, {}
    try:
        previous.LABELS = {**old_labels, **HEADINGS}
        with inspect_figures(original_layout, False):
            original = previous.render(output / "original_labels", True, values, truth, channels)
        previous.LABELS = {**old_labels, **HEADINGS, **R1_LABELS}
        with inspect_figures(revised_layout, True):
            revised = previous.render(output / "r1_labels", True, values, truth, channels)
    finally:
        previous.NAMES, previous.LABELS, previous.memory.main = old_names, old_labels, old_memory

    assert original == revised, "Relabeling changed plotted coordinates"
    assert revised == {stem: expected[stem] for stem in NAMES}, "Frozen figure signatures differ"
    for stem in NAMES:
        for key in ("size_inches", "axes_positions"):
            assert original_layout[stem][key] == revised_layout[stem][key], (stem, key)
    assert source_hashes == {str(path.relative_to(ROOT)): previous.digest(path)
                             for path in sorted(previous.SOURCES)}

    products = []
    for old, new in NAMES.items():
        for suffix in ("pdf", "png"):
            target = output / f"{new}.{suffix}"
            shutil.copy2(output / "r1_labels" / f"{old}.{suffix}", target)
            if args.publish and suffix == "pdf":
                shutil.copy2(target, figure_dir / target.name)
            products.append({"file": target.name, "sha256": previous.digest(target),
                             "bytes": target.stat().st_size})
    assert all(previous.digest(ROOT / path) == digest for path, digest in unchanged.items())
    manifest = {
        "purpose": "Name the original matched reset control R1 and distinguish its local retraining",
        "sources": source_hashes,
        "prior_manifests": {str(path.relative_to(ROOT)): previous.digest(path) for path in prior_paths},
        "label_substitutions": R1_LABELS,
        "figure_names": NAMES,
        "plotted_coordinates_identical": True,
        "matches_prior_coordinate_signatures": True,
        "coordinate_signatures": revised,
        "figure_and_axes_geometry_unchanged": True,
        "revised_labels": {stem: data["labels"] for stem, data in revised_layout.items()},
        "source_exports_unchanged": True,
        "other_active_pdfs_unchanged": True,
        "other_active_pdf_hashes": unchanged,
        "new_evaluations_or_metric_calculations": False,
        "figures": products,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        archive = ROOT / "Overleaf/archive/Graphs/v5_r1_labels_20261114"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "manifest.json", archive / "manifest.json")
        shutil.copy2(Path(__file__), archive / Path(__file__).name)
        (archive / "README.md").write_text(
            "# R1 model labels in v5\n\n"
            "The six retained comparison figures name the original matched reset/reset "
            "SWA control R1. The independently trained seed-22 raw control in Figure 5 "
            "is labeled R1 retraining (seed 22). Existing M1 labels, provisional spectral "
            "headings, values, curve styles, and axes geometry are preserved.\n\n"
            "Every plotted coordinate and uncertainty-band path matches both the "
            "preceding render and its archived signature. Source exports and all "
            "other active PDFs are hash-checked unchanged. Deleted physical-power "
            "figures are not republished. No evaluations or metric estimates are run. "
            "Run this wrapper from the parent Weather_global workspace through "
            "scripts/graphcast_env.sh with --publish.\n"
        )
    print(json.dumps({"output": str(output), "pdfs": len(NAMES),
                      "coordinates_identical": True, "sources_unchanged": True,
                      "other_pdfs_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
