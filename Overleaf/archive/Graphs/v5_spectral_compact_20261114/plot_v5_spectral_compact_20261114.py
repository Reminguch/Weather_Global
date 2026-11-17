#!/usr/bin/env python3
"""Apply the compact Figure 6 style to retained Appendix C spectra only.

No forecast or metric computations: reuse frozen curves and preserve all panel
titles, mismatch annotations, axis limits, and original label font sizes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.lines import Line2D

import plot_v5_m1_labels_20261114 as previous
from plot_v5_spectral_pending122_20261114 import HEADINGS
from plot_v5_r1_labels_20261114 import R1_LABELS

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "fig13_smoothing_feature_amplitude_matched36": "fig11_smoothing_feature_amplitude_v5_pending122",
    "fig14_smoothing_feature_coherence_matched36": "fig12_smoothing_feature_coherence_v5_pending122",
    "fig15_smoothing_lead_spectra_matched36": "fig13_smoothing_lead_spectra_v5_pending122",
}


@contextmanager
def compact_hooks(records, compact):
    original_save = Figure.savefig
    adjusted = set()

    def save(self, filename, *args, **kwargs):
        stem = Path(filename).stem
        if stem in NAMES:
            if compact and id(self) not in adjusted:
                lead_plot = stem == "fig15_smoothing_lead_spectra_matched36"
                self.set_size_inches(6.25, 5.6 if lead_plot else 3.8)
                for title in list(self.texts):
                    if title.get_text() == "Provisional evaluation":
                        title.remove()
                for legend in list(self.legends):
                    legend.remove()
                handles = [Line2D([], [], color=color, lw=1.5,
                                  label=previous.LABELS.get(label, label))
                           for _, label, color in previous.appendix.MODELS]
                self.legend(handles=handles, loc="lower center",
                            bbox_to_anchor=(.5, .009 if lead_plot else .012),
                            ncol=4, frameon=False, fontsize=7.5, handlelength=1.8,
                            handletextpad=.55, columnspacing=1.25, borderaxespad=0.)
                if lead_plot:
                    self.subplots_adjust(left=.095, right=.965, bottom=.115,
                                         top=.85, hspace=.50, wspace=.32)
                else:
                    self.subplots_adjust(left=.09, right=.965, bottom=.17,
                                         top=.92, hspace=.55, wspace=.32)
                adjusted.add(id(self))
            self.canvas.draw()
            labels = sorted({artist.get_text() for artist in self.findobj(Text) if artist.get_text()})
            assert all(label in labels for label in ("GraphCast", "M1", "GC + optimal filter", "R1"))
            if compact:
                assert "Provisional evaluation" not in labels
                assert len(self.legends) == 1
            records[stem] = {
                "coordinate_signature": previous.numerical_signature(self),
                "size_inches": self.get_size_inches().tolist(),
                "axes_positions": [list(axis.get_position().bounds) for axis in self.axes],
                "panel_titles": [axis.get_title() for axis in self.axes],
                "annotations": [[(text.get_text(), text.get_fontsize()) for text in axis.texts]
                                for axis in self.axes],
                "label_fonts": [[axis.xaxis.label.get_fontsize(), axis.yaxis.label.get_fontsize(),
                                 axis.title.get_fontsize()] for axis in self.axes],
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
                        default=ROOT / "artifacts/manuscripts/v5_spectral_compact_20261114")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    values, truth, channels = previous.frozen_values()
    modules = (previous, previous.appendix, previous.appendix.base,
               previous.appendix.layout, previous.appendix.physical)
    previous.SOURCES.update(Path(module.__file__).resolve() for module in modules)
    previous.SOURCES.update((Path(__file__).resolve(),
                             ROOT / "scripts/analyze_models/plot_v5_spectral_pending122_20261114.py",
                             ROOT / "scripts/analyze_models/plot_v5_r1_labels_20261114.py"))
    source_hashes = {str(path.relative_to(ROOT)): previous.digest(path)
                     for path in sorted(previous.SOURCES)}
    figure_dir = ROOT / "Overleaf/arxiv_pics"
    target_names = {f"{name}.pdf" for name in NAMES.values()}
    unchanged = {str(path.relative_to(ROOT)): previous.digest(path)
                 for path in sorted(figure_dir.glob("*.pdf")) if path.name not in target_names}
    before = output / "before_figures"
    before.mkdir(exist_ok=True)
    for name in target_names:
        if not (before / name).exists():
            shutil.copy2(figure_dir / name, before / name)
    prior_path = ROOT / "Overleaf/archive/Graphs/v5_r1_labels_20261114/manifest.json"
    expected = json.loads(prior_path.read_text())["coordinate_signatures"]
    old_names, old_labels = previous.NAMES, previous.LABELS
    other_modules = (previous.memory, previous.spectra, previous.loss)
    old_mains = [module.main for module in other_modules]
    original, compact = {}, {}
    previous.NAMES = NAMES
    previous.LABELS = {**old_labels, **HEADINGS, **R1_LABELS}
    for module in other_modules:
        module.main = lambda: None
    try:
        with compact_hooks(original, False):
            original_signatures = previous.render(output / "original", True, values, truth, channels)
        with compact_hooks(compact, True):
            revised_signatures = previous.render(output / "compact", True, values, truth, channels)
    finally:
        previous.NAMES, previous.LABELS = old_names, old_labels
        for module, main_function in zip(other_modules, old_mains):
            module.main = main_function
    assert original_signatures == revised_signatures == {stem: expected[stem] for stem in NAMES}
    for stem in NAMES:
        assert original[stem]["coordinate_signature"] == compact[stem]["coordinate_signature"] == expected[stem]
        for key in ("panel_titles", "annotations", "label_fonts"):
            assert original[stem][key] == compact[stem][key], (stem, key)
    assert source_hashes == {str(path.relative_to(ROOT)): previous.digest(path)
                             for path in sorted(previous.SOURCES)}
    products = []
    for old, new in NAMES.items():
        for suffix in ("pdf", "png"):
            target = output / f"{new}.{suffix}"
            shutil.copy2(output / "compact" / f"{old}.{suffix}", target)
            if args.publish and suffix == "pdf":
                shutil.copy2(target, figure_dir / target.name)
            products.append({"file": target.name, "sha256": previous.digest(target),
                             "bytes": target.stat().st_size})
    assert all(previous.digest(ROOT / path) == value for path, value in unchanged.items())
    manifest = {
        "purpose": "Apply compact Figure 6 legend/header style to v5 Figures 11–13",
        "sources": source_hashes,
        "prior_manifest": {"path": str(prior_path.relative_to(ROOT)), "sha256": previous.digest(prior_path)},
        "figure_names": NAMES,
        "prior_layout": original,
        "compact_layout": compact,
        "coordinate_signature_matches_prior_archive": True,
        "plotted_coordinates_identical": True,
        "panel_titles_annotations_and_fonts_unchanged": True,
        "source_exports_unchanged": True,
        "other_active_figures_unchanged": True,
        "other_active_figure_hashes": unchanged,
        "provisional_status": "Retained in the main-text draft note covering Appendix C",
        "new_evaluations_or_metric_calculations": False,
        "figures": products,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        archive = ROOT / "Overleaf/archive/Graphs/v5_spectral_compact_20261114"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), archive / Path(__file__).name)
        shutil.copy2(output / "manifest.json", archive / "manifest.json")
        (archive / "README.md").write_text(
            "# Compact retained Appendix C spectra\n\n"
            "Figures 11–13 use the same compact, single-row bottom legend as Figure 6, "
            "without a duplicate provisional heading. The two feature plots are "
            "6.25 × 3.8 inches; the three-lead plot is 6.25 × 5.6 inches. The manuscript's "
            "draft-status note still applies to Appendix C.\n\n"
            "Every curve, axis limit, field/lead title, mismatch annotation, and label "
            "font size is preserved. Coordinate signatures match the previous R1 "
            "archive exactly. Figure 6, the Figure 10 loss plot, all other active PDFs, "
            "and the saved source exports are unchanged. No evaluations or metrics "
            "were computed. Run in the parent Weather_global workspace through "
            "scripts/graphcast_env.sh with --publish.\n")
    print(json.dumps({"output": str(output), "pdfs": len(NAMES),
                      "coordinates_identical": True, "annotations_and_fonts_unchanged": True,
                      "other_figures_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
