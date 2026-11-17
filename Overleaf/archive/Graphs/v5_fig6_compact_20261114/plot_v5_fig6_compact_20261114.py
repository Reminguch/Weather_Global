#!/usr/bin/env python3
"""Compact Figure 6 without altering frozen spectra or axis limits.

Remove the redundant in-figure draft heading and move the four-model legend
to one line below the panels. Provisional status remains in the manuscript's
caption and draft-status paragraph. No forecasts or metrics are recomputed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys

from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.lines import Line2D

import plot_v4_spectra_day5_day10_20261114 as source
import plot_v5_m1_labels_20261114 as previous

ROOT = Path(__file__).resolve().parents[2]
STEM = source.STEM
TARGET = "fig6_smoothing_day5_day10_v5_pending122"
LABELS = {"Carry/carry (M1)": "M1", "Reset/reset": "R1",
          "36 matched forecast dates": "Provisional evaluation"}


@contextmanager
def figure_hooks(records, compact):
    original_text, original_save = Text.set_text, Figure.savefig
    adjusted = set()

    def set_text(self, value):
        return original_text(self, LABELS.get(value, value) if isinstance(value, str) else value)

    def save(self, filename, *args, **kwargs):
        if Path(filename).stem == STEM:
            if compact and id(self) not in adjusted:
                self.set_size_inches(6.25, 3.8)
                for title in list(self.texts):
                    if title.get_text() == "Provisional evaluation":
                        title.remove()
                for legend in list(self.legends):
                    legend.remove()
                handles = [Line2D([], [], color=color, lw=1.5, label=LABELS.get(label, label))
                           for _, label, color in source.MODELS]
                self.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .012),
                            ncol=4, frameon=False, fontsize=7.5, handlelength=1.8,
                            handletextpad=.55, columnspacing=1.25, borderaxespad=0.)
                self.subplots_adjust(left=.09, right=.965, bottom=.17, top=.80,
                                     hspace=.47, wspace=.25)
                adjusted.add(id(self))
            self.canvas.draw()
            records["coordinate_signature"] = previous.numerical_signature(self)
            records["size_inches"] = self.get_size_inches().tolist()
            records["axes_positions"] = [list(axis.get_position().bounds) for axis in self.axes]
            labels = sorted({artist.get_text() for artist in self.findobj(Text) if artist.get_text()})
            records["labels"] = labels
            assert all(label in labels for label in ("GraphCast", "M1", "GC + optimal filter", "R1"))
            if compact:
                assert "Provisional evaluation" not in labels
                assert len(self.legends) == 1
        return original_save(self, filename, *args, **kwargs)

    Text.set_text, Figure.savefig = set_text, save
    try:
        yield
    finally:
        Text.set_text, Figure.savefig = original_text, original_save


def render(folder, compact):
    records = {}
    old_argv = sys.argv
    try:
        sys.argv = [str(source.__file__), "--output", str(folder)]
        with figure_hooks(records, compact), redirect_stdout(io.StringIO()):
            source.main()
    finally:
        sys.argv = old_argv
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/manuscripts/v5_fig6_compact_20261114")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = ROOT / "Overleaf/arxiv_pics"
    prior_path = ROOT / "Overleaf/archive/Graphs/v5_r1_labels_20261114/manifest.json"
    expected = json.loads(prior_path.read_text())["coordinate_signatures"][STEM]
    sources = [source.SOURCE, Path(source.__file__), Path(previous.__file__),
               Path(__file__), prior_path]
    source_hashes = {str(path.relative_to(ROOT)): previous.digest(path) for path in sources}
    unchanged = {str(path.relative_to(ROOT)): previous.digest(path)
                 for path in sorted(figure_dir.glob("*.pdf")) if path.stem != TARGET}
    shutil.copy2(figure_dir / f"{TARGET}.pdf", output / "before.pdf")
    original = render(output / "original", False)
    compact = render(output / "compact", True)
    assert original["coordinate_signature"] == compact["coordinate_signature"] == expected
    assert source_hashes == {str(path.relative_to(ROOT)): previous.digest(path) for path in sources}
    products = []
    for suffix in ("pdf", "png"):
        target = output / f"{TARGET}.{suffix}"
        shutil.copy2(output / "compact" / f"{STEM}.{suffix}", target)
        if args.publish and suffix == "pdf":
            shutil.copy2(target, figure_dir / target.name)
        products.append({"file": target.name, "sha256": previous.digest(target),
                         "bytes": target.stat().st_size})
    assert all(previous.digest(ROOT / path) == value for path, value in unchanged.items())
    manifest = {
        "purpose": "Compact Figure 6 header and legend; all spectra and limits unchanged",
        "sources": source_hashes,
        "prior_layout": original,
        "compact_layout": compact,
        "coordinate_signature_matches_prior_archive": True,
        "plotted_coordinates_identical": True,
        "source_exports_unchanged": True,
        "other_active_figures_unchanged": True,
        "other_active_figure_hashes": unchanged,
        "provisional_status": "Retained in manuscript caption and main-text draft-status paragraph",
        "new_evaluations_or_metric_calculations": False,
        "figures": products,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        archive = ROOT / "Overleaf/archive/Graphs/v5_fig6_compact_20261114"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), archive / Path(__file__).name)
        shutil.copy2(output / "manifest.json", archive / "manifest.json")
        (archive / "README.md").write_text(
            "# Compact v5 Figure 6\n\n"
            "The redundant in-figure provisional heading is removed, and the four-model "
            "legend is one compact row beneath the panels. The canvas changes from "
            "6.25 × 4.9 to 6.25 × 3.8 inches with the original label font sizes retained. "
            "Provisional status remains in the manuscript caption and draft-status paragraph.\n\n"
            "Every plotted value, uncertainty path, and axis limit matches the prior R1 "
            "archive exactly. Source exports and all other active figure PDFs are unchanged. "
            "No evaluations or metrics were computed. Run in the parent Weather_global "
            "workspace through scripts/graphcast_env.sh with --publish.\n")
    print(json.dumps({"output": str(output), "figure_inches": compact["size_inches"],
                      "coordinates_identical": True, "other_figures_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
