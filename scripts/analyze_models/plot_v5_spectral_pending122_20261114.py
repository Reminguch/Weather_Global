#!/usr/bin/env python3
"""Mark the frozen v5 spectral plots provisional while full evaluation runs.

The displayed values remain the archived 36-date results. This script changes
only draft headings and filenames; it never labels them completed 122-date
results or computes new forecasts, metrics, or uncertainty estimates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import plot_v5_m1_labels_20261114 as previous

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "fig7_smoothing_day5_day10_v4_matched36": "fig6_smoothing_day5_day10_v5_pending122",
    "fig12_smoothing_loss_v4_matched36": "fig10_smoothing_loss_v5_pending122",
    "fig13_smoothing_feature_amplitude_matched36": "fig11_smoothing_feature_amplitude_v5_pending122",
    "fig14_smoothing_feature_coherence_matched36": "fig12_smoothing_feature_coherence_v5_pending122",
    "fig15_smoothing_lead_spectra_matched36": "fig13_smoothing_lead_spectra_v5_pending122",
    "fig17_memory_power_day5_matched36": "fig14_memory_power_day5_v5_pending122",
    "fig18_memory_power_day10_matched36": "fig15_memory_power_day10_v5_pending122",
}
HEADINGS = {
    "36 matched forecast dates": "Provisional evaluation",
    "Day 5 · 36 matched forecast dates": "Day 5 · provisional evaluation",
    "Day 10 · 36 matched forecast dates": "Day 10 · provisional evaluation",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manuscripts/v5_spectral_pending122_20261114")
    parser.add_argument("--publish", action="store_true", help="Copy checked draft PDFs and provenance into Overleaf")
    args = parser.parse_args()
    output = args.output.resolve()

    values, truth, channels = previous.frozen_values()
    for module in (previous, previous.spectra, previous.loss, previous.appendix,
                   previous.appendix.base, previous.appendix.layout, previous.appendix.physical):
        previous.SOURCES.add(Path(module.__file__).resolve())
    previous.SOURCES.add(Path(__file__).resolve())
    source_hashes = {str(path.relative_to(ROOT)): previous.digest(path) for path in sorted(previous.SOURCES)}
    fig5 = ROOT / "Overleaf/arxiv_pics/fig5_memory_policies_v5_matched_only.pdf"
    fig5_hash = previous.digest(fig5)
    old_names, old_labels, old_memory = previous.NAMES, previous.LABELS, previous.memory.main
    previous.NAMES = NAMES
    previous.LABELS = {**old_labels, **HEADINGS}
    # The established 122-date memory figure is outside this task.
    previous.memory.main = lambda: None
    try:
        original = previous.render(output / "original-labels", False, values, truth, channels)
        revised = previous.render(output / "provisional-labels", True, values, truth, channels)
    finally:
        previous.NAMES, previous.LABELS, previous.memory.main = old_names, old_labels, old_memory
    assert original == revised, "Changing draft headings altered plotted coordinates"
    assert source_hashes == {str(path.relative_to(ROOT)): previous.digest(path) for path in sorted(previous.SOURCES)}
    assert previous.digest(fig5) == fig5_hash

    products = []
    for old, new in NAMES.items():
        for suffix in ("pdf", "png"):
            source = output / "provisional-labels" / f"{old}.{suffix}"
            target = output / f"{new}.{suffix}"
            shutil.copy2(source, target)
            if args.publish and suffix == "pdf":
                shutil.copy2(target, ROOT / "Overleaf/arxiv_pics" / target.name)
            products.append({"file": target.name, "sha256": previous.digest(target), "bytes": target.stat().st_size})
    manifest = {
        "purpose": "Clearly provisional spectral graphics pending the full evaluation",
        "source_cohort": {"forecast_count": 36, "start": "2023-01-01", "end": "2023-04-16", "status": "frozen_interim_matched_cohort"},
        "planned_evaluation": {"forecast_count": 122, "status": "in_progress_per_author", "replace_before_camera_ready": True},
        "sources": source_hashes,
        "draft_heading_substitutions": HEADINGS,
        "figure_names": NAMES,
        "plotted_coordinates_identical": True,
        "coordinate_signatures": revised,
        "source_exports_unchanged": True,
        "new_evaluations_or_metric_calculations": False,
        "figure5_unchanged": True,
        "figures": products,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        archive = ROOT / "Overleaf/archive/Graphs/v5_spectral_pending122_20261114"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "manifest.json", archive / "manifest.json")
        shutil.copy2(Path(__file__), archive / Path(__file__).name)
        (archive / "README.md").write_text(
            "# Provisional spectral figures for the 122-date draft\n\n"
            "The seven PDFs are explicitly labeled provisional while the full 122-date "
            "evaluation is in progress. Their numerical contents remain the frozen "
            "36-date matched exports (1 January–16 April 2023), with source hashes "
            "and exact coordinate comparisons in manifest.json. No values have been "
            "relabeled as completed 122-date measurements. Replace the plots and "
            "associated table/prose values when the full evaluation is ready.\n\n"
            "The prior v4 and v5 source archives are preserved. Figure 5 is untouched. "
            "Run the generator from the parent Weather_global workspace through the "
            "GraphCast environment with --publish.\n"
        )
    print(json.dumps({"output": str(output), "pdfs": len(NAMES), "coordinates_identical": True,
                      "source_exports_unchanged": True, "figure5_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
