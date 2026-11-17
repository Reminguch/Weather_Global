#!/usr/bin/env python3
"""Clarify Figure 15's fields without changing the frozen plotted values.

Only the day-10 physical-power figure is published. Reuses the pending-122
headings and original plotting pipeline; no forecasts or metrics are computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

import plot_v5_spectral_pending122_20261114 as pending

previous = pending.previous
ROOT = previous.ROOT
STEM = "fig18_memory_power_day10_matched36"
NAME = pending.NAMES[STEM]
PANEL_TITLES = {
    "T2m": "(a) 2 m temperature\n(T2m)",
    "Z500": "(b) 500 hPa geopotential\n(Z500)",
    "Q700": "(c) 700 hPa specific humidity\n(Q700)",
    "TP6h": "(d) 6-hour accumulated\nprecipitation (TP6h)",
}
EXPECTED_FIELDS = [
    ("T2m", "2m_temperature", None, 1., "K^2"),
    ("Z500", "geopotential", 500., 1., "m^4 s^-4"),
    ("Q700", "specific_humidity", 700., 1e6, "(g kg^-1)^2"),
    ("TP6h", "total_precipitation_6hr", None, 1e6, "mm^2"),
]


def render(folder, expanded, values, truth, channels):
    """Restrict the historical pipeline to Figure 15, then change its titles."""
    old_names, old_labels = previous.NAMES, previous.LABELS
    modules = (previous.memory, previous.spectra, previous.loss)
    old_mains = [module.main for module in modules]
    old_export = previous.appendix.base.export
    checked = []

    def checked_export(fig, images, stem):
        assert stem == STEM, "Only Figure 15 may be exported"
        if expanded:
            # Additional vertical space keeps both title lines clear of the
            # wavelength axes, the shared legend, and the first-row x labels.
            fig.set_size_inches(6.25, 5.15)
            fig.subplots_adjust(left=.115, right=.965, bottom=.095, top=.735,
                                wspace=.42, hspace=.75)
        assert len(fig.axes) == 4
        for axis, (short, variable, pressure, conversion, unit) in zip(fig.axes, EXPECTED_FIELDS):
            assert axis.get_title() == (PANEL_TITLES[short] if expanded else short)
            assert axis.get_yscale() == "log"
            assert len(axis.lines) == 5
            channel = channels.index([variable, pressure])
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), conversion * truth[39, channel, 1:])
            for model in range(4):
                np.testing.assert_array_equal(axis.lines[model + 1].get_ydata(),
                                              conversion * values["prediction_power"][39, model, channel, 1:])
            checked.append({"panel": chr(ord("a") + len(checked)), "title": axis.get_title(),
                            "variable": variable, "pressure_hpa": pressure,
                            "power_conversion": conversion, "power_unit": unit,
                            "npz_channel_index": channel, "lead_hours": 240,
                            "curves_equal_frozen_exports": True})
        assert any(t.get_text() == "Day 10 · provisional evaluation" for t in fig.texts)
        return old_export(fig, images, stem)

    previous.NAMES = {STEM: NAME}
    previous.LABELS = {**old_labels, **pending.HEADINGS, **(PANEL_TITLES if expanded else {})}
    for module in modules:
        module.main = lambda: None
    previous.appendix.base.export = checked_export
    try:
        signatures = previous.render(folder, True, values, truth, channels)
    finally:
        previous.NAMES, previous.LABELS = old_names, old_labels
        previous.appendix.base.export = old_export
        for module, main in zip(modules, old_mains):
            module.main = main
    assert len(checked) == 4
    assert set(p.name for p in folder.iterdir()) == {f"{STEM}.pdf", f"{STEM}.png"}
    return signatures, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manuscripts/v5_fig15_features_20261114")
    parser.add_argument("--publish", action="store_true", help="Publish only the checked Figure 15 PDF")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    values, truth, channels = previous.frozen_values()
    actual_fields = [tuple(field[:5]) for field in previous.appendix.physical.FIELDS]
    assert actual_fields == EXPECTED_FIELDS, "Physical field identities or units changed"
    with np.load(previous.FROZEN / "all_channels_selected_leads_curves.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        assert metadata["channels"] == channels
        assert metadata["lead_hours"][2] == 240
        for _, variable, pressure, _, _ in EXPECTED_FIELDS:
            assert [variable, pressure] in metadata["channels"]

    for module in (pending, previous, previous.appendix, previous.appendix.base,
                   previous.appendix.layout, previous.appendix.physical):
        previous.SOURCES.add(Path(module.__file__).resolve())
    previous.SOURCES.add(Path(__file__).resolve())
    sources = {str(path.relative_to(ROOT)): previous.digest(path) for path in sorted(previous.SOURCES)}
    active = ROOT / "Overleaf/arxiv_pics"
    unaffected = {str(path.relative_to(ROOT)): previous.digest(path) for path in sorted(active.glob("*.pdf"))
                  if path.name != f"{NAME}.pdf"}
    original, _ = render(output / "original-labels", False, values, truth, channels)
    revised, panels = render(output / "explicit-feature-labels", True, values, truth, channels)
    assert original == revised, "Figure 15's coordinates or axis limits changed"
    prior_manifest_path = ROOT / "Overleaf/archive/Graphs/v5_spectral_pending122_20261114/manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text())
    assert revised[STEM] == prior_manifest["coordinate_signatures"][STEM], "Plot differs from archived Figure 15"
    assert sources == {str(path.relative_to(ROOT)): previous.digest(path) for path in sorted(previous.SOURCES)}
    products = []
    for suffix in ("pdf", "png"):
        target = output / f"{NAME}.{suffix}"
        shutil.copy2(output / "explicit-feature-labels" / f"{STEM}.{suffix}", target)
        if args.publish and suffix == "pdf":
            shutil.copy2(target, active / target.name)
        products.append({"file": target.name, "sha256": previous.digest(target), "bytes": target.stat().st_size})
    assert unaffected == {str(path.relative_to(ROOT)): previous.digest(path) for path in sorted(active.glob("*.pdf"))
                          if path.name != f"{NAME}.pdf"}
    manifest = {
        "purpose": "Figure 15 full variable names and panel letters only",
        "source_cohort": prior_manifest["source_cohort"],
        "planned_evaluation": prior_manifest["planned_evaluation"],
        "sources": sources,
        "prior_manifest": {"path": str(prior_manifest_path.relative_to(ROOT)), "sha256": previous.digest(prior_manifest_path)},
        "panels": panels,
        "figure_name": NAME,
        "plotted_coordinates_identical": True,
        "coordinate_signatures": revised,
        "matches_prior_figure15_coordinate_signature": True,
        "source_exports_unchanged": True,
        "other_active_figures_unchanged": True,
        "other_active_figure_hashes": unaffected,
        "new_evaluations_or_metric_calculations": False,
        "figures": products,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.publish:
        archive = ROOT / "Overleaf/archive/Graphs/v5_fig15_features_20261114"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / "manifest.json", archive / "manifest.json")
        shutil.copy2(Path(__file__), archive / Path(__file__).name)
        (archive / "README.md").write_text(
            "# Figure 15 explicit feature labels\n\n"
            "Only Figure 15 is republished: panel letters and full field names identify "
            "2 m temperature, 500 hPa geopotential, 700 hPa specific humidity, and "
            "six-hour accumulated precipitation. Additional vertical spacing accommodates "
            "the two-line titles. Physical units and the provisional evaluation heading "
            "are preserved.\n\n"
            "The manifest verifies every field against the frozen NPZ channel metadata, "
            "each displayed curve against its saved values, and the complete coordinate "
            "signature against the preceding Figure 15 archive. Other active PDFs are "
            "hash-checked unchanged. No evaluations or metric calculations are run.\n\n"
            "Values remain the frozen 36-date interim export pending replacement by the "
            "full 122-date evaluation. Run this generator in the parent Weather_global "
            "workspace through scripts/graphcast_env.sh with --publish.\n"
        )
    print(json.dumps({"output": str(output), "pdfs": 1, "coordinates_identical": True,
                      "matches_prior_figure15": True, "other_figures_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
