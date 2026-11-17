#!/usr/bin/env python3
"""Appendix E figures on four common dates with the matched reset baseline.

Reads saved exact losses and spherical moments only. No forecasts, training,
filter fitting, bootstrap draws, or amplitude rescaling are performed. The
four compared predictions are raw GraphCast, raw carry/carry, filtered
GraphCast and raw reset/reset; historical outputs remain unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_v4_smoothing_paper_20261114 as base


ROOT = base.ROOT
NAME = "v4_appendix_e_four_models_20261114"
MODELS = [("GC", "GraphCast", "#2864A5"),
          ("M1", "Carry/carry (M1)", "#D97720"),
          ("GC_filter", "GC + optimal filter", "#238B8E"),
          ("reset", "Reset/reset", "#9467BD")]
SELECTED_LEADS = (0, 19, 39)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def style(index, metric):
    settings = {"color": MODELS[index][2], "lw": 1.35, "label": MODELS[index][1]}
    if metric == "coherence":
        settings.update(ls=(5 if index == 2 else 0, (5, 5)), dash_capstyle="butt")
    return settings


def legend(fig):
    fig.text(.5, .995, "Four matched forecast dates", ha="center", va="top", fontsize=8)
    handles = [Line2D([], [], color=color, lw=1.5, label=label) for _, label, color in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95),
               ncol=2, frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.3)


def draw_loss(axis, values, *, relative=False):
    for mi in range(4):
        curve = 100 * (1 - values[:, mi] / values[:, 0]) if relative else values[:, mi]
        axis.plot(base.DAYS, curve, **style(mi, "loss"))
    axis.axvline(5, color=".5", ls=":", lw=.7)
    if relative:
        axis.axhline(0, color=".4", ls=":", lw=.6)
    axis.set(xlim=(.25, 10), xlabel="Forecast lead (days)", xticks=[1, 3, 5, 7, 10],
             ylabel="Loss reduction vs raw GC (%)" if relative else "Exact GraphCast loss")
    axis.grid(alpha=.18, linewidth=.5)


def draw_spectrum(axis, curves, metric, *, top=True):
    for mi in range(4):
        axis.plot(base.DEGREE, curves[mi], **style(mi, metric))
    axis.axhline(1., color=".4", ls=":", lw=.6)
    lower = min(-.025, float(curves.min()) - .025) if metric == "coherence" else 0
    axis.set(ylim=(lower, max(1.05, float(curves.max()) * 1.04)),
             ylabel="Coherence" if metric == "coherence" else "Amplitude ratio")
    base.spatial_axis(axis, top=top)


def make_figures(values, channels, images):
    outputs = []
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.65))
    draw_loss(axes[0], values["loss"])
    draw_loss(axes[1], values["loss"], relative=True)
    legend(fig)
    fig.subplots_adjust(left=.095, right=.965, bottom=.19, top=.745, wspace=.35)
    outputs += base.export(fig, images, "smoothing_loss_four_models")

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.9))
    for axis, metric in zip(axes, ("amplitude", "coherence")):
        draw_spectrum(axis, values[metric][19], metric)
    legend(fig)
    fig.subplots_adjust(left=.09, right=.965, bottom=.18, top=.61, wspace=.30)
    outputs += base.export(fig, images, "smoothing_spectra_day5_four_models")

    for metric in ("amplitude", "coherence"):
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 4.75))
        for axis, (short, variable, pressure) in zip(axes.flat, base.FEATURES):
            ci = channels.index([variable, pressure])
            draw_spectrum(axis, values[metric + "_by_channel"][19, :, ci], metric, top=False)
            axis.set_title(short, pad=5)
            mismatch_gc, mismatch_m1 = values["power_error_by_channel"][19, :2, ci]
            high = metric == "amplitude" and short == "Z500"
            axis.text(.035, .94 if high else .06,
                      f"Mismatch: M1 {mismatch_m1:.1f}% | GC {mismatch_gc:.1f}%",
                      transform=axis.transAxes, fontsize=7, ha="left", va="top" if high else "bottom",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
        legend(fig)
        fig.subplots_adjust(left=.09, right=.965, bottom=.105, top=.81, wspace=.32, hspace=.47)
        outputs += base.export(fig, images, f"smoothing_features_{metric}_day5_four_models")

    fig, axes = plt.subplots(3, 2, figsize=(6.25, 7.2))
    for row, lead in enumerate(SELECTED_LEADS):
        for col, metric in enumerate(("amplitude", "coherence")):
            draw_spectrum(axes[row, col], values[metric][lead], metric, top=row == 0)
            axes[row, col].set_title("6 hours" if lead == 0 else f"{base.DAYS[lead]:g} days",
                                     pad=41 if row == 0 else 5)
    legend(fig)
    fig.subplots_adjust(left=.095, right=.965, bottom=.07, top=.80, wspace=.32, hspace=.45)
    outputs += base.export(fig, images, "smoothing_spectra_6h_5d_10d_four_models")
    return outputs


def main():
    data = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
    images = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
    data.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    filters = json.loads(base.FILTERS.read_text())
    layout = filters["layout"]
    channels = layout["channels"]
    weights = np.asarray(layout["weights"], dtype=float)
    expected = []
    for variable, pressure in channels:
        variable_weight = .1 if variable in ("10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "total_precipitation_6hr") else 1.
        levels = [p for name, p in channels if name == variable and p is not None]
        expected.append(variable_weight * pressure / sum(levels) if pressure is not None else variable_weight)
    np.testing.assert_allclose(weights, expected, rtol=1e-14, atol=1e-14)
    weights /= weights.sum()
    source, arrays, pins = base.load(base.PREVIEW / "gc_m1_reference", 4)
    reset_source, reset, reset_pins = base.load(base.PREVIEW / "report", 4)
    if (source["initialization_ids"] != base.PREVIEW_IDS or reset_source["initialization_ids"] != base.PREVIEW_IDS
            or source["layout"] != layout or reset_source["layout"] != layout
            or source["branches"] != ["GC", "M1"] or source["variants"] != ["raw", "projected", "optimal"]
            or reset_source["branches"] != ["GC", "R_matched"]
            or arrays["prediction_power"].shape != (40, 2, 3, 83, 180)
            or reset["prediction_power"].shape != (40, 2, 83, 180)
            or source["campaign_hash"] != filters["campaign_hash"]
            or reset_source["checkpoint"]["sha256"] != base.RESET_SHA
            or not reset_source["reset_state_every_step"]):
        raise ValueError("Unexpected common cohort, model, checkpoint or channel layout")
    values = base.prepare(source, arrays, weights, reset=reset)
    np.testing.assert_array_equal(values["prediction_power"][:, 1], arrays["prediction_power"][:, 1, 0])
    np.testing.assert_array_equal(values["prediction_power"][:, 3], reset["prediction_power"][:, 1])
    for mi, source_loss in enumerate((arrays["exact_gc_loss"][:, 0, 0], arrays["exact_gc_loss"][:, 1, 0],
                                      arrays["exact_gc_loss"][:, 0, 2], reset["exact_gc_loss"][:, 1])):
        np.testing.assert_array_equal(values["loss"][:, mi], source_loss)

    contrasts = base.contrasts(values)
    previous_path = ROOT / "plots/analyze_models/data/resolution_eval/v4_memory_spectra_appendix_20261114/manifest.json"
    previous = json.loads(previous_path.read_text())
    assert contrasts == previous["contrasts"], "Previously reported four-date scores changed"
    table_rows = []
    for mi, (model_id, label, _) in enumerate(MODELS):
        table_rows.append({"forecast_count": 4, "lead_count": 40, "model": model_id, "label": label,
                           "exact_gc_rollout_loss": contrasts["rollout"]["exact_loss_by_model"][model_id],
                           "power_mismatch_pct": contrasts["rollout"]["power_error_by_model_pct"][model_id]})
    loss_rows, spectral_rows, feature_rows, summary_rows = [], [], [], []
    for lead in range(40):
        for mi, (model_id, label, _) in enumerate(MODELS):
            loss_rows.append({"forecast_count": 4, "lead_hours": (lead + 1) * 6, "model": model_id, "label": label,
                              "exact_gc_loss": values["loss"][lead, mi],
                              "reduction_vs_raw_gc_pct": 100 * (1 - values["loss"][lead, mi] / values["loss"][lead, 0])})
    for lead in SELECTED_LEADS:
        for mi, (model_id, _, _) in enumerate(MODELS):
            for di, degree in enumerate(base.DEGREE):
                spectral_rows.append({"forecast_count": 4, "lead_hours": (lead + 1) * 6, "model": model_id, "degree": degree,
                                      "weighted_amplitude_ratio": values["amplitude"][lead, mi, di],
                                      "weighted_coherence": values["coherence"][lead, mi, di]})
            for ci, (variable, pressure) in enumerate(channels):
                summary_rows.append({"forecast_count": 4, "lead_hours": (lead + 1) * 6, "model": model_id,
                                     "variable": variable, "pressure_hpa": pressure,
                                     "power_mismatch_equal_degree_pct": values["power_error_by_channel"][lead, mi, ci],
                                     "power_mismatch_truth_power_weighted_pct": values["power_weighted_error_by_channel"][lead, mi, ci],
                                     "coherence_equal_degree_mean": values["coherence_by_channel"][lead, mi, ci].mean()})
    for short, variable, pressure in base.FEATURES:
        ci = channels.index([variable, pressure])
        for mi, (model_id, _, _) in enumerate(MODELS):
            for di, degree in enumerate(base.DEGREE):
                feature_rows.append({"forecast_count": 4, "lead_hours": 120, "model": model_id, "field": short,
                                     "degree": degree, "amplitude_ratio": values["amplitude_by_channel"][19, mi, ci, di],
                                     "coherence": values["coherence_by_channel"][19, mi, ci, di]})
    exports = {"table7_accuracy_power.csv": table_rows, "loss_by_lead.csv": loss_rows,
               "weighted_spectral_curves.csv": spectral_rows, "feature_day5_curves.csv": feature_rows,
               "all_channels_selected_leads_summary.csv": summary_rows}
    for filename, rows in exports.items():
        write_csv(data / filename, rows)
    array_metadata = {"cohort": "four_matched_dates", "forecast_count": 4,
                      "initialization_ids": base.PREVIEW_IDS, "models": [x[0] for x in MODELS],
                      "channels": channels, "lead_hours": [6, 120, 240],
                      "array_axes": "lead, model, channel, degree; truth omits model", "degrees": "1..179"}
    np.savez_compressed(data / "all_channels_selected_leads_curves.npz", degree=base.DEGREE,
                        amplitude_ratio=values["amplitude_by_channel"][list(SELECTED_LEADS)],
                        coherence=values["coherence_by_channel"][list(SELECTED_LEADS)],
                        prediction_power=values["prediction_power"][list(SELECTED_LEADS), :, :, 1:],
                        truth_power=arrays["truth_power"][list(SELECTED_LEADS), :, 1:],
                        metadata=json.dumps(array_metadata))
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        outputs = make_figures(values, channels, images)
    gc_raw = float(values["loss"][:, 0].mean())
    gc_reset_rerun = float(reset["exact_gc_loss"][:, 0].mean())
    summary = {"created": "2026-11-14", "description": "All Appendix E comparisons share four models and four forecast dates",
               "cohort": {"count": 4, "initialization_ids": base.PREVIEW_IDS, "lead_count": 40},
               "models": [{"id": model_id, "label": label} for model_id, label, _ in MODELS],
               "contrasts": contrasts, "table7_rows": table_rows,
               "projection_control": {"model": "GC", "raw_exact_rollout_loss": gc_raw,
                                      "unit_gain_projected_exact_rollout_loss": float(arrays["exact_gc_loss"][:, 0, 1].mean()),
                                      "optimal_filter_exact_rollout_loss": float(values["loss"][:, 2].mean())},
               "numerical_diagnostics": {"reset_report": reset_source.get("numerical_diagnostics", {}),
                                         "reference_report": source.get("numerical_diagnostics", {}),
                                         "repeated_gc_rollout_loss": gc_reset_rerun,
                                         "gc_reference_rollout_loss": gc_raw,
                                         "repeated_gc_relative_difference_pct": 100 * (gc_reset_rerun / gc_raw - 1)},
               "definitions": {"loss": "Exact saved native-grid GraphCast weighted MSE; pool losses before ratios",
                               "power_mismatch": "100*mean abs(date-pooled Pf/Py-1), equally over 40 leads, 83 channels and degrees1..179",
                               "amplitude_coherence": "Per-channel ratios from date-pooled moments, averaged with normalized variable/pressure weights",
                               "filter": "GraphCast output-only filter fitted on 32 starts in2022; no feedback alteration",
                               "coherence_dashes": "GC/filtered-GC use complementary(5,5) phases0/5; no offsets, arrows or markers",
                               "field_selection": "T2m,Z500,Q700,TP6h, fixed before this presentation revision",
                               "feature_annotations": "Same equal-degree mismatch for raw M1 and raw GC in amplitude/coherence plots",
                               "uncertainty": "Four-date fixed-checkpoint point estimates; no new confidence intervals or training-seed variation",
                               "raw_outputs": "Carry/carry and reset/reset remain unfiltered, without amplitude rescaling"},
               "checks": {**values["checks"], "four_models_in_every_figure": True,
                          "same_truth_and_dates": True, "retained_four_date_contrasts_identical": True,
                          "raw_carry_reset_powers_unchanged": True, "normalized_weight_sum": float(weights.sum())},
               "sources": pins + reset_pins + [base.pin(base.FILTERS), base.pin(previous_path),
                                                base.pin(Path(__file__)), base.pin(Path(base.__file__))],
               "figures": outputs,
               "curves": [base.pin(data / name) for name in [*exports, "all_channels_selected_leads_curves.npz"]]}
    (data / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    manifest = dict(summary)
    manifest["summary"] = base.pin(data / "summary.json")
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"images": str(images.relative_to(ROOT)), "data": str(data.relative_to(ROOT)),
                      "table7": table_rows, "projection_control": summary["projection_control"]}, indent=2))


if __name__ == "__main__":
    main()
