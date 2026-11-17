#!/usr/bin/env python3
"""Frozen 36-date Appendix E figures; saved losses/moments only.

No forecasts, filter fitting, repooling of newly completed shards, bootstrap
draws, or amplitude matching. Historical four-date products remain unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

import plot_v4_smoothing_paper_20261114 as base
import plot_v4_appendix_e_four_models_20261114 as layout
import plot_v4_memory_spectra_appendix_20261114 as physical

ROOT = base.ROOT
NAME = "v4_appendix_e_matched36_20261114"
SOURCE = ROOT / "artifacts/evaluations/v24_matched_spectra_completed_20261114"
COHORT_SHA = "be9f3aa4c8e9fb9ae6bd132a86f66708334c151e033d9bc91d576aab2471fbdc"
SCORES = ROOT / "plots/analyze_models/data/resolution_eval/v24_literature_spectral_matched36_20261114/comparison_scores.csv"
MODELS = layout.MODELS
SCORE_IDS = ("GC", "M1", "GC_filtered", "R_matched")
WINDOWS = (("rollout", slice(None)), ("day5", slice(19, 20)), ("day10", slice(39, 40)))
LEADS = (0, 19, 39)
STEMS = (
    "fig11_smoothing_accuracy_matched36", "fig12_smoothing_amplitude_coherence_matched36",
    "fig13_smoothing_feature_amplitude_matched36", "fig14_smoothing_feature_coherence_matched36",
    "fig15_smoothing_lead_spectra_matched36", "fig16_accuracy_power_summary_matched36",
    "fig17_memory_power_day5_matched36", "fig18_memory_power_day10_matched36")


def legend(fig, *, truth=False, day=None):
    title = "36 matched forecast dates" if day is None else f"Day {day} · 36 matched forecast dates"
    fig.text(.5, .995, title, ha="center", va="top", fontsize=8)
    handles = [Line2D([], [], color=color, lw=1.5, label=label) for _, label, color in MODELS]
    if truth:
        handles.insert(0, Line2D([], [], color="black", lw=1.2, label="ERA5"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .967 if truth else .95),
               ncol=3 if truth else 2, frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.3)


def make_summary(values, images):
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.5))
    order = (2, 1, 3)  # Filtered GC, carry/carry, reset/reset.
    windows_y = np.array([2., 1., 0.])
    for axis, key, title, xlabel in zip(
            axes, ("loss", "power_error_by_lead"), ("Forecast accuracy", "Spectral power fidelity"),
            ("MSE reduction vs raw GC (%)", "Power-mismatch reduction vs raw GC (%)")):
        matrix = np.stack([values[key][indices].mean(axis=0) for _, indices in WINDOWS])
        reductions = 100 * (1 - matrix / matrix[:, :1])
        span = max(float(np.ptp(reductions)), 1.)
        for row, model in enumerate(order):
            y = windows_y + (.22 - .22 * row)
            x = reductions[:, model]
            axis.barh(y, x, height=.18, color=MODELS[model][2], zorder=3)
            for yy, xx in zip(y, x):
                axis.annotate(f"{xx:+.2f}", (xx, yy), xytext=(3 if xx >= 0 else -3, 0),
                              textcoords="offset points", va="center", ha="left" if xx >= 0 else "right", fontsize=7)
        axis.axvline(0, color=MODELS[0][2], lw=1.1, ls=(0, (4, 3)), zorder=4)
        lower, upper = min(0., float(reductions.min())), max(0., float(reductions.max()))
        axis.set(xlim=(lower - .28 * span, upper + .28 * span), ylim=(-.5, 2.5),
                 yticks=windows_y, yticklabels=["Rollout", "Day 5", "Day 10"], xlabel=xlabel)
        axis.set_title(title, fontsize=8, pad=7)
        axis.grid(axis="x", alpha=.18, linewidth=.5)
        axis.set_axisbelow(True)
    handles = [Line2D([], [], color=MODELS[0][2], lw=1.2, ls=(0, (4, 3)), label="Raw GraphCast (0%)")]
    handles += [Patch(facecolor=MODELS[i][2], label=MODELS[i][1]) for i in (1, 2, 3)]
    fig.text(.5, .995, "36 matched forecast dates", ha="center", va="top", fontsize=8)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .95), ncol=2,
               frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.3)
    fig.text(.5, .025, "Positive: improvement. Negative: deterioration. The panels measure distinct quantities.",
             ha="center", fontsize=7)
    fig.subplots_adjust(left=.105, right=.98, bottom=.18, top=.735, wspace=.43)
    return base.export(fig, images, STEMS[5])


def make_figures(values, truth, channels, images):
    outputs = []
    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.65))
    layout.draw_loss(axes[0], values["loss"])
    layout.draw_loss(axes[1], values["loss"], relative=True)
    legend(fig)
    fig.subplots_adjust(left=.095, right=.965, bottom=.19, top=.745, wspace=.35)
    outputs += base.export(fig, images, STEMS[0])

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.9))
    for axis, metric in zip(axes, ("amplitude", "coherence")):
        layout.draw_spectrum(axis, values[metric][19], metric)
    legend(fig)
    fig.subplots_adjust(left=.09, right=.965, bottom=.18, top=.61, wspace=.30)
    outputs += base.export(fig, images, STEMS[1])

    for fi, metric in enumerate(("amplitude", "coherence")):
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 4.75))
        for axis, (short, variable, pressure) in zip(axes.flat, base.FEATURES):
            ci = channels.index([variable, pressure])
            layout.draw_spectrum(axis, values[metric + "_by_channel"][19, :, ci], metric, top=False)
            axis.set_title(short, pad=5)
            gc, carry = values["power_error_by_channel"][19, :2, ci]
            high = metric == "amplitude" and short == "Z500"
            axis.text(.035, .94 if high else .06, f"Mismatch: M1 {carry:.1f}% | GC {gc:.1f}%",
                      transform=axis.transAxes, fontsize=7, ha="left", va="top" if high else "bottom",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
        legend(fig)
        fig.subplots_adjust(left=.09, right=.965, bottom=.105, top=.81, wspace=.32, hspace=.47)
        outputs += base.export(fig, images, STEMS[2 + fi])

    fig, axes = plt.subplots(3, 2, figsize=(6.25, 7.2))
    for row, lead in enumerate(LEADS):
        for col, metric in enumerate(("amplitude", "coherence")):
            layout.draw_spectrum(axes[row, col], values[metric][lead], metric, top=row == 0)
            axes[row, col].set_title("6 hours" if lead == 0 else f"{base.DAYS[lead]:g} days",
                                    pad=41 if row == 0 else 5)
    legend(fig)
    fig.subplots_adjust(left=.095, right=.965, bottom=.07, top=.80, wspace=.32, hspace=.45)
    outputs += base.export(fig, images, STEMS[4])
    outputs += make_summary(values, images)

    limits = {}
    for short, variable, pressure, conversion, _, _ in physical.FIELDS:
        ci = channels.index([variable, pressure])
        curves = np.concatenate([values["prediction_power"][[19, 39], :, ci, 1:].reshape(-1, 179),
                                 truth[[19, 39], ci, 1:]], axis=0) * conversion
        limits[short] = (float(curves.min()) / 1.5, float(curves.max()) * 1.5)
    for li, lead in enumerate((19, 39)):
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 4.65))
        for fi, (axis, field) in enumerate(zip(axes.flat, physical.FIELDS)):
            short, variable, pressure, conversion, _, unit_label = field
            ci = channels.index([variable, pressure])
            axis.plot(base.DEGREE, conversion * truth[lead, ci, 1:], color="black", lw=1.1, label="ERA5")
            for mi in range(4):
                axis.plot(base.DEGREE, conversion * values["prediction_power"][lead, mi, ci, 1:],
                          **physical.line_style(mi, "power"))
            axis.set_yscale("log")
            axis.set(ylim=limits[short], ylabel=f"Power per degree ({unit_label})")
            axis.set_title(short, pad=37 if fi < 2 else 5)
            base.spatial_axis(axis, top=fi < 2)
        legend(fig, truth=True, day=int(base.DAYS[lead]))
        fig.subplots_adjust(left=.115, right=.965, bottom=.105, top=.735, wspace=.42, hspace=.48)
        outputs += base.export(fig, images, STEMS[6 + li])
    return outputs


def main():
    data = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
    images = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
    data.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    cohort_path = SOURCE / "cohort.json"
    if base.pin(cohort_path)["sha256"] != COHORT_SHA:
        raise ValueError("Frozen 36-date cohort changed; do not adopt later completed shards")
    cohort = json.loads(cohort_path.read_text())
    source, arrays, source_pins = base.load(SOURCE / "gc_m1_reference", 36)
    reset_source, reset, reset_pins = base.load(SOURCE / "reset", 36)
    filters = json.loads(base.FILTERS.read_text())
    channels = filters["layout"]["channels"]
    if (cohort["count"] != 36 or source["initialization_ids"] != cohort["initialization_ids"]
            or reset_source["initialization_ids"] != cohort["initialization_ids"]
            or source["status"] != "interim_matched_cohort" or reset_source["status"] != "interim_matched_cohort"
            or source["layout"] != filters["layout"] or reset_source["layout"] != filters["layout"]
            or source["branches"] != ["GC", "M1"] or source["variants"] != ["raw", "projected", "optimal"]
            or reset_source["branches"] != ["GC", "R_matched"]
            or reset_source["checkpoint"]["sha256"] != base.RESET_SHA or not reset_source["reset_state_every_step"]):
        raise ValueError("Unexpected frozen cohort, layout, model, or reset policy")
    weights = np.asarray(filters["layout"]["weights"], dtype=float)
    weights /= weights.sum()
    values = base.prepare(source, arrays, weights, reset=reset)
    truth = arrays["truth_power"]
    np.testing.assert_array_equal(values["prediction_power"][:, 1], arrays["prediction_power"][:, 1, 0])
    np.testing.assert_array_equal(values["prediction_power"][:, 3], reset["prediction_power"][:, 1])
    references = list(csv.DictReader(SCORES.open(newline="")))
    lookup = {(row["window"], row["model"]): row for row in references}
    if len(lookup) != 12 or len(references) != 12:
        raise ValueError("Expected exactly four models in three reference windows")
    scores = []
    for window, indices in WINDOWS:
        loss = values["loss"][indices].mean(axis=0)
        mismatch = values["power_error_by_lead"][indices].mean(axis=0)
        for mi, (model, label, _) in enumerate(MODELS):
            row = {"window": window, "model": model, "label": label, "forecast_count": 36,
                   "lead_count": 40 if window == "rollout" else 1, "exact_gc_loss": loss[mi],
                   "loss_reduction_vs_raw_gc_pct": 100 * (1 - loss[mi] / loss[0]),
                   "power_mismatch_pct": mismatch[mi],
                   "power_mismatch_reduction_vs_raw_gc_pct": 100 * (1 - mismatch[mi] / mismatch[0])}
            ref = lookup[window, SCORE_IDS[mi]]
            for key in ("forecast_count", "lead_count", "exact_gc_loss", "loss_reduction_vs_raw_gc_pct",
                        "power_mismatch_pct", "power_mismatch_reduction_vs_raw_gc_pct"):
                np.testing.assert_allclose(row[key], float(ref[key]), rtol=1e-12, atol=1e-12)
            scores.append(row)
    losses, spectra, feature_curves, powers, channel_scores = [], [], [], [], []
    for lead in range(40):
        for mi, (model, label, _) in enumerate(MODELS):
            losses.append({"forecast_count": 36, "lead_hours": 6 * (lead + 1), "model": model, "label": label,
                           "exact_gc_loss": values["loss"][lead, mi],
                           "loss_reduction_vs_raw_gc_pct": 100 * (1 - values["loss"][lead, mi] / values["loss"][lead, 0])})
    for lead in LEADS:
        for mi, (model, _, _) in enumerate(MODELS):
            for di, degree in enumerate(base.DEGREE):
                spectra.append({"forecast_count": 36, "lead_hours": (lead + 1) * 6, "model": model, "degree": degree,
                                "weighted_amplitude_ratio": values["amplitude"][lead, mi, di],
                                "weighted_coherence": values["coherence"][lead, mi, di]})
            for ci, (variable, pressure) in enumerate(channels):
                channel_scores.append({"forecast_count": 36, "lead_hours": (lead + 1) * 6, "model": model,
                                       "variable": variable, "pressure_hpa": pressure,
                                       "power_mismatch_equal_degree_pct": values["power_error_by_channel"][lead, mi, ci],
                                       "power_mismatch_truth_power_weighted_pct": values["power_weighted_error_by_channel"][lead, mi, ci],
                                       "mean_coherence_equal_degree": values["coherence_by_channel"][lead, mi, ci].mean()})
    for short, variable, pressure, conversion, unit, _ in physical.FIELDS:
        ci = channels.index([variable, pressure])
        for mi, (model, _, _) in enumerate(MODELS):
            for di, degree in enumerate(base.DEGREE):
                feature_curves.append({"forecast_count": 36, "lead_hours": 120, "model": model, "field": short,
                                       "degree": degree, "amplitude_ratio": values["amplitude_by_channel"][19, mi, ci, di],
                                       "coherence": values["coherence_by_channel"][19, mi, ci, di]})
                for lead in (19, 39):
                    powers.append({"forecast_count": 36, "lead_hours": (lead + 1) * 6, "model": model, "field": short,
                                   "degree": degree, "power_units": unit,
                                   "forecast_power": conversion * values["prediction_power"][lead, mi, ci, degree],
                                   "era5_power": conversion * truth[lead, ci, degree]})
    exports = {"comparison_scores.csv": scores, "table7_rollout_scores.csv": scores[:4],
               "loss_by_lead.csv": losses, "weighted_spectral_curves.csv": spectra,
               "feature_day5_curves.csv": feature_curves, "physical_power_selected_fields.csv": powers,
               "all_channels_selected_leads_summary.csv": channel_scores}
    for name, rows in exports.items():
        layout.write_csv(data / name, rows)
    np.savez_compressed(data / "all_channels_selected_leads_curves.npz", degree=base.DEGREE,
                        amplitude_ratio=values["amplitude_by_channel"][list(LEADS)],
                        coherence=values["coherence_by_channel"][list(LEADS)],
                        prediction_power=values["prediction_power"][list(LEADS), :, :, 1:],
                        truth_power=truth[list(LEADS), :, 1:], metadata=json.dumps({
                            "forecast_count": 36, "initialization_ids": cohort["initialization_ids"],
                            "models": [m[0] for m in MODELS], "channels": channels, "lead_hours": [6, 120, 240],
                            "axes": "lead,model,channel,degree; truth omits model", "degrees": "1..179"}))
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figures = make_figures(values, truth, channels, images)
    summary = {"created": "2026-11-14", "cohort": {"count": 36, "initialization_ids": cohort["initialization_ids"],
                "status": "frozen_interim_matched_cohort", "planned_source_count": 92,
                "later_completed_dates": "Excluded; source cohort SHA256 is fixed in this script"},
               "models": [{"id": model, "label": label} for model, label, _ in MODELS],
               "contrasts": base.contrasts(values), "comparison_scores": scores,
               "definitions": {"loss": "Exact saved native-grid GraphCast weighted MSE; pool losses before ratios",
                               "power_mismatch": "100*mean abs(date-pooled Pf/Py-1), equally over channels/degrees1..179 and requested leads",
                               "score_reductions": "100*(1-model_score/raw_GC_score), separately for loss and power mismatch; distinct quantities",
                               "amplitude_coherence": "Per-channel ratios from date-pooled moments, averaged with normalized variable/pressure weights",
                               "physical_power": "Separate physical squared units; Q700 power converted x1e6 to(g/kg)^2 and precipitation x1e6 to mm^2",
                               "filter": "Existing output-only GC filter fitted on32 dates in2022; no refitting or feedback change",
                               "raw_models": "Carry/carry and reset/reset are unfiltered; no amplitude matching",
                               "uncertainty": "Fixed-checkpoint point estimates for36 dates; no new intervals or seed estimates"},
               "checks": {**values["checks"], "raw_carry_reset_powers_unchanged": True,
                          "same_truth_and_dates": True, "frozen36_source_checksum_matches": True,
                          "all12_reference_score_rows_match": True},
               "preserved_numerical_diagnostics": reset_source.get("numerical_diagnostics", {}),
               "projection_control": {"raw_gc_rollout_loss": float(arrays["exact_gc_loss"][:, 0, 0].mean()),
                                      "unit_gain_projected_gc_rollout_loss": float(arrays["exact_gc_loss"][:, 0, 1].mean())},
               "sources": source_pins + reset_pins + [base.pin(cohort_path), base.pin(base.FILTERS), base.pin(SCORES),
                          *[base.pin(Path(module.__file__)) for module in (base, layout, physical)], base.pin(Path(__file__))],
               "figures": figures,
               "curves": [base.pin(data / name) for name in [*exports, "all_channels_selected_leads_curves.npz"]]}
    (data / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    manifest = {**summary, "summary": base.pin(data / "summary.json")}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"images": str(images.relative_to(ROOT)), "figures": len(figures),
                      "data": str(data.relative_to(ROOT)), "checks": summary["checks"],
                      "contrasts": summary["contrasts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
