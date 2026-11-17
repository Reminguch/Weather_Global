#!/usr/bin/env python3
"""Manuscript appendix: original carry/reset spectra on four matched dates.

Only saved physical moments and exact native-grid losses are read. There are
no forecasts, refitted filters, equal-power rescaling, or uncertainty draws.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_v4_smoothing_paper_20261114 as base

ROOT = base.ROOT
NAME = "v4_memory_spectra_appendix_20261114"
MODELS = [("GC", "GraphCast", "#2864A5"),
          ("M1", "Carry/carry (M1)", "#D97720"),
          ("GC_filter", "GC + optimal filter", "#238B8E"),
          ("reset", "Reset/reset", "#9467BD")]
FIELDS = [("T2m", "2m_temperature", None, 1., "K^2", r"K$^2$"),
          ("Z500", "geopotential", 500., 1., "m^4 s^-4", r"m$^4$ s$^{-4}$"),
          ("Q700", "specific_humidity", 700., 1e6, "(g kg^-1)^2", r"(g kg$^{-1}$)$^2$"),
          ("TP6h", "total_precipitation_6hr", None, 1e6, "mm^2", r"mm$^2$")]
LEADS = (19, 39)


def line_style(index, metric):
    settings = {"color": MODELS[index][2], "lw": 1.2, "label": MODELS[index][1]}
    if metric == "coherence":
        settings.update(ls=(5 if index == 2 else 0, (5, 5)), dash_capstyle="butt")
    return settings


def legend(fig, *, truth=False):
    handles = [Line2D([], [], color=color, lw=1.4, label=label) for _, label, color in MODELS]
    if truth:
        handles = [Line2D([], [], color="black", lw=1.2, label="ERA5")] + handles
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .967),
               ncol=3 if truth else 2, frameon=False, fontsize=7.5, handlelength=2., columnspacing=1.3)


def draw_loss(axis, values, *, relative=False):
    for index in range(4):
        curve = 100 * (1 - values[:, index] / values[:, 0]) if relative else values[:, index]
        axis.plot(base.DAYS, curve, **line_style(index, "loss"))
    axis.axvline(5, color=".5", ls=":", lw=.7)
    if relative:
        axis.axhline(0, color=".4", ls=":", lw=.6)
    axis.set(xlim=(.25, 10), xlabel="Forecast lead (days)", xticks=[1, 3, 5, 7, 10],
             ylabel="Loss reduction vs raw GC (%)" if relative else "Exact GraphCast loss")
    axis.grid(alpha=.18, linewidth=.5)


def draw_spectrum(axis, curves, metric):
    for index in range(4):
        axis.plot(base.DEGREE, curves[index], **line_style(index, metric))
    axis.axhline(1, color=".4", ls=":", lw=.6)
    axis.set(ylim=(min(-.025, float(curves.min()) - .025) if metric == "coherence" else 0,
                   max(1.05, float(curves.max()) * 1.04)),
             ylabel="Coherence" if metric == "coherence" else "Amplitude ratio")
    base.spatial_axis(axis, top=True)


def make_figures(values, truth, channels, images):
    outputs = []
    fig, axes = plt.subplots(3, 2, figsize=(6.25, 7.1))
    draw_loss(axes[0, 0], values["loss"])
    draw_loss(axes[0, 1], values["loss"], relative=True)
    for row, lead in enumerate(LEADS, start=1):
        for col, metric in enumerate(("amplitude", "coherence")):
            draw_spectrum(axes[row, col], values[metric][lead], metric)
            axes[row, col].set_title(f"Day {int(base.DAYS[lead])}", pad=37)
    fig.text(.5, .995, "Matched four-date preview", ha="center", va="top", fontsize=8.5)
    legend(fig)
    fig.subplots_adjust(left=.10, right=.965, bottom=.065, top=.865, wspace=.36, hspace=.95)
    outputs += base.export(fig, images, "memory_loss_amplitude_coherence_preview4")

    # Use the same physical limits at both leads so their spectral amplitudes
    # can be compared directly. Each panel retains every positive degree.
    limits = {}
    for short, variable, pressure, conversion, _, _ in FIELDS:
        ci = channels.index([variable, pressure])
        curves = np.concatenate([values["prediction_power"][list(LEADS), :, ci, 1:].reshape(-1, 179),
                                 truth[list(LEADS), ci, 1:]], axis=0) * conversion
        limits[short] = (float(curves.min()) / 1.5, float(curves.max()) * 1.5)
    for lead in LEADS:
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 4.65))
        for fi, (axis, field) in enumerate(zip(axes.flat, FIELDS)):
            short, variable, pressure, conversion, _, unit_label = field
            ci = channels.index([variable, pressure])
            axis.plot(base.DEGREE, conversion * truth[lead, ci, 1:], color="black", lw=1.1, label="ERA5")
            for mi in range(4):
                axis.plot(base.DEGREE, conversion * values["prediction_power"][lead, mi, ci, 1:], **line_style(mi, "power"))
            axis.set_yscale("log")
            axis.set(ylim=limits[short], ylabel=f"Power per degree ({unit_label})")
            axis.set_title(short, pad=37 if fi < 2 else 5)
            base.spatial_axis(axis, top=fi < 2)
        fig.text(.5, .995, f"Day {int(base.DAYS[lead])} · matched four-date preview", ha="center", va="top", fontsize=8.5)
        legend(fig, truth=True)
        fig.subplots_adjust(left=.115, right=.965, bottom=.105, top=.735, wspace=.42, hspace=.48)
        outputs += base.export(fig, images, f"memory_physical_power_day{int(base.DAYS[lead])}_preview4")
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
        weight = .1 if variable in ("10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "total_precipitation_6hr") else 1.
        levels = [p for name, p in channels if name == variable and p is not None]
        expected.append(weight * pressure / sum(levels) if pressure is not None else weight)
    np.testing.assert_allclose(weights, expected, rtol=1e-14, atol=1e-14)
    weights /= weights.sum()
    source, arrays, source_pins = base.load(base.PREVIEW / "gc_m1_reference", 4)
    reset_source, reset, reset_pins = base.load(base.PREVIEW / "report", 4)
    if (source["initialization_ids"] != base.PREVIEW_IDS or reset_source["initialization_ids"] != base.PREVIEW_IDS
            or source["layout"] != layout or reset_source["layout"] != layout
            or source["branches"] != ["GC", "M1"] or source["variants"] != ["raw", "projected", "optimal"]
            or reset_source["branches"] != ["GC", "R_matched"]
            or arrays["prediction_power"].shape != (40, 2, 3, 83, 180)
            or reset["prediction_power"].shape != (40, 2, 83, 180)
            or reset_source["checkpoint"]["sha256"] != base.RESET_SHA
            or not reset_source["reset_state_every_step"]):
        raise ValueError("Unexpected cohort, channel layout, model order, or reset checkpoint/policy")
    values = base.prepare(source, arrays, weights, reset=reset)
    truth = arrays["truth_power"]
    # The raw model powers are copied directly, with no normalization to ERA5.
    np.testing.assert_array_equal(values["prediction_power"][:, 1], arrays["prediction_power"][:, 1, 0])
    np.testing.assert_array_equal(values["prediction_power"][:, 3], reset["prediction_power"][:, 1])
    loss_rows, spectrum_rows, power_rows, channel_rows = [], [], [], []
    for lead in range(40):
        for mi, (model_id, label, _) in enumerate(MODELS):
            loss_rows.append({"cohort": "preview4", "forecast_count": 4, "lead_hours": 6 * (lead + 1),
                              "model": model_id, "model_label": label, "exact_gc_loss": values["loss"][lead, mi],
                              "loss_reduction_vs_same_cohort_raw_gc_pct": 100 * (1 - values["loss"][lead, mi] / values["loss"][lead, 0])})
    for lead in LEADS:
        for mi, (model_id, label, _) in enumerate(MODELS):
            for di, degree in enumerate(base.DEGREE):
                spectrum_rows.append({"cohort": "preview4", "forecast_count": 4, "lead_hours": 6 * (lead + 1),
                                      "model": model_id, "model_label": label, "degree": degree,
                                      "weighted_amplitude_ratio": values["amplitude"][lead, mi, di],
                                      "weighted_coherence": values["coherence"][lead, mi, di]})
            for ci, (variable, pressure) in enumerate(channels):
                channel_rows.append({"cohort": "preview4", "forecast_count": 4, "lead_hours": 6 * (lead + 1),
                                     "model": model_id, "variable": variable, "pressure_hpa": pressure,
                                     "power_mismatch_equal_degree_pct": values["power_error_by_channel"][lead, mi, ci],
                                     "power_mismatch_truth_power_weighted_pct": values["power_weighted_error_by_channel"][lead, mi, ci],
                                     "coherence_equal_degree_mean": values["coherence_by_channel"][lead, mi, ci].mean()})
        for short, variable, pressure, conversion, unit, _ in FIELDS:
            ci = channels.index([variable, pressure])
            for di, degree in enumerate(base.DEGREE):
                for mi, (model_id, label, _) in enumerate(MODELS):
                    power_rows.append({"cohort": "preview4", "forecast_count": 4, "lead_hours": 6 * (lead + 1),
                                       "field": short, "model": model_id, "model_label": label, "degree": degree,
                                       "power_units": unit, "forecast_power": conversion * values["prediction_power"][lead, mi, ci, degree],
                                       "era5_power": conversion * truth[lead, ci, degree]})
    rows_by_file = {"exact_loss_by_lead.csv": loss_rows, "weighted_amplitude_coherence.csv": spectrum_rows,
                    "physical_power_selected_fields.csv": power_rows, "all_channels_selected_leads_summary.csv": channel_rows}
    for filename, rows in rows_by_file.items():
        base.write_csv(data / filename, rows)
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False}):
        figures = make_figures(values, truth, channels, images)
    manifest = {"created": "2026-11-14", "description": "Original carry/carry and reset/reset outputs: matched four-date preview",
                "cohort": {"count": 4, "initialization_ids": base.PREVIEW_IDS, "status": "preliminary_four_date_preview"},
                "models": [{"id": model_id, "label": label} for model_id, label, _ in MODELS],
                "sources": source_pins + reset_pins + [base.pin(base.FILTERS), base.pin(Path(__file__)), base.pin(Path(base.__file__)),
                                                        base.pin(ROOT / "src/analysis/optimal_spherical_filter.py")],
                "definitions": {"loss": "Saved exact native-grid GraphCast weighted MSE; raw unfiltered M1 and reset forecasts",
                                "power": "Date-pooled physical power per total degree: sum_m |coefficient(l,m)|^2 with positive m doubled; normalized continuous spherical mean-square convention",
                                "degree_range": "1..179; degree zero excluded; no climatological-mean subtraction",
                                "amplitude_ratio": "Per-channel sqrt(date-pooled forecast power / date-pooled truth power), then loss variable/pressure-weighted average",
                                "coherence": "Per-channel pooled real cross-power / sqrt(pooled forecast power * pooled truth power), then the same weighted average",
                                "channel_weights": "Original GC variable weights and pressure-proportional level weights, normalized to sum one; no difference-stddev weighting of dimensionless ratios",
                                "units": "T2m K^2; Z500 geopotential m^4 s^-4; Q700 converted kg/kg to g/kg before squaring (power x1e6); TP6h converted m to mm before squaring (power x1e6)",
                                "filter": "Existing GC filter fitted on 32 development starts in 2022; output-only filtering; raw M1/reset are unchanged",
                                "equal_power_rescaling": "Not applied; raw carry/reset power arrays are exactly equal to the saved input arrays",
                                "uncertainty": "Four dates and fixed checkpoints; no spectral confidence intervals or training-seed uncertainty",
                                "channel_mismatch": "Equal-degree 100*mean_l |Pf/Py-1| and truth-power-weighted 100*sum_l |Pf-Py|/sum_l Py, l=1..179"},
                "checks": {**values["checks"], "original_raw_model_powers_unchanged": True, "same_truth_and_dates": True,
                           "normalized_weight_sum": float(weights.sum())},
                "preserved_numerical_diagnostics": reset_source.get("numerical_diagnostics", {}),
                "contrasts": base.contrasts(values),
                "figures": figures, "curves": [base.pin(data / filename) for filename in rows_by_file]}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"images": str(images.relative_to(ROOT)), "figures": len(figures),
                      "checks": manifest["checks"], "contrasts": manifest["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
