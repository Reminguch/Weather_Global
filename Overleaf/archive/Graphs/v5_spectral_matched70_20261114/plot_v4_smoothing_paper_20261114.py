#!/usr/bin/env python3
"""Compact v4 smoothing figures from existing 92-date and four-date moments.

No forecasts, model selection, filters, or bootstrap samples are generated.
The four-date reset preview remains separate from the 92-date comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "v4_smoothing_paper_20261114"
FULL = ROOT / "plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113"
PREVIEW = ROOT / "artifacts/evaluations/v24_matched_reset_preview4_20261113"
FILTERS = ROOT / "artifacts/evaluations/v24_optimal_smoothing_v4_20261112/filters.json"
RESET_SHA = "db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6"
PREVIEW_IDS = ["20230101T000000Z", "20230401T000000Z", "20230703T000000Z", "20231001T000000Z"]
MODELS = [("GC", "GraphCast", "#2864A5"), ("M1", "M1", "#D97720"),
          ("GC_filter", "GC + optimal filter", "#238B8E"), ("reset", "Reset/reset", "#9467BD")]
FEATURES = [("T2m", "2m_temperature", None), ("Z500", "geopotential", 500.),
            ("Q700", "specific_humidity", 700.), ("TP6h", "total_precipitation_6hr", None)]
DEGREE = np.arange(1, 180)
DAYS = np.arange(1, 41) / 4


def pin(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load(directory, count):
    report_path, path = directory / "summary.json", directory / "power_pooled.npz"
    report = json.loads(report_path.read_text())
    if report["count"] != count or len(set(report["initialization_ids"])) != count:
        raise ValueError("Unexpected or duplicate forecast dates")
    if pin(path)["sha256"] != report["data"][path.name]["sha256"]:
        raise ValueError("Pooled input fingerprint mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("prediction_power", "truth_power", "cross_power", "error_power", "exact_gc_loss")}
        metadata = json.loads(str(archive["metadata"]))
    if metadata["initialization_ids"] != report["initialization_ids"] or metadata["campaign_hash"] != report["campaign_hash"]:
        raise ValueError("Report and statistics provenance differ")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite moment or loss")
    if np.any(arrays["prediction_power"] <= 0) or np.any(arrays["truth_power"] <= 0) or np.any(arrays["error_power"] < 0):
        raise ValueError("Invalid physical moments")
    return report, arrays, [pin(report_path), pin(path)]


def prepare(source, arrays, weights, *, reset=None):
    keys = ("prediction_power", "cross_power", "error_power")
    values = {key: np.stack([arrays[key][:, 0, 0], arrays[key][:, 1, 0], arrays[key][:, 0, 2]], axis=1) for key in keys}
    values["loss"] = np.stack([arrays["exact_gc_loss"][:, 0, 0], arrays["exact_gc_loss"][:, 1, 0], arrays["exact_gc_loss"][:, 0, 2]], axis=1)
    truth = arrays["truth_power"]
    if reset is not None:
        for key in keys:
            values[key] = np.concatenate([values[key], reset[key][:, 1:2]], axis=1)
        values["loss"] = np.concatenate([values["loss"], reset["exact_gc_loss"][:, 1:2]], axis=1)
        np.testing.assert_allclose(reset["truth_power"], truth, rtol=1e-9, atol=1e-12)
    forecast, cross, error = (values[key] for key in keys)
    ratio = forecast / truth[:, None]
    values["amplitude_by_channel"] = np.sqrt(ratio[..., 1:])
    values["coherence_by_channel"] = (cross / np.sqrt(forecast * truth[:, None]))[..., 1:]
    for metric in ("amplitude", "coherence"):
        values[metric] = np.sum(values[metric + "_by_channel"] * weights[None, None, :, None], axis=2)
    values["power_error_by_channel"] = 100 * np.mean(np.abs(ratio[..., 1:] - 1), axis=-1)
    values["power_weighted_error_by_channel"] = (100 * np.sum(np.abs(forecast[..., 1:] - truth[:, None, :, 1:]), axis=-1)
                                               / np.sum(truth[:, None, :, 1:], axis=-1))
    values["power_error_by_lead"] = 100 * np.mean(np.abs(ratio[..., 1:] - 1), axis=(-1, -2))
    residual = np.abs(forecast + truth[:, None] - 2 * cross - error) / (forecast + truth[:, None])
    closure = float(residual.max())
    if closure > 1e-10:
        raise ValueError("Saved moment identity is inconsistent")
    if source["count"] == 92:
        for row in source["rows"]:
            bi, vi = source["branches"].index(row["model"]), source["variants"].index(row["variant"])
            np.testing.assert_allclose(arrays["exact_gc_loss"][:, bi, vi].mean(), row["exact_gc_rollout_loss"], rtol=1e-12)
    values["checks"] = {"spectral_identity_relative_max": closure,
                        "maximum_absolute_channel_coherence": float(np.abs(values["coherence_by_channel"]).max()),
                        "finite_values_retained": True, "spectral_intervals": "not estimated"}
    return values


def reciprocal(value):
    value = np.asarray(value, dtype=float)
    return np.divide(40030., value, out=np.full_like(value, np.inf), where=value != 0)


def spatial_axis(axis, *, top=True):
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


def style(index, metric):
    result = {"color": MODELS[index][2], "lw": 1.35, "label": MODELS[index][1]}
    if metric == "coherence":
        phase = 5 if index == 2 else 0
        result.update(ls=(phase, (5, 5)), dash_capstyle="butt")
    return result


def legend(fig, count, *, top=.995):
    handles = [Line2D([], [], color=color, lw=1.5, label=name) for _, name, color in MODELS[:count]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, top),
               ncol=count if count == 3 else 2, frameon=False, fontsize=7.8, handlelength=2.2, columnspacing=1.1)


def export(fig, images, stem):
    paths = []
    for suffix in ("pdf", "png"):
        path = images / f"{stem}.{suffix}"
        fig.savefig(path, dpi=230)
        paths.append(pin(path))
    plt.close(fig)
    return paths


def draw_loss(axis, values, *, relative=False):
    for index in range(values.shape[1]):
        line = 100 * (1 - values[:, index] / values[:, 0]) if relative else values[:, index]
        axis.plot(DAYS, line, **style(index, "loss"))
    axis.axvline(5, color=".50", ls=":", lw=.7)
    axis.set(xlim=(.25, 10), xlabel="Forecast lead (days)", xticks=[1, 3, 5, 7, 10])
    axis.set_ylabel("Loss reduction vs raw GC (%)" if relative else "Exact GraphCast loss")
    if relative:
        axis.axhline(0, color=".4", ls=":", lw=.6)
    axis.grid(alpha=.18, linewidth=.5)


def draw_spectrum(axis, values, metric, *, top=True):
    for index in range(values.shape[0]):
        axis.plot(DEGREE, values[index], **style(index, metric))
    axis.axhline(1., color=".4", ls=":", lw=.6)
    lower = min(-.025, float(values.min()) - .025) if metric == "coherence" else 0
    upper = max(1.05, float(values.max()) * 1.04)
    axis.set_ylim(lower, upper)
    axis.set_ylabel("Coherence" if metric == "coherence" else "Amplitude ratio")
    spatial_axis(axis, top=top)


def figures(values92, values4, channels, images, *, features_only=False):
    outputs = []

    def save(fig, stem):
        if features_only and not stem.startswith("smoothing_features_"):
            paths = [images / f"{stem}.{suffix}" for suffix in ("pdf", "png")]
            if not all(path.exists() for path in paths):
                raise ValueError("--features-only requires the other figures to exist")
            plt.close(fig)
            return [pin(path) for path in paths]
        return export(fig, images, stem)

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.65))
    draw_loss(axes[0], values92["loss"])
    draw_loss(axes[1], values92["loss"], relative=True)
    legend(fig, 3)
    fig.subplots_adjust(left=.095, right=.965, bottom=.19, top=.83, wspace=.35)
    outputs += save(fig, "smoothing_loss_92")

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.9))
    for axis, metric in zip(axes, ("amplitude", "coherence")):
        draw_spectrum(axis, values92[metric][19], metric)
    legend(fig, 3)
    fig.subplots_adjust(left=.09, right=.965, bottom=.18, top=.71, wspace=.30)
    outputs += save(fig, "smoothing_spectra_day5_92")

    fig, axes = plt.subplots(2, 2, figsize=(6.25, 5.1))
    draw_loss(axes[0, 0], values4["loss"])
    draw_loss(axes[0, 1], values4["loss"], relative=True)
    for axis, metric in zip(axes[1], ("amplitude", "coherence")):
        draw_spectrum(axis, values4[metric][19], metric)
    fig.text(.5, .995, "Four-date preview", ha="center", va="top", fontsize=8.5)
    legend(fig, 4, top=.96)
    fig.subplots_adjust(left=.095, right=.965, bottom=.095, top=.82, wspace=.34, hspace=.82)
    outputs += save(fig, "smoothing_reset_preview4")

    for metric in ("amplitude", "coherence"):
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 4.75))
        for axis, (short, variable, pressure) in zip(axes.flat, FEATURES):
            index = channels.index([variable, pressure])
            draw_spectrum(axis, values92[metric + "_by_channel"][19, :, index], metric, top=False)
            axis.set_title(short, pad=5)
            mismatch_gc, mismatch_m1 = values92["power_error_by_channel"][19, :2, index]
            high_annotation = metric == "amplitude" and short == "Z500"
            axis.text(.035, .94 if high_annotation else .06,
                      f"Mismatch: M1 {mismatch_m1:.1f}% | GC {mismatch_gc:.1f}%",
                      transform=axis.transAxes, fontsize=7, ha="left", va="top" if high_annotation else "bottom",
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
        legend(fig, 3)
        fig.subplots_adjust(left=.09, right=.965, bottom=.105, top=.86, wspace=.32, hspace=.47)
        outputs += save(fig, f"smoothing_features_{metric}_day5_92")

    fig, axes = plt.subplots(3, 2, figsize=(6.25, 7.2))
    for row, lead in enumerate((0, 19, 39)):
        for column, metric in enumerate(("amplitude", "coherence")):
            draw_spectrum(axes[row, column], values92[metric][lead], metric, top=row == 0)
            axes[row, column].set_title("6 hours" if lead == 0 else f"{DAYS[lead]:g} days", pad=41 if row == 0 else 5)
    legend(fig, 3)
    fig.subplots_adjust(left=.095, right=.965, bottom=.07, top=.85, wspace=.32, hspace=.45)
    outputs += save(fig, "smoothing_spectra_6h_5d_10d_92")
    return outputs


def contrasts(values):
    results = {}
    for label, leads in (("rollout", list(range(40))), ("day5", [19]), ("day10", [39])):
        loss = values["loss"][leads].mean(axis=0)
        results[label] = {"exact_loss_by_model": {MODELS[i][0]: float(v) for i, v in enumerate(loss)},
                          "m1_vs_raw_gc_reduction_pct": float(100 * (1 - loss[1] / loss[0])),
                          "m1_vs_optimal_gc_reduction_pct": float(100 * (1 - loss[1] / loss[2])),
                          "optimal_gc_vs_raw_gc_reduction_pct": float(100 * (1 - loss[2] / loss[0])),
                          "power_error_by_model_pct": {MODELS[i][0]: float(v) for i, v in enumerate(values["power_error_by_lead"][leads].mean(axis=0))}}
        if len(loss) == 4:
            results[label]["m1_vs_reset_reduction_pct"] = float(100 * (1 - loss[1] / loss[3]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-only", action="store_true", help="Refresh only the two feature figures; preserve all other figures")
    args = parser.parse_args()
    data = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
    images = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
    data.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    filters = json.loads(FILTERS.read_text())
    layout = filters["layout"]
    channels = layout["channels"]
    weights = np.asarray(layout["weights"], dtype=float)
    expected_weights = []
    for variable, pressure in channels:
        variable_weight = .1 if variable in ("10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "total_precipitation_6hr") else 1.
        pressures = [level for name, level in channels if name == variable and level is not None]
        expected_weights.append(variable_weight * pressure / sum(pressures) if pressure is not None else variable_weight)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-14, atol=1e-14)
    normalized_weights = weights / weights.sum()
    source92, arrays92, pins92 = load(FULL, 92)
    source4, arrays4, pins4 = load(PREVIEW / "gc_m1_reference", 4)
    reset_report, reset, reset_pins = load(PREVIEW / "report", 4)
    for source, arrays in ((source92, arrays92), (source4, arrays4)):
        if (source["branches"] != ["GC", "M1"] or source["variants"] != ["raw", "projected", "optimal"]
                or arrays["prediction_power"].shape != (40, 2, 3, 83, 180)
                or arrays["truth_power"].shape != (40, 83, 180) or arrays["exact_gc_loss"].shape != (40, 2, 3)):
            raise ValueError("Unexpected reference model order or array dimensions")
    if reset_report["branches"] != ["GC", "R_matched"] or reset["prediction_power"].shape != (40, 2, 83, 180):
        raise ValueError("Unexpected reset branch order or array dimensions")
    if (source92["campaign_hash"] != filters["campaign_hash"] or source4["layout"] != layout
            or reset_report["layout"] != layout or source4["initialization_ids"] != PREVIEW_IDS
            or reset_report["initialization_ids"] != PREVIEW_IDS
            or reset_report["checkpoint"]["sha256"] != RESET_SHA
            or reset_report["model_id"] != "R_matched" or not reset_report["reset_state_every_step"]):
        raise ValueError("Mismatched cohort, checkpoint, policy or channel layout")
    values92 = prepare(source92, arrays92, normalized_weights)
    values4 = prepare(source4, arrays4, normalized_weights, reset=reset)
    loss_rows, spectrum_rows, feature_rows = [], [], []
    for cohort, values in (("interim92", values92), ("preview4", values4)):
        for li in range(40):
            for mi in range(values["loss"].shape[1]):
                loss_rows.append({"cohort": cohort, "model": MODELS[mi][0], "lead_hours": 6 * (li + 1),
                                  "exact_gc_loss": values["loss"][li, mi],
                                  "loss_reduction_vs_same_cohort_raw_gc_pct": 100 * (1 - values["loss"][li, mi] / values["loss"][li, 0])})
                if li in (0, 19, 39):
                    for di, degree in enumerate(DEGREE):
                        spectrum_rows.append({"cohort": cohort, "model": MODELS[mi][0], "lead_hours": 6 * (li + 1), "degree": degree,
                                              "amplitude_ratio": values["amplitude"][li, mi, di], "coherence": values["coherence"][li, mi, di]})
    for name, variable, pressure in FEATURES:
        ci = channels.index([variable, pressure])
        for mi in range(3):
            for di, degree in enumerate(DEGREE):
                feature_rows.append({"cohort": "interim92", "model": MODELS[mi][0], "field": name, "lead_hours": 120, "degree": degree,
                                     "amplitude_ratio": values92["amplitude_by_channel"][19, mi, ci, di],
                                     "coherence": values92["coherence_by_channel"][19, mi, ci, di]})
    for filename, rows in (("loss_by_lead.csv", loss_rows), ("weighted_spectral_curves.csv", spectrum_rows), ("feature_day5_curves.csv", feature_rows)):
        write_csv(data / filename, rows)
    channel_rows = []
    for ci, (variable, pressure) in enumerate(channels):
        for mi in range(3):
            channel_rows.append({"cohort": "interim92", "forecast_count": 92, "lead_hours": 120,
                                 "channel_index": ci, "variable": variable, "pressure_hpa": pressure,
                                 "model": MODELS[mi][0],
                                 "power_mismatch_equal_degree_pct": values92["power_error_by_channel"][19, mi, ci],
                                 "power_mismatch_truth_power_weighted_pct": values92["power_weighted_error_by_channel"][19, mi, ci],
                                 "coherence_equal_degree_mean": values92["coherence_by_channel"][19, mi, ci].mean(),
                                 "amplitude_ratio_equal_degree_mean": values92["amplitude_by_channel"][19, mi, ci].mean()})
    write_csv(data / "all_channels_day5_summary.csv", channel_rows)
    np.savez_compressed(data / "all_channels_day5_curves.npz", degree=DEGREE,
                        models=np.asarray([row[0] for row in MODELS[:3]]),
                        channels_json=json.dumps(channels), normalized_channel_weights=normalized_weights,
                        forecast_power=values92["prediction_power"][19, :, :, 1:], truth_power=arrays92["truth_power"][19, :, 1:],
                        amplitude_ratio=values92["amplitude_by_channel"][19], coherence=values92["coherence_by_channel"][19],
                        metadata=json.dumps({"cohort": "interim92", "count": 92, "planned_count": 122, "lead_hours": 120,
                                             "initialization_ids": source92["initialization_ids"],
                                             "array_axes": "model, channel, degree; truth_power omits model axis",
                                             "degrees": "1..179", "sources": pins92}))
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False}):
        outputs = figures(values92, values4, channels, images, features_only=args.features_only)
    result = {"created": "2026-11-14", "description": "Compact v4 smoothing figures; saved-data analysis only",
              "cohorts": {"interim92": {"count": 92, "planned_count": 122, "initialization_ids": source92["initialization_ids"],
                                        "status": "interim_incomplete_annual_evaluation", "models": [x[0] for x in MODELS[:3]], "sources": pins92},
                          "preview4": {"count": 4, "initialization_ids": PREVIEW_IDS, "status": "preliminary_four_date_preview",
                                       "models": [x[0] for x in MODELS], "sources": pins4 + reset_pins}},
              "filter_fit": "32 development starts in 2022; unchanged fitted filters, output-only application",
              "definitions": {"loss": "Saved exact native-grid GraphCast weighted MSE, pooled over dates; ratios of pooled losses",
                              "amplitude_ratio": "Per-channel sqrt(pooled forecast power / pooled truth power), then loss variable/pressure-weighted mean",
                              "coherence": "Per-channel pooled real cross-power / sqrt(pooled forecast power * pooled truth power), then loss variable/pressure-weighted mean",
                              "weights": "Frozen loss variable and pressure weights normalized to sum one; no stddev normalization for dimensionless ratios",
                              "degree_range": "1..179; positive degrees only for spectral curves; physical fields without climatological-mean removal",
                              "power_error": "100*mean abs(pooled forecast/truth power-1), equal over positive degrees and channels, then leads",
                              "channel_power_mismatch_equal_degree_pct": "100*mean_l abs(Pforecast(l)/Ptruth(l)-1), l=1..179; powers pooled over 92 dates before division",
                              "channel_power_mismatch_truth_power_weighted_pct": "100*sum_l abs(Pforecast(l)-Ptruth(l))/sum_l Ptruth(l), l=1..179; pooled powers",
                              "channel_mean_coherence": "Equal-degree mean of per-channel pooled cross-power/sqrt(Pforecast*Ptruth), l=1..179",
                              "feature_annotations": "Day-5 equal-degree power mismatch for raw M1 and raw GC, rounded to one decimal percent; identical scalar diagnostic in amplitude and coherence panels",
                              "uncertainty": "No spectral confidence intervals or seed variation; fixed original M1 and matched reset checkpoint",
                              "coherence_dashes": "GC blue and filtered GC teal use complementary (5,5) dash phases; no markers, arrows, offsets or clipping"},
              "checks": {"interim92": values92["checks"], "preview4": values4["checks"], "normalized_weight_sum": float(normalized_weights.sum()),
                         "same_truth_in_preview": True, "source_exact_losses_reproduced": True},
              "contrasts": {"interim92": contrasts(values92), "preview4": contrasts(values4)},
              "sources": [pin(FILTERS), pin(Path(__file__))], "figures": outputs,
              "curves": [pin(data / name) for name in ("loss_by_lead.csv", "weighted_spectral_curves.csv", "feature_day5_curves.csv",
                                                       "all_channels_day5_summary.csv", "all_channels_day5_curves.npz")]}
    (data / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"images": str(images.relative_to(ROOT)), "figure_files": len(outputs), "contrasts": result["contrasts"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
