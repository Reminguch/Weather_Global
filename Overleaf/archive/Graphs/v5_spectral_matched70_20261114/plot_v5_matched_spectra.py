#!/usr/bin/env python3
"""Render the five active v5 spectral figures from an explicit matched cohort.

Read pooled GC/M1 and matched-reset moments; never launch forecasts or write to
Overleaf. The actual cohort is retained in every numerical export and manifest.
The manuscript supplies the provisional-status note, as in the existing v5
layout. Historical 36-date scripts and artifacts are not modified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import plot_v4_smoothing_paper_20261114 as math_source

ROOT = Path(__file__).resolve().parents[2]
MODELS = (("GC", "GraphCast", "#2864A5"), ("M1", "M1", "#D97720"),
          ("GC_filter", "GC + optimal filter", "#238B8E"), ("reset", "R1", "#9467BD"))
STEMS = ("fig6_smoothing_day5_day10_v5_pending122", "fig10_smoothing_loss_v5_pending122",
         "fig11_smoothing_feature_amplitude_v5_pending122",
         "fig12_smoothing_feature_coherence_v5_pending122",
         "fig13_smoothing_lead_spectra_v5_pending122")
LEAD_HOURS = list(range(6, 241, 6))
SELECTED_LEADS = (0, 19, 39)
WINDOWS = (("rollout", np.arange(40)), ("day5", np.array([19])), ("day10", np.array([39])))
STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
         "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2.5,
         "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.spines.top": False,
         "axes.spines.right": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": digest(path)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_pool(directory, ids, *, reset):
    summary_path, npz_path = directory / "summary.json", directory / "power_pooled.npz"
    summary = json.loads(summary_path.read_text())
    if summary["count"] != len(ids) or summary["initialization_ids"] != ids:
        raise ValueError(f"Pooled cohort differs: {directory}")
    if digest(npz_path) != summary["data"]["power_pooled.npz"]["sha256"]:
        raise ValueError(f"Pooled input fingerprint mismatch: {directory}")
    shapes = {key: ((40, 2, 83, 180) if reset else (40, 2, 3, 83, 180))
              for key in ("prediction_power", "cross_power", "error_power")}
    shapes.update(truth_power=(40, 83, 180), exact_gc_loss=(40, 2) if reset else (40, 2, 3))
    with np.load(npz_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {key: archive[key] for key in shapes}
    for key in ("initialization_ids", "campaign_hash", "layout", "branches", "variants"):
        if metadata[key] != summary[key]:
            raise ValueError(f"Pooled report/statistics {key} differs: {directory}")
    for key, shape in shapes.items():
        values = arrays[key]
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key} shape or nonfinite values: {directory}")
        if key in ("prediction_power", "truth_power") and np.any(values <= 0):
            raise ValueError(f"Nonpositive physical power: {directory}:{key}")
        if key in ("error_power", "exact_gc_loss") and np.any(values < 0):
            raise ValueError(f"Negative squared error: {directory}:{key}")
    if summary.get("lead_hours", LEAD_HOURS) != LEAD_HOURS or summary.get("lmax", 179) != 179:
        raise ValueError("Expected 40 six-hour leads and degree 0 through 179")
    return summary, arrays, [pin(summary_path), pin(npz_path)]


def load_inputs(args):
    cohort = json.loads(args.cohort.read_text())
    ids = cohort["initialization_ids"]
    annual = [(datetime(2023, 1, 1) + timedelta(days=3 * i)).strftime("%Y%m%dT000000Z")
              for i in range(122)]
    if (cohort["count"] != args.count or len(ids) != args.count or len(set(ids)) != args.count
            or ids != [stamp for stamp in annual if stamp in set(ids)]):
        raise ValueError("Cohort must match the explicit count and ordered 2023 three-day grid")
    if args.count == 122 and ids != annual:
        raise ValueError("A full-year claim requires exactly the standard 122 starts")
    filters = json.loads(args.filters.read_text())
    source, arrays, source_pins = load_pool(args.source, ids, reset=False)
    reset_source, reset, reset_pins = load_pool(args.reset_source, ids, reset=True)
    layout = filters["layout"]
    channels = layout["channels"]
    if (source["branches"] != ["GC", "M1"] or source["variants"] != ["raw", "projected", "optimal"]
            or reset_source["branches"] != ["GC", "R_matched"] or reset_source["variants"] != ["raw"]
            or source["layout"] != layout or reset_source["layout"] != layout or len(channels) != 83
            or source["campaign_hash"] != filters["campaign_hash"]
            or reset_source["checkpoint"]["sha256"] != math_source.RESET_SHA
            or reset_source.get("model_id") != "R_matched" or reset_source.get("reset_state_every_step") is not True
            or filters["lead_hours"] != LEAD_HOURS or filters["lmax"] != 179
            or filters["fit_year"] != 2022 or filters["development_count"] != 32):
        raise ValueError("Incorrect model, reset policy, filter provenance or channel layout")
    if source.get("filter") and source["filter"]["sha256"] != digest(args.filters):
        raise ValueError("The supplied filter differs from the pooled reference")
    expected_weights = []
    for variable, pressure in channels:
        variable_weight = .1 if variable in ("10m_u_component_of_wind", "10m_v_component_of_wind",
                                            "mean_sea_level_pressure", "total_precipitation_6hr") else 1.
        levels = [level for name, level in channels if name == variable and level is not None]
        expected_weights.append(variable_weight * pressure / sum(levels) if pressure is not None else variable_weight)
    weights = np.asarray(layout["weights"], dtype=float)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-14, atol=1e-14)
    weights /= weights.sum()
    # The established computation includes the spectral closure/truth checks;
    # supply a verified loss summary for generic pooled reports without rows.
    source_for_math = dict(source)
    source_for_math.setdefault("rows", [
        {"model": model, "variant": variant,
         "exact_gc_rollout_loss": float(arrays["exact_gc_loss"][:, bi, vi].mean())}
        for bi, model in enumerate(source["branches"]) for vi, variant in enumerate(source["variants"])])
    values = math_source.prepare(source_for_math, arrays, weights, reset=reset)
    for mi, exact in enumerate((arrays["exact_gc_loss"][:, 0, 0], arrays["exact_gc_loss"][:, 1, 0],
                                arrays["exact_gc_loss"][:, 0, 2], reset["exact_gc_loss"][:, 1])):
        np.testing.assert_array_equal(values["loss"][:, mi], exact)
    np.testing.assert_array_equal(values["prediction_power"][:, 1], arrays["prediction_power"][:, 1, 0])
    np.testing.assert_array_equal(values["prediction_power"][:, 3], reset["prediction_power"][:, 1])
    pins = source_pins + reset_pins + [pin(args.cohort), pin(args.filters), pin(Path(__file__)),
                                     pin(Path(math_source.__file__))]
    return cohort, source, reset_source, values, arrays["truth_power"], channels, weights, pins


def line_style(index, metric):
    settings = {"color": MODELS[index][2], "lw": 1.35, "label": MODELS[index][1]}
    if metric == "coherence":
        settings.update(ls=(5 if index == 2 else 0, (5, 5)), dash_capstyle="butt")
    return settings


def checked_line(axis, x, y, **style):
    line, = axis.plot(x, y, **style)
    np.testing.assert_array_equal(line.get_xdata(), x)
    np.testing.assert_array_equal(line.get_ydata(), y)
    return line


def draw_spectrum(axis, curves, metric, *, top, limits=None):
    for mi in range(4):
        checked_line(axis, math_source.DEGREE, curves[mi], **line_style(mi, metric))
    axis.axhline(1., color=".4", ls=":", lw=.6)
    if limits is None:
        lower = min(-.025, float(curves.min()) - .025) if metric == "coherence" else 0.
        limits = lower, max(1.05, float(curves.max()) * 1.04)
    axis.set(ylim=limits, ylabel="Coherence" if metric == "coherence" else "Amplitude ratio")
    math_source.spatial_axis(axis, top=top)


def legend(fig, *, y=.012):
    handles = [Line2D([], [], color=color, lw=1.5, label=label) for _, label, color in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=4,
               frameon=False, fontsize=7.5, handlelength=1.8, handletextpad=.55,
               columnspacing=1.25, borderaxespad=0.)


def render(values, channels, output):
    products, records = [], {}
    def save(fig, stem):
        fig.canvas.draw()
        signature = hashlib.sha256()
        curves = 0
        for axis in fig.axes:
            for line in axis.lines:
                for values_ in (line.get_xdata(), line.get_ydata()):
                    data = np.asarray(values_, dtype="<f8")
                    signature.update(str(data.shape).encode())
                    signature.update(data.tobytes())
                curves += 1
        labels = [text.get_text() for lg in fig.legends for text in lg.get_texts()]
        if labels != [row[1] for row in MODELS]:
            raise ValueError(f"Four-model legend differs: {stem}")
        records[stem] = {"size_inches": fig.get_size_inches().tolist(),
                         "axes": len(fig.axes), "lines_including_references": curves,
                         "coordinate_sha256": signature.hexdigest(), "legend": labels,
                         "axes_positions": [list(axis.get_position().bounds) for axis in fig.axes]}
        for suffix in ("pdf", "png"):
            path = output / f"{stem}.{suffix}"
            fig.savefig(path, dpi=230)
            products.append(pin(path))
        plt.close(fig)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 3.8), sharey="row")
        for row, metric in enumerate(("amplitude", "coherence")):
            selected = values[metric][[19, 39]]
            lower = min(-.025, float(selected.min()) - .025) if metric == "coherence" else 0.
            limits = lower, max(1.05, float(selected.max()) * 1.04)
            for col, lead in enumerate((19, 39)):
                draw_spectrum(axes[row, col], values[metric][lead], metric, top=row == 0, limits=limits)
                if col:
                    axes[row, col].set_ylabel("")
                if row == 0:
                    axes[row, col].set_title(f"Day {(lead + 1) // 4}", fontsize=8.5, pad=39)
        legend(fig)
        fig.subplots_adjust(left=.09, right=.965, bottom=.17, top=.80, hspace=.47, wspace=.25)
        save(fig, STEMS[0])

        fig, axis = plt.subplots(figsize=(6.25, 2.65))
        for mi in range(4):
            checked_line(axis, math_source.DAYS, values["loss"][:, mi], **line_style(mi, "loss"))
        axis.axvline(5, color=".5", ls=":", lw=.7)
        axis.set(xlim=(.25, 10), xlabel="Forecast lead (days)", xticks=[1, 3, 5, 7, 10],
                 ylabel="Exact GraphCast loss")
        axis.grid(alpha=.18, linewidth=.5)
        legend(fig)
        fig.subplots_adjust(left=.095, right=.965, bottom=.27, top=.96)
        save(fig, STEMS[1])

        for metric, stem in zip(("amplitude", "coherence"), STEMS[2:4]):
            fig, axes = plt.subplots(2, 2, figsize=(6.25, 3.8))
            for axis, (short, variable, pressure) in zip(axes.flat, math_source.FEATURES):
                ci = channels.index([variable, pressure])
                draw_spectrum(axis, values[metric + "_by_channel"][19, :, ci], metric, top=False)
                axis.set_title(short, pad=5)
                gc, m1 = values["power_error_by_channel"][19, :2, ci]
                high = metric == "amplitude" and short == "Z500"
                axis.text(.035, .94 if high else .06, f"Mismatch: M1 {m1:.1f}% | GC {gc:.1f}%",
                          transform=axis.transAxes, fontsize=7, ha="left", va="top" if high else "bottom",
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
            legend(fig)
            fig.subplots_adjust(left=.09, right=.965, bottom=.17, top=.92, hspace=.55, wspace=.32)
            save(fig, stem)

        fig, axes = plt.subplots(3, 2, figsize=(6.25, 5.6))
        for row, lead in enumerate(SELECTED_LEADS):
            for col, metric in enumerate(("amplitude", "coherence")):
                draw_spectrum(axes[row, col], values[metric][lead], metric, top=row == 0)
                axes[row, col].set_title("6 hours" if lead == 0 else f"{math_source.DAYS[lead]:g} days",
                                        pad=41 if row == 0 else 5)
        legend(fig, y=.009)
        fig.subplots_adjust(left=.095, right=.965, bottom=.115, top=.85, hspace=.50, wspace=.32)
        save(fig, STEMS[4])
    if len(products) != 10 or set(records) != set(STEMS):
        raise ValueError("Expected exactly the five retained v5 figures in PDF/PNG")
    return products, records


def export_values(values, truth, channels, cohort, output):
    count = cohort["count"]
    scores, loss_rows, spectral_rows, feature_rows = [], [], [], []
    contrasts = math_source.contrasts(values)
    for window, leads in WINDOWS:
        loss = values["loss"][leads].mean(axis=0)
        mismatch = values["power_error_by_lead"][leads].mean(axis=0)
        contrasts[window]["m1_minus_reset_power_mismatch_pp"] = float(mismatch[1] - mismatch[3])
        for mi, (model, label, _) in enumerate(MODELS):
            scores.append({"forecast_count": count, "window": window, "model": model, "label": label,
                           "exact_gc_loss": float(loss[mi]), "power_mismatch_pct": float(mismatch[mi]),
                           "relative_mse": float(loss[mi] / loss[0]),
                           "relative_power_mismatch": float(mismatch[mi] / mismatch[0]),
                           "mse_reduction_vs_raw_gc_pct": float(100 * (1 - loss[mi] / loss[0])),
                           "power_mismatch_reduction_vs_raw_gc_pct": float(100 * (1 - mismatch[mi] / mismatch[0]))})
    for li, lead in enumerate(LEAD_HOURS):
        for mi, (model, label, _) in enumerate(MODELS):
            loss_rows.append({"forecast_count": count, "lead_hours": lead, "model": model, "label": label,
                              "exact_gc_loss": float(values["loss"][li, mi]),
                              "power_mismatch_pct": float(values["power_error_by_lead"][li, mi])})
    for li in SELECTED_LEADS:
        for mi, (model, _, _) in enumerate(MODELS):
            for di, degree in enumerate(math_source.DEGREE):
                spectral_rows.append({"forecast_count": count, "lead_hours": LEAD_HOURS[li], "model": model,
                                      "degree": int(degree), "weighted_amplitude_ratio": float(values["amplitude"][li, mi, di]),
                                      "weighted_coherence": float(values["coherence"][li, mi, di])})
    for short, variable, pressure in math_source.FEATURES:
        ci = channels.index([variable, pressure])
        for mi, (model, _, _) in enumerate(MODELS):
            for di, degree in enumerate(math_source.DEGREE):
                feature_rows.append({"forecast_count": count, "lead_hours": 120, "model": model, "field": short,
                                     "degree": int(degree), "amplitude_ratio": float(values["amplitude_by_channel"][19, mi, ci, di]),
                                     "coherence": float(values["coherence_by_channel"][19, mi, ci, di])})
    outputs = {"comparison_scores.csv": scores, "loss_by_lead.csv": loss_rows,
               "weighted_spectral_curves.csv": spectral_rows, "feature_day5_curves.csv": feature_rows}
    for name, rows in outputs.items():
        write_csv(output / name, rows)
    table_lines = []
    lookup = {(row["window"], row["model"]): row for row in scores}
    for model, label, _ in (MODELS[0], MODELS[2], MODELS[1], MODELS[3]):
        columns = [lookup[window, model][metric] for window, _ in WINDOWS
                   for metric in ("relative_mse", "relative_power_mismatch")]
        table_lines.append(label + " & " + " & ".join(f"{value:.2f}" for value in columns) + r" \\")
    (output / "table1_rows.tex").write_text("\n".join(table_lines) + "\n")
    metadata = {"forecast_count": count, "initialization_ids": cohort["initialization_ids"],
                "models": [row[0] for row in MODELS], "channels": channels,
                "lead_hours": [LEAD_HOURS[i] for i in SELECTED_LEADS], "degree": "1..179",
                "array_axes": "lead,model,channel,degree; truth omits model"}
    np.savez_compressed(output / "all_channels_selected_leads_curves.npz",
                        metadata=json.dumps(metadata), degree=math_source.DEGREE,
                        amplitude_ratio=values["amplitude_by_channel"][list(SELECTED_LEADS)],
                        coherence=values["coherence_by_channel"][list(SELECTED_LEADS)],
                        prediction_power=values["prediction_power"][list(SELECTED_LEADS), :, :, 1:],
                        truth_power=truth[list(SELECTED_LEADS), :, 1:])
    return scores, contrasts, [pin(output / name) for name in (*outputs, "table1_rows.tex",
                                                             "all_channels_selected_leads_curves.npz")]


def replay_exports(directory, output, expected_count):
    """Reproduce the figures using only the small archived numerical exports."""
    manifest_path, summary_path = directory / "manifest.json", directory / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    if digest(summary_path) != manifest["summary"]["sha256"]:
        raise ValueError("Archived summary fingerprint differs")
    summary = json.loads(summary_path.read_text())
    if summary["cohort"]["count"] != expected_count:
        raise ValueError("Archived cohort differs from --count")
    for record in summary["exports"]:
        path = directory / Path(record["path"]).name
        if digest(path) != record["sha256"]:
            raise ValueError(f"Archived numerical export changed: {path}")
    with np.load(directory / "all_channels_selected_leads_curves.npz", allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if (metadata["forecast_count"] != expected_count
                or metadata["initialization_ids"] != summary["cohort"]["initialization_ids"]
                or metadata["models"] != [row[0] for row in MODELS]
                or metadata["lead_hours"] != [6, 120, 240]):
            raise ValueError("Archived channel-curve metadata differs")
        channels = metadata["channels"]
        values = {}
        for key, field in (("amplitude_by_channel", "amplitude_ratio"), ("coherence_by_channel", "coherence")):
            values[key] = np.full((40, 4, len(channels), 179), np.nan)
            values[key][list(SELECTED_LEADS)] = archive[field]
        mismatch = 100 * np.mean(np.abs(archive["prediction_power"] / archive["truth_power"][:, None] - 1), axis=-1)
        values["power_error_by_channel"] = np.full((40, 4, len(channels)), np.nan)
        values["power_error_by_channel"][list(SELECTED_LEADS)] = mismatch
    models = [row[0] for row in MODELS]
    with (directory / "loss_by_lead.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 160 or any(int(row["forecast_count"]) != expected_count for row in rows):
        raise ValueError("Expected all 40 leads and four models in archived losses")
    values["loss"] = np.full((40, 4), np.nan)
    for row in rows:
        values["loss"][int(row["lead_hours"]) // 6 - 1, models.index(row["model"])] = float(row["exact_gc_loss"])
    if not np.isfinite(values["loss"]).all():
        raise ValueError("Incomplete archived loss curves")
    with (directory / "weighted_spectral_curves.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 3 * 4 * 179 or any(int(row["forecast_count"]) != expected_count for row in rows):
        raise ValueError("Expected all selected leads and four models in archived spectra")
    for metric, column in (("amplitude", "weighted_amplitude_ratio"), ("coherence", "weighted_coherence")):
        values[metric] = np.full((40, 4, 179), np.nan)
        for row in rows:
            values[metric][int(row["lead_hours"]) // 6 - 1, models.index(row["model"]), int(row["degree"]) - 1] = float(row[column])
        if not np.isfinite(values[metric][list(SELECTED_LEADS)]).all():
            raise ValueError("Incomplete archived spectral curves")
    output.mkdir(parents=True, exist_ok=True)
    products, layout = render(values, channels, output)
    if layout != summary["figure_layout"]:
        raise ValueError("Reproduced coordinates, labels or layout differ from the archived figures")
    write_json(output / "reproduction.json", {
        "status": "verified", "cohort": summary["cohort"], "source_manifest": pin(manifest_path),
        "source_summary": pin(summary_path), "renderer": pin(Path(__file__)), "figures": products,
        "figure_layout": layout, "plotted_coordinates_and_layout_match": True,
        "original_pooled_inputs_required": False, "new_forecasts_or_metric_estimates": False})
    print(json.dumps({"output": str(output), "count": expected_count, "pdf_count": 5,
                      "portable_reproduction_verified": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="GC/M1 pooled summary.json and power_pooled.npz directory")
    inputs.add_argument("--export-source", type=Path, help="Replay a previously exported small numerical bundle; no pooled inputs needed")
    parser.add_argument("--reset-source", type=Path, help="Matched R1 pooled summary.json and power_pooled.npz directory")
    parser.add_argument("--cohort", type=Path, help="Explicit ordered initialization_ids and count")
    parser.add_argument("--filters", type=Path, help="Original 2022-fit filters.json")
    parser.add_argument("--count", type=int, required=True, help="Expected matched sample count, not planned final count")
    parser.add_argument("--output", type=Path, required=True, help="Artifact directory without prior render outputs; publication is separate")
    args = parser.parse_args()
    for name in ("source", "export_source", "reset_source", "cohort", "filters", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if not 0 < args.count <= 122:
        raise ValueError("Expected between 1 and 122 standard 2023 initializations")
    output_names = [f"{stem}.{suffix}" for stem in STEMS for suffix in ("pdf", "png")]
    output_names += ["summary.json", "manifest.json", "comparison_scores.csv", "loss_by_lead.csv",
                     "weighted_spectral_curves.csv", "feature_day5_curves.csv", "table1_rows.tex",
                     "all_channels_selected_leads_curves.npz", Path(__file__).name]
    source_dirs = [path for path in (args.source, args.reset_source, args.export_source) if path is not None]
    if (args.output.is_relative_to(ROOT / "Overleaf") or any(args.output.is_relative_to(path) for path in source_dirs)
            or any((args.output / name).exists() for name in output_names)):
        raise ValueError("Use an artifact directory without prior render outputs, separate from source pools and Overleaf")
    if args.export_source:
        replay_exports(args.export_source, args.output, args.count)
        return
    if any(getattr(args, name) is None for name in ("reset_source", "cohort", "filters")):
        parser.error("--source requires --reset-source, --cohort and --filters")
    cohort, source, reset_source, values, truth, channels, weights, pins = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    products, layout = render(values, channels, args.output)
    scores, contrasts, exports = export_values(values, truth, channels, cohort, args.output)
    ids = cohort["initialization_ids"]
    summary = {"format": "v5_matched_spectral_replot_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "complete_matched_2023" if args.count == 122 else "interim_matched_cohort",
               "cohort": {"count": args.count, "planned_final_count": 122, "initialization_ids": ids,
                          "first_initialization": ids[0], "last_initialization": ids[-1],
                          "source_cohort_status": cohort.get("status"), "lead_hours": LEAD_HOURS},
               "models": [{"id": model, "label": label} for model, label, _ in MODELS],
               "table1_rows": scores, "contrasts": contrasts,
               "definitions": {"loss": "Exact saved native-grid GraphCast weighted MSE; pool losses before ratios",
                               "power_mismatch": "100*mean abs(date-pooled Pf/Py-1), equally over selected leads,83channels,degrees1..179",
                               "amplitude_coherence": "Ratios from date-pooled moments, averaged with normalized GraphCast variable/pressure weights",
                               "filter": "GraphCast output-only 2022 development32 filter; unfiltered autoregressive feedback",
                               "table": "Each error divided by same-cohort raw GraphCast error;1=baseline;lower is better",
                               "uncertainty": "Fixed-checkpoint point estimates; no new confidence intervals",
                               "provisional_status": "Manuscript note and caption; no redundant in-figure heading"},
               "checks": {**values["checks"], "same_dates_and_truth": True,
                          "raw_m1_r1_predictions_and_exact_losses_preserved": True,
                          "each_plotted_curve_equals_source_values": True, "normalized_weight_sum": float(weights.sum())},
               "numerical_diagnostics": {"reference": source.get("numerical_diagnostics", {}),
                                         "reset": reset_source.get("numerical_diagnostics", {})},
               "sources": pins, "figures": products, "figure_layout": layout, "exports": exports,
               "new_forecasts_filter_fits_or_bootstrap_draws": False,
               "removed_physical_power_figures_restored": False}
    write_json(args.output / "summary.json", summary)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    write_json(args.output / "manifest.json", {**summary, "summary": pin(args.output / "summary.json"),
                                               "source_snapshot": pin(args.output / Path(__file__).name)})
    for record in pins:
        path = Path(record["path"])
        path = path if path.is_absolute() else ROOT / path
        if digest(path) != record["sha256"]:
            raise ValueError(f"Source changed during rendering: {path}")
    print(json.dumps({"output": str(args.output), "count": args.count, "date_range": [ids[0], ids[-1]],
                      "pdf_count": 5, "status": summary["status"], "contrasts": contrasts}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
