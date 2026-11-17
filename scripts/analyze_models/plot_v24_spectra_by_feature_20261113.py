#!/usr/bin/env python3
"""Per-feature spectral diagnostics from matched, already pooled forecasts.

Reuses saved moments only. Keeps power, amplitude and coherence in separate
figures, and exports every channel to prevent selected examples hiding failures.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

import plot_v24_literature_spectral_comparison_20261113 as base

ROOT = base.ROOT
CAMPAIGN = ROOT / "artifacts/evaluations/v24_matched_reset_preview4_20261113"
NAME = "v24_literature_spectral_preview4_20261113"
PREVIEW_CAPTION = "PRELIMINARY: 4 matched 2023 starts (Jan 1, Apr 1, Jul 3, Oct 1); fixed checkpoints; no uncertainty bands."
COHORT_CAPTION = PREVIEW_CAPTION
DEGREES = np.arange(1, 180)
MODELS = [
    ("GC", "GraphCast (raw)", "#2864A5", 0),
    ("M1", "M1 (raw)", "#D97720", None),
    ("GC_filtered", "GraphCast + optimal filter", "#238B8E", 5),
    ("R_matched", "Reset/reset (raw)", "#9467BD", None),
]
# Name, short label, physical power unit. Stored fields retain original SI units.
VARIABLES = {
    "10m_u_component_of_wind": ("10 m zonal wind", "U10m", r"m$^2$ s$^{-2}$"),
    "10m_v_component_of_wind": ("10 m meridional wind", "V10m", r"m$^2$ s$^{-2}$"),
    "2m_temperature": ("2 m temperature", "T2m", r"K$^2$"),
    "geopotential": ("Geopotential", "Z", r"m$^4$ s$^{-4}$"),
    "mean_sea_level_pressure": ("Mean sea-level pressure", "MSLP", r"Pa$^2$"),
    "specific_humidity": ("Specific humidity", "Q", r"(kg kg$^{-1}$)$^2$"),
    "temperature": ("Temperature", "T", r"K$^2$"),
    "total_precipitation_6hr": ("6-hour precipitation", "TP6h", r"m$^2$"),
    "u_component_of_wind": ("Zonal wind", "U", r"m$^2$ s$^{-2}$"),
    "v_component_of_wind": ("Meridional wind", "V", r"m$^2$ s$^{-2}$"),
    "vertical_velocity": ("Vertical pressure velocity", "W", r"Pa$^2$ s$^{-2}$"),
}
METRICS = {
    "amplitude": ("Spectral amplitude", r"$\sqrt{P_{\rm forecast}/P_{\rm ERA5}}$"),
    "coherence": ("Spectral coherence", "Coherence"),
    "power": ("Physical power spectra", "Power"),
}


def load_inputs(source_directory=None, reset_directory=None, *, planned_count=92, return_cohort=False):
    """Keep the historical five-item return value unless cohort data are requested."""
    default_preview = source_directory is None and reset_directory is None
    directories = [Path(source_directory or CAMPAIGN / "gc_m1_reference").resolve(),
                   Path(reset_directory or CAMPAIGN / "report").resolve()]
    summaries, arrays, pins = [], [], []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        path = directory / "power_pooled.npz"
        if base.pin(path)["sha256"] != summary["data"][path.name]["sha256"]:
            raise ValueError(f"Pooled moment hash mismatch: {path}")
        ids = summary["initialization_ids"]
        if (summary["status"] not in {"complete", "preliminary_four_date_preview", "interim_matched_cohort"}
                or summary["count"] != len(ids) or not ids or len(set(ids)) != len(ids)
                or ids != sorted(ids)):
            raise ValueError("Invalid or incomplete pooled comparison cohort")
        if default_preview and ids != base.PREVIEW_IDS:
            raise ValueError("Expected the four predeclared matched preview dates")
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in ("truth_power", "prediction_power", "cross_power", "error_power")}
            metadata = json.loads(str(archive["metadata"]))
        if (metadata["initialization_ids"] != ids
                or metadata["layout"] != summary["layout"]
                or metadata["campaign_hash"] != summary["campaign_hash"]):
            raise ValueError("NPZ/report provenance differs")
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("Nonfinite moment")
        if any(np.any(values[key] <= 0) for key in ("truth_power", "prediction_power")):
            raise ValueError("Nonpositive power; ratio is undefined")
        summaries.append(summary)
        arrays.append(values)
        pins.extend([base.pin(directory / "summary.json"), base.pin(path)])
    source, reset = summaries
    if source["initialization_ids"] != reset["initialization_ids"]:
        raise ValueError("GC/M1 and reset/reset must use exactly the same ordered dates")
    if source["branches"] != ["GC", "M1"] or source["variants"] != ["raw", "projected", "optimal"]:
        raise ValueError("Unexpected reference branch order")
    if (reset["branches"] != ["GC", "R_matched"] or reset["model_id"] != "R_matched"
            or not reset["reset_state_every_step"] or reset["checkpoint"]["sha256"] != base.RESET_SHA256
            or reset["layout"] != source["layout"] or reset["lmax"] != 179
            or reset["lead_hours"] != list(range(6, 241, 6))):
        raise ValueError("Unexpected matched-reset provenance")
    if (metadata.get("checkpoint_sha256", base.RESET_SHA256) != base.RESET_SHA256
            or source["count"] > planned_count or planned_count < 1):
        raise ValueError("Unexpected reset checkpoint or planned cohort size")
    if arrays[0]["prediction_power"].shape != (40, 2, 3, 83, 180) or arrays[1]["prediction_power"].shape != (40, 2, 83, 180):
        raise ValueError("Unexpected spectral array shape")
    np.testing.assert_allclose(arrays[0]["truth_power"], arrays[1]["truth_power"], rtol=1e-9, atol=1e-12)
    closure = []
    for i, values in enumerate(arrays):
        truth = values["truth_power"][:, None, None] if i == 0 else values["truth_power"][:, None]
        residual = values["prediction_power"] + truth - 2 * values["cross_power"] - values["error_power"]
        closure.append(float(np.max(np.abs(residual) / (values["prediction_power"] + truth))))
        if closure[-1] > 1e-10 or np.any(values["error_power"] < 0):
            raise ValueError("Inconsistent spectral moments")
    result = (source["layout"]["channels"], arrays, pins, closure, reset.get("numerical_diagnostics", {}))
    ids = source["initialization_ids"]
    preview = default_preview or (ids == base.PREVIEW_IDS and source["status"] == "preliminary_four_date_preview")
    cohort = {"count": len(ids), "planned_count": planned_count, "initialization_ids": ids,
              "status": "preliminary_four_date_preview" if preview else
                        ("interim_matched_cohort" if len(ids) < planned_count else "complete"),
              "source_status": source["status"], "reset_status": reset["status"],
              "date_span": [datetime.strptime(identity, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d")
                            for identity in (ids[0], ids[-1])]}
    return result + (cohort,) if return_cohort else result


def label(channel):
    variable, level = channel
    return VARIABLES[variable][1] + (f"{level:g}" if level is not None else "")


def line_style(index, metric):
    _, name, color, phase = MODELS[index]
    style = {"color": color, "lw": 1.6, "label": name}
    if metric == "coherence":
        style.update(ls="--" if phase is None else (phase, (5, 5)), dash_capstyle="butt")
    return style


def finish(fig, title, subtitle, metric):
    height = fig.get_figheight()
    # Fixed physical spacing also works for both the 6-panel and 13-panel pages.
    fig.suptitle(title, fontsize=14, y=1 - .12 / height, va="top")
    fig.text(.5, 1 - .46 / height, subtitle, ha="center", va="top", fontsize=9)
    handles = [Line2D([], [], **line_style(i, metric)) for i in range(4)]
    if metric == "power":
        handles.insert(0, Line2D([], [], color=".18", lw=1.8, label="ERA5"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1 - .70 / height),
               ncol=len(handles), frameon=False, fontsize=9, handlelength=3)
    fig.text(.5, .035, COHORT_CAPTION,
             ha="center", fontsize=8)
    fig.text(.5, .015, "Dates pooled before ratios; degrees 1–179, no climatology subtraction. GC filter fitted on 32 starts in 2022.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(.005, .065, .995, 1 - 1.15 / height), h_pad=2.4, w_pad=1.5)


def draw_grid(indices, channels, values, scores, metric, title, subtitle, ranks=None):
    columns = 3 if len(indices) <= 6 else 4
    rows = math.ceil(len(indices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13.5, 3.25 * rows + 2), squeeze=False)
    for panel, (axis, index) in enumerate(zip(axes.flat, indices)):
        if metric == "power":
            axis.plot(DEGREES, values["truth"][index], color=".18", lw=1.8)
        displayed = values[metric][:, index]
        for mi in range(4):
            axis.plot(DEGREES, displayed[mi], **line_style(mi, metric))
        if metric == "power":
            axis.set_yscale("log")
            axis.set_ylabel(f"Power ({VARIABLES[channels[index][0]][2]})")
        else:
            axis.axhline(1, color=".35", lw=.8, ls=":")
            high = max(1.06, float(displayed.max()) * 1.06)
            low = min(-.035, float(displayed.min()) - .035) if metric == "coherence" else -.035
            axis.set_ylim(low, high)
            axis.set_ylabel(METRICS[metric][1])
        base.spatial_axis(axis, top=panel < columns)
        axis.tick_params(labelsize=8)
        text = label(channels[index])
        if ranks is not None:
            text += f" · rank {ranks[index]}/83"
        axis.set_title(text, fontsize=11, pad=45 if panel < columns else 8)
        axis.text(.03, .04, f"Mismatch: M1 {scores['equal_degree'][1, index]:.1f}% | GC {scores['equal_degree'][0, index]:.1f}%",
                  transform=axis.transAxes, fontsize=8,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 2})
    for axis in list(axes.flat)[len(indices):]:
        axis.set_visible(False)
    finish(fig, title, subtitle, metric)
    return fig


def export(fig, images, stem, book=None):
    outputs = []
    for suffix in ("pdf", "png"):
        path = images / f"{stem}.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        outputs.append(path)
    if book is not None:
        book.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main():
    global COHORT_CAPTION
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "For short render batches, run --metrics amplitude, coherence and power "
        "sequentially for each lead. Their manifests are merged automatically."))
    parser.add_argument("--lead-hours", type=int, default=120, choices=range(6, 241, 6))
    parser.add_argument("--source", type=Path, help="Pooled GC/M1 report directory")
    parser.add_argument("--reset-source", type=Path, help="Pooled matched-reset report directory")
    parser.add_argument("--name", default=NAME, help="Output campaign directory name")
    parser.add_argument("--planned-count", type=int, default=92)
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRICS), default=list(METRICS),
                        help="Render selected metrics; verified per-metric manifests are merged")
    args = parser.parse_args()
    if (args.source is None) != (args.reset_source is None):
        parser.error("Provide both --source and --reset-source for a new cohort")
    if args.source is not None and args.name == NAME:
        parser.error("Use a new --name to preserve the historical four-date preview")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("--name must be a single directory name")
    args.metrics = list(dict.fromkeys(args.metrics))
    lead = args.lead_hours // 6 - 1
    day = args.lead_hours / 24
    suffix = f"{day:g}d"
    images = ROOT / "plots/analyze_models/images/resolution_eval" / args.name / "by_feature"
    data = ROOT / "plots/analyze_models/data/resolution_eval" / args.name / "by_feature"
    images.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    channels, arrays, pins, closure, numerical_diagnostics, cohort = load_inputs(
        args.source, args.reset_source, planned_count=args.planned_count, return_cohort=True)
    if cohort["status"] == "preliminary_four_date_preview":
        COHORT_CAPTION = PREVIEW_CAPTION
    else:
        count_text = f"{cohort['count']}/{cohort['planned_count']}" if cohort["status"].startswith("interim") else str(cohort["count"])
        prefix = "INTERIM: " if cohort["status"].startswith("interim") else ""
        COHORT_CAPTION = (f"{prefix}{count_text} matched starts ({cohort['date_span'][0]}–{cohort['date_span'][1]}); "
                          "fixed checkpoints; no uncertainty bands.")
    pooled, reset = arrays
    truth = pooled["truth_power"][lead, :, 1:]
    def stack(key):
        return np.stack([pooled[key][lead, 0, 0, :, 1:], pooled[key][lead, 1, 0, :, 1:],
                         pooled[key][lead, 0, 2, :, 1:], reset[key][lead, 1, :, 1:]])
    power, cross = stack("prediction_power"), stack("cross_power")
    ratio = power / truth[None]
    values = {"power": power, "truth": truth, "amplitude": np.sqrt(ratio),
              "coherence": cross / np.sqrt(power * truth[None])}
    scores = {
        "equal_degree": 100 * np.abs(ratio - 1).mean(axis=-1),
        "energy": 100 * np.abs(power - truth[None]).sum(axis=-1) / truth.sum(axis=-1)[None],
        "total_power_ratio": power.sum(axis=-1) / truth.sum(axis=-1)[None],
        "mean_coherence": values["coherence"].mean(axis=-1),
    }
    order = np.argsort(scores["equal_degree"][1], kind="stable")
    ranks = np.empty(83, dtype=int)
    ranks[order] = np.arange(1, 84)
    energy_order = np.argsort(scores["energy"][1], kind="stable")
    energy_ranks = np.empty(83, dtype=int)
    energy_ranks[energy_order] = np.arange(1, 84)
    selected = list(order[:3]) + list(order[-3:][::-1])
    familiar = [channels.index([v, p]) for v, p in [
        ("2m_temperature", None), ("geopotential", 500.), ("temperature", 850.),
        ("specific_humidity", 700.), ("total_precipitation_6hr", None), ("vertical_velocity", 500.)]]
    ranking = []
    for index in order:
        row = {"feature": label(channels[index]), "variable": channels[index][0],
               "pressure_hpa": channels[index][1], "lead_hours": args.lead_hours,
               "m1_equal_degree_rank": int(ranks[index]), "m1_energy_weighted_rank": int(energy_ranks[index])}
        for mi, (key, _, _, _) in enumerate(MODELS):
            for metric in scores:
                row[f"{key}_{metric}"] = float(scores[metric][mi, index])
        ranking.append(row)
    csv_path = data / f"feature_ranking_{suffix}.csv"
    curve_path = data / f"feature_curves_{suffix}.npz"
    curve_metadata = {"channels": channels, "models": [m[0] for m in MODELS],
                      "initialization_ids": cohort["initialization_ids"],
                      "lead_hours": args.lead_hours, "sources": pins}
    curve_values = {**values, "power_ratio": ratio, "cross_power": cross, "degrees": DEGREES}
    if curve_path.exists():
        # Keep shared-data hashes stable across separately rendered metric batches.
        with np.load(curve_path, allow_pickle=False) as saved:
            if json.loads(str(saved["metadata"])) != curve_metadata:
                raise ValueError("Existing feature data use another source/cohort; choose a new --name")
            for key, value in curve_values.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        np.savez_compressed(curve_path, **curve_values, metadata=json.dumps(curve_metadata))
    base.write_csv(csv_path, ranking)
    groups = [("surface", "Surface variables", [i for i, (_, p) in enumerate(channels) if p is None])]
    for variable in dict.fromkeys(v for v, p in channels if p is not None):
        groups.append((variable, VARIABLES[variable][0], [i for i, (v, _) in enumerate(channels) if v == variable]))
    outputs = []
    metric_outputs = {}
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False}):
        for metric in args.metrics:
            name = METRICS[metric][0]
            first_output = len(outputs)
            for tag, indices, title, subtitle in [
                ("best_worst", selected, "Closest and poorest M1 spectra by equal-degree mismatch",
                 "Top row: three closest; bottom: three poorest M1 spectra. Labels compare M1 / raw GC using equal-degree power mismatch."),
                ("familiar", familiar, "Common fields: contrasts in M1 spectral fidelity",
                 "Power mismatch: M1 and raw GC, equal weight per wavenumber (1–179); smaller is closer to ERA5. Fixed illustrative fields.")]:
                fig = draw_grid(indices, channels, values, scores, metric,
                                f"Day {day:g} · {name} · {title}", subtitle, ranks)
                outputs += export(fig, images, f"{tag}_{metric}_{suffix}")
            atlas_path = images / f"all_features_{metric}_{suffix}.pdf"
            with PdfPages(atlas_path) as book:
                for stem, title, indices in groups:
                    fig = draw_grid(indices, channels, values, scores, metric,
                                    f"Day {day:g} · {name} · {title}",
                                    "Power mismatch: M1 and raw GC, equal weight per wavenumber (1–179); smaller is closer to ERA5. Separate pressure levels.")
                    outputs += export(fig, images, f"{stem}_{metric}_{suffix}", book)
            outputs.append(atlas_path)
            metric_outputs[metric] = outputs[first_output:]
    summary = {
        "status": cohort["status"], "lead_hours": args.lead_hours, "cohort": cohort,
        "initialization_ids": cohort["initialization_ids"], "channels": channels,
        "models": [m[0] for m in MODELS], "sources": pins + [base.pin(Path(__file__)), base.pin(Path(base.__file__))],
        "definitions": {
            "equal_degree": "100 * mean_l(abs(Pforecast/PERA5 - 1)), l=1..179; each degree has equal weight",
            "energy": "100 * sum_l(abs(Pforecast-PERA5)) / sum_l(PERA5), l=1..179; ERA5-energy-weighted relative mismatch",
            "total_power_ratio": "sum_l(Pforecast)/sum_l(PERA5), l=1..179; differs from mismatch because signed differences can cancel",
            "amplitude": "sqrt(date-pooled forecast power / date-pooled ERA5 power), per channel",
            "coherence": "date-pooled cross-power / sqrt(date-pooled forecast power * date-pooled ERA5 power)",
            "mean_coherence": "Unweighted mean coherence over degrees1..179; diagnostic, not an exact forecast score",
            "scope": "T179 physical full fields, no climatology subtraction, l0 excluded; no mixed-unit averaging across channels",
            "ranking": "All83 channels ranked by M1 raw equal-degree mismatch; closest/poorest are relative ranks, not pass/fail thresholds",
            "selection": "best_worst shows exact top3 and bottom3; familiar shows fixed T2m/Z500/T850/Q700/TP6h/W500; atlas includes all83",
            "panel_annotations": "M1 raw and GraphCast raw equal-degree power mismatch percentages, explicitly labelled by model; energy-weighted scores remain in numeric tables only",
            "limits": f"{cohort['count']} fixed dates and one checkpoint per method; no significance/seed claim. Equal-degree scores emphasize low-energy spectral tails; energy-weighted scores emphasize large-scale power. These are not GraphCast loss."},
        "closest_three": [label(channels[i]) for i in order[:3]],
        "poorest_three": [label(channels[i]) for i in order[-3:][::-1]],
        "relative_counts": {key: {metric: int(np.sum(scores[metric][1] < scores[metric][mi]))
                                  for metric in ("equal_degree", "energy")}
                            for mi, (key, _, _, _) in enumerate(MODELS) if mi != 1},
        "checks": {"input_hashes_and_cohorts_match": True, "moment_identity_relative_max": closure,
                   "channel_count": len(channels), "atlas_pages_each": len(groups),
                   "maximum_absolute_coherence": float(np.abs(values["coherence"]).max())},
        "finite_reset_rerun_diagnostics": numerical_diagnostics,
        "data": [base.pin(csv_path), base.pin(curve_path)],
        "figures": [base.pin(p) for p in outputs],
    }
    # A partial rendering batch never claims absent or stale metrics. Each batch
    # retains its own provenance; sequential batches consolidate the same lead.
    context = {key: summary[key] for key in ("cohort", "lead_hours", "models", "sources", "data")}
    for metric, paths in metric_outputs.items():
        batch = {"context": context, "metric": metric, "figures": [base.pin(p) for p in paths]}
        (data / f"figures_{suffix}_{metric}.json").write_text(json.dumps(batch, indent=2) + "\n")
    available, figures, skipped = [], [], []
    for metric in METRICS:
        batch_path = data / f"figures_{suffix}_{metric}.json"
        if not batch_path.exists():
            continue
        batch = json.loads(batch_path.read_text())
        if batch.get("context") != context:
            skipped.append(metric)
            continue
        if any(base.pin(ROOT / item["path"])["sha256"] != item["sha256"] for item in batch["figures"]):
            raise ValueError(f"Previously rendered {metric} figures changed")
        available.append(metric)
        figures.extend(batch["figures"])
    summary.update(figures=figures, metrics_rendered_this_run=args.metrics,
                   metrics_available=available, all_metrics_complete=set(available) == set(METRICS),
                   ignored_stale_metric_manifests=skipped)
    summary_path = data / f"summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = [f"Day {day:g} per-feature spectral diagnostics", "", COHORT_CAPTION,
        "Fixed original M1 and matched reset/reset SWAs.",
        "Models: raw GraphCast, raw M1, GraphCast + development-fitted optimal filter, raw matched reset/reset.", "",
        "Verified plot metrics currently available: " + ", ".join(available) + ".",
        "Read power, amplitude ratio and coherence separately. Every atlas includes all83 channels on7 pages.",
        "Best/worst panels select the exact top/bottom3 by M1 equal-degree power mismatch. Familiar panels use fixed common fields.",
        "Panel labels compare M1 and original raw GraphCast using the same equal-degree power mismatch (%).",
        "Equal-degree mismatch = 100 mean_l |Pf/Pt - 1|; energy-weighted mismatch = 100 sum_l |Pf-Pt| / sum_l Pt.",
        "Both use degrees1..179 after pooling dates. Neither is forecast MSE or exact GraphCast loss.",
        "A low-energy spectral tail can dominate equal-degree mismatch while contributing little total power.",
        "Energy-weighted matching alone can conceal high-wavenumber smoothing. Inspect both plus the full curves.", "",
        "Selected fields: equal-degree power mismatch (%), M1 raw and GraphCast raw", ""]
    for index in sorted(set(selected + familiar), key=lambda i: ranks[i]):
        report.append(f"{label(channels[index]):8s}: M1 {scores['equal_degree'][1,index]:8.3f}%; GC {scores['equal_degree'][0,index]:8.3f}%; M1 rank {ranks[index]}/83")
    report += ["", "Relative counts (descriptive only):", json.dumps(summary["relative_counts"], indent=2), "",
        f"All scores and plots use the same {cohort['count']} fixed dates; they do not establish annual representativeness or statistical significance.",
        "The earlier reset report retains finite rerun drift; this script does not introduce a noise-based gate.",
        "No new forecast, training, filter fitting or model selection was performed.", "",
        f"Numeric ranking: {csv_path.relative_to(ROOT)}", f"Figure folder: {images.relative_to(ROOT)}",
        "Templates: AMSE arXiv:2501.19374v2 Fig3 (amplitude/coherence), GraphCast arXiv:2212.12794v2 Fig38 (physical power)."]
    (data / f"REPORT_{suffix}.txt").write_text("\n".join(report) + "\n")
    print(json.dumps({"figures_rendered": len(outputs), "figures_available": len(figures),
                      "metrics_available": available, "cohort": cohort,
                      "closest": summary["closest_three"], "poorest": summary["poorest_three"],
                      "relative_counts": summary["relative_counts"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
