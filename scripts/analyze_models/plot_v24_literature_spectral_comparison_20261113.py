#!/usr/bin/env python3
"""Literature-style diagnostics from pinned, already pooled GC/M1 statistics.

No forecasts, filter fitting, bootstrap resampling, or checkpoint selection.
Layouts follow GraphCast (arXiv:2212.12794v2), Figures 31/38, and AMSE
(arXiv:2501.19374v2), Figure 3.  This is our 1-degree experiment, not an
evaluation of the AMSE model from that paper.
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
NAME = "v24_literature_spectral_comparison_20261113"
SOURCE = ROOT / "plots/analyze_models/data/resolution_eval/v24_optimal_smoothing_v4_interim92_power_20261113"
FILTERS = ROOT / "artifacts/evaluations/v24_optimal_smoothing_v4_20261112/filters.json"
RESET_CAMPAIGN = "artifacts/evaluations/v24_matched_reset_spectra_20261113"
RESET_ORIGINAL = "artifacts/evaluations/v24_matched_reset_2023_20261018"
RESET_SHA256 = "db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6"
RESET_COLOR = "#9467BD"
RESET_LABEL = "Reset/reset (raw)"
PREVIEW_IDS = ["20230101T000000Z", "20230401T000000Z", "20230703T000000Z", "20231001T000000Z"]
MODELS = (("GraphCast", 0, "#2864A5"), ("M1", 1, "#D97720"))
VARIANTS = (("Raw", 0, "-"), ("Optimal filter", 2, "--"))
FIELDS = (("Z500", "geopotential", 500., r"m$^2$ s$^{-2}$", r"m$^4$ s$^{-4}$"),
          ("T850", "temperature", 850., "K", r"K$^2$"),
          ("Q700", "specific_humidity", 700., r"g kg$^{-1}$", r"(g kg$^{-1}$)$^2$"),
          ("T2m", "2m_temperature", None, "K", r"K$^2$"))
CAPTION = "92/122 matched 2023 starts (Jan 1–Oct 1): interim annual evaluation; fixed original M1 checkpoint."
DETAIL = "Filters fitted on 32 starts in 2022; output filtering only. Point estimates; no seed or date-uncertainty bands."
DEGREES = np.arange(1, 180)
DAYS = np.arange(1, 41) / 4


def pin(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_reset_losses(initialization_ids):
    directory = ROOT / RESET_ORIGINAL
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if (protocol["model_id"] != "R_matched" or not protocol["reset_state_every_step"]
            or protocol["training_state_policy"] != "reset_every_anchor"
            or protocol["checkpoint"]["sha256"] != RESET_SHA256):
        raise ValueError("Expected the matched reset-trained/reset-evaluated SWA control")
    if pin(protocol["checkpoint"]["path"])["sha256"] != RESET_SHA256:
        raise ValueError("Matched reset checkpoint hash mismatch")
    path = directory / "report/per_initialization_losses.csv"
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model_id"] != "R_matched":
                continue
            identity = row["initialization_time"].replace("-", "").replace(":", "")
            if identity not in initialization_ids:
                continue
            key = (identity, int(row["lead_hours"]))
            if key in rows or row["protocol_hash"] != protocol["protocol_hash"]:
                raise ValueError("Duplicate or mismatched reset loss row")
            rows[key] = float(row["model_loss"])
    expected = [(identity, lead) for identity in initialization_ids for lead in range(6, 241, 6)]
    if set(rows) != set(expected):
        raise ValueError("Reset loss dates/leads differ from the spectral comparison")
    losses = np.asarray([rows[key] for key in expected]).reshape(len(initialization_ids), 40)
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("Invalid saved reset loss")
    return losses.mean(axis=0), {"model_id": "R_matched", "checkpoint": protocol["checkpoint"],
        "training_policy": "reset_every_anchor", "forecast_policy": "reset_every_step",
        "count": len(initialization_ids), "sources": [pin(protocol_path), pin(path)],
        "exact_loss_source": "archived matched-reset evaluation, restricted to identical comparison dates"}


def load_reset_spectra(directory, initialization_ids, layout, truth, fields, weights):
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if (summary["status"] not in ("complete", "interim_matched_cohort") or summary["count"] != len(initialization_ids)
            or summary["initialization_ids"] != initialization_ids
            or summary["model_id"] != "R_matched" or not summary["reset_state_every_step"]
            or summary["checkpoint"]["sha256"] != RESET_SHA256
            or summary["layout"] != layout or summary["branches"] != ["GC", "R_matched"]
            or summary["lead_hours"] != list(range(6, 241, 6)) or summary["lmax"] != 179):
        raise ValueError("Reset spectra do not match the checkpoint, cohort, policy or transform")
    path = directory / "power_pooled.npz"
    if pin(path)["sha256"] != summary["data"][path.name]["sha256"]:
        raise ValueError("Reset pooled spectrum hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {key: archive[key] for key in ("prediction_power", "truth_power", "cross_power", "error_power", "exact_gc_loss")}
    if (metadata["initialization_ids"] != initialization_ids
            or metadata["checkpoint_sha256"] != RESET_SHA256
            or metadata["layout"] != layout or metadata["campaign_hash"] != summary["campaign_hash"]):
        raise ValueError("Reset NPZ provenance differs from its report")
    if (any(not np.isfinite(value).all() for value in arrays.values())
            or arrays["prediction_power"].shape != (40, 2, 83, 180)
            or arrays["truth_power"].shape != truth.shape
            or arrays["exact_gc_loss"].shape != (40, 2)
            or np.any(arrays["prediction_power"] <= 0) or np.any(arrays["error_power"] < 0)):
        raise ValueError("Invalid reset spectral arrays")
    # The deterministic verification data must match; forecast drift is recorded,
    # not used as a finite-loss or repeatability gate.
    np.testing.assert_allclose(arrays["truth_power"], truth, rtol=1e-9, atol=1e-12)
    prediction, cross, error = (arrays[key][:, 1] for key in ("prediction_power", "cross_power", "error_power"))
    identity_error = float(np.max(np.abs(prediction + truth - 2 * cross - error) / (prediction + truth)))
    if identity_error > 1e-10:
        raise ValueError("Reset spectral error/cross identity is inconsistent")
    amplitude = np.sum(np.sqrt(prediction[..., 1:] / truth[..., 1:]) * weights[None, :, None], axis=1)
    channel_coherence = cross / np.sqrt(prediction * truth)
    coherence = np.sum(channel_coherence[..., 1:] * weights[None, :, None], axis=1)
    rmse = np.sqrt(error[:, fields].sum(axis=-1))
    rmse[:, 2] *= 1000.
    return {"prediction_power": prediction, "coherence": coherence, "amplitude": amplitude,
        "rmse": rmse, "loss": arrays["exact_gc_loss"][:, 1], "paired_gc_loss": arrays["exact_gc_loss"][:, 0],
        "macro_power_error_pct": 100 * np.abs(prediction[..., 1:] / truth[..., 1:] - 1).mean(axis=(1, 2)),
        "sources": [pin(summary_path), pin(path)], "spectral_error_cross_identity_relative_max": identity_error,
        "maximum_absolute_channel_coherence": float(np.abs(channel_coherence[..., 1:]).max())}


def reciprocal(values):
    values = np.asarray(values, dtype=float)
    return np.divide(40030., values, out=np.full_like(values, np.inf), where=values != 0)


def spatial_axis(axis, *, top=False):
    axis.set_xscale("log")
    axis.set_xlim(1, 179)
    axis.set_xticks([1, 5, 10, 20, 50, 100, 179], ["1", "5", "10", "20", "50", "100", "179"])
    axis.set_xlabel("Total spherical wavenumber ℓ")
    axis.grid(alpha=.18, which="major")
    if top:
        secondary = axis.secondary_xaxis("top", functions=(reciprocal, reciprocal))
        secondary.set_xticks([10000, 4000, 1000, 400, 224], ["10,000", "4,000", "1,000", "400", "224"])
        secondary.set_xlabel("Wavelength (km)", labelpad=5)
        secondary.tick_params(labelsize=8)


def save(fig, images, stem, *, template):
    height = fig.get_figheight()
    # Physical spacing keeps the three footer lines distinct in short panels.
    fig.text(.5, .50 / height, CAPTION, ha="center", fontsize=8)
    fig.text(.5, .32 / height, DETAIL, ha="center", fontsize=8)
    fig.text(.5, .14 / height, template, ha="center", fontsize=7.5, color=".30")
    fig.tight_layout(rect=(0, .80 / height, 1, .925))
    paths = []
    for suffix in ("png", "pdf"):
        path = images / f"{stem}.{suffix}"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_amplitude_comparison(amplitude, coherence, images, lead, stem, reset=None):
    comparisons = [
        {"label": "GraphCast (raw)", "color": "#2864A5", "phase": 0,
         "amplitude": amplitude[lead, 0, 0], "coherence": coherence[lead, 0, 0]},
        {"label": "M1 (raw)", "color": "#D97720", "phase": None,
         "amplitude": amplitude[lead, 1, 0], "coherence": coherence[lead, 1, 0]},
        {"label": "GraphCast + optimal filter", "color": "#238B8E", "phase": 5,
         "amplitude": amplitude[lead, 0, 2], "coherence": coherence[lead, 0, 2]},
    ]
    if reset is not None:
        comparisons.append({"label": RESET_LABEL, "color": RESET_COLOR, "phase": None,
                            "amplitude": reset["amplitude"][lead], "coherence": reset["coherence"][lead]})

    def draw_panel(axis, metric):
        displayed = np.stack([item[metric] for item in comparisons])
        for item in comparisons:
            style = {"color": item["color"], "label": item["label"], "lw": 1.8}
            if metric == "coherence":
                style["ls"] = "--" if item["phase"] is None else (item["phase"], (5, 5))
                style["dash_capstyle"] = "butt"
            axis.plot(DEGREES, item[metric], **style)
        axis.set_ylim(min(-.04, float(displayed.min()) - .04),
                      max(1.06, float(displayed.max()) + .06))
        axis.axhline(1., color=".35", ls=":", lw=1)
        axis.set_ylabel("Coherence" if metric == "coherence" else
                        r"Amplitude ratio $\sqrt{P_{\mathrm{forecast}} / P_{\mathrm{ERA5}}}$")
        spatial_axis(axis, top=True)
        axis.legend(loc="lower left", frameon=False, fontsize=9.5, handlelength=3)

    template = "Adapted from AMSE Fig. 3 (arXiv:2501.19374v2). Variable/pressure-weighted per-channel ratios; no normalization by field standard deviations."
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    for axis, metric, title in zip(axes, ("coherence", "amplitude"),
                                   ("Coherence", "Power-spectrum amplitude")):
        draw_panel(axis, metric)
        axis.set_title(title, pad=50, fontsize=12)
    fig.suptitle(f"Day {DAYS[lead]:g}: spectral amplitude and pattern agreement",
                 y=.995, fontsize=14)
    paths = save(fig, images, stem, template=template)
    for metric, title in (("coherence", "spectral coherence"), ("amplitude", "power-spectrum amplitude")):
        fig, axis = plt.subplots(figsize=(9.3, 5.8))
        draw_panel(axis, metric)
        fig.suptitle(f"Day {DAYS[lead]:g}: {title}", y=.995, fontsize=14)
        paths += save(fig, images, stem.replace("amplitude_coherence", metric), template=template)
    return paths


def plot_amplitude(amplitude, coherence, images, lead_indices, stem, reset=None):
    if len(lead_indices) == 1:
        return plot_amplitude_comparison(amplitude, coherence, images, lead_indices[0], stem, reset)
    fig, axes = plt.subplots(len(lead_indices), 2, figsize=(11.4, 3.3 * len(lead_indices) + 1.2),
                             sharex=True, squeeze=False)
    comparisons = [("GraphCast (raw)", 0, 0, "#2864A5", 0),
                   ("M1 (raw)", 1, 0, "#D97720", None),
                   ("GraphCast + optimal filter", 0, 2, "#238B8E", 5)]
    for ri, lead in enumerate(lead_indices):
        for ci, (metric, values) in enumerate((("amplitude", amplitude), ("coherence", coherence))):
            axis = axes[ri, ci]
            displayed = []
            for _, bi, vi, color, phase in comparisons:
                curve = values[lead, bi, vi]
                displayed.append(curve)
                style = "-" if metric == "amplitude" else ("--" if phase is None else (phase, (5, 5)))
                axis.plot(DEGREES, curve, color=color, ls=style, lw=1.8, dash_capstyle="butt")
            if reset is not None:
                displayed.append(reset[metric][lead])
                axis.plot(DEGREES, reset[metric][lead], color=RESET_COLOR,
                          ls="-" if metric == "amplitude" else "--", lw=1.8)
            axis.axhline(1., color=".3", ls=":", lw=1)
            displayed = np.stack(displayed)
            axis.set_ylim(min(-.04, float(displayed.min()) - .04), max(1.06, float(displayed.max()) + .06))
            lead_label = "6 hours" if lead == 0 else f"{DAYS[lead]:g} days"
            axis.set_title(f"{lead_label} · {metric.capitalize()}", fontsize=11, pad=45 if ri == 0 else 8)
            axis.set_ylabel("Amplitude ratio" if metric == "amplitude" else "Coherence")
            spatial_axis(axis, top=ri == 0)
    handles = [Line2D([0], [0], color=color, lw=2, label=model) for model, _, _, color, _ in comparisons]
    if reset is not None:
        handles.append(Line2D([0], [0], color=RESET_COLOR, lw=2, label=RESET_LABEL))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .955), ncol=len(handles), frameon=False)
    fig.suptitle("Spectral amplitude and pattern agreement", y=.995, fontsize=14)
    return save(fig, images, stem, template="Layout: AMSE Fig. 3 (arXiv:2501.19374v2). Variable/pressure-weighted per-channel ratios; no normalization by field standard deviations.")


def plot_rmse(rmse, images, reset=None):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.0), sharex=True)
    for fi, (label, _, _, unit, _) in enumerate(FIELDS):
        for model, bi, color in MODELS:
            for variant, vi, style in VARIANTS:
                if bi == 1 and vi != 0:
                    continue
                line_color = "#238B8E" if vi == 2 else color
                values = rmse[:, bi, vi, fi]
                axes[0, fi].plot(DAYS, values, color=line_color, ls=style, label=f"{model} · {variant}", lw=1.7)
                axes[1, fi].plot(DAYS, 100 * values / rmse[:, 0, 0, fi], color=line_color, ls=style, lw=1.7)
        if reset is not None:
            axes[0, fi].plot(DAYS, reset["rmse"][:, fi], color=RESET_COLOR, lw=1.7, label=RESET_LABEL)
            axes[1, fi].plot(DAYS, 100 * reset["rmse"][:, fi] / rmse[:, 0, 0, fi], color=RESET_COLOR, lw=1.7)
        axes[0, fi].set_title(label)
        axes[0, fi].set_ylabel(f"Spectral RMSE ({unit})")
        axes[1, fi].set_ylabel("RMSE / raw GC RMSE (%)")
        axes[1, fi].axhline(100, color=".35", ls=":", lw=1)
        for axis in axes[:, fi]:
            axis.axvspan(5, 10, color=".5", alpha=.055)
            axis.axvline(5, color=".5", ls=":", lw=.8)
            axis.set(xlim=(.25, 10), xlabel="Forecast lead (days)", xticks=[1, 3, 5, 7, 10])
            axis.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .952), ncol=len(handles), frameon=False)
    fig.suptitle("Forecast accuracy before and after optimal filtering", y=.995, fontsize=14)
    return save(fig, images, "graphcast_style_spectral_rmse", template="Layout: GraphCast Fig. 31 (arXiv:2212.12794v2). RMSE of represented T179 coefficients, including degree 0; not exact native-grid RMSE. Gray region: beyond M1's 5-day feedback tail.")


def plot_spectra(forecast, truth, fields, lead, images, reset=None):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.7), sharex=True)
    for fi, (label, _, _, _, power_unit) in enumerate(FIELDS):
        channel = fields[fi]
        power_scale = 1e6 if label == "Q700" else 1.
        axes[0, fi].plot(DEGREES, power_scale * truth[lead, channel, 1:], color=".1", lw=1.7, label="ERA5")
        for model, bi, color in MODELS:
            for variant, vi, style in VARIANTS:
                if bi == 1 and vi != 0:
                    continue
                line_color = "#238B8E" if vi == 2 else color
                values = forecast[lead, bi, vi, channel, 1:]
                axes[0, fi].plot(DEGREES, power_scale * values, color=line_color, ls=style, lw=1.6, label=f"{model} · {variant}")
                axes[1, fi].plot(DEGREES, values / truth[lead, channel, 1:], color=line_color, ls=style, lw=1.6)
        if reset is not None:
            values = reset["prediction_power"][lead, channel, 1:]
            axes[0, fi].plot(DEGREES, power_scale * values, color=RESET_COLOR, lw=1.6, label=RESET_LABEL)
            axes[1, fi].plot(DEGREES, values / truth[lead, channel, 1:], color=RESET_COLOR, lw=1.6)
        axes[0, fi].set_title(label, pad=8)
        axes[0, fi].set_yscale("log")
        axes[0, fi].set_ylabel(f"Power per degree ({power_unit})")
        axes[1, fi].set_ylabel("Forecast / ERA5 power")
        axes[1, fi].axhline(1, color=".25", ls=":", lw=.9)
        for row in range(2):
            spatial_axis(axes[row, fi], top=row == 0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=len(handles), frameon=False)
    lead_label = "6 hours" if lead == 0 else f"{DAYS[lead]:g} days"
    fig.suptitle(f"Physical power spectra and amplitude fidelity · {lead_label}", y=.995, fontsize=14)
    return save(fig, images, f"graphcast_style_spectra_{'6h' if lead == 0 else str(int(DAYS[lead])) + 'd'}",
                template="Layout: GraphCast Fig. 38 (arXiv:2212.12794v2). Physical-field spectra; no climatology subtraction. Positive degrees 1–179; all model values retained.")


def plot_exact_loss(loss, images, reset_loss):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for model, bi, color in MODELS:
        for variant, vi, style in VARIANTS:
            if bi == 1 and vi != 0:
                continue
            line_color = "#238B8E" if vi == 2 else color
            axes[0].plot(DAYS, loss[:, bi, vi], color=line_color, ls=style, lw=1.8, label=f"{model} · {variant}")
            axes[1].plot(DAYS, 100 * (1 - loss[:, bi, vi] / loss[:, 0, 0]), color=line_color, ls=style, lw=1.8)
    axes[0].plot(DAYS, reset_loss, color=RESET_COLOR, lw=1.8, label=RESET_LABEL)
    axes[1].plot(DAYS, 100 * (1 - reset_loss / loss[:, 0, 0]), color=RESET_COLOR, lw=1.8)
    for axis in axes:
        axis.axvspan(5, 10, color=".5", alpha=.055)
        axis.axvline(5, color=".5", ls=":", lw=.8)
        axis.set(xlim=(.25, 10), xlabel="Forecast lead (days)", xticks=[1, 3, 5, 7, 10])
        axis.grid(alpha=.18)
    axes[0].set_ylabel("Exact original GraphCast loss")
    axes[1].set_ylabel("Loss reduction vs paired raw GC (%)")
    axes[1].axhline(0, color=".35", ls=":", lw=.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .953), ncol=len(handles), frameon=False, fontsize=8.5)
    fig.suptitle("Exact native-grid loss before and after optimal filtering", y=.995, fontsize=14)
    return save(fig, images, "exact_gc_loss_before_after_filtering", template="Lead-time presentation follows GraphCast Fig. 31; ordinate is exact original GraphCast weighted MSE, not RMSE. Gray region: beyond M1's 5-day feedback tail.")


def main():
    global ROOT, CAPTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--filters", type=Path)
    parser.add_argument("--reset-source", type=Path)
    parser.add_argument("--preview", action="store_true",
                        help="Use the four predeclared preview dates in a separate output directory")
    parser.add_argument("--completed-subset", action="store_true",
                        help="Use a pooled same-date interim cohort in a new output directory")
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args()
    if (args.preview or args.completed_subset) and args.name == NAME:
        raise ValueError("A preview/subset must not overwrite the 92-date figure set")
    if args.preview and args.completed_subset:
        raise ValueError("Choose preview or completed subset, not both")
    code_root = ROOT
    ROOT = args.workspace.resolve()
    args.source = (args.source or ROOT / SOURCE.relative_to(code_root)).resolve()
    args.filters = (args.filters or ROOT / FILTERS.relative_to(code_root)).resolve()
    reset_source = (args.reset_source or ROOT / RESET_CAMPAIGN / "report").resolve()
    if args.reset_source is not None and not (reset_source / "summary.json").exists():
        raise FileNotFoundError("Requested reset spectral report is not complete: " + str(reset_source))
    data_dir = ROOT / "plots/analyze_models/data/resolution_eval" / args.name
    images = ROOT / "plots/analyze_models/images/resolution_eval" / args.name
    data_dir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source / "summary.json").read_text())
    filters = json.loads(args.filters.read_text())
    saved_pin = source["data"]["power_pooled.npz"]
    npz_path = args.source / "power_pooled.npz"
    if pin(npz_path)["sha256"] != saved_pin["sha256"]:
        raise ValueError("Pooled-source hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("prediction_power", "truth_power", "cross_power", "error_power", "exact_gc_loss", "spectral_weighted_gc_loss")}
        metadata = json.loads(str(archive["metadata"]))
    if source["campaign_hash"] != filters["campaign_hash"] or metadata["campaign_hash"] != source["campaign_hash"]:
        raise ValueError("Filter/statistics campaign mismatch")
    expected_count = int(source["count"]) if args.completed_subset else (4 if args.preview else 92)
    if args.completed_subset and (expected_count < 1 or expected_count > 92
            or source["status"] != "interim_matched_cohort"):
        raise ValueError("Expected a pinned completed subset of the 92-date spectral campaign")
    if (source["count"] != expected_count or metadata["initialization_ids"] != source["initialization_ids"]
            or len(set(source["initialization_ids"])) != expected_count
            or (args.preview and source["initialization_ids"] != PREVIEW_IDS)):
        raise ValueError("Unexpected or mismatched comparison cohort")
    forecast, truth, cross, error, loss = (arrays[key] for key in ("prediction_power", "truth_power", "cross_power", "error_power", "exact_gc_loss"))
    if forecast.shape != (40, 2, 3, 83, 180) or truth.shape != (40, 83, 180):
        raise ValueError("Unexpected spectral layout")
    if any(not np.isfinite(value).all() for value in arrays.values()) or np.any(forecast <= 0) or np.any(truth <= 0) or np.any(error < 0):
        raise ValueError("Nonfinite or invalid physical moments")
    weights = np.asarray(filters["layout"]["weights"], dtype=float)
    channels = filters["layout"]["channels"]
    expected = []
    for variable, level in channels:
        variable_weight = .1 if variable in ("10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure", "total_precipitation_6hr") else 1.
        levels = [p for name, p in channels if name == variable and p is not None]
        expected.append(variable_weight * level / sum(levels) if level is not None else variable_weight)
    np.testing.assert_allclose(weights, expected, rtol=1e-14, atol=1e-14)
    normalized_weights = weights / weights.sum()
    ratio = forecast / truth[:, None, None]
    amp_by_channel = np.sqrt(ratio)
    coh_by_channel = cross / np.sqrt(forecast * truth[:, None, None])
    amplitude = np.sum(amp_by_channel[..., 1:] * normalized_weights[None, None, None, :, None], axis=-2)
    coherence = np.sum(coh_by_channel[..., 1:] * normalized_weights[None, None, None, :, None], axis=-2)
    fields = [channels.index([variable, level]) for _, variable, level, _, _ in FIELDS]
    rmse = np.sqrt(error[..., fields, :].sum(axis=-1))
    rmse[..., 2] *= 1000.  # Q700: kg/kg to g/kg; ratios are unchanged.
    relative_identity = np.abs(forecast + truth[:, None, None] - 2 * cross - error) / (forecast + truth[:, None, None])
    identity_max = float(relative_identity.max())
    if identity_max > 1e-10:
        raise ValueError("Saved spectral error/cross identity is inconsistent")
    scales = np.asarray(filters["layout"]["scales"])
    reconstructed = np.sum(error.sum(axis=-1) * (weights / scales**2)[None, None, None], axis=-1)
    np.testing.assert_allclose(reconstructed, arrays["spectral_weighted_gc_loss"], rtol=1e-12, atol=1e-12)
    for row in source["rows"]:
        bi, vi = source["branches"].index(row["model"]), source["variants"].index(row["variant"])
        np.testing.assert_allclose(loss[:, bi, vi].mean(), row["exact_gc_rollout_loss"], rtol=1e-12)
    reset_loss, reset_info = load_reset_losses(source["initialization_ids"])
    reset = load_reset_spectra(reset_source, source["initialization_ids"], filters["layout"],
                               truth, fields, normalized_weights)
    reset_info["spectral_status"] = "complete" if reset is not None else "pending_evaluation"
    reset_info["expected_spectral_report"] = str(reset_source.relative_to(ROOT))
    reset_info["filtered_reset"] = "not evaluated; only raw reset/reset is shown"
    if reset is not None:
        reset_info["archived_vs_spectral_rerun_loss"] = {
            "archived_by_lead": reset_loss.tolist(), "rerun_by_lead": reset["loss"].tolist(),
            "difference_by_lead": (reset["loss"] - reset_loss).tolist(),
            "policy": "finite rerun differences are reported, not an acceptance gate"}
        reset_info["paired_gc_rerun_loss_difference_by_lead"] = (reset["paired_gc_loss"] - loss[:, 0, 0]).tolist()
        reset_loss = reset["loss"]
        reset_info["exact_loss_source"] = "same rerun as reset spectral moments"
        reset_info["sources"] += reset["sources"]
        reset_info["spectral_error_cross_identity_relative_max"] = reset["spectral_error_cross_identity_relative_max"]
        CAPTION = "92/122 matched 2023 starts (Jan 1–Oct 1): interim annual evaluation; fixed M1 and reset/reset checkpoints."
    if args.preview:
        CAPTION = "PRELIMINARY: 4 matched 2023 starts (Jan 1, Apr 1, Jul 3, Oct 1); fixed checkpoints."
    if args.completed_subset:
        if reset is None:
            raise ValueError("Every completed-subset comparison requires reset/reset spectra")
        ids = source["initialization_ids"]
        def date_label(identity):
            return f"{identity[:4]}-{identity[4:6]}-{identity[6:8]}"
        CAPTION = (f"INTERIM: {expected_count}/92 matched starts ({date_label(ids[0])} to {date_label(ids[-1])}); "
                   "same dates for all four predictors.")
    records, amplitudes, spectra, lead_scores, weights_rows = [], [], [], [], []
    for channel, weight, normalized in zip(channels, weights, normalized_weights):
        weights_rows.append({"variable": channel[0], "pressure_hpa": channel[1], "loss_channel_weight": weight, "normalized_ratio_weight": normalized})
    for li in range(40):
        for model, bi, _ in MODELS:
            for variant, vi, _ in VARIANTS:
                if bi == 1 and vi != 0:
                    continue
                lead_scores.append({"lead_hours": 6 * (li + 1), "model": model, "variant": variant,
                                    "exact_gc_loss": loss[li, bi, vi],
                                    "exact_gc_loss_reduction_vs_paired_raw_gc_pct": 100 * (1 - loss[li, bi, vi] / loss[li, 0, 0])})
                for fi, field in enumerate(FIELDS):
                    records.append({"lead_hours": 6 * (li + 1), "model": model, "variant": variant, "field": field[0],
                                    "represented_t179_physical_rmse": rmse[li, bi, vi, fi], "rmse_units": field[3],
                                    "rmse_pct_of_paired_raw_gc": 100 * rmse[li, bi, vi, fi] / rmse[li, 0, 0, fi]})
                for ki, degree in enumerate(DEGREES):
                    amplitudes.append({"lead_hours": 6 * (li + 1), "model": model, "variant": variant, "degree": degree,
                                       "wavelength_km": 40030. / degree, "weighted_mean_amplitude_ratio": amplitude[li, bi, vi, ki],
                                       "weighted_mean_coherence": coherence[li, bi, vi, ki]})
                if li in (0, 19, 39):
                    for fi, channel in enumerate(fields):
                        for degree in DEGREES:
                            spectra.append({"lead_hours": 6 * (li + 1), "field": FIELDS[fi][0], "model": model, "variant": variant,
                                            "degree": degree, "power_units": "SI squared units before Q700 plot conversion", "forecast_physical_power": forecast[li, bi, vi, channel, degree],
                                            "truth_physical_power": truth[li, channel, degree], "power_ratio": ratio[li, bi, vi, channel, degree]})
        lead_scores.append({"lead_hours": 6 * (li + 1), "model": "Reset/reset", "variant": "Raw",
                            "exact_gc_loss": reset_loss[li],
                            "exact_gc_loss_reduction_vs_paired_raw_gc_pct": 100 * (1 - reset_loss[li] / loss[li, 0, 0])})
        if reset is not None:
            for fi, field in enumerate(FIELDS):
                records.append({"lead_hours": 6 * (li + 1), "model": "Reset/reset", "variant": "Raw", "field": field[0],
                                "represented_t179_physical_rmse": reset["rmse"][li, fi], "rmse_units": field[3],
                                "rmse_pct_of_paired_raw_gc": 100 * reset["rmse"][li, fi] / rmse[li, 0, 0, fi]})
            for ki, degree in enumerate(DEGREES):
                amplitudes.append({"lead_hours": 6 * (li + 1), "model": "Reset/reset", "variant": "Raw", "degree": degree,
                                   "wavelength_km": 40030. / degree, "weighted_mean_amplitude_ratio": reset["amplitude"][li, ki],
                                   "weighted_mean_coherence": reset["coherence"][li, ki]})
            if li in (0, 19, 39):
                for fi, channel in enumerate(fields):
                    for degree in DEGREES:
                        spectra.append({"lead_hours": 6 * (li + 1), "field": FIELDS[fi][0], "model": "Reset/reset", "variant": "Raw",
                                        "degree": degree, "power_units": "SI squared units before Q700 plot conversion",
                                        "forecast_physical_power": reset["prediction_power"][li, channel, degree],
                                        "truth_physical_power": truth[li, channel, degree],
                                        "power_ratio": reset["prediction_power"][li, channel, degree] / truth[li, channel, degree]})
    for filename, rows in (("spectral_rmse_by_lead.csv", records), ("amplitude_coherence_by_degree.csv", amplitudes),
                           ("physical_spectra_selected_leads.csv", spectra), ("exact_gc_loss_by_lead.csv", lead_scores),
                           ("channel_weights.csv", weights_rows)):
        write_csv(data_dir / filename, rows)
    outputs = []
    with plt.rc_context({"font.size": 9.5, "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False}):
        outputs += plot_amplitude(amplitude, coherence, images, [19], "amse_style_amplitude_coherence_5d", reset)
        outputs += plot_amplitude(amplitude, coherence, images, [39], "amse_style_amplitude_coherence_10d", reset)
        outputs += plot_amplitude(amplitude, coherence, images, [0, 19, 39], "amse_style_amplitude_coherence_6h_5d_10d", reset)
        outputs += plot_rmse(rmse, images, reset)
        for lead in (0, 19, 39):
            outputs += plot_spectra(forecast, truth, fields, lead, images, reset)
        outputs += plot_exact_loss(loss, images, reset_loss)
    contrasts, comparison_rows = {}, []
    for label, indices in (("rollout", np.arange(40)), ("day5", np.asarray([19])), ("day10", np.asarray([39]))):
        pooled_loss = loss[indices].mean(axis=0)
        contrasts[label] = {"raw_m1_vs_raw_gc_loss_reduction_pct": float(100 * (1 - pooled_loss[1, 0] / pooled_loss[0, 0])),
                            "raw_m1_vs_optimal_gc_loss_reduction_pct": float(100 * (1 - pooled_loss[1, 0] / pooled_loss[0, 2])),
                            "exact_losses": {"GC": float(pooled_loss[0, 0]), "M1": float(pooled_loss[1, 0]),
                                             "GC_filtered": float(pooled_loss[0, 2]),
                                             "R_matched": float(reset_loss[indices].mean())},
                            "reset_exact_loss": float(reset_loss[indices].mean()),
                            "raw_m1_vs_reset_loss_reduction_pct": float(100 * (1 - pooled_loss[1, 0] / reset_loss[indices].mean()))}
        if reset is not None:
            contrasts[label]["reset_macro_power_error_pct"] = float(reset["macro_power_error_pct"][indices].mean())
            contrasts[label]["raw_m1_macro_power_error_pct"] = float(100 * np.abs(ratio[indices, 1, 0, :, 1:] - 1).mean())
            baseline_power_mismatch = float(100 * np.abs(ratio[indices, 0, 0, :, 1:] - 1).mean())
            contrasts[label]["power_mismatch_pct"] = {}
            for name, bi, vi in (("GC", 0, 0), ("GC_filtered", 0, 2), ("M1", 1, 0), ("R_matched", None, None)):
                native_loss = float(pooled_loss[bi, vi]) if bi is not None else float(reset_loss[indices].mean())
                mismatch = (float(100 * np.abs(ratio[indices, bi, vi, :, 1:] - 1).mean()) if bi is not None
                            else float(reset["macro_power_error_pct"][indices].mean()))
                contrasts[label]["power_mismatch_pct"][name] = mismatch
                comparison_rows.append({"window": label, "model": name, "forecast_count": expected_count,
                    "lead_count": len(indices), "exact_gc_loss": native_loss,
                    "loss_reduction_vs_raw_gc_pct": 100 * (1 - native_loss / pooled_loss[0, 0]),
                    "power_mismatch_pct": mismatch,
                    "power_mismatch_reduction_vs_raw_gc_pct": 100 * (1 - mismatch / baseline_power_mismatch)})
    if comparison_rows:
        write_csv(data_dir / "comparison_scores.csv", comparison_rows)
    warning = []
    max_coh = float(np.abs(coh_by_channel[..., 1:]).max())
    if max_coh > 1.0003:
        warning.append("Finite coherence exceeds reference 1.0003; values retained without clipping.")
    if reset is None:
        warning.append("Reset/reset spectra pending: only the exact-loss figure currently includes this control.")
    elif reset["maximum_absolute_channel_coherence"] > 1.0003:
        warning.append("Finite reset coherence exceeds reference 1.0003; values retained without clipping.")
    summary = {"description": "Literature-style original-M1/GC/matched-reset diagnostics, not an AMSE model evaluation",
               "status": "interim_matched_cohort" if args.completed_subset else ("preliminary_four_date_preview" if args.preview else "interim_incomplete_annual_evaluation"),
               "count": expected_count, "planned_count": 92 if args.completed_subset else 122,
               "initialization_ids": source["initialization_ids"], "filter_fit": "32 development starts in 2022; frozen before 2023 evaluation",
               "sources": [pin(args.source / "summary.json"), pin(npz_path), pin(args.filters), pin(Path(__file__))],
               "reset_control": reset_info,
               "templates": {"amplitude_coherence": "AMSE arXiv:2501.19374v2 Figure 3", "spectral_rmse": "GraphCast arXiv:2212.12794v2 Figure 31", "power_spectra": "GraphCast arXiv:2212.12794v2 Figure 38"},
               "five_day_amplitude_display": {"panels": 2, "panel_order": ["coherence", "amplitude ratio"],
                   "standalone_exports": ["amse_style_coherence_5d", "amse_style_amplitude_5d"],
                   "models": ["GraphCast raw", "M1 raw", "GraphCast optimal filter"] + (["Reset/reset raw"] if reset is not None else []),
                   "line_styles": {"amplitude_ratio": "solid", "coherence": "dashed",
                       "gc_coherence_overlap": "equal dash/gap lengths (5,5), raw GC phase 0 and filtered GC phase 5, matching linewidth and butt caps; no markers, annotation or value offsets"}},
               "definitions": {"amplitude_ratio": "sqrt(date-pooled forecast power / date-pooled truth power), computed per channel before weighted averaging",
                   "coherence": "date-pooled real cross-power / sqrt(date-pooled forecast power * date-pooled truth power), per channel before weighted averaging",
                   "channel_aggregation": "Loss variable and pressure weights normalized to sum one; no field standard-deviation normalization for dimensionless ratios",
                   "spectral_rmse": "sqrt(sum directly saved physical error power over degrees 0..179)); Q700 converted kg/kg to g/kg; represented spherical mean-square norm, not exact native-grid RMSE",
                   "exact_gc_loss": "Saved exact original native-grid GC weighted MSE; headline contrasts pool losses before division",
                   "power_spectra": "Physical per-degree power, degrees 1..179, no climatology subtraction; no mixed-unit power averaging",
                   "uncertainty": "Fixed-checkpoint descriptive point estimates; no spectral/date confidence intervals, no five-seed replication"},
               "checks": {"saved_pooled_hash_matches": True, "same_cohort_all_panels": True, "channel_weights_match_gc_variable_pressure_definition": True,
                          "source_exact_losses_reproduced": True, "spectral_weighted_loss_reproduced": True,
                          "spectral_error_cross_identity_relative_max": identity_max, "maximum_absolute_channel_coherence": max_coh,
                          "weights_sum": float(weights.sum()), "normalized_weights_sum": float(normalized_weights.sum()), "warnings": warning},
               "contrasts": contrasts, "figures": [pin(path) for path in outputs]}
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (images / "RESET_STATUS.txt").write_text(
        ("Matched reset/reset is included in every figure. Raw reset only; no fitted reset filter.\n"
         if reset is not None else
         "Matched reset/reset is currently included in exact_gc_loss_before_after_filtering only.\n"
         "The six spectral figures await the matched reset spectral evaluation.\n")
        + f"All comparisons use the same {expected_count} initialization dates and original matched SWA checkpoint.\n"
        + "Report: " + str(reset_source) + "\n")
    print(json.dumps({"figures": len(outputs), "contrasts": contrasts, "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
