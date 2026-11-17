#!/usr/bin/env python3
"""Replay the audited 122-date spectral exports as compact manuscript figures.

No forecast files are read and no metrics are redefined. The portable archive
contains this renderer and its inputs; use --input-dir to replay that archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAG = "v5_variable_spectral_presentation_20261115"
MODELS = ("GC", "M1", "GC_filter", "R1")
LABELS = ("GraphCast", "M1", "GC + optimal filter", "R1")
COLORS = ("#2864A5", "#D97720", "#238B8E", "#9467BD")
FIELDS = ("T2m", "Z500", "U850", "Q100")
DEGREE = np.arange(1, 180)
REFERENCE = np.sqrt(0.75)
STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .6,
         "xtick.major.width": .6, "ytick.major.width": .6,
         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
         "pdf.fonttype": 42, "axes.spines.top": False,
         "axes.spines.right": False, "savefig.facecolor": "white"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, values):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, values[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def prepare_archive(archive):
    original = ROOT / "Overleaf/archive/Graphs/v5_spectral_matched122_20261115"
    audit = ROOT / "plots/analyze_models/data/resolution_eval/v5_variable_spectral_audit_20261115"
    sources = []
    copies = [(original / "weighted_spectral_curves.csv", "weighted_spectral_curves.csv"),
              (original / "all_channels_selected_leads_curves.npz", "all_channels_selected_leads_curves.npz"),
              (original / "summary.json", "aggregate_source_summary.json")]
    copies += [(audit / name, name) for name in (
        "plot_data.npz", "channel_metrics.csv", "channel_contrasts.csv",
        "lead_metrics.csv", "scale_crossings.csv", "independent_validation.json")]
    copies.append((audit / "summary.json", "channel_source_summary.json"))
    for source, name in copies:
        shutil.copy2(source, archive / name)
        sources.append({"path": str(source.relative_to(ROOT)), "archive_name": name,
                        "sha256": sha(source)})
    write_json(archive / "input_manifest.json", sources)


def load_data(archive):
    for entry in json.loads((archive / "input_manifest.json").read_text()):
        if sha(archive / entry["archive_name"]) != entry["sha256"]:
            raise ValueError(f"Input changed: {entry['archive_name']}")
    aggregate = np.full((3, 2, 4, 179), np.nan)
    leads = (6, 120, 240)
    for row in rows(archive / "weighted_spectral_curves.csv"):
        assert int(row["forecast_count"]) == 122
        li = leads.index(int(row["lead_hours"]))
        mi = MODELS.index("R1" if row["model"] == "reset" else row["model"])
        di = int(row["degree"]) - 1
        aggregate[li, :, mi, di] = [float(row["weighted_amplitude_ratio"]),
                                    float(row["weighted_coherence"])]
    with np.load(archive / "plot_data.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(archive / "all_channels_selected_leads_curves.npz", allow_pickle=False) as source:
        # Independent export routes must agree exactly for every displayed
        # channel, model, degree and lead, including unfavorable examples.
        np.testing.assert_allclose(source["amplitude_ratio"][1:], data["selected_amplitude"],
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(source["coherence"][1:], data["selected_coherence"],
                                   rtol=1e-13, atol=1e-13)
    assert np.isfinite(aggregate).all()
    assert data["selected_amplitude"].shape == (2, 4, 83, 179)
    channels = list(data["channels"])
    selected = []
    contrasts = rows(archive / "channel_contrasts.csv")
    for field in FIELDS:
        ci = channels.index(field)
        for window, li in (("day5", 19), ("day10", 39)):
            values = data["rmse"][li, :, ci]
            contrast = next(r for r in contrasts if r["channel"] == field and
                            r["window"] == window and r["reference"] == "R1")
            reduction = 100 * (1 - values[1] / values[3])
            np.testing.assert_allclose(reduction, float(contrast["rmse_reduction_pct"]),
                                       rtol=1e-13, atol=1e-13)
            selected.append({"channel": field, "window": window,
                             "M1_native_grid_rmse": float(values[1]),
                             "R1_native_grid_rmse": float(values[3]),
                             "M1_rmse_reduction_vs_R1_pct": float(reduction)})
    write_csv(archive / "selected_native_grid_rmse.csv", selected)
    return aggregate, data


def line_style(mi, coherence):
    style = {"color": COLORS[mi], "lw": 1.25, "label": LABELS[mi]}
    # Positive gains preserve coherence: stagger only GC and filtered GC.
    if coherence and mi in (0, 2):
        style.update(ls=(5 if mi == 2 else 0, (5, 5)), dash_capstyle="butt")
    return style


def reciprocal(value):
    value = np.asarray(value, dtype=float)
    return np.divide(40030., value, out=np.full_like(value, np.inf), where=value != 0)


def draw(axis, curves, metric, limits, *, top=False, small=False, xlabel=True):
    for mi in range(4):
        line, = axis.plot(DEGREE, curves[mi], **line_style(mi, metric == "coherence"))
        np.testing.assert_array_equal(line.get_ydata(), curves[mi])
        assert limits[0] <= float(curves[mi].min()) <= float(curves[mi].max()) <= limits[1]
    axis.axhline(1, color=".55", ls=":", lw=.65, zorder=0)
    if metric == "amplitude":
        axis.axhline(REFERENCE, color=".3", ls=(0, (4, 2, 1, 2)), lw=.75, zorder=0)
    axis.set_xscale("log")
    axis.set(xlim=(1, 179), ylim=limits)
    if small:
        axis.set_xticks([1, 10, 100], ["1", "10", "100"])
    else:
        axis.set_xticks([1, 10, 100, 179], ["1", "10", "100", "179"])
    axis.tick_params(which="minor", bottom=False)
    if xlabel:
        axis.set_xlabel(r"Total wavenumber $\ell$")
    axis.grid(alpha=.17, linewidth=.5)
    if top:
        secondary = axis.secondary_xaxis("top", functions=(reciprocal, reciprocal))
        secondary.set_xticks([10000, 1000, 224], ["10,000", "1,000", "224"])
        secondary.set_xlabel("Wavelength (km)", labelpad=3)
        secondary.tick_params(labelsize=7, length=2)


def legend(fig, *, y=.012):
    handles = [Line2D([], [], color=color, lw=1.5, label=label)
               for label, color in zip(LABELS, COLORS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=4,
               frameon=False, fontsize=7.5, handlelength=1.8,
               handletextpad=.55, columnspacing=1.25, borderaxespad=0.)


def render(aggregate, data, output):
    checks = {}

    def save(fig, stem):
        fig.canvas.draw()
        digest = hashlib.sha256()
        count = 0
        for axis in fig.axes:
            for line in axis.lines:
                for values in (line.get_xdata(), line.get_ydata()):
                    digest.update(np.asarray(values, dtype="<f8").tobytes())
                count += 1
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=240)
        checks[stem] = {"size_inches": fig.get_size_inches().tolist(), "axes": len(fig.axes),
                        "lines_including_references": count,
                        "coordinate_sha256": digest.hexdigest()}
        plt.close(fig)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(6.25, 3.8), sharey="row")
        for row, metric in enumerate(("amplitude", "coherence")):
            curves = aggregate[1:, row]
            limits = (0 if row == 0 else min(-.025, curves.min() - .025),
                      max(1.05, curves.max() * 1.04))
            for col, day in enumerate((5, 10)):
                draw(axes[row, col], curves[col], metric, limits, top=row == 0)
                if col == 0:
                    axes[row, col].set_ylabel("Amplitude ratio" if row == 0 else "Coherence")
                if row == 0:
                    axes[row, col].set_title(f"Day {day}", fontsize=8.5, pad=39)
        legend(fig)
        fig.subplots_adjust(left=.09, right=.965, bottom=.17, top=.80, hspace=.47, wspace=.25)
        save(fig, "fig6_smoothing_day5_day10_v5_variables122")

        channels = list(data["channels"])
        for li, day in enumerate((5, 10)):
            fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.15), sharex=True)
            for col, field in enumerate(FIELDS):
                ci = channels.index(field)
                all_amp = data["selected_amplitude"][:, :, ci]
                for row, metric in enumerate(("amplitude", "coherence")):
                    curves = data[f"selected_{metric}"][li, :, ci]
                    # Identical limits at days 5 and 10; Z500 tails remain visible.
                    limits = (0, max(1.08, all_amp.max() * 1.04)) if row == 0 else (-.1, 1.05)
                    draw(axes[row, col], curves, metric, limits, small=True, xlabel=row == 1)
                    if row == 0:
                        axes[row, col].set_title(field, pad=5)
                    if col == 0:
                        axes[row, col].set_ylabel("Amplitude ratio" if row == 0 else "Coherence")
                    if row == 1 and col > 0:
                        axes[row, col].tick_params(labelleft=False)
            legend(fig)
            fig.subplots_adjust(left=.075, right=.985, bottom=.21, top=.91, hspace=.18, wspace=.28)
            save(fig, f"fig{11 + li}_smoothing_variables_day{day}_v5_matched122")

        fig, axes = plt.subplots(3, 2, figsize=(6.25, 5.6))
        for li, title in enumerate(("6 hours", "5 days", "10 days")):
            for col, metric in enumerate(("amplitude", "coherence")):
                curves = aggregate[li, col]
                limits = (0 if col == 0 else min(-.025, curves.min() - .025),
                          max(1.05, curves.max() * 1.04))
                draw(axes[li, col], curves, metric, limits, top=li == 0)
                axes[li, col].set_ylabel("Amplitude ratio" if col == 0 else "Coherence")
                axes[li, col].set_title(title, pad=41 if li == 0 else 5)
        legend(fig, y=.009)
        fig.subplots_adjust(left=.095, right=.965, bottom=.115, top=.85, hspace=.50, wspace=.32)
        save(fig, "fig13_smoothing_lead_spectra_v5_variables122")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Replay a previously prepared portable archive")
    parser.add_argument("--archive-dir", type=Path,
                        default=ROOT / "Overleaf/archive/Graphs" / TAG)
    parser.add_argument("--image-dir", type=Path,
                        default=ROOT / "plots/analyze_models/images/resolution_eval" / TAG)
    parser.add_argument("--install-paper-figures", action="store_true")
    args = parser.parse_args()
    archive = args.input_dir or args.archive_dir
    archive.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    if not args.input_dir:
        prepare_archive(archive)
    aggregate, data = load_data(archive)
    checks = render(aggregate, data, args.image_dir)
    write_json(archive / "figure_checks.json", checks)
    write_json(archive / "selection.json", {
        "fields": FIELDS, "days": [5, 10], "models": MODELS,
        "selection": "Illustrative contrasting cases; not a ranking or independent test set.",
        "rationale": {"T2m": "Relatively retained spectrum; surface/terrain caveat.",
                      "Z500": "Standard circulation diagnostic with nontrivial spectral tails.",
                      "U850": "Accuracy/coherence gain accompanied by attenuation.",
                      "Q100": "Adverse day-10 example against matched R1."},
        "reference_amplitude": float(REFERENCE),
        "reference_interpretation": "Per channel, sqrt(0.75) means 25% power deficit. On combined curves it is a weighted-amplitude reference, not 25% mean power deficit.",
        "preserved": "Original equal-degree scores, separate truth-power-weighted scores, all 83 channels, 20/25/30% crossing sensitivity, and finite numerical warnings.",
        "earlier_exploration": "The earlier T2m/T400/U850/Q100 selection is unchanged in v5_variable_spectral_audit_20261115."})
    renderer = archive / Path(__file__).name
    if renderer.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, renderer)
    products = []
    for stem in checks:
        for suffix in ("pdf", "png"):
            source = args.image_dir / f"{stem}.{suffix}"
            shutil.copy2(source, archive / source.name)
            products.append({"filename": source.name, "sha256": sha(source)})
            if args.install_paper_figures and suffix == "pdf":
                shutil.copy2(source, ROOT / "Overleaf/arxiv_pics" / source.name)
    write_json(archive / "manifest.json", {
        "cohort_count": 122, "channels": 83, "source_pins": "input_manifest.json",
        "renderer_sha256": sha(renderer), "products": products,
        "checks": {"full_channel_export_agreement": True, "native_grid_rmse_contrast_agreement": True,
                   "all_plotted_curve_values_verified": True, "no_displayed_data_clipped": True},
        "notes": "Fixed-checkpoint point estimates. No new evaluation or training. Full source cohort and finite warnings are in the archived summaries."})
    (archive / "README.txt").write_text(
        "V5 variable spectral presentation, 15 November 2026\n\n"
        "Figures 6 and 13 retain the original GraphCast variable/pressure-weighted\n"
        "83-channel amplitude and coherence curves. Figures 11 and 12 show T2m,\n"
        "Z500, U850, Q100 at days 5 and 10. Scalar metrics are not drawn on\n"
        "coherence panels. Dotted line: amplitude matching / unit coherence;\n"
        "dash-dot line on amplitude panels: sqrt(0.75). This is a 25% power\n"
        "deficit only for an individual channel, not for the combined curve.\n\n"
        "Native-grid RMSE contrasts: selected_native_grid_rmse.csv. All 83\n"
        "channels and original metric weightings are retained in the source\n"
        "CSVs. The two channel coherence exports agree to rtol=atol=1e-13.\n"
        "Finite rerun warnings remain in channel_source_summary.json.\n\n"
        "Replay with Python/NumPy/Matplotlib:\n"
        "python plot_v5_variable_presentation_20261115.py --input-dir . --image-dir /tmp/v5-variable-replay\n")
    print(json.dumps({"archive": str(archive), "images": str(args.image_dir),
                      "figures": list(checks)}, indent=2))


if __name__ == "__main__":
    main()
