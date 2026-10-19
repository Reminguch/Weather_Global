"""Plot the four-model AMSE feature report from prepared, saved spectra.

Run with::

    source scripts/graphcast_env.sh && python scripts/analyze_models/plot_v24_amse_feature_report.py

The input NPZ is prepared separately from the existing evaluations. No model
evaluation is performed here. Each line plot contains exactly four model curves;
absolute power also includes a separately labeled ERA5 truth reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
NAME = "v24_amse_feature_report_20261004"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
LEADS = (24, 72, 168, 240)
MODELS = (
    ("original_gc", "Original GC", "#606871"),
    ("mse", "MSE", "#0072B2"),
    ("full_amse", "Full AMSE", "#D55E00"),
    ("amse_lambda0p1", "AMSE λ=0.1", "#009E73"),
)
FIELDS = {
    "T2m": ("2 m temperature", "K", "K²"),
    "U10": ("10 m zonal wind", "m s⁻¹", "m² s⁻²"),
    "V10": ("10 m meridional wind", "m s⁻¹", "m² s⁻²"),
    "MSLP": ("Mean sea-level pressure", "Pa", "Pa²"),
    "TP6": ("6-hour accumulated precipitation", "m", "m²"),
    "T850": ("Temperature at 850 hPa", "K", "K²"),
    "Q700": ("Specific humidity at 700 hPa", "kg kg⁻¹", "(kg kg⁻¹)²"),
    "U850": ("Zonal wind at 850 hPa", "m s⁻¹", "m² s⁻²"),
    "V850": ("Meridional wind at 850 hPa", "m s⁻¹", "m² s⁻²"),
    "WS10": ("10 m wind speed", "m s⁻¹", "m² s⁻²"),
    "WS850": ("Wind speed at 850 hPa", "m s⁻¹", "m² s⁻²"),
}
METRICS = (
    ("power", "Spectral power", "Squared Fourier amplitudes, averaged over latitude and starts"),
    ("relative_power_error_pct", "Relative spectral power error",
     "100 × |forecast power − ERA5 power| / ERA5 power · Lower is better"),
    ("coherence", "Forecast–truth spectral coherence",
     "Real cospectrum / sqrt(forecast power × ERA5 power) · Higher is better"),
)
SCOPE = "1° grid · 8 fixed 2023 starts · MSE/AMSE SWA checkpoints · 30–60° latitude in both hemispheres"


def identity(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_curves():
    with np.load(DATA / "spectra.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    require(set(raw["model_ids"].tolist()) == {model[0] for model in MODELS}, "Wrong models")
    require(set(raw["variables"].tolist()) == set(FIELDS), "Wrong spectral fields")
    np.testing.assert_array_equal(raw["lead_hours"], np.arange(6, 241, 6))
    np.testing.assert_array_equal(raw["wavenumber"], np.arange(181))
    require(raw["truth_power"].shape == (40, 11, 181), "Unexpected truth shape")
    order = [list(raw["model_ids"]).index(model[0]) for model in MODELS]
    truth = raw["truth_power"][..., 1:]
    require(np.isfinite(truth).all() and (truth > 0).all(), "Invalid truth powers")
    require(np.all(raw["truth_power"][..., 0] == 0), "Zonal means have not been removed")
    values = {}
    for key in ("power", "cross_power", "error_power"):
        require(raw[key].shape == (4, 40, 11, 181), f"Unexpected {key} shape")
        values[key] = raw[key][order, ..., 1:]
        require(np.isfinite(values[key]).all(), f"Nonfinite {key}")
    require(np.all(values["power"] > 0), "Nonpositive forecast power")
    require(np.all(values["error_power"] >= 0), "Negative forecast error power")
    np.testing.assert_allclose(values["error_power"], values["power"] + truth - 2 * values["cross_power"],
                               rtol=1e-10, atol=1e-10)
    values["relative_power_error_pct"] = 100 * np.abs(values["power"] - truth) / truth
    values["coherence"] = values["cross_power"] / np.sqrt(values["power"] * truth)
    require(np.isfinite(values["coherence"]).all(), "Nonfinite coherence")
    require(np.all(np.abs(values["coherence"]) <= 1 + 1e-6), "Coherence outside [-1, 1]")
    require(np.isfinite(values["relative_power_error_pct"]).all(), "Nonfinite power error")
    indices = [list(raw["lead_hours"]).index(lead) for lead in LEADS]
    curves = {key: array[:, indices] for key, array in values.items()}
    curves["truth_power"] = truth[indices]
    # No coherence clipping: preserve negative values wherever they occur.
    return curves, raw["variables"].tolist(), raw["wavenumber"][1:]


def save_figure(fig, filename, title, metric, variable=None):
    paths = {}
    for suffix in ("png", "pdf"):
        path = IMAGES / f"{filename}.{suffix}"
        fig.savefig(path, dpi=160, facecolor="white")
        paths[suffix] = str(path.relative_to(ROOT))
    plt.close(fig)
    return dict(title=title, metric=metric, variable=variable, **paths)


def plot_field(curves, variables, k, variable, metric, metric_title, formula):
    vi = variables.index(variable)
    name, unit, power_unit = FIELDS[variable]
    title = f"{name} ({variable}; {unit})"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
    fig.suptitle(title, y=.971, fontsize=16, weight="bold")
    fig.text(.5, .926, metric_title, ha="center", fontsize=13)
    fig.text(.5, .888, formula, ha="center", fontsize=9.5, color="#485360")
    ylabel = {"power": f"Zonal power ({power_unit})",
              "relative_power_error_pct": "Relative power error (%)",
              "coherence": "Forecast–truth coherence"}[metric]
    for li, ax in enumerate(axes.ravel()):
        for mi, (_, label, color) in enumerate(MODELS):
            ax.plot(k, curves[metric][mi, li, vi], color=color, lw=1.65, label=label)
        require(len(ax.lines) == 4, "A panel must contain exactly four model curves")
        if metric == "power":
            ax.plot(k, curves["truth_power"][li, vi], color="#151515", lw=1.4,
                    ls="--", label="ERA5 truth (reference)", zorder=5)
        ax.set_title(f"Day {LEADS[li] // 24}", fontsize=12, pad=8)
        ax.set_xscale("log")
        ax.set_xlim(1, 180)
        ax.set_xlabel("Zonal wavenumber (cycles / latitude circle)")
        ax.set_ylabel(ylabel)
        ax.tick_params(labelbottom=True)
        ax.set_xticks([1, 10, 100])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(which="major", color="#DDE2E6", alpha=.8, lw=.7)
        ax.grid(which="minor", axis="x", color="#E9ECEF", alpha=.6, lw=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "power":
            ax.set_yscale("log")
    if metric == "relative_power_error_pct":
        axes[0, 0].set_ylim(bottom=0)
    elif metric == "coherence":
        minimum = float(np.min(curves[metric][:, :, vi]))
        bottom = min(0, np.floor((minimum - .02) * 10) / 10) if minimum < 0 else 0
        axes[0, 0].set_ylim(bottom, 1.025)
        axes[0, 0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="lower center", bbox_to_anchor=(.5, .097), ncol=4,
               frameon=False, fontsize=11, handlelength=2.6, columnspacing=2.1)
    if metric == "power":
        fig.legend(handles[4:], labels[4:], loc="lower center", bbox_to_anchor=(.5, .068),
                   frameon=False, fontsize=9, handlelength=2.8)
    fig.text(.5, .045 if metric == "power" else .065, SCOPE, ha="center", fontsize=9)
    fig.text(.5, .019 if metric == "power" else .037, "Raw powers pooled before ratios · Zonal mean removed · k = 1–180 · No smoothing",
             ha="center", fontsize=9, color="#485360")
    fig.subplots_adjust(left=.095, right=.973, top=.821, bottom=.22, hspace=.46, wspace=.24)
    return save_figure(fig, f"{metric}_{variable}", f"{title}: {metric_title.lower()}", metric, variable)


def plot_overview(curves, variables, metric):
    power_error = metric == "relative_power_error_pct"
    means = curves[metric].mean(axis=-1)
    field_order = [variables.index(variable) for variable in FIELDS]
    matrices = means[:, :, field_order].transpose(0, 2, 1)
    title = "Spectral power error across all fields" if power_error else "Forecast–truth coherence across all fields"
    fig, axes = plt.subplots(1, 4, figsize=(13, 8.5), sharey=True)
    fig.suptitle(title, y=.956, fontsize=17, weight="bold")
    text = ("Mean relative power error (%) · Lower is better" if power_error else
            "Mean real spectral coherence · Higher is better")
    fig.text(.5, .906, text, ha="center", fontsize=12)
    fig.text(.5, .871, "Each cell weights wavenumbers k = 1–180 equally. A common color scale is used for all models.",
             ha="center", fontsize=9.5, color="#485360")
    if power_error:
        low, high, cmap = 0., max(10., np.ceil(matrices.max() / 10) * 10), "YlOrRd"
    else:
        low, high, cmap = min(0., np.floor(matrices.min() * 10) / 10), 1., "YlGnBu"
    for mi, ax in enumerate(axes):
        _, label, color = MODELS[mi]
        image = ax.imshow(matrices[mi], vmin=low, vmax=high, cmap=cmap, aspect="auto")
        ax.set_title(label, color=color, fontsize=12, weight="bold", pad=12)
        ax.set_xticks(range(4), [f"D{lead // 24}" for lead in LEADS])
        ax.set_xlabel("Forecast day", labelpad=9)
        ax.set_yticks(range(len(FIELDS)), list(FIELDS))
        ax.tick_params(length=0)
        for row in range(len(FIELDS)):
            for col in range(4):
                value = matrices[mi, row, col]
                norm = (value - low) / (high - low)
                ax.text(col, row, f"{value:.1f}" if power_error else f"{value:.2f}", ha="center", va="center",
                        color="white" if norm > .58 else "#17222E", fontsize=9)
        ax.spines[:].set_visible(False)
    colorbar_ax = fig.add_axes((.931, .241, .013, .552))
    fig.colorbar(image, cax=colorbar_ax, label="Relative power error (%)" if power_error else "Coherence")
    fig.text(.5, .13, SCOPE, ha="center", fontsize=9)
    fig.text(.5, .095, "Field names and physical units are given on the individual diagnostic pages. No uncertainty intervals.",
             ha="center", fontsize=9, color="#485360")
    fig.subplots_adjust(left=.087, right=.909, top=.793, bottom=.24, wspace=.2)
    return save_figure(fig, f"overview_{metric}", title, f"overview_{metric}")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    curves, variables, k = load_curves()
    IMAGES.mkdir(parents=True, exist_ok=True)
    figures = [plot_overview(curves, variables, metric)
               for metric in ("relative_power_error_pct", "coherence")]
    for variable in FIELDS:
        for metric, title, formula in METRICS:
            figures.append(plot_field(curves, variables, k, variable, metric, title, formula))
    manifest = {
        "script": identity(Path(__file__)), "source": identity(DATA / "spectra.npz"),
        "model_labels": [model[1] for model in MODELS], "lead_days": [lead // 24 for lead in LEADS],
        "variable_order": list(FIELDS), "figures": figures,
        "validation": {
            "four_models_per_line_panel": True,
            "absolute_power_has_separate_era5_truth_reference": True,
            "positive_finite_powers": True,
            "finite_power_errors_and_coherences": True,
            "coherence_range_valid_and_no_clipping": True,
            "spectral_error_identity_rtol_atol": [1e-10, 1e-10],
            "coherence_minimum": float(curves["coherence"].min()),
            "coherence_maximum": float(curves["coherence"].max()),
        },
    }
    (IMAGES / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figure_count": len(figures), "manifest": str(IMAGES / "figures_manifest.json"),
                      "validation": manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
