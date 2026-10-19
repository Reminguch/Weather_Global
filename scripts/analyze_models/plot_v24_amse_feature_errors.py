"""Plot saved four-model AMSE forecast errors without running model evaluation.

Run through scripts/graphcast_env.sh. Inputs are assembled by the companion
feature-report data script; every plotted value comes from report_data.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = "v24_amse_feature_report_20261004"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / REPORT
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / REPORT
MODEL_ORDER = ["original_gc", "mse", "amse_lambda0p1", "full_amse"]
STYLES = {
    "original_gc": {"linestyle": "--", "linewidth": 1.8, "zorder": 3},
    "mse": {"linestyle": "-", "linewidth": 1.9, "zorder": 4},
    "amse_lambda0p1": {"linestyle": (0, (5, 1.5, 1, 1.5)), "linewidth": 1.9, "zorder": 5},
    "full_amse": {"linestyle": "-", "linewidth": 1.9, "zorder": 4},
}
VARIABLES = {
    "2m_temperature": ("2 m temperature", "K", 1.0, False),
    "10m_u_component_of_wind": ("10 m zonal wind", "m s⁻¹", 1.0, False),
    "10m_v_component_of_wind": ("10 m meridional wind", "m s⁻¹", 1.0, False),
    "mean_sea_level_pressure": ("Mean sea-level pressure", "hPa", 0.01, False),
    "total_precipitation_6hr": ("6 h precipitation", "mm", 1000.0, False),
    "temperature": ("Temperature · all pressure levels", "K", 1.0, True),
    "specific_humidity": ("Specific humidity · all levels", "g kg⁻¹", 1000.0, True),
    "u_component_of_wind": ("Zonal wind · all pressure levels", "m s⁻¹", 1.0, True),
    "v_component_of_wind": ("Meridional wind · all levels", "m s⁻¹", 1.0, True),
    "geopotential": ("Geopotential · all pressure levels", "m² s⁻²", 1.0, True),
    "vertical_velocity": ("Vertical velocity · all levels", "Pa s⁻¹", 1.0, True),
}
PRIMITIVE_FEATURES = ["T2m", "U10", "V10", "MSLP", "TP6", "T850", "Q700", "U850", "V850"]


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titlepad": 9, "savefig.facecolor": "white",
    })


def style(ax, *, zero=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E2E6E9", linewidth=0.7)
    ax.set_axisbelow(True)
    if zero:
        ax.set_ylim(bottom=0)
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Forecast lead (days)")


def handles(data):
    return [Line2D([], [], color=data["models"][m]["color"],
                   label=data["models"][m]["label"], **STYLES[m])
            for m in MODEL_ORDER]


def heading(fig, title, dataset, data):
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.981)
    fig.text(0.5, 0.948,
             f"{dataset['n_samples']} matched {dataset['year']} starts · "
             "6-hourly leads through day 10 · residual models use SWA checkpoints",
             ha="center", fontsize=10.5, color="#485460")
    fig.legend(handles=handles(data), loc="upper center", bbox_to_anchor=(0.5, 0.932),
               ncol=4, frameon=False, fontsize=10.5, handlelength=3.3,
               columnspacing=2.1)


def vector(values, n, description):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"Invalid values for {description}: shape={arr.shape}")
    return arr


def save(fig, stem, directory, manifest, description, dataset, **extra):
    outputs = {}
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=170)
        outputs[extension] = str(path.relative_to(ROOT))
    plt.close(fig)
    manifest["figures"].append({
        "stem": stem, "description": description, "dataset": dataset,
        "model_order": MODEL_ORDER, "files": outputs, **extra,
    })


def variable_rmse(data, name, directory, manifest):
    dataset = data["evaluations"][name]
    days = np.asarray(dataset["lead_hours"]) / 24
    fig, axes = plt.subplots(4, 3, figsize=(13.3, 13.0))
    heading(fig, "Forecast RMSE by variable", dataset, data)
    all_values = set(dataset["models"]["mse"]["per_variable_per_step"])
    missing = set(VARIABLES) - all_values
    if missing:
        raise ValueError(f"Missing variables: {missing}")
    for ax, (variable, (label, units, scale, _)) in zip(axes.flat, VARIABLES.items()):
        for model in MODEL_ORDER:
            values = dataset["models"][model]["per_variable_per_step"][variable]["rmse"]
            values = scale * vector(values, len(days), f"{name}/{model}/{variable}")
            ax.plot(days, values, color=data["models"][model]["color"], **STYLES[model])
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_ylabel(f"RMSE ({units})")
        style(ax)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.96,
        "How to read these panels\n\n"
        "Lower RMSE is better.\n"
        "Each panel has its own physical units.\n\n"
        "Upper-air groups pool squared error\n"
        "equally across all 13 pressure levels\n"
        "before taking the square root.",
        va="top", fontsize=10.5, color="#485460", linespacing=1.45)
    fig.text(0.065, 0.047,
             "Global latitude-weighted errors pooled over the saved starts. "
             "These physical-variable RMSEs differ from the exact weighted GraphCast loss.",
             fontsize=9.2)
    fig.text(0.065, 0.029,
             "One common Original GC reference is shown; repeated baseline evaluations have small differences. "
             "No uncertainty intervals are available.", fontsize=9.2, color="#586572")
    fig.subplots_adjust(left=0.075, right=0.978, top=0.877, bottom=0.10,
                        hspace=0.49, wspace=0.30)
    save(fig, f"rmse_variables_{name}", directory, manifest,
         "All 11 model-output variable groups; upper-air errors pool levels equally in MSE space.",
         name, variables=list(VARIABLES), upper_air_level_weighting="equal MSE pooling")


def feature_series(record, feature, n):
    key = feature["metric_key"]
    if key is None:
        raise ValueError("A derived wind-speed feature has no saved global RMSE")
    for source in ("per_channel_per_step", "per_variable_per_step"):
        if key in record[source]:
            return vector(record[source][key]["rmse"], n, key)
    raise KeyError(f"Missing saved RMSE channel: {key}")


def feature_rmse(data, name, directory, manifest):
    dataset = data["evaluations"][name]
    days = np.asarray(dataset["lead_hours"]) / 24
    fig, axes = plt.subplots(3, 3, figsize=(13.3, 10.7))
    heading(fig, "Global RMSE for the spectral diagnostic fields", dataset, data)
    for ax, feature_id in zip(axes.flat, PRIMITIVE_FEATURES):
        feature = data["features"][feature_id]
        scale = float(feature.get("display_scale", 1))
        units = feature.get("display_units", feature["units"])
        for model in MODEL_ORDER:
            values = feature_series(dataset["models"][model], feature, len(days)) * scale
            ax.plot(days, values, color=data["models"][model]["color"], **STYLES[model])
        ax.set_title(f"{feature_id} · {feature['label']}", loc="left", fontweight="bold")
        ax.set_ylabel(f"RMSE ({units})")
        style(ax)
    fig.text(0.065, 0.052,
             "Global latitude-weighted physical RMSE at the specified level; "
             "the companion spectral curves use only 30–60° latitude in each hemisphere.", fontsize=9.1)
    fig.text(0.065, 0.031,
             "Global wind-speed RMSE (WS10 / WS850) was not saved; it cannot be reconstructed from component RMSEs. "
             "No uncertainty intervals.", fontsize=9.1, color="#586572")
    fig.subplots_adjust(left=0.075, right=0.978, top=0.869, bottom=0.117,
                        hspace=0.46, wspace=0.31)
    save(fig, f"rmse_spectral_fields_{name}", directory, manifest,
         "Nine primitive spectral fields with saved global physical RMSE; derived wind speeds excluded.",
         name, features=PRIMITIVE_FEATURES,
         missing_global_rmse_features=["WS10", "WS850"])


def exact_loss(data, directory, manifest):
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.7))
    fig.suptitle("Exact GraphCast forecast loss", fontsize=20, weight="bold", y=0.974)
    fig.text(0.5, 0.929,
             "Original variable weights, difference-standard-deviation normalization, "
             "pressure-level weights and latitude weights",
             ha="center", fontsize=10.5, color="#485460")
    fig.legend(handles=handles(data), loc="upper center", bbox_to_anchor=(0.5, 0.912),
               ncol=4, frameon=False, fontsize=11, handlelength=3.3)
    comparisons = {}
    for row, name in enumerate(("exact32", "spectral8")):
        dataset = data["evaluations"][name]
        days = np.asarray(dataset["lead_hours"]) / 24
        baseline = dataset["models"]["original_gc"]["exact_graphcast_loss"]
        base_steps = vector(baseline["per_step"], len(days), f"{name}/baseline")
        base_rollout = float(baseline["rollout"])
        comparisons[name] = {}
        for model in MODEL_ORDER:
            metric = dataset["models"][model]["exact_graphcast_loss"]
            steps = vector(metric["per_step"], len(days), f"{name}/{model}/exact")
            rollout = float(metric["rollout"])
            if not np.isclose(rollout, steps.mean(), rtol=1e-6, atol=1e-9):
                raise ValueError(f"Rollout loss disagrees with saved per-step mean: {name}/{model}")
            color = data["models"][model]["color"]
            axes[row, 0].plot(days, steps, color=color, **STYLES[model])
            axes[row, 1].plot(days, 100 * (1 - steps / base_steps), color=color, **STYLES[model])
            comparisons[name][model] = {
                "rollout_loss": rollout,
                "reduction_pct_vs_common_gc": 100 * (1 - rollout / base_rollout),
            }
            paired = dataset["models"][model].get("paired_baseline")
            if paired:
                paired_loss = float(paired["exact_graphcast_loss"]["rollout"])
                comparisons[name][model]["reduction_pct_vs_own_paired_gc"] = 100 * (1 - rollout / paired_loss)
        for column in (0, 1):
            style(axes[row, column], zero=column == 0)
        axes[row, 0].set_title(f"{dataset['n_samples']} starts in {dataset['year']} · loss by lead",
                               loc="left", weight="bold", fontsize=12)
        axes[row, 0].set_ylabel("Exact weighted GraphCast MSE")
        axes[row, 1].set_title("Loss reduction vs common Original GC", loc="left", weight="bold", fontsize=11)
        axes[row, 1].set_ylabel("Loss reduction (%) · higher is better")
        labels = [data["models"][m]["label"] for m in MODEL_ORDER]
        values = [comparisons[name][m]["rollout_loss"] for m in MODEL_ORDER]
        ax = axes[row, 2]
        ypos = np.arange(len(MODEL_ORDER))
        ax.barh(ypos, values, color=[data["models"][m]["color"] for m in MODEL_ORDER],
                height=0.61, alpha=0.92)
        ax.set_yticks(ypos, labels, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean exact loss over all 40 leads")
        ax.set_title("10-day rollout loss", loc="left", weight="bold", fontsize=12)
        ax.set_xlim(0, max(values) * 1.57)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#E2E6E9", linewidth=0.7)
        ax.set_axisbelow(True)
        for index, model in enumerate(MODEL_ORDER):
            result = comparisons[name][model]
            suffix = "reference" if model == "original_gc" else f"{result['reduction_pct_vs_common_gc']:+.2f}%"
            ax.text(values[index] + max(values) * 0.025, index,
                    f"{values[index]:.3f}\n{suffix}", va="center", fontsize=9.5)
    fig.text(0.055, 0.067,
             "All plotted reductions use the same displayed Original GC within each dataset. "
             "They are ratios of losses; they are not RMSE reductions.", fontsize=9.8)
    fig.text(0.055, 0.043,
             "Rollout reduction is computed after pooling all 40 leads. "
             "Own-paired baseline results and baseline repeatability differences are retained in the data audit.",
             fontsize=9.5, color="#586572")
    fig.subplots_adjust(left=0.064, right=0.985, top=0.818, bottom=0.135,
                        hspace=0.55, wspace=0.44)
    save(fig, "exact_graphcast_loss", directory, manifest,
         "Exact per-lead and rollout loss for both saved datasets; reductions use a common GC reference.",
         ["exact32", "spectral8"], comparison=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "report_data.json")
    parser.add_argument("--image-dir", type=Path, default=IMAGES)
    parser.add_argument("--manifest", type=Path, default=DATA / "error_figures_manifest.json")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    if data["model_order"] != MODEL_ORDER:
        raise ValueError(f"Unexpected model order: {data['model_order']}")
    configure()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(args.data.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "model_order": MODEL_ORDER, "figures": []}
    for name in ("exact32", "spectral8"):
        variable_rmse(data, name, args.image_dir, manifest)
        feature_rmse(data, name, args.image_dir, manifest)
    exact_loss(data, args.image_dir, manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(manifest['figures'])} error figures as PNG/PDF to {args.image_dir}")


if __name__ == "__main__":
    main()
