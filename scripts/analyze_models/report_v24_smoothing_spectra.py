#!/usr/bin/env python3
"""Plot archived GC/M1 smoothing spectra without rerunning forecasts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def divide(numerator, denominator):
    output = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path(
        "artifacts/evaluations/v24_smoothing_m1_20261107"))
    parser.add_argument("--name", default="v24_smoothing_m1_spectra_20261109")
    args = parser.parse_args()
    data_dir = Path("plots/analyze_models/data/resolution_eval") / args.name
    image_dir = Path("plots/analyze_models/images/resolution_eval") / args.name
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.campaign / "report.json"
    report = json.loads(report_path.read_text())
    pooled_path = args.campaign / "annual_pooled.npz"
    if sha256(pooled_path) != report["pooled_statistics"]["sha256"]:
        raise ValueError("Pooled statistics do not match the completed report")
    with np.load(pooled_path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        forecast = saved["prediction_power"]
        truth = saved["truth_power"]
        cross = saved["cross_power"]
        loss = saved["exact_gc_loss"]
        archived_macro = saved["macro_power_mismatch"]
    if forecast.shape != (40, 2, 3, 83, 180) or truth.shape != (40, 83, 180):
        raise ValueError("Unexpected spherical-spectrum layout")
    if metadata["branches"] != ["GC", "M1"] or metadata["variants"] != ["raw", "gaussian", "band"]:
        raise ValueError("Unexpected model/filter order")
    if report["count"] != 122 or len(metadata["initialization_ids"]) != 122:
        raise ValueError("Unexpected initialization coverage")
    for array in (forecast, truth, cross, loss):
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite saved physical statistics")

    ratio = divide(forecast, truth[:, None, None])
    correlation = divide(cross, np.sqrt(forecast * truth[:, None, None]))
    # Average ratios only after the archived powers have been pooled over starts.
    mismatch = np.abs(ratio[..., 1:] - 1)
    macro = np.nanmean(mismatch, axis=(-2, -1))
    macro_reproduction_error = float(np.max(np.abs(macro - archived_macro)))
    mean_ratio = np.nanmean(ratio[..., 1:], axis=-2)
    rollout_loss = loss.mean(axis=0)
    rollout_reduction = 100 * (1 - rollout_loss / rollout_loss[0, 0])
    day10_reduction = 100 * (1 - loss[-1] / loss[-1, 0, 0])
    degrees = np.arange(1, 180)
    lead_indices = [3, 11, 27, 39]
    variants = ["raw", "Gaussian", "band"]
    colors = ["#2471A3", "#D55E00"]
    styles = ["-", ":", "--"]
    labels = [f"{branch} {variant}" for branch in ("GC", "M1") for variant in variants]

    def curves(axis, values):
        for branch in range(2):
            for variant in range(3):
                axis.plot(degrees, values[branch, variant], color=colors[branch],
                          linestyle=styles[variant], linewidth=2.1 if variant == 0 else 1.65,
                          alpha=1 if variant != 1 else .8,
                          label=labels[branch * 3 + variant])

    def save(fig, name):
        fig.savefig(image_dir / f"{name}.png", dpi=190, bbox_inches="tight")
        fig.savefig(image_dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, index in zip(axes.flat, lead_indices):
        curves(axis, mean_ratio[index])
        axis.axhline(1, color="#222222", linewidth=1, linestyle="-.", label="ERA5 = 1")
        axis.set_title(f"Day {(index + 1) // 4}")
        axis.set_xlim(1, 179)
        axis.set_ylim(0, max(1.1, float(np.nanmax(mean_ratio[lead_indices])) * 1.07))
        axis.set_xticks([1, 40, 80, 120, 160, 179])
        axis.grid(alpha=.18)
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean forecast / ERA5 power\n(equal weight across 83 channels)")
    for axis in axes[-1]:
        axis.set_xlabel("Spherical harmonic degree ℓ — higher means finer scales")
    fig.suptitle("GC and M1: power spectra before and after smoothing", fontsize=15)
    fig.text(.5, .94, "122 paired 2023 initializations; powers pooled over starts before ratios; degree zero excluded",
             ha="center", fontsize=10)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, .09, 1, .92])
    save(fig, "power_ratio_by_degree")

    channels = [tuple(channel) for channel in metadata["layout"]["channels"]]
    fields = [
        ("2m_temperature", None, "2-m temperature", "K²"),
        ("temperature", 850., "850-hPa temperature", "K²"),
        ("geopotential", 500., "500-hPa geopotential", "m⁴ s⁻⁴"),
        ("u_component_of_wind", 850., "850-hPa zonal wind", "m² s⁻²"),
        ("specific_humidity", 700., "700-hPa specific humidity", "(kg/kg)²"),
        ("total_precipitation_6hr", None, "6-hour precipitation", "m²"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    field_rows = []
    for axis, (variable, level, title, unit) in zip(axes.flat, fields):
        channel = channels.index((variable, level))
        curves(axis, forecast[-1, :, :, channel, 1:])
        axis.plot(degrees, truth[-1, channel, 1:], color="#222222", linewidth=2, label="ERA5 truth")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1, 179)
        axis.set_title(title)
        axis.set_xlabel("Spherical harmonic degree ℓ")
        axis.set_ylabel(f"Power per degree ({unit})")
        axis.grid(alpha=.18, which="both")
        for branch in range(2):
            for variant in range(3):
                for degree in degrees:
                    field_rows.append({"variable": variable, "level_hpa": level,
                        "lead_days": 10, "model_filter": labels[branch * 3 + variant],
                        "degree": int(degree), "forecast_power": forecast[-1, branch, variant, channel, degree],
                        "truth_power": truth[-1, channel, degree],
                        "power_ratio": ratio[-1, branch, variant, channel, degree],
                        "spectral_correlation": correlation[-1, branch, variant, channel, degree]})
    fig.suptitle("Day-10 physical power spectra: GC, M1 and their smoothing baselines", fontsize=15)
    fig.text(.5, .94, "122 paired 2023 initializations; physical spherical powers, no climatology subtraction",
             ha="center", fontsize=10)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, .09, 1, .92])
    save(fig, "physical_power_day10")

    rows = []
    for branch in range(2):
        for variant in range(3):
            rows.append({"model_filter": labels[branch * 3 + variant],
                "exact_gc_rollout_loss": float(rollout_loss[branch, variant]),
                "rollout_loss_reduction_pct": float(rollout_reduction[branch, variant]),
                "day10_loss_reduction_pct": float(day10_reduction[branch, variant]),
                "rollout_mean_relative_power_error_pct": float(100 * macro[:, branch, variant].mean()),
                "day10_relative_power_error_pct": float(100 * macro[-1, branch, variant])})
    write_csv(data_dir / "summary.csv", rows)
    write_csv(data_dir / "physical_power_day10.csv", field_rows)
    degree_rows = []
    for index in lead_indices:
        for branch in range(2):
            for variant in range(3):
                for degree in degrees:
                    degree_rows.append({"lead_days": (index + 1) / 4,
                        "model_filter": labels[branch * 3 + variant], "degree": int(degree),
                        "mean_channel_power_ratio": mean_ratio[index, branch, variant, degree - 1],
                        "mean_channel_absolute_relative_power_error": float(np.nanmean(
                            mismatch[index, branch, variant, :, degree - 1]))})
    write_csv(data_dir / "power_ratio_by_degree.csv", degree_rows)
    provenance = {
        "campaign": str(args.campaign), "campaign_hash": metadata["campaign_hash"],
        "source_report_sha256": sha256(report_path), "pooled_statistics_sha256": sha256(pooled_path),
        "script_sha256": sha256(__file__), "initializations": report["count"],
        "macro_reproduction_max_absolute_error": macro_reproduction_error,
        "numerical_diagnostics": report["numerical_diagnostics"],
        "metric": "100 * mean over 40 leads, 83 channels and degrees1..179 of abs(pooled_forecast_power / pooled_truth_power - 1)",
        "power_ratio_figure": "Equal-channel mean of per-channel ratios; not total atmospheric power. Excesses and deficits can cancel in this curve, but not in the absolute-error table.",
        "uncertainty": "Descriptive fixed-checkpoint point estimates; no new bootstrap intervals calculated.",
        "summary": rows,
    }
    (data_dir / "summary.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    table = ["| Model/filter | Rollout loss reduction (%) ↑ | Mean power error (%) ↓ | Day-10 power error (%) ↓ |",
             "| --- | ---: | ---: | ---: |"]
    for row in rows:
        table.append(f"| {row['model_filter']} | {row['rollout_loss_reduction_pct']:.2f} | "
                     f"{row['rollout_mean_relative_power_error_pct']:.2f} | {row['day10_relative_power_error_pct']:.2f} |")
    # data/<resolution_eval>/<name> -> analyze_models, then images/...
    relative_images = f"../../../images/resolution_eval/{args.name}"
    text = "\n".join([
        "# GC/M1 smoothing: power-spectrum comparison", "", "Recorded: 2026-11-09 (America/New_York).", "",
        "122 paired 2023 initializations, 40 six-hour leads, principal M1 checkpoint. Gaussian and band filters were fitted independently for GC and M1 on the 32-start 2022 development cohort and applied to forecast outputs, without feeding filtered fields back into the rollout.", "",
        *table, "",
        "Loss reduction uses raw GC as the common denominator, pooling losses before taking ratios. Mean power error pools physical spherical powers across starts first, then averages 100×|forecast/truth power−1| equally over 83 channels, degrees1–179 and all 40 leads. Degree zero is excluded; no climatological mean is removed. Lower power error means better amplitude fidelity, not necessarily better spatial alignment. This metric differs from the weighted GC loss.", "",
        "Raw M1 has lower forecast loss than both smoothed-GC baselines and lower macro power error than either, but its power error is higher than raw GC. Both filter families lower rollout loss while worsening overall power matching for each model. The effect varies by degree and channel; some high-degree bins have excess power, so the curves do not support uniform smoothing at every scale.", "",
        f"![Power ratios]({relative_images}/power_ratio_by_degree.png)", "",
        "The first figure averages dimensionless channelwise power ratios, not powers in incompatible physical units. This mean can hide offsetting deficits/excesses; use the absolute-error table and individual spectra alongside it.", "",
        f"![Day-10 physical spectra]({relative_images}/physical_power_day10.png)", "",
        "Point estimates describe these fixed checkpoints on the reused 2023 cohort. No spectral bootstrap intervals were computed for this report. Finite numerical diagnostics are preserved in summary.json. These results do not identify a physical mechanism or establish that the M1 advantage is entirely explained by smoothing.", "",
        "Reproduce from the repository root:", "", "```bash", "source scripts/graphcast_env.sh",
        f"python scripts/analyze_models/report_v24_smoothing_spectra.py --campaign {args.campaign} --name {args.name}", "```", "",
    ])
    (data_dir / "report.md").write_text(text)
    print(json.dumps({"data_dir": str(data_dir), "image_dir": str(image_dir),
                      "macro_reproduction_error": macro_reproduction_error, "summary": rows}, indent=2))


if __name__ == "__main__":
    main()
