#!/usr/bin/env python3
"""Summarize saved M1 training-step evaluations; never run model inference."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "v24_training_steps_and_k_20261113"
SOURCE = ROOT / "plots/analyze_models/data/resolution_eval/v24_matched_budget_comparison_20261112"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
RAW = ["m1_total1000_raw", "m1_total1250_raw", "M1_raw"]
STYLES = {
    "m1_total1000_raw": ("Raw 1,000 updates", "#0072B2", "-"),
    "m1_total1250_raw": ("Raw 1,250 updates", "#D55E00", "-"),
    "M1_raw": ("Raw 2,500 updates", "#009E73", "-"),
    "M1_swa": ("SWA at 2,500-update budget", "#9B59B6", "--"),
    "public_gc": ("Public GraphCast", "#777777", ":"),
}
FIELDS = [("2m_temperature", None, "2 m temperature", "K²"),
          ("geopotential", 500, "500 hPa geopotential", "m⁴ s⁻⁴"),
          ("u_component_of_wind", 850, "850 hPa zonal wind", "m² s⁻²")]


def pinned(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_csv(name, rows):
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    result = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan)
    np.divide(a, b, out=result, where=b > 0)
    return result


def save(fig, name):
    paths = []
    for extension in ("png", "pdf"):
        path = IMAGES / f"{name}.{extension}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        paths.append(pinned(path))
    plt.close(fig)
    return paths


def main():
    report_path, spectra_path = SOURCE / "report.json", SOURCE / "pooled_spectra.npz"
    report = json.loads(report_path.read_text())
    if report["status"] != "complete" or report["count"] != 4:
        raise ValueError("Expected the completed four-date comparison")
    if pinned(spectra_path)["sha256"] != report["pooled_spectra"]["sha256"]:
        raise ValueError("Saved spectra checksum mismatch")
    manifest_path = Path(report["sources"][0]["path"])
    if pinned(manifest_path)["sha256"] != report["sources"][0]["sha256"]:
        raise ValueError("Saved manifest checksum mismatch")
    with np.load(spectra_path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        channels = [tuple(c) for c in metadata["layout"]["channels"]]
        identities = saved["models"].tolist()
        degree = saved["degree"].copy()
        lead_hours = saved["lead_hours"].copy()
        truth = saved["truth_power"].copy()
        mask = saved["common_positive_truth_mask"].copy()
        powers = {key: saved["prediction_power"][identities.index(key)].copy() for key in STYLES}
        losses = {key: saved["exact_gc_loss"][identities.index(key)].copy() for key in STYLES}
    if metadata["initialization_times"] != report["initialization_times"]:
        raise ValueError("Spectral cohort does not match report")
    if truth.shape != (40, 83, 180) or not np.array_equal(degree, np.arange(180)):
        raise ValueError("Unexpected spherical spectrum layout")
    if not np.array_equal(lead_hours, np.arange(1, 41) * 6):
        raise ValueError("Unexpected forecast leads")
    if not all(np.isfinite(a).all() for a in [truth, *powers.values(), *losses.values()]):
        raise ValueError("Nonfinite saved values")
    scores = {row["model"]: row for row in report["summary"] if row["model"] in STYLES}
    errors = {}
    for identity in STYLES:
        error = 100 * np.abs(ratio(powers[identity][..., 1:], truth[..., 1:]) - 1)
        errors[identity] = np.array([error[i][mask[i]].mean() for i in range(40)])
        np.testing.assert_allclose(losses[identity].mean(), scores[identity]["exact_gc_mse_rollout"], rtol=1e-7)
        np.testing.assert_allclose(error[mask].mean(), scores[identity]["power_error_rollout_pct"], rtol=1e-10)
        np.testing.assert_allclose(errors[identity][-1], scores[identity]["power_error_day10_pct"], rtol=1e-10)
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    write_csv("m1_steps.csv", [scores[key] for key in [*RAW, "M1_swa", "public_gc"]])
    curves = [{"model": key, "lead_days": float(hours / 24),
               "exact_gc_mse": float(losses[key][i]),
               "exact_gc_loss_reduction_pct": float(100 * (1 - losses[key][i] / losses["public_gc"][i])),
               "power_mismatch_pct": float(errors[key][i])}
              for key in STYLES for i, hours in enumerate(lead_hours)]
    write_csv("m1_lead_curves.csv", curves)
    deltas = []
    for previous, current in zip(RAW[:-1], RAW[1:]):
        old, new = scores[previous], scores[current]
        decrease = old["exact_gc_mse_rollout"] - new["exact_gc_mse_rollout"]
        deltas.append({"from_total_updates": old["training_updates"], "to_total_updates": new["training_updates"],
                       "mse_decrease": decrease, "mse_reduction_pct": 100 * decrease / old["exact_gc_mse_rollout"],
                       "power_mismatch_change_pp": new["power_error_rollout_pct"] - old["power_error_rollout_pct"],
                       "public_gc_repeat_spread": report["baseline_repeat_spread_rollout"],
                       "decrease_exceeds_public_gc_repeat_spread": decrease > report["baseline_repeat_spread_rollout"],
                       "absolute_change_exceeds_public_gc_repeat_spread": abs(decrease) > report["baseline_repeat_spread_rollout"],
                       "reference_scope": "baseline-only replay difference; not full model noise or an acceptance gate"})
    write_csv("m1_step_deltas.csv", deltas)

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "semibold", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.9))
    updates = [scores[key]["training_updates"] for key in RAW]
    for ax, metric, label in [(axes[0, 0], "exact_gc_mse_rollout", "Exact GC rollout MSE ↓"),
                               (axes[0, 1], "power_error_rollout_pct", "Mean spherical power mismatch (%) ↓")]:
        values = [scores[key][metric] for key in RAW]
        ax.plot(updates, values, color="#A3A3A3", lw=1.6, zorder=1)
        for key, x, y in zip(RAW, updates, values):
            ax.scatter(x, y, s=65, color=STYLES[key][1], zorder=3)
            ax.annotate(f"{y:.3f}" if "mse" in metric else f"{y:.2f}", (x, y),
                        xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10)
        swa = scores["M1_swa"][metric]
        ax.scatter(2500, swa, marker="D", s=75, color=STYLES["M1_swa"][1], zorder=3)
        ax.annotate(f"SWA {swa:.3f}" if "mse" in metric else f"SWA {swa:.2f}",
                    (2500, swa), xytext=(-8, -18), textcoords="offset points", ha="right", fontsize=10)
        ax.set(xlabel="Total optimizer updates (includes initial 500)", ylabel=label,
               xticks=[1000, 1250, 1500, 2000, 2500], xlim=(850, 2700))
        ax.margins(y=.32)
        ax.grid(alpha=.18)
    axes[0, 0].set_title("A   Accuracy across the raw training trajectory", loc="left")
    axes[0, 1].set_title("B   Spectrum fidelity across the same trajectory", loc="left")
    for key, (label, color, linestyle) in STYLES.items():
        if key != "public_gc":
            axes[1, 0].plot(lead_hours / 24, 100 * (1 - losses[key] / losses["public_gc"]),
                            label=label, color=color, ls=linestyle, lw=1.8)
        axes[1, 1].plot(lead_hours / 24, errors[key], label=label, color=color, ls=linestyle, lw=1.8)
    axes[1, 0].axhline(0, color="#777777", lw=1, ls=":")
    axes[1, 0].set(title="C   Forecast accuracy by lead", ylabel="Exact GC loss reduction vs public GC (%) ↑")
    axes[1, 1].set(title="D   Spectrum fidelity by lead", ylabel="Spherical power mismatch (%) ↓")
    for ax in axes[1]:
        ax.set(xlabel="Forecast lead (days); distinct from training K", xlim=(.25, 10))
        ax.grid(alpha=.18)
    fig.suptitle("M1: training duration changes accuracy and spectral fidelity", y=.995, fontsize=17, weight="semibold")
    fig.text(.5, .949, "Training K = 20 · 4 matched 2022 development dates · 40 six-hour forecast leads · one seed",
             ha="center", fontsize=11)
    fig.legend(*axes[1, 1].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .927),
               ncols=3, frameon=False, fontsize=10)
    fig.text(.5, .035, "SWA averages total updates 1,000 / 1,500 / 2,000 / 2,500; the diamond marks its training budget.\n"
             "Power mismatch: date-pooled powers, equal mean over 83 channels and degrees 1–179; panels A/B also average 40 leads.\n"
             "Point estimates, no confidence bands. Public-GC replay spread: 0.0794 rollout-MSE units; not full model uncertainty.",
             ha="center", va="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .095, 1, .85), h_pad=2.2, w_pad=2.8)
    figures = save(fig, "m1_training_steps")

    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.7))
    rows = []
    day10 = int(np.flatnonzero(lead_hours == 240)[0])
    for column, (variable, level, label, units) in enumerate(FIELDS):
        channel = channels.index((variable, level))
        reference = truth[day10, channel, 1:]
        axes[0, column].plot(degree[1:], reference, color="black", lw=2, label="ERA5 truth")
        axes[1, column].axhline(1, color="black", lw=1.3)
        for key, (legend, color, style) in STYLES.items():
            power = powers[key][day10, channel, 1:]
            power_ratio = ratio(power, reference)
            axes[0, column].plot(degree[1:], power, color=color, ls=style, lw=1.6, label=legend)
            axes[1, column].plot(degree[1:], power_ratio, color=color, ls=style, lw=1.6)
            rows.extend({"model": key, "variable": variable, "level_hpa": level, "lead_hours": 240,
                         "spherical_degree": int(ell), "forecast_power": float(power[i]),
                         "truth_power": float(reference[i]), "power_units": units,
                         "forecast_truth_power_ratio": float(power_ratio[i])} for i, ell in enumerate(degree[1:]))
        axes[0, column].set(title=label, ylabel=f"Physical power ({units})")
        axes[1, column].set(ylabel="Forecast / ERA5 power", xlabel="Spherical harmonic degree ℓ")
        for ax in axes[:, column]:
            ax.set(xscale="log", yscale="log", xlim=(1, 179))
            ax.grid(alpha=.18, which="major")
    fig.suptitle("M1 day-10 power spectra at three raw training checkpoints", y=.995, fontsize=17, weight="semibold")
    fig.text(.5, .948, "Fixed training K = 20 · same four 2022 dates · powers pooled before forming ratios", ha="center", fontsize=11)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .928), ncols=3, frameon=False, fontsize=10)
    fig.text(.5, .035, "Positive degrees 1–179; physical fields, no climatology subtraction. Both axes use logarithmic scales.\n"
             "Power ratios describe amplitude fidelity, not forecast-error power or spatial/phase accuracy. SWA is a separate averaged checkpoint.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .075, 1, .85), h_pad=2, w_pad=2)
    figures.extend(save(fig, "m1_day10_power_spectra"))
    write_csv("m1_day10_power_spectra.csv", rows)
    provenance = {
        "date": "2026-11-13", "scope": "Replot existing M1 evaluations; no new training or inference",
        "sources": [pinned(report_path), pinned(spectra_path), pinned(manifest_path)],
        "script": pinned(Path(__file__).resolve()), "training_k": 20, "seed": 22,
        "initialization_times": report["initialization_times"], "raw_total_updates": updates,
        "swa_source_total_updates": [1000, 1500, 2000, 2500],
        "power_error_definition": report["power_error_definition"],
        "baseline_repeat_spread_rollout": report["baseline_repeat_spread_rollout"],
        "baseline_repeat_spread_day10": report["baseline_repeat_spread_day10"],
        "noise_scope": report["noise_scope"],
        "numerical_reference_differences": [x for x in report["numerical_reference_differences"] if x["model"] in STYLES],
        "warnings": [x for x in report["warnings"] if x["model"] in STYLES],
        "validation": "Saved manifest/spectra hashes and cohort verified; every plotted rollout MSE and power score reproduced from NPZ arrays",
        "limitations": "Four development dates, one seed, sparse raw checkpoints. No seed or date uncertainty bands. Cannot identify an optimum or training-K effect.",
        "figures": figures,
    }
    (DATA / "m1_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "complete", "images": [f["path"] for f in figures], "raw_step_changes": deltas}, indent=2))


if __name__ == "__main__":
    main()
