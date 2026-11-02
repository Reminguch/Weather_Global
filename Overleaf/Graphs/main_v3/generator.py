#!/usr/bin/env python3
"""Rebuild main_v3 variable and memory figures from saved RMSE evaluations.

Run from the repository root:
    source scripts/graphcast_env.sh
    python Overleaf/Graphs/main_v3/generator.py

This script does not evaluate any checkpoint or reconstruct GraphCast loss.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/main_v3_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global_experiments_data/results/2026-08-10-ablation-swa-select32")
VARIABLES = (
    ("geopotential", "Geopotential"),
    ("temperature", "Temperature"),
    ("specific_humidity", "Specific humidity"),
    ("u_component_of_wind", "Zonal wind"),
    ("v_component_of_wind", "Meridional wind"),
    ("vertical_velocity", "Vertical velocity"),
    ("10m_u_component_of_wind", "10 m zonal wind"),
    ("10m_v_component_of_wind", "10 m meridional wind"),
    ("2m_temperature", "2 m temperature"),
    ("mean_sea_level_pressure", "Mean sea-level pressure"),
    ("total_precipitation_6hr", "6 h precipitation"),
)
ARMS = (
    ("M0", "Memory off, K=0", "#D55E00", "--"),
    ("S0", "Memory on, K=0", "#CC79A7", ":"),
    ("M20", "Memory off, K=20", "#009E73", "-."),
    ("S20", "Memory on, K=20", "#0072B2", "-"),
)
DAYS = np.arange(1, 41) / 4
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def reduction(metrics):
    baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
    model = np.asarray(metrics["rmse_full"], dtype=float)
    assert baseline.shape == model.shape == (40,)
    assert np.isfinite(baseline).all() and (baseline > 0).all()
    assert np.isfinite(model).all() and (model >= 0).all()
    values = 100 * (1 - model / baseline)
    if "improvement_pct_rmse" in metrics:
        np.testing.assert_allclose(values, metrics["improvement_pct_rmse"], atol=1e-10)
    return values


def validate(doc, indices):
    assert doc["n_samples"] == 32 and doc["target_steps"] == 40
    assert doc["chosen_idx"] == indices
    assert doc["warmup_steps"] == 0 and doc["feedback_mode"] == "full"
    assert len(doc["per_channel_per_step"]) == 83
    assert len(doc["per_variable_per_step"]) == 11
    for metrics in doc["per_channel_per_step"].values():
        reduction(metrics)
    for key, metrics in doc["per_variable_per_step"].items():
        reduction(metrics)
        channels = [c for k, c in doc["per_channel_per_step"].items()
                    if k == key or k.startswith(key + "_level")]
        assert len(channels) in (1, 13)
        for field in ("rmse_baseline", "rmse_full"):
            pooled = np.sqrt(np.mean([np.square(c[field]) for c in channels], axis=0))
            # Aggregated fields have float32 reduction noise; agreement confirms
            # equal-level MSE pooling rather than a mean of channel percentages.
            np.testing.assert_allclose(pooled, metrics[field], rtol=2e-7, atol=0)


def save(fig, stem, description):
    fig.savefig(HERE / f"{stem}.pdf", metadata={
        "Title": stem.replace("_", " "), "Subject": description,
        "Creator": "main_v3/generator.py; saved evaluation artifacts only",
        "CreationDate": None, "ModDate": None,
    })
    fig.savefig(HERE / f"{stem}.png", dpi=190)
    plt.close(fig)


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lead_axes(ax):
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Forecast lead (days)")
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)


def breadth(doc):
    matrix = np.array([reduction(doc["per_variable_per_step"][k]) for k, _ in VARIABLES])
    channel_values = np.array([reduction(m) for m in doc["per_channel_per_step"].values()])
    counts = (channel_values > 0).sum(axis=0)
    assert (matrix[:, 19:] > 0).all(), "All groups must improve at 120--240 h"
    assert counts[30] == 83 and (counts[30:] == 83).all()
    assert (counts[:30] < 83).all(), "186 h must be the first all-channel positive lead"
    fig = plt.figure(figsize=(6.5, 5.7))
    ax = fig.add_axes([.295, .375, .56, .545])
    heatmap = matrix[:, [19, 23, 27, 31, 35, 39]]
    im = ax.imshow(heatmap, cmap="YlGnBu", vmin=0, vmax=20, aspect="auto")
    ax.set_yticks(range(11), [label for _, label in VARIABLES])
    ax.set_xticks(range(6), ["5", "6", "7", "8", "9", "10"])
    ax.set_xlabel("Forecast lead (days)", labelpad=4)
    ax.set_title("(a) RMSE reduction by variable group", loc="left", pad=9)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), value in np.ndenumerate(heatmap):
        ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8.5,
                color="white" if value >= 12 else "#202020")
    cax = fig.add_axes([.875, .375, .016, .545])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 5, 10, 15, 20])
    cb.set_label("RMSE reduction (%)", labelpad=4, fontsize=8.5)
    cb.outline.set_visible(False)
    bottom = fig.add_axes([.12, .085, .785, .16])
    bottom.plot(DAYS, counts, color="#0072B2", linewidth=1.65)
    bottom.axhline(83, color=".5", linewidth=.8, linestyle="--")
    bottom.scatter([7.75], [83], s=16, color="#0072B2", zorder=3)
    bottom.annotate("83/83 from day 7.75", xy=(7.75, 83), xytext=(5.4, 42),
                    fontsize=8, arrowprops={"arrowstyle": "-", "lw": .8, "color": ".4"})
    bottom.set_ylim(0, 90)
    bottom.set_yticks([0, 40, 83])
    bottom.set_ylabel("Improved\nchannels")
    bottom.set_title("(b) Coverage across all 83 channels", loc="left", pad=7)
    lead_axes(bottom)
    save(fig, "breadth_15yr", "15-year S128 K20, SWA6k--11k; 32 select32 starts. "
         "Paired RMSE reduction: equal-pressure-level MSE pooling for groups; "
         "each channel is evaluated separately for the positive-reduction count.")
    write_csv("breadth_15yr_all_leads.csv", [
        {"variable": key, "lead_hours": (i + 1) * 6, "lead_days": day,
         "rmse_baseline": doc["per_variable_per_step"][key]["rmse_baseline"][i],
         "rmse_model": doc["per_variable_per_step"][key]["rmse_full"][i],
         "rmse_reduction_pct": matrix[v, i]}
        for v, (key, _) in enumerate(VARIABLES) for i, day in enumerate(DAYS)
    ])
    write_csv("breadth_15yr_channels.csv", [
        {"channel": key, "lead_hours": (i + 1) * 6, "lead_days": day,
         "rmse_baseline": metrics["rmse_baseline"][i],
         "rmse_model": metrics["rmse_full"][i], "rmse_reduction_pct": value}
        for key, metrics in doc["per_channel_per_step"].items()
        for i, (day, value) in enumerate(zip(DAYS, reduction(metrics)))
    ])
    write_csv("breadth_15yr_channel_counts.csv", [
        {"lead_hours": (i + 1) * 6, "lead_days": day,
         "improved_channels": int(count), "total_channels": 83}
        for i, (day, count) in enumerate(zip(DAYS, counts))
    ])
    return {"all_groups_improve_from_120h": True, "all_channels_improve_from_hours": 186,
            "channel_counts": counts.tolist(),
            "day10_group_reduction_min_pct": float(matrix[:, -1].min()),
            "day10_group_reduction_max_pct": float(matrix[:, -1].max())}


def memory(docs):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    fig.subplots_adjust(left=.09, right=.985, top=.85, bottom=.30, wspace=.28)
    main_keys = (("2m_temperature", "(a) 2 m temperature"),
                 ("geopotential_level500", "(b) 500 hPa geopotential"))
    for ax, (key, title) in zip(axes, main_keys):
        for arm, label, color, style in ARMS:
            ax.plot(DAYS, reduction(docs[arm]["per_channel_per_step"][key]),
                    color=color, linestyle=style, label=label, linewidth=1.65)
        lead_axes(ax)
        ax.axhline(0, color=".5", linewidth=.8)
        ax.set_ylim(-9, 15)
        ax.set_yticks([-5, 0, 5, 10, 15])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("RMSE reduction (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(.54, .005), handlelength=2.6,
               columnspacing=2.5)
    save(fig, "memory_ar_15yr", "15-year memory x autoregressive-training ablation; "
         "lr1e-4, SWA6k--11k, same32 starts. Each arm uses its own paired baseline. "
         "Memory off resets both SSM state and convolution cache during training and evaluation.")
    fig, axes = plt.subplots(4, 3, figsize=(6.5, 7.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.13, top=.955, hspace=.39, wspace=.18)
    for i, (key, title) in enumerate(VARIABLES):
        ax = axes.flat[i]
        for arm, label, color, style in ARMS:
            ax.plot(DAYS, reduction(docs[arm]["per_variable_per_step"][key]),
                    color=color, linestyle=style, label=label, linewidth=1.3)
        lead_axes(ax)
        ax.axhline(0, color=".5", linewidth=.65)
        ax.set_ylim(-11, 16)
        ax.set_yticks([-10, 0, 10])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("")
    axes.flat[-1].set_visible(False)
    for ax in axes[-1, :2]:
        ax.set_xlabel("Lead (days)")
    axes[-2, -1].tick_params(labelbottom=True)
    axes[-2, -1].set_xlabel("Lead (days)")
    fig.supylabel("RMSE reduction (%)", x=.008, fontsize=9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(.53, .015), handlelength=2.6,
               columnspacing=2.5)
    save(fig, "memory_ar_15yr_all_variables", "All11 variable groups for the same four-arm "
         "15-year memory x autoregressive-training ablation; equal-level MSE pooling, "
         "paired RMSE reductions, not GraphCast-loss reduction.")
    rows = []
    for arm, _, _, _ in ARMS:
        for scope, metrics_dict in (("variable_group", docs[arm]["per_variable_per_step"]),
                                    ("channel", docs[arm]["per_channel_per_step"])):
            for key, metrics in metrics_dict.items():
                for i, value in enumerate(reduction(metrics)):
                    rows.append({"arm": arm, "scope": scope, "variable": key,
                                 "lead_hours": (i + 1) * 6, "lead_days": DAYS[i],
                                 "rmse_baseline": metrics["rmse_baseline"][i],
                                 "rmse_model": metrics["rmse_full"][i],
                                 "rmse_reduction_pct": value})
    write_csv("memory_ar_15yr_all_leads.csv", rows)
    day10 = []
    for key, label in VARIABLES + (("geopotential_level500", "500 hPa geopotential"),):
        scope = "per_channel_per_step" if "_level" in key else "per_variable_per_step"
        values = {arm: float(reduction(docs[arm][scope][key])[-1]) for arm, *_ in ARMS}
        day10.append({"variable": key, "label": label, **values,
                      "memory_effect_K0_pp": values["S0"] - values["M0"],
                      "memory_effect_K20_pp": values["S20"] - values["M20"]})
    write_csv("memory_ar_15yr_day10.csv", day10)
    base = docs["M0"]["per_channel_per_step"]
    baseline_diff = {
        arm: max(float(np.max(np.abs(np.asarray(metrics["rmse_baseline"]) /
                                    base[key]["rmse_baseline"] - 1) * 100))
                 for key, metrics in doc["per_channel_per_step"].items())
        for arm, doc in docs.items()
    }
    return {"day10_results": day10,
            "max_relative_baseline_rmse_deviation_from_M0_pct": baseline_diff,
            "day10_K20_memory_benefits_all_11_groups": all(r["memory_effect_K20_pp"] > 0 for r in day10[:11])}


def main():
    paths = {"S128": SOURCE / "sel32_S128_SWA6k-11k.json"}
    paths.update({arm: SOURCE / f"sel32_{arm}_lr1e-4_SWA6k-11k.json" for arm, *_ in ARMS})
    docs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    indices = docs["S128"]["chosen_idx"]
    for name, doc in docs.items():
        validate(doc, indices)
        assert doc["ablation_mode"] == ("reset_all_every_step" if name in ("M0", "M20") else "none")
    findings = {"breadth": breadth(docs["S128"]),
                "memory": memory({arm: docs[arm] for arm, *_ in ARMS})}
    provenance = {
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formula": "100*(1-rmse_full/rmse_baseline)",
        "metric": "cosine-latitude-weighted RMSE reduction; not GraphCast loss",
        "group_aggregation": "mean of physical-unit squared errors over starts and equally over pressure levels, then square root",
        "n_samples": 32, "chosen_idx": indices, "target_steps": 40,
        "sources": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "ckpt": docs[name]["ckpt"], "ablation_mode": docs[name]["ablation_mode"],
                           "anchor_manifest": docs[name]["anchor_manifest"]}
                    for name, path in paths.items()},
        "findings": findings,
        "baseline_caveat": "Evaluation runs have slightly different baseline RMSEs despite matching start indices. Each curve retains its own paired baseline. The cause is not established by these artifacts.",
        "uncertainty": "No confidence intervals or significance claims. One selected training run per ablation arm; starts are clustered and used in model selection.",
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
