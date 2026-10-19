#!/usr/bin/env python3
"""Report saved RMSEs for the paper-selected v24 checkpoint; no inference.

Run from the repository root after sourcing scripts/graphcast_env.sh.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/v24_best_variable_rmse_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = "v24_best_variable_rmse_20260925"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
REPORT = ROOT / "docs/reports/v24_best_variable_rmse_2026-09-25.md"
SPECTRAL = ROOT / "plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917"
SOURCES = {
    "development_2022_n32": ROOT / "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908/runs/di16_fast_mamba/eval/cold_full_zero_exact_gc/matched32/swa_step00500-02000.json",
    "pilot_2023_n8": SPECTRAL / "models/M1/evaluation.json",
}
LABELS = {"development_2022_n32": "2022 development · 32 starts", "pilot_2023_n8": "2023 pilot · 8 starts"}
VARIABLES = {
    "2m_temperature": ("2 m temperature", "K", 1),
    "mean_sea_level_pressure": ("Mean sea-level pressure", "hPa", .01),
    "10m_u_component_of_wind": ("10 m zonal wind (u)", "m/s", 1),
    "10m_v_component_of_wind": ("10 m meridional wind (v)", "m/s", 1),
    "total_precipitation_6hr": ("6 h precipitation", "mm", 1000),
    "geopotential": ("Geopotential", "m²/s²", 1),
    "temperature": ("Air temperature", "K", 1),
    "specific_humidity": ("Specific humidity", "g/kg", 1000),
    "u_component_of_wind": ("Zonal wind (u)", "m/s", 1),
    "v_component_of_wind": ("Meridional wind (v)", "m/s", 1),
    "vertical_velocity": ("Pressure vertical velocity", "Pa/s", 1),
}
SURFACE = set(list(VARIABLES)[:5])
DAYS = np.arange(1, 41) / 4


def relative(path):
    return str(path.relative_to(ROOT))


def link(path, label):
    return f"[{label}]({os.path.relpath(path, REPORT.parent)})"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (DATA / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate(document, count):
    assert document["evaluation_status"] == "complete"
    assert document["metrics_complete"] and document["evaluated_samples"] == count
    assert document["remaining_samples"] == 0 and document["target_steps"] == 40
    assert document["resolution"] == 1 and document["eval_mode"] == "cold_full"
    assert document["residual_state_init"] == "zero" and not document["rs_reset_every_step"]
    assert len(document["per_channel_per_step"]) == 83
    assert set(document["per_variable_per_step"]) == set(VARIABLES)
    for category in ("per_variable_per_step", "per_channel_per_step"):
        for values in document[category].values():
            baseline = np.array(values["rmse_baseline"])
            model = np.array(values["rmse_full"])
            assert baseline.shape == model.shape == (40,)
            assert np.isfinite(baseline).all() and np.isfinite(model).all()
            assert (baseline > 0).all() and (model >= 0).all()
            np.testing.assert_allclose(100 * (1 - model / baseline), values["improvement_pct_rmse"], rtol=0, atol=1e-10)
    # Variable-group MSE pools the 13 levels equally before the square root.
    for variable in set(VARIABLES) - SURFACE:
        levels = [v for k, v in document["per_channel_per_step"].items() if k.startswith(variable + "_level")]
        assert len(levels) == 13
        for branch in ("baseline", "full"):
            pooled = np.sqrt(np.mean([np.array(v[f"rmse_{branch}"])**2 for v in levels], axis=0))
            np.testing.assert_allclose(pooled, document["per_variable_per_step"][variable][f"rmse_{branch}"], rtol=2e-6)
    loss = document["original_graphcast_loss"]
    np.testing.assert_allclose(100 * (1 - np.mean(loss["full_per_step"]) / np.mean(loss["baseline_per_step"])), loss["improvement_pct_rollout"], atol=1e-10)


def records(dataset, document, category):
    rows = []
    for name, metrics in document[category].items():
        if name in VARIABLES:
            variable, level = name, ""
        else:
            variable, _, level = name.rpartition("_level")
        label, units, scale = VARIABLES[variable]
        for step in range(40):
            baseline = metrics["rmse_baseline"][step] * scale
            model = metrics["rmse_full"][step] * scale
            rows.append(dict(dataset=dataset, evaluated_starts=document["evaluated_samples"],
                             variable=variable, pressure_level_hpa=level,
                             aggregation="equal_level_mse" if category == "per_variable_per_step" and variable not in SURFACE else "single_channel",
                             lead_hours=(step + 1) * 6, lead_days=(step + 1) / 4,
                             units=units, baseline_rmse=baseline, model_rmse=model,
                             absolute_rmse_reduction=baseline - model,
                             rmse_reduction_pct=100 * (1 - model / baseline),
                             source=relative(SOURCES[dataset])))
    return rows


def save(fig, name):
    fig.savefig(IMAGES / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(IMAGES / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def figures(documents):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True, sharey=True)
    for ax, (variable, (label, _, _)) in zip(axes.flat, VARIABLES.items()):
        for (dataset, document), color in zip(documents.items(), ["#0072B2", "#D55E00"]):
            ax.plot(DAYS, document["per_variable_per_step"][variable]["improvement_pct_rmse"], label=LABELS[dataset], color=color)
        ax.axhline(0, color=".55", lw=.7)
        ax.set_title(label + (" · all levels" if variable not in SURFACE else ""), fontsize=9)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.grid(alpha=.2)
        ax.tick_params(labelbottom=True)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False)
    fig.suptitle("RMSE reduction relative to paired GraphCast · di16 fast-Mamba SWA 500–2000")
    fig.supxlabel("Forecast lead (days)")
    fig.supylabel("RMSE reduction (%) · positive means lower error")
    fig.tight_layout(rect=(.02, .02, 1, .96))
    save(fig, "rmse_reduction_vs_lead")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, layout="constrained")
    for ax, (dataset, document) in zip(axes, documents.items()):
        values = np.array([document["per_variable_per_step"][v]["improvement_pct_rmse"][3::4] for v in VARIABLES])
        im = ax.imshow(values, cmap="RdBu", vmin=-18, vmax=18, aspect="auto")
        ax.set_title(LABELS[dataset])
        ax.set_xticks(range(10), range(1, 11))
        ax.set_yticks(range(11), [v[0] for v in VARIABLES.values()])
        ax.set_xlabel("Forecast lead (days)")
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="white" if abs(val) > 10 else "black")
    fig.colorbar(im, ax=axes, label="RMSE reduction (%)", shrink=.8)
    fig.suptitle("Daily RMSE reductions · upper-air groups pool all 13 levels equally")
    save(fig, "daily_rmse_reduction")
    fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True)
    for ax, (variable, (label, units, scale)) in zip(axes.flat, VARIABLES.items()):
        for (dataset, document), color in zip(documents.items(), ["#0072B2", "#D55E00"]):
            metrics = document["per_variable_per_step"][variable]
            for branch, style in [("baseline", "--"), ("full", "-")]:
                ax.plot(DAYS, np.array(metrics[f"rmse_{branch}"]) * scale, color=color, ls=style,
                        label=LABELS[dataset] + (" · GraphCast" if branch == "baseline" else " · corrected"))
        ax.set_title(label, fontsize=9)
        ax.set_ylabel(f"RMSE ({units})")
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 10)
        ax.grid(alpha=.2)
        ax.tick_params(labelbottom=True)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=8)
    fig.suptitle("Absolute RMSE · each dataset uses its own paired baseline")
    fig.supxlabel("Forecast lead (days)")
    fig.tight_layout(rect=(0, .02, 1, .96))
    save(fig, "absolute_rmse_vs_lead")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    documents = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
    for document, count in zip(documents.values(), [32, 8]):
        validate(document, count)
    dev, pilot = documents.values()
    assert Path(dev["ckpt"]).resolve() == Path(pilot["ckpt"]).resolve()
    assert (ROOT / dev["ckpt_in"]).resolve() == (ROOT / pilot["ckpt_in"]).resolve()
    protocol = json.loads((SPECTRAL / "protocol.json").read_text())
    assert pilot["protocol_hash"] == protocol["protocol_hash"]
    assert pilot["checkpoint_sha256"] == protocol["models"][0]["checkpoint_sha256"]
    assert pilot["global_initialization_times"] == protocol["initialization_times"]
    assert sha(Path(dev["ckpt"])) == pilot["checkpoint_sha256"]
    variable_rows = [row for key, doc in documents.items() for row in records(key, doc, "per_variable_per_step")]
    channel_rows = [row for key, doc in documents.items() for row in records(key, doc, "per_channel_per_step")]
    write_csv("per_variable_all_leads.csv", variable_rows)
    write_csv("per_channel_all_leads.csv", channel_rows)
    write_csv("per_variable_integer_days.csv", [r for r in variable_rows if r["lead_hours"] % 24 == 0])
    counts = [dict(dataset=key, lead_days=(i+1)/4,
                   variables_improved=sum(v["improvement_pct_rmse"][i] > 0 for v in doc["per_variable_per_step"].values()),
                   channels_improved=sum(v["improvement_pct_rmse"][i] > 0 for v in doc["per_channel_per_step"].values()))
              for key, doc in documents.items() for i in range(40)]
    write_csv("improved_counts.csv", counts)
    provenance = dict(generator=relative(Path(__file__)), generator_sha256=sha(Path(__file__)),
                      checkpoint=dev["ckpt"], checkpoint_sha256=pilot["checkpoint_sha256"],
                      rmse_definition="sqrt(mean over starts and equally over pressure levels of global cosine-latitude-weighted squared error))",
                      reduction_definition="100 * (1 - model_rmse / paired_baseline_rmse)",
                      datasets={key: dict(source=relative(SOURCES[key]), sha256=sha(SOURCES[key]),
                           n_samples=doc["evaluated_samples"], chosen_idx=doc["chosen_idx"],
                           initialization_times=doc.get("global_initialization_times"),
                           original_graphcast_loss=doc["original_graphcast_loss"]) for key, doc in documents.items()})
    (DATA / "sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    figures(documents)
    lines = ["# Best-model RMSE by forecast lead — 2026-09-25", "",
             "The requested RMSEs are already calculated. No new evaluation is needed for the saved 32-start 2022 development set or the eight-start 2023 pilot.", "",
             "Model: 1° v24 `di16_fast_mamba`, Mamba1/BCG1, residual width512/MP2, threefold Mamba learning rate, SWA updates 500/1000/1500/2000. This is the selected model in the current paper figures and the September 24 recipe comparison. Best means highest observed aggregate exact GraphCast loss reduction in the compared completed recipes, not proven superiority or best at every variable/lead. Its 16.3434% aggregate loss reduction is close to di32 joint SWA (16.2940%).", "",
             "Both evaluations use 40 six-hour leads, native 1° verification, cold full-feedback rollouts, zero initial recurrent memory, state carry thereafter, and their own paired frozen GraphCast-small baseline. The 2022 starts were used for model selection. The 2023 pilot has eight preselected dates: January 15, February 15, April 15, May 15, July 15, August 15, October 15 and November 15, all at 00 UTC. These datasets are reported separately; neither has variable-level confidence intervals here.", "",
             "RMSE reduction (%) = `100 × (1 − model RMSE / baseline RMSE)`. Positive values mean lower error; negative values mean degradation. RMSE pools squared errors across starts before taking the square root and uses normalized cosine-latitude weights. Upper-air group results additionally pool squared errors equally across all 13 pressure levels; they are not pressure-weighted or a single standard-level score. Exact GraphCast loss uses a different weighting/normalization and is reported separately.", ""]
    for key, doc in documents.items():
        lines += [f"**{LABELS[key]}: RMSE reduction (%)**", "", "| Variable | Day 1 | Day 3 | Day 5 | Day 7 | Day 10 |", "|---|---:|---:|---:|---:|---:|"]
        for variable, (label, _, _) in VARIABLES.items():
            vals = doc["per_variable_per_step"][variable]["improvement_pct_rmse"]
            lines.append("| " + label + " | " + " | ".join(f"{vals[d*4-1]:.2f}" for d in [1,3,5,7,10]) + " |")
        loss = doc["original_graphcast_loss"]
        exceptions = [(k, v["improvement_pct_rmse"][-1]) for k,v in doc["per_channel_per_step"].items() if v["improvement_pct_rmse"][-1] <= 0]
        lines += ["", f"All 11 variable groups improve at day 10; {83-len(exceptions)}/83 individual channels improve. Day-10 exceptions: " + ", ".join(f"`{k}` ({v:.2f}% RMSE reduction)" for k,v in exceptions) + ".",
                  f"Exact original GraphCast loss reduction, a separate metric: {loss['improvement_pct_rollout']:.4f}% over all leads and {loss['improvement_pct_per_step'][-1]:.4f}% at day 10.", ""]
    lines += ["**Absolute day-10 RMSE**", "", "| Variable | Unit | 2022 baseline | 2022 model | 2023 baseline | 2023 model |", "|---|---|---:|---:|---:|---:|"]
    for variable, (label, units, scale) in VARIABLES.items():
        vals = [doc["per_variable_per_step"][variable][f"rmse_{branch}"][-1]*scale for doc in documents.values() for branch in ["baseline", "full"]]
        lines.append(f"| {label} | {units} | " + " | ".join(f"{v:.4g}" for v in vals) + " |")
    lines += ["", "**Individual standard-level channels: RMSE reduction (%)**", "", "| Channel | 2022 day 5 | 2022 day 10 | 2023 day 5 | 2023 day 10 |", "|---|---:|---:|---:|---:|"]
    for channel in ["geopotential_level500", "temperature_level850", "specific_humidity_level700", "u_component_of_wind_level850", "v_component_of_wind_level850", "vertical_velocity_level500"]:
        vals = [doc["per_channel_per_step"][channel]["improvement_pct_rmse"][i] for doc in documents.values() for i in [19,39]]
        lines.append(f"| {channel} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines += ["", "**Evaluation still needed for full-year confirmation**", "",
              "The separate 365-start 2023 campaign is incomplete. Slurm accounting checked on September 25 shows preflight tasks `14077246_0–2` timed out (1:02 limits; elapsed about 1:07). Gate `14077255`, production array `14077262_[0–68]`, and report `14077269` were cancelled. Each saved preflight partial JSON has zero completed starts. There is no completed full-year result in that campaign directory and these jobs are no longer queued. Recover the preflight, validate runtime/numerical settings, and rerun the dependent campaign to obtain full-year evidence. This report did not submit or modify jobs.", "",
              "**Saved outputs**", "",
              "- " + link(IMAGES / "rmse_reduction_vs_lead.pdf", "All-variable RMSE-reduction curves (PDF)"),
              "- " + link(IMAGES / "daily_rmse_reduction.pdf", "Daily RMSE-reduction heatmap (PDF)"),
              "- " + link(IMAGES / "absolute_rmse_vs_lead.pdf", "Absolute RMSE curves (PDF)"),
              "- " + link(DATA / "per_variable_all_leads.csv", "11 variables × 40 leads × 2 datasets (CSV)"),
              "- " + link(DATA / "per_channel_all_leads.csv", "83 channels × 40 leads × 2 datasets (CSV)"),
              "- " + link(DATA / "per_variable_integer_days.csv", "Integer-day table (CSV)"),
              "- " + link(DATA / "improved_counts.csv", "Improved variable/channel counts (CSV)"),
              "- " + link(DATA / "sources.json", "Source hashes, checkpoint identity, protocol and exact losses"), "",
              "Validation: checked complete 32/8-start outputs, matching checkpoint and baseline identities, checkpoint SHA-256 against the frozen 2023 protocol, all 7,520 variable/channel lead reductions against paired RMSEs, and group RMSEs against equally pooled pressure-level MSEs. Source evaluation files are unchanged.", "",
              "Reproduce: `source scripts/graphcast_env.sh && python scripts/analyze_models/report_v24_best_variable_rmse.py`.", "",
              "Sources: " + "; ".join(link(p, key) for key, p in SOURCES.items()) + "; " + link(ROOT / "docs/experiments/v24_res1_ar_sweep_2026-09-24.md", "September 24 recipe selection") + ".", ""]
    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}; {len(variable_rows)} variable rows; {len(channel_rows)} channel rows; three PNG/PDF figure pairs.")


if __name__ == "__main__":
    main()
