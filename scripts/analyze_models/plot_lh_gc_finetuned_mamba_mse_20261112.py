#!/usr/bin/env python3
"""Plot saved LH 2023 exact GraphCast MSE; no forecasts or training are run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weather-global-lh-mse-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/LH_rebuttal_results/weatherdata20261111"
NAME = "lh_gc_finetuned_mamba_mse_20261112"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
LEADS = np.arange(6, 241, 6)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    sources = [Path(__file__).resolve()]

    def read(relative):
        path = SOURCE / relative
        sources.append(path)
        return json.loads(path.read_text())

    protocol = read("setup/evaluation_protocol.json")
    models = {}
    baselines = {}
    dates = None
    for rate in ("3em7", "1em5", "3em5"):
        result = read(f"01_graphcast_finetuning/evaluations/lr{rate}_step2500_2023.json")
        rows = result["original_graphcast_loss_per_initialization"]
        current_dates = pd.to_datetime([row["initialization_time"] for row in rows], utc=True)
        if dates is None:
            dates = current_dates
        require(current_dates.equals(dates) and len(dates) == 122 and dates.is_unique,
                "GCFT initialization cohort mismatch")
        require(all(row["lead_hours"] == LEADS.tolist() for row in rows), "GCFT lead mismatch")
        require(result["identity"]["evaluation_data_metadata"]["sha256"] ==
                protocol["verification"]["zarr_metadata"]["sha256"], "Verification identity mismatch")
        require(result["identity"]["completed_training_updates"] == 2500, "Checkpoint update mismatch")
        models[f"gcft_lr{rate}"] = np.array([row["full_per_step"] for row in rows])[None, ...]
        baselines[f"gcft_lr{rate}"] = np.array([row["baseline_per_step"] for row in rows])

    scores_path = SOURCE / "03_seed_replication/per_start_scores.csv.gz"
    sources.append(scores_path)
    scores = pd.read_csv(scores_path)
    scores["initialization_time"] = pd.to_datetime(scores["initialization_time"], utc=True)
    require(not scores.duplicated(["model", "initialization_time", "lead_hours"]).any(),
            "Duplicate per-start score cells")
    for mode, suffix in (("raw", "raw2000"), ("swa", "swa")):
        arrays = []
        for seed in range(22, 27):
            model = f"s{seed}_carry_{suffix}"
            selected = scores[scores.model == model]
            require(len(selected) == 122 * 40, f"Incomplete {model}")
            for column in ("model_exact_loss", "public_graphcast_exact_loss"):
                values = selected.pivot(index="initialization_time", columns="lead_hours", values=column)
                require(values.index.equals(dates) and values.columns.tolist() == LEADS.tolist(),
                        f"Cohort/lead mismatch for {model}")
                if column == "model_exact_loss":
                    arrays.append(values.to_numpy())
                else:
                    baselines[model] = values.to_numpy()
        models[f"m1_{mode}"] = np.stack(arrays)
    models["public_gc"] = baselines["gcft_lr3em7"][None, ...]
    for name, values in {**models, **baselines}.items():
        require(np.isfinite(values).all() and (values >= 0).all(), f"Invalid loss in {name}")

    labels = {
        "public_gc": "Original GraphCast",
        "gcft_lr3em7": r"GC fine-tuned, LR $3\times10^{-7}$",
        "gcft_lr1em5": r"GC fine-tuned, LR $1\times10^{-5}$",
        "gcft_lr3em5": r"GC fine-tuned, LR $3\times10^{-5}$",
        "m1_raw": "GC–Mamba M1, raw (5 seeds)",
        "m1_swa": "GC–Mamba M1, SWA (5 seeds)",
    }
    colors = {"public_gc": "#222222", "gcft_lr3em7": "#d97706", "gcft_lr1em5": "#b5462f",
              "gcft_lr3em5": "#79933c", "m1_raw": "#0072b2", "m1_swa": "#8b4c9f"}
    order = list(labels)
    table, summary = [], {}
    reference = models["public_gc"].mean(axis=(0, 1))
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    for name in order:
        per_seed = models[name].mean(axis=1)
        mean = per_seed.mean(axis=0)
        sd = per_seed.std(axis=0, ddof=1) if len(per_seed) > 1 else np.zeros(40)
        style = "--" if name in ("public_gc", "m1_swa") else "-"
        ax.plot(LEADS / 24, mean, color=colors[name], linestyle=style,
                linewidth=2.6 if name.startswith("m1") else 2.0, label=labels[name])
        if len(per_seed) > 1:
            ax.fill_between(LEADS / 24, mean - sd, mean + sd, color=colors[name], alpha=0.14, linewidth=0)
        for lead, value, spread in zip(LEADS, mean, sd):
            table.append({"model": name, "lead_hours": int(lead), "lead_days": float(lead / 24),
                          "exact_gc_mse_mean": float(value),
                          "training_seed_sample_sd": float(spread) if len(per_seed) > 1 else None,
                          "training_seed_count": len(per_seed), "initialization_count": 122})
        pooled = per_seed.mean(axis=1)
        summary[name] = {"rollout_mse_mean": float(pooled.mean()), "day10_mse_mean": float(mean[-1]),
                         "rollout_mse_seed_sd": float(pooled.std(ddof=1)) if len(pooled) > 1 else None,
                         "day10_mse_seed_sd": float(sd[-1]) if len(pooled) > 1 else None}
    ax.set(xlabel="Forecast lead time (days)", ylabel="Exact GraphCast weighted, normalized MSE",
           xlim=(0, 10), ylim=(0, None), xticks=np.arange(0, 11))
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, ncol=2)
    ax.set_title("Forecast MSE over the 10-day rollout", fontsize=17, pad=30)
    ax.text(0.5, 1.025, "122 initializations in 2023 · native 1° grid · 2,500 total training updates",
            transform=ax.transAxes, ha="center", fontsize=11)
    fig.text(0.5, 0.025, "M1: 500 warm-up + 2,000 main updates. Bands: ±1 training-seed SD; not confidence intervals.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(IMAGES / f"mse_vs_lead.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(table).to_csv(DATA / "mse_vs_lead.csv", index=False)
    drift = {}
    for name, values in baselines.items():
        curve = values.mean(axis=0)
        drift[name] = {"pooled_relative_difference_pct": float(100 * (curve.mean() / reference.mean() - 1)),
                       "max_abs_lead_relative_difference_pct": float(np.max(np.abs(100 * (curve / reference - 1))))}
    metadata = {
        "date": "2026-11-12", "metric": "Original GraphCast normalized weighted MSE; dimensionless",
        "aggregation": "Average exact per-start losses at each lead; average M1 curves across seeds 22–26",
        "uncertainty": "M1 shading is sample SD of the five date-averaged seed curves, not date uncertainty",
        "public_reference": "Independent public GC trajectory from LH GCFT LR3e-7 evaluator",
        "checkpoint_policy": "GCFT raw2500; M1 raw main2000 and SWA(main500,1000,1500,2000), following warm-up500",
        "scope": "LH local 2023 evaluation. Separate from the four-start 2022 Della power-spectrum pilot.",
        "baseline_repeatability": {"policy": "Finite differences are warnings, not gates", "measurements": drift},
        "summary": summary,
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sources],
    }
    (DATA / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"images": str(IMAGES.relative_to(ROOT)), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
