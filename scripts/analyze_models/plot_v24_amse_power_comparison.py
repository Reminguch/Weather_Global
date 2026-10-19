"""Summarize saved AMSE SWA zonal power diagnostics; no model evaluation.

Run with: source scripts/graphcast_env.sh && python scripts/analyze_models/plot_v24_amse_power_comparison.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "plots/analyze_models/data/resolution_eval"
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval"
OUTPUT = "amse_spectral_power_comparison_20261004"
TAG = "swa_step00500-02000"
LEADS = [24, 72, 168, 240]
CAMPAIGNS = [
    ("res1_amse_blend_lambda0p1_20260924", "Blend, λ = 0.1", "#007F79"),
    ("res1_amse_pair_20260919", "Full AMSE, λ = 1", "#C65324"),
]
MSE_COLOR = "#34455A"
FIELDS = ["T2m", "U10", "V10", "MSLP", "TP6", "T850", "Q700", "U850", "V850", "WS10", "WS850"]


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_campaign(name, title, color):
    path = DATA / name / "spectral_curves.csv"
    frame = pd.read_csv(path)
    frame = frame[(frame.branch == "full") & frame.model.isin([f"mse_fp32_{TAG}", f"amse_{TAG}"])]
    keys = ["variable", "lead_hours", "wavenumber"]
    assert not frame.duplicated(keys + ["model"]).any()
    mse = frame[frame.model == f"mse_fp32_{TAG}"].set_index(keys).sort_index()
    amse = frame[frame.model == f"amse_{TAG}"].set_index(keys).sort_index()
    assert mse.index.equals(amse.index)
    assert len(mse) == len(FIELDS) * len(LEADS) * 180
    assert set(mse.index.get_level_values("variable")) == set(FIELDS)
    assert set(mse.index.get_level_values("lead_hours")) == set(LEADS)
    assert set(mse.index.get_level_values("wavenumber")) == set(range(1, 181))
    table = pd.DataFrame({
        "mse_power_error_pct": mse.relative_power_error_pct,
        "amse_power_error_pct": amse.relative_power_error_pct,
        "mse_coherence": mse.coherence,
        "amse_coherence": amse.coherence,
    })
    assert np.isfinite(table.values).all()
    table["power_error_decrease_pp"] = table.mse_power_error_pct - table.amse_power_error_pct
    pooled_path = DATA / name / "pooled_spectra.npz"
    ratios = {}
    with np.load(pooled_path, allow_pickle=False) as pooled:
        indices = [list(pooled["lead_hours"]).index(lead) for lead in LEADS]
        variables = list(pooled["variables"])
        truth = pooled[f"mse_fp32_{TAG}_truth_power"][indices, :, 1:]
        assert np.array_equal(truth, pooled[f"amse_{TAG}_truth_power"][indices, :, 1:])
        for arm, label in (("mse_fp32", "mse"), ("amse", "amse")):
            ratio = pooled[f"{arm}_{TAG}_full_power"][indices, :, 1:] / truth
            # Independently verify that the normalized spectra reproduce the saved CSV.
            grid = pd.MultiIndex.from_product([variables, LEADS, range(1, 181)], names=keys)
            error = pd.Series(100 * np.abs(ratio.transpose(1, 0, 2) - 1).ravel(), index=grid)
            np.testing.assert_allclose(error.reindex(table.index), table[f"{label}_power_error_pct"], rtol=1e-10, atol=1e-10)
            ratios[label] = ratio[:, variables.index("T850"), :]
    means = table.mean()
    summary = {
        "experiment": name,
        "label": title,
        "mse_mean_power_error_pct": float(means.mse_power_error_pct),
        "amse_mean_power_error_pct": float(means.amse_power_error_pct),
        "power_error_decrease_pp": float(means.power_error_decrease_pp),
        "relative_power_error_reduction_pct": float(100 * (1 - means.amse_power_error_pct / means.mse_power_error_pct)),
        "bins_with_lower_power_error_pct": float(100 * (table.power_error_decrease_pp > 0).mean()),
        "mse_mean_coherence": float(means.mse_coherence),
        "amse_mean_coherence": float(means.amse_coherence),
        "sources": [identity(path), identity(pooled_path)],
    }
    return {"name": name, "title": title, "color": color, "table": table, "summary": summary, "ratios": ratios}


def save(fig, stem):
    directory = IMAGES / OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E8ED", linewidth=0.8)
    ax.set_axisbelow(True)


def overview(campaigns):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("How much does AMSE improve spectral power matching?", x=.5, y=.965, fontsize=18, weight="bold")
    fig.text(.5, .923, "SWA checkpoints · 8 matched 2023 starts · zonal spectra at 30–60° in both hemispheres", ha="center", fontsize=10.5)
    for col, record in enumerate(campaigns):
        table, summary, color = record["table"], record["summary"], record["color"]
        ax = axes[0, col]
        by_lead = table.groupby("lead_hours").mean()
        x = by_lead.index.values / 24
        ax.plot(x, by_lead.mse_power_error_pct, "o-", color=MSE_COLOR, label="Matched MSE control", lw=2, ms=5)
        ax.plot(x, by_lead.amse_power_error_pct, "o-", color=color, label=record["title"], lw=2, ms=5)
        ax.set_title(record["title"], loc="left", fontsize=14, weight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xlabel("Forecast lead (days)")
        ax.set_ylabel("Mean relative power error (%)\nlower is better")
        ax.set_ylim(0, 83)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
        ax.text(.98, .04,
                f"{summary['mse_mean_power_error_pct']:.2f}% → {summary['amse_mean_power_error_pct']:.2f}%\n"
                f"{summary['relative_power_error_reduction_pct']:.2f}% lower mean power error",
                ha="right", va="bottom", transform=ax.transAxes, color=color, fontsize=11, weight="bold")
        style(ax)
        ax = axes[1, col]
        by_k = table.groupby("wavenumber").power_error_decrease_pp.mean()
        ax.axhline(0, color="#75808C", lw=1)
        ax.plot(by_k.index, by_k, color=color, lw=1.6)
        ax.fill_between(by_k.index, 0, by_k, color=color, alpha=.13)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 90, 180])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(1, 180)
        # A common scale makes the contrast between full AMSE and the blend explicit.
        ax.set_ylim(-1, 33)
        ax.set_title("Where power error decreases", loc="left", fontsize=12, pad=10)
        ax.set_ylabel("MSE error − AMSE error (pp)\npositive is better")
        ax.set_xlabel("Zonal wavenumber k  (larger k = smaller spatial scales)")
        ax.text(.04, .94, f"Lower error in {summary['bins_with_lower_power_error_pct']:.1f}% of saved bins",
                transform=ax.transAxes, va="top", fontsize=10, color=color)
        style(ax)
    fig.text(.07, .077, "Power error = 100 × |forecast power − ERA5 power| / ERA5 power. Raw powers are pooled across starts first.", fontsize=9)
    fig.text(.07, .055, "Equal weighting of 11 fields and k = 1–180; headline and lower panels also average days 1, 3, 7, 10. No smoothing.", fontsize=9)
    fig.text(.07, .033, "These are spectral diagnostics, not GraphCast loss reductions. Exploratory sample; no uncertainty intervals.", fontsize=9, color="#586574")
    fig.subplots_adjust(left=.08, right=.975, top=.855, bottom=.17, wspace=.27, hspace=.58)
    save(fig, "spectral_power_summary")


def normalized_spectra(campaigns):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    fig.suptitle("Temperature at 850 hPa: how much ERA5 spectral power is retained?", y=.96, fontsize=17, weight="bold")
    fig.text(.5, .915, "Normalized zonal power · ideal = 1 · below 1 indicates a power deficit · SWA, 8 pooled 2023 starts", ha="center", fontsize=11)
    for row, record in enumerate(campaigns):
        for col, lead in enumerate(LEADS):
            ax = axes[row, col]
            k = np.arange(1, 181)
            ax.axhline(1, color="#7A838C", lw=1, ls="--", label="ERA5 power")
            ax.plot(k, record["ratios"]["mse"][col], color=MSE_COLOR, lw=1.7, label="Matched MSE control")
            ax.plot(k, record["ratios"]["amse"][col], color=record["color"], lw=1.7, label=record["title"])
            ax.set_xscale("log")
            ax.set_xlim(1, 180)
            ax.set_ylim(0, 1.35)
            ax.set_xticks([1, 10, 100])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_title(f"Day {lead // 24}", fontsize=12)
            if col == 0:
                ax.set_ylabel(record["title"] + "\nForecast power / ERA5 power")
                ax.legend(loc="lower left", fontsize=8, frameon=False)
            if row == 1:
                ax.set_xlabel("Zonal wavenumber k")
            style(ax)
    fig.text(.065, .052, "30–60° latitude bands in both hemispheres. Curves are unsmoothed. T850 is the example field used in the existing experiment report.", fontsize=10)
    fig.text(.065, .025, "Closer power does not imply correct phase or lower forecast error; full AMSE loses deterministic skill in the matched evaluation.", fontsize=10, color="#586574")
    fig.subplots_adjust(left=.07, right=.985, top=.845, bottom=.145, hspace=.32, wspace=.13)
    save(fig, "T850_normalized_power")


def field_heatmap(campaigns):
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    fig.suptitle("Power-error decrease by field and forecast lead", y=.97, fontsize=17, weight="bold")
    fig.text(.5, .925, "MSE error − AMSE error, in percentage points · positive is better · each cell averages k = 1–180", ha="center", fontsize=10)
    for ax, record in zip(axes, campaigns):
        values = record["table"].groupby(["variable", "lead_hours"]).power_error_decrease_pp.mean().unstack().reindex(FIELDS)
        limit = float(np.abs(values.values).max())
        im = ax.imshow(values, cmap="BrBG", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(record["title"], fontsize=13, weight="bold")
        ax.set_yticks(range(len(FIELDS)), FIELDS)
        ax.set_xticks(range(len(LEADS)), [f"Day {h // 24}" for h in LEADS])
        ax.tick_params(length=0)
        for i in range(len(FIELDS)):
            for j in range(len(LEADS)):
                value = values.iloc[i, j]
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(value) > .65 * limit else "#14212B")
        fig.colorbar(im, ax=ax, fraction=.045, pad=.03, label="Power-error decrease (pp)")
    fig.text(.07, .07, "Separate color scales; read the numeric values to compare panels. SWA · 8 exploratory 2023 starts · 30–60° in both hemispheres.", fontsize=9)
    fig.text(.07, .04, "T: temperature; U/V: wind components; WS: wind speed; Q: specific humidity; MSLP: sea-level pressure; TP6: 6-hour precipitation.", fontsize=9)
    fig.subplots_adjust(left=.07, right=.965, top=.86, bottom=.13, wspace=.29)
    save(fig, "spectral_power_by_field")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    records = [load_campaign(*values) for values in CAMPAIGNS]
    destination = DATA / OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    description = {
        "checkpoint": TAG,
        "initializations": 8,
        "evaluation_year": 2023,
        "lead_days": [h // 24 for h in LEADS],
        "fields": FIELDS,
        "wavenumbers": [1, 180],
        "power_error_definition": "100 * abs(mean_start(forecast_power) - mean_start(truth_power)) / mean_start(truth_power)",
        "aggregation": "Equal weight per field, selected lead, and wavenumber; ratios computed after pooling raw powers across starts.",
        "relative_power_error_reduction_definition": "100 * (1 - mean_amse_power_error / mean_mse_power_error)",
        "scope": "Midlatitude zonal spectra (30–60 degrees, both hemispheres), not global spherical harmonics. Exploratory sample, no uncertainty intervals. This is not the original GraphCast loss metric.",
        "campaigns": [record["summary"] for record in records],
        "script": identity(Path(__file__)),
    }
    (destination / "summary.json").write_text(json.dumps(description, indent=2) + "\n")
    for record in records:
        record["table"].to_csv(destination / f"{record['name']}_matched_bins.csv")
        for grouping in ("lead_hours", "variable"):
            table = record["table"].groupby(grouping).mean()
            table["relative_power_error_reduction_pct"] = 100 * (1 - table.amse_power_error_pct / table.mse_power_error_pct)
            table.to_csv(destination / f"{record['name']}_by_{grouping}.csv")
    overview(records)
    normalized_spectra(records)
    field_heatmap(records)
    print(json.dumps([record["summary"] for record in records], indent=2))
    print(IMAGES / OUTPUT)


if __name__ == "__main__":
    main()
