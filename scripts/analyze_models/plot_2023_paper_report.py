"""Reproduce the paper's figure layouts from saved 2023 results; no inference.

Run with: source scripts/graphcast_env.sh && python scripts/analyze_models/plot_2023_paper_report.py
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Overleaf/Graphs"))
from paper_style import apply_style

ACCURACY = ROOT / "artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009"
SPECTRA = ROOT / "artifacts/evaluations/v24_spectral_pilot_report_recovery_20261007/report"
OUT = ROOT / "plots/analyze_models/images/resolution_eval/2023_paper_report"
DATA = ROOT / "plots/analyze_models/data/resolution_eval/2023_paper_report"
BLUE, GRAY = "#0072B2", "#737373"
VARIABLES = [
    ("geopotential", "Geopotential", "m²/s²", 1),
    ("temperature", "Temperature", "K", 1),
    ("specific_humidity", "Specific humidity", "g/kg", 1000),
    ("u_component_of_wind", "Zonal wind", "m/s", 1),
    ("v_component_of_wind", "Meridional wind", "m/s", 1),
    ("vertical_velocity", "Pressure vertical velocity", "Pa/s", 1),
    ("10m_u_component_of_wind", "10 m zonal wind", "m/s", 1),
    ("10m_v_component_of_wind", "10 m meridional wind", "m/s", 1),
    ("2m_temperature", "2 m temperature", "K", 1),
    ("mean_sea_level_pressure", "Mean sea-level pressure", "hPa", .01),
    ("total_precipitation_6hr", "6 h precipitation", "mm", 1000),
]


def lead_axis(ax):
    ax.set(xlim=(0, 10), xticks=[0, 2, 4, 6, 8, 10], xlabel="Forecast lead time (days)")
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    loss_file = ACCURACY / "report/loss_summary.csv"
    rmse_file = ACCURACY / "report/variable_rmse.csv"
    spectra_file = SPECTRA / "pooled_spectra.csv"
    loss = pd.read_csv(loss_file).query("analysis == 'primary'")
    rmse = pd.read_csv(rmse_file).query("season == 'annual' and model_id == 'M1'")
    spectra = pd.read_csv(spectra_file)
    protocol = json.loads((ACCURACY / "protocol.json").read_text())
    spectral_protocol_file = ROOT / "plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/protocol.json"
    spectral_protocol = json.loads(spectral_protocol_file.read_text())
    assert len(protocol["initialization_times"]) == 122
    spectral_summary = json.loads((SPECTRA / "summary.json").read_text())
    assert spectral_summary["samples_per_model"] == 8
    for model in spectral_protocol["models"]:
        other = next(m for m in protocol["models"] if m["model_id"] == model["model_id"])
        assert model["checkpoint_sha256"] == other["checkpoint_sha256"]
    np.testing.assert_allclose(loss.improvement_pct, 100 * (1 - loss.model_loss / loss.baseline_loss))
    np.testing.assert_allclose(rmse.rmse_reduction_pct, 100 * (1 - rmse.model_rmse / rmse.baseline_rmse))
    assert np.isfinite(spectra.select_dtypes(include="number").to_numpy()).all()
    curves = loss.query("endpoint == 'lead'")
    m1 = curves.query("model_id == 'M1'").sort_values("lead_hours")
    assert np.array_equal(m1.lead_hours, np.arange(6, 241, 6))
    days = m1.lead_hours.to_numpy() / 24
    temp = rmse.query("endpoint == 'lead' and variable == '2m_temperature'").sort_values("lead_hours")
    exports = []
    with PdfPages(OUT / "2023_paper_figures.pdf") as book:
        def save(fig, name):
            fig.savefig(OUT / f"{name}.png", dpi=210)
            fig.savefig(OUT / f"{name}.pdf")
            book.savefig(fig)
            exports.append(name)
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.8))
        fig.subplots_adjust(left=.115, right=.975, bottom=.23, top=.88, wspace=.43, hspace=.78)
        fig.suptitle("2023 forecast skill · 122 year-round starts", y=.985)
        for ax, base, full in [(axes[0, 0], m1.baseline_loss, m1.model_loss),
                               (axes[1, 0], temp.baseline_rmse, temp.model_rmse)]:
            ax.plot(days, base, color=GRAY, ls="--", lw=1.5, label="GraphCast")
            ax.plot(days, full, color=BLUE, lw=1.6, label="GraphCast–Mamba (M1)")
            ax.set_ylim(bottom=0)
        axes[0, 1].plot(days, m1.improvement_pct, color=BLUE, lw=1.6)
        axes[1, 1].plot(days, temp.rmse_reduction_pct, color=BLUE, lw=1.6)
        axes[0, 1].set_ylim(-2, 26)
        axes[1, 1].set_ylim(-1, 12)
        titles = ["(a) Forecast loss", "(b) Loss reduction", "(c) 2 m temperature RMSE", "(d) 2 m temperature\nRMSE reduction"]
        labels = ["GraphCast loss", "GraphCast loss\nreduction (%)", "2 m temperature\nRMSE (K)", "RMSE reduction (%)"]
        for ax, title, label in zip(axes.flat, titles, labels):
            lead_axis(ax)
            ax.set_title(title, loc="left")
            ax.set_ylabel(label)
        for ax in axes[:, 1]:
            ax.axhline(0, color=".5", lw=.8)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .08), ncol=2, frameon=False)
        aggregate = loss.query("model_id == 'M1' and endpoint == 'rollout'").iloc[0].improvement_pct
        fig.text(.5, .052, f"Exact loss reduction: rollout {aggregate:.2f}% · day 10 {m1.improvement_pct.iloc[-1]:.2f}%", ha="center", fontsize=9)
        fig.text(.5, .022, "Numerical-repeatability warning retained.", ha="center", fontsize=8)
        save(fig, "forecast_skill_2023")

        selected = rmse.query("endpoint == 'lead' and lead_hours in [120,144,168,192,216,240]")
        matrix = selected.pivot(index="variable", columns="lead_hours", values="rmse_reduction_pct").reindex([v[0] for v in VARIABLES]).to_numpy()
        assert matrix.shape == (11, 6) and np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix <= 20).all()
        fig = plt.figure(figsize=(6.5, 3.8))
        ax = fig.add_axes([.325, .19, .52, .65])
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=20, aspect="auto")
        ax.set_yticks(range(11), [v[1] for v in VARIABLES])
        ax.set_xticks(range(6), [5, 6, 7, 8, 9, 10])
        ax.set_xlabel("Forecast lead time (days)")
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
        ax.grid(which="minor", color="white", lw=.8)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9, color="white" if value >= 12 else "#202020")
        cb = fig.colorbar(im, cax=fig.add_axes([.875, .19, .016, .65]), ticks=[0, 5, 10, 15, 20])
        cb.set_label("RMSE reduction (%)")
        cb.outline.set_visible(False)
        fig.suptitle("2023 variable RMSE reduction · M1 · 122 starts", y=.96)
        fig.text(.5, .045, "Physical MSE pooled before square root; upper-air groups use equal pressure levels.", ha="center", fontsize=8)
        save(fig, "variable_rmse_heatmap_2023")

        fig, axes = plt.subplots(4, 3, figsize=(9, 9.1))
        fig.subplots_adjust(left=.08, right=.98, bottom=.11, top=.91, wspace=.42, hspace=.64)
        for ax, (key, label, unit, scale) in zip(axes.flat, VARIABLES):
            group = rmse.query("endpoint == 'lead' and variable == @key").sort_values("lead_hours")
            ax.plot(days, group.baseline_rmse * scale, color=GRAY, ls="--", label="GraphCast")
            ax.plot(days, group.model_rmse * scale, color=BLUE, label="GraphCast–Mamba (M1)")
            lead_axis(ax)
            ax.set(title=label, ylabel=f"RMSE ({unit})", ylim=(0, None))
        axes.flat[-1].axis("off")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .025), ncol=2, frameon=False)
        fig.suptitle("2023 physical-unit RMSE · M1 · 122 year-round starts", y=.975)
        save(fig, "variable_rmse_curves_2023")

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1), sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.31, top=.78, wspace=.25)
        sets = [[("M1", "M1: carry", BLUE, "-"), ("M1_reset", "M1: reset at inference", "#D55E00", "--")],
                [("C_carry", "Carry / carry", BLUE, "-"), ("C_reset", "Carry / reset", "#D55E00", "--"), ("R_reset", "Reset / reset", "#009E73", "-.")]]
        for ax, specs in zip(axes, sets):
            for model, label, color, style in specs:
                group = curves.query("model_id == @model").sort_values("lead_hours")
                ax.plot(days, group.improvement_pct, label=label, color=color, ls=style, lw=1.7)
            lead_axis(ax)
            ax.axhline(0, color=".5", lw=.8)
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        axes[0].set_ylabel("GraphCast loss reduction (%)")
        axes[0].set_title("(a) M1 inference-memory ablation", loc="left")
        axes[1].set_title("(b) V22 train / evaluate controls", loc="left")
        axes[0].set_ylim(min(-3, curves.improvement_pct.min() - 1), 26)
        fig.suptitle("2023 memory dependence · 122 year-round starts", y=.96)
        fig.text(.5, .15, "Exact original GraphCast loss throughout; common canonical baseline.", ha="center", fontsize=9)
        fig.text(.5, .08, "V22 controls are separate training runs; they are not matched capacity controls for M1.", ha="center", fontsize=8)
        save(fig, "memory_policies_2023")

        fields = ["T2m", "U10", "V10", "MSLP", "TP6", "T850", "Q700", "U850", "V850"]
        fig, axes = plt.subplots(3, 3, figsize=(8.2, 7.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.87, wspace=.23, hspace=.38)
        for ax, field in zip(axes.flat, fields):
            for model, label, color, style in [("GraphCast", "GraphCast", GRAY, "--"), ("M1", "GraphCast–Mamba (M1)", BLUE, "-")]:
                group = spectra.query("lead_hours == 240 and variable == @field and model_id == @model").sort_values("zonal_wavenumber")
                assert len(group) == 180
                ax.plot(group.zonal_wavenumber, group.prediction_truth_power_ratio, color=color, ls=style, lw=1.35, label=label)
            ax.axhline(1, color="black", ls=":", lw=.8)
            ax.set(title=field, xscale="log", xlim=(1, 180), ylim=(0, 1.65))
            ax.grid(alpha=.18, linewidth=.6)
            ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        max_ratio = spectra.query("lead_hours == 240 and variable in @fields and model_id in ['GraphCast','M1']").prediction_truth_power_ratio.max()
        assert max_ratio <= 1.65, f"Spectrum y limit would clip data: {max_ratio}"
        fig.suptitle("2023 day-10 spectral power · eight-start pilot", y=.97)
        fig.text(.5, .915, "Midlatitude zonal anomalies (30–60° N/S); unsmoothed positive modes", ha="center", fontsize=9)
        fig.supylabel("Forecast / ERA5 spectral power", x=.015, fontsize=10)
        fig.supxlabel("Zonal wavenumber (cycles around a latitude circle)", y=.105, fontsize=9)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .045), ncol=2, frameon=False)
        fig.text(.5, .014, "Dotted line: ERA5 power. Separate pilot sample; no uncertainty or preservation claim.", ha="center", fontsize=8)
        save(fig, "spectral_power_ratio_day10_2023")

    loss.to_csv(DATA / "exact_loss_primary.csv", index=False)
    rmse.to_csv(DATA / "m1_variable_rmse.csv", index=False)
    spectra.query("lead_hours == 240 and model_id in ['GraphCast','M1']").to_csv(DATA / "m1_day10_spectra.csv", index=False)
    inputs = [loss_file, rmse_file, spectra_file, ACCURACY / "protocol.json", spectral_protocol_file, SPECTRA / "summary.json", ROOT / "Overleaf/Graphs/paper_style.py", Path(__file__)]
    manifest = {
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
        "figures": exports, "accuracy_starts": 122, "spectral_starts": 8,
        "model": "M1 (prespecified, same checkpoint as paper's 2022 primary figure)",
        "loss_intervals": "pointwise 95%, paired block bootstrap; not simultaneous intervals",
        "numerical_warnings": "Saved repeatability warnings retained; no new inference performed.",
        "coverage": "No 83-channel coverage figure: this report exports variable-group RMSE, not per-channel MSE.",
        "metric_checks": "Saved loss and RMSE reductions recomputed and verified; spectral and accuracy checkpoint identities match.",
    }
    (DATA / "figure_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(OUT / "2023_paper_figures.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in exports:
            for ext in ("png", "pdf"):
                archive.write(OUT / f"{name}.{ext}", f"{name}.{ext}")
        archive.write(OUT / "2023_paper_figures.pdf", "2023_paper_figures.pdf")
        for path in sorted(DATA.glob("*")):
            archive.write(path, f"data/{path.name}")
        archive.write(Path(__file__), "plot_2023_paper_report.py")
    print(f"Saved {len(exports)} figures, combined PDF, ZIP and source data: {OUT}")


if __name__ == "__main__":
    main()
