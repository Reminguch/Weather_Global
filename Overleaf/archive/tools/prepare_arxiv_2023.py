"""Prepare arXiv figures from saved 2023 evaluations, without running inference.

From Weather_global: source scripts/graphcast_env.sh && python Overleaf/tools/prepare_arxiv_2023.py
The compact CSV exports and source hashes accompany the manuscript figures.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Overleaf/Graphs"))
from paper_style import apply_style

ACCURACY = ROOT / "artifacts/evaluations/v24_res1_2023_stride3_memory_recovery_20261009"
MEMORY = ROOT / "artifacts/evaluations/v24_matched_reset_2023_20261018"
SPECTRA = ROOT / "artifacts/evaluations/v24_spectral_pilot_report_recovery_20261007/report"
OUT = ROOT / "plots/analyze_models/images/resolution_eval/arxiv_2023"
DATA = ROOT / "plots/analyze_models/data/resolution_eval/arxiv_2023"
PAPER = ROOT / "Overleaf/Graphs/arxiv_2023"
BLUE, GRAY, ORANGE, GREEN = "#0072B2", "#737373", "#D55E00", "#009E73"
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


def matched_memory():
    """Validate all paired forecast files, then export the frozen report's CIs."""
    protocol_path = MEMORY / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    helper_path = MEMORY / "control/report_v24_matched_reset_2023.py"
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == protocol["source_files"][str(helper_path)]
    spec = importlib.util.spec_from_file_location("_arxiv_matched_reset_report", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    protocol, reference, frozen, source_digest = helper.load_workflow(MEMORY)
    baseline, full, audit = helper.gather(MEMORY, protocol, reference, frozen, source_digest)
    summary_path = MEMORY / "report/summary.json"
    summary = json.loads(summary_path.read_text())
    receipt_path = MEMORY / "submissions/jobs.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "complete"
    assert receipt["report_summary_sha256"] == helper.sha256(summary_path)
    assert summary["protocol_hash"] == protocol["protocol_hash"]
    assert summary["reference_protocol_hash"] == reference["protocol_hash"]
    assert summary["model_ids"] == list(helper.MODEL_IDS)
    assert summary["initialization_count"] == 122
    assert baseline.shape == full.shape == (3, 122, 40)
    assert summary["lead_hours"] == reference["lead_hours"] == list(range(6, 241, 6))
    assert summary["audit"]["sources"] == audit["sources"]
    primary = summary["primary"]
    pooled_baseline = np.r_[baseline[0].mean(), baseline[0].mean(axis=0)]
    pooled_models = np.column_stack([full.mean(axis=(1, 2)), full.mean(axis=1)])
    np.testing.assert_allclose(primary["canonical_baseline_per_step"], pooled_baseline[1:], rtol=1e-13)
    np.testing.assert_allclose(primary["model_per_step"], pooled_models[:, 1:], rtol=1e-13)
    np.testing.assert_allclose(primary["improvement_pct"], 100 * (1 - pooled_models / pooled_baseline), rtol=1e-11)
    rows = []
    for m, model in enumerate(helper.MODEL_IDS):
        for j in range(41):
            rows.append({"analysis": "primary", "model_id": model,
                         "endpoint": "rollout" if j == 0 else "lead",
                         "lead_hours": None if j == 0 else reference["lead_hours"][j - 1],
                         "block_starts": primary["block_starts"], "segmentation": primary["segmentation"],
                         "initialization_count": 122, "baseline_loss": pooled_baseline[j],
                         "model_loss": pooled_models[m, j], "improvement_pct": primary["improvement_pct"][m][j],
                         "pointwise_ci95_lower": primary["pointwise_ci95_lower"][m][j],
                         "pointwise_ci95_upper": primary["pointwise_ci95_upper"][m][j]})
    contrasts = []
    for contrast in primary["contrasts"]:
        first = helper.MODEL_IDS.index(contrast["first_arm"])
        second = helper.MODEL_IDS.index(contrast["second_arm"])
        np.testing.assert_allclose(contrast["effect"],
            100 * (pooled_models[second] - pooled_models[first]) / pooled_baseline, rtol=1e-11)
        for j in range(41):
            contrasts.append({"analysis": "matched_memory_extension", "contrast_id": contrast["contrast_id"],
                              "endpoint": "rollout" if j == 0 else "lead",
                              "lead_hours": None if j == 0 else reference["lead_hours"][j - 1],
                              "effect_percentage_points": contrast["effect"][j],
                              "pointwise_ci95_lower": contrast["pointwise_ci95_lower"][j],
                              "pointwise_ci95_upper": contrast["pointwise_ci95_upper"][j]})
    sources = [protocol_path, helper_path, summary_path, receipt_path, MEMORY / "report/report.md"]
    sources.extend(Path(item["path"]) for item in audit["sources"])
    return pd.DataFrame(rows), pd.DataFrame(contrasts), protocol, summary, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-only", action="store_true", help="Replace the memory figure and exports, preserving other figure files.")
    args = parser.parse_args()
    apply_style()
    for directory in (OUT, DATA, PAPER):
        directory.mkdir(parents=True, exist_ok=True)
    sources = [ACCURACY / "protocol.json", ACCURACY / "report/loss_summary.csv",
               ACCURACY / "report/variable_rmse.csv", ACCURACY / "report/named_endpoints.csv",
               ACCURACY / "report/seasonal_exact_loss.csv", SPECTRA / "pooled_spectra.csv",
               SPECTRA / "summary.json",
               ROOT / "plots/analyze_models/data/resolution_eval/res1_spectral_pilot_2023_20260917/protocol.json"]
    protocol = json.loads(sources[0].read_text())
    original_loss = pd.read_csv(sources[1]).query("analysis == 'primary' and model_id in ['M1', 'M1_reset']")
    loss, memory_contrasts, memory_protocol, memory_summary, memory_sources = matched_memory()
    sources.extend(memory_sources)
    for model in ("M1", "M1_reset"):
        old = original_loss.query("model_id == @model").sort_values(["endpoint", "lead_hours"])
        new = loss.query("model_id == @model").sort_values(["endpoint", "lead_hours"])
        np.testing.assert_allclose(new.improvement_pct, old.improvement_pct, rtol=1e-11)
        np.testing.assert_allclose(new[["pointwise_ci95_lower", "pointwise_ci95_upper"]],
                                   old[["pointwise_ci95_lower", "pointwise_ci95_upper"]], rtol=1e-11)
    rmse = pd.read_csv(sources[2]).query("season == 'annual' and model_id == 'M1'")
    endpoints = pd.read_csv(sources[3]).query("analysis == 'primary' and first_arm == 'M1'")
    seasonal = pd.read_csv(sources[4]).query("model_id == 'M1'")
    spectra = pd.read_csv(sources[5]).query("lead_hours == 240 and model_id in ['GraphCast','M1']")
    spectral_summary = json.loads(sources[6].read_text())
    spectral_protocol = json.loads(sources[7].read_text())
    assert len(protocol["initialization_times"]) == 122
    assert spectral_summary["samples_per_model"] == 8
    for model in spectral_protocol["models"]:
        other = next(m for m in protocol["models"] if m["model_id"] == model["model_id"])
        assert model["checkpoint_sha256"] == other["checkpoint_sha256"]
    np.testing.assert_allclose(loss.improvement_pct, 100 * (1 - loss.model_loss / loss.baseline_loss))
    np.testing.assert_allclose(rmse.rmse_reduction_pct, 100 * (1 - rmse.model_rmse / rmse.baseline_rmse))
    curves = loss.query("endpoint == 'lead'")
    m1 = curves.query("model_id == 'M1'").sort_values("lead_hours")
    assert np.array_equal(m1.lead_hours, np.arange(6, 241, 6))
    days = m1.lead_hours.to_numpy() / 24

    shards = sorted((ACCURACY / "shards").glob("M1_[0-9][0-9].json"))
    assert len(shards) == 8
    sources.extend(shards)
    records = [json.loads(p.read_text()) for p in shards]
    expected = next(m for m in protocol["models"] if m["model_id"] == "M1")
    for record in records:
        assert record["metrics_complete"] and record["remaining_samples"] == 0
        assert record["checkpoint_sha256"] == expected["checkpoint_sha256"]
        assert record["protocol_hash"] == protocol["protocol_hash"]
    counts = np.array([r["evaluated_samples"] for r in records])
    starts = [s["initialization_time"] for r in records for s in r["original_graphcast_loss_per_initialization"]]
    assert counts.sum() == len(starts) == len(set(starts)) == 122
    assert set(pd.to_datetime(starts)) == set(pd.to_datetime(protocol["initialization_times"]))
    reset_model = next(m for m in protocol["models"] if m["model_id"] == "M1_reset")
    assert reset_model["checkpoint_sha256"] == expected["checkpoint_sha256"]
    assert reset_model["reset_state_every_step"] and not expected["reset_state_every_step"]
    reset_shards = sorted((ACCURACY / "shards").glob("M1_reset_[0-9][0-9].json"))
    assert len(reset_shards) == 8
    sources.extend(reset_shards)
    reset_starts = []
    for path in reset_shards:
        record = json.loads(path.read_text())
        assert record["metrics_complete"] and record["remaining_samples"] == 0
        assert record["checkpoint_sha256"] == expected["checkpoint_sha256"]
        assert record["protocol_hash"] == protocol["protocol_hash"]
        reset_starts.extend(s["initialization_time"] for s in record["original_graphcast_loss_per_initialization"])
    assert len(reset_starts) == len(set(reset_starts)) == 122
    assert set(pd.to_datetime(reset_starts)) == set(pd.to_datetime(starts))
    channels = sorted(records[0]["per_channel_per_step"])
    assert len(channels) == 83
    pooled = {}
    for metric in ("rmse_baseline", "rmse_full"):
        values = np.array([[r["per_channel_per_step"][c][metric] for c in channels] for r in records])
        assert values.shape == (8, 83, 40)
        pooled[metric] = np.sqrt(np.average(values ** 2, weights=counts, axis=0))
    reductions = 100 * (1 - pooled["rmse_full"] / pooled["rmse_baseline"])
    improved = (reductions > 0).sum(axis=0)
    for key, *_ in VARIABLES:
        indices = [i for i, c in enumerate(channels) if c == key or c.startswith(key + "_level")]
        assert len(indices) in (1, 13)
        group = rmse.query("endpoint == 'lead' and variable == @key").sort_values("lead_hours")
        for metric, column in (("rmse_baseline", "baseline_rmse"), ("rmse_full", "model_rmse")):
            np.testing.assert_allclose(np.sqrt(np.mean(pooled[metric][indices] ** 2, axis=0)), group[column], rtol=3e-7)
    pd.DataFrame({"lead_hours": m1.lead_hours, "channels_improved": improved, "channels_total": 83}).to_csv(DATA / "channel_coverage.csv", index=False)
    pd.DataFrame([{"channel": c, "lead_hours": (j + 1) * 6,
                   "baseline_rmse": pooled["rmse_baseline"][i, j], "model_rmse": pooled["rmse_full"][i, j],
                   "rmse_reduction_pct": reductions[i, j]} for i, c in enumerate(channels) for j in range(40)]).to_csv(DATA / "channel_rmse.csv", index=False)
    exports = []

    def save(fig, name):
        if not args.memory_only or name == "memory_policies":
            for ext in ("pdf", "png"):
                fig.savefig(OUT / f"{name}.{ext}", dpi=210)
            shutil.copy2(OUT / f"{name}.pdf", PAPER / f"{name}.pdf")
        else:
            assert (PAPER / f"{name}.pdf").is_file(), "Run once without --memory-only to create all figures"
        exports.append(name)
        plt.close(fig)

    temp = rmse.query("endpoint == 'lead' and variable == '2m_temperature'").sort_values("lead_hours")
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.3))
    fig.subplots_adjust(left=.12, right=.975, bottom=.17, top=.94, wspace=.43, hspace=.70)
    for ax, base, full in [(axes[0, 0], m1.baseline_loss, m1.model_loss), (axes[1, 0], temp.baseline_rmse, temp.model_rmse)]:
        ax.plot(days, base, color=GRAY, ls="--", label="GraphCast")
        ax.plot(days, full, color=BLUE, label="GraphCast–Mamba")
        ax.set_ylim(bottom=0)
    axes[0, 1].plot(days, m1.improvement_pct, color=BLUE)
    axes[1, 1].plot(days, temp.rmse_reduction_pct, color=BLUE)
    axes[0, 1].set_ylim(-2, 26)
    axes[1, 1].set_ylim(-1, 12)
    for ax, title, label in zip(axes.flat, ["(a) Forecast loss", "(b) Loss reduction", "(c) 2 m temperature", "(d) 2 m temperature"],
                               ["GraphCast loss", "GraphCast loss\nreduction (%)", "RMSE (K)", "RMSE reduction (%)"]):
        lead_axis(ax)
        ax.set_title(title, loc="left")
        ax.set_ylabel(label)
    for ax in axes[:, 1]:
        ax.axhline(0, color=".5", lw=.8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .025), ncol=2, frameon=False)
    save(fig, "forecast_skill")

    selected = rmse.query("endpoint == 'lead' and lead_hours in [120,144,168,192,216,240]")
    matrix = selected.pivot(index="variable", columns="lead_hours", values="rmse_reduction_pct").reindex([v[0] for v in VARIABLES]).to_numpy()
    assert matrix.shape == (11, 6) and np.isfinite(matrix).all() and (matrix > 0).all() and (matrix < 20).all()
    fig = plt.figure(figsize=(6.5, 2.75))
    ax = fig.add_axes([.325, .18, .52, .79])
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=20, aspect="auto")
    ax.set_yticks(range(11), [v[1] for v in VARIABLES])
    ax.set_xticks(range(6), [5, 6, 7, 8, 9, 10])
    ax.set_xlabel("Forecast lead time (days)")
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value >= 12 else "#202020")
    cb = fig.colorbar(im, cax=fig.add_axes([.875, .18, .016, .79]), ticks=[0, 5, 10, 15, 20])
    cb.set_label("RMSE reduction (%)")
    cb.outline.set_visible(False)
    save(fig, "overview_heatmap")

    fig, ax = plt.subplots(figsize=(6.5, 1.55))
    fig.subplots_adjust(left=.12, right=.975, bottom=.33, top=.90)
    ax.plot(days, improved, color=BLUE, marker="o", markersize=2.5, lw=1.2)
    ax.axhline(83, color=GRAY, ls=":", lw=.8)
    lead_axis(ax)
    ax.set(ylim=(0, 90), yticks=[0, 40, 83], ylabel="Channels\nimproved")
    ax.annotate(f"{improved[-1]}/83", xy=(10, improved[-1]), xytext=(9.1, 50), ha="center", fontsize=9,
                arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .7})
    save(fig, "overview_coverage")

    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    fig.subplots_adjust(left=.12, right=.98, bottom=.20, top=.89)
    for model, label, color, style in [
        ("M1", "Carry / carry", BLUE, "-"),
        ("M1_reset", "Carry / reset", ORANGE, "--"),
        ("R_matched", "Reset / reset (matched)", GREEN, "-."),
    ]:
        group = curves.query("model_id == @model").sort_values("lead_hours")
        assert np.array_equal(group.lead_hours, np.arange(6, 241, 6))
        ax.plot(days, group.improvement_pct, label=label, color=color, ls=style)
    lead_axis(ax)
    ax.axhline(0, color=".5", lw=.8)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set(ylabel="GraphCast loss reduction (%)", ylim=(min(-3, curves.pointwise_ci95_lower.min() - 1), 27))
    ax.set_title("Training / forecasting memory policies", loc="left")
    save(fig, "memory_policies")

    fig, axes = plt.subplots(4, 3, figsize=(8.2, 8.1))
    fig.subplots_adjust(left=.085, right=.98, bottom=.12, top=.96, wspace=.46, hspace=.70)
    for ax, (key, label, unit, scale) in zip(axes.flat, VARIABLES):
        group = rmse.query("endpoint == 'lead' and variable == @key").sort_values("lead_hours")
        ax.plot(days, group.baseline_rmse * scale, color=GRAY, ls="--", label="GraphCast")
        ax.plot(days, group.model_rmse * scale, color=BLUE, label="GraphCast–Mamba")
        lead_axis(ax)
        ax.set(title=label, ylabel=f"RMSE ({unit})", ylim=(0, None))
    axes.flat[-1].axis("off")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .025), ncol=2, frameon=False)
    save(fig, "variable_rmse")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
    fig.subplots_adjust(left=.10, right=.98, bottom=.21, top=.89, wspace=.36)
    for season, color in zip(["DJF", "MAM", "JJA", "SON"], [BLUE, GREEN, ORANGE, "#CC79A7"]):
        curve = seasonal.query("season == @season and endpoint == 'lead'").sort_values("lead_hours")
        point = seasonal.query("season == @season and endpoint == 'rollout'").iloc[0]
        axes[0].plot(days, curve.improvement_pct, color=color, label=f"{season} (n={int(point.initialization_count)})")
        axes[1].bar(season, point.improvement_pct, color=color, width=.65)
        axes[1].text(season, point.improvement_pct + .45, f"{point.improvement_pct:.2f}", ha="center", fontsize=8)
    lead_axis(axes[0])
    axes[0].set(ylabel="GraphCast loss reduction (%)", ylim=(-2, 28))
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_title("(a) Lead-time dependence", loc="left")
    annual = seasonal.query("season == 'annual' and endpoint == 'rollout'").iloc[0].improvement_pct
    axes[1].axhline(annual, color=GRAY, ls="--", lw=.8)
    axes[1].set(ylim=(0, 19), ylabel="Rollout loss reduction (%)", xlabel="Initialization season")
    axes[1].set_title("(b) Ten-day aggregate", loc="left")
    save(fig, "seasonal_loss")

    fields = ["T2m", "U10", "V10", "MSLP", "TP6", "T850", "Q700", "U850", "V850"]
    for metric, name, ylabel, ylim in [("prediction_truth_power_ratio", "spectral_power", "Forecast / ERA5 spectral power", (0, 1.65)),
                                      ("spectral_correlation", "spectral_correlation", "Forecast–ERA5 spectral correlation", (-.2, 1.05))]:
        fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.1), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.95, wspace=.23, hspace=.34)
        for ax, field in zip(axes.flat, fields):
            for model, label, color, style in [("GraphCast", "GraphCast", GRAY, "--"), ("M1", "GraphCast–Mamba", BLUE, "-")]:
                group = spectra.query("variable == @field and model_id == @model").sort_values("zonal_wavenumber")
                assert len(group) == 180 and group[metric].between(*ylim).all()
                ax.plot(group.zonal_wavenumber, group[metric], color=color, ls=style, lw=1.2, label=label)
            ax.axhline(1 if metric == "prediction_truth_power_ratio" else 0, color="black", ls=":", lw=.8)
            ax.set(title=field, xscale="log", xlim=(1, 180), ylim=ylim)
            ax.grid(alpha=.18, linewidth=.6)
            ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        fig.supylabel(ylabel, x=.01, fontsize=10)
        fig.supxlabel("Zonal wavenumber (cycles around a latitude circle)", y=.095, fontsize=9)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False)
        save(fig, name)

    for frame, name in [(loss, "exact_loss_primary"), (memory_contrasts, "memory_paired_contrasts"),
                        (rmse, "variable_rmse"), (endpoints, "named_endpoints"),
                        (seasonal, "seasonal_loss"), (spectra, "day10_spectra")]:
        frame.to_csv(DATA / f"{name}.csv", index=False)
    sources.extend([ACCURACY / "pilot_validation.json", ACCURACY / "report/report.md",
                    ACCURACY / "report/claim_sensitivity.csv", Path(__file__),
                    ROOT / "Overleaf/Graphs/paper_style.py"])
    manifest = {"sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(sources)],
                "figures": exports, "accuracy_starts": 122, "spectral_starts": 8,
                "primary_checkpoint_sha256": expected["checkpoint_sha256"],
                "coverage_pooling": "Shard RMSE squared, weighted by evaluated_samples, then square root; groups cross-checked against report.",
                "loss_intervals": "Lead-wise CSV confidence intervals are pointwise 95%; named_endpoints.csv retains simultaneous_ci95 fields from the original four-endpoint family. The matched memory extension and memory_paired_contrasts.csv use pointwise intervals only.",
                "displayed_memory_models": memory_summary["model_ids"],
                "matched_memory_control": {
                    "model_id": "R_matched", "checkpoint_sha256": memory_protocol["checkpoint"]["sha256"],
                    "protocol_hash": memory_protocol["protocol_hash"],
                    "reference_protocol_hash": memory_protocol["reference_protocol_hash"],
                    "coverage": {"policies": 3, "initializations_per_policy": 122, "leads": 40},
                    "canonical_baseline": "Original M1 execution of GraphCast, pooled before ratios",
                    "analysis_scope": memory_summary["analysis_scope"],
                    "primary_contrast": memory_summary["primary_endpoint"],
                    "bootstrap": {key: memory_summary["primary"][key] for key in
                                  ("method", "block_starts", "block_elapsed_days", "replicates", "seed")},
                    "training_source_limit": memory_summary["training_source_limit"],
                    "numerical_status": memory_summary["audit"]["numerical_status"],
                    "limitations": memory_summary["limitations"]},
                "numerics": "Saved repeatability warnings remain; no new inference.",
                "main_model_selection": "Prespecified M1; 2023 is not used to select another checkpoint."}
    (DATA / "figure_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in DATA.iterdir():
        shutil.copy2(path, PAPER / path.name)
    print(f"Saved {len(exports)} figures with data and provenance to {PAPER}")
    print("Day-10 improved channels:", improved[-1], "; exceptions:", [(c, reductions[i, -1]) for i, c in enumerate(channels) if reductions[i, -1] <= 0])


if __name__ == "__main__":
    main()
