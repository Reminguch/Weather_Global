#!/usr/bin/env python3
"""Render v4 seed-uncertainty figures from existing, explicitly allowed scores.

No forecasts or bootstrap resampling are run. Original memory intervals are
reused verbatim; local bands describe sample standard deviation over seeds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "results/LH_rebuttal_results/weatherdata20261111/03_seed_replication"
ARCHIVE = ROOT / "Overleaf/archive/Graphs/arxiv_2023"
RUN_NAME = "v4_seed_uncertainty_20261113"
SEEDS = tuple(range(22, 27))
SWA_MODELS = tuple(f"s{seed}_carry_swa" for seed in SEEDS)
RAW_MODELS = tuple(f"s{seed}_carry_raw2000" for seed in SEEDS)
CONTROLS = ("s22_carry_reset", "s22_reset_reset")
ALLOWED = {("five_seed_swa", model) for model in SWA_MODELS} | {
    ("fig6_comparison", model) for model in (*RAW_MODELS, *CONTROLS)
}
LEADS = np.arange(6, 241, 6)
BLUE, GRAY, ORANGE, GREEN = "#0072B2", "#737373", "#D55E00", "#009E73"
VALUE_COLUMNS = [
    "public_graphcast_exact_loss", "model_exact_loss",
    "public_graphcast_t2m_mse_k2", "model_t2m_mse_k2",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def read_scores() -> tuple[pd.DataFrame, dict]:
    # Reject unrelated rows before parsing numerical values. This export also
    # contains other experiment families that are outside the authorized scope.
    with gzip.open(CAMPAIGN / "per_start_scores.csv.gz", "rt") as handle:
        rows = [row for row in csv.DictReader(handle)
                if (row["group"], row["model"]) in ALLOWED]
    scores = pd.DataFrame(rows)
    scores["lead_hours"] = scores.lead_hours.astype(int)
    scores["training_seed"] = scores.training_seed.astype(int)
    scores[VALUE_COLUMNS] = scores[VALUE_COLUMNS].astype(float)
    expected_starts = pd.date_range("2023-01-01", "2023-12-30", freq="3D", tz="UTC")
    checks = {}
    assert set(zip(scores.group, scores.model)) == ALLOWED
    for model, frame in scores.groupby("model", sort=True):
        assert len(frame) == 122 * 40, (model, len(frame))
        assert not frame.duplicated(["initialization_time", "lead_hours"]).any()
        starts = pd.DatetimeIndex(pd.to_datetime(frame.initialization_time.unique(), utc=True)).sort_values()
        assert starts.equals(expected_starts), model
        assert np.array_equal(np.sort(frame.lead_hours.unique()), LEADS), model
        assert np.isfinite(frame[VALUE_COLUMNS].to_numpy()).all(), model
        assert (frame[VALUE_COLUMNS] >= 0).all().all(), model
        assert (frame.public_graphcast_exact_loss > 0).all(), model
        valid = pd.to_datetime(frame.initialization_time, utc=True) + pd.to_timedelta(frame.lead_hours, unit="h")
        assert (valid == pd.to_datetime(frame.valid_time, utc=True)).all(), model
        checks[model] = {"rows": len(frame), "starts": len(starts), "leads": 40,
                         "unique_start_lead_pairs": True, "finite_scores": True}
    return scores, checks


def lead_scores(scores: pd.DataFrame, models: tuple[str, ...], common_baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    output = []
    for model in models:
        selected = scores.loc[scores.model == model]
        pooled = selected.groupby("lead_hours")[VALUE_COLUMNS].mean().reindex(LEADS)
        if common_baseline is not None:
            pooled["public_graphcast_exact_loss"] = common_baseline.public_graphcast_exact_loss
            pooled["public_graphcast_t2m_mse_k2"] = common_baseline.public_graphcast_t2m_mse_k2
        pooled["model_t2m_rmse_k"] = np.sqrt(pooled.model_t2m_mse_k2)
        pooled["public_graphcast_t2m_rmse_k"] = np.sqrt(pooled.public_graphcast_t2m_mse_k2)
        pooled["loss_reduction_pct"] = 100 * (1 - pooled.model_exact_loss / pooled.public_graphcast_exact_loss)
        pooled["t2m_rmse_reduction_pct"] = 100 * (1 - pooled.model_t2m_rmse_k / pooled.public_graphcast_t2m_rmse_k)
        pooled["model"] = model
        pooled["training_seed"] = int(selected.training_seed.iloc[0])
        pooled["baseline_policy"] = "common_raw_seed22" if common_baseline is not None else "each_seed_paired"
        output.append(pooled.reset_index())
    return pd.concat(output, ignore_index=True)


def aggregate_curves(curves: pd.DataFrame) -> pd.DataFrame:
    metrics = ["model_exact_loss", "public_graphcast_exact_loss", "loss_reduction_pct",
               "model_t2m_rmse_k", "public_graphcast_t2m_rmse_k", "t2m_rmse_reduction_pct"]
    grouped = curves.groupby("lead_hours")[metrics]
    result = grouped.mean().add_suffix("_mean")
    result = result.join(grouped.std(ddof=1).add_suffix("_sample_sd"))
    result["training_seed_count"] = len(SEEDS)
    return result.reset_index()


def endpoint_scores(scores: pd.DataFrame, models: tuple[str, ...], common_model: str | None = None) -> pd.DataFrame:
    rows = []
    for model in models:
        for endpoint in ("rollout", "day10"):
            frame = scores.loc[scores.model == model]
            base = scores.loc[scores.model == common_model] if common_model else frame
            if endpoint == "day10":
                frame, base = frame.loc[frame.lead_hours == 240], base.loc[base.lead_hours == 240]
            baseline_loss, model_loss = base.public_graphcast_exact_loss.mean(), frame.model_exact_loss.mean()
            rows.append({"model": model, "training_seed": int(frame.training_seed.iloc[0]),
                         "endpoint": endpoint, "baseline_loss": baseline_loss,
                         "model_loss": model_loss, "loss_reduction_pct": 100 * (1 - model_loss / baseline_loss),
                         "baseline_policy": "common_raw_seed22" if common_model else "each_seed_paired"})
    return pd.DataFrame(rows)


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 10,
                         "legend.fontsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "lines.linewidth": 1.6})


def lead_axis(ax: plt.Axes) -> None:
    ax.set(xlim=(0, 10), xticks=[0, 2, 4, 6, 8, 10], xlabel="Forecast lead (days)")
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, name: str, image_dir: Path, paper_dir: Path) -> list[str]:
    paths = []
    for suffix in ("pdf", "png"):
        path = image_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=240, facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    shutil.copy2(image_dir / f"{name}.pdf", paper_dir / f"{name}.pdf")
    plt.close(fig)
    return paths


def plot_skill(curves: pd.DataFrame, aggregate: pd.DataFrame, image_dir: Path, paper_dir: Path) -> list[str]:
    fig, axes = plt.subplots(2, 2, figsize=(6, 5.9))
    fig.subplots_adjust(left=.12, right=.98, bottom=.21, top=.945, wspace=.48, hspace=.64)
    x = aggregate.lead_hours.to_numpy() / 24
    panels = [
        ("model_exact_loss", "(a) Forecast loss", "GraphCast loss", "public_graphcast_exact_loss"),
        ("loss_reduction_pct", "(b) Loss reduction", "Loss reduction (%)", None),
        ("model_t2m_rmse_k", "(c) 2 m temperature", "RMSE (K)", "public_graphcast_t2m_rmse_k"),
        ("t2m_rmse_reduction_pct", "(d) 2 m temperature", "RMSE reduction (%)", None),
    ]
    for ax, (metric, title, ylabel, baseline_metric) in zip(axes.flat, panels):
        for _, frame in curves.groupby("model"):
            frame = frame.sort_values("lead_hours")
            ax.plot(frame.lead_hours / 24, frame[metric], color=BLUE, alpha=.24, lw=.75)
        mean = aggregate[f"{metric}_mean"].to_numpy()
        sd = aggregate[f"{metric}_sample_sd"].to_numpy()
        ax.fill_between(x, mean - sd, mean + sd, color=BLUE, alpha=.16, lw=0)
        ax.plot(x, mean, color=BLUE, lw=2)
        if baseline_metric:
            ax.plot(x, aggregate[f"{baseline_metric}_mean"], color=GRAY, ls="--", lw=1.6)
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(0, color=GRAY, lw=.7)
        lead_axis(ax)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
    handles = [Line2D([], [], color=BLUE, lw=2, label="M1 mean (5 seeds)"),
               Line2D([], [], color=GRAY, ls="--", label="Mean paired GraphCast"),
               Line2D([], [], color=BLUE, alpha=.35, lw=.8, label="Individual seeds"),
               Patch(facecolor=BLUE, alpha=.16, label="±1 seed SD")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .045),
               ncol=2, frameon=False, columnspacing=1.0, handlelength=1.6)
    fig.text(.53, .014, "122 starts in 2023 · SWA · repaired verification fields", ha="center", fontsize=10)
    return save_figure(fig, "fig4_forecast_skill_v4_seeds", image_dir, paper_dir)


def plot_memory(original: pd.DataFrame, raw: pd.DataFrame, raw_aggregate: pd.DataFrame,
                image_dir: Path, paper_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(6, 4.05), sharey=True)
    fig.subplots_adjust(left=.115, right=.985, bottom=.29, top=.84, wspace=.21)
    for model, label, color, linestyle in [
        ("M1", "Carry / carry", BLUE, "-"),
        ("M1_reset", "Carry / reset", ORANGE, "--"),
        ("R_matched", "Reset / reset", GREEN, "-.")]:
        frame = original.loc[(original.model_id == model) & (original.endpoint == "lead")].sort_values("lead_hours")
        x = frame.lead_hours.to_numpy() / 24
        axes[0].fill_between(x, frame.pointwise_ci95_lower, frame.pointwise_ci95_upper,
                             color=color, alpha=.12, lw=0)
        axes[0].plot(x, frame.improvement_pct, color=color, ls=linestyle, label=label)
    for model in RAW_MODELS:
        frame = raw.loc[raw.model == model].sort_values("lead_hours")
        axes[1].plot(frame.lead_hours / 24, frame.loss_reduction_pct, color=BLUE, lw=.7, alpha=.22)
    x = raw_aggregate.lead_hours.to_numpy() / 24
    mean = raw_aggregate.loss_reduction_pct_mean.to_numpy()
    sd = raw_aggregate.loss_reduction_pct_sample_sd.to_numpy()
    axes[1].fill_between(x, mean - sd, mean + sd, color=BLUE, alpha=.16, lw=0)
    axes[1].plot(x, mean, color=BLUE, lw=2, label="Carry / carry (n=5)")
    for model, label, color, linestyle in [
        ("s22_carry_reset", "Carry / reset (n=1)", ORANGE, "--"),
        ("s22_reset_reset", "Reset / reset (n=1)", GREEN, "-.")]:
        frame = raw.loc[raw.model == model].sort_values("lead_hours")
        axes[1].plot(frame.lead_hours / 24, frame.loss_reduction_pct, color=color, ls=linestyle, label=label)
    axes[0].set_title("(a) Original SWA", loc="left", pad=17)
    axes[1].set_title("(b) Independent replication\n(raw checkpoints)", loc="left", pad=8)
    axes[0].set_ylabel("GraphCast loss reduction (%)")
    lower = min(float(original.pointwise_ci95_lower.min()),
                float(raw.loss_reduction_pct.min()), float((mean - sd).min()))
    upper = max(float(original.pointwise_ci95_upper.max()),
                float(raw.loss_reduction_pct.max()), float((mean + sd).max()))
    for ax in axes:
        lead_axis(ax)
        ax.axhline(0, color=GRAY, lw=.7)
        ax.set_ylim(min(-2, np.floor(lower - .5)), max(27, np.ceil(upper + .5)))
        ax.legend(loc="upper left", frameon=False, fontsize=10,
                  borderaxespad=.1, labelspacing=.25, handlelength=1.8)
    fig.text(.33, .14, "Bands: pointwise 95% date CI", ha="center", fontsize=10)
    fig.text(.765, .14, "Band: ±1 SD over carry seeds", ha="center", fontsize=10)
    fig.text(.54, .073, "Each panel uses one common GraphCast baseline;", ha="center", fontsize=10)
    fig.text(.54, .035, "the replication uses repaired verification fields.", ha="center", fontsize=10)
    return save_figure(fig, "fig6_memory_policies_v4_seeds", image_dir, paper_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "plots/analyze_models/data/resolution_eval" / RUN_NAME)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "plots/analyze_models/images/resolution_eval" / RUN_NAME)
    args = parser.parse_args()
    for path in (args.data_dir, args.image_dir):
        path.mkdir(parents=True, exist_ok=True)
    paper_dir = ROOT / "Overleaf/arxiv_pics"
    scores, checks = read_scores()
    swa = lead_scores(scores, SWA_MODELS)
    swa_aggregate = aggregate_curves(swa)
    swa_endpoints = endpoint_scores(scores, SWA_MODELS)
    baseline = scores.loc[scores.model == RAW_MODELS[0]].groupby("lead_hours")[VALUE_COLUMNS].mean().reindex(LEADS)
    raw = lead_scores(scores, (*RAW_MODELS, *CONTROLS), baseline)
    raw_aggregate = aggregate_curves(raw.loc[raw.model.isin(RAW_MODELS)])
    raw_endpoints = endpoint_scores(scores, (*RAW_MODELS, *CONTROLS), RAW_MODELS[0])
    reference = json.loads((CAMPAIGN / "seed_summary.json").read_text())
    summary = {}
    for endpoint in ("rollout", "day10"):
        values = swa_endpoints.loc[swa_endpoints.endpoint == endpoint, "loss_reduction_pct"].to_numpy()
        expected = reference["between_seed_variation"][endpoint]
        np.testing.assert_allclose([values.mean(), values.std(ddof=1)],
                                   [expected["mean_pct"], expected["sample_sd_percentage_points"]],
                                   atol=1e-10, rtol=0)
        summary[endpoint] = {"mean_pct": float(values.mean()), "sample_sd_pp": float(values.std(ddof=1)),
                             "min_pct": float(values.min()), "max_pct": float(values.max()), "n": 5}
    original = pd.read_csv(ARCHIVE / "exact_loss_primary.csv")
    original = original.loc[original.model_id.isin(["M1", "M1_reset", "R_matched"])].copy()
    original_contrasts = pd.read_csv(ARCHIVE / "memory_paired_contrasts.csv")
    for model, frame in original.groupby("model_id"):
        assert len(frame.loc[frame.endpoint == "rollout"]) == 1, model
        leads = frame.loc[frame.endpoint == "lead"].sort_values("lead_hours")
        assert np.array_equal(leads.lead_hours, LEADS), model
        assert np.isfinite(leads[["baseline_loss", "model_loss", "improvement_pct",
                                  "pointwise_ci95_lower", "pointwise_ci95_upper"]]).all().all()
        np.testing.assert_allclose(100 * (1 - frame.model_loss / frame.baseline_loss), frame.improvement_pct, atol=1e-10)
    base_matrix = original.loc[original.endpoint == "lead"].pivot(index="lead_hours", columns="model_id", values="baseline_loss")
    np.testing.assert_allclose(base_matrix.to_numpy(), np.repeat(base_matrix[["M1"]].to_numpy(), 3, axis=1), atol=1e-12)
    for contrast, left, right in [("M1_vs_R_matched", "M1", "R_matched"),
                                  ("M1_reset_vs_R_matched", "M1_reset", "R_matched"),
                                  ("M1_vs_M1_reset", "M1", "M1_reset")]:
        for endpoint in ("rollout", "lead"):
            direct = [original.loc[(original.model_id == model) & (original.endpoint == endpoint)]
                      .sort_values("lead_hours").improvement_pct.to_numpy() for model in (left, right)]
            saved = original_contrasts.loc[(original_contrasts.contrast_id == contrast) &
                                           (original_contrasts.endpoint == endpoint)].sort_values("lead_hours")
            np.testing.assert_allclose(direct[0] - direct[1], saved.effect_percentage_points, atol=1e-10)
    style()
    outputs = plot_skill(swa, swa_aggregate, args.image_dir, paper_dir)
    outputs += plot_memory(original, raw, raw_aggregate, args.image_dir, paper_dir)
    tables = {"figure4_per_seed_curves": swa, "figure4_mean_sd_curves": swa_aggregate,
              "swa_seed_endpoints": swa_endpoints, "figure6_raw_curves": raw,
              "figure6_raw_carry_mean_sd_curves": raw_aggregate, "raw_memory_endpoints": raw_endpoints,
              "figure6_original_swa_curves": original, "original_memory_paired_contrasts": original_contrasts}
    for name, frame in tables.items():
        frame.to_csv(args.data_dir / f"{name}.csv", index=False)
    identities = json.loads((CAMPAIGN / "provenance/checkpoint_identities.json").read_text())
    identities["records"] = {key: value for key, value in identities["records"].items()
                             if key in (*SWA_MODELS, *RAW_MODELS, *CONTROLS)}
    assert set(identities["records"]) == set((*SWA_MODELS, *RAW_MODELS, *CONTROLS))
    assert identities["records"]["s22_carry_raw2000"]["checkpoint_sha256"] == identities["records"]["s22_carry_reset"]["checkpoint_sha256"]
    sources = [Path(__file__), CAMPAIGN / "per_start_scores.csv.gz", CAMPAIGN / "seed_summary.json",
               CAMPAIGN / "provenance/seed_training_manifest.json", CAMPAIGN / "provenance/checkpoint_identities.json",
               CAMPAIGN / "provenance/native_truth_audit.json", CAMPAIGN / "provenance/evaluation_job.json",
               CAMPAIGN / "provenance/initializations_2023_stride3.json", CAMPAIGN / "provenance/fig6_reset_manifest.json",
               ARCHIVE / "exact_loss_primary.csv", ARCHIVE / "memory_paired_contrasts.csv"]
    sources += sorted((CAMPAIGN / "provenance/resolved_configs").glob("s*_carry_*.json"))
    sources += sorted((CAMPAIGN / "provenance/resolved_configs").glob("s22_reset_reset_*.json"))
    notes = [
        "Five new local retrainings (seeds 22–26); local seed22 is distinct from the historical selected checkpoint.",
        "Recipe reconstructed from Appendix A.1 and public configs; historical Della training configs/environment unavailable locally.",
        "Figure4 averages per-seed metrics; each seed first pools all starts (and leads for rollout), then forms loss ratios or RMSE.",
        "Figure4 dashed baseline is the arithmetic mean of the five paired baseline curves; baseline T2m RMSE is calculated within each run before averaging.",
        "Shading in Figure4 and Figure6 replication is ±1 sample SD across training seeds, not a confidence interval or an ensemble forecast.",
        "Original Figure6 shading reuses paired date-block 95% intervals conditional on fixed historical checkpoints; it is not seed uncertainty.",
        "Figure6 replication uses raw main-step 2000 weights after 500 warmup updates; its controls each have one training seed, so it does not measure multi-seed memory contrasts.",
        "Every Figure6 replication curve uses the public GraphCast baseline from local raw seed22; the historical panel uses its original common baseline.",
        "Local verification replaces stale pressure T/Z/q fields at 2023-11-10 06/12/18 UTC using NCAR ERA5 (nine start/lead cells, zero initial-input overlap).",
        "Historical and local panels retain distinct training/environment and verification provenance; historical model is not a sixth replication seed.",
        "No new training, forecasts, or bootstrap jobs were performed; no full-GraphCast fine-tuning rows enter these outputs.",
    ]
    report = {"schema_version": 1, "allowed_models": sorted(identities["records"]),
              "checks": {"per_model": checks, "seed_summary_matches_existing_export": True,
                         "original_common_baseline": True, "carry_reset_same_raw_weights": True},
              "swa_between_seed_summary": summary, "swa_seed_endpoints": json_records(swa_endpoints),
              "raw_memory_endpoints": json_records(raw_endpoints),
              "original_memory_paired_contrasts": json_records(original_contrasts),
              "original_endpoints": json_records(original.loc[original.endpoint == "rollout"]),
              "existing_date_bootstrap": reference["date_bootstrap"],
              "per_seed_existing_date_intervals": reference["seeds"],
              "checkpoint_identities": identities, "sources": {str(path.relative_to(ROOT)): digest(path) for path in sources},
              "curve_tables": {name: f"{name}.csv" for name in tables},
              "caveats": notes, "figures": outputs}
    (args.data_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": summary, "figures": outputs,
                      "data_dir": str(args.data_dir.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
