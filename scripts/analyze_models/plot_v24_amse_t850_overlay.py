"""Add lambda=0.1 to the saved full-AMSE T850 power diagnostic.

Run from the repository with::

    source scripts/graphcast_env.sh && python scripts/analyze_models/plot_v24_amse_t850_overlay.py

This reads existing pooled spectra and specifications; no forecasts are run.
The original power-error PNG is preserved once before replacing it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "plots/analyze_models/data/resolution_eval"
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval"
CHECKPOINTS = ROOT / "artifacts/checkpoints/v24_Ilya"
PAIR = "res1_amse_pair_20260919"
BLEND = "res1_amse_blend_lambda0p1_20260924"
TAG = "swa_step00500-02000"
LEADS = (24, 72, 168, 240)
VARIABLE = "T850"
ARMS = (
    (PAIR, "mse_fp32", "MSE control (full-AMSE campaign)", "#0072B2", "-", 1.7),
    (BLEND, "mse_fp32", "MSE control (blend campaign)", "#0072B2", "--", 1.0),
    (PAIR, "amse", "Full AMSE (λ = 1)", "#D55E00", "-", 1.7),
    (BLEND, "amse", "AMSE blend (λ = 0.1)", "#009E73", "-", 1.7),
)
SPEC_MATCH_KEYS = (
    "stage", "tag", "architecture", "target_steps", "lead_hours", "baseline",
    "statistics", "forecast", "data_path", "evaluation_year", "n_samples",
    "warmup_steps", "initialization_times", "initialization_manifest_sha256",
    "spectral_variables", "spectral_metadata", "data_metadata",
)


def identity(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_records():
    archives = {}
    sources = []
    for campaign in (PAIR, BLEND):
        path = DATA / campaign / "pooled_spectra.npz"
        with np.load(path, allow_pickle=False) as archive:
            archives[campaign] = {key: archive[key] for key in archive.files}
        sources.append(identity(path))
    reference_spec = None
    reference_truth = None
    records = []
    for campaign, arm, label, color, linestyle, linewidth in ARMS:
        arrays = archives[campaign]
        spec_path = CHECKPOINTS / campaign / "eval/spectral" / arm / TAG / "specification.json"
        spec = json.loads(spec_path.read_text())
        sources.append(identity(spec_path))
        require(spec["arm"] == arm, f"Unexpected arm in {spec_path}")
        require(spec["stage"] == "spectral" and spec["tag"] == TAG, "Wrong evaluation stage/tag")
        if reference_spec is None:
            reference_spec = spec
        for key in SPEC_MATCH_KEYS:
            require(spec[key] == reference_spec[key], f"Evaluation mismatch for {campaign}/{arm}: {key}")
        np.testing.assert_array_equal(arrays["lead_hours"], spec["lead_hours"])
        np.testing.assert_array_equal(arrays["variables"], spec["spectral_variables"])
        np.testing.assert_array_equal(arrays["wavenumber"], archives[PAIR]["wavenumber"])
        np.testing.assert_array_equal(arrays["wavenumber"], np.arange(181))
        prefix = f"{arm}_{TAG}"
        truth = arrays[f"{prefix}_truth_power"]
        if reference_truth is None:
            reference_truth = truth
        np.testing.assert_array_equal(truth, reference_truth)
        require(truth.shape == (40, 11, 181), f"Unexpected truth shape {truth.shape}")
        require(np.all(truth[..., 1:] > 0), "Nonpositive truth power in nonzero modes")
        lead_indices = [list(arrays["lead_hours"]).index(lead) for lead in LEADS]
        vi = list(arrays["variables"]).index(VARIABLE)
        selection = np.ix_(lead_indices, [vi], np.arange(1, 181))
        selected_truth = truth[selection][:, 0, :]
        record = dict(campaign=campaign, arm=arm, label=label, color=color,
                      linestyle=linestyle, linewidth=linewidth,
                      checkpoint=spec["checkpoint"], checkpoint_sha256=spec["checkpoint_sha256"])
        record["truth_power"] = selected_truth
        for branch in ("baseline", "full"):
            power = arrays[f"{prefix}_{branch}_power"][selection][:, 0, :]
            cross = arrays[f"{prefix}_{branch}_cross_power"][selection][:, 0, :]
            error = arrays[f"{prefix}_{branch}_error_power"][selection][:, 0, :]
            require(np.isfinite(power).all() and np.all(power > 0), "Invalid forecast powers")
            coherence = cross / np.sqrt(power * selected_truth)
            require(np.isfinite(coherence).all() and np.all(np.abs(coherence) <= 1 + 1e-6),
                    "Invalid forecast–truth coherence")
            np.testing.assert_allclose(error, power + selected_truth - 2 * cross,
                                       rtol=1e-10, atol=1e-10)
            record[branch] = {
                "power": power,
                "normalized_power": power / selected_truth,
                "relative_power_error_pct": 100 * np.abs(power - selected_truth) / selected_truth,
                "coherence": np.clip(coherence, -1, 1),
                "error_power": error,
            }
        # Verify the overlay reproduces the campaign's saved report metrics.
        report_rows = {}
        with (DATA / campaign / "spectral_curves.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model"] == prefix and row["variable"] == VARIABLE:
                    report_rows[(row["branch"], int(row["lead_hours"]), int(row["wavenumber"]))] = row
        for branch in ("baseline", "full"):
            for li, lead in enumerate(LEADS):
                for ki, k in enumerate(range(1, 181)):
                    row = report_rows[(branch, lead, k)]
                    for metric in ("power", "relative_power_error_pct", "coherence"):
                        np.testing.assert_allclose(record[branch][metric][li, ki], float(row[metric]),
                                                   rtol=1e-10, atol=1e-10)
        sources.append(identity(DATA / campaign / "spectral_curves.csv"))
        records.append(record)
    require(len(reference_spec["initialization_times"]) == 8, "Expected eight matched starts")
    return records, reference_spec, sources


def plot(records, metric, directory):
    power = metric == "relative_power_error_pct"
    ylabel = "Relative power-spectrum error (%)" if power else "Forecast–truth spectral coherence"
    formula = ("100 × |forecast power − ERA5 power| / ERA5 power · Lower is better" if power else
               "Real cospectrum / sqrt(forecast power × ERA5 power) · Higher is better")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    k = np.arange(1, 181)
    baseline_styles = ("--", ":", "-.", (0, (4, 2, 1, 2)))
    for ax, li, lead in zip(axes.ravel(), range(len(LEADS)), LEADS, strict=True):
        ax.axhline(0 if power else 1, color="black", lw=.7)
        for record, linestyle in zip(records, baseline_styles, strict=True):
            ax.plot(k, record["baseline"][metric][li], color="#808080", alpha=.52,
                    ls=linestyle, lw=1.0, zorder=2)
        for record in records:
            ax.plot(k, record["full"][metric][li], color=record["color"],
                    ls=record["linestyle"], lw=record["linewidth"], label=record["label"], zorder=3)
        ax.set_xscale("log")
        ax.tick_params(labelbottom=True)
        ax.set_title(f"Day {lead // 24}")
        ax.set_xlabel("Zonal wavenumber (cycles / latitude circle)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2, which="both")
    if power:
        axes[0, 0].set_ylim(bottom=0)
    else:
        axes[0, 0].set_ylim(-1.03, 1.03)
    fig.suptitle(f"T850: {ylabel.lower()} · {TAG}", y=.975, fontsize=15)
    fig.text(.5, .932, formula, ha="center", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .101), ncol=2,
               frameon=False, fontsize=10, columnspacing=2.2)
    baseline_handle = Line2D([], [], color="#808080", ls="--", lw=1.2, alpha=.65)
    fig.legend([baseline_handle], ["Original GC (4 separate evaluations)"], loc="lower center",
               bbox_to_anchor=(.5, .071), frameon=False, fontsize=10)
    fig.text(.5, .046,
             "1° · Eight matched 2023 starts · 30–60° both hemispheres · Pooled powers before ratios",
             ha="center", fontsize=9)
    fig.text(.5, .023,
             "Repeated GC baselines differ; all four curves are shown separately in gray. No averaging or smoothing.",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.855, bottom=.265, left=.095, right=.97, hspace=.43, wspace=.24)
    stem = f"{metric}_{TAG}_T850" + ("" if power else "_with_lambda0p1")
    outputs = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, dpi=160, facecolor="white")
        outputs.append(identity(path))
    plt.close(fig)
    return outputs


def save_tables(records, directory):
    rows, summaries = [], []
    for record in records:
        for branch in ("baseline", "full"):
            for li, lead in enumerate(LEADS):
                for ki, k in enumerate(range(1, 181)):
                    rows.append(dict(campaign=record["campaign"], arm=record["arm"], branch=branch,
                                     checkpoint=TAG, variable=VARIABLE, lead_hours=lead, wavenumber=k,
                                     truth_power=record["truth_power"][li, ki],
                                     **{key: value[li, ki] for key, value in record[branch].items()}))
                summaries.append(dict(campaign=record["campaign"], arm=record["arm"], branch=branch,
                                      checkpoint=TAG, variable=VARIABLE, lead_days=lead // 24,
                                      mean_relative_power_error_pct=record[branch]["relative_power_error_pct"][li].mean(),
                                      mean_coherence=record[branch]["coherence"][li].mean(),
                                      zonal_anomaly_mse=record[branch]["error_power"][li].sum()))
    for filename, table in (("spectral_curves.csv", rows), ("lead_summary.csv", summaries)):
        with (directory / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    records, spec, sources = load_records()
    directory = IMAGES / PAIR
    data_directory = DATA / PAIR / "t850_lambda_overlay"
    data_directory.mkdir(parents=True, exist_ok=True)
    original = directory / f"relative_power_error_pct_{TAG}_T850.png"
    backup = directory / "before_lambda0p1" / original.name
    if original.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, backup)
    outputs = []
    for metric in ("relative_power_error_pct", "coherence"):
        outputs.extend(plot(records, metric, directory))
    save_tables(records, data_directory)
    provenance = {
        "script": identity(Path(__file__)),
        "sources": list({source["path"]: source for source in sources}.values()),
        "outputs": outputs,
        "original_power_plot_backup": identity(backup) if backup.exists() else None,
        "checkpoint": TAG,
        "variable": VARIABLE,
        "lead_hours": list(LEADS),
        "wavenumbers": [1, 180],
        "initialization_times": spec["initialization_times"],
        "spectral_metadata": spec["spectral_metadata"],
        "validation": {
            "matched_specification_keys": list(SPEC_MATCH_KEYS),
            "truth_all_leads_variables_and_wavenumbers": "exact array equality in all four arms",
            "report_metric_reproduction": "rtol=1e-10, atol=1e-10 versus both saved spectral_curves.csv files",
            "error_power_identity": "error_power = forecast_power + truth_power - 2 * real_cross_power",
        },
        "baseline_policy": "Four repeated original GC evaluations plotted separately in gray; never averaged.",
        "summary_weighting": "Equal weight across k=1..180 for each T850 lead; zonal_anomaly_mse sums error power.",
        "summary_scope": "Midlatitude zonal-anomaly diagnostics, not global GraphCast loss or full-field RMSE.",
        "models": [{key: record[key] for key in ("campaign", "arm", "label", "checkpoint", "checkpoint_sha256")}
                   for record in records],
    }
    (data_directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"plots": [value["path"] for value in outputs],
                      "data": str(data_directory.relative_to(ROOT)),
                      "backup": str(backup.relative_to(ROOT)), "validation": "passed"}, indent=2))


if __name__ == "__main__":
    main()
