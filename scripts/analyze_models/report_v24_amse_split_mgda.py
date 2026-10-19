"""Report the prescribed di16/di32 AMSE-split MGDA endpoints; no selection.

Consumes the frozen arbitrary-checkpoint evaluation manifest. Exact32 and
spectral8 remain separate cohorts. All spectral comparisons use the accuracy
measurements saved on those same starts, and every width uses its own step 0.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_models import evaluate_v24_amse_regime as evaluation

BRANCHES = ("baseline", "full")
SPECTRAL_KEYS = ("truth_power", "baseline_power", "full_power",
                 "baseline_cross_power", "full_cross_power",
                 "baseline_error_power", "full_error_power")
WARNINGS = [
    "Both cohorts are previously inspected development data, not fresh confirmation.",
    "Step 2000 is primary; step 500 describes the trajectory. No checkpoint is selected.",
    "Each width is compared with its own initialization and one designated GC baseline. "
    "There is no matched fixed-alpha or MSE continuation arm; adaptive weighting is not isolated.",
    "Spherical observer A/C are physical per-field diagnostics, not the normalized, "
    "variable/level-weighted all-channel training objectives. Degree zero is retained in data; "
    "positive-degree spectral plots explicitly omit it.",
    "Absolute power is pooled before display ratios and coherence. Nonlinear A/C terms "
    "are computed per initialization before averaging. No powers are divided by truth "
    "power in the training objective.",
]


def ratio(value, reference):
    """An undefined zero-reference comparison stays missing."""
    return None if reference == 0 else float(value / reference)


def improvement(value, reference):
    value = ratio(value, reference)
    return None if value is None else 100 * (1 - value)


def vector(value, steps, label):
    result = np.asarray(value, dtype=float)
    evaluation.require(result.shape == (steps,) and np.isfinite(result).all(),
                       f"Invalid {label}")
    return result


def accuracy_summary(result, initial, canonical, *, branch="full"):
    """Pool raw GC loss, and squared RMSE, before forming aggregate ratios."""
    steps = int(result["target_steps"])
    loss = vector(result["original_graphcast_loss"][f"{branch}_per_step"], steps, "GC loss")
    initial_loss = vector(initial["original_graphcast_loss"]["full_per_step"], steps, "initial GC loss")
    gc = vector(canonical["original_graphcast_loss"]["baseline_per_step"], steps, "baseline GC loss")
    evaluation.require(steps >= 40, "Day 10 requires 40 six-hour leads")
    result_summary = {
        "loss_rollout": float(loss.mean()), "loss_day10": float(loss[39]),
        "gc_loss_rollout": float(gc.mean()), "gc_loss_day10": float(gc[39]),
        "loss_improvement_vs_gc_pct": improvement(loss.mean(), gc.mean()),
        "day10_loss_improvement_vs_gc_pct": improvement(loss[39], gc[39]),
        "loss_improvement_vs_initialization_pct": improvement(loss.mean(), initial_loss.mean()),
        "day10_loss_improvement_vs_initialization_pct": improvement(loss[39], initial_loss[39]),
    }
    fields = []
    for level, key in (("variable", "per_variable_per_step"), ("channel", "per_channel_per_step")):
        evaluation.require(set(result[key]) == set(initial[key]) == set(canonical[key]),
                           f"Different {level} coordinates")
        for name, values in result[key].items():
            current = vector(values[f"rmse_{branch}"], steps, name)
            start = vector(initial[key][name]["rmse_full"], steps, name)
            baseline = vector(canonical[key][name]["rmse_baseline"], steps, name)
            evaluation.require(np.all(current >= 0) and np.all(start >= 0) and np.all(baseline >= 0),
                               f"Negative physical RMSE: {name}")
            # Inputs are already pooled across starts by the evaluator. Never
            # average RMSE across starts or leads to create a rollout RMSE.
            fields.append({"kind": level, "field": name,
                           "rmse_rollout": float(np.sqrt(np.mean(current ** 2))),
                           "initialization_rmse_rollout": float(np.sqrt(np.mean(start ** 2))),
                           "gc_rmse_rollout": float(np.sqrt(np.mean(baseline ** 2))),
                           "rmse_day10": float(current[39]),
                           "rmse_day10_ratio_vs_initialization": ratio(current[39], start[39]),
                           "rmse_day10_ratio_vs_gc": ratio(current[39], baseline[39])})
    result_summary["physical_fields"] = fields
    return result_summary


def physical_components(power, truth, cross):
    """Unregularized observational A/C, with zero-coherence at one zero power.

    Both-zero power contributes zero regardless of the coherence convention.
    These physical diagnostics intentionally do not reproduce the training
    epsilon after physical-unit conversion or its variable/level weighting.
    """
    power, truth, cross = (np.asarray(value, dtype=float) for value in (power, truth, cross))
    evaluation.require(power.shape == truth.shape == cross.shape and
                       all(np.isfinite(value).all() for value in (power, truth, cross)) and
                       np.all(power >= 0) and np.all(truth >= 0), "Invalid spectral moments")
    denominator = np.sqrt(power) * np.sqrt(truth)
    coherence = np.divide(cross, denominator, out=np.zeros_like(cross), where=denominator > 0)
    # Transform roundoff can move a nearly perfect correlation outside [-1,1].
    coherence = np.clip(coherence, -1, 1)
    return (np.sqrt(power) - np.sqrt(truth)) ** 2, 2 * np.maximum(power, truth) * (1 - coherence)


def load_spherical(directory, spec):
    stamps = [evaluation._time(stamp) for stamp in spec["initialization_times"]]
    records, sources, coordinates = {}, [], None
    for path in sorted((Path(directory) / "spherical").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            stamp = evaluation._time(metadata["initialization_time"])
            evaluation.require(metadata.get("status") == "complete" and stamp in stamps and stamp not in records,
                               "Unexpected, incomplete or duplicate spherical initialization")
            for key in ("protocol_hash", "model_id", "checkpoint_sha256"):
                evaluation.require(metadata[key] == spec[key], f"Spherical {key} mismatch")
            evaluation.require(metadata["baseline_checkpoint_sha256"] == spec["baseline"]["checkpoint_sha256"],
                               "Spherical GC identity mismatch")
            current = {key: archive[key].tolist() for key in ("variables", "units", "degree", "lead_hours")}
            evaluation.require(current["lead_hours"] == list(range(6, 6 * spec["target_steps"] + 1, 6)),
                               "Spherical leads differ")
            if coordinates is None:
                coordinates = current
            evaluation.require(coordinates == current and current["degree"] == list(range(len(current["degree"]))),
                               "Spherical coordinates differ or degree zero is missing")
            arrays = {key: np.asarray(archive[key], dtype=float) for key in SPECTRAL_KEYS}
            expected = (spec["target_steps"], len(current["variables"]), len(current["degree"]))
            evaluation.require(all(value.shape == expected and np.isfinite(value).all() for value in arrays.values()),
                               "Invalid spherical moment arrays")
            # Keep discrete fitted/unresolved diagnostics distinct from the
            # continuous fitted spectral norm; neither difference substitutes
            # for the other definition of unresolved grid error.
            arrays.update({key: np.asarray(archive[key], dtype=float) for key in archive.files
                           if key.endswith(("_grid_mse", "_grid_power"))})
            evaluation.require(all(np.isfinite(value).all() for value in arrays.values()),
                               "Nonfinite spherical grid diagnostics")
            records[stamp] = arrays
        sources.append(evaluation.frozen_file(path))
    evaluation.require(set(records) == set(stamps), "Incomplete spherical observer outputs")
    stacked = {key: np.stack([records[stamp][key] for stamp in stamps]) for key in records[stamps[0]]}
    components = {}
    for branch in BRANCHES:
        amplitude, pattern = physical_components(stacked[f"{branch}_power"], stacked["truth_power"],
                                                 stacked[f"{branch}_cross_power"])
        components[f"{branch}_amplitude"] = amplitude.mean(axis=0)
        components[f"{branch}_pattern"] = pattern.mean(axis=0)
    return {**coordinates, "times": stamps, "sources": sources, "stacked": stacked,
            "pooled": {key: value.mean(axis=0) for key, value in stacked.items()}, "components": components}


def drift(actual, canonical):
    actual, canonical = np.asarray(actual, float), np.asarray(canonical, float)
    evaluation.require(actual.shape == canonical.shape, "Baseline audit coordinates differ")
    difference = actual - canonical
    return {"max_abs": float(np.max(np.abs(difference))),
            "relative_l2": ratio(float(np.linalg.norm(difference)), float(np.linalg.norm(canonical)))}


def training_records(training_root):
    """Read only canonical training histories, excluding shadow restart updates."""
    output = {}
    for family in ("di16", "di32"):
        path = Path(training_root) / family / "train_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        steps = [row["step"] for row in rows]
        evaluation.require(steps == list(range(1, 2001)), f"Expected canonical updates 1..2000: {path}")
        required = ("amplitude_loss", "pattern_loss", "gc_loss", "mgda_alpha", "update_norm")
        evaluation.require(all(all(key in row and np.isfinite(row[key]) for key in required) for row in rows),
                           f"Missing/nonfinite split-objective metrics: {path}")
        norms = np.asarray([row["update_norm"] for row in rows])
        median = float(np.median(norms))
        statistics = {}
        summary_keys = (*required, "amse_loss", "amplitude_gradient_norm", "pattern_gradient_norm",
                        "gradient_cosine", "relative_combined_gradient_norm")
        for key in summary_keys:
            if not all(key in row for row in rows):
                continue
            values = np.asarray([row[key] for row in rows])
            evaluation.require(np.isfinite(values).all(), f"Nonfinite training summary {path}/{key}")
            statistics[key] = {"first100_mean": float(values[:100].mean()),
                               "last100_mean": float(values[-100:].mean()),
                               "minimum": float(values.min()), "median": float(np.median(values)),
                               "maximum": float(values.max())}
        output[family] = {"source": evaluation.frozen_file(path), "steps": 2000,
                          "exact_zero_update_count": int(np.count_nonzero(norms == 0)),
                          "small_update_threshold": median * 1e-3,
                          "small_update_count": int(np.count_nonzero(norms <= median * 1e-3)),
                          "small_update_definition": "update_norm <= 0.001 * this run's median; descriptive only",
                          "statistics": statistics, "rows": rows}
    return output


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def overview_fields(variables):
    """Keep the spherical overview readable while saving every channel in NPZ."""
    if len(variables) <= 16:
        return list(variables)
    desired = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
               "mean_sea_level_pressure", "total_precipitation_6hr", "temperature_level850",
               "temperature_level500", "specific_humidity_level700", "specific_humidity_level500",
               "geopotential_level500", "geopotential_level850", "u_component_of_wind_level850",
               "v_component_of_wind_level850", "u_component_of_wind_level250",
               "v_component_of_wind_level250", "vertical_velocity_level500")
    selected = [name for name in desired if name in variables]
    return (selected + [name for name in variables if name not in selected])[:16]


def build_report(manifest_path, *, image_dir=None, training_root=None):
    """Write JSON, Markdown, CSV and figures; return the report directory."""
    manifest_path = Path(manifest_path)
    manifest = evaluation.load_manifest(manifest_path)
    evaluation.require(manifest["evaluation_role"] == "development", "Expected the development cohort")
    expected = {(family, step) for family in ("di16", "di32") for step in (0, 500, 2000)}
    evaluation.require({(task["family"], task["step"]) for task in manifest["task_specs"]} == expected and
                       len(manifest["task_specs"]) == 6 and
                       len({task["id"] for task in manifest["task_specs"]}) == 6,
                       "Expected both widths at 0, 500 and 2000 with distinct IDs")
    loaded, sources = {}, []
    for index, task in enumerate(manifest["task_specs"]):
        stages = {stage: evaluation.resolve_result(manifest_path, index, stage) for stage in ("exact", "spectral")}
        directory, spec, result, _ = stages["spectral"]
        zonal = evaluation.load_spectral_result(directory, spec, result)
        spherical = load_spherical(directory, spec)
        for folder, _, _, _ in stages.values():
            sources.extend(evaluation.frozen_file(folder / name) for name in ("specification.json", "evaluation.json"))
        sources.extend(zonal["sources"] + spherical["sources"])
        loaded[task["id"]] = {"task": task, "stages": stages, "zonal": zonal, "spherical": spherical}
    canonical_id = manifest["canonical_mse_id"]
    evaluation.require(canonical_id in loaded, "Missing designated GC evaluation")
    canonical = loaded[canonical_id]
    records, audit, accuracy_rows, rmse_rows, spectral_rows = [], [], [], [], []
    destination = Path(manifest["output_dir"]) / "report"
    destination.mkdir(parents=True, exist_ok=True)
    for identity, item in loaded.items():
        task = item["task"]
        initial_id = task.get("paired_initialization_id", f"{task['family']}_step00000")
        evaluation.require(initial_id in loaded and loaded[initial_id]["task"]["family"] == task["family"] and
                           loaded[initial_id]["task"]["step"] == 0, "Missing own-width initialization")
        initial = loaded[initial_id]
        record = {**task, "paired_initialization_id": initial_id, "accuracy": {}}
        for stage in ("exact", "spectral"):
            _, spec, result, _ = item["stages"][stage]
            _, baseline_spec, baseline_result, _ = canonical["stages"][stage]
            coordinate = "chosen_idx" if stage == "exact" else "initialization_times"
            evaluation.require(spec[coordinate] == baseline_spec[coordinate], f"Different {stage} cohort")
            summary = accuracy_summary(result, initial["stages"][stage][2], baseline_result)
            record["accuracy"][stage] = summary
            accuracy_rows.append({"id": identity, "cohort": stage,
                                  **{key: value for key, value in summary.items() if key != "physical_fields"}})
            audit.append({"id": identity, "cohort": stage, "metric": "exact_gc_loss",
                          **drift(result["original_graphcast_loss"]["baseline_per_step"],
                                  baseline_result["original_graphcast_loss"]["baseline_per_step"])})
            for kind, key in (("variable", "per_variable_per_step"), ("channel", "per_channel_per_step")):
                for name, values in result[key].items():
                    baseline = baseline_result[key][name]["rmse_baseline"]
                    start = initial["stages"][stage][2][key][name]["rmse_full"]
                    audit.append({"id": identity, "cohort": stage, "metric": f"{kind}/{name}/rmse",
                                  **drift(values["rmse_baseline"], baseline)})
                    for lead, (value, reference, gc) in enumerate(zip(values["rmse_full"], start, baseline, strict=True), 1):
                        rmse_rows.append({"id": identity, "cohort": stage, "kind": kind, "field": name,
                                          "lead_hours": 6 * lead, "rmse": value, "gc_rmse": gc,
                                          "initialization_rmse": reference, "ratio_vs_gc": ratio(value, gc),
                                          "ratio_vs_initialization": ratio(value, reference)})
        for domain in ("zonal", "spherical"):
            data, baseline = item[domain], canonical[domain]
            evaluation.require(data["times"] == baseline["times"] and data["variables"] == baseline["variables"],
                               f"Different {domain} coordinates")
            np.testing.assert_allclose(data["stacked"]["truth_power"], baseline["stacked"]["truth_power"], rtol=0, atol=0)
            for key in ("baseline_power", "baseline_cross_power", "baseline_error_power"):
                audit.append({"id": identity, "cohort": "spectral", "metric": f"{domain}/{key}",
                              **drift(data["pooled"][key], baseline["pooled"][key])})
            payload = {**data["pooled"], "variables": np.asarray(data["variables"]),
                       "coordinate": np.asarray(data["degree"] if domain == "spherical" else data["wavenumber"])}
            if domain == "spherical":
                payload.update(data["components"])
            np.savez_compressed(destination / f"{identity}_{domain}_pooled.npz", **payload)
        spherical = item["spherical"]
        for branch in BRANCHES:
            for vi, name in enumerate(spherical["variables"]):
                for li, hours in enumerate(spherical["lead_hours"]):
                    amplitude = spherical["components"][f"{branch}_amplitude"][li, vi]
                    pattern = spherical["components"][f"{branch}_pattern"][li, vi]
                    spectral_rows.append({"id": identity, "branch": branch, "field": name, "lead_hours": hours,
                                          "amplitude_all_degrees": float(amplitude.sum()),
                                          "pattern_all_degrees": float(pattern.sum()),
                                          "amplitude_positive_degrees": float(amplitude[1:].sum()),
                                          "pattern_positive_degrees": float(pattern[1:].sum())})
        record["zonal_selected_diagnostics"] = evaluation.feature_rows(item["zonal"])
        records.append(record)
    training = training_records(training_root) if training_root is not None else {}
    warnings = list(WARNINGS)
    if not training:
        warnings.append("Training history was not provided; optimizer diagnostics are unavailable.")
    if any(row["relative_l2"] is not None and row["relative_l2"] > 1e-5 for row in audit):
        warnings.append("Repeated GC calculations drift by more than 1e-5 relative L2 for some diagnostics; "
                        "see baseline_audit.csv. All displayed GC references use the designated run, never an average.")
    report = {"schema_version": 1, "format": "v24_amse_split_mgda_report_v1", "status": "complete",
              "manifest": evaluation.frozen_file(manifest_path), "sources": sources,
              "primary_endpoint": "step02000", "selection_allowed": False,
              "canonical_gc_id": canonical_id, "records": records, "warnings": warnings,
              "baseline_audit": audit,
              "training": {family: {key: value for key, value in values.items() if key != "rows"}
                           for family, values in training.items()},
              "accounting": {"canonical_optimizer_updates": 4000, "shadow_restart_updates": 100,
                             "forward_calibration_batches": 64,
                             "note": "Prescribed campaign totals; shadow/calibration are not counted as production learning."},
              "spectral_fields": canonical["spherical"]["variables"]}
    write_csv(destination / "accuracy_summary.csv", accuracy_rows)
    write_csv(destination / "rmse_rollouts.csv", rmse_rows)
    write_csv(destination / "spherical_components.csv", spectral_rows)
    write_csv(destination / "baseline_audit.csv", audit)
    write_csv(destination / "zonal_diagnostics.csv", [{"id": row["id"], **field}
              for row in records for field in row["zonal_selected_diagnostics"]])
    for family, values in training.items():
        write_csv(destination / f"{family}_training_metrics.csv", values["rows"])
    evaluation.write_json(destination / "split_mgda_report.json", report)
    if image_dir is not None:
        render(loaded, canonical_id, training, Path(image_dir), destination / "split_mgda_report.json")
    lines = ["# AMSE split MGDA: di16 and di32", "", "Native step 2000 is the primary endpoint; step 500 is intermediate.", "",
             "| Cohort | Checkpoint | GC loss, rollout | GC loss, day 10 | Loss reduction vs GC (%) | Loss reduction vs own initialization (%) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in accuracy_rows:
        numbers = ["undefined" if row[key] is None else f"{row[key]:.5f}" for key in
                   ("loss_rollout", "loss_day10", "loss_improvement_vs_gc_pct", "loss_improvement_vs_initialization_pct")]
        lines.append(f"| {row['cohort']} | {row['id']} | " + " | ".join(numbers) + " |")
    lines += ["", "Exact: 32 starts in 2022. Spectral: 8 starts in 2023, with accuracy on those same cases.", "",
              "All saved variables and channels have 40-lead RMSE curves in `rmse_rollouts.csv`. "
              "Rollout RMSE pools squared errors before taking a square root. "
              "`spherical_components.csv` retains both degree-zero-inclusive and anomaly-only A/C. "
              "Pooled NPZ files retain power, cross-power, fitted error and discrete projection diagnostics.", ""]
    if image_dir is not None:
        lines += [f"Figures: `{Path(image_dir).resolve()}` (absolute power first, ratios and coherence second).", ""]
    lines += ["Interpretation limits:", "", *[f"- {warning}" for warning in warnings], ""]
    (destination / "README.md").write_text("\n".join(lines))
    return destination


def render(loaded, canonical_id, training, image_dir, report_path):
    from scripts.analyze_models import plot_v24_amse_regime as plotting
    import matplotlib.pyplot as plt

    image_dir.mkdir(parents=True, exist_ok=True)
    plotting.style()
    figures = {"source": plotting.identity(report_path), "figures": [], "canonical_gc_id": canonical_id}
    canonical = loaded[canonical_id]
    labels = {identity: f"{item['task']['family']}, update {item['task']['step']}" for identity, item in loaded.items()}
    days = np.arange(1, 41) / 4
    for stage in ("exact", "spectral"):
        baseline = canonical["stages"][stage][2]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(days, baseline["original_graphcast_loss"]["baseline_per_step"], color="black", label="Original GC")
        for identity, item in loaded.items():
            ax.plot(days, item["stages"][stage][2]["original_graphcast_loss"]["full_per_step"], label=labels[identity])
        ax.set(xlabel="Forecast day", ylabel="Exact GraphCast loss", title=f"{stage}: original GC loss")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plotting.save(fig, f"{stage}_gc_loss", image_dir, figures, cohort=stage)
        for kind, key in (("variable", "per_variable_per_step"), ("channel", "per_channel_per_step")):
            names = list(baseline[key])
            for page, start in enumerate(range(0, len(names), 12), 1):
                selected = names[start:start + 12]
                fig, axes = plt.subplots(4, 3, figsize=(16, 14))
                for ax, name in zip(axes.flat, selected):
                    ax.plot(days, baseline[key][name]["rmse_baseline"], color="black", label="Original GC")
                    for identity, item in loaded.items():
                        ax.plot(days, item["stages"][stage][2][key][name]["rmse_full"], label=labels[identity], linewidth=.9)
                    ax.set(title=name.replace("_", " "), xlabel="Forecast day", ylabel="RMSE (physical units)")
                    ax.grid(alpha=.2)
                for ax in list(axes.flat)[len(selected):]:
                    ax.set_visible(False)
                axes.flat[0].legend(fontsize=6)
                fig.suptitle(f"{stage}: {kind} RMSE; pooled squared errors before square root")
                fig.tight_layout()
                plotting.save(fig, f"{stage}_{kind}_rmse_{page:02d}", image_dir, figures, cohort=stage, fields=selected)
    for domain in ("spherical", "zonal"):
        baseline = canonical[domain]
        coordinate = np.asarray(baseline["degree"] if domain == "spherical" else baseline["wavenumber"])
        positive = coordinate > 0
        for name in overview_fields(baseline["variables"]):
            vi = baseline["variables"].index(name)
            fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True)
            for column, day in enumerate((1, 3, 7, 10)):
                li = day * 4 - 1
                truth = baseline["pooled"]["truth_power"][li, vi, positive]
                axes[0, column].loglog(coordinate[positive], truth, color="black", label="Truth", linewidth=1.7)
                series = [("Original GC", baseline["pooled"], "baseline")]
                series += [(labels[identity], item[domain]["pooled"], "full") for identity, item in loaded.items()]
                for label, values, branch in series:
                    power = values[f"{branch}_power"][li, vi, positive]
                    cross = values[f"{branch}_cross_power"][li, vi, positive]
                    axes[0, column].loglog(coordinate[positive], power, label=label, linewidth=.9)
                    axes[1, column].semilogx(coordinate[positive], np.divide(power, truth,
                        out=np.full_like(power, np.nan), where=truth > 0), label=label, linewidth=.9)
                    denominator = np.sqrt(power * truth)
                    axes[2, column].semilogx(coordinate[positive], np.divide(cross, denominator,
                        out=np.full_like(cross, np.nan), where=denominator > 0), label=label, linewidth=.9)
                axes[0, column].set_title(f"Day {day}")
                axes[1, column].axhline(1, color="gray", linestyle=":")
                axes[2, column].set(xlabel="Spherical degree" if domain == "spherical" else "Zonal wavenumber", ylim=(-1.05, 1.05))
            for ax, label in zip(axes[:, 0], ("Absolute power (physical units squared)", "Power / truth power", "Coherence"), strict=True):
                ax.set_ylabel(label)
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(f"{name}: {domain} spectra on the same 8 starts; degree/wavenumber zero omitted")
            fig.tight_layout()
            plotting.save(fig, f"{domain}_{name}_spectra", image_dir, figures, field=name, cohort="spectral",
                          panel_order=["absolute_power", "power_ratio", "coherence"], degree_zero="retained in NPZ, omitted in plot")
    panels = (("Raw amplitude and pattern", ("amplitude_loss", "pattern_loss", "amse_loss")),
              ("Original GC loss (diagnostic only)", ("gc_loss",)),
              ("Normalized objectives", ("amplitude_loss_normalized", "pattern_loss_normalized")),
              ("MGDA alpha and gradient cosine", ("mgda_alpha", "gradient_cosine")),
              ("Objective gradient norms", ("amplitude_gradient_norm", "pattern_gradient_norm")),
              ("Actual parameter displacement", ("update_norm",)),
              ("Clipping and stalls", ("clip_factor", "stalled")),
              ("Spatial learning rate", ("learning_rate",)))
    for family, values in training.items():
        fig, axes = plt.subplots(4, 2, figsize=(13, 13))
        missing = []
        for ax, (title, keys) in zip(axes.flat, panels, strict=True):
            for key in keys:
                points = np.asarray([(row["step"], row[key]) for row in values["rows"] if key in row], dtype=float)
                if not points.size:
                    missing.append(key)
                    continue
                evaluation.require(np.isfinite(points).all(), f"Nonfinite training plot {family}/{key}")
                ax.plot(points[:, 0], points[:, 1], label=key, linewidth=.7)
            ax.set(title=title, xlabel="Completed optimizer updates")
            if ax.lines:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, "Diagnostic unavailable", transform=ax.transAxes, ha="center")
        fig.suptitle(f"{family}: canonical trajectory, excluding shadow restart")
        fig.tight_layout()
        plotting.save(fig, f"{family}_optimizer", image_dir, figures, source=values["source"], missing_metrics=missing)
    evaluation.write_json(image_dir / "figures_manifest.json", figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--training-root", type=Path)
    args = parser.parse_args()
    print(build_report(args.manifest, image_dir=args.image_dir, training_root=args.training_root))


if __name__ == "__main__":
    main()
