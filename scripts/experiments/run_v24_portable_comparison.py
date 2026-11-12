#!/usr/bin/env python3
"""Portable raw/SWA MSE and spherical-spectrum comparisons; never submit jobs."""
from __future__ import annotations

import argparse
from dataclasses import fields
import glob
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments.frozen_runtime import freeze_source, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, unseal
from scripts.experiments.run_v24_raw_checkpoint_pilot import PilotObserver, pool_spectra, POWER_KEYS
from scripts.analyze_models.report_v24_raw_checkpoint_pilot import (
    divide, finite_difference, power_error, paired_summary, write_csv,
)

FORMAT = "v24_portable_comparison_v1"
DRIVER = "scripts/experiments/run_v24_portable_comparison.py"
WRAPPER = "scripts/experiments/v24_portable_comparison.sh"
FULL_GC_POINTS = (500, 1000, 1500, 2000, 2500)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dates_from(cohort, year):
    dates = [row["initialization_time"] for row in cohort["initializations"]]
    require(dates and dates == sorted(set(dates)), "Cohort dates must be sorted and unique")
    require(all(value.startswith(f"{year}-") for value in dates), "Cohort year differs")
    return dates


def evaluation_data_record(path):
    path = Path(path)
    if path.is_file():
        return {"path": str(path), "kind": "file", "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns}
    require(path.is_dir(), "Evaluation data is neither a file nor a directory")
    metadata = [pinned(path / name) for name in (".zmetadata", ".zgroup", ".zattrs", "zarr.json")
                if (path / name).is_file()]
    require(metadata, "Directory evaluation data must be a Zarr store with root metadata")
    return {"path": str(path), "kind": "zarr", "metadata": metadata,
            "verification_scope": "Root metadata identity; array data must remain immutable across evaluations"}


def verify_evaluation_data(item):
    path = Path(item["path"])
    if item["kind"] == "file":
        require(path.is_file() and path.stat().st_size == item["bytes"]
                and path.stat().st_mtime_ns == item["mtime_ns"], "Evaluation data file changed")
    else:
        require(item["kind"] == "zarr" and path.is_dir(), "Evaluation Zarr store disappeared")
        for metadata in item["metadata"]:
            check(metadata)


def model_specs(campaign, manifest):
    """Checkpoint counters are stored/main counters; total counters are explicit."""
    specs = []
    for task in manifest["tasks"]:
        family = task["model_family"]
        require(family in ("gc_mamba", "full_gc"), "Unsupported model family")
        steps = [500, 1000, 1500, 2000] if family == "gc_mamba" else [1000, 1500, 2000, 2500]
        run = campaign / "runs" / task["id"]
        spec = {"task": task, "family": family, "run": run, "steps": steps,
                "final_step": steps[-1], "config": task["config"],
                "proof": campaign / "checks" / f"{task['id']}_total00002500.json"}
        specs.append(spec)
    return specs


def selected_full_gc_steps(development, tasks):
    """Select only on complete 2022 scores; exact ties choose the earlier raw step."""
    require(development["status"] == "complete" and development["year"] == 2022,
            "Annual selection requires a completed 2022 report")
    selected = {}
    for task in tasks:
        rows = [row for row in development["summary"]
                if row.get("training_run_id") == task["id"] and row["checkpoint_kind"] == "raw"]
        require(sorted(row["checkpoint_step"] for row in rows) == list(FULL_GC_POINTS),
                "Development report lacks one of the five raw full-GC checkpoints")
        require(all(np.isfinite(row["exact_gc_mse_rollout"]) for row in rows), "Nonfinite development score")
        best = min(rows, key=lambda row: (row["exact_gc_mse_rollout"], row["checkpoint_step"]))
        selected[task["id"]] = sorted({2500, best["checkpoint_step"]})
    return selected


def build_swa(root, spec):
    import jax
    from src.models.mamba.v24_Ilya.checkpoint import build_v24_Ilya_swa, load_v24_Ilya_checkpoint
    sources = [pinned(spec["run"] / "checkpoints" / f"checkpoint_step{step:08d}.pkl") for step in spec["steps"]]
    target = root / "swa" / f"{spec['task']['id']}.pkl"
    payload = build_v24_Ilya_swa([Path(row["path"]) for row in sources], spec["steps"], target,
                               residual_state_policy="zero")
    require(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(payload["residual_params"])),
            "Nonfinite SWA parameters")
    require(load_v24_Ilya_checkpoint(target).model_family == spec["family"], "SWA model family differs")
    for source in sources:
        check(source)
    seal(root / "checks" / f"swa_{spec['task']['id']}.json", {
        "status": "complete", "sources": sources, "source_steps": spec["steps"],
        "source_total_updates": [1000, 1500, 2000, 2500], "checkpoint": pinned(target)})
    return pinned(target)


def prepare(root, package, campaigns, data_path, cohort_path, year, *, development_report=None,
            m1_raw=None, m1_swa=None, m1_config=None):
    root, package = Path(root).resolve(), Path(package).resolve()
    require(not root.exists(), "Use a new comparison output directory")
    require(campaigns, "At least one completed training campaign is required")
    cohort_path = Path(cohort_path).resolve()
    dates = dates_from(read(cohort_path), year)
    data_path = str(Path(data_path).expanduser().absolute())
    data_files = sorted(glob.glob(data_path))
    require(data_files, "Evaluation data path/glob matched no files")
    require(not m1_swa or m1_raw, "Supply M1 raw alongside M1 SWA")
    require(not m1_raw or m1_config, "M1 evaluation requires its architecture configuration")
    manifests, specs, inputs = [], [], [pinned(cohort_path)]
    for value in campaigns:
        campaign = Path(value).resolve()
        manifest = unseal(campaign / "manifest.json")
        require(Path(manifest["experiment"]) == campaign, "Training campaign was relocated after preparation")
        require(verify_source(manifest["source_root"])["source_digest"] == manifest["source_digest"],
                "Training source changed")
        manifests.append(pinned(campaign / "manifest.json"))
        for spec in model_specs(campaign, manifest):
            proof = unseal(spec["proof"])
            require(proof["status"] == "complete" and proof["manifest_hash"] == manifest["record_hash"]
                    and proof["task_id"] == spec["task"]["id"] and proof["completed_steps"] == spec["final_step"]
                    and proof["total_training_updates"] == 2500, "Incomplete/mismatched final training proof")
            require(Path(proof["checkpoint"]["path"]) == spec["run"] / "checkpoints" /
                    f"checkpoint_step{spec['final_step']:08d}.pkl", "Final checkpoint location differs")
            check(proof["checkpoint"]); check(spec["config"])
            inputs.extend([pinned(spec["proof"]), proof["checkpoint"], spec["config"]])
            specs.append(spec)
    require(len({spec["task"]["id"] for spec in specs}) == len(specs), "Duplicate training identity")
    full_gc_tasks = [spec["task"] for spec in specs if spec["family"] == "full_gc"]
    selected = {task["id"]: list(FULL_GC_POINTS) for task in full_gc_tasks}
    if year == 2023 and full_gc_tasks:
        require(development_report is not None, "2023 full-GC selection requires --development-report")
        development = unseal(Path(development_report).resolve())
        require(development["training_campaigns"] == manifests, "Development used different training campaigns")
        selected = selected_full_gc_steps(development, full_gc_tasks)
        inputs.append(pinned(development_report))
    configs = [read(spec["config"]["path"]) for spec in specs]
    baseline = pinned(configs[0]["data"]["baseline_checkpoint"])
    stats_dir = Path(configs[0]["data"]["stats_dir"]).resolve()
    stats = [pinned(stats_dir / name) for name in ("mean_by_level.nc", "stddev_by_level.nc", "diffs_stddev_by_level.nc")]
    for config in configs:
        require(pinned(config["data"]["baseline_checkpoint"])["sha256"] == baseline["sha256"], "Different public GC initializers")
        require([pinned(Path(config["data"]["stats_dir"]) / Path(row["path"]).name)["sha256"] for row in stats]
                == [row["sha256"] for row in stats], "Normalization differs between models")
    inputs += manifests + [baseline, *stats]
    for directory in ("swa", "checks", "evaluation", "report", "logs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    models = []
    for spec in specs:
        task = spec["task"]
        label = f"GC–Mamba λ={task['lambda']:g}" if spec["family"] == "gc_mamba" else f"Full GC K{task['horizon']}"
        points = [spec["final_step"]] if spec["family"] == "gc_mamba" else selected[task["id"]]
        for point in [*points, "swa"]:
            is_swa = point == "swa"
            step = spec["final_step"] if is_swa else point
            total_updates = step + (500 if spec["family"] == "gc_mamba" else 0)
            point_label = "SWA" if is_swa else f"raw {total_updates}"
            checkpoint = build_swa(root, spec) if is_swa else pinned(spec["run"] / "checkpoints" / f"checkpoint_step{step:08d}.pkl")
            model = {"id": f"{task['id']}_{'swa' if is_swa else f'raw{step:05d}'}",
                "label": f"{label} {point_label}",
                "model_family": spec["family"], "lambda": task["lambda"], "horizon": task.get("horizon", 20),
                "step": step, "checkpoint_kind": "swa" if is_swa else "raw", "checkpoint": checkpoint,
                "training_run_id": task["id"], "run_config": spec["config"],
                "training_updates": total_updates}
            if is_swa or step == spec["final_step"]:
                model["comparison_pair"] = task["id"]
            if is_swa:
                model["swa_steps"] = spec["steps"]
                model["swa_total_updates"] = [1000, 1500, 2000, 2500]
            models.append(model)
    if m1_raw:
        m1_models = []
        for kind, path in (("raw", m1_raw), ("swa", m1_swa)):
            if path is None:
                continue
            row = {"id": f"M1_{kind}", "label": f"M1 {kind}", "model_family": "gc_mamba", "lambda": None,
                "horizon": 20, "step": 2000, "checkpoint_kind": kind, "training_updates": 2500,
                "training_run_id": "M1", "checkpoint": pinned(Path(path).resolve()),
                "run_config": pinned(Path(m1_config).resolve())}
            if m1_swa:
                row["comparison_pair"] = "M1"
                if kind == "swa":
                    row["swa_steps"] = [500, 1000, 1500, 2000]
            m1_models.append(row)
        models = m1_models + models
    for model in models:
        inputs.extend([model["checkpoint"], model["run_config"]])
    source = freeze_source(package, root / "source")
    return seal(root / "manifest.json", {"format": FORMAT, "experiment": str(root), "workspace": str(package),
        "source_root": str(root / "source"), "source_digest": source["source_digest"],
        "inputs": inputs, "models": models, "training_campaigns": manifests, "baseline": baseline,
        "stats_dir": str(stats_dir), "cohort": pinned(cohort_path), "year": year, "count": len(dates),
        "target_steps": 40, "development_data": data_path,
        "data_files": [evaluation_data_record(path) for path in data_files],
        "selected_full_gc_steps": selected,
        "selection": "2022: all five full-GC raw checkpoints plus SWA; 2023: final raw/SWA plus lowest-2022-MSE raw (earlier step breaks ties). GC–Mamba: prescribed final raw/SWA.",
        "limits": "Fixed checkpoints and one training seed. Full-GC and GC–Mamba use different documented training recipes. Prior 2023 exposure must be disclosed; forecast-sample intervals do not measure training variability. Finite worsening and numerical drift remain warnings."})


def verify(root, *, frozen=False):
    manifest = unseal(root / "manifest.json")
    require(manifest["format"] == FORMAT and Path(manifest["experiment"]) == root, "Comparison identity differs")
    require(not frozen or ROOT == Path(manifest["source_root"]), "Execute the frozen comparison source")
    require(verify_source(manifest["source_root"])["source_digest"] == manifest["source_digest"], "Frozen source changed")
    for item in manifest["inputs"]:
        check(item)
    for item in manifest["data_files"]:
        verify_evaluation_data(item)
    return manifest


def evaluate(root, manifest, task):
    from src.models.graphcast.training.core.model import load_stats
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from scripts.experiments.run_v24_scalar_lambda import gpu_required
    gpu_required()
    model = manifest["models"][task]
    dest = root / "evaluation" / model["id"]
    require(not dest.exists(), "Evaluation output already exists; retain it and use a new comparison for retries")
    dest.mkdir()
    allowed = {field.name for field in fields(V24IlyaEvalConfig)}
    architecture = {k: v for k, v in read(model["run_config"]["path"])["architecture"].items() if k in allowed}
    observer = PilotObserver(dest / "spectra", load_stats(Path(manifest["stats_dir"]))["diffs_stddev_by_level"],
        {"campaign_hash": manifest["record_hash"], "model_id": model["id"], "checkpoint_sha256": model["checkpoint"]["sha256"],
         "cohort_sha256": manifest["cohort"]["sha256"], "source_digest": manifest["source_digest"]})
    cfg = V24IlyaEvalConfig(**architecture, ckpt=Path(model["checkpoint"]["path"]), out_json=dest / "evaluation.json",
        model_id=model["id"], model_family=model["model_family"], ckpt_in=Path(manifest["baseline"]["path"]),
        stats_dir=Path(manifest["stats_dir"]), data_path=manifest["development_data"],
        initialization_manifest=Path(manifest["cohort"]["path"]), target_steps=40, n_samples=manifest["count"],
        val_year=manifest["year"], train_start_year=2015, train_end_year=2021, warmup_steps=0,
        eval_mode="cold_full", residual_state_init="zero", omit_rms_bias=True)
    started = time.monotonic()
    result = evaluate_v24_Ilya(cfg, prediction_observer=observer)
    dates = dates_from(read(manifest["cohort"]["path"]), manifest["year"])
    require(result["evaluation_status"] == "complete" and result["metrics_complete"]
            and result["remaining_samples"] == 0 and result["evaluated_samples"] == len(dates)
            and result["initialization_manifest_sha256"] == manifest["cohort"]["sha256"]
            and result["global_initialization_times"] == dates, "Incomplete/mismatched exact-MSE coverage")
    require(all(np.isfinite(np.asarray(result["original_graphcast_loss"][key])).all() for key in
                ("baseline_rollout", "full_rollout", "baseline_per_step", "full_per_step")), "Nonfinite exact MSE")
    pooled = pool_spectra(dest, manifest, model)
    check(model["checkpoint"])
    return seal(dest / "complete.json", {"status": "complete", "manifest_hash": manifest["record_hash"],
        "checkpoint": model["checkpoint"], "evaluation": pinned(cfg.out_json), "pooled_spectra": pinned(pooled),
        "count": len(dates), "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2})


def load_results(root, manifest):
    items, layout = {}, None
    dates = dates_from(read(manifest["cohort"]["path"]), manifest["year"])
    for model in manifest["models"]:
        record = unseal(root / "evaluation" / model["id"] / "complete.json")
        require(record["status"] == "complete" and record["manifest_hash"] == manifest["record_hash"]
                and record["checkpoint"] == model["checkpoint"] and record["count"] == len(dates), "Evaluation completion differs")
        for key in ("checkpoint", "evaluation", "pooled_spectra"):
            check(record[key])
        result = read(record["evaluation"]["path"])
        per_start = result["original_graphcast_loss_per_initialization"]
        require(result["model_id"] == model["id"] and result["evaluation_status"] == "complete"
                and result["metrics_complete"] and result["remaining_samples"] == 0
                and result["evaluated_samples"] == len(dates)
                and result["initialization_manifest_sha256"] == manifest["cohort"]["sha256"]
                and [row["initialization_time"] for row in per_start] == dates
                and all(row["lead_hours"] == list(range(6, 241, 6)) for row in per_start), "MSE date/lead coverage differs")
        with np.load(record["pooled_spectra"]["path"], allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            require(metadata["campaign_hash"] == manifest["record_hash"] and metadata["count"] == len(dates)
                    and metadata["cohort_sha256"] == manifest["cohort"]["sha256"]
                    and metadata["checkpoint_sha256"] == model["checkpoint"]["sha256"]
                    and metadata["source_digest"] == manifest["source_digest"]
                    and metadata["initialization_ids"] == [row["initialization_id"] for row in per_start]
                    and metadata["branches"] == ["public_gc", model["id"]]
                    and metadata["lead_hours"] == list(range(6, 241, 6)), "Spectrum identities differ")
            layout = layout or metadata["layout"]
            require(layout == metadata["layout"] and len(layout["channels"]) == 83, "Spectrum layout differs")
            powers = {key: np.asarray(saved[key], dtype=float) for key in POWER_KEYS}
        for key, value in powers.items():
            shape = (40, 83, 180) if key == "truth_power" else (40, 2, 83, 180)
            require(value.shape == shape and np.isfinite(value).all(), "Invalid spectrum shape/values")
            require(key == "cross_power" or (value >= 0).all(), "Negative physical power")
        loss = {branch: np.asarray(result["original_graphcast_loss"][branch + "_per_step"], dtype=float)
                for branch in ("baseline", "full")}
        loss.update({branch + "_rollout": float(result["original_graphcast_loss"][branch + "_rollout"])
                     for branch in ("baseline", "full")})
        require(all(np.isfinite(value).all() and (np.asarray(value) >= 0).all() for value in loss.values()), "Invalid exact losses")
        items[model["id"]] = {"model": model, "record": record, "result": result, "power": powers, "loss": loss}
    return items, layout, dates


def annual_intervals(items, dates, canonical):
    from scripts.analyze_models.merge_v24_2023_eval import cadence_block_counts
    counts = cadence_block_counts(dates, stride_days=3, block_starts=5, replicates=2000, seed=20261112)
    baseline = np.asarray([row["baseline_per_step"] for row in items[canonical]["result"]["original_graphcast_loss_per_initialization"]])
    values = {name: np.asarray([row["full_per_step"] for row in item["result"]["original_graphcast_loss_per_initialization"]])
              for name, item in items.items()}
    references = {"public_gc": baseline, **{name: value for name, value in values.items() if name.startswith("M1_")}}
    rows = []
    for name, value in values.items():
        for reference, other in references.items():
            if reference == name:
                continue
            for scope, selected in (("rollout", slice(None)), ("day10", slice(-1, None))):
                a, b, denominator = value[:, selected].mean(1), other[:, selected].mean(1), baseline[:, selected].mean(1)
                draws = 100 * (counts @ (b - a)) / (counts @ denominator)
                low, high = np.quantile(draws, [.025, .975])
                rows.append({"model": name, "reference": reference, "scope": scope,
                    "advantage_pp_of_gc_loss": float(100 * (b.mean() - a.mean()) / denominator.mean()),
                    "ci95_low": float(low), "ci95_high": float(high)})
    return {"method": "Paired non-wrapping moving-block percentile bootstrap", "block_starts": 5,
            "stride_days": 3, "replicates": 2000, "seed": 20261112,
            "scope": "Forecast-sample uncertainty for fixed checkpoints, not training-seed uncertainty", "contrasts": rows}


def figures(root, manifest, series, items, channels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = root / "report" / "figures"
    output.mkdir(exist_ok=True)
    files = []
    references = [name for name in ("public_gc", "M1_raw", "M1_swa") if name in series]
    groups = {}
    for model in manifest["models"]:
        if model["id"].startswith("M1_"):
            continue
        group = "gc_mamba" if model["model_family"] == "gc_mamba" else f"full_gc_k{model['horizon']:02d}"
        groups.setdefault(group, []).append(model["id"])
    for group, members in groups.items():
        identities = references + members
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for identity in identities:
            item = series[identity]
            dashed = item.get("checkpoint_kind") == "swa"
            axes[0].plot(np.arange(1, 41) / 4, item["loss"], label=item["label"], linestyle="--" if dashed else "-")
            row = item["summary"]
            axes[1].scatter(row["power_error_rollout_pct"], row["exact_gc_mse_rollout"], label=item["label"], marker="s" if dashed else "o")
        axes[0].set(xlabel="Forecast lead (days)", ylabel="Exact GC weighted MSE")
        axes[1].set(xlabel="Spherical power mismatch (%)", ylabel="Exact GC weighted MSE")
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle(f"{group}: {manifest['count']} starts in {manifest['year']}")
        path = output / f"{group}_mse_power.png"; fig.savefig(path, dpi=150); plt.close(fig); files.append(str(path))
        for kind in ("forecast_power", "power_ratio", "error_power", "spectral_correlation"):
            fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
            for row, channel in enumerate((("2m_temperature", None), ("geopotential", 500), ("u_component_of_wind", 850))):
                ci = channels.index(channel)
                for col, lead in enumerate((19, 39)):
                    ax = axes[row, col]
                    truth = series["public_gc"]["truth"][lead, ci, 1:]
                    if kind == "forecast_power":
                        ax.loglog(np.arange(1, 180), truth, color="black", label="ERA5")
                    elif kind in ("power_ratio", "spectral_correlation"):
                        ax.axhline(1, color="black", linewidth=.7)
                    for identity in identities:
                        item = series[identity]
                        value = item["power"][lead, ci, 1:]
                        if kind == "power_ratio":
                            value = divide(value, truth)
                        elif kind == "error_power":
                            value = item["error_power"][lead, ci, 1:]
                        elif kind == "spectral_correlation":
                            value = divide(item["cross_power"][lead, ci, 1:], np.sqrt(value * truth))
                        ax.plot(np.arange(1, 180), value, label=item["label"], linestyle="--" if item.get("checkpoint_kind") == "swa" else "-")
                    ax.set_xscale("log")
                    if kind in ("forecast_power", "error_power"):
                        ax.set_yscale("log")
                    ax.set(title=f"{channel[0]} {channel[1] or ''}, day {(lead + 1) / 4:g}", xlabel="Spherical degree", ylabel=kind.replace("_", " "))
                    ax.grid(alpha=.2)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=3, fontsize=7)
            path = output / f"{group}_{kind}.png"; fig.savefig(path, dpi=150); plt.close(fig); files.append(str(path))
    return files


def report(root, manifest):
    items, layout, dates = load_results(root, manifest)
    canonical = manifest["models"][0]["id"]
    reference = items[canonical]
    truth = reference["power"]["truth_power"]
    mask = np.logical_and.reduce([item["power"]["truth_power"][..., 1:] > 0 for item in items.values()])
    require(mask.any() and mask[-1].any(), "No positive truth-power bins")
    baseline, baseline_rollout = reference["loss"]["baseline"], reference["loss"]["baseline_rollout"]
    require((baseline > 0).all() and baseline_rollout > 0, "Nonpositive public-GC denominator")
    spread = max(abs(item["loss"]["baseline_rollout"] - baseline_rollout) for item in items.values())
    day10_spread = max(abs(item["loss"]["baseline"][-1] - baseline[-1]) for item in items.values())
    series = {"public_gc": {"label": "Public DeepMind GC", "loss": baseline, "rollout": baseline_rollout,
        "truth": truth, "power": reference["power"]["prediction_power"][:, 0],
        "error_power": reference["power"]["error_power"][:, 0], "cross_power": reference["power"]["cross_power"][:, 0]}}
    diagnostics = []
    for identity, item in items.items():
        series[identity] = {**item["model"], "loss": item["loss"]["full"], "rollout": item["loss"]["full_rollout"],
            "truth": truth, "power": item["power"]["prediction_power"][:, 1],
            "error_power": item["power"]["error_power"][:, 1], "cross_power": item["power"]["cross_power"][:, 1]}
        diagnostics += [finite_difference(identity, "truth_power", item["power"]["truth_power"], truth),
            finite_difference(identity, "public_gc_power", item["power"]["prediction_power"][:, 0], series["public_gc"]["power"]),
            finite_difference(identity, "public_gc_loss", item["loss"]["baseline"], baseline)]
    summary, curves = [], []
    for identity, item in series.items():
        mismatch, per_lead = power_error(item["power"], truth, mask)
        row = {"model": identity, "label": item["label"], "model_family": item.get("model_family", "reference"),
            "training_run_id": item.get("training_run_id"), "lambda": item.get("lambda"), "horizon": item.get("horizon"),
            "checkpoint_kind": item.get("checkpoint_kind"), "checkpoint_step": item.get("step"),
            "training_updates": item.get("training_updates"), "exact_gc_mse_rollout": item["rollout"],
            "exact_gc_mse_day10": float(item["loss"][-1]), "mse_reduction_rollout_pct": 100 * (1 - item["rollout"] / baseline_rollout),
            "mse_reduction_day10_pct": float(100 * (1 - item["loss"][-1] / baseline[-1])),
            "power_error_rollout_pct": mismatch, "power_error_day10_pct": float(per_lead[-1]),
            "rollout_decrease_exceeds_baseline_repeat_spread": bool(baseline_rollout - item["rollout"] > spread),
            "day10_decrease_exceeds_baseline_repeat_spread": bool(baseline[-1] - item["loss"][-1] > day10_spread)}
        item["summary"] = row; summary.append(row)
        for lead in range(40):
            curves.append({"model": identity, "lead_days": (lead + 1) / 4, "exact_gc_mse": float(item["loss"][lead]),
                           "power_error_pct": float(per_lead[lead]) if np.isfinite(per_lead[lead]) else None})
    destination = root / "report"
    write_csv(destination / "summary.csv", summary); write_csv(destination / "lead_curves.csv", curves)
    pairs = paired_summary(manifest["models"], series, spread, day10_spread)
    if pairs:
        write_csv(destination / "raw_vs_swa.csv", pairs)
    intervals = annual_intervals(items, dates, canonical) if manifest["year"] == 2023 else None
    if intervals:
        write_csv(destination / "paired_intervals.csv", intervals["contrasts"])
    generated = figures(root, manifest, series, items, [tuple(c) for c in layout["channels"]])
    payload = {"status": "complete", "year": manifest["year"], "count": len(dates), "manifest_hash": manifest["record_hash"],
        "training_campaigns": manifest["training_campaigns"], "summary": summary, "raw_vs_swa": pairs,
        "canonical_baseline": f"Public GC branch of {canonical}", "baseline_repeat_spread_rollout": spread,
        "baseline_repeat_spread_day10": float(day10_spread), "numerical_reference_differences": diagnostics,
        "warnings": [row for row in diagnostics if row["exceeds_reference"]], "paired_intervals": intervals,
        "excluded_zero_truth_bins": int(mask.size - mask.sum()),
        "power_definition": "Powers pooled across starts first; 100*abs(Pforecast/Ptruth-1), equally averaged over 83 channels, degrees1–179 and leads; one common positive-truth mask.",
        "limits": manifest["limits"], "figures": generated}
    seal(destination / "report.json", payload)
    lines = [f"# Portable comparison: {len(dates)} starts in {manifest['year']}", "",
        "40 six-hour leads. Exact original GraphCast weighted MSE and physical spherical T179 spectra.", "",
        "| Model | Rollout MSE | Reduction vs public GC | Power mismatch | Day-10 MSE |",
        "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {row['label']} | {row['exact_gc_mse_rollout']:.6f} | {row['mse_reduction_rollout_pct']:+.3f}% | {row['power_error_rollout_pct']:.3f}% | {row['exact_gc_mse_day10']:.6f} |" for row in summary]
    lines += ["", payload["power_definition"], "", manifest["limits"], "",
        f"Public-GC repeat spread: rollout {spread:.6g}; day10 {day10_spread:.6g}. These are warnings, not gates or model-noise estimates.", "",
        "Raw/SWA differences are in `raw_vs_swa.csv`; spectra retain forecast, truth, cross and error powers for every channel/lead."]
    if intervals:
        lines += ["", "`paired_intervals.csv` contains paired 95% date-block intervals against public GC and available M1 references; positive differences favor the named model."]
    for path in generated:
        lines += ["", f"![{Path(path).stem}](figures/{Path(path).name})"]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    return {"status": "complete", "models": len(items), "report": str(destination / "report.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan", "evaluate", "report"))
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--package-root", type=Path, default=ROOT)
    parser.add_argument("--gcm-campaign", type=Path)
    parser.add_argument("--full-gc-campaign", type=Path)
    parser.add_argument("--data-path")
    parser.add_argument("--initialization-manifest", type=Path)
    parser.add_argument("--year", type=int, choices=(2022, 2023), default=2022)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--m1-raw", type=Path); parser.add_argument("--m1-swa", type=Path)
    parser.add_argument("--m1-config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args(); root = args.experiment.resolve()
    if args.stage == "prepare":
        require(args.data_path and args.initialization_manifest, "Prepare requires data and initialization manifest")
        manifest = prepare(root, args.package_root, [p for p in (args.gcm_campaign, args.full_gc_campaign) if p],
            args.data_path, args.initialization_manifest, args.year, development_report=args.development_report,
            m1_raw=args.m1_raw, m1_swa=args.m1_swa, m1_config=args.m1_config)
        result = {"status": "prepared", "models": len(manifest["models"]), "record_hash": manifest["record_hash"]}
    else:
        manifest = verify(root, frozen=args.stage in ("evaluate", "report"))
        if args.stage == "plan":
            result = {"array": f"0-{len(manifest['models']) - 1}", "evaluate": ["bash", str(Path(manifest["source_root"]) / WRAPPER), "evaluate", str(root), "<task>"],
                      "report": ["bash", str(Path(manifest["source_root"]) / WRAPPER), "report", str(root), "0"]}
        elif args.stage == "evaluate":
            require(0 <= args.task < len(manifest["models"]), "Invalid task index")
            result = evaluate(root, manifest, args.task)
        else:
            result = report(root, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
