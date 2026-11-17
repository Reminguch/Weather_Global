#!/usr/bin/env python3
"""Raw matched reset/reset spectra on the existing 92-date plotting cohort.

No training, filter fitting, checkpoint selection, or changes to computational
source. Only the explicit submit stage submits jobs. Reports refresh the seven
literature-style workspace figures after every required forecast is complete.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments import run_v24_smoothing as original
from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, frozen_command, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/run_v24_matched_reset_spectra.py"
WRAPPER = "scripts/experiments/v24_matched_reset_spectra.slurm"
PLOTTER = "scripts/analyze_models/plot_v24_literature_spectral_comparison_20261113.py"
OVERLAYS = (DRIVER, WRAPPER, "src/analysis/raw_spectral_observer.py", PLOTTER)
PARENT = "artifacts/evaluations/v24_optimal_smoothing_v4_20261112"
MATCHED = "artifacts/evaluations/v24_matched_reset_2023_20261018"
LITERATURE = "plots/analyze_models/data/resolution_eval/v24_literature_spectral_comparison_20261113/summary.json"
DEFAULT_ROOT = "artifacts/evaluations/v24_matched_reset_spectra_20261113"
RESET_SHA256 = "db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6"
LEADS = list(range(6, 241, 6))


def copy_pin(source, destination):
    before = pinned(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    check(before)
    result = pinned(destination)
    if result["sha256"] != before["sha256"]:
        raise ValueError("Reference changed while copying")
    return result


def freeze_extension(workspace, parent, destination):
    if destination.exists():
        raise FileExistsError(destination)
    previous = verify_source(parent)
    recorded = {}
    for relative in sorted(set(previous["files"]) | set(OVERLAYS)):
        source = (workspace if relative in OVERLAYS else parent) / relative
        proof = copy_pin(source, destination / relative)
        recorded[relative] = proof["sha256"]
    verify_source(parent)
    write(destination / "snapshot.json", {
        "format": FORMAT, "workspace": str(workspace), "files": recorded,
        "source_digest": _digest(recorded), "parent_snapshot": pinned(parent / "snapshot.json"),
        "overlays": list(OVERLAYS),
        "forecast_computation": "All inherited computational source is byte-identical to optimal-filter campaign"})
    return verify_source(destination)


def prepare(root, workspace, parent_root):
    root, workspace, parent_root = [Path(p).resolve() for p in (root, workspace, parent_root)]
    if root.exists():
        raise FileExistsError("Prepare into a new reset spectral campaign directory")
    parent = unseal(parent_root / "manifest.json")
    if parent["format"] != "v24_optimal_smoothing_campaign_v1":
        raise ValueError("Expected the original optimal-filter campaign")
    parent_source = verify_source(parent["source_root"])
    if parent_source["source_digest"] != parent["source_digest"]:
        raise ValueError("Parent computational source differs")
    matched_root = workspace / MATCHED
    protocol = read(matched_root / "protocol.json")
    if (protocol["checkpoint"]["sha256"] != RESET_SHA256 or protocol["model_id"] != "R_matched"
            or not protocol["reset_state_every_step"]
            or protocol["training_state_policy"] != "reset_every_anchor"
            or protocol["architecture"] != parent["architecture"]):
        raise ValueError("Expected the main architecture-matched reset/reset SWA")
    check(protocol["checkpoint"])
    literature = read(workspace / LITERATURE)
    ids = literature["initialization_ids"]
    annual = read(parent["cohorts"]["evaluate"]["path"])
    annual_ids = original.expected_ids(annual)
    if len(ids) != 92 or len(set(ids)) != 92 or ids != annual_ids[:92]:
        raise ValueError("Expected the pinned 92-date January1–October1 plotting cohort")
    root.mkdir(parents=True)
    for name in ("logs", "checks", "submissions", "manifests", "references", "evaluate", "report"):
        (root / name).mkdir()
    # Preserve mutable plotting metadata before the completion report refreshes it.
    references = {
        "literature_summary": copy_pin(workspace / LITERATURE, root / "references/literature_summary.json"),
        "matched_protocol": copy_pin(matched_root / "protocol.json", root / "references/matched_protocol.json"),
        "parent_manifest": copy_pin(parent_root / "manifest.json", root / "references/parent_manifest.json")}
    canonical = [copy_pin(matched_root / f"shards/R_matched_{i:02d}.json",
                          root / f"references/R_matched_{i:02d}.json") for i in range(8)]
    # The inherited source manifest identifies every numerical dependency. The
    # only replacements are observation/orchestration and the finalized plotter.
    frozen = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    run_config = Path(protocol["checkpoint"]["path"]).parent.parent / "run_config.json"
    configuration = read(run_config)
    if configuration["architecture"] != protocol["architecture"]:
        raise ValueError("Reset run architecture differs from its evaluation protocol")
    cohort = {**annual, "initializations": annual["initializations"][:92]}
    write(root / "manifests/evaluate.json", cohort)
    cohort_pin = pinned(root / "manifests/evaluate.json")
    shards = []
    for i, part in enumerate(np.array_split(np.asarray(cohort["initializations"], dtype=object), 8)):
        path = root / f"manifests/evaluate_{i:02d}.json"
        write(path, {**cohort, "initializations": list(part)})
        shards.append({"manifest": pinned(path), "count": len(part)})
    baseline_pin = {"path": parent["baseline"]["path"], "sha256": parent["baseline"]["checkpoint_sha256"]}
    inputs = [*references.values(), *canonical, cohort_pin, protocol["checkpoint"], pinned(run_config),
              baseline_pin, *[s["manifest"] for s in shards]]
    # Retain exact source-campaign data/statistic pins, without requiring unrelated
    # development evaluations or changing the parent's campaign while it runs.
    for item in parent["inputs"]:
        path = Path(item["path"])
        if path.parent == Path(parent["stats_dir"]) or str(path).startswith(parent["data"]["evaluate"]):
            inputs.append(item)
    scientific_inputs = []
    for item in literature["sources"]:
        if Path(item["path"]).suffix in (".json", ".npz"):
            path = Path(item["path"])
            record = {"path": str(path if path.is_absolute() else workspace / path), "sha256": item["sha256"]}
            check(record)
            scientific_inputs.append(record)
    inputs.extend(scientific_inputs)
    for item in inputs:
        check(item)
    return seal(root / "manifest.json", {
        "format": "v24_matched_reset_spectra_campaign_v1", "workspace": str(workspace),
        "source_root": str(root / "source"), "source_digest": frozen["source_digest"],
        "model_id": "R_matched", "model_family": configuration.get("model_family", "auto"),
        "checkpoint": protocol["checkpoint"], "run_config": pinned(run_config),
        "architecture": protocol["architecture"], "baseline": parent["baseline"],
        "stats_dir": parent["stats_dir"], "data_path": parent["data"]["evaluate"],
        "target_steps": 40, "reset_state_every_step": True, "training_state_policy": "reset_every_anchor",
        "cohorts": {"evaluate": cohort_pin}, "shards": {"evaluate": shards}, "inputs": inputs,
        "initialization_ids": ids, "references": references, "canonical_reset_metrics": canonical,
        "literature_scientific_inputs": scientific_inputs,
        "policy": {"lmax": 179, "channels": 83, "lead_hours": LEADS, "filter_fitting": False,
                   "variants": ["raw"], "feedback": "raw full forecast", "numerical_policy": "advisory",
                   "scope": "fixed original reset/reset checkpoint on existing92 dates; no new selection"},
        "resources": {"memory_gib": 16, "cpus": 4, "minutes": 62,
                      "constraint": "gpu80&pcie", "report_memory_gib": 12, "report_minutes": 30,
                      "evidence": "17430308_0..3 raw-style8-date calibration:22:00–23:06,9.59–9.82GiB,4CPU; "
                                  "17430310_0..5 full-filter15–16-date evaluation:37:15–39:14,9.63–10.00GiB. "
                                  "New raw-only shards11–12dates; no extra qualification job."}})


def verify(root, *, frozen=True):
    manifest = unseal(root / "manifest.json")
    if (manifest["format"] != "v24_matched_reset_spectra_campaign_v1"
            or manifest["checkpoint"]["sha256"] != RESET_SHA256
            or manifest["reset_state_every_step"] is not True
            or manifest["model_id"] != "R_matched"):
        raise ValueError("Incorrect matched-reset campaign identity or state policy")
    for item in manifest["inputs"]:
        check(item)
    source = verify_source(manifest["source_root"])
    if source["source_digest"] != manifest["source_digest"]:
        raise ValueError("Source identity changed")
    if frozen and ROOT.resolve() != Path(manifest["source_root"]).resolve():
        raise ValueError("Run workers/report/submission from the frozen source snapshot")
    cohort = original.expected_ids(read(manifest["cohorts"]["evaluate"]["path"]))
    shard_ids = [value for shard in manifest["shards"]["evaluate"]
                 for value in original.expected_ids(read(shard["manifest"]["path"]))]
    if cohort != manifest["initialization_ids"] or len(cohort) != 92 or shard_ids != cohort:
        raise ValueError("Missing or duplicate frozen cohort entries")
    return manifest


def make_eval_config(manifest, task, output):
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    if not 0 <= task < len(manifest["shards"]["evaluate"]):
        raise ValueError("Invalid reset spectral shard index")
    if manifest["reset_state_every_step"] is not True:
        raise ValueError("This evaluation requires reset at every forecast step")
    shard = manifest["shards"]["evaluate"][task]
    allowed = {field.name for field in fields(V24IlyaEvalConfig)}
    architecture = {key: value for key, value in manifest["architecture"].items() if key in allowed}
    return V24IlyaEvalConfig(**architecture, ckpt=Path(manifest["checkpoint"]["path"]),
        out_json=output, eval_mode="cold_full", ckpt_in=Path(manifest["baseline"]["path"]),
        stats_dir=Path(manifest["stats_dir"]), data_path=manifest["data_path"],
        target_steps=40, warmup_steps=0, n_samples=shard["count"],
        initialization_manifest=Path(shard["manifest"]["path"]), val_year=2023,
        train_start_year=2015, train_end_year=2021, model_id="R_matched",
        model_family=manifest["model_family"], residual_state_init="zero",
        reset_state_every_step=True, residual_alpha=1.0, omit_rms_bias=True, seed=0)


def drift_diagnostic(measured, reference, *, rtol, label):
    measured, reference = np.asarray(measured), np.asarray(reference)
    if measured.shape != reference.shape or not np.isfinite(measured).all() or not np.isfinite(reference).all():
        raise ValueError(f"Invalid {label} metric arrays")
    delta = measured - reference
    bound = rtol * np.maximum(np.abs(reference), np.finfo(float).tiny)
    warnings = [f"Finite {label} drift exceeds {rtol:g} relative reference; values retained"] if np.any(np.abs(delta) > bound) else []
    return {"reference_relative_bound": rtol, "maximum_absolute_difference": float(np.abs(delta).max()),
            "maximum_relative_difference": float(np.max(np.abs(delta) / np.maximum(np.abs(reference), np.finfo(float).tiny))),
            "pooled_measured": float(measured.mean()), "pooled_reference": float(reference.mean()),
            "pooled_difference": float(delta.mean()), "warnings": warnings}


def evaluate(root, manifest, task):
    import jax
    from src.analysis.raw_spectral_observer import RawSpectralObserver
    from src.models.graphcast.training.core.model import load_stats
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Production reset forecasts require one allocated GPU")
    directory = root / "evaluate" / f"{task:02d}"
    config = make_eval_config(manifest, task, directory / "exact_metrics.json")
    if directory.exists():
        raise FileExistsError("Refuse to overwrite an existing or partial spectral shard")
    directory.mkdir()
    shard = manifest["shards"]["evaluate"][task]
    ids = original.expected_ids(read(shard["manifest"]["path"]))
    identity = {"campaign_hash": manifest["record_hash"], "model_id": "R_matched",
                "checkpoint_sha256": RESET_SHA256, "reset_state_every_step": True,
                "protocol_hash": shard["manifest"]["sha256"], "source_digest": manifest["source_digest"]}
    observer = RawSpectralObserver(directory / "statistics",
        load_stats(Path(manifest["stats_dir"]))["diffs_stddev_by_level"], identity)
    started = time.monotonic()
    metrics = evaluate_v24_Ilya(config, prediction_observer=observer)
    check(manifest["checkpoint"])
    rows = metrics.get("original_graphcast_loss_per_initialization", [])
    if (not metrics.get("metrics_complete") or metrics.get("evaluation_status") != "complete"
            or metrics.get("rs_reset_every_step") is not True
            or [row["initialization_id"] for row in rows] != ids):
        raise ValueError("Incomplete reset/reset metrics or wrong state policy/cohort")
    paths = sorted((directory / "statistics").glob("*.npz"))
    if [path.stem for path in paths] != ids:
        raise ValueError("Missing or unexpected raw spectral dates")
    observed = []
    for path in paths:
        arrays, _ = original.load_statistics(path, manifest, "evaluate")
        observed.append(arrays["exact_gc_loss"])
    engine = np.asarray([[row["baseline_per_step"], row["full_per_step"]] for row in rows]).swapaxes(1, 2)
    roundoff = drift_diagnostic(np.asarray(observed), engine, rtol=1e-5,
                               label="FP64 observer/FP32 evaluator")
    return seal(root / f"checks/evaluate_{task:02d}.json", {
        "status": "complete", "campaign_hash": manifest["record_hash"], "count": len(ids),
        "reset_state_every_step": True, "initialization_ids": ids,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "observer_step_seconds": observer.step_seconds, "grid_mse_roundoff": roundoff,
        "metrics": pinned(config.out_json), "statistics": [pinned(path) for path in paths]})


def canonical_reset_losses(manifest):
    lookup = {}
    for record in manifest["canonical_reset_metrics"]:
        check(record)
        payload = read(record["path"])
        if (payload["checkpoint_sha256"] != RESET_SHA256 or payload["model_id"] != "R_matched"
                or payload["rs_reset_every_step"] is not True or not payload["metrics_complete"]):
            raise ValueError("Canonical reset metric provenance differs")
        for row in payload["original_graphcast_loss_per_initialization"]:
            key = row["initialization_id"]
            if key in lookup or row["lead_hours"] != LEADS:
                raise ValueError("Duplicate canonical reset date or different leads")
            lookup[key] = row["full_per_step"]
    return np.asarray([lookup[key] for key in manifest["initialization_ids"]], dtype=float)


def report(root, manifest):
    from src.analysis.raw_spectral_observer import STATISTICS_SCHEMA
    from src.analysis.smoothing_observer import atomic_npz
    paths = original.statistics_paths(root, manifest, "evaluate")
    if [path.stem for path in paths] != manifest["initialization_ids"]:
        raise ValueError("Report requires all92 pinned dates in order")
    aggregate, layout, individual = None, None, []
    expected_shapes = {"prediction_power": (40, 2, 83, 180), "error_power": (40, 2, 83, 180),
        "cross_power": (40, 2, 83, 180), "truth_power": (40, 83, 180),
        "exact_gc_loss": (40, 2), "spectral_weighted_gc_loss": (40, 2),
        "normalized_grid_mse": (40, 2, 83)}
    for path in paths:
        arrays, metadata = original.load_statistics(path, manifest, "evaluate")
        if (metadata["schema"] != STATISTICS_SCHEMA or metadata["checkpoint_sha256"] != RESET_SHA256
                or metadata["reset_state_every_step"] is not True or metadata["lmax"] != 179
                or metadata["branches"] != ["GC", "R_matched"] or metadata["lead_hours"] != LEADS
                or metadata["source_digest"] != manifest["source_digest"]):
            raise ValueError("Raw spectral provenance differs")
        if layout is None:
            layout = metadata["layout"]
            aggregate = {key: np.zeros(shape, dtype=np.float64) for key, shape in expected_shapes.items()}
        if metadata["layout"] != layout or set(arrays) != set(expected_shapes):
            raise ValueError("Raw spectral channel layout or keys differ")
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape or (key != "cross_power" and np.any(arrays[key] < 0)):
                raise ValueError(f"Invalid raw spectral shape/values: {key}")
            aggregate[key] += arrays[key]
        individual.append(arrays["exact_gc_loss"])
    pooled = {key: value / len(paths) for key, value in aggregate.items()}
    individual = np.asarray(individual)
    diagnostics = {"canonical_reset_metric_drift": drift_diagnostic(
        individual[:, :, 1], canonical_reset_losses(manifest), rtol=1e-3,
        label="new reset forecast/archived reset checkpoint evaluation"),
        "worker_grid_mse_roundoff": [unseal(root / f"checks/evaluate_{i:02d}.json")["grid_mse_roundoff"]
                                    for i in range(8)]}
    references = [pinned(root / "manifest.json"), *[pinned(path) for path in paths]]
    metadata = {"format": "v24_matched_reset_spectra_pooled_v1", "status": "complete",
        "campaign_hash": manifest["record_hash"], "count": len(paths), "model_id": "R_matched",
        "reset_state_every_step": True, "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": RESET_SHA256, "source_digest": manifest["source_digest"],
        "initialization_ids": manifest["initialization_ids"], "branches": ["GC", "R_matched"],
        "lead_hours": LEADS, "layout": layout, "lmax": 179, "variants": ["raw"],
        "sources": references, "numerical_diagnostics": diagnostics,
        "filter_fitting": False, "uncertainty": "fixed checkpoint point estimates; no spectral intervals"}
    pooled_path = root / "report/power_pooled.npz"
    if pooled_path.exists():
        with np.load(pooled_path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"])) != metadata:
                raise ValueError("Existing pooled metadata differs")
            for key, value in pooled.items():
                if not np.array_equal(archive[key], value):
                    raise ValueError("Existing pooled statistics differ")
    else:
        atomic_npz(pooled_path, pooled, metadata)
    summary = seal(root / "report/summary.json", {
        **metadata, "format": "v24_matched_reset_spectra_report_v1",
        "data": {"power_pooled.npz": pinned(pooled_path)},
        "exact_gc_loss_by_lead": pooled["exact_gc_loss"].tolist()})
    command, options = frozen_command(Path(manifest["source_root"]), Path(manifest["workspace"]),
                                     PLOTTER, ["--workspace", manifest["workspace"],
                                               "--reset-source", str(root / "report")])
    subprocess.run(command, check=True, **options)
    write(root / "checks/plot_refresh.json", {"status": "complete", "campaign_hash": manifest["record_hash"],
        "summary": pinned(root / "report/summary.json"), "plotter": pinned(Path(manifest["source_root"]) / PLOTTER),
        "workspace_plot_summary": pinned(Path(manifest["workspace"]) / LITERATURE)})
    return summary


def sbatch(root, manifest, stage, *, array=None, dependency=None):
    budget = manifest["resources"]
    gpu = stage == "evaluate"
    command = ["sbatch", "--parsable", "--account=dabanin", f"--job-name=reset-spectra-{stage}",
        f"--chdir={manifest['workspace']}", f"--mem={budget['memory_gib'] if gpu else budget['report_memory_gib']}G",
        f"--time={budget['minutes'] if gpu else budget['report_minutes']}",
        f"--cpus-per-task={budget['cpus'] if gpu else 2}",
        f"--output={root}/logs/{stage}-%A_%a.out", f"--error={root}/logs/{stage}-%A_%a.err"]
    if gpu:
        command += ["--gres=gpu:1", f"--constraint={budget['constraint']}", "--qos=gpu-short"]
    if array is not None:
        command.append(f"--array={array}")
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    return command + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(root), manifest["source_root"], "0"]


def submit(root, manifest):
    evaluation = submit_once(root, "evaluate", sbatch(root, manifest, "evaluate", array="0-7"))
    reporting = submit_once(root, "report", sbatch(root, manifest, "report", dependency=original.job_id(evaluation)))
    return {"evaluate": evaluation, "report": reporting}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "evaluate", "report", "submit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--parent", type=Path, default=ROOT / PARENT)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == "prepare":
        result = prepare(root, args.workspace, args.parent)
    else:
        manifest = verify(root, frozen=args.stage != "verify")
        if args.stage == "verify":
            result = {"status": "verified", "campaign_hash": manifest["record_hash"],
                      "planned_commands": [sbatch(root, manifest, "evaluate", array="0-7"),
                                           sbatch(root, manifest, "report", dependency="<evaluation_job>")]}
        elif args.stage == "evaluate":
            result = evaluate(root, manifest, args.task)
        else:
            result = {"report": report, "submit": submit}[args.stage](root, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
