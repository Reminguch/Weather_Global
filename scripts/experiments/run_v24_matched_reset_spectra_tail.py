#!/usr/bin/env python3
"""Complete the matched reset/reset 2023 spectra with nine short tail jobs.

The original 92-date snapshot supplies every computational file. Only this
orchestrator and its wrapper are overlaid. Reports preserve the original
campaign and write separate tail30 and annual122 products; no plot is refreshed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments import run_v24_matched_reset_spectra as reset
from scripts.experiments.frozen_runtime import FORMAT, _digest, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, submit_once, unseal, write

DRIVER = "scripts/experiments/run_v24_matched_reset_spectra_tail.py"
WRAPPER = "scripts/experiments/v24_matched_reset_spectra_tail.slurm"
OVERLAYS = (DRIVER, WRAPPER)
DEFAULT_ROOT = "artifacts/evaluations/v24_matched_reset_spectra_tail30_20261114"
CAMPAIGN_FORMAT = "v24_matched_reset_spectra_tail_campaign_v1"
SHARD_COUNTS = [4, 4, 4, 3, 3, 3, 3, 3, 3]
EXPECTED_SHAPES = {
    "prediction_power": (40, 2, 83, 180), "error_power": (40, 2, 83, 180),
    "cross_power": (40, 2, 83, 180), "truth_power": (40, 83, 180),
    "exact_gc_loss": (40, 2), "spectral_weighted_gc_loss": (40, 2),
    "normalized_grid_mse": (40, 2, 83),
}


def annual_ids():
    return [(datetime(2023, 1, 1) + timedelta(days=3 * i)).strftime("%Y%m%dT%H%M%SZ")
            for i in range(122)]


def validate_union(prefix, tail, annual):
    if (annual != annual_ids() or len(prefix) != 92 or len(tail) != 30
            or len(set(prefix)) != 92 or len(set(tail)) != 30
            or set(prefix) & set(tail) or prefix + tail != annual):
        raise ValueError("Expected disjoint original92 + missing30 = complete ordered annual122")


def freeze_extension(workspace, parent, destination):
    if destination.exists():
        raise FileExistsError(destination)
    previous = verify_source(parent)
    recorded = {}
    for relative in sorted(set(previous["files"]) | set(OVERLAYS)):
        source = (workspace if relative in OVERLAYS else parent) / relative
        recorded[relative] = reset.copy_pin(source, destination / relative)["sha256"]
    verify_source(parent)
    write(destination / "snapshot.json", {
        "format": FORMAT, "workspace": str(workspace), "files": recorded,
        "source_digest": _digest(recorded), "parent_snapshot": pinned(parent / "snapshot.json"),
        "overlays": list(OVERLAYS),
        "forecast_computation": "Every inherited file is byte-identical to original92 reset snapshot"})
    return verify_source(destination)


def prepare(root, workspace, parent_root, annual_root):
    root, workspace, parent_root, annual_root = [Path(p).resolve()
        for p in (root, workspace, parent_root, annual_root)]
    if root.exists():
        raise FileExistsError("Prepare requires a new tail campaign directory")
    parent = reset.verify(parent_root, frozen=False)
    annual = unseal(annual_root / "manifest.json")
    if (annual["format"] != "v24_optimal_smoothing_campaign_v1"
            or parent["references"]["parent_manifest"]["sha256"] != pinned(annual_root / "manifest.json")["sha256"]
            or parent["data_path"] != annual["data"]["evaluate"]
            or parent["architecture"] != annual["architecture"]
            or parent["baseline"] != annual["baseline"]):
        raise ValueError("Annual cohort is not the original reset campaign's pinned parent")
    check(annual["cohorts"]["evaluate"])
    cohort = read(annual["cohorts"]["evaluate"]["path"])
    all_ids = reset.original.expected_ids(cohort)
    tail = {**cohort, "initializations": cohort["initializations"][92:]}
    ids = reset.original.expected_ids(tail)
    validate_union(parent["initialization_ids"], ids, all_ids)
    # Canonical archive covers all122; fail before preparing if any tail row is absent.
    reset.canonical_reset_losses({**parent, "initialization_ids": ids})
    root.mkdir(parents=True)
    for name in ("logs", "checks", "submissions", "manifests", "references", "evaluate", "report", "annual122_report"):
        (root / name).mkdir()
    references = {
        "original92_manifest": reset.copy_pin(parent_root / "manifest.json", root / "references/original92_manifest.json"),
        "annual_parent_manifest": reset.copy_pin(annual_root / "manifest.json", root / "references/annual_parent_manifest.json"),
        "annual_cohort": reset.copy_pin(Path(annual["cohorts"]["evaluate"]["path"]), root / "manifests/annual122.json"),
    }
    frozen = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    write(root / "manifests/evaluate.json", tail)
    cohort_pin = pinned(root / "manifests/evaluate.json")
    shards = []
    for task, part in enumerate(np.array_split(np.asarray(tail["initializations"], dtype=object), 9)):
        path = root / f"manifests/evaluate_{task:02d}.json"
        write(path, {**tail, "initializations": list(part)})
        shards.append({"manifest": pinned(path), "count": len(part)})
    if [shard["count"] for shard in shards] != SHARD_COUNTS:
        raise ValueError("Unexpected nine-shard partition")
    manifest = copy.deepcopy(parent)
    manifest.pop("record_hash")
    manifest.update({
        "format": CAMPAIGN_FORMAT, "workspace": str(workspace),
        "source_root": str(root / "source"), "source_digest": frozen["source_digest"],
        "original92_root": str(parent_root), "original92_campaign_hash": parent["record_hash"],
        "original92_source_digest": parent["source_digest"],
        "cohorts": {"evaluate": cohort_pin, "annual122": references["annual_cohort"]},
        "shards": {"evaluate": shards}, "initialization_ids": ids,
        "annual_initialization_ids": all_ids,
        "references": {**parent["references"], **references},
        "inputs": [*parent["inputs"], *references.values(), cohort_pin,
                   *[shard["manifest"] for shard in shards]],
        "policy": {**parent["policy"], "scope": "missing30 October4–December30; fixed matched reset checkpoint; annual union122"},
        "resources": {"memory_gib": 16, "cpus": 4, "minutes": 50, "constraint": "gpu80&pcie",
            "qos": "gpu-test", "report_memory_gib": 12, "report_minutes": 30,
            "qos_max_running_per_user": 3, "qos_max_submitted_per_user": 25,
            "evidence": "Reuse original raw-reset evaluation shape; only3–4 dates/job, versus11–12 original. "
                        "Original calibration/evaluation used9.59–10.00GiB. User requests nine50min GPU-test jobs; "
                        "Slurm enforces three running, with nine below25 submitted."},
    })
    return seal(root / "manifest.json", manifest)


def verify(root, *, frozen=True):
    root = Path(root).resolve()
    manifest = unseal(root / "manifest.json")
    if (manifest["format"] != CAMPAIGN_FORMAT or manifest["checkpoint"]["sha256"] != reset.RESET_SHA256
            or manifest["reset_state_every_step"] is not True or manifest["model_id"] != "R_matched"
            or manifest["training_state_policy"] != "reset_every_anchor" or manifest["target_steps"] != 40):
        raise ValueError("Incorrect tail campaign/checkpoint/reset policy")
    for item in manifest["inputs"]:
        check(item)
    parent = unseal(manifest["references"]["original92_manifest"]["path"])
    check({"path": str(Path(manifest["original92_root"]) / "manifest.json"),
           "sha256": manifest["references"]["original92_manifest"]["sha256"]})
    if (parent["record_hash"] != manifest["original92_campaign_hash"]
            or parent["source_digest"] != manifest["original92_source_digest"]):
        raise ValueError("Original92 campaign identity changed")
    for key in ("checkpoint", "run_config", "architecture", "baseline", "stats_dir", "data_path",
                "canonical_reset_metrics", "model_family", "training_state_policy", "reset_state_every_step"):
        if manifest[key] != parent[key]:
            raise ValueError(f"Tail computational input differs: {key}")
    source = verify_source(manifest["source_root"])
    previous = verify_source(parent["source_root"])
    if (source["source_digest"] != manifest["source_digest"]
            or previous["source_digest"] != parent["source_digest"]
            or source["overlays"] != list(OVERLAYS)
            or set(source["files"]) != set(previous["files"]) | set(OVERLAYS)
            or any(source["files"][name] != digest for name, digest in previous["files"].items()
                   if name not in OVERLAYS)):
        raise ValueError("Inherited computational source changed")
    if frozen and ROOT.resolve() != Path(manifest["source_root"]).resolve():
        raise ValueError("Run workers/report/submission from the frozen tail snapshot")
    cohort = reset.original.expected_ids(read(manifest["cohorts"]["evaluate"]["path"]))
    annual = reset.original.expected_ids(read(manifest["cohorts"]["annual122"]["path"]))
    shards = manifest["shards"]["evaluate"]
    shard_ids = [value for shard in shards
                 for value in reset.original.expected_ids(read(shard["manifest"]["path"]))]
    if ([shard["count"] for shard in shards] != SHARD_COUNTS or shard_ids != cohort
            or cohort != manifest["initialization_ids"] or annual != manifest["annual_initialization_ids"]
            or any(len(reset.original.expected_ids(read(shard["manifest"]["path"]))) != shard["count"] for shard in shards)):
        raise ValueError("Tail shard coverage/count mismatch")
    validate_union(parent["initialization_ids"], cohort, annual)
    if (manifest["resources"]["minutes"] != 50 or manifest["resources"]["qos"] != "gpu-test"
            or manifest["resources"]["memory_gib"] != 16 or manifest["resources"]["cpus"] != 4
            or manifest["resources"]["constraint"] != "gpu80&pcie"):
        raise ValueError("Unexpected tail Slurm budget")
    return manifest


def collect(root, manifest):
    """Validate completed shards and preserve each date's separate source identity."""
    from src.analysis.raw_spectral_observer import STATISTICS_SCHEMA
    paths, per_date, worker_diagnostics, checks = [], [], [], []
    for task, shard in enumerate(manifest["shards"]["evaluate"]):
        proof_path = root / f"checks/evaluate_{task:02d}.json"
        proof = unseal(proof_path)
        ids = reset.original.expected_ids(read(shard["manifest"]["path"]))
        if (proof["status"] != "complete" or proof["campaign_hash"] != manifest["record_hash"]
                or proof["count"] != shard["count"] or proof["initialization_ids"] != ids
                or proof["reset_state_every_step"] is not True
                or [Path(item["path"]).stem for item in proof["statistics"]] != ids):
            raise ValueError("Incomplete or incompatible spectral shard proof")
        check(proof["metrics"])
        checks.append(pinned(proof_path))
        worker_diagnostics.append(proof["grid_mse_roundoff"])
        for item in proof["statistics"]:
            check(item)
            paths.append((Path(item["path"]), shard["manifest"]["sha256"], item, pinned(proof_path)))
    if [path.stem for path, *_ in paths] != manifest["initialization_ids"]:
        raise ValueError("Completed statistics do not cover exactly the frozen cohort")
    aggregate = {key: np.zeros(shape, dtype=np.float64) for key, shape in EXPECTED_SHAPES.items()}
    layout, individual = None, []
    for path, protocol_hash, item, proof_pin in paths:
        arrays, metadata = reset.original.load_statistics(path, manifest, "evaluate")
        if (metadata["schema"] != STATISTICS_SCHEMA or metadata["checkpoint_sha256"] != reset.RESET_SHA256
                or metadata["reset_state_every_step"] is not True or metadata["lmax"] != 179
                or metadata["branches"] != ["GC", "R_matched"] or metadata["lead_hours"] != reset.LEADS
                or metadata["variants"] != ["raw"] or metadata["protocol_hash"] != protocol_hash
                or metadata["source_digest"] != manifest["source_digest"]):
            raise ValueError("Raw per-date spectral provenance differs")
        if layout is None:
            layout = metadata["layout"]
        if metadata["layout"] != layout or set(arrays) != set(EXPECTED_SHAPES):
            raise ValueError("Raw spectral channel layout or keys differ")
        for key, shape in EXPECTED_SHAPES.items():
            if arrays[key].shape != shape or (key != "cross_power" and np.any(arrays[key] < 0)):
                raise ValueError(f"Invalid raw spectral array: {key}")
            aggregate[key] += arrays[key]
        individual.append(arrays["exact_gc_loss"])
        per_date.append({"initialization_id": path.stem, "statistics": item, "completion_check": proof_pin,
                         "campaign_hash": manifest["record_hash"], "source_digest": manifest["source_digest"],
                         "protocol_hash": protocol_hash, "checkpoint_sha256": reset.RESET_SHA256})
    return {"sums": aggregate, "individual": np.asarray(individual), "layout": layout,
            "per_date": per_date, "checks": checks, "worker_diagnostics": worker_diagnostics}


def save_pool(directory, manifest, campaigns, data, ids, *, scope):
    from src.analysis.smoothing_observer import atomic_npz
    count = len(ids)
    if [row["initialization_id"] for row in data["per_date"]] != ids:
        raise ValueError("Pool provenance is not an exact ordered cohort")
    pooled = {key: value / count for key, value in data["sums"].items()}
    diagnostics = {
        "canonical_reset_metric_drift": reset.drift_diagnostic(
            data["individual"][:, :, 1], reset.canonical_reset_losses({**manifest, "initialization_ids": ids}),
            rtol=1e-3, label="new reset forecast/archived reset checkpoint evaluation"),
        "worker_grid_mse_roundoff": data["worker_diagnostics"],
        "numerical_policy": "All finite drift is advisory; measured values and warnings retained",
    }
    metadata = {"format": "v24_matched_reset_spectra_pooled_v1", "status": "complete", "scope": scope,
        "count": count, "model_id": "R_matched", "checkpoint": manifest["checkpoint"],
        "checkpoint_sha256": reset.RESET_SHA256, "reset_state_every_step": True,
        "campaign_hash": manifest["record_hash"], "source_digest": manifest["source_digest"],
        "source_digest_scope": "supplemental orchestration snapshot; each forecast source is recorded per date",
        "campaigns": campaigns, "per_date_provenance": data["per_date"],
        "initialization_ids": ids, "branches": ["GC", "R_matched"], "lead_hours": reset.LEADS,
        "layout": data["layout"], "lmax": 179, "variants": ["raw"],
        "sources": [*[item["manifest"] for item in campaigns], *data["checks"],
                    *[row["statistics"] for row in data["per_date"]]],
        "numerical_diagnostics": diagnostics, "filter_fitting": False,
        "uncertainty": "fixed checkpoint point estimates; no spectral intervals"}
    path = directory / "power_pooled.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"])) != metadata or set(archive.files) != set(pooled) | {"metadata"}:
                raise ValueError("Existing pooled metadata/keys differ; refuse overwrite")
            if any(not np.array_equal(archive[key], value) for key, value in pooled.items()):
                raise ValueError("Existing pooled statistics differ; refuse overwrite")
    else:
        atomic_npz(path, pooled, metadata)
    return seal(directory / "summary.json", {
        **metadata, "format": "v24_matched_reset_spectra_report_v1",
        "data": {"power_pooled.npz": pinned(path)},
        "exact_gc_loss_by_lead": pooled["exact_gc_loss"].tolist()})


def report(root, manifest):
    # One writer also protects atomic_npz's shared .partial name on retries.
    with (root / "report/report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_root = Path(manifest["original92_root"])
        parent = reset.verify(parent_root, frozen=False)
        tail = collect(root, manifest)
        prefix = collect(parent_root, parent)
        validate_union(parent["initialization_ids"], manifest["initialization_ids"], manifest["annual_initialization_ids"])
        if prefix["layout"] != tail["layout"]:
            raise ValueError("Original92 and tail30 spectral layouts differ")
        campaigns = [{"manifest": pinned(directory / "manifest.json"), "campaign_hash": item["record_hash"],
                      "source_digest": item["source_digest"], "count": len(item["initialization_ids"])}
                     for directory, item in ((parent_root, parent), (root, manifest))]
        tail_summary = save_pool(root / "report", manifest, campaigns[1:], tail,
                                 manifest["initialization_ids"], scope="tail30")
        combined = {
            "sums": {key: prefix["sums"][key] + tail["sums"][key] for key in EXPECTED_SHAPES},
            "individual": np.concatenate((prefix["individual"], tail["individual"]), axis=0),
            "layout": tail["layout"], "per_date": prefix["per_date"] + tail["per_date"],
            "checks": prefix["checks"] + tail["checks"],
            "worker_diagnostics": prefix["worker_diagnostics"] + tail["worker_diagnostics"],
        }
        annual_summary = save_pool(root / "annual122_report", manifest, campaigns, combined,
                                   manifest["annual_initialization_ids"], scope="annual122")
        return seal(root / "checks/report.json", {
            "status": "complete", "campaign_hash": manifest["record_hash"], "tail_count": tail_summary["count"],
            "annual_count": annual_summary["count"], "tail_summary": pinned(root / "report/summary.json"),
            "annual_summary": pinned(root / "annual122_report/summary.json"),
            "original92_preserved": True, "plot_refresh": False})


def sbatch(root, manifest, stage, *, dependency=None):
    budget, gpu = manifest["resources"], stage == "evaluate"
    command = ["sbatch", "--parsable", "--account=dabanin", f"--job-name=reset-tail-{stage}",
        f"--chdir={manifest['workspace']}", f"--mem={budget['memory_gib'] if gpu else budget['report_memory_gib']}G",
        f"--time={budget['minutes'] if gpu else budget['report_minutes']}",
        f"--cpus-per-task={budget['cpus'] if gpu else 2}",
        f"--output={root}/logs/{stage}-%A_%a.out", f"--error={root}/logs/{stage}-%A_%a.err"]
    if gpu:
        command += ["--gres=gpu:1", f"--constraint={budget['constraint']}", "--qos=gpu-test", "--array=0-8"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    return command + [str(Path(manifest["source_root"]) / WRAPPER), stage, str(root), manifest["source_root"], "0"]


def submit(root, manifest):
    evaluation = submit_once(root, "evaluate", sbatch(root, manifest, "evaluate"))
    reporting = submit_once(root, "report", sbatch(root, manifest, "report", dependency=reset.original.job_id(evaluation)))
    return {"evaluate": evaluation, "report": reporting, "gpu_tasks": 9, "tail_dates": 30}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "evaluate", "report", "submit"))
    parser.add_argument("--root", type=Path, default=ROOT / DEFAULT_ROOT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--parent", type=Path, default=ROOT / reset.DEFAULT_ROOT)
    parser.add_argument("--annual-parent", type=Path, default=ROOT / reset.PARENT)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.stage == "prepare":
        result = prepare(root, args.workspace, args.parent, args.annual_parent)
    else:
        manifest = verify(root, frozen=args.stage != "verify")
        if args.stage == "verify":
            result = {"status": "verified", "campaign_hash": manifest["record_hash"], "tail_count": 30,
                "annual_count": 122, "shard_counts": SHARD_COUNTS,
                "planned_commands": [sbatch(root, manifest, "evaluate"),
                                     sbatch(root, manifest, "report", dependency="<evaluation_job>")]}
        elif args.stage == "evaluate":
            result = reset.evaluate(root, manifest, args.task)
        else:
            result = {"report": report, "submit": submit}[args.stage](root, manifest)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
