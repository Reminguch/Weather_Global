#!/usr/bin/env python3
"""Evaluate the matched reset/reset SWA with the completed 122-date extremes protocol."""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import fcntl
import importlib.metadata
import json
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source

DRIVER = "scripts/experiments/run_v24_matched_reset_extremes.py"
WRAPPER = "scripts/experiments/v24_matched_reset_extremes.slurm"
REPORTER = "scripts/analyze_models/report_v24_matched_reset_extremes.py"
OVERLAYS = (DRIVER, WRAPPER, REPORTER, "scripts/experiments/frozen_runtime.py")
REFERENCE = "artifacts/evaluations/v24_extremes_2023_20261106"
MATCHED = "artifacts/evaluations/v24_matched_reset_2023_20261018/protocol.json"
DEFAULT_ROOT = "artifacts/evaluations/v24_matched_reset_extremes_20261114"
RESET_SHA256 = "db0e8032ac47daebd4e9c8869e4eac3c325613156507512567f20f6382086ab6"
LEADS = list(range(6, 241, 6))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pinned(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def check(record):
    if file_sha256(record["path"]) != record["sha256"]:
        raise ValueError(f"Pinned input changed: {record['path']}")


def initialization_ids(payload):
    return [datetime.fromisoformat(row["initialization_time"].replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
            for row in payload["initializations"]]


def freeze(workspace, reference, destination):
    hashes = read(reference / "source_hashes.json")
    recorded = {}
    # The legacy hash list also pins its own inherited snapshot manifest.
    # Verify that metadata, then create a new manifest for this extension.
    check({"path": str(reference / "source/snapshot.json"), "sha256": hashes["snapshot.json"]})
    for relative in sorted((set(hashes) - {"snapshot.json"}) | set(OVERLAYS)):
        if relative in hashes:
            check({"path": str(reference / "source" / relative), "sha256": hashes[relative]})
        source = (workspace if relative in OVERLAYS else reference / "source") / relative
        proof = pinned(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        check(proof)
        if file_sha256(target) != proof["sha256"]:
            raise ValueError("Source changed during copy")
        recorded[relative] = proof["sha256"]
    write(destination / "snapshot.json", {
        "format": FORMAT, "workspace": str(workspace), "files": recorded,
        "source_digest": _digest(recorded), "parent_source_hashes": pinned(reference / "source_hashes.json"),
        "overlays": list(OVERLAYS),
        "forecast_computation": "Byte-identical to canonical extreme-event source; explicit reset/reset policy."})
    return verify_source(destination)


def prepare(root, workspace):
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    if root.exists():
        raise FileExistsError("Prepare into a new campaign directory; preserve prior results")
    reference = workspace / REFERENCE
    parent = read(reference / "campaign.json")
    completed = read(reference / "complete.json")
    if completed.get("status") != "complete" or completed.get("starts") != 122 or completed.get("leads") != 40:
        raise ValueError("Canonical extreme evaluation is incomplete")
    reference_report = {"path": completed["report"], "sha256": completed["report_sha256"]}
    check(reference_report)
    protocol = read(workspace / MATCHED)
    prototype = read(reference / "shards/00/manifest.json")["protocols"]["spectral"]
    if (protocol["checkpoint"]["sha256"] != RESET_SHA256
            or protocol["model_id"] != "R_matched"
            or protocol["reset_state_every_step"] is not True
            or protocol["training_state_policy"] != "reset_every_anchor"
            or protocol["architecture"] != prototype["architecture"]
            or any(protocol["baseline"][key] != prototype["baseline"][key]
                   for key in ("path", "checkpoint_sha256"))
            or protocol["data_path"] != prototype["data_path"]):
        raise ValueError("Matched reset checkpoint, architecture or evaluation inputs differ")
    check(protocol["checkpoint"])
    check(protocol["initialization_manifest"])
    annual = read(protocol["initialization_manifest"]["path"])
    dates = [row["initialization_time"] for row in annual["initializations"]]
    if dates != parent["dates"] or len(dates) != 122 or dates != sorted(set(dates)):
        raise ValueError("Expected the identical canonical 122-date cohort")
    config_path = Path(protocol["checkpoint"]["path"]).parent.parent / "run_config.json"
    configuration = read(config_path)
    if configuration["architecture"] != protocol["architecture"]:
        raise ValueError("Checkpoint run configuration differs from matched architecture")
    packages = {name: importlib.metadata.version(name) for name in parent["packages"]}
    if packages != parent["packages"]:
        raise ValueError("Evaluation package versions differ from original extreme-event campaign")
    climate_pin = pinned(reference / "climatology/metadata.json")
    climate = read(climate_pin["path"])
    if climate.get("status") != "complete" or climate.get("quantiles") != [.02, .05, .95, .98]:
        raise ValueError("Expected completed original climatological thresholds")
    inputs = [pinned(reference / "campaign.json"), pinned(reference / "complete.json"),
              reference_report, pinned(reference / "source_hashes.json"), pinned(workspace / MATCHED),
              protocol["checkpoint"], pinned(config_path), protocol["initialization_manifest"],
              prototype["data_metadata"], *prototype["statistics"], climate_pin,
              pinned(reference / "land_mask.npy"),
              {"path": prototype["baseline"]["path"], "sha256": prototype["baseline"]["checkpoint_sha256"]}]
    for item in inputs:
        check(item)
    for item in read(reference_report["path"])["sources"]:
        if "/results/M1/" in item["path"]:
            check(item)
    # Validate the existing threshold arrays once now; every worker's frozen
    # observer independently checks their recorded hashes before inference.
    climate_records = climate["arrays"]
    if isinstance(climate_records, dict):
        climate_records = list(climate_records.values())
    for item in climate_records:
        path = Path(item["path"])
        check({**item, "path": str(path if path.is_absolute() else reference / "climatology" / path)})
    root.mkdir(parents=True)
    for name in ("logs", "submissions", "manifests", "evaluate"):
        (root / name).mkdir()
    snapshot = freeze(workspace, reference, root / "source")
    write(root / "manifests/initializations.json", annual)
    annual_pin = pinned(root / "manifests/initializations.json")
    inputs.append(annual_pin)
    shards = []
    for i, rows in enumerate(np.array_split(np.asarray(annual["initializations"], dtype=object), 8)):
        path = root / f"manifests/shard_{i:02d}.json"
        write(path, {**annual, "initializations": list(rows)})
        record = pinned(path)
        inputs.append(record)
        shards.append({"initializations": record, "count": len(rows), "output_dir": str(root / f"evaluate/{i:02d}")})
    manifest = {
        "format": "v24_matched_reset_extremes_v1", "workspace": str(workspace),
        "source_root": str(root / "source"), "source_digest": snapshot["source_digest"],
        "reference_root": str(reference), "reference_campaign": pinned(reference / "campaign.json"),
        "reference_complete": pinned(reference / "complete.json"), "reference_report": reference_report,
        "initializations": annual_pin, "initialization_ids": initialization_ids(annual),
        "dates": dates, "shards": shards, "checkpoint": protocol["checkpoint"],
        "run_config": pinned(config_path), "architecture": protocol["architecture"],
        "model_id": "R_matched", "model_family": configuration.get("model_family", "auto"),
        "baseline": prototype["baseline"], "stats_dir": prototype["stats_dir"],
        "data_path": prototype["data_path"], "target_steps": 40,
        "forecast": {**protocol["forecast"], "reset_state_every_step": True},
        "reset_state_every_step": True, "training_state_policy": "reset_every_anchor",
        "climatology_metadata": climate_pin, "climatology_dir": str(reference / "climatology"),
        "land_mask": pinned(reference / "land_mask.npy"), "events": parent["events"],
        "uncertainty": parent["uncertainty"], "packages": packages, "inputs": inputs,
        "comparison_models": ["GraphCast", "M1", "R_matched"],
        "scope": "Original fixed-threshold occurrence and intensity; no spatial filtering or gain-swept PR curves.",
        "numerical_policy": "Finite drift and weak metrics retained as warnings; no performance gate.",
        "resources": {"gpu": {"tasks": 8, "cpus": 8, "mem_gib": 16, "minutes": 61, "qos": "gpu-short"},
                      "report": {"cpus": 2, "mem_gib": 4, "minutes": 10},
                      "evidence": "Canonical extremes17016910_0–15:23:04–24:18,7.39–9.50GiB,15/16dates. "
                                  "Reset spectra17553195_0–2:24:33–24:55,7.70–8.04GiB,12dates. "
                                  "Allow30–35min per shard;61min/gpu-short routes to regular GPU; user excludes GPU-test. "
                                  "Canonical report17016911:49s/0.27GiB."},
        "output_data_dir": str(workspace / "plots/analyze_models/data/resolution_eval" / root.name),
        "output_image_dir": str(workspace / "plots/analyze_models/images/resolution_eval" / root.name)}
    manifest["manifest_hash"] = _digest(manifest)
    write(root / "manifest.json", manifest)
    return manifest


def verify(root, *, frozen=True):
    manifest = read(Path(root) / "manifest.json")
    if manifest["manifest_hash"] != _digest({k: v for k, v in manifest.items() if k != "manifest_hash"}):
        raise ValueError("Campaign manifest changed")
    if (manifest["format"] != "v24_matched_reset_extremes_v1"
            or manifest["checkpoint"]["sha256"] != RESET_SHA256
            or manifest["reset_state_every_step"] is not True):
        raise ValueError("Incorrect matched-reset checkpoint or state policy")
    for item in manifest["inputs"]:
        check(item)
    source = verify_source(manifest["source_root"])
    if source["source_digest"] != manifest["source_digest"]:
        raise ValueError("Frozen computational source differs")
    if frozen and ROOT.resolve() != Path(manifest["source_root"]).resolve():
        raise ValueError("Execute worker/report/submit from frozen source")
    ids = initialization_ids(read(manifest["initializations"]["path"]))
    shard_ids = [key for shard in manifest["shards"] for key in initialization_ids(read(shard["initializations"]["path"]))]
    if len(ids) != 122 or ids != sorted(set(ids)) or ids != manifest["initialization_ids"] or shard_ids != ids:
        raise ValueError("Missing or duplicate forecast dates")
    if {name: importlib.metadata.version(name) for name in manifest["packages"]} != manifest["packages"]:
        raise ValueError("Evaluation environment differs from frozen reference")
    return manifest


def make_eval_config(manifest, task):
    from src.models.mamba.v24_Ilya.config import V24IlyaEvalConfig
    if not 0 <= task < len(manifest["shards"]) or manifest["reset_state_every_step"] is not True:
        raise ValueError("Invalid reset/reset shard or state policy")
    shard = manifest["shards"][task]
    allowed = {field.name for field in fields(V24IlyaEvalConfig)}
    if set(manifest["architecture"]) - allowed:
        raise ValueError("Frozen evaluator does not support the full matched architecture")
    architecture = manifest["architecture"]
    return V24IlyaEvalConfig(**architecture, ckpt=Path(manifest["checkpoint"]["path"]),
        out_json=Path(shard["output_dir"]) / "metrics.json", eval_mode="cold_full",
        ckpt_in=Path(manifest["baseline"]["path"]), stats_dir=Path(manifest["stats_dir"]),
        data_path=manifest["data_path"], target_steps=40, warmup_steps=0, n_samples=shard["count"],
        initialization_manifest=Path(shard["initializations"]["path"]), val_year=2023,
        train_start_year=2015, train_end_year=2021, model_id="R_matched",
        residual_state_init="zero", reset_state_every_step=True, residual_alpha=1., omit_rms_bias=True, seed=0)


def evaluate(root, manifest, task):
    import jax
    from src.analysis.extremes_observer import ExtremesObserver, FrozenClimatology
    from src.models.mamba.v24_Ilya.evaluation import evaluate_v24_Ilya
    from scripts.analyze_models.report_v24_extremes_2023 import _check_record
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Production forecasts require one allocated GPU")
    config = make_eval_config(manifest, task)
    shard = manifest["shards"][task]
    directory = Path(shard["output_dir"])
    directory.mkdir(exist_ok=False)
    annual = read(shard["initializations"]["path"])
    ids = initialization_ids(annual)
    dates = [row["initialization_time"] for row in annual["initializations"]]
    identity = {"protocol_hash": manifest["manifest_hash"], "model_id": "R_matched",
                "checkpoint_sha256": RESET_SHA256, "baseline_checkpoint_sha256": manifest["baseline"]["checkpoint_sha256"],
                "reset_state_every_step": True, "source_digest": manifest["source_digest"]}
    extremes = {key: manifest[key] for key in ("climatology_dir", "climatology_metadata", "land_mask")}
    observer = ExtremesObserver(directory, identity,
        FrozenClimatology(manifest["climatology_dir"], manifest["climatology_metadata"]),
        np.load(manifest["land_mask"]["path"], allow_pickle=False), target_steps=40,
        provenance={"extremes_protocol": extremes, "campaign_manifest": pinned(root / "manifest.json")})
    started = time.monotonic()
    metrics = evaluate_v24_Ilya(config, prediction_observer=observer)
    check(manifest["checkpoint"])
    rows = metrics.get("original_graphcast_loss_per_initialization", [])
    if (not metrics.get("metrics_complete") or metrics.get("evaluation_status") != "complete"
            or metrics.get("rs_reset_every_step") is not True
            or [row["initialization_id"] for row in rows] != ids
            or any(row["lead_hours"] != LEADS for row in rows)):
        raise ValueError("Incomplete reset metrics or incorrect cohort/state policy")
    losses = np.asarray([[row["baseline_per_step"], row["full_per_step"]] for row in rows])
    if losses.shape != (len(ids), 2, 40) or not np.isfinite(losses).all():
        raise ValueError("Nonfinite or incomplete exact forecast losses")
    paths = sorted((directory / "extremes").glob("*.npz"))
    if [path.stem for path in paths] != ids:
        raise ValueError("Missing or unexpected extreme-event records")
    for path in paths:
        _check_record(path, {**identity, "baseline": manifest["baseline"]}, dates, manifest["climatology_metadata"]["sha256"])
    metrics.update(identity)
    write(config.out_json, metrics)
    write(directory / "complete.json", {"status": "complete", "manifest_hash": manifest["manifest_hash"],
        "initialization_ids": ids, "initialization_times": dates, "reset_state_every_step": True, "metrics": pinned(config.out_json),
        "records": [pinned(path) for path in paths], "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2})


def submit(root, manifest):
    active_path = root / "submissions.json"
    if active_path.exists() and read(active_path).get("scheduling_override"):
        active = read(active_path)
        return {key: active[key] for key in ("worker", "report")}
    common = ["sbatch", "--parsable", "--account=dabanin", f"--chdir={manifest['workspace']}"]
    wrapper = str(Path(manifest["source_root"]) / WRAPPER)
    def once(name, arguments):
        receipt = root / "submissions" / f"{name}.json"
        with receipt.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if receipt.exists():
                prior = read(receipt)
                if prior.get("job_id"):
                    return prior["job_id"]
                raise RuntimeError("Submission began without recorded job ID; inspect Slurm before retrying")
            command = common + arguments
            print(shlex.join(command), flush=True)
            write(receipt, {"status": "submitting", "command": command, "manifest_hash": manifest["manifest_hash"]})
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job = result.stdout.strip().split(";")[0]
            if not job.isdigit():
                raise ValueError(f"Unrecognized sbatch response: {result.stdout}")
            write(receipt, {"status": "submitted", "job_id": job, "command": command,
                            "manifest_hash": manifest["manifest_hash"], "stderr": result.stderr})
            return job
    # Site policy selects the partition automatically; explicit --partition is forbidden.
    worker = once("worker", ["--job-name=v24-reset-extremes", "--qos=gpu-short", "--time=61",
        "--gres=gpu:1", "--constraint=gpu80", "--cpus-per-task=8", "--mem=16G", "--array=0-7",
        f"--output={root}/logs/worker_%A_%a.out", f"--error={root}/logs/worker_%A_%a.err", wrapper, "worker", str(root)])
    report_job = once("report", ["--job-name=v24-reset-extremes-report", "--time=10", "--cpus-per-task=2", "--mem=4G",
        f"--dependency=afterok:{worker}", f"--output={root}/logs/report_%j.out", f"--error={root}/logs/report_%j.err",
        wrapper, "report", str(root)])
    write(root / "submissions.json", {"worker": worker, "report": report_job, "manifest_hash": manifest["manifest_hash"]})
    return {"worker": worker, "report": report_job}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "verify", "worker", "report", "submit"))
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "prepare":
        manifest = prepare(root, args.workspace)
        print(json.dumps({"manifest_hash": manifest["manifest_hash"], "root": str(root)}))
        return
    manifest = verify(root)
    if args.stage == "worker":
        evaluate(root, manifest, args.task)
    elif args.stage == "report":
        from scripts.analyze_models.report_v24_matched_reset_extremes import report
        report(root)
    elif args.stage == "submit":
        print(json.dumps(submit(root, manifest)))
    else:
        print(json.dumps({"status": "verified", "dates": len(manifest["initialization_ids"])}))


if __name__ == "__main__":
    main()
