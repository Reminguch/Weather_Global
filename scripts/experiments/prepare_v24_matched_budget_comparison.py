#!/usr/bin/env python3
"""Prepare matched-total-update M1/November checkpoint evaluations, without training."""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, unseal, write

REFERENCE = "artifacts/evaluations/v24_raw_checkpoint_pilot_20261110"
DEFAULT = "artifacts/evaluations/v24_matched_budget_comparison_20261112"
DRIVER = "scripts/experiments/prepare_v24_matched_budget_comparison.py"
OVERLAYS = (DRIVER, "scripts/experiments/run_v24_raw_checkpoint_pilot.py",
            "scripts/analyze_models/report_v24_raw_checkpoint_pilot.py")
GROUP = "Matched training updates"
LIMITS = (
    "Four matched 2022 development starts and 40 six-hour leads, not annual skill "
    "or checkpoint selection. Compare raw checkpoints at 1000 and 1250 total "
    "training updates. M1 has 500 initial updates: main-step500/750 therefore "
    "mean total1000/1250; November checkpoint steps equal total updates. "
    "M1 raw/SWA at total2500 are contextual references, not budget-matched "
    "competitors. Existing November1250 and M1 endpoint forecasts are reused. "
    "Evaluation dates, truth, normalization, cold initial state, full-feedback "
    "rollouts and computational source match; training recipes do not. "
    "M1 uses stopped weather-feedback gradients and a two-stage optimizer "
    "schedule; November uses full weather-feedback gradients and one schedule. "
    "Full-GC lambda0 has peakLR3e-7; November GC-Mamba lambda0 has3e-5. "
    "Power mismatch measures amplitude fidelity, not spatial alignment. "
    "Finite drift and deterioration remain warnings, not completion gates."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze_extension(workspace, parent, destination):
    """Keep forecast/metric code identical to the completed comparison."""
    if destination.exists():
        raise FileExistsError(destination)
    previous = verify_source(parent)
    files = {}
    for relative in sorted(set(previous["files"]) | set(OVERLAYS)):
        source = (workspace if relative in OVERLAYS else parent) / relative
        digest = file_sha256(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(file_sha256(target) == file_sha256(source) == digest,
                f"Source changed while freezing: {source}")
        files[relative] = digest
    verify_source(parent)
    write(destination / "snapshot.json", {
        "format": FORMAT, "workspace": str(workspace), "files": files,
        "source_digest": _digest(files), "parent_snapshot": pinned(parent / "snapshot.json"),
        "overlays": list(OVERLAYS)})
    return verify_source(destination)


def parent_model(parent, identity):
    matches = [m for m in parent["models"] if m["id"] == identity]
    require(len(matches) == 1, f"Missing/duplicate parent model: {identity}")
    return copy.deepcopy(matches[0])


def reused_model(parent, parent_record, reference_root, identity):
    model = parent_model(parent, identity)
    directory = reference_root / "evaluation" / identity
    completion_path = directory / "complete.json"
    completion = unseal(completion_path)
    require(completion["status"] == "complete"
            and completion["manifest_hash"] == parent["record_hash"]
            and completion["checkpoint"] == model["checkpoint"]
            and completion["count"] == 4,
            f"Reused completion does not match: {identity}")
    inputs = [pinned(completion_path), model["run_config"]]
    for key in ("checkpoint", "evaluation", "pooled_spectra"):
        check(completion[key])
        inputs.append(completion[key])
    model["reuse"] = {"campaign": parent_record, "directory": str(directory),
                      "source_digest": parent["source_digest"], "manifest_hash": parent["record_hash"]}
    if identity in ("M1_raw", "M1_swa"):
        model.update(training_updates=2500, total_training_updates=2500,
                     checkpoint_kind="raw" if identity == "M1_raw" else "swa")
        model["label"] = ("M1 raw, total 2500 (main step 2000)" if identity == "M1_raw"
                          else "M1 SWA, total 1000–2500")
    else:
        model.update(training_updates=1250, total_training_updates=1250,
                     checkpoint_kind="raw", plot_group=GROUP)
        model["label"] += ", raw total 1250"
    return model, inputs


def earlier_model(parent, original_id, identity, total_updates):
    """Select by total training cost, retaining M1's preceding 500 updates."""
    model = parent_model(parent, original_id)
    is_m1 = original_id == "M1_raw"
    initial_updates = 500 if is_m1 else 0
    stored_step = total_updates - initial_updates
    require(total_updates in (1000, 1250) and stored_step > 0, "Unsupported matched budget")
    run = Path(model["run_config"]["path"]).parent
    model.update(id=identity, step=stored_step, training_updates=total_updates,
                 total_training_updates=total_updates, preceding_training_updates=initial_updates,
                 checkpoint=pinned(run / "checkpoints" / f"checkpoint_step{stored_step:08d}.pkl"),
                 checkpoint_kind="raw", plot_group=GROUP,
                 selection="Predetermined raw checkpoint at matched total training updates")
    base_label = "M1" if is_m1 else model["label"]
    model["label"] = f"{base_label}, raw total {total_updates}"
    if is_m1:
        model["label"] += f" (main step {stored_step})"
    return model


def verify_checkpoint(model, baseline_hash):
    """Inspect serialized progress and identity instead of trusting filenames."""
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    checkpoint = load_v24_Ilya_training_checkpoint(Path(model["checkpoint"]["path"]))
    require(checkpoint.completed_step == model["step"], "Checkpoint completed step differs")
    require(checkpoint.model_family == model["model_family"], "Checkpoint model family differs")
    require(checkpoint.baseline_checkpoint_fingerprint == baseline_hash, "Checkpoint backbone differs")
    config = read(model["run_config"]["path"])
    require(all(key in config and config[key] == value
                for key, value in checkpoint.resolved_training_config.items()),
            "Checkpoint configuration differs from pinned run")
    proof = {"checkpoint": model["checkpoint"], "completed_step": checkpoint.completed_step,
             "total_training_updates": model["total_training_updates"],
             "preceding_training_updates": model.get("preceding_training_updates", 0),
             "model_family": checkpoint.model_family,
             "baseline_checkpoint_fingerprint": checkpoint.baseline_checkpoint_fingerprint,
             "run_config": model["run_config"]}
    del checkpoint
    gc.collect()
    check(model["checkpoint"])
    return proof


def prepare(root, workspace, reference_root):
    from scripts.analyze_models.report_v24_raw_checkpoint_pilot import verify_computational_sources
    root, workspace, reference_root = [Path(p).resolve() for p in (root, workspace, reference_root)]
    if root.exists():
        raise FileExistsError("Use a new campaign directory; preserve existing evidence")
    parent_record = pinned(reference_root / "manifest.json")
    parent = unseal(parent_record["path"])
    check(parent["cohort"])
    cohort = read(parent["cohort"]["path"])
    require(parent["target_steps"] == 40 and len(cohort["initializations"]) == 4,
            "Expected the completed four-date/40-lead pilot")
    require(verify_source(Path(parent["source_root"]))["source_digest"] == parent["source_digest"],
            "Reference runtime differs")
    models, inputs = [], [parent_record, *parent["inputs"]]
    for identity in ("M1_raw", "M1_swa"):
        model, records = reused_model(parent, parent_record, reference_root, identity)
        models.append(model); inputs.extend(records)
    models += [earlier_model(parent, "M1_raw", "m1_total1000_raw", 1000),
               earlier_model(parent, "M1_raw", "m1_total1250_raw", 1250),
               earlier_model(parent, "gcm_s22_l000", "gc_mamba_l000_total1000_raw", 1000),
               earlier_model(parent, "gcft_s22_l000", "full_gc_l000_total1000_raw", 1000)]
    for identity in ("gcm_s22_l000", "gcft_s22_l000"):
        model, records = reused_model(parent, parent_record, reference_root, identity)
        models.append(model); inputs.extend(records)
    require(len(models) == len({m["id"] for m in models}) == 8, "Unexpected models")
    proofs = []
    for model in models:
        check(model["checkpoint"]); check(model["run_config"])
        cfg = read(model["run_config"]["path"])
        require(pinned(workspace / cfg["data"]["baseline_checkpoint"])["sha256"]
                == parent["baseline"]["sha256"], "Public baseline differs")
        require((workspace / cfg["data"]["stats_dir"]).resolve() == Path(parent["stats_dir"]).resolve(),
                "Normalization statistics differ")
        if "reuse" not in model:
            proofs.append(verify_checkpoint(model, parent["baseline"]["sha256"]))
        inputs.extend([model["checkpoint"], model["run_config"]])
    for item in inputs:
        check(item)
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "evaluation", "checks"):
        (root / sub).mkdir()
    snapshot = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    proof_path = root / "checks" / "checkpoint_identity.json"
    seal(proof_path, {"status": "complete", "checkpoints": proofs})
    manifest = copy.deepcopy(parent)
    manifest.pop("record_hash")
    manifest.update(format="v24_matched_budget_comparison_v1", models=models,
        source_root=str(root / "source"), source_digest=snapshot["source_digest"],
        parent_campaign=parent_record, inputs=[*inputs, pinned(proof_path)], report_dependencies=[],
        image_root=str(workspace / "plots/analyze_models/images/resolution_eval" / root.name),
        data_root=str(workspace / "plots/analyze_models/data/resolution_eval" / root.name),
        report_title="Matched training budgets: M1 and November lambda-zero MSE and power spectra",
        figure_title="Matched total updates: 1000 and 1250", report_limits=LIMITS,
        resources={"gpu": 1, "cpus": 4, "mem_gib": 16, "minutes": 62,
                   "evidence": "Completed four-date pilot17249273: 14m44s–15m46s/model, maximum8.75GiB host RSS;62min avoids the documented1h gputest routing boundary."})
    manifest["protocol"].update(limitations=LIMITS,
        checkpoint_selection="Raw total1000 and1250, chosen before their scores; endpoint references reused",
        matched_total_updates=[1000, 1250])
    verify_computational_sources(parent, manifest)
    return seal(root / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--reference-root", type=Path, default=ROOT / REFERENCE)
    args = parser.parse_args()
    result = prepare(args.root, args.workspace, args.reference_root)
    print(json.dumps({"manifest": str(args.root / "manifest.json"),
                      "models": [{k: m.get(k) for k in ("id", "step", "total_training_updates", "reuse")}
                                 for m in result["models"]]}, indent=2))


if __name__ == "__main__":
    main()
