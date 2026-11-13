"""Add historical raw lambda endpoints without repeating the M1 references."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, unseal, write

OVERLAYS = ("scripts/experiments/prepare_v24_legacy_raw_pilot.py",
            "scripts/experiments/run_v24_raw_checkpoint_pilot.py",
            "scripts/analyze_models/report_v24_raw_checkpoint_pilot.py",
            "scripts/analyze_models/comparison_policy.py")
BASE = "artifacts/checkpoints/v24_Ilya"
SPECS = (
    ("sep24_mse", "Sept24 grid-MSE control", "res1_amse_blend_lambda0p1_20260924", "mse_fp32", 2000, None, "September λ0.1 pair"),
    ("sep24_l010", "Sept24 λ=0.1", "res1_amse_blend_lambda0p1_20260924", "amse", 2000, .1, "September λ0.1 pair"),
    ("sep19_mse", "Sept19 grid-MSE control", "res1_amse_pair_20260919", "mse_fp32", 2000, None, "September λ1 pair"),
    ("sep19_l100", "Sept19 λ=1", "res1_amse_pair_20260919", "amse", 2000, 1., "September λ1 pair"),
    ("oct04_mse", "Oct04 grid-MSE control", "res1_amse_joint_regime_20261004", "mse", 500, None, "October 500-update continuation"),
    ("oct04_l000", "Oct04 λ=0 (spectral MSE)", "res1_amse_joint_regime_20261004", "spectral_mse", 500, 0., "October 500-update continuation"),
    ("oct04_l025", "Oct04 λ=0.25", "res1_amse_joint_regime_20261004", "amse_l025", 500, .25, "October 500-update continuation"),
    ("oct04_l050", "Oct04 λ=0.5", "res1_amse_joint_regime_20261004", "amse_l050", 500, .5, "October 500-update continuation"),
)


def freeze_extension(workspace, parent, destination):
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
        if file_sha256(target) != digest or file_sha256(source) != digest:
            raise ValueError(f"Source changed while freezing: {source}")
        files[relative] = digest
    verify_source(parent)
    write(destination / "snapshot.json", {"format": FORMAT, "workspace": str(workspace),
        "files": files, "source_digest": _digest(files),
        "parent_snapshot": pinned(parent / "snapshot.json"), "overlays": list(OVERLAYS)})
    return verify_source(destination)


def legacy_models(workspace):
    models = []
    for identity, label, campaign, name, step, weight, group in SPECS:
        run = workspace / BASE / campaign / "runs" / name
        config = pinned(run / "run_config.json")
        checkpoint = pinned(run / "checkpoints" / f"checkpoint_step{step:08d}.pkl")
        models.append({"id": identity, "label": label, "model_family": "gc_mamba",
                       "lambda": weight, "step": step, "plot_group": group,
                       "checkpoint": checkpoint, "run_config": config,
                       "selection": "Historical prescribed raw endpoint; no pilot-score selection",
                       "training_run_id": f"{campaign}/{name}",
                       "objective": read(config["path"])["objective"]})
    return models


def prepare_legacy(root, workspace, reference_root):
    root, workspace, reference_root = map(lambda p: Path(p).resolve(), (root, workspace, reference_root))
    if root.exists():
        raise FileExistsError("Use a new legacy pilot directory")
    parent_record = pinned(reference_root / "manifest.json")
    parent = unseal(parent_record["path"])
    check(parent["cohort"])
    if parent["target_steps"] != 40 or len(read(parent["cohort"]["path"])["initializations"]) != 4:
        raise ValueError("Reference is not the requested four-date/40-lead pilot")
    if verify_source(Path(parent["source_root"]))["source_digest"] != parent["source_digest"]:
        raise ValueError("Reference runtime mismatch")
    models, reference_indices = [], []
    for identity in ("M1_raw", "M1_swa"):
        matches = [(i, m) for i, m in enumerate(parent["models"]) if m["id"] == identity]
        if len(matches) != 1:
            raise ValueError("Missing/duplicate M1 reference")
        index, original = matches[0]
        model = copy.deepcopy(original)
        model["reuse"] = {"campaign": parent_record, "directory": str(reference_root / "evaluation" / identity),
                          "source_digest": parent["source_digest"], "manifest_hash": parent["record_hash"]}
        models.append(model); reference_indices.append(index)
    models += legacy_models(workspace)
    for model in models:
        check(model["checkpoint"]); check(model["run_config"])
        config = read(model["run_config"]["path"])
        if pinned(workspace / config["data"]["baseline_checkpoint"])["sha256"] != parent["baseline"]["sha256"]:
            raise ValueError("Legacy model public backbone mismatch")
        if (workspace / config["data"]["stats_dir"]).resolve() != Path(parent["stats_dir"]).resolve():
            raise ValueError("Legacy normalization mismatch")
    root.mkdir(parents=True)
    for sub in ("logs", "submissions", "evaluation", "checks"):
        (root / sub).mkdir()
    snapshot = freeze_extension(workspace, Path(parent["source_root"]), root / "source")
    submission = read(reference_root / "submissions/evaluate.json")
    job = str(submission["job_id"])
    manifest = copy.deepcopy(parent)
    manifest.pop("record_hash")
    manifest.update(format="v24_legacy_raw_checkpoint_pilot_v1", models=models,
        source_root=str(root / "source"), source_digest=snapshot["source_digest"],
        parent_campaign=parent_record,
        inputs=[*parent["inputs"], parent_record],
        report_dependencies=[f"{job}_{index}" for index in reference_indices],
        image_root=str(workspace / "plots/analyze_models/images/resolution_eval" / root.name),
        data_root=str(workspace / "plots/analyze_models/data/resolution_eval" / root.name),
        report_limits="Four-date development pilot, not annual skill or checkpoint selection. Historical September λ0.1/λ1 pairs and October500-update continuations have different training histories and the older spectral interpolation objective; they are not one controlled lambda sweep or the fresh scalar objective. All legacy evaluated endpoints are raw and unaveraged; October training nevertheless started from a September MSE SWA. October spectral-MSE λ0 is distinct from its exact-grid-MSE control. M1 raw2000/SWA references are reused from the original matched pilot. Finite worse metrics do not stop evaluation.")
    manifest["protocol"]["limitations"] = manifest["report_limits"]
    return seal(root / "manifest.json", manifest)
