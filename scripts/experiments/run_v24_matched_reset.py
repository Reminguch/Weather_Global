"""Train a reset-policy counterpart of the selected batch-four di16 model.

Keep the two original optimization stages and their saved implementation. The
only scientific configuration change is reset_every_anchor in both stages.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.frozen_runtime import (
    FORMAT, _digest, _source_files, file_sha256, frozen_command, verify_source,
)

PARENT = "artifacts/checkpoints/v24_Ilya/res1_batch4_temporal_lr_20260908"
DEFAULT = "artifacts/checkpoints/v24_Ilya/res1_selected_matched_reset_20261015"
SWA_STEPS = (500, 1000, 1500, 2000)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(workspace, experiment):
    if experiment.exists():
        raise FileExistsError(f"Refusing to replace an experiment: {experiment}")
    parent = workspace / PARENT
    legacy = parent / "source"
    experiment.mkdir(parents=True)
    configs, inputs = {}, {}
    for stage, filename in [("warmup", "warmup_di16_fast_mamba.json"),
                            ("branch", "di16_fast_mamba.json")]:
        original = parent / "configs" / filename
        cfg = read(original)
        inputs[str(original)] = file_sha256(original)
        new = copy.deepcopy(cfg)
        new["sequence"]["temporal_state_policy"] = "reset_every_anchor"
        new["output"] = {"output_root": str(experiment / "runs"), "run_name": stage}
        # Explicitly reject any extra scientific change.
        comparable = copy.deepcopy(new)
        comparable["sequence"]["temporal_state_policy"] = "carry"
        comparable["output"] = cfg["output"]
        if comparable != cfg:
            raise ValueError("Unexpected configuration change")
        path = experiment / "configs" / f"{stage}.json"
        write(path, new)
        configs[stage] = {"path": str(path), "sha256": file_sha256(path)}

    source = experiment / "source"
    origins = {}
    for original in _source_files(workspace):
        relative = original.relative_to(workspace)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = file_sha256(original)
        shutil.copyfile(original, target)
        if file_sha256(target) != before or file_sha256(original) != before:
            raise ValueError(f"Source changed during copy: {relative}")
        origins[str(relative)] = "workspace"
    for line in (legacy / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        original = legacy / relative
        if file_sha256(original) != expected:
            raise ValueError(f"Archived source hash mismatch: {relative}")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        origins[relative] = "original_saved_source"
    # Restore the one changed dependency outside the partial legacy snapshot.
    revision = (legacy / "git_commit.txt").read_text().strip()
    dependency = "src/models/graphcast/training/core/prepared_array.py"
    content = subprocess.check_output(["git", "show", f"{revision}:{dependency}"], cwd=workspace)
    (source / dependency).write_bytes(content)
    origins[dependency] = f"git:{revision}"
    files = {str(p.relative_to(source)): file_sha256(p) for p in _source_files(source)}
    snapshot = {"format": FORMAT, "workspace": str(workspace), "files": files,
                "source_digest": _digest(files)}
    write(source / "snapshot.json", snapshot)
    verify_source(source)
    write(experiment / "source_provenance.json", {
        "original_revision": revision, "origins": origins,
        "note": "Archived training/model code restored; unarchived dependencies frozen from workspace; prepared_array restored from original revision.",
        "historical_limit": "Original snapshot omitted dependencies and environment; bitwise historical reproduction is not claimed.",
    })
    reference = parent / "runs/di16_fast_mamba/run_config.json"
    inputs[str(reference)] = file_sha256(reference)
    data = read(reference)["data"]
    for name in ("metadata.json", "validity.json"):
        path = workspace / data["prepared_root"] / name
        inputs[str(path)] = file_sha256(path)
    for path in sorted((workspace / data["stats_dir"]).glob("*.nc")):
        inputs[str(path)] = file_sha256(path)
    for directory in ("logs", "checks", "submissions"):
        (experiment / directory).mkdir()
    manifest = {
        "workspace": str(workspace), "source_root": str(source),
        "source_manifest_sha256": file_sha256(source / "snapshot.json"),
        "configs": configs, "input_hashes": inputs,
        "reference_run_config": str(reference),
        "reference_checkpoint": str(parent / "runs/di16_fast_mamba/swa/swa_step00500-02000.pkl"),
        "reference_checkpoint_sha256": "b224e69b76608894e768715072674656f07024f8f48cd3d14cad688e8f7d35a0",
        "scientific_changes": {"sequence.temporal_state_policy": ["carry", "reset_every_anchor"]},
        "stages": ["500-update independent reset warmup", "2000-update reset branch", "SWA"],
        "swa_steps": list(SWA_STEPS),
        "evaluation": "Not launched here; evaluate completed SWA with per-step reset on the original 122-start 2023 protocol.",
    }
    write(experiment / "manifest.json", manifest)
    print(experiment)


def require_runtime(experiment):
    manifest = read(experiment / "manifest.json")
    if ROOT != Path(manifest["source_root"]):
        raise RuntimeError("Run from the prepared source snapshot")
    verify_source(ROOT)
    if file_sha256(ROOT / "snapshot.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Source identity changed")
    for item in manifest["configs"].values():
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Configuration changed: {item['path']}")
    for path, expected in manifest["input_hashes"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"Reference input changed: {path}")
    return manifest


def call(manifest, script, *args):
    command, options = frozen_command(ROOT, manifest["workspace"], script, args)
    subprocess.run(command, check=True, **options)


def preflight(manifest, experiment):
    import importlib.metadata
    from src.models.mamba.v24_Ilya.checkpoint import manifest_sha256
    from src.models.mamba.v24_Ilya.training.config import load_training_config
    from src.models.mamba.v24_Ilya.training.runner import validate_input_paths
    reference = read(manifest["reference_run_config"])
    reports = {}
    for stage, item in manifest["configs"].items():
        cfg = load_training_config(Path(item["path"]))
        validate_input_paths(cfg)
        if cfg.temporal_state_policy != "reset_every_anchor" or not cfg.architecture.temporal_stateful:
            raise ValueError("Must retain architecture and reset memory before every prediction")
        if file_sha256(cfg.baseline_checkpoint) != reference["derived"]["baseline_checkpoint_fingerprint"]:
            raise ValueError("Baseline differs from selected model")
        if manifest_sha256(cfg.anchor_manifest_root) != reference["derived"]["anchor_manifest_fingerprint"]:
            raise ValueError("Training anchors differ from selected model")
        reports[stage] = cfg.to_dict()
    if file_sha256(manifest["reference_checkpoint"]) != manifest["reference_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    write(experiment / "checks/preflight.json", {"passed": True, "resolved_configs": reports,
          "source_digest": verify_source(ROOT)["source_digest"],
          "software": {name: importlib.metadata.version(name) for name in
                       ("jax", "jaxlib", "numpy", "xarray", "dm-haiku", "optax")}})
    return reports


def train(manifest, experiment):
    import numpy as np
    from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
    preflight(manifest, experiment)
    prior = None
    for stage, steps in [("warmup", 500), ("branch", 2000)]:
        config = manifest["configs"][stage]["path"]
        directory = experiment / "runs" / stage
        final = directory / "checkpoints" / f"checkpoint_step{steps:08d}.pkl"
        if directory.exists():
            raise FileExistsError(f"Use explicit audited recovery instead of overwriting {directory}")
        args = ["--config", config]
        if prior is not None:
            args += ["--init-from", prior]
        call(manifest, "scripts/training/train_v24_Ilya.py", *args)
        checkpoint = load_v24_Ilya_training_checkpoint(final)
        if checkpoint.completed_step != steps:
            raise ValueError(f"Incomplete {stage}")
        if checkpoint.resolved_training_config["sequence"]["temporal_state_policy"] != "reset_every_anchor":
            raise ValueError("Checkpoint memory policy differs")
        from src.models.mamba.v24_Ilya.training.config import load_training_config
        if checkpoint.resolved_training_config != load_training_config(Path(config)).to_dict():
            raise ValueError("Completed checkpoint configuration differs")
        import jax
        if any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree_util.tree_leaves(checkpoint.residual_state)):
            raise ValueError("Reset-trained checkpoint contains nonzero recurrent memory")
        prior = final
    output = experiment / "runs/branch/swa/swa_step00500-02000.pkl"
    call(manifest, "scripts/training/build_v24_Ilya_swa.py", "--inputs",
         *[experiment / "runs/branch/checkpoints" / f"checkpoint_step{n:08d}.pkl" for n in SWA_STEPS],
         "--source-steps", *SWA_STEPS, "--output", output)
    write(experiment / "checks/training_complete.json", {
        "completed_updates": {"warmup": 500, "branch": 2000},
        "swa_checkpoint": str(output), "swa_sha256": file_sha256(output),
        "evaluation_pending": True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "check", "train"])
    parser.add_argument("--experiment", type=Path, default=ROOT / DEFAULT)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if args.stage == "prepare":
        prepare(ROOT, experiment)
    else:
        manifest = require_runtime(experiment)
        os.chdir(manifest["workspace"])
        (preflight if args.stage == "check" else train)(manifest, experiment)


if __name__ == "__main__":
    main()
