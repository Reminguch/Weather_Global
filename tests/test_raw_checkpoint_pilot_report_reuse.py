"""Reuse must preserve forecast identities while grouping distinct training histories."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_models import report_v24_raw_checkpoint_pilot as report
from scripts.experiments.frozen_runtime import _digest
from scripts.experiments.run_v24_mgda import pinned, read, seal, write


def snapshot(directory, report_text):
    files = {}
    for relative, contents in {
        "src/evaluator.py": "same computational runtime",
        "third_party/graphcast/graphcast/graphcast.py": "same GraphCast runtime",
        "scripts/analyze_models/report_v24_raw_checkpoint_pilot.py": report_text,
    }.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        files[relative] = pinned(path)["sha256"]
    value = {"files": files, "source_digest": _digest(files)}
    write(directory / "snapshot.json", value)
    return value["source_digest"]


def fixture(tmp_path):
    parent_root, root = tmp_path / "parent", tmp_path / "extension"
    parent_root.mkdir(); root.mkdir()
    cohort_path = parent_root / "cohort.json"
    dates = [f"2022-{month:02d}-02T00:00:00Z" for month in (1, 4, 7, 10)]
    ids = [stamp.replace("-", "").replace(":", "") for stamp in dates]
    write(cohort_path, {"initializations": [{"initialization_time": stamp} for stamp in dates]})
    checkpoint, config, baseline = [parent_root / name for name in ("weights.pkl", "config.json", "gc.npz")]
    for path in (checkpoint, config, baseline):
        path.write_text("fixture")
    model = {"id": "M1_raw", "label": "M1 raw", "model_family": "gc_mamba", "lambda": None,
             "step": 2000, "checkpoint": pinned(checkpoint), "run_config": pinned(config)}
    parent = seal(parent_root / "manifest.json", {
        "models": [model], "cohort": pinned(cohort_path), "target_steps": 40,
        "baseline": pinned(baseline), "source_root": str(parent_root / "source"),
        "source_digest": snapshot(parent_root / "source", "original report")})
    reused = {**model, "reuse": {"campaign": pinned(parent_root / "manifest.json"),
              "directory": str(parent_root / "evaluation/M1_raw"),
              "manifest_hash": parent["record_hash"], "source_digest": parent["source_digest"]}}
    manifest = seal(root / "manifest.json", {
        "models": [reused], "cohort": parent["cohort"], "target_steps": 40,
        "baseline": parent["baseline"], "source_root": str(root / "source"),
        "source_digest": snapshot(root / "source", "new grouping report")})
    directory = Path(reused["reuse"]["directory"])
    directory.mkdir(parents=True)
    leads = list(range(6, 241, 6))
    evaluation = {"evaluation_status": "complete", "metrics_complete": True,
        "remaining_samples": 0, "model_id": "M1_raw", "evaluated_samples": 4,
        "target_steps": 40, "initialization_manifest_sha256": manifest["cohort"]["sha256"],
        "ckpt_in": str(baseline), "original_graphcast_loss_per_initialization": [
            {"initialization_id": identity, "initialization_time": stamp, "lead_hours": leads}
            for identity, stamp in zip(ids, dates)], "original_graphcast_loss": {
                "baseline_per_step": [2.] * 40, "full_per_step": [2.01] * 40,
                "baseline_rollout": 2., "full_rollout": 2.01}}
    write(directory / "evaluation.json", evaluation)
    channels = [["fixture", level] for level in range(83)]
    metadata = {"layout": {"channels": channels}, "count": 4, "initialization_ids": ids,
        "lead_hours": leads, "campaign_hash": parent["record_hash"],
        "cohort_sha256": manifest["cohort"]["sha256"], "source_digest": parent["source_digest"],
        "lmax": 179, "checkpoint_sha256": model["checkpoint"]["sha256"],
        "branches": ["public_gc", "M1_raw"]}
    np.savez_compressed(directory / "pooled_spectra.npz", metadata=json.dumps(metadata),
        truth_power=np.ones((40, 83, 180)), prediction_power=np.ones((40, 2, 83, 180)),
        error_power=np.ones((40, 2, 83, 180)), cross_power=-np.ones((40, 2, 83, 180)))
    seal(directory / "complete.json", {"status": "complete", "manifest_hash": parent["record_hash"],
        "checkpoint": model["checkpoint"], "count": 4,
        "evaluation": pinned(directory / "evaluation.json"),
        "pooled_spectra": pinned(directory / "pooled_spectra.npz")})
    return root, manifest, reused, directory


def test_load_reused_parent_outputs_accepts_report_only_source_changes(tmp_path):
    root, manifest, _, _ = fixture(tmp_path)
    loaded, _, dates, _ = report.load_models(root, manifest)
    assert len(dates) == 4
    assert loaded["M1_raw"]["loss"]["full_rollout"] == 2.01
    assert loaded["M1_raw"]["power"]["cross_power"].min() == -1


@pytest.mark.parametrize("field", ["cohort", "checkpoint", "directory"])
def test_reuse_rejects_changed_identity(tmp_path, field):
    root, manifest, model, _ = fixture(tmp_path)
    manifest, model = deepcopy(manifest), deepcopy(model)
    if field == "cohort":
        manifest["cohort"]["sha256"] = "different cohort"
    elif field == "checkpoint":
        model["checkpoint"]["sha256"] = "different weights"
    else:
        model["reuse"]["directory"] += "/different-model"
    with pytest.raises(ValueError, match="Reused"):
        report.evaluation_context(root, manifest, model, set())


def test_reuse_rejects_changed_computational_source_even_with_new_digest(tmp_path):
    root, manifest, model, _ = fixture(tmp_path)
    path = root / "source/src/evaluator.py"
    path.write_text("a changed forecast algorithm")
    saved = read(root / "source/snapshot.json")
    saved["files"]["src/evaluator.py"] = pinned(path)["sha256"]
    saved["source_digest"] = _digest(saved["files"])
    write(root / "source/snapshot.json", saved)
    manifest["source_digest"] = saved["source_digest"]
    with pytest.raises(ValueError, match="computational source differs"):
        report.evaluation_context(root, manifest, model, set())


def test_plot_groups_separate_training_histories_and_omit_references():
    models = [{"id": "M1_raw", "model_family": "gc_mamba"},
              {"id": "M1_swa", "model_family": "gc_mamba"},
              {"id": "old01", "model_family": "gc_mamba", "plot_group": "September λ0.1 pair"},
              {"id": "old01_control", "model_family": "gc_mamba", "plot_group": "September λ0.1 pair"},
              {"id": "old1", "model_family": "gc_mamba", "plot_group": "September λ1 pair"},
              {"id": "fresh", "model_family": "full_gc"}]
    assert report.plot_groups(models) == {"September λ0.1 pair": ["old01", "old01_control"],
        "September λ1 pair": ["old1"], "full_gc": ["fresh"]}
    assert report.figure_group_name("September λ0.1 pair") == "september_lambda0_1_pair"
