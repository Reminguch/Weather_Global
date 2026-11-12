"""Keep stored steps, total budgets and mixed campaign identities distinct."""
from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts.experiments import prepare_v24_mixed_final_comparison as mixed
from scripts.experiments import run_v24_m1_matched_downstream as downstream


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_new_gcm_preparation_does_not_invent_run_config_or_checkpoint_hashes(tmp_path):
    tasks = []
    for index, weight in enumerate((0, .25, .5, .75, 1)):
        config = tmp_path / "configs" / f"arm{index}.json"
        write(config, {"optimizer": {"max_steps": 2000},
                       "sequence": {"feedback_mode": "closed_loop_sg"}})
        tasks.append({"id": f"arm{index}", "logical_id": f"arm{index}",
            "model_family": "gc_mamba", "lambda": weight, "learning_rate": 1e-4,
            "initial_config": mixed.pinned(config), "main_config": mixed.pinned(config)})
    mixed.seal(tmp_path / "manifest.json", {"tasks": tasks, "source_digest": "training-source"})
    specs, records = mixed.gcm_specs(tmp_path)
    assert len(specs) == 5
    for spec in specs:
        assert spec["swa_steps"] == [500, 1000, 1500, 2000]
        assert spec["final_stored_step"] == 2000
        assert "run_config" not in spec
        assert not Path(spec["run_config_path"]).exists()
        assert spec["final_proof_path"].endswith("_total00002500.json")
    assert all(Path(record["path"]).exists() for record in records)


def final_fixture(tmp_path):
    config = tmp_path / "main_config.json"
    write(config, {})
    actual = tmp_path / "run_config.json"
    write(actual, {"config_source": str(config)})
    metrics = tmp_path / "metrics.jsonl"
    write(metrics, {"loss": 100, "scientific_improvement_required": False})
    initial = tmp_path / "initial.json"
    mixed.seal(initial, {"status": "complete", "completed_steps": 500,
        "total_training_updates": 500, "manifest_hash": "new-training", "metrics": mixed.pinned(metrics)})
    spec = {"id": "gcm", "training_run_id": "gcm", "model_family": "gc_mamba", "lambda": .75,
        "final_stored_step": 2000, "swa_steps": [500, 1000, 1500, 2000],
        "training_manifest_hash": "new-training", "training_source_digest": "new-source",
        "training_config": mixed.pinned(config), "run_config_path": str(actual),
        "final_proof_path": str(tmp_path / "final.json"), "checkpoint_paths": {}}
    for step in spec["swa_steps"]:
        path = tmp_path / f"checkpoint_step{step:08d}.pkl"
        path.write_bytes(str(step).encode())
        spec["checkpoint_paths"][str(step)] = str(path)
    proof = {"status": "complete", "task_id": "gcm", "model_family": "gc_mamba", "lambda": .75,
        "completed_steps": 2000, "total_training_updates": 2500, "manifest_hash": "new-training",
        "source_digest": "new-source", "metrics": mixed.pinned(metrics),
        "initial_stage_proof": mixed.pinned(initial), "run_config": mixed.pinned(actual),
        "checkpoint": mixed.pinned(spec["checkpoint_paths"]["2000"])}
    mixed.seal(spec["final_proof_path"], proof)
    return spec, proof


def test_final_main2000_requires_initial500_and_total2500(tmp_path):
    spec, proof = final_fixture(tmp_path)
    resolved, sources, records = mixed.final_sources(spec)
    assert resolved["run_config"] == proof["run_config"]
    assert sources[-1] == proof["checkpoint"]
    assert len(sources) == 4
    changed = deepcopy(proof)
    changed["total_training_updates"] = 2000
    Path(spec["final_proof_path"]).unlink()
    mixed.seal(spec["final_proof_path"], changed)
    with pytest.raises(ValueError, match="2500 total"):
        mixed.final_sources(spec)


def test_mixed_completion_rejects_foreign_training_or_changed_weights(tmp_path):
    spec, proof = final_fixture(tmp_path)
    proof["manifest_hash"] = "old-campaign"
    Path(spec["final_proof_path"]).unlink()
    mixed.seal(spec["final_proof_path"], proof)
    with pytest.raises(ValueError, match="completion differs"):
        mixed.final_sources(spec)
    proof["manifest_hash"] = "new-training"
    Path(spec["final_proof_path"]).unlink()
    mixed.seal(spec["final_proof_path"], proof)
    Path(spec["checkpoint_paths"]["2000"]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="Final checkpoint differs"):
        mixed.final_sources(spec)


def test_swa_averages_main_steps_with_500_preceding_updates(tmp_path):
    from src.models.mamba.v24_Ilya.checkpoint import training_checkpoint_payload, load_v24_Ilya_checkpoint
    root = tmp_path / "campaign"
    (root / "swa").mkdir(parents=True)
    (root / "checks").mkdir()
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps({"model_family": "gc_mamba"}))
    sources = []
    for step in mixed.GCM_SWA_STEPS:
        payload = training_checkpoint_payload(completed_step=step,
            residual_params={"m": {"w": np.array([step], np.float32)}},
            residual_state={"m": {"state": np.ones(1, np.float32)}}, optimizer_state={},
            rng_key=np.array([0, 22], np.uint32),
            training_cursor={"epoch": 0, "segment_index": 0, "segment_offset": 0},
            resolved_training_config={"model_family": "gc_mamba"},
            baseline_checkpoint_path="baseline.npz", baseline_checkpoint_fingerprint="baseline",
            anchor_manifest_fingerprint="anchors", parameter_overlay_metadata={})
        path = tmp_path / f"step{step}.pkl"
        path.write_bytes(pickle.dumps(payload)); sources.append(mixed.pinned(path))
    spec = {"id": "gcm", "model_family": "gc_mamba", "run_config": mixed.pinned(config_path),
            "swa_steps": list(mixed.GCM_SWA_STEPS)}
    plan = {"record_hash": "plan", "manifest_template": {"baseline": {"sha256": "baseline"}}}
    record, proof = mixed.original.build_swa(root, plan, spec, sources)
    loaded = load_v24_Ilya_checkpoint(Path(record["path"]))
    np.testing.assert_array_equal(loaded.residual_params["m"]["w"], [1250])
    assert not loaded.has_residual_state
    assert mixed.unseal(proof["path"])["source_steps"] == [500, 1000, 1500, 2000]


def test_mixed_submission_waits_only_unchanged_gc_and_new_gcm(tmp_path):
    plan = {"workspace": "/workspace", "source_root": "/source",
        "training_dependencies": ["17049163", "99999999"],
        "resources": {"materialize": {"cpus": 2, "mem_gib": 8, "minutes": 30},
                      "evaluate": {"cpus": 4, "mem_gib": 16, "minutes": 62},
                      "report": {"cpus": 2, "mem_gib": 4, "minutes": 15}}}
    commands = mixed.submission_plan(tmp_path, plan)
    assert "--dependency=afterok:17049163:99999999" in commands["materialize"]
    assert "--array=2-21" in commands["evaluate"]
    assert all("17062358" not in arg for cmd in commands.values() for arg in cmd)


def test_annual_join_follows_actual_releases_and_skips_completed_stages(tmp_path):
    a, b = tmp_path / "new_gcm", tmp_path / "unchanged_gc"
    write(a / "submissions/release-calibration.json", {"job_id": "100"})
    write(b / "submissions/release-calibration.json", {"job_id": "200"})
    write(b / "submissions/release-evaluation.json", {"job_id": "300"})
    manifest = {"directories": {"a": str(a), "b": str(b)}}
    assert downstream.pending_release_dependencies(manifest, "release-calibration.json", "release-evaluation.json") == ["100"]
    mixed.seal(a / "report.json", {"status": "complete"})
    assert downstream.pending_release_dependencies(manifest, "release-calibration.json", "release-evaluation.json") == []
    assert downstream.pending_release_dependencies(manifest, "release-evaluation.json", "report.json") == ["300"]


def test_annual_join_does_not_silently_ignore_missing_submission(tmp_path):
    with pytest.raises(FileNotFoundError):
        downstream.pending_release_dependencies({"directories": {"gc": str(tmp_path)}},
                                                "release-calibration.json", "release-evaluation.json")
