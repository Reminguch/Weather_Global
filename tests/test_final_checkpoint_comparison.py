"""Final-endpoint staging, immutable sources and actual raw/SWA contracts."""
from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts.experiments import prepare_v24_final_checkpoint_comparison as campaign
from scripts.experiments.frozen_runtime import FORMAT, _digest, file_sha256, verify_source


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def training_fixture(tmp_path):
    training = tmp_path / "training"
    for family in ("gc_mamba", "full_gc"):
        tasks = []
        for index, weight in enumerate((0, .25, .5, .75, 1)):
            identity = f"{family}_{index}"
            actual = identity + ("_lr3em7_override" if family == "full_gc" and index == 0 else "")
            run = training / "runs" / actual
            config = {"model_family": family, "optimizer": {"max_steps": 2500}}
            for path in (training / "configs" / f"{actual}.json", run / "run_config.json"):
                write_file(path, json.dumps(config))
            for step in (1000, 1250):
                write_file(run / "checkpoints" / f"checkpoint_step{step:08d}.pkl", f"{actual}_{step}")
            campaign.seal(training / "checks" / f"{actual}_step00001250.json", {
                "status": "complete", "completed_steps": 1250, "task_id": actual,
                "model_family": family,
                "checkpoint": campaign.pinned(run / "checkpoints/checkpoint_step00001250.pkl")})
            tasks.append({"id": actual, "logical_id": identity, "model_family": family,
                "lambda": weight, "learning_rate": 3e-7 if actual.endswith("override") else 3e-5,
                "config": campaign.pinned(training / "configs" / f"{actual}.json")})
        campaign.seal(training / f"selected_arms_{family}.json", {"tasks": tasks})
    return training


def test_preparation_pins_midpoints_but_defers_final_checkpoint_hashes(tmp_path):
    training = training_fixture(tmp_path)
    specs, inputs = campaign.training_specs(tmp_path, training)
    assert len(specs) == 10 and len({s["id"] for s in specs}) == 10
    assert specs[5]["training_run_id"].endswith("lr3em7_override")
    assert specs[5]["learning_rate"] == 3e-7
    for spec in specs:
        assert set(spec["checkpoint_paths"]) == {"1000", "1500", "2000", "2500"}
        assert not Path(spec["checkpoint_paths"]["2500"]).exists()
        assert spec["available_sources"] == [campaign.pinned(spec["checkpoint_paths"]["1000"])]
    for record in inputs:
        campaign.check(record)
    # A changed identity, even if the checkpoint itself remains, is an error.
    write_file(Path(specs[0]["run_config"]["path"]), "{}")
    with pytest.raises(ValueError, match="Pinned input changed"):
        campaign.check(specs[0]["run_config"])


def final_fixture(tmp_path, identity="gcm_0"):
    spec = {"id": identity, "training_run_id": identity, "lambda": 0,
            "model_family": "gc_mamba", "final_proof_path": str(tmp_path / f"{identity}_proof.json"),
            "checkpoint_paths": {}}
    for step in campaign.SWA_STEPS:
        path = tmp_path / identity / f"checkpoint_step{step:08d}.pkl"
        write_file(path, f"checkpoint_{step}")
        spec["checkpoint_paths"][str(step)] = str(path)
    metrics = tmp_path / identity / "final_metrics.jsonl"
    write_file(metrics, '{"finite": true, "loss_increased": true}\n')
    proof = {"status": "complete", "completed_steps": 2500, "task_id": identity,
        "model_family": "gc_mamba", "lambda": 0, "manifest_hash": "training-hash",
        "source_digest": "training-source", "checkpoint": campaign.pinned(spec["checkpoint_paths"]["2500"]),
        "metrics": campaign.pinned(metrics), "scientific_improvement_required": False}
    campaign.seal(spec["final_proof_path"], proof)
    plan = {"training_manifest_hash": "training-hash", "training_source_digest": "training-source"}
    return plan, spec, proof


def test_final_sources_requires_2500_and_identity_but_not_descent(tmp_path):
    plan, spec, proof = final_fixture(tmp_path)
    sources, records = campaign.final_sources(plan, spec)
    assert len(sources) == 4 and sources[-1] == proof["checkpoint"]
    assert records[-1] == proof["metrics"]
    changed = deepcopy(proof)
    changed["completed_steps"] = 1250
    Path(spec["final_proof_path"]).unlink()
    campaign.seal(spec["final_proof_path"], changed)
    with pytest.raises(ValueError, match="Final training completion"):
        campaign.final_sources(plan, spec)


def test_final_sources_rejects_changed_final_checkpoint(tmp_path):
    plan, spec, _ = final_fixture(tmp_path)
    Path(spec["checkpoint_paths"]["2500"]).write_text("replaced endpoint")
    with pytest.raises(ValueError, match="Final training checkpoint differs"):
        campaign.final_sources(plan, spec)


def test_materialization_checks_all_endpoints_before_building_swa(tmp_path, monkeypatch):
    plan, first, _ = final_fixture(tmp_path, "first")
    _, second, _ = final_fixture(tmp_path, "second")
    Path(second["final_proof_path"]).unlink()
    plan["specs"] = [first, second]
    monkeypatch.setattr(campaign, "verify_plan", lambda root: plan)
    monkeypatch.setattr(campaign, "build_swa", lambda *args: pytest.fail("Do not build before all endpoints exist"))
    with pytest.raises(FileNotFoundError):
        campaign.materialize(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("family", ["gc_mamba", "full_gc"])
def test_real_swa_uses_four_prescribed_steps_and_zero_state(tmp_path, family):
    from src.models.mamba.v24_Ilya.checkpoint import training_checkpoint_payload, load_v24_Ilya_checkpoint
    root = tmp_path / "campaign"
    (root / "swa").mkdir(parents=True)
    (root / "checks").mkdir()
    sources = []
    for step in campaign.SWA_STEPS:
        payload = training_checkpoint_payload(completed_step=step,
            residual_params={"module": {"w": np.array([step, 2 * step], np.float32)}},
            residual_state={} if family == "full_gc" else {"module": {"s": np.ones(2, np.float32)}},
            optimizer_state={}, rng_key=np.array([0, 22], np.uint32),
            training_cursor={"epoch": 0, "segment_index": 0, "segment_offset": 0},
            resolved_training_config={"model_family": family}, baseline_checkpoint_path="baseline.npz",
            baseline_checkpoint_fingerprint="baseline-hash", anchor_manifest_fingerprint="anchors",
            parameter_overlay_metadata={})
        path = tmp_path / f"step{step}.pkl"
        path.write_bytes(pickle.dumps(payload))
        sources.append(campaign.pinned(path))
    plan = {"record_hash": "plan", "manifest_template": {"baseline": {"sha256": "baseline-hash"}}}
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps({"model_family": family}))
    spec = {"id": "arm", "model_family": family, "run_config": campaign.pinned(config_path)}
    record, proof_record = campaign.build_swa(root, plan, spec, sources)
    loaded = load_v24_Ilya_checkpoint(Path(record["path"]))
    assert loaded.model_family == family and not loaded.has_residual_state
    np.testing.assert_array_equal(loaded.residual_params["module"]["w"], [1750, 3500])
    proof = campaign.unseal(proof_record["path"])
    assert proof["source_steps"] == [1000, 1500, 2000, 2500]
    assert proof["weights"] == [.25] * 4 and proof["residual_state_policy"] == "zero"
    assert campaign.build_swa(root, plan, spec, sources) == (record, proof_record)
    sources[-1] = {**sources[-1], "sha256": "different"}
    with pytest.raises(ValueError, match="different inputs"):
        campaign.build_swa(root, plan, spec, sources)


def test_freezing_keeps_parent_computation_and_does_not_change_parent(tmp_path):
    workspace, parent, destination = [tmp_path / name for name in ("workspace", "parent", "destination")]
    files = {"src/forecast.py": "frozen prediction", "third_party/graphcast/graphcast/graphcast.py": "frozen GC"}
    for relative, text in files.items():
        write_file(parent / relative, text)
        write_file(workspace / relative, "unrelated workspace edit")
    hashes = {name: file_sha256(parent / name) for name in files}
    campaign.write(parent / "snapshot.json", {"format": FORMAT, "workspace": str(workspace),
        "files": hashes, "source_digest": _digest(hashes)})
    before = (parent / "snapshot.json").read_bytes()
    for relative in campaign.OVERLAYS:
        write_file(workspace / relative, "new orchestration/report")
    snapshot = campaign.freeze_extension(workspace, parent, destination)
    assert (parent / "snapshot.json").read_bytes() == before
    assert verify_source(parent)["source_digest"] == _digest(hashes)
    assert snapshot["files"]["src/forecast.py"] == hashes["src/forecast.py"]
    for relative in campaign.OVERLAYS:
        assert (destination / relative).read_bytes() == (workspace / relative).read_bytes()


def test_submission_preserves_training_and_reuses_m1_without_throttle(tmp_path):
    plan = {"workspace": "/workspace", "source_root": "/snapshot",
        "training_dependencies": ["17049163", "17062358"],
        "resources": {"materialize": {"cpus": 2, "mem_gib": 8, "minutes": 30},
                      "evaluate": {"cpus": 4, "mem_gib": 16, "minutes": 62},
                      "report": {"cpus": 2, "mem_gib": 4, "minutes": 15}}}
    commands = campaign.submission_plan(tmp_path, plan)
    assert "--dependency=afterok:17049163:17062358" in commands["materialize"]
    assert "--dependency=afterok:<materialize-job>" in commands["evaluate"]
    assert "--dependency=afterok:<evaluate-job>" in commands["report"]
    assert "--array=2-21" in commands["evaluate"]
    assert all("%" not in arg for cmd in commands.values() for arg in cmd if arg.startswith("--array="))
    assert not any(arg.startswith("--gres=") for arg in commands["materialize"] + commands["report"])
    assert set(commands) == {"materialize", "evaluate", "report"}
