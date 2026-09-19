"""Checkpoint continuity and production-gate tests without model compilation."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.models.mamba.v24_Ilya.checkpoint import atomic_json_dump, atomic_pickle_dump, load_v24_Ilya_training_checkpoint, training_checkpoint_payload
from src.models.mamba.v24_Ilya.training import cached_checkpoint as ck
from src.models.mamba.v24_Ilya.training.config import load_training_config
from scripts.experiments import run_v24_cached_pair as pair


@pytest.fixture
def config(tmp_path):
    value = load_training_config(Path("configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json"))
    return dataclasses.replace(value, output_root=tmp_path / "runs", run_name="cached",
                               max_steps=4000, checkpoint_every=500, learning_rate_schedule="constant",
                               end_learning_rate=None, warmup_steps=0)


@pytest.fixture
def shared(tmp_path, config):
    source_config = dataclasses.replace(config, output_root=tmp_path / "shared", run_name="canonical")
    path = tmp_path / "shared.pkl"
    atomic_pickle_dump(training_checkpoint_payload(
        completed_step=0, residual_params={"module": {"w": np.asarray([1., 2.], np.float32)}},
        residual_state={"memory": {"state": np.asarray([3.], np.float32)}},
        optimizer_state=(np.asarray(0, np.int32), np.asarray([4.], np.float32)),
        rng_key=np.asarray([6, 7], np.uint32), training_cursor={"epoch": 0, "segment_index": 0, "segment_offset": 0},
        resolved_training_config=source_config.to_dict(), baseline_checkpoint_path=str(config.baseline_checkpoint),
        baseline_checkpoint_fingerprint="baseline", anchor_manifest_fingerprint="anchors", parameter_overlay_metadata={}), path)
    return path


def test_projection_preserves_native_numerical_state_and_detects_modification(shared, config):
    provenance, execution = {"source": "pinned"}, {"backend": "cached_stepwise"}
    target = ck.project_shared_initialization(shared, config, provenance=provenance, execution=execution)
    loaded = ck.validate_cached_checkpoint(target, provenance, execution)
    np.testing.assert_array_equal(loaded.residual_params["module"]["w"], [1., 2.])
    np.testing.assert_array_equal(loaded.residual_state["memory"]["state"], [3.])
    np.testing.assert_array_equal(loaded.rng_key, [6, 7])
    assert loaded.resolved_training_config["output"] == config.to_dict()["output"]
    assert ck.project_shared_initialization(shared, config, provenance=provenance, execution=execution) == target
    target.write_bytes(target.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="modified"):
        ck.project_shared_initialization(shared, config, provenance=provenance, execution=execution)


def test_projection_rejects_scientific_changes_but_allows_operational_smoke(shared, config):
    smoke = dataclasses.replace(config, max_steps=20, checkpoint_every=20)
    ck.project_shared_initialization(shared, smoke)
    changed = dataclasses.replace(config, run_name="different", learning_rate=2e-4)
    with pytest.raises(ValueError, match="scientific"):
        ck.project_shared_initialization(shared, changed)


def test_sidecar_rejects_stale_execution_or_provenance(shared, config):
    target = ck.project_shared_initialization(shared, config, provenance={"id": 1}, execution={"backend": "cached_stepwise"})
    with pytest.raises(ValueError, match="provenance/execution"):
        ck.validate_cached_checkpoint(target, {"id": 2}, {"backend": "cached_stepwise"})
    with pytest.raises(ValueError, match="provenance/execution"):
        ck.validate_cached_checkpoint(target, {"id": 1}, {"backend": "cached_sequence"})


def test_latest_resume_does_not_restart_at_step_zero(shared, config):
    ck.project_shared_initialization(shared, config)
    source = load_v24_Ilya_training_checkpoint(shared)
    from src.models.mamba.v24_Ilya.training.data import TrainingCursor
    target = config.run_dir / "checkpoints/checkpoint_step00000020.pkl"
    ck.write_cached_checkpoint(target, config=config, completed_step=20, residual_params=source.residual_params,
        residual_state=source.residual_state, optimizer_state=source.optimizer_state, rng_key=source.rng_key,
        cursor=TrainingCursor(segment_index=5), baseline_fingerprint="baseline", anchor_fingerprint="anchors",
        overlay_metadata={}, provenance={}, execution={})
    assert ck.latest_checkpoint_for_run(config) == target
    marker = config.run_dir / "latest_checkpoint.json"
    atomic_json_dump({"completed_step": 21, "checkpoint": str(target)}, marker)
    with pytest.raises(ValueError, match="inconsistent step"):
        ck.latest_checkpoint_for_run(config)


def test_production_gates_require_twenty_steps_and_pinned_identity(tmp_path):
    provenance = {"code": {"runner": "abc"}, "init": "def"}
    parity, paired = tmp_path / "full.json", tmp_path / "paired.json"
    atomic_json_dump({"kind": "full_parity", "passed": True, "provenance": provenance}, parity)
    atomic_json_dump({"kind": "paired20", "passed": True, "provenance": provenance, "completed_steps": 19}, paired)
    with pytest.raises(ValueError, match="twenty"):
        ck.require_production_gates(parity, paired, provenance)
    atomic_json_dump({"kind": "paired20", "passed": True, "provenance": provenance, "completed_steps": 20}, paired)
    result = ck.require_production_gates(parity, paired, provenance)
    assert set(result) == {"full_parity", "paired20"}
    with pytest.raises(ValueError, match="Stale"):
        ck.require_production_gates(parity, paired, {**provenance, "init": "changed"})


def test_modest_production_budget_requires_measured_overhead():
    report = {"backends": {"cached_4cpu": {"peak_host_gib": 40., "p90_end_to_end_seconds": 7.,
                                        "startup_seconds": 700., "checkpoint_seconds": 3.}}, "validation_seconds": 240.}
    budget = pair.measured_budget(report, "cached")
    assert budget["memory_gib"] == 64
    assert budget["cpus"] == 4
    assert budget["gpu"] == "gpu40&nomig"
    assert budget["time_minutes"] == 690
    report.pop("validation_seconds")
    with pytest.raises(KeyError):
        pair.measured_budget(report, "cached")


def test_submission_is_idempotent_and_stops_at_unresolved_intent(tmp_path, monkeypatch):
    manifest = {"experiment_root": str(tmp_path), "source_root": str(tmp_path / "source"), "source_digest": "source"}
    calls = []
    monkeypatch.setattr(pair, "submit_command", lambda args: calls.append(args) or "12345")
    first = pair.submit_stage(manifest, "smoke")
    assert pair.submit_stage(manifest, "smoke") == first
    assert len(calls) == 1
    assert first["requests"]["smoke"]["memory_gib"] == 128
    assert "--constraint=gpu40&nomig" in calls[0]
    (tmp_path / "submission_smoke.json").unlink()
    (tmp_path / ".submission_smoke_smoke.intent.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Unresolved submission intent"):
        pair.submit_stage(manifest, "smoke")
    assert len(calls) == 1


def test_failed_gate_never_submits_production(tmp_path, monkeypatch):
    manifest = {"experiment_root": str(tmp_path), "source_root": str(tmp_path / "source")}
    monkeypatch.setattr(pair, "check_gates", lambda _: (_ for _ in ()).throw(ValueError("gate failed")))
    monkeypatch.setattr(pair, "submit_command", lambda _: pytest.fail("must not submit"))
    with pytest.raises(ValueError, match="gate failed"):
        pair.submit_stage(manifest, "production")


def test_pair_configuration_holds_science_fixed(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    relative = Path("configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json")
    destination = workspace / relative
    destination.parent.mkdir(parents=True)
    destination.write_text(relative.read_text())
    cache = tmp_path / "cache"
    cache.mkdir()
    atomic_json_dump({"manifest_sha256": "pinned"}, cache / "manifest.json")
    atomic_json_dump({"manifest_sha256": "pinned", "partial": False}, cache / "READY.json")
    monkeypatch.setattr(pair, "freeze_source", lambda *args: {"source_digest": "source"})
    manifest = pair.prepare_experiment(tmp_path / "experiment", cache_root=cache, workspace=workspace)
    configs = {name: load_training_config(Path(path)) for name, path in manifest["configs"].items()}
    assert all(ck.scientific_config(value) == ck.scientific_config(configs["online"]) for value in configs.values())
    assert configs["online"].max_steps == configs["cached"].max_steps == 4000
    assert configs["online"].learning_rate == 1e-4 and configs["online"].warmup_steps == 0
    assert configs["online"].validation.num_segments is None
    assert configs["online"].validation.every_steps == 500
    assert configs["smoke_cached"].max_steps == 20 and not configs["smoke_cached"].validation.enabled
    assert manifest["execution"]["expected_manifest_sha256"] == "pinned"


def test_final_report_depends_on_both_production_jobs(tmp_path, monkeypatch):
    benchmark_path = tmp_path / "benchmark.json"
    measurements = {"peak_host_gib": 40., "p90_end_to_end_seconds": 7., "startup_seconds": 700., "checkpoint_seconds": 3.}
    atomic_json_dump({"passed": True, "provenance": {"pinned": True}, "validation_seconds": 240.,
                     "backends": {"online": measurements, "cached_4cpu": measurements}}, benchmark_path)
    preflight_path = tmp_path / "preflight.json"
    atomic_json_dump({"passed": True, "provenance": {"pinned": True},
                     "backends": {"online": measurements}}, preflight_path)
    manifest = {"experiment_root": str(tmp_path), "source_root": str(tmp_path / "source"),
                "source_digest": "source", "reports": {"benchmark": str(benchmark_path), "preflight": str(preflight_path)}}
    monkeypatch.setattr(pair, "check_gates", lambda _: {"passed": True})
    monkeypatch.setattr(pair, "experiment_provenance", lambda _: {"pinned": True})
    monkeypatch.setattr(pair, "require_resource_check", lambda *args: {"passed": True})
    calls = []
    monkeypatch.setattr(pair, "submit_command", lambda args: calls.append(args) or str(100 + len(calls)))
    record = pair.submit_stage(manifest, "production")
    assert record["jobs"] == {"online": "101", "cached": "102", "finalize": "103"}
    assert "--dependency=afterok:101:102" in calls[2]
    assert "--cpus-per-task=4" in calls[1]
    assert len(calls) == 3
    pair.submit_stage(manifest, "production")
    assert len(calls) == 3


def test_resource_check_requires_actual_allocation_and_full_validation(tmp_path, monkeypatch):
    report = {"completed_steps": 70, "full_validation_completed": True,
              "allocation": {"memory_gib": 64, "cpus": 4}}
    monkeypatch.setattr(pair, "require_report", lambda *args: report)
    assert pair.require_resource_check({}, {"memory_gib": 64}) == report
    with pytest.raises(ValueError, match="allocation"):
        pair.require_resource_check({}, {"memory_gib": 80})
    report["full_validation_completed"] = False
    with pytest.raises(ValueError, match="full cached validation"):
        pair.require_resource_check({}, {"memory_gib": 64})


def test_resume_archives_uncheckpointed_metrics_before_replay(tmp_path):
    path = tmp_path / "train_metrics.jsonl"
    path.write_text('{"step": 20, "loss": 1}\n{"step": 21, "loss": 2}\n{"step":')
    assert ck.reconcile_run_metrics(tmp_path, 20) == {"train_metrics.jsonl": 2}
    assert path.read_text() == '{"step": 20, "loss": 1}\n'
    archives = list((tmp_path / "recovery").iterdir())
    assert len(archives) == 1
    assert '"step": 21' in archives[0].read_text()
    assert ck.reconcile_run_metrics(tmp_path, 20) == {}


def test_existing_cached_run_requires_explicit_resume(config, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.models.mamba.v24_Ilya import checkpoint as native_checkpoint
    from src.models.mamba.v24_Ilya.training import cached_config, cached_diagnostics, cached_runner
    monkeypatch.setattr(cached_diagnostics, "require_matched_gpu_environment", lambda: {}, raising=False)
    monkeypatch.setattr(cached_config, "load_cached_training_config", lambda *args, **kwargs:
                        SimpleNamespace(common=config, execution={"backend": "cached_stepwise"}))
    monkeypatch.setattr(cached_runner, "build_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(cached_runner, "latest_checkpoint_for_run", lambda *args: tmp_path / "existing.pkl")
    monkeypatch.setattr(native_checkpoint, "load_v24_Ilya_training_checkpoint", lambda *args: SimpleNamespace(completed_step=20))
    with pytest.raises(ValueError, match="explicit --resume"):
        cached_runner.run_cached_training(tmp_path / "config.json", tmp_path / "shared.pkl",
                                         cache_root=tmp_path, max_steps=20)


def test_auto_advance_is_opt_in_pinned_and_afterok_current_job(tmp_path, monkeypatch):
    config_path = tmp_path / "cached.json"
    config_path.write_text("{}")
    manifest = {"experiment_root": str(tmp_path), "source_digest": "source", "configs": {"cached": str(config_path)}}
    calls = []
    monkeypatch.setattr(pair, "submit_stage", lambda manifest, stage, **kwargs:
                        calls.append((stage, kwargs)) or {"jobs": {stage: "567"}})
    assert pair.advance_pipeline(manifest, "smoke") is None
    pair.enable_automation(manifest)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    pair.advance_pipeline(manifest, "smoke")
    assert calls == [("gates", {"dependency": "123"})]
    assert json.loads((tmp_path / "pipeline_status.json").read_text())["jobs"] == {"gates": "567"}
    config_path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="identity changed"):
        pair.advance_pipeline(manifest, "gates")
    assert len(calls) == 1
