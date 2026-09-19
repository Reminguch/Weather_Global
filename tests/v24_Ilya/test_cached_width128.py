"""Verify the requested scientific arms and gated cached-only submissions."""
import json

import pytest

from scripts.experiments import run_v24_cached_pair as pair
from scripts.experiments import run_v24_cached_width128 as sweep


def measurements(seconds=7):
    return {"validation_seconds": 240., "backends": {
        "cached_4cpu": {"peak_host_gib": 40., "p90_end_to_end_seconds": seconds,
                        "startup_seconds": 700., "checkpoint_seconds": 3.}}}


def test_arms_preserve_fresh_reference_except_requested_architecture_and_cache_protocol():
    reference = json.loads(sweep.REFERENCE.read_text())
    for _, di, stateful in sweep.ARMS:
        config = sweep.arm_config(reference, di=di, stateful=stateful)
        assert config["optimizer"] == reference["optimizer"]
        assert config["architecture"]["residual_width"] == 128
        assert config["architecture"]["residual_initialization"] == "fresh"
        assert config["architecture"]["temporal_d_inner"] == di
        assert config["architecture"]["temporal_stateful"] is stateful
        assert config["architecture"]["temporal_bc_groups"] == 1
        assert config["sequence"]["feedback_mode"] == "baseline"
        assert config["validation"]["num_segments"] is None
    assert [item[1:] for item in sweep.ARMS] == [(32, True), (64, True), (16, False)]


def test_budget_counts_twenty_thousand_updates_and_actual_cadence():
    schedule = {"max_steps": 20000, "checkpoint_every": 1000, "validation_every": 2000}
    budget = pair.measured_budget(measurements(), "cached", schedule)
    assert budget["time_minutes"] == 3240
    assert pair.job_qos(budget["time_minutes"]) == "gpu-medium"


def test_long_trajectory_splits_at_checkpoints_and_preserves_all_steps():
    manifest = {"production_schedule": {"max_steps": 20000, "checkpoint_every": 1000, "validation_every": 2000}}
    chunks = pair.cached_production_chunks(manifest, measurements(50))
    assert len(chunks) > 1
    assert chunks[-1][0] == 20000
    assert all(stop % 1000 == 0 and budget["time_minutes"] <= 8640 for stop, budget in chunks)
    assert [stop for stop, _ in chunks] == sorted(set(stop for stop, _ in chunks))
    with pytest.raises(ValueError, match="144-hour"):
        pair.job_qos(8641)


def test_cached_only_production_has_no_online_job_and_chains_native_continuations(tmp_path, monkeypatch):
    manifest = {"experiment_root": str(tmp_path), "source_root": str(tmp_path / "source"),
                "source_digest": "source", "production_backends": ["cached"],
                "production_schedule": {"max_steps": 20000, "checkpoint_every": 1000, "validation_every": 2000}}
    monkeypatch.setattr(pair, "check_gates", lambda _: {"passed": True})
    monkeypatch.setattr(pair, "require_report", lambda *args: measurements(50))
    monkeypatch.setattr(pair, "require_resource_check", lambda *args: {"passed": True})
    calls = []
    monkeypatch.setattr(pair, "submit_command", lambda args: calls.append(args) or str(100 + len(calls)))
    result = pair.submit_stage(manifest, "production", dependency="99")
    assert len(calls) > 1
    assert "--dependency=afterok:99" in calls[0]
    assert calls[0][-1] == "0"
    for index, command in enumerate(calls[1:], start=1):
        assert f"--dependency=afterok:{100 + index}" in command
        assert command[-1] == "1"
    assert all(launch["backend"] == "cached" for launch in result["launches"].values())
    assert "online" not in result["jobs"] and "finalize" not in result["jobs"]
    pair.submit_stage(manifest, "production", dependency="99")
    assert len(calls) == len(result["jobs"])
