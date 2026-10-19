"""Diagnostic controls and evidence must distinguish causes without releasing jobs."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.experiments import run_v24_recovery_diagnostics as probe


def test_runtime_matrix_changes_one_control_at_a_time():
    assert probe.flags("ordinary") == probe.BASE_FLAGS
    assert probe.flags("deterministic").split()[1:] == ["--xla_gpu_deterministic_ops=true"]
    assert probe.flags("autotune_off").split()[1:] == ["--xla_gpu_autotune_level=0"]
    assert set(probe.flags("deterministic_autotune_off").split()) == {
        probe.BASE_FLAGS, "--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0"}
    with pytest.raises(ValueError):
        probe.flags("unknown")


def test_comparison_records_drift_but_rejects_invalid_evidence():
    report = probe.array_comparison(np.array([1., 2.01]), np.array([1., 2.]))
    assert report["differing_elements"] == 1
    assert report["relative_l2"] > 0
    with pytest.raises(ValueError, match="Nonfinite"):
        probe.array_comparison(np.array([np.nan]), np.array([1.]))
    with pytest.raises(ValueError, match="shapes"):
        probe.array_comparison(np.array([1.]), np.array([1., 2.]))


def test_matrix_uses_same_allocation_for_process_and_gpu_comparisons(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    calls = []
    monkeypatch.setattr(probe, "child", lambda *args, **kwargs: calls.append((args[2], kwargs)))
    probe.matrix(tmp_path, {"source_root": "/frozen"})
    assert len(calls) == 25
    assert [c[1]["device"] for c in calls[:3]] == ["GPU-first", "GPU-first", "GPU-second"]
    assert all("--root" in c[0] for c in calls)
    assert "--instrument" in calls[-1][0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first")
    with pytest.raises(ValueError, match="two allocated"):
        probe.matrix(tmp_path, {"source_root": "/frozen"})


def test_partial_summary_preserves_incompleteness_without_production_dependencies(tmp_path):
    probe.summarize(tmp_path, {})
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["completed_fixed_cases"] == 0
    assert result["expected_fixed_cases"] == 24
    assert result["production_dependencies"] == []
    assert "incomplete" in result["limitations"][0]


def test_nonfinite_record_cannot_be_published(tmp_path):
    with pytest.raises(ValueError):
        probe.write(tmp_path / "invalid.json", {"value": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_summary_distinguishes_same_gpu_and_true_cross_gpu(tmp_path):
    for name, gpu in (("first", "GPU-A"), ("fresh", "GPU-A"), ("other_gpu", "GPU-B")):
        directory = tmp_path / "fixed" / ("ordinary_standalone_" + name)
        report = {"status": "complete", "instrumented": False, "runtime": {"gpu_uuid": gpu},
            "input_hash": "inputs", "parameter_hash": "params", "state_hash": "state",
            "rng_hash": "rng", "hlo_sha256": "hlo"}
        probe.write(directory / "result.json", report)
        np.savez(directory / "baseline_reference.npz", leaf_0=np.array([1.]))
    probe.summarize(tmp_path, {})
    result = probe.read(tmp_path / "summary.json")
    assert len(result["cross_process_gpu"]) == 2
    assert all(x["hardware_design_satisfied"] for x in result["cross_process_gpu"])
    assert result["cross_process_gpu"][0]["same_gpu"] is True
    assert result["cross_process_gpu"][1]["same_gpu"] is False
