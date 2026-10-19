"""Release advisory recoveries without confusing numerical drift with integrity."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.experiments import run_v24_cached_pair as pair
from src.models.mamba.v24_Ilya.training.cached_checkpoint import require_production_gates
from src.models.mamba.v24_Ilya.training.cached_diagnostics import (
    logical_dataset_digest, comparison_integrity, tree_comparison,
)


def test_logical_raw_identity_ignores_representation_but_detects_changed_weather():
    source = xr.Dataset({"weather": (("lat", "lon"), np.array([[1., 2.]], np.float32))},
                        coords={"lat": np.array([0.], np.float32), "lon": np.array([1., 2.], np.float32)})
    reconstructed = source.astype(np.float64).assign_coords(lat=source.lat.astype(np.float64), lon=source.lon.astype(np.float64))
    reconstructed.attrs["producer"] = "different metadata"
    assert logical_dataset_digest(source) == logical_dataset_digest(reconstructed.transpose("lon", "lat"))
    reconstructed["weather"].values[0, 1] += 1e-6
    assert logical_dataset_digest(source) != logical_dataset_digest(reconstructed)
    reconstructed["weather"].values[0, 1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite raw"):
        logical_dataset_digest(reconstructed)


def test_numerical_drift_is_separate_from_shape_and_nonfinite_failures():
    drift = tree_comparison(np.array([1.1]), np.array([1.]))
    assert not drift["passed"] and comparison_integrity(drift)
    assert not comparison_integrity(tree_comparison(np.array([np.nan]), np.array([1.])))
    assert not comparison_integrity(tree_comparison(np.ones(2), np.ones(1)))


def test_advisory_release_ignores_missing_and_updated_diagnostic_reports(tmp_path):
    provenance = {"source": "frozen", "inputs": "pinned"}
    integrity = tmp_path / "integrity.json"
    value = {"kind": "cached_integrity", "numerical_policy": "advisory", "integrity_passed": True,
             "checks": {"raw_data": True, "exact_serialization": True}, "provenance": provenance}
    integrity.write_text(json.dumps(value))
    missing = tmp_path / "missing.json"
    first = require_production_gates(missing, missing, provenance, numerical_policy="advisory", integrity_report=integrity)
    missing.write_text(json.dumps({"passed": False, "numeric_error": 0.03}))
    assert require_production_gates(missing, missing, provenance, numerical_policy="advisory", integrity_report=integrity) == first
    with pytest.raises(ValueError, match="gate did not pass"):
        require_production_gates(missing, missing, provenance)
    value["checks"]["raw_data"] = False
    integrity.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity checks failed"):
        require_production_gates(missing, missing, provenance, numerical_policy="advisory", integrity_report=integrity)


def test_advisory_manifest_keeps_real_source_identity_blocking(monkeypatch):
    from src.models.mamba.v24_Ilya.training import cached_data
    monkeypatch.setattr(cached_data.baseline_cache, "validate_manifest", lambda _: None)
    saved = {"manifest_sha256": "producer", "compatibility_sha256": "p", "chunks": [1],
             "compatibility": {"generation_code_sha256": {"model.py": "old"}, "baseline_sha256": "same", "raw_files": "same"}}
    consumer = copy.deepcopy(saved)
    consumer["compatibility"]["generation_code_sha256"]["model.py"] = "new"
    consumer["manifest_sha256"] = "consumer"
    cached_data.validate_consumer_manifest(saved, consumer, numerical_policy="advisory")
    assert saved["compatibility"]["generation_code_sha256"]["model.py"] == "old"
    with pytest.raises(ValueError, match="source/configuration/code"):
        cached_data.validate_consumer_manifest(saved, consumer)
    consumer["compatibility"]["baseline_sha256"] = "wrong"
    with pytest.raises(ValueError, match="raw data/model"):
        cached_data.validate_consumer_manifest(saved, consumer, numerical_policy="advisory")


def measurements():
    record = {"peak_host_gib": 20., "p90_end_to_end_seconds": 50., "startup_seconds": 700., "checkpoint_seconds": 3.}
    return {"numerical_policy": "advisory", "prepared_working_set_gib": 239., "validation_seconds": 240.,
            "backends": {"cached_4cpu": record, "online": record}}


def test_full_working_set_is_included_in_advisory_production_memory():
    assert pair.measured_budget(measurements(), "cached")["memory_gib"] == 336


def test_advisory_production_chunks_both_backends_and_finalizes_without_diagnostics(tmp_path, monkeypatch):
    manifest = {"experiment_root": str(tmp_path), "source_root": str(tmp_path / "source"),
                "source_digest": "source", "numerical_policy": "advisory", "production_backends": ["online", "cached"],
                "production_schedule": {"max_steps": 20000, "checkpoint_every": 1000, "validation_every": 2000}}
    monkeypatch.setattr(pair, "check_gates", lambda _: {"integrity": "immutable"})
    monkeypatch.setattr(pair, "require_report", lambda *args: measurements())
    monkeypatch.setattr(pair, "require_resource_check", lambda *args: {"integrity_passed": True})
    calls = []
    monkeypatch.setattr(pair, "submit_command", lambda args: calls.append(args) or str(100 + len(calls)))
    result = pair.submit_stage(manifest, "production", dependency="99")
    for backend in ("online", "cached"):
        launches = [record for record in result["launches"].values() if record["backend"] == backend]
        assert len(launches) > 1
        assert launches[0]["resume_latest"] is False
        assert launches[-1]["stop_step"] == 20000
    assert result["jobs"]["finalize"]
    assert "--dependency=afterok:" in " ".join(calls[-1])
    assert not any("diagnostics" in " ".join(call) for call in calls)
    pair.submit_stage(manifest, "production", dependency="99")
    assert len(calls) == len(result["jobs"])


def test_final_report_accepts_numeric_drift_and_missing_diagnostics_but_rejects_nonfinite(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.models.mamba.v24_Ilya import checkpoint
    from src.models.mamba.v24_Ilya.training import cached_report, cached_checkpoint, config
    monkeypatch.setattr(cached_report, "FINAL_STEP", 2)
    monkeypatch.setattr(cached_report, "VALIDATION_STEPS", (1, 2))
    monkeypatch.setattr(cached_report, "REPLAY_STEPS", (1, 2))
    provenance = {"frozen": "same"}
    monkeypatch.setattr(cached_checkpoint, "build_provenance", lambda *args, **kwargs: provenance)
    monkeypatch.setattr(config, "load_training_config", lambda *args: {})
    scientific = {"optimizer": {"learning_rate_schedule": "constant"}}
    monkeypatch.setattr(checkpoint, "load_v24_Ilya_training_checkpoint", lambda path:
        SimpleNamespace(completed_step=int(Path(path).stem.removeprefix("checkpoint_step")), resolved_training_config=scientific))
    shared = tmp_path / "shared.pkl"
    shared.write_bytes(b"shared")
    configs = {}
    for name in ("online", "cached"):
        directory = tmp_path / "runs" / name
        (directory / "checkpoints").mkdir(parents=True)
        for step in (0, 1, 2):
            cached_report.checkpoint_path(directory, step).write_bytes(b"checkpoint")
        initial = cached_report.checkpoint_path(directory, 0)
        (directory / "shared_initialization.json").write_text(json.dumps({"shared_init_sha256": cached_report._sha256(shared),
            "projected_checkpoint_sha256": cached_report._sha256(initial)}))
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({**scientific, "output": {"output_root": str(tmp_path / "runs"), "run_name": name}}))
        configs[name] = str(path)
        (directory / "train_metrics.jsonl").write_text("\n".join(json.dumps({"step": step, "loss": 1., "gradient_norm": 1.}) for step in (1, 2)))
        records = []
        for step in (1, 2):
            records.append({"step": step, "role": "fixed_checkpoint", "loss": 1.,
                "validation_protocol": cached_report.VALIDATION_PROTOCOL, "eval_feedback": "baseline",
                "selected_segments": 15, "available_segments": 15, "num_chunks": 60, "num_anchors": 1440,
                "subset_policy": "all_complete_segments", "physical_metric_precision": "fp32", "training_loss_precision": "bf16",
                "subset_fingerprint": "raw", "cache_manifest_sha256": "cache", "segment_ids": list(range(15)),
                "original_graphcast_loss": {"baseline_rollout": 10., "full_rollout": 4. if name == "online" else 5.,
                                            "improvement_pct_rollout": 60. if name == "online" else 50.}})
        (directory / "validation_metrics.jsonl").write_text("\n".join(map(json.dumps, records)))
    reports = tmp_path / "reports"
    reports.mkdir()
    integrity = {"kind": "cached_integrity", "numerical_policy": "advisory", "integrity_passed": True,
                 "checks": {"raw_data": True}, "provenance": provenance}
    (reports / "integrity.json").write_text(json.dumps(integrity))
    (reports / "benchmark.json").write_text(json.dumps({"passed": True, "provenance": provenance}))
    manifest = {"format": "v24_cached_pair_v1", "numerical_policy": "advisory", "configs": configs,
                "shared_init": str(shared), "cache_root": str(tmp_path / "cache"), "source_root": str(tmp_path / "source"),
                "source_digest": "source", "execution": {"expected_manifest_sha256": "cache"},
                "reports": {name: str(reports / (name + ".json")) for name in ("integrity", "benchmark", "full_parity", "paired20")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = cached_report.build_report(tmp_path, collect_accounting=False, make_plots=False)
    assert result["passed"] and result["integrity_passed"]
    assert result["status"] == "complete_with_warnings"
    assert not result["numerical_checks"]["final_quality_within_limits"]
    assert not result["numerical_checks"]["full_parity"]
    path = tmp_path / "runs/cached/train_metrics.jsonl"
    path.write_text(path.read_text().replace('"loss": 1.0', '"loss": NaN'))
    result = cached_report.build_report(tmp_path, collect_accounting=False, make_plots=False)
    assert not result["passed"] and result["status"] == "failed_integrity"
