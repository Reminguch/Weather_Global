from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "third_party" / "graphcast"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest

from src.models.mamba.v24_Ilya.training.cached_report import (
    REPLAY_PROTOCOL, VALIDATION_PROTOCOL, VALIDATION_STEPS,
    _plot, assess_replay, assess_run, build_report, cache_generation_cost, checkpoint_path, final_quality,
)


def _validation(step=4000, loss=5.0, baseline=10.0):
    return {
        "step": step, "role": "fixed_checkpoint", "loss": 1.25,
        "validation_protocol": VALIDATION_PROTOCOL, "eval_feedback": "baseline",
        "selected_segments": 15, "available_segments": 15,
        "num_chunks": 60, "num_anchors": 1440, "subset_policy": "all_complete_segments",
        "physical_metric_precision": "fp32", "training_loss_precision": "bf16",
        "original_graphcast_loss": {"baseline_rollout": baseline, "full_rollout": loss,
                                    "improvement_pct_rollout": 100 * (1 - loss / baseline)},
    }


@pytest.mark.parametrize("online,cached,expected", [(5.0, 5.04, True), (1.0, 1.011, False), (9.0, 9.06, False)])
def test_exact_loss_and_percentage_point_limits_are_both_enforced(online, cached, expected):
    result = final_quality(_validation(loss=online), _validation(loss=cached))
    assert result["passed"] is expected
    assert result["relative_exact_loss_difference"] == pytest.approx(abs(cached - online) / online)
    assert result["improvement_difference_pp"] == pytest.approx(abs(cached - online) * 10)


def test_quality_rejects_different_baseline_and_nonfinite_values():
    assert not final_quality(_validation(), _validation(baseline=10.01))["passed"]
    assert not final_quality(_validation(), _validation(loss=float("nan")))["passed"]


def _complete_run():
    training = [{"step": step, "loss": 1.0, "gradient_norm": 0.2} for step in range(1, 4001)]
    validation = [_validation(step) for step in VALIDATION_STEPS]
    checkpoints = {str(step): {"valid": True, "completed_step": step} for step in VALIDATION_STEPS}
    return training, validation, checkpoints


def test_completion_requires_all_updates_validations_and_native_checkpoints():
    training, validation, checkpoints = _complete_run()
    assert assess_run(training, validation, checkpoints)["passed"]
    assert not assess_run(training[1:], validation, checkpoints)["passed"]
    assert not assess_run(training, validation[:-1], checkpoints)["passed"]
    assert not assess_run(training, validation + [validation[-1]], checkpoints)["passed"]
    checkpoints["4000"] = {"valid": True, "completed_step": 3500}
    assert not assess_run(training, validation, checkpoints)["passed"]


def test_partial_validation_and_nonfinite_training_never_pass():
    training, validation, checkpoints = _complete_run()
    validation[0]["num_chunks"] = 32
    assert not assess_run(training, validation, checkpoints)["passed"]
    validation[0]["num_chunks"] = 60
    training[200]["loss"] = float("nan")
    assert not assess_run(training, validation, checkpoints)["passed"]


def _replay():
    identity = {"protocol": REPLAY_PROTOCOL, "checkpoint_sha256": "a" * 64, "step": 4000, "run": "cached"}
    chunks = [{
        "passed": True, "data_identity": True, "state_reset": index == 0,
        "cursor": {"segment_index": 0, "segment_offset": index * 24},
        "loss_components": {"passed": True},
        "per_step": [{"step_index": index, **{key: {"passed": True} for key in ("prediction", "state", "loss")}}
                     for index in range(24)],
    } for index in range(2)]
    return {**identity, "passed": True, "initial_state": "zero", "chunks": chunks}, identity


def test_replay_requires_current_weights_complete_steps_and_chunk_carry():
    record, identity = _replay()
    assert assess_replay(record, identity)["passed"]
    stale = {**identity, "checkpoint_sha256": "b" * 64}
    assert not assess_replay(record, stale)["passed"]
    incomplete = copy.deepcopy(record)
    incomplete["chunks"][1]["per_step"].pop()
    assert not assess_replay(incomplete, identity)["passed"]
    reset = copy.deepcopy(record)
    reset["chunks"][1]["state_reset"] = True
    assert not assess_replay(reset, identity)["passed"]
    assert not assess_replay({**identity, "passed": True}, identity)["passed"]


def test_checkpoint_filename_uses_native_eight_digit_steps(tmp_path):
    assert checkpoint_path(tmp_path, 500).name == "checkpoint_step00000500.pkl"
    assert checkpoint_path(tmp_path, 4000).name == "checkpoint_step00004000.pkl"


def test_missing_experiment_outputs_write_a_failed_report(tmp_path):
    configs = {}
    for name in ("online", "cached"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"output": {"output_root": str(tmp_path / "runs"), "run_name": name},
                                   "optimizer": {"learning_rate_schedule": "constant"}}))
        configs[name] = str(path)
    manifest = {
        "format": "v24_cached_pair_v1", "configs": configs,
        "shared_init": str(tmp_path / "missing_shared.pkl"), "cache_root": str(tmp_path / "missing_cache"),
        "source_root": str(tmp_path / "source"), "source_digest": "c" * 64,
        "execution": {"expected_manifest_sha256": "d" * 64},
        "reports": {name: str(tmp_path / "reports" / f"{name}.json") for name in ("full_parity", "paired20", "benchmark")},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = build_report(tmp_path, collect_accounting=False, make_plots=False)
    assert result["passed"] is False
    assert result["status"] == "incomplete_or_failed"
    assert not result["checks"]["all_six_same_checkpoint_replays"]
    assert (tmp_path / "reports/final_report.json").is_file()
    assert "FAIL / MISSING" in (tmp_path / "reports/final_report.md").read_text()


def test_learning_curves_export_png_and_pdf(tmp_path):
    training = {name: [{"step": index, "loss": 2 / index} for index in range(1, 5)] for name in ("online", "cached")}
    validation = {name: [_validation(500), _validation(1000, loss=4.9)] for name in ("online", "cached")}
    _plot(tmp_path / "curves", training, validation)
    assert (tmp_path / "curves.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "curves.pdf").read_bytes().startswith(b"%PDF")


def test_cache_cost_includes_writes_once_and_requires_all_chunks(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"manifest_sha256": "abc", "num_shards": 1, "chunks": [{"id": 0}, {"id": 1}]}))
    directory = tmp_path / "shard_000"
    directory.mkdir()
    marker = {"manifest_sha256": "abc", "initialization_seconds": 5,
              "timings": [{"chunk_id": 0, "seconds": 10, "write_seconds": 2},
                          {"chunk_id": 1, "seconds": 20, "write_seconds": 3}], "files": {}}
    (directory / "COMPLETE.json").write_text(json.dumps(marker))
    result = cache_generation_cost(tmp_path)
    assert result["available"]
    assert result["summed_recorded_worker_seconds"] == 35
    marker["timings"].pop()
    (directory / "COMPLETE.json").write_text(json.dumps(marker))
    assert not cache_generation_cost(tmp_path)["available"]
