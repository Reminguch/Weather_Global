"""Scientific controls, fresh-state transfer and checkpoint selection."""
import copy
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.experiments import fresh_data_scale_training as training
from src.models.mamba.v24_Ilya import checkpoint
from src.models.mamba.v24_Ilya.training.cached_checkpoint import numerical_checkpoint_identity
from src.models.mamba.v24_Ilya.training.config import load_training_config
from src.models.mamba.v24_Ilya.training.endpoint_step import build_optimizer


CONTROL = training.ROOT / "artifacts/checkpoints/v24_Ilya/res1_cached_w128_di_sweep_recovery_20261007/di64_carry/runs/cached/run_config.json"
CLOSED = training.ROOT / "artifacts/checkpoints/v24_Ilya/res1_cached_w128_closed_loop_k20_20261013/runs/di64/run_config.json"


@pytest.mark.parametrize("width", (128, 512))
def test_open_configs_preserve_actual_historical_optimizer_and_only_change_width_data_output(tmp_path, width):
    original = training.read(CONTROL)
    small = training.open_config(original, tmp_path / "small", "cached", width)
    large = training.open_config(original, tmp_path / "large", "cached", width,
                                 {"prepared_root": "/prepared40", "anchor_manifest_root": "/anchors40"})
    assert small["optimizer"] == original["optimizer"]
    assert small["optimizer"]["warmup_steps"] == 200
    assert small["optimizer"]["checkpoint_every"] == 1000
    assert training.initialization_config_identity(small) == training.initialization_config_identity(large)
    assert small["architecture"]["residual_width"] == width
    small["architecture"]["residual_width"] = 128
    small["output"] = original["output"]
    assert small == training._clean(original)


def test_closed_config_keeps_common_seven_year_data_and_historical_schedule(tmp_path):
    original = training.read(CLOSED)
    updated = training.closed_config(original, tmp_path / "runs", "fresh512", 512)
    assert updated["data"] == original["data"]
    assert updated["optimizer"] == original["optimizer"]
    assert updated["validation"] == original["validation"]
    assert updated["sequence"]["feedback_mode"] == "closed_loop_sg"
    assert updated["architecture"]["residual_initialization"] == "fresh"


def test_previously_incorrect_no_warmup_template_is_rejected(tmp_path):
    value = training.read(CONTROL)
    value["optimizer"]["warmup_steps"] = 0
    with pytest.raises(ValueError, match="Optimizer"):
        training.open_config(value, tmp_path, "cached", 512)


@pytest.fixture
def fresh_source(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    value = training.open_config(training.read(CONTROL), tmp_path / "runs", "cached", 128)
    training.write(config_path, value)
    config = load_training_config(config_path)
    params = {"spatial": {"w": np.ones((2, 2), np.float32)},
              "temporal_residual_head": {"w": np.zeros((2, 2), np.float32)},
              "temporal/out_proj": {"w": np.zeros((2, 2), np.float32)}}
    optimizer, _ = build_optimizer(config)
    source = tmp_path / "source.pkl"
    monkeypatch.setattr(checkpoint, "manifest_sha256", lambda _: "new-anchor-manifest")
    monkeypatch.setattr(checkpoint, "file_sha256", lambda p: "sha:" + Path(p).name)
    payload = checkpoint.training_checkpoint_payload(
        completed_step=0, residual_params=params, residual_state={"memory": np.zeros(2, np.float32)},
        optimizer_state=optimizer.init(params), rng_key=jax.random.PRNGKey(22),
        training_cursor=training.ZERO_CURSOR, resolved_training_config=config.to_dict(),
        baseline_checkpoint_path=str(config.baseline_checkpoint),
        baseline_checkpoint_fingerprint=checkpoint.file_sha256(config.baseline_checkpoint),
        anchor_manifest_fingerprint="old-anchor-manifest", parameter_overlay_metadata={
            "residual_initialization": "fresh", "residual_copied": 0, "residual_fresh": 3,
            "baseline_copied": 10, "baseline_fresh": 0})
    checkpoint.atomic_pickle_dump(payload, source)
    return config_path, config, source, payload


def test_step_zero_rebinding_keeps_every_numerical_component_exact(fresh_source, tmp_path):
    config_path, _, source, _ = fresh_source
    target = training.prepare_init(config_path, tmp_path / "new_init", source)
    old = checkpoint.load_v24_Ilya_training_checkpoint(source)
    new = checkpoint.load_v24_Ilya_training_checkpoint(target)
    assert numerical_checkpoint_identity(old) == numerical_checkpoint_identity(new)
    assert new.anchor_manifest_fingerprint == "new-anchor-manifest"
    assert new.parameter_overlay_metadata["matched_fresh_source_step"] == 0
    assert new.parameter_overlay_metadata["residual_copied"] == 0
    assert "init_from" not in new.parameter_overlay_metadata
    assert training.prepare_init(config_path, tmp_path / "new_init", source) == target


@pytest.mark.parametrize("change", ("trained", "optimizer", "state", "copied", "schedule"))
def test_trained_or_incompatible_initializations_cannot_enter_fresh_arms(fresh_source, tmp_path, change):
    config_path, _, source, payload = fresh_source
    if change == "trained":
        payload["completed_step"] = 10
    elif change == "state":
        payload["residual_state"]["memory"][:] = 1
    elif change == "copied":
        payload["parameter_overlay_metadata"]["residual_copied"] = 1
    elif change == "schedule":
        payload["resolved_training_config"]["optimizer"]["warmup_steps"] = 0
    else:
        payload["optimizer_state"] = jax.tree_util.tree_map(lambda v: v + 1, payload["optimizer_state"])
    checkpoint.atomic_pickle_dump(payload, source)
    with pytest.raises(ValueError):
        training.prepare_init(config_path, tmp_path / "bad_init", source)


def records():
    return [{"role": "fixed_checkpoint", "step": step, "selected_segments": 15, "available_segments": 15,
             "validation_protocol": "cached_stepwise_all_segments_v1", "subset_fingerprint": "all2022",
             "original_graphcast_loss": {"full_rollout": 2. if step in (6000, 14000) else 3.},
             "loss": 10000.-step} for step in training.SELECTION_STEPS]


def test_selection_uses_exact_metric_and_earliest_tie_not_legacy_loss():
    assert training.select_records(records())["step"] == 6000


@pytest.mark.parametrize("change", ("missing", "duplicate", "subset", "segments", "nan"))
def test_selection_rejects_incomplete_or_changed_validation_grid(change):
    rows = records()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "subset":
        rows[-1]["subset_fingerprint"] = "different"
    elif change == "segments":
        rows[-1]["selected_segments"] = 8
    else:
        rows[-1]["original_graphcast_loss"]["full_rollout"] = float("nan")
    with pytest.raises(ValueError):
        training.select_records(rows)


def test_exposure_counts_partial_epochs_and_year_boundaries():
    manifest = {"chunks": [
        {"id": 1, "split": "train", "target_timestamps": ["1982-12-31T18:00", "1983-01-01T00:00"]},
        {"id": 2, "split": "train", "target_timestamps": ["2015-01-01T00:00", "2015-01-01T06:00"]}]}
    result = training.year_exposure([{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 1}], manifest)
    assert result["weather_steps_by_year"] == {"1982": 2, "1983": 2, "2015": 2}
    assert result["chunk_uses_by_first_target_year"] == {"1982": 2, "2015": 1}
    assert result["total_weather_steps"] == 6


def test_workers_refuse_unbounded_weather_reads(monkeypatch):
    monkeypatch.delenv("WEATHER_PREPARED_BOUNDED_READS", raising=False)
    with pytest.raises(ValueError, match="BOUNDED_READS"):
        training._bounded_environment()


def test_advisory_execution_pins_original_manifest_consistently(tmp_path):
    fingerprint = "1" * 64
    training.write(tmp_path / "manifest.json", {"manifest_sha256": fingerprint})
    training.write(tmp_path / "READY.json", {"manifest_sha256": fingerprint, "partial": False})
    kwargs = training.cache_execution(tmp_path)
    assert kwargs == {"cache_root": tmp_path, "prefetch_batches": 0,
                      "expected_manifest_sha256": fingerprint}
    training.write(tmp_path / "READY.json", {"manifest_sha256": "other", "partial": False})
    with pytest.raises(ValueError, match="READY"):
        training.cache_execution(tmp_path)


def test_production_wrapper_passes_legacy_gate_arguments_for_frozen_control_runner(fresh_source, tmp_path, monkeypatch):
    from src.models.mamba.v24_Ilya.training import cached_runner, cached_checkpoint
    config_path, _, source, _ = fresh_source
    monkeypatch.setenv("WEATHER_PREPARED_BOUNDED_READS", "1")
    cache = tmp_path / "cache"
    training.write(cache / "manifest.json", {"manifest_sha256": "1" * 64})
    training.write(cache / "READY.json", {"manifest_sha256": "1" * 64, "partial": False})
    monkeypatch.setattr(cached_checkpoint, "latest_checkpoint_for_run", lambda _: None)
    integrity = tmp_path / "qualification.json"
    class ReachedRunner(Exception):
        pass
    def run(*args, **kwargs):
        assert kwargs["numerical_policy"] == "advisory"
        assert kwargs["parity_report"] == kwargs["paired_report"] == kwargs["integrity_report"] == integrity
        assert kwargs["expected_manifest_sha256"] == "1" * 64
        raise ReachedRunner
    monkeypatch.setattr(cached_runner, "run_cached_training", run)
    with pytest.raises(ReachedRunner):
        training.train_open(config_path, source, cache, tmp_path / "output.json",
                            stop_step=4000, integrity_report=integrity)
