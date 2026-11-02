"""Actual runner exact resume, operational stop and active-log boundaries."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.models.mamba.v24_Ilya.checkpoint import load_v24_Ilya_training_checkpoint
from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training import runner
from src.models.mamba.v24_Ilya.training.config import V24IlyaTrainConfig, V24IlyaTrainInvocation
from src.models.mamba.v24_Ilya.training.data import TrainingCursor
from src.models.mamba.v24_Ilya.training.mgda_step import MultiObjectiveResult
from src.models.mamba.v24_Ilya.training.power_loss import make_power_calibration


def mock_run(tmp_path, monkeypatch, *, full_feedback=False, seen_states=None):
    config = V24IlyaTrainConfig(prepared_root=tmp_path / "prepared", anchor_manifest_root=tmp_path / "anchors",
        baseline_checkpoint=tmp_path / "baseline.npz", output_root=tmp_path / "runs", run_name="mgda",
        architecture=V24IlyaArchitectureConfig(residual_initialization="fresh"), input_duration=None,
        segment_steps=4, bptt_steps=2, ar_tail_k=1, feedback_mode="closed_loop_sg", loss_mode="all_steps",
        loss_kind="mse_fp32", optimizer_kind="mgda_rms", max_steps=6, checkpoint_every=2, seed=22,
        learning_rate_schedule="warmup_cosine_constant", learning_rate=1e-5, end_learning_rate=3e-6,
        warmup_steps=2, decay_end_step=4, weight_decay=0,
        power_calibration=make_power_calibration(np.ones((9, 2)), 1., 1.))
    if full_feedback:
        config = replace(config, segment_steps=96, bptt_steps=24, ar_tail_k=20,
                         feedback_mode="closed_loop", objective_pair="mse_amse_excess",
                         loss_kind="amse", power_calibration=None, mgda_backtracking={})
    config.baseline_checkpoint.write_bytes(b"baseline fixture")
    params = {"module": {"w": jnp.array([.2, .4], jnp.float32)}}
    state = {"s": jnp.zeros((1, 2), jnp.float32)}
    baseline = SimpleNamespace(task_config=None, model_config=None, params=params)
    transform = SimpleNamespace(init=lambda *args: (params, state))
    transforms = SimpleNamespace(residual_loss=transform, residual_predict=transform, baseline_predict=transform)
    @dataclass
    class DataReport:
        frames: int = 1

    def chunk(cursor, *args):
        offset = cursor.segment_offset + config.bptt_steps
        next_cursor = (TrainingCursor(segment_index=cursor.segment_index + 1)
                       if offset == config.segment_steps else
                       TrainingCursor(segment_index=cursor.segment_index, segment_offset=offset))
        return SimpleNamespace(input_frames=(), static_inputs=None, truths=(), forcings=(),
                               data_report=DataReport(), next_cursor=next_cursor)
    data = SimpleNamespace(anchor_indices=np.array([0]), train_split=np.array([0]), val_split=np.array([1]),
        input_steps=2, time_step=np.timedelta64(6, "h"), segments=[0], validation_segments=[1],
        manifest_fingerprint="anchor-fixture", build_chunk=chunk, validate_cursor=lambda *args: None,
        store=SimpleNamespace(build_batch_from_indices=lambda **kw: (None, None, None), selection_metadata={}))
    monkeypatch.setattr(runner, "validate_input_paths", lambda config: None)
    monkeypatch.setattr(runner, "load_graphcast_checkpoint", lambda path: baseline)
    monkeypatch.setattr(runner, "load_stats", lambda path: None)
    monkeypatch.setattr(runner, "validate_stats_coverage", lambda *args: None)
    monkeypatch.setattr(runner, "open_training_data", lambda *args: data)
    monkeypatch.setattr(runner, "build_model_configs", lambda *args: None)
    monkeypatch.setattr(runner, "build_training_transforms", lambda *args: transforms)
    monkeypatch.setattr(runner, "cast_physical_boundary_fp32", lambda value: value)
    monkeypatch.setattr(runner, "memory_contract", lambda **kw: {})
    monkeypatch.setattr(runner, "_device_memory_snapshot", lambda: None)

    def make_step(optimizer, **kw):
        def step(p, s, opt_state, keys, *args):
            if seen_states is not None:
                seen_states.append(np.asarray(s["s"]).copy())
            g = jax.tree_util.tree_map(lambda value: 2 * value + jnp.mean(s["s"]) +
                                      .01 * jax.random.normal(keys[0], value.shape), p)
            g2 = jax.tree_util.tree_map(lambda value: .5 * value, g)
            updates, next_opt, diagnostics = optimizer.update_pair(g, g2, opt_state, p)
            next_p = optax.apply_updates(p, updates)
            next_s = {"s": .7 * s["s"] + p["module"]["w"][None] + .01 * jax.random.normal(keys[1], s["s"].shape)}
            diagnostics = dict(diagnostics, gc_loss=jnp.array(1.), power_loss=jnp.array(2.))
            return MultiObjectiveResult(next_p, next_s, next_opt, jnp.array(1.), jnp.array(1.),
                                        jnp.array([1., 1.]), diagnostics)
        return step
    monkeypatch.setattr(runner, "make_train_step", make_step)
    return config


@pytest.mark.parametrize("full_feedback", [False, True])
def test_runner_resume_matches_uninterrupted_and_archives_tail(tmp_path, monkeypatch, full_feedback):
    seen_states = []
    config = mock_run(tmp_path, monkeypatch, full_feedback=full_feedback, seen_states=seen_states)
    invocation = V24IlyaTrainInvocation(config, tmp_path / "config.json")
    final = runner.run_training(invocation)
    expected = load_v24_Ilya_training_checkpoint(final)
    windows_per_segment = config.segment_steps // config.bptt_steps
    for index, state in enumerate(seen_states):
        assert bool(np.any(state)) == bool(index % windows_per_segment)
    expected_rng = jax.random.PRNGKey(config.seed)
    for _ in range(2):  # Residual and frozen-baseline initialization keys.
        expected_rng = jax.random.split(expected_rng)[0]
    for _ in range(config.max_steps):
        expected_rng = jax.random.split(expected_rng, config.bptt_steps + 1)[0]
    np.testing.assert_array_equal(expected.rng_key, expected_rng)
    assert expected.training_cursor == TrainingCursor(
        segment_index=6 // windows_per_segment,
        segment_offset=(6 % windows_per_segment) * config.bptt_steps).to_dict()
    early = config.run_dir / "checkpoints/checkpoint_step00000002.pkl"
    final = runner.run_training(V24IlyaTrainInvocation(config, invocation.config_path, resume=early))
    actual = load_v24_Ilya_training_checkpoint(final)
    for field in ("residual_params", "residual_state", "optimizer_state", "rng_key"):
        for a, b in zip(jax.tree_util.tree_leaves(getattr(expected, field)),
                        jax.tree_util.tree_leaves(getattr(actual, field)), strict=True):
            np.testing.assert_array_equal(a, b)
    assert actual.training_cursor == expected.training_cursor
    assert int(actual.optimizer_state.count) == 6
    metrics = [json.loads(line) for line in (config.run_dir / "train_metrics.jsonl").read_text().splitlines()]
    assert [item["step"] for item in metrics] == list(range(1, 7))
    assert all("gc_loss" in item and "power_loss" in item for item in metrics)
    archives = list((config.run_dir / "resume_archives").glob("*/train_metrics.jsonl"))
    assert len(archives) == 1
    assert len(archives[0].read_text().splitlines()) == 6


@pytest.mark.parametrize("full_feedback", [False, True])
def test_operational_stop_saves_without_changing_schedule_or_budget(tmp_path, monkeypatch, full_feedback):
    config = mock_run(tmp_path, monkeypatch, full_feedback=full_feedback)
    path = runner.run_training(V24IlyaTrainInvocation(config, tmp_path / "config.json", stop_after_steps=3))
    saved = load_v24_Ilya_training_checkpoint(path)
    assert saved.completed_step == 3
    assert saved.resolved_training_config["optimizer"]["max_steps"] == 6
    assert saved.resolved_training_config["optimizer"]["decay_end_step"] == 4
    assert int(saved.optimizer_state.count) == 3


def test_resume_count_mismatch_fails_before_accepting_metrics(tmp_path, monkeypatch):
    import pickle
    config = mock_run(tmp_path, monkeypatch)
    path = runner.run_training(V24IlyaTrainInvocation(config, tmp_path / "config.json", stop_after_steps=3))
    payload = pickle.loads(path.read_bytes())
    payload["optimizer_state"] = payload["optimizer_state"]._replace(count=np.asarray(1, np.int32))
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="update count differs"):
        runner.run_training(V24IlyaTrainInvocation(config, tmp_path / "config.json", resume=path))


def test_calibration_provenance_rejects_wrong_warmstart(tmp_path):
    calibration = {"init_from_sha256": "expected"}
    config = SimpleNamespace(power_calibration=calibration)
    with pytest.raises(ValueError, match="init_from_sha256"):
        runner._validate_calibration_provenance(config, "baseline", "anchors", {"init_from_sha256": "wrong"})


def test_archive_handles_partial_uncommitted_json(tmp_path):
    path = tmp_path / "train_metrics.jsonl"
    original = '{"step": 1}\n{"step": 2}\n{"step":'
    path.write_text(original)
    archive = runner._archive_resume_tails(tmp_path, 1)
    assert path.read_text() == '{"step": 1}\n'
    assert (archive / path.name).read_text() == original
