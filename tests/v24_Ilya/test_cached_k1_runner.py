"""Exact restart, sample accounting and validation cadence for the k1 runner."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr

from src.models.mamba.v24_Ilya.training import cached_k1_runner as runner
from src.models.mamba.v24_Ilya.training.cached_k1_data import CachedK1Cursor
from src.models.mamba.v24_Ilya.training.config import load_training_config


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    latent_size: int = 512
    gnn_msg_steps: int = 16


class TinySampler:
    """Nine real segments exercise an eight-lane full and one-lane padded group."""

    def __init__(self, cached, *, batch_size, seed, shuffle=True):
        self.batch_size = batch_size
        self.num_segments = 9
        self.chunks_per_segment = 4
        self.groups_per_epoch = (self.num_segments + batch_size - 1) // batch_size
        self.steps_per_epoch = self.groups_per_epoch * 4
        self.seed = seed

    def completed_updates(self, cursor):
        return cursor.epoch * self.steps_per_epoch + cursor.group_index * 4 + cursor.chunk_index

    def cursor_for_update(self, step):
        epoch, inner = divmod(step, self.steps_per_epoch)
        group, chunk = divmod(inner, 4)
        return CachedK1Cursor(epoch, group, chunk)

    def build_batch(self, cursor):
        step = self.completed_updates(cursor)
        valid = min(self.batch_size, self.num_segments - cursor.group_index * self.batch_size)
        mask = np.arange(self.batch_size) < valid
        values = tuple(jnp.arange(self.batch_size, dtype=jnp.float32) * .01 + .02 * (step + i)
                       for i in range(24))
        ids = np.where(mask, cursor.group_index * self.batch_size + np.arange(self.batch_size), -1)
        return SimpleNamespace(inputs=values, targets=values, forcings=values, valid_mask=mask,
            segment_ids=ids, valid_weather_steps=valid * 24,
            reset_state=cursor.chunk_index == 0, next_cursor=self.cursor_for_update(step + 1))


class TinyTrainer:
    """Actual stochastic recurrent gradients/Adam state, without weather graphs."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def initialize(self, key, inputs, targets, forcings):
        return {"module": {"w": jax.random.normal(key, (), dtype=jnp.float32)}}, {
            "memory": {"state": jnp.zeros_like(inputs)}}

    def evaluate_step(self, params, state, key, inputs, targets, forcings, valid_mask):
        mask = jnp.asarray(valid_mask, jnp.float32)
        memory = (.8 * state["memory"]["state"] + params["module"]["w"] * inputs
                  + .001 * jax.random.normal(key, inputs.shape)) * mask
        loss = jnp.sum(jnp.square(memory - targets) * mask) / jnp.sum(mask)
        return loss, {"memory": {"state": memory}}, memory

    def __call__(self, params, state, optimizer_state, keys, inputs, targets, forcings, valid_mask):
        mask = jnp.asarray(valid_mask, jnp.float32)

        def objective(p):
            memory = state["memory"]["state"]
            components = []
            for key, value in zip(keys, inputs):
                memory = (.8 * memory + p["module"]["w"] * value
                          + .001 * jax.random.normal(key, value.shape)) * mask
                components.append(jnp.sum(jnp.square(memory - value) * mask) / jnp.sum(mask))
            return jnp.mean(jnp.stack(components)), (memory, jnp.stack(components))

        (loss, (memory, components)), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, optimizer_state = self.optimizer.update(gradients, optimizer_state, params)
        return (optax.apply_updates(params, updates), {"memory": {"state": memory}},
                optimizer_state, loss, optax.global_norm(gradients), components)


@pytest.fixture
def setup_runner(tmp_path, monkeypatch):
    from src.models.mamba.v24_Ilya.training import (
        cached_config, cached_data, cached_diagnostics, cached_k1_data, cached_k1_step)
    base = load_training_config(Path("configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json"))
    base = dataclasses.replace(base, ar_tail_k=0, max_steps=200, checkpoint_every=8,
        warmup_steps=5, architecture=dataclasses.replace(base.architecture,
            residual_initialization="fresh", residual_width=128),
        distributed=dataclasses.replace(base.distributed, per_device_batch_size=8),
        output_root=tmp_path, run_name="full")
    producer = dataclasses.replace(base, distributed=dataclasses.replace(base.distributed, per_device_batch_size=1))
    cached = SimpleNamespace(common=producer,
        manifest={"manifest_sha256": "cache", "compatibility_sha256": "compatibility",
                  "compatibility": {"baseline_sha256": "baseline"}},
        identity={"cache": "verified"},
        training_data=SimpleNamespace(manifest_fingerprint="anchors"),
        context=(None, None, None, None, None, ModelConfig()))
    configs = {"full": base, "split": dataclasses.replace(base, run_name="split")}
    monkeypatch.setattr(runner, "load_training_config", lambda path: configs[Path(path).stem])
    monkeypatch.setattr(cached_config, "load_cached_training_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_data, "open_cached_training_data", lambda *args: cached)
    monkeypatch.setattr(runner, "require_k1_runtime", lambda: {"device": "cpu-test"})
    monkeypatch.setattr(cached_k1_data, "CachedK1Data", TinySampler)
    monkeypatch.setattr(cached_k1_step, "make_cached_k1_step", lambda **kwargs: TinyTrainer(kwargs["optimizer"]))
    monkeypatch.setattr(runner, "build_k1_provenance", lambda config, *args, **kwargs:
                        {"scientific_config_sha256": runner.digest_json(runner.scientific_config(config))})
    return configs, cached


def _run(name, **kwargs):
    return runner.run_cached_k1_training(Path(name + ".json"), Path("producer.json"),
        cache_root=Path("cache"), expected_manifest_sha256="pinned", **kwargs)


def test_cpu_runner_exact_resume_preserves_state_rng_optimizer_schedule_and_masked_counts(setup_runner):
    configs, _ = setup_runner
    full = _run("full", max_steps=6, verify_roundtrip=True)
    interrupted = _run("split", max_steps=3, verify_roundtrip=True)
    split = _run("split", max_steps=6, resume=interrupted, verify_roundtrip=True)
    provenance = json.loads(full.with_suffix(".cached_k1.json").read_text())["provenance"]
    report = runner.compare_k1_checkpoints(full, split, provenance)
    assert report["passed"]
    assert report["actual"]["cursor"] == {"epoch": 0, "group_index": 1, "chunk_index": 2}
    assert report["actual"]["samples_seen"] == (8 * 4 + 2) * 24
    full_rows = [json.loads(line) for line in (configs["full"].run_dir / "train_metrics.jsonl").read_text().splitlines()]
    split_rows = [json.loads(line) for line in (configs["split"].run_dir / "train_metrics.jsonl").read_text().splitlines()]
    assert [row["loss"] for row in full_rows] == [row["loss"] for row in split_rows]
    assert [row["learning_rate"] for row in full_rows] == [row["learning_rate"] for row in split_rows]
    assert full_rows[-1]["valid_lanes"] == 1 and full_rows[-1]["valid_weather_steps"] == 24
    assert full_rows[-1]["samples_seen"] == 816
    roundtrip = json.loads((configs["full"].run_dir / "checkpoint_roundtrip.json").read_text())
    assert roundtrip["passed"] and roundtrip["prediction_roundtrip"]["passed"]
    for name in ("repeat", "reload"):
        comparison = roundtrip["prediction_roundtrip"][name]
        assert comparison["passed"]
        assert set(comparison["components"]) == {"loss", "state", "prediction"}


def test_advisory_roundtrip_separates_numerical_warning_and_serialization():
    prediction = {'passed': False, 'integrity_passed': True, 'repeat': {'relative_l2': .03}}
    warning = runner.roundtrip_result({'params': True}, prediction, numerical_policy='advisory')
    assert warning['passed'] and warning['integrity_passed']
    assert warning['status'] == 'complete_with_warnings'
    assert not runner.roundtrip_result({'params': True}, prediction)['passed']
    assert not runner.roundtrip_result({'params': False}, prediction, numerical_policy='advisory')['passed']
    assert not runner.roundtrip_result({'params': True}, {**prediction, 'integrity_passed': False},
                                       numerical_policy='advisory')['passed']


def test_advisory_runner_completes_despite_finite_forward_drift(setup_runner, monkeypatch):
    configs, _ = setup_runner
    original = TinyTrainer.evaluate_step
    calls = [0]
    def varied(self, *args, **kwargs):
        loss, state, prediction = original(self, *args, **kwargs)
        calls[0] += 1
        return loss + .1 * calls[0], state, prediction + .1 * calls[0]
    monkeypatch.setattr(TinyTrainer, 'evaluate_step', varied)
    path = _run('full', max_steps=2, verify_roundtrip=True, numerical_policy='advisory')
    report = json.loads((configs['full'].run_dir / 'checkpoint_roundtrip.json').read_text())
    assert report['passed'] and report['serialization_passed'] and report['numerical_status'] == 'warning'
    assert path.with_suffix('.serialization.json').exists()
    assert path.with_suffix('.roundtrip.json').exists()
    assert path.with_name('checkpoint_step00000000.serialization.json').exists()


def test_advisory_runner_rejects_nonfinite_forward_output(setup_runner, monkeypatch):
    original = TinyTrainer.evaluate_step
    def broken(self, *args, **kwargs):
        loss, state, prediction = original(self, *args, **kwargs)
        return loss, state, prediction * np.nan
    monkeypatch.setattr(TinyTrainer, 'evaluate_step', broken)
    with pytest.raises(ValueError, match='checkpoint roundtrip'):
        _run('full', max_steps=2, verify_roundtrip=True, numerical_policy='advisory')


def test_fresh_and_resumed_training_use_committed_device_arrays(setup_runner, monkeypatch):
    original = TinyTrainer.__call__
    signatures = []

    def train(self, params, state, optimizer_state, keys, *args):
        leaves = jax.tree_util.tree_leaves((params, state, optimizer_state, keys))
        assert all(isinstance(leaf, jax.Array) and leaf.committed for leaf in leaves)
        signatures.append([(leaf.shape, leaf.dtype, leaf.weak_type, leaf.sharding) for leaf in leaves])
        return original(self, params, state, optimizer_state, keys, *args)

    monkeypatch.setattr(TinyTrainer, "__call__", train)
    checkpoint = _run("full", max_steps=1)
    _run("full", max_steps=2, resume=checkpoint)
    assert signatures[0] == signatures[1]


def test_roundtrip_reuses_one_jit_signature_for_live_and_reloaded_trees():
    traces = []
    device = jax.devices()[0]
    params = {"w": jnp.asarray(2, jnp.float32)}
    state = {"memory": np.ones(4, np.float32)}
    checkpoint = SimpleNamespace(residual_params=jax.device_get(params), residual_state=state)
    rng = jax.random.PRNGKey(42)
    inputs = jnp.arange(4, dtype=jnp.float32)
    batch = SimpleNamespace(inputs=(inputs,), targets=(inputs,), forcings=(inputs,),
                            valid_mask=np.ones(4, bool))

    @jax.jit
    def evaluate(params, state, key, inputs, targets, forcings, mask):
        traces.append(1)
        prediction = params["w"] * inputs + state["memory"]
        return jnp.mean(prediction ** 2), {"memory": prediction}, prediction

    # Prime the exact device signature used by normal live execution.
    params, state, rng = jax.device_put((params, state, rng), device)
    evaluate(params, state, jax.random.fold_in(rng, 0x4B31), inputs, inputs, inputs, batch.valid_mask)
    result = runner.prediction_roundtrip_check(train_step=SimpleNamespace(evaluate_step=evaluate),
        checkpoint=checkpoint, params=params, state=state, rng=rng, batch=batch)
    assert result["passed"]
    assert len(traces) == 1
    assert result["relative_l2"] == 0


def _scripted_roundtrip(expected, repeated, reloaded):
    outputs = iter((expected, repeated, reloaded))
    checkpoint = SimpleNamespace(residual_params={}, residual_state={})
    batch = SimpleNamespace(inputs=(None,), targets=(None,), forcings=(None,), valid_mask=None)
    return runner.prediction_roundtrip_check(
        train_step=SimpleNamespace(evaluate_step=lambda *args: next(outputs)),
        checkpoint=checkpoint, params={}, state={}, rng=jax.random.PRNGKey(0), batch=batch)


@pytest.mark.parametrize("component,index", [("loss", 0), ("state", 1), ("prediction", 2)])
def test_roundtrip_component_error_cannot_hide_in_large_other_component(component, index):
    expected = [np.asarray(1, np.float32), np.ones(4, np.float32), np.ones(4, np.float32)]
    expected[2 if index == 1 else 1] *= 1e6
    changed = [value.copy() for value in expected]
    changed[index] *= 2
    result = _scripted_roundtrip(tuple(expected), tuple(expected), tuple(changed))
    assert result["reload"]["aggregate"]["passed"]
    assert not result["reload"]["components"][component]["passed"]
    assert result["repeat"]["passed"]
    assert not result["passed"]


def test_roundtrip_unchanged_forward_noise_fails_even_when_reload_matches():
    expected = (np.float32(1), {"memory": np.ones(4, np.float32)}, np.ones(4, np.float32))
    repeated = (np.float32(1), expected[1], 2 * expected[2])
    result = _scripted_roundtrip(expected, repeated, expected)
    assert not result["repeat"]["components"]["prediction"]["passed"]
    assert not result["repeat"]["passed"]
    assert result["reload"]["passed"]
    assert not result["passed"]


@pytest.mark.parametrize("phase", ["reference", "repeat", "reload"])
@pytest.mark.parametrize("component,index", [("loss", 0), ("state", 1), ("prediction", 2)])
def test_roundtrip_rejects_nonfinite_component(phase, component, index):
    expected = (np.float32(1), np.ones(4, np.float32), np.ones(4, np.float32))
    changed = list(expected)
    changed[index] = np.full_like(changed[index], np.nan)
    outputs = [expected, expected, expected]
    outputs[("reference", "repeat", "reload").index(phase)] = tuple(changed)
    result = _scripted_roundtrip(*outputs)
    comparison = result["repeat" if phase == "reference" else phase]
    assert comparison["components"][component]["reason"] == "nonfinite value"
    assert not result["passed"]


def test_roundtrip_detects_changed_reloaded_parameters():
    trainer = TinyTrainer(None)
    batch = TinySampler(None, batch_size=4, seed=1).build_batch(CachedK1Cursor())
    rng = jax.random.PRNGKey(0)
    params, state = trainer.initialize(rng, batch.inputs[0], None, None)
    checkpoint = SimpleNamespace(residual_params={"module": {"w": np.float32(100)}},
                                 residual_state=jax.device_get(state))
    result = runner.prediction_roundtrip_check(train_step=trainer, checkpoint=checkpoint,
        params=params, state=state, rng=rng, batch=batch)
    assert result["repeat"]["passed"]
    assert not result["reload"]["passed"]
    assert not result["passed"]


@pytest.fixture
def update_checkpoints(setup_runner):
    configs, _ = setup_runner
    modules = ("grid2mesh_gnn/encoder", "mesh_gnn/processor", "mesh2grid_gnn/decoder",
               "mesh_interleaved_temporal_r0_s0/mamba", "temporal_residual_head")
    before = {name: {"w": np.zeros(3, np.float32)} for name in modules}
    after = {name: {"w": np.array([0, 0, .25], np.float32)} for name in modules}

    def write(params, name):
        return runner.write_k1_checkpoint(configs["full"].run_dir / "checkpoints" / f"{name}.pkl",
            config=configs["full"], completed_step=1, params=params, state={}, optimizer_state={},
            rng=jax.random.PRNGKey(0), cursor=CachedK1Cursor(), samples_seen=0,
            baseline_fingerprint="baseline", anchor_fingerprint="anchors", provenance={})

    return before, after, write


def test_parameter_updates_require_groups_not_every_leaf(update_checkpoints):
    before, after, write = update_checkpoints
    report = runner.parameter_update_report(write(before, "before"), write(after, "after"))
    assert report["passed"]
    assert set(report["groups"]) == {"grid2mesh", "mesh_processor", "mesh2grid", "temporal", "correction_head"}
    for item in report["groups"].values():
        assert item["passed"] and item["finite"]
        assert item["absolute_l2"] == .25 and item["max_abs"] == .25
        assert item["elements"] == 3 and item["changed_elements"] == 1


@pytest.mark.parametrize("module,group", [("grid2mesh_gnn/encoder", "grid2mesh"),
    ("mesh_gnn/processor", "mesh_processor"), ("mesh2grid_gnn/decoder", "mesh2grid"),
    ("mesh_interleaved_temporal_r0_s0/mamba", "temporal"), ("temporal_residual_head", "correction_head")])
def test_parameter_updates_reject_unchanged_group(update_checkpoints, module, group):
    before, after, write = update_checkpoints
    after[module] = before[module]
    report = runner.parameter_update_report(write(before, "before"), write(after, "after"))
    assert not report["passed"]
    assert not report["groups"][group]["passed"]
    assert report["groups"][group]["absolute_l2"] == 0


@pytest.mark.parametrize("corruption,reason", [("tree", "tree structure"), ("shape", "shape or dtype"),
    ("nonfinite", "Nonfinite"), ("unknown", "Unclassified"), ("missing", None)])
def test_parameter_updates_reject_malformed_or_missing_groups(update_checkpoints, corruption, reason):
    before, after, write = update_checkpoints
    if corruption == "tree":
        after["temporal_residual_head"]["extra"] = np.zeros(1, np.float32)
    elif corruption == "shape":
        after["temporal_residual_head"]["w"] = np.zeros(4, np.float32)
    elif corruption == "nonfinite":
        after["temporal_residual_head"]["w"][0] = np.nan
    elif corruption == "unknown":
        before["unknown/module"] = {"w": np.zeros(1, np.float32)}
        after["unknown/module"] = {"w": np.ones(1, np.float32)}
    elif corruption == "missing":
        before.pop("temporal_residual_head")
        after.pop("temporal_residual_head")
    report = runner.parameter_update_report(write(before, "before"), write(after, "after"))
    assert not report["passed"]
    if reason:
        assert reason in report["reason"]
    else:
        assert report["groups"]["correction_head"]["elements"] == 0


def test_epoch_checkpoint_validation_and_pending_validation_resume(setup_runner, monkeypatch):
    configs, _ = setup_runner
    calls = []

    def validation(**kwargs):
        calls.append(kwargs["step"])
        return {"step": kwargs["step"], "loss": .5, "role": "fixed_checkpoint",
                "subset_fingerprint": "all15", "selected_segments": 15}

    monkeypatch.setattr(runner, "run_k1_validation", validation)
    checkpoint = _run("full", max_steps=8)
    assert calls == [8]
    assert checkpoint.name == "checkpoint_step00000008.pkl"
    best = json.loads((configs["full"].run_dir / "best_validation.json").read_text())
    assert best["step"] == 8 and best["checkpoint"] == str(checkpoint)
    _run("full", max_steps=8, resume=checkpoint)
    assert calls == [8]
    (configs["full"].run_dir / "validation_metrics.jsonl").unlink()
    _run("full", max_steps=8, resume=checkpoint)
    assert calls == [8, 8]


def test_existing_run_and_changed_provenance_fail_before_replay(setup_runner):
    configs, _ = setup_runner
    checkpoint = _run("full", max_steps=1)
    with pytest.raises(FileExistsError, match="explicit resume"):
        _run("full", max_steps=2)
    configs["full"] = dataclasses.replace(configs["full"], learning_rate=2e-4)
    with pytest.raises(ValueError, match="provenance differs"):
        _run("full", max_steps=2, resume=checkpoint)


def test_corrupt_checkpoint_bytes_and_true_sample_count_are_rejected(setup_runner):
    _, cached = setup_runner
    path = _run("full", max_steps=1)
    provenance = json.loads(path.with_suffix(".cached_k1.json").read_text())["provenance"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        runner.load_k1_checkpoint(path, provenance)


@pytest.mark.parametrize("batch_size,updates", [(8, 56), (4, 108)])
def test_true_samples_before_cursor_exclude_padding(batch_size, updates):
    data = SimpleNamespace(batch_size=batch_size, num_segments=106)
    final_group = (106 + batch_size - 1) // batch_size - 1
    assert runner.samples_before_cursor(data, CachedK1Cursor(epoch=25)) == 25 * 106 * 96
    previous = runner.samples_before_cursor(data, CachedK1Cursor(group_index=final_group, chunk_index=3))
    assert 106 * 96 - previous == 2 * 24


def test_full_schedule_is_immutable_and_fresh_initialization_required(setup_runner):
    configs, cached = setup_runner
    config = configs["full"]
    data = TinySampler(cached, batch_size=8, seed=22)
    runner.validate_k1_config(config, cached, data, epochs=25)
    with pytest.raises(ValueError, match="max_steps=200"):
        runner.validate_k1_config(dataclasses.replace(config, max_steps=20), cached, data, epochs=25)
    with pytest.raises(ValueError, match="fresh"):
        runner.validate_k1_config(dataclasses.replace(config, architecture=dataclasses.replace(
            config.architecture, residual_initialization="baseline_overlay")), cached, data, epochs=25)
    assert runner.scientific_config(config)["optimizer"]["max_steps"] == 200


def test_production_rejects_preflight_validation_subset(setup_runner):
    with pytest.raises(ValueError, match="all complete segments"):
        _run("full", validation_segments=2)


def test_validation_aggregates_all15_real_segments_and_exact_physical_loss(setup_runner):
    configs, _ = setup_runner
    lat = np.linspace(-90, 90, 5, dtype=np.float32)

    def field(value):
        return xr.Dataset({"2m_temperature": (("batch", "time", "lat", "lon"),
            np.full((8, 1, 5, 4), value, np.float32))}, coords={
                "batch": np.arange(8), "time": np.array([6], dtype="timedelta64[h]"),
                "lat": lat, "lon": np.arange(4, dtype=np.float32) * 90})

    class ValidationData:
        batch_size = 8
        chunks_per_segment = 4

        def build_validation_batch(self, group, chunk, **kwargs):
            ids = np.arange(group * 8, (group + 1) * 8)
            mask = ids < 15
            return SimpleNamespace(valid_mask=mask, segment_ids=np.where(mask, ids, -1),
                truths=(field(1),) * 24, baselines=(field(3),) * 24,
                inputs=(field(group),) * 24, targets=(field(-2),) * 24, forcings=(field(0),) * 24)

    class ValidationTrainer:
        def evaluate(self, params, state, keys, inputs, targets, forcings, valid_mask, *, prediction_callback):
            group = int(inputs[0]["2m_temperature"].values.flat[0])
            loss = 10 if group == 0 else 100
            prediction = field(-1 if group == 0 else 0)
            for index in range(24):
                prediction_callback(index, prediction)
            return np.float32(loss), {}, np.full(24, loss, np.float32), ()

    stats = {"diffs_stddev_by_level": xr.Dataset({"2m_temperature": np.float32(1)})}
    cached = SimpleNamespace(training_data=SimpleNamespace(validation_segments=range(15),
        fingerprint_validation_subset=lambda ids: "all15"), context=(None, None, None, stats),
        manifest={"coordinates": {"lat": lat}, "manifest_sha256": "cache"})
    result = runner.run_k1_validation(train_step=ValidationTrainer(), data=ValidationData(), cached=cached,
        config=configs["full"], params={}, zero_state={}, step=56)
    assert result["selected_segments"] == 15 and result["num_anchors"] == 15 * 96
    assert result["loss"] == (8 * 10 + 7 * 100) / 15
    exact = result["original_graphcast_loss"]
    assert exact["baseline_rollout"] == pytest.approx(4, abs=1e-6)
    assert exact["full_rollout"] == pytest.approx((8 * 1 + 7 * 4) / 15, abs=1e-6)
    assert exact["improvement_pct_rollout"] == pytest.approx(40, abs=1e-5)
