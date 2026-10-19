"""Independent cached lanes, native resume, and batch-one validation."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.training.cached_batching import compare_batch_lanes, state_lane
from src.models.mamba.v24_Ilya.training.cached_config import CachedTrainingConfig, CachedExecutionConfig
from src.models.mamba.v24_Ilya.training.cached_data import CachedSequenceBatch, CachedTrainingData
from src.models.mamba.v24_Ilya.training.cached_stepwise import CachedStepwiseTrainer, CachedResidualTransforms
from src.models.mamba.v24_Ilya.training.config import load_training_config
from src.models.mamba.v24_Ilya.training.data import TrainingCursor, V24IlyaTrainingData


def config_for(tmp_path, batch_size):
    reference = load_training_config(Path("configs/experiments/v24_Ilya/res1_open_loop_mamba1_di64_bcg2_all24_10k.json"))
    return dataclasses.replace(reference, output_root=tmp_path, run_name=f"batch{batch_size}",
        distributed=dataclasses.replace(reference.distributed, per_device_batch_size=batch_size),
        bptt_steps=4, segment_steps=8, ar_tail_k=2, precision="fp32", max_steps=5,
        checkpoint_every=2, learning_rate_schedule="constant", end_learning_rate=None,
        warmup_steps=0, validation=dataclasses.replace(reference.validation, enabled=False))


def dataset(values):
    return xr.Dataset({"x": (("batch", "time"), np.asarray(values, np.float32).reshape(-1, 1))},
                      coords={"batch": np.arange(len(values)), "time": [np.timedelta64(6, "h")]})


def adapter_for(config, segment_count=9):
    adapter = CachedTrainingData.__new__(CachedTrainingData)
    adapter.common = config
    adapter.training_data = SimpleNamespace(segments=tuple(np.arange(8) for _ in range(segment_count)))
    adapter.training_data.validate_cursor = lambda cursor, cfg: V24IlyaTrainingData.validate_cursor(adapter.training_data, cursor, cfg)

    def materialize(*, split, segment_id, segment_offset, next_cursor, include_validation_fields):
        inputs = tuple(dataset([1 + segment_id + (segment_offset + step) / 10]) for step in range(4))
        return CachedSequenceBatch(inputs=inputs, targets=tuple(dataset([2.]) for _ in range(4)),
            forcings=tuple(dataset([0.]) for _ in range(4)), truths=None, baselines=None,
            chunk_id=segment_id * 2 + segment_offset // 4, split=split, segment_id=segment_id,
            segment_offset=segment_offset, chunk_index=segment_offset // 4,
            raw_anchor_indices=np.arange(4) + segment_id * 8 + segment_offset,
            timestamps=tuple(str(segment_id * 8 + segment_offset + step) for step in range(4)),
            next_cursor=next_cursor)
    adapter._materialize = materialize
    return adapter


class Residual:
    def apply(self, params, state, key, inputs, target, forcing):
        del key, forcing
        value = xarray_jax.unwrap_data(inputs.x)
        memory = .7 * state["memory"] + params["gain"] * value
        prediction = xr.Dataset({"x": xarray_jax.DataArray(params["head"] * memory,
                                   dims=target.x.dims, coords=target.x.coords)})
        return prediction, {"memory": memory}


class ResidualLoss:
    def apply(self, params, state, key, inputs, target, forcing):
        prediction, state = Residual().apply(params, state, key, inputs, target, forcing)
        error = xarray_jax.unwrap_data(prediction.x) - xarray_jax.unwrap_data(target.x)
        loss = xarray_jax.DataArray(jnp.mean(error ** 2, axis=1), dims=("batch",), coords={"batch": target.batch})
        return ((loss, {}), prediction), state


def trainer_for(config, optimizer=None):
    return CachedStepwiseTrainer(config=config, optimizer=optimizer or optax.adam(.01),
        transforms=CachedResidualTransforms(Residual(), ResidualLoss()))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_lanes_cover_aligned_groups_with_dropped_tail(tmp_path, batch_size):
    config = config_for(tmp_path, batch_size)
    CachedTrainingConfig(config, CachedExecutionConfig(tmp_path), tmp_path / "config.json")
    adapter = adapter_for(config)
    groups = 9 // batch_size
    batches = list(adapter.iter_chunks(TrainingCursor(), max_chunks=2 * groups + 1))
    assert batches[-1].segment_offset == 0 and batches[-1].segment_id == 0
    assert batches[-2].next_cursor == TrainingCursor(epoch=1)
    for group in range(groups):
        first, second = batches[group * 2:group * 2 + 2]
        assert first.reset_state and not second.reset_state
        assert first.inputs[0].sizes["batch"] == batch_size
        np.testing.assert_allclose(first.inputs[0].x[:, 0], 1 + np.arange(batch_size) + group * batch_size)
        np.testing.assert_allclose(second.inputs[0].x[:, 0], 1.4 + np.arange(batch_size) + group * batch_size)
    if batch_size > 1:
        with pytest.raises(ValueError, match="aligned segment group"):
            adapter.build_chunk(TrainingCursor(segment_index=1))
        with pytest.raises(ValueError, match="aligned segment group"):
            adapter.build_chunk(TrainingCursor(segment_index=8))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batched_gradient_loss_and_state_match_independent_lanes(tmp_path, batch_size):
    config = config_for(tmp_path, batch_size)
    trainer = trainer_for(config)
    batch = adapter_for(config).build_chunk(TrainingCursor())
    params = {"gain": jnp.asarray(.11), "head": jnp.asarray(.23)}
    state = {"memory": jnp.arange(batch_size, dtype=jnp.float32).reshape(-1, 1) / 10}
    keys = jax.random.split(jax.random.PRNGKey(22), 4)
    result = compare_batch_lanes(trainer=trainer, params=params, state=state, keys=keys,
                                 batch=batch, batch_size=batch_size)
    assert result["passed"] and result["integrity_passed"]
    assert len(result["lane_losses"]) == batch_size
    np.testing.assert_array_equal(state["memory"], np.arange(batch_size).reshape(-1, 1).astype(np.float32) / 10)


def test_perturbing_one_lane_preserves_other_lane_states(tmp_path):
    config = config_for(tmp_path, 4)
    trainer = trainer_for(config)
    batch = adapter_for(config).build_chunk(TrainingCursor())
    params = {"gain": jnp.asarray(.11), "head": jnp.asarray(.23)}
    state = {"memory": jnp.zeros((4, 1))}
    keys = jax.random.split(jax.random.PRNGKey(22), 4)
    changed = tuple(value.copy(deep=True) for value in batch.inputs)
    for value in changed:
        value.x.data[2] += 10
    original = trainer.evaluate(params, state, keys, batch.inputs, batch.targets, batch.forcings)[1]
    perturbed = trainer.evaluate(params, state, keys, changed, batch.targets, batch.forcings)[1]
    np.testing.assert_array_equal(original["memory"][[0, 1, 3], :], perturbed["memory"][[0, 1, 3], :])
    assert not np.array_equal(original["memory"][2], perturbed["memory"][2])
    np.testing.assert_array_equal(state_lane(original, 2, 4)["memory"], original["memory"][2:3])


def test_batch4_native_checkpoint_resume_across_group_reset(tmp_path, monkeypatch):
    from src.models.mamba.v24_Ilya import checkpoint as native
    from src.models.mamba.v24_Ilya.training import cached_runner, cached_config, cached_data, cached_stepwise, cached_diagnostics
    config = config_for(tmp_path, 4)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    adapter = adapter_for(config)
    # run_cached_training replaces fields of a real dataclass model config.
    from graphcast.graphcast import ModelConfig
    adapter.context = (config, None, None, None, adapter.training_data,
        ModelConfig(resolution=1., mesh_size=1, latent_size=4, gnn_msg_steps=2,
                    hidden_layers=1, radius_query_fraction_edge_length=.6))
    adapter.training_data.manifest_fingerprint = "anchors"
    adapter.manifest = {"compatibility": {"baseline_sha256": "baseline"}}
    adapter.identity = {"test": "independent-lanes"}
    monkeypatch.setattr(cached_diagnostics, "require_matched_gpu_environment", lambda *_: {})
    monkeypatch.setattr(cached_runner, "build_provenance", lambda *args, **kwargs: {"test": "pinned"})
    monkeypatch.setattr(cached_config, "load_cached_training_config", lambda *args, **kwargs:
        CachedTrainingConfig(config, CachedExecutionConfig(tmp_path), config_path))
    monkeypatch.setattr(cached_data, "open_cached_training_data", lambda *args, **kwargs: adapter)
    monkeypatch.setattr(cached_stepwise, "make_cached_stepwise_train_step",
        lambda **kwargs: trainer_for(kwargs["config"], kwargs["optimizer"]))
    params = {"gain": jnp.asarray(.11), "head": jnp.asarray(.23)}
    optimizer, _ = cached_runner.build_optimizer(config)
    shared = tmp_path / "shared.pkl"
    native.atomic_pickle_dump(native.training_checkpoint_payload(
        completed_step=0, residual_params=params, residual_state={"memory": jnp.zeros((4, 1))},
        optimizer_state=optimizer.init(params), rng_key=jax.random.PRNGKey(22),
        training_cursor=TrainingCursor().to_dict(), resolved_training_config=config.to_dict(),
        baseline_checkpoint_path=str(config.baseline_checkpoint), baseline_checkpoint_fingerprint="baseline",
        anchor_manifest_fingerprint="anchors", parameter_overlay_metadata={}), shared)
    interrupted = cached_runner.run_cached_training(config_path, shared, cache_root=tmp_path, max_steps=2)
    saved = native.load_v24_Ilya_training_checkpoint(interrupted)
    assert saved.training_cursor == TrainingCursor(segment_index=4).to_dict()
    resumed = cached_runner.run_cached_training(config_path, shared, cache_root=tmp_path, resume=interrupted)
    resumed_checkpoint = native.load_v24_Ilya_training_checkpoint(resumed)
    config = dataclasses.replace(config, run_name="continuous")
    adapter.common = config
    continuous = cached_runner.run_cached_training(config_path, shared, cache_root=tmp_path)
    continuous_checkpoint = native.load_v24_Ilya_training_checkpoint(continuous)
    from src.models.mamba.v24_Ilya.training.cached_checkpoint import numerical_checkpoint_identity
    assert numerical_checkpoint_identity(resumed_checkpoint) == numerical_checkpoint_identity(continuous_checkpoint)
    import json
    rows = [json.loads(line) for line in (config.run_dir / "train_metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["batch_size"] == 4 and rows[-1]["anchors_seen"] == 80
    assert rows[2]["segment_ids"] == [4, 5, 6, 7]
