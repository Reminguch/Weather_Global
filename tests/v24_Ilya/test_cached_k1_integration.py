"""Consumer/producer separation and padded sequence restart integration."""

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
from graphcast import xarray_jax

from src.models.mamba.v24_Ilya.config import V24IlyaArchitectureConfig
from src.models.mamba.v24_Ilya.training.cached_data import CachedSequenceBatch
from src.models.mamba.v24_Ilya.training.cached_k1_data import CachedK1Cursor, CachedK1Data
from src.models.mamba.v24_Ilya.training.cached_k1_runner import (
    load_k1_checkpoint, samples_before_cursor, validate_k1_config, write_k1_checkpoint,
)
from src.models.mamba.v24_Ilya.training.cached_k1_step import CachedK1Trainer
from src.models.mamba.v24_Ilya.training.cached_stepwise import CachedResidualTransforms
from src.models.mamba.v24_Ilya.training.config import V24IlyaDistributedConfig, V24IlyaTrainConfig


def _config(root: Path) -> V24IlyaTrainConfig:
    return V24IlyaTrainConfig(
        prepared_root=root / "prepared", anchor_manifest_root=root / "anchors",
        baseline_checkpoint=root / "baseline.npz", output_root=root / "out", run_name="toy",
        architecture=V24IlyaArchitectureConfig(width=512, residual_width=128,
                                              residual_initialization="fresh"),
        segment_steps=96, bptt_steps=24, ar_tail_k=0, feedback_mode="baseline",
        loss_mode="all_steps", learning_rate_schedule="cosine", max_steps=200,
        warmup_steps=0, end_learning_rate=1e-5, precision="fp32")


def _field(values) -> xr.Dataset:
    return xr.Dataset({"u": (("batch", "time"), np.asarray(values, np.float32)[:, None])},
                      coords={"time": np.array([6], dtype="timedelta64[h]")})


class _Producer:
    def __init__(self, config):
        self.common = config
        self.training_data = SimpleNamespace(segments=tuple(range(6)), validation_segments=())
        self.manifest = {
            "partial": False, "manifest_sha256": "saved-producer-manifest",
            "compatibility_sha256": "saved-producer-identity",
            "compatibility": {"ar_tail_k": 0, "feedback_mode": "baseline",
                              "bptt_steps": 24, "truth_prefix_steps": 24},
            "chunks": [{"split": "train", "segment_id": s, "segment_offset": 24 * c}
                       for s in range(6) for c in range(4)]}
        self.identity = {"manifest_sha256": self.manifest["manifest_sha256"]}

    def build_chunk(self, cursor, *, include_validation_fields=False):
        segment, offset = cursor.segment_index, cursor.segment_offset
        start = segment * 96 + offset
        inputs = tuple(_field([(start + i) / 100]) for i in range(24))
        targets = tuple(_field([.01 * (start + i) + .2]) for i in range(24))
        forcing = tuple(_field([.03]) for _ in range(24))
        return CachedSequenceBatch(inputs, targets, forcing, None, None,
            segment * 4 + offset // 24, "train", segment, offset, offset // 24,
            np.arange(start, start + 24), tuple(map(str, range(start, start + 24))), None)


def _consumer(producer, batch=4, epochs=25):
    config = dataclasses.replace(producer.common,
        architecture=dataclasses.replace(producer.common.architecture, residual_width=512),
        distributed=V24IlyaDistributedConfig(per_device_batch_size=batch))
    data = CachedK1Data(producer, batch_size=batch, seed=22, shuffle=False)
    return dataclasses.replace(config, max_steps=epochs * data.steps_per_epoch), data


def test_consumer_width_and_batch_change_without_mutating_producer_identity(tmp_path):
    producer = _Producer(_config(tmp_path))
    original = json.dumps(producer.common.to_dict(), sort_keys=True)
    manifest = json.dumps(producer.manifest, sort_keys=True)
    config, data = _consumer(producer)
    validate_k1_config(config, producer, data, epochs=25)
    assert producer.common.architecture.residual_width == 128
    assert config.architecture.residual_width == 512
    assert producer.common.distributed.global_batch_size == 1
    assert config.distributed.global_batch_size == 4
    assert json.dumps(producer.common.to_dict(), sort_keys=True) == original
    assert json.dumps(producer.manifest, sort_keys=True) == manifest


@pytest.mark.parametrize("name,value", [("width", 256), ("mesh_size", 4), ("baseline_msg_steps", 8)])
def test_baseline_architecture_change_cannot_masquerade_as_residual_width_change(tmp_path, name, value):
    producer = _Producer(_config(tmp_path))
    config, data = _consumer(producer)
    wrong = dataclasses.replace(config,
        architecture=dataclasses.replace(config.architecture, **{name: value}))
    with pytest.raises(ValueError):
        validate_k1_config(wrong, producer, data, epochs=25)


def _values(dataset):
    return xarray_jax.unwrap_data(dataset.u)[:, 0]


class _Predict:
    def apply(self, params, state, key, inputs, targets, forcings):
        del key, targets
        memory = .7 * state["memory"] + params["toy"]["w"] * (_values(inputs) + _values(forcings))
        prediction = params["toy"]["head"] * memory
        output = xr.Dataset({"u": (("batch", "time"), xarray_jax.wrap(prediction[:, None]))})
        return output, {"memory": memory}


class _Loss:
    def apply(self, params, state, key, inputs, targets, forcings):
        prediction, next_state = _Predict().apply(params, state, key, inputs, targets, forcings)
        values = (_values(prediction) - _values(targets)) ** 2
        loss = xr.DataArray(xarray_jax.wrap(values), dims=("batch",))
        return ((loss, {}), prediction), next_state


def _assert_tree_exact(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


def test_padded_midsegment_checkpoint_preserves_next_updates_and_epoch_reset(tmp_path):
    producer = _Producer(_config(tmp_path))
    config, data = _consumer(producer)
    config.run_dir.mkdir(parents=True)
    optimizer = optax.adam(1e-3)
    trainer = CachedK1Trainer(config=config, optimizer=optimizer,
        transforms=CachedResidualTransforms(_Predict(), _Loss()))
    params = {"toy": {"w": jnp.asarray(.3, jnp.float32), "head": jnp.asarray(.4, jnp.float32)}}
    state = {"memory": jnp.zeros((4,), dtype=jnp.float32)}
    initial = (params, state, optimizer.init(params), jax.random.PRNGKey(8), CachedK1Cursor(), 0)

    def advance(values, count):
        params, state, opt_state, rng, cursor, samples = values
        resets = []
        for _ in range(count):
            batch = data.build_batch(cursor)
            resets.append(batch.reset_state)
            if batch.reset_state:
                state = jax.tree.map(jnp.zeros_like, state)
            rng, *keys = jax.random.split(rng, 25)
            params, state, opt_state, loss, _, _ = trainer(
                params, state, opt_state, jnp.stack(keys), batch.inputs,
                batch.targets, batch.forcings, batch.valid_mask)
            assert np.isfinite(loss)
            invalid = np.asarray(batch.valid_mask) == 0
            np.testing.assert_array_equal(np.asarray(state["memory"])[invalid], 0)
            cursor = batch.next_cursor
            samples += batch.valid_weather_steps
            assert samples == samples_before_cursor(data, cursor)
        return (params, state, opt_state, rng, cursor, samples), resets

    middle, _ = advance(initial, 6)
    params, state, opt_state, rng, cursor, samples = middle
    assert cursor == CachedK1Cursor(group_index=1, chunk_index=2)
    assert samples == 4 * 96 + 2 * 48
    provenance = {"cache_manifest_sha256": producer.manifest["manifest_sha256"]}
    path = write_k1_checkpoint(config.run_dir / "checkpoints/step6.pkl", config=config,
        completed_step=6, params=params, state=state, optimizer_state=opt_state,
        rng=rng, cursor=cursor, samples_seen=samples,
        baseline_fingerprint="teacher", anchor_fingerprint="anchors", provenance=provenance)
    checkpoint, backend = load_k1_checkpoint(path, provenance, data=data)
    restored = (checkpoint.residual_params, checkpoint.residual_state,
                checkpoint.optimizer_state, checkpoint.rng_key,
                CachedK1Cursor.from_mapping(backend["cursor"]), backend["samples_seen"])
    uninterrupted, expected_resets = advance(middle, 3)
    resumed, actual_resets = advance(restored, 3)
    assert actual_resets == expected_resets == [False, False, True]
    _assert_tree_exact(resumed[:4], uninterrupted[:4])
    assert resumed[4:] == uninterrupted[4:]
    assert resumed[4] == CachedK1Cursor(epoch=1, chunk_index=1)
