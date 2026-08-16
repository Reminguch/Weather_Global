from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest
import xarray as xr
from graphcast import xarray_jax

from src.models.mamba.v23_Ilya.checkpoint import (
    atomic_pickle_dump,
    data_parallel_training_checkpoint_payload,
    load_v23_Ilya_checkpoint,
    load_v23_Ilya_data_parallel_training_checkpoint,
)
from src.models.mamba.v23_Ilya.config import V23IlyaArchitectureConfig
from src.models.mamba.v23_Ilya.training.config import (
    V23IlyaDistributedConfig,
    V23IlyaTrainConfig,
)
from src.models.mamba.v23_Ilya.training.data import (
    ReplicaGroupCursor,
    V23IlyaTrainingData,
    advance_replica_group_cursor,
)
from src.models.mamba.v23_Ilya.training.data_parallel import (
    derive_replica_step_keys,
    pack_replica_trees,
    replica_max_abs_difference,
    replica_tree_to_host,
    replicate_tree,
    shard_replica_tree,
    unreplicate_tree,
    unpack_replica_trees,
    validate_data_parallel_runtime,
)
from src.models.mamba.v23_Ilya.training.endpoint_step import (
    V23IlyaTrainingTransforms,
    make_data_parallel_train_step,
    make_train_step,
)


class FrozenBaseline:
    def apply(self, params, state, key, inputs, template, forcing):
        del params, key, inputs, forcing
        return jax.tree_util.tree_map(jnp.zeros_like, template), state


class StatefulResidual:
    def apply(self, params, state, key, inputs, template, forcing):
        del key, forcing
        value = jnp.mean(xarray_jax.unwrap_data(inputs["x"]))
        next_state = {"s": 0.7 * state["s"] + params["a"] * value}
        prediction = jax.tree_util.tree_map(
            lambda target: jnp.ones_like(target) * params["b"] * next_state["s"],
            template,
        )
        return prediction, next_state


class StatefulResidualLoss:
    def apply(self, params, state, key, inputs, target, forcing):
        prediction, next_state = StatefulResidual().apply(
            params,
            state,
            key,
            inputs,
            target,
            forcing,
        )
        error = (
            xarray_jax.unwrap_data(prediction["x"])
            - xarray_jax.unwrap_data(target["x"])
        )
        loss = xr.DataArray(
            jnp.mean(error**2)[None],
            dims=("batch",),
            coords={"batch": [0]},
        )
        return ((loss, {}), prediction), next_state


def _config() -> V23IlyaTrainConfig:
    return V23IlyaTrainConfig(
        prepared_root=Path("prepared"),
        anchor_manifest_root=Path("anchors"),
        baseline_checkpoint=Path("baseline"),
        output_root=Path("output"),
        run_name="dp-test",
        architecture=V23IlyaArchitectureConfig(width=4),
        segment_steps=4,
        bptt_steps=4,
        ar_tail_k=2,
        feedback_mode="closed_loop_sg",
        loss_mode="sparse_steps",
        supervised_horizons=(1, 3),
        supervised_weights=(1, 3),
        weather_tape_precision="fp32",
        distributed=V23IlyaDistributedConfig(
            mode="data_parallel",
            num_devices=4,
            per_device_batch_size=1,
            drop_incomplete_replica_group=True,
        ),
    )


def _dataset(value: float, name: str = "x", hour: int = 0) -> xr.Dataset:
    return xr.Dataset(
        {
            name: (
                ("batch", "time"),
                jnp.asarray([[value]], dtype=jnp.float32),
            )
        },
        coords={"batch": [0], "time": [np.timedelta64(hour, "h")]},
    )


def _input_frame(value: float) -> xr.Dataset:
    return xr.merge([_dataset(value), _dataset(0.0, "forcing")])


def _lane_arguments(lane: int):
    scale = float(lane + 1)
    return (
        (_input_frame(scale), _input_frame(2.0 * scale), _input_frame(3.0 * scale)),
        xr.Dataset(),
        (
            _dataset(2.0 + scale, hour=6),
            _dataset(3.0 + 2.0 * scale, hour=6),
        ),
        tuple(_dataset(0.0, "forcing", 6) for _ in range(4)),
    )


def _transforms() -> V23IlyaTrainingTransforms:
    residual = StatefulResidual()
    residual_loss = StatefulResidualLoss()
    return V23IlyaTrainingTransforms(
        FrozenBaseline(),
        residual,
        residual_loss,
        residual_loss,
    )


def _tree_allclose(left, right, *, rtol=1e-6, atol=1e-6) -> None:
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def _require_four_cpu_devices():
    if jax.local_device_count() != 4:
        pytest.skip("requires --xla_force_host_platform_device_count=4")
    return validate_data_parallel_runtime(4, require_gpu=False)


def test_runtime_validation_is_strict() -> None:
    count = jax.local_device_count()
    assert len(validate_data_parallel_runtime(count, require_gpu=False)) == count
    with pytest.raises(RuntimeError, match="Expected exactly"):
        validate_data_parallel_runtime(count + 1, require_gpu=False)
    if jax.default_backend() == "cpu":
        with pytest.raises(RuntimeError, match="requires GPU"):
            validate_data_parallel_runtime(count, require_gpu=True)


def test_replica_group_cursor_advances_chronologically_and_drops_tail() -> None:
    config = V23IlyaTrainConfig(
        **{
            **_config().__dict__,
            "segment_steps": 120,
            "bptt_steps": 24,
            "ar_tail_k": 20,
            "loss_mode": "last_step",
            "supervised_horizons": (),
            "supervised_weights": (),
        }
    )
    data = V23IlyaTrainingData(
        store=None,
        anchor_indices=np.arange(1080, dtype=np.int64),
        train_split=np.arange(1080, dtype=np.int64),
        val_split=np.asarray([], dtype=np.int64),
        segments=tuple(
            np.arange(start, start + 120, dtype=np.int64)
            for start in range(0, 1080, 120)
        ),
        validation_segments=(),
        fixed_validation_segment_ids=np.asarray([], dtype=np.int64),
        validation_subset_policy="disabled",
        validation_subset_fingerprint=None,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
        manifest_fingerprint="test",
    )
    cursor = ReplicaGroupCursor()
    assert data.replica_group_count(4) == 2
    assert data.dropped_replica_segments(4) == 1
    assert data.active_replica_segment_ids(cursor, 4) == (0, 1, 2, 3)
    for expected_offset in (24, 48, 72, 96):
        cursor = advance_replica_group_cursor(cursor, config, 2)
        assert cursor.segment_offset == expected_offset
        assert cursor.group_index == 0
    cursor = advance_replica_group_cursor(cursor, config, 2)
    assert cursor == ReplicaGroupCursor(epoch=0, group_index=1, segment_offset=0)
    assert data.active_replica_segment_ids(cursor, 4) == (4, 5, 6, 7)


def test_rng_derivation_is_deterministic_and_lane_distinct() -> None:
    master = jax.random.PRNGKey(22)
    next_a, keys_a = derive_replica_step_keys(
        master, num_replicas=4, bptt_steps=24
    )
    next_b, keys_b = derive_replica_step_keys(
        master, num_replicas=4, bptt_steps=24
    )
    np.testing.assert_array_equal(next_a, next_b)
    np.testing.assert_array_equal(keys_a, keys_b)
    assert keys_a.shape == (4, 24, 2)
    assert len({tuple(value) for value in np.asarray(keys_a).reshape(-1, 2)}) == 96


def test_xarray_replica_pack_roundtrip() -> None:
    devices = _require_four_cpu_devices()
    trees = tuple(
        xr.Dataset(
            {
                "x": (
                    ("batch", "time"),
                    jnp.asarray([[replica + 0.5]], dtype=jnp.float32),
                )
            },
            coords={"batch": [0], "time": [np.timedelta64(0, "h")]},
        )
        for replica in range(4)
    )
    treedef, leaves = pack_replica_trees(trees, devices)
    unpacked = unpack_replica_trees(treedef, leaves, 4)
    for actual, expected in zip(unpacked, trees, strict=True):
        xr.testing.assert_identical(actual, expected)


def test_data_parallel_gradient_state_and_replicas_match_serial() -> None:
    devices = _require_four_cpu_devices()
    config = _config()
    params = {"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)}
    initial_states = tuple(
        {"s": jnp.asarray(0.05 * lane, dtype=jnp.float32)} for lane in range(4)
    )
    master, keys = derive_replica_step_keys(
        jax.random.PRNGKey(7), num_replicas=4, bptt_steps=4
    )
    del master

    serial_step = make_train_step(
        transforms=_transforms(),
        optimizer=optax.sgd(1.0),
        baseline_params={},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
    )
    serial_gradients = []
    serial_states = []
    serial_losses = []
    for lane in range(4):
        frames, static, truths, forcings = _lane_arguments(lane)
        next_params, next_state, _, loss, _, loss_components = serial_step(
            params,
            initial_states[lane],
            optax.sgd(1.0).init(params),
            keys[lane],
            frames,
            static,
            truths,
            forcings,
        )
        serial_gradients.append(
            jax.tree_util.tree_map(lambda before, after: before - after, params, next_params)
        )
        serial_states.append(next_state)
        serial_losses.append(float(loss))
        np.testing.assert_allclose(
            loss,
            np.sum(np.asarray(loss_components) * np.asarray([0.25, 0.75])),
            rtol=1e-6,
        )

    clip_norm = 0.05
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.sgd(0.1))
    parallel_step = make_data_parallel_train_step(
        transforms=_transforms(),
        optimizer=optimizer,
        baseline_params={},
        baseline_state={},
        config=config,
        time_step=pd.Timedelta("6h"),
        input_steps=2,
        devices=devices,
    )
    replicated_params = replicate_tree(params, devices)
    replicated_optimizer = replicate_tree(optimizer.init(params), devices)
    state_host = jax.tree_util.tree_map(
        lambda *values: np.stack(values), *initial_states
    )
    replica_states = shard_replica_tree(state_host, devices)
    lane_args = tuple(_lane_arguments(lane) for lane in range(4))
    (
        next_replicated_params,
        next_replica_states,
        next_replicated_optimizer,
        mean_loss,
        gradient_norm,
        lane_losses,
        lane_loss_components,
    ) = parallel_step(
        replicated_params,
        replica_states,
        replicated_optimizer,
        keys,
        tuple(value[0] for value in lane_args),
        tuple(value[1] for value in lane_args),
        tuple(value[2] for value in lane_args),
        tuple(value[3] for value in lane_args),
    )

    mean_gradient = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values).mean(axis=0), *serial_gradients
    )
    expected_updates, _ = optimizer.update(
        mean_gradient,
        optimizer.init(params),
        params,
    )
    expected_params = optax.apply_updates(params, expected_updates)
    actual_params = jax.tree_util.tree_map(
        lambda value: np.asarray(value[0]), next_replicated_params
    )
    _tree_allclose(actual_params, expected_params)
    np.testing.assert_allclose(lane_losses, serial_losses, rtol=1e-6, atol=1e-6)
    assert np.asarray(lane_loss_components).shape == (2, 4)
    np.testing.assert_allclose(mean_loss, np.mean(serial_losses), rtol=1e-6)
    np.testing.assert_allclose(gradient_norm, optax.global_norm(mean_gradient), rtol=1e-6)

    actual_states = replica_tree_to_host(next_replica_states)
    for lane, expected_state in enumerate(serial_states):
        _tree_allclose(
            jax.tree_util.tree_map(lambda value: value[lane], actual_states),
            expected_state,
        )
    assert replica_max_abs_difference(next_replicated_params) == 0.0
    assert replica_max_abs_difference(next_replicated_optimizer) == 0.0

    individually_clipped = []
    for gradient in serial_gradients:
        norm = optax.global_norm(gradient)
        individually_clipped.append(
            jax.tree_util.tree_map(
                lambda value: value * jnp.minimum(1.0, clip_norm / (norm + 1e-12)),
                gradient,
            )
        )
    wrong_gradient = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values).mean(axis=0), *individually_clipped
    )
    wrong_params = jax.tree_util.tree_map(
        lambda value, gradient: value - 0.1 * gradient,
        params,
        wrong_gradient,
    )
    assert any(
        not np.allclose(actual, wrong)
        for actual, wrong in zip(
            jax.tree_util.tree_leaves(actual_params),
            jax.tree_util.tree_leaves(wrong_params),
            strict=True,
        )
    )


def test_data_parallel_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    devices = _require_four_cpu_devices()
    optimizer = optax.adam(0.01)
    initial_params = {"w": jnp.asarray(0.3, dtype=jnp.float32)}
    initial_state = {"s": np.arange(4, dtype=np.float32) * 0.1}
    lane_inputs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

    def local_step(params, state, optimizer_state, key, value):
        def loss_fn(candidate):
            noise = 0.01 * jax.random.normal(key)
            prediction = candidate["w"] * value + state["s"] + noise
            return (prediction - 1.5) ** 2

        loss, gradient = jax.value_and_grad(loss_fn)(params)
        mean_gradient = jax.tree_util.tree_map(
            lambda leaf: jax.lax.pmean(leaf, "data"), gradient
        )
        updates, next_optimizer_state = optimizer.update(
            mean_gradient, optimizer_state, params
        )
        next_params = optax.apply_updates(params, updates)
        next_state = {"s": 0.8 * state["s"] + 0.2 * value}
        return (
            next_params,
            next_state,
            next_optimizer_state,
            jax.lax.pmean(loss, "data"),
        )

    mapped_step = jax.pmap(local_step, axis_name="data", devices=list(devices))

    def fresh():
        return (
            jax.random.PRNGKey(91),
            replicate_tree(initial_params, devices),
            shard_replica_tree(initial_state, devices),
            replicate_tree(optimizer.init(initial_params), devices),
        )

    def advance(master_key, params, states, optimizer_state):
        next_master, keys = derive_replica_step_keys(
            master_key, num_replicas=4, bptt_steps=1
        )
        next_params, next_states, next_optimizer, _ = mapped_step(
            params, states, optimizer_state, keys[:, 0], lane_inputs
        )
        return next_master, next_params, next_states, next_optimizer

    uninterrupted = fresh()
    for _ in range(3):
        uninterrupted = advance(*uninterrupted)

    interrupted = advance(*fresh())
    path = tmp_path / "distributed.pkl"
    atomic_pickle_dump(
        data_parallel_training_checkpoint_payload(
            completed_step=1,
            residual_params=unreplicate_tree(interrupted[1]),
            replica_states=replica_tree_to_host(interrupted[2]),
            optimizer_state=unreplicate_tree(interrupted[3]),
            rng_key=interrupted[0],
            replica_group_cursor={"epoch": 0, "group_index": 0, "segment_offset": 1},
            active_segment_ids=(0, 1, 2, 3),
            num_devices=4,
            resolved_training_config=_config().to_dict(),
            baseline_checkpoint_path="baseline.npz",
            baseline_checkpoint_fingerprint="baseline-sha",
            anchor_manifest_fingerprint="manifest-sha",
            parameter_overlay_metadata={},
        ),
        path,
    )
    checkpoint = load_v23_Ilya_data_parallel_training_checkpoint(path)
    generic = load_v23_Ilya_checkpoint(path)
    assert generic.residual_state is None
    assert checkpoint.completed_step == 1
    assert checkpoint.active_segment_ids == (0, 1, 2, 3)

    resumed = (
        checkpoint.rng_key,
        replicate_tree(checkpoint.residual_params, devices),
        shard_replica_tree(checkpoint.replica_states, devices),
        replicate_tree(checkpoint.optimizer_state, devices),
    )
    for _ in range(2):
        resumed = advance(*resumed)

    np.testing.assert_array_equal(resumed[0], uninterrupted[0])
    _tree_allclose(
        replica_tree_to_host(resumed[1]),
        replica_tree_to_host(uninterrupted[1]),
    )
    _tree_allclose(
        replica_tree_to_host(resumed[2]),
        replica_tree_to_host(uninterrupted[2]),
    )
    _tree_allclose(
        replica_tree_to_host(resumed[3]),
        replica_tree_to_host(uninterrupted[3]),
     )
