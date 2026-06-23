from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.training.param_utils import (
    build_optimizer_group_labels,
    build_trainable_labels,
    is_lora_param,
    is_temporal_param,
    merge_param_partitions,
    overlay_matching_params,
    partition_params_by_trainable_part,
)


def test_overlay_matching_params_copies_gc_and_keeps_new_mamba_params() -> None:
    initial = {
        "graph_cast/encoder": {"w": np.zeros((2, 2)), "b": np.zeros((2,))},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": np.ones((2, 2))
        },
    }
    source = {
        "graph_cast/encoder": {"w": np.full((2, 2), 3.0), "b": np.full((2,), 4.0)}
    }

    merged, stats = overlay_matching_params(initial, source)

    np.testing.assert_allclose(merged["graph_cast/encoder"]["w"], np.full((2, 2), 3.0))
    np.testing.assert_allclose(merged["graph_cast/encoder"]["b"], np.full((2,), 4.0))
    np.testing.assert_allclose(
        merged["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
        np.ones((2, 2)),
    )
    assert stats.copied == 2
    assert stats.initialized == 1


def test_overlay_matching_params_keeps_new_lora_params_when_source_lacks_them() -> None:
    initial = {
        "graph_cast/encoder": {"w": np.zeros((2, 2))},
        "mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora": {
            "a": np.ones((2, 1)),
            "b": np.zeros((1, 2)),
        },
    }
    source = {"graph_cast/encoder": {"w": np.full((2, 2), 3.0)}}

    merged, stats = overlay_matching_params(initial, source)

    np.testing.assert_allclose(merged["graph_cast/encoder"]["w"], np.full((2, 2), 3.0))
    np.testing.assert_allclose(
        merged["mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora"]["a"],
        np.ones((2, 1)),
    )
    assert stats.copied == 1
    assert stats.initialized == 2


def test_overlay_matching_params_fails_on_shape_mismatch() -> None:
    initial = {"graph_cast/encoder": {"w": np.zeros((2, 2))}}
    source = {"graph_cast/encoder": {"w": np.zeros((3, 2))}}

    with pytest.raises(ValueError, match="shape"):
        overlay_matching_params(initial, source)


def test_mamba_only_mask_freezes_gc_even_with_weight_decay() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")

    params = {
        "graph_cast/encoder": {"w": jnp.ones((2, 2))},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": jnp.ones((2, 2))
        },
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = optax.multi_transform(
        {
            "train": optax.adamw(1e-2, weight_decay=1e-1),
            "freeze": optax.set_to_zero(),
        },
        build_trainable_labels(params, "mamba"),
    )

    updates, _ = opt.update(grads, opt.init(params), params)
    new_params = optax.apply_updates(params, updates)

    np.testing.assert_allclose(new_params["graph_cast/encoder"]["w"], params["graph_cast/encoder"]["w"])
    assert not np.allclose(
        new_params["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
        params["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
    )


def test_optimizer_group_labels_split_graphcast_and_mamba_params() -> None:
    params = {
        "graph_cast/encoder": {"w": np.ones((2, 2))},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": np.ones((2, 2))
        },
    }

    labels = build_optimizer_group_labels(params, "all")

    assert labels["graph_cast/encoder"]["w"] == "graphcast"
    assert labels["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"] == "mamba"


def test_optimizer_group_labels_split_lora_params() -> None:
    params = {
        "graph_cast/encoder": {"w": np.ones((2, 2))},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": np.ones((2, 2))
        },
        "mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora": {
            "a": np.ones((2, 1)),
            "b": np.ones((1, 2)),
        },
    }

    labels = build_optimizer_group_labels(params, "mamba_lora")

    assert labels["graph_cast/encoder"]["w"] == "freeze"
    assert labels["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"] == "mamba"
    assert labels["mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora"]["a"] == "lora"
    assert labels["mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora"]["b"] == "lora"


def test_optimizer_group_lrs_apply_larger_mamba_update() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")

    params = {
        "graph_cast/encoder": {"w": jnp.ones((2, 2))},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": jnp.ones((2, 2))
        },
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = optax.multi_transform(
        {
            "graphcast": optax.adamw(learning_rate=1e-3, b1=0.0, b2=0.0, weight_decay=0.0),
            "mamba": optax.adamw(learning_rate=1e-2, b1=0.0, b2=0.0, weight_decay=0.0),
        },
        build_optimizer_group_labels(params, "all"),
    )

    updates, _ = opt.update(grads, opt.init(params), params)

    graphcast_update = abs(float(updates["graph_cast/encoder"]["w"][0, 0]))
    mamba_update = abs(
        float(updates["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"][0, 0])
    )
    np.testing.assert_allclose(mamba_update / graphcast_update, 10.0, rtol=1e-4)


def test_partition_and_merge_keeps_frozen_params_unchanged() -> None:
    params = {
        "graph_cast/encoder": {"w": np.full((2, 2), 3.0)},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": np.ones((2, 2))
        },
    }

    trainable, frozen = partition_params_by_trainable_part(params, "mamba")
    trainable["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"] = np.full((2, 2), 5.0)
    merged = merge_param_partitions(trainable, frozen)

    assert "graph_cast/encoder" not in trainable
    assert "graph_cast/encoder" in frozen
    np.testing.assert_allclose(merged["graph_cast/encoder"]["w"], params["graph_cast/encoder"]["w"])
    np.testing.assert_allclose(
        merged["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
        np.full((2, 2), 5.0),
    )


def test_mamba_lora_mask_freezes_base_gc_even_with_weight_decay() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")

    params = {
        "graph_cast/encoder": {"w": jnp.ones((2, 2))},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": jnp.ones((2, 2))
        },
        "mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora": {
            "a": jnp.ones((2, 1)),
            "b": jnp.ones((1, 2)),
        },
    }
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    opt = optax.multi_transform(
        {
            "mamba": optax.adamw(1e-2, weight_decay=1e-1),
            "lora": optax.adamw(1e-2, weight_decay=1e-1),
            "freeze": optax.set_to_zero(),
        },
        build_optimizer_group_labels(params, "mamba_lora"),
    )

    updates, _ = opt.update(grads, opt.init(params), params)
    new_params = optax.apply_updates(params, updates)

    np.testing.assert_allclose(new_params["graph_cast/encoder"]["w"], params["graph_cast/encoder"]["w"])
    assert not np.allclose(
        new_params["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
        params["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
    )
    assert not np.allclose(
        new_params["mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora"]["a"],
        params["mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora"]["a"],
    )


def test_partitioned_gradients_only_include_trainable_tree() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    params = {
        "graph_cast/encoder": {"w": jnp.full((2,), 3.0)},
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj": {
            "w": jnp.full((2,), 2.0)
        },
    }
    trainable, frozen = partition_params_by_trainable_part(params, "mamba")

    def loss_fn(trainable_params, frozen_params):
        full_params = merge_param_partitions(trainable_params, frozen_params)
        frozen_w = full_params["graph_cast/encoder"]["w"]
        trainable_w = full_params["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"]
        return jnp.sum(frozen_w * trainable_w)

    grads = jax.grad(loss_fn, argnums=0)(trainable, frozen)
    frozen_alt = {
        "graph_cast/encoder": {"w": jnp.full((2,), 5.0)},
    }
    grads_alt = jax.grad(loss_fn, argnums=0)(trainable, frozen_alt)

    assert set(grads) == {"graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"}
    np.testing.assert_allclose(
        grads["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
        np.full((2,), 3.0),
    )
    assert set(grads_alt) == {"graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"}
    np.testing.assert_allclose(
        grads_alt["graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj"]["w"],
        np.full((2,), 5.0),
    )


def test_is_temporal_param_matches_expected_mamba_names() -> None:
    assert is_temporal_param(
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj", "w"
    )
    assert not is_temporal_param("graph_cast/encoder", "w")


def test_is_lora_param_matches_expected_lora_names() -> None:
    assert is_lora_param(
        "mesh_gnn/~_networks_builder/processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora", "a"
    )
    assert is_lora_param(
        "mesh_gnn/~_networks_builder/processor_edges_0_mesh_mlp/~/linear_1_lora", "b"
    )
    assert not is_lora_param("mesh_gnn/~_networks_builder/processor_edges_0_mesh_mlp/~/linear_1", "w")
