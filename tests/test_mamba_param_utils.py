from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.training.param_utils import (
    build_trainable_labels,
    is_temporal_param,
    overlay_matching_params,
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


def test_is_temporal_param_matches_expected_mamba_names() -> None:
    assert is_temporal_param(
        "graph_cast/mesh_interleaved_temporal_r0_s0/mamba_block_0/out_proj", "w"
    )
    assert not is_temporal_param("graph_cast/encoder", "w")
