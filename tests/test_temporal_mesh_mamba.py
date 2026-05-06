from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from src.models.mamba.modules.temporal_mesh_mamba import (
    TemporalMeshBlock,
    TemporalMeshConfig,
)


def _run_block(cfg: TemporalMeshConfig, x_tnbd: jax.Array, *, is_training: bool) -> jax.Array:
    def forward(x):
        block = TemporalMeshBlock(cfg)
        return block(x, is_training=is_training)

    transformed = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    params = transformed.init(rng, x_tnbd)
    return transformed.apply(params, rng, x_tnbd)


def test_temporal_mesh_block_none_returns_last_step() -> None:
    x = jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5)
    y = _run_block(TemporalMeshConfig(backbone="none"), x, is_training=False)
    np.testing.assert_allclose(np.asarray(y), np.asarray(x[-1]))


def test_temporal_mesh_block_mamba_returns_expected_shape_and_finite_values() -> None:
    x = jnp.ones((4, 6, 2, 8), dtype=jnp.float32)
    y = _run_block(
        TemporalMeshConfig(backbone="mamba", d_inner=8, layers=2, dropout=0.0),
        x,
        is_training=False,
    )
    assert y.shape == (6, 2, 8)
    assert np.isfinite(np.asarray(y)).all()


def test_temporal_mesh_block_mamba_accepts_single_step_3d_latents() -> None:
    x = jnp.ones((6, 2, 8), dtype=jnp.float32)
    y = _run_block(
        TemporalMeshConfig(backbone="mamba", d_inner=8, layers=1, dropout=0.0),
        x,
        is_training=False,
    )
    assert y.shape == x.shape
    assert np.isfinite(np.asarray(y)).all()
