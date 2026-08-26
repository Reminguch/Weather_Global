from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.models.mamba.modules.temporal_mesh_mamba_Ilya import (
    TemporalMeshBlock,
    TemporalMeshConfig,
)


def _transform(cfg: TemporalMeshConfig):
    def forward(x_tnbd, prev_state=None):
        return TemporalMeshBlock(cfg)(x_tnbd, prev_state=prev_state, is_training=False)

    return hk.transform(forward)


def _state_leaves(state):
    return [
        np.asarray(value)
        for layer_state in state
        for value in (layer_state.ssm_state, layer_state.conv_cache)
    ]


def _single_module_params(params, suffix: str):
    matches = [
        module_params
        for module_name, module_params in params.items()
        if module_name.endswith(suffix)
    ]
    assert len(matches) == 1
    return matches[0]


def test_mamba1_dt_initialization_and_zero_output_preserve_identity() -> None:
    cfg = TemporalMeshConfig(
        backbone="mamba",
        location="mesh_processor_interleaved",
        d_inner=4,
        d_state=3,
        d_conv=2,
        dt_rank=2,
        layers=1,
        init_scheme="mamba1",
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
        zero_init_output=True,
    )
    transformed = _transform(cfg)
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (3, 5, 2, 6))

    params = transformed.init(rng, x, None)
    output, state = transformed.apply(params, rng, x, None)
    dt_proj = _single_module_params(params, "dt_proj")
    out_proj = _single_module_params(params, "out_proj")

    realized_dt = jax.nn.softplus(dt_proj["b"])
    assert np.min(realized_dt) >= cfg.dt_min
    assert np.max(realized_dt) <= cfg.dt_max
    weight_bound = cfg.dt_scale / np.sqrt(cfg.dt_rank)
    assert np.min(dt_proj["w"]) >= -weight_bound
    assert np.max(dt_proj["w"]) <= weight_bound
    np.testing.assert_array_equal(out_proj["w"], np.zeros_like(out_proj["w"]))
    np.testing.assert_array_equal(output, x)
    assert all(np.isfinite(value).all() for value in _state_leaves(state))


def test_legacy_dt_initialization_retains_zero_bias() -> None:
    cfg = TemporalMeshConfig(
        backbone="mamba",
        location="mesh_processor_interleaved",
        d_inner=4,
        d_state=3,
        d_conv=2,
        dt_rank=2,
        layers=1,
        init_scheme="legacy_haiku",
    )
    transformed = _transform(cfg)
    rng = jax.random.PRNGKey(43)
    x = jax.random.normal(rng, (2, 3, 1, 6))

    params = transformed.init(rng, x, None)
    dt_proj = _single_module_params(params, "dt_proj")
    np.testing.assert_array_equal(dt_proj["b"], np.zeros_like(dt_proj["b"]))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"init_scheme": "unknown"}, "init_scheme"),
        ({"dt_init": "unknown"}, "dt_init"),
        ({"dt_min": 0.0}, "dt_min"),
        ({"dt_min": 0.1, "dt_max": 0.01}, "dt_max"),
        ({"dt_scale": 0.0}, "dt_scale"),
        ({"dt_init_floor": 0.0}, "dt_init_floor"),
        ({"dt_max": 0.01, "dt_init_floor": 0.02}, "dt_init_floor"),
    ],
)
def test_mamba_initialization_rejects_invalid_values(
    replacement: dict,
    message: str,
) -> None:
    cfg = dataclasses.replace(
        TemporalMeshConfig(
            backbone="mamba",
            location="mesh_processor_interleaved",
            d_inner=4,
            d_state=3,
            d_conv=2,
            dt_rank=2,
        ),
        **replacement,
    )
    transformed = _transform(cfg)
    x = jnp.ones((2, 3, 1, 6), dtype=jnp.float32)
    with pytest.raises(ValueError, match=message):
        transformed.init(jax.random.PRNGKey(44), x, None)


def test_grouped_bc_has_finite_outputs_and_preserves_external_state_shape() -> None:
    cfg = TemporalMeshConfig(
        backbone="mamba",
        location="mesh_processor_interleaved",
        d_inner=4,
        bc_groups=2,
        d_state=3,
        d_conv=2,
        layers=1,
    )
    transformed = _transform(cfg)
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (3, 5, 2, 6))

    params = transformed.init(rng, x, None)
    output, state = transformed.apply(params, rng, x, None)

    assert output.shape == x.shape
    assert state[0].ssm_state.shape == (2, 5, 4, 3)
    assert state[0].conv_cache.shape == (2, 5, 4, 1)
    assert np.isfinite(np.asarray(output)).all()
    assert all(np.isfinite(value).all() for value in _state_leaves(state))

    x_proj_weights = [
        module_params["w"]
        for module_name, module_params in params.items()
        if module_name.endswith("x_proj")
    ]
    assert len(x_proj_weights) == 1
    assert x_proj_weights[0].shape == (4, 13)


@pytest.mark.parametrize("bc_groups", [0, 5, 3])
def test_grouped_bc_rejects_invalid_group_counts(bc_groups: int) -> None:
    cfg = TemporalMeshConfig(
        backbone="mamba",
        location="mesh_processor_interleaved",
        d_inner=4,
        bc_groups=bc_groups,
        d_state=3,
        d_conv=1,
    )
    transformed = _transform(cfg)
    x = jnp.ones((2, 3, 1, 6), dtype=jnp.float32)

    with pytest.raises(ValueError, match="bc_groups|divisible"):
        transformed.init(jax.random.PRNGKey(1), x, None)


def test_bc_groups_one_is_identical_when_implicit_or_explicit() -> None:
    default_cfg = TemporalMeshConfig(
        backbone="mamba",
        location="mesh_processor_interleaved",
        d_inner=4,
        d_state=3,
        d_conv=2,
    )
    explicit_cfg = dataclasses.replace(default_cfg, bc_groups=1)
    default_transform = _transform(default_cfg)
    explicit_transform = _transform(explicit_cfg)
    rng = jax.random.PRNGKey(2)
    x = jax.random.normal(rng, (4, 3, 2, 6))

    default_params = default_transform.init(rng, x, None)
    explicit_params = explicit_transform.init(rng, x, None)
    default_output, default_state = default_transform.apply(default_params, rng, x, None)
    explicit_output, explicit_state = explicit_transform.apply(explicit_params, rng, x, None)

    assert jax.tree_util.tree_structure(default_params) == jax.tree_util.tree_structure(
        explicit_params
    )
    for default_leaf, explicit_leaf in zip(
        jax.tree_util.tree_leaves(default_params),
        jax.tree_util.tree_leaves(explicit_params),
        strict=True,
    ):
        np.testing.assert_array_equal(default_leaf, explicit_leaf)
    np.testing.assert_array_equal(default_output, explicit_output)
    for default_leaf, explicit_leaf in zip(
        _state_leaves(default_state),
        _state_leaves(explicit_state),
        strict=True,
    ):
        np.testing.assert_array_equal(default_leaf, explicit_leaf)

    x_proj_weights = [
        module_params["w"]
        for module_name, module_params in default_params.items()
        if module_name.endswith("x_proj")
    ]
    assert len(x_proj_weights) == 1
    assert x_proj_weights[0].shape == (4, 7)


def test_grouped_bc_chunked_execution_matches_full_sequence() -> None:
    cfg = TemporalMeshConfig(
        backbone="mamba",
        location="mesh_processor_interleaved",
        d_inner=4,
        bc_groups=2,
        d_state=3,
        d_conv=3,
        layers=2,
    )
    transformed = _transform(cfg)
    rng = jax.random.PRNGKey(3)
    x = jax.random.normal(rng, (5, 4, 2, 6))
    params = transformed.init(rng, x, None)

    full_output, full_state = transformed.apply(params, rng, x, None)
    first_output, first_state = transformed.apply(params, rng, x[:2], None)
    second_output, chunked_state = transformed.apply(params, rng, x[2:], first_state)

    np.testing.assert_allclose(
        np.asarray(jnp.concatenate([first_output, second_output], axis=0)),
        np.asarray(full_output),
        rtol=1e-5,
        atol=1e-5,
    )
    for full_leaf, chunked_leaf in zip(
        _state_leaves(full_state),
        _state_leaves(chunked_state),
        strict=True,
    ):
        np.testing.assert_allclose(chunked_leaf, full_leaf, rtol=1e-5, atol=1e-5)
