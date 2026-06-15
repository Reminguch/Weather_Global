from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))

from graphcast import deep_typed_graph_net as dtgn  # noqa: E402
from graphcast import graphcast as gc  # noqa: E402
from graphcast import typed_graph  # noqa: E402


def test_temporal_processor_group_sizes_for_mp6_sweep() -> None:
    assert gc._temporal_processor_group_sizes(6, 2) == [3, 3]
    assert gc._temporal_processor_group_sizes(6, 3) == [2, 2, 2]
    assert gc._temporal_processor_group_sizes(6, 6) == [1, 1, 1, 1, 1, 1]


def test_lora_mlp_preserves_base_linear_paths_and_starts_as_noop() -> None:
    output_sizes = [4, 3]

    def lora_forward(x):
        mlp = dtgn._LoRAMLP(
            output_sizes=output_sizes,
            activation=jax.nn.swish,
            lora_rank=2,
            lora_alpha=2,
            name="processor_nodes_0_mesh_nodes_mlp",
        )
        return mlp(x)

    def base_forward(x):
        mlp = hk.nets.MLP(
            output_sizes=output_sizes,
            activation=jax.nn.swish,
            name="processor_nodes_0_mesh_nodes_mlp",
        )
        return mlp(x)

    lora_transformed = hk.transform(lora_forward)
    base_transformed = hk.transform(base_forward)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((2, 5), dtype=jnp.float32)

    params = lora_transformed.init(rng, x)
    flat = hk.data_structures.to_mutable_dict(params)

    assert "processor_nodes_0_mesh_nodes_mlp/~/linear_0" in flat
    assert "processor_nodes_0_mesh_nodes_mlp/~/linear_1" in flat
    assert "processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora" in flat
    assert "processor_nodes_0_mesh_nodes_mlp/~/linear_1_lora" in flat
    np.testing.assert_allclose(
        np.asarray(flat["processor_nodes_0_mesh_nodes_mlp/~/linear_0_lora"]["b"]),
        np.zeros((2, 4), dtype=np.float32),
    )

    lora_y = lora_transformed.apply(params, rng, x)
    base_y = base_transformed.apply(params, rng, x)
    np.testing.assert_allclose(np.asarray(lora_y), np.asarray(base_y), rtol=1e-6, atol=1e-6)


def test_processor_lora_only_adds_processor_mlp_params_and_is_disabled_by_default() -> None:
    graph = typed_graph.TypedGraph(
        context=typed_graph.Context(n_graph=jnp.asarray([1]), features=()),
        nodes={
            "mesh_nodes": typed_graph.NodeSet(
                n_node=jnp.asarray([2]),
                features=jnp.ones((2, 3), dtype=jnp.float32),
            )
        },
        edges={
            typed_graph.EdgeSetKey("mesh_edges", ("mesh_nodes", "mesh_nodes")): typed_graph.EdgeSet(
                n_edge=jnp.asarray([2]),
                indices=typed_graph.EdgesIndices(
                    senders=jnp.asarray([0, 1]),
                    receivers=jnp.asarray([1, 0]),
                ),
                features=jnp.ones((2, 2), dtype=jnp.float32),
            )
        },
    )

    def forward(lora_rank: int):
        net = dtgn.DeepTypedGraphNet(
            node_latent_size={"mesh_nodes": 4},
            edge_latent_size={"mesh_edges": 4},
            node_output_size={"mesh_nodes": 2},
            mlp_hidden_size=4,
            mlp_num_hidden_layers=1,
            num_message_passing_steps=1,
            use_layer_norm=False,
            activation="swish",
            lora_rank=lora_rank,
            lora_alpha=4,
            lora_scope=dtgn.LORA_SCOPE_PROCESSOR_MLP,
            name="mesh_gnn",
        )
        return net(graph).nodes["mesh_nodes"].features

    transformed = hk.transform(forward)
    rng = jax.random.PRNGKey(0)

    disabled_params = transformed.init(rng, 0)
    disabled_flat = hk.data_structures.to_mutable_dict(disabled_params)
    assert not any("_lora" in name for name in disabled_flat)

    lora_params = transformed.init(rng, 4)
    lora_flat = hk.data_structures.to_mutable_dict(lora_params)
    lora_names = {name for name in lora_flat if "_lora" in name}
    assert lora_names
    assert all("/processor_" in name for name in lora_names)
    assert all(name.endswith("_lora") for name in lora_names)
    assert any("/processor_edges_0_mesh_edges_mlp/~/linear_0_lora" in name for name in lora_names)
    assert any("/processor_nodes_0_mesh_nodes_mlp/~/linear_1_lora" in name for name in lora_names)
    assert not any("/encoder_" in name or "/decoder_" in name for name in lora_names)

    for name in lora_names:
        np.testing.assert_allclose(np.asarray(lora_flat[name]["b"]), np.zeros_like(lora_flat[name]["b"]))

    lora_y = transformed.apply(lora_params, rng, 4)
    disabled_y_with_lora_params = transformed.apply(lora_params, rng, 0)
    np.testing.assert_allclose(
        np.asarray(lora_y),
        np.asarray(disabled_y_with_lora_params),
        rtol=1e-6,
        atol=1e-6,
    )


def test_mamba_call_uses_vanilla_stacked_input_encoder() -> None:
    calls: list[str] = []
    predictor = object.__new__(gc.GraphCast)
    predictor._temporal_backbone = "mamba"
    predictor._temporal_location = "mesh_processor_interleaved"

    def maybe_init(inputs):
        calls.append("maybe_init")

    def inputs_to_grid_node_features(inputs, forcings):
        calls.append("stacked_inputs")
        return "grid_features"

    def run_grid2mesh_gnn(grid_features):
        calls.append(f"grid2mesh:{grid_features}")
        return "latent_mesh", "latent_grid"

    def run_mesh_gnn(latent_mesh, *, is_training):
        calls.append(f"mesh_gnn:{latent_mesh}:{is_training}")
        return "updated_mesh"

    def run_mesh2grid_gnn(updated_mesh, latent_grid):
        calls.append(f"mesh2grid:{updated_mesh}:{latent_grid}")
        return "grid_output"

    def grid_node_outputs_to_prediction(grid_output, targets_template):
        calls.append(f"prediction:{grid_output}:{targets_template}")
        return "prediction"

    predictor._maybe_init = maybe_init
    predictor._inputs_to_grid_node_features = inputs_to_grid_node_features
    predictor._run_grid2mesh_gnn = run_grid2mesh_gnn
    predictor._run_mesh_gnn = run_mesh_gnn
    predictor._run_mesh2grid_gnn = run_mesh2grid_gnn
    predictor._grid_node_outputs_to_prediction = grid_node_outputs_to_prediction

    out = gc.GraphCast.__call__(
        predictor,
        inputs="inputs",
        targets_template="targets",
        forcings="forcings",
        is_training=True,
    )

    assert out == "prediction"
    assert calls == [
        "maybe_init",
        "stacked_inputs",
        "grid2mesh:grid_features",
        "mesh_gnn:latent_mesh:True",
        "mesh2grid:updated_mesh:latent_grid",
        "prediction:grid_output:targets",
    ]


def test_residual_output_head_zero_initializes_grid_outputs() -> None:
    def forward(enabled: bool):
        predictor = object.__new__(gc.GraphCast)
        predictor._temporal_backbone = "none"
        predictor._temporal_location = "mesh_post_encoder"
        predictor._residual_output_head_enabled = enabled
        predictor._maybe_init = lambda inputs: None
        predictor._inputs_to_grid_node_features = lambda inputs, forcings: jnp.ones((3, 2, 4))
        predictor._run_grid2mesh_gnn = lambda grid_features: ("latent_mesh", "latent_grid")
        predictor._run_mesh_gnn = lambda latent_mesh, *, is_training: "updated_mesh"
        predictor._run_mesh2grid_gnn = lambda updated_mesh, latent_grid: jnp.ones((3, 2, 4))
        predictor._grid_node_outputs_to_prediction = lambda grid_output, targets_template: grid_output
        return gc.GraphCast.__call__(
            predictor,
            inputs=None,
            targets_template=None,
            forcings=None,
            is_training=True,
        )

    transformed = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    params = transformed.init(rng, True)
    output = transformed.apply(params, rng, True)

    np.testing.assert_allclose(np.asarray(output), np.zeros((3, 2, 4), dtype=np.float32))
    assert "residual_output_head" in hk.data_structures.to_mutable_dict(params)


def test_residual_output_head_is_absent_when_disabled() -> None:
    def forward():
        predictor = object.__new__(gc.GraphCast)
        predictor._temporal_backbone = "none"
        predictor._temporal_location = "mesh_post_encoder"
        predictor._maybe_init = lambda inputs: None
        predictor._inputs_to_grid_node_features = lambda inputs, forcings: jnp.ones((3, 2, 4))
        predictor._run_grid2mesh_gnn = lambda grid_features: ("latent_mesh", "latent_grid")
        predictor._run_mesh_gnn = lambda latent_mesh, *, is_training: "updated_mesh"
        predictor._run_mesh2grid_gnn = lambda updated_mesh, latent_grid: jnp.ones((3, 2, 4))
        predictor._grid_node_outputs_to_prediction = lambda grid_output, targets_template: grid_output
        return gc.GraphCast.__call__(
            predictor,
            inputs=None,
            targets_template=None,
            forcings=None,
            is_training=True,
        )

    transformed = hk.transform(forward)
    rng = jax.random.PRNGKey(0)
    params = transformed.init(rng)
    output = transformed.apply(params, rng)

    np.testing.assert_allclose(np.asarray(output), np.ones((3, 2, 4), dtype=np.float32))
    assert "residual_output_head" not in hk.data_structures.to_mutable_dict(params)


class _NodeSet(NamedTuple):
    features: jax.Array


class _EdgeSet(NamedTuple):
    features: jax.Array


class _TinyGraph(NamedTuple):
    nodes: dict[str, _NodeSet]
    edges: dict[str, _EdgeSet]

    def edge_key_by_name(self, name: str) -> str:
        return name


class _TinyMeshGNN:
    _num_processor_repetitions = 1

    def _networks_builder(self, input_graph):
        del input_graph
        return None, [None], None

    def _embed(self, graph, embedder_network):
        del embedder_network
        return graph

    def _process_step(self, processor_network, graph):
        del processor_network
        mesh_nodes = graph.nodes["mesh_nodes"]
        return graph._replace(
            nodes={
                "mesh_nodes": mesh_nodes._replace(
                    features=mesh_nodes.features + jnp.asarray(0.1, mesh_nodes.features.dtype)
                )
            }
        )


def test_stateful_interleaved_temporal_uses_3d_mesh_latents() -> None:
    n_mesh = 5
    batch_size = 2
    channels = 8

    graph = _TinyGraph(
        nodes={"mesh_nodes": _NodeSet(features=jnp.zeros((n_mesh, 1), dtype=jnp.float32))},
        edges={"mesh": _EdgeSet(features=jnp.zeros((4, 1), dtype=jnp.float32))},
    )

    def forward(x):
        predictor = object.__new__(gc.GraphCast)
        predictor._mesh_graph_structure = graph
        predictor._mesh_gnn = _TinyMeshGNN()
        predictor._temporal_backbone = "mamba"
        predictor._temporal_location = "mesh_processor_interleaved"
        predictor._temporal_stateful = True
        predictor._temporal_d_inner = 4
        predictor._temporal_d_state = 3
        predictor._temporal_d_conv = 1
        predictor._temporal_dt_rank = "auto"
        predictor._temporal_bias = False
        predictor._temporal_conv_bias = True
        predictor._temporal_layers = 1
        predictor._temporal_dropout = 0.0
        predictor._temporal_zero_init_out = False
        return gc.GraphCast._run_mesh_gnn_interleaved(predictor, x)

    transformed = hk.transform_with_state(forward)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((n_mesh, batch_size, channels), dtype=jnp.float32)

    params, state = transformed.init(rng, x)
    y, next_state = transformed.apply(params, state, rng, x)

    assert y.shape == (n_mesh, batch_size, channels)
    flat_state = hk.data_structures.to_mutable_dict(next_state)
    ssm_states = [
        value
        for module_state in flat_state.values()
        for name, value in module_state.items()
        if name.endswith("_ssm_state")
    ]
    assert len(ssm_states) == 1
    assert ssm_states[0].shape == (batch_size, n_mesh, 4, 3)
    assert np.isfinite(np.asarray(y)).all()
    assert np.isfinite(np.asarray(ssm_states[0])).all()


def test_zero_init_stateful_interleaved_temporal_starts_as_identity() -> None:
    n_mesh = 5
    batch_size = 2
    channels = 8

    graph = _TinyGraph(
        nodes={"mesh_nodes": _NodeSet(features=jnp.zeros((n_mesh, 1), dtype=jnp.float32))},
        edges={"mesh": _EdgeSet(features=jnp.zeros((4, 1), dtype=jnp.float32))},
    )

    def forward(x):
        predictor = object.__new__(gc.GraphCast)
        predictor._mesh_graph_structure = graph
        predictor._mesh_gnn = _TinyMeshGNN()
        predictor._temporal_backbone = "mamba"
        predictor._temporal_location = "mesh_processor_interleaved"
        predictor._temporal_stateful = True
        predictor._temporal_d_inner = 4
        predictor._temporal_d_state = 3
        predictor._temporal_d_conv = 1
        predictor._temporal_dt_rank = "auto"
        predictor._temporal_bias = False
        predictor._temporal_conv_bias = True
        predictor._temporal_layers = 1
        predictor._temporal_dropout = 0.0
        predictor._temporal_zero_init_out = True
        predictor._temporal_insert_count = None
        return gc.GraphCast._run_mesh_gnn_interleaved(predictor, x)

    transformed = hk.transform_with_state(forward)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((n_mesh, batch_size, channels), dtype=jnp.float32)

    params, state = transformed.init(rng, x)
    y, _ = transformed.apply(params, state, rng, x)

    np.testing.assert_allclose(
        np.asarray(y),
        np.asarray(x + jnp.asarray(0.1, x.dtype)),
        rtol=1e-6,
        atol=1e-6,
    )
    zero_out_proj = [
        value
        for module_name, param_name, value in hk.data_structures.traverse(params)
        if "mamba_block_0/out_proj" in module_name and param_name == "w"
    ]
    assert zero_out_proj
    for value in zero_out_proj:
        np.testing.assert_allclose(np.asarray(value), 0.0, rtol=0.0, atol=0.0)


def test_interleaved_processor_remat_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    remat_calls = 0

    def fake_remat(fn):
        nonlocal remat_calls
        remat_calls += 1
        return fn

    class _NoopTemporalBlock:
        def __init__(self, cfg, name=None):
            del name
            self.cfg = cfg

        def __call__(self, node_features, **kwargs):
            del kwargs
            return node_features

    monkeypatch.setattr(gc.hk, "remat", fake_remat)
    monkeypatch.setattr(gc, "_get_temporal_block_cls", lambda _stateful: _NoopTemporalBlock)

    def run(enabled: bool) -> None:
        graph = _TinyGraph(
            nodes={"mesh_nodes": _NodeSet(features=jnp.zeros((5, 1), dtype=jnp.float32))},
            edges={"mesh": _EdgeSet(features=jnp.zeros((4, 1), dtype=jnp.float32))},
        )
        predictor = object.__new__(gc.GraphCast)
        predictor._mesh_graph_structure = graph
        predictor._mesh_gnn = _TinyMeshGNN()
        predictor._temporal_backbone = "none"
        predictor._temporal_location = "mesh_processor_interleaved"
        predictor._temporal_stateful = False
        predictor._temporal_d_inner = None
        predictor._temporal_d_state = 16
        predictor._temporal_d_conv = 4
        predictor._temporal_dt_rank = "auto"
        predictor._temporal_bias = False
        predictor._temporal_conv_bias = True
        predictor._temporal_layers = 1
        predictor._temporal_dropout = 0.0
        predictor._temporal_zero_init_out = False
        predictor._temporal_insert_count = None
        predictor._remat_processor_steps = enabled
        x = jnp.ones((5, 2, 8), dtype=jnp.float32)
        gc.GraphCast._run_mesh_gnn_interleaved(predictor, x, is_training=True)

    run(False)
    assert remat_calls == 0
    run(True)
    assert remat_calls == 1


def test_mesh2grid_remat_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    remat_calls = 0

    def fake_remat(fn):
        nonlocal remat_calls
        remat_calls += 1
        return fn

    monkeypatch.setattr(gc.hk, "remat", fake_remat)
    graph = _TinyGraph(
        nodes={
            "mesh_nodes": _NodeSet(features=jnp.zeros((2, 1), dtype=jnp.float32)),
            "grid_nodes": _NodeSet(features=jnp.zeros((3, 1), dtype=jnp.float32)),
        },
        edges={"mesh2grid": _EdgeSet(features=jnp.zeros((4, 1), dtype=jnp.float32))},
    )

    def run(enabled: bool):
        predictor = object.__new__(gc.GraphCast)
        predictor._mesh2grid_graph_structure = graph
        predictor._mesh2grid_gnn = lambda input_graph: input_graph
        predictor._remat_mesh2grid = enabled
        return gc.GraphCast._run_mesh2grid_gnn(
            predictor,
            jnp.ones((2, 2, 4), dtype=jnp.float32),
            jnp.ones((3, 2, 4), dtype=jnp.float32),
        )

    run(False)
    assert remat_calls == 0
    out = run(True)
    assert remat_calls == 1
    assert out.shape == (3, 2, 4)
