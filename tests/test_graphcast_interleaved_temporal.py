from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))

from graphcast import graphcast as gc  # noqa: E402


def test_temporal_processor_group_sizes_for_mp6_sweep() -> None:
    assert gc._temporal_processor_group_sizes(6, 2) == [3, 3]
    assert gc._temporal_processor_group_sizes(6, 3) == [2, 2, 2]
    assert gc._temporal_processor_group_sizes(6, 6) == [1, 1, 1, 1, 1, 1]


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
