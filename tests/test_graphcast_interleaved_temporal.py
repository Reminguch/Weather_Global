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


def test_stateful_interleaved_temporal_keeps_mesh_and_batch_axes_distinct() -> None:
    time_size = 3
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
        predictor._temporal_hidden_size = channels
        predictor._temporal_d_inner = 4
        predictor._temporal_d_state = 3
        predictor._temporal_d_conv = 1
        predictor._temporal_dt_rank = "auto"
        predictor._temporal_bias = False
        predictor._temporal_conv_bias = True
        predictor._temporal_layers = 1
        predictor._temporal_dropout = 0.0
        return gc.GraphCast._run_mesh_gnn_interleaved(predictor, x)

    transformed = hk.transform_with_state(forward)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((time_size, n_mesh, batch_size, channels), dtype=jnp.float32)

    params, state = transformed.init(rng, x)
    y, next_state = transformed.apply(params, state, rng, x)

    assert y.shape == (time_size, n_mesh, batch_size, channels)
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
