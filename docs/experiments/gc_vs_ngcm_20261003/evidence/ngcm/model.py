"""GraphCast connectivity and interleaved Mamba on one native Gaussian frame.

The small subclass only changes the array/schema boundary. No GC baseline,
pressure-level task features, pretrained GC overlay, or predicted baseline input.
"""
from __future__ import annotations

from pathlib import Path
import sys
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/graphcast"))
from graphcast import graphcast as gc  # noqa: E402


class NativeResidualGraph(gc.GraphCast):
    def __init__(self, architecture, latitude, longitude, output_size):
        # Each named native channel is one output, independent of GC's field registry.
        task = gc.TaskConfig(input_variables=(), target_variables=tuple(
            f"native_channel_{i}" for i in range(output_size)), forcing_variables=(),
            pressure_levels=(), input_duration="6h")
        config = gc.ModelConfig(resolution=0.0, mesh_size=architecture.mesh_size,
                               latent_size=architecture.width, gnn_msg_steps=architecture.msg_steps,
                               hidden_layers=1, radius_query_fraction_edge_length=0.6)
        super().__init__(config, task)
        self._temporal_backbone = "mamba"
        self._temporal_location = "mesh_processor_interleaved"
        self._temporal_stateful = True
        self._temporal_use_full_mamba = True
        self._temporal_d_inner = architecture.d_inner
        self._temporal_d_state = architecture.d_state
        self._temporal_d_conv = architecture.d_conv
        self._temporal_bc_groups = architecture.bc_groups
        self._temporal_dt_rank = architecture.dt_rank
        self._temporal_layers = architecture.temporal_layers
        self._temporal_init_scheme = architecture.init_scheme
        self._temporal_dropout = architecture.dropout
        self._temporal_zero_init_out = False
        self._maybe_init(xr.Dataset(coords={"lat": np.asarray(latitude), "lon": np.asarray(longitude)}))
        self.output_size = output_size
        self.geometry = {"grid_nodes": self._num_grid_nodes, "mesh_nodes": self._num_mesh_nodes,
                         "query_radius": float(self._query_radius), "edges": {}}
        for graph in (self._grid2mesh_graph_structure, self._mesh_graph_structure, self._mesh2grid_graph_structure):
            for key, edge in graph.edges.items():
                count = int(np.asarray(graph.nodes[key.node_sets[1]].n_node).sum())
                degree = np.bincount(edge.indices.receivers, minlength=count)
                if degree.shape != (count,) or not np.all(degree > 0):
                    raise ValueError(f"Disconnected receiver nodes in {key.name}; revise the fixed mesh before production")
                self.geometry["edges"][key.name] = {"count": int(len(edge.indices.receivers)),
                    "receiver_degree_min": int(degree.min()), "receiver_degree_max": int(degree.max())}

    def __call__(self, native_features, known_features):
        x = jnp.concatenate((native_features, known_features), axis=-1).astype(jnp.float32)
        if x.ndim != 3:
            raise ValueError("Single chronological lane requires [lon,lat,channel]")
        x = jnp.transpose(x, (1, 0, 2)).reshape(self._num_grid_nodes, 1, -1)
        mesh, grid = self._run_grid2mesh_gnn(x)
        mesh = self._run_mesh_gnn(mesh, is_training=False)
        output = self._run_mesh2grid_gnn(mesh, grid)
        output = hk.Linear(self.output_size, w_init=hk.initializers.Constant(0),
                           b_init=hk.initializers.Constant(0), name="native_zero_head")(output)
        return output[:, 0].reshape(len(self._grid_lat), len(self._grid_lon), -1).transpose(1, 0, 2)


def make_branch(architecture, latitude_radians, longitude_radians, output_size):
    lat, lon = np.rad2deg(latitude_radians), np.rad2deg(longitude_radians)
    def forward(features, known):
        return NativeResidualGraph(architecture, lat, lon, output_size)(features, known)
    transformed = hk.transform_with_state(forward)
    return hk.TransformedWithState(transformed.init, jax.jit(transformed.apply))


def zero_memory(state):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), state)


def geometry_manifest(architecture, latitude_radians, longitude_radians, output_size):
    @hk.transform
    def inspect():
        graph = NativeResidualGraph(architecture, np.rad2deg(latitude_radians),
                                    np.rad2deg(longitude_radians), output_size)
        return graph.geometry
    return inspect.apply({}, None)


def parameter_counts(params, state):
    return {"trainable_parameters": sum(int(x.size) for x in jax.tree_util.tree_leaves(params)),
            "recurrent_bytes": sum(int(x.size * x.dtype.itemsize) for x in jax.tree_util.tree_leaves(state))}
