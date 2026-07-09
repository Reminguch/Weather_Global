"""Legacy GraphCast+Mamba predictor for 98-channel encoder checkpoints that treated single time-step."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import xarray
from graphcast import graphcast as gc


class LegacyGraphCastMamba(gc.GraphCast):
    """GraphCast+Mamba path used by old per-timestep encoder checkpoints."""

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: xarray.Dataset,
        is_training: bool = False,
    ) -> xarray.Dataset:
        self._maybe_init(inputs)

        if self._temporal_backbone == "none":
            grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
            latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(grid_node_features)
        else:
            grid_node_features = self._inputs_to_grid_node_features_by_time(inputs, forcings)
            latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn_over_time(grid_node_features)
            latent_grid_nodes = latent_grid_nodes[-1]
            if self._temporal_location == "mesh_post_encoder":
                latent_mesh_nodes = self._run_temporal_mesh_block(latent_mesh_nodes, is_training=is_training)

        updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes, is_training=is_training)
        if updated_latent_mesh_nodes.ndim == 4:
            updated_latent_mesh_nodes = updated_latent_mesh_nodes[-1]

        output_grid_nodes = self._run_mesh2grid_gnn(updated_latent_mesh_nodes, latent_grid_nodes)
        return self._grid_node_outputs_to_prediction(output_grid_nodes, targets_template)

    def _inputs_to_grid_node_features_by_time(
        self,
        inputs: xarray.Dataset,
        forcings: xarray.Dataset,
    ) -> chex.Array:
        """Converts xarrays to [time, num_grid_nodes, batch, channels]."""
        stacked_inputs = gc.model_utils.dataset_to_stacked(
            inputs, preserved_dims=("batch", "time", "lat", "lon")
        )
        grid_xarray_lat_lon_leading = gc.model_utils.lat_lon_to_leading_axes(stacked_inputs)
        input_features = gc.xarray_jax.unwrap(grid_xarray_lat_lon_leading.data).reshape(
            (-1,) + grid_xarray_lat_lon_leading.data.shape[2:]
        )
        input_features = jnp.transpose(input_features, (2, 0, 1, 3))

        stacked_forcings = gc.model_utils.dataset_to_stacked(forcings)
        forcings_lat_lon_leading = gc.model_utils.lat_lon_to_leading_axes(stacked_forcings)
        forcing_features = gc.xarray_jax.unwrap(forcings_lat_lon_leading.data).reshape(
            (-1,) + forcings_lat_lon_leading.data.shape[2:]
        )
        forcing_features = jnp.broadcast_to(
            forcing_features[None, ...],
            (input_features.shape[0],) + forcing_features.shape,
        )
        return jnp.concatenate([input_features, forcing_features], axis=-1)

    def _run_grid2mesh_gnn_over_time(self, grid_node_features: chex.Array) -> tuple[chex.Array, chex.Array]:
        """Runs grid2mesh independently per input timestep."""
        if grid_node_features.ndim != 4:
            raise ValueError(
                "Expected [time, num_grid_nodes, batch, channels], got "
                f"shape={grid_node_features.shape}"
            )

        latent_mesh_nodes = []
        latent_grid_nodes = []
        for time_i in range(grid_node_features.shape[0]):
            latent_mesh_t, latent_grid_t = self._run_grid2mesh_gnn(grid_node_features[time_i])
            latent_mesh_nodes.append(latent_mesh_t)
            latent_grid_nodes.append(latent_grid_t)
        return jnp.stack(latent_mesh_nodes, axis=0), jnp.stack(latent_grid_nodes, axis=0)

    def _run_mesh_gnn(
        self,
        latent_mesh_nodes: chex.Array,
        *,
        is_training: bool = False,
    ) -> chex.Array:
        if (
            latent_mesh_nodes.ndim == 4
            and self._temporal_backbone != "none"
            and self._temporal_location == "mesh_processor_interleaved"
        ):
            return self._run_mesh_gnn_interleaved(latent_mesh_nodes, is_training=is_training)
        return super()._run_mesh_gnn(latent_mesh_nodes, is_training=is_training)

    def _run_mesh_gnn_interleaved(
        self,
        latent_mesh_nodes: chex.Array,
        *,
        is_training: bool = False,
    ) -> chex.Array:
        """Runs mesh processor steps interleaved with temporal sequence blocks."""
        if latent_mesh_nodes.ndim != 4:
            raise ValueError(
                "Expected [time, num_mesh_nodes, batch, channels], got "
                f"shape={latent_mesh_nodes.shape}"
            )
        time_size, n_mesh, batch_size, _ = latent_mesh_nodes.shape

        mesh_graph = self._mesh_graph_structure
        assert mesh_graph is not None
        mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
        edges = mesh_graph.edges[mesh_edges_key]
        edge_features = gc._add_batch_second_axis(edges.features.astype(latent_mesh_nodes.dtype), batch_size)

        def build_graph(node_features):
            return mesh_graph._replace(
                edges={mesh_edges_key: edges._replace(features=edge_features)},
                nodes={"mesh_nodes": mesh_graph.nodes["mesh_nodes"]._replace(features=node_features)},
            )

        input_graph = build_graph(latent_mesh_nodes[0])
        embedder_network, processor_networks, _ = self._mesh_gnn._networks_builder(input_graph)

        latent_graphs = [
            self._mesh_gnn._embed(build_graph(latent_mesh_nodes[time_i]), embedder_network)
            for time_i in range(time_size)
        ]

        for repetition_i in range(self._mesh_gnn._num_processor_repetitions):
            for step_i, processor_network in enumerate(processor_networks):
                latent_graphs = [
                    self._mesh_gnn._process_step(processor_network, latent_graph_t)
                    for latent_graph_t in latent_graphs
                ]
                node_sequence = jnp.stack(
                    [graph_t.nodes["mesh_nodes"].features for graph_t in latent_graphs],
                    axis=0,
                )
                expected_prefix = (time_size, n_mesh, batch_size)
                if node_sequence.shape[:3] != expected_prefix:
                    raise ValueError(
                        "Unexpected interleaved mesh node sequence shape before temporal block: "
                        f"expected prefix {expected_prefix}, got {node_sequence.shape}."
                    )

                temporal_block_name = f"mesh_interleaved_temporal_r{repetition_i}_s{step_i}"
                temporal_block = gc._get_temporal_block_cls(self._temporal_stateful)(
                    gc.TemporalMeshConfig(
                        backbone=self._temporal_backbone,
                        location=self._temporal_location,
                        hidden_size=self._temporal_hidden_size,
                        d_inner=self._temporal_d_inner,
                        d_state=self._temporal_d_state,
                        dt_rank=self._temporal_dt_rank,
                        d_conv=self._temporal_d_conv,
                        layers=self._temporal_layers,
                        bias=self._temporal_bias,
                        conv_bias=self._temporal_conv_bias,
                        dropout=self._temporal_dropout,
                        zero_init_output=self._temporal_zero_init_out,
                    ),
                    name=temporal_block_name,
                )
                if self._temporal_stateful:
                    temporal_state = gc.load_temporal_state_from_haiku(
                        f"{temporal_block_name}_state",
                        temporal_block.cfg,
                        batch_size=batch_size,
                        n_mesh=node_sequence.shape[1],
                        dtype=node_sequence.dtype,
                    )
                    node_sequence, next_temporal_state = temporal_block(
                        node_sequence,
                        prev_state=temporal_state,
                        is_training=is_training,
                    )
                    gc.store_temporal_state_to_haiku(f"{temporal_block_name}_state", next_temporal_state)
                else:
                    node_sequence = temporal_block(node_sequence, is_training=is_training)

                if node_sequence.shape[:3] != expected_prefix:
                    raise ValueError(
                        "Unexpected interleaved mesh node sequence shape after temporal block: "
                        f"expected prefix {expected_prefix}, got {node_sequence.shape}."
                    )
                latent_graphs = [
                    graph_t._replace(
                        nodes={
                            "mesh_nodes": graph_t.nodes["mesh_nodes"]._replace(features=node_sequence[time_i])
                        }
                    )
                    for time_i, graph_t in enumerate(latent_graphs)
                ]

        return jnp.stack([graph_t.nodes["mesh_nodes"].features for graph_t in latent_graphs], axis=0)

    def _run_temporal_mesh_block(
        self,
        latent_mesh_nodes: chex.Array,
        is_training: bool = False,
    ) -> chex.Array:
        if latent_mesh_nodes.ndim not in (3, 4):
            raise ValueError(
                "Expected mesh latent rank 3 or 4, got "
                f"{latent_mesh_nodes.ndim} with shape={latent_mesh_nodes.shape}"
            )
        if not hasattr(self, "_temporal_block"):
            self._temporal_block = gc._get_temporal_block_cls(self._temporal_stateful)(
                gc.TemporalMeshConfig(
                    backbone=self._temporal_backbone,
                    location=self._temporal_location,
                    hidden_size=self._temporal_hidden_size,
                    d_inner=self._temporal_d_inner,
                    d_state=self._temporal_d_state,
                    dt_rank=self._temporal_dt_rank,
                    d_conv=self._temporal_d_conv,
                    layers=self._temporal_layers,
                    bias=self._temporal_bias,
                    conv_bias=self._temporal_conv_bias,
                    dropout=self._temporal_dropout,
                    zero_init_output=self._temporal_zero_init_out,
                ),
                name="temporal_mesh_block",
            )
        if not self._temporal_stateful:
            return self._temporal_block(latent_mesh_nodes, is_training=is_training)

        batch_size = latent_mesh_nodes.shape[2] if latent_mesh_nodes.ndim == 4 else latent_mesh_nodes.shape[1]
        n_mesh = latent_mesh_nodes.shape[1] if latent_mesh_nodes.ndim == 4 else latent_mesh_nodes.shape[0]
        temporal_state = gc.load_temporal_state_from_haiku(
            "temporal_mesh_block_state",
            self._temporal_block.cfg,
            batch_size=batch_size,
            n_mesh=n_mesh,
            dtype=latent_mesh_nodes.dtype,
        )
        output, next_temporal_state = self._temporal_block(
            latent_mesh_nodes,
            prev_state=temporal_state,
            is_training=is_training,
        )
        gc.store_temporal_state_to_haiku("temporal_mesh_block_state", next_temporal_state)
        return output
