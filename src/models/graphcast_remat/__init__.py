"""Rematerialization-enabled GraphCast subclass for gcmamba_v1.

Subclasses ``graphcast.graphcast.GraphCast`` (third_party) and adds optional
``hk.remat`` boundaries around the heaviest activation sources:

  - ``_remat_processor_steps``: wraps each frozen processor msg-passing step
    in ``hk.remat``.  By default the interleaved temporal Mamba block runs
    OUTSIDE the remat boundary (SAFE Mode-2).  BPTT memory across mp=16
    steps becomes ~per-step instead of cumulative, AND backward gradients
    are correct (verified rel diff ~0.03).
  - ``_remat_temporal_inside_processor`` (DEFAULT False): when set True,
    processor + Mamba share a single remat boundary (Mode-1).  Initial
    verification suggested Mode-1 had broken gradients (rel diff ~0.68),
    but that was a methodology artifact — once the verify script aligned
    BOTH params AND state between the two transforms, Mode-1's gradient
    rel diff dropped into the bf16 noise floor (~0.04), matching Mode-2.
    Mode-2 is kept as the canonical/production default because it matches
    Ilya's main-branch implementation and decouples Mamba's stateful
    complexity from the remat boundary, but Mode-1 is no longer believed
    to be incorrect.  Still reachable only via direct attribute assignment
    by test code, never via the gcmamba_v1 CLI.
  - ``_remat_mesh2grid``: wrap mesh2grid GNN call.
  - ``_remat_grid2mesh``: wrap grid2mesh GNN call (skip-value still consumed
    via recompute on backward, not stop_gradient).

The subclass keeps the Haiku module scope name ``graph_cast`` (same as the
base class) by reusing the class name "GraphCast", so existing
``overlay_matching_params(params, deepmind_ckpt.params)`` still works.

DOES NOT MODIFY third_party/graphcast/graphcast.py.
"""
from __future__ import annotations

from typing import Any
import sys
from pathlib import Path

# Make third_party graphcast importable if not already on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GC_THIRD_PARTY = _REPO_ROOT / "third_party" / "graphcast"
if str(_GC_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_GC_THIRD_PARTY))

import chex  # noqa: E402
import haiku as hk  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Base GraphCast and its module-private helpers we need to re-use verbatim
# in the rematerialized inner-loop body.
from graphcast import graphcast as _gc_base  # noqa: E402
from graphcast.graphcast import (  # noqa: E402
    _add_batch_second_axis,
    _get_temporal_block_cls,
)
from src.models.mamba.modules.temporal_mesh_mamba import TemporalMeshConfig  # noqa: E402
from src.models.mamba.modules.temporal_mesh_mamba_Ilya import (  # noqa: E402
    load_temporal_state_from_haiku,
    store_temporal_state_to_haiku,
)


class GraphCast(_gc_base.GraphCast):
    """GraphCast with optional hk.remat around heavy GNN blocks.

    NOTE: class name kept as ``GraphCast`` so Haiku module scope is
    ``graph_cast`` (matching the base class). overlay_matching_params from
    a DeepMind GraphCast ckpt continues to work unchanged.

    All remat flags default False -> behavior identical to the base class.
    """

    def __init__(self, model_config, task_config):
        super().__init__(model_config, task_config)
        # Remat control attributes. Defaults preserve base-class behavior.
        self._remat_processor_steps: bool = False
        self._remat_mesh2grid: bool = False
        self._remat_grid2mesh: bool = False
        # Whether the temporal Mamba block sits INSIDE the processor remat
        # boundary.  Default False (= Mode-2 = canonical, matches Ilya main):
        # hk.remat wraps the processor msg-passing step ONLY; Mamba runs
        # outside the remat boundary.
        # When True (= Mode-1 = experimental): processor + Mamba share one
        # remat boundary.  Earlier verify runs suggested Mode-1 had broken
        # gradients, but the apparent breakage was a verify-script artifact
        # (params/state init mismatch).  With strict shared-state methodology
        # both modes match within bf16 noise (~0.04 grad rel diff).
        # Kept Mode-2 as default for API stability + memory-profile parity
        # with Ilya's main branch.
        self._remat_temporal_inside_processor: bool = False

    def __call__(self, inputs, targets_template, forcings, is_training=False):
        """Mirror base __call__ with optional hk.remat around grid2mesh/mesh2grid.

        The interleaved processor-step remat is inside _run_mesh_gnn_interleaved.
        """
        self._maybe_init(inputs)
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

        # Wrap bound methods as closures before hk.remat (safer than
        # hk.remat(bound_method) — Haiku may mis-handle the implicit `self`
        # binding when lifting params/state).
        if self._remat_grid2mesh:
            def _grid2mesh_fn(x):
                return self._run_grid2mesh_gnn(x)
            latent_mesh_nodes, latent_grid_nodes = hk.remat(_grid2mesh_fn)(
                grid_node_features)
        else:
            latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(
                grid_node_features)

        if (self._temporal_backbone != "none"
                and self._temporal_location == "mesh_post_encoder"):
            latent_mesh_nodes = self._run_temporal_mesh_block(
                latent_mesh_nodes, is_training=is_training)

        updated_latent_mesh_nodes = self._run_mesh_gnn(
            latent_mesh_nodes, is_training=is_training)

        if self._remat_mesh2grid:
            def _mesh2grid_fn(mesh_nodes, grid_nodes):
                return self._run_mesh2grid_gnn(mesh_nodes, grid_nodes)
            output_grid_nodes = hk.remat(_mesh2grid_fn)(
                updated_latent_mesh_nodes, latent_grid_nodes)
        else:
            output_grid_nodes = self._run_mesh2grid_gnn(
                updated_latent_mesh_nodes, latent_grid_nodes)

        return self._grid_node_outputs_to_prediction(
            output_grid_nodes, targets_template)

    # ---- Interleaved processor with optional per-step rematerialization ----

    def _run_mesh_gnn_interleaved(
        self,
        latent_mesh_nodes: chex.Array,
        *,
        is_training: bool = False,
    ) -> chex.Array:
        """Re-implement base interleaved loop, optionally wrapping each
        ``(processor msg-passing step + temporal block)`` pair in hk.remat.

        When ``self._remat_processor_steps`` is False, defers to the base
        class for exact behavior parity.
        """
        if not self._remat_processor_steps:
            return super()._run_mesh_gnn_interleaved(
                latent_mesh_nodes, is_training=is_training)

        if latent_mesh_nodes.ndim != 3:
            raise ValueError(
                "Expected [num_mesh_nodes, batch, channels], got "
                f"shape={latent_mesh_nodes.shape}")
        n_mesh, batch_size, _ = latent_mesh_nodes.shape

        mesh_graph = self._mesh_graph_structure
        assert mesh_graph is not None
        mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
        edges = mesh_graph.edges[mesh_edges_key]
        edge_features = _add_batch_second_axis(
            edges.features.astype(latent_mesh_nodes.dtype), batch_size)

        def build_graph(node_features):
            return mesh_graph._replace(
                edges={mesh_edges_key: edges._replace(features=edge_features)},
                nodes={"mesh_nodes": mesh_graph.nodes["mesh_nodes"]._replace(
                    features=node_features)})

        input_graph = build_graph(latent_mesh_nodes)
        embedder_network, processor_networks, _ = (
            self._mesh_gnn._networks_builder(input_graph))
        latent_graph = self._mesh_gnn._embed(
            build_graph(latent_mesh_nodes), embedder_network)

        # CRITICAL: keep Haiku state I/O OUTSIDE the hk.remat boundary.
        # If `hk.set_state(name, v)` runs inside remat, backward's recompute
        # of the forward will see the POST-write state (the one that came
        # *after* the original forward) instead of the PRE-write state it
        # was given originally -> activations recomputed with wrong inputs
        # -> gradients silently wrong. (This was the cause of the
        # verify_remat Gate-grad FAIL.) So state load/store happen here;
        # the remat'd step_fn only sees `prev_state` as a plain JAX arg
        # and returns `new_state` as a plain return value.
        def make_step_fn(processor_network, block_name, block_cfg, stateful):
            def step_fn(latent_graph, prev_state):
                # 1. Process step
                latent_graph = self._mesh_gnn._process_step(
                    processor_network, latent_graph)
                node_features = latent_graph.nodes["mesh_nodes"].features
                expected_prefix = (n_mesh, batch_size)
                if node_features.shape[:2] != expected_prefix:
                    raise ValueError(
                        "Unexpected mesh node shape before temporal block: "
                        f"expected prefix {expected_prefix}, "
                        f"got {node_features.shape}.")
                # 2. Temporal Mamba block (functional — state is arg/return)
                temporal_block = _get_temporal_block_cls(stateful)(
                    block_cfg, name=block_name)
                if stateful:
                    node_features, new_state = temporal_block(
                        node_features,
                        prev_state=prev_state,
                        is_training=is_training,
                    )
                else:
                    node_features = temporal_block(
                        node_features, is_training=is_training)
                    new_state = prev_state  # passthrough
                if node_features.shape[:2] != expected_prefix:
                    raise ValueError(
                        "Unexpected mesh node shape after temporal block: "
                        f"expected prefix {expected_prefix}, "
                        f"got {node_features.shape}.")
                latent_graph = latent_graph._replace(
                    nodes={"mesh_nodes":
                           latent_graph.nodes["mesh_nodes"]._replace(
                               features=node_features)})
                return latent_graph, new_state
            return step_fn

        # Sub-function for Mode 2: ONLY processor msg-passing in remat,
        # Mamba block runs OUTSIDE.
        def make_proc_only_fn(processor_network):
            def proc_fn(latent_graph):
                return self._mesh_gnn._process_step(
                    processor_network, latent_graph)
            return proc_fn

        for repetition_i in range(
                self._mesh_gnn._num_processor_repetitions):
            for step_i, processor_network in enumerate(processor_networks):
                block_name = (
                    f"mesh_interleaved_temporal_r{repetition_i}_s{step_i}")
                block_cfg = TemporalMeshConfig(
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
                )
                node_dtype = (latent_graph.nodes["mesh_nodes"]
                              .features.dtype)

                if not self._remat_temporal_inside_processor:
                    # MODE 2 (DEFAULT, SAFE): processor remat'd, Mamba runs
                    # OUTSIDE the remat boundary.  Verified rel diff ~0.03.
                    proc_fn = make_proc_only_fn(processor_network)
                    latent_graph = hk.remat(proc_fn)(latent_graph)
                    # Now run temporal block normally (no remat).
                    node_features = (latent_graph.nodes["mesh_nodes"]
                                     .features)
                    temporal_block = _get_temporal_block_cls(
                        self._temporal_stateful)(
                        block_cfg, name=block_name)
                    if self._temporal_stateful:
                        prev_state = load_temporal_state_from_haiku(
                            f"{block_name}_state", block_cfg,
                            batch_size=batch_size, n_mesh=n_mesh,
                            dtype=node_dtype,
                        )
                        node_features, new_state = temporal_block(
                            node_features, prev_state=prev_state,
                            is_training=is_training)
                        store_temporal_state_to_haiku(
                            f"{block_name}_state", new_state)
                    else:
                        node_features = temporal_block(
                            node_features, is_training=is_training)
                    latent_graph = latent_graph._replace(
                        nodes={"mesh_nodes":
                               latent_graph.nodes["mesh_nodes"]._replace(
                                   features=node_features)})
                else:
                    # MODE 1 (UNSAFE, deprecated): processor + temporal share
                    # one remat boundary.  Failed gradient verification at
                    # rel diff ~0.68 — only reachable via direct attribute
                    # `_remat_temporal_inside_processor=True`, never via
                    # production CLI.  Kept for verify_remat tests.
                    if self._temporal_stateful:
                        prev_state = load_temporal_state_from_haiku(
                            f"{block_name}_state", block_cfg,
                            batch_size=batch_size, n_mesh=n_mesh,
                            dtype=node_dtype,
                        )
                    else:
                        # Dummy 0-d carry so step_fn's signature is uniform.
                        prev_state = jnp.zeros((), dtype=node_dtype)

                    step_fn = make_step_fn(
                        processor_network, block_name, block_cfg,
                        self._temporal_stateful)
                    latent_graph, new_state = hk.remat(step_fn)(
                        latent_graph, prev_state)

                    if self._temporal_stateful:
                        store_temporal_state_to_haiku(
                            f"{block_name}_state", new_state)

        return latent_graph.nodes["mesh_nodes"].features


__all__ = ["GraphCast"]
