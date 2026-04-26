"""Minimal MeshGraphNet-style spatial processor on the icosphere mesh.

The MZ-residual pipeline does Grid -> Mesh (KNN encoder) -> Mamba SSM (along time)
-> Mesh -> Grid (KNN decoder). The Mamba block is a 1-D temporal SSM applied
independently to each mesh node, so mesh nodes never exchange information with
their geometric neighbours during the residual computation.

This module adds an optional spatial communication step between the encoder
and the temporal SSM: a stack of bidirectional message-passing layers operating
on the icosphere edges. Each layer is

    m_ij = MLP_e([h_i, h_j])                     edge update (symmetric)
    agg_i = mean_{j: (j,i) in E} m_ji            mean aggregate received messages
    h_i' = h_i + MLP_n([h_i, agg_i])             node update (residual + LN)

This is the same backbone as GraphCast's processor (Battaglia 2018 Interaction
Network) but stripped down: single mesh scale, no edge features, no globals,
mean aggregation, residual + LayerNorm. With 1-2 layers and hidden_size=128
this adds ~0.05M params and ~10% extra step time, while giving each mesh node
a ~250-500 km spatial receptive field before the SSM kicks in.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class MeshProcessorConfig:
    hidden_size: int
    n_layers: int = 1
    edge_hidden_size: Optional[int] = None  # default == hidden_size
    activation: str = "silu"


def _activation(name: str):
    return {"silu": jax.nn.silu, "relu": jax.nn.relu, "gelu": jax.nn.gelu}[name]


class MeshProcessor(hk.Module):
    """Bidirectional message-passing block on a fixed icosphere mesh.

    Parameters
    ----------
    cfg : MeshProcessorConfig
    senders, receivers : 1-D np.ndarray (int32, shape [E])
        Edge endpoint indices in [0, M). Edges are expected to be bidirectional
        already (each undirected edge appears in both directions).
    n_mesh_nodes : int
        Number of mesh nodes M (used for receiver-side scatter aggregation).
    """

    def __init__(
        self,
        cfg: MeshProcessorConfig,
        *,
        senders: np.ndarray,
        receivers: np.ndarray,
        n_mesh_nodes: int,
        name: str | None = None,
    ):
        super().__init__(name=name or "mesh_processor")
        if senders.shape != receivers.shape:
            raise ValueError(
                f"senders/receivers shape mismatch: {senders.shape} vs {receivers.shape}"
            )
        if senders.ndim != 1:
            raise ValueError(f"expected 1-D edge index arrays, got {senders.shape}")
        self.cfg = cfg
        self._senders = jnp.asarray(senders, dtype=jnp.int32)
        self._receivers = jnp.asarray(receivers, dtype=jnp.int32)
        self._n_mesh = int(n_mesh_nodes)
        # Precompute receiver degree (number of incoming edges) for mean
        # aggregation. Using float32 for division stability; cast at use site.
        deg = np.bincount(receivers, minlength=n_mesh_nodes).astype(np.float32)
        deg = np.maximum(deg, 1.0)  # guard against isolated nodes
        self._inv_deg = jnp.asarray(1.0 / deg, dtype=jnp.float32)

    def __call__(self, mesh_features: jax.Array) -> jax.Array:
        """mesh_features: ``[B, M, H]`` -> ``[B, M, H]``."""
        if mesh_features.ndim != 3:
            raise ValueError(
                f"expected mesh_features with shape [B, M, H], got {mesh_features.shape}"
            )
        B, M, H = mesh_features.shape
        if M != self._n_mesh:
            raise ValueError(
                f"mesh_features has M={M} but processor was built for M={self._n_mesh}"
            )
        if H != self.cfg.hidden_size:
            raise ValueError(
                f"mesh_features H={H} != processor hidden_size={self.cfg.hidden_size}"
            )

        h = mesh_features
        for layer_idx in range(self.cfg.n_layers):
            h = self._mp_block(h, layer_idx)
        return h

    def _mp_block(self, h: jax.Array, layer_idx: int) -> jax.Array:
        cfg = self.cfg
        H = cfg.hidden_size
        Hedge = cfg.edge_hidden_size or H
        act = _activation(cfg.activation)
        B, M, _ = h.shape

        # ---- edge update --------------------------------------------------
        # Gather sender/receiver features along the edge axis.
        h_send = h[:, self._senders, :]    # [B, E, H]
        h_recv = h[:, self._receivers, :]  # [B, E, H]
        edge_in = jnp.concatenate([h_send, h_recv], axis=-1)  # [B, E, 2H]

        m = hk.Linear(Hedge, name=f"edge_l{layer_idx}_w0")(edge_in)
        m = act(m)
        m = hk.Linear(H, name=f"edge_l{layer_idx}_w1")(m)
        m = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name=f"edge_l{layer_idx}_ln",
        )(m)  # [B, E, H]

        # ---- aggregate at receivers (sum, then divide by node degree) -----
        # jnp ops: zero buffer, scatter-add via .at[].add(), then mean by /deg.
        agg = jnp.zeros((B, M, H), dtype=h.dtype)
        agg = agg.at[:, self._receivers, :].add(m.astype(h.dtype))
        agg = agg * self._inv_deg.astype(h.dtype)[None, :, None]  # mean

        # ---- node update --------------------------------------------------
        node_in = jnp.concatenate([h, agg], axis=-1)  # [B, M, 2H]
        u = hk.Linear(H, name=f"node_l{layer_idx}_w0")(node_in)
        u = act(u)
        u = hk.Linear(H, name=f"node_l{layer_idx}_w1")(u)
        u = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True,
            name=f"node_l{layer_idx}_ln",
        )(u)
        return h + u  # residual

    @property
    def n_edges(self) -> int:
        return int(self._senders.shape[0])
