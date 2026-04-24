"""MZ-residual Mamba with grid <-> mesh <-> grid pathway.

Pipeline (same Mori-Zwanzig decomposition, different spatial operator):

    Grid [B, lat, lon, 2F]
        -> input_proj (2F -> H)        # per grid point, no mixing
        -> Grid2Mesh aggregate           # fixed geometric KNN, [B, M, H]
        -> optional mesh_pre MLP (H -> H)
        -> Mamba over time on [B*M, T, H]
        -> optional mesh_post MLP (H -> H)
        -> Mesh2Grid aggregate           # fixed geometric KNN, [B, lat, lon, H]
        -> residual_head (H -> F)       # per grid point

Crucially, the per-grid-point Mamba becomes a per-mesh-node Mamba. Mesh nodes
are O(100)-O(10k) vs grid points O(10k)-O(100k), so the parallel dimension P_mesh
shrinks dramatically and H can grow to compensate within the same memory
budget. The Grid2Mesh / Mesh2Grid operators also let grid points "talk" through
the mesh bottleneck, which the original per-grid-point module could not do.

All variables share the same pathway — no per-variable heads. At inference
time the caller is expected to split the output feature axis back to per-
variable slices for plotting.

Parameter sharing between ``__call__`` (teacher-forced, parallel over T) and
``rollout_ar`` (autoregressive) follows the same pattern as the existing
module: all sub-modules are instantiated in ``__init__`` and reused across
modes/time steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from ..grid_mamba.mz_grid_mamba import _SelectiveSSMBlock  # reuse the Mamba block


@dataclass(frozen=True)
class MZResidualMeshedConfig:
    input_size: int                  # 2 * F (concatenation of current_state, prev_residual)
    output_size: int                 # F (all resolved features)
    hidden_size: int = 128           # can go much bigger than the grid variant
    layers: int = 1
    dropout: float = 0.0
    a_log_init: float = -0.1
    use_mesh_pre_mlp: bool = True    # small MLP after Grid2Mesh
    use_mesh_post_mlp: bool = True   # small MLP before Mesh2Grid


# -----------------------------------------------------------------------------
# Geometric aggregation primitives. The index and weight tensors are fixed
# (precomputed in numpy). gather+weighted-sum compiles cleanly under jit.
# -----------------------------------------------------------------------------


def _grid_to_mesh_aggregate(
    x_b_pgrid_h: jax.Array,
    g2m_indices: jax.Array,
    g2m_weights: jax.Array,
) -> jax.Array:
    """(B, P_grid, H) -> (B, M, H) weighted by fixed KNN edges."""
    # x_b_pgrid_h:  [B, P_grid, H]
    # g2m_indices:  [M, K]
    # g2m_weights:  [M, K]
    gathered = x_b_pgrid_h[:, g2m_indices, :]          # [B, M, K, H]
    weighted = gathered * g2m_weights[None, :, :, None]
    return weighted.sum(axis=2)                        # [B, M, H]


def _mesh_to_grid_aggregate(
    x_b_m_h: jax.Array,
    m2g_indices: jax.Array,
    m2g_weights: jax.Array,
) -> jax.Array:
    """(B, M, H) -> (B, P_grid, H) weighted by fixed KNN edges."""
    gathered = x_b_m_h[:, m2g_indices, :]              # [B, P_grid, K, H]
    weighted = gathered * m2g_weights[None, :, :, None]
    return weighted.sum(axis=2)                        # [B, P_grid, H]


# -----------------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------------


class MZResidualMeshedMamba(hk.Module):
    """Meshed MZ-residual Mamba; same external interface as MZResidualMamba.

    Parameters
    ----------
    cfg : MZResidualMeshedConfig
    g2m_indices, g2m_weights : jnp.ndarray
        Fixed geometric tensors from ``mesh_ops.build_grid_mesh_projections``.
    m2g_indices, m2g_weights : jnp.ndarray
        Same, for the mesh->grid direction.
    n_mesh_nodes : int
        Number of mesh nodes (M). Pass explicitly to make shape checks robust.
    """

    def __init__(
        self,
        cfg: MZResidualMeshedConfig,
        *,
        g2m_indices: np.ndarray | jax.Array,
        g2m_weights: np.ndarray | jax.Array,
        m2g_indices: np.ndarray | jax.Array,
        m2g_weights: np.ndarray | jax.Array,
        n_mesh_nodes: int,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.cfg = cfg
        self._n_mesh = int(n_mesh_nodes)
        # Stash geometric tensors. Cast to jax/bf16-friendly dtypes.
        self._g2m_idx = jnp.asarray(g2m_indices, dtype=jnp.int32)
        self._g2m_w = jnp.asarray(g2m_weights, dtype=jnp.float32)
        self._m2g_idx = jnp.asarray(m2g_indices, dtype=jnp.int32)
        self._m2g_w = jnp.asarray(m2g_weights, dtype=jnp.float32)

        self._input_proj = hk.Linear(cfg.hidden_size, name="input_proj")
        self._mesh_pre = (
            hk.Linear(cfg.hidden_size, name="mesh_pre") if cfg.use_mesh_pre_mlp else None
        )
        self._mesh_post = (
            hk.Linear(cfg.hidden_size, name="mesh_post") if cfg.use_mesh_post_mlp else None
        )
        self._ssm_blocks = [
            _SelectiveSSMBlock(
                hidden_size=cfg.hidden_size,
                dropout=cfg.dropout,
                a_log_init=cfg.a_log_init,
                name=f"mz_mamba_block_{i}",
            )
            for i in range(cfg.layers)
        ]
        self._residual_head = hk.Linear(
            cfg.output_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="residual_head",
        )

    # ------------------------------------------------------------------ helpers
    def _encode_to_mesh(self, x_grid_b_p_f: jax.Array) -> jax.Array:
        """[B, P_grid, 2F] -> [B, M, H]"""
        h = self._input_proj(x_grid_b_p_f)
        h = _grid_to_mesh_aggregate(h, self._g2m_idx, self._g2m_w.astype(h.dtype))
        if self._mesh_pre is not None:
            h = jax.nn.silu(self._mesh_pre(h))
        return h

    def _decode_from_mesh(self, h_mesh_b_m_h: jax.Array) -> jax.Array:
        """[B, M, H] -> [B, P_grid, F_out]"""
        h = h_mesh_b_m_h
        if self._mesh_post is not None:
            h = jax.nn.silu(self._mesh_post(h))
        h = _mesh_to_grid_aggregate(h, self._m2g_idx, self._m2g_w.astype(h.dtype))
        return self._residual_head(h)

    # ---------------------------------------------------- teacher-forced mode
    def __call__(self, seq_tblnf: jax.Array, *, is_training: bool) -> jax.Array:
        if seq_tblnf.ndim != 5:
            raise ValueError(
                "Expected [time, batch, lat, lon, features], got "
                f"shape={seq_tblnf.shape}"
            )
        T, B, lat, lon, Fin = seq_tblnf.shape
        if Fin != self.cfg.input_size:
            raise ValueError(
                f"Expected input feature dim {self.cfg.input_size}, got {Fin}"
            )
        P_grid = lat * lon
        H = self.cfg.hidden_size

        x = seq_tblnf.reshape(T, B, P_grid, Fin)                 # [T, B, P, 2F]
        # Encode every time step independently grid -> mesh.
        x = x.reshape(T * B, P_grid, Fin)
        x_mesh = self._encode_to_mesh(x)                         # [T*B, M, H]
        x_mesh = x_mesh.reshape(T, B, self._n_mesh, H)

        # Mamba over time for each (batch, mesh) sequence.
        x_mesh = jnp.transpose(x_mesh, (1, 2, 0, 3))             # [B, M, T, H]
        x_mesh = x_mesh.reshape(B * self._n_mesh, T, H)
        for ssm in self._ssm_blocks:
            x_mesh = ssm(x_mesh, is_training=is_training)
        x_mesh = x_mesh.reshape(B, self._n_mesh, T, H)
        x_mesh = jnp.transpose(x_mesh, (2, 0, 1, 3))             # [T, B, M, H]

        # Decode each time step back to grid.
        x_mesh = x_mesh.reshape(T * B, self._n_mesh, H)
        y = self._decode_from_mesh(x_mesh)                       # [T*B, P, F_out]
        y = y.reshape(T, B, lat, lon, self.cfg.output_size)
        return y

    # ------------------------------------------------------- autoregressive
    def rollout_ar(
        self,
        current_state_n_tblnf: jax.Array,
        *,
        is_training: bool,
        true_prev_residual_n_tblnf: jax.Array | None = None,
        teacher_forcing_prob: float = 0.0,
        tf_mask_per_step: jax.Array | None = None,
        residual_clip: float | None = None,
        baseline_absolute_n_tblnf: jax.Array | None = None,
        residual_to_state_rescale_f: jax.Array | None = None,
    ) -> jax.Array:
        """Autoregressive rollout. Semantics mirror
        ``mz_residual_mamba.MZResidualMamba.rollout_ar``; only the spatial
        operator changes.
        """
        if current_state_n_tblnf.ndim != 5:
            raise ValueError(
                "Expected [time, batch, lat, lon, features], got "
                f"shape={current_state_n_tblnf.shape}"
            )
        T, B, lat, lon, F = current_state_n_tblnf.shape
        if F != self.cfg.output_size:
            raise ValueError(
                "rollout_ar expects current_state_n feature dim == output_size, "
                f"got {F} vs output_size={self.cfg.output_size}"
            )
        if 2 * F != self.cfg.input_size:
            raise ValueError(
                "rollout_ar assumes the network was built with "
                f"input_size = 2 * output_size. Got input_size={self.cfg.input_size}, "
                f"output_size={self.cfg.output_size}."
            )
        P_grid = lat * lon
        H = self.cfg.hidden_size
        M = self._n_mesh
        dtype = current_state_n_tblnf.dtype

        # Flatten spatial dims, keep [T, B, P_grid, F] layout.
        cs = current_state_n_tblnf.reshape(T, B, P_grid, F)
        if true_prev_residual_n_tblnf is not None:
            tpr = true_prev_residual_n_tblnf.reshape(T, B, P_grid, F)
        else:
            tpr = None
        if baseline_absolute_n_tblnf is not None:
            bsln_n = baseline_absolute_n_tblnf.reshape(T, B, P_grid, F)
        else:
            bsln_n = None

        use_state_feedback = (
            bsln_n is not None and residual_to_state_rescale_f is not None
        )
        rescale_f = (
            residual_to_state_rescale_f.astype(dtype)[None, None, :]
            if use_state_feedback
            else None
        )

        if tf_mask_per_step is not None:
            if tf_mask_per_step.shape != (T,):
                raise ValueError(
                    f"tf_mask_per_step must have shape (T={T},), got "
                    f"{tf_mask_per_step.shape}"
                )
            tf_mask_per_step = tf_mask_per_step.astype(dtype)

        prev_residual = jnp.zeros((B, P_grid, F), dtype=dtype)
        h_states: list[jax.Array] = [
            jnp.zeros((B * M, H), dtype=dtype) for _ in range(self.cfg.layers)
        ]

        preds: list[jax.Array] = []
        for t in range(T):
            # ------- choose prev_residual (grid-space) -------------------
            if t == 0 or tpr is None:
                r_prev = prev_residual
            elif tf_mask_per_step is not None:
                m_t = tf_mask_per_step[t]
                r_prev = m_t * tpr[t] + (1.0 - m_t) * prev_residual
            else:
                key = hk.next_rng_key()
                mask = (
                    jax.random.uniform(key, shape=(B, 1, 1)) < teacher_forcing_prob
                ).astype(dtype)
                r_prev = mask * tpr[t] + (1.0 - mask) * prev_residual

            # ------- choose current_state (grid-space, Option-2) ---------
            if use_state_feedback and t > 0:
                state_from_feedback = bsln_n[t - 1] + prev_residual * rescale_f
                if tf_mask_per_step is not None:
                    m_t = tf_mask_per_step[t]
                    cs_t = m_t * cs[t] + (1.0 - m_t) * state_from_feedback
                else:
                    cs_t = state_from_feedback
            else:
                cs_t = cs[t]

            # ------- grid -> mesh (per-step encoding) --------------------
            x_t = jnp.concatenate([cs_t, r_prev], axis=-1)       # [B, P, 2F]
            h_mesh = self._encode_to_mesh(x_t)                   # [B, M, H]
            h_mesh = h_mesh.reshape(B * M, H)

            # ------- one Mamba step per layer ----------------------------
            new_h_states: list[jax.Array] = []
            y_t = h_mesh
            for ssm, h_prev in zip(self._ssm_blocks, h_states):
                y_t, h_new = ssm.step(y_t, h_prev, is_training=is_training)
                new_h_states.append(h_new)
            h_states = new_h_states

            # ------- mesh -> grid + residual_head ------------------------
            y_t = y_t.reshape(B, M, H)
            pred_t = self._decode_from_mesh(y_t)                 # [B, P, F]
            if residual_clip is not None and residual_clip > 0:
                pred_t = jnp.clip(pred_t, -residual_clip, residual_clip)
            preds.append(pred_t)
            prev_residual = pred_t

        preds_tbpf = jnp.stack(preds, axis=0)                    # [T, B, P, F]
        preds_tbpf = preds_tbpf.reshape(T, B, lat, lon, -1)
        return preds_tbpf
