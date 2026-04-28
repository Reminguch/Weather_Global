"""MZ-residual model using the full (S6-style) Mamba block, meshed variant.

Same outer structure as MZResidualMeshedMamba (Grid->Mesh->Mamba->Mesh->Grid),
but the temporal block is FullMambaBlock instead of _SelectiveSSMBlock. The
per-step assembly of current_state + prev_residual, the Option-2 state
feedback in rollout_ar, and the TF-mask handling are all unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from .full_mamba_block import FullMambaBlock
from ..processor import MeshProcessor, MeshProcessorConfig


@dataclass(frozen=True)
class MZResidualFullMambaConfig:
    input_size: int                  # 2F
    output_size: int                 # F
    hidden_size: int = 128           # d_model (H)
    d_state: int = 16                # new: SSM state dim per inner channel
    expand: int = 2                  # inner-channel expansion factor
    layers: int = 1
    dropout: float = 0.0
    a_log_init_min: float = -3.0
    a_log_init_max: float = -0.1
    use_mesh_pre_mlp: bool = True
    use_mesh_post_mlp: bool = True
    # Optional spatial GNN processor between encoder and SSM. n_layers=0 -> off
    # (default, identical to pre-processor model). n_layers>=1 inserts a stack
    # of bidirectional MeshGraphNet blocks operating on icosphere edges, so
    # mesh nodes can exchange information with their geometric neighbours
    # before the temporal SSM kicks in.
    processor_layers: int = 0
    processor_hidden_size: int | None = None  # None -> hidden_size
    # Specialist heads (Mod G). When enabled, the final residual_head is split
    # into two parallel Linear projections: one for upper-air channels and one
    # for surface channels. Each gets its own weights and bias, so surface
    # gradients no longer pull on upper-air capacity (and vice versa). The
    # shared body up through the SSM remains identical.
    use_specialist_heads: bool = False
    # Channel indices, in the same order as the model's flat output axis F.
    # Required when use_specialist_heads=True. Concatenation order at output
    # is preserved by scatter according to these indices, so the model's
    # output shape stays [..., F] and downstream code is unchanged.
    upper_channel_indices: tuple = ()
    surface_channel_indices: tuple = ()


# ---- geometric aggregation (unchanged from mz_meshed) -----------------------


def _grid_to_mesh_aggregate(x_b_pgrid_h, g2m_indices, g2m_weights):
    gathered = x_b_pgrid_h[:, g2m_indices, :]
    weighted = gathered * g2m_weights[None, :, :, None]
    return weighted.sum(axis=2)


def _mesh_to_grid_aggregate(x_b_m_h, m2g_indices, m2g_weights):
    gathered = x_b_m_h[:, m2g_indices, :]
    weighted = gathered * m2g_weights[None, :, :, None]
    return weighted.sum(axis=2)


# ---- main module ------------------------------------------------------------


class MZResidualFullMambaMeshed(hk.Module):
    """Meshed MZ with full-Mamba (d_state>1, input-dep B/C) temporal block."""

    def __init__(
        self,
        cfg: MZResidualFullMambaConfig,
        *,
        g2m_indices: np.ndarray | jax.Array,
        g2m_weights: np.ndarray | jax.Array,
        m2g_indices: np.ndarray | jax.Array,
        m2g_weights: np.ndarray | jax.Array,
        n_mesh_nodes: int,
        mesh_senders: np.ndarray | None = None,
        mesh_receivers: np.ndarray | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.cfg = cfg
        self._n_mesh = int(n_mesh_nodes)
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

        # Optional spatial GNN processor on the icosphere mesh.
        if cfg.processor_layers > 0:
            if mesh_senders is None or mesh_receivers is None:
                raise ValueError(
                    "processor_layers > 0 requires mesh_senders and mesh_receivers "
                    "(call build_mesh_edges in mesh_ops to obtain them)."
                )
            proc_cfg = MeshProcessorConfig(
                hidden_size=cfg.processor_hidden_size or cfg.hidden_size,
                n_layers=cfg.processor_layers,
            )
            if proc_cfg.hidden_size != cfg.hidden_size:
                raise ValueError(
                    "processor_hidden_size must equal hidden_size for now "
                    "(no projection layer between encoder and processor)."
                )
            self._processor = MeshProcessor(
                proc_cfg,
                senders=mesh_senders,
                receivers=mesh_receivers,
                n_mesh_nodes=self._n_mesh,
                name="mesh_processor",
            )
        else:
            self._processor = None

        self._ssm_blocks = [
            FullMambaBlock(
                hidden_size=cfg.hidden_size,
                d_state=cfg.d_state,
                expand=cfg.expand,
                dropout=cfg.dropout,
                a_log_init_min=cfg.a_log_init_min,
                a_log_init_max=cfg.a_log_init_max,
                name=f"full_mamba_block_{i}",
            )
            for i in range(cfg.layers)
        ]
        if cfg.use_specialist_heads:
            n_up = len(cfg.upper_channel_indices)
            n_sf = len(cfg.surface_channel_indices)
            if n_up + n_sf != cfg.output_size:
                raise ValueError(
                    f"specialist heads: upper({n_up}) + surface({n_sf}) "
                    f"!= output_size ({cfg.output_size})"
                )
            self._residual_head = None
            self._upper_head = hk.Linear(
                n_up, w_init=hk.initializers.Constant(0.0),
                b_init=hk.initializers.Constant(0.0), name="upper_head",
            )
            self._surface_head = hk.Linear(
                n_sf, w_init=hk.initializers.Constant(0.0),
                b_init=hk.initializers.Constant(0.0), name="surface_head",
            )
            # Precompute a single permutation that turns concat([up, sf])
            # into the canonical channel order, so _decode_from_mesh can do
            # one concat + one take (= one gather) instead of two scatter
            # ops + zero-allocation. Scatter is ~3x slower than gather on
            # GPU and dominates the per-step cost when on the model's hot
            # path; this fix removes that without changing semantics.
            combined = list(cfg.upper_channel_indices) + list(cfg.surface_channel_indices)
            perm = np.zeros(cfg.output_size, dtype=np.int32)
            for j, ci in enumerate(combined):
                perm[ci] = j
            self._canonical_perm = jnp.asarray(perm, dtype=jnp.int32)
        else:
            self._residual_head = hk.Linear(
                cfg.output_size,
                w_init=hk.initializers.Constant(0.0),
                b_init=hk.initializers.Constant(0.0),
                name="residual_head",
            )
            self._upper_head = None
            self._surface_head = None

    # ------------------------------------------------- grid <-> mesh helpers
    def _encode_to_mesh(self, x_grid_b_p_f: jax.Array) -> jax.Array:
        h = self._input_proj(x_grid_b_p_f)
        h = _grid_to_mesh_aggregate(h, self._g2m_idx, self._g2m_w.astype(h.dtype))
        if self._mesh_pre is not None:
            h = jax.nn.silu(self._mesh_pre(h))
        return h

    def _decode_from_mesh(self, h_mesh_b_m_h: jax.Array) -> jax.Array:
        h = h_mesh_b_m_h
        if self._mesh_post is not None:
            h = jax.nn.silu(self._mesh_post(h))
        h = _mesh_to_grid_aggregate(h, self._m2g_idx, self._m2g_w.astype(h.dtype))
        if self._residual_head is not None:
            return self._residual_head(h)
        # Specialist-heads path: compute upper and surface independently,
        # then reorder via a single precomputed permutation gather. Faster
        # than two scatters into a zero-allocated buffer.
        up = self._upper_head(h)         # [..., n_up]
        sf = self._surface_head(h)       # [..., n_sf]
        combined = jnp.concatenate([up, sf], axis=-1)             # [..., F]
        return jnp.take(combined, self._canonical_perm, axis=-1)  # [..., F]

    # ------------------------------------------------------ teacher-forced
    def __call__(self, seq_tblnf: jax.Array, *, is_training: bool) -> jax.Array:
        if seq_tblnf.ndim != 5:
            raise ValueError(f"Expected 5D input, got shape={seq_tblnf.shape}")
        T, B, lat, lon, Fin = seq_tblnf.shape
        if Fin != self.cfg.input_size:
            raise ValueError(
                f"Expected input feature dim {self.cfg.input_size}, got {Fin}"
            )
        P_grid = lat * lon
        H = self.cfg.hidden_size
        x = seq_tblnf.reshape(T * B, P_grid, Fin)
        x_mesh = self._encode_to_mesh(x)                      # [T*B, M, H]
        if self._processor is not None:
            # Spatial message passing per (T*B) snapshot, no time mixing.
            x_mesh = self._processor(x_mesh)
        x_mesh = x_mesh.reshape(T, B, self._n_mesh, H)
        x_mesh = jnp.transpose(x_mesh, (1, 2, 0, 3))          # [B, M, T, H]
        x_mesh = x_mesh.reshape(B * self._n_mesh, T, H)
        for ssm in self._ssm_blocks:
            x_mesh = ssm(x_mesh, is_training=is_training)
        x_mesh = x_mesh.reshape(B, self._n_mesh, T, H)
        x_mesh = jnp.transpose(x_mesh, (2, 0, 1, 3))          # [T, B, M, H]
        x_mesh = x_mesh.reshape(T * B, self._n_mesh, H)
        y = self._decode_from_mesh(x_mesh)                    # [T*B, P, F_out]
        y = y.reshape(T, B, lat, lon, self.cfg.output_size)
        return y

    # ------------------------------------------------------- AR rollout
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
        allow_tf_at_t0: bool = False,
    ) -> jax.Array:
        if current_state_n_tblnf.ndim != 5:
            raise ValueError(f"Expected 5D, got {current_state_n_tblnf.shape}")
        T, B, lat, lon, F = current_state_n_tblnf.shape
        if F != self.cfg.output_size:
            raise ValueError(
                f"rollout_ar current_state feature dim {F} != output_size "
                f"{self.cfg.output_size}"
            )
        if 2 * F != self.cfg.input_size:
            raise ValueError(
                "rollout_ar assumes input_size = 2 * output_size."
            )
        P_grid = lat * lon
        H = self.cfg.hidden_size
        M = self._n_mesh
        D_inner = H * self.cfg.expand
        N = self.cfg.d_state
        dtype = current_state_n_tblnf.dtype

        cs = current_state_n_tblnf.reshape(T, B, P_grid, F)
        tpr = (true_prev_residual_n_tblnf.reshape(T, B, P_grid, F)
               if true_prev_residual_n_tblnf is not None else None)
        bsln_n = (baseline_absolute_n_tblnf.reshape(T, B, P_grid, F)
                  if baseline_absolute_n_tblnf is not None else None)
        use_state_feedback = (
            bsln_n is not None and residual_to_state_rescale_f is not None
        )
        rescale_f = (residual_to_state_rescale_f.astype(dtype)[None, None, :]
                     if use_state_feedback else None)

        if tf_mask_per_step is not None:
            if tf_mask_per_step.shape != (T,):
                raise ValueError(f"tf_mask shape {tf_mask_per_step.shape} != ({T},)")
            tf_mask_per_step = tf_mask_per_step.astype(dtype)

        prev_residual = jnp.zeros((B, P_grid, F), dtype=dtype)
        h_states: list[jax.Array] = [
            jnp.zeros((B * M, D_inner, N), dtype=dtype) for _ in range(self.cfg.layers)
        ]

        preds: list[jax.Array] = []
        for t in range(T):
            # Patch 2: under anchor-as-batch each anchor's first physical step
            # lands at t=0, and we want tf_mask[0]=1 to actually inject the
            # observable previous-anchor residual via tpr[0]. allow_tf_at_t0=True
            # opts in to this; allow_tf_at_t0=False preserves the v1 behaviour
            # where t=0 always uses prev_residual=zero (since legacy anchor_0
            # has no observable predecessor).
            if tpr is None:
                r_prev = prev_residual
            elif tf_mask_per_step is not None:
                if t == 0 and not allow_tf_at_t0:
                    r_prev = prev_residual
                else:
                    m_t = tf_mask_per_step[t]
                    r_prev = m_t * tpr[t] + (1.0 - m_t) * prev_residual
            else:
                if t == 0:
                    r_prev = prev_residual
                else:
                    key = hk.next_rng_key()
                    mask = (
                        jax.random.uniform(key, shape=(B, 1, 1)) < teacher_forcing_prob
                    ).astype(dtype)
                    r_prev = mask * tpr[t] + (1.0 - mask) * prev_residual

            if use_state_feedback and t > 0:
                state_from_feedback = bsln_n[t - 1] + prev_residual * rescale_f
                if tf_mask_per_step is not None:
                    m_t = tf_mask_per_step[t]
                    cs_t = m_t * cs[t] + (1.0 - m_t) * state_from_feedback
                else:
                    cs_t = state_from_feedback
            else:
                cs_t = cs[t]

            x_t = jnp.concatenate([cs_t, r_prev], axis=-1)         # [B, P, 2F]
            h_mesh = self._encode_to_mesh(x_t)                      # [B, M, H]
            if self._processor is not None:
                h_mesh = self._processor(h_mesh)
            h_mesh = h_mesh.reshape(B * M, H)

            new_h_states: list[jax.Array] = []
            y_t = h_mesh
            for ssm, h_prev in zip(self._ssm_blocks, h_states):
                y_t, h_new = ssm.step(y_t, h_prev, is_training=is_training)
                new_h_states.append(h_new)
            h_states = new_h_states

            y_t = y_t.reshape(B, M, H)
            pred_t = self._decode_from_mesh(y_t)                    # [B, P, F]
            if residual_clip is not None and residual_clip > 0:
                pred_t = jnp.clip(pred_t, -residual_clip, residual_clip)
            preds.append(pred_t)
            prev_residual = pred_t

        preds_tbpf = jnp.stack(preds, axis=0)
        preds_tbpf = preds_tbpf.reshape(T, B, lat, lon, -1)
        return preds_tbpf
