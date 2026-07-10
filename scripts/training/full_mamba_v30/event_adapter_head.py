"""v30 Contrastive Event Adapter Head — SKELETON.

Architecture (on top of v22 frozen residual Mamba):

    output_grid_nodes      [num_grid_nodes, batch, channels]   # from mesh2grid
    r_base   = base_head(h)                                    # v22, overlaid
    u        = GELU(W1 h)             trunk     [N, B, trunk_dim]
    z_pred   = norm(W_z u)            embedding [N, B, proj_dim]
    g        = sigmoid(W_g u + b_g)   gate      [N, B, out_size]
    delta    = cap * tanh(W_d u)      correction[N, B, out_size]
    d        = sigmoid(W_dmp u + b_dmp)  damp   [N, B, out_size]    NEW (Step 1.1)
    adapter  = g * delta - d * r_base                              NEW (Step 1.1)
    r_final  = r_base + adapter_scale * adapter

`adapter_scale=0` uses an EXPLICIT branch (Step 1.2) so r_final is byte-identical
to v22 (no 0*NaN / dtype promotion / XLA folding hazards). New heads are still
instantiated (params land in the tree) but their compute doesn't reach r_final.

Damp head (Step 1.1) lets the adapter REDUCE r_base, not just add to it. At
init bias=-4, damp≈0.018 → near-no-op; trainable so the model can learn to
partially undo v22 residual where it hurts (e.g. record-extreme regions).

Naming so v22 K=18 overlay still works:
    base_head        : name="temporal_residual_head"          (overlaid from v22)
    event_trunk      : name="temporal_event_trunk"            (NEW)
    event_proj       : name="temporal_event_proj"             (NEW)
    event_gate       : name="temporal_event_gate"             (NEW)
    event_delta      : name="temporal_event_delta"            (NEW)
    event_damp       : name="temporal_event_damp"             (NEW)
    target_encoder.* : name="temporal_event_target_encoder/*" (NEW, training-only)

`target_encoder` lives INSIDE this predictor's loss_and_predictions, so its
params are in the same Haiku tree (no separate hk.transform). If
`force_init_target_encoder=True` (Step 1.3), the encoder is invoked even when
lambda_con=0 (with 0-weighted contribution to total) so its params land in the
tree at init time — this lets you smoke-test with lambda_con=0 and then resume
with lambda_con>0 without optimizer-mask shape mismatches.

State published via hk.set_state (for loss + diagnostics):
    z_pred_grid          : [num_grid_nodes, batch, proj_dim]
    gate_grid            : [num_grid_nodes, batch, out_size]
    delta_grid           : [num_grid_nodes, batch, out_size]
    damp_grid            : [num_grid_nodes, batch, out_size]
    effective_adapter    : [num_grid_nodes, batch, out_size]    # scale*(g*d - damp*r_base)
    r_base_grid          : [num_grid_nodes, batch, out_size]    # for damp regularizer

Loss (skeleton): currently just weighted-MSE on r_final, identical to v22.
InfoNCE plumbing is laid out but lambda_con=0 default → no-op (unless
force_init_target_encoder=True, in which case 0-weighted forward keeps params).
"""
from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from graphcast import graphcast as gc
from graphcast import losses
from graphcast import xarray_jax


def _xr(da):
    return xarray_jax.unwrap_data(da)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class TargetEncoder(hk.Module):
    """Tiny CNN/MLP over a truth tube patch -> normalized embedding.

    Input  : [batch, T_win, P, P, C_vars]
    Output : [batch, proj_dim]  (L2-normalized)

    First-version is a flatten+MLP. A small Conv2D over (P,P) followed by GAP +
    MLP is the obvious upgrade once the skeleton is verified.
    """

    def __init__(self, proj_dim: int = 64, hidden: int = 128,
                 name: str = "temporal_event_target_encoder"):
        super().__init__(name=name)
        self._proj_dim = proj_dim
        self._hidden = hidden

    def __call__(self, truth_tube):
        # truth_tube: [B, T, P, P, C]
        b = truth_tube.shape[0]
        x = truth_tube.reshape(b, -1)
        x = hk.Linear(self._hidden, name="fc1")(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(self._proj_dim, name="fc2")(x)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class GCResidualWithEventAdapter(gc.GraphCast):
    """Residual Mamba + tiny contrastive-trainable event adapter head."""

    def __init__(self, model_cfg, task_cfg, *,
                 adapter_scale: float = 0.0,     # set to 1.0 once verified
                 trunk_dim: int = 64,
                 proj_dim: int = 64,
                 gate_bias_init: float = -2.0,
                 damp_bias_init: float = -4.0,   # NEW (Step 1.1): damp ≈ 0.018 at init
                 delta_cap: float = 0.5,
                 lambda_con: float = 0.0,        # InfoNCE weight (0 = off)
                 lambda_delta_reg: float = 0.0,  # |effective_adapter|^2 regularizer
                 tau_contrastive: float = 0.1,
                 force_init_target_encoder: bool = False,   # NEW (Step 1.3)
                 event_patch_size: int = 16):    # NEW (Step 1.5): Python static int
        super().__init__(model_cfg, task_cfg)
        self._adapter_scale = adapter_scale
        self._trunk_dim = trunk_dim
        self._proj_dim = proj_dim
        self._gate_bias_init = gate_bias_init
        self._damp_bias_init = damp_bias_init
        self._delta_cap = delta_cap
        self._lambda_con = lambda_con
        self._lambda_delta_reg = lambda_delta_reg
        self._tau_con = tau_contrastive
        self._force_init_target_encoder = force_init_target_encoder
        # patch_size MUST be a static Python int (used as jnp.arange bound). Don't
        # take this from event_batch dict — that would become a traced value inside
        # jit and break gather_event_patch.
        self._event_patch_size = int(event_patch_size)

    def __call__(self, inputs, targets_template, forcings, is_training=False):
        self._maybe_init(inputs)
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(
            grid_node_features)
        if (self._temporal_backbone != "none"
                and self._temporal_location == "mesh_post_encoder"):
            latent_mesh_nodes = self._run_temporal_mesh_block(
                latent_mesh_nodes, is_training=is_training)
        updated_latent_mesh_nodes = self._run_mesh_gnn(
            latent_mesh_nodes, is_training=is_training)
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes)

        out_size = output_grid_nodes.shape[-1]

        # --- Base v22 residual head (overlay-compatible name) ---
        base_head = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_residual_head",
        )
        r_base = base_head(output_grid_nodes)

        # --- Event adapter trunk + heads ---
        u = hk.Linear(self._trunk_dim, name="temporal_event_trunk")(output_grid_nodes)
        u = jax.nn.gelu(u)

        z_pred = hk.Linear(self._proj_dim, name="temporal_event_proj")(u)
        z_pred = z_pred / (jnp.linalg.norm(z_pred, axis=-1, keepdims=True) + 1e-6)

        gate = jax.nn.sigmoid(hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(self._gate_bias_init),
            name="temporal_event_gate",
        )(u))

        delta_raw = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_event_delta",
        )(u)
        delta = self._delta_cap * jnp.tanh(delta_raw)

        # NEW (Step 1.1): damping head — lets adapter REDUCE r_base in cells
        # where v22 residual is hurting. At init bias=-4, damp ≈ 0.018 (≈ no-op).
        damp = jax.nn.sigmoid(hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(self._damp_bias_init),
            name="temporal_event_damp",
        )(u))

        adapter = gate * delta - damp * r_base   # [N, B, out_size]

        # Step 1.2: explicit branch for scale==0 → byte-identical v22.
        if self._adapter_scale == 0.0:
            r_final = r_base
            effective_adapter = jnp.zeros_like(r_base)
        else:
            effective_adapter = self._adapter_scale * adapter
            r_final = r_base + effective_adapter

        # State for loss + diagnostics. Grid-node ordering preserved
        # (lat_idx * num_lon + lon_idx). gather_event_patch reshapes as needed.
        hk.set_state("z_pred_grid",       z_pred.astype(jnp.float32))
        hk.set_state("gate_grid",         gate.astype(jnp.float32))
        hk.set_state("delta_grid",        delta.astype(jnp.float32))
        hk.set_state("damp_grid",         damp.astype(jnp.float32))
        hk.set_state("r_base_grid",       r_base.astype(jnp.float32))
        hk.set_state("effective_adapter", effective_adapter.astype(jnp.float32))

        return self._grid_node_outputs_to_prediction(r_final, targets_template)

    # ---------- loss ----------

    def loss_and_predictions(self, inputs, targets, forcings,
                             event_batch=None):
        """Loss = weighted MSE  [+ lambda_con * InfoNCE]  [+ lambda_delta * |g*delta|^2].

        Args:
            event_batch: optional dict of JAX arrays:
                'truth_tube_pos':  [B, T_win, P, P, C_vars]   positive truth tubes
                'truth_tube_neg':  [B, Nneg, T_win, P, P, C]  in-batch negatives
                                   (currently unused — defaults to in-batch other-i diag)
                'center_lat_idx':  [B]   patch center lat index
                'center_lon_idx':  [B]   patch center lon index
                NOTE: patch_size is a STATIC int on the predictor
                (`self._event_patch_size`), NOT a key in event_batch. Don't put
                Python ints into the pytree — they'd get traced inside jit.
            If event_batch is None OR self._lambda_con == 0, contrastive is skipped.
        """
        predictions = self(
            inputs, targets_template=targets, forcings=forcings, is_training=True)

        # Standard v22 weighted-MSE on r_final.
        base_loss, base_scalars = losses.weighted_mse_per_level(
            predictions, targets,
            per_variable_weights={
                "2m_temperature": 1.0,
                "10m_u_component_of_wind": 0.1,
                "10m_v_component_of_wind": 0.1,
                "mean_sea_level_pressure": 0.1,
                "total_precipitation_6hr": 0.1,
            })

        base_dtype = xarray_jax.unwrap_data(base_loss).dtype
        # IMPORTANT: cast `ones` to base_dtype. Bfloat16Cast.tree_map_cast walks
        # the aux pytree and replaces any leaf whose dtype != bf16 with None.
        # If `ones` is fp32 it gets None'd → the whole (value, ones) tuple ends
        # up with None elements. Same applies to the value side, so always cast
        # both via `_aux_scalar` below.
        ones = jnp.array(1.0, dtype=base_dtype)
        def _aux_scalar(v):
            # Force every aux value to base_dtype so tree_map_cast handles it.
            return jnp.asarray(v).astype(base_dtype)
        aux = dict(base_scalars)

        total = base_loss

        # ---- InfoNCE path (Step 1.3: force-init also runs encoder with 0 weight) ----
        # Encoder is invoked when:
        #   (a) event_batch is provided AND lambda_con > 0 → real contrastive loss
        #   (b) event_batch is provided AND force_init_target_encoder → 0-weight
        #       forward, so target_encoder params land in tree at init time
        # Otherwise the encoder is skipped and inference / smoke runs don't touch
        # its params.
        run_target_encoder = (event_batch is not None) and (
            self._lambda_con > 0.0 or self._force_init_target_encoder)
        if run_target_encoder:
            z_pred_grid = hk.get_state("z_pred_grid")   # [N, B, proj_dim]
            z_pred_patch = gather_event_patch(
                z_pred_grid,
                num_lat=self._grid_lat.shape[0],
                num_lon=self._grid_lon.shape[0],
                center_lat_idx=event_batch["center_lat_idx"],
                center_lon_idx=event_batch["center_lon_idx"],
                patch_size=self._event_patch_size,   # STATIC int on predictor
            )   # [B, proj_dim]

            target_encoder = TargetEncoder(proj_dim=self._proj_dim)
            z_pos = target_encoder(event_batch["truth_tube_pos"])   # [B, proj_dim]

            # Explicit-neg path (C0): only used when lambda_con > 0 AND
            # truth_tube_neg is provided. Otherwise this is the force-init or
            # pos-only debug path and InfoNCE is replaced by a 0-weighted stub.
            has_negs = (event_batch.get("truth_tube_neg") is not None)

            if self._lambda_con > 0.0 and has_negs:
                t_neg = event_batch["truth_tube_neg"]   # [B, Nneg, T, P, P, C]
                B_, Nneg = t_neg.shape[0], t_neg.shape[1]
                # Encode all negatives in one batched forward.
                t_neg_flat = t_neg.reshape((B_ * Nneg,) + t_neg.shape[2:])
                z_neg = target_encoder(t_neg_flat).reshape(B_, Nneg, self._proj_dim)

                L_con = info_nce_loss(z_pred_patch, z_pos, z_neg, tau=self._tau_con)
                L_con_da = xarray_jax.DataArray(
                    data=L_con.astype(base_dtype), dims=())
                total = total + self._lambda_con * L_con_da
                aux["aux_L_con"] = (_aux_scalar(L_con), ones)

                # Retrieval diagnostics (explicit pos/neg).
                pos_per = jnp.sum(z_pred_patch * z_pos, axis=-1)     # [B]
                neg_all = jnp.einsum("bd,bnd->bn", z_pred_patch, z_neg)  # [B, Nneg]
                pos_sim = jnp.mean(pos_per)
                neg_sim_mean = jnp.mean(neg_all)
                neg_sim_max = jnp.mean(jnp.max(neg_all, axis=-1))
                aux["aux_con_pos_sim"]     = (_aux_scalar(pos_sim), ones)
                aux["aux_con_neg_sim"]     = (_aux_scalar(neg_sim_mean), ones)
                aux["aux_con_neg_sim_max"] = (_aux_scalar(neg_sim_max), ones)
                aux["aux_con_gap"]         = (_aux_scalar(pos_sim - neg_sim_mean), ones)
                # Retrieval acc: 1 if pos-sim > all neg-sims, else 0.
                top1 = jnp.mean(
                    (pos_per > jnp.max(neg_all, axis=-1)).astype(jnp.float32))
                aux["aux_con_top1"] = (_aux_scalar(top1), ones)
                # Embedding health: std across the D dimensions of a
                # (L2-normalized) vector. Random Gaussian norm-to-unit in D dims
                # gives ~1/sqrt(D)=0.125 for D=64. A collapsed embedding (all
                # dims equal magnitude) has std → 0.
                aux["aux_z_pred_std"]  = (
                    _aux_scalar(jnp.mean(jnp.std(z_pred_patch, axis=-1))), ones)
                aux["aux_z_truth_std"] = (
                    _aux_scalar(jnp.mean(jnp.std(z_pos,        axis=-1))), ones)
            else:
                # Force-init / pos-only path: add 0*sum(z_pos) so target_encoder
                # grads are traced and params get init'd. No effect on loss.
                stub = (0.0 * jnp.sum(z_pos)).astype(base_dtype)
                stub_da = xarray_jax.DataArray(data=stub, dims=())
                total = total + stub_da
                aux["aux_L_con"] = (_aux_scalar(0.0), ones)
                pos_sim = jnp.mean(jnp.sum(z_pred_patch * z_pos, axis=-1))
                aux["aux_con_pos_sim"] = (_aux_scalar(pos_sim), ones)

        # ---- Effective-adapter amplitude regularizer (Step 1 / point 7) ----
        # Penalizes the actual contribution to r_final, not the pre-scale product.
        if self._lambda_delta_reg > 0.0:
            eff = hk.get_state("effective_adapter")
            L_delta = jnp.mean(eff ** 2)
            L_delta_da = xarray_jax.DataArray(
                data=L_delta.astype(base_dtype), dims=())
            total = total + self._lambda_delta_reg * L_delta_da
            aux["aux_L_delta"] = (_aux_scalar(L_delta), ones)

        # ---- Diagnostics (always logged) ----
        try:
            gate_grid = hk.get_state("gate_grid")
            delta_grid = hk.get_state("delta_grid")
            damp_grid  = hk.get_state("damp_grid")
            eff_grid   = hk.get_state("effective_adapter")
            aux["aux_gate_mean"]  = (_aux_scalar(jnp.mean(gate_grid)), ones)
            aux["aux_damp_mean"]  = (_aux_scalar(jnp.mean(damp_grid)), ones)
            aux["aux_delta_rms"]  = (_aux_scalar(jnp.sqrt(jnp.mean(delta_grid ** 2))), ones)
            aux["aux_adapter_rms"] = (
                _aux_scalar(jnp.sqrt(jnp.mean((gate_grid * delta_grid) ** 2))), ones)
            aux["aux_effective_adapter_rms"] = (
                _aux_scalar(jnp.sqrt(jnp.mean(eff_grid ** 2))), ones)
        except (LookupError, ValueError, KeyError):
            pass

        return (total, aux), predictions

    def loss(self, inputs, targets, forcings, event_batch=None):
        (loss_val, scalars), _ = self.loss_and_predictions(
            inputs, targets, forcings, event_batch=event_batch)
        return loss_val, scalars


# ---------------------------------------------------------------------------
# Patch gather + InfoNCE helpers
# ---------------------------------------------------------------------------

def gather_event_patch(z_grid, num_lat: int, num_lon: int,
                       center_lat_idx, center_lon_idx, patch_size: int):
    """Mean-pool a [P, P] patch from a grid-node-flat embedding map.

    Args:
        z_grid: [num_grid_nodes, batch, proj_dim]. Grid nodes are flattened as
                lat_idx * num_lon + lon_idx (matches
                GraphCast._grid_node_outputs_to_prediction).
        num_lat / num_lon: grid dimensions.
        center_lat_idx / center_lon_idx: [batch] index arrays.
        patch_size: int, side length of the patch (e.g. 16).

    Returns:
        [batch, proj_dim] — patch-mean-pooled embedding (L2-renormalized).

    Notes:
        - Lon is wrapped circularly. Lat is clipped to [0, num_lat-1].
        - dynamic_slice keeps this jit-friendly per-sample.
        - First-version uses jax.vmap; patch size is static.
    """
    # z_grid -> [batch, num_lat, num_lon, proj_dim]
    proj_dim = z_grid.shape[-1]
    batch = z_grid.shape[1]
    z_blf = jnp.transpose(z_grid, (1, 0, 2))             # [B, N, D]
    z_bll = z_blf.reshape(batch, num_lat, num_lon, proj_dim)

    half = patch_size // 2

    def one(z_ll, lat_c, lon_c):
        # Clip lat; wrap lon via gather with modular indices.
        lat_start = jnp.clip(lat_c - half, 0, num_lat - patch_size)
        lon_inds = (lon_c - half + jnp.arange(patch_size)) % num_lon
        lat_inds = lat_start + jnp.arange(patch_size)
        patch = z_ll[lat_inds[:, None], lon_inds[None, :]]   # [P, P, D]
        pooled = jnp.mean(patch, axis=(0, 1))                # [D]
        return pooled

    z_patch = jax.vmap(one)(z_bll, center_lat_idx, center_lon_idx)
    z_patch = z_patch / (jnp.linalg.norm(z_patch, axis=-1, keepdims=True) + 1e-6)
    return z_patch


def info_nce_loss(z_pred, z_pos, z_neg, tau: float = 0.1):
    """Asymmetric InfoNCE with EXPLICIT negatives (pred → pos vs neg).

    Args:
        z_pred:  [B, D]         L2-normalized forecast-latent patch embedding
        z_pos:   [B, D]         L2-normalized positive truth-tube embedding
        z_neg:   [B, Nneg, D]   L2-normalized explicit-negative truth-tube embeddings
        tau:     temperature

    Returns:
        scalar loss = mean over batch of  -log( exp(s_pos/tau) / sum_k exp(s_k/tau) ).

    Why explicit negatives (not in-batch / symmetric CLIP):
        - GraphCast train batches are tiny (B=1 in BPTT). In-batch negatives give
          no contrastive signal at B=1.
        - Explicit negatives let the sampler control climate-matching (same
          month / lat band, different time/lon) without depending on batch size.
        - Asymmetric (pred → truth) is simpler than symmetric CLIP; the reverse
          direction (truth → pred) adds little when negatives are already
          carefully matched.
    """
    # logit_pos: [B, 1]
    logit_pos = jnp.sum(z_pred * z_pos, axis=-1, keepdims=True) / tau
    # logit_neg: [B, Nneg]
    logit_neg = jnp.einsum("bd,bnd->bn", z_pred, z_neg) / tau
    logits = jnp.concatenate([logit_pos, logit_neg], axis=1)   # [B, 1+Nneg]
    # Positive is always at index 0 → CE = -log_softmax[:, 0]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -log_probs[..., 0]
    return jnp.mean(ce)
