"""Gated extreme residual boost head for ResMamba.

Architecture change on top of GCResidualWithZeroHead:

    Original:  r = base_head(latent)
               pred = bp + r
    New:       r_base = base_head(latent)
               r_ext  = ext_head(latent)
               g_ext  = sigmoid(gate_head(latent))
               r = r_base + g_ext * r_ext

Initial-equivalent guarantee:
  base_head  : zero-init weights (or overlaid from existing K=18 ckpt)
  ext_head   : zero-init weights → r_ext ≡ 0 at start
  gate_head  : w=0, b=GATE_BIAS_INIT (default -3.5) → sigmoid ≈ 0.029 at start
So at step 0:  r ≈ r_base  (existing-ckpt behavior preserved within ~3%).

Loss (computed inside the predictor's `loss_and_predictions`):

    L = L_mse                            # standard weighted_mse_per_level
        + β · BCE(gate, extreme_mask)    # per-cell extreme supervision
        + λ_g · E[gate]                  # sparsity prior

where extreme_mask = 1{|y - bp| > q_α(|y - bp|)} per variable. Note `targets`
inside this loss is the NORMALIZED residual (DirectResidualNormalizer pre-norm).
α-weighted MSE (Lₐ = (1 + α·a) · MSE) is left for v2 — BCE + sparsity provide
the gate supervision needed for v1.

Gate is published per-variable via hk.set_state so the loss can read the
correctly-aligned [..., lat, lon, level?] gate slice for each variable.
"""
from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import xarray as xr

from graphcast import graphcast as gc
from graphcast import losses
from graphcast import xarray_jax


def _xr_data(da):
    """Raw JAX array out of an xarray DataArray (unwraps JaxArrayWrapper)."""
    return xarray_jax.unwrap_data(da)


class GCResidualWithGatedExtremeHead(gc.GraphCast):
    """gc.GraphCast subclass: base residual head + gated extreme residual head."""

    def __init__(self, model_cfg, task_cfg, *,
                 gate_bias_init: float = -3.5,
                 beta: float = 0.05,
                 lambda_g: float = 0.003,
                 extreme_quantile: float = 0.95,
                 alpha_extreme: float = 0.0):
        super().__init__(model_cfg, task_cfg)
        self._gate_bias_init = gate_bias_init
        self._beta = beta
        self._lambda_g = lambda_g
        self._extreme_quantile = extreme_quantile
        # NEW (phase 1.5): alpha-weighted MSE term on extreme cells.
        # L_extreme = alpha * E[a · (pred - target)^2]  added to global MSE.
        # This gives ext_head a value-learning signal (not just gate-position
        # supervision from BCE).
        self._alpha_extreme = alpha_extreme
        # Cache of variable order for gate_state key building.
        self._gate_state_var_order: list[str] = []

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
        # Base head — name matches GCResidualWithZeroHead so existing K=18
        # checkpoint's residual_head weights overlay cleanly.
        base_head = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_residual_head",
        )
        ext_head = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_residual_ext_head",
        )
        gate_head = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(self._gate_bias_init),
            name="temporal_residual_gate_head",
        )

        r_base = base_head(output_grid_nodes)
        r_ext  = ext_head(output_grid_nodes)
        gate_logits = gate_head(output_grid_nodes)
        gate = jax.nn.sigmoid(gate_logits)
        residual_output = r_base + gate * r_ext

        # To publish gate ALIGNED with each target variable's spatial layout,
        # we route `gate` through the same _grid_node_outputs_to_prediction
        # converter that builds `predictions`. Gate is unitless [0,1] so it
        # doesn't need normalization treatment.
        gate_ds = self._grid_node_outputs_to_prediction(
            gate.astype(jnp.float32), targets_template)
        # Persist per-variable gate as state for the loss.
        self._gate_state_var_order = list(gate_ds.data_vars)
        for var_name in gate_ds.data_vars:
            hk.set_state(f"gate__{var_name}",
                         _xr_data(gate_ds[var_name]).astype(jnp.float32))

        return self._grid_node_outputs_to_prediction(residual_output, targets_template)

    def loss_and_predictions(self, inputs, targets, forcings):
        """Override gc.GraphCast.loss_and_predictions to add gate aux losses."""
        predictions = self(
            inputs, targets_template=targets, forcings=forcings, is_training=True)

        # Standard weighted-MSE on residual.
        base_loss, base_scalars = losses.weighted_mse_per_level(
            predictions, targets,
            per_variable_weights={
                "2m_temperature": 1.0,
                "10m_u_component_of_wind": 0.1,
                "10m_v_component_of_wind": 0.1,
                "mean_sea_level_pressure": 0.1,
                "total_precipitation_6hr": 0.1,
            })

        # === Auxiliary gate losses ===
        # extreme label  a = 1{|target| > q_α(|target|)}  per variable
        # BCE(gate, a)  averaged across (cells, levels, batch, vars)
        # sparsity      = E[gate]
        # NEW: alpha-weighted MSE on extreme cells = alpha · E[a · (pred-target)²]
        bce_per_var = []
        sparsity_per_var = []
        extreme_rate_per_var = []
        gate_mean_extreme_per_var = []
        ext_mse_per_var = []
        gate_non_extreme_per_var = []
        for var_name in targets.data_vars:
            t = _xr_data(targets[var_name])
            abs_t = jnp.abs(t)
            # Global per-variable q95 over (time, level?, lat, lon, batch).
            q_a = jnp.quantile(abs_t.reshape(-1), self._extreme_quantile)
            a = (abs_t > q_a).astype(jnp.float32)

            gate_key = f"gate__{var_name}"
            try:
                g_var = hk.get_state(gate_key)
            except (LookupError, ValueError, KeyError):
                continue

            g_var = g_var.astype(jnp.float32)
            g_clip = jnp.clip(g_var, 1e-7, 1.0 - 1e-7)
            bce = -(a * jnp.log(g_clip) + (1.0 - a) * jnp.log(1.0 - g_clip))
            bce_per_var.append(jnp.mean(bce))
            sparsity_per_var.append(jnp.mean(g_var))
            extreme_rate_per_var.append(jnp.mean(a))
            denom_e = jnp.maximum(jnp.sum(a),       1.0)
            denom_n = jnp.maximum(jnp.sum(1.0 - a), 1.0)
            gate_mean_extreme_per_var.append(jnp.sum(g_var * a) / denom_e)
            gate_non_extreme_per_var.append(jnp.sum(g_var * (1.0 - a)) / denom_n)

            # NEW: extreme-weighted MSE — gives ext_head a value-learning target.
            # Need access to predictions for this variable.
            pred_var = _xr_data(predictions[var_name]).astype(jnp.float32)
            target_var = t.astype(jnp.float32)
            sq_err = (pred_var - target_var) ** 2
            ext_mse_per_var.append(jnp.sum(a * sq_err) / denom_e)

        if bce_per_var:
            bce_loss      = jnp.mean(jnp.stack(bce_per_var))
            sparsity_loss = jnp.mean(jnp.stack(sparsity_per_var))
            gate_mean_extreme = jnp.mean(jnp.stack(gate_mean_extreme_per_var))
            gate_mean_non_extreme = jnp.mean(jnp.stack(gate_non_extreme_per_var))
            extreme_rate_obs = jnp.mean(jnp.stack(extreme_rate_per_var))
            ext_mse_loss = jnp.mean(jnp.stack(ext_mse_per_var))
        else:
            zero = jnp.array(0.0, dtype=jnp.float32)
            bce_loss = sparsity_loss = gate_mean_extreme = zero
            gate_mean_non_extreme = extreme_rate_obs = ext_mse_loss = zero

        # base_loss is an xarray DataArray (data wrapped via JaxArrayWrapper).
        # Wrap aux scalars the same way so xarray arithmetic stays consistent.
        base_dtype = xarray_jax.unwrap_data(base_loss).dtype
        bce_loss_c      = bce_loss.astype(base_dtype)
        sparsity_loss_c = sparsity_loss.astype(base_dtype)
        ext_mse_loss_c  = ext_mse_loss.astype(base_dtype)
        bce_da      = xarray_jax.DataArray(data=bce_loss_c,      dims=())
        sparsity_da = xarray_jax.DataArray(data=sparsity_loss_c, dims=())
        ext_mse_da  = xarray_jax.DataArray(data=ext_mse_loss_c,  dims=())
        total_loss = (base_loss
                      + self._beta * bce_da
                      + self._lambda_g * sparsity_da
                      + self._alpha_extreme * ext_mse_da)

        aux_scalars = dict(base_scalars)
        ones = jnp.array(1.0)
        aux_scalars["aux_bce"]                   = (bce_loss, ones)
        aux_scalars["aux_sparsity"]              = (sparsity_loss, ones)
        aux_scalars["aux_gate_mean_extreme"]     = (gate_mean_extreme, ones)
        aux_scalars["aux_gate_mean_non_extreme"] = (gate_mean_non_extreme, ones)
        aux_scalars["aux_gate_contrast"]         = (gate_mean_extreme - gate_mean_non_extreme, ones)
        aux_scalars["aux_extreme_rate"]          = (extreme_rate_obs, ones)
        aux_scalars["aux_ext_mse"]               = (ext_mse_loss, ones)
        return (total_loss, aux_scalars), predictions

    def loss(self, inputs, targets, forcings):
        (loss_val, scalars), _ = self.loss_and_predictions(inputs, targets, forcings)
        return loss_val, scalars
