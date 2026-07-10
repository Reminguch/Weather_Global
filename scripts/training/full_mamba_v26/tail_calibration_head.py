"""v26 Physical Tail Calibration Head.

Architecture:
    y_hat = y_GC + r_base + Delta_tail

where:
    r_base    : original v22 residual Mamba head (overlaid from K=18 ckpt)
    Delta_tail: NEW bounded tail calibration, ONLY non-zero in physical tail region

Tail correction is decomposed as:
    Delta_var = cap_var · m_phys_var · tanh(a_tail_var)
where:
    a_tail_var : raw output of tail_head (free, bounded by tanh)
    m_phys_var : physical mask in [0,1], gated on input state's anomaly z-score
    cap_var    : per-variable amplitude cap (normalized space)

For v1 we calibrate ONLY:
    2m_temperature       : high/low tail masks (m_high, m_low from z = norm(input_T))
    10m_u/v wind         : wind-speed mask m_wind from sqrt(u^2 + v^2) at last input slot
                           (For v1: applied to u/v separately; direction-preserving radial
                            wind calibration is a v2 refinement.)

Loss:
    L = L_global  +  alpha_tail · L_tail  +  lambda_quiet · L_quiet
    L_global = weighted_mse_per_level(r_final, residual_targets)   # standard v22 loss
    L_tail   = E[ m_truth_tail · (Delta_tail - sg(residual_target - r_base))² ]
    L_quiet  = E[ (1 - m_phys_var) · Delta_tail² ]

Inputs to model are already z-score normalized via DirectResidualNormalizer wrapper.
So "anomaly" = input value itself; z_95 ≈ 1.64 used as tail threshold.

NOTE: this is a v1 implementation; physical climatology (per doy/lat/lon p95) is
approximated by global p95 = 1.64 in normalized space.
"""
from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import xarray as xr

from graphcast import graphcast as gc
from graphcast import losses
from graphcast import xarray_jax


def _xr(da):
    return xarray_jax.unwrap_data(da)


class GCResidualWithTailCalibrationHead(gc.GraphCast):
    """Residual Mamba + bounded physical tail calibration head."""

    def __init__(self, model_cfg, task_cfg, *,
                 alpha_tail: float = 1.0,
                 lambda_quiet: float = 0.01,
                 cap_T: float = 0.5,         # normalized cap for 2m_T
                 cap_wind: float = 0.5,       # normalized cap for wind u/v
                 z95: float = 1.64,           # normalized z-score for p95
                 tau: float = 0.5,            # sigmoid steepness for mask
                 extreme_quantile: float = 0.95):
        super().__init__(model_cfg, task_cfg)
        self._alpha_tail = alpha_tail
        self._lambda_quiet = lambda_quiet
        self._cap_T = cap_T
        self._cap_wind = cap_wind
        self._z95 = z95
        self._tau = tau
        self._extreme_quantile = extreme_quantile

    def __call__(self, inputs, targets_template, forcings, is_training=False):
        self._maybe_init(inputs)
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(grid_node_features)
        if (self._temporal_backbone != "none"
                and self._temporal_location == "mesh_post_encoder"):
            latent_mesh_nodes = self._run_temporal_mesh_block(
                latent_mesh_nodes, is_training=is_training)
        updated_latent_mesh_nodes = self._run_mesh_gnn(
            latent_mesh_nodes, is_training=is_training)
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes)

        out_size = output_grid_nodes.shape[-1]
        # Base residual head — name matches GCResidualWithZeroHead so K=18 overlay works.
        base_head = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_residual_head",
        )
        # Tail calibration head — outputs full residual field; physical decoder masks it.
        tail_head = hk.Linear(
            out_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_tail_calibration_head",
        )

        r_base = base_head(output_grid_nodes)
        r_tail_raw = tail_head(output_grid_nodes)

        # Convert both to xarray Datasets (variables/levels separated).
        r_base_ds = self._grid_node_outputs_to_prediction(r_base, targets_template)
        r_tail_raw_ds = self._grid_node_outputs_to_prediction(
            r_tail_raw.astype(jnp.float32), targets_template)

        # Compute physical masks from inputs (last time slot), all in normalized space.
        # inputs are pre-normalized via DirectResidualNormalizer wrapper, so values are
        # roughly N(0,1) anomalies relative to global climatology.
        delta_vars = {}
        # Track state for loss computation.
        for vname in r_base_ds.data_vars:
            r_base_var = _xr(r_base_ds[vname]).astype(jnp.float32)
            r_tail_var = _xr(r_tail_raw_ds[vname]).astype(jnp.float32)

            if vname == "2m_temperature":
                z_T = _xr(inputs["2m_temperature"].isel(time=-1)).astype(jnp.float32)
                # Expand to match (batch, time=1, lat, lon) shape of r_base
                # inputs.isel(time=-1) has shape (batch, lat, lon); r_base has time dim.
                z_T_b = z_T[:, None, ...]   # (batch, 1, lat, lon)
                m_high = jax.nn.sigmoid((z_T_b - self._z95) / self._tau)
                m_low  = jax.nn.sigmoid((-z_T_b - self._z95) / self._tau)
                # Tail-direction handling: when high, only allow positive tanh; vice versa.
                # Simpler: combined mask in [0, 2] then divide
                mask_T = jnp.maximum(m_high, m_low)   # in [0,1]
                # signed correction: extend along the active tail direction
                signed_tanh = m_high * jnp.tanh(r_tail_var) - m_low * jnp.tanh(r_tail_var)
                # Above isn't right; simpler:
                #   high region: correction sign should be free (extreme could be undershot or overshot)
                # Use just `mask_T * cap * tanh(r_tail)` allowing either sign.
                delta_T = self._cap_T * mask_T * jnp.tanh(r_tail_var)
                delta_vars[vname] = delta_T
                hk.set_state(f"mask__{vname}", mask_T)
                hk.set_state(f"Delta__{vname}", delta_T)
                hk.set_state(f"rbase__{vname}", r_base_var)

            elif vname in ("10m_u_component_of_wind", "10m_v_component_of_wind"):
                # For wind: use the L2 wind-speed mask computed from BOTH u/v of last input.
                z_u = _xr(inputs["10m_u_component_of_wind"].isel(time=-1)).astype(jnp.float32)
                z_v = _xr(inputs["10m_v_component_of_wind"].isel(time=-1)).astype(jnp.float32)
                s_norm = jnp.sqrt(z_u * z_u + z_v * z_v)
                s_norm_b = s_norm[:, None, ...]
                m_wind = jax.nn.sigmoid((s_norm_b - self._z95) / self._tau)
                delta_w = self._cap_wind * m_wind * jnp.tanh(r_tail_var)
                delta_vars[vname] = delta_w
                hk.set_state(f"mask__{vname}", m_wind)
                hk.set_state(f"Delta__{vname}", delta_w)
                hk.set_state(f"rbase__{vname}", r_base_var)

            else:
                # No tail correction for other variables. Delta = 0.
                pass

        # Compose r_final = r_base + Delta
        r_final_data_vars = {}
        for vname in r_base_ds.data_vars:
            r_base_var = r_base_ds[vname]
            if vname in delta_vars:
                base_arr = _xr(r_base_var).astype(jnp.float32)
                final_arr = base_arr + delta_vars[vname]
                r_final_data_vars[vname] = (
                    r_base_var.dims, xarray_jax.wrap(final_arr.astype(_xr(r_base_var).dtype)))
            else:
                r_final_data_vars[vname] = (r_base_var.dims, _xr(r_base_var))
        r_final_ds = xarray_jax.Dataset(data_vars=r_final_data_vars)
        # Align coords/attrs to r_base_ds.
        r_final_ds = r_final_ds.assign_coords(r_base_ds.coords)
        return r_final_ds

    def loss_and_predictions(self, inputs, targets, forcings):
        predictions = self(
            inputs, targets_template=targets, forcings=forcings, is_training=True)
        # Standard weighted-MSE on r_final.
        base_loss, base_scalars = losses.weighted_mse_per_level(
            predictions, targets,
            per_variable_weights={
                "2m_temperature": 1.0,
                "10m_u_component_of_wind": 0.1,
                "10m_v_component_of_wind": 0.1,
                "mean_sea_level_pressure": 0.1,
                "total_precipitation_6hr": 0.1,
            })

        # Tail losses per calibrated variable.
        tail_terms = []
        quiet_terms = []
        for vname in ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"):
            if vname not in predictions.data_vars: continue
            try:
                delta = hk.get_state(f"Delta__{vname}")
                mask  = hk.get_state(f"mask__{vname}")
                r_base_var = hk.get_state(f"rbase__{vname}")
            except (LookupError, ValueError, KeyError):
                continue
            target_var = _xr(targets[vname]).astype(jnp.float32)

            # truth tail mask: cells where |target residual| > q95 (per var)
            abs_t = jnp.abs(target_var)
            q = jnp.quantile(abs_t.reshape(-1), self._extreme_quantile)
            m_truth = (abs_t > q).astype(jnp.float32)

            # target_remain = (target - r_base) = what Delta_tail should ideally be.
            target_remain = target_var - r_base_var
            tail_err_sq = (delta - jax.lax.stop_gradient(target_remain)) ** 2

            denom = jnp.maximum(jnp.sum(m_truth), 1.0)
            L_tail_v = jnp.sum(m_truth * tail_err_sq) / denom
            tail_terms.append(L_tail_v)

            # Quiet: in non-tail (mask near 0) regions, delta should be near 0.
            L_quiet_v = jnp.mean((1.0 - mask) * delta * delta)
            quiet_terms.append(L_quiet_v)

        if tail_terms:
            L_tail  = jnp.mean(jnp.stack(tail_terms))
            L_quiet = jnp.mean(jnp.stack(quiet_terms))
        else:
            zero = jnp.array(0.0, dtype=jnp.float32)
            L_tail = L_quiet = zero

        base_dtype = xarray_jax.unwrap_data(base_loss).dtype
        L_tail_da  = xarray_jax.DataArray(data=L_tail.astype(base_dtype),  dims=())
        L_quiet_da = xarray_jax.DataArray(data=L_quiet.astype(base_dtype), dims=())
        total_loss = (base_loss
                      + self._alpha_tail   * L_tail_da
                      + self._lambda_quiet * L_quiet_da)

        aux = dict(base_scalars)
        ones = jnp.array(1.0)
        aux["aux_L_tail"]  = (L_tail,  ones)
        aux["aux_L_quiet"] = (L_quiet, ones)
        # Diagnostic: max |Delta| per var
        for vname in ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"):
            try:
                d = hk.get_state(f"Delta__{vname}")
                m = hk.get_state(f"mask__{vname}")
                aux[f"aux_delta_rms__{vname}"] = (jnp.sqrt(jnp.mean(d*d)), ones)
                aux[f"aux_mask_mean__{vname}"] = (jnp.mean(m), ones)
            except (LookupError, ValueError, KeyError):
                pass
        return (total_loss, aux), predictions

    def loss(self, inputs, targets, forcings):
        (loss_val, scalars), _ = self.loss_and_predictions(inputs, targets, forcings)
        return loss_val, scalars
