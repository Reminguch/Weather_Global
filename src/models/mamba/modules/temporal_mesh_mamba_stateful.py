"""Stateful Mamba temporal block for mesh-latent sequences.

Unlike temporal_mesh_mamba.py which resets hidden state every sample,
this version preserves SSM hidden state across autoregressive steps
via hk.get_state/hk.set_state. When used with target_steps > 1,
the autoregressive.Predictor's hk.scan automatically threads Haiku
state between steps, giving Mamba true long-term memory.

State is stored per mesh node (shape: n_mesh x hidden_size), shared
across the batch dimension. This ensures state shape is independent
of batch size, avoiding shape mismatches between init and apply.
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TemporalMeshConfig:
    backbone: str = "none"
    location: str = "mesh_post_encoder"
    hidden_size: int = 128
    layers: int = 1
    dropout: float = 0.0
    zero_init_output: bool = False


class _StatefulSSMBlock(hk.Module):
    """Mamba-style selective SSM block that preserves hidden state across calls.

    State shape is (n_mesh, hidden_size) — independent of batch size.
    Input x_btd has shape (batch*n_mesh, time, channels). We process all
    batch elements together but store/load state only for n_mesh nodes
    (using the mean across batch elements for the stored state).
    """

    def __init__(
        self,
        hidden_size: int,
        n_mesh: int,
        dropout: float,
        zero_init_output: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._hidden_size = hidden_size
        self._n_mesh = n_mesh
        self._dropout = dropout
        self._zero_init_output = zero_init_output

    def __call__(self, x_btd: jax.Array, *, is_training: bool, batch_size: int) -> jax.Array:
        """x_btd: (batch*n_mesh, time, channels)."""
        input_dim = x_btd.shape[-1]
        x_dtype = x_btd.dtype
        residual = x_btd

        x_btd = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x_btd)
        x_btd = x_btd.astype(x_dtype)

        projected = hk.Linear(2 * self._hidden_size, name="in_proj")(x_btd).astype(x_dtype)
        u_btd, gate_btd = jnp.split(projected, 2, axis=-1)
        dt_btd = jax.nn.softplus(
            hk.Linear(self._hidden_size, name="dt_proj")(u_btd)).astype(x_dtype)

        a_log = hk.get_parameter(
            "a_log",
            shape=(self._hidden_size,),
            init=hk.initializers.Constant(-1.0),
        ).astype(x_dtype)
        skip = hk.get_parameter(
            "skip",
            shape=(self._hidden_size,),
            init=hk.initializers.Constant(1.0),
        ).astype(x_dtype)

        a = (-jnp.exp(a_log)).astype(x_dtype)[None, :]

        def step_fn(state_bh, inputs):
            u_bh, gate_bh, dt_bh = inputs
            decay = jnp.exp(a * dt_bh)
            next_state = decay * state_bh + (1.0 - decay) * u_bh
            y_bh = next_state * jax.nn.sigmoid(gate_bh) + skip[None, :] * u_bh
            return next_state.astype(x_dtype), y_bh.astype(x_dtype)

        # Load stored state: (n_mesh, hidden_size) — batch-independent
        stored_state = hk.get_state(
            "ssm_state",
            shape=(self._n_mesh, self._hidden_size),
            dtype=x_dtype,
            init=jnp.zeros,
        )
        # Broadcast to (batch*n_mesh, hidden_size) by tiling for each batch element
        init_state = jnp.tile(stored_state, (batch_size, 1))

        final_state, y_tbh = jax.lax.scan(
            step_fn,
            init_state,
            (
                jnp.swapaxes(u_btd, 0, 1),
                jnp.swapaxes(gate_btd, 0, 1),
                jnp.swapaxes(dt_btd, 0, 1),
            ),
        )

        # Average across batch elements and save: (batch*n_mesh, H) -> (n_mesh, H)
        new_stored = final_state.reshape(batch_size, self._n_mesh, self._hidden_size).mean(axis=0)
        hk.set_state("ssm_state", new_stored)

        y_btd = jnp.swapaxes(y_tbh, 0, 1)
        y_btd = hk.Linear(
            input_dim,
            w_init=hk.initializers.Constant(0.0) if self._zero_init_output else None,
            b_init=hk.initializers.Constant(0.0) if self._zero_init_output else None,
            name="out_proj",
        )(y_btd)
        if is_training and self._dropout > 0.0:
            y_btd = hk.dropout(hk.next_rng_key(), self._dropout, y_btd)
        return residual + y_btd


class TemporalMeshBlock(hk.Module):
    """Temporal block over `[time, n_mesh_nodes, batch, channels]` mesh latents.

    Same API as the stateless version in temporal_mesh_mamba.py, but uses
    _StatefulSSMBlock so hidden state persists across autoregressive steps.
    """

    def __init__(self, cfg: TemporalMeshConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, mesh_latent_tnbd: jax.Array, *, is_training: bool) -> jax.Array:
        if mesh_latent_tnbd.ndim != 4:
            raise ValueError(
                "Expected [time, n_mesh_nodes, batch, channels], got "
                f"shape={mesh_latent_tnbd.shape}"
            )

        if self.cfg.location == "mesh_processor_interleaved":
            if self.cfg.backbone == "none":
                return mesh_latent_tnbd
            if self.cfg.backbone != "mamba":
                raise ValueError(f"Unsupported temporal backbone: {self.cfg.backbone}")
            return self._run_sequence(mesh_latent_tnbd, is_training=is_training)

        if self.cfg.backbone == "none":
            return mesh_latent_tnbd[-1]
        if self.cfg.backbone != "mamba":
            raise ValueError(f"Unsupported temporal backbone: {self.cfg.backbone}")

        sequence = self._run_sequence(mesh_latent_tnbd, is_training=is_training)
        return sequence[-1]

    def _run_sequence(self, mesh_latent_tnbd: jax.Array, *, is_training: bool) -> jax.Array:
        time_steps, n_mesh, batch_size, channels = mesh_latent_tnbd.shape
        x_bntd = jnp.transpose(mesh_latent_tnbd, (2, 1, 0, 3))
        x_bntd = x_bntd.reshape(batch_size * n_mesh, time_steps, channels)

        for layer_idx in range(self.cfg.layers):
            x_bntd = _StatefulSSMBlock(
                hidden_size=self.cfg.hidden_size,
                n_mesh=n_mesh,
                dropout=self.cfg.dropout,
                zero_init_output=self.cfg.zero_init_output,
                name=f"mamba_block_{layer_idx}",
            )(x_bntd, is_training=is_training, batch_size=batch_size)

        x_bntd = x_bntd.reshape(batch_size, n_mesh, time_steps, channels)
        return jnp.transpose(x_bntd, (2, 1, 0, 3))
