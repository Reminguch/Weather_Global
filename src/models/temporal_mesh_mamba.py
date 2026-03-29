"""Minimal JAX/Haiku temporal block for mesh-latent sequences.

This is a lightweight Mamba-style state space block used to validate the
time-preserving GraphCast path before any deeper processor changes.
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


class _SelectiveSSMBlock(hk.Module):
    """Small Mamba-style selective state space block over a time axis."""

    def __init__(self, hidden_size: int, dropout: float, name: str | None = None):
        super().__init__(name=name)
        self._hidden_size = hidden_size
        self._dropout = dropout

    def __call__(self, x_btd: jax.Array, *, is_training: bool) -> jax.Array:
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

        init_state = jnp.zeros((x_btd.shape[0], self._hidden_size), dtype=x_btd.dtype)
        _, y_tbh = jax.lax.scan(
            step_fn,
            init_state,
            (
                jnp.swapaxes(u_btd, 0, 1),
                jnp.swapaxes(gate_btd, 0, 1),
                jnp.swapaxes(dt_btd, 0, 1),
            ),
        )
        y_btd = jnp.swapaxes(y_tbh, 0, 1)
        y_btd = hk.Linear(input_dim, name="out_proj")(y_btd)
        if is_training and self._dropout > 0.0:
            y_btd = hk.dropout(hk.next_rng_key(), self._dropout, y_btd)
        return residual + y_btd


class TemporalMeshBlock(hk.Module):
    """Temporal block over `[time, n_mesh_nodes, batch, channels]` mesh latents."""

    def __init__(self, cfg: TemporalMeshConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, mesh_latent_tnbd: jax.Array, *, is_training: bool) -> jax.Array:
        if mesh_latent_tnbd.ndim == 3:
            return mesh_latent_tnbd
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
        return jnp.transpose(sequence[-1], (1, 0, 2))

    def _run_sequence(self, mesh_latent_tnbd: jax.Array, *, is_training: bool) -> jax.Array:
        time_steps, n_mesh, batch_size, channels = mesh_latent_tnbd.shape
        x_bntd = jnp.transpose(mesh_latent_tnbd, (2, 1, 0, 3))
        x_bntd = x_bntd.reshape(batch_size * n_mesh, time_steps, channels)

        for layer_idx in range(self.cfg.layers):
            x_bntd = _SelectiveSSMBlock(
                hidden_size=self.cfg.hidden_size,
                dropout=self.cfg.dropout,
                name=f"mamba_block_{layer_idx}",
            )(x_bntd, is_training=is_training)

        x_bntd = x_bntd.reshape(batch_size, n_mesh, time_steps, channels)
        return jnp.transpose(x_bntd, (2, 1, 0, 3))
