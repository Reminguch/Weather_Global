"""Original pre-refactor MZResidualMamba (teacher-forced only).

This is the ORIGINAL structure used for the ``mz_r4_m3_i32_seg32_h16_fullnorm``
pilot that produced the 7.84% overall-MAE improvement on 2022 val. All Haiku
sub-modules are created lazily inside the method bodies (as in the pre-refactor
version), so parameter names are flat:

    mz_residual_mamba/input_proj/w
    mz_residual_mamba/mz_mamba_block_0/layer_norm/scale
    ...

This file is kept so that:
  * the existing ``mz_residual_step400.pkl`` checkpoints (produced by this
    exact module) can be re-loaded unchanged,
  * the teacher-forced one-step-ahead assimilated setup --- which is the
    deployment-consistent regime for operational NWP --- remains a frozen
    reference implementation independent of later rollout/scheduled-sampling
    experiments,
  * future comparisons between teacher-forced and autoregressive setups
    always have a fixed, bit-reproducible "v1 teacher" anchor.

Use this module when you want the pre-2026-04-22 behaviour. The newer module
``mz_residual_mamba.py`` supports autoregressive rollout and scheduled
sampling but has different Haiku parameter names (``~/`` segments) and is
therefore not a drop-in replacement for v1 checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class MZResidualConfig:
    input_size: int
    output_size: int
    hidden_size: int = 16
    layers: int = 1
    dropout: float = 0.0
    a_log_init: float = -0.1


class _SelectiveSSMBlock(hk.Module):
    """Small Mamba-style selective state-space block over time."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        a_log_init: float,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._a_log_init = a_log_init

    def __call__(self, x_btd: jax.Array, *, is_training: bool) -> jax.Array:
        input_dim = x_btd.shape[-1]
        x_dtype = x_btd.dtype
        residual = x_btd
        x_btd = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x_btd)
        x_btd = x_btd.astype(x_dtype)
        projected = hk.Linear(2 * self._hidden_size, name="in_proj")(x_btd).astype(x_dtype)
        u_btd, gate_btd = jnp.split(projected, 2, axis=-1)
        dt_btd = jax.nn.softplus(hk.Linear(self._hidden_size, name="dt_proj")(u_btd)).astype(x_dtype)

        a_log = hk.get_parameter(
            "a_log",
            shape=(self._hidden_size,),
            init=hk.initializers.Constant(self._a_log_init),
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

        init_state = jnp.zeros((x_btd.shape[0], self._hidden_size), dtype=x_dtype)
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
        y_btd = hk.Linear(input_dim, name="out_proj")(y_btd).astype(x_dtype)
        if is_training and self._dropout > 0.0:
            y_btd = hk.dropout(hk.next_rng_key(), self._dropout, y_btd)
        return residual + y_btd


class MZResidualMamba(hk.Module):
    """Shared temporal model over grid-pointwise resolved-variable histories.

    Inputs are shaped ``[time, batch, lat, lon, features]``. The same temporal
    block is applied independently to each grid point.
    """

    def __init__(self, cfg: MZResidualConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(self, seq_tblnf: jax.Array, *, is_training: bool) -> jax.Array:
        if seq_tblnf.ndim != 5:
            raise ValueError(
                "Expected [time, batch, lat, lon, features], got "
                f"shape={seq_tblnf.shape}"
            )
        time_steps, batch_size, lat, lon, feat_dim = seq_tblnf.shape
        if feat_dim != self.cfg.input_size:
            raise ValueError(
                f"Expected input feature dim {self.cfg.input_size}, got {feat_dim}"
            )

        x_bptd = jnp.transpose(seq_tblnf, (1, 2, 3, 0, 4))
        x_bptd = x_bptd.reshape(batch_size * lat * lon, time_steps, feat_dim)

        x_bptd = hk.Linear(self.cfg.hidden_size, name="input_proj")(x_bptd)
        for layer_idx in range(self.cfg.layers):
            x_bptd = _SelectiveSSMBlock(
                hidden_size=self.cfg.hidden_size,
                dropout=self.cfg.dropout,
                a_log_init=self.cfg.a_log_init,
                name=f"mz_mamba_block_{layer_idx}",
            )(x_bptd, is_training=is_training)

        out_bptd = hk.Linear(
            self.cfg.output_size,
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="residual_head",
        )(x_bptd)

        out_bptd = out_bptd.reshape(batch_size, lat, lon, time_steps, self.cfg.output_size)
        return jnp.transpose(out_bptd, (3, 0, 1, 2, 4))


def shift_residual_history(residual_tblnf: jax.Array) -> jax.Array:
    """Teacher-forcing shift: previous residuals with zero first step."""
    if residual_tblnf.ndim != 5:
        raise ValueError(
            "Expected [time, batch, lat, lon, features], got "
            f"shape={residual_tblnf.shape}"
        )
    zeros0 = jnp.zeros_like(residual_tblnf[:1])
    return jnp.concatenate([zeros0, residual_tblnf[:-1]], axis=0)
