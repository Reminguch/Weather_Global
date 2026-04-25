"""Stateful temporal Mamba block with explicit per-sample recurrent state.

This module ports the ordering and core ingredients from
`mamba-minimal/model.py` to JAX/Haiku while supporting chunked autoregressive
execution. Persistent memory is private per `(batch, mesh_node)` and consists
of both:

* the true Mamba SSM hidden state `(d_inner, d_state)`, and
* a causal conv cache of the last `d_conv - 1` projected `x` tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TemporalMeshConfig:
    backbone: str = "none"
    location: str = "mesh_post_encoder"
    hidden_size: int | None = None
    d_inner: int | None = None
    d_state: int = 16
    dt_rank: int | str = "auto"
    d_conv: int = 4
    layers: int = 1
    bias: bool = False
    conv_bias: bool = True
    dropout: float = 0.0
    zero_init_output: bool = False


class TemporalLayerState(NamedTuple):
    ssm_state: jax.Array
    conv_cache: jax.Array


TemporalMeshState = tuple[TemporalLayerState, ...]


def _cfg_value(cfg: object, name: str, default):
    value = getattr(cfg, name, default)
    return default if value is None else value


def _resolve_d_inner(cfg: object) -> int:
    d_inner = getattr(cfg, "d_inner", None)
    if d_inner is not None:
        return int(d_inner)
    hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)
    raise ValueError("Temporal Mamba requires either `d_inner` or `hidden_size`.")


def _resolve_dt_rank(cfg: object, d_model: int) -> int:
    dt_rank = _cfg_value(cfg, "dt_rank", "auto")
    if dt_rank == "auto":
        return math.ceil(d_model / 16)
    return int(dt_rank)


def init_temporal_state(
    cfg: object,
    *,
    batch_size: int,
    n_mesh: int,
    dtype: jnp.dtype,
) -> TemporalMeshState:
    if _cfg_value(cfg, "backbone", "none") == "none":
        return ()
    d_inner = _resolve_d_inner(cfg)
    d_state = int(_cfg_value(cfg, "d_state", 16))
    d_conv = int(_cfg_value(cfg, "d_conv", 4))
    conv_width = max(d_conv - 1, 0)
    return tuple(
        TemporalLayerState(
            ssm_state=jnp.zeros((batch_size, n_mesh, d_inner, d_state), dtype=dtype),
            conv_cache=jnp.zeros((batch_size, n_mesh, d_inner, conv_width), dtype=dtype),
        )
        for _ in range(int(_cfg_value(cfg, "layers", 1)))
    )


def reset_temporal_state(
    state: TemporalMeshState,
    reset_mask: jax.Array | None = None,
) -> TemporalMeshState:
    if reset_mask is None:
        return tuple(
            TemporalLayerState(
                ssm_state=jnp.zeros_like(layer_state.ssm_state),
                conv_cache=jnp.zeros_like(layer_state.conv_cache),
            )
            for layer_state in state
        )

    reset_mask = jnp.asarray(reset_mask, dtype=bool)
    reset_mask_ssm = reset_mask[:, None, None, None]
    reset_mask_conv = reset_mask[:, None, None, None]
    return tuple(
        TemporalLayerState(
            ssm_state=jnp.where(
                reset_mask_ssm, jnp.zeros_like(layer_state.ssm_state), layer_state.ssm_state
            ),
            conv_cache=jnp.where(
                reset_mask_conv, jnp.zeros_like(layer_state.conv_cache), layer_state.conv_cache
            ),
        )
        for layer_state in state
    )


def load_temporal_state_from_haiku(
    prefix: str,
    cfg: object,
    *,
    batch_size: int,
    n_mesh: int,
    dtype: jnp.dtype,
) -> TemporalMeshState:
    d_inner = _resolve_d_inner(cfg)
    d_state = int(_cfg_value(cfg, "d_state", 16))
    d_conv = int(_cfg_value(cfg, "d_conv", 4))
    conv_width = max(d_conv - 1, 0)
    return tuple(
        TemporalLayerState(
            ssm_state=hk.get_state(
                f"{prefix}_layer_{layer_idx}_ssm_state",
                shape=(batch_size, n_mesh, d_inner, d_state),
                dtype=dtype,
                init=jnp.zeros,
            ),
            conv_cache=hk.get_state(
                f"{prefix}_layer_{layer_idx}_conv_cache",
                shape=(batch_size, n_mesh, d_inner, conv_width),
                dtype=dtype,
                init=jnp.zeros,
            ),
        )
        for layer_idx in range(int(_cfg_value(cfg, "layers", 1)))
    )


def store_temporal_state_to_haiku(prefix: str, state: TemporalMeshState) -> None:
    for layer_idx, layer_state in enumerate(state):
        hk.set_state(f"{prefix}_layer_{layer_idx}_ssm_state", layer_state.ssm_state)
        hk.set_state(f"{prefix}_layer_{layer_idx}_conv_cache", layer_state.conv_cache)


class _DepthwiseCausalConv1D(hk.Module):
    """Depthwise causal Conv1D with explicit cache for chunked execution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        with_bias: bool,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._channels = channels
        self._kernel_size = kernel_size
        self._with_bias = with_bias

    def __call__(
        self,
        x_bct: jax.Array,
        prev_cache: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if prev_cache.shape[-1] != max(self._kernel_size - 1, 0):
            raise ValueError(
                "Expected conv cache width "
                f"{max(self._kernel_size - 1, 0)}, got {prev_cache.shape[-1]}."
            )
        x_dtype = x_bct.dtype
        kernel = hk.get_parameter(
            "kernel",
            shape=(self._channels, self._kernel_size),
            dtype=x_dtype,
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        bias = None
        if self._with_bias:
            bias = hk.get_parameter(
                "bias",
                shape=(self._channels,),
                dtype=x_dtype,
                init=jnp.zeros,
            )

        full_sequence = jnp.concatenate([prev_cache, x_bct], axis=-1)
        outputs = []
        for step_idx in range(x_bct.shape[-1]):
            window = jax.lax.dynamic_slice_in_dim(
                full_sequence, start_index=step_idx, slice_size=self._kernel_size, axis=-1
            )
            outputs.append(jnp.sum(window * kernel[None, :, :], axis=-1))
        y_bct = jnp.stack(outputs, axis=-1)
        if bias is not None:
            y_bct = y_bct + bias[None, :, None]
        if self._kernel_size == 1:
            next_cache = full_sequence[:, :, :0]
        else:
            next_cache = full_sequence[:, :, -(self._kernel_size - 1) :]
        return y_bct.astype(x_dtype), next_cache.astype(x_dtype)


class _StatefulSSMBlock(hk.Module):
    """Faithful minimal-Mamba block with explicit SSM state and conv cache."""

    def __init__(self, cfg: object, name: str | None = None):
        super().__init__(name=name)
        self._cfg = cfg

    def __call__(
        self,
        x_btd: jax.Array,
        prev_state: jax.Array,
        prev_conv_cache: jax.Array,
        *,
        is_training: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del is_training
        d_model = x_btd.shape[-1]
        d_inner = _resolve_d_inner(self._cfg)
        x_dtype = x_btd.dtype

        projected = hk.Linear(
            2 * d_inner,
            with_bias=bool(_cfg_value(self._cfg, "bias", False)),
            name="in_proj",
        )(x_btd).astype(x_dtype)
        x_branch_btd, res_branch_btd = jnp.split(projected, 2, axis=-1)

        x_branch_bct = jnp.transpose(x_branch_btd, (0, 2, 1))
        conv_out_bct, next_conv_cache = _DepthwiseCausalConv1D(
            d_inner,
            int(_cfg_value(self._cfg, "d_conv", 4)),
            with_bias=bool(_cfg_value(self._cfg, "conv_bias", True)),
            name="conv1d",
        )(x_branch_bct, prev_conv_cache.astype(x_dtype))
        conv_out_btd = jnp.transpose(conv_out_bct, (0, 2, 1))
        conv_out_btd = jax.nn.silu(conv_out_btd)

        ssm_out_btd, next_state = self.ssm(
            conv_out_btd, prev_state=prev_state, d_model=d_model
        )
        y_btd = ssm_out_btd * jax.nn.silu(res_branch_btd)
        output = hk.Linear(
            d_model,
            with_bias=bool(_cfg_value(self._cfg, "bias", False)),
            w_init=(
                hk.initializers.Constant(0.0)
                if bool(_cfg_value(self._cfg, "zero_init_output", False))
                else None
            ),
            b_init=(
                hk.initializers.Constant(0.0)
                if bool(_cfg_value(self._cfg, "zero_init_output", False))
                else None
            ),
            name="out_proj",
        )(y_btd)
        return output.astype(x_dtype), next_state.astype(x_dtype), next_conv_cache.astype(x_dtype)

    def ssm(
        self,
        x_btd: jax.Array,
        *,
        prev_state: jax.Array,
        d_model: int,
    ) -> tuple[jax.Array, jax.Array]:
        d_inner = _resolve_d_inner(self._cfg)
        d_state = int(_cfg_value(self._cfg, "d_state", 16))
        dt_rank = _resolve_dt_rank(self._cfg, d_model)
        x_dtype = x_btd.dtype

        def _a_log_init(shape: tuple[int, int], dtype: jnp.dtype) -> jax.Array:
            del shape
            base = jnp.arange(1, d_state + 1, dtype=dtype)
            return jnp.log(jnp.tile(base[None, :], (d_inner, 1)))

        a_log = hk.get_parameter(
            "A_log",
            shape=(d_inner, d_state),
            dtype=jnp.float32,
            init=_a_log_init,
        )
        d_skip = hk.get_parameter(
            "D",
            shape=(d_inner,),
            dtype=jnp.float32,
            init=jnp.ones,
        )

        x_dbl = hk.Linear(
            dt_rank + 2 * d_state,
            with_bias=False,
            name="x_proj",
        )(x_btd)
        delta_raw_btr, b_btn, c_btn = jnp.split(x_dbl, [dt_rank, dt_rank + d_state], axis=-1)
        delta_btd = jax.nn.softplus(
            hk.Linear(d_inner, with_bias=True, name="dt_proj")(delta_raw_btr)
        )

        u_btd = x_btd.astype(jnp.float32)
        delta_btd = delta_btd.astype(jnp.float32)
        b_btn = b_btn.astype(jnp.float32)
        c_btn = c_btn.astype(jnp.float32)
        a_dn = -jnp.exp(a_log.astype(jnp.float32))
        d_skip = d_skip.astype(jnp.float32)

        delta_a_btdn = jnp.exp(jnp.einsum("btd,dn->btdn", delta_btd, a_dn))
        delta_b_u_btdn = jnp.einsum("btd,btn,btd->btdn", delta_btd, b_btn, u_btd)

        scan_inputs = (
            jnp.swapaxes(delta_a_btdn, 0, 1),
            jnp.swapaxes(delta_b_u_btdn, 0, 1),
            jnp.swapaxes(c_btn, 0, 1),
        )

        def scan_step(
            state_bdn: jax.Array,
            inputs_t: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            delta_a_bdn, delta_b_u_bdn, c_bn = inputs_t
            next_state = delta_a_bdn * state_bdn + delta_b_u_bdn
            y_bd = jnp.einsum("bdn,bn->bd", next_state, c_bn)
            return next_state, y_bd

        next_state_bdn, y_tbd = jax.lax.scan(
            scan_step,
            prev_state.astype(jnp.float32),
            scan_inputs,
        )
        y_btd = jnp.swapaxes(y_tbd, 0, 1)
        y_btd = y_btd + u_btd * d_skip[None, None, :]
        return y_btd.astype(x_dtype), next_state_bdn.astype(x_dtype)


class TemporalMeshBlock(hk.Module):
    """Temporal block over `[time, n_mesh_nodes, batch, channels]` mesh latents."""

    def __init__(self, cfg: object, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

    def __call__(
        self,
        mesh_latent_tnbd: jax.Array,
        prev_state: TemporalMeshState | None = None,
        *,
        is_training: bool,
        reset_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, TemporalMeshState]:
        if mesh_latent_tnbd.ndim == 3:
            if _cfg_value(self.cfg, "backbone", "none") == "none":
                return mesh_latent_tnbd, prev_state or ()
            mesh_4d = mesh_latent_tnbd[None]
            out_4d, next_state = self._run_sequence(
                mesh_4d,
                prev_state=prev_state,
                is_training=is_training,
                reset_mask=reset_mask,
            )
            return out_4d[0], next_state

        if mesh_latent_tnbd.ndim != 4:
            raise ValueError(
                "Expected [time, n_mesh_nodes, batch, channels], got "
                f"shape={mesh_latent_tnbd.shape}"
            )

        if _cfg_value(self.cfg, "location", "mesh_post_encoder") == "mesh_processor_interleaved":
            if _cfg_value(self.cfg, "backbone", "none") == "none":
                return mesh_latent_tnbd, prev_state or ()
            if _cfg_value(self.cfg, "backbone", "none") != "mamba":
                raise ValueError(f"Unsupported temporal backbone: {_cfg_value(self.cfg, 'backbone', 'none')}")
            return self._run_sequence(
                mesh_latent_tnbd,
                prev_state=prev_state,
                is_training=is_training,
                reset_mask=reset_mask,
            )

        if _cfg_value(self.cfg, "backbone", "none") == "none":
            return mesh_latent_tnbd[-1], prev_state or ()
        if _cfg_value(self.cfg, "backbone", "none") != "mamba":
            raise ValueError(f"Unsupported temporal backbone: {_cfg_value(self.cfg, 'backbone', 'none')}")

        sequence, next_state = self._run_sequence(
            mesh_latent_tnbd,
            prev_state=prev_state,
            is_training=is_training,
            reset_mask=reset_mask,
        )
        return sequence[-1], next_state

    def _run_sequence(
        self,
        mesh_latent_tnbd: jax.Array,
        *,
        prev_state: TemporalMeshState | None,
        is_training: bool,
        reset_mask: jax.Array | None,
    ) -> tuple[jax.Array, TemporalMeshState]:
        time_steps, n_mesh, batch_size, channels = mesh_latent_tnbd.shape
        del time_steps
        d_inner = _resolve_d_inner(self.cfg)
        d_state = int(_cfg_value(self.cfg, "d_state", 16))
        conv_width = max(int(_cfg_value(self.cfg, "d_conv", 4)) - 1, 0)

        if prev_state is None:
            prev_state = init_temporal_state(
                self.cfg,
                batch_size=batch_size,
                n_mesh=n_mesh,
                dtype=mesh_latent_tnbd.dtype,
            )
        if reset_mask is not None:
            prev_state = reset_temporal_state(prev_state, reset_mask)
        if len(prev_state) != int(_cfg_value(self.cfg, "layers", 1)):
            raise ValueError(
                f"Expected {int(_cfg_value(self.cfg, 'layers', 1))} layer states, got {len(prev_state)}."
            )

        x_btd = jnp.transpose(mesh_latent_tnbd, (2, 1, 0, 3))
        x_btd = x_btd.reshape(batch_size * n_mesh, mesh_latent_tnbd.shape[0], channels)

        next_state_layers: list[TemporalLayerState] = []
        for layer_idx, layer_state in enumerate(prev_state):
            if layer_state.ssm_state.shape != (batch_size, n_mesh, d_inner, d_state):
                raise ValueError(
                    "Unexpected SSM state shape for layer "
                    f"{layer_idx}: expected {(batch_size, n_mesh, d_inner, d_state)}, "
                    f"got {layer_state.ssm_state.shape}."
                )
            if layer_state.conv_cache.shape != (batch_size, n_mesh, d_inner, conv_width):
                raise ValueError(
                    "Unexpected conv cache shape for layer "
                    f"{layer_idx}: expected {(batch_size, n_mesh, d_inner, conv_width)}, "
                    f"got {layer_state.conv_cache.shape}."
                )

            residual_btd = x_btd
            x_norm_btd = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True,
                name=f"layer_norm_{layer_idx}",
            )(x_btd).astype(x_btd.dtype)
            y_btd, next_ssm_bdn, next_conv_bdc = _StatefulSSMBlock(
                self.cfg,
                name=f"mamba_block_{layer_idx}",
            )(
                x_norm_btd,
                prev_state=layer_state.ssm_state.reshape(batch_size * n_mesh, d_inner, d_state),
                prev_conv_cache=layer_state.conv_cache.reshape(batch_size * n_mesh, d_inner, conv_width),
                is_training=is_training,
            )
            if is_training and float(_cfg_value(self.cfg, "dropout", 0.0)) > 0.0:
                y_btd = hk.dropout(hk.next_rng_key(), float(_cfg_value(self.cfg, "dropout", 0.0)), y_btd)
            x_btd = residual_btd + y_btd
            next_state_layers.append(
                TemporalLayerState(
                    ssm_state=next_ssm_bdn.reshape(batch_size, n_mesh, d_inner, d_state),
                    conv_cache=next_conv_bdc.reshape(batch_size, n_mesh, d_inner, conv_width),
                )
            )

        x_btd = x_btd.reshape(batch_size, n_mesh, mesh_latent_tnbd.shape[0], channels)
        return jnp.transpose(x_btd, (2, 1, 0, 3)), tuple(next_state_layers)
