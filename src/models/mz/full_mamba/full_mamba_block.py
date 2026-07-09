"""Full (S6-style) Mamba block.

Per-channel diagonal SSM with multi-dim state and input-dependent B, C, dt.
Scans along the time axis with jax.lax.scan. Provides both a parallel
teacher-forced __call__ and a one-step step() method so the wrapper can
reuse the same Haiku parameters for autoregressive rollout.

Shapes inside __call__:
  input  x_btd:  [B*P, T, H]            (H = hidden_size = d_model)
  output y_btd:  [B*P, T, H]            (same as input, plus residual connection)
  hidden state:  [B*P, D_inner, N]      (D_inner = H*expand, N = d_state)

Parameter tensors:
  layer_norm/scale, offset                 (H,)
  in_proj/w, b                             (H -> 2 * D_inner)     # splits into x, z
  x_proj/w, b                              (D_inner -> D_inner + 2N)  # [dt_in, B, C]
  dt_proj/w, b                             (D_inner -> D_inner)   # dt from dt_in
  A_log                                    (D_inner, N)           # -exp(A_log) = A
  D                                        (D_inner,)             # skip (scalar per ch)
  out_proj/w, b                            (D_inner -> H)
"""

from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp


class FullMambaBlock(hk.Module):
    """Closer-to-original Mamba SSM block. Tier 1 upgrades.

    Parameters
    ----------
    hidden_size : int
        Input/output feature dim H.
    d_state : int
        State dim per inner channel (N in Mamba notation). With d_state=16 each
        inner channel can represent a linear combination of 16 independent
        exponential-memory modes.
    expand : int
        Width multiplier for the inner representation. D_inner = H * expand.
        Original Mamba uses expand=2.
    dropout : float
        Applied on the output before the final residual add.
    a_log_init_min, a_log_init_max : float
        Uniform init range for log(-A). Different channels get different
        time constants by construction (the "log-uniform" HIPPO-style init).
    """

    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.0,
        a_log_init_min: float = -3.0,
        a_log_init_max: float = -0.1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._H = hidden_size
        self._N = d_state
        self._E = expand
        self._D_inner = hidden_size * expand
        self._dropout = dropout
        self._a_log_init_min = float(a_log_init_min)
        self._a_log_init_max = float(a_log_init_max)

        self._ln = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="layer_norm"
        )
        # in_proj produces [x_path, z_path] each of size D_inner.
        self._in_proj = hk.Linear(2 * self._D_inner, name="in_proj")
        # x_proj produces [dt_input (D_inner), B (N), C (N)] given x_path.
        self._x_proj = hk.Linear(self._D_inner + 2 * self._N, name="x_proj")
        # dt_proj maps dt_input (D_inner) -> softplus -> dt (D_inner).
        self._dt_proj = hk.Linear(self._D_inner, name="dt_proj")
        self._out_proj = hk.Linear(self._H, name="out_proj")

    # ---------------------------------------------------------------- params
    def _get_A_log_and_D(self, dtype):
        A_log = hk.get_parameter(
            "A_log",
            shape=(self._D_inner, self._N),
            dtype=dtype,
            init=hk.initializers.RandomUniform(
                minval=self._a_log_init_min, maxval=self._a_log_init_max
            ),
        ).astype(dtype)
        D = hk.get_parameter(
            "D",
            shape=(self._D_inner,),
            dtype=dtype,
            init=hk.initializers.Constant(1.0),
        ).astype(dtype)
        return A_log, D

    # ---------------------------------------------------------------- helpers
    def _proj_inputs(self, x_btd: jax.Array, dtype) -> tuple[jax.Array, jax.Array]:
        """LN -> in_proj -> split into (x_path, z_path).

        x_btd: [B*P, T, H]
        returns x_path, z_path, each [B*P, T, D_inner]
        """
        x = self._ln(x_btd).astype(dtype)
        xz = self._in_proj(x).astype(dtype)
        x_path, z_path = jnp.split(xz, 2, axis=-1)
        x_path = jax.nn.silu(x_path)
        return x_path, z_path

    def _compute_dt_B_C(self, x_path_btd: jax.Array, dtype):
        """x_proj + dt_proj to get input-dependent dt, B, C.

        x_path_btd: [..., D_inner]   (works for both [B*P, T, D_inner] and [B*P, D_inner])
        Returns:
          dt : [..., D_inner]     (strictly > 0)
          B  : [..., N]
          C  : [..., N]
        """
        proj = self._x_proj(x_path_btd).astype(dtype)
        dt_input, B, C = jnp.split(proj, [self._D_inner, self._D_inner + self._N], axis=-1)
        dt = jax.nn.softplus(self._dt_proj(dt_input)).astype(dtype)
        return dt, B, C

    # -------------------------------------------------- parallel teacher mode
    def __call__(self, x_btd: jax.Array, *, is_training: bool) -> jax.Array:
        """Run one block over a full sequence of length T.

        x_btd : [B*P, T, H]
        returns [B*P, T, H]   (includes the block's outer residual connection)
        """
        dtype = x_btd.dtype
        residual = x_btd

        x_path, z_path = self._proj_inputs(x_btd, dtype)   # both [B*P, T, D_inner]
        dt_btd, B_btd, C_btd = self._compute_dt_B_C(x_path, dtype)

        A_log, D = self._get_A_log_and_D(dtype)
        A = -jnp.exp(A_log)                                # [D_inner, N]

        # Prepare inputs shaped [T, B*P, ...]
        u_tbi = jnp.swapaxes(x_path, 0, 1)                 # [T, B*P, D_inner]
        dt_tbi = jnp.swapaxes(dt_btd, 0, 1)                # [T, B*P, D_inner]
        B_tbn = jnp.swapaxes(B_btd, 0, 1)                  # [T, B*P, N]
        C_tbn = jnp.swapaxes(C_btd, 0, 1)                  # [T, B*P, N]

        B_size = x_btd.shape[0]

        def step_fn(state_bin, inputs):
            u_bi, dt_bi, B_bn, C_bn = inputs
            # Discretize: decay[b, i, n] = exp(dt[b, i] * A[i, n])
            # Need (B, D_inner, N) broadcasting.
            decay = jnp.exp(dt_bi[:, :, None] * A[None, :, :])       # [B, D_inner, N]
            # Input contribution: dt * B * u broadcast
            # dt_bi: [B, D_inner] , B_bn: [B, N] , u_bi: [B, D_inner]
            input_term = (dt_bi[:, :, None]
                          * B_bn[:, None, :]
                          * u_bi[:, :, None])                         # [B, D_inner, N]
            new_state = decay * state_bin + input_term
            # Output: sum over N of state * C_broadcast
            y = jnp.sum(new_state * C_bn[:, None, :], axis=-1)        # [B, D_inner]
            return new_state.astype(dtype), y.astype(dtype)

        init_state = jnp.zeros(
            (B_size, self._D_inner, self._N), dtype=dtype
        )
        _, y_tbi = jax.lax.scan(
            step_fn,
            init_state,
            (u_tbi, dt_tbi, B_tbn, C_tbn),
        )                                                            # [T, B, D_inner]
        y_btd = jnp.swapaxes(y_tbi, 0, 1)                            # [B, T, D_inner]

        # Add skip (D * u)
        y_btd = y_btd + D[None, None, :] * x_path

        # Output gate (SiLU-gated by z_path), then project back to H.
        y_btd = y_btd * jax.nn.silu(z_path)
        y_btd = self._out_proj(y_btd).astype(dtype)
        if is_training and self._dropout > 0.0:
            y_btd = hk.dropout(hk.next_rng_key(), self._dropout, y_btd)
        return residual + y_btd

    # -------------------------------------------------- one-step AR mode
    def step(
        self,
        x_bd: jax.Array,
        h_prev_bin: jax.Array,
        *,
        is_training: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """One autoregressive step.

        x_bd        : [B*P, H]
        h_prev_bin  : [B*P, D_inner, N]
        returns (y_bd, h_new_bin)
          y_bd       : [B*P, H]  (includes outer residual connection)
          h_new_bin  : [B*P, D_inner, N]
        """
        dtype = x_bd.dtype
        residual = x_bd

        # Apply LN + in_proj on a 2D array (no time axis). Add/remove dummy time.
        x_path_1d, z_path_1d = self._proj_inputs(x_bd[:, None, :], dtype)
        x_path = x_path_1d[:, 0, :]                                   # [B, D_inner]
        z_path = z_path_1d[:, 0, :]                                   # [B, D_inner]

        dt_bi, B_bn, C_bn = self._compute_dt_B_C(x_path, dtype)       # dt: [B, D_inner], B/C: [B, N]

        A_log, D = self._get_A_log_and_D(dtype)
        A = -jnp.exp(A_log)                                           # [D_inner, N]
        decay = jnp.exp(dt_bi[:, :, None] * A[None, :, :])            # [B, D_inner, N]
        input_term = (dt_bi[:, :, None]
                      * B_bn[:, None, :]
                      * x_path[:, :, None])                           # [B, D_inner, N]
        h_new = decay * h_prev_bin + input_term
        y_bi = jnp.sum(h_new * C_bn[:, None, :], axis=-1)             # [B, D_inner]

        y_bi = y_bi + D[None, :] * x_path
        y_bi = y_bi * jax.nn.silu(z_path)
        y_bi = self._out_proj(y_bi).astype(dtype)
        if is_training and self._dropout > 0.0:
            y_bi = hk.dropout(hk.next_rng_key(), self._dropout, y_bi)
        return (residual + y_bi).astype(dtype), h_new.astype(dtype)
