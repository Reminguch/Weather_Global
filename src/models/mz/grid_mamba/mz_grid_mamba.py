"""MZ-lite residual memory model over selected resolved variables.

This module treats a contiguous time segment as a single sample. A frozen
baseline provides the Markov prediction; this module learns the residual memory
correction on top of selected resolved variables.

Two forward modes share identical parameters:
  * ``__call__`` (parallel, teacher-forced): the full time axis is scanned in
    one pass with ``prev_residual`` input at step t taken from the
    ground-truth residual at step t-1.
  * ``rollout_ar`` (autoregressive / scheduled-sampling): ``prev_residual`` is
    the model's own output from step t-1, optionally mixed with the ground
    truth via a Bernoulli teacher-forcing probability.

Parameter sharing between the two modes is achieved by pre-instantiating every
Haiku sub-module in ``__init__`` and reusing the same module instances across
all time steps. This avoids Haiku's automatic name suffixing that would
otherwise create ``input_proj_1``, ``input_proj_2``... when a Linear is
instantiated inside a for-loop.
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
    """Small Mamba-style selective state-space block over time.

    Sub-layers are instantiated once here in ``__init__`` and reused by both
    ``__call__`` (parallel scan) and ``step`` (one-step autoregressive) so the
    two modes share the same parameter tensors.
    """

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
        self._ln = hk.LayerNorm(
            axis=-1, create_scale=True, create_offset=True, name="layer_norm"
        )
        self._in_proj = hk.Linear(2 * hidden_size, name="in_proj")
        self._dt_proj = hk.Linear(hidden_size, name="dt_proj")
        # SSM block is used after `input_proj` has already mapped the raw input
        # to `hidden_size`, so the SSM's out_proj is H -> H.
        self._out_proj = hk.Linear(hidden_size, name="out_proj")

    def _get_a_skip(self, x_dtype):
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
        return a_log, skip

    def _pre_step(self, x_bd_or_btd, x_dtype):
        """Shared prefix for __call__ and step: LN + in_proj + dt_proj."""
        x = self._ln(x_bd_or_btd).astype(x_dtype)
        projected = self._in_proj(x).astype(x_dtype)
        u, gate = jnp.split(projected, 2, axis=-1)
        dt = jax.nn.softplus(self._dt_proj(u)).astype(x_dtype)
        return u, gate, dt

    def __call__(self, x_btd: jax.Array, *, is_training: bool) -> jax.Array:
        x_dtype = x_btd.dtype
        residual = x_btd
        u_btd, gate_btd, dt_btd = self._pre_step(x_btd, x_dtype)

        a_log, skip = self._get_a_skip(x_dtype)
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
        y_btd = self._out_proj(y_btd).astype(x_dtype)
        if is_training and self._dropout > 0.0:
            y_btd = hk.dropout(hk.next_rng_key(), self._dropout, y_btd)
        return residual + y_btd

    def step(
        self,
        x_bd: jax.Array,
        h_prev_bh: jax.Array,
        *,
        is_training: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """One-step version for autoregressive rollout.

        Returns (y_bd, h_new_bh) where y_bd already includes the input residual.
        Uses the same Haiku sub-modules as ``__call__`` for parameter sharing.
        """
        x_dtype = x_bd.dtype
        residual = x_bd
        u_bh, gate_bh, dt_bh = self._pre_step(x_bd, x_dtype)

        a_log, skip = self._get_a_skip(x_dtype)
        a = (-jnp.exp(a_log)).astype(x_dtype)[None, :]

        decay = jnp.exp(a * dt_bh)
        h_new = decay * h_prev_bh + (1.0 - decay) * u_bh
        y_bh = h_new * jax.nn.sigmoid(gate_bh) + skip[None, :] * u_bh
        y_bh = self._out_proj(y_bh).astype(x_dtype)
        if is_training and self._dropout > 0.0:
            y_bh = hk.dropout(hk.next_rng_key(), self._dropout, y_bh)
        return (residual + y_bh).astype(x_dtype), h_new.astype(x_dtype)


class MZResidualMamba(hk.Module):
    """Shared temporal model over grid-pointwise resolved-variable histories.

    Inputs are shaped ``[time, batch, lat, lon, features]``.
    The same temporal block is applied independently to each grid point.

    Sub-modules (input_proj, SSM blocks, residual_head) are instantiated in
    ``__init__`` so both ``__call__`` and ``rollout_ar`` reuse the same
    parameter tensors.
    """

    def __init__(self, cfg: MZResidualConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg
        self._input_proj = hk.Linear(cfg.hidden_size, name="input_proj")
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

    # ----- Parallel (teacher-forced) mode -------------------------------------
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

        x_bptd = self._input_proj(x_bptd)
        for ssm in self._ssm_blocks:
            x_bptd = ssm(x_bptd, is_training=is_training)

        out_bptd = self._residual_head(x_bptd)
        out_bptd = out_bptd.reshape(batch_size, lat, lon, time_steps, self.cfg.output_size)
        return jnp.transpose(out_bptd, (3, 0, 1, 2, 4))

    # ----- Autoregressive mode ------------------------------------------------
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
        """Autoregressive rollout over T time steps.

        Two mutually exclusive ways to control teacher forcing:
          * ``teacher_forcing_prob`` (scalar, default) — per-step Bernoulli:
            mask ~ Bernoulli(tf_prob). Used for scheduled sampling.
          * ``tf_mask_per_step`` (shape ``[T]``, deterministic) — the mask
            value at each rollout step is hard-coded. Overrides the scalar
            Bernoulli path. Typical usage: for target_steps=K rollout
            the caller passes mask = [1, 0, 0, ..., 0] repeated (1 at the first
            step of each K-step sub-rollout, 0 elsewhere), so inter-sample
            transitions are teacher-forced (observable at deployment) while
            intra-sample steps self-feed (honest rollout error).

        At step t, ``prev_residual`` is chosen as:
          * t == 0                       : zeros,
          * t > 0, mask_t = 1            : true residual r^*_{t-1} / diffs_stddev,
          * t > 0, mask_t = 0            : model's own prediction r_hat_{t-1}.

        Option 2 (corrected-state feedback, requires both
        ``baseline_absolute_n_tblnf`` and ``residual_to_state_rescale_f``):
        at intra-sample steps (mask_t = 0), override the ``current_state`` input
        for step t with the self-corrected state from the previous step:

            current_state_n[t] = baseline_absolute_n[t-1]
                                 + pred_residual_n[t-1] * residual_to_state_rescale

        where ``baseline_absolute_n = (baseline_raw - input_mean) / input_std``
        and ``residual_to_state_rescale = output_denorm / input_std`` are both
        computed in normalized space by the caller. With Option 2, MZ's own
        correction from step t-1 propagates into its input at step t, matching
        how a deployed system would feed its best forecast into the next cycle.

        Shared-parameter guarantee: this method uses the same ``input_proj``,
        ``mz_mamba_block_*`` and ``residual_head`` Haiku module instances as
        ``__call__``, so params learned in one mode are usable by the other.
        """
        if current_state_n_tblnf.ndim != 5:
            raise ValueError(
                "Expected [time, batch, lat, lon, features], got "
                f"shape={current_state_n_tblnf.shape}"
            )
        T, B, lat, lon, F = current_state_n_tblnf.shape
        if F != self.cfg.output_size:
            raise ValueError(
                f"rollout_ar expects current_state_n feature dim == output_size, "
                f"got {F} vs output_size={self.cfg.output_size}"
            )
        expected_input_dim = 2 * F
        if expected_input_dim != self.cfg.input_size:
            raise ValueError(
                "rollout_ar assumes the network was built with "
                f"input_size = 2 * output_size. Got input_size={self.cfg.input_size}, "
                f"output_size={self.cfg.output_size}."
            )
        P = B * lat * lon
        H = self.cfg.hidden_size
        dtype = current_state_n_tblnf.dtype

        cs = jnp.transpose(current_state_n_tblnf, (1, 2, 3, 0, 4))  # [B, lat, lon, T, F]
        cs = cs.reshape(P, T, F)
        cs = jnp.swapaxes(cs, 0, 1)  # [T, B*P, F]

        if true_prev_residual_n_tblnf is not None:
            tpr = jnp.transpose(true_prev_residual_n_tblnf, (1, 2, 3, 0, 4))
            tpr = tpr.reshape(P, T, F)
            tpr = jnp.swapaxes(tpr, 0, 1)
        else:
            tpr = None

        prev_residual = jnp.zeros((P, F), dtype=dtype)
        h_states: list[jax.Array] = [
            jnp.zeros((P, H), dtype=dtype) for _ in range(self.cfg.layers)
        ]

        # Validate / coerce the deterministic per-step mask if provided.
        if tf_mask_per_step is not None:
            if tf_mask_per_step.shape != (T,):
                raise ValueError(
                    f"tf_mask_per_step must have shape (T={T},), got "
                    f"{tf_mask_per_step.shape}"
                )
            tf_mask_per_step = tf_mask_per_step.astype(dtype)

        # Option-2 feedback bookkeeping: normalized baseline trajectory and
        # residual->state rescale factor. Both must be provided together.
        use_state_feedback = (
            baseline_absolute_n_tblnf is not None
            and residual_to_state_rescale_f is not None
        )
        if use_state_feedback:
            bsln_n = jnp.transpose(baseline_absolute_n_tblnf, (1, 2, 3, 0, 4))
            bsln_n = bsln_n.reshape(P, T, F)
            bsln_n = jnp.swapaxes(bsln_n, 0, 1)  # [T, B*P, F]
            rescale_f = residual_to_state_rescale_f.astype(dtype)[None, :]  # [1, F]
        else:
            bsln_n = None
            rescale_f = None

        preds: list[jax.Array] = []
        for t in range(T):
            if t == 0:
                # First step: no "previous residual" exists; matches
                # shift_residual_history which zero-pads index 0.
                r_prev = prev_residual
            elif tpr is None:
                # Pure autoregressive path (eval / deployment).
                r_prev = prev_residual
            elif tf_mask_per_step is not None:
                # Deterministic per-step mask (used for target_steps>1 rollout).
                # mask = 1 -> teacher, 0 -> self-feed. Shared across grid points.
                m_t = tf_mask_per_step[t]
                r_prev = m_t * tpr[t] + (1.0 - m_t) * prev_residual
            else:
                # Scheduled-sampling Bernoulli path (scalar tf_prob).
                # tf_prob = 0 -> mask all-zero -> prev_residual (pure AR);
                # tf_prob = 1 -> mask all-one -> tpr[t] (pure teacher);
                # intermediate -> mixed. Written without Python comparisons on
                # teacher_forcing_prob so it stays jit-friendly when tf_prob is
                # a traced jnp array.
                key = hk.next_rng_key()
                mask = (jax.random.uniform(key, shape=(P, 1)) < teacher_forcing_prob).astype(dtype)
                r_prev = mask * tpr[t] + (1.0 - mask) * prev_residual

            # Select the per-step anchor state input:
            #   * use precomputed current_state_n at anchor-step-0 positions
            #     (mask_t = 1) or whenever Option-2 feedback is disabled;
            #   * with Option-2 active and mask_t = 0 (intra-sample rollout),
            #     use the previous step's corrected state in normalized units.
            if use_state_feedback and t > 0:
                # state_from_feedback_n = baseline_absolute_n[t-1]
                #   + prev_pred_residual_n * (output_denorm / input_std)
                state_from_feedback = (
                    bsln_n[t - 1] + prev_residual * rescale_f
                )
                if tf_mask_per_step is not None:
                    m_t = tf_mask_per_step[t]
                    cs_t = m_t * cs[t] + (1.0 - m_t) * state_from_feedback
                else:
                    cs_t = state_from_feedback
            else:
                cs_t = cs[t]

            x_t = jnp.concatenate([cs_t, r_prev], axis=-1)
            x_t = self._input_proj(x_t)
            new_h_states: list[jax.Array] = []
            for ssm, h_prev in zip(self._ssm_blocks, h_states):
                x_t, h_new = ssm.step(x_t, h_prev, is_training=is_training)
                new_h_states.append(h_new)
            h_states = new_h_states

            pred_t = self._residual_head(x_t)
            if residual_clip is not None and residual_clip > 0:
                pred_t = jnp.clip(pred_t, -residual_clip, residual_clip)
            preds.append(pred_t)
            prev_residual = pred_t

        preds_tbpd = jnp.stack(preds, axis=0)  # [T, B*P, F_out]
        preds_tbpd = preds_tbpd.reshape(T, B, lat, lon, -1)
        return preds_tbpd


def shift_residual_history(residual_tblnf: jax.Array) -> jax.Array:
    """Teacher-forcing shift: previous residuals with zero first step."""
    if residual_tblnf.ndim != 5:
        raise ValueError(
            "Expected [time, batch, lat, lon, features], got "
            f"shape={residual_tblnf.shape}"
        )
    zeros0 = jnp.zeros_like(residual_tblnf[:1])
    return jnp.concatenate([zeros0, residual_tblnf[:-1]], axis=0)
