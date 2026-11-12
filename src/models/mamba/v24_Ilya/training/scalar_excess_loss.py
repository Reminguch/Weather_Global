"""Exact grid GraphCast MSE plus a separately weighted AMSE excess.

The inner AMSE coefficient is fixed at one. The outer coefficient is applied
once to its stable excess diagnostic. At outer zero, no spherical transform is
constructed or evaluated; missing spectral diagnostics are genuinely uncomputed.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from .spectral_loss import build_full_field_loss


def build_gc_mse_amse_excess_loss(stats, *, spectral_excess_weight: float,
                                lmax: int | None = None, power_epsilon: float = 1e-12):
    weight = float(spectral_excess_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("spectral_excess_weight must be finite and in [0, 1]")
    raw_loss = build_full_field_loss(
        stats, "amse" if weight else "mse_fp32", lmax=lmax,
        power_epsilon=power_epsilon, amse_lambda=1.0,
    )

    def loss(predictions, targets):
        _, diagnostics = raw_loss(predictions, targets)
        grid = diagnostics["grid_mse_fp32"]
        weighted_excess = (jnp.asarray(weight, jnp.float32) * diagnostics["amse_excess"]
                           if weight else jnp.zeros_like(grid))
        total = grid + weighted_excess
        diagnostics = {**diagnostics, "weighted_amse_excess": weighted_excess,
                       "scalar_total": total}
        return total, diagnostics

    return loss


def build_native_mse_excess_penalty(stats, *, spectral_excess_weight: float,
                                   lmax: int | None = None, power_epsilon: float = 1e-12):
    """Return only the spectral addition; the native predictor supplies grid MSE.

    In particular, the FP32 grid diagnostic must never replace the historical
    BF16 native loss. The returned penalty uses the physical forecast produced
    by that same native loss-and-predictions call.
    """
    weight = float(spectral_excess_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("spectral_excess_weight must be finite and in [0, 1]")
    raw_loss = (build_full_field_loss(stats, "amse", lmax=lmax,
                                    power_epsilon=power_epsilon, amse_lambda=1.)
                if weight else None)

    def penalty(predictions, targets):
        if raw_loss is None:
            zero = jnp.zeros((targets.sizes["batch"],), dtype=jnp.float32)
            return zero, {"amse_excess": zero, "weighted_amse_excess": zero}
        _, diagnostics = raw_loss(predictions, targets)
        weighted = jnp.asarray(weight, jnp.float32) * diagnostics["amse_excess"]
        return weighted, {**diagnostics, "weighted_amse_excess": weighted}

    return penalty
