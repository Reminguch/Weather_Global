"""Exact grid MSE and the nonnegative spectral AMSE adjustment.

Both objectives score the same physical full fields with GraphCast's original
difference-standard-deviation, pressure and variable weights. The second
objective subtracts spectral MSE, never grid MSE, and is evaluated directly in
the spectral loss to avoid cancellation. No truth-power division or separate
objective normalization is introduced.
"""
from __future__ import annotations

import jax.numpy as jnp


def build_mse_amse_excess_objectives(full_field_loss):
    """Return raw weighted (exact GC MSE, AMSE minus spectral MSE)."""
    def objectives(prediction, truth):
        amse, diagnostics = full_field_loss(prediction, truth)
        gc = diagnostics["grid_mse_fp32"]
        excess = diagnostics["amse_excess"]
        return jnp.stack((gc, excess)), {
            "gc_loss": gc,
            "amse_excess": excess,
            "spectral_mse": diagnostics["spectral_mse"],
            "amse_loss": amse,
        }
    return objectives
