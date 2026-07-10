from __future__ import annotations

import jax
import jax.numpy as jnp
import xarray as xr
from graphcast import xarray_jax


RESIDUAL_AR_FEEDBACK_BASELINE = "baseline"
RESIDUAL_AR_FEEDBACK_BASELINE_PLUS_RESIDUAL = "baseline_plus_residual"
# NEW: gated closed-loop. baseline + lambda * residual, lambda scheduled per step.
RESIDUAL_AR_FEEDBACK_GATED = "gated"
RESIDUAL_AR_FEEDBACK = RESIDUAL_AR_FEEDBACK_BASELINE
RESIDUAL_AR_FEEDBACK_CHOICES = (
    RESIDUAL_AR_FEEDBACK_BASELINE,
    RESIDUAL_AR_FEEDBACK_BASELINE_PLUS_RESIDUAL,
    RESIDUAL_AR_FEEDBACK_GATED,
)


def validate_residual_ar_feedback(mode: str) -> str:
    if mode not in RESIDUAL_AR_FEEDBACK_CHOICES:
        raise ValueError(
            f"Unknown residual AR feedback mode {mode!r}; "
            f"expected one of {', '.join(RESIDUAL_AR_FEEDBACK_CHOICES)}."
        )
    return mode


def _unwrap(da: xr.DataArray) -> jnp.ndarray:
    """Get the raw JAX array out of a xarray_jax DataArray."""
    return xarray_jax.unwrap_data(da)


def _wrap_like(arr: jnp.ndarray, like: xr.DataArray) -> xr.DataArray:
    """Wrap a raw JAX array back into a xarray_jax DataArray matching `like`'s dims/coords.

    Uses the xarray_jax.DataArray constructor (not raw xr.DataArray) — that
    constructor calls `wrap()` on the data and avoids triggering xarray's
    `__array__` introspection during construction (which would call
    np.asarray on a tracer and explode under jit).
    """
    return xarray_jax.DataArray(
        arr,
        dims=like.dims,
        coords={d: like.coords[d] for d in like.dims if d in like.coords},
        name=like.name,
    )


def _scale_dataset(ds: xr.Dataset, scalar) -> xr.Dataset:
    """Scale every data_var in xr.Dataset by a (jax) scalar (jit-safe)."""
    new = {}
    for name, da in ds.data_vars.items():
        scaled = _unwrap(da) * scalar
        new[name] = _wrap_like(scaled, da)
    return xr.Dataset(new, coords=ds.coords, attrs=ds.attrs)


def _add_datasets(a: xr.Dataset, b: xr.Dataset) -> xr.Dataset:
    """Add corresponding data_vars between two xr.Datasets (jit-safe, raw jax)."""
    new = {}
    for name in a.data_vars:
        if name in b.data_vars:
            summed = _unwrap(a[name]) + _unwrap(b[name])
            new[name] = _wrap_like(summed, a[name])
        else:
            new[name] = a[name]
    return xr.Dataset(new, coords=a.coords, attrs=a.attrs)


def _stop_grad_dataset(ds: xr.Dataset) -> xr.Dataset:
    new = {}
    for name, da in ds.data_vars.items():
        stopped = jax.lax.stop_gradient(_unwrap(da))
        new[name] = _wrap_like(stopped, da)
    return xr.Dataset(new, coords=ds.coords, attrs=ds.attrs)


def residual_physical_feedback(
    *,
    baseline_pred: xr.Dataset,
    full_pred: xr.Dataset,
    mode: str,
    lam: float | jnp.ndarray | None = None,
    stop_grad: bool = False,
) -> xr.Dataset:
    """Compute the field that becomes next step's input.

    mode='baseline': open-loop, next_input = baseline (residual NOT fed back)
    mode='baseline_plus_residual': closed-loop, next_input = baseline + residual
    mode='gated': closed-loop with rampable lambda,
                  next_input = baseline + lam * residual
                  (lam=0 → open-loop, lam=1 → baseline_plus_residual)

    stop_grad=True: jax.lax.stop_gradient on the returned dataset.
        Forward feedback stays closed-loop; backward gradient does not
        propagate through the feedback path. Stabilises BPTT.
    """
    validate_residual_ar_feedback(mode)
    if mode == RESIDUAL_AR_FEEDBACK_BASELINE:
        out = baseline_pred
    elif mode == RESIDUAL_AR_FEEDBACK_BASELINE_PLUS_RESIDUAL:
        out = full_pred
    elif mode == RESIDUAL_AR_FEEDBACK_GATED:
        if lam is None:
            raise ValueError("mode='gated' requires lam (the feedback strength).")
        residual = _add_datasets(full_pred,
                                 _scale_dataset(baseline_pred, -1.0))  # residual = full - baseline
        gated_residual = _scale_dataset(residual, lam)
        out = _add_datasets(baseline_pred, gated_residual)
    else:
        raise ValueError(mode)
    if stop_grad:
        out = _stop_grad_dataset(out)
    return out
