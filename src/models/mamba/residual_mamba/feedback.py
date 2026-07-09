from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
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

# Feedback-only clipping (Version B): clip the perturbation that gets fed back
# into the frozen GraphCast input, without touching the output residual.
FEEDBACK_CLIP_MODE_NONE = "none"
FEEDBACK_CLIP_MODE_TANH = "tanh"
FEEDBACK_CLIP_MODE_HARD = "hard"
FEEDBACK_CLIP_MODE_CHOICES = (
    FEEDBACK_CLIP_MODE_NONE,
    FEEDBACK_CLIP_MODE_TANH,
    FEEDBACK_CLIP_MODE_HARD,
)


def validate_residual_ar_feedback(mode: str) -> str:
    if mode not in RESIDUAL_AR_FEEDBACK_CHOICES:
        raise ValueError(
            f"Unknown residual AR feedback mode {mode!r}; "
            f"expected one of {', '.join(RESIDUAL_AR_FEEDBACK_CHOICES)}."
        )
    return mode


def validate_feedback_clip_mode(mode: str) -> str:
    if mode not in FEEDBACK_CLIP_MODE_CHOICES:
        raise ValueError(
            f"Unknown feedback clip mode {mode!r}; "
            f"expected one of {', '.join(FEEDBACK_CLIP_MODE_CHOICES)}."
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


def _scale_gradient_dataset(ds: xr.Dataset, gamma: float | jnp.ndarray) -> xr.Dataset:
    """Forward pass unchanged; backward gradient scaled by gamma.

    gamma=0 ⇔ jax.lax.stop_gradient.
    gamma=1 ⇔ identity (full gradient).
    """
    new = {}
    for name, da in ds.data_vars.items():
        arr = _unwrap(da)
        arr_sg = jax.lax.stop_gradient(arr)
        scaled = arr_sg + gamma * (arr - arr_sg)
        new[name] = _wrap_like(scaled, da)
    return xr.Dataset(new, coords=ds.coords, attrs=ds.attrs)


def make_feedback_var_stddev_dict(
    stddev_ds: xr.Dataset,
    *,
    pressure_levels: tuple[int, ...] | None = None,
) -> dict[str, jnp.ndarray]:
    """Build per-var stddev arrays for normalized feedback clipping.

    For each var in `stddev_ds` (typically `stddev_by_level` from norm stats),
    returns a jnp array pre-shaped to broadcast against the residual prediction
    arrays. Level-dependent vars (e.g. temperature, geopotential) get shape
    (1, 1, level, 1, 1) which broadcasts against (batch, time, level, lat, lon);
    surface vars get a scalar.

    `pressure_levels`: when provided, level-dependent vars are subset to these
    levels (in this order) before flattening — required because stats files
    typically carry all 37 ECMWF levels but the task config uses a subset
    (e.g. 13 levels for GraphCast).

    The residual is expected to come from the GraphCast prediction Dataset,
    which has consistent dim ordering (batch, time, [level,] lat, lon).
    """
    out: dict[str, jnp.ndarray] = {}
    for name, da in stddev_ds.data_vars.items():
        if "level" in da.dims:
            sub = da.sel(level=list(pressure_levels)) if pressure_levels is not None else da
            vals = np.asarray(sub.values, dtype=np.float32)
            # (level,) -> (1, 1, level, 1, 1)
            out[name] = jnp.asarray(vals.reshape(1, 1, -1, 1, 1))
        else:
            vals = np.asarray(da.values, dtype=np.float32)
            out[name] = jnp.asarray(float(vals))
    return out


def _clip_residual_normalized(
    residual_ds: xr.Dataset,
    *,
    var_stddevs: dict[str, jnp.ndarray],
    c: float,
    mode: str,
) -> xr.Dataset:
    """Clip per-variable residual in normalized (input-std) units.

    For each var with std σ:
        r_norm = r / σ
        r_clipped_norm = c * tanh(r_norm / c)      (mode='tanh')
        r_clipped_norm = clip(r_norm, -c, c)        (mode='hard')
        r_clipped     = r_clipped_norm * σ
    """
    new = {}
    for name, da in residual_ds.data_vars.items():
        arr = _unwrap(da)
        std = var_stddevs.get(name)
        if std is None:
            new[name] = da
            continue
        arr_norm = arr / std
        if mode == FEEDBACK_CLIP_MODE_TANH:
            clipped_norm = c * jnp.tanh(arr_norm / c)
        elif mode == FEEDBACK_CLIP_MODE_HARD:
            clipped_norm = jnp.clip(arr_norm, -c, c)
        else:  # NONE — shouldn't reach here if caller checks, but be defensive
            clipped_norm = arr_norm
        clipped = clipped_norm * std
        new[name] = _wrap_like(clipped, da)
    return xr.Dataset(new, coords=residual_ds.coords, attrs=residual_ds.attrs)


def residual_feedback_components(
    *,
    baseline_pred: xr.Dataset,
    full_pred: xr.Dataset,
    mode: str,
    lam: float | jnp.ndarray | None = None,
    stop_grad: bool = False,
    feedback_clip_mode: str = FEEDBACK_CLIP_MODE_NONE,
    feedback_clip_c: float = 1.0,
    feedback_grad_scale: float = 1.0,
    var_stddevs: dict[str, jnp.ndarray] | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Compute physical-input feedback AND the matching Mamba-memory residual.

    Returns:
        feedback_preds: the new physical rolling-input field
                         = baseline                                  (mode='baseline')
                         = full_pred = baseline + raw_residual       (mode='baseline_plus_residual')
                         = baseline + lam * processed_residual       (mode='gated')
        feedback_residual: the residual *actually injected on top of baseline*.
            CALLER SHOULD USE THIS for advance_residual_inputs() so the Mamba
            memory stream stays consistent with the physical input stream.
            None for 'baseline' mode (no residual was injected).

    Why the second return value matters: when feedback path is clipped/scaled
    (`feedback_clip_mode`, `feedback_grad_scale`, or `lam<1`), the residual that
    physically ends up in the next GraphCast input is NOT the raw `residual_preds`
    Mamba produced. Passing raw `residual_preds` into Mamba memory creates a
    distribution mismatch between what Mamba "remembers" and what GraphCast
    actually saw. Use `feedback_residual` (= λ * processed_residual) instead.

    Version B controls (mode='gated' only):
        feedback_clip_mode: 'none' | 'tanh' | 'hard'
            Clip the residual that gets fed back in normalized (input-std) units,
            per variable. Output residual is NOT clipped — only the feedback path.
        feedback_clip_c: clip threshold in std units.
        feedback_grad_scale: 0.0 = stop_gradient; 1.0 = full gradient;
            (0, 1) = partial (V-B2). `stop_grad=True` overrides to 0.0.
        var_stddevs: required when feedback_clip_mode != 'none'.
    """
    validate_residual_ar_feedback(mode)
    if mode == RESIDUAL_AR_FEEDBACK_BASELINE:
        out = baseline_pred
        if stop_grad:
            out = _stop_grad_dataset(out)
        return out, None
    if mode == RESIDUAL_AR_FEEDBACK_BASELINE_PLUS_RESIDUAL:
        # next_input = baseline + raw_residual (no λ, no clip)
        residual = _add_datasets(full_pred, _scale_dataset(baseline_pred, -1.0))
        if stop_grad:
            residual = _stop_grad_dataset(residual)
            out = _add_datasets(_stop_grad_dataset(baseline_pred), residual)
        else:
            out = full_pred
        return out, residual
    # mode == RESIDUAL_AR_FEEDBACK_GATED
    if lam is None:
        raise ValueError("mode='gated' requires lam (the feedback strength).")
    validate_feedback_clip_mode(feedback_clip_mode)
    residual = _add_datasets(full_pred, _scale_dataset(baseline_pred, -1.0))

    if feedback_clip_mode != FEEDBACK_CLIP_MODE_NONE:
        if var_stddevs is None:
            raise ValueError(
                f"feedback_clip_mode={feedback_clip_mode!r} requires var_stddevs "
                "(call make_feedback_var_stddev_dict(stats['stddev_by_level'])."
            )
        residual = _clip_residual_normalized(
            residual,
            var_stddevs=var_stddevs,
            c=feedback_clip_c,
            mode=feedback_clip_mode,
        )

    # Gradient-flow control on the feedback path:
    #   stop_grad=True  -> full stop (legacy V-A behavior)
    #   else use feedback_grad_scale:
    #       0.0 -> full stop
    #       1.0 -> full gradient
    #       (0,1) -> partial (V-B2)
    effective_grad_scale = 0.0 if stop_grad else feedback_grad_scale
    if effective_grad_scale == 0.0:
        residual = _stop_grad_dataset(residual)
    elif effective_grad_scale != 1.0:
        residual = _scale_gradient_dataset(residual, effective_grad_scale)

    gated_residual = _scale_dataset(residual, lam)
    out = _add_datasets(baseline_pred, gated_residual)
    return out, gated_residual


def residual_physical_feedback(
    *,
    baseline_pred: xr.Dataset,
    full_pred: xr.Dataset,
    mode: str,
    lam: float | jnp.ndarray | None = None,
    stop_grad: bool = False,
    feedback_clip_mode: str = FEEDBACK_CLIP_MODE_NONE,
    feedback_clip_c: float = 1.0,
    feedback_grad_scale: float = 1.0,
    var_stddevs: dict[str, jnp.ndarray] | None = None,
) -> xr.Dataset:
    """Backwards-compatible wrapper: returns only the physical feedback prediction.

    Callers that also need the matching Mamba-memory residual (to keep the two
    streams consistent under clipping/scaling) should use
    `residual_feedback_components` instead.
    """
    feedback_preds, _ = residual_feedback_components(
        baseline_pred=baseline_pred,
        full_pred=full_pred,
        mode=mode,
        lam=lam,
        stop_grad=stop_grad,
        feedback_clip_mode=feedback_clip_mode,
        feedback_clip_c=feedback_clip_c,
        feedback_grad_scale=feedback_grad_scale,
        var_stddevs=var_stddevs,
    )
    return feedback_preds
