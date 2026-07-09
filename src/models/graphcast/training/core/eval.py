from __future__ import annotations

import concurrent.futures
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from .batching import BatchBuilder, build_batch_from_indices_vectorized
from .eval_selection import (
    EVAL_SUBSET_STRATIFIED_FIXED,
    select_eval_subset,
)
from .model import gc, scalarize_loss


def _times_for_indices(eval_ds: xr.Dataset, indices: np.ndarray) -> pd.DatetimeIndex | None:
    if not hasattr(eval_ds, "time"):
        return None
    try:
        time_index = pd.DatetimeIndex(pd.to_datetime(eval_ds.time.values))
        return pd.DatetimeIndex(time_index[np.asarray(indices, dtype=np.int64)])
    except (IndexError, TypeError, ValueError):
        return None


def _map_temporal_state_leaves(state: hk.State, fn) -> hk.State:
    mutable_state = hk.data_structures.to_mutable_dict(state)
    for module_state in mutable_state.values():
        for state_name, leaf in module_state.items():
            if not isinstance(leaf, jax.Array):
                continue
            if state_name.endswith("_ssm_state") or state_name.endswith("_conv_cache"):
                module_state[state_name] = fn(leaf)
    return hk.data_structures.to_immutable_dict(mutable_state)


def _reset_temporal_eval_state(state: hk.State) -> hk.State:
    return _map_temporal_state_leaves(state, jnp.zeros_like)


def run_eval(
    transformed_eval_loss,
    params: hk.Params,
    state: hk.State,
    rng: jax.Array,
    eval_ds: xr.Dataset,
    eval_indices: np.ndarray,
    *,
    eval_batch_size: int,
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
    progress_label: str = "eval",
    batch_builder: BatchBuilder = build_batch_from_indices_vectorized,
    prefetch_workers: int = 4,
    prefetch_depth: int = 4,
    max_batches: int | None = None,
    subset_policy: str = EVAL_SUBSET_STRATIFIED_FIXED,
    subset_role: str = "fixed_checkpoint",
    subset_fold: int | None = None,
) -> dict[str, float]:
    del state

    losses: list[float] = []

    max_windows = None if max_batches is None else max_batches * eval_batch_size
    selection = select_eval_subset(
        np.asarray(eval_indices, dtype=np.int64),
        max_windows,
        times=_times_for_indices(eval_ds, eval_indices),
        policy=subset_policy,
        role=subset_role,
        fold=subset_fold,
    )
    selected_indices = selection.item_ids
    index_batches = [
        selected_indices[i : i + eval_batch_size]
        for i in range(0, len(selected_indices), eval_batch_size)
    ]
    available_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    n_batches = len(index_batches)
    if n_batches == 0:
        raise ValueError("No eval batches selected.")
    if selection.capped:
        readable_policy = selection.policy.replace("_", " ")
        print(
            f"[{progress_label}] using {readable_policy} "
            f"{len(selected_indices)}/{len(eval_indices)} validation windows "
            f"({n_batches}/{available_batches} batches)"
        )

    def _build(idx):
        return batch_builder(
            eval_ds,
            indices=idx,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )

    t_eval0 = time.time()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=prefetch_workers)
    pending: list[concurrent.futures.Future] = []
    next_submit = 0

    def _fill():
        nonlocal next_submit
        while len(pending) < prefetch_depth and next_submit < n_batches:
            pending.append(executor.submit(_build, index_batches[next_submit]))
            next_submit += 1

    _fill()

    loss_state_by_batch_size: dict[int, hk.State] = {}
    loss_eval_fn_cache: dict[int, callable] = {}

    def _loss_eval_batch_fn(batch_size: int, batch_state: hk.State):
        fn = loss_eval_fn_cache.get(batch_size)
        if fn is None:
            @jax.jit
            def fn(params, key, inputs, targets, forcings):
                loss_and_diag, _ = transformed_eval_loss.apply(
                    params,
                    batch_state,
                    key,
                    inputs,
                    targets,
                    forcings,
                    False,
                )
                return scalarize_loss(loss_and_diag[0])

            loss_eval_fn_cache[batch_size] = fn
        return loss_eval_fn_cache[batch_size]

    for batch_i in range(1, n_batches + 1):
        inputs, targets, forcings = pending.pop(0).result()
        _fill()

        batch_size = int(inputs.sizes["batch"])
        rng, loss_init_key, loss_apply_key = jax.random.split(rng, 3)
        if batch_size not in loss_state_by_batch_size:
            _, batch_state = transformed_eval_loss.init(loss_init_key, inputs, targets, forcings, False)
            loss_state_by_batch_size[batch_size] = _reset_temporal_eval_state(batch_state)

        loss_value = _loss_eval_batch_fn(
            batch_size,
            _reset_temporal_eval_state(loss_state_by_batch_size[batch_size]),
        )(params, loss_apply_key, inputs, targets, forcings)
        loss = float(loss_value)
        losses.append(loss)
        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_loss {loss:.6f}"
            )

    executor.shutdown(wait=False)
    return {
        "total": float(np.mean(losses)),
        "batches": float(n_batches),
        **selection.metadata(item_name="windows"),
    }
