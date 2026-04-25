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
from .model import gc, scalarize_loss


def run_eval(
    transformed_eval,
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
) -> dict[str, float]:
    del state

    losses: list[float] = []
    # Per-variable accumulators for RMSE and MAE.
    var_se_sums: dict[str, float] = {}
    var_ae_sums: dict[str, float] = {}
    var_counts: dict[str, int] = {}

    n_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    index_batches = [
        eval_indices[i : i + eval_batch_size]
        for i in range(0, len(eval_indices), eval_batch_size)
    ]

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

    state_by_batch_size: dict[int, hk.State] = {}
    eval_fn_cache: dict[int, callable] = {}

    def _eval_batch_fn(batch_size: int, batch_state: hk.State):
        fn = eval_fn_cache.get(batch_size)
        if fn is None:
            @jax.jit
            def fn(params, key, inputs, targets, forcings):
                (loss_and_diag, preds), _ = transformed_eval.apply(
                    params,
                    batch_state,
                    key,
                    inputs,
                    targets,
                    forcings,
                    False,
                )
                return scalarize_loss(loss_and_diag[0]), preds

            eval_fn_cache[batch_size] = fn
        return eval_fn_cache[batch_size]

    for batch_i in range(1, n_batches + 1):
        inputs, targets, forcings = pending.pop(0).result()
        _fill()

        batch_size = int(inputs.sizes["batch"])
        rng, init_key, apply_key = jax.random.split(rng, 3)
        if batch_size not in state_by_batch_size:
            _, batch_state = transformed_eval.init(init_key, inputs, targets, forcings, False)
            batch_state = jax.tree_util.tree_map(
                lambda leaf: jnp.zeros_like(leaf) if isinstance(leaf, jax.Array) else leaf,
                batch_state,
            )
            state_by_batch_size[batch_size] = batch_state
        loss_value, preds = _eval_batch_fn(batch_size, state_by_batch_size[batch_size])(
            params, apply_key, inputs, targets, forcings
        )
        loss = float(loss_value)
        losses.append(loss)

        for var_name in preds.data_vars:
            if var_name not in targets.data_vars:
                continue
            common_dims = [d for d in targets[var_name].dims if d in preds[var_name].dims]
            pred_aligned = preds[var_name].transpose(*common_dims)
            tgt_aligned = targets[var_name].transpose(*common_dims)
            pred_vals = np.asarray(pred_aligned.values, dtype=np.float32)
            tgt_vals = np.asarray(tgt_aligned.values, dtype=np.float32)
            diff = pred_vals - tgt_vals
            n = diff.size
            var_se_sums[var_name] = var_se_sums.get(var_name, 0.0) + float(np.sum(diff ** 2))
            var_ae_sums[var_name] = var_ae_sums.get(var_name, 0.0) + float(np.sum(np.abs(diff)))
            var_counts[var_name] = var_counts.get(var_name, 0) + n
        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_loss {loss:.6f}"
            )

    executor.shutdown(wait=False)

    results: dict[str, float] = {"total": float(np.mean(losses))}

    if var_counts:
        total_se, total_ae, total_n = 0.0, 0.0, 0
        print(f"[{progress_label}] === Per-variable metrics ===")
        for var_name in sorted(var_counts.keys()):
            n = var_counts[var_name]
            rmse = float(np.sqrt(var_se_sums[var_name] / n))
            mae = float(var_ae_sums[var_name] / n)
            results[f"{var_name}_RMSE"] = rmse
            results[f"{var_name}_MAE"] = mae
            print(f"[{progress_label}]   {var_name}: RMSE={rmse:.4f}  MAE={mae:.4f}")
            total_se += var_se_sums[var_name]
            total_ae += var_ae_sums[var_name]
            total_n += n
        overall_rmse = float(np.sqrt(total_se / total_n))
        overall_mae = float(total_ae / total_n)
        results["overall_RMSE"] = overall_rmse
        results["overall_MAE"] = overall_mae
        print(f"[{progress_label}]   OVERALL: RMSE={overall_rmse:.4f}  MAE={overall_mae:.4f}")

    return results
