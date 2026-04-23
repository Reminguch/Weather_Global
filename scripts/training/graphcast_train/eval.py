from __future__ import annotations

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
    transformed,
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
    transformed_predict=None,
    batch_builder: BatchBuilder = build_batch_from_indices_vectorized,
) -> dict[str, float]:
    # Reset SSM hidden states so eval starts from clean zeros
    state = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros_like(leaf) if isinstance(leaf, jax.Array) else leaf,
        state,
    )

    losses: list[float] = []
    # Per-variable accumulators for RMSE and MAE
    var_se_sums: dict[str, float] = {}   # sum of squared errors
    var_ae_sums: dict[str, float] = {}   # sum of absolute errors
    var_counts: dict[str, int] = {}      # number of elements
    compute_metrics = transformed_predict is not None

    n_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    t_eval0 = time.time()

    for batch_i, i in enumerate(range(0, len(eval_indices), eval_batch_size), start=1):
        idx = eval_indices[i : i + eval_batch_size]
        inputs, targets, forcings = batch_builder(
            eval_ds,
            indices=idx,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )

        rng, key = jax.random.split(rng)

        if compute_metrics:
            # Get loss from original transformed
            (loss_and_diag, _state_after) = transformed.apply(
                params, state, key, inputs, targets, forcings, False)
            loss = float(scalarize_loss(loss_and_diag[0]))
            # Get predictions from predict fn
            (preds, _pred_state) = transformed_predict.apply(
                params, state, key, inputs, targets, forcings, False)
            # Accumulate per-variable SE and AE in original (denormalized) space
            for var_name in preds.data_vars:
                if var_name not in targets.data_vars:
                    continue
                # Align dimensions: preds and targets may have different dim order
                # when target_steps > 1 (e.g. batch,time,lat,lon vs time,batch,lat,lon)
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
        else:
            (loss_and_diag, _state_after) = transformed.apply(
                params, state, key, inputs, targets, forcings, False)
            loss = float(scalarize_loss(loss_and_diag[0]))

        losses.append(loss)

        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_loss {loss:.6f}"
            )

    results: dict[str, float] = {"total": float(np.mean(losses))}

    if compute_metrics and var_counts:
        # Compute and print per-variable RMSE and MAE
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
