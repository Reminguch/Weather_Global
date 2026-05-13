#!/usr/bin/env python3
"""Residual Mamba equivalence diagnostic for one prepared warm chunk.

This is meant to answer one very specific question:

    A = loss(residual_pred, target - baseline_pred)
    B = loss(baseline_pred + residual_pred, target)

For squared-error GraphCast metrics, A and B should be identical when they are
computed from the exact same baseline and residual predictions.  If they are
not, the reconstruction/metric path is wrong.  If they match but one eval path
is much worse than another, the problem is earlier in the residual eval path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if GRAPHCAST_LOCAL.exists() and str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))

from graphcast import checkpoint, graphcast  # noqa: E402

from scripts.analyze_models.legacy.analysis_metrics import (  # noqa: E402
    GRAPHCAST_PER_VARIABLE_WEIGHTS,
    normalized_per_variable_mse,
    normalized_weighted_mse_allvars,
)
from scripts.analyze_models.unified_resolution_eval import (  # noqa: E402
    DEFAULT_PREPARED_DATA_ROOT,
    HOURS_PER_STEP,
    N_INPUT_STEPS,
    RES_GRID_STRIDE,
    STATS_DIR,
    _is_residual_mamba_run,
    _load_stats,
    _open_prepared_eval_store_for_checkpoint,
    _prepared_chunk_start_indices,
    _iter_prepared_stream_chunks,
)
from src.models.graphcast.evaluation.device_resolution_eval import (  # noqa: E402
    PreparedDeviceResolutionEvaluator,
    add_device_accumulator_to_host,
)
from src.models.graphcast.runtime import (  # noqa: E402
    _clone_state,
    _constant_inputs,
    _dataset_to_numpy,
    _update_inputs,
    infer_family,
    load_run_config,
)
from src.models.graphcast.training.core.prepared_block_batches import PreparedBlockBatchLoader  # noqa: E402
from src.models.graphcast.training.core.prepared_data import load_prepared_metric_grid  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    advance_residual_inputs,
    build_zero_residual_inputs,
)
from src.models.mamba.residual_mamba.runtime import _build_residual_rollout_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test residual Mamba A/B loss equivalence on one prepared warm chunk."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Residual Mamba ckpt_best.npz path.")
    parser.add_argument("--resolution", type=int, default=None, help="Prepared resolution; inferred when omitted.")
    parser.add_argument("--prepared-data-root", type=Path, default=ROOT / DEFAULT_PREPARED_DATA_ROOT)
    parser.add_argument("--stats-dir", type=Path, default=ROOT / STATS_DIR)
    parser.add_argument(
        "--metric-grid-resolution",
        type=int,
        default=RES_GRID_STRIDE,
        help=(
            "Prepared grid resolution used for metric sampling. "
            f"Default {RES_GRID_STRIDE} matches resolution eval; pass model resolution for full-grid loss."
        ),
    )
    parser.add_argument("--eval-start", type=str, default=None)
    parser.add_argument("--eval-end", type=str, default=None)
    parser.add_argument("--eval-year", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=24)
    parser.add_argument("--trunk-steps", type=int, default=1)
    parser.add_argument("--lead-steps", type=int, nargs="+", default=[1])
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--chunk-index", type=int, default=0, help="Warm chunk to inspect, zero-based.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--paths",
        choices=["device_like", "runtime", "both"],
        default="both",
        help="Which residual forward path(s) to diagnose.",
    )
    parser.add_argument(
        "--skip-device-accumulator",
        action="store_true",
        help="Skip comparison against PreparedDeviceResolutionEvaluator.evaluate_warm_chunk.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _infer_resolution(ckpt_path: Path, run_cfg: dict[str, Any]) -> int:
    import re

    candidates = [ckpt_path.parent.name, ckpt_path.parent.parent.name]
    baseline = run_cfg.get("residual_training", {}).get("baseline_checkpoint")
    if baseline:
        candidates.append(str(baseline))
    for candidate in candidates:
        match = re.search(r"(?:^|[_/])res(\d+)(?:[_.]|$)", candidate)
        if match:
            return int(match.group(1))
    raise ValueError("Could not infer resolution from checkpoint path; pass --resolution.")


def _as_float(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(arr.mean())


def _select_metric_grid(ds: xr.Dataset, lats: xr.DataArray, lons: xr.DataArray) -> xr.Dataset:
    if "lat" in ds.coords and "lon" in ds.coords:
        return ds.sel(lat=lats, lon=lons, method="nearest")
    return ds


def _weighted_loss(
    pred: xr.Dataset,
    target: xr.Dataset,
    *,
    metric_lats: xr.DataArray,
    metric_lons: xr.DataArray,
    stats: dict[str, xr.Dataset],
) -> tuple[float, list[float]]:
    pred_grid = _select_metric_grid(pred, metric_lats, metric_lons)
    target_grid = _select_metric_grid(target, metric_lats, metric_lons)
    loss = normalized_weighted_mse_allvars(
        pred_grid,
        target_grid,
        per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
        use_latitude_weights=True,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    values = np.asarray(loss.values, dtype=np.float64).reshape(-1)
    return float(values.mean()), [float(v) for v in values.tolist()]


def _per_variable_loss(
    pred: xr.Dataset,
    target: xr.Dataset,
    *,
    metric_lats: xr.DataArray,
    metric_lons: xr.DataArray,
    stats: dict[str, xr.Dataset],
) -> dict[str, float]:
    pred_grid = _select_metric_grid(pred, metric_lats, metric_lons)
    target_grid = _select_metric_grid(target, metric_lats, metric_lons)
    losses = normalized_per_variable_mse(
        pred_grid,
        target_grid,
        use_latitude_weights=True,
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    return {name: _as_float(loss.values) for name, loss in sorted(losses.items())}


def _max_abs_identity_error(
    *,
    baseline_pred: xr.Dataset,
    residual_pred: xr.Dataset,
    target: xr.Dataset,
    residual_target: xr.Dataset,
) -> float:
    max_abs = 0.0
    for name in target.data_vars:
        if name not in baseline_pred or name not in residual_pred or name not in residual_target:
            continue
        full_error = (baseline_pred[name] + residual_pred[name]) - target[name]
        residual_error = residual_pred[name] - residual_target[name]
        diff = np.asarray((full_error - residual_error).values, dtype=np.float64)
        if diff.size:
            max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))
    return max_abs


def _record_step(
    *,
    path_name: str,
    anchor_index: int,
    lead_step: int,
    baseline_pred: xr.Dataset,
    residual_pred: xr.Dataset,
    target_step: xr.Dataset,
    metric_lats: xr.DataArray,
    metric_lons: xr.DataArray,
    stats: dict[str, xr.Dataset],
) -> dict[str, Any]:
    baseline_pred = _dataset_to_numpy(baseline_pred)
    residual_pred = _dataset_to_numpy(residual_pred)
    target_step = _dataset_to_numpy(target_step)
    residual_target = _dataset_to_numpy(target_step - baseline_pred)
    full_pred = _dataset_to_numpy(baseline_pred + residual_pred)

    residual_loss, residual_loss_by_batch = _weighted_loss(
        residual_pred,
        residual_target,
        metric_lats=metric_lats,
        metric_lons=metric_lons,
        stats=stats,
    )
    full_loss, full_loss_by_batch = _weighted_loss(
        full_pred,
        target_step,
        metric_lats=metric_lats,
        metric_lons=metric_lons,
        stats=stats,
    )
    baseline_loss, baseline_loss_by_batch = _weighted_loss(
        baseline_pred,
        target_step,
        metric_lats=metric_lats,
        metric_lons=metric_lons,
        stats=stats,
    )
    per_var_a = _per_variable_loss(
        residual_pred,
        residual_target,
        metric_lats=metric_lats,
        metric_lons=metric_lons,
        stats=stats,
    )
    per_var_b = _per_variable_loss(
        full_pred,
        target_step,
        metric_lats=metric_lats,
        metric_lons=metric_lons,
        stats=stats,
    )
    per_var_delta = {name: per_var_b[name] - per_var_a[name] for name in per_var_a.keys() & per_var_b.keys()}
    return {
        "path": path_name,
        "anchor_index": int(anchor_index),
        "lead_step": int(lead_step),
        "lead_hours": int(lead_step * HOURS_PER_STEP),
        "A_residual_loss": residual_loss,
        "B_full_reconstruction_loss": full_loss,
        "baseline_only_loss": baseline_loss,
        "B_minus_A": full_loss - residual_loss,
        "abs_B_minus_A": abs(full_loss - residual_loss),
        "relative_B_minus_A": (full_loss - residual_loss) / residual_loss if residual_loss else float("nan"),
        "identity_max_abs_error": _max_abs_identity_error(
            baseline_pred=baseline_pred,
            residual_pred=residual_pred,
            target=target_step,
            residual_target=residual_target,
        ),
        "A_by_batch": residual_loss_by_batch,
        "B_by_batch": full_loss_by_batch,
        "baseline_by_batch": baseline_loss_by_batch,
        "per_variable_A": per_var_a,
        "per_variable_B": per_var_b,
        "per_variable_B_minus_A": per_var_delta,
    }


def _roll_warmup_and_branch_runtime(
    *,
    residual_bundle: dict[str, Any],
    baseline_bundle: dict[str, Any],
    inputs_by_step: tuple[xr.Dataset, ...],
    targets_by_step: tuple[xr.Dataset, ...],
    forcings_by_step: tuple[xr.Dataset, ...],
    warmup_steps: int,
    trunk_steps: int,
    max_lead_steps: int,
    seed: int,
    metric_lats: xr.DataArray,
    metric_lons: xr.DataArray,
    stats: dict[str, xr.Dataset],
) -> list[dict[str, Any]]:
    inputs0 = inputs_by_step[0]
    targets0 = targets_by_step[0]
    forcings0 = forcings_by_step[0]
    constant_inputs = _constant_inputs(inputs0, targets0, forcings0)
    rolling_inputs = inputs0.drop_vars(constant_inputs.keys())
    residual_inputs = build_zero_residual_inputs(inputs0, targets0.isel(time=slice(0, 1)))
    baseline_state = None
    residual_state = None

    step_keys = jax.random.split(
        jax.random.PRNGKey(seed),
        2 * int(warmup_steps) + int(trunk_steps) * (2 + max_lead_steps * 2),
    )
    key_i = 0
    for step_i in range(int(warmup_steps)):
        target_step = targets_by_step[step_i].isel(time=slice(0, 1))
        forcings_step = forcings_by_step[step_i].isel(time=slice(0, 1))
        all_inputs = xr.merge([constant_inputs, rolling_inputs])
        baseline_pred, baseline_state = baseline_bundle["step"](
            rng=step_keys[key_i],
            inputs=all_inputs,
            targets_template=target_step,
            forcings=forcings_step,
            state=baseline_state,
        )
        residual_pred, residual_state = residual_bundle["step"](
            rng=step_keys[key_i + 1],
            inputs=residual_inputs,
            targets_template=target_step,
            forcings=forcings_step,
            state=residual_state,
        )
        residual_inputs = advance_residual_inputs(residual_inputs, target_step - baseline_pred)
        rolling_inputs = _update_inputs(rolling_inputs, xr.merge([target_step, forcings_step]))
        key_i += 2

    records = []
    for anchor_i in range(int(trunk_steps)):
        branch_rolling = rolling_inputs.copy(deep=False)
        branch_residual_inputs = residual_inputs.copy(deep=False)
        branch_baseline_state = _clone_state(baseline_state)
        branch_residual_state = _clone_state(residual_state)
        branch_keys = jax.random.split(step_keys[key_i], max_lead_steps * 2)
        key_i += 1
        branch_targets = targets_by_step[int(warmup_steps) + anchor_i]
        branch_forcings = forcings_by_step[int(warmup_steps) + anchor_i]
        for lead_i in range(max_lead_steps):
            target_step = branch_targets.isel(time=slice(lead_i, lead_i + 1))
            forcings_step = branch_forcings.isel(time=slice(lead_i, lead_i + 1))
            all_inputs = xr.merge([constant_inputs, branch_rolling])
            baseline_pred, branch_baseline_state = baseline_bundle["step"](
                rng=branch_keys[2 * lead_i],
                inputs=all_inputs,
                targets_template=target_step,
                forcings=forcings_step,
                state=branch_baseline_state,
            )
            residual_pred, branch_residual_state = residual_bundle["step"](
                rng=branch_keys[2 * lead_i + 1],
                inputs=branch_residual_inputs,
                targets_template=target_step,
                forcings=forcings_step,
                state=branch_residual_state,
            )
            records.append(
                _record_step(
                    path_name="runtime",
                    anchor_index=anchor_i,
                    lead_step=lead_i + 1,
                    baseline_pred=baseline_pred,
                    residual_pred=residual_pred,
                    target_step=target_step,
                    metric_lats=metric_lats,
                    metric_lons=metric_lons,
                    stats=stats,
                )
            )
            branch_residual_inputs = advance_residual_inputs(branch_residual_inputs, target_step - baseline_pred)
            branch_rolling = _update_inputs(branch_rolling, xr.merge([target_step, forcings_step]))
        truth_target = branch_targets.isel(time=slice(0, 1))
        truth_forcings = branch_forcings.isel(time=slice(0, 1))
        all_inputs = xr.merge([constant_inputs, rolling_inputs])
        baseline_pred, baseline_state = baseline_bundle["step"](
            rng=step_keys[key_i],
            inputs=all_inputs,
            targets_template=truth_target,
            forcings=truth_forcings,
            state=baseline_state,
        )
        _residual_pred, residual_state = residual_bundle["step"](
            rng=step_keys[key_i + 1],
            inputs=residual_inputs,
            targets_template=truth_target,
            forcings=truth_forcings,
            state=residual_state,
        )
        residual_inputs = advance_residual_inputs(residual_inputs, truth_target - baseline_pred)
        rolling_inputs = _update_inputs(rolling_inputs, xr.merge([truth_target, truth_forcings]))
        key_i += 2
    return records


def _device_like_residual_step(
    evaluator: PreparedDeviceResolutionEvaluator,
    *,
    rng: tuple[jax.Array, jax.Array],
    rolling_inputs: xr.Dataset,
    constant_inputs: xr.Dataset,
    residual_inputs: xr.Dataset,
    target_step: xr.Dataset,
    forcings_step: xr.Dataset,
    residual_state: Any,
    baseline_state: Any,
):
    bundle = evaluator.bundle
    assert bundle.baseline is not None
    all_inputs = xr.merge([constant_inputs, rolling_inputs])
    baseline_pred, baseline_next = bundle.baseline.transformed.apply(
        bundle.baseline.params,
        baseline_state,
        rng[0],
        all_inputs,
        target_step,
        forcings_step,
    )
    residual_pred, residual_next = bundle.primary.transformed.apply(
        bundle.primary.params,
        residual_state,
        rng[1],
        residual_inputs,
        target_step,
        forcings_step,
    )
    residual_target = target_step - baseline_pred
    return (
        baseline_pred,
        residual_pred,
        residual_next,
        baseline_next,
        advance_residual_inputs(residual_inputs, residual_target),
    )


def _roll_warmup_and_branch_device_like(
    *,
    evaluator: PreparedDeviceResolutionEvaluator,
    inputs_by_step: tuple[xr.Dataset, ...],
    targets_by_step: tuple[xr.Dataset, ...],
    forcings_by_step: tuple[xr.Dataset, ...],
    warmup_steps: int,
    trunk_steps: int,
    max_lead_steps: int,
    seed: int,
    metric_lats: xr.DataArray,
    metric_lons: xr.DataArray,
    stats: dict[str, xr.Dataset],
) -> list[dict[str, Any]]:
    inputs0 = inputs_by_step[0]
    targets0 = targets_by_step[0]
    forcings0 = forcings_by_step[0]
    rng = jax.random.PRNGKey(seed)
    states = evaluator._initial_states(rng, inputs0, targets0, forcings0)
    residual_state, baseline_state = states
    constant_inputs = _constant_inputs(inputs0, targets0, forcings0)
    rolling_inputs = inputs0.drop_vars(constant_inputs.keys())
    residual_inputs = build_zero_residual_inputs(inputs0, targets0.isel(time=slice(0, 1)))

    step_keys = jax.random.split(
        rng,
        2 * int(warmup_steps) + int(trunk_steps) * (2 + max_lead_steps * 2),
    )
    key_i = 0
    for step_i in range(int(warmup_steps)):
        target_step = targets_by_step[step_i].isel(time=slice(0, 1))
        forcings_step = forcings_by_step[step_i].isel(time=slice(0, 1))
        _baseline_pred, _residual_pred, residual_state, baseline_state, residual_inputs = _device_like_residual_step(
            evaluator,
            rng=(step_keys[key_i], step_keys[key_i + 1]),
            rolling_inputs=rolling_inputs,
            constant_inputs=constant_inputs,
            residual_inputs=residual_inputs,
            target_step=target_step,
            forcings_step=forcings_step,
            residual_state=residual_state,
            baseline_state=baseline_state,
        )
        rolling_inputs = _update_inputs(rolling_inputs, xr.merge([target_step, forcings_step]))
        key_i += 2

    records = []
    for anchor_i in range(int(trunk_steps)):
        branch_rolling = rolling_inputs.copy(deep=False)
        branch_residual_inputs = residual_inputs.copy(deep=False)
        branch_residual_state = jax.tree_util.tree_map(lambda x: x, residual_state)
        branch_baseline_state = jax.tree_util.tree_map(lambda x: x, baseline_state)
        branch_keys = jax.random.split(step_keys[key_i], max_lead_steps * 2)
        key_i += 1
        branch_targets = targets_by_step[int(warmup_steps) + anchor_i]
        branch_forcings = forcings_by_step[int(warmup_steps) + anchor_i]
        for lead_i in range(max_lead_steps):
            target_step = branch_targets.isel(time=slice(lead_i, lead_i + 1))
            forcings_step = branch_forcings.isel(time=slice(lead_i, lead_i + 1))
            baseline_pred, residual_pred, branch_residual_state, branch_baseline_state, branch_residual_inputs = (
                _device_like_residual_step(
                    evaluator,
                    rng=(branch_keys[2 * lead_i], branch_keys[2 * lead_i + 1]),
                    rolling_inputs=branch_rolling,
                    constant_inputs=constant_inputs,
                    residual_inputs=branch_residual_inputs,
                    target_step=target_step,
                    forcings_step=forcings_step,
                    residual_state=branch_residual_state,
                    baseline_state=branch_baseline_state,
                )
            )
            records.append(
                _record_step(
                    path_name="device_like",
                    anchor_index=anchor_i,
                    lead_step=lead_i + 1,
                    baseline_pred=baseline_pred,
                    residual_pred=residual_pred,
                    target_step=target_step,
                    metric_lats=metric_lats,
                    metric_lons=metric_lons,
                    stats=stats,
                )
            )
            branch_rolling = _update_inputs(branch_rolling, xr.merge([target_step, forcings_step]))
        truth_target = branch_targets.isel(time=slice(0, 1))
        truth_forcings = branch_forcings.isel(time=slice(0, 1))
        _baseline_pred, _residual_pred, residual_state, baseline_state, residual_inputs = _device_like_residual_step(
            evaluator,
            rng=(step_keys[key_i], step_keys[key_i + 1]),
            rolling_inputs=rolling_inputs,
            constant_inputs=constant_inputs,
            residual_inputs=residual_inputs,
            target_step=truth_target,
            forcings_step=truth_forcings,
            residual_state=residual_state,
            baseline_state=baseline_state,
        )
        rolling_inputs = _update_inputs(rolling_inputs, xr.merge([truth_target, truth_forcings]))
        key_i += 2
    return records


def _device_accumulator_records(
    *,
    evaluator: PreparedDeviceResolutionEvaluator,
    inputs0: xr.Dataset,
    targets_by_step: tuple[xr.Dataset, ...],
    forcings_by_step: tuple[xr.Dataset, ...],
    warmup_steps: int,
    trunk_steps: int,
    max_lead_steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    device_acc = evaluator.evaluate_warm_chunk(
        jax.random.PRNGKey(seed),
        inputs0,
        targets_by_step,
        forcings_by_step,
        warmup_steps=warmup_steps,
        trunk_steps=trunk_steps,
    )
    host_acc = {
        "weighted_sum": np.zeros(max_lead_steps, dtype=float),
        "weighted_count": np.zeros(max_lead_steps, dtype=int),
        "per_variable_sum": {},
        "per_variable_count": {},
    }
    add_device_accumulator_to_host(
        host_acc,
        device_acc,
        variable_names=evaluator.metric_variable_names,
    )
    values = np.divide(
        host_acc["weighted_sum"],
        host_acc["weighted_count"],
        out=np.full(max_lead_steps, np.nan),
        where=host_acc["weighted_count"] > 0,
    )
    return [
        {
            "path": "device_accumulator",
            "anchor_index": None,
            "lead_step": lead_i + 1,
            "lead_hours": int((lead_i + 1) * HOURS_PER_STEP),
            "B_full_reconstruction_loss": float(values[lead_i]),
            "count": int(host_acc["weighted_count"][lead_i]),
        }
        for lead_i in range(max_lead_steps)
    ]


def _filter_leads(records: list[dict[str, Any]], lead_steps: list[int]) -> list[dict[str, Any]]:
    keep = set(int(step) for step in lead_steps)
    return [record for record in records if int(record["lead_step"]) in keep]


def _print_summary(payload: dict[str, Any]) -> None:
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
    print("\nloss equivalence:")
    for record in payload["records"]:
        if record["path"] == "device_accumulator":
            print(
                f"  {record['path']:18s} lead={record['lead_hours']:>2d}h "
                f"B={record['B_full_reconstruction_loss']:.6f} "
                f"n={record['count']}"
            )
            continue
        print(
            f"  {record['path']:18s} anchor={record['anchor_index']:>2d} lead={record['lead_hours']:>2d}h "
            f"A={record['A_residual_loss']:.6f} "
            f"B={record['B_full_reconstruction_loss']:.6f} "
            f"B-A={record['B_minus_A']:.3e} "
            f"baseline={record['baseline_only_loss']:.6f} "
            f"identity_max_abs={record['identity_max_abs_error']:.3e}"
        )

    for path_name in ("device_like", "runtime"):
        path_records = [record for record in payload["records"] if record["path"] == path_name]
        for step in payload["metadata"]["lead_steps"]:
            lead_records = [record for record in path_records if record["lead_step"] == step]
            if not lead_records:
                continue
            b_values = np.asarray([record["B_full_reconstruction_loss"] for record in lead_records], dtype=float)
            print(
                f"  summary {path_name:11s} lead={step * HOURS_PER_STEP:>2d}h "
                f"B_mean={float(b_values.mean()):.6f} "
                f"B_min={float(b_values.min()):.6f} "
                f"B_max={float(b_values.max()):.6f} "
                f"anchors={len(lead_records)}"
            )

    by_key = {
        (record["path"], record.get("anchor_index"), record["lead_step"]): record
        for record in payload["records"]
    }
    for path_name in ("device_like", "runtime"):
        for step in payload["metadata"]["lead_steps"]:
            path_records = [record for record in payload["records"] if record["path"] == path_name and record["lead_step"] == step]
            device_record = by_key.get(("device_accumulator", None, step))
            if not path_records or device_record is None:
                continue
            mean_b = float(np.mean([record["B_full_reconstruction_loss"] for record in path_records]))
            delta = device_record["B_full_reconstruction_loss"] - mean_b
            print(
                f"  compare accumulator vs {path_name:11s} lead={step * HOURS_PER_STEP:>2d}h "
                f"delta={delta:.3e}"
            )


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    run_cfg = load_run_config(ckpt_path)
    if not _is_residual_mamba_run(run_cfg):
        raise ValueError(f"Expected residual_mamba checkpoint, found {infer_family(run_cfg)}: {ckpt_path}")
    resolution = int(args.resolution or _infer_resolution(ckpt_path, run_cfg))
    lead_steps = sorted({int(step) for step in args.lead_steps})
    if any(step <= 0 for step in lead_steps):
        raise ValueError("--lead-steps must be positive.")
    if args.window_batch_size <= 0:
        raise ValueError("--window-batch-size must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.trunk_steps <= 0:
        raise ValueError("--trunk-steps must be positive.")

    max_lead_steps = max(lead_steps)
    total_horizon_steps = int(args.warmup_steps) + int(args.trunk_steps) + max_lead_steps
    stats = _load_stats(args.stats_dir)
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)

    residual_bundle, baseline_bundle = _build_residual_rollout_bundle(ckpt_obj, stats, ckpt_path)
    task_cfg = residual_bundle["task_cfg"]
    store, eval_metadata = _open_prepared_eval_store_for_checkpoint(
        args.prepared_data_root,
        resolution=resolution,
        task_cfg=task_cfg,
        run_cfg=run_cfg,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        eval_year=args.eval_year,
    )
    chunk_start_indices = _prepared_chunk_start_indices(
        store.sizes["time"],
        trunk_steps=args.trunk_steps,
        total_horizon_steps=total_horizon_steps,
    )
    segments = [
        np.arange(
            int(chunk_start) + N_INPUT_STEPS - 1,
            int(chunk_start) + N_INPUT_STEPS - 1 + int(args.warmup_steps) + int(args.trunk_steps),
            dtype=np.int64,
        )
        for chunk_start in chunk_start_indices
    ]
    stream_chunks = list(_iter_prepared_stream_chunks(segments, batch_size=args.window_batch_size))
    if args.chunk_index < 0 or args.chunk_index >= len(stream_chunks):
        raise IndexError(f"--chunk-index {args.chunk_index} outside 0..{len(stream_chunks) - 1}")
    chunk = stream_chunks[args.chunk_index]
    loader = PreparedBlockBatchLoader(
        store,
        segments,
        input_steps=N_INPUT_STEPS,
        target_steps=max_lead_steps,
        task_cfg=task_cfg,
        dt=pd.Timedelta(hours=HOURS_PER_STEP),
        label="residual-smoking-gun",
    )
    inputs_by_step, targets_by_step, forcings_by_step, load_stats = loader.load_chunk(chunk)
    metric_lats, metric_lons = load_prepared_metric_grid(
        args.prepared_data_root,
        resolution=args.metric_grid_resolution,
    )

    records: list[dict[str, Any]] = []
    evaluator = PreparedDeviceResolutionEvaluator(
        ckpt_obj,
        stats,
        ckpt_path,
        res_grid_lats=metric_lats,
        res_grid_lons=metric_lons,
        per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
        max_lead_steps=max_lead_steps,
    )
    if args.paths in ("device_like", "both"):
        records.extend(
            _roll_warmup_and_branch_device_like(
                evaluator=evaluator,
                inputs_by_step=inputs_by_step,
                targets_by_step=targets_by_step,
                forcings_by_step=forcings_by_step,
                warmup_steps=args.warmup_steps,
                trunk_steps=args.trunk_steps,
                max_lead_steps=max_lead_steps,
                seed=args.seed,
                metric_lats=metric_lats,
                metric_lons=metric_lons,
                stats=stats,
            )
        )
    if args.paths in ("runtime", "both"):
        records.extend(
            _roll_warmup_and_branch_runtime(
                residual_bundle=residual_bundle,
                baseline_bundle=baseline_bundle,
                inputs_by_step=inputs_by_step,
                targets_by_step=targets_by_step,
                forcings_by_step=forcings_by_step,
                warmup_steps=args.warmup_steps,
                trunk_steps=args.trunk_steps,
                max_lead_steps=max_lead_steps,
                seed=args.seed,
                metric_lats=metric_lats,
                metric_lons=metric_lons,
                stats=stats,
            )
        )
    if not args.skip_device_accumulator:
        records.extend(
            _device_accumulator_records(
                evaluator=evaluator,
                inputs0=inputs_by_step[0],
                targets_by_step=targets_by_step,
                forcings_by_step=forcings_by_step,
                warmup_steps=args.warmup_steps,
                trunk_steps=args.trunk_steps,
                max_lead_steps=max_lead_steps,
                seed=args.seed,
            )
        )

    payload = {
        "metadata": {
            "checkpoint": str(ckpt_path),
            "baseline_checkpoint": run_cfg.get("residual_training", {}).get("baseline_checkpoint"),
            "resolution": resolution,
            "metric_grid_resolution": int(args.metric_grid_resolution),
            "lead_steps": lead_steps,
            "warmup_steps": int(args.warmup_steps),
            "trunk_steps": int(args.trunk_steps),
            "chunk_index": int(args.chunk_index),
            "stream_chunks": len(stream_chunks),
            "lane_segment_ids": [int(value) for value in np.asarray(chunk.lane_segment_ids).tolist()],
            "lane_offsets": [int(value) for value in np.asarray(chunk.lane_offsets).tolist()],
            "seed": int(args.seed),
            "eval_metadata": eval_metadata,
            "load_stats": {
                "load_s": float(load_stats.load_s),
                "cache_hits": int(load_stats.cache_hits),
                "cache_misses": int(load_stats.cache_misses),
                "loaded_gib": float(load_stats.loaded_gib),
            },
        },
        "records": _filter_leads(records, lead_steps),
    }
    _print_summary(payload)
    if args.output_json is not None:
        output_path = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
