#!/usr/bin/env python3
"""Unified resolution evaluation across GraphCast, GC-Mamba, and residual Mamba."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections.abc import Hashable
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import xarray

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if GRAPHCAST_LOCAL.exists() and str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))

from src.data_operations.loaders.graphcast_dataset import open_graphcast_era5
from src.models.graphcast.runtime import infer_family, load_run_config


def _require_graphcast() -> None:
    try:
        import graphcast  # noqa: F401
    except Exception as exc:
        raise ImportError("graphcast is required. Activate env via scripts/graphcast_env.sh.") from exc


_require_graphcast()
from graphcast import checkpoint, data_utils, graphcast

from scripts.analyze_models.legacy.analysis_metrics import (
    GRAPHCAST_PER_VARIABLE_WEIGHTS,
    normalized_per_variable_mse,
    normalized_weighted_mse_allvars,
)
from scripts.analyze_models.legacy.graphcast_analysis_utils import (
    build_run_jitted,
    build_truth_anchored_residual_runner,
    suppress_graphcast_future_warnings,
)


DATASET_PATH = "data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc"
STATS_DIR = "data/graphcast/graphcast/stats"
DEFAULT_OUTPUT_DATA_DIR = "plots/analyze_models/data/resolution_eval"
DEFAULT_OUTPUT_IMAGE_DIR = "plots/analyze_models/images/resolution_eval"

FAMILIES = ["graphcast", "gc_mamba", "residual_mamba"]
METRICS = ["weighted_allvars", "per_variable"]
EVAL_MODES = ["cold", "warm"]
LEAD_DAYS = [1, 2, 4]
N_EVAL_DAYS = 365
HOURS_PER_STEP = 6
N_INPUT_STEPS = 2
N_EXTRA_STEPS = 14
WINDOW_BATCH_SIZE = 8
WARMUP_STEPS = 24
TRUNK_STEPS = 32
RES_GRID_STRIDE = 15


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    family: str
    variant: str
    model_type: str
    di: int | None
    res: int
    ckpt_path: Path
    run_name: str


@dataclasses.dataclass
class EvalShardCache:
    ds_res_by_stride: dict[int, xarray.Dataset]
    metric_grid_by_stride: dict[int, tuple[xarray.DataArray, xarray.DataArray]]
    task_cfg_kwargs_by_key: dict[Hashable, dict]
    cold_batches: dict[tuple[Hashable, int, int, int], tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset] | None]
    warm_batches: dict[tuple[Hashable, int, int, int, int], tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset] | None]
    cold_hits: int = 0
    cold_misses: int = 0
    warm_hits: int = 0
    warm_misses: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified resolution evaluation across model families.")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--resolutions", type=int, nargs="+", default=None)
    parser.add_argument(
        "--checkpoint-roots",
        type=Path,
        nargs="+",
        default=None,
        help="Optional checkpoint root directories to search instead of scanning all artifacts/checkpoints.",
    )
    parser.add_argument("--lead-days", type=int, nargs="+", default=LEAD_DAYS)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    parser.add_argument("--eval-modes", nargs="+", choices=EVAL_MODES, default=EVAL_MODES)
    parser.add_argument("--n-eval-days", type=int, default=N_EVAL_DAYS)
    parser.add_argument("--window-batch-size", type=int, default=WINDOW_BATCH_SIZE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--trunk-steps", type=int, default=TRUNK_STEPS)
    parser.add_argument("--output-data-dir", type=Path, default=ROOT / DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=ROOT / DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-csv-name", type=str, default="resolution_eval.csv")
    parser.add_argument("--print-shards", action="store_true", help="Print one available family:res shard per line.")
    return parser.parse_args()


def _load_stats(stats_dir: Path) -> dict[str, xarray.Dataset]:
    return {
        "diffs_stddev_by_level": xarray.open_dataset(stats_dir / "diffs_stddev_by_level.nc").compute(),
        "mean_by_level": xarray.open_dataset(stats_dir / "mean_by_level.nc").compute(),
        "stddev_by_level": xarray.open_dataset(stats_dir / "stddev_by_level.nc").compute(),
    }


def _ensure_datetime_coord(ds: xarray.Dataset) -> xarray.Dataset:
    if np.issubdtype(ds["time"].dtype, np.integer) or ds["time"].dtype.kind in "iu":
        times = pd.to_datetime(ds["time"].values, unit="h", origin="1959-01-01")
        ds = ds.assign_coords(time=("time", times))
    if "batch" in ds.dims:
        dt = xarray.DataArray(ds["time"].values, dims=["time"]).expand_dims(batch=ds.sizes["batch"])
        ds = ds.assign_coords(datetime=dt)
    else:
        ds = ds.assign_coords(datetime=("time", ds["time"].values))
    return ds


def _prepare_dataset(max_lead_steps: int, n_eval_days: int) -> tuple[xarray.Dataset, int, int]:
    n_steps_per_day = 24 // HOURS_PER_STEP
    n_target_times = n_eval_days * n_steps_per_day
    n_context_steps = N_INPUT_STEPS + max_lead_steps
    n_total_steps = n_context_steps + n_target_times + N_EXTRA_STEPS

    ds = open_graphcast_era5(DATASET_PATH, time_slice=slice(-n_total_steps, None)).compute()
    if ds.sizes.get("lat") == 721 and ds.sizes.get("lon") == 1440:
        ds = ds.isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
    ds = _ensure_datetime_coord(ds)

    n_steps = ds.sizes["time"]
    if n_steps < n_context_steps + n_target_times:
        n_target_times = n_steps - n_context_steps
    if n_target_times <= 0:
        raise ValueError(f"Dataset has only {n_steps} steps; not enough context for lead={max_lead_steps}.")
    start_target_idx = n_steps - n_target_times
    return ds, start_target_idx, n_steps


def _freeze_cache_key(value):
    if dataclasses.is_dataclass(value):
        return _freeze_cache_key(dataclasses.asdict(value))
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_cache_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_key(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return tuple(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _task_cfg_cache_key(task_cfg) -> tuple[Hashable, dict]:
    task_cfg_kwargs = dataclasses.asdict(task_cfg)
    return _freeze_cache_key(task_cfg_kwargs), task_cfg_kwargs


def _target_indices(start_target_idx: int, n_steps: int, max_lead_steps: int) -> list[int]:
    min_target_idx = N_INPUT_STEPS + max_lead_steps - 1
    return np.arange(max(start_target_idx, min_target_idx), n_steps, dtype=int).tolist()


def _chunk_start_indices(
    start_target_idx: int,
    n_steps: int,
    *,
    warmup_steps: int,
    trunk_steps: int,
    total_horizon_steps: int,
) -> list[int]:
    min_chunk_start = max(0, start_target_idx - (N_INPUT_STEPS + warmup_steps))
    max_chunk_start = n_steps - (N_INPUT_STEPS + total_horizon_steps)
    if max_chunk_start < min_chunk_start:
        raise ValueError(
            f"Not enough timesteps for truth-anchored eval: need {N_INPUT_STEPS + total_horizon_steps}, have {n_steps}"
        )
    return np.arange(min_chunk_start, max_chunk_start + 1, trunk_steps, dtype=int).tolist()


def _get_resolved_dataset(
    ds_base: xarray.Dataset,
    stride: int,
    cache: EvalShardCache,
) -> tuple[xarray.Dataset, xarray.DataArray, xarray.DataArray]:
    if stride not in cache.ds_res_by_stride:
        ds_res = ds_base.isel(lat=slice(0, None, stride), lon=slice(0, None, stride)) if stride > 1 else ds_base
        res_grid_lats = xarray.DataArray(np.asarray(ds_base["lat"].values)[::RES_GRID_STRIDE], dims=["lat"])
        res_grid_lons = xarray.DataArray(np.asarray(ds_base["lon"].values)[::RES_GRID_STRIDE], dims=["lon"])
        cache.ds_res_by_stride[stride] = ds_res
        cache.metric_grid_by_stride[stride] = (res_grid_lats, res_grid_lons)
    return cache.ds_res_by_stride[stride], *cache.metric_grid_by_stride[stride]


def _build_batch(ds_res: xarray.Dataset, idx_batch: list[int], task_cfg_kwargs: dict, max_lead_steps: int):
    lead_slice = slice(f"{HOURS_PER_STEP}h", f"{max_lead_steps * HOURS_PER_STEP}h")
    inputs_list, targets_list, forcings_list = [], [], []
    for target_idx in idx_batch:
        window = ds_res.isel(time=slice(target_idx - (N_INPUT_STEPS + max_lead_steps) + 1, target_idx + 1))
        if window.sizes["time"] < N_INPUT_STEPS + max_lead_steps:
            continue

        in_i, tgt_i, forc_i = data_utils.extract_inputs_targets_forcings(
            window, target_lead_times=lead_slice, **task_cfg_kwargs
        )
        for name in getattr(graphcast, "STATIC_VARS", ("geopotential_at_surface", "land_sea_mask")):
            if name in in_i and "time" in in_i[name].dims:
                in_i = in_i.assign({name: in_i[name].isel(time=0, drop=True)})

        inputs_list.append(in_i.isel(batch=0, drop=True))
        targets_list.append(tgt_i.isel(batch=0, drop=True))
        forcings_list.append(forc_i.isel(batch=0, drop=True))

    if not inputs_list:
        return None
    batch_size = len(inputs_list)
    inputs_b = xarray.concat(inputs_list, dim="batch").assign_coords(batch=np.arange(batch_size))
    targets_b = xarray.concat(targets_list, dim="batch").assign_coords(batch=np.arange(batch_size))
    forcings_b = xarray.concat(forcings_list, dim="batch").assign_coords(batch=np.arange(batch_size))
    return inputs_b, targets_b, forcings_b


def _build_warm_chunk_batch(
    ds_res: xarray.Dataset,
    chunk_start_indices: list[int],
    task_cfg_kwargs: dict,
    total_horizon_steps: int,
):
    lead_slice = slice(f"{HOURS_PER_STEP}h", f"{total_horizon_steps * HOURS_PER_STEP}h")
    inputs_list, targets_list, forcings_list = [], [], []
    for chunk_start in chunk_start_indices:
        window = ds_res.isel(time=slice(chunk_start, chunk_start + N_INPUT_STEPS + total_horizon_steps))
        if window.sizes["time"] < N_INPUT_STEPS + total_horizon_steps:
            continue

        in_i, tgt_i, forc_i = data_utils.extract_inputs_targets_forcings(
            window, target_lead_times=lead_slice, **task_cfg_kwargs
        )
        for name in getattr(graphcast, "STATIC_VARS", ("geopotential_at_surface", "land_sea_mask")):
            if name in in_i and "time" in in_i[name].dims:
                in_i = in_i.assign({name: in_i[name].isel(time=0, drop=True)})

        inputs_list.append(in_i.isel(batch=0, drop=True))
        targets_list.append(tgt_i.isel(batch=0, drop=True))
        forcings_list.append(forc_i.isel(batch=0, drop=True))

    if not inputs_list:
        return None
    batch_size = len(inputs_list)
    inputs_b = xarray.concat(inputs_list, dim="batch").assign_coords(batch=np.arange(batch_size))
    targets_b = xarray.concat(targets_list, dim="batch").assign_coords(batch=np.arange(batch_size))
    forcings_b = xarray.concat(forcings_list, dim="batch").assign_coords(batch=np.arange(batch_size))
    return inputs_b, targets_b, forcings_b


def _empty_metric_accumulator(max_lead_steps: int) -> dict[str, object]:
    return {
        "weighted_sum": np.zeros(max_lead_steps, dtype=float),
        "weighted_count": np.zeros(max_lead_steps, dtype=int),
        "per_variable_sum": {},
        "per_variable_count": {},
    }


def _accumulate_metrics(
    acc: dict[str, object],
    pred_b: xarray.Dataset,
    target_b: xarray.Dataset,
    *,
    res_grid_lats: xarray.DataArray,
    res_grid_lons: xarray.DataArray,
    stats: dict[str, xarray.Dataset],
) -> None:
    n_steps = pred_b.sizes.get("time", 0)
    if n_steps == 0:
        return

    weighted_sum = acc["weighted_sum"]
    weighted_count = acc["weighted_count"]
    per_variable_sum = acc["per_variable_sum"]
    per_variable_count = acc["per_variable_count"]

    for step_i in range(n_steps):
        pred_step = pred_b.isel(time=step_i)
        target_step = target_b.isel(time=step_i)
        grid_pred = pred_step.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")
        grid_target = target_step.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")

        weighted_loss_batch = normalized_weighted_mse_allvars(
            grid_pred,
            grid_target,
            per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
            use_latitude_weights=True,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        )
        weighted_values = np.asarray(weighted_loss_batch.values)
        weighted_sum[step_i] += float(weighted_values.sum())
        weighted_count[step_i] += int(weighted_values.size)

        per_var_losses = normalized_per_variable_mse(
            grid_pred,
            grid_target,
            use_latitude_weights=True,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        )
        for name, loss_batch in per_var_losses.items():
            values = np.asarray(loss_batch.values)
            per_variable_sum.setdefault(name, np.zeros_like(weighted_sum))
            per_variable_count.setdefault(name, np.zeros_like(weighted_count))
            per_variable_sum[name][step_i] += float(values.sum())
            per_variable_count[name][step_i] += int(values.size)


def _finalize_metrics(
    acc: dict[str, object],
    lead_days: list[int],
    lead_steps: list[int],
) -> tuple[dict[int, float], dict[str, dict[int, float]], dict[int, int]]:
    weighted_curve = np.divide(
        acc["weighted_sum"],
        acc["weighted_count"],
        out=np.full(len(acc["weighted_sum"]), np.nan),
        where=acc["weighted_count"] > 0,
    )

    weighted_by_day: dict[int, float] = {}
    n_by_day: dict[int, int] = {}
    for day, step in zip(lead_days, lead_steps):
        weighted_by_day[day] = float(weighted_curve[step - 1])
        n_by_day[day] = int(acc["weighted_count"][step - 1])

    per_variable_by_day: dict[str, dict[int, float]] = {}
    for name, sums in sorted(acc["per_variable_sum"].items()):
        counts = acc["per_variable_count"][name]
        curve = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts > 0)
        per_variable_by_day[name] = {day: float(curve[step - 1]) for day, step in zip(lead_days, lead_steps)}
    return weighted_by_day, per_variable_by_day, n_by_day


def _evaluate_checkpoint(
    ckpt_path: Path,
    stats: dict[str, xarray.Dataset],
    ds_base: xarray.Dataset,
    start_target_idx: int,
    n_steps: int,
    lead_days: list[int],
    lead_steps: list[int],
    seed_base: int,
    *,
    window_batch_size: int,
    target_indices: list[int],
    cache: EvalShardCache,
) -> tuple[dict[int, float], dict[str, dict[int, float]], dict[int, int]]:
    max_lead_steps = max(lead_steps)
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    run_jitted, task_cfg, model_cfg, _run_cfg = build_run_jitted(ckpt_obj, stats, ckpt_path)
    task_key, task_cfg_kwargs = _task_cfg_cache_key(task_cfg)
    cache.task_cfg_kwargs_by_key.setdefault(task_key, task_cfg_kwargs)

    stride = max(1, int(round(float(getattr(model_cfg, "resolution", 1.0)))))
    ds_res, res_grid_lats, res_grid_lons = _get_resolved_dataset(ds_base, stride, cache)
    n_batches = (len(target_indices) + window_batch_size - 1) // window_batch_size
    acc = _empty_metric_accumulator(max_lead_steps)

    for batch_i in range(n_batches):
        i0 = batch_i * window_batch_size
        i1 = min((batch_i + 1) * window_batch_size, len(target_indices))
        batch_key = (task_key, stride, max_lead_steps, batch_i)
        if batch_key in cache.cold_batches:
            batch = cache.cold_batches[batch_key]
            cache.cold_hits += 1
        else:
            batch = _build_batch(ds_res, target_indices[i0:i1], task_cfg_kwargs, max_lead_steps)
            cache.cold_batches[batch_key] = batch
            cache.cold_misses += 1
        if batch is None:
            continue
        inputs_b, targets_b, forcings_b = batch
        with suppress_graphcast_future_warnings():
            pred_b = run_jitted(
                rng=jax.random.PRNGKey(seed_base + batch_i),
                inputs=inputs_b,
                targets_template=targets_b * np.nan,
                forcings=forcings_b,
            )
        _accumulate_metrics(
            acc,
            pred_b.isel(time=slice(0, max_lead_steps)),
            targets_b.isel(time=slice(0, max_lead_steps)),
            res_grid_lats=res_grid_lats,
            res_grid_lons=res_grid_lons,
            stats=stats,
        )

    return _finalize_metrics(acc, lead_days, lead_steps)


def _evaluate_checkpoint_truth_anchored(
    ckpt_path: Path,
    stats: dict[str, xarray.Dataset],
    ds_base: xarray.Dataset,
    start_target_idx: int,
    n_steps: int,
    lead_days: list[int],
    lead_steps: list[int],
    seed_base: int,
    *,
    warmup_steps: int,
    trunk_steps: int,
    window_batch_size: int,
    cache: EvalShardCache,
) -> tuple[dict[int, float], dict[str, dict[int, float]], dict[int, int]]:
    max_lead_steps = max(lead_steps)
    total_horizon_steps = warmup_steps + trunk_steps + max_lead_steps
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    runner = build_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path)
    task_cfg = runner["task_cfg"]
    model_cfg = runner["model_cfg"]
    task_key, task_cfg_kwargs = _task_cfg_cache_key(task_cfg)
    cache.task_cfg_kwargs_by_key.setdefault(task_key, task_cfg_kwargs)

    stride = max(1, int(round(float(getattr(model_cfg, "resolution", 1.0)))))
    ds_res, res_grid_lats, res_grid_lons = _get_resolved_dataset(ds_base, stride, cache)
    chunk_start_indices = _chunk_start_indices(
        start_target_idx,
        n_steps,
        warmup_steps=warmup_steps,
        trunk_steps=trunk_steps,
        total_horizon_steps=total_horizon_steps,
    )
    n_batches = (len(chunk_start_indices) + window_batch_size - 1) // window_batch_size
    acc = _empty_metric_accumulator(max_lead_steps)

    for batch_i in range(n_batches):
        i0 = batch_i * window_batch_size
        i1 = min((batch_i + 1) * window_batch_size, len(chunk_start_indices))
        batch_key = (task_key, stride, total_horizon_steps, trunk_steps, batch_i)
        if batch_key in cache.warm_batches:
            batch = cache.warm_batches[batch_key]
            cache.warm_hits += 1
        else:
            batch = _build_warm_chunk_batch(ds_res, chunk_start_indices[i0:i1], task_cfg_kwargs, total_horizon_steps)
            cache.warm_batches[batch_key] = batch
            cache.warm_misses += 1
        if batch is None:
            continue
        inputs_b, targets_b, forcings_b = batch
        with suppress_graphcast_future_warnings():
            context = runner["initialize_context"](
                inputs=inputs_b,
                targets_template=targets_b * np.nan,
                forcings=forcings_b,
            )
            step_keys = jax.random.split(
                jax.random.PRNGKey(seed_base + batch_i),
                2 * warmup_steps + trunk_steps * (2 + max_lead_steps * 2),
            )
            key_i = 0
            for step_i in range(warmup_steps):
                _truth_pred, context = runner["truth_step"](
                    rng=(step_keys[key_i], step_keys[key_i + 1]),
                    context=context,
                    target_step=targets_b.isel(time=slice(step_i, step_i + 1)),
                    forcings_step=forcings_b.isel(time=slice(step_i, step_i + 1)),
                )
                key_i += 2

            for anchor_i in range(trunk_steps):
                branch_start = warmup_steps + anchor_i
                branch_targets = targets_b.isel(time=slice(branch_start, branch_start + max_lead_steps))
                branch_forcings = forcings_b.isel(time=slice(branch_start, branch_start + max_lead_steps))
                branch_pred = runner["branch_rollout"](
                    rng=step_keys[key_i],
                    context=context,
                    targets_template=branch_targets * np.nan,
                    forcings=branch_forcings,
                )
                _accumulate_metrics(
                    acc,
                    branch_pred,
                    branch_targets,
                    res_grid_lats=res_grid_lats,
                    res_grid_lons=res_grid_lons,
                    stats=stats,
                )
                key_i += 1
                _truth_pred, context = runner["truth_step"](
                    rng=(step_keys[key_i], step_keys[key_i + 1]),
                    context=context,
                    target_step=targets_b.isel(time=slice(branch_start, branch_start + 1)),
                    forcings_step=forcings_b.isel(time=slice(branch_start, branch_start + 1)),
                )
                key_i += 2

    return _finalize_metrics(acc, lead_days, lead_steps)


def _parse_res_from_path(ckpt_path: Path, run_cfg: dict) -> int | None:
    candidates = [ckpt_path.parent.name, ckpt_path.parent.parent.name]
    baseline_ckpt = run_cfg.get("residual_training", {}).get("baseline_checkpoint")
    if baseline_ckpt:
        candidates.append(str(baseline_ckpt))
    for candidate in candidates:
        match = re.search(r"(?:^|[_/])res(\d+)(?:[_.]|$)", candidate)
        if match:
            return int(match.group(1))
    return None


def _parse_di_from_name(run_name: str) -> int | None:
    for pattern in (r"_di(\d+)(?:_|$)", r"_dh(\d+)(?:_|$)"):
        match = re.search(pattern, run_name)
        if match:
            return int(match.group(1))
    return None


def _iter_checkpoint_paths(checkpoint_roots: list[Path] | None = None):
    if checkpoint_roots:
        for root in checkpoint_roots:
            resolved_root = root if root.is_absolute() else ROOT / root
            if not resolved_root.exists():
                continue
            yield from sorted(resolved_root.glob("**/ckpt_best.npz"))
        return
    yield from sorted((ROOT / "artifacts/checkpoints").glob("**/ckpt_best.npz"))


def discover_model_entries(
    families: list[str],
    resolutions: list[int] | None = None,
    checkpoint_roots: list[Path] | None = None,
) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    for ckpt_path in _iter_checkpoint_paths(checkpoint_roots):
        run_cfg = load_run_config(ckpt_path)
        family = infer_family(run_cfg)
        if family not in families:
            continue
        res = _parse_res_from_path(ckpt_path, run_cfg)
        if res is None:
            continue
        if resolutions is not None and res not in resolutions:
            continue
        run_name = ckpt_path.parent.name
        di = _parse_di_from_name(run_name)
        entries.append(
            ModelEntry(
                family=family,
                variant=run_name,
                model_type=family,
                di=di,
                res=res,
                ckpt_path=ckpt_path,
                run_name=run_name,
            )
        )
    entries.sort(key=lambda entry: (entry.family, entry.res, entry.variant))
    return entries


def discover_shards(
    families: list[str],
    resolutions: list[int] | None = None,
    checkpoint_roots: list[Path] | None = None,
) -> list[str]:
    shard_pairs = sorted(
        {(entry.family, entry.res) for entry in discover_model_entries(families, resolutions, checkpoint_roots)}
    )
    return [f"{family}:{res}" for family, res in shard_pairs]


def _append_metric_rows(
    rows: list[dict],
    entry: ModelEntry,
    lead_days: list[int],
    metrics: list[str],
    eval_mode: str,
    weighted_by_day: dict[int, float],
    per_variable_by_day: dict[str, dict[int, float]],
    n_by_day: dict[int, int],
    *,
    warmup_steps: int,
    trunk_steps: int,
) -> None:
    for day in lead_days:
        if "weighted_allvars" in metrics:
            rows.append(
                {
                    "family": entry.family,
                    "variant": entry.variant,
                    "model_type": entry.model_type,
                    "di": entry.di,
                    "res": entry.res,
                    "lead_days": day,
                    "eval_mode": eval_mode,
                    "metric_kind": "weighted_allvars",
                    "variable": "",
                    "value": weighted_by_day.get(day, np.nan),
                    "n_points": n_by_day.get(day, 0),
                    "run_name": entry.run_name,
                    "ckpt_path": str(entry.ckpt_path),
                    "warmup_steps": warmup_steps if entry.family == "residual_mamba" else np.nan,
                    "trunk_steps": trunk_steps if entry.family == "residual_mamba" else np.nan,
                }
            )
        if "per_variable" in metrics:
            for variable, by_day in sorted(per_variable_by_day.items()):
                rows.append(
                    {
                        "family": entry.family,
                        "variant": entry.variant,
                        "model_type": entry.model_type,
                        "di": entry.di,
                        "res": entry.res,
                        "lead_days": day,
                        "eval_mode": eval_mode,
                        "metric_kind": "per_variable",
                        "variable": variable,
                        "value": by_day.get(day, np.nan),
                        "n_points": n_by_day.get(day, 0),
                        "run_name": entry.run_name,
                        "ckpt_path": str(entry.ckpt_path),
                        "warmup_steps": warmup_steps if entry.family == "residual_mamba" else np.nan,
                        "trunk_steps": trunk_steps if entry.family == "residual_mamba" else np.nan,
                    }
                )


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.trunk_steps <= 0:
        raise ValueError("--trunk-steps must be positive.")

    if args.print_shards:
        for shard in discover_shards(args.families, args.resolutions, args.checkpoint_roots):
            print(shard)
        return

    entries = discover_model_entries(args.families, args.resolutions, args.checkpoint_roots)
    if not entries:
        raise RuntimeError("No checkpoints found for the requested family/resolution filters.")

    lead_steps = [int((24 * day) // HOURS_PER_STEP) for day in args.lead_days]
    ds_base, start_target_idx, n_steps = _prepare_dataset(max(lead_steps), args.n_eval_days)
    stats = _load_stats(ROOT / STATS_DIR)
    target_indices = _target_indices(start_target_idx, n_steps, max(lead_steps))
    cache = EvalShardCache(
        ds_res_by_stride={},
        metric_grid_by_stride={},
        task_cfg_kwargs_by_key={},
        cold_batches={},
        warm_batches={},
    )

    print(f"Evaluating {len(entries)} checkpoints across families={args.families} resolutions={args.resolutions}")
    rows: list[dict] = []
    for index, entry in enumerate(entries, start=1):
        print(f"\n[{index}/{len(entries)}] {entry.family} res={entry.res} {entry.run_name}")
        cold_weighted_by_day, cold_per_variable_by_day, cold_n_by_day = _evaluate_checkpoint(
            entry.ckpt_path,
            stats,
            ds_base,
            start_target_idx,
            n_steps,
            args.lead_days,
            lead_steps,
            seed_base=100000 * index,
            window_batch_size=args.window_batch_size,
            target_indices=target_indices,
            cache=cache,
        )
        if "cold" in args.eval_modes:
            _append_metric_rows(
                rows,
                entry,
                args.lead_days,
                args.metrics,
                "cold",
                cold_weighted_by_day,
                cold_per_variable_by_day,
                cold_n_by_day,
                warmup_steps=args.warmup_steps,
                trunk_steps=args.trunk_steps,
            )

        if "warm" in args.eval_modes:
            if entry.family == "residual_mamba":
                warm_weighted_by_day, warm_per_variable_by_day, warm_n_by_day = _evaluate_checkpoint_truth_anchored(
                    entry.ckpt_path,
                    stats,
                    ds_base,
                    start_target_idx,
                    n_steps,
                    args.lead_days,
                    lead_steps,
                    seed_base=200000 * index,
                    warmup_steps=args.warmup_steps,
                    trunk_steps=args.trunk_steps,
                    window_batch_size=args.window_batch_size,
                    cache=cache,
                )
            else:
                warm_weighted_by_day = {day: np.nan for day in args.lead_days}
                warm_per_variable_by_day = {
                    variable: {day: np.nan for day in args.lead_days}
                    for variable in cold_per_variable_by_day
                }
                warm_n_by_day = {day: 0 for day in args.lead_days}
            _append_metric_rows(
                rows,
                entry,
                args.lead_days,
                args.metrics,
                "warm",
                warm_weighted_by_day,
                warm_per_variable_by_day,
                warm_n_by_day,
                warmup_steps=args.warmup_steps,
                trunk_steps=args.trunk_steps,
            )

    df = pd.DataFrame(rows).sort_values(
        ["family", "variant", "res", "lead_days", "eval_mode", "metric_kind", "variable"]
    ).reset_index(drop=True)
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_data_dir / args.output_csv_name
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    print(
        "Cache summary: "
        f"cold_hits={cache.cold_hits} cold_misses={cache.cold_misses} "
        f"warm_hits={cache.warm_hits} warm_misses={cache.warm_misses} "
        f"stride_cache={len(cache.ds_res_by_stride)} task_cfg_cache={len(cache.task_cfg_kwargs_by_key)}"
    )


if __name__ == "__main__":
    main()
