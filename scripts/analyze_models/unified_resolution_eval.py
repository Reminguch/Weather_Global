#!/usr/bin/env python3
"""Unified resolution evaluation across GraphCast, GC-Mamba, and residual Mamba."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import re
import sys
import time
from collections.abc import Hashable, Iterable
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
from src.models.graphcast.evaluation.device_resolution_eval import (
    PreparedDeviceResolutionEvaluator,
    add_device_accumulator_to_host,
)
from src.models.graphcast.runtime import infer_family, load_run_config
from src.models.graphcast.training.core.prepared_array import PreparedArrayStore
from src.models.graphcast.training.core.prepared_block_batches import PreparedBlockBatchLoader
from src.models.graphcast.training.core.prepared_data import (
    PreparedDataError,
    load_prepared_metric_grid,
    open_prepared_store,
    prepared_eval_metadata,
    prepared_store_path_from_root,
    resolution_tag,
    select_prepared_eval_window,
)
from src.models.graphcast.training.core.segments import SegmentChunk


def _require_graphcast() -> None:
    try:
        import graphcast  # noqa: F401
    except Exception as exc:
        raise ImportError("graphcast is required. Activate env via scripts/graphcast_env.sh.") from exc


_require_graphcast()
from graphcast import checkpoint, data_utils, graphcast

from scripts.analyze_models.legacy.analysis_metrics import (
    GRAPHCAST_PER_VARIABLE_WEIGHTS,
    latitude_weights_with_fallback,
    normalized_per_variable_mse,
    normalized_weighted_mse_allvars,
)
from scripts.analyze_models.legacy.graphcast_analysis_utils import (
    build_run_jitted,
    build_truth_anchored_runner,
    suppress_graphcast_future_warnings,
)
from src.models.mamba.gc_mamba.legacy_runtime import is_legacy_gc_mamba_checkpoint
from src.models.mamba.residual_mamba.feedback import RESIDUAL_AR_FEEDBACK, RESIDUAL_AR_FEEDBACK_CHOICES
from src.models.mamba.residual_mamba.runtime import (
    build_run_jitted as build_residual_run_jitted,
    build_training_equivalent_run_jitted as build_residual_training_equivalent_run_jitted,
    build_training_equivalent_truth_anchored_residual_runner,
    build_truth_anchored_residual_runner,
)


DATASET_PATH = "data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc"
STATS_DIR = "data/graphcast/graphcast/stats"
DEFAULT_OUTPUT_DATA_DIR = "plots/analyze_models/data/resolution_eval"
DEFAULT_OUTPUT_IMAGE_DIR = "plots/analyze_models/images/resolution_eval"
DEFAULT_PREPARED_DATA_ROOT = "data/graphcast/graphcast/dataset/prepared_stream"

FAMILIES = ["graphcast", "gc_mamba", "legacy_gc_mamba", "residual_mamba"]
METRICS = ["weighted_allvars", "per_variable", "rmse_k"]
DEFAULT_METRICS = ["rmse_k"]
EVAL_MODES = ["cold", "warm"]
DATA_SOURCES = ["prepared_array", "raw"]
LEAD_DAYS = [1, 2, 4]
DEFAULT_RESOLUTIONS = [1, 2, 3, 4, 6, 9, 18]
N_EVAL_DAYS = 365
HOURS_PER_STEP = 6
N_INPUT_STEPS = 2
N_EXTRA_STEPS = 14
WINDOW_BATCH_SIZE = 8
WARMUP_STEPS = 24
TRUNK_STEPS = 32
PREPARED_STREAM_BLOCK_STEPS = 32
RES_GRID_STRIDE = 18
CSV_SORT_COLUMNS = ["family", "variant", "res", "lead_steps", "lead_days", "eval_mode", "metric_kind", "variable"]
RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED = "teacher_forced_training_equivalent"
RESIDUAL_EVAL_SEMANTICS_ROLLOUT = "rollout"
RESIDUAL_EVAL_SEMANTICS = RESIDUAL_EVAL_SEMANTICS_ROLLOUT
RESIDUAL_EVAL_SEMANTICS_CHOICES = [RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED, RESIDUAL_EVAL_SEMANTICS_ROLLOUT]
METRIC_SEMANTICS = "graphcast_weighted_mse_per_level_projected_grid"
PHYSICAL_RMSE_METRIC_SEMANTICS = "physical_rmse_projected_grid_no_std"
LEAD_AGGREGATION_ENDPOINT = "endpoint"
LEAD_AGGREGATION_MEAN_TO_LEAD = "mean_to_lead"
LEAD_AGGREGATION_CHOICES = [LEAD_AGGREGATION_ENDPOINT, LEAD_AGGREGATION_MEAN_TO_LEAD]
NYC_LAT = 40.7
NYC_LON = 286.0
NYC_POINT_VARIABLE = "2m_temperature_nyc"


@dataclasses.dataclass(frozen=True)
class ModelEntry:
    family: str
    variant: str
    model_type: str
    di: int | None
    res: float
    ckpt_path: Path
    run_name: str


@dataclasses.dataclass
class EvalShardCache:
    ds_res_by_stride: dict[int, xarray.Dataset]
    metric_grid_by_stride: dict[int, tuple[xarray.DataArray, xarray.DataArray]]
    task_cfg_kwargs_by_key: dict[Hashable, dict]
    cold_batches: dict[tuple, tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset] | None]
    warm_batches: dict[tuple, tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset] | None]
    cold_hits: int = 0
    cold_misses: int = 0
    warm_hits: int = 0
    warm_misses: int = 0
    cold_cache_mode: str = "enabled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified resolution evaluation across model families.")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--resolutions", type=float, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="prepared_array")
    parser.add_argument("--prepared-data-root", type=Path, default=ROOT / DEFAULT_PREPARED_DATA_ROOT)
    parser.add_argument("--stats-dir", type=Path, default=ROOT / STATS_DIR)
    parser.add_argument("--eval-start", type=str, default=None, help="Prepared-array eval start timestamp, inclusive.")
    parser.add_argument("--eval-end", type=str, default=None, help="Prepared-array eval end timestamp, exclusive.")
    parser.add_argument("--eval-year", type=int, default=None, help="Prepared-array eval year override.")
    parser.add_argument(
        "--checkpoint-roots",
        type=Path,
        nargs="+",
        default=None,
        help="Optional checkpoint root directories to search instead of scanning all artifacts/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        type=Path,
        nargs="+",
        default=None,
        help="Optional exact checkpoint/params files to evaluate in addition to any discovered checkpoint roots.",
    )
    parser.add_argument("--lead-days", type=int, nargs="+", default=LEAD_DAYS)
    parser.add_argument(
        "--lead-steps",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate explicit autoregressive lead steps. Overrides --lead-days when set.",
    )
    parser.add_argument(
        "--lead-aggregation",
        choices=LEAD_AGGREGATION_CHOICES,
        default=LEAD_AGGREGATION_ENDPOINT,
        help=(
            "How to reduce per-step rollout metrics at requested leads. "
            "'endpoint' reports the loss exactly at k; 'mean_to_lead' reports "
            "the mean loss over rollout steps 1..k."
        ),
    )
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=DEFAULT_METRICS)
    parser.add_argument(
        "--metric-variables",
        nargs="+",
        default=None,
        help=(
            "Optional variable allow-list for per-variable/rmse metric accumulation. "
            "Use '2m_temperature' to score only global 2m-temperature rows."
        ),
    )
    parser.add_argument("--eval-modes", nargs="+", choices=EVAL_MODES, default=EVAL_MODES)
    parser.add_argument(
        "--residual-eval-semantics",
        choices=RESIDUAL_EVAL_SEMANTICS_CHOICES,
        default=RESIDUAL_EVAL_SEMANTICS,
        help=(
            "Residual Mamba eval behavior. The default 'rollout' is the proper AR forecast "
            "eval: warm context is truth-fed, then scored branches feed predictions back. "
            "'teacher_forced_training_equivalent' is a diagnostic/training-equivalence mode "
            "and should not be compared as an error-vs-lead rollout."
        ),
    )
    parser.add_argument(
        "--residual-ar-feedback",
        choices=RESIDUAL_AR_FEEDBACK_CHOICES,
        default=RESIDUAL_AR_FEEDBACK,
        help=(
            "Physical autoregressive feedback for residual Mamba rollout eval. "
            "'baseline_plus_residual' feeds the full corrected forecast back; "
            "'baseline' scores baseline+residual output but feeds only the frozen "
            "baseline forecast into the next physical input step. No effect for "
            "teacher-forced residual eval semantics."
        ),
    )
    parser.add_argument("--n-eval-days", type=int, default=N_EVAL_DAYS)
    parser.add_argument("--window-batch-size", type=int, default=WINDOW_BATCH_SIZE)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--trunk-steps", type=int, default=TRUNK_STEPS)
    parser.add_argument("--prepared-load-workers", type=int, default=1)
    parser.add_argument("--prepared-stream-block-steps", type=int, default=PREPARED_STREAM_BLOCK_STEPS)
    parser.add_argument(
        "--metric-grid-resolution",
        type=float,
        default=RES_GRID_STRIDE,
        help=(
            "Prepared-store resolution used for metric sampling. "
            f"Default {RES_GRID_STRIDE} preserves common-grid resolution eval; "
            "use 2 to score on the native res2 grid."
        ),
    )
    parser.add_argument(
        "--disable-prepared-device-eval",
        action="store_true",
        help="Use the streamed host/xarray prepared eval path instead of jitted device-side metric accumulation.",
    )
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


def _warn_skip(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _print_progress(label: str, current: int, total: int, start_time: float) -> None:
    elapsed = time.time() - start_time
    frac = float(current) / float(total) if total else 0.0
    rate = float(current) / elapsed if elapsed > 0 else 0.0
    eta = (float(total - current) / rate) if rate > 0 else float("nan")
    eta_text = "unknown" if not np.isfinite(eta) else f"{eta:.1f}s"
    print(
        f"[{label}] {current}/{total} ({100.0 * frac:.1f}%) "
        f"elapsed {elapsed:.1f}s eta {eta_text}",
        flush=True,
    )


def _prepared_target_indices(n_steps: int, max_lead_steps: int) -> list[int]:
    first_anchor = N_INPUT_STEPS - 1
    stop = n_steps - max_lead_steps
    if stop <= first_anchor:
        raise PreparedDataError(
            f"Prepared eval split has {n_steps} steps; need at least {N_INPUT_STEPS + max_lead_steps}."
        )
    return np.arange(first_anchor, stop, dtype=int).tolist()


def _prepared_chunk_start_indices(
    n_steps: int,
    *,
    trunk_steps: int,
    total_horizon_steps: int,
) -> list[int]:
    stop = n_steps - (N_INPUT_STEPS + total_horizon_steps) + 1
    if stop <= 0:
        raise PreparedDataError(
            f"Prepared eval split has {n_steps} steps; need at least {N_INPUT_STEPS + total_horizon_steps}."
        )
    return np.arange(0, stop, trunk_steps, dtype=int).tolist()


def _prepared_stream_segments(indices: list[int], block_steps: int) -> list[np.ndarray]:
    if block_steps <= 0:
        raise ValueError("--prepared-stream-block-steps must be positive.")
    return [
        np.asarray(indices[start : start + block_steps], dtype=np.int64)
        for start in range(0, len(indices), block_steps)
        if indices[start : start + block_steps]
    ]


def _iter_prepared_stream_chunks(
    segments: list[np.ndarray],
    *,
    batch_size: int,
) -> Iterable[SegmentChunk]:
    if batch_size <= 0:
        raise ValueError("--window-batch-size must be positive.")
    for start in range(0, len(segments), batch_size):
        segment_group = segments[start : start + batch_size]
        if not segment_group:
            continue
        max_len = max(len(segment) for segment in segment_group)
        chunk_indices: list[np.ndarray] = []
        lane_offsets: list[np.ndarray] = []
        for offset in range(max_len):
            step_indices = []
            step_offsets = []
            for segment in segment_group:
                if offset >= len(segment):
                    continue
                step_indices.append(int(segment[offset]))
                step_offsets.append(offset)
            if step_indices:
                chunk_indices.append(np.asarray(step_indices, dtype=np.int64))
                lane_offsets.append(np.asarray(step_offsets, dtype=np.int64))
        if not chunk_indices:
            continue
        lane_count = len(chunk_indices[0])
        yield SegmentChunk(
            chunk_indices=tuple(chunk_indices),
            reset_mask=np.zeros(lane_count, dtype=np.bool_),
            lane_segment_ids=np.arange(start, start + lane_count, dtype=np.int64),
            lane_offsets=lane_offsets[0],
            epoch=0,
        )


@contextlib.contextmanager
def _prepared_load_executor(max_workers: int):
    if max_workers <= 1:
        yield None
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield executor


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


def _is_residual_mamba_run(run_cfg: dict) -> bool:
    return infer_family(run_cfg) == "residual_mamba"


def _add_residual_eval_semantics(
    metadata: dict[str, object],
    run_cfg: dict,
    residual_eval_semantics: str,
    residual_ar_feedback: str,
) -> dict[str, object]:
    if not _is_residual_mamba_run(run_cfg):
        return metadata
    out = dict(metadata)
    out["residual_eval_semantics"] = residual_eval_semantics
    out["residual_ar_feedback"] = residual_ar_feedback
    return out


def _task_cfg_cache_key(task_cfg) -> tuple[Hashable, dict]:
    task_cfg_kwargs = dataclasses.asdict(task_cfg)
    return _freeze_cache_key(task_cfg_kwargs), task_cfg_kwargs


def _lead_days_from_steps(lead_steps: list[int]) -> list[float]:
    return [float(step * HOURS_PER_STEP) / 24.0 for step in lead_steps]


def _format_resolution_value(resolution: float) -> str:
    value = float(resolution)
    if np.isclose(value, round(value), atol=1e-6):
        return str(int(round(value)))
    return f"{value:g}"


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
        "physical_mse_sum": {},
        "physical_mse_count": {},
    }


def _accumulate_metrics(
    acc: dict[str, object],
    pred_b: xarray.Dataset,
    target_b: xarray.Dataset,
    *,
    res_grid_lats: xarray.DataArray,
    res_grid_lons: xarray.DataArray,
    stats: dict[str, xarray.Dataset],
    metric_variables: set[str] | None = None,
) -> None:
    n_steps = pred_b.sizes.get("time", 0)
    if n_steps == 0:
        return

    weighted_sum = acc["weighted_sum"]
    weighted_count = acc["weighted_count"]
    per_variable_sum = acc["per_variable_sum"]
    per_variable_count = acc["per_variable_count"]
    physical_mse_sum = acc["physical_mse_sum"]
    physical_mse_count = acc["physical_mse_count"]

    for step_i in range(n_steps):
        pred_step = pred_b.isel(time=step_i)
        target_step = target_b.isel(time=step_i)
        grid_pred = pred_step.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")
        grid_target = target_step.sel(lat=res_grid_lats, lon=res_grid_lons, method="nearest")
        if metric_variables is not None:
            keep = [name for name in metric_variables if name in grid_pred and name in grid_target]
            if not keep:
                continue
            grid_pred = grid_pred[keep]
            grid_target = grid_target[keep]

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

        include_2m = metric_variables is None or "2m_temperature" in metric_variables
        if include_2m and "2m_temperature" in grid_pred and "2m_temperature" in grid_target:
            physical_loss = (grid_pred["2m_temperature"] - grid_target["2m_temperature"]) ** 2
            if "lat" in physical_loss.dims:
                lat_w = latitude_weights_with_fallback(grid_target["2m_temperature"]).astype(physical_loss.dtype)
                physical_loss = physical_loss * lat_w
            reduce_dims = [dim for dim in physical_loss.dims if dim not in ("batch",)]
            if reduce_dims:
                physical_loss = physical_loss.mean(reduce_dims, skipna=False)
            values = np.asarray(physical_loss.values)
            physical_mse_sum.setdefault("2m_temperature", np.zeros_like(weighted_sum))
            physical_mse_count.setdefault("2m_temperature", np.zeros_like(weighted_count))
            physical_mse_sum["2m_temperature"][step_i] += float(values.sum())
            physical_mse_count["2m_temperature"][step_i] += int(values.size)

        include_nyc = metric_variables is None or NYC_POINT_VARIABLE in metric_variables
        if include_nyc and "2m_temperature" in pred_step and "2m_temperature" in target_step:
            metric_lat = float(res_grid_lats.values[np.abs(np.asarray(res_grid_lats.values) - NYC_LAT).argmin()])
            metric_lon = float(res_grid_lons.values[np.abs(np.asarray(res_grid_lons.values) - NYC_LON).argmin()])
            point_pred = pred_step["2m_temperature"].sel(lat=metric_lat, lon=metric_lon, method="nearest")
            point_target = target_step["2m_temperature"].sel(lat=metric_lat, lon=metric_lon, method="nearest")
            physical_point_loss = (point_pred - point_target) ** 2
            reduce_dims = [dim for dim in physical_point_loss.dims if dim not in ("batch",)]
            if reduce_dims:
                physical_point_loss = physical_point_loss.mean(reduce_dims, skipna=False)
            values = np.asarray(physical_point_loss.values)
            physical_mse_sum.setdefault(NYC_POINT_VARIABLE, np.zeros_like(weighted_sum))
            physical_mse_count.setdefault(NYC_POINT_VARIABLE, np.zeros_like(weighted_count))
            physical_mse_sum[NYC_POINT_VARIABLE][step_i] += float(values.sum())
            physical_mse_count[NYC_POINT_VARIABLE][step_i] += int(values.size)

            point_loss = (point_pred - point_target) ** 2
            if "2m_temperature" in stats["diffs_stddev_by_level"]:
                scale = stats["diffs_stddev_by_level"]["2m_temperature"]
                if "lat" in scale.dims and "lon" in scale.dims:
                    scale = scale.sel(lat=metric_lat, lon=metric_lon, method="nearest")
                point_loss = point_loss / (scale.astype(point_loss.dtype) ** 2)
            reduce_dims = [dim for dim in point_loss.dims if dim not in ("batch",)]
            if reduce_dims:
                point_loss = point_loss.mean(reduce_dims, skipna=False)
            values = np.asarray(point_loss.values)
            per_variable_sum.setdefault(NYC_POINT_VARIABLE, np.zeros_like(weighted_sum))
            per_variable_count.setdefault(NYC_POINT_VARIABLE, np.zeros_like(weighted_count))
            per_variable_sum[NYC_POINT_VARIABLE][step_i] += float(values.sum())
            per_variable_count[NYC_POINT_VARIABLE][step_i] += int(values.size)


def _finalize_metrics(
    acc: dict[str, object],
    lead_days: list[float],
    lead_steps: list[int],
    *,
    lead_aggregation: str = LEAD_AGGREGATION_ENDPOINT,
) -> tuple[
    dict[float, float],
    dict[str, dict[float, float]],
    dict[str, dict[float, float]],
    dict[float, int],
]:
    if lead_aggregation == LEAD_AGGREGATION_ENDPOINT:
        weighted_sum = acc["weighted_sum"]
        weighted_count = acc["weighted_count"]
    elif lead_aggregation == LEAD_AGGREGATION_MEAN_TO_LEAD:
        weighted_sum = np.cumsum(acc["weighted_sum"])
        weighted_count = np.cumsum(acc["weighted_count"])
    else:
        raise ValueError(f"Unknown lead aggregation: {lead_aggregation!r}")

    weighted_curve = np.divide(
        weighted_sum,
        weighted_count,
        out=np.full(len(weighted_sum), np.nan),
        where=weighted_count > 0,
    )

    weighted_by_day: dict[float, float] = {}
    n_by_day: dict[float, int] = {}
    for day, step in zip(lead_days, lead_steps):
        weighted_by_day[day] = float(weighted_curve[step - 1])
        n_by_day[day] = int(weighted_count[step - 1])

    per_variable_by_day: dict[str, dict[float, float]] = {}
    for name, sums in sorted(acc["per_variable_sum"].items()):
        counts = acc["per_variable_count"][name]
        if lead_aggregation == LEAD_AGGREGATION_MEAN_TO_LEAD:
            sums = np.cumsum(sums)
            counts = np.cumsum(counts)
        curve = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts > 0)
        per_variable_by_day[name] = {day: float(curve[step - 1]) for day, step in zip(lead_days, lead_steps)}

    rmse_by_day: dict[str, dict[float, float]] = {}
    for name, sums in sorted(acc["physical_mse_sum"].items()):
        counts = acc["physical_mse_count"][name]
        if lead_aggregation == LEAD_AGGREGATION_MEAN_TO_LEAD:
            sums = np.cumsum(sums)
            counts = np.cumsum(counts)
        mse_curve = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts > 0)
        rmse_curve = np.sqrt(mse_curve)
        rmse_by_day[name] = {day: float(rmse_curve[step - 1]) for day, step in zip(lead_days, lead_steps)}
    return weighted_by_day, per_variable_by_day, rmse_by_day, n_by_day


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
    residual_eval_semantics: str,
    residual_ar_feedback: str,
    lead_aggregation: str,
    metric_variables: set[str] | None,
    cache: EvalShardCache,
) -> tuple[
    dict[float, float],
    dict[str, dict[float, float]],
    dict[str, dict[float, float]],
    dict[float, int],
]:
    max_lead_steps = max(lead_steps)
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    run_cfg = load_run_config(ckpt_path)
    is_residual = _is_residual_mamba_run(run_cfg)
    if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED:
        run_jitted, task_cfg, model_cfg, _run_cfg = build_residual_training_equivalent_run_jitted(
            ckpt_obj, stats, ckpt_path
        )
    elif is_residual:
        run_jitted, task_cfg, model_cfg, _run_cfg = build_residual_run_jitted(
            ckpt_obj,
            stats,
            ckpt_path,
            residual_ar_feedback=residual_ar_feedback,
        )
    else:
        run_jitted, task_cfg, model_cfg, _run_cfg = build_run_jitted(ckpt_obj, stats, ckpt_path)
    task_key, task_cfg_kwargs = _task_cfg_cache_key(task_cfg)
    cache.task_cfg_kwargs_by_key.setdefault(task_key, task_cfg_kwargs)

    stride = max(1, int(round(float(getattr(model_cfg, "resolution", 1.0)))))
    ds_res, res_grid_lats, res_grid_lons = _get_resolved_dataset(ds_base, stride, cache)
    n_batches = (len(target_indices) + window_batch_size - 1) // window_batch_size
    acc = _empty_metric_accumulator(max_lead_steps)
    use_cold_batch_cache = stride > 1
    if not use_cold_batch_cache:
        cache.cold_cache_mode = "bypass_res1"

    for batch_i in range(n_batches):
        i0 = batch_i * window_batch_size
        i1 = min((batch_i + 1) * window_batch_size, len(target_indices))
        batch_key = (task_key, stride, max_lead_steps, batch_i)
        if use_cold_batch_cache and batch_key in cache.cold_batches:
            batch = cache.cold_batches[batch_key]
            cache.cold_hits += 1
        else:
            batch = _build_batch(ds_res, target_indices[i0:i1], task_cfg_kwargs, max_lead_steps)
            if use_cold_batch_cache:
                cache.cold_batches[batch_key] = batch
                cache.cold_misses += 1
        if batch is None:
            continue
        inputs_b, targets_b, forcings_b = batch
        with suppress_graphcast_future_warnings():
            pred_b = run_jitted(
                rng=jax.random.PRNGKey(seed_base + batch_i),
                inputs=inputs_b,
                targets_template=(
                    targets_b
                    if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
                    else targets_b * np.nan
                ),
                forcings=forcings_b,
            )
        _accumulate_metrics(
            acc,
            pred_b.isel(time=slice(0, max_lead_steps)),
            targets_b.isel(time=slice(0, max_lead_steps)),
            res_grid_lats=res_grid_lats,
            res_grid_lons=res_grid_lons,
            stats=stats,
            metric_variables=metric_variables,
        )

    return _finalize_metrics(acc, lead_days, lead_steps, lead_aggregation=lead_aggregation)


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
    residual_eval_semantics: str,
    residual_ar_feedback: str,
    lead_aggregation: str,
    metric_variables: set[str] | None,
    cache: EvalShardCache,
) -> tuple[
    dict[float, float],
    dict[str, dict[float, float]],
    dict[str, dict[float, float]],
    dict[float, int],
]:
    max_lead_steps = max(lead_steps)
    total_horizon_steps = warmup_steps + trunk_steps + max_lead_steps
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    run_cfg = load_run_config(ckpt_path)
    is_residual = _is_residual_mamba_run(run_cfg)
    runner = (
        (
            build_training_equivalent_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path)
            if residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
            else build_truth_anchored_residual_runner(
                ckpt_obj,
                stats,
                ckpt_path,
                residual_ar_feedback=residual_ar_feedback,
            )
        )
        if is_residual
        else build_truth_anchored_runner(ckpt_obj, stats, ckpt_path)
    )
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
                targets_template=(
                    targets_b
                    if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
                    else targets_b * np.nan
                ),
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
                    targets_template=(
                        branch_targets
                        if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
                        else branch_targets * np.nan
                    ),
                    forcings=branch_forcings,
                )
                _accumulate_metrics(
                    acc,
                    branch_pred,
                    branch_targets,
                    res_grid_lats=res_grid_lats,
                    res_grid_lons=res_grid_lons,
                    stats=stats,
                    metric_variables=metric_variables,
                )
                key_i += 1
                _truth_pred, context = runner["truth_step"](
                    rng=(step_keys[key_i], step_keys[key_i + 1]),
                    context=context,
                    target_step=targets_b.isel(time=slice(branch_start, branch_start + 1)),
                    forcings_step=forcings_b.isel(time=slice(branch_start, branch_start + 1)),
                )
                key_i += 2

    return _finalize_metrics(acc, lead_days, lead_steps, lead_aggregation=lead_aggregation)


def _open_prepared_eval_store_for_checkpoint(
    prepared_data_root: Path,
    *,
    resolution: float,
    task_cfg,
    run_cfg: dict,
    eval_start: str | None,
    eval_end: str | None,
    eval_year: int | None,
) -> tuple[PreparedArrayStore, dict[str, object]]:
    store_path = prepared_store_path_from_root(prepared_data_root, resolution)
    store = open_prepared_store(
        prepared_data_root,
        resolution,
        task_cfg,
        label=f"prepared-{resolution_tag(resolution)}",
    )
    selected_year = eval_year
    if eval_start is None and eval_end is None and selected_year is None:
        raw_year = run_cfg.get("val_year")
        selected_year = None if raw_year is None else int(raw_year)

    selected_window = select_prepared_eval_window(
        store,
        eval_start=eval_start,
        eval_end=eval_end,
        eval_year=selected_year,
    )
    return selected_window.store, prepared_eval_metadata(store_path, selected_window)


def _evaluate_checkpoint_prepared(
    ckpt_path: Path,
    stats: dict[str, xarray.Dataset],
    prepared_data_root: Path,
    entry_res: float,
    lead_days: list[int],
    lead_steps: list[int],
    seed_base: int,
    *,
    window_batch_size: int,
    prepared_load_workers: int,
    prepared_stream_block_steps: int,
    use_device_eval: bool,
    cache: EvalShardCache,
    res_grid_lats: xarray.DataArray,
    res_grid_lons: xarray.DataArray,
    eval_start: str | None,
    eval_end: str | None,
    eval_year: int | None,
    residual_eval_semantics: str,
    residual_ar_feedback: str,
    lead_aggregation: str,
    metric_variables: set[str] | None,
) -> tuple[
    dict[float, float],
    dict[str, dict[float, float]],
    dict[str, dict[float, float]],
    dict[float, int],
    dict[str, object],
]:
    max_lead_steps = max(lead_steps)
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    ckpt_run_cfg = load_run_config(ckpt_path)
    is_residual = _is_residual_mamba_run(ckpt_run_cfg)
    device_evaluator = None
    if use_device_eval:
        device_evaluator = PreparedDeviceResolutionEvaluator(
            ckpt_obj,
            stats,
            ckpt_path,
            res_grid_lats=res_grid_lats,
            res_grid_lons=res_grid_lons,
            per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
            max_lead_steps=max_lead_steps,
            metric_variables=None if metric_variables is None else tuple(sorted(metric_variables)),
            nyc_lat=NYC_LAT if metric_variables is None or NYC_POINT_VARIABLE in metric_variables else None,
            nyc_lon=NYC_LON if metric_variables is None or NYC_POINT_VARIABLE in metric_variables else None,
            nyc_output_name=NYC_POINT_VARIABLE if metric_variables is None or NYC_POINT_VARIABLE in metric_variables else None,
            residual_eval_semantics=residual_eval_semantics,
            residual_ar_feedback=residual_ar_feedback,
        )
        task_cfg = device_evaluator.task_cfg
        run_cfg = device_evaluator.run_cfg
    else:
        if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED:
            run_jitted, task_cfg, _model_cfg, run_cfg = build_residual_training_equivalent_run_jitted(
                ckpt_obj, stats, ckpt_path
            )
        elif is_residual:
            run_jitted, task_cfg, _model_cfg, run_cfg = build_residual_run_jitted(
                ckpt_obj,
                stats,
                ckpt_path,
                residual_ar_feedback=residual_ar_feedback,
            )
        else:
            run_jitted, task_cfg, _model_cfg, run_cfg = build_run_jitted(ckpt_obj, stats, ckpt_path)
    task_key, task_cfg_kwargs = _task_cfg_cache_key(task_cfg)
    cache.task_cfg_kwargs_by_key.setdefault(task_key, task_cfg_kwargs)

    store, eval_metadata = _open_prepared_eval_store_for_checkpoint(
        prepared_data_root,
        resolution=entry_res,
        task_cfg=task_cfg,
        run_cfg=run_cfg,
        eval_start=eval_start,
        eval_end=eval_end,
        eval_year=eval_year,
    )
    eval_metadata = _add_residual_eval_semantics(
        eval_metadata,
        run_cfg,
        residual_eval_semantics,
        residual_ar_feedback,
    )
    eval_metadata["lead_aggregation"] = lead_aggregation
    target_indices = _prepared_target_indices(store.sizes["time"], max_lead_steps)
    acc = _empty_metric_accumulator(max_lead_steps)

    segments = _prepared_stream_segments(target_indices, prepared_stream_block_steps)
    stream_chunks = list(_iter_prepared_stream_chunks(segments, batch_size=window_batch_size))
    total_batches = sum(len(chunk.chunk_indices) for chunk in stream_chunks)
    progress_label = f"prepared-cold-res{entry_res}:{ckpt_path.parent.name}"
    print(
        f"[{progress_label}] start anchors={len(target_indices)} "
        f"stream_chunks={len(stream_chunks)} batches={total_batches} "
        f"device_eval={use_device_eval}",
        flush=True,
    )
    if is_residual:
        print(f"[{progress_label}] residual_eval_semantics={residual_eval_semantics}", flush=True)
        print(f"[{progress_label}] residual_ar_feedback={residual_ar_feedback}", flush=True)
    progress_t0 = time.time()
    next_progress = 1
    with _prepared_load_executor(prepared_load_workers) as load_executor:
        loader = PreparedBlockBatchLoader(
            store,
            segments,
            input_steps=N_INPUT_STEPS,
            target_steps=max_lead_steps,
            task_cfg=task_cfg,
            dt=pd.Timedelta(hours=HOURS_PER_STEP),
            load_executor=load_executor,
            max_workers=prepared_load_workers,
            label=f"prepared-cold-res{entry_res}",
        )
        batch_i = 0
        for chunk_i, chunk in enumerate(stream_chunks, start=1):
            for batch, _load_stats in loader.iter_chunk_batches(chunk):
                inputs_b, targets_b, forcings_b = batch
                if device_evaluator is not None:
                    device_acc = device_evaluator.evaluate_cold_batch(
                        jax.random.PRNGKey(seed_base + batch_i),
                        inputs_b,
                        targets_b,
                        forcings_b,
                    )
                    add_device_accumulator_to_host(
                        acc,
                        device_acc,
                        variable_names=device_evaluator.metric_variable_names,
                    )
                else:
                    with suppress_graphcast_future_warnings():
                        pred_b = run_jitted(
                            rng=jax.random.PRNGKey(seed_base + batch_i),
                            inputs=inputs_b,
                            targets_template=(
                                targets_b
                                if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
                                else targets_b * np.nan
                            ),
                            forcings=forcings_b,
                        )
                    _accumulate_metrics(
                        acc,
                        pred_b.isel(time=slice(0, max_lead_steps)),
                        targets_b.isel(time=slice(0, max_lead_steps)),
                        res_grid_lats=res_grid_lats,
                        res_grid_lons=res_grid_lons,
                        stats=stats,
                        metric_variables=metric_variables,
                    )
                batch_i += 1
                if (
                    batch_i == next_progress
                    or batch_i == total_batches
                    or batch_i % 25 == 0
                ):
                    _print_progress(progress_label, batch_i, total_batches, progress_t0)
                    while next_progress <= batch_i:
                        next_progress *= 2
            if chunk_i == 1 or chunk_i == len(stream_chunks) or chunk_i % 10 == 0:
                print(
                    f"[{progress_label}] stream_chunk {chunk_i}/{len(stream_chunks)} "
                    f"batches_done={batch_i}/{total_batches}",
                    flush=True,
                )

    weighted_by_day, per_variable_by_day, rmse_by_day, n_by_day = _finalize_metrics(
        acc,
        lead_days,
        lead_steps,
        lead_aggregation=lead_aggregation,
    )
    return weighted_by_day, per_variable_by_day, rmse_by_day, n_by_day, eval_metadata


def _evaluate_checkpoint_truth_anchored_prepared(
    ckpt_path: Path,
    stats: dict[str, xarray.Dataset],
    prepared_data_root: Path,
    entry_res: float,
    lead_days: list[int],
    lead_steps: list[int],
    seed_base: int,
    *,
    warmup_steps: int,
    trunk_steps: int,
    window_batch_size: int,
    prepared_load_workers: int,
    use_device_eval: bool,
    cache: EvalShardCache,
    res_grid_lats: xarray.DataArray,
    res_grid_lons: xarray.DataArray,
    eval_start: str | None,
    eval_end: str | None,
    eval_year: int | None,
    residual_eval_semantics: str,
    residual_ar_feedback: str,
    lead_aggregation: str,
    metric_variables: set[str] | None,
) -> tuple[
    dict[float, float],
    dict[str, dict[float, float]],
    dict[str, dict[float, float]],
    dict[float, int],
    dict[str, object],
]:
    max_lead_steps = max(lead_steps)
    total_horizon_steps = warmup_steps + trunk_steps + max_lead_steps
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    ckpt_run_cfg = load_run_config(ckpt_path)
    is_residual = _is_residual_mamba_run(ckpt_run_cfg)
    device_evaluator = None
    if use_device_eval:
        device_evaluator = PreparedDeviceResolutionEvaluator(
            ckpt_obj,
            stats,
            ckpt_path,
            res_grid_lats=res_grid_lats,
            res_grid_lons=res_grid_lons,
            per_variable_weights=GRAPHCAST_PER_VARIABLE_WEIGHTS,
            max_lead_steps=max_lead_steps,
            metric_variables=None if metric_variables is None else tuple(sorted(metric_variables)),
            nyc_lat=NYC_LAT if metric_variables is None or NYC_POINT_VARIABLE in metric_variables else None,
            nyc_lon=NYC_LON if metric_variables is None or NYC_POINT_VARIABLE in metric_variables else None,
            nyc_output_name=NYC_POINT_VARIABLE if metric_variables is None or NYC_POINT_VARIABLE in metric_variables else None,
            residual_eval_semantics=residual_eval_semantics,
            residual_ar_feedback=residual_ar_feedback,
        )
        task_cfg = device_evaluator.task_cfg
        run_cfg = device_evaluator.run_cfg
    else:
        runner = (
            (
                build_training_equivalent_truth_anchored_residual_runner(ckpt_obj, stats, ckpt_path)
                if residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
                else build_truth_anchored_residual_runner(
                    ckpt_obj,
                    stats,
                    ckpt_path,
                    residual_ar_feedback=residual_ar_feedback,
                )
            )
            if is_residual
            else build_truth_anchored_runner(ckpt_obj, stats, ckpt_path)
        )
        task_cfg = runner["task_cfg"]
        run_cfg = load_run_config(ckpt_path)
    task_key, task_cfg_kwargs = _task_cfg_cache_key(task_cfg)
    cache.task_cfg_kwargs_by_key.setdefault(task_key, task_cfg_kwargs)

    store, eval_metadata = _open_prepared_eval_store_for_checkpoint(
        prepared_data_root,
        resolution=entry_res,
        task_cfg=task_cfg,
        run_cfg=run_cfg,
        eval_start=eval_start,
        eval_end=eval_end,
        eval_year=eval_year,
    )
    eval_metadata = _add_residual_eval_semantics(
        eval_metadata,
        run_cfg,
        residual_eval_semantics,
        residual_ar_feedback,
    )
    eval_metadata["lead_aggregation"] = lead_aggregation
    chunk_start_indices = _prepared_chunk_start_indices(
        store.sizes["time"],
        trunk_steps=trunk_steps,
        total_horizon_steps=total_horizon_steps,
    )
    acc = _empty_metric_accumulator(max_lead_steps)

    segments = [
        np.arange(
            int(chunk_start) + N_INPUT_STEPS - 1,
            int(chunk_start) + N_INPUT_STEPS - 1 + warmup_steps + trunk_steps,
            dtype=np.int64,
        )
        for chunk_start in chunk_start_indices
    ]
    stream_chunks = list(_iter_prepared_stream_chunks(segments, batch_size=window_batch_size))
    total_chunks = len(stream_chunks)
    progress_label = f"prepared-warm-res{entry_res}:{ckpt_path.parent.name}"
    print(
        f"[{progress_label}] start anchors={len(chunk_start_indices)} "
        f"stream_chunks={total_chunks} warmup={warmup_steps} trunk={trunk_steps} "
        f"max_lead={max_lead_steps} device_eval={use_device_eval}",
        flush=True,
    )
    if is_residual:
        print(f"[{progress_label}] residual_eval_semantics={residual_eval_semantics}", flush=True)
        print(f"[{progress_label}] residual_ar_feedback={residual_ar_feedback}", flush=True)
    progress_t0 = time.time()
    next_progress = 1
    with _prepared_load_executor(prepared_load_workers) as load_executor:
        loader = PreparedBlockBatchLoader(
            store,
            segments,
            input_steps=N_INPUT_STEPS,
            target_steps=max_lead_steps,
            task_cfg=task_cfg,
            dt=pd.Timedelta(hours=HOURS_PER_STEP),
            load_executor=load_executor,
            max_workers=prepared_load_workers,
            label=f"prepared-warm-res{entry_res}",
        )
        for batch_i, chunk in enumerate(stream_chunks):
            step_iter = loader.iter_chunk_batches(chunk)
            try:
                first_batch, _load_stats = next(step_iter)
            except StopIteration:
                continue
            inputs_b, targets_b, forcings_b = first_batch
            if device_evaluator is not None:
                target_batches = [targets_b]
                forcing_batches = [forcings_b]
                for _step_i in range(1, warmup_steps + trunk_steps):
                    _inputs_b, next_targets, next_forcings = next(step_iter)[0]
                    target_batches.append(next_targets)
                    forcing_batches.append(next_forcings)
                device_acc = device_evaluator.evaluate_warm_chunk(
                    jax.random.PRNGKey(seed_base + batch_i),
                    inputs_b,
                    tuple(target_batches),
                    tuple(forcing_batches),
                    warmup_steps=warmup_steps,
                    trunk_steps=trunk_steps,
                )
                add_device_accumulator_to_host(
                    acc,
                    device_acc,
                    variable_names=device_evaluator.metric_variable_names,
                )
                current = batch_i + 1
                if (
                    current == next_progress
                    or current == total_chunks
                    or current % 10 == 0
                ):
                    _print_progress(progress_label, current, total_chunks, progress_t0)
                    while next_progress <= current:
                        next_progress *= 2
                continue
            with suppress_graphcast_future_warnings():
                context = runner["initialize_context"](
                    inputs=inputs_b,
                    targets_template=(
                        targets_b
                        if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
                        else targets_b * np.nan
                    ),
                    forcings=forcings_b,
                )
                step_keys = jax.random.split(
                    jax.random.PRNGKey(seed_base + batch_i),
                    2 * warmup_steps + trunk_steps * (2 + max_lead_steps * 2),
                )
                key_i = 0
                current_targets = targets_b
                current_forcings = forcings_b
                for step_i in range(warmup_steps):
                    _truth_pred, context = runner["truth_step"](
                        rng=(step_keys[key_i], step_keys[key_i + 1]),
                        context=context,
                        target_step=current_targets.isel(time=slice(0, 1)),
                        forcings_step=current_forcings.isel(time=slice(0, 1)),
                    )
                    key_i += 2
                    if step_i != warmup_steps - 1 or trunk_steps > 0:
                        _inputs_b, current_targets, current_forcings = next(step_iter)[0]

                for anchor_i in range(trunk_steps):
                    branch_targets = current_targets.isel(time=slice(0, max_lead_steps))
                    branch_forcings = current_forcings.isel(time=slice(0, max_lead_steps))
                    branch_pred = runner["branch_rollout"](
                        rng=step_keys[key_i],
                        context=context,
                        targets_template=(
                            branch_targets
                            if is_residual and residual_eval_semantics == RESIDUAL_EVAL_SEMANTICS_TEACHER_FORCED
                            else branch_targets * np.nan
                        ),
                        forcings=branch_forcings,
                    )
                    _accumulate_metrics(
                        acc,
                        branch_pred,
                        branch_targets,
                        res_grid_lats=res_grid_lats,
                        res_grid_lons=res_grid_lons,
                        stats=stats,
                        metric_variables=metric_variables,
                    )
                    key_i += 1
                    _truth_pred, context = runner["truth_step"](
                        rng=(step_keys[key_i], step_keys[key_i + 1]),
                        context=context,
                        target_step=current_targets.isel(time=slice(0, 1)),
                        forcings_step=current_forcings.isel(time=slice(0, 1)),
                    )
                    key_i += 2
                    if anchor_i != trunk_steps - 1:
                        _inputs_b, current_targets, current_forcings = next(step_iter)[0]
            current = batch_i + 1
            if (
                current == next_progress
                or current == total_chunks
                or current % 10 == 0
            ):
                _print_progress(progress_label, current, total_chunks, progress_t0)
                while next_progress <= current:
                    next_progress *= 2

    weighted_by_day, per_variable_by_day, rmse_by_day, n_by_day = _finalize_metrics(
        acc,
        lead_days,
        lead_steps,
        lead_aggregation=lead_aggregation,
    )
    return weighted_by_day, per_variable_by_day, rmse_by_day, n_by_day, eval_metadata


def _parse_res_from_path(ckpt_path: Path, run_cfg: dict) -> float | None:
    candidates = [ckpt_path.name, ckpt_path.stem, ckpt_path.parent.name, ckpt_path.parent.parent.name]
    baseline_ckpt = run_cfg.get("residual_training", {}).get("baseline_checkpoint")
    if baseline_ckpt:
        candidates.append(str(baseline_ckpt))
    for candidate in candidates:
        match = re.search(r"(?:^|[_/])res(?P<tag>\d+p\d+)(?:[_.]|$)", candidate)
        if match:
            return float(match.group("tag").replace("p", "."))
        match = re.search(r"(?:^|[_/])res(\d+)(?:[_.]|$)", candidate)
        if match:
            return float(match.group(1))
        match = re.search(r"resolution\s+(\d+(?:\.\d+)?)", candidate)
        if match:
            return float(match.group(1))
    return None


def _parse_di_from_name(run_name: str) -> int | None:
    for pattern in (r"_di(\d+)(?:_|$)", r"_dh(\d+)(?:_|$)"):
        match = re.search(pattern, run_name)
        if match:
            return int(match.group(1))
    return None


def _iter_checkpoint_paths(
    checkpoint_roots: list[Path] | None = None,
    checkpoint_paths: list[Path] | None = None,
):
    seen: set[Path] = set()
    if checkpoint_paths:
        for path in checkpoint_paths:
            resolved_path = path if path.is_absolute() else ROOT / path
            if not resolved_path.exists():
                continue
            resolved_path = resolved_path.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            yield resolved_path
    if checkpoint_roots:
        for root in checkpoint_roots:
            resolved_root = root if root.is_absolute() else ROOT / root
            if not resolved_root.exists():
                continue
            for path in sorted(resolved_root.glob("**/ckpt_best.npz")):
                resolved_path = path.resolve()
                if resolved_path in seen:
                    continue
                seen.add(resolved_path)
                yield resolved_path
        return
    if not checkpoint_paths:
        for path in sorted((ROOT / "artifacts/checkpoints").glob("**/ckpt_best.npz")):
            resolved_path = path.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            yield resolved_path


def _checkpoint_family(ckpt_path: Path, run_cfg: dict) -> str:
    family = infer_family(run_cfg)
    if family != "gc_mamba":
        return family
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    return "legacy_gc_mamba" if is_legacy_gc_mamba_checkpoint(ckpt_obj) else family


def _variant_name_for_checkpoint(ckpt_path: Path) -> str:
    if ckpt_path.name == "ckpt_best.npz":
        return ckpt_path.parent.name
    if ckpt_path.parent.name == "intermediate_checkpoints":
        run_name = ckpt_path.parent.parent.name
    else:
        run_name = ckpt_path.parent.name
    return f"{run_name}_{ckpt_path.stem}"


def discover_model_entries(
    families: list[str],
    resolutions: list[float] | None = None,
    checkpoint_roots: list[Path] | None = None,
    checkpoint_paths: list[Path] | None = None,
) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    for ckpt_path in _iter_checkpoint_paths(checkpoint_roots, checkpoint_paths):
        run_cfg = load_run_config(ckpt_path)
        family = _checkpoint_family(ckpt_path, run_cfg)
        if family not in families:
            continue
        res = _parse_res_from_path(ckpt_path, run_cfg)
        if res is None:
            continue
        if resolutions is not None and not any(np.isclose(res, requested, atol=1e-6) for requested in resolutions):
            continue
        run_name = _variant_name_for_checkpoint(ckpt_path)
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
    resolutions: list[float] | None = None,
    checkpoint_roots: list[Path] | None = None,
    checkpoint_paths: list[Path] | None = None,
) -> list[str]:
    shard_pairs = sorted(
        {
            (entry.family, entry.res)
            for entry in discover_model_entries(families, resolutions, checkpoint_roots, checkpoint_paths)
        }
    )
    return [f"{family}:{_format_resolution_value(res)}" for family, res in shard_pairs]


def _append_metric_rows(
    rows: list[dict],
    entry: ModelEntry,
    lead_days: list[float],
    lead_steps: list[int],
    metrics: list[str],
    eval_mode: str,
    weighted_by_day: dict[float, float],
    per_variable_by_day: dict[str, dict[float, float]],
    rmse_by_day: dict[str, dict[float, float]],
    n_by_day: dict[float, int],
    *,
    warmup_steps: int,
    trunk_steps: int,
    eval_metadata: dict[str, object],
) -> None:
    metadata = {
        "data_source": eval_metadata.get("data_source", ""),
        "prepared_store": eval_metadata.get("prepared_store", ""),
        "eval_start": eval_metadata.get("eval_start", ""),
        "eval_end": eval_metadata.get("eval_end", ""),
        "eval_year": eval_metadata.get("eval_year", np.nan),
        "stats_dir": eval_metadata.get("stats_dir", ""),
        "metric_grid_resolution": eval_metadata.get("metric_grid_resolution", np.nan),
        "metric_variables": eval_metadata.get("metric_variables", ""),
        "metric_semantics": eval_metadata.get("metric_semantics", METRIC_SEMANTICS),
        "lead_aggregation": eval_metadata.get("lead_aggregation", LEAD_AGGREGATION_ENDPOINT),
        "residual_eval_semantics": eval_metadata.get("residual_eval_semantics", ""),
        "residual_ar_feedback": eval_metadata.get("residual_ar_feedback", ""),
    }
    for day, step in zip(lead_days, lead_steps):
        if "weighted_allvars" in metrics:
            rows.append(
                {
                    "family": entry.family,
                    "variant": entry.variant,
                    "model_type": entry.model_type,
                    "di": entry.di,
                    "res": entry.res,
                    "lead_days": day,
                    "lead_steps": step,
                    "eval_mode": eval_mode,
                    "metric_kind": "weighted_allvars",
                    "variable": "",
                    "value": weighted_by_day.get(day, np.nan),
                    "n_points": n_by_day.get(day, 0),
                    "run_name": entry.run_name,
                    "ckpt_path": str(entry.ckpt_path),
                    "warmup_steps": warmup_steps if eval_mode == "warm" else np.nan,
                    "trunk_steps": trunk_steps if eval_mode == "warm" else np.nan,
                    **metadata,
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
                        "lead_steps": step,
                        "eval_mode": eval_mode,
                        "metric_kind": "per_variable",
                        "variable": variable,
                        "value": by_day.get(day, np.nan),
                        "n_points": n_by_day.get(day, 0),
                        "run_name": entry.run_name,
                        "ckpt_path": str(entry.ckpt_path),
                        "warmup_steps": warmup_steps if eval_mode == "warm" else np.nan,
                        "trunk_steps": trunk_steps if eval_mode == "warm" else np.nan,
                        **metadata,
                    }
                )
        if "rmse_k" in metrics:
            for variable, by_day in sorted(rmse_by_day.items()):
                rows.append(
                    {
                        "family": entry.family,
                        "variant": entry.variant,
                        "model_type": entry.model_type,
                        "di": entry.di,
                        "res": entry.res,
                        "lead_days": day,
                        "lead_steps": step,
                        "eval_mode": eval_mode,
                        "metric_kind": "rmse_k",
                        "variable": variable,
                        "value": by_day.get(day, np.nan),
                        "n_points": n_by_day.get(day, 0),
                        "run_name": entry.run_name,
                        "ckpt_path": str(entry.ckpt_path),
                        "warmup_steps": warmup_steps if eval_mode == "warm" else np.nan,
                        "trunk_steps": trunk_steps if eval_mode == "warm" else np.nan,
                        **metadata,
                        "metric_semantics": PHYSICAL_RMSE_METRIC_SEMANTICS,
                    }
                )


def _write_rows_csv(rows: list[dict], csv_path: Path) -> Path:
    if not rows:
        raise ValueError("No metric rows to write.")
    df = pd.DataFrame(rows).sort_values(CSV_SORT_COLUMNS).reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(csv_path)
    return csv_path


def _metric_variable_label(metric_variables: set[str] | None) -> str:
    return "" if metric_variables is None else " ".join(sorted(metric_variables))


def _add_common_eval_metadata(
    metadata: dict[str, object],
    *,
    stats_dir: Path,
    metric_grid_resolution: float,
    metric_variables: set[str] | None,
) -> dict[str, object]:
    out = dict(metadata)
    out["stats_dir"] = str(stats_dir)
    out["metric_grid_resolution"] = float(metric_grid_resolution)
    out["metric_variables"] = _metric_variable_label(metric_variables)
    return out


def _filter_entries_with_prepared_stores(entries: list[ModelEntry], prepared_data_root: Path) -> list[ModelEntry]:
    out: list[ModelEntry] = []
    warned_resolutions: set[float] = set()
    for entry in entries:
        store_path = prepared_store_path_from_root(prepared_data_root, entry.res)
        if store_path.exists():
            out.append(entry)
            continue
        if entry.res not in warned_resolutions:
            _warn_skip(f"skipping res{entry.res}: prepared array store not found at {store_path}")
            warned_resolutions.add(entry.res)
    return out


def main() -> None:
    args = parse_args()
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.trunk_steps <= 0:
        raise ValueError("--trunk-steps must be positive.")
    if args.prepared_load_workers <= 0:
        raise ValueError("--prepared-load-workers must be positive.")
    if args.prepared_stream_block_steps <= 0:
        raise ValueError("--prepared-stream-block-steps must be positive.")
    if (args.eval_start is None) != (args.eval_end is None):
        raise ValueError("Provide both --eval-start and --eval-end, or neither.")
    metric_variables = None if args.metric_variables is None else set(args.metric_variables)

    if args.print_shards:
        entries = discover_model_entries(
            args.families, args.resolutions, args.checkpoint_roots, args.checkpoint_paths
        )
        if args.data_source == "prepared_array":
            entries = _filter_entries_with_prepared_stores(entries, args.prepared_data_root)
        for family, res in sorted({(entry.family, entry.res) for entry in entries}):
            print(f"{family}:{_format_resolution_value(res)}")
        return

    entries = discover_model_entries(args.families, args.resolutions, args.checkpoint_roots, args.checkpoint_paths)
    if args.data_source == "prepared_array":
        entries = _filter_entries_with_prepared_stores(entries, args.prepared_data_root)
    if not entries:
        raise RuntimeError("No checkpoints found for the requested family/resolution filters.")

    if args.lead_steps is not None:
        lead_steps = sorted({int(step) for step in args.lead_steps})
        if any(step <= 0 for step in lead_steps):
            raise ValueError("--lead-steps values must be positive.")
        lead_days = _lead_days_from_steps(lead_steps)
    else:
        lead_days = [float(day) for day in args.lead_days]
        lead_steps = [int((24 * day) // HOURS_PER_STEP) for day in args.lead_days]
    stats_dir = args.stats_dir if args.stats_dir.is_absolute() else ROOT / args.stats_dir
    stats = _load_stats(stats_dir)
    ds_base = None
    start_target_idx = 0
    n_steps = 0
    target_indices: list[int] = []
    prepared_metric_grid: tuple[xarray.DataArray, xarray.DataArray] | None = None
    if args.data_source == "raw":
        ds_base, start_target_idx, n_steps = _prepare_dataset(max(lead_steps), args.n_eval_days)
        target_indices = _target_indices(start_target_idx, n_steps, max(lead_steps))
    else:
        prepared_metric_grid = load_prepared_metric_grid(
            args.prepared_data_root,
            resolution=args.metric_grid_resolution,
        )
    cache = EvalShardCache(
        ds_res_by_stride={},
        metric_grid_by_stride={},
        task_cfg_kwargs_by_key={},
        cold_batches={},
        warm_batches={},
    )

    print(
        f"Evaluating {len(entries)} checkpoints across families={args.families} "
        f"resolutions={args.resolutions} data_source={args.data_source} "
        f"lead_aggregation={args.lead_aggregation} metric_grid_resolution={args.metric_grid_resolution} "
        f"stats_dir={stats_dir} metric_variables={_metric_variable_label(metric_variables) or '<all>'}"
    )
    rows: list[dict] = []
    csv_path = args.output_data_dir / args.output_csv_name
    for index, entry in enumerate(entries, start=1):
        print(f"\n[{index}/{len(entries)}] {entry.family} res={entry.res} {entry.run_name}")
        row_count_before = len(rows)
        if "cold" in args.eval_modes:
            try:
                if args.data_source == "raw":
                    assert ds_base is not None
                    cold_weighted_by_day, cold_per_variable_by_day, cold_rmse_by_day, cold_n_by_day = _evaluate_checkpoint(
                        entry.ckpt_path,
                        stats,
                        ds_base,
                        start_target_idx,
                        n_steps,
                        lead_days,
                        lead_steps,
                        seed_base=100000 * index,
                        window_batch_size=args.window_batch_size,
                        target_indices=target_indices,
                        residual_eval_semantics=args.residual_eval_semantics,
                        residual_ar_feedback=args.residual_ar_feedback,
                        lead_aggregation=args.lead_aggregation,
                        metric_variables=metric_variables,
                        cache=cache,
                    )
                    cold_metadata = {"data_source": "raw", "lead_aggregation": args.lead_aggregation}
                    if entry.family == "residual_mamba":
                        cold_metadata["residual_eval_semantics"] = args.residual_eval_semantics
                        cold_metadata["residual_ar_feedback"] = args.residual_ar_feedback
                else:
                    assert prepared_metric_grid is not None
                    cold_weighted_by_day, cold_per_variable_by_day, cold_rmse_by_day, cold_n_by_day, cold_metadata = (
                        _evaluate_checkpoint_prepared(
                            entry.ckpt_path,
                            stats,
                            args.prepared_data_root,
                            entry.res,
                            lead_days,
                            lead_steps,
                            seed_base=100000 * index,
                            window_batch_size=args.window_batch_size,
                            prepared_load_workers=args.prepared_load_workers,
                            prepared_stream_block_steps=args.prepared_stream_block_steps,
                            use_device_eval=not args.disable_prepared_device_eval,
                            cache=cache,
                            res_grid_lats=prepared_metric_grid[0],
                            res_grid_lons=prepared_metric_grid[1],
                            eval_start=args.eval_start,
                            eval_end=args.eval_end,
                            eval_year=args.eval_year,
                            residual_eval_semantics=args.residual_eval_semantics,
                            residual_ar_feedback=args.residual_ar_feedback,
                            lead_aggregation=args.lead_aggregation,
                            metric_variables=metric_variables,
                        )
                    )
                cold_metadata = _add_common_eval_metadata(
                    cold_metadata,
                    stats_dir=stats_dir,
                    metric_grid_resolution=args.metric_grid_resolution,
                    metric_variables=metric_variables,
                )
                _append_metric_rows(
                    rows,
                    entry,
                    lead_days,
                    lead_steps,
                    args.metrics,
                    "cold",
                    cold_weighted_by_day,
                    cold_per_variable_by_day,
                    cold_rmse_by_day,
                    cold_n_by_day,
                    warmup_steps=args.warmup_steps,
                    trunk_steps=args.trunk_steps,
                    eval_metadata=cold_metadata,
                )
            except (FileNotFoundError, PreparedDataError) as exc:
                if args.data_source != "prepared_array":
                    raise
                _warn_skip(f"skipping cold eval for {entry.run_name}: {exc}")

        if "warm" in args.eval_modes:
            try:
                if args.data_source == "raw":
                    assert ds_base is not None
                    warm_weighted_by_day, warm_per_variable_by_day, warm_rmse_by_day, warm_n_by_day = (
                        _evaluate_checkpoint_truth_anchored(
                            entry.ckpt_path,
                            stats,
                            ds_base,
                            start_target_idx,
                            n_steps,
                            lead_days,
                            lead_steps,
                            seed_base=200000 * index,
                            warmup_steps=args.warmup_steps,
                            trunk_steps=args.trunk_steps,
                            window_batch_size=args.window_batch_size,
                            residual_eval_semantics=args.residual_eval_semantics,
                            residual_ar_feedback=args.residual_ar_feedback,
                            lead_aggregation=args.lead_aggregation,
                            metric_variables=metric_variables,
                            cache=cache,
                        )
                    )
                    warm_metadata = {"data_source": "raw", "lead_aggregation": args.lead_aggregation}
                    if entry.family == "residual_mamba":
                        warm_metadata["residual_eval_semantics"] = args.residual_eval_semantics
                        warm_metadata["residual_ar_feedback"] = args.residual_ar_feedback
                else:
                    assert prepared_metric_grid is not None
                    warm_weighted_by_day, warm_per_variable_by_day, warm_rmse_by_day, warm_n_by_day, warm_metadata = (
                        _evaluate_checkpoint_truth_anchored_prepared(
                            entry.ckpt_path,
                            stats,
                            args.prepared_data_root,
                            entry.res,
                            lead_days,
                            lead_steps,
                            seed_base=200000 * index,
                            warmup_steps=args.warmup_steps,
                            trunk_steps=args.trunk_steps,
                            window_batch_size=args.window_batch_size,
                            prepared_load_workers=args.prepared_load_workers,
                            use_device_eval=not args.disable_prepared_device_eval,
                            cache=cache,
                            res_grid_lats=prepared_metric_grid[0],
                            res_grid_lons=prepared_metric_grid[1],
                            eval_start=args.eval_start,
                            eval_end=args.eval_end,
                            eval_year=args.eval_year,
                            residual_eval_semantics=args.residual_eval_semantics,
                            residual_ar_feedback=args.residual_ar_feedback,
                            lead_aggregation=args.lead_aggregation,
                            metric_variables=metric_variables,
                        )
                    )
                warm_metadata = _add_common_eval_metadata(
                    warm_metadata,
                    stats_dir=stats_dir,
                    metric_grid_resolution=args.metric_grid_resolution,
                    metric_variables=metric_variables,
                )
                _append_metric_rows(
                    rows,
                    entry,
                    lead_days,
                    lead_steps,
                    args.metrics,
                    "warm",
                    warm_weighted_by_day,
                    warm_per_variable_by_day,
                    warm_rmse_by_day,
                    warm_n_by_day,
                    warmup_steps=args.warmup_steps,
                    trunk_steps=args.trunk_steps,
                    eval_metadata=warm_metadata,
                )
            except (FileNotFoundError, PreparedDataError) as exc:
                if args.data_source != "prepared_array":
                    raise
                _warn_skip(f"skipping warm eval for {entry.run_name}: {exc}")

        if len(rows) > row_count_before:
            _write_rows_csv(rows, csv_path)
            print(f"Updated CSV after {index}/{len(entries)} checkpoints: {csv_path}")

    if not rows:
        raise RuntimeError("No checkpoints were evaluated successfully.")
    _write_rows_csv(rows, csv_path)
    print(f"\nSaved CSV: {csv_path}")
    print(
        "Cache summary: "
        f"cold_hits={cache.cold_hits} cold_misses={cache.cold_misses} "
        f"warm_hits={cache.warm_hits} warm_misses={cache.warm_misses} "
        f"stride_cache={len(cache.ds_res_by_stride)} task_cfg_cache={len(cache.task_cfg_kwargs_by_key)} "
        f"cold_cache_mode={cache.cold_cache_mode}"
    )


if __name__ == "__main__":
    main()
