#!/usr/bin/env python3
"""Fine-tune GraphCast_small on ERA5.

This script:
 - loads local train/eval datasets in GraphCast-compatible layout
 - restores GraphCast_small params + configs from a GraphCast checkpoint
 - runs one-step (+6h) training with optional short rollout loss
 - logs train/eval loss, step time, memory usage
 - saves checkpoints and plots validation loss
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graphcast_dataset import open_graphcast_era5


def _require_graphcast() -> None:
    try:
        import graphcast  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "graphcast is required. Activate env via scripts/graphcast_env.sh or install graphcast+jax+haiku."
        ) from exc


_require_graphcast()
import haiku as hk
import jax
import jax.numpy as jnp
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast as gc,
    normalization,
    xarray_jax,
)

# Silence known upstream xarray FutureWarning emitted inside graphcast.autoregressive.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"graphcast\.autoregressive",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The return type of `Dataset\.dims` will be changed.*",
)


DEFAULT_LOCAL_DATASET = "data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
DEFAULT_TRAIN = DEFAULT_LOCAL_DATASET
DEFAULT_EVAL = ""
DEFAULT_CKPT = (
    "data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
DEFAULT_STATS_DIR = "data/graphcast/graphcast/stats"
DEFAULT_OUT_DIR = "artifacts/checkpoints/graphcast_finetune_1y"


@dataclasses.dataclass
class RunConfig:
    train_path: str
    eval_path: str | None
    train_start_year: int | None
    train_end_year: int | None
    val_days: int
    ckpt_in: str
    out_dir: str
    run_name: str
    batch_size: int
    epochs: int
    tiny: bool
    tiny_days: int
    eval_every: int
    eval_batch_size: int
    checkpoint_every: int
    lr: float
    weight_decay: float
    seed: int
    rollout_steps: int
    rollout_weight: float
    enable_rollout: bool
    precision: str


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Fine-tune GraphCast_small on ERA5.")
    parser.add_argument(
        "--train-path",
        default=DEFAULT_TRAIN,
        help="Local dataset path (.zarr or .nc) used for training.",
    )
    parser.add_argument(
        "--eval-path",
        default=DEFAULT_EVAL,
        help="Optional local evaluation dataset path. If omitted, eval is trailing --val-days from train years.",
    )
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--ckpt-in", default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-name", default="h100_full")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--tiny-days", type=int, default=14)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--rollout-weight", type=float, default=0.5)
    parser.add_argument("--no-rollout", action="store_true", help="Disable rollout loss term.")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()
    if args.val_days <= 0:
        raise ValueError("--val-days must be > 0")
    if (args.train_start_year is None) ^ (args.train_end_year is None):
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_start_year is not None and args.train_start_year > args.train_end_year:
        raise ValueError("--train-start-year must be <= --train-end-year")
    return RunConfig(
        train_path=args.train_path,
        eval_path=args.eval_path if args.eval_path else None,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        val_days=args.val_days,
        ckpt_in=args.ckpt_in,
        out_dir=args.out_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        tiny=args.tiny,
        tiny_days=args.tiny_days,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        rollout_weight=args.rollout_weight,
        enable_rollout=not args.no_rollout,
        precision=args.precision,
    )


def _assert_local_path(path: str, arg_name: str) -> None:
    if "://" in path:
        raise ValueError(f"{arg_name} must be a local path; remote URIs are disabled: {path}")


def load_stats(stats_dir: Path) -> dict[str, xr.Dataset]:
    def open_nc(name: str) -> xr.Dataset:
        return xr.open_dataset(stats_dir / f"{name}.nc")

    return {
        "stddev_by_level": open_nc("stddev_by_level"),
        "mean_by_level": open_nc("mean_by_level"),
        "diffs_stddev_by_level": open_nc("diffs_stddev_by_level"),
    }


def load_graphcast_checkpoint(path: Path) -> gc.CheckPoint:
    with path.open("rb") as f:
        return checkpoint.load(f, gc.CheckPoint)


def build_predictor(
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    stats: dict[str, xr.Dataset],
    *,
    use_bf16: bool,
    gradient_checkpointing: bool,
):
    predictor = gc.GraphCast(model_cfg, task_cfg)
    if use_bf16:
        predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=gradient_checkpointing)
    return predictor


def validate_stats_coverage(task_cfg: gc.TaskConfig, stats: dict[str, xr.Dataset]) -> None:
    required_inputs = set(task_cfg.input_variables) | set(task_cfg.forcing_variables)
    required_targets = set(task_cfg.target_variables)

    stddev_vars = set(stats["stddev_by_level"].data_vars)
    mean_vars = set(stats["mean_by_level"].data_vars)
    diffs_vars = set(stats["diffs_stddev_by_level"].data_vars)

    missing_stddev = sorted(required_inputs - stddev_vars)
    missing_mean = sorted(required_inputs - mean_vars)
    missing_diffs = sorted(required_targets - diffs_vars)

    if missing_stddev or missing_mean or missing_diffs:
        raise ValueError(
            "Normalization stats missing required variables: "
            f"stddev_missing={missing_stddev}, "
            f"mean_missing={missing_mean}, "
            f"diffs_stddev_missing={missing_diffs}"
        )


def _ensure_datetime_coord(ds: xr.Dataset) -> xr.Dataset:
    def _with_datetime_coord(dataset: xr.Dataset, time_values: np.ndarray) -> xr.Dataset:
        if "batch" in dataset.dims:
            # GraphCast derived forcing helpers expect datetime to match (batch, time) when batch exists.
            bt = np.broadcast_to(np.asarray(time_values)[None, :], (dataset.sizes["batch"], len(time_values)))
            return dataset.assign_coords(datetime=(("batch", "time"), bt))
        return dataset.assign_coords(datetime=("time", time_values))

    if np.issubdtype(ds.time.dtype, np.datetime64):
        if "datetime" not in ds.coords:
            ds = _with_datetime_coord(ds, ds.time.values)
        elif "batch" in ds.dims and ds.coords["datetime"].dims == ("time",):
            ds = _with_datetime_coord(ds, ds.time.values)
        return ds

    # Handle encoded numeric times from some Zarr stores.
    decoded = xr.decode_cf(ds)
    if "datetime" not in decoded.coords:
        decoded = _with_datetime_coord(decoded, decoded.time.values)
    elif "batch" in decoded.dims and decoded.coords["datetime"].dims == ("time",):
        decoded = _with_datetime_coord(decoded, decoded.time.values)
    return decoded


def prepare_dataset_for_task(ds: xr.Dataset, task_cfg: gc.TaskConfig) -> xr.Dataset:
    ds = _ensure_datetime_coord(ds)
    forcing_vars = set(task_cfg.forcing_variables)
    static_input_vars = (
        set(task_cfg.input_variables)
        - set(task_cfg.target_variables)
        - set(task_cfg.forcing_variables)
    )

    # Add derived forcings once at dataset level (not per-sample).
    if forcing_vars & {"year_progress_sin", "year_progress_cos", "day_progress_sin", "day_progress_cos"}:
        data_utils.add_derived_vars(ds)
    if "toa_incident_solar_radiation" in forcing_vars and "toa_incident_solar_radiation" not in ds.data_vars:
        data_utils.add_tisr_var(ds)

    # Keep static inputs truly time-independent so model input channels match checkpoint structure.
    # If static vars were broadcast along time by dataset loaders, collapse them back.
    for name in sorted(static_input_vars):
        if name in ds.data_vars and "time" in ds[name].dims:
            ds[name] = ds[name].isel(time=0, drop=True)
    return ds


def infer_time_step(ds: xr.Dataset) -> pd.Timedelta:
    if ds.sizes["time"] < 2:
        raise ValueError("Dataset must contain at least two time steps.")
    delta = pd.Timedelta(ds.time.values[1] - ds.time.values[0])
    if delta <= pd.Timedelta(0):
        raise ValueError(f"Invalid non-positive time step: {delta}.")
    return delta


def input_steps_from_duration(input_duration: str, dt: pd.Timedelta) -> int:
    duration = pd.Timedelta(input_duration)
    if duration % dt != pd.Timedelta(0):
        raise ValueError(f"input_duration={duration} is not divisible by dt={dt}.")
    steps = int(duration // dt)
    if steps < 1:
        raise ValueError(f"input_duration={duration} produced invalid input steps={steps}.")
    return steps


def lead_times(target_steps: int, dt: pd.Timedelta) -> Sequence[pd.Timedelta]:
    return [dt * (i + 1) for i in range(target_steps)]


def valid_final_input_indices(total_time_steps: int, input_steps: int, target_steps: int) -> np.ndarray:
    start = input_steps - 1
    stop = total_time_steps - target_steps
    if stop <= start:
        return np.array([], dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


def build_single_sample(
    ds: xr.Dataset,
    *,
    final_input_idx: int,
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    window_start = final_input_idx - input_steps + 1
    window_stop = final_input_idx + target_steps
    if window_start < 0 or window_stop >= ds.sizes["time"]:
        raise IndexError(
            f"Requested sample idx={final_input_idx} outside valid range for "
            f"input_steps={input_steps}, target_steps={target_steps}, total={ds.sizes['time']}."
        )

    # Inclusive right endpoint via +1 in slice stop.
    window = ds.isel(time=slice(window_start, window_stop + 1))
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        window,
        input_variables=task_cfg.input_variables,
        target_variables=task_cfg.target_variables,
        forcing_variables=task_cfg.forcing_variables,
        pressure_levels=task_cfg.pressure_levels,
        input_duration=task_cfg.input_duration,
        target_lead_times=lead_times(target_steps, dt),
    )
    return inputs, targets, forcings


def build_batch_from_indices(
    ds: xr.Dataset,
    *,
    indices: Iterable[int],
    input_steps: int,
    target_steps: int,
    task_cfg: gc.TaskConfig,
    dt: pd.Timedelta,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    inputs_list = []
    targets_list = []
    forcings_list = []

    for idx in indices:
        inputs, targets, forcings = build_single_sample(
            ds,
            final_input_idx=int(idx),
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )
        # Each sample has batch=1. Drop it, then stack samples back into batch.
        inputs_list.append(inputs.isel(batch=0, drop=True))
        targets_list.append(targets.isel(batch=0, drop=True))
        forcings_list.append(forcings.isel(batch=0, drop=True))

    batch_inputs = xr.concat(inputs_list, dim="batch").assign_coords(batch=np.arange(len(inputs_list)))
    batch_targets = xr.concat(targets_list, dim="batch").assign_coords(batch=np.arange(len(targets_list)))
    batch_forcings = xr.concat(forcings_list, dim="batch").assign_coords(batch=np.arange(len(forcings_list)))
    return batch_inputs.load(), batch_targets.load(), batch_forcings.load()


def scalarize_loss(loss_da: xr.DataArray) -> jax.Array:
    return jnp.mean(xarray_jax.unwrap_data(loss_da))


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
    use_rollout_loss: bool,
    rollout_weight: float,
    progress_label: str = "eval",
) -> dict[str, float]:
    total_losses: list[float] = []
    one_losses: list[float] = []
    rollout_losses: list[float] = []
    n_batches = (len(eval_indices) + eval_batch_size - 1) // eval_batch_size
    t_eval0 = time.time()

    for batch_i, i in enumerate(range(0, len(eval_indices), eval_batch_size), start=1):
        idx = eval_indices[i : i + eval_batch_size]
        inputs, rollout_targets, rollout_forcings = build_batch_from_indices(
            eval_ds,
            indices=idx,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt,
        )
        one_targets = rollout_targets.isel(time=[0])
        one_forcings = rollout_forcings.isel(time=[0])

        rng, key_one, key_roll = jax.random.split(rng, 3)
        (one_loss_and_diag, _state_after_one) = transformed.apply(
            params, state, key_one, inputs, one_targets, one_forcings, False
        )
        one_loss = float(scalarize_loss(one_loss_and_diag[0]))
        rollout_loss = 0.0
        total = one_loss

        if use_rollout_loss:
            (roll_loss_and_diag, _state_after_rollout) = transformed.apply(
                params, state, key_roll, inputs, rollout_targets, rollout_forcings, False
            )
            rollout_loss = float(scalarize_loss(roll_loss_and_diag[0]))
            total = one_loss + rollout_weight * rollout_loss

        one_losses.append(one_loss)
        rollout_losses.append(rollout_loss)
        total_losses.append(total)

        if batch_i == 1 or batch_i % 10 == 0 or batch_i == n_batches:
            elapsed = time.time() - t_eval0
            print(
                f"[{progress_label}] batch {batch_i}/{n_batches} "
                f"elapsed {elapsed:.1f}s current_total {total:.6f}"
            )

    return {
        "total": float(np.mean(total_losses)),
        "one_step": float(np.mean(one_losses)),
        "rollout": float(np.mean(rollout_losses)) if use_rollout_loss else 0.0,
    }


def save_checkpoint(
    out_dir: Path,
    *,
    params: hk.Params,
    step: int,
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    description: str,
    license_text: str,
) -> None:
    path = out_dir / f"ckpt_step{step}.npz"
    ckpt_out = gc.CheckPoint(
        params=params,
        model_config=model_cfg,
        task_config=task_cfg,
        description=description,
        license=license_text,
    )
    with path.open("wb") as f:
        checkpoint.dump(f, ckpt_out)
    print(f"saved checkpoint: {path}")


def save_logs(
    out_dir: Path,
    train_losses: list[tuple[int, float]],
    eval_losses: list[tuple[int, float]],
    eval_details: list[dict[str, Any]],
    step_times: list[tuple[int, float]],
    mem_usage: list[tuple[int, float]],
    actual_usage: list[dict[str, Any]],
    epoch_summaries: list[dict[str, Any]],
) -> None:
    with (out_dir / "train_loss.json").open("w", encoding="utf-8") as f:
        json.dump(train_losses, f)
    with (out_dir / "eval_loss.json").open("w", encoding="utf-8") as f:
        json.dump(eval_losses, f)
    with (out_dir / "eval_details.json").open("w", encoding="utf-8") as f:
        json.dump(eval_details, f, indent=2)
    with (out_dir / "step_times.json").open("w", encoding="utf-8") as f:
        json.dump(step_times, f)
    with (out_dir / "memory_gib.json").open("w", encoding="utf-8") as f:
        json.dump(mem_usage, f)
    with (out_dir / "actual_usage.json").open("w", encoding="utf-8") as f:
        json.dump(actual_usage, f, indent=2)
    rss_vals = [float(x["proc_rss_gib"]) for x in actual_usage if x.get("proc_rss_gib") is not None]
    gpu_vals = [float(x["gpu_mem_gib"]) for x in actual_usage if x.get("gpu_mem_gib") is not None]
    with (out_dir / "actual_usage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rss_gib_peak": float(np.max(rss_vals)) if rss_vals else None,
                "rss_gib_avg": float(np.mean(rss_vals)) if rss_vals else None,
                "gpu_mem_gib_peak": float(np.max(gpu_vals)) if gpu_vals else None,
                "gpu_mem_gib_avg": float(np.mean(gpu_vals)) if gpu_vals else None,
                "samples": len(actual_usage),
            },
            f,
            indent=2,
        )
    with (out_dir / "epoch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(epoch_summaries, f, indent=2)


def _read_proc_mem_gib() -> tuple[float | None, float | None]:
    """Return (current_rss_gib, peak_hwm_gib) from /proc/self/status."""
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as f:
            lines = f.readlines()
        rss_kib: int | None = None
        hwm_kib: int | None = None
        for line in lines:
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kib = int(line.split()[1])
        rss_gib = float(rss_kib) / (1024**2) if rss_kib is not None else None
        hwm_gib = float(hwm_kib) / (1024**2) if hwm_kib is not None else None
        return rss_gib, hwm_gib
    except Exception:
        return None, None


def _read_gpu_mem_by_device() -> tuple[list[dict[str, float | int]], float | None]:
    """Return per-device GPU memory stats and total used GiB from nvidia-smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        devices: list[dict[str, float | int]] = []
        total_mib = 0.0
        for raw in proc.stdout.splitlines():
            row = raw.strip()
            if not row:
                continue
            # format: "index, used_mib, total_mib"
            parts = [x.strip() for x in row.split(",") if x.strip() != ""]
            if len(parts) < 3:
                continue
            index = int(parts[0])
            used_mib = float(parts[1])
            total_dev_mib = float(parts[2])
            total_mib += used_mib
            devices.append(
                {
                    "index": index,
                    "used_gib": used_mib / 1024.0,
                    "total_gib": total_dev_mib / 1024.0,
                }
            )
        return devices, (total_mib / 1024.0 if devices else None)
    except Exception:
        return [], None


def sample_actual_usage(step: int, pid: int) -> dict[str, Any]:
    del pid
    proc_rss_gib, proc_hwm_gib = _read_proc_mem_gib()
    gpu_devices, gpu_mem_gib = _read_gpu_mem_by_device()
    return {
        "step": step,
        "timestamp": time.time(),
        "proc_rss_gib": proc_rss_gib,
        "proc_hwm_gib": proc_hwm_gib,
        "gpu_mem_gib": gpu_mem_gib,
        "gpu_mem_total_gib": gpu_mem_gib,
        "gpu_devices": gpu_devices,
    }


def plot_val_loss(out_dir: Path, eval_losses: list[tuple[int, float]]) -> None:
    if not eval_losses:
        return
    steps, vals = zip(*eval_losses)
    plt.figure()
    plt.plot(steps, vals, marker="o")
    y_max = max(vals) * 2.0
    if y_max > 0:
        plt.ylim(0.0, y_max)
    plt.xlabel("step")
    plt.ylabel("eval loss")
    plt.title("Validation loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss.png")
    plt.close()


def _slice_dataset_years(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    start = pd.Timestamp(year=start_year, month=1, day=1, hour=0)
    end = pd.Timestamp(year=end_year, month=12, day=31, hour=23, minute=59, second=59)
    return ds.sel(time=slice(start, end))


def _split_train_eval_trailing_days(ds: xr.Dataset, val_days: int) -> tuple[xr.Dataset, xr.Dataset]:
    val_steps = val_days * 4  # 6-hour cadence
    if ds.sizes.get("time", 0) <= val_steps:
        raise ValueError(
            f"Not enough timesteps ({ds.sizes.get('time', 0)}) for trailing validation of "
            f"{val_days} days ({val_steps} steps)."
        )
    train_ds = ds.isel(time=slice(None, -val_steps))
    eval_ds = ds.isel(time=slice(-val_steps, None))

    train_times = pd.DatetimeIndex(pd.to_datetime(train_ds.time.values))
    eval_times = pd.DatetimeIndex(pd.to_datetime(eval_ds.time.values))
    if train_times.intersection(eval_times).size > 0:
        raise ValueError("Train/eval overlap detected in trailing validation split.")
    return train_ds, eval_ds


def main() -> None:
    cfg = parse_args()
    _assert_local_path(cfg.train_path, "--train-path")
    if cfg.eval_path:
        _assert_local_path(cfg.eval_path, "--eval-path")
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_in = load_graphcast_checkpoint(Path(cfg.ckpt_in))
    model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    params_from_ckpt = ckpt_in.params
    norm_stats = load_stats(Path(DEFAULT_STATS_DIR))
    validate_stats_coverage(task_cfg, norm_stats)

    train_full = open_graphcast_era5(cfg.train_path)
    train_full = _ensure_datetime_coord(train_full)
    years_all = pd.DatetimeIndex(pd.to_datetime(train_full.time.values)).year.astype(int)
    min_year = int(years_all.min())
    max_year = int(years_all.max())
    train_start_year = cfg.train_start_year if cfg.train_start_year is not None else min_year
    train_end_year = cfg.train_end_year if cfg.train_end_year is not None else max_year
    if train_start_year < min_year or train_end_year > max_year:
        raise ValueError(
            f"Requested train years [{train_start_year}, {train_end_year}] outside dataset years "
            f"[{min_year}, {max_year}]"
        )

    train_range_ds = _slice_dataset_years(train_full, train_start_year, train_end_year)
    if train_range_ds.sizes.get("time", 0) == 0:
        raise ValueError("Selected training year range produced an empty dataset.")

    if cfg.eval_path:
        train_ds = train_range_ds
        eval_ds = open_graphcast_era5(cfg.eval_path)
    else:
        train_ds, eval_ds = _split_train_eval_trailing_days(train_range_ds, cfg.val_days)

    if train_ds.sizes.get("time", 0) == 0:
        raise ValueError("Train dataset is empty after split.")
    if eval_ds.sizes.get("time", 0) == 0:
        raise ValueError("Eval dataset is empty after split.")

    print(
        "Data split: "
        f"train_path={cfg.train_path}, eval_path={cfg.eval_path or '<derived-from-train>'}, "
        f"train_years={train_start_year}-{train_end_year}, val_days={cfg.val_days}, "
        f"train_time={train_ds.sizes['time']}, eval_time={eval_ds.sizes['time']}"
    )

    if cfg.tiny:
        slice_steps = cfg.tiny_days * 4  # 6h steps/day
        train_ds = train_ds.isel(time=slice(-slice_steps, None))
        batch_size = 1
    else:
        batch_size = cfg.batch_size

    # Ensure required forcing variables exist before sample extraction.
    train_ds = prepare_dataset_for_task(train_ds, task_cfg)
    eval_ds = prepare_dataset_for_task(eval_ds, task_cfg)

    dt_train = infer_time_step(train_ds)
    dt_eval = infer_time_step(eval_ds)
    if dt_train != dt_eval:
        raise ValueError(f"Train/eval time step mismatch: train={dt_train}, eval={dt_eval}")

    input_steps = input_steps_from_duration(task_cfg.input_duration, dt_train)
    target_steps = max(1, cfg.rollout_steps if cfg.enable_rollout else 1)
    use_rollout_loss = cfg.enable_rollout and target_steps > 1

    train_final_indices = valid_final_input_indices(train_ds.sizes["time"], input_steps, target_steps)
    eval_final_indices = valid_final_input_indices(eval_ds.sizes["time"], input_steps, target_steps)
    if len(train_final_indices) == 0:
        raise ValueError("No train samples after applying input/target window requirements.")
    if len(eval_final_indices) == 0:
        raise ValueError("No eval samples after applying input/target window requirements.")

    print(
        "Prepared dataset windows: "
        f"train_time={train_ds.sizes['time']}, eval_time={eval_ds.sizes['time']}, "
        f"train_samples={len(train_final_indices)}, eval_samples={len(eval_final_indices)}, "
        f"input_steps={input_steps}, target_steps={target_steps}"
    )

    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=True,
        )
        # For autoregressive.Predictor, training goes through .loss(), not .loss_and_predictions().
        return predictor.loss(inputs, targets, forcings)

    transformed = hk.transform_with_state(forward_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    # Initialize model/state structure with the first train sample.
    sample_inputs, sample_roll_targets, sample_roll_forcings = build_batch_from_indices(
        train_ds,
        indices=[int(train_final_indices[0])],
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
    )
    sample_one_targets = sample_roll_targets.isel(time=[0])
    sample_one_forcings = sample_roll_forcings.isel(time=[0])
    init_params, state = transformed.init(rng, sample_inputs, sample_one_targets, sample_one_forcings, True)

    # Keep model structure from init, then overlay pretrained checkpoint values.
    params = hk.data_structures.merge(init_params, params_from_ckpt)

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        rng_key: jax.Array,
        inputs: xr.Dataset,
        one_targets: xr.Dataset,
        one_forcings: xr.Dataset,
        rollout_targets: xr.Dataset,
        rollout_forcings: xr.Dataset,
    ):
        def loss_fn(p, s, key):
            key_one, key_roll = jax.random.split(key)
            (one_loss_and_diag, state_after_one) = transformed.apply(
                p, s, key_one, inputs, one_targets, one_forcings, True
            )
            one_loss = scalarize_loss(one_loss_and_diag[0])
            total_loss = one_loss
            rollout_loss = jnp.array(0.0, dtype=one_loss.dtype)
            state_after = state_after_one

            if use_rollout_loss:
                (roll_loss_and_diag, state_after_rollout) = transformed.apply(
                    p, state_after_one, key_roll, inputs, rollout_targets, rollout_forcings, True
                )
                rollout_loss = scalarize_loss(roll_loss_and_diag[0])
                total_loss = one_loss + jnp.asarray(cfg.rollout_weight, dtype=one_loss.dtype) * rollout_loss
                state_after = state_after_rollout

            return total_loss, (state_after, one_loss, rollout_loss)

        (total_loss, (new_state, one_loss, rollout_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, rng_key
        )
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, new_opt_state, total_loss, one_loss, rollout_loss

    train_losses: list[tuple[int, float]] = []
    eval_losses: list[tuple[int, float]] = []
    eval_details: list[dict[str, Any]] = []
    step_times: list[tuple[int, float]] = []
    mem_usage: list[tuple[int, float]] = []
    actual_usage: list[dict[str, Any]] = []
    epoch_summaries: list[dict[str, Any]] = []
    pid = os.getpid()

    step = 0
    for epoch in range(cfg.epochs):
        epoch_indices = train_final_indices.copy()
        if not cfg.tiny:
            np.random.default_rng(cfg.seed + epoch).shuffle(epoch_indices)

        epoch_loss_accum: list[float] = []
        epoch_one_accum: list[float] = []
        epoch_roll_accum: list[float] = []
        epoch_start_step = step + 1

        for i in range(0, len(epoch_indices), batch_size):
            batch_idx = epoch_indices[i : i + batch_size]
            batch_inputs, batch_roll_targets, batch_roll_forcings = build_batch_from_indices(
                train_ds,
                indices=batch_idx,
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt_train,
            )
            batch_one_targets = batch_roll_targets.isel(time=[0])
            batch_one_forcings = batch_roll_forcings.isel(time=[0])

            rng, step_key = jax.random.split(rng)
            t0 = time.time()
            params, state, opt_state, total_loss, one_loss, rollout_loss = train_step(
                params,
                state,
                opt_state,
                step_key,
                batch_inputs,
                batch_one_targets,
                batch_one_forcings,
                batch_roll_targets,
                batch_roll_forcings,
            )
            step_time = time.time() - t0

            step += 1
            total_loss_f = float(total_loss)
            one_loss_f = float(one_loss)
            rollout_loss_f = float(rollout_loss)
            train_losses.append((step, total_loss_f))
            epoch_loss_accum.append(total_loss_f)
            epoch_one_accum.append(one_loss_f)
            epoch_roll_accum.append(rollout_loss_f)
            step_times.append((step, step_time))

            usage = sample_actual_usage(step=step, pid=pid)
            actual_usage.append(usage)
            if usage.get("gpu_mem_gib") is not None:
                mem_usage.append((step, float(usage["gpu_mem_gib"])))

            if step % 10 == 0:
                if use_rollout_loss:
                    print(
                        f"step {step} total {total_loss_f:.6f} "
                        f"one_step {one_loss_f:.6f} rollout {rollout_loss_f:.6f}"
                    )
                else:
                    print(f"step {step} loss {total_loss_f:.6f}")

            if step % cfg.eval_every == 0:
                eval_metrics = run_eval(
                    transformed,
                    params,
                    state,
                    rng,
                    eval_ds,
                    eval_final_indices,
                    eval_batch_size=cfg.eval_batch_size,
                    input_steps=input_steps,
                    target_steps=target_steps,
                    task_cfg=task_cfg,
                    dt=dt_train,
                    use_rollout_loss=use_rollout_loss,
                    rollout_weight=cfg.rollout_weight,
                    progress_label=f"eval@step{step}",
                )
                eval_losses.append((step, eval_metrics["total"]))
                eval_details.append(
                    {
                        "step": step,
                        "total": eval_metrics["total"],
                        "one_step": eval_metrics["one_step"],
                        "rollout": eval_metrics["rollout"],
                    }
                )
                print(
                    f"[eval] step {step} total {eval_metrics['total']:.6f} "
                    f"one_step {eval_metrics['one_step']:.6f} rollout {eval_metrics['rollout']:.6f}"
                )
                # Persist metadata immediately after each evaluation.
                save_logs(
                    out_dir,
                    train_losses,
                    eval_losses,
                    eval_details,
                    step_times,
                    mem_usage,
                    actual_usage,
                    epoch_summaries,
                )

            if step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    out_dir,
                    params=params,
                    step=step,
                    model_cfg=model_cfg,
                    task_cfg=task_cfg,
                    description=ckpt_in.description,
                    license_text=ckpt_in.license,
                )

        epoch_mean = float(np.mean(epoch_loss_accum)) if epoch_loss_accum else float("nan")
        epoch_one_mean = float(np.mean(epoch_one_accum)) if epoch_one_accum else float("nan")
        epoch_roll_mean = float(np.mean(epoch_roll_accum)) if epoch_roll_accum else float("nan")

        epoch_eval = run_eval(
            transformed,
            params,
            state,
            rng,
            eval_ds,
            eval_final_indices,
            eval_batch_size=cfg.eval_batch_size,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt_train,
            use_rollout_loss=use_rollout_loss,
            rollout_weight=cfg.rollout_weight,
            progress_label=f"eval@epoch{epoch+1}",
        )
        eval_losses.append((step, epoch_eval["total"]))
        eval_details.append(
            {
                "step": step,
                "epoch": epoch + 1,
                "total": epoch_eval["total"],
                "one_step": epoch_eval["one_step"],
                "rollout": epoch_eval["rollout"],
            }
        )

        epoch_summaries.append(
            {
                "epoch": epoch + 1,
                "steps": step - epoch_start_step + 1,
                "train_loss_mean": epoch_mean,
                "train_one_step_mean": epoch_one_mean,
                "train_rollout_mean": epoch_roll_mean,
                "eval_total_end": epoch_eval["total"],
                "eval_one_step_end": epoch_eval["one_step"],
                "eval_rollout_end": epoch_eval["rollout"],
                "time_per_step_mean": float(
                    np.mean([t for s, t in step_times if s >= epoch_start_step] or [float("nan")])
                ),
                "mem_gib_max": float(np.max([m for s, m in mem_usage if s >= epoch_start_step] or [float("nan")])),
                "rss_gib_max": float(
                    np.max(
                        [
                            float(x["proc_rss_gib"])
                            for x in actual_usage
                            if int(x["step"]) >= epoch_start_step and x.get("proc_rss_gib") is not None
                        ]
                        or [float("nan")]
                    )
                ),
                "gpu_mem_gib_max": float(
                    np.max(
                        [
                            float(x["gpu_mem_gib"])
                            for x in actual_usage
                            if int(x["step"]) >= epoch_start_step and x.get("gpu_mem_gib") is not None
                        ]
                        or [float("nan")]
                    )
                ),
            }
        )
        print(
            f"[epoch {epoch + 1}] train_total {epoch_mean:.6f} "
            f"train_one_step {epoch_one_mean:.6f} train_rollout {epoch_roll_mean:.6f} "
            f"eval_total {epoch_eval['total']:.6f}"
        )
        save_logs(
            out_dir,
            train_losses,
            eval_losses,
            eval_details,
            step_times,
            mem_usage,
            actual_usage,
            epoch_summaries,
        )

    final_eval = run_eval(
        transformed,
        params,
        state,
        rng,
        eval_ds,
        eval_final_indices,
        eval_batch_size=cfg.eval_batch_size,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
        use_rollout_loss=use_rollout_loss,
        rollout_weight=cfg.rollout_weight,
        progress_label="eval@final",
    )
    eval_losses.append((step, final_eval["total"]))
    eval_details.append(
        {
            "step": step,
            "final": True,
            "total": final_eval["total"],
            "one_step": final_eval["one_step"],
            "rollout": final_eval["rollout"],
        }
    )
    save_checkpoint(
        out_dir,
        params=params,
        step=step,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        description=ckpt_in.description,
        license_text=ckpt_in.license,
    )
    save_logs(
        out_dir,
        train_losses,
        eval_losses,
        eval_details,
        step_times,
        mem_usage,
        actual_usage,
        epoch_summaries,
    )
    plot_val_loss(out_dir, eval_losses)
    rss_vals = [float(x["proc_rss_gib"]) for x in actual_usage if x.get("proc_rss_gib") is not None]
    gpu_vals = [float(x["gpu_mem_gib"]) for x in actual_usage if x.get("gpu_mem_gib") is not None]
    print(
        "Actual usage summary: "
        f"rss_peak={float(np.max(rss_vals)) if rss_vals else float('nan'):.3f} GiB, "
        f"rss_avg={float(np.mean(rss_vals)) if rss_vals else float('nan'):.3f} GiB, "
        f"gpu_peak={float(np.max(gpu_vals)) if gpu_vals else float('nan'):.3f} GiB, "
        f"gpu_avg={float(np.mean(gpu_vals)) if gpu_vals else float('nan'):.3f} GiB"
    )
    print(
        f"Done. Final eval total {final_eval['total']:.6f}, "
        f"one_step {final_eval['one_step']:.6f}, rollout {final_eval['rollout']:.6f}. "
        f"Outputs in {out_dir}"
    )


if __name__ == "__main__":
    main()
