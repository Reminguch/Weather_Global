#!/usr/bin/env python3
"""Plot January 2022 NYC 2m-temperature fixed-lead trajectories for res2 GC-Mamba."""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
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
from scripts.analyze_models.legacy.graphcast_analysis_utils import (  # noqa: E402
    build_run_jitted,
    suppress_graphcast_future_warnings,
)
from src.models.graphcast.training.core.prepared_data import open_prepared_store  # noqa: E402


EXPERIMENT = "res2_ds16_gc_mamba_target_steps_bptt16_warm_leads1_9d"
DEFAULT_OUTPUT_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT / "ny_trajectory"
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT / "ny_trajectory"
DEFAULT_PREPARED_DATA_ROOT = ROOT / "data/graphcast/graphcast/dataset/prepared_stream"
DEFAULT_STATS_DIR = ROOT / "data/graphcast/graphcast/stats"
BASELINE_CKPT = (
    ROOT
    / "artifacts/checkpoints/7_years/vanilla_gc_mp6_continue20k"
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_continue20k/ckpt_best.npz"
)
MAMBA_ROOT = ROOT / "artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16"
MAMBA_CKPTS = {
    4: MAMBA_ROOT
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_target_step4_bptt16/ckpt_best.npz",
    8: MAMBA_ROOT
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_target_step8_bptt16/ckpt_best.npz",
    12: MAMBA_ROOT
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_target_step12_bptt16/ckpt_best.npz",
}
MAMBA_DI_K12_CKPTS = {
    16: MAMBA_ROOT
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di16_ds16_20k_target_step12_bptt16/ckpt_best.npz",
    64: MAMBA_ROOT
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_target_step12_bptt16/ckpt_best.npz",
    128: MAMBA_ROOT
    / "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di128_ds64_20k_target_step12_bptt16/ckpt_best.npz",
}
MAMBA_DI_K12_DS = {
    16: 16,
    64: 16,
    128: 64,
}
NYC_LAT = 40.7
NYC_LON = 286.0
HOURS_PER_STEP = 6
N_INPUT_STEPS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data-root", type=Path, default=DEFAULT_PREPARED_DATA_ROOT)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--lead-days", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument(
        "--plot-mode",
        choices=("target_steps", "di_k12"),
        default="target_steps",
        help="Which GC-Mamba comparison to plot.",
    )
    parser.add_argument("--eval-start", default="2022-01-01")
    parser.add_argument("--eval-end", default="2022-02-01")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lat", type=float, default=NYC_LAT)
    parser.add_argument("--lon", type=float, default=NYC_LON)
    parser.add_argument("--seed-base", type=int, default=9200)
    return parser.parse_args()


def _load_stats(stats_dir: Path) -> dict[str, xr.Dataset]:
    return {
        "diffs_stddev_by_level": xr.open_dataset(stats_dir / "diffs_stddev_by_level.nc").compute(),
        "mean_by_level": xr.open_dataset(stats_dir / "mean_by_level.nc").compute(),
        "stddev_by_level": xr.open_dataset(stats_dir / "stddev_by_level.nc").compute(),
    }


def _to_celsius(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size and np.nanmedian(values) > 200:
        return values - 273.15
    return values


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.abs(np.asarray(values, dtype=float) - float(target)).argmin())


def _load_checkpoint_runner(ckpt_path: Path, stats: dict[str, xr.Dataset]):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with ckpt_path.open("rb") as f:
        ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
    run_jitted, task_cfg, model_cfg, _run_cfg = build_run_jitted(ckpt_obj, stats, ckpt_path)
    if not np.isclose(float(model_cfg.resolution), 2.0, atol=1e-6):
        raise ValueError(f"Expected res2 checkpoint, got resolution={model_cfg.resolution}: {ckpt_path}")
    return run_jitted, task_cfg


def _truth_series(store, valid_indices: np.ndarray, *, lat_idx: int, lon_idx: int) -> np.ndarray:
    source = store.data_vars["2m_temperature"]
    time_axis = source.dims.index("time")
    lat_axis = source.dims.index("lat")
    lon_axis = source.dims.index("lon")
    arr = np.asarray(source.data)
    arr = np.take(arr, valid_indices, axis=time_axis)
    if time_axis != 0:
        arr = np.moveaxis(arr, time_axis, 0)
    # After moving time to front, lat/lon axes shift if they originally preceded time.
    dims = ["time", *[dim for dim in source.dims if dim != "time"]]
    lat_axis = dims.index("lat")
    lon_axis = dims.index("lon")
    return _to_celsius(np.take(np.take(arr, lat_idx, axis=lat_axis), lon_idx, axis=lon_axis - (1 if lat_axis < lon_axis else 0)))


def _predict_series(
    *,
    label: str,
    ckpt_path: Path,
    stats: dict[str, xr.Dataset],
    prepared_data_root: Path,
    valid_indices: np.ndarray,
    lead_steps: int,
    lat: float,
    lon: float,
    batch_size: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_jitted, task_cfg = _load_checkpoint_runner(ckpt_path, stats)
    store = open_prepared_store(prepared_data_root, 2, task_cfg, label=f"ny-{label}")
    lat_idx = _nearest_index(store.coords["lat"], lat)
    lon_idx = _nearest_index(store.coords["lon"], lon)
    origin_indices = valid_indices - int(lead_steps)
    if np.any(origin_indices < N_INPUT_STEPS - 1):
        raise ValueError(f"Lead {lead_steps} has origins before available context.")

    preds: list[np.ndarray] = []
    for start in range(0, len(origin_indices), batch_size):
        origins = origin_indices[start : start + batch_size]
        inputs, targets, forcings = store.build_batch_from_indices(
            indices=origins,
            input_steps=N_INPUT_STEPS,
            target_steps=lead_steps,
            task_cfg=task_cfg,
            dt=pd.Timedelta(hours=HOURS_PER_STEP),
        )
        with suppress_graphcast_future_warnings():
            pred = run_jitted(
                rng=jax.random.PRNGKey(seed_base + start),
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
            )
        point = pred["2m_temperature"].isel(time=-1, lat=lat_idx, lon=lon_idx).values
        preds.append(np.asarray(point, dtype=float))
        print(f"[{label}] lead_steps={lead_steps} {min(start + batch_size, len(origin_indices))}/{len(origin_indices)}", flush=True)
    pred_c = _to_celsius(np.concatenate(preds, axis=0))
    lat_value = float(np.asarray(store.coords["lat"])[lat_idx])
    lon_value = float(np.asarray(store.coords["lon"])[lon_idx])
    return pred_c, np.asarray([lat_value]), np.asarray([lon_value])


def _valid_indices_for_window(store, *, start: str, end: str, lead_steps: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times = pd.DatetimeIndex(pd.to_datetime(store.time.values))
    mask = (times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))
    valid = np.where(mask)[0]
    valid = valid[valid - int(lead_steps) >= N_INPUT_STEPS - 1]
    if valid.size == 0:
        raise ValueError(f"No valid target times for {start} to {end} at lead_steps={lead_steps}.")
    return times[valid], valid.astype(np.int64)


def _plot_target_steps(df: pd.DataFrame, *, lead_days: int, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.plot(df["valid_time"], df["truth_c"], color="#111111", linewidth=2.4, label="ERA5 truth")
    ax.plot(
        df["valid_time"],
        df["baseline_continue20k_c"],
        color="#767676",
        linewidth=2.0,
        linestyle="--",
        label="Continue20k vanilla",
    )
    colors = {4: "#2f6f9f", 8: "#b14b2d", 12: "#2f7d4f"}
    for target_step in (4, 8, 12):
        col = f"gc_mamba_di64_ds16_ts{target_step}_c"
        ax.plot(
            df["valid_time"],
            df[col],
            linewidth=1.8,
            color=colors[target_step],
            label=f"GC-Mamba di64/ds16 target {target_step}",
        )
    ax.set_title(f"NYC 2m temperature | fixed lead {lead_days}d | January 2022")
    ax.set_xlabel("Valid time")
    ax.set_ylabel("2m temperature (C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(image_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {image_path}")


def _plot_di_k12(df: pd.DataFrame, *, lead_days: int, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.plot(df["valid_time"], df["truth_c"], color="#111111", linewidth=2.4, label="ERA5 truth")
    ax.plot(
        df["valid_time"],
        df["baseline_continue20k_c"],
        color="#767676",
        linewidth=2.0,
        linestyle="--",
        label="Continue20k vanilla",
    )
    colors = {16: "#2f6f9f", 64: "#2f7d4f", 128: "#b14b2d"}
    for di in (16, 64, 128):
        ds = MAMBA_DI_K12_DS[di]
        col = f"gc_mamba_di{di}_ds{ds}_ts12_c"
        ax.plot(
            df["valid_time"],
            df[col],
            linewidth=1.8,
            color=colors[di],
            label=f"GC-Mamba di{di}/ds{ds} target 12",
        )
    ax.set_title(f"NYC 2m temperature | fixed lead {lead_days}d | January 2022 | target 12")
    ax.set_xlabel("Valid time")
    ax.set_ylabel("2m temperature (C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(image_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {image_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    stats = _load_stats(args.stats_dir)

    # Use the baseline task config to validate and index the common res2 prepared store.
    _baseline_runner, baseline_task_cfg = _load_checkpoint_runner(BASELINE_CKPT, stats)
    common_store = open_prepared_store(args.prepared_data_root, 2, baseline_task_cfg, label="ny-common")
    lat_idx = _nearest_index(common_store.coords["lat"], args.lat)
    lon_idx = _nearest_index(common_store.coords["lon"], args.lon)
    lat_value = float(np.asarray(common_store.coords["lat"])[lat_idx])
    lon_value = float(np.asarray(common_store.coords["lon"])[lon_idx])

    summary_rows: list[dict[str, object]] = []
    for lead_days in args.lead_days:
        lead_steps = int(lead_days) * 24 // HOURS_PER_STEP
        times, valid_indices = _valid_indices_for_window(
            common_store,
            start=args.eval_start,
            end=args.eval_end,
            lead_steps=lead_steps,
        )
        df = pd.DataFrame(
            {
                "valid_time": times,
                "truth_c": _truth_series(common_store, valid_indices, lat_idx=lat_idx, lon_idx=lon_idx),
            }
        )
        baseline_pred, _, _ = _predict_series(
            label="baseline_continue20k",
            ckpt_path=BASELINE_CKPT,
            stats=stats,
            prepared_data_root=args.prepared_data_root,
            valid_indices=valid_indices,
            lead_steps=lead_steps,
            lat=args.lat,
            lon=args.lon,
            batch_size=args.batch_size,
            seed_base=args.seed_base + lead_steps,
        )
        df["baseline_continue20k_c"] = baseline_pred
        if args.plot_mode == "target_steps":
            for target_step, ckpt_path in MAMBA_CKPTS.items():
                pred, _, _ = _predict_series(
                    label=f"gc_mamba_di64_ds16_ts{target_step}",
                    ckpt_path=ckpt_path,
                    stats=stats,
                    prepared_data_root=args.prepared_data_root,
                    valid_indices=valid_indices,
                    lead_steps=lead_steps,
                    lat=args.lat,
                    lon=args.lon,
                    batch_size=args.batch_size,
                    seed_base=args.seed_base + lead_steps + target_step * 100,
                )
                df[f"gc_mamba_di64_ds16_ts{target_step}_c"] = pred
            csv_name = f"nyc_2m_temperature_lead{lead_days}d_jan2022_timeseries.csv"
            image_name = f"nyc_2m_temperature_lead{lead_days}d_jan2022.png"
        elif args.plot_mode == "di_k12":
            for di, ckpt_path in MAMBA_DI_K12_CKPTS.items():
                ds = MAMBA_DI_K12_DS[di]
                pred, _, _ = _predict_series(
                    label=f"gc_mamba_di{di}_ds{ds}_ts12",
                    ckpt_path=ckpt_path,
                    stats=stats,
                    prepared_data_root=args.prepared_data_root,
                    valid_indices=valid_indices,
                    lead_steps=lead_steps,
                    lat=args.lat,
                    lon=args.lon,
                    batch_size=args.batch_size,
                    seed_base=args.seed_base + lead_steps + di * 100,
                )
                df[f"gc_mamba_di{di}_ds{ds}_ts12_c"] = pred
            csv_name = f"nyc_2m_temperature_lead{lead_days}d_jan2022_di_k12_timeseries.csv"
            image_name = f"nyc_2m_temperature_lead{lead_days}d_jan2022_di_k12.png"
        else:
            raise ValueError(f"Unknown plot mode: {args.plot_mode}")

        csv_path = args.output_dir / csv_name
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        image_path = args.image_dir / image_name
        if args.plot_mode == "target_steps":
            _plot_target_steps(df, lead_days=lead_days, image_path=image_path)
        else:
            _plot_di_k12(df, lead_days=lead_days, image_path=image_path)
        summary_rows.append(
            {
                "lead_days": lead_days,
                "lead_steps": lead_steps,
                "plot_mode": args.plot_mode,
                "n_points": len(df),
                "lat": lat_value,
                "lon": lon_value,
                "csv": str(csv_path),
                "image": str(image_path),
            }
        )

    if len(args.lead_days) == 1:
        summary_path = args.output_dir / f"nyc_2m_temperature_jan2022_{args.plot_mode}_lead{args.lead_days[0]}d_summary.csv"
    else:
        summary_path = args.output_dir / f"nyc_2m_temperature_jan2022_{args.plot_mode}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
