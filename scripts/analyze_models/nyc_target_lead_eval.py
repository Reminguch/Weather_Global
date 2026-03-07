#!/usr/bin/env python3
"""Evaluate NYC temperature at a fixed target lead for res4 checkpoints.

Replicates the notebook workflow:
- discover latest checkpoint per run under artifacts/checkpoints/graphcast_res4_stream/res4_*
- evaluate 2m temperature at NYC for a fixed lead (default 6 days)
- save trajectory plot and MAE summary under plots/analyze_models
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import re
import sys
from pathlib import Path

import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graphcast_dataset import open_graphcast_era5


def _require_graphcast() -> None:
    try:
        import graphcast  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "graphcast is required. Activate env via scripts/graphcast_env.sh."
        ) from exc


_require_graphcast()
from graphcast import autoregressive, casting, checkpoint, data_utils, graphcast, normalization, rollout


DATASET_DEFAULT = "data/graphcast/graphcast/dataset/source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc"
STATS_DIR_DEFAULT = "data/graphcast/graphcast/stats"
OUTPUT_DIR_DEFAULT = "plots/analyze_models"
CKPT_ROOT_DEFAULT = "artifacts/checkpoints/graphcast_res4_stream"
RUN_PREFIX_DEFAULT = "res4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NYC trajectory at fixed target lead.")
    p.add_argument("--dataset-dir", default=DATASET_DEFAULT, help="Path/URI for ERA5 dataset.")
    p.add_argument("--stats-dir", default=STATS_DIR_DEFAULT, help="Normalization stats directory.")
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Output directory.")
    p.add_argument("--ckpt-root", default=CKPT_ROOT_DEFAULT, help="Checkpoint root directory.")
    p.add_argument("--run-prefix", default=RUN_PREFIX_DEFAULT, help="Run prefix (default: res4).")
    p.add_argument("--target-lead-days", type=int, default=6, help="Forecast lead in days.")
    p.add_argument("--hours-per-step", type=int, default=6, help="Hours per forecast step.")
    p.add_argument("--n-input-steps", type=int, default=2, help="Input context steps.")
    p.add_argument("--n-eval-days", type=int, default=40, help="How many recent days to score.")
    p.add_argument("--n-extra-steps", type=int, default=14, help="Extra buffer steps before scoring.")
    p.add_argument("--lat-nyc", type=float, default=40.7, help="NYC latitude.")
    p.add_argument("--lon-nyc", type=float, default=286.0, help="NYC longitude in 0..360.")
    p.add_argument("--output-stem", default=None, help="Optional output stem override.")
    return p.parse_args()


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


def _prepare_dataset(
    dataset_dir: str,
    *,
    n_eval_days: int,
    hours_per_step: int,
    n_input_steps: int,
    target_lead_steps: int,
    n_extra_steps: int,
) -> tuple[xarray.Dataset, int, int]:
    n_steps_per_day = 24 // hours_per_step
    n_target_times = n_eval_days * n_steps_per_day
    n_context_steps = n_input_steps + target_lead_steps
    n_total_steps = n_context_steps + n_target_times + n_extra_steps

    ds = open_graphcast_era5(dataset_dir, time_slice=slice(-n_total_steps, None)).compute()
    if ds.sizes.get("lat") == 721 and ds.sizes.get("lon") == 1440:
        ds = ds.isel(lat=slice(0, None, 4), lon=slice(0, None, 4))
    ds = _ensure_datetime_coord(ds)

    n_steps = ds.sizes["time"]
    if n_steps < n_context_steps + n_target_times:
        n_target_times = n_steps - n_context_steps
    if n_target_times <= 0:
        raise ValueError(
            f"Dataset has only {n_steps} steps; requires at least {n_context_steps + n_steps_per_day}."
        )
    start_target_idx = n_steps - n_target_times
    return ds, start_target_idx, n_steps


def _sort_key_from_run_name(run_name: str) -> tuple[int, int, str]:
    m_w = re.search(r"_w(\d+)_", run_name)
    m_mp = re.search(r"_mp(\d+)_", run_name)
    width = int(m_w.group(1)) if m_w else 10**9
    mp = int(m_mp.group(1)) if m_mp else 10**9
    return (width, mp, run_name)


def _discover_latest_checkpoints(root_dir: Path, prefix: str) -> list[Path]:
    ckpts: list[Path] = []
    for run_dir in sorted(root_dir.glob(f"{prefix}_*")):
        if not run_dir.is_dir():
            continue
        candidates: list[tuple[int, Path]] = []
        for p in run_dir.glob("ckpt_step*.npz"):
            m = re.search(r"ckpt_step(\d+)$", p.stem)
            if m:
                candidates.append((int(m.group(1)), p))
        if candidates:
            ckpts.append(max(candidates, key=lambda x: x[0])[1])
    return sorted(ckpts, key=lambda p: _sort_key_from_run_name(p.parent.name))


def _build_run_jitted(ckpt_obj: graphcast.CheckPoint, stats: dict[str, xarray.Dataset]):
    model_cfg = ckpt_obj.model_config
    task_cfg = ckpt_obj.task_config
    params = ckpt_obj.params
    state = {}

    def construct_wrapped_graphcast(model_config, task_config):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"],
            stddev_by_level=stats["stddev_by_level"],
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=False)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_cfg, task_config=task_cfg)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    return drop_state(with_params(jax.jit(with_configs(run_forward.apply)))), task_cfg, model_cfg


def _to_celsius_if_needed(arr: np.ndarray) -> np.ndarray:
    if arr.size > 0 and np.median(arr) > 200:
        return arr - 273.15
    return arr


def main() -> None:
    args = parse_args()

    target_lead_hours = args.target_lead_days * 24
    if target_lead_hours % args.hours_per_step != 0:
        raise ValueError(
            f"target_lead_days={args.target_lead_days} is not divisible by hours_per_step={args.hours_per_step}."
        )
    target_lead_steps = target_lead_hours // args.hours_per_step
    n_context_steps = args.n_input_steps + target_lead_steps
    target_lead_times = f"{target_lead_hours}h"

    ds_nyc, start_target_idx, n_steps = _prepare_dataset(
        args.dataset_dir,
        n_eval_days=args.n_eval_days,
        hours_per_step=args.hours_per_step,
        n_input_steps=args.n_input_steps,
        target_lead_steps=target_lead_steps,
        n_extra_steps=args.n_extra_steps,
    )
    all_times_nyc = ds_nyc["time"].values[start_target_idx:n_steps]
    stats = _load_stats(ROOT / args.stats_dir)

    ckpt_paths = _discover_latest_checkpoints(ROOT / args.ckpt_root, args.run_prefix)
    if not ckpt_paths:
        raise RuntimeError(f"No ckpt_step*.npz found under {ROOT / args.ckpt_root} for prefix={args.run_prefix}")
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    print(
        f"Evaluating {len(ckpt_paths)} checkpoints at target lead {args.target_lead_days} days "
        f"({target_lead_hours}h, {target_lead_steps} steps)."
    )

    results: list[dict[str, np.ndarray | str]] = []
    for m_i, ckpt_path in enumerate(ckpt_paths, start=1):
        with ckpt_path.open("rb") as f:
            ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)
        run_jitted, task_cfg, model_cfg = _build_run_jitted(ckpt_obj, stats)

        res = float(getattr(model_cfg, "resolution", 1.0))
        stride = max(1, int(round(res)))
        if stride > 1:
            ds_res = ds_nyc.isel(lat=slice(0, None, stride), lon=slice(0, None, stride))
        else:
            ds_res = ds_nyc

        lat_idx = int(np.argmin(np.abs(ds_res["lat"].values - args.lat_nyc)))
        lon_idx = int(np.argmin(np.abs(ds_res["lon"].values - args.lon_nyc)))

        pred_nyc: list[float] = []
        real_nyc_res: list[float] = []
        for target_idx in range(start_target_idx, n_steps):
            window = ds_res.isel(time=slice(target_idx - n_context_steps + 1, target_idx + 1))
            in_i, tgt_i, forc_i = data_utils.extract_inputs_targets_forcings(
                window,
                target_lead_times=target_lead_times,
                **dataclasses.asdict(task_cfg),
            )
            for name in getattr(graphcast, "STATIC_VARS", ("geopotential_at_surface", "land_sea_mask")):
                if name in in_i and "time" in in_i[name].dims:
                    in_i = in_i.assign({name: in_i[name].isel(time=0, drop=True)})

            pred_i = rollout.chunked_prediction(
                run_jitted,
                rng=jax.random.PRNGKey(target_idx + 4000 + m_i),
                inputs=in_i,
                targets_template=tgt_i * np.nan,
                forcings=forc_i,
            )

            pred_nyc.append(float(pred_i["2m_temperature"].isel(time=-1, batch=0, lat=lat_idx, lon=lon_idx).values))
            real_nyc_res.append(float(ds_res["2m_temperature"].isel(time=target_idx, batch=0, lat=lat_idx, lon=lon_idx).values))

        pred_arr = _to_celsius_if_needed(np.asarray(pred_nyc, dtype=float))
        real_arr = _to_celsius_if_needed(np.asarray(real_nyc_res, dtype=float))

        step_tag = ckpt_path.stem.replace("ckpt_step", "")
        label = f"{ckpt_path.parent.name} (step {step_tag})"
        results.append({"label": label, "pred_nyc": pred_arr, "real_nyc_res": real_arr})
        print(f"[{m_i}/{len(ckpt_paths)}] {label}: N={len(pred_arr)} stride={stride} resolution={res}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or f"nyc_targetlead_{args.run_prefix}_{args.target_lead_days}d"

    fig, ax = plt.subplots(figsize=(12, 4))
    n_time = 0
    for item in results:
        pred_arr = np.asarray(item["pred_nyc"])
        real_arr = np.asarray(item["real_nyc_res"])
        n = min(len(all_times_nyc), len(pred_arr), len(real_arr))
        if n == 0:
            continue
        n_time = max(n_time, n)
        ax.plot(all_times_nyc[:n], pred_arr[:n], alpha=0.75, label=str(item["label"]))

    if results:
        real_arr = np.asarray(results[0]["real_nyc_res"])
        n = min(len(all_times_nyc), len(real_arr), n_time if n_time > 0 else len(real_arr))
        if n > 0:
            ax.plot(all_times_nyc[:n], real_arr[:n], linestyle="--", alpha=1.0, label="ERA5@grid", color="black")

    ax.set_ylabel("2 m temperature (C)")
    ax.set_xlabel("Valid time")
    ax.set_title(
        f"NYC (~40.7N, 74W) - {args.target_lead_days}-day lead, "
        f"{args.run_prefix}_stream checkpoints vs matched ERA5"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    plot_path = out_dir / f"{output_stem}.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    metrics_rows: list[dict[str, float | int | str]] = []
    for item in results:
        label = str(item["label"])
        pred_arr = np.asarray(item["pred_nyc"])
        real_arr = np.asarray(item["real_nyc_res"])
        n = min(len(all_times_nyc), len(pred_arr), len(real_arr))
        if n == 0:
            print(
                f"Skipping {label}: empty series "
                f"(pred={len(pred_arr)}, real={len(real_arr)}, time={len(all_times_nyc)})"
            )
            continue
        err = np.abs(pred_arr[:n] - real_arr[:n])
        metrics_rows.append(
            {
                "label": label,
                "mae_c": float(np.mean(err)),
                "max_error_c": float(np.max(err)),
                "n_points": int(n),
            }
        )

    metrics_rows.sort(key=lambda x: float(x["mae_c"]))
    csv_path = out_dir / f"{output_stem}_mae.csv"
    pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)

    print(f"Saved plot: {plot_path}")
    print(f"Saved MAE summary: {csv_path}")
    if metrics_rows:
        print(f"NYC {args.target_lead_days}-day lead ({args.run_prefix}) - mean absolute error (C), lower is better")
        for row in metrics_rows:
            print(
                f"{str(row['label']):55s} MAE={float(row['mae_c']):6.3f}  "
                f"MAX={float(row['max_error_c']):6.3f}  N={int(row['n_points'])}"
            )
    else:
        print("No comparable prediction/ERA5 points to score.")


if __name__ == "__main__":
    main()
