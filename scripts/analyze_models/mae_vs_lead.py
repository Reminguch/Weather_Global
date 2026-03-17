#!/usr/bin/env python3
"""Compute NYC MAE-vs-lead curves for GraphCast checkpoints.

Supports model groups:
- res1: latest checkpoints from artifacts/checkpoints/graphcast_res1_stream/res1_*
- res2: latest checkpoints from artifacts/checkpoints/graphcast_res2_stream/res2_*
- res4: latest checkpoints from artifacts/checkpoints/graphcast_res4_stream/res4_*
- res8: latest checkpoints from artifacts/checkpoints/graphcast_res8_stream/res8_*
- baseline: single baseline GraphCast_small checkpoint (res=1)

Outputs:
- plot PNG under plots/analyze_models/
- CSV with MAE per lead step and model label under plots/analyze_models/
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
BASELINE_CKPT_DEFAULT = (
    "data/graphcast/graphcast/params/"
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - "
    "precipitation input and output.npz"
)
STATS_DIR_DEFAULT = "data/graphcast/graphcast/stats"
OUTPUT_DIR_DEFAULT = "plots/analyze_models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MAE vs lead step for GraphCast model groups.")
    p.add_argument("--model-group", choices=["res1", "res2", "res4", "res8", "baseline"], required=True)
    p.add_argument("--dataset-dir", default=DATASET_DEFAULT, help="Path/URI for ERA5 dataset.")
    p.add_argument("--stats-dir", default=STATS_DIR_DEFAULT, help="Normalization stats directory.")
    p.add_argument("--baseline-ckpt", default=BASELINE_CKPT_DEFAULT, help="Baseline checkpoint file path.")
    p.add_argument("--n-eval-days", type=int, default=40, help="Number of days used for scoring window.")
    p.add_argument("--hours-per-step", type=int, default=6, help="Hours per forecast step.")
    p.add_argument("--n-input-steps", type=int, default=2, help="Input context steps.")
    p.add_argument(
        "--max-lead-steps",
        type=int,
        default=None,
        help="Max lead steps (default: 24 for res1/res2/res4/res8, 48 baseline).",
    )
    p.add_argument("--n-extra-steps", type=int, default=14, help="Extra buffer steps before scoring window.")
    p.add_argument("--discard-start-steps", type=int, default=None, help="Discard earliest origins from scoring.")
    p.add_argument("--discard-end-steps", type=int, default=0, help="Discard latest origins from scoring.")
    p.add_argument("--window-batch-size", type=int, default=8, help="Number of origins per rollout call.")
    p.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT, help="Output directory for plots/csv.")
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


def _prepare_nyc_dataset(
    dataset_dir: str,
    *,
    n_eval_days: int,
    hours_per_step: int,
    n_input_steps: int,
    max_lead_steps: int,
    n_extra_steps: int,
) -> tuple[xarray.Dataset, int, int]:
    n_steps_per_day = 24 // hours_per_step
    n_target_times = n_eval_days * n_steps_per_day
    n_context_steps = n_input_steps + max_lead_steps
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


def _build_eval_batch(
    ds_res: xarray.Dataset,
    *,
    idx_batch: list[int],
    task_cfg,
    n_input_steps: int,
    max_lead_steps: int,
    hours_per_step: int,
    lat_idx: int,
    lon_idx: int,
):
    lead_times_eval = slice(f"{hours_per_step}h", f"{max_lead_steps * hours_per_step}h")
    inputs_list, targets_list, forcings_list, real_list = [], [], [], []
    for target_idx in idx_batch:
        window = ds_res.isel(time=slice(target_idx - (n_input_steps + max_lead_steps) + 1, target_idx + 1))
        if window.sizes["time"] < n_input_steps + max_lead_steps:
            continue

        in_i, tgt_i, forc_i = data_utils.extract_inputs_targets_forcings(
            window,
            target_lead_times=lead_times_eval,
            **dataclasses.asdict(task_cfg),
        )
        for name in getattr(graphcast, "STATIC_VARS", ("geopotential_at_surface", "land_sea_mask")):
            if name in in_i and "time" in in_i[name].dims:
                in_i = in_i.assign({name: in_i[name].isel(time=0, drop=True)})

        inputs_list.append(in_i.isel(batch=0, drop=True))
        targets_list.append(tgt_i.isel(batch=0, drop=True))
        forcings_list.append(forc_i.isel(batch=0, drop=True))
        real_list.append(
            np.asarray(
                ds_res["2m_temperature"].isel(
                    time=slice(target_idx - max_lead_steps + 1, target_idx + 1),
                    batch=0,
                    lat=lat_idx,
                    lon=lon_idx,
                ).values
            )
        )

    if not inputs_list:
        return None

    b = len(inputs_list)
    inputs_b = xarray.concat(inputs_list, dim="batch").assign_coords(batch=np.arange(b))
    targets_b = xarray.concat(targets_list, dim="batch").assign_coords(batch=np.arange(b))
    forcings_b = xarray.concat(forcings_list, dim="batch").assign_coords(batch=np.arange(b))
    real_b = np.stack(real_list, axis=0)
    return inputs_b, targets_b, forcings_b, real_b


def _eval_group(
    *,
    ckpt_paths: list[Path],
    stats: dict[str, xarray.Dataset],
    ds_nyc: xarray.Dataset,
    start_target_idx: int,
    n_steps: int,
    max_lead_steps: int,
    hours_per_step: int,
    n_input_steps: int,
    window_batch_size: int,
    discard_start_steps: int,
    discard_end_steps: int,
    lat_nyc: float,
    lon_nyc: float,
):
    min_target_idx = n_input_steps + max_lead_steps - 1
    raw_target_indices = np.arange(max(start_target_idx, min_target_idx), n_steps, dtype=int)
    if discard_start_steps + discard_end_steps >= len(raw_target_indices):
        raise RuntimeError("Boundary discard removed all target indices.")
    target_indices = raw_target_indices[
        discard_start_steps : len(raw_target_indices) - discard_end_steps if discard_end_steps > 0 else None
    ].tolist()
    num_batches = (len(target_indices) + window_batch_size - 1) // window_batch_size
    print(
        f"Scoring origins: total={len(raw_target_indices)}, used={len(target_indices)} "
        f"(discard_start={discard_start_steps}, discard_end={discard_end_steps})"
    )

    results: list[tuple[str, np.ndarray]] = []
    for m_i, ckpt_path in enumerate(ckpt_paths, start=1):
        with ckpt_path.open("rb") as f:
            ckpt_obj = checkpoint.load(f, graphcast.CheckPoint)

        run_jitted, task_cfg, model_cfg = _build_run_jitted(ckpt_obj, stats)
        run_name = ckpt_path.parent.name if ckpt_path.parent != ckpt_path else ckpt_path.stem
        step_tag = ckpt_path.stem.replace("ckpt_step", "")
        label = f"{run_name} (step {step_tag})" if "ckpt_step" in ckpt_path.stem else run_name

        stride = max(1, int(round(float(getattr(model_cfg, "resolution", 1.0)))))
        ds_res = ds_nyc.isel(lat=slice(0, None, stride), lon=slice(0, None, stride)) if stride > 1 else ds_nyc
        lat_idx = int(np.argmin(np.abs(ds_res["lat"].values - lat_nyc)))
        lon_idx = int(np.argmin(np.abs(ds_res["lon"].values - lon_nyc)))

        abs_err_sum = np.zeros(max_lead_steps, dtype=float)
        counts = np.zeros(max_lead_steps, dtype=int)

        for b_i in range(num_batches):
            i0 = b_i * window_batch_size
            i1 = min((b_i + 1) * window_batch_size, len(target_indices))
            idx_batch = target_indices[i0:i1]
            batch_pack = _build_eval_batch(
                ds_res,
                idx_batch=idx_batch,
                task_cfg=task_cfg,
                n_input_steps=n_input_steps,
                max_lead_steps=max_lead_steps,
                hours_per_step=hours_per_step,
                lat_idx=lat_idx,
                lon_idx=lon_idx,
            )
            if batch_pack is None:
                continue
            inputs_b, targets_b, forcings_b, real_b = batch_pack
            pred_b = rollout.chunked_prediction(
                run_jitted,
                rng=jax.random.PRNGKey(100000 * m_i + b_i),
                inputs=inputs_b,
                targets_template=targets_b * np.nan,
                forcings=forcings_b,
            )
            pred_bt = np.asarray(
                pred_b["2m_temperature"].isel(lat=lat_idx, lon=lon_idx).transpose("batch", "time").values
            )
            n = min(max_lead_steps, pred_bt.shape[1], real_b.shape[1])
            if n == 0:
                continue
            err_bt = np.abs(pred_bt[:, :n] - real_b[:, :n])
            abs_err_sum[:n] += np.sum(err_bt, axis=0)
            counts[:n] += err_bt.shape[0]

        mae = np.divide(
            abs_err_sum,
            counts,
            out=np.full(max_lead_steps, np.nan, dtype=float),
            where=counts > 0,
        )
        results.append((label, mae))
        print(f"[{m_i}/{len(ckpt_paths)}] {label}: step1={mae[0]:.3f}, step{max_lead_steps}={mae[max_lead_steps-1]:.3f}")
    return results


def main() -> None:
    args = parse_args()
    model_group = args.model_group
    max_lead_steps = args.max_lead_steps or (48 if model_group == "baseline" else 24)
    discard_start_steps = args.discard_start_steps
    if discard_start_steps is None:
        discard_start_steps = args.n_extra_steps
    discard_end_steps = args.discard_end_steps

    ds_nyc, start_target_idx, n_steps = _prepare_nyc_dataset(
        args.dataset_dir,
        n_eval_days=args.n_eval_days,
        hours_per_step=args.hours_per_step,
        n_input_steps=args.n_input_steps,
        max_lead_steps=max_lead_steps,
        n_extra_steps=args.n_extra_steps,
    )
    stats = _load_stats(ROOT / args.stats_dir)
    lat_nyc, lon_nyc = 40.7, 360 - 74

    if model_group == "baseline":
        ckpt_paths = [ROOT / args.baseline_ckpt]
    elif model_group == "res1":
        ckpt_paths = _discover_latest_checkpoints(ROOT / "artifacts/checkpoints/graphcast_res1_stream", "res1")
    elif model_group == "res2":
        ckpt_paths = _discover_latest_checkpoints(ROOT / "artifacts/checkpoints/graphcast_res2_stream", "res2")
    elif model_group == "res8":
        ckpt_paths = _discover_latest_checkpoints(ROOT / "artifacts/checkpoints/graphcast_res8_stream", "res8")
    else:
        ckpt_paths = _discover_latest_checkpoints(ROOT / "artifacts/checkpoints/graphcast_res4_stream", "res4")
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoints found for group={model_group}")
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    results = _eval_group(
        ckpt_paths=ckpt_paths,
        stats=stats,
        ds_nyc=ds_nyc,
        start_target_idx=start_target_idx,
        n_steps=n_steps,
        max_lead_steps=max_lead_steps,
        hours_per_step=args.hours_per_step,
        n_input_steps=args.n_input_steps,
        window_batch_size=args.window_batch_size,
        discard_start_steps=discard_start_steps,
        discard_end_steps=discard_end_steps,
        lat_nyc=lat_nyc,
        lon_nyc=lon_nyc,
    )

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lead_steps = np.arange(1, max_lead_steps + 1)

    # Plot (blue gradient, ordered as checkpoint list).
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.cm.Blues
    colors = [cmap(v) for v in np.linspace(0.45, 0.95, len(results))]
    for i, (label, mae) in enumerate(results):
        ax.plot(lead_steps, mae, marker="o", markersize=2, alpha=0.9, color=colors[i], label=label)
    ax.set_xlabel(f"Lead step ({args.hours_per_step}h each)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title(f"NYC MAE vs lead step (1..{max_lead_steps}) for {model_group}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    plot_path = out_dir / f"nyc_mae_vs_lead_{model_group}.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    df = pd.DataFrame({"lead_step": lead_steps})
    for label, mae in results:
        df[label] = mae
    csv_path = out_dir / f"nyc_mae_vs_lead_{model_group}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved plot: {plot_path}")
    print(f"Saved csv:  {csv_path}")


if __name__ == "__main__":
    main()
