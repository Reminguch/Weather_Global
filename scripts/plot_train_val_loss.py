#!/usr/bin/env python3
"""
Update val_loss.png in each run directory under given checkpoint roots.
Reads eval_loss.json and train_loss.json, plots both train and validation loss
vs step, and saves val_loss.png (overwriting existing).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_VARIABLE_LOSS_WEIGHTS = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


def _load_pairs(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _load_train_series(out_dir: Path) -> tuple[list[int], list[float]]:
    train_data = _load_pairs(out_dir / "train_loss.json")
    if not train_data:
        return [], []

    first = train_data[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        steps = [int(x[0]) for x in train_data if isinstance(x, (list, tuple)) and len(x) >= 2]
        vals = [float(x[1]) for x in train_data if isinstance(x, (list, tuple)) and len(x) >= 2]
        return steps, vals

    # Backward compatibility: old train_loss.json format was just [loss, ...].
    vals = [float(x) for x in train_data]

    # Prefer step_times.json for true step axis if available.
    step_times = _load_pairs(out_dir / "step_times.json")
    step_axis = [int(x[0]) for x in step_times if isinstance(x, (list, tuple)) and len(x) >= 2]
    if len(step_axis) >= len(vals):
        return step_axis[: len(vals)], vals

    # Final fallback for legacy files with no step metadata.
    steps = [i + 1 for i in range(len(vals))]
    return steps, vals


def _load_legacy_train_log(out_dir: Path) -> tuple[list[int], list[float]]:
    """Load the per-step object records written by legacy residual-Mamba jobs."""
    path = out_dir / "train_log.json"
    if not path.exists():
        return [], []
    try:
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return [], []
    if not isinstance(records, list):
        return [], []
    pairs = [
        (int(record["step"]), float(record["loss"]))
        for record in records
        if isinstance(record, dict) and "step" in record and "loss" in record
    ]
    return [step for step, _ in pairs], [loss for _, loss in pairs]


def _load_rollout_eval_loss(
    out_dir: Path, stats_dir: Path
) -> tuple[list[int], list[float]]:
    """Reconstruct the residual training objective from v20/v22 rollout logs.

    The rollout evaluators save physical-unit, latitude-weighted RMSE for each
    channel.  Squaring it and dividing by ``diffs_stddev_by_level`` recovers
    the normalized residual MSE.  We then apply GraphCast's variable and
    pressure-level weights and average over the run's BPTT horizon, matching
    the scalar logged in ``train_log.json``.
    """
    eval_files = sorted(out_dir.glob("eval/*/step*.json"))
    if not eval_files or not stats_dir.is_dir():
        return [], []

    try:
        import numpy as np
        import xarray as xr

        with (out_dir / "run_config.json").open("r", encoding="utf-8") as f:
            config = json.load(f).get("config", {})
        feedback_mode = config.get("feedback_mode")
        expected_mode = "cold_full" if feedback_mode == "closed_loop_sg" else "cold_bp"
        selected = [p for p in eval_files if p.parent.name == expected_mode]
        if selected:
            eval_files = selected
        rollout_leads = int(config.get("bptt_steps", 0)) or None

        scales = xr.open_dataset(stats_dir / "diffs_stddev_by_level.nc")
        level_values = np.asarray(scales.level.values, dtype=float)
        level_weights = {
            int(level) if float(level).is_integer() else float(level): float(level / level_values.mean())
            for level in level_values
        }
    except Exception:
        return [], []

    pairs: list[tuple[int, float]] = []
    for path in eval_files:
        match = re.fullmatch(r"step(\d+)\.json", path.name)
        if not match:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                rollout = json.load(f)
            channels = rollout["per_channel_per_step"]
            total_by_lead = None
            for variable, scale_da in scales.data_vars.items():
                weight = _VARIABLE_LOSS_WEIGHTS.get(variable, 1.0)
                if "level" in scale_da.dims:
                    # GraphCast's loss averages across the level dimension
                    # after applying normalized pressure weights.  Do the
                    # same here; summing would inflate every atmospheric
                    # variable by the number of pressure levels (13 here).
                    variable_by_lead = None
                    n_levels = 0
                    for level in level_values:
                        level_key = int(level) if float(level).is_integer() else float(level)
                        key = f"{variable}_level{level_key}"
                        if key not in channels:
                            continue
                        scale = float(scale_da.sel(level=level).values)
                        mse = np.square(np.asarray(channels[key]["rmse_full"], dtype=float))
                        term = level_weights[level_key] * mse / (scale * scale)
                        variable_by_lead = term if variable_by_lead is None else variable_by_lead + term
                        n_levels += 1
                    if variable_by_lead is not None:
                        term = weight * variable_by_lead / n_levels
                        total_by_lead = term if total_by_lead is None else total_by_lead + term
                elif variable in channels:
                    scale = float(scale_da.values)
                    mse = np.square(np.asarray(channels[variable]["rmse_full"], dtype=float))
                    term = weight * mse / (scale * scale)
                    total_by_lead = term if total_by_lead is None else total_by_lead + term
            if total_by_lead is None:
                continue
            if rollout_leads is not None:
                total_by_lead = total_by_lead[:rollout_leads]
            pairs.append((int(match.group(1)), float(np.mean(total_by_lead))))
        except (KeyError, TypeError, ValueError, OSError):
            continue
    pairs.sort()
    return [step for step, _ in pairs], [value for _, value in pairs]


def plot_train_and_val_loss(out_dir: Path, stats_dir: Path) -> bool:
    """Plot train + val loss from JSONs in out_dir. Save val_loss.png. Return True if done."""
    eval_data = _load_pairs(out_dir / "eval_loss.json")
    eval_steps = [int(x[0]) for x in eval_data if isinstance(x, (list, tuple)) and len(x) >= 2]
    eval_vals = [float(x[1]) for x in eval_data if isinstance(x, (list, tuple)) and len(x) >= 2]
    if not eval_steps:
        eval_steps, eval_vals = _load_rollout_eval_loss(out_dir, stats_dir)
    train_steps, train_vals = _load_train_series(out_dir)
    if not train_steps:
        train_steps, train_vals = _load_legacy_train_log(out_dir)
    if not eval_steps and not train_steps:
        return False

    # Downsample train for plotting if very long (max ~5000 points).
    max_train_points = 5000
    if len(train_steps) > max_train_points:
        stride = max(1, len(train_steps) // max_train_points)
        train_steps = train_steps[::stride]
        train_vals = train_vals[::stride]

    fig, ax = plt.subplots()
    train_ax = ax
    use_secondary_train_axis = False
    if train_vals and eval_vals:
        eval_lo, eval_hi = min(eval_vals), max(eval_vals)
        use_secondary_train_axis = max(train_vals) < eval_lo or min(train_vals) > eval_hi
        if use_secondary_train_axis:
            train_ax = ax.twinx()
    if train_steps and train_vals:
        train_ax.plot(train_steps, train_vals, alpha=0.7, label="Train loss", color="C0")
        if use_secondary_train_axis:
            train_ax.set_ylabel("train loss", color="C0")
            train_ax.tick_params(axis="y", labelcolor="C0")
    if eval_steps:
        ax.plot(eval_steps, eval_vals, marker="o", linestyle="-", label="Val loss", color="C1")
        # Use a shared range whenever both curves overlap, so neither trace is
        # clipped.  A separate train axis is used above for disjoint ranges.
        y_values = list(eval_vals)
        if not use_secondary_train_axis:
            y_values.extend(train_vals)
        y_min = min(y_values)
        y_max = max(y_values)
        y_span = y_max - y_min
        pad = 0.1 * y_span if y_span > 0 else max(1e-8, 0.1 * abs(y_max))
        lo = max(0.0, y_min - pad)
        hi = y_max + pad
        if hi > lo:
            ax.set_ylim(lo, hi)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss" if use_secondary_train_axis else "loss")
    title = "Train & validation loss" if eval_steps else "Training loss (validation not recorded)"
    if use_secondary_train_axis:
        title += " (separate axes)"
    ax.set_title(title)
    if use_secondary_train_axis:
        lines = ax.get_lines() + train_ax.get_lines()
        ax.legend(lines, [line.get_label() for line in lines])
    else:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss.png")
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Checkpoint root dirs (e.g. artifacts/checkpoints/graphcast_res2_stream)",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("data/graphcast/graphcast/stats"),
        help="GraphCast statistics used to reconstruct v20/v22 rollout validation loss.",
    )
    args = parser.parse_args()

    updated = 0
    for root in args.roots:
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            continue
        # Include both modern evaluation records and legacy training records.
        run_dirs = sorted({p.parent for p in root.rglob("eval_loss.json")} | {p.parent for p in root.rglob("train_log.json")})
        if (root / "eval_loss.json").exists() or (root / "train_log.json").exists():
            run_dirs = sorted(set(run_dirs + [root]))
        for run_dir in run_dirs:
            if plot_train_and_val_loss(run_dir, args.stats_dir):
                print(f"Updated: {run_dir}")
                updated += 1
    print(f"Updated {updated} run(s).")


if __name__ == "__main__":
    main()
