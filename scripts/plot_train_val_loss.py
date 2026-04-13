#!/usr/bin/env python3
"""
Update val_loss.png in each run directory under given checkpoint roots.
Reads eval_loss.json and train_loss.json, plots both train and validation loss
vs step, and saves val_loss.png (overwriting existing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def plot_train_and_val_loss(out_dir: Path) -> bool:
    """Plot train + val loss from JSONs in out_dir. Save val_loss.png. Return True if done."""
    eval_data = _load_pairs(out_dir / "eval_loss.json")
    if not eval_data:
        return False

    eval_steps = [int(x[0]) for x in eval_data if isinstance(x, (list, tuple)) and len(x) >= 2]
    eval_vals = [float(x[1]) for x in eval_data if isinstance(x, (list, tuple)) and len(x) >= 2]
    if not eval_steps:
        return False

    train_steps, train_vals = _load_train_series(out_dir)

    # Downsample train for plotting if very long (max ~5000 points).
    max_train_points = 5000
    if len(train_steps) > max_train_points:
        stride = max(1, len(train_steps) // max_train_points)
        train_steps = train_steps[::stride]
        train_vals = train_vals[::stride]

    fig, ax = plt.subplots()
    if train_steps and train_vals:
        ax.plot(train_steps, train_vals, alpha=0.7, label="Train loss", color="C0")
    ax.plot(eval_steps, eval_vals, marker="o", linestyle="-", label="Val loss", color="C1")
    # Scale y-axis from validation-loss dynamics only.
    # Train loss may be clipped by this range, per intended behavior.
    y_min = min(eval_vals)
    y_max = max(eval_vals)
    y_span = y_max - y_min
    pad = 0.1 * y_span if y_span > 0 else max(1e-8, 0.1 * abs(y_max))
    lo = max(0.0, y_min - pad)
    hi = y_max + pad
    if hi > lo:
        ax.set_ylim(lo, hi)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Train & validation loss")
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
    args = parser.parse_args()

    updated = 0
    for root in args.roots:
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            continue
        # Include nested run directories that contain eval_loss.json.
        run_dirs = sorted({p.parent for p in root.rglob("eval_loss.json")})
        if (root / "eval_loss.json").exists():
            run_dirs = sorted(set(run_dirs + [root]))
        for run_dir in run_dirs:
            if plot_train_and_val_loss(run_dir):
                print(f"Updated: {run_dir}")
                updated += 1
    print(f"Updated {updated} run(s).")


if __name__ == "__main__":
    main()
