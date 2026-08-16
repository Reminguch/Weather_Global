#!/usr/bin/env python3
"""Summarize position-matched v23 sparse-loss learning-rate smokes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


HORIZONS = (1, 4, 8, 12, 16, 20)
WEIGHTS = np.asarray((1, 1, 2, 2, 4, 8), dtype=np.float64) / 18.0
EXPECTED_STEPS = 200
WARMUP_STEPS = 20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for config_path in sorted(root.glob("*/run_config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "train_metrics.jsonl"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        objective = config["objective"]
        if objective.get("loss_mode") != "sparse_steps":
            continue
        if tuple(objective.get("supervised_horizons", ())) != HORIZONS:
            raise ValueError(f"Unexpected horizons in {config_path}")
        configured_weights = np.asarray(
            objective.get("supervised_weights", ()), dtype=np.float64
        )
        if not np.array_equal(configured_weights, WEIGHTS * 18.0):
            raise ValueError(f"Unexpected weights in {config_path}")
        records = []
        if metrics_path.is_file():
            records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        steps = [int(record["step"]) for record in records]
        if steps != list(range(1, len(records) + 1)):
            raise ValueError(f"Non-consecutive steps in {metrics_path}")
        runs.append(
            {
                "learning_rate": float(config["optimizer"]["learning_rate"]),
                "run_name": run_dir.name,
                "run_dir": run_dir,
                "records": records,
            }
        )
    if not runs:
        raise ValueError(f"No sparse-loss runs found under {root}")
    return sorted(runs, key=lambda run: run["learning_rate"])


def _metric_values(records: list[dict[str, Any]], name: str) -> np.ndarray:
    if name == "loss":
        return np.asarray([record["loss"] for record in records], dtype=np.float64)
    if name == "gradient_norm":
        return np.asarray(
            [record["gradient_norm"] for record in records], dtype=np.float64
        )
    horizon = name.removeprefix("h")
    return np.asarray(
        [record["loss_by_horizon"][horizon] for record in records],
        dtype=np.float64,
    )


def _rolling_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    if values.size < window:
        return np.asarray([], dtype=np.float64)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    records = run["records"]
    components = np.asarray(
        [
            [record["loss_by_horizon"][str(horizon)] for horizon in HORIZONS]
            for record in records
        ],
        dtype=np.float64,
    )
    aggregate = _metric_values(records, "loss")
    gradient_norm = _metric_values(records, "gradient_norm")
    reconstructed = components @ WEIGHTS if records else np.asarray([])
    all_values = (
        np.concatenate((aggregate, gradient_norm, components.reshape(-1)))
        if records
        else np.asarray([])
    )
    finite = bool(np.all(np.isfinite(all_values)))
    reconstruction_error = (
        float(np.max(np.abs(aggregate - reconstructed))) if records else None
    )

    h20 = components[:, -1] if records else np.asarray([])
    post_warmup = h20[WARMUP_STEPS:]
    h20_slope = (
        float(np.polyfit(np.arange(post_warmup.size), post_warmup, 1)[0])
        if post_warmup.size >= 2
        else None
    )
    early_mean = float(np.mean(h20[20:60])) if h20.size >= 60 else None
    late_mean = float(np.mean(h20[160:200])) if h20.size >= 200 else None
    sustained_h20_growth = (
        bool(h20_slope > 0 and late_mean > early_mean)
        if h20_slope is not None and early_mean is not None and late_mean is not None
        else None
    )
    complete = len(records) == EXPECTED_STEPS
    rejected = bool(not finite or (sustained_h20_growth is True))
    return {
        "learning_rate": run["learning_rate"],
        "run_name": run["run_name"],
        "completed_steps": len(records),
        "complete": complete,
        "all_finite": finite,
        "max_aggregate_reconstruction_error": reconstruction_error,
        "post_warmup_h20_slope_per_step": h20_slope,
        "h20_mean_steps_21_60": early_mean,
        "h20_mean_steps_161_200": late_mean,
        "sustained_h20_growth": sustained_h20_growth,
        "rejected": rejected if complete else None,
    }


def _write_csv(runs: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "learning_rate",
        "run_name",
        "step",
        "loss",
        "gradient_norm",
        *(f"loss_h{horizon}" for horizon in HORIZONS),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for record in run["records"]:
                row = {
                    "learning_rate": run["learning_rate"],
                    "run_name": run["run_name"],
                    "step": record["step"],
                    "loss": record["loss"],
                    "gradient_norm": record["gradient_norm"],
                }
                row.update(
                    {
                        f"loss_h{horizon}": record["loss_by_horizon"][str(horizon)]
                        for horizon in HORIZONS
                    }
                )
                writer.writerow(row)


def _plot(runs: list[dict[str, Any]], path: Path) -> None:
    panels = (
        ("loss", "Weighted aggregate"),
        *((f"h{horizon}", f"Horizon {horizon}") for horizon in HORIZONS),
        ("gradient_norm", "Gradient norm"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True)
    colors = plt.get_cmap("viridis")(
        np.linspace(0.08, 0.9, max(len(runs), 2))[: len(runs)]
    )
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for color, run in zip(colors, runs, strict=True):
            values = _metric_values(run["records"], metric)
            steps = np.arange(1, values.size + 1)
            label = f"{run['learning_rate']:.0e}"
            axis.plot(steps, values, color=color, alpha=0.22, linewidth=0.8)
            rolling = _rolling_mean(values)
            if rolling.size:
                axis.plot(
                    np.arange(20, values.size + 1),
                    rolling,
                    color=color,
                    linewidth=2.0,
                    label=label,
                )
            elif values.size:
                axis.plot(steps, values, color=color, linewidth=1.5, label=label)
        axis.axvline(WARMUP_STEPS, color="0.35", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Update position")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Loss / norm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, title="Peak LR", loc="upper center", ncol=4)
    figure.suptitle(
        "v23 res0.25 sparse multi-horizon LR sweep (raw + 20-step mean)",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or args.root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.root)
    summaries = [_summarize_run(run) for run in runs]
    summary = {
        "expected_steps": EXPECTED_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "horizons": list(HORIZONS),
        "normalized_weights": WEIGHTS.tolist(),
        "rejection_rule": (
            "reject non-finite runs, or complete runs with both positive post-warmup "
            "horizon-20 slope and steps-161:200 mean above steps-21:60 mean"
        ),
        "runs": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(runs, output_dir / "position_matched_metrics.csv")
    _plot(runs, output_dir / "position_matched_curves.png")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
