#!/usr/bin/env python3
"""Plot the cold-full accuracy trajectory across di64/BCG4 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EVAL_DIR = Path(
    "artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck/"
    "di64_bcg4_closed_sg_stateful_20k/eval/cold_full_zero"
)
DEFAULT_OUTPUT_DIR = Path(
    "plots/analyze_models/images/resolution_eval/"
    "v22_final_res2_gc500k_k12_di_crosscheck_20k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument(
        "--steps", type=int, nargs="+",
        default=(2000, 4000, 6000, 8000, 10000, 12000, 14000),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)
    required = {"chosen_idx", "target_steps", "per_variable_per_step"}
    missing = sorted(required - set(evaluation))
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    if "2m_temperature" not in evaluation["per_variable_per_step"]:
        raise ValueError(f"{path} has no 2m-temperature metrics")
    return evaluation


def allvar_improvement(evaluation: dict) -> np.ndarray:
    curves = []
    for metrics in evaluation["per_variable_per_step"].values():
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        residual = np.asarray(metrics["rmse_full"], dtype=float)
        curves.append(100.0 * (1.0 - residual / baseline))
    return np.mean(np.stack(curves), axis=0)


def two_m_rmse(evaluation: dict) -> np.ndarray:
    return np.asarray(
        evaluation["per_variable_per_step"]["2m_temperature"]["rmse_full"],
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    if len(set(args.steps)) != len(args.steps):
        raise ValueError(f"Duplicate checkpoint steps: {args.steps}")
    paths = [args.eval_dir / f"step{step}.json" for step in args.steps]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing evaluation JSONs:\n" + "\n".join(missing))
    evaluations = [load(path) for path in paths]
    chosen_indices = {tuple(item["chosen_idx"]) for item in evaluations}
    horizons = {int(item["target_steps"]) for item in evaluations}
    if len(chosen_indices) != 1:
        raise ValueError("Checkpoint evaluations do not use the same sampled anchors")
    if len(horizons) != 1:
        raise ValueError(f"Checkpoint horizons differ: {sorted(horizons)}")

    horizon = horizons.pop()
    lead_days = np.arange(1, horizon + 1) * 0.25
    colours = plt.cm.plasma(np.linspace(0.12, 0.90, len(evaluations)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    caption = (
        "Stateful di64, B/C groups=4; closed-SG; initialized from vanilla GC500k.\n"
        "32 matched cold anchors; 40 six-hour leads; zero initial temporal state."
    )

    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    axis.axhline(0.0, color="0.35", lw=1.0, label="Vanilla GC500k")
    for colour, step, evaluation in zip(colours, args.steps, evaluations, strict=True):
        axis.plot(
            lead_days,
            allvar_improvement(evaluation),
            color=colour,
            lw=2.15,
            label=f"checkpoint {step // 1000}k",
        )
    axis.set(
        xlabel="forecast lead time (days)",
        ylabel="equal-variable relative RMSE reduction (%)",
        title="di64 BCG4: checkpoint step sweep, all variables",
        xlim=(0.25, lead_days[-1]),
    )
    axis.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1))
    axis.grid(alpha=0.28)
    axis.legend(loc="best", ncol=2, fontsize=8.7)
    axis.text(0.99, 0.015, caption, transform=axis.transAxes, ha="right", va="bottom", fontsize=7.8, color="0.25")
    fig.tight_layout()
    allvars_output = args.output_dir / "di64_bcg4_checkpoint_steps_allvars_relative_rmse_improvement.png"
    fig.savefig(allvars_output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    for colour, step, evaluation in zip(colours, args.steps, evaluations, strict=True):
        axis.plot(
            lead_days,
            two_m_rmse(evaluation),
            color=colour,
            lw=2.15,
            label=f"checkpoint {step // 1000}k",
        )
    axis.set(
        xlabel="forecast lead time (days)",
        ylabel="2 m-temperature RMSE (K)",
        title="di64 BCG4: checkpoint step sweep, 2 m temperature",
        xlim=(0.25, lead_days[-1]),
    )
    axis.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1))
    axis.grid(alpha=0.28)
    axis.legend(loc="best", ncol=2, fontsize=8.7)
    axis.text(0.99, 0.015, caption, transform=axis.transAxes, ha="right", va="bottom", fontsize=7.8, color="0.25")
    fig.tight_layout()
    two_m_output = args.output_dir / "di64_bcg4_checkpoint_steps_2m_temperature_rmse.png"
    fig.savefig(two_m_output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {allvars_output}")
    print(f"Saved {two_m_output}")


if __name__ == "__main__":
    main()
