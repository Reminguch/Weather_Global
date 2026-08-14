#!/usr/bin/env python3
"""Compare matched stateful di16, di32, and di64/BCG4 checkpoint ages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/checkpoints/v22_final/res2_gc500k_k12_20k_di_crosscheck")
RUNS = {
    "di16": ROOT / "di16_closed_sg_stateful_20k/eval/cold_full_zero",
    "di32": ROOT / "di32_closed_sg_stateful_20k/eval/cold_full_zero",
    "di64 BCG4": ROOT / "di64_bcg4_closed_sg_stateful_20k/eval/cold_full_zero",
}
DEFAULT_OUTPUT_DIR = Path(
    "plots/analyze_models/images/resolution_eval/"
    "v22_final_res2_gc500k_k12_di_crosscheck_20k"
)
COLOURS = {"di16": "#31688E", "di32": "#35B779", "di64 BCG4": "#E76F51"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=(2000, 4000, 6000, 8000))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"chosen_idx", "target_steps", "per_variable_per_step"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return value


def allvar_improvement(evaluation: dict) -> np.ndarray:
    curves = []
    for metrics in evaluation["per_variable_per_step"].values():
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        residual = np.asarray(metrics["rmse_full"], dtype=float)
        curves.append(100.0 * (1.0 - residual / baseline))
    return np.mean(np.stack(curves), axis=0)


def two_m_full(evaluation: dict) -> np.ndarray:
    return np.asarray(
        evaluation["per_variable_per_step"]["2m_temperature"]["rmse_full"], dtype=float
    )


def two_m_baseline(evaluation: dict) -> np.ndarray:
    return np.asarray(
        evaluation["per_variable_per_step"]["2m_temperature"]["rmse_baseline"], dtype=float
    )


def validate(evaluations: dict[str, dict[int, dict]]) -> tuple[int, np.ndarray]:
    flat = [item for by_step in evaluations.values() for item in by_step.values()]
    chosen = {tuple(item["chosen_idx"]) for item in flat}
    horizons = {int(item["target_steps"]) for item in flat}
    if len(chosen) != 1:
        raise ValueError("Comparison evaluations do not use the same sampled anchors")
    if len(horizons) != 1:
        raise ValueError(f"Comparison horizons differ: {sorted(horizons)}")
    baselines = [two_m_baseline(item) for item in flat]
    if not all(np.allclose(baselines[0], candidate, rtol=1e-7, atol=1e-7) for candidate in baselines[1:]):
        raise ValueError("Comparison evaluations do not have identical GraphCast baselines")
    return horizons.pop(), baselines[0]


def make_figure(
    *,
    evaluations: dict[str, dict[int, dict]],
    steps: tuple[int, ...],
    lead_days: np.ndarray,
    output: Path,
    metric: str,
    baseline_2m: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), sharex=True)
    for axis, step in zip(axes.flat, steps, strict=True):
        if metric == "allvar":
            axis.axhline(0.0, color="0.2", lw=1.3, label="Vanilla GC500k")
            for label, by_step in evaluations.items():
                axis.plot(lead_days, allvar_improvement(by_step[step]), color=COLOURS[label], lw=2.25, label=label)
            axis.set_ylabel("equal-variable RMSE reduction (%)")
        else:
            axis.plot(lead_days, baseline_2m, color="0.2", lw=1.6, label="Vanilla GC500k")
            for label, by_step in evaluations.items():
                axis.plot(lead_days, two_m_full(by_step[step]), color=COLOURS[label], lw=2.25, label=label)
            axis.set_ylabel("2 m-temperature RMSE (K)")
        axis.set_title(f"checkpoint {step // 1000}k")
        axis.set_xlim(0.25, lead_days[-1])
        axis.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1))
        axis.grid(alpha=0.28)
    for axis in axes[-1]:
        axis.set_xlabel("forecast lead time (days)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    title = (
        "Stateful residual-Mamba width comparison: di16 vs di32 vs di64/BCG4"
        if metric == "allvar"
        else "Stateful residual-Mamba width comparison: 2 m temperature"
    )
    figure.suptitle(title, y=0.995)
    figure.text(
        0.5, 0.015,
        "Closed-SG; initialized from vanilla GC500k; 32 matched cold anchors; "
        "40 six-hour leads; zero initial temporal state.",
        ha="center", va="bottom", fontsize=8.4, color="0.25",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.885))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    steps = tuple(args.steps)
    if len(steps) != 4 or len(set(steps)) != 4:
        raise ValueError("This four-panel comparison requires four distinct checkpoint steps")
    evaluations: dict[str, dict[int, dict]] = {}
    for label, directory in RUNS.items():
        paths = {step: directory / f"step{step}.json" for step in steps}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {label} evaluation JSONs:\n" + "\n".join(missing))
        evaluations[label] = {step: load(path) for step, path in paths.items()}
    horizon, baseline_2m = validate(evaluations)
    lead_days = np.arange(1, horizon + 1) * 0.25
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_figure(
        evaluations=evaluations,
        steps=steps,
        lead_days=lead_days,
        output=args.output_dir / "stateful_di16_di32_di64_bcg4_common_steps_allvars.png",
        metric="allvar",
        baseline_2m=baseline_2m,
    )
    make_figure(
        evaluations=evaluations,
        steps=steps,
        lead_days=lead_days,
        output=args.output_dir / "stateful_di16_di32_di64_bcg4_common_steps_2m_temperature.png",
        metric="two_m",
        baseline_2m=baseline_2m,
    )
    print(f"Saved plots in {args.output_dir}")


if __name__ == "__main__":
    main()
