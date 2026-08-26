#!/usr/bin/env python3
"""Rank the matched eight-anchor res1 cosine-pilot screening evaluations."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import xarray as xr

from scripts.analyze_models.summarize_v22_final_res1_swa_low_lr import (
    equal_variable_rmse,
    load,
    weighted_allvars_mse,
)


ROOT = Path("artifacts/checkpoints/v22_final")
RUN_ROOT = ROOT / "res1_cosine_di16_di64_8k_20260817"
REFERENCE_ROOT = (
    ROOT
    / "res1_dm_7yr_k20_di_bcg_20k_20260813"
    / "di16_bcg1_closed_sg_stateful_20k"
)
STATS = Path("data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
OUTPUT = RUN_ROOT / "screen8_evaluation_summary.csv"
ARTIFACTS = (
    *(f"step{step}" for step in range(1_000, 8_001, 1_000)),
    "swa_step02000-04000",
    "swa_step02000-06000",
    "swa_step04000-08000",
    "swa_step02000-08000",
)
RUNS = ("di16_bcg1_cosine_8k", "di64_bcg4_cosine_8k")


def specs() -> list[tuple[str, Path]]:
    result = []
    for run in RUNS:
        for artifact in ARTIFACTS:
            result.append(
                (run + "/" + artifact, RUN_ROOT / run / "eval/screen8_cold_full_zero" / f"{artifact}.json")
            )
    result.append(
        (
            "incumbent_di16_bcg1/swa_step02000-08000",
            REFERENCE_ROOT
            / "eval/screen8_cold_full_zero/incumbent_swa_step02000-08000.json",
        )
    )
    return result


def main() -> None:
    paths = specs()
    missing = [str(path) for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing screening evaluations:\n" + "\n".join(missing))
    loaded = [(label, path, load(path)) for label, path in paths]
    anchors = {tuple(value["chosen_idx"]) for _, _, value in loaded}
    horizons = {int(value["target_steps"]) for _, _, value in loaded}
    samples = {int(value["evaluated_samples"]) for _, _, value in loaded}
    if len(anchors) != 1 or horizons != {40} or samples != {8}:
        raise ValueError("Screens are not matched on eight anchors and 40 leads")

    scales = xr.load_dataset(STATS)
    rows = []
    for label, path, value in loaded:
        baseline_mse, full_mse = weighted_allvars_mse(value, scales)
        weighted_curve = 100.0 * (1.0 - full_mse / baseline_mse)
        rmse_curve = equal_variable_rmse(value)
        original = value["original_graphcast_loss"]
        rows.append(
            {
                "label": label,
                "original_graphcast_loss_improvement_pct": float(
                    original["improvement_pct_rollout"]
                ),
                "weighted_allvars_mse_improvement_pct": float(
                    100.0 * (1.0 - np.mean(full_mse) / np.mean(baseline_mse))
                ),
                "equal_variable_rmse_improvement_pct": float(np.mean(rmse_curve)),
                "day4_original_graphcast_loss_improvement_pct": float(
                    original["improvement_pct_per_step"][15]
                ),
                "day10_original_graphcast_loss_improvement_pct": float(
                    original["improvement_pct_per_step"][39]
                ),
                "day4_weighted_allvars_mse_improvement_pct": float(weighted_curve[15]),
                "day10_weighted_allvars_mse_improvement_pct": float(weighted_curve[39]),
                "source_json": str(path),
            }
        )
    rows.sort(
        key=lambda row: float(row["original_graphcast_loss_improvement_pct"]),
        reverse=True,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank:2d}. {row['label']}: original GC "
            f"{row['original_graphcast_loss_improvement_pct']:.3f}%, "
            f"weighted_allvars MSE {row['weighted_allvars_mse_improvement_pct']:.3f}%, "
            f"equal-variable RMSE {row['equal_variable_rmse_improvement_pct']:.3f}%"
        )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
