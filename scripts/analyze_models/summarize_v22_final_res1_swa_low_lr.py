#!/usr/bin/env python3
"""Summarize matched res1 SWA-continuation evaluations without mixing metrics."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path("artifacts/checkpoints/v22_final")
CONTINUATION_ROOT = ROOT / "res1_swa_low_lr_continuation_20260816"
REFERENCE_ROOT = (
    ROOT
    / "res1_dm_7yr_k20_di_bcg_20k_20260813"
    / "di16_bcg1_closed_sg_stateful_20k"
)
STATS = Path("data/graphcast/graphcast/stats/diffs_stddev_by_level.nc")
OUTPUT = CONTINUATION_ROOT / "evaluation_summary_allvars_mse.csv"
VARIABLE_WEIGHTS = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


def evaluation_specs() -> list[tuple[str, Path]]:
    specs = [
        (
            "initial_swa_2k_8k",
            REFERENCE_ROOT
            / "eval/cold_full_zero_exact/swa_step02000-08000.json",
        )
    ]
    for lr_tag in ("lr1em5", "lr3em5"):
        run = CONTINUATION_ROOT / f"di16_bcg1_swa2k8k_{lr_tag}_2k"
        for artifact in (
            "step500",
            "step1000",
            "step1500",
            "step2000",
            "swa_step00500-02000",
        ):
            specs.append((f"{lr_tag}_{artifact}", run / f"eval/cold_full_zero/{artifact}.json"))
    return specs


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"chosen_idx", "target_steps", "per_variable_per_step", "per_channel_per_step"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return value


def equal_variable_rmse(value: dict) -> np.ndarray:
    curves = []
    for metrics in value["per_variable_per_step"].values():
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        full = np.asarray(metrics["rmse_full"], dtype=float)
        curves.append(100.0 * (1.0 - full / baseline))
    return np.mean(np.stack(curves), axis=0)


def weighted_allvars_mse(value: dict, scales: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct legacy weighted_allvars normalized MSE from channel RMSEs.

    This uses equal level weighting within each atmospheric variable, matching
    ``normalized_weighted_mse_allvars``. It is distinct from original
    GraphCast loss, which uses pressure-proportional level weights.
    """
    grouped: dict[str, list[tuple[int | None, dict]]] = defaultdict(list)
    for channel, metrics in value["per_channel_per_step"].items():
        match = re.fullmatch(r"(.+)_level(\d+)", channel)
        if match:
            grouped[match.group(1)].append((int(match.group(2)), metrics))
        else:
            grouped[channel].append((None, metrics))

    totals = {
        "baseline": np.zeros(int(value["target_steps"]), dtype=float),
        "full": np.zeros(int(value["target_steps"]), dtype=float),
    }
    total_weight = 0.0
    for variable, channels in grouped.items():
        weight = float(VARIABLE_WEIGHTS.get(variable, 1.0))
        total_weight += weight
        for branch, field in (("baseline", "rmse_baseline"), ("full", "rmse_full")):
            normalized_channels = []
            for level, metrics in channels:
                scale_value = scales[variable]
                if level is not None:
                    scale_value = scale_value.sel(level=level)
                scale = float(scale_value.values)
                rmse = np.asarray(metrics[field], dtype=float)
                normalized_channels.append(rmse**2 / scale**2)
            totals[branch] += weight * np.mean(np.stack(normalized_channels), axis=0)

    return totals["baseline"] / total_weight, totals["full"] / total_weight


def main() -> None:
    available = [(label, path, load(path)) for label, path in evaluation_specs() if path.exists()]
    if not available:
        raise FileNotFoundError("No matching evaluation JSONs exist")

    anchors = {tuple(value["chosen_idx"]) for _, _, value in available}
    horizons = {int(value["target_steps"]) for _, _, value in available}
    if len(anchors) != 1 or horizons != {40}:
        raise ValueError("Evaluations are not matched on anchors and 40-lead horizon")

    scales = xr.load_dataset(STATS)
    rows = []
    for label, path, value in available:
        baseline_mse, full_mse = weighted_allvars_mse(value, scales)
        weighted_allvars_improvement = 100.0 * (1.0 - full_mse / baseline_mse)
        rmse_improvement = equal_variable_rmse(value)
        original = value.get("original_graphcast_loss")
        rows.append(
            {
                "label": label,
                "weighted_allvars_baseline_mse": float(np.mean(baseline_mse)),
                "weighted_allvars_full_mse": float(np.mean(full_mse)),
                "weighted_allvars_mse_improvement_pct": float(
                    100.0 * (1.0 - np.mean(full_mse) / np.mean(baseline_mse))
                ),
                "day4_weighted_allvars_mse_improvement_pct": float(
                    weighted_allvars_improvement[15]
                ),
                "day10_weighted_allvars_mse_improvement_pct": float(
                    weighted_allvars_improvement[39]
                ),
                "original_graphcast_loss_improvement_pct": (
                    float(original["improvement_pct_rollout"]) if original else ""
                ),
                "day4_original_graphcast_loss_improvement_pct": (
                    float(original["improvement_pct_per_step"][15]) if original else ""
                ),
                "day10_original_graphcast_loss_improvement_pct": (
                    float(original["improvement_pct_per_step"][39]) if original else ""
                ),
                "mean_equal_variable_rmse_improvement_pct": float(
                    np.mean(rmse_improvement)
                ),
                "day4_equal_variable_rmse_improvement_pct": float(rmse_improvement[15]),
                "day10_equal_variable_rmse_improvement_pct": float(rmse_improvement[39]),
                "source_json": str(path),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['label']}: weighted_allvars MSE "
            f"{row['weighted_allvars_mse_improvement_pct']:.4f}%, "
            f"original GC loss {row['original_graphcast_loss_improvement_pct']}, "
            f"equal-variable RMSE {row['mean_equal_variable_rmse_improvement_pct']:.4f}%"
        )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
