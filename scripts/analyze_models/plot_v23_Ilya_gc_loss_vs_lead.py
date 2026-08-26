#!/usr/bin/env python3
"""Plot approximate original GraphCast-loss improvement from v23 JSON metrics.

The v23 selection JSONs predate exact ``original_graphcast_loss`` output.  This
reconstructs the objective from saved latitude-weighted channel MSEs, GraphCast
difference scales, pressure weights, and per-variable weights.  It is not exact
because the saved MSEs used cosine latitude weights, which assign zero weight
to pole centers, rather than GraphCast's finite pole-cell area weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


VARIABLE_WEIGHTS = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSON"),
        required=True,
        help="Curve label and v23 evaluation JSON; repeat for each curve.",
    )
    parser.add_argument("--diffs-stddev", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-loss-csv", type=Path)
    parser.add_argument("--output-2m-image", type=Path)
    parser.add_argument("--output-2m-csv", type=Path)
    return parser.parse_args()


def _scale_at_level(scales: xr.DataArray, level: int | None) -> float:
    if level is not None:
        scales = scales.sel(level=level)
    value = np.asarray(scales.values).squeeze()
    if value.size != 1:
        raise ValueError(f"Expected scalar difference scale, got shape {value.shape}")
    return float(value)


def _loss_by_lead(metrics: dict, scales: xr.Dataset, branch: str) -> np.ndarray:
    per_variable = metrics["per_variable_per_step"]
    per_channel = metrics["per_channel_per_step"]
    total = np.zeros(metrics["target_steps"], dtype=np.float64)

    for variable in per_variable:
        if variable not in scales:
            raise KeyError(f"No GraphCast difference scale for {variable!r}")
        scale = scales[variable]
        if "level" in scale.dims:
            levels = np.asarray(scale.level.values, dtype=np.float64)
            level_weights = levels / levels.sum()
            variable_loss = np.zeros_like(total)
            for level, weight in zip(levels.astype(int), level_weights, strict=True):
                channel = per_channel[f"{variable}_level{level}"]
                mse = np.square(np.asarray(channel[f"rmse_{branch}"], dtype=np.float64))
                variable_loss += weight * mse / _scale_at_level(scale, int(level)) ** 2
        else:
            mse = np.square(
                np.asarray(per_channel[variable][f"rmse_{branch}"], dtype=np.float64)
            )
            variable_loss = mse / _scale_at_level(scale, None) ** 2
        total += VARIABLE_WEIGHTS.get(variable, 1.0) * variable_loss
    return total


def main() -> None:
    args = _args()
    scales = xr.load_dataset(args.diffs_stddev)
    curves: dict[str, np.ndarray] = {}
    loss_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    temperature_curves: dict[str, np.ndarray] = {}
    lead_days: np.ndarray | None = None

    for label, json_path_string in args.series:
        metrics = json.loads(Path(json_path_string).read_text())
        baseline = _loss_by_lead(metrics, scales, "baseline")
        full = _loss_by_lead(metrics, scales, "full")
        loss_curves[label] = (baseline, full)
        curves[label] = 100.0 * (1.0 - full / baseline)
        temperature_curves[label] = np.asarray(
            metrics["per_variable_per_step"]["2m_temperature"][
                "improvement_pct_rmse"
            ],
            dtype=np.float64,
        )
        candidate_leads = 0.25 * np.arange(1, metrics["target_steps"] + 1)
        if lead_days is None:
            lead_days = candidate_leads
        elif not np.array_equal(lead_days, candidate_leads):
            raise ValueError("All series must have matching lead times")

    assert lead_days is not None
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lead_days", *curves])
        for index, lead in enumerate(lead_days):
            writer.writerow([f"{lead:.2f}", *(f"{values[index]:.10g}" for values in curves.values())])

    if args.output_loss_csv is not None:
        args.output_loss_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_loss_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "lead_days",
                    *(
                        column
                        for label in loss_curves
                        for column in (f"{label} baseline", f"{label} full")
                    ),
                ]
            )
            for index, lead in enumerate(lead_days):
                writer.writerow(
                    [
                        f"{lead:.2f}",
                        *(
                            f"{values[index]:.10g}"
                            for pair in loss_curves.values()
                            for values in pair
                        ),
                    ]
                )

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for label, values in curves.items():
        ax.plot(lead_days, values, linewidth=2.0, label=label)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("GraphCast loss improvement (%)")
    ax.set_title("v23 res0.25 cold full-feedback: approximate GraphCast loss improvement")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Checkpoint", ncols=2)
    fig.text(
        0.5,
        0.025,
        "Approximation reconstructed from saved channel MSEs; exact GraphCast pole-cell weighting is unavailable.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.19)
    fig.savefig(args.output_image, dpi=180)
    plt.close(fig)

    if (args.output_2m_image is None) != (args.output_2m_csv is None):
        raise ValueError("--output-2m-image and --output-2m-csv must be used together")
    if args.output_2m_image is not None and args.output_2m_csv is not None:
        args.output_2m_image.parent.mkdir(parents=True, exist_ok=True)
        args.output_2m_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_2m_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["lead_days", *temperature_curves])
            for index, lead in enumerate(lead_days):
                writer.writerow(
                    [
                        f"{lead:.2f}",
                        *(f"{values[index]:.10g}" for values in temperature_curves.values()),
                    ]
                )

        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        for label, values in temperature_curves.items():
            ax.plot(lead_days, values, linewidth=2.0, label=label)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.65)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel("2 m temperature RMSE improvement (%)")
        ax.set_title("v23 res0.25 cold full-feedback: 2 m temperature RMSE improvement")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Checkpoint", ncols=2)
        fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.13)
        fig.savefig(args.output_2m_image, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
