#!/usr/bin/env python3
"""Plot exact original GraphCast-loss improvement from v24 evaluation JSONs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("LABEL", "JSON"),
        required=True,
        help="Curve label and v24 evaluation JSON; repeat for each curve.",
    )
    parser.add_argument("--output-image", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-loss-csv", type=Path)
    parser.add_argument("--output-2m-image", type=Path)
    parser.add_argument("--output-2m-csv", type=Path)
    return parser.parse_args()


def _loss_by_lead(metrics: dict, branch: str) -> np.ndarray:
    section = metrics.get("original_graphcast_loss")
    if not isinstance(section, dict):
        raise ValueError("Evaluation JSON has no exact original_graphcast_loss section")
    key = f"{branch}_per_step"
    if key not in section:
        raise ValueError(f"Evaluation JSON has no {key!r} exact loss values")
    values = np.asarray(section[key], dtype=np.float64)
    if values.shape != (int(metrics["target_steps"]),):
        raise ValueError(f"Unexpected {key} shape: {values.shape}")
    return values


def main() -> None:
    args = _args()
    curves: dict[str, np.ndarray] = {}
    loss_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    temperature_curves: dict[str, np.ndarray] = {}
    lead_days: np.ndarray | None = None

    for label, json_path_string in args.series:
        metrics = json.loads(Path(json_path_string).read_text())
        baseline = _loss_by_lead(metrics, "baseline")
        full = _loss_by_lead(metrics, "full")
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
    ax.set_title("v24 res0.25 cold full-feedback: exact GraphCast loss improvement")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Checkpoint", ncols=2)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.13)
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
        ax.set_title("v24 res0.25 cold full-feedback: 2 m temperature RMSE improvement")
        ax.grid(True, alpha=0.25)
        ax.legend(title="Checkpoint", ncols=2)
        fig.subplots_adjust(left=0.10, right=0.985, top=0.91, bottom=0.13)
        fig.savefig(args.output_2m_image, dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
