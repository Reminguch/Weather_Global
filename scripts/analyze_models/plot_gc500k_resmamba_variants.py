#!/usr/bin/env python3
"""Plot the two completed gc500k residual-Mamba cold-rollout evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIABLE = "2m_temperature"
HOURS_PER_STEP = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-json", type=Path, required=True)
    parser.add_argument("--closed-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    return parser.parse_args()


def load_eval(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)
    values = np.asarray(
        evaluation["per_channel_per_step"][VARIABLE]["rmse_full"],
        dtype=float,
    )
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid {VARIABLE} RMSE curve in {path}")
    evaluation["_rmse"] = values
    return evaluation


def main() -> None:
    args = parse_args()
    open_eval = load_eval(args.open_json)
    closed_eval = load_eval(args.closed_json)

    if tuple(open_eval["chosen_idx"]) != tuple(closed_eval["chosen_idx"]):
        raise ValueError("Residual evaluations do not use identical sampled anchors")
    if int(open_eval["target_steps"]) != int(closed_eval["target_steps"]):
        raise ValueError("Residual evaluation horizons differ")

    open_rmse = open_eval["_rmse"]
    closed_rmse = closed_eval["_rmse"]
    if len(open_rmse) != len(closed_rmse):
        raise ValueError("Residual RMSE curve lengths differ")

    lead_steps = np.arange(1, len(open_rmse) + 1)
    lead_days = lead_steps * HOURS_PER_STEP / 24
    n_samples = len(open_eval["chosen_idx"])

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.plot(
        lead_days,
        open_rmse,
        lw=2.5,
        color="#0072B2",
        label=f"Open feedback, trained AR-tail k=12 (cold, n={n_samples})",
    )
    ax.plot(
        lead_days,
        closed_rmse,
        lw=2.5,
        color="#D55E00",
        label=f"Closed-SG feedback, trained AR-tail k=12 (cold, n={n_samples})",
    )
    ax.set(
        xlabel="lead time (days)",
        ylabel="2 m temperature RMSE (K)",
        title="Residual-Mamba variants initialized from vanilla GraphCast 500k",
        xlim=(0.25, lead_days[-1]),
    )
    ax.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Final residual checkpoints at step 50k; same 32 cold anchors; "
        "40-step (10-day) evaluation horizon.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.28",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "lead_steps": lead_steps,
            "lead_hours": lead_steps * HOURS_PER_STEP,
            "lead_days": lead_days,
            "residual_mamba_open_rmse_k": open_rmse,
            "residual_mamba_closed_sg_rmse_k": closed_rmse,
            "closed_sg_minus_open_rmse_k": closed_rmse - open_rmse,
        }
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    print(f"Saved plot: {args.output}")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
