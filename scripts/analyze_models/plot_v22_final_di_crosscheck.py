#!/usr/bin/env python3
"""Plot GC500k residual-Mamba cold-rollout checkpoint comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HOURS_PER_STEP = 6
VARIABLE = "2m_temperature"
DEFAULT_NEW_DIR = Path(
    "artifacts/checkpoints/v22_final/"
    "res2_gc500k_k12_20k_di_crosscheck_20260812/"
    "di16_closed_sg_20k/eval/cold_full_zero"
)
DEFAULT_STATEFUL_DI16_DIR = Path(
    "artifacts/checkpoints/v22_final/"
    "res2_gc500k_k12_20k_di_crosscheck/"
    "di16_closed_sg_stateful_20k/eval/cold_full_zero"
)
DEFAULT_STATEFUL_DI32_DIR = Path(
    "artifacts/checkpoints/v22_final/"
    "res2_gc500k_k12_20k_di_crosscheck/"
    "di32_closed_sg_stateful_20k/eval/cold_full_zero"
)
DEFAULT_STATEFUL_DI64_BCG4_DIR = Path(
    "artifacts/checkpoints/v22_final/"
    "res2_gc500k_k12_20k_di_crosscheck/"
    "di64_bcg4_closed_sg_stateful_20k/eval/cold_full_zero"
)
DEFAULT_VANILLA = Path(
    "plots/analyze_models/data/resolution_eval/"
    "resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k/"
    "v22_final_eval_json/source_vanilla_step500k.json"
)
DEFAULT_EXISTING = Path(
    "plots/analyze_models/data/resolution_eval/"
    "resmamba_res2_v20_dc4_stateless_k12_from_gc500k_50k/"
    "v22_final_eval_json/current_closed_step50k.json"
)
DEFAULT_NEW_STEPS = (8000, 10000, 12000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-dir", type=Path, default=DEFAULT_NEW_DIR)
    parser.add_argument("--stateful-di16-dir", type=Path, default=DEFAULT_STATEFUL_DI16_DIR)
    parser.add_argument("--stateful-di32-dir", type=Path, default=DEFAULT_STATEFUL_DI32_DIR)
    parser.add_argument(
        "--stateful-di64-bcg4-dir", type=Path,
        default=DEFAULT_STATEFUL_DI64_BCG4_DIR,
    )
    parser.add_argument("--vanilla-json", type=Path, default=DEFAULT_VANILLA)
    parser.add_argument("--existing-json", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument(
        "--new-steps", type=int, nargs="+", default=DEFAULT_NEW_STEPS,
        help="Stateless new-pipeline checkpoint steps to retain.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "plots/analyze_models/images/resolution_eval/"
            "v22_final_res2_gc500k_k12_di_crosscheck_20k"
        ),
    )
    parser.add_argument(
        "--summary-csv", type=Path,
        default=Path(
            "plots/analyze_models/data/resolution_eval/"
            "v22_final_res2_gc500k_k12_di_crosscheck_20k/accuracy_curves.csv"
        ),
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    required = {"chosen_idx", "target_steps", "per_variable_per_step"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    if VARIABLE not in value["per_variable_per_step"]:
        raise ValueError(f"{path} has no {VARIABLE!r} metrics")
    return value


def checkpoint_step(path: Path) -> int:
    if not path.stem.startswith("step"):
        raise ValueError(f"Expected a stepNNNN.json file, got {path}")
    return int(path.stem.removeprefix("step"))


def selected(paths: list[Path], requested: tuple[int, ...], label: str) -> list[Path]:
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate requested {label} steps: {requested}")
    available = {checkpoint_step(path): path for path in paths}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"Missing {label} evaluations: {missing}")
    return [available[step] for step in requested]


def verify_matched(evaluations: list[dict]) -> int:
    chosen = {tuple(item["chosen_idx"]) for item in evaluations}
    if len(chosen) != 1:
        raise ValueError("Evaluation JSONs use different sampled anchors")
    horizons = {int(item["target_steps"]) for item in evaluations}
    if len(horizons) != 1:
        raise ValueError(f"Evaluation horizons differ: {sorted(horizons)}")
    return horizons.pop()


def allvar(evaluation: dict) -> np.ndarray:
    values = []
    for metrics in evaluation["per_variable_per_step"].values():
        baseline = np.asarray(metrics["rmse_baseline"], dtype=float)
        full = np.asarray(metrics["rmse_full"], dtype=float)
        values.append(100.0 * (1.0 - full / baseline))
    return np.mean(np.stack(values), axis=0)


def two_m(evaluation: dict) -> np.ndarray:
    value = np.asarray(
        evaluation["per_variable_per_step"][VARIABLE]["rmse_full"], dtype=float
    )
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Invalid 2m-temperature RMSE")
    return value


def write_summary(
    path: Path, lead_days: np.ndarray, vanilla: dict, existing: dict,
    stateless: list[tuple[int, dict]], stateful: list[tuple[int, dict]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    curves = [("stateless", step, item) for step, item in stateless]
    curves += [("stateful", step, item) for step, item in stateful]
    fields = ["lead_steps", "lead_days", "vanilla_gc500k_2m_rmse_k",
              "existing_stateless_di16_step50000_2m_rmse_k",
              "existing_stateless_di16_step50000_relative_allvar_rmse_improvement_pct"]
    for kind, step, _ in curves:
        fields += [f"{kind}_di16_step{step}_2m_rmse_k",
                   f"{kind}_di16_step{step}_relative_allvar_rmse_improvement_pct"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, day in enumerate(lead_days):
            row = {
                "lead_steps": index + 1, "lead_days": day,
                "vanilla_gc500k_2m_rmse_k": two_m(vanilla)[index],
                "existing_stateless_di16_step50000_2m_rmse_k": two_m(existing)[index],
                "existing_stateless_di16_step50000_relative_allvar_rmse_improvement_pct": allvar(existing)[index],
            }
            for kind, step, item in curves:
                row[f"{kind}_di16_step{step}_2m_rmse_k"] = two_m(item)[index]
                row[f"{kind}_di16_step{step}_relative_allvar_rmse_improvement_pct"] = allvar(item)[index]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    stateless_paths = selected(
        sorted(args.new_dir.glob("step*.json"), key=checkpoint_step),
        tuple(args.new_steps), "stateless new-pipeline",
    )
    stateful_paths = sorted(
        args.stateful_di16_dir.glob("step*.json"), key=checkpoint_step
    )
    if not stateful_paths:
        raise FileNotFoundError(
            f"No completed stateful di16 evaluations in {args.stateful_di16_dir}"
        )
    stateful_di32_paths = sorted(
        args.stateful_di32_dir.glob("step*.json"), key=checkpoint_step
    )
    stateful_di64_bcg4_paths = sorted(
        args.stateful_di64_bcg4_dir.glob("step*.json"), key=checkpoint_step
    )
    if not stateful_di32_paths or not stateful_di64_bcg4_paths:
        raise FileNotFoundError("Expected completed stateful di32 and di64-bcg4 evaluations")
    stateless = [(checkpoint_step(path), load(path)) for path in stateless_paths]
    stateful = [(checkpoint_step(path), load(path)) for path in stateful_paths]
    stateful_di32 = [(checkpoint_step(path), load(path)) for path in stateful_di32_paths]
    stateful_di64_bcg4 = [
        (checkpoint_step(path), load(path)) for path in stateful_di64_bcg4_paths
    ]
    vanilla, existing = load(args.vanilla_json), load(args.existing_json)
    horizon = verify_matched([
        vanilla, existing, *(x[1] for x in stateless), *(x[1] for x in stateful),
        *(x[1] for x in stateful_di32), *(x[1] for x in stateful_di64_bcg4),
    ])
    lead_days = np.arange(1, horizon + 1) * HOURS_PER_STEP / 24
    stateless_colours = plt.cm.viridis(np.linspace(0.18, 0.70, len(stateless)))
    stateful_colours = plt.cm.plasma(np.linspace(0.25, 0.75, len(stateful)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(14.5, 6.3))
    axis.plot(lead_days, np.zeros(horizon), color="0.18", lw=1.8, label="Vanilla GC500k")
    axis.plot(lead_days, allvar(existing), color="#984EA3", lw=1.9, label="Existing stateless di16, 50k")
    for colour, (step, item) in zip(stateless_colours, stateless, strict=True):
        axis.plot(lead_days, allvar(item), color=colour, lw=2.1, label=f"New stateless di16, {step // 1000}k")
    for colour, (step, item) in zip(stateful_colours, stateful, strict=True):
        axis.plot(lead_days, allvar(item), color=colour, lw=2.1, ls="--", label=f"New stateful di16, {step // 1000}k")
    for colour, (step, item) in zip(
        plt.cm.cividis(np.linspace(0.25, 0.80, len(stateful_di32))),
        stateful_di32,
        strict=True,
    ):
        axis.plot(lead_days, allvar(item), color=colour, lw=2.1, ls=":", label=f"New stateful di32, {step // 1000}k")
    for colour, (step, item) in zip(
        plt.cm.autumn(np.linspace(0.30, 0.85, len(stateful_di64_bcg4))),
        stateful_di64_bcg4,
        strict=True,
    ):
        axis.plot(lead_days, allvar(item), color=colour, lw=2.2, ls="-.", label=f"New stateful di64 B/C=4, {step // 1000}k")
    axis.axhline(0.0, color="0.55", lw=0.8)
    axis.set(xlabel="forecast lead time (days)", ylabel="equal-variable relative RMSE reduction (%)",
             title="All variables: stateful vs stateless residual-Mamba", xlim=(0.25, lead_days[-1]))
    axis.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1)); axis.grid(alpha=0.28)
    axis.legend(fontsize=7.1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.text(0.99, 0.015, "32 matched cold anchors; 40 six-hour leads; zero initial temporal state.\nAll residual-Mamba curves are closed-SG models initialized from vanilla GC500k.", transform=axis.transAxes, ha="right", va="bottom", fontsize=7.7, color="0.25")
    fig.tight_layout(rect=(0, 0, 0.73, 1))
    allvars_output = args.output_dir / "allvars_relative_rmse_improvement_vs_lead_day.png"
    fig.savefig(allvars_output, dpi=180, bbox_inches="tight"); plt.close(fig)

    di16_step4k = dict(stateful).get(4000)
    di64_bcg4_step4k = dict(stateful_di64_bcg4).get(4000)
    if di16_step4k is None or di64_bcg4_step4k is None:
        raise FileNotFoundError("Focused plot requires stateful di16 and di64-bcg4 step4000")
    fig, axis = plt.subplots(figsize=(9.8, 5.7))
    axis.plot(lead_days, np.zeros(horizon), color="0.18", lw=1.8, label="Vanilla GC500k")
    axis.plot(
        lead_days, allvar(di16_step4k), color="#31688E", lw=2.5,
        label="New stateful di16, 4k",
    )
    axis.plot(
        lead_days, allvar(di64_bcg4_step4k), color="#E76F51", lw=2.5, ls="-.",
        label="New stateful di64, B/C groups=4, 4k",
    )
    axis.axhline(0.0, color="0.55", lw=0.8)
    axis.set(
        xlabel="forecast lead time (days)",
        ylabel="equal-variable relative RMSE reduction (%)",
        title="All variables: stateful di16 vs B/C-grouped di64 at 4k",
        xlim=(0.25, lead_days[-1]),
    )
    axis.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1))
    axis.grid(alpha=0.28)
    axis.legend(fontsize=9.0, loc="best")
    axis.text(
        0.99, 0.015,
        "32 matched cold anchors; 40 six-hour leads; zero initial temporal state.\n"
        "Both closed-SG models are initialized from vanilla GC500k.",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=7.8, color="0.25",
    )
    fig.tight_layout()
    focused_output = args.output_dir / "allvars_stateful_di16_vs_di64_bcg4_step4k.png"
    fig.savefig(focused_output, dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.4, 6.1))
    axis.plot(lead_days, two_m(vanilla), color="0.18", lw=2.4, label="Vanilla GraphCast 500k")
    axis.plot(lead_days, two_m(existing), color="#984EA3", lw=1.9, label="Existing stateless di16, 50k")
    for colour, (step, item) in zip(stateless_colours, stateless, strict=True):
        axis.plot(lead_days, two_m(item), color=colour, lw=2.1, label=f"New stateless di16, {step // 1000}k")
    for colour, (step, item) in zip(stateful_colours, stateful, strict=True):
        axis.plot(lead_days, two_m(item), color=colour, lw=2.1, ls="--", label=f"New stateful di16, {step // 1000}k")
    axis.set(xlabel="forecast lead time (days)", ylabel="2 m temperature RMSE (K)",
             title="2 m-temperature cold rollout: stateful vs stateless residual-Mamba", xlim=(0.25, lead_days[-1]))
    axis.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1)); axis.grid(alpha=0.28)
    axis.legend(fontsize=8.2, loc="upper left")
    axis.text(0.99, 0.015, "32 matched cold anchors; 40 six-hour leads; zero initial temporal state.\nAll residual-Mamba curves are closed-SG di16 models initialized from vanilla GC500k.", transform=axis.transAxes, ha="right", va="bottom", fontsize=7.7, color="0.25")
    fig.tight_layout()
    two_m_output = args.output_dir / "cold_rmse_2m_temperature_vs_lead_day.png"
    fig.savefig(two_m_output, dpi=180, bbox_inches="tight"); plt.close(fig)

    write_summary(args.summary_csv, lead_days, vanilla, existing, stateless, stateful)
    print(f"Saved {allvars_output}")
    print(f"Saved {focused_output}")
    print(f"Saved {two_m_output}")
    print(f"Saved {args.summary_csv}")


if __name__ == "__main__":
    main()
