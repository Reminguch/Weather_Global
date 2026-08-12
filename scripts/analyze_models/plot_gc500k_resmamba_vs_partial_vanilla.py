#!/usr/bin/env python3
"""Plot residual-Mamba, vanilla controls, and DeepMind res1 cold rollouts."""

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
    parser.add_argument(
        "--earlier-open-json",
        type=Path,
        help="Optional earlier residual-Mamba open-feedback evaluation.",
    )
    parser.add_argument(
        "--earlier-closed-json",
        type=Path,
        help="Optional earlier residual-Mamba closed-SG evaluation.",
    )
    parser.add_argument("--ar-json", type=Path, required=True)
    parser.add_argument("--swa-open-json", type=Path)
    parser.add_argument("--swa-closed-json", type=Path)
    parser.add_argument("--swa-ar-json", type=Path)
    parser.add_argument("--vanilla-json", type=Path, required=True)
    parser.add_argument(
        "--deepmind-csv",
        type=Path,
        required=True,
        help="DeepMind GraphCast-small res1 2 m-temperature RMSE reference CSV.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    return parser.parse_args()


def load_eval(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)
    if VARIABLE not in evaluation.get("per_channel_per_step", {}):
        raise ValueError(f"{path} has no {VARIABLE!r} channel metrics")
    return evaluation


def rmse_curve(evaluation: dict, key: str) -> np.ndarray:
    values = np.asarray(
        evaluation["per_channel_per_step"][VARIABLE][key],
        dtype=float,
    )
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid {VARIABLE} {key} curve")
    return values


def load_deepmind_reference(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    rows = pd.read_csv(path)
    required = {"lead_steps", "eval_mode", "metric_kind", "variable", "value", "n_points"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    rows = rows.loc[
        (rows["eval_mode"] == "cold")
        & (rows["metric_kind"] == "rmse_k")
        & (rows["variable"] == VARIABLE)
    ].copy()
    if rows.empty:
        raise ValueError(f"No cold {VARIABLE} rmse_k rows in {path}")
    rows["lead_steps"] = pd.to_numeric(rows["lead_steps"], errors="raise")
    rows["value"] = pd.to_numeric(rows["value"], errors="raise")
    rows = rows.sort_values("lead_steps")
    if rows["lead_steps"].duplicated().any() or not np.isfinite(rows["value"]).all():
        raise ValueError(f"Invalid DeepMind reference rows in {path}")
    point_counts = rows["n_points"].dropna().unique()
    if len(point_counts) != 1:
        raise ValueError(f"DeepMind reference has inconsistent n_points: {point_counts.tolist()}")
    return (
        rows["lead_steps"].to_numpy(dtype=int),
        rows["value"].to_numpy(dtype=float),
        int(point_counts[0]),
    )


def main() -> None:
    args = parse_args()
    open_eval = load_eval(args.open_json)
    closed_eval = load_eval(args.closed_json)
    ar_eval = load_eval(args.ar_json)
    vanilla_eval = load_eval(args.vanilla_json)
    swa_args = (args.swa_open_json, args.swa_closed_json, args.swa_ar_json)
    if any(path is None for path in swa_args) and any(path is not None for path in swa_args):
        raise ValueError("Pass all three SWA evaluation JSONs, or none")
    swa_open_eval = load_eval(args.swa_open_json) if args.swa_open_json is not None else None
    swa_closed_eval = load_eval(args.swa_closed_json) if args.swa_closed_json is not None else None
    swa_ar_eval = load_eval(args.swa_ar_json) if args.swa_ar_json is not None else None
    if (args.earlier_open_json is None) != (args.earlier_closed_json is None):
        raise ValueError("Pass both earlier residual-Mamba evaluation JSONs, or neither")
    earlier_open_eval = (
        load_eval(args.earlier_open_json) if args.earlier_open_json is not None else None
    )
    earlier_closed_eval = (
        load_eval(args.earlier_closed_json)
        if args.earlier_closed_json is not None
        else None
    )
    deepmind_steps, deepmind_rmse, deepmind_n_points = load_deepmind_reference(
        args.deepmind_csv
    )

    matched_evaluations = [
        open_eval,
        closed_eval,
        ar_eval,
        vanilla_eval,
    ]
    if earlier_open_eval is not None and earlier_closed_eval is not None:
        matched_evaluations.extend((earlier_open_eval, earlier_closed_eval))
    if swa_open_eval is not None and swa_closed_eval is not None and swa_ar_eval is not None:
        matched_evaluations.extend((swa_open_eval, swa_closed_eval, swa_ar_eval))
    chosen = [
        tuple(item["chosen_idx"])
        for item in matched_evaluations
    ]
    if len(set(chosen)) != 1:
        raise ValueError("Evaluations do not use identical sampled anchors")
    horizons = {
        int(item["target_steps"])
        for item in matched_evaluations
    }
    if len(horizons) != 1:
        raise ValueError(f"Evaluation horizons differ: {sorted(horizons)}")

    open_rmse = rmse_curve(open_eval, "rmse_full")
    closed_rmse = rmse_curve(closed_eval, "rmse_full")
    ar_rmse = rmse_curve(ar_eval, "rmse_baseline")
    vanilla_rmse = rmse_curve(vanilla_eval, "rmse_baseline")
    swa_open_rmse = rmse_curve(swa_open_eval, "rmse_full") if swa_open_eval is not None else None
    swa_closed_rmse = (
        rmse_curve(swa_closed_eval, "rmse_full") if swa_closed_eval is not None else None
    )
    swa_ar_rmse = rmse_curve(swa_ar_eval, "rmse_baseline") if swa_ar_eval is not None else None
    earlier_open_rmse = (
        rmse_curve(earlier_open_eval, "rmse_full")
        if earlier_open_eval is not None
        else None
    )
    earlier_closed_rmse = (
        rmse_curve(earlier_closed_eval, "rmse_full")
        if earlier_closed_eval is not None
        else None
    )
    earlier_vanilla_rmse = (
        rmse_curve(earlier_open_eval, "rmse_baseline")
        if earlier_open_eval is not None
        else None
    )
    lengths = {len(open_rmse), len(closed_rmse), len(ar_rmse), len(vanilla_rmse)}
    if earlier_open_rmse is not None and earlier_closed_rmse is not None:
        lengths.update(
            (len(earlier_open_rmse), len(earlier_closed_rmse), len(earlier_vanilla_rmse))
        )
    if swa_open_rmse is not None and swa_closed_rmse is not None and swa_ar_rmse is not None:
        lengths.update((len(swa_open_rmse), len(swa_closed_rmse), len(swa_ar_rmse)))
    if len(lengths) != 1:
        raise ValueError(f"RMSE curve lengths differ: {sorted(lengths)}")
    if np.any(deepmind_steps < 1) or np.any(deepmind_steps > len(open_rmse)):
        raise ValueError("DeepMind reference leads fall outside the residual rollout horizon")

    lead_steps = np.arange(1, len(open_rmse) + 1)
    lead_days = lead_steps * HOURS_PER_STEP / 24
    n_samples = len(chosen[0])

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.plot(
        lead_days,
        open_rmse,
        lw=2.4,
        color="#0072B2",
        label=f"Residual-Mamba open feedback, trained AR-tail k=12 (cold, n={n_samples})",
    )
    ax.plot(
        lead_days,
        closed_rmse,
        lw=2.4,
        color="#D55E00",
        label=f"Residual-Mamba closed-SG feedback, trained AR-tail k=12 (cold, n={n_samples})",
    )
    if swa_open_rmse is not None and swa_closed_rmse is not None and swa_ar_rmse is not None:
        ax.plot(
            lead_days,
            swa_open_rmse,
            lw=2.0,
            ls="-.",
            color="#0072B2",
            label=f"Residual-Mamba open SWA 42k–50k (cold, n={n_samples})",
        )
        ax.plot(
            lead_days,
            swa_closed_rmse,
            lw=2.0,
            ls="-.",
            color="#D55E00",
            label=f"Residual-Mamba closed-SG SWA 42k–50k (cold, n={n_samples})",
        )
    if earlier_open_rmse is not None and earlier_closed_rmse is not None:
        ax.plot(
            lead_days,
            earlier_vanilla_rmse,
            lw=2.2,
            ls=":",
            color="0.18",
            label=(
                "Earlier vanilla GraphCast 50k run, best step 48k "
                f"(cold, n={n_samples})"
            ),
        )
        ax.plot(
            lead_days,
            earlier_open_rmse,
            lw=2.1,
            ls="--",
            color="#56B4E9",
            label=(
                "Earlier residual-Mamba open feedback, trained from vanilla 50k "
                f"(best step 16k; cold, n={n_samples})"
            ),
        )
        ax.plot(
            lead_days,
            earlier_closed_rmse,
            lw=2.1,
            ls="--",
            color="#984EA3",
            label=(
                "Earlier residual-Mamba closed-SG, trained from vanilla 50k "
                f"(best step 16k; cold, n={n_samples})"
            ),
        )
    ax.plot(
        lead_days,
        ar_rmse,
        lw=2.4,
        color="#E69F00",
        label=f"Vanilla GraphCast AR-k12, best step 24k (cold, n={n_samples})",
    )
    if swa_ar_rmse is not None:
        ax.plot(
            lead_days,
            swa_ar_rmse,
            lw=2.0,
            ls="-.",
            color="#E69F00",
            label=f"Vanilla GraphCast AR-k12 SWA 18k–26k (cold, n={n_samples})",
        )
    ax.plot(
        lead_days,
        vanilla_rmse,
        lw=2.4,
        color="#009E73",
        label=f"Source vanilla GraphCast, checkpoint 500k (cold, n={n_samples})",
    )
    ax.plot(
        deepmind_steps * HOURS_PER_STEP / 24,
        deepmind_rmse,
        lw=2.2,
        ls="--",
        marker="o",
        ms=4,
        color="#CC79A7",
        label=(
            "DeepMind GraphCast-small, res1 "
            f"(cold 2022 reference, n={deepmind_n_points})"
        ),
    )
    ax.set(
        xlabel="lead time (days)",
        ylabel="2 m temperature RMSE (K)",
        title="Cold 2 m-temperature rollout: res2 models and DeepMind res1 reference",
        xlim=(0.25, lead_days[-1]),
    )
    ax.set_xticks(np.arange(1, int(np.ceil(lead_days[-1])) + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.6, loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Res2 curves: same 32 anchors and 40 six-hour leads. Earlier vanilla/residual "
        "runs use the 50k training run's validation-best step-48k checkpoint; current "
        "residual variants: final step 50k; AR-k12: validation-best step 24k; source "
        "vanilla GraphCast: checkpoint at step 500k. SWA curves uniformly average the "
        "five latest selected checkpoints. "
        "Dashed DeepMind res1 curve: full-year 2022 cold evaluation (not anchor-matched).",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.28",
        wrap=True,
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
            "vanilla_graphcast_ar_k12_best_step24000_rmse_k": ar_rmse,
            "vanilla_graphcast_source_step500000_rmse_k": vanilla_rmse,
        }
    )
    if swa_open_rmse is not None and swa_closed_rmse is not None and swa_ar_rmse is not None:
        summary["residual_mamba_open_swa_step42k_50k_rmse_k"] = swa_open_rmse
        summary["residual_mamba_closed_sg_swa_step42k_50k_rmse_k"] = swa_closed_rmse
        summary["vanilla_graphcast_ar_k12_swa_step18k_26k_rmse_k"] = swa_ar_rmse
    if earlier_open_rmse is not None and earlier_closed_rmse is not None:
        summary["earlier_vanilla_graphcast_50k_run_best_step48000_rmse_k"] = (
            earlier_vanilla_rmse
        )
        summary["earlier_residual_mamba_open_from_vanilla50k_best_step16000_rmse_k"] = (
            earlier_open_rmse
        )
        summary[
            "earlier_residual_mamba_closed_sg_from_vanilla50k_best_step16000_rmse_k"
        ] = earlier_closed_rmse
    deepmind_column = "deepmind_graphcast_small_res1_full_year_2022_rmse_k"
    summary[deepmind_column] = np.nan
    summary.loc[deepmind_steps - 1, deepmind_column] = deepmind_rmse
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    print(f"Saved plot: {args.output}")
    print(f"Saved summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
