#!/usr/bin/env python3
"""Plot 2 m-temperature physical RMSE for d_conv=4 SWA and GC references.

This combines the already-computed residual rollout JSON logs with the
resolution-eval CSVs.  The residual curves are the d_conv=4 SWA models; the
three GraphCast curves are selected by their exact checkpoint variants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESIDUAL_ROOT = ROOT / (
    "artifacts/checkpoints/7_years/"
    "resmamba_res2_v20_dinner16_layers2_loop_dconv_sweep_20k"
)
RES2_CSV = ROOT / (
    "plots/analyze_models/data/resolution_eval/"
    "res2_ds16_gc_mamba_release_all20k_warm_leads1_9d/resolution_eval.csv"
)
RES1_CSV = ROOT / (
    "plots/analyze_models/data/resolution_eval/"
    "gcsmall_full_year_2022_res1grid_leads1_10d/resolution_eval.csv"
)
DEFAULT_OUTPUT = ROOT / (
    "plots/analyze_models/images/resolution_eval/"
    "resmamba_res2_v20_dconv_loop_sweep/"
    "resmamba_dc4_swa_absolute_rmse_2m_temperature.png"
)
DEFAULT_COMPARISON_CSV = ROOT / (
    "plots/analyze_models/data/resolution_eval/"
    "res2_di16_ds16_ts12_frozen_cold_leads1_9d/resolution_eval.csv"
)
DEFAULT_COMPARISON_VARIANT = (
    "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_"
    "gc_mamba_tc2_di16_ds16_20k_target_step12_bptt16"
)

VANILLA_VARIANT = "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k"
VANILLA_CONTINUE_VARIANT = f"{VANILLA_VARIANT}_continue20k"


def load_swa(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read physical RMSE (K) from a residual rollout summary JSON."""
    with path.open(encoding="utf-8") as f:
        evaluation = json.load(f)
    values = np.asarray(
        evaluation["per_channel_per_step"]["2m_temperature"]["rmse_full"],
        dtype=float,
    )
    # The residual evaluator rolls forward in 6-hour increments.
    return 6 * np.arange(1, len(values) + 1), values, int(evaluation["n_samples"])


def load_csv_curve(csv_path: Path, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one physical 2 m-temperature RMSE curve from resolution-eval CSV."""
    data = pd.read_csv(csv_path)
    selected = data.loc[
        (data["variant"] == variant)
        & (data["metric_kind"] == "rmse_k")
        & (data["variable"] == "2m_temperature")
        & (data["metric_semantics"] == "physical_rmse_projected_grid_no_std")
    ].sort_values("lead_steps")
    if selected.empty:
        raise ValueError(f"No matching physical RMSE rows for {variant!r} in {csv_path}")
    return 6 * selected["lead_steps"].to_numpy(dtype=float), selected["value"].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="Optional cold GC-Mamba resolution-eval CSV to overlay.",
    )
    parser.add_argument("--comparison-variant", default=DEFAULT_COMPARISON_VARIANT)
    parser.add_argument("--comparison-label", default="Residual_Mamba (res2, cold)")
    parser.add_argument(
        "--comparison-summary-csv",
        type=Path,
        default=None,
        help="Optional CSV containing daily differences relative to the DC4-SWA curves.",
    )
    args = parser.parse_args()

    open_hours, open_rmse, open_samples = load_swa(
        RESIDUAL_ROOT
        / "resmamba_res2_v20_k14_di16_l2_dc4_open_20000steps/eval/cold_bp/"
        "swa_step4k-12k.json"
    )
    closed_hours, closed_rmse, closed_samples = load_swa(
        RESIDUAL_ROOT
        / "resmamba_res2_v20_k14_di16_l2_dc4_closed_sg_20000steps/eval/cold_full/"
        "swa_step2k-14k.json"
    )
    vanilla_hours, vanilla_rmse = load_csv_curve(RES2_CSV, VANILLA_VARIANT)
    continue_hours, continue_rmse = load_csv_curve(RES2_CSV, VANILLA_CONTINUE_VARIANT)
    res1 = pd.read_csv(RES1_CSV)
    res1_variant = res1.loc[
        (res1["family"] == "graphcast")
        & (res1["metric_kind"] == "rmse_k")
        & (res1["variable"] == "2m_temperature"),
        "variant",
    ].unique()
    if len(res1_variant) != 1:
        raise ValueError(f"Expected one res1 GraphCast variant, found: {res1_variant}")
    res1_hours, res1_rmse = load_csv_curve(RES1_CSV, res1_variant[0])

    comparison_hours: np.ndarray | None = None
    comparison_rmse: np.ndarray | None = None
    if args.comparison_csv is not None:
        comparison_hours, comparison_rmse = load_csv_curve(args.comparison_csv, args.comparison_variant)

    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.plot(open_hours / 24, open_rmse, lw=2.5, color="#0072B2",
            label=f"Additiva_Mamba d_conv=4 SWA, open ({open_samples} cold samples)")
    ax.plot(closed_hours / 24, closed_rmse, lw=2.5, color="#D55E00",
            label=f"Additiva_Mamba d_conv=4 SWA, closed ({closed_samples} cold samples)")
    ax.plot(vanilla_hours / 24, vanilla_rmse, "o-", ms=5, lw=1.9, color="#555555",
            label="Vanilla GC (res2, warm)")
    ax.plot(continue_hours / 24, continue_rmse, "s-", ms=5, lw=1.9, color="#009E73",
            label="Vanilla GC continue 20k (res2, warm)")
    ax.plot(res1_hours / 24, res1_rmse, "^-", ms=5, lw=1.9, color="#CC79A7",
            label="resolution 1 GC (projected, cold)")
    if comparison_hours is not None and comparison_rmse is not None:
        ax.plot(comparison_hours / 24, comparison_rmse, "P-", ms=6, lw=2.3, color="#7B3294",
                label=args.comparison_label)

    ax.set(
        xlabel="lead time (days)",
        ylabel="2 m temperature RMSE (K)",
        title="Absolute forecast error by lead time",
        xlim=(0.25, 10),
    )
    ax.set_xticks(np.arange(1, 11))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Physical RMSE; the res2 GC reference CSVs are warm while SWA and res1 GC are cold.\n"
        "The optional GC-Mamba overlay uses the evaluation mode recorded in its source CSV. "
        "Markers show 24-hour CSV endpoints; SWA is every 6 hours.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.28",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved: {args.output}")

    if args.comparison_summary_csv is not None:
        if comparison_hours is None or comparison_rmse is None:
            raise ValueError("--comparison-summary-csv requires --comparison-csv")
        open_at_comparison = np.interp(comparison_hours, open_hours, open_rmse)
        closed_at_comparison = np.interp(comparison_hours, closed_hours, closed_rmse)
        summary = pd.DataFrame(
            {
                "lead_days": comparison_hours / 24,
                "gc_mamba_rmse_k": comparison_rmse,
                "resmamba_dc4_swa_open_rmse_k": open_at_comparison,
                "resmamba_dc4_swa_closed_rmse_k": closed_at_comparison,
            }
        )
        for label in ("open", "closed"):
            baseline = summary[f"resmamba_dc4_swa_{label}_rmse_k"]
            summary[f"gc_mamba_minus_{label}_rmse_k"] = summary["gc_mamba_rmse_k"] - baseline
            summary[f"gc_mamba_vs_{label}_relative_error_pct"] = 100 * (
                summary["gc_mamba_rmse_k"] / baseline - 1
            )
        args.comparison_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.comparison_summary_csv, index=False)
        print(f"Saved comparison summary: {args.comparison_summary_csv}")


if __name__ == "__main__":
    main()
