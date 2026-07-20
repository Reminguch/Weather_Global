#!/usr/bin/env python3
"""Plot checkpoint rollout evaluations for the residual-Mamba d-conv sweep.

Each run gets one image with paper/GraphCast-weighted normalized-MSE
improvement over the frozen baseline.  The source metrics are the existing
``eval/<mode>/step*.json`` rollout logs; no evaluation is launched here.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
SELECTED_LEADS_HOURS = (24, 72, 120, 240)


def _level_key(level: float) -> int | float:
    return int(level) if float(level).is_integer() else float(level)


def weighted_mse_improvement(
    evaluation: dict, scales: xr.Dataset
) -> np.ndarray:
    """Return GraphCast-weighted normalized-MSE improvement for every lead."""
    channels = evaluation["per_channel_per_step"]
    levels = np.asarray(scales.level.values, dtype=float)
    level_weights = {_level_key(level): level / levels.mean() for level in levels}
    baseline_total = full_total = None

    for variable, scale_da in scales.data_vars.items():
        variable_weight = VARIABLE_WEIGHTS.get(variable, 1.0)
        if "level" in scale_da.dims:
            baseline_var = full_var = None
            n_levels = 0
            for level in levels:
                key = f"{variable}_level{_level_key(level)}"
                if key not in channels:
                    continue
                scale = float(scale_da.sel(level=level).values)
                item = channels[key]
                baseline_mse = np.square(np.asarray(item["rmse_baseline"], dtype=float))
                full_mse = np.square(np.asarray(item["rmse_full"], dtype=float))
                factor = level_weights[_level_key(level)] / (scale * scale)
                baseline_var = baseline_mse * factor if baseline_var is None else baseline_var + baseline_mse * factor
                full_var = full_mse * factor if full_var is None else full_var + full_mse * factor
                n_levels += 1
            if not n_levels:
                continue
            baseline_term = variable_weight * baseline_var / n_levels
            full_term = variable_weight * full_var / n_levels
        elif variable in channels:
            scale = float(scale_da.values)
            item = channels[variable]
            baseline_term = variable_weight * np.square(np.asarray(item["rmse_baseline"], dtype=float)) / (scale * scale)
            full_term = variable_weight * np.square(np.asarray(item["rmse_full"], dtype=float)) / (scale * scale)
        else:
            continue
        baseline_total = baseline_term if baseline_total is None else baseline_total + baseline_term
        full_total = full_term if full_total is None else full_total + full_term

    if baseline_total is None or full_total is None:
        raise ValueError("No scored channels matched the statistics dataset.")
    return 100.0 * (1.0 - full_total / baseline_total)


def _load_run(
    run_dir: Path, scales: xr.Dataset
) -> tuple[list[int], list[np.ndarray], list[tuple[str, np.ndarray]], dict]:
    files = sorted(run_dir.glob("eval/*/step*.json"))
    steps, curves, metadata = [], [], {}
    for path in files:
        match = re.fullmatch(r"step(\d+)\.json", path.name)
        if not match:
            continue
        with path.open("r", encoding="utf-8") as f:
            evaluation = json.load(f)
        steps.append(int(match.group(1)))
        curves.append(weighted_mse_improvement(evaluation, scales))
        metadata = evaluation
    swa_curves: list[tuple[str, np.ndarray]] = []
    for path in sorted(run_dir.glob("eval/*/swa_step*.json")):
        with path.open("r", encoding="utf-8") as f:
            evaluation = json.load(f)
        tag = path.stem.removeprefix("swa_").replace("step", "step ").replace("k-", "k–")
        swa_curves.append((f"SWA {tag}", weighted_mse_improvement(evaluation, scales)))
        metadata = evaluation
    if not steps:
        raise ValueError(f"No evaluation JSON files in {run_dir}")
    pairs = sorted(zip(steps, curves), key=lambda pair: pair[0])
    return [step for step, _ in pairs], [curve for _, curve in pairs], swa_curves, metadata


def _run_label(run_dir: Path) -> str:
    name = run_dir.name
    dconv = re.search(r"_dc(\d+)_", name)
    feedback = "closed-loop" if "_closed_sg_" in name else "baseline-feedback"
    return f"d_conv={dconv.group(1) if dconv else '?'} • {feedback}"


def _plot_lead_time_panel(ax: plt.Axes, steps: list[int], curves: list[np.ndarray],
                          swa_curves: list[tuple[str, np.ndarray]], lead_hours: np.ndarray) -> None:
    """Render the lead-time-improvement panel shared by both output layouts."""
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(steps) - 1)) for i in range(len(steps))]
    for step, curve, color in zip(steps, curves, colors):
        ax.plot(lead_hours, curve, color=color, lw=1.8, label=f"step {step:,}")
    for label, curve in swa_curves:
        ax.plot(lead_hours, curve, color="black", lw=2.8, ls="--", label=label, zorder=5)
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set(xlabel="lead time (hours)", ylabel="weighted normalized-MSE improvement (%)",
           title="Improvement over frozen GraphCast baseline")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")


def plot_run(run_dir: Path, output_dir: Path, scales: xr.Dataset) -> Path:
    steps, curves, swa_curves, metadata = _load_run(run_dir, scales)
    lead_hours = 6 * np.arange(1, len(curves[0]) + 1)
    mode = metadata.get("eval_mode", "unknown")
    samples = metadata.get("n_samples", "?")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    _plot_lead_time_panel(axes[0], steps, curves, swa_curves, lead_hours)

    ax = axes[1]
    for lead_hours_value in SELECTED_LEADS_HOURS:
        index = lead_hours_value // 6 - 1
        if index >= len(curves[0]):
            continue
        values = [curve[index] for curve in curves]
        ax.plot(steps, values, "-o", lw=1.8, ms=4.5, label=f"{lead_hours_value}h")
        for swa_label, curve in swa_curves:
            ax.axhline(
                curve[index], color="black", lw=1.1, ls="--", alpha=0.8,
                label=f"{swa_label} ({lead_hours_value}h)",
            )
    ax.axhline(0.0, color="black", lw=0.7)
    ax.set(xlabel="checkpoint step", ylabel="weighted normalized-MSE improvement (%)",
           title="Checkpoint selection by lead time")
    ax.set_xticks(steps)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Residual-Mamba res2 rollout evaluation — {_run_label(run_dir)}\n"
        f"{mode}, zero residual state, {samples} held-out samples",
        fontsize=12,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_dir.name}_rollout_eval_summary.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_lead_time_only(run_dir: Path, output_dir: Path, scales: xr.Dataset) -> Path:
    """Write the standalone counterpart of the summary figure's left panel."""
    steps, curves, swa_curves, metadata = _load_run(run_dir, scales)
    lead_hours = 6 * np.arange(1, len(curves[0]) + 1)
    mode = metadata.get("eval_mode", "unknown")
    samples = metadata.get("n_samples", "?")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _plot_lead_time_panel(ax, steps, curves, swa_curves, lead_hours)
    fig.suptitle(
        f"Residual-Mamba res2 rollout evaluation — {_run_label(run_dir)}\n"
        f"{mode}, zero residual state, {samples} held-out samples",
        fontsize=12,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_dir.name}_rollout_eval_lead_time.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("plots/analyze_models/images/resolution_eval/resmamba_res2_v20_dconv_loop_sweep"),
    )
    parser.add_argument(
        "--stats-dir", type=Path,
        default=Path("data/graphcast/graphcast/stats"),
    )
    parser.add_argument(
        "--lead-time-only", action="store_true",
        help="Write only the standalone lead-time panel from each summary figure.",
    )
    parser.add_argument(
        "--run-names", nargs="+",
        help="Optional checkpoint-directory names to plot (defaults to every evaluated run).",
    )
    args = parser.parse_args()

    scales = xr.open_dataset(args.stats_dir / "diffs_stddev_by_level.nc")
    run_dirs = sorted(path for path in args.checkpoint_root.iterdir() if path.is_dir())
    if args.run_names:
        requested = set(args.run_names)
        run_dirs = [path for path in run_dirs if path.name in requested]
        missing = requested - {path.name for path in run_dirs}
        if missing:
            raise ValueError(f"Requested run directories not found: {sorted(missing)}")
    for run_dir in run_dirs:
        if not list(run_dir.glob("eval/*/step*.json")):
            continue
        plotter = plot_lead_time_only if args.lead_time_only else plot_run
        output = plotter(run_dir, args.output_dir, scales)
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
