#!/usr/bin/env python3
"""Plot res1 BPTT16 k-sweep lead curves with the DeepMind baseline."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "res1_official_bptt16_k_sweep_warm_res1grid_k1_40"
DEFAULT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT
DEFAULT_BASELINE_WEIGHTED_DIR = (
    ROOT / "plots/analyze_models/data/resolution_eval/deepmind_graphcast_small_res1_cold/shards"
)
LEAD_STEPS = [1, 8, 16, 24, 32, 40]
LEAD_LABELS = {1: "6h", 8: "2d", 16: "4d", 24: "6d", 32: "8d", 40: "10d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-subdir", default="lead_curves_res1_existing_partial")
    parser.add_argument("--output-prefix", default="res1_bptt16_k_sweep")
    parser.add_argument(
        "--mamba-glob",
        default="shards/resolution_eval_*mamba_res1_warm_res1grid_k1_40*.csv",
        help="Glob, relative to --data-dir, for GC/Residual Mamba shard CSVs.",
    )
    parser.add_argument(
        "--mamba-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional exact GC/Residual Mamba CSV paths. When set, these replace --mamba-glob discovery.",
    )
    parser.add_argument(
        "--baseline-weighted-dir",
        type=Path,
        default=DEFAULT_BASELINE_WEIGHTED_DIR,
        help="Directory containing existing DeepMind baseline weighted-allvars shard CSVs.",
    )
    parser.add_argument(
        "--baseline-rmse-csv",
        type=Path,
        default=None,
        help="Optional DeepMind baseline RMSE-K CSV to overlay on the 2m-temperature plot.",
    )
    parser.add_argument(
        "--allow-teacher-forced-residual",
        action="store_true",
        help=(
            "Allow residual_mamba rows marked teacher_forced_training_equivalent. "
            "By default these diagnostic rows are rejected because they are not "
            "proper error-vs-lead rollout curves."
        ),
    )
    return parser.parse_args()


def _ensure_lead_steps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lead_days = pd.to_numeric(df["lead_days"], errors="coerce")
    if "lead_steps" not in df.columns:
        df["lead_steps"] = (lead_days * 24.0 / 6.0).round().astype(int)
    else:
        lead_steps = pd.to_numeric(df["lead_steps"], errors="coerce")
        df["lead_steps"] = lead_steps.fillna((lead_days * 24.0 / 6.0).round()).astype(int)
    df["lead_days"] = lead_days
    return df


def _curve_label(row: pd.Series) -> str:
    family_label = {
        "gc_mamba": "GC-Mamba",
        "residual_mamba": "Residual Mamba",
        "graphcast": "DeepMind GraphCast-small baseline",
    }.get(str(row["family"]), str(row["family"]))
    if str(row["family"]) == "graphcast":
        return family_label
    match = re.search(r"_k(\d+)_", str(row["variant"]))
    return f"{family_label} k{match.group(1)}" if match else family_label


def _reject_teacher_forced_residual_rows(df: pd.DataFrame) -> None:
    if "residual_eval_semantics" not in df.columns:
        return
    residual_rows = df["family"].astype(str).eq("residual_mamba")
    teacher_forced_rows = df["residual_eval_semantics"].astype(str).eq("teacher_forced_training_equivalent")
    bad = df[residual_rows & teacher_forced_rows]
    if bad.empty:
        return
    variants = ", ".join(sorted(bad["variant"].astype(str).unique()))
    raise ValueError(
        "Refusing to plot teacher-forced residual_mamba rows as lead curves. "
        "Rerun residual eval with --residual-eval-semantics rollout, or pass "
        f"--allow-teacher-forced-residual for diagnostic plots only. Variants: {variants}"
    )


def _resolve_csv_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_mamba_rows(
    data_dir: Path,
    mamba_glob: str,
    *,
    mamba_csvs: list[Path] | None = None,
    allow_teacher_forced_residual: bool = False,
) -> pd.DataFrame:
    paths = [_resolve_csv_path(path) for path in mamba_csvs] if mamba_csvs is not None else sorted(data_dir.glob(mamba_glob))
    if not paths:
        raise FileNotFoundError(f"No Mamba CSVs found under {data_dir} matching {mamba_glob}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError("Mamba CSVs were found, but all were empty.")
    df = _ensure_lead_steps(pd.concat(frames, ignore_index=True))
    df = df[pd.to_numeric(df["res"], errors="coerce").eq(1)].copy()
    df = df[df["lead_steps"].astype(int).isin(LEAD_STEPS)].copy()
    if not allow_teacher_forced_residual:
        _reject_teacher_forced_residual_rows(df)
    df["curve"] = df.apply(_curve_label, axis=1)
    return df


def _load_baseline_weighted_rows(baseline_dir: Path) -> pd.DataFrame:
    paths = sorted(baseline_dir.glob("*.csv"))
    if not paths:
        return pd.DataFrame()
    df = _ensure_lead_steps(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    df = df[
        pd.to_numeric(df["res"], errors="coerce").eq(1)
        & df["lead_steps"].astype(int).isin(LEAD_STEPS)
        & df["metric_kind"].astype(str).eq("weighted_allvars")
        & df["variable"].fillna("").astype(str).eq("")
    ].copy()
    if df.empty:
        return df
    df["curve"] = "DeepMind GraphCast-small baseline"
    return df


def _load_baseline_rmse_rows(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = _ensure_lead_steps(pd.read_csv(path))
    df = df[
        pd.to_numeric(df["res"], errors="coerce").eq(1)
        & df["lead_steps"].astype(int).isin(LEAD_STEPS)
        & df["metric_kind"].astype(str).eq("rmse_k")
        & df["variable"].astype(str).eq("2m_temperature")
    ].copy()
    if df.empty:
        return df
    df["curve"] = "DeepMind GraphCast-small baseline"
    return df


def _style(curve: str) -> dict[str, object]:
    styles: dict[str, dict[str, object]] = {
        "DeepMind GraphCast-small baseline": {
            "color": "#222222",
            "linestyle": "--",
            "marker": "D",
            "linewidth": 2.4,
        },
        "GC-Mamba k1": {"color": "#1f77b4", "marker": "o"},
        "GC-Mamba k4": {"color": "#2ca02c", "marker": "s"},
        "GC-Mamba k8": {"color": "#17becf", "marker": "^"},
        "Residual Mamba k1": {"color": "#d62728", "marker": "o"},
        "Residual Mamba k4": {"color": "#ff7f0e", "marker": "s"},
        "Residual Mamba k8": {"color": "#9467bd", "marker": "^"},
    }
    default: dict[str, object] = {"linewidth": 1.9, "markersize": 5.5}
    default.update(styles.get(curve, {}))
    return default


def _plot(rows: pd.DataFrame, *, title: str, ylabel: str, out_path: Path) -> None:
    if rows.empty:
        raise ValueError(f"No rows to plot for {out_path.name}")
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=180)
    for curve, sub in rows.groupby("curve", sort=False):
        sub = sub.sort_values("lead_steps")
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["value"].astype(float),
            label=curve,
            **_style(str(curve)),
        )
    ax.set_xticks(LEAD_STEPS)
    ax.set_xticklabels([LEAD_LABELS[step] for step in LEAD_STEPS])
    ax.set_xlabel("Lead time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    out_data = args.data_dir / args.output_subdir
    out_img = args.image_dir / args.output_subdir
    out_data.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

    mamba = _load_mamba_rows(
        args.data_dir,
        args.mamba_glob,
        mamba_csvs=args.mamba_csvs,
        allow_teacher_forced_residual=args.allow_teacher_forced_residual,
    )
    baseline_weighted = _load_baseline_weighted_rows(args.baseline_weighted_dir)
    baseline_rmse = _load_baseline_rmse_rows(args.baseline_rmse_csv)

    weighted = pd.concat(
        [
            mamba[
                mamba["metric_kind"].astype(str).eq("weighted_allvars")
                & mamba["variable"].fillna("").astype(str).eq("")
            ],
            baseline_weighted,
        ],
        ignore_index=True,
    ).sort_values(["curve", "lead_steps"])
    weighted.to_csv(out_data / f"{args.output_prefix}_weighted_allvars_plotted_rows.csv", index=False)
    _plot(
        weighted,
        title="Res1 lead curves: weighted allvars",
        ylabel="Weighted allvars",
        out_path=out_img / f"{args.output_prefix}_weighted_allvars_vs_lead_with_baseline.png",
    )

    temp = pd.concat(
        [
            mamba[
                mamba["metric_kind"].astype(str).eq("rmse_k")
                & mamba["variable"].astype(str).eq("2m_temperature")
            ],
            baseline_rmse,
        ],
        ignore_index=True,
    ).sort_values(["curve", "lead_steps"])
    temp.to_csv(out_data / f"{args.output_prefix}_rmse_k_2m_temperature_plotted_rows.csv", index=False)
    _plot(
        temp,
        title="Res1 lead curves: 2m temperature RMSE-K",
        ylabel="2m temperature RMSE (K)",
        out_path=out_img / f"{args.output_prefix}_rmse_k_2m_temperature_vs_lead_with_baseline.png",
    )


if __name__ == "__main__":
    main()
