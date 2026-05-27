#!/usr/bin/env python3
"""Plot BPTT16 GC-Mamba res2 lead curves against BPTT6 and vanilla baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "7y_mp6_gc_mamba20k_ds_quarter_vs_vanilla_continue20k_warm"
DEFAULT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT
DEFAULT_BPTT16_CSV = DEFAULT_DATA_DIR / "shards/resolution_eval_gc_mamba_res2_bptt16.csv"
DEFAULT_BASELINE_CSV = DEFAULT_DATA_DIR / "resolution_eval.csv"
PLOT_PREFIX = f"{EXPERIMENT}_lead_curve_res2_bptt16_vs_bptt6_vanilla"

LEAD_STEPS = [1, 4, 8, 16, 24, 32]
LEAD_LABELS = {1: "6h", 4: "1d", 8: "2d", 16: "4d", 24: "6d", 32: "8d"}
CURVES = {
    "GC-Mamba BPTT16 di64/ds16": {
        "variant": "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di64_ds16_20k_bptt16",
        "source": "bptt16",
        "style": {"color": "#1f77b4", "marker": "s", "linestyle": "-"},
    },
    "GC-Mamba BPTT16 di256/ds64": {
        "variant": "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di256_ds64_20k_bptt16",
        "source": "bptt16",
        "style": {"color": "#d62728", "marker": "^", "linestyle": "-"},
    },
    "GC-Mamba BPTT6 di256/ds64": {
        "variant": "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_gc_mamba_tc2_di256_ds64_20k",
        "source": "baseline",
        "style": {"color": "#ff7f0e", "marker": "^", "linestyle": "--"},
    },
    "Vanilla GC continue20k": {
        "variant": "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_continue20k",
        "source": "baseline",
        "style": {"color": "#2f2f2f", "marker": "o", "linestyle": "-"},
    },
}
METRICS = [
    ("weighted_allvars", None, "weighted_allvars", "Normalized weighted MSE", "Weighted all variables"),
    ("per_variable", "2m_temperature", "per_variable_2m_temperature", "2m temperature normalized MSE", "2m temperature"),
    (
        "per_variable",
        "2m_temperature_nyc",
        "per_variable_2m_temperature_nyc",
        "NYC 2m temperature normalized MSE",
        "NYC 2m temperature",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bptt16-csv", type=Path, default=DEFAULT_BPTT16_CSV)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    return parser.parse_args()


def _ensure_lead_steps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lead_days = pd.to_numeric(df["lead_days"], errors="coerce")
    derived_steps = (lead_days * 24.0 / 6.0).round()
    if "lead_steps" not in df.columns:
        df["lead_steps"] = derived_steps
    else:
        lead_steps = pd.to_numeric(df["lead_steps"], errors="coerce")
        df["lead_steps"] = lead_steps.fillna(derived_steps)
    df["lead_steps"] = df["lead_steps"].astype(int)
    df["lead_days"] = lead_days
    return df


def _load_selected_rows(bptt16_csv: Path, baseline_csv: Path) -> pd.DataFrame:
    if not bptt16_csv.exists():
        raise FileNotFoundError(f"Missing BPTT16 shard: {bptt16_csv}")
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {baseline_csv}")

    frames = {
        "bptt16": _ensure_lead_steps(pd.read_csv(bptt16_csv)),
        "baseline": _ensure_lead_steps(pd.read_csv(baseline_csv)),
    }
    selected = []
    for label, spec in CURVES.items():
        df = frames[str(spec["source"])]
        rows = df[
            (df["res"].astype(int) == 2)
            & (df["eval_mode"].astype(str) == "warm")
            & (df["lead_steps"].astype(int).isin(LEAD_STEPS))
            & (df["variant"].astype(str) == str(spec["variant"]))
        ].copy()
        rows["plot_label"] = label
        selected.append(rows)

    out = pd.concat(selected, ignore_index=True)
    if out.empty:
        raise ValueError("No rows selected for plotting.")
    out["plot_label"] = pd.Categorical(out["plot_label"], categories=list(CURVES), ordered=True)
    return out.sort_values(["plot_label", "lead_steps", "metric_kind", "variable"]).reset_index(drop=True)


def _metric_rows(df: pd.DataFrame, metric_kind: str, variable: str | None, metric_slug: str) -> pd.DataFrame:
    rows = df[df["metric_kind"].astype(str) == metric_kind].copy()
    if variable is None:
        rows = rows[rows["variable"].fillna("").astype(str) == ""]
    else:
        rows = rows[rows["variable"].astype(str) == variable]
    rows["plot_metric_slug"] = metric_slug

    missing = []
    for label in CURVES:
        got = set(rows[rows["plot_label"] == label]["lead_steps"].astype(int))
        want = set(LEAD_STEPS)
        if got != want:
            missing.append(f"{label}: missing {sorted(want - got)}")
    if missing:
        raise ValueError(f"{metric_slug} missing expected rows: " + "; ".join(missing))
    return rows


def _plot(rows: pd.DataFrame, title_metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for label, spec in CURVES.items():
        sub = rows[rows["plot_label"] == label].sort_values("lead_steps")
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["value"].astype(float),
            label=label,
            linewidth=2.0,
            markersize=6,
            **spec["style"],
        )
    ax.set_xticks(LEAD_STEPS)
    ax.set_xticklabels([LEAD_LABELS[step] for step in LEAD_STEPS])
    ax.set_xlabel("Lead time")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Res2 warm lead curve | {title_metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)

    df = _load_selected_rows(args.bptt16_csv, args.baseline_csv)
    plotted_frames = []
    for metric_kind, variable, metric_slug, ylabel, title_metric in METRICS:
        rows = _metric_rows(df, metric_kind, variable, metric_slug)
        plotted_frames.append(rows)
        _plot(
            rows,
            title_metric=title_metric,
            ylabel=ylabel,
            out_path=args.image_dir / f"{PLOT_PREFIX}_{metric_slug}.png",
        )

    audit = pd.concat(plotted_frames, ignore_index=True)
    audit_path = args.image_dir / f"{PLOT_PREFIX}_plotted_rows.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Saved plotted rows: {audit_path}")


if __name__ == "__main__":
    main()
