#!/usr/bin/env python3
"""Plot GC-Mamba resolution evals against same-width GraphCast baselines."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_CSV = (
    ROOT / "plots/analyze_models/data/resolution_eval/pilot_graphcast_segments_ms_sweep_warm/resolution_eval.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "plots/analyze_models/data/resolution_eval/gc_mamba_vs_graphcast_baseline"
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval/gc_mamba_vs_graphcast_baseline"
VARIANT_RE = re.compile(r"res(?P<res>\d+)_m(?P<mesh>\d+)_w(?P<width>\d+)_mp(?P<mp>\d+)(?:_(?:di|dh|h)(?P<di>\d+))?")
RES_MESH_TOKEN_RE = re.compile(r"_res\d+_m\d+(?=_)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-prefix", default="gc_mamba_vs_graphcast")
    parser.add_argument("--source-label", default="GC-Mamba")
    parser.add_argument("--lead-days", type=int, nargs="+", default=None)
    parser.add_argument("--eval-modes", nargs="+", default=None)
    parser.add_argument("--metric-kind", default="weighted_allvars")
    return parser.parse_args()


def _parse_variant(value: str) -> dict[str, int | None]:
    match = VARIANT_RE.search(str(value))
    if not match:
        raise ValueError(f"Could not parse variant tokens from {value!r}")
    out: dict[str, int | None] = {}
    for key, raw in match.groupdict().items():
        out[key] = None if raw is None else int(raw)
    return out


def _annotate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    parsed = df["variant"].astype(str).map(_parse_variant).apply(pd.Series)
    for col in ["mesh", "width", "mp"]:
        df[col] = parsed[col].astype(int)
    df["di"] = parsed["di"]
    df["source"] = source
    df["model_key"] = df["variant"].astype(str).map(lambda value: RES_MESH_TOKEN_RE.sub("", value, count=1))
    return df


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "value"


def _label(curve: pd.DataFrame, source_label: str) -> str:
    source = str(curve["source"].iloc[0])
    width = int(curve["width"].iloc[0])
    mp = int(curve["mp"].iloc[0])
    if source == "baseline":
        return f"GraphCast baseline w{width} mp{mp}"
    di = curve["di"].iloc[0]
    suffix = "" if pd.isna(di) else f" di{int(di)}"
    return f"{source_label} w{width} mp{mp}{suffix}"


def main() -> None:
    args = parse_args()
    eval_df = _annotate(pd.read_csv(args.eval_csv), "eval")
    baseline_df = _annotate(pd.read_csv(args.baseline_csv), "baseline")

    eval_df = eval_df[eval_df["metric_kind"].astype(str) == args.metric_kind]
    baseline_df = baseline_df[baseline_df["metric_kind"].astype(str) == args.metric_kind]
    if args.metric_kind == "weighted_allvars":
        eval_df = eval_df[eval_df["variable"].fillna("") == ""]
        baseline_df = baseline_df[baseline_df["variable"].fillna("") == ""]
    if args.lead_days is not None:
        eval_df = eval_df[eval_df["lead_days"].astype(int).isin(args.lead_days)]
        baseline_df = baseline_df[baseline_df["lead_days"].astype(int).isin(args.lead_days)]
    if args.eval_modes is not None:
        eval_df = eval_df[eval_df["eval_mode"].astype(str).isin(args.eval_modes)]
        baseline_df = baseline_df[baseline_df["eval_mode"].astype(str).isin(args.eval_modes)]

    widths = set(eval_df["width"].astype(int).unique())
    mps = set(eval_df["mp"].astype(int).unique())
    baseline_df = baseline_df[
        baseline_df["width"].astype(int).isin(widths) & baseline_df["mp"].astype(int).isin(mps)
    ]
    combined = pd.concat([eval_df, baseline_df], ignore_index=True)
    combined = combined.sort_values(["eval_mode", "lead_days", "width", "mp", "source", "di", "res", "variant"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_prefix}_matched.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    args.image_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for (eval_mode, lead_day, width, mp), sub in combined.groupby(["eval_mode", "lead_days", "width", "mp"], sort=True):
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        plotted = False
        for _, curve in sub.groupby(["source", "model_key"], sort=True):
            curve = curve.sort_values("res")
            if curve["value"].notna().sum() == 0:
                continue
            is_baseline = str(curve["source"].iloc[0]) == "baseline"
            ax.plot(
                curve["res"],
                curve["value"],
                marker="o",
                linestyle="--" if is_baseline else "-",
                linewidth=2.4 if is_baseline else 1.8,
                label=_label(curve, args.source_label),
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Resolution group (res)")
        ax.set_ylabel("Normalized weighted MSE")
        ax.set_title(f"{args.source_label} vs same-width GraphCast | w={int(width)} mp={int(mp)} | lead={int(lead_day)}d | {eval_mode}")
        ax.set_xticks(sorted(sub["res"].astype(int).unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = args.image_dir / (
            f"{args.output_prefix}_w{int(width)}_mp{int(mp)}_lead{int(lead_day)}d_{_safe(str(eval_mode))}.png"
        )
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        print(f"Saved image: {out_path}")
        saved.append(out_path)
    print(f"Saved {len(saved)} images")


if __name__ == "__main__":
    main()
