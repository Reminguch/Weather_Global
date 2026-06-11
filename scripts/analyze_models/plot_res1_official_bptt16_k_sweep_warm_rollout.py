#!/usr/bin/env python3
"""Plot res1 official BPTT16 k-sweep warm rollout lead curves."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "res1_official_bptt16_k_sweep_warm_res1grid_k1_40"
DEFAULT_INPUT_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT / "shards"
DEFAULT_OUTPUT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT
DEFAULT_OUTPUT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT

DEFAULT_LEAD_STEPS = [1, 8, 16, 24, 32, 40]
FAMILIES = ["gc_mamba", "residual_mamba"]
FAMILY_LABELS = {"gc_mamba": "GC-Mamba", "residual_mamba": "Residual Mamba"}
TARGET_COLORS = {
    1: "#2f6f9f",
    4: "#b14b2d",
    8: "#2f7d4f",
    12: "#7b4fa3",
    16: "#8a6d2f",
}
PARAM_MARKERS = ["o", "s", "^", "D", "P", "X"]
VARIANT_RE = re.compile(
    r"(?P<model>gc_mamba|residual_mamba).*_di(?P<di>\d+)_ds(?P<ds>\d+)_k(?P<target_step>\d+)_bptt16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--shard-glob", default="resolution_eval_*_warm_res1grid_k1_40.csv")
    parser.add_argument("--output-prefix", default="res1_official_bptt16_k_sweep_warm_rollout")
    parser.add_argument("--merged-csv-name", default="resolution_eval.csv")
    parser.add_argument("--lead-steps", type=int, nargs="+", default=DEFAULT_LEAD_STEPS)
    parser.add_argument("--title-prefix", default="Res1 official warm rollout")
    return parser.parse_args()


def _lead_label(step: int) -> str:
    hours = int(step) * 6
    if hours % 24 == 0:
        return f"k={step}\n{hours // 24}d"
    return f"k={step}\n{hours}h"


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
    return df


def _annotate_variants(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant in df["variant"].astype(str):
        match = VARIANT_RE.search(variant)
        if match is None:
            records.append({"parsed_model": None, "parsed_di": None, "parsed_ds": None, "target_step": None})
            continue
        records.append(
            {
                "parsed_model": match.group("model"),
                "parsed_di": int(match.group("di")),
                "parsed_ds": int(match.group("ds")),
                "target_step": int(match.group("target_step")),
            }
        )
    parsed = pd.DataFrame.from_records(records, index=df.index)
    return pd.concat([df, parsed], axis=1)


def _load_rows(input_dir: Path, shard_glob: str, lead_steps: list[int]) -> pd.DataFrame:
    paths = sorted(input_dir.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"No shard CSVs matching {shard_glob} under {input_dir}")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = _annotate_variants(_ensure_lead_steps(df))
    rows = df[
        (df["res"].astype(int) == 1)
        & (df["eval_mode"].astype(str) == "warm")
        & (df["lead_steps"].astype(int).isin(lead_steps))
        & (df["parsed_model"].isin(FAMILIES))
    ].copy()
    if rows.empty:
        raise ValueError("No matching res1 warm rollout rows found in the shard CSVs.")
    rows["parsed_di"] = rows["parsed_di"].astype(int)
    rows["parsed_ds"] = rows["parsed_ds"].astype(int)
    rows["target_step"] = rows["target_step"].astype(int)
    rows["family_label"] = rows["parsed_model"].map(FAMILY_LABELS)
    rows["param_label"] = "di" + rows["parsed_di"].astype(str) + "/ds" + rows["parsed_ds"].astype(str)
    return rows.sort_values(
        ["parsed_model", "target_step", "parsed_di", "parsed_ds", "lead_steps", "metric_kind", "variable"]
    ).reset_index(drop=True)


def _metric_rows(df: pd.DataFrame, *, metric_kind: str, variable: str | None, lead_steps: list[int]) -> pd.DataFrame:
    rows = df[df["metric_kind"].astype(str) == metric_kind].copy()
    if variable is None:
        rows = rows[rows["variable"].fillna("").astype(str) == ""]
    else:
        rows = rows[rows["variable"].astype(str) == variable]
    if rows.empty:
        var_label = "" if variable is None else f"/{variable}"
        raise ValueError(f"No rows for {metric_kind}{var_label}.")

    want_leads = set(lead_steps)
    missing = []
    for (family, target_step, di, ds), group in rows.groupby(
        ["parsed_model", "target_step", "parsed_di", "parsed_ds"],
        sort=True,
    ):
        got = set(group["lead_steps"].astype(int))
        if got != want_leads:
            label = f"{family} train k={int(target_step)} di{int(di)}/ds{int(ds)}"
            missing.append(f"{label}: missing {sorted(want_leads - got)}")
    if missing:
        raise ValueError("; ".join(missing))
    return rows


def _style_maps(rows: pd.DataFrame) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
    param_specs = sorted({(int(row.parsed_di), int(row.parsed_ds)) for row in rows.itertuples()})
    marker_by_param = {
        spec: PARAM_MARKERS[index % len(PARAM_MARKERS)]
        for index, spec in enumerate(param_specs)
    }
    target_steps = sorted({int(value) for value in rows["target_step"].dropna().unique()})
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_by_target = {}
    for index, target_step in enumerate(target_steps):
        color_by_target[target_step] = TARGET_COLORS.get(
            target_step,
            fallback_colors[index % len(fallback_colors)] if fallback_colors else "C0",
        )
    return marker_by_param, color_by_target


def _plot_family_metric(
    rows: pd.DataFrame,
    *,
    family: str,
    lead_steps: list[int],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    family_rows = rows[rows["parsed_model"] == family].copy()
    if family_rows.empty:
        raise ValueError(f"No rows to plot for family {family!r}.")

    marker_by_param, color_by_target = _style_maps(family_rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for (target_step, di, ds), sub in family_rows.groupby(
        ["target_step", "parsed_di", "parsed_ds"],
        sort=True,
    ):
        sub = sub.sort_values("lead_steps")
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["value"].astype(float),
            color=color_by_target[int(target_step)],
            marker=marker_by_param[(int(di), int(ds))],
            label=f"train k={int(target_step)}, di{int(di)}/ds{int(ds)}",
            linewidth=2.0,
            markersize=5,
        )

    ax.set_xticks(lead_steps)
    ax.set_xticklabels([_lead_label(step) for step in lead_steps])
    ax.set_xlabel("Rollout lead step k")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{FAMILY_LABELS[family]} | {title}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    lead_steps = [int(step) for step in args.lead_steps]
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.input_dir, args.shard_glob, lead_steps)
    merged_csv = args.output_data_dir / args.merged_csv_name
    rows.to_csv(merged_csv, index=False)
    print(f"Saved merged CSV: {merged_csv}")

    weighted = _metric_rows(rows, metric_kind="weighted_allvars", variable=None, lead_steps=lead_steps)
    temp2m = _metric_rows(rows, metric_kind="rmse_k", variable="2m_temperature", lead_steps=lead_steps)
    weighted.to_csv(args.output_data_dir / "plotted_rows_weighted_allvars.csv", index=False)
    temp2m.to_csv(args.output_data_dir / "plotted_rows_2m_temperature_rmse_k.csv", index=False)

    for family in FAMILIES:
        _plot_family_metric(
            weighted,
            family=family,
            lead_steps=lead_steps,
            title=f"{args.title_prefix} loss vs k",
            ylabel="Normalized weighted MSE",
            out_path=args.output_image_dir / f"{args.output_prefix}_{family}_weighted_allvars_vs_k.png",
        )
        _plot_family_metric(
            temp2m,
            family=family,
            lead_steps=lead_steps,
            title=f"{args.title_prefix} 2m temperature RMSE vs k",
            ylabel="2m temperature RMSE (K)",
            out_path=args.output_image_dir / f"{args.output_prefix}_{family}_2m_temperature_rmse_k_vs_k.png",
        )


if __name__ == "__main__":
    main()
