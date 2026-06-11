#!/usr/bin/env python3
"""Plot res2 target-step BPTT16 warm rollout lead curves."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "7y_mp6_mamba_res2_target_steps_bptt16_warm_rollout"
DEFAULT_INPUT_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT / "shards"
DEFAULT_OUTPUT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT
DEFAULT_OUTPUT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT

LEAD_STEPS = [4, 12, 20, 28, 36]
LEAD_LABELS = {4: "k=4\n1d", 12: "k=12\n3d", 20: "k=20\n5d", 28: "k=28\n7d", 36: "k=36\n9d"}
TARGET_STEPS = [4, 8, 12]
PARAM_SPECS = [(64, 32), (128, 64)]
FAMILIES = ["gc_mamba", "residual_mamba"]
FAMILY_LABELS = {"gc_mamba": "GC-Mamba", "residual_mamba": "Residual Mamba"}
BASELINE_VARIANT = "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k"
TARGET_COLORS = {4: "#2f6f9f", 8: "#b14b2d", 12: "#2f7d4f"}
PARAM_STYLES = {
    (64, 32): {"marker": "o", "linestyle": "-"},
    (128, 64): {"marker": "s", "linestyle": "--"},
}
VARIANT_RE = re.compile(
    r"(?P<model>gc_mamba|residual_mamba).*_di(?P<di>\d+)_ds(?P<ds>\d+)_20k_target_step(?P<target_step>\d+)_bptt16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--shard-glob", default="resolution_eval_*_res2_target_steps_bptt16_warm_rollout_k4_36*.csv")
    parser.add_argument("--output-prefix", default="res2_target_steps_bptt16_warm_rollout")
    parser.add_argument("--merged-csv-name", default="resolution_eval.csv")
    parser.add_argument("--title-prefix", default="Res2 warm rolling")
    parser.add_argument("--temperature-metric-kind", default="rmse_k")
    parser.add_argument("--temperature-variable", default="2m_temperature")
    parser.add_argument("--temperature-ylabel", default="2m temperature RMSE (K)")
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


def _load_rows(input_dir: Path, shard_glob: str) -> pd.DataFrame:
    paths = sorted(input_dir.glob(shard_glob))
    if not paths:
        raise FileNotFoundError(f"No shard CSVs matching {shard_glob} under {input_dir}")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = _annotate_variants(_ensure_lead_steps(df))
    mamba_rows = df[
        (df["res"].astype(int) == 2)
        & (df["eval_mode"].astype(str) == "warm")
        & (df["lead_steps"].astype(int).isin(LEAD_STEPS))
        & (df["parsed_model"].isin(FAMILIES))
        & (df["target_step"].isin(TARGET_STEPS))
        & (df[["parsed_di", "parsed_ds"]].apply(tuple, axis=1).isin(PARAM_SPECS))
    ].copy()
    baseline_rows = df[
        (df["family"].astype(str) == "graphcast")
        & (df["variant"].astype(str) == BASELINE_VARIANT)
        & (df["res"].astype(int) == 2)
        & (df["eval_mode"].astype(str) == "warm")
        & (df["lead_steps"].astype(int).isin(LEAD_STEPS))
    ].copy()
    baseline_rows["parsed_model"] = "graphcast"
    baseline_rows["parsed_di"] = pd.NA
    baseline_rows["parsed_ds"] = pd.NA
    baseline_rows["target_step"] = pd.NA

    rows = pd.concat([mamba_rows, baseline_rows], ignore_index=True)
    if rows.empty:
        raise ValueError("No matching warm rollout rows found for the target-step BPTT16 variants.")
    rows["family_label"] = rows["parsed_model"].map({**FAMILY_LABELS, "graphcast": "Vanilla GC"})
    mamba_mask = rows["parsed_model"].isin(FAMILIES)
    rows["param_label"] = ""
    rows.loc[mamba_mask, "param_label"] = (
        "di"
        + rows.loc[mamba_mask, "parsed_di"].astype(int).astype(str)
        + "/ds"
        + rows.loc[mamba_mask, "parsed_ds"].astype(int).astype(str)
    )
    return rows.sort_values(
        ["parsed_model", "target_step", "parsed_di", "parsed_ds", "lead_steps", "metric_kind", "variable"]
    ).reset_index(drop=True)


def _metric_rows(df: pd.DataFrame, *, metric_kind: str, variable: str | None) -> pd.DataFrame:
    rows = df[(df["parsed_model"].isin(FAMILIES)) & (df["metric_kind"].astype(str) == metric_kind)].copy()
    if variable is None:
        rows = rows[rows["variable"].fillna("").astype(str) == ""]
    else:
        rows = rows[rows["variable"].astype(str) == variable]
    rows = rows.dropna(subset=["target_step", "parsed_di", "parsed_ds"]).copy()
    rows["target_step"] = rows["target_step"].astype(int)
    rows["parsed_di"] = rows["parsed_di"].astype(int)
    rows["parsed_ds"] = rows["parsed_ds"].astype(int)

    want_leads = set(LEAD_STEPS)
    missing = []
    for (family, target_step, di, ds), group in rows.groupby(
        ["parsed_model", "target_step", "parsed_di", "parsed_ds"],
        dropna=True,
        sort=True,
    ):
        got = set(group["lead_steps"].astype(int))
        if got != want_leads:
            label = f"{family} target_step={int(target_step)} di{int(di)}/ds{int(ds)}"
            missing.append(f"{label}: missing {sorted(want_leads - got)}")
    if missing:
        raise ValueError("; ".join(missing))
    return rows


def _baseline_metric_rows(df: pd.DataFrame, *, metric_kind: str, variable: str | None) -> pd.DataFrame:
    rows = df[(df["parsed_model"] == "graphcast") & (df["metric_kind"].astype(str) == metric_kind)].copy()
    if variable is None:
        rows = rows[rows["variable"].fillna("").astype(str) == ""]
    else:
        rows = rows[rows["variable"].astype(str) == variable]
    got = set(rows["lead_steps"].astype(int))
    want = set(LEAD_STEPS)
    if got != want:
        raise ValueError(f"Vanilla GC baseline missing lead steps {sorted(want - got)}")
    return rows


def _plot_family_metric(
    rows: pd.DataFrame,
    *,
    baseline_rows: pd.DataFrame,
    family: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    family_rows = rows[rows["parsed_model"] == family]
    baseline = baseline_rows.sort_values("lead_steps")
    ax.plot(
        baseline["lead_steps"].astype(int),
        baseline["value"].astype(float),
        color="#1f1f1f",
        label="Vanilla GC baseline",
        linewidth=2.4,
        linestyle=":",
        marker="D",
        markersize=5,
        zorder=5,
    )
    for target_step in TARGET_STEPS:
        for di, ds in PARAM_SPECS:
            sub = family_rows[
                (family_rows["target_step"].astype(int) == target_step)
                & (family_rows["parsed_di"].astype(int) == di)
                & (family_rows["parsed_ds"].astype(int) == ds)
            ].sort_values("lead_steps")
            if sub.empty:
                continue
            style = PARAM_STYLES[(di, ds)]
            ax.plot(
                sub["lead_steps"].astype(int),
                sub["value"].astype(float),
                color=TARGET_COLORS[target_step],
                label=f"target {target_step}, di{di}/ds{ds}",
                linewidth=2.0,
                markersize=5,
                **style,
            )
    ax.set_xticks(LEAD_STEPS)
    ax.set_xticklabels([LEAD_LABELS[step] for step in LEAD_STEPS])
    ax.set_xlabel("Lead step k")
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
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.input_dir, args.shard_glob)
    merged_csv = args.output_data_dir / args.merged_csv_name
    rows.to_csv(merged_csv, index=False)
    print(f"Saved merged CSV: {merged_csv}")

    has_weighted = (
        (rows["metric_kind"].astype(str) == "weighted_allvars")
        & (rows["variable"].fillna("").astype(str) == "")
    ).any()
    if has_weighted:
        weighted = _metric_rows(rows, metric_kind="weighted_allvars", variable=None)
        weighted_baseline = _baseline_metric_rows(rows, metric_kind="weighted_allvars", variable=None)
        weighted.to_csv(args.output_data_dir / "plotted_rows_weighted_allvars.csv", index=False)
        weighted_baseline.to_csv(args.output_data_dir / "plotted_rows_vanilla_gc_weighted_allvars.csv", index=False)
        for family in FAMILIES:
            _plot_family_metric(
                weighted,
                baseline_rows=weighted_baseline,
                family=family,
                title=f"{args.title_prefix} eval loss vs k",
                ylabel="Normalized weighted MSE",
                out_path=args.output_image_dir
                / f"{args.output_prefix}_{family}_weighted_allvars_vs_k.png",
            )

    temp2m = _metric_rows(rows, metric_kind=args.temperature_metric_kind, variable=args.temperature_variable)
    temp2m_baseline = _baseline_metric_rows(
        rows,
        metric_kind=args.temperature_metric_kind,
        variable=args.temperature_variable,
    )
    if args.temperature_metric_kind == "per_variable" and args.temperature_variable == "2m_temperature":
        temp_suffix = "2m_temperature"
    elif args.temperature_metric_kind == "rmse_k" and args.temperature_variable == "2m_temperature":
        temp_suffix = "2m_temperature_rmse_k"
    else:
        temp_suffix = f"{args.temperature_variable}_{args.temperature_metric_kind}"
    temp2m.to_csv(args.output_data_dir / f"plotted_rows_{temp_suffix}.csv", index=False)
    temp2m_baseline.to_csv(args.output_data_dir / f"plotted_rows_vanilla_gc_{temp_suffix}.csv", index=False)
    for family in FAMILIES:
        _plot_family_metric(
            temp2m,
            baseline_rows=temp2m_baseline,
            family=family,
            title=f"{args.title_prefix} 2m temperature loss vs k",
            ylabel=args.temperature_ylabel,
            out_path=args.output_image_dir
            / f"{args.output_prefix}_{family}_{temp_suffix}_vs_k.png",
        )


if __name__ == "__main__":
    main()
