#!/usr/bin/env python3
"""Compare res2 ds16 release/frozen evals with an existing legacy BPTT16 CSV."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".tmp/matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENT = "res2_ds16_existing_csv_comparison"
RELEASE_EXPERIMENT = "res2_ds16_gc_mamba_release_all20k_warm_leads1_9d"
LEGACY_EXPERIMENT = "7y_mp6_gc_mamba20k_ds_quarter_vs_vanilla_continue20k_warm"

DEFAULT_RELEASE_CSV = (
    ROOT / "plots/analyze_models/data/resolution_eval" / RELEASE_EXPERIMENT / "resolution_eval.csv"
)
DEFAULT_LEGACY_CSV = (
    ROOT
    / "plots/analyze_models/data/resolution_eval"
    / LEGACY_EXPERIMENT
    / "shards/resolution_eval_gc_mamba_res2_bptt16.csv"
)
DEFAULT_OUTPUT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT
DEFAULT_OUTPUT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT

BASE_NAME = "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k"
CONTINUE_VARIANT = f"{BASE_NAME}_continue20k"
LEGACY_VARIANT = f"{BASE_NAME}_gc_mamba_tc2_di64_ds16_20k_bptt16"

TARGET_STEPS = [4, 8, 12]
DI_VALUES = [16, 64, 256]
TARGET_COLORS = {4: "#2f6f9f", 8: "#b14b2d", 12: "#2f7d4f"}
DI_MARKERS = {16: "o", 64: "s", 256: "^"}
STAGE_STYLES = {
    "frozen20k": {"label": "Frozen20k", "linestyle": ":", "linewidth": 1.35, "alpha": 0.58, "zorder": 3},
    "release_all20k": {"label": "Release all +20k", "linestyle": "-", "linewidth": 2.0, "alpha": 0.92, "zorder": 4},
}
VARIANT_RE = re.compile(
    r"(?P<model>gc_mamba).*_di(?P<di>\d+)_ds(?P<ds>\d+)_20k_target_step(?P<target_step>\d+)_bptt16"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-csv", type=Path, default=DEFAULT_RELEASE_CSV)
    parser.add_argument("--legacy-csv", type=Path, default=DEFAULT_LEGACY_CSV)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-prefix", default=EXPERIMENT)
    return parser.parse_args()


def _lead_label(step: int) -> str:
    hours = int(step) * 6
    if hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


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
    df["variable"] = df["variable"].fillna("").astype(str)
    return df


def _annotate_release_rows(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant in df["variant"].astype(str):
        match = VARIANT_RE.search(variant)
        if match is None:
            records.append(
                {
                    "comparison_group": "vanilla_continue20k" if variant == CONTINUE_VARIANT else None,
                    "display_label": "Vanilla continue20k" if variant == CONTINUE_VARIANT else None,
                    "parsed_di": None,
                    "parsed_ds": None,
                    "target_step": None,
                    "stage": None,
                }
            )
            continue
        di = int(match.group("di"))
        ds = int(match.group("ds"))
        target_step = int(match.group("target_step"))
        stage = "release_all20k" if "_release_all20k" in variant else "frozen20k"
        stage_label = STAGE_STYLES[stage]["label"]
        records.append(
            {
                "comparison_group": stage,
                "display_label": f"{stage_label} target {target_step}, di{di}/ds{ds}",
                "parsed_di": di,
                "parsed_ds": ds,
                "target_step": target_step,
                "stage": stage,
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def _load_release_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing release CSV: {path}")
    df = _annotate_release_rows(_ensure_lead_steps(pd.read_csv(path)))
    df = df[
        (pd.to_numeric(df["res"], errors="coerce").sub(2.0).abs() < 1e-6)
        & (df["eval_mode"].astype(str) == "warm")
    ].copy()
    target_mamba = (
        (df["family"].astype(str) == "gc_mamba")
        & (pd.to_numeric(df["parsed_ds"], errors="coerce") == 16)
        & (pd.to_numeric(df["parsed_di"], errors="coerce").isin(DI_VALUES))
        & (pd.to_numeric(df["target_step"], errors="coerce").isin(TARGET_STEPS))
    )
    vanilla_continue = (df["family"].astype(str) == "graphcast") & (
        df["variant"].astype(str) == CONTINUE_VARIANT
    )
    rows = df[target_mamba | vanilla_continue].copy()
    if rows.empty:
        raise ValueError(f"No res2 warm release/frozen or continue20k rows found in {path}")
    rows["source_csv"] = str(path)
    return rows


def _load_legacy_weighted_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing legacy CSV: {path}")
    df = _ensure_lead_steps(pd.read_csv(path))
    rows = df[
        (pd.to_numeric(df["res"], errors="coerce").sub(2.0).abs() < 1e-6)
        & (df["eval_mode"].astype(str) == "warm")
        & (df["variant"].astype(str) == LEGACY_VARIANT)
        & (df["metric_kind"].astype(str) == "weighted_allvars")
        & (df["variable"] == "")
    ].copy()
    if rows.empty:
        raise ValueError(f"No weighted legacy BPTT16 rows for {LEGACY_VARIANT} in {path}")
    rows["comparison_group"] = "legacy_bptt16"
    rows["display_label"] = "Legacy BPTT16 di64/ds16"
    rows["parsed_di"] = 64
    rows["parsed_ds"] = 16
    rows["target_step"] = None
    rows["stage"] = "legacy_bptt16"
    rows["source_csv"] = str(path)
    return rows


def _metric_rows(rows: pd.DataFrame, metric_kind: str, variable: str = "") -> pd.DataFrame:
    return rows[
        (rows["metric_kind"].astype(str) == metric_kind)
        & (rows["variable"].fillna("").astype(str) == variable)
    ].copy()


def _common_lead_steps(rows: pd.DataFrame) -> list[int]:
    by_variant = {
        str(variant): set(group["lead_steps"].astype(int).tolist())
        for variant, group in rows.groupby("variant", sort=True)
    }
    if not by_variant:
        raise ValueError("No rows available to determine common lead steps.")
    common = set.intersection(*by_variant.values())
    if not common:
        details = {variant: sorted(steps) for variant, steps in by_variant.items()}
        raise ValueError(f"No common lead steps across selected curves: {details}")
    return sorted(common)


def _assert_complete(rows: pd.DataFrame, lead_steps: list[int]) -> None:
    want = set(lead_steps)
    missing = []
    for variant, group in rows.groupby("variant", sort=True):
        got = set(group["lead_steps"].astype(int).tolist())
        if got != want:
            missing.append(f"{variant}: missing {sorted(want - got)}, extra {sorted(got - want)}")
    if missing:
        raise ValueError("; ".join(missing))


def _plot_weighted(rows: pd.DataFrame, lead_steps: list[int], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.4))

    baseline = rows[rows["variant"].astype(str) == CONTINUE_VARIANT].sort_values("lead_steps")
    if baseline.empty:
        raise ValueError("Missing vanilla continue20k weighted rows.")
    ax.plot(
        baseline["lead_steps"].astype(int),
        baseline["value"].astype(float),
        label="Vanilla continue20k",
        color="#2f2f2f",
        linestyle="--",
        linewidth=2.8,
        marker="D",
        markersize=5,
        zorder=6,
    )

    mamba = rows[
        (rows["family"].astype(str) == "gc_mamba")
        & (rows["comparison_group"].isin(["frozen20k", "release_all20k"]))
    ].copy()
    for stage in ["frozen20k", "release_all20k"]:
        style = STAGE_STYLES[stage]
        stage_rows = mamba[mamba["comparison_group"] == stage]
        for target_step in TARGET_STEPS:
            for di in DI_VALUES:
                sub = stage_rows[
                    (pd.to_numeric(stage_rows["target_step"], errors="coerce") == target_step)
                    & (pd.to_numeric(stage_rows["parsed_di"], errors="coerce") == di)
                ].sort_values("lead_steps")
                if sub.empty:
                    continue
                ax.plot(
                    sub["lead_steps"].astype(int),
                    sub["value"].astype(float),
                    label=f"{style['label']} target {target_step}, di{di}/ds16",
                    color=TARGET_COLORS[target_step],
                    marker=DI_MARKERS[di],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=5,
                    alpha=style["alpha"],
                    zorder=style["zorder"],
                )

    legacy = rows[rows["variant"].astype(str) == LEGACY_VARIANT].sort_values("lead_steps")
    if legacy.empty:
        raise ValueError("Missing legacy BPTT16 weighted rows.")
    ax.plot(
        legacy["lead_steps"].astype(int),
        legacy["value"].astype(float),
        label="Legacy BPTT16 di64/ds16",
        color="#8a4b08",
        linestyle="-.",
        linewidth=2.7,
        marker="X",
        markersize=7,
        zorder=7,
    )

    ax.set_xticks(lead_steps)
    ax.set_xticklabels([_lead_label(step) for step in lead_steps])
    ax.set_xlabel("Lead time")
    ax.set_ylabel("Normalized weighted MSE")
    ax.set_title("Res2 ds16 GC-Mamba warm rollout | existing CSV comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def _plot_rmse(rows: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rmse = _metric_rows(rows, "rmse_k", "2m_temperature")
    if rmse.empty:
        print("No 2m_temperature rmse_k rows found; skipping RMSE plot.")
        return rmse

    lead_steps = sorted(rmse["lead_steps"].astype(int).unique().tolist())
    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    baseline = rmse[rmse["variant"].astype(str) == CONTINUE_VARIANT].sort_values("lead_steps")
    if not baseline.empty:
        ax.plot(
            baseline["lead_steps"].astype(int),
            baseline["value"].astype(float),
            label="Vanilla continue20k",
            color="#2f2f2f",
            linestyle="--",
            linewidth=2.8,
            marker="D",
            markersize=5,
            zorder=6,
        )

    mamba = rmse[rmse["family"].astype(str) == "gc_mamba"].copy()
    for stage in ["frozen20k", "release_all20k"]:
        style = STAGE_STYLES[stage]
        stage_rows = mamba[mamba["comparison_group"] == stage]
        for target_step in TARGET_STEPS:
            for di in DI_VALUES:
                sub = stage_rows[
                    (pd.to_numeric(stage_rows["target_step"], errors="coerce") == target_step)
                    & (pd.to_numeric(stage_rows["parsed_di"], errors="coerce") == di)
                ].sort_values("lead_steps")
                if sub.empty:
                    continue
                ax.plot(
                    sub["lead_steps"].astype(int),
                    sub["value"].astype(float),
                    label=f"{style['label']} target {target_step}, di{di}/ds16",
                    color=TARGET_COLORS[target_step],
                    marker=DI_MARKERS[di],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=5,
                    alpha=style["alpha"],
                    zorder=style["zorder"],
                )

    ax.set_xticks(lead_steps)
    ax.set_xticklabels([_lead_label(step) for step in lead_steps])
    ax.set_xlabel("Lead time")
    ax.set_ylabel("2m temperature RMSE (K)")
    ax.set_title("Res2 ds16 GC-Mamba warm rollout | 2m temperature")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")
    return rmse


def main() -> None:
    args = parse_args()
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    release_rows = _load_release_rows(args.release_csv)
    legacy_weighted_rows = _load_legacy_weighted_rows(args.legacy_csv)

    weighted = pd.concat(
        [_metric_rows(release_rows, "weighted_allvars", ""), legacy_weighted_rows],
        ignore_index=True,
        sort=False,
    )
    common_leads = _common_lead_steps(weighted)
    weighted = weighted[weighted["lead_steps"].astype(int).isin(common_leads)].copy()
    _assert_complete(weighted, common_leads)
    weighted = weighted.sort_values(["comparison_group", "variant", "lead_steps"]).reset_index(drop=True)

    audit_csv = args.output_data_dir / f"{args.output_prefix}_weighted_allvars_rows.csv"
    weighted.to_csv(audit_csv, index=False)
    print(f"Saved weighted plotted-row audit CSV: {audit_csv}")
    print(f"Common weighted lead steps: {' '.join(str(step) for step in common_leads)}")

    _plot_weighted(
        weighted,
        common_leads,
        args.output_image_dir / f"{args.output_prefix}_weighted_allvars_vs_lead.png",
    )

    rmse = _plot_rmse(
        release_rows,
        args.output_image_dir / f"{args.output_prefix}_2m_temperature_rmse_k_vs_lead.png",
    )
    if not rmse.empty:
        rmse_csv = args.output_data_dir / f"{args.output_prefix}_2m_temperature_rmse_k_rows.csv"
        rmse.sort_values(["comparison_group", "variant", "lead_steps"]).to_csv(rmse_csv, index=False)
        print(f"Saved RMSE plotted-row audit CSV: {rmse_csv}")


if __name__ == "__main__":
    main()
