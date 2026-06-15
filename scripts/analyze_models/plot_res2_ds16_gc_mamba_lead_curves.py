#!/usr/bin/env python3
"""Plot res2 ds16 GC-Mamba lead curves against vanilla baselines."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "res2_ds16_gc_mamba_target_steps_bptt16_warm_leads1_9d"
DEFAULT_INPUT_CSV = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT / "resolution_eval.csv"
DEFAULT_OUTPUT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval" / EXPERIMENT
DEFAULT_OUTPUT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval" / EXPERIMENT

LEAD_STEPS = [4, 8, 12, 16, 20, 24, 28, 32, 36]
LEAD_LABELS = {
    4: "1d",
    8: "2d",
    12: "3d",
    16: "4d",
    20: "5d",
    24: "6d",
    28: "7d",
    32: "8d",
    36: "9d",
}
DI_VALUES = [16, 64, 256]
TARGET_STEPS = [4, 8, 12]
BASELINE_LABELS = {
    "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k": "Init vanilla",
    "vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_continue20k": "Continue20k vanilla",
}
TARGET_COLORS = {4: "#2f6f9f", 8: "#b14b2d", 12: "#2f7d4f"}
DI_MARKERS = {16: "o", 64: "s", 256: "^"}
FLAVOR_COLORS = {
    "gclr3e7_mlr1e5_tdrop10": "#7b3294",
    "gclr1e6_mlr1e5": "#008837",
}
CHECKPOINT_STYLES = {
    "best": "-",
    "step40000": "--",
}
VARIANT_RE = re.compile(
    r"(?P<model>gc_mamba).*_di(?P<di>\d+)_ds(?P<ds>\d+)_20k_target_step(?P<target_step>\d+)_bptt16"
)
STEP_RE = re.compile(r"_ckpt_step(?P<step>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-data-dir", type=Path, default=DEFAULT_OUTPUT_DATA_DIR)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_OUTPUT_IMAGE_DIR)
    parser.add_argument("--output-prefix", default=EXPERIMENT)
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


def _annotate(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant in df["variant"].astype(str):
        match = VARIANT_RE.search(variant)
        if match is None:
            records.append(
                {
                    "parsed_di": None,
                    "parsed_ds": None,
                    "target_step": None,
                    "stage": None,
                    "release_flavor": None,
                    "checkpoint_label": None,
                    "plot_label": None,
                }
            )
        else:
            stage = "release_all20k" if "_release_all20k" in variant else "frozen20k"
            if "gclr3e7_mlr1e5_tdrop0p10" in variant:
                flavor = "gclr3e7_mlr1e5_tdrop10"
                flavor_label = "GCLR3e-7 MLR1e-5 tdrop0.10"
            elif "gclr1e6_mlr1e5" in variant:
                flavor = "gclr1e6_mlr1e5"
                flavor_label = "GCLR1e-6 MLR1e-5"
            elif stage == "release_all20k":
                flavor = "release_all20k"
                flavor_label = "Release all +20k"
            else:
                flavor = "frozen20k"
                flavor_label = "Frozen20k"
            step_match = STEP_RE.search(variant)
            checkpoint_label = f"step{step_match.group('step')}" if step_match else "best"
            checkpoint_suffix = f" {checkpoint_label}" if stage == "release_all20k" else ""
            records.append(
                {
                    "parsed_di": int(match.group("di")),
                    "parsed_ds": int(match.group("ds")),
                    "target_step": int(match.group("target_step")),
                    "stage": stage,
                    "release_flavor": flavor,
                    "checkpoint_label": checkpoint_label,
                    "plot_label": (
                        f"{flavor_label}{checkpoint_suffix} target {int(match.group('target_step'))}, "
                        f"di{int(match.group('di'))}/ds{int(match.group('ds'))}"
                    ),
                }
            )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(records)], axis=1)


def _load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = _annotate(_ensure_lead_steps(pd.read_csv(path)))
    df = df[
        (pd.to_numeric(df["res"], errors="coerce").sub(2.0).abs() < 1e-6)
        & (df["eval_mode"].astype(str) == "warm")
        & (df["lead_steps"].astype(int).isin(LEAD_STEPS))
    ].copy()
    mamba = df[
        (df["family"].astype(str) == "gc_mamba")
        & (df["parsed_ds"] == 16)
        & (df["parsed_di"].isin(DI_VALUES))
        & (df["target_step"].isin(TARGET_STEPS))
    ].copy()
    baseline = df[
        (df["family"].astype(str) == "graphcast")
        & (df["variant"].astype(str).isin(BASELINE_LABELS))
    ].copy()
    rows = pd.concat([mamba, baseline], ignore_index=True)
    if rows.empty:
        raise ValueError("No res2 warm ds16 GC-Mamba or baseline rows found.")
    return rows.sort_values(["family", "stage", "variant", "lead_steps", "metric_kind", "variable"]).reset_index(drop=True)


def _metric_rows(df: pd.DataFrame, *, metric_kind: str, variable: str | None) -> pd.DataFrame:
    rows = df[df["metric_kind"].astype(str) == metric_kind].copy()
    if variable is None:
        rows = rows[rows["variable"].fillna("").astype(str) == ""]
    else:
        rows = rows[rows["variable"].astype(str) == variable]
    missing = []
    for (variant, family), group in rows.groupby(["variant", "family"], sort=True):
        got = set(group["lead_steps"].astype(int))
        want = set(LEAD_STEPS)
        if got != want:
            missing.append(f"{family}:{variant} missing {sorted(want - got)}")
    if missing:
        raise ValueError("; ".join(missing))
    return rows


def _plot_metric(rows: pd.DataFrame, *, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.2))

    for variant, label in BASELINE_LABELS.items():
        sub = rows[rows["variant"].astype(str) == variant].sort_values("lead_steps")
        if sub.empty:
            continue
        linestyle = ":" if "continue20k" not in variant else "--"
        color = "#222222" if "continue20k" not in variant else "#767676"
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["value"].astype(float),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker="D",
            markersize=5,
            zorder=5,
        )

    mamba = rows[rows["family"].astype(str) == "gc_mamba"].copy()
    mamba["parsed_di"] = mamba["parsed_di"].astype(int)
    mamba["target_step"] = mamba["target_step"].astype(int)
    mamba["stage"] = mamba["stage"].fillna("frozen20k").astype(str)
    for variant, sub in mamba.groupby("variant", sort=True):
        sub = sub.sort_values("lead_steps")
        if sub.empty:
            continue
        first = sub.iloc[0]
        stage = str(first["stage"])
        target_step = int(first["target_step"])
        di = int(first["parsed_di"])
        flavor = str(first.get("release_flavor") or "")
        checkpoint_label = str(first.get("checkpoint_label") or "best")
        if stage == "frozen20k":
            color = TARGET_COLORS.get(target_step, "#555555")
            linestyle = ":"
            linewidth = 1.45
            alpha = 0.65
            zorder = 3
        else:
            color = FLAVOR_COLORS.get(flavor, TARGET_COLORS.get(target_step, "#333333"))
            linestyle = CHECKPOINT_STYLES.get(checkpoint_label, "-.")
            linewidth = 2.0
            alpha = 0.95
            zorder = 4
        ax.plot(
            sub["lead_steps"].astype(int),
            sub["value"].astype(float),
            label=str(first["plot_label"]),
            color=color,
            marker=DI_MARKERS.get(di, "o"),
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=5,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xticks(LEAD_STEPS)
    ax.set_xticklabels([LEAD_LABELS[step] for step in LEAD_STEPS])
    ax.set_xlabel("Lead time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncols=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    args.output_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_image_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.input_csv)
    audit_csv = args.output_data_dir / f"{args.output_prefix}_annotated_plotted_rows.csv"
    rows.to_csv(audit_csv, index=False)
    print(f"Saved plotted-row audit CSV: {audit_csv}")

    weighted = _metric_rows(rows, metric_kind="weighted_allvars", variable=None)
    weighted.to_csv(args.output_data_dir / f"{args.output_prefix}_weighted_allvars_rows.csv", index=False)
    _plot_metric(
        weighted,
        title="Res2 ds16 GC-Mamba warm rollout | weighted allvars",
        ylabel="Normalized weighted MSE",
        out_path=args.output_image_dir / f"{args.output_prefix}_weighted_allvars_vs_lead.png",
    )

    temp2m = _metric_rows(rows, metric_kind="rmse_k", variable="2m_temperature")
    temp2m.to_csv(args.output_data_dir / f"{args.output_prefix}_2m_temperature_rmse_k_rows.csv", index=False)
    _plot_metric(
        temp2m,
        title="Res2 ds16 GC-Mamba warm rollout | 2m temperature",
        ylabel="2m temperature RMSE (K)",
        out_path=args.output_image_dir / f"{args.output_prefix}_2m_temperature_rmse_k_vs_lead.png",
    )


if __name__ == "__main__":
    main()
