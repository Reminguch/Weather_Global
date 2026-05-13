#!/usr/bin/env python3
"""Plot small frozen/release Mamba runs against vanilla GraphCast."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval"
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval/small_mamba_freeze_release_warm"
DEFAULT_GC_MAMBA_CSV = DEFAULT_DATA_DIR / "small_gc_mamba_freeze_release_warm/resolution_eval.csv"
DEFAULT_RESIDUAL_MAMBA_CSV = DEFAULT_DATA_DIR / "small_residual_mamba_freeze_release_warm/resolution_eval.csv"
RUN_TOKEN_RE = re.compile(
    r"_res(?P<res>\d+)_m(?P<mesh>\d+)_w(?P<width>\d+)_mp(?P<mp>\d+).*?(?P<mamba>gc_mamba|residual_mamba)?"
    r"(?:_tc(?P<tc>\d+)_di(?P<di>\d+)_ds(?P<ds>\d+))?"
    r"(?P<frozen>_frozen50k)?(?P<release>_release20k)?$"
)


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    color: str
    marker: str
    linestyle: str


CURVES = [
    CurveSpec("vanilla", "Vanilla 200k", "#2f2f2f", "o", "-"),
    CurveSpec("frozen_di128_ds64", "Frozen50k di128/ds64", "#1f77b4", "s", "-"),
    CurveSpec("frozen_di256_ds128", "Frozen50k di256/ds128", "#1f77b4", "^", "-"),
    CurveSpec("release_di128_ds64", "Frozen50k + release20k di128/ds64", "#d62728", "s", "--"),
    CurveSpec("release_di256_ds128", "Frozen50k + release20k di256/ds128", "#d62728", "^", "--"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gc-mamba-csv", type=Path, default=DEFAULT_GC_MAMBA_CSV)
    parser.add_argument("--residual-mamba-csv", type=Path, default=DEFAULT_RESIDUAL_MAMBA_CSV)
    parser.add_argument("--output-image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--lead-days", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--lead-steps",
        type=int,
        nargs="+",
        default=None,
        help="Render explicit autoregressive lead steps. Overrides --lead-days when set.",
    )
    parser.add_argument("--resolutions", type=int, nargs="+", default=[2, 3, 6])
    parser.add_argument("--eval-mode", default="warm")
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


def _lead_label(lead_step: int) -> str:
    hours = int(lead_step) * 6
    if hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


def _lead_steps_from_days(lead_days: list[int]) -> list[int]:
    return [int((24 * int(day)) // 6) for day in lead_days]


def _parse_variant(variant: str) -> dict[str, int | str | bool | None]:
    match = RUN_TOKEN_RE.search(str(variant))
    if not match:
        raise ValueError(f"Could not parse small experiment variant: {variant}")
    parsed: dict[str, int | str | bool | None] = {}
    for key, raw in match.groupdict().items():
        if key in {"res", "mesh", "width", "mp", "tc", "di", "ds"}:
            parsed[key] = None if raw is None else int(raw)
        elif key in {"frozen", "release"}:
            parsed[key] = raw is not None
        else:
            parsed[key] = raw
    return parsed


def _curve_key(row: pd.Series) -> str | None:
    family = str(row["family"])
    if family == "graphcast":
        return "vanilla"
    di = row["di"]
    ds = row["ds"]
    if pd.isna(di) or pd.isna(ds):
        return None
    stage = "release" if bool(row["release"]) else "frozen" if bool(row["frozen"]) else None
    if stage is None:
        return None
    if int(di) == 128 and int(ds) == 64:
        return f"{stage}_di128_ds64"
    if int(di) == 256 and int(ds) == 128:
        return f"{stage}_di256_ds128"
    return None


def _load_plot_rows(csv_path: Path, family: str, *, eval_mode: str, resolutions: set[int]) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing merged eval CSV: {csv_path}")
    df = _ensure_lead_steps(pd.read_csv(csv_path))
    df = df[
        (df["metric_kind"].astype(str) == "weighted_allvars")
        & (df["eval_mode"].astype(str) == eval_mode)
        & (df["variable"].fillna("").astype(str) == "")
        & (df["res"].astype(int).isin(resolutions))
    ].copy()
    if df.empty:
        raise ValueError(f"No weighted_allvars {eval_mode} rows found in {csv_path}")

    parsed = df["variant"].astype(str).map(_parse_variant).apply(pd.Series)
    for col in ["di", "ds", "frozen", "release"]:
        df[col] = parsed[col]
    df["curve_key"] = df.apply(_curve_key, axis=1)
    df = df[df["curve_key"].notna()].copy()
    allowed_families = {"graphcast", family}
    df = df[df["family"].astype(str).isin(allowed_families)].copy()
    if df.empty:
        raise ValueError(f"No requested curves found in {csv_path}")
    return df.sort_values(["lead_steps", "curve_key", "res", "variant"]).reset_index(drop=True)


def _plot_suite(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    audit_csv_path: Path,
    lead_steps: list[int],
    resolutions: list[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(audit_csv_path, index=False)
    print(f"Saved plotted rows: {audit_csv_path}")

    fig, axes = plt.subplots(1, len(lead_steps), figsize=(5.2 * len(lead_steps), 4.8), sharey=True)
    if len(lead_steps) == 1:
        axes = [axes]

    for ax, lead_step in zip(axes, lead_steps):
        sub = df[df["lead_steps"].astype(int) == int(lead_step)]
        for curve in CURVES:
            curve_df = sub[sub["curve_key"] == curve.key].sort_values("res")
            if curve_df.empty:
                continue
            ax.plot(
                curve_df["res"].astype(int),
                curve_df["value"].astype(float),
                color=curve.color,
                marker=curve.marker,
                linestyle=curve.linestyle,
                linewidth=2.0,
                markersize=5.5,
                label=curve.label,
            )
        ax.set_title(f"lead {_lead_label(lead_step)}")
        ax.set_xlabel("Resolution group (res)")
        ax.set_xticks(resolutions)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Normalized weighted MSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    resolutions = [int(value) for value in args.resolutions]
    resolution_set = set(resolutions)
    lead_steps = (
        sorted({int(step) for step in args.lead_steps})
        if args.lead_steps is not None
        else _lead_steps_from_days(args.lead_days)
    )
    output_dir = args.output_image_dir

    gc_df = _load_plot_rows(args.gc_mamba_csv, "gc_mamba", eval_mode=args.eval_mode, resolutions=resolution_set)
    residual_df = _load_plot_rows(
        args.residual_mamba_csv,
        "residual_mamba",
        eval_mode=args.eval_mode,
        resolutions=resolution_set,
    )

    _plot_suite(
        gc_df,
        title="Small GC-Mamba frozen/release vs vanilla | warm eval",
        out_path=output_dir / "small_gc_mamba_freeze50k_release20k_vs_vanilla_warm.png",
        audit_csv_path=output_dir / "small_gc_mamba_freeze50k_release20k_vs_vanilla_warm_plotted_rows.csv",
        lead_steps=lead_steps,
        resolutions=resolutions,
    )
    _plot_suite(
        residual_df,
        title="Small residual Mamba frozen/release vs vanilla | warm eval",
        out_path=output_dir / "small_residual_mamba_freeze50k_release20k_vs_vanilla_warm.png",
        audit_csv_path=output_dir / "small_residual_mamba_freeze50k_release20k_vs_vanilla_warm_plotted_rows.csv",
        lead_steps=lead_steps,
        resolutions=resolutions,
    )


if __name__ == "__main__":
    main()
