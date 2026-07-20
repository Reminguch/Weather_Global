#!/usr/bin/env python3
"""Plot vanilla-GraphCast resolution curves, retaining mesh variants as points."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "plots/analyze_models/data/resolution_eval/vanilla_gc_3y_7y_nores1"
DEFAULT_INPUT_CSV = DEFAULT_DATA_DIR / "resolution_eval.csv"
DEFAULT_OUTPUT_CSV = DEFAULT_DATA_DIR / "filtered_res4_m3_only_no_res9_m2_no_mp2_resolution_eval.csv"
DEFAULT_IMAGE_DIR = ROOT / "plots/analyze_models/images/resolution_eval/vanilla_gc_3y_7y_nores1/mp_mesh_curves_res4_m3_only_no_res9_m2_no_mp2"

VARIANT_RE = re.compile(
    r"vanilla_gc_(?P<years>\d+)y_.*?res(?P<res>\d+)_m(?P<mesh>\d+)_w(?P<width>\d+)_mp(?P<mp>\d+)(?:_|$)"
)
MESH_MARKERS = {2: "s", 3: "^", 4: "o"}
YEAR_COLORS = {3: "#2166ac", 7: "#d95f02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--no-output-csv", action="store_true")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--years", type=int, nargs="+", default=[3, 7])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[2, 3, 4, 6, 9, 15, 18])
    parser.add_argument("--mp", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--lead-day", type=int, default=4)
    parser.add_argument("--eval-mode", default="cold")
    parser.add_argument(
        "--exclude-res-mesh",
        type=int,
        nargs=2,
        action="append",
        metavar=("RES", "MESH"),
        default=[[4, 4], [9, 2]],
        help="Remove a resolution/mesh variant from every plotted curve (default: 4 4 and 9 2).",
    )
    parser.add_argument(
        "--exclude-mp",
        type=int,
        nargs="+",
        default=[2],
        help="Remove message-passing counts from the filtered CSV and plots (default: 2).",
    )
    return parser.parse_args()


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["variant"].astype(str).str.extract(VARIANT_RE)
    if parsed.isna().any(axis=None):
        bad = df.loc[parsed.isna().any(axis=1), "variant"].unique().tolist()
        raise ValueError(f"Could not parse variants: {bad}")
    parsed = parsed.astype(int)
    if not (df["res"].astype(int) == parsed["res"]).all():
        raise ValueError("The CSV res column disagrees with a variant res token.")
    return df.assign(years=parsed["years"], mesh=parsed["mesh"], mp=parsed["mp"])


def offset_x(sub: pd.DataFrame) -> pd.DataFrame:
    """Separate same-resolution mesh variants, matching the source figure convention."""
    out = sub.copy()
    out["plot_res"] = out["res"].astype(float)
    for res, group in out.groupby("res"):
        meshes = sorted(group["mesh"].astype(int).unique())
        if len(meshes) > 1:
            offsets = {mesh: (i - (len(meshes) - 1) / 2) * 0.08 for i, mesh in enumerate(meshes)}
            out.loc[group.index, "plot_res"] = group["res"].astype(float) + group["mesh"].map(offsets)
    return out


def plot_mp(sub: pd.DataFrame, *, mp: int, y_limits: tuple[float, float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.3))
    legend_handles: list[Line2D] = []
    for year in sorted(sub["years"].astype(int).unique()):
        by_year = sub[sub["years"].astype(int) == year]
        ax.plot(
            by_year.sort_values("plot_res")["plot_res"],
            by_year.sort_values("plot_res")["value"],
            color=YEAR_COLORS.get(year),
            linewidth=2.1,
            label=f"{year}y",
        )
        for mesh in sorted(by_year["mesh"].astype(int).unique(), reverse=True):
            points = by_year[by_year["mesh"].astype(int) == mesh].sort_values("plot_res")
            ax.scatter(
                points["plot_res"],
                points["value"],
                color=YEAR_COLORS.get(year),
                marker=MESH_MARKERS.get(mesh, "o"),
                s=70,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
    for mesh in sorted(sub["mesh"].astype(int).unique(), reverse=True):
        legend_handles.append(
            Line2D(
                [], [], color="#555555", marker=MESH_MARKERS.get(mesh, "o"),
                linestyle="None", markersize=8.5, markeredgecolor="black", markeredgewidth=0.5,
                label=f"mesh={mesh}",
            )
        )

    ax.set_title(f"Vanilla GraphCast MP={mp} weighted_allvars vs res | lead=4d | cold")
    ax.set_xlabel("Resolution group (res); mesh variants are offset")
    ax.set_ylabel("Normalized weighted all-variable MSE")
    ax.set_xticks(sorted(sub["res"].astype(int).unique()))
    ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.35)
    year_legend = ax.legend(title="Training duration", loc="upper left")
    ax.add_artist(year_legend)
    ax.legend(handles=legend_handles, title="Mesh", fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Saved image: {out_path}")


def main() -> None:
    args = parse_args()
    df = annotate(pd.read_csv(args.input_csv))
    exclusions = {(int(res), int(mesh)) for res, mesh in args.exclude_res_mesh}
    df = df[~df.apply(lambda row: (int(row.res), int(row.mesh)) in exclusions, axis=1)].copy()
    df = df[~df["mp"].astype(int).isin(args.exclude_mp)].copy()
    if not args.no_output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved filtered CSV: {args.output_csv}")

    metric = df[
        (df["metric_kind"] == "weighted_allvars")
        & (df["variable"].fillna("") == "")
        & (df["lead_days"].astype(int) == args.lead_day)
        & (df["eval_mode"].astype(str) == args.eval_mode)
        & (df["years"].astype(int).isin(args.years))
        & (df["res"].astype(int).isin(args.resolutions))
        & (df["mp"].astype(int).isin(args.mp))
    ].copy()
    if metric.empty:
        raise ValueError("No rows left after applying plot filters.")
    metric = offset_x(metric)
    y_limits = (metric["value"].min() - 0.25, metric["value"].max() + 0.25)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    for mp in args.mp:
        sub = metric[metric["mp"].astype(int) == mp]
        if not sub.empty:
            plot_mp(
                sub,
                mp=mp,
                y_limits=y_limits,
                out_path=args.image_dir / f"vanilla_gc_3y_7y_weighted_allvars_mp{mp}_mesh_vs_res_lead4d_cold.png",
            )


if __name__ == "__main__":
    main()
