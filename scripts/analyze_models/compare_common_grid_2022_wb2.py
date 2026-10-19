"""Compare our 2022 GC025 rerun with the published WeatherBench2 2022 GraphCast scores.

Both are verified against ERA5 on the 1.5° grid and expressed as % of the
published 2022 ENS-mean RMSE, i.e. the units of the old Figs. 5/9 green curve.
Our RMSE pools per-start MSE over the 2022 every-third-day 00 UTC starts before
the square root. WB2's start set is not recorded in its score file.

    python scripts/analyze_models/compare_common_grid_2022_wb2.py [--shards DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Overleaf/Graphs"))
from paper_style import apply_style, FIGURE_WIDTH, LABEL_SIZE

WORKFLOW = ROOT / "artifacts/evaluations/graphcast_common_grid_2022_20261016"
WB2 = ROOT / "artifacts/evaluations/graphcast_common_grid_2023_20261015/audits/recovered_wb2"
SUMMARY_2023 = ROOT / "artifacts/evaluations/graphcast_common_grid_2023_20261015/report/summary.json"
OUT = ROOT / "plots/analyze_models/images/resolution_eval/common_grid_2022_20261016"
GREEN, BLACK = "#1fa450", "#222222"
WB2_NAMES = {"Z500": ("geopotential", 500), "Z850": ("geopotential", 850),
             "T500": ("temperature", 500), "T850": ("temperature", 850),
             "T2m": ("2m_temperature", None), "MSL": ("mean_sea_level_pressure", None),
             "Q700": ("specific_humidity", 700), "Q850": ("specific_humidity", 850),
             "U850": ("u_component_of_wind", 850), "V850": ("v_component_of_wind", 850),
             "U10": ("10m_u_component_of_wind", None), "V10": ("10m_v_component_of_wind", None)}
DIAGNOSTICS = list(WB2_NAMES)
LEADS = np.arange(6, 241, 6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pooled_rmse(shard_dir):
    starts, mse = [], []
    for path in sorted(Path(shard_dir).glob("gc025_*.json")):
        body = json.loads(path.read_text())
        assert body["status"] == "complete" and body["model"] == "gc025"
        for record in body["per_initialization"]:
            starts.append(record["initialization_time"])
            scores = record["mse"]["gc025"]
            mse.append([scores[d] for d in DIAGNOSTICS])
    assert starts and len(starts) == len(set(starts)), "Missing or duplicate starts"
    # (start, diagnostic, lead) -> (lead, diagnostic)
    return sorted(starts), np.sqrt(np.mean(np.asarray(mse), axis=0)).T


def published(name):
    data = xr.open_dataset(WB2 / name, decode_timedelta=True).sel(region="global")
    hours = (data.lead_time.values / np.timedelta64(1, "h")).astype(int)
    index = [int(np.flatnonzero(hours == h)[0]) for h in LEADS]
    columns = []
    for diagnostic in DIAGNOSTICS:
        variable, level = WB2_NAMES[diagnostic]
        field = data["rmse." + variable]
        if level is not None:
            field = field.sel(level=level)
        columns.append(field.values[index])
    return np.stack(columns, axis=1)


def write_figure(path, days, ours, wb2, labels, *, full, n_starts):
    rows, cols, height = (3, 4, 5.3) if full else (1, 3, 2.35)
    fig, axes = plt.subplots(rows, cols, figsize=(FIGURE_WIDTH, height), squeeze=False)
    if full:
        fig.subplots_adjust(left=.112, right=.988, bottom=.12, top=.845, wspace=.39, hspace=.53)
        fig.suptitle("GraphCast (0.25°), 2022, common 1.5° grid", x=.52, y=.994)
        legend_y = .937
    else:
        fig.subplots_adjust(left=.112, right=.988, bottom=.265, top=.80, wspace=.29)
        legend_y = .985
    for ax, label in zip(axes.flat, labels, strict=True):
        j = DIAGNOSTICS.index(label)
        ax.axhline(100, color="#8b919b", lw=.8, ls=":")
        ax.plot(days, wb2[:, j], color=GREEN, lw=1.7)
        ax.plot(days, ours[:, j], color=BLACK, lw=1.3, ls="--")
        ax.set_xlim(0, 10)
        ax.set_xticks(range(0, 11, 2))
        ax.tick_params(length=2.5, width=.65, pad=2.5)
        ax.set_title(label, pad=5)
    fig.legend([Line2D([], [], color=GREEN, lw=1.7), Line2D([], [], color=BLACK, lw=1.3, ls="--")],
               ["Published WB2 2022 (old green curve)", f"Our rerun, {n_starts} starts"],
               loc="upper center", bbox_to_anchor=(.52, legend_y), ncol=2, frameon=False,
               handlelength=2.1, columnspacing=1.4, borderaxespad=0)
    fig.supxlabel("Forecast lead time (days)", x=.55, y=.025 if full else .035, fontsize=LABEL_SIZE)
    fig.supylabel("RMSE vs ENS 2022 [%]", x=.015, y=.49, fontsize=LABEL_SIZE)
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=WORKFLOW / "shards")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    starts, ours = pooled_rmse(args.shards)
    wb2_gc = published("graphcast_vs_era5_240x121_2022.nc")
    wb2_ens = published("ens_mean_vs_era5_240x121_2022.nc")
    rmse_2023 = np.asarray(json.loads(SUMMARY_2023.read_text())["rmse"]["gc025"])
    ratio = 100 * (ours / wb2_gc - 1)
    rows = {}
    for j, d in enumerate(DIAGNOSTICS):
        rows[d] = {f"day{k}": {"ours_2022": float(ours[4*k-1, j]), "wb2_2022": float(wb2_gc[4*k-1, j]),
                               "ours_2023": float(rmse_2023[4*k-1, j]),
                               "ours_vs_wb2_pct": float(ratio[4*k-1, j])} for k in (1, 3, 5, 10)}
        rows[d]["ours_vs_wb2_pct_range_all_leads"] = [float(ratio[:, j].min()), float(ratio[:, j].max())]
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style()
    days = LEADS / 24
    tag = f"_{args.tag}" if args.tag else ""
    figures = [
        write_figure(OUT / f"gc025_2022_vs_wb2_all_variables{tag}.pdf", days, 100 * ours / wb2_ens,
                     100 * wb2_gc / wb2_ens, DIAGNOSTICS, full=True, n_starts=len(starts)),
        write_figure(OUT / f"gc025_2022_vs_wb2_three_panels{tag}.pdf", days, 100 * ours / wb2_ens,
                     100 * wb2_gc / wb2_ens, ["Z500", "T2m", "U850"], full=False, n_starts=len(starts)),
    ]
    report = {"n_starts": len(starts), "first_start": starts[0], "last_start": starts[-1],
              "shards": str(args.shards), "wb2_sources": {n: sha(WB2 / n) for n in (
                  "graphcast_vs_era5_240x121_2022.nc", "ens_mean_vs_era5_240x121_2022.nc")},
              "rmse_units": "physical SI", "diagnostics": rows, "figures": figures}
    (OUT / f"gc025_2022_vs_wb2{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(starts)} starts; ours vs published WB2 2022 RMSE (%):")
    print("diag    day1    day3    day5   day10 | WB2 day3  ours22 day3  ours23 day3")
    for d, r in rows.items():
        print(f"{d:5s} " + " ".join(f"{r[f'day{k}']['ours_vs_wb2_pct']:+7.1f}" for k in (1, 3, 5, 10))
              + f" | {r['day3']['wb2_2022']:9.4g} {r['day3']['ours_2022']:12.4g} {r['day3']['ours_2023']:12.4g}")
    for item in figures:
        print(item["path"])


if __name__ == "__main__":
    main()
