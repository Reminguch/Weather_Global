#!/usr/bin/env python3
"""Plot the completed November lambda pilot and historical M1 references."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_v24_mgda import check, unseal
from scripts.analyze_models.comparison_policy import POLICY, partition_models

NAME = "v24_lambda_mse_power_with_m1_20261112_eligible_20261112_v2"
SOURCE = ROOT / "plots/analyze_models/data/resolution_eval/v24_raw_checkpoint_pilot_20261110"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
FAMILIES = {"gc_mamba": ("GC–Mamba", "#0072B2", "o"),
            "full_gc": ("GC fine-tuned", "#D55E00", "s")}
LAMBDAS = (0., .25, .5, .75, 1.)


def pinned(path):
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_points():
    report = unseal(SOURCE / "report.json")
    if report["status"] != "complete" or report["count"] != 4 or report["target_steps"] != 40:
        raise ValueError("Expected the completed four-start, 40-lead pilot")
    with (SOURCE / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["model"]: row for row in report["summary"]}
    if len(rows) != 13 or len({row["model"] for row in rows}) != 13:
        raise ValueError("Expected exactly ten sweep models, original GC and two M1 references")
    points = []
    for row in rows:
        identity = row["model"]
        power, mse = float(row["power_error_rollout_pct"]), float(row["exact_gc_mse_rollout"])
        if not all(math.isfinite(value) for value in (power, mse)):
            raise ValueError("Nonfinite plotted score")
        for column in ("power_error_rollout_pct", "exact_gc_mse_rollout"):
            if not math.isclose(float(row[column]), expected[identity][column], rel_tol=1e-12):
                raise ValueError("CSV differs from the sealed source report")
        reference = identity in ("public_gc", "M1_raw", "M1_swa")
        if not reference and int(row["checkpoint_step"]) != 1250:
            raise ValueError("Unexpected November checkpoint budget")
        points.append({"model": identity, "family": row["model_family"],
            "experiment": "original_gc" if identity == "public_gc" else "historical_m1" if reference else "november_sweep",
            "lambda": None if identity == "public_gc" else 0. if reference else float(row["lambda"]),
            "checkpoint_kind": "swa" if identity == "M1_swa" else "public_pretrained" if identity == "public_gc" else "raw",
            "total_training_updates": None if identity == "public_gc" else 2500 if reference else 1250,
            "power_mismatch_pct": power, "exact_gc_mse": mse})
    for family in FAMILIES:
        if sorted(p["lambda"] for p in points if p["experiment"] == "november_sweep"
                  and p["family"] == family) != list(LAMBDAS):
            raise ValueError("Lambda coverage differs")
    manifest_record = next(source for source in report["sources"]
                           if Path(source["path"]).name == "manifest.json")
    check(manifest_record)
    manifest = unseal(manifest_record["path"])
    if manifest["record_hash"] != report["manifest_hash"]:
        raise ValueError("Source report and model manifest differ")
    included, excluded = partition_models(manifest["models"])
    eligible = {model["id"] for model in included} | {"public_gc"}
    points = [point for point in points if point["model"] in eligible]
    return points, dict(report, comparison_excluded_models=excluded)


def main():
    points, source_report = load_points()
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    with (DATA / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.8, 8.4))
    fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.20)
    ax.set_xlim(57.3, 78.5)
    ax.set_ylim(8.85, 11.60)
    ax.set_xlabel("Power-spectrum mismatch to ERA5 (%) ↓", labelpad=12)
    ax.set_ylabel("Exact GraphCast weighted MSE ↓", labelpad=12)
    ax.grid(color="#d6dce2", alpha=.65, linewidth=.7)
    ax.set_axisbelow(True)

    baseline = next(p for p in points if p["model"] == "public_gc")
    ax.axvline(baseline["power_mismatch_pct"], color="#77808a", alpha=.5, lw=1, ls=(0, (4, 4)))
    ax.axhline(baseline["exact_gc_mse"], color="#77808a", alpha=.5, lw=1, ls=(0, (4, 4)))
    offsets = {"gc_mamba": {0.: (12, -20), .25: (23, 24), .5: (-18, 40),
                            .75: (-39, -34), 1.: (20, 5)},
               "full_gc": {0.: (12, -17), .25: (12, 8), .5: (10, 12),
                           .75: (10, 10), 1.: (10, 10)}}
    for family, (label, color, marker) in FAMILIES.items():
        selected = [p for p in points if p["experiment"] == "november_sweep" and p["family"] == family]
        ax.scatter([p["power_mismatch_pct"] for p in selected], [p["exact_gc_mse"] for p in selected],
                   s=100, marker=marker, color=color, edgecolors="white", linewidths=1.2,
                   zorder=5, label=label + " · raw, 1,250 updates")
        for point in selected:
            offset = offsets[family][point["lambda"]]
            ax.annotate(f"λ = {point['lambda']:g}", (point["power_mismatch_pct"], point["exact_gc_mse"]),
                xytext=offset, textcoords="offset points", color=color, fontsize=11,
                ha="left" if offset[0] >= 0 else "right", va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": .6},
                arrowprops={"arrowstyle": "-", "color": color, "alpha": .6, "lw": .8}, zorder=6)
    ax.scatter(baseline["power_mismatch_pct"], baseline["exact_gc_mse"],
               marker="X", s=155, color="#222222", edgecolors="white", linewidths=1, zorder=7)
    ax.annotate("Original GC", (baseline["power_mismatch_pct"], baseline["exact_gc_mse"]),
                xytext=(-18, 18), textcoords="offset points", ha="right", weight="bold", fontsize=11)
    for identity, label, marker, size, filled in (
        ("M1_raw", "M1 raw · λ = 0", "D", 120, False),
        ("M1_swa", "M1 SWA · λ = 0", "*", 290, True)):
        point = next(p for p in points if p["model"] == identity)
        ax.scatter(point["power_mismatch_pct"], point["exact_gc_mse"], s=size, marker=marker,
                   facecolors="#7B3294" if filled else "white", edgecolors="#7B3294", linewidths=1.6, zorder=7)
        ax.annotate(label + "\n2,500 total updates", (point["power_mismatch_pct"], point["exact_gc_mse"]),
                    xytext=(0, 19), textcoords="offset points", ha="center", color="#7B3294",
                    fontsize=11, linespacing=1.5, weight="medium")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#dce0e4",
              framealpha=1, fontsize=11, borderpad=.8, labelspacing=.9)
    ax.text(.018, .035, "Lower-left is better", transform=ax.transAxes, color="#4e5964", fontsize=11)
    fig.suptitle("MSE vs power-spectrum mismatch", fontsize=21, weight="bold", x=.10, ha="left", y=.97)
    fig.text(.10, .902, "November λ sweep + historical M1  |  4 matched 2022 starts × 40 six-hour leads",
             fontsize=12, color="#4e5964")
    fig.text(.10, .087, "Power mismatch: pooled spherical powers; 83 channels, degrees 1–179, all 40 leads.",
             fontsize=10, color="#4e5964")
    fig.text(.10, .056, "Training recipes and budgets differ. Fine-tuned GC: λ = 0, LR 3e−7. Rejected LR 1e−5 runs excluded.",
             fontsize=10, color="#4e5964")
    files = []
    for extension in ("png", "pdf", "svg"):
        path = IMAGES / f"mse_power_lambda_with_m1.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        files.append(pinned(path))
    plt.close(fig)
    provenance = {"source_report": pinned(SOURCE / "report.json"), "source_csv": pinned(SOURCE / "summary.csv"),
        "script": pinned(Path(__file__)), "points": pinned(DATA / "points.csv"), "figures": files,
        "point_count": len(points), "initialization_times": source_report["initialization_times"],
        "comparison_policy": POLICY, "excluded_models": source_report["comparison_excluded_models"],
        "target_steps": 40, "power_definition": source_report["power_error_definition"],
        "m1_lambda_label": "0: historical native MSE without spectral penalty; distinct recipe from November lambda 0",
        "baseline_repeat_spread_rollout": source_report["baseline_repeat_spread_rollout"],
        "uncertainty": "Point estimates from four development starts; no spectral uncertainty intervals.",
        "new_evaluations": False}
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"points": len(points), "figures": [x["path"] for x in files]}, indent=2))


if __name__ == "__main__":
    main()
