#!/usr/bin/env python3
"""Make a consistent better/worse view of saved LH extreme-event scores."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weather-global-lh-extremes-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/LH_rebuttal_results/weatherdata20261111/02_extreme_events/per_lead.csv"
NAME = "lh_extremes_better_worse_20261112"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
EVENTS = (("hot", "Heat", "K"), ("cold", "Cold", "K"),
          ("wind", "Strong wind", "m/s"), ("rain", "Heavy rain (24 h)", "mm"))


def main():
    frame = pd.read_csv(SOURCE)
    frame = frame[(frame.threshold_set == "primary") & frame.metric.isin(["csi", "tail_mae"])].copy()
    if frame.duplicated(["event", "region", "metric", "lead_hours"]).any():
        raise ValueError("Duplicate source score cells")
    if not np.allclose(frame.m1_raw - frame.gc, frame.delta_m1_minus_gc, rtol=1e-12, atol=1e-12):
        raise ValueError("Saved differences disagree with model scores")
    factors = np.where(frame.metric == "csi", 100.0, -1.0)
    frame["m1_advantage"] = factors * frame.delta_m1_minus_gc
    ends = np.stack([factors * frame.delta_ci95_lower, factors * frame.delta_ci95_upper])
    frame["advantage_ci95_lower"] = ends.min(axis=0)
    frame["advantage_ci95_upper"] = ends.max(axis=0)
    frame["advantage_unit"] = np.where(frame.metric == "csi", "CSI percentage points", frame.unit)
    frame["pointwise_ci_direction"] = np.select(
        [frame.advantage_ci95_lower > 0, frame.advantage_ci95_upper < 0],
        ["M1 better", "GraphCast better"], default="interval includes zero")
    check = frame[["m1_advantage", "advantage_ci95_lower", "advantage_ci95_upper"]].to_numpy()
    if not np.isfinite(check).all():
        raise ValueError("Nonfinite plotted scores or intervals")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.1), sharex=True)
    detection = frame[frame.metric == "csi"]
    common_csi_limit = 1.13 * np.abs(detection[["advantage_ci95_lower", "advantage_ci95_upper"]]).max().max()
    for col, (event, title, unit) in enumerate(EVENTS):
        axes[0, col].set_title(title, fontsize=14, pad=10)
        for row, metric in enumerate(("csi", "tail_mae")):
            ax = axes[row, col]
            subset = frame[(frame.event == event) & (frame.metric == metric)]
            limit = common_csi_limit if row == 0 else 1.13 * np.abs(
                subset[["advantage_ci95_lower", "advantage_ci95_upper"]]).max().max()
            ax.set(xlim=(0, 10.25), ylim=(-limit, limit), xticks=(0, 2, 4, 6, 8, 10))
            ax.axhspan(0, limit, color="#dff1e7", zorder=0)
            ax.axhspan(-limit, 0, color="#f9e3df", zorder=0)
            ax.axhline(0, color="#4c4c4c", linewidth=1)
            ax.grid(axis="y", color="white", linewidth=.9)
            for region, style, color, marker, offset in (
                ("global", "-", "#143e63", "o", -.06),
                ("land", "--", "#774b8c", "s", .06),
            ):
                points = subset[subset.region == region].sort_values("lead_hours")
                expected = np.arange(24 if event == "rain" else 6, 241, 6)
                if not np.array_equal(points.lead_hours.to_numpy(), expected):
                    raise ValueError(f"Incomplete leads: {event}/{metric}/{region}")
                ax.plot(points.lead_hours / 24, points.m1_advantage,
                        linestyle=style, color=color, linewidth=1.7)
                endpoints = points[points.lead_hours.isin([24, 120, 240])]
                # Draw saved percentile intervals directly: they need not contain the point estimate.
                ax.vlines(endpoints.lead_hours / 24 + offset, endpoints.advantage_ci95_lower,
                          endpoints.advantage_ci95_upper, color=color, linewidth=1.3)
                ax.scatter(endpoints.lead_hours / 24 + offset, endpoints.m1_advantage,
                           color=color, s=15, marker=marker, zorder=4)
            if row == 0 and col == 0:
                ax.set_ylabel("CSI advantage (percentage points)\nM1 − GraphCast")
            elif row == 0:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.set_ylabel(f"MAE improvement ({unit})\nGraphCast − M1")
            if col == 0:
                ax.text(.025, .94, "M1 better ↑", color="#216441", transform=ax.transAxes,
                        fontsize=10, va="top", fontweight="bold")
                ax.text(.025, .04, "GraphCast better ↓", color="#9b3b32", transform=ax.transAxes,
                        fontsize=10, va="bottom", fontweight="bold")
    fig.suptitle("Extreme-event skill: positive always favors M1", fontsize=18, y=.99)
    fig.text(.5, .944, "2023 · 122 starts · local seed 22 · raw main step 2,000 after 500 warm-up updates",
             ha="center", fontsize=11)
    handles = [Line2D([0], [0], color="#143e63", label="Global", linewidth=2),
               Line2D([0], [0], color="#774b8c", linestyle="--", label="Land", linewidth=2),
               Patch(facecolor="#dff1e7", label="M1 better (> 0)"),
               Patch(facecolor="#f9e3df", label="GraphCast better (< 0)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .918),
               ncol=4, frameon=False, fontsize=11)
    fig.supxlabel("Forecast lead time (days)", y=.078)
    fig.text(.5, .035, "Bars: paired pointwise 95% intervals at days 1, 5 and 10. Crossing zero means the direction is uncertain.\n"
             "CSI measures event detection; MAE measures intensity error where ERA5 is extreme. MAE panels use different physical units/scales.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .105, 1, .865), h_pad=2, w_pad=1.7)
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"extremes_better_worse.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(DATA / "advantage_scores.csv", index=False)
    metadata = {
        "created_date": "2026-11-12", "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "code": str(Path(__file__).resolve().relative_to(ROOT)),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csi_transform": "100 * (M1_CSI - GC_CSI); absolute percentage-point difference, not relative percent",
        "mae_transform": "GC_tail_MAE - M1_tail_MAE; negate source difference and reverse interval endpoints",
        "intervals": "Reuse paired pointwise date-bootstrap intervals; no new uncertainty calculation",
        "scope": "Primary thresholds, global and land, local raw seed22; same evidence as original figE1",
        "bias_panels": "Original frequency-bias target is 1; original signed tail-bias target is 0. Omitted from this detection/intensity companion.",
        "row_count": len(frame),
    }
    (DATA / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(frame[(frame.region == "global") & frame.lead_hours.isin([24,120,240])]
          [["event", "lead_hours", "metric", "m1_advantage", "pointwise_ci_direction"]].to_string(index=False))
    print(IMAGES)


if __name__ == "__main__":
    main()
