#!/usr/bin/env python3
"""Plot saved absolute extreme-event scores separately over global and land cells."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weather-global-lh-extremes-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/LH_rebuttal_results/weatherdata20261111/02_extreme_events/per_lead.csv"
SCORER = SOURCE.parent / "provenance/score_extremes.py"
NAME = "lh_extremes_absolute_20261112"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
EVENTS = (("hot", "Heat", "K"), ("cold", "Cold", "K"),
          ("wind", "Strong wind", "m/s"), ("rain", "Heavy rain (24 h)", "mm"))
MODELS = (("gc", "Original GraphCast", "#0072b2", "-", "o", -.055),
          ("m1_raw", "GC–Mamba M1, raw", "#d55e00", "--", "s", .055))


def main():
    scores = pd.read_csv(SOURCE)
    scores = scores[(scores.threshold_set == "primary") & scores.metric.isin(["csi", "tail_mae"])].copy()
    columns = [f"{model}{suffix}" for model, *_ in MODELS
               for suffix in ("", "_ci95_lower", "_ci95_upper")]
    if scores.duplicated(["event", "region", "metric", "lead_hours"]).any():
        raise ValueError("Duplicate scores")
    if not np.isfinite(scores[columns].to_numpy()).all():
        raise ValueError("Nonfinite plotted scores/intervals")
    if (scores[["gc", "m1_raw"]] < 0).any().any():
        raise ValueError("Negative CSI/MAE")
    if (scores.loc[scores.metric == "csi", ["gc", "m1_raw"]] > 1).any().any():
        raise ValueError("CSI outside [0,1]")
    expected_total = 2 * 2 * (3 * 40 + 37)
    if len(scores) != expected_total:
        raise ValueError("Unexpected score count")
    max_mae = {event: float(scores[(scores.event == event) & (scores.metric == "tail_mae")]
                            [["gc_ci95_upper", "m1_raw_ci95_upper", "gc", "m1_raw"]].max().max()) * 1.08
               for event, *_ in EVENTS}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    with PdfPages(IMAGES / "extremes_absolute_global_and_land.pdf") as book:
        for region, scope in (("global", "Global: land + ocean"), ("land", "Land only: land mask ≥ 0.5")):
            fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.6), sharex=True)
            for col, (event, title, unit) in enumerate(EVENTS):
                axes[0, col].set_title(title, fontsize=14, pad=9)
                for row, metric in enumerate(("csi", "tail_mae")):
                    ax = axes[row, col]
                    values = scores[(scores.region == region) & (scores.event == event)
                                    & (scores.metric == metric)].sort_values("lead_hours")
                    if not np.array_equal(values.lead_hours.to_numpy(), np.arange(24 if event == "rain" else 6, 241, 6)):
                        raise ValueError(f"Missing lead in {region}/{event}/{metric}")
                    ax.set(xlim=(0, 10.2), xticks=(0, 2, 4, 6, 8, 10),
                           ylim=(0, 1.03 if metric == "csi" else max_mae[event]))
                    ax.grid(alpha=.2)
                    ax.axhline(1 if metric == "csi" else 0, color="black", linestyle="-.",
                               linewidth=1.7, zorder=3, clip_on=False)
                    for model, label, color, style, marker, offset in MODELS:
                        ax.plot(values.lead_hours / 24, values[model], color=color,
                                linestyle=style, linewidth=2.1, label=label)
                        endpoints = values[values.lead_hours.isin([24, 120, 240])]
                        ax.vlines(endpoints.lead_hours / 24 + offset, endpoints[f"{model}_ci95_lower"],
                                  endpoints[f"{model}_ci95_upper"], color=color, linewidth=1.1, alpha=.85)
                        ax.scatter(endpoints.lead_hours / 24 + offset, endpoints[model],
                                   color=color, marker=marker, s=17, zorder=4)
                    if metric == "csi":
                        ax.set_yticks(np.arange(0, 1.01, .2))
                        if col == 0:
                            ax.set_ylabel("Event-detection score (CSI)\nHigher is better ↑")
                        else:
                            ax.tick_params(labelleft=False)
                        if col == 3:
                            ax.text(.98, .95, "ERA5 vs itself: CSI = 1", transform=ax.transAxes,
                                    fontsize=9, ha="right", va="top", color=".35")
                    else:
                        ax.set_ylabel(f"Extreme-event MAE ({unit})\nLower is better ↓")
                        ax.text(.98, .04, "ERA5 vs itself: MAE = 0", transform=ax.transAxes,
                                fontsize=8, ha="right", va="bottom", color=".35")
            fig.suptitle(f"Absolute extreme-event scores — {scope}", fontsize=17, y=.985)
            fig.text(.5, .936, "2023 · 122 starts · area weighted · M1 seed 22, raw main step 2,000 after 500 warm-up updates",
                     ha="center", fontsize=10.5)
            handles = [Line2D([0], [0], color=color, linestyle=style, linewidth=2.2, label=label)
                       for _, label, color, style, _, _ in MODELS]
            handles.append(Line2D([0], [0], color="black", linestyle="-.", linewidth=1.7,
                                  label="ERA5 self-reference (ideal score)"))
            fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.5, .916), frameon=False)
            fig.supxlabel("Forecast lead time (days)", y=.077)
            fig.text(.5, .032, "Actual model scores, without model-to-model subtraction. Axes are identical between the global and land figures.\n"
                     "MAE is evaluated where ERA5 is extreme. Bars: each model's pointwise 95% interval at days 1, 5 and 10.",
                     ha="center", fontsize=9.5)
            fig.tight_layout(rect=(0, .11, 1, .868), h_pad=2, w_pad=1.4)
            for ext in ("png", "pdf"):
                fig.savefig(IMAGES / f"extremes_absolute_{region}.{ext}", dpi=180, bbox_inches="tight")
            book.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    scores.to_csv(DATA / "absolute_scores.csv", index=False)
    endpoints = scores[scores.lead_hours == 240]
    endpoints.to_csv(DATA / "day10_scores.csv", index=False)
    def pin(path):
        return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = {
        "date": "2026-11-12", "sources": [pin(SOURCE), pin(SCORER)], "code": pin(Path(__file__).resolve()),
        "global": "All eligible land and ocean cells; exact spherical grid-cell area weights",
        "land": "Subset with land_mask >= 0.5, weighted by full grid-cell area; not fractional-land weighting",
        "rain_eligibility": "Climatological 24h-rain q95 > 1 mm; scores begin at lead 24h",
        "plotted_values": "Untransformed saved gc and m1_raw CSI and tail_mae values, primary thresholds",
        "tail_mae": "Area-weighted mean absolute physical error on observed ERA5 threshold exceedances, including missed events",
        "intervals": "Saved per-model pointwise 95% date-block intervals; overlap is not a paired difference test",
        "truth_reference": "ERA5 scored against itself has CSI=1 and MAE=0; black ideal lines, not an extra measured forecast",
        "axis_limits": {"csi": [0, 1.03], "tail_mae_max_by_event": max_mae},
        "scope": "Local raw seed22 versus independent public GC; these are scores, not mean forecast physical fields",
        "row_count": len(scores),
    }
    (DATA / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(endpoints[["region", "event", "metric", "unit", "gc", "m1_raw"]].to_string(index=False))
    print(IMAGES)


if __name__ == "__main__":
    main()
