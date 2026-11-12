#!/usr/bin/env python3
"""Replot LH record-exceedance bins with the supplied style and ideal truth line."""
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/LH_rebuttal_results/weatherdata20261111/02_extreme_events/record_exceedance_bins.csv"
NAME = "lh_record_exceedance_20261112"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
EVENTS = (("heat", "Heat records", "K"), ("cold", "Cold records", "K"), ("wind", "Wind records", "m/s"))
MODELS = (("public_graphcast", "Public GraphCast", "#777777"),
          ("m1_seed22_raw2000", "GraphCast–Mamba, raw main step 2,000", "#0072b2"))


def main():
    frame = pd.read_csv(SOURCE)
    frame = frame[frame.lead_hours == 240].copy()
    if len(frame) != 3 * 2 * 8 or frame.duplicated(["event", "model", "bin"]).any():
        raise ValueError("Incomplete or duplicate record-bin data")
    if not np.isfinite(frame[["bias", "rmse", "count"]]).all().all() or (frame["count"] <= 0).any():
        raise ValueError("Invalid record-bin scores")
    if not np.allclose(frame.bias, frame.sum_signed_error / frame["count"], rtol=1e-12, atol=1e-12):
        raise ValueError("Bias does not match saved sufficient statistics")
    if not np.allclose(frame.rmse, np.sqrt(frame.sum_squared_error / frame["count"]), rtol=1e-12, atol=1e-12):
        raise ValueError("RMSE does not match saved sufficient statistics")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), sharex="col")
    positions = np.arange(8)
    for col, (event, title, unit) in enumerate(EVENTS):
        reference = frame[(frame.event == event) & (frame.model == "public_graphcast")].sort_values("lower_edge")
        bins = reference["bin"].tolist()
        counts = reference["count"].to_numpy()
        for row, metric in enumerate(("rmse", "bias")):
            ax = axes[row, col]
            ax.set(xlim=(-.4, 7.4))
            ax.grid(axis="y", alpha=.22)
            ax.set_axisbelow(True)
            for model, label, color in MODELS:
                scores = frame[(frame.event == event) & (frame.model == model)].sort_values("lower_edge")
                if scores["bin"].tolist() != bins or not np.array_equal(scores["count"], counts):
                    raise ValueError("Model bins or truth-selected samples differ")
                ax.plot(positions, scores[metric], "o-", color=color, label=label, linewidth=1.8, markersize=4)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=3)
            if row == 0:
                ax.set_ylabel(f"RMSE ({unit})\nLower is better ↓")
                ax.set_ylim(bottom=-.35)
                ax.set_title(title, fontsize=14, pad=31)
                top = ax.secondary_xaxis("top")
                top.set_xticks(positions, [f"{n:,}" for n in counts])
                top.tick_params(length=0, labelsize=8, pad=4, colors=".35")
                top.spines["top"].set_visible(False)
            else:
                ax.set_ylabel(f"Mean forecast − ERA5 ({unit})\nCloser to zero is better")
                ax.set_xticks(positions, bins)
                ax.set_xlabel(f"Observed record-exceedance bin ({unit})")
                if event == "cold":
                    ax.text(.03, .94, "Positive = forecast too warm", transform=ax.transAxes,
                            fontsize=8.5, va="top", color=".35")
                else:
                    ax.text(.03, .94, "Negative = underprediction", transform=ax.transAxes,
                            fontsize=8.5, va="top", color=".35")
    fig.suptitle("10-day forecast errors on record-breaking weather", y=.985, fontsize=17)
    fig.text(.5, .944, "Fixed lead: 240 h · ERA5-selected record breaks · 122 starts · records from 2015–2021",
             ha="center", fontsize=10.5)
    handles = [Line2D([0], [0], color=color, marker="o", linewidth=1.8, label=label)
               for _, label, color in MODELS]
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                          label="ERA5 self-reference: zero error"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .922), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .033, "Counts above panels are grid-cell/forecast-start pairs, with equal weight per pair; no land/ocean split.\n"
             "Local monthly record: maximum for heat/wind, minimum for cold. No uncertainty intervals are supplied for these bins.",
             ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, .075, 1, .84), h_pad=2, w_pad=1.5)
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"record_exceedance_240h_truth.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(DATA / "record_exceedance_240h.csv", index=False)
    def pin(path):
        return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata = {
        "date": "2026-11-12", "source": pin(SOURCE), "code": pin(Path(__file__).resolve()),
        "source_methods": pin(SOURCE.parent / "provenance/plot_record_exceedance.py"),
        "lead_hours": 240,
        "record": "Per-grid-cell/calendar-month maximum or minimum over all 6-hourly ERA5 samples during 2015–2021",
        "exceedance": "Heat/wind: ERA5 minus historical maximum; cold: historical minimum minus ERA5. Strictly positive only.",
        "aggregation": "Same ERA5-selected cell/start pairs for both models, equal pair weights, global grid without regional split",
        "scores": "RMSE=sqrt(sum(error^2)/count), bias=sum(forecast-ERA5)/count; source scores verified and unchanged",
        "truth_reference": "ERA5 scored against itself has zero RMSE and bias, not a raw temperature/wind curve",
        "limits": "Correlated cell/start pairs; no per-date bin statistics or confidence intervals supplied; no event-detection/false-alarm assessment",
    }
    (DATA / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(IMAGES)


if __name__ == "__main__":
    main()
