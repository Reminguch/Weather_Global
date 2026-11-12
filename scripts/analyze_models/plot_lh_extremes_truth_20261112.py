#!/usr/bin/env python3
"""Plot measured ERA5 event frequency and each forecast's event frequency."""
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
SOURCE = ROOT / "results/LH_rebuttal_results/weatherdata20261111/02_extreme_events/per_lead.csv"
NAME = "lh_extremes_truth_20261112"
DATA = ROOT / "plots/analyze_models/data/resolution_eval" / NAME
IMAGES = ROOT / "plots/analyze_models/images/resolution_eval" / NAME
EVENTS = (("hot", "Heat"), ("cold", "Cold"), ("wind", "Strong wind"), ("rain", "Heavy rain (24 h)"))
SERIES = (("era5", "ERA5 observed", "black", "-", "o", -.09),
          ("gc", "Original GraphCast", "#0072b2", "-", "o", 0),
          ("m1_raw", "GC–Mamba M1, raw", "#d55e00", "--", "s", .09))


def main():
    original = pd.read_csv(SOURCE)
    source = original[original.threshold_set == "primary"]
    observed = source[source.metric == "observed_frequency"].copy()
    forecast = source[source.metric == "forecast_frequency"].copy()
    if not np.array_equal(observed.gc.to_numpy(), observed.m1_raw.to_numpy()):
        raise ValueError("Observed truth differs between the two model records")
    keys = ["event", "region", "lead_hours"]
    obs = observed[keys + ["gc", "gc_ci95_lower", "gc_ci95_upper"]].rename(
        columns={"gc": "era5", "gc_ci95_lower": "era5_ci95_lower", "gc_ci95_upper": "era5_ci95_upper"})
    fields = [model + suffix for model in ("gc", "m1_raw") for suffix in ("", "_ci95_lower", "_ci95_upper")]
    values = forecast[keys + fields].merge(obs, on=keys, validate="one_to_one")
    numeric = fields + ["era5", "era5_ci95_lower", "era5_ci95_upper"]
    if len(values) != 2 * (3 * 40 + 37) or not np.isfinite(values[numeric].to_numpy()).all():
        raise ValueError("Incomplete/nonfinite frequency records")
    if ((values[numeric] < 0) | (values[numeric] > 1)).any().any():
        raise ValueError("Frequency outside [0,1]")
    values[numeric] *= 100
    values["unit"] = "percent of eligible area, pooled across starts"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.5), sharex=True)
    for col, (event, title) in enumerate(EVENTS):
        axes[0, col].set_title(title, fontsize=14, pad=9)
        maximum = float(values[values.event == event][numeric].max().max()) * 1.10
        for row, region in enumerate(("global", "land")):
            ax = axes[row, col]
            rows = values[(values.event == event) & (values.region == region)].sort_values("lead_hours")
            if not np.array_equal(rows.lead_hours.to_numpy(), np.arange(24 if event == "rain" else 6, 241, 6)):
                raise ValueError("Incomplete frequency lead coverage")
            ax.set(xlim=(0, 10.2), ylim=(0, maximum), xticks=(0, 2, 4, 6, 8, 10))
            ax.grid(alpha=.2)
            for field, label, color, style, marker, offset in SERIES:
                ax.plot(rows.lead_hours / 24, rows[field], label=label, color=color,
                        linestyle=style, linewidth=2.1)
                endpoints = rows[rows.lead_hours.isin([24, 120, 240])]
                ax.vlines(endpoints.lead_hours / 24 + offset, endpoints[field + "_ci95_lower"],
                          endpoints[field + "_ci95_upper"], color=color, linewidth=1, alpha=.65)
                ax.scatter(endpoints.lead_hours / 24 + offset, endpoints[field], color=color,
                           marker=marker, s=16, zorder=4)
            label = "Global: land + ocean" if region == "global" else "Land only"
            if col == 0:
                ax.set_ylabel(label + "\nEvent frequency (%)")
            else:
                ax.set_ylabel("Event frequency (%)")
    fig.suptitle("How often extremes occur: ERA5 truth and forecasts", y=.985, fontsize=17)
    fig.text(.5, .938, "2023 · 122 starts · area weighted · M1 seed 22, raw main step 2,000 after 500 warm-up updates",
             ha="center", fontsize=10.5)
    handles = [Line2D([0], [0], color=color, linestyle=style, linewidth=2.1, label=label)
               for _, label, color, style, _, _ in SERIES]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.5, .918), frameon=False)
    fig.supxlabel("Forecast lead time (days)", y=.078)
    fig.text(.5, .032, "Frequency = extreme-event area / eligible area, pooled over starts. Bars: pointwise 95% date-block intervals.\n"
             "ERA5 verifies each lead; its dates/hours change with lead. Matching frequency alone does not imply correct event location or timing.",
             ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, .11, 1, .868), h_pad=2, w_pad=1.4)
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(IMAGES / f"event_frequency_truth.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    values.to_csv(DATA / "event_frequency_percent.csv", index=False)
    metadata = {
        "date": "2026-11-12", "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "code": str(Path(__file__).resolve().relative_to(ROOT)),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "truth": "100 * saved observed_frequency, verified identical in both model records",
        "forecasts": "100 * saved forecast_frequency for GC and local raw M1",
        "denominator": "Eligible spherical cell area pooled over all 122 starts, separately for each event/region/lead",
        "regions": "Global includes eligible land and ocean; land restricts mask>=0.5",
        "rain": "Lead>=24h; eligibility requires climatological 24h-rain q95>1mm",
        "thresholds": "Primary: climatological q98 heat/wind/rain, q2 cold, derived from 1991–2020",
        "limits": "Event prevalence, not event location/timing skill or mean physical intensity. Raw fields/intensity sums are absent from LH handoff.",
        "row_count": len(values),
    }
    (DATA / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(values[values.lead_hours == 240][["event", "region", "era5", "gc", "m1_raw"]].to_string(index=False))
    print(IMAGES)


if __name__ == "__main__":
    main()
