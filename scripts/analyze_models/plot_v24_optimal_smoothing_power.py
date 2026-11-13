#!/usr/bin/env python3
"""Plot paired native-grid loss and scale-dependent power for GC/M1 filtering.

Reads an already pooled interim/final analysis. No forecasts or filter fitting
are performed here. Physical powers from different channels are never added:
the degree plots average dimensionless forecast/truth ratios across channels.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

MODELS = (("GC", 0, "#2864A5"), ("M1", 1, "#DF7F22"))
VARIANTS = (("raw", 0, "-", "o"), ("optimal", 2, "--", "D"))
LEAD_DAYS = (1, 3, 7, 10)


def _date(value):
    text = str(value)
    if "-" in text:
        return text.split("T")[0]
    return datetime.strptime(text, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d")


def _mean_defined(values, axis):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    output = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(np.where(valid, values, 0.).sum(axis=axis), count,
              out=output, where=count > 0)
    return output


def _caption(summary):
    count = int(summary["count"])
    first, last = (_date(summary[key]) for key in ("first_initialization", "last_initialization"))
    extent = "incomplete 2023 evaluation" if count < 122 else "2023 evaluation"
    return (f"{count} matched initialization dates, {first} to {last}; {extent}. "
            "Filters fitted on 2022; raw forecasts feed subsequent steps.")


def _save(figure, directory, stem, caption, extra):
    # Reserve two explicit caption lines rather than allowing layout engines to
    # shrink labels or place them over the data.
    figure.text(.5, .038, caption, ha="center", va="center", fontsize=8)
    figure.text(.5, .014, extra, ha="center", va="center", fontsize=8)
    figure.tight_layout(rect=(0, .073, 1, .96))
    outputs = []
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=200, bbox_inches="tight")
        outputs.append(str(path))
    return outputs


def plot(data_dir, image_dir):
    """Create three PNG/PDF figure pairs from power_pooled.npz and summary.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir, image_dir = Path(data_dir), Path(image_dir)
    summary = json.loads((data_dir / "summary.json").read_text())
    rows = summary["rows"]
    endpoints = {(row["model"], row["variant"]): row for row in rows}
    if len(endpoints) != len(rows):
        raise ValueError("Duplicate predictor/filter summary rows")
    with np.load(data_dir / "power_pooled.npz", allow_pickle=False) as archive:
        power_ratio = archive["power_ratio"]
        power_error = archive["macro_power_error_pct"]
        native_loss = archive["exact_gc_loss"]
    if (power_ratio.ndim != 5 or power_ratio.shape[:3] != (40, 2, 3) or
            power_ratio.shape[-1] != 180 or power_error.shape != (40, 2, 3) or
            native_loss.shape != (40, 2, 3)):
        raise ValueError("Expected 40 leads, GC/M1, raw/projected/optimal, and T179 spectra")
    if not np.isfinite(native_loss).all() or np.isinf(power_ratio).any() or np.isinf(power_error).any():
        raise ValueError("Nonfinite loss or infinite spectral ratio")
    for model, _, _ in MODELS:
        for variant, _, _, _ in VARIANTS:
            if (model, variant) not in endpoints:
                raise ValueError(f"Missing {model}/{variant} summary")
    image_dir.mkdir(parents=True, exist_ok=True)
    caption = _caption(summary)
    channel_count = power_ratio.shape[-2]
    degrees, days = np.arange(1, 180), np.arange(1, 41) / 4
    outputs = []
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(9, 6))
        for model, _, color in MODELS:
            points = []
            for variant, _, _, marker in VARIANTS:
                row = endpoints[model, variant]
                x, y = row["rollout_power_error_pct"], row["exact_gc_rollout_loss"]
                if x is None or y is None or not np.isfinite([x, y]).all():
                    raise ValueError("Undefined native-loss/power-error endpoint")
                points.append((x, y))
                axis.scatter(x, y, color=color, marker=marker, s=70, zorder=3)
                offset = (7, 9) if variant == "raw" else (7, -15)
                axis.annotate(f"{model} {variant}", (x, y), xytext=offset,
                              textcoords="offset points", color=color, fontsize=10)
            axis.annotate("", xy=points[1], xytext=points[0],
                          arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6,
                                      "shrinkA": 8, "shrinkB": 8}, zorder=2)
        axis.set(xlabel="Mean absolute relative power error (%)",
                 ylabel="Exact GraphCast native-grid rollout loss")
        axis.margins(x=.25, y=.25)
        axis.grid(alpha=.2)
        figure.suptitle("Native-grid MSE and spectral-power fidelity", y=.99)
        outputs += _save(figure, image_dir, "loss_power_tradeoff", caption,
            "Arrows: raw → independently fitted optimal filter. Lower on both axes is better; power error excludes degree zero.")
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        for axis, day in zip(axes.flat, LEAD_DAYS):
            lead = 4 * day - 1
            for model, bi, color in MODELS:
                for variant, vi, style, _ in VARIANTS:
                    curve = _mean_defined(power_ratio[lead, bi, vi, :, 1:], axis=0)
                    axis.plot(degrees, curve, color=color, linestyle=style, lw=1.65,
                              label=f"{model} {variant}")
            axis.axhline(1., color="0.35", linestyle=":", lw=1., label="Ideal power ratio = 1")
            axis.set(xscale="log", xlim=(1, 179), title=f"Day {day}",
                     xlabel="Spherical degree ℓ", ylabel="Equal-channel mean forecast / truth power")
            axis.grid(alpha=.18, which="both")
        axes[0, 0].legend(fontsize=8, frameon=False)
        figure.suptitle("Forecast power relative to truth across spatial scales", y=.99)
        outputs += _save(figure, image_dir, "power_ratio_by_degree", caption,
            f"Equal mean of defined ratios across {channel_count} channels, degrees 1–179. Deficits and excesses can cancel; see absolute power error.")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 5.8))
        for model, bi, color in MODELS:
            for variant, vi, style, _ in VARIANTS:
                axis.plot(days, power_error[:, bi, vi], color=color, linestyle=style,
                          lw=1.7, label=f"{model} {variant}")
        axis.axhline(0., color="0.45", linestyle=":", lw=.9)
        axis.set(xlabel="Forecast lead (days)", ylabel="Mean absolute relative power error (%)",
                 xlim=(.25, 10))
        axis.grid(alpha=.2)
        axis.legend(fontsize=9, frameon=False)
        figure.suptitle("Power mismatch through the forecast", y=.99)
        outputs += _save(figure, image_dir, "power_error_by_lead", caption,
            f"100 × mean |forecast/truth power − 1| over {channel_count} channels and degrees 1–179, after pooling initialization dates.")
        plt.close(figure)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"figures": plot(args.data_dir, args.image_dir)}, indent=2))


if __name__ == "__main__":
    main()
