#!/usr/bin/env python3
"""Plot forecast error against exceedance of local monthly ERA5 records."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score_extremes import CAMPAIGN, LEADS, VARIABLES, WORKSPACE, read_forecast, read_receipts, ratio, utc


HISTORICAL = WORKSPACE / "data/graphcast/graphcast/dataset/prepared_stream/res1"
FIRST_MONTH = np.datetime64("2015-01", "M")
END_MONTH = np.datetime64("2022-01", "M")
BIN_EDGES = np.array([0, 1, 2, 3, 4, 5, 6, 8, np.inf])
BIN_LABELS = ("0–1", "1–2", "2–3", "3–4", "4–5", "5–6", "6–8", "≥8")
EVENTS = (("Heat records", "K"), ("Cold records", "K"), ("Wind records", "m/s"))


def monthly_records(lat, lon):
    """Reduce all 6-hourly reference frames to grid-cell extrema by calendar month."""
    for name, expected in (("lat", lat), ("lon", lon)):
        if not np.array_equal(np.load(HISTORICAL / f"coords/{name}.npy", allow_pickle=False), expected):
            raise ValueError(f"Historical and forecast grids disagree on {name}")
    times = np.load(HISTORICAL / "coords/time.npy", allow_pickle=False).astype("datetime64[h]")
    start, end = FIRST_MONTH.astype("datetime64[h]"), END_MONTH.astype("datetime64[h]")
    begin, stop = np.searchsorted(times, [start, end])
    if not np.array_equal(times[begin:stop], np.arange(start, end, np.timedelta64(6, "h"))):
        raise ValueError("Incomplete 2015–2021 six-hourly ERA5 reference")
    fields = [np.load(HISTORICAL / f"vars/{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in VARIABLES[:3]]
    if any(array.shape != (len(times), len(lat), len(lon)) or array.dtype != np.float32 for array in fields):
        raise ValueError("Unexpected historical surface array shape or dtype")

    records = np.full((12, len(EVENTS), len(lat), len(lon)), -np.inf, dtype=np.float32)
    records[:, 1] = np.inf
    months = np.arange(FIRST_MONTH, END_MONTH, dtype="datetime64[M]")
    offsets = np.searchsorted(times, np.append(months, END_MONTH).astype("datetime64[h]"))
    for index, (first, last) in enumerate(zip(offsets[:-1], offsets[1:])):
        temperature, u, v = (field[first:last] for field in fields)
        month = records[index % 12]
        np.maximum(month[0], temperature.max(axis=0), out=month[0])
        np.minimum(month[1], temperature.min(axis=0), out=month[1])
        np.maximum(month[2], np.hypot(u, v).max(axis=0), out=month[2])
    if not np.isfinite(records).all():
        raise ValueError("Nonfinite historical record")
    return records


def bin_errors(fields, records):
    temperature = fields[:, 0].astype(np.float64)
    wind = np.hypot(fields[:, 1], fields[:, 2]).astype(np.float64)
    exceedances = (temperature[0] - records[0], records[1] - temperature[0], wind[0] - records[2])
    counts = np.zeros((len(EVENTS), len(BIN_LABELS)), dtype=np.int64)
    signed = np.zeros((len(EVENTS), 2, len(BIN_LABELS)))
    squared = np.zeros_like(signed)
    for event, (excess, values) in enumerate(zip(exceedances, (temperature, temperature, wind))):
        record_cells = excess > 0
        bins = np.searchsorted(BIN_EDGES, excess[record_cells], side="right") - 1
        counts[event] = np.bincount(bins, minlength=len(BIN_LABELS))
        for model, error in enumerate(values[1:, record_cells] - values[0, record_cells]):
            signed[event, model] = np.bincount(bins, weights=error, minlength=len(BIN_LABELS))
            squared[event, model] = np.bincount(bins, weights=error ** 2, minlength=len(BIN_LABELS))
    return counts, signed, squared


def draw(output, counts, bias, rmse):
    fig, axes = plt.subplots(2, 3, sharex="col", figsize=(11, 6.6), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, .88))
    positions = np.arange(len(BIN_LABELS))
    models = (("Public GraphCast", "#777777"), ("GraphCast–Mamba (raw step 2,000)", "#0072B2"))
    for column, (title, unit) in enumerate(EVENTS):
        for row, (ylabel, scores) in enumerate((("RMSE", rmse), ("Forecast bias", bias))):
            ax = axes[row, column]
            ax.set(ylabel=f"{ylabel} ({unit})", xlim=(-.4, len(positions) - .6))
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
            for model, (label, color) in enumerate(models):
                ax.plot(positions, scores[column, model], "o-", color=color, label=label, linewidth=1.6, markersize=4)
            if row == 0:
                ax.set_title(title, pad=26)
                top = ax.secondary_xaxis("top")
                top.set_xticks(positions, [f"{count:,}" for count in counts[column]])
                top.tick_params(length=0, labelsize=7, pad=4, colors=".35")
                top.spines["top"].set_visible(False)
            else:
                ax.axhline(0, color=".45", linewidth=.8)
                ax.set_xticks(positions, BIN_LABELS)
                ax.set_xlabel(f"Record exceedance ({unit})")
    fig.suptitle("Forecast errors on record-breaking weather", y=.982, fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .930),
               borderaxespad=0, ncol=2, frameon=False)
    for extension in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-hours", type=int, choices=(24, 240), default=24)
    args = parser.parse_args()
    cache = CAMPAIGN / "forecasts"
    summary = json.loads((CAMPAIGN / "summary.json").read_bytes())
    with np.load(CAMPAIGN / "sufficient_statistics.npz", allow_pickle=False) as saved:
        dates = saved["initializations"].tolist()
    manifest, receipts = read_receipts(cache, dates)
    if manifest["provenance_sha256"] != summary["forecast_provenance_sha256"]:
        raise ValueError("The forecast caches differ from the completed evaluation")
    lat, lon = (np.load(CAMPAIGN / f"climatology/{name}.npy", allow_pickle=False) for name in ("lat", "lon"))
    records = monthly_records(lat, lon)
    print("Built 2015–2021 monthly historical records", flush=True)
    lead, = np.flatnonzero(LEADS == args.lead_hours)

    def score(start):
        fields, times = read_forecast(receipts[start], manifest, lat, lon)
        date = utc(times[lead])
        return bin_errors(fields[:, :3, lead], records[date.month - 1])

    counts = np.zeros((len(EVENTS), len(BIN_LABELS)), dtype=np.int64)
    signed = np.zeros((len(EVENTS), 2, len(BIN_LABELS)))
    squared = np.zeros_like(signed)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done, (event_counts, event_signed, event_squared) in enumerate(pool.map(score, dates), 1):
            counts += event_counts
            signed += event_signed
            squared += event_squared
            if done % 30 == 0 or done == len(dates):
                print(f"Scored {done}/{len(dates)} forecast starts", flush=True)
    bias = ratio(signed, counts[:, None, :])
    rmse = np.sqrt(ratio(squared, counts[:, None, :]))
    if np.any(rmse + 1e-12 < np.abs(bias)) or not np.all(counts.sum(axis=1) > 0):
        raise ValueError("Invalid record-exceedance scores")
    print("Bin counts for heat, cold and wind:", counts.tolist(), flush=True)
    with TemporaryDirectory(dir=CAMPAIGN) as staging:
        draw(Path(staging) / f"record_exceedance_{args.lead_hours}h", counts, bias, rmse)
        for path in Path(staging).iterdir():
            path.replace(CAMPAIGN / path.name)


if __name__ == "__main__":
    main()
