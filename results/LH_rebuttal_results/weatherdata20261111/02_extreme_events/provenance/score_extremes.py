#!/usr/bin/env python3
"""Score raw M1 extreme events against independent public GraphCast forecasts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


CAMPAIGN = Path(__file__).resolve().parent
WORKSPACE = CAMPAIGN.parents[2]
sys.path.insert(0, str(WORKSPACE))
from scripts.experiments.run_v24_2023_eval import make_initializations


LEADS = np.arange(6, 241, 6)
ENDPOINTS = (24, 120, 240)
BRANCHES = ("truth", "baseline", "full")
VARIABLES = ("2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation_6hr")
EVENTS = (
    ("hot", "High temperature", "t2m", (98, 95), "K"),
    ("cold", "Low temperature", "t2m", (2, 5), "K"),
    ("wind", "Strong wind", "ws10", (98, 95), "m/s"),
    ("rain", "Heavy precipitation", "tp24", (98, 95), "mm"),
)
REGIONS = ("global", "land")
FAMILIES = ("primary", "sensitivity")
STATS = ("eligible_area", "observed_area", "forecast_area", "hit_area", "tail_absolute_error_area",
         "tail_signed_error_area", "tail_squared_error_area", "eligible_count", "observed_count",
         "forecast_count", "hit_count")
BOOTSTRAP = {"replicates": 2000, "block_starts": 5, "seed": 20261106}
METRIC_NAMES = ("csi", "recall", "precision", "far", "frequency_bias", "observed_frequency",
                "forecast_frequency", "tail_mae", "tail_bias", "tail_rmse")
CSV_FIELDS = ("event", "threshold_set", "quantile", "region", "lead_hours", "metric", "unit", "gc", "m1_raw",
              "delta_m1_minus_gc", "delta_ci95_lower", "delta_ci95_upper", "gc_ci95_lower", "gc_ci95_upper",
              "m1_raw_ci95_lower", "m1_raw_ci95_upper", "paired_valid_draws", "initialization_count")


def utc(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.utcoffset() != timedelta(0):
        raise ValueError(f"Expected UTC timestamp: {value}")
    return date


def area_weights(latitude, longitude):
    """Fraction of the sphere in each regular grid cell, including the poles."""
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    lat_order = np.argsort(latitude)
    ascending = latitude[lat_order]
    lat_diffs = np.diff(ascending)
    lon_diffs = np.diff(np.r_[np.sort(longitude % 360), np.min(longitude % 360) + 360])
    if (len(latitude) < 2 or len(longitude) < 2 or ascending[0] != -90 or ascending[-1] != 90
            or not np.allclose(lat_diffs, 180 / (len(latitude) - 1))
            or not np.allclose(lon_diffs, 360 / len(longitude))):
        raise ValueError("Expected a regular global grid with both poles")
    edges = np.r_[-90, (ascending[:-1] + ascending[1:]) / 2, 90]
    weights = np.empty_like(latitude)
    weights[lat_order] = np.diff(np.sin(np.deg2rad(edges))) / (2 * len(longitude))
    return np.broadcast_to(weights[:, None], (len(latitude), len(longitude)))


def score_start(fields, valid_times, climatology, weights, land):
    """Keep additive, paired statistics for each event, threshold, region, model and lead."""
    hours = []
    days = []
    for value in valid_times:
        date = utc(value)
        if date.hour % 6 or date.minute or date.second or date.microsecond:
            raise ValueError(f"Climatology requires synoptic UTC times: {value}")
        hours.append(date.hour // 6)
        days.append(date.replace(year=2001).timetuple().tm_yday - 1)
    days, hours = np.asarray(days), np.asarray(hours)
    rain = sum(fields[:, 3, offset:offset + len(LEADS) - 3].astype(np.float64) for offset in range(4)) * 1000
    observables = (fields[:, 0], np.hypot(fields[:, 1], fields[:, 2]), rain)
    rain_eligible = climatology["tp24_q95"][days[3:], hours[3:]] > 1
    output = np.zeros((len(EVENTS), 2, 2, 2, len(LEADS), len(STATS)), dtype=np.float64)
    for event_index, (event, _, variable, quantiles, _) in enumerate(EVENTS):
        rain_event = event == "rain"
        eligible_leads = slice(3 if rain_event else 0, None)
        data = observables[{"t2m": 0, "ws10": 1, "tp24": 2}[variable]]
        compare = np.less if event == "cold" else np.greater
        errors = np.subtract(data[1:], data[0], dtype=np.float64)
        for family, quantile in enumerate(quantiles):
            threshold = climatology[f"{variable}_q{quantile:02d}"][days[eligible_leads], hours[eligible_leads]]
            if not np.isfinite(threshold).all():
                raise ValueError(f"Nonfinite {variable} q{quantile} threshold")
            valid = rain_eligible if rain_event else np.ones_like(threshold, dtype=bool)
            observed = compare(data[0], threshold) & valid
            for branch, error in enumerate(errors):
                forecast = compare(data[branch + 1], threshold) & valid
                hit = observed & forecast
                tail = np.where(observed, error, 0)
                # The first axis of weights contains global and land weights.
                def weighted(value):
                    return np.einsum("lyx,ryx->rl", value, weights, optimize=False)

                masks = (valid, observed, forecast, hit)
                area = [weighted(value) for value in (*masks, np.abs(tail), tail, tail**2)]
                count = [np.stack((mask.sum(axis=(1, 2)), (mask & land).sum(axis=(1, 2)))) for mask in masks]
                output[event_index, family, :, branch, eligible_leads, :] = np.stack((*area, *count), axis=-1)
    return output


def circular_counts(size, *, replicates, block_starts, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, size, size=(replicates, (size + block_starts - 1) // block_starts))
    indices = ((starts[..., None] + np.arange(block_starts)) % size).reshape(replicates, -1)[:, :size]
    counts = np.zeros((replicates, size), dtype=np.float64)
    np.add.at(counts, (np.arange(replicates)[:, None], indices), 1)
    return counts


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def metrics(stats):
    area, obs, forecast, hit, absolute, signed, squared, count, obs_count, forecast_count, hit_count = np.moveaxis(stats, -1, 0)
    return {
        "csi": ratio(hit, obs + forecast - hit), "recall": ratio(hit, obs),
        "precision": ratio(hit, forecast), "far": ratio(forecast - hit, forecast),
        "frequency_bias": ratio(forecast, obs), "observed_frequency": ratio(obs, area),
        "forecast_frequency": ratio(forecast, area), "tail_mae": ratio(absolute, obs),
        "tail_bias": ratio(signed, obs), "tail_rmse": np.sqrt(ratio(squared, obs)),
        "hit_area": hit, "miss_area": obs - hit, "false_alarm_area": forecast - hit,
        "correct_negative_area": np.maximum(area - obs - forecast + hit, 0), "eligible_area": area,
        "observed_area": obs, "forecast_area": forecast,
        "hit_count": hit_count, "miss_count": obs_count - hit_count, "false_alarm_count": forecast_count - hit_count,
        "correct_negative_count": count - obs_count - forecast_count + hit_count, "eligible_count": count,
        "observed_count": obs_count, "forecast_count": forecast_count,
    }


def interval(values):
    valid = values[np.isfinite(values)]
    return (np.quantile(valid, [.025, .975]).tolist() if valid.size else [None, None], int(valid.size))


def table_rows(samples, counts):
    n = samples.shape[0]
    for event_idx, (event, _, _, quantiles, unit) in enumerate(EVENTS):
        for family_idx, family in enumerate(FAMILIES):
            for region_idx, region in enumerate(REGIONS):
                cell = samples[:, event_idx, family_idx, region_idx]
                point = metrics(cell.sum(axis=0))
                draws = metrics((counts @ cell.reshape(n, -1)).reshape(-1, *cell.shape[1:]))
                for lead_idx in range(3 if event == "rain" else 0, len(LEADS)):
                    for name, values in point.items():
                        gc, m1 = values[:, lead_idx]
                        delta = m1 - gc
                        ci, gc_ci, m1_ci, valid = [None, None], [None, None], [None, None], None
                        if name in METRIC_NAMES:
                            gc_draws, m1_draws = draws[name][:, :, lead_idx].T
                            ci, valid = interval(m1_draws - gc_draws)
                            gc_ci, _ = interval(gc_draws)
                            m1_ci, _ = interval(m1_draws)
                        yield {
                            "event": event, "threshold_set": family, "quantile": quantiles[family_idx], "region": region,
                            "lead_hours": int(LEADS[lead_idx]), "metric": name,
                            "unit": unit if name.startswith("tail_") else ("cells" if name.endswith("_count") else
                                    "fraction_of_globe_summed_over_starts" if name.endswith("_area") else "dimensionless"),
                            "gc": float(gc) if np.isfinite(gc) else None,
                            "m1_raw": float(m1) if np.isfinite(m1) else None,
                            "delta_m1_minus_gc": float(delta) if np.isfinite(delta) else None,
                            "delta_ci95_lower": ci[0], "delta_ci95_upper": ci[1], "paired_valid_draws": valid,
                            "gc_ci95_lower": gc_ci[0], "gc_ci95_upper": gc_ci[1],
                            "m1_raw_ci95_lower": m1_ci[0], "m1_raw_ci95_upper": m1_ci[1], "initialization_count": n,
                        }


def read_receipts(cache, dates):
    manifest = json.loads((cache.parent / "forecast_cache_manifest.json").read_bytes())
    provenance = manifest["provenance_sha256"]
    if (manifest["initialization_times"] != dates or tuple(manifest["branches"]) != BRANCHES
            or tuple(manifest["variables"]) != VARIABLES or manifest["lead_hours"] != LEADS.tolist()
            or manifest["schema_version"] != 1 or manifest["shard_count"] != 8
            or manifest["shape"] != [3, 4, 40, 181, 360]):
        raise ValueError("Forecast cache manifest differs from the fixed A2 protocol")
    receipts = {}
    for shard in range(manifest["shard_count"]):
        record = json.loads((cache / f"shard-{shard:02d}.complete.json").read_bytes())
        if (record["status"] != "complete" or record["shard_index"] != shard
                or record["provenance_sha256"] != provenance
                or {item["initialization_time"] for item in record["files"]} != set(dates[shard::8])):
            raise ValueError(f"Invalid completion receipt for shard {shard}")
        for item in record["files"]:
            date = item["initialization_time"]
            identifier = utc(date).strftime("%Y%m%dT%H%M%SZ")
            if (item["initialization_id"] != identifier or Path(item["path"]).resolve() != cache / f"{identifier}.npz"
                    or date in receipts):
                raise ValueError(f"Invalid or duplicate forecast receipt for {date}")
            receipts[date] = item
    return manifest, receipts


def read_forecast(receipt, manifest, latitude, longitude):
    date = receipt["initialization_time"]
    path = Path(receipt["path"])
    with path.open("rb") as stream:
        if (path.stat().st_size != receipt["size_bytes"]
                or hashlib.file_digest(stream, "sha256").hexdigest() != receipt["sha256"]):
            raise ValueError(f"Forecast file differs from its completion receipt: {path}")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            valid_times = [(utc(date) + timedelta(hours=int(h))).isoformat().replace("+00:00", "Z") for h in LEADS]
            if (archive["initialization_time"].item() != date or archive["valid_times"].tolist() != valid_times
                    or archive["provenance_sha256"].item() != manifest["provenance_sha256"]
                    or tuple(archive["branches"]) != BRANCHES or tuple(archive["variables"]) != VARIABLES
                    or not np.array_equal(archive["lead_hours"], LEADS)
                    or not np.array_equal(archive["lat"], latitude) or not np.array_equal(archive["lon"], longitude)):
                raise ValueError(f"Forecast date or grid differs from the climatology and campaign: {path}")
            fields = archive["fields"]
    if fields.dtype != np.float32 or list(fields.shape) != manifest["shape"] or not np.isfinite(fields).all():
        raise ValueError(f"Invalid forecast fields: {path}")
    return fields, valid_times


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    index = {(r["event"], r["region"], r["metric"], r["lead_hours"]): r for r in rows if r["threshold_set"] == "primary"}
    fig, axes = plt.subplots(4, 4, sharex=True, figsize=(12, 9.3), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, .875))
    metrics_to_plot = (
        ("csi", "Δ CSI\n(higher is better)", 0),
        ("frequency_bias", "Frequency bias\n(target = 1)", 1),
        ("tail_bias", "Tail bias\n(target = 0)", 0),
        ("tail_mae", "Δ tail MAE\n(lower is better)", 0),
    )
    for col, (event, label, _, _, unit) in enumerate(EVENTS):
        axes[0, col].set_title(label)
        leads = LEADS[3:] if event == "rain" else LEADS
        days = leads / 24
        for row, (metric, label, guide) in enumerate(metrics_to_plot):
            ax = axes[row, col]
            ax.axhline(guide, color="0.55", linewidth=.7, linestyle=":")
            ax.set(xlim=(0, 10), xticks=(0, 2, 4, 6, 8, 10))
            ax.grid(axis="y", alpha=.2)
            if col == 0:
                ax.set_ylabel(label)
            if row >= 2:
                ax.text(.98, .96, unit, transform=ax.transAxes, ha="right", va="top", fontsize=8)
            for region, style in (("global", "-"), ("land", "--")):
                records = [index[event, region, metric, int(lead)] for lead in leads]
                lines = (("delta_m1_minus_gc", "delta", "#7A3E99"),) if row in (0, 3) else (
                    ("gc", "gc", "#0072B2"), ("m1_raw", "m1_raw", "#D55E00"),
                )
                for value, ci, color in lines:
                    ax.plot(days, np.asarray([r[value] for r in records], dtype=float), style, color=color, linewidth=1.2)
                    for r in records:
                        if r["lead_hours"] in ENDPOINTS and r[f"{ci}_ci95_lower"] is not None:
                            day = r["lead_hours"] / 24 - (.04 if region == "land" else .12)
                            ax.vlines(day, r[f"{ci}_ci95_lower"], r[f"{ci}_ci95_upper"], color=color, linewidth=.75)
    fig.suptitle("2023 extreme-event evaluation", y=.982, fontsize=14)
    fig.supxlabel("Forecast lead time (days)")
    legend = [Line2D([0], [0], color=color, label=label) for color, label in (
        ("#0072B2", "Public GraphCast"),
        ("#D55E00", "M1 (step 2,000, seed 22)"),
        ("#7A3E99", "Score difference (Δ)"),
        ("0.2", "Global"),
    )]
    legend.append(Line2D([0], [0], color="0.2", linestyle="--", label="Land"))
    fig.legend(handles=legend, ncol=5, loc="upper center", bbox_to_anchor=(.5, .942),
               borderaxespad=0, frameon=False)
    fig.text(.5, .904, "Δ = M1 − GraphCast. Tail errors use observed extreme-event cells. "
             "Bars show pointwise 95% intervals at days 1, 5 and 10.",
             ha="center", va="center", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"figE1_m1_raw2k_vs_gc_per_lead.{ext}", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-cache", type=Path, default=CAMPAIGN / "forecasts")
    parser.add_argument("--climatology", type=Path, default=CAMPAIGN / "climatology")
    parser.add_argument("--land-mask", type=Path)
    parser.add_argument("--initializations", type=Path, default=WORKSPACE / "artifacts/checkpoints/v24_Ilya/res1_m1_seed_repeats_local/initializations_2023_stride3.json")
    parser.add_argument("--output-dir", type=Path, default=CAMPAIGN)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    initialization_bytes = args.initializations.read_bytes()
    dates = [row["initialization_time"] if isinstance(row, dict) else row
             for row in json.loads(initialization_bytes)["initializations"]]
    if dates != make_initializations(2023, stride_days=3):
        raise ValueError("Expected the 122 fixed 2023 initializations")
    manifest, receipts = read_receipts(args.forecast_cache.resolve(), dates)
    climate_bytes = (args.climatology / "metadata.json").read_bytes()
    latitude, longitude = (np.load(args.climatology / name, allow_pickle=False) for name in ("lat.npy", "lon.npy"))
    land_mask = np.load(args.land_mask or args.climatology / "land_mask.npy", allow_pickle=False)
    if ((len(latitude), len(longitude)) != (181, 360) or land_mask.shape != (181, 360)
            or not np.all(np.isfinite(land_mask) & (land_mask >= 0) & (land_mask <= 1))):
        raise ValueError("Unexpected native 1° climatology grid or land mask")
    land = land_mask >= .5
    area = area_weights(latitude, longitude)
    weights = np.stack((area, np.where(land, area, 0)))
    names = {f"{variable}_q{quantile:02d}" for _, _, variable, quantiles, _ in EVENTS for quantile in quantiles}
    climatology = {name: np.load(args.climatology / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in names}
    if any(value.shape != (365, 4, 181, 360) or value.dtype != np.float32 for value in climatology.values()):
        raise ValueError("Expected float32 climatology on the 365-day, four-UTC-hour native grid")

    samples = []
    for number, date in enumerate(dates, 1):
        fields, valid_times = read_forecast(receipts[date], manifest, latitude, longitude)
        samples.append(score_start(fields, valid_times, climatology, weights, land))
        print(f"[{number}/{len(dates)}] scored {date}", flush=True)
    samples = np.stack(samples)
    counts = circular_counts(len(dates), **BOOTSTRAP)
    rows = list(table_rows(samples, counts))
    endpoints = [row for row in rows if row["lead_hours"] in ENDPOINTS]
    summary = {
        "status": "complete", "model": "M1 carry seed 22 raw main step 2,000", "initializations": len(dates),
        "initialization_manifest_sha256": hashlib.sha256(initialization_bytes).hexdigest(),
        "forecast_provenance_sha256": manifest["provenance_sha256"],
        "climatology_metadata_sha256": hashlib.sha256(climate_bytes).hexdigest(),
        "climatology_metadata": json.loads(climate_bytes),
        "bootstrap": {**BOOTSTRAP, "method": "paired circular moving blocks of 5 starts at a 3-day cadence",
                      "intervals": "pointwise percentile 95%", "scope": "forecast-date uncertainty for fixed checkpoints"},
        "reference": "independent public GraphCast rollout", "contrast": "M1 raw minus public GraphCast",
        "scorer_provenance": "Local implementation of the A2 handoff. Original frozen observer and scorer unavailable.",
        "weighting": "exact spherical grid-cell area including polar caps",
        "thresholds": {"primary": "q98/q2", "sensitivity": "q95/q5", "rain_eligibility": "local TP24 q95 > 1 mm"},
        "tail": "forecast minus truth on truth-event cells, in K, m/s or mm",
        "headline_endpoints": [r for r in endpoints if r["threshold_set"] == "primary" and
                               r["metric"] in {"csi", "frequency_bias", "tail_bias", "tail_mae"}],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir) as temporary:
        output = Path(temporary)
        write_table(output / "per_lead.csv", rows)
        write_table(output / "endpoints.csv", endpoints)
        np.savez_compressed(output / "sufficient_statistics.npz", stats=samples, initializations=dates, lead_hours=LEADS,
                            events=[event[0] for event in EVENTS], threshold_sets=FAMILIES, regions=REGIONS,
                            branches=BRANCHES[1:], statistic_names=STATS)
        if not args.no_figures:
            plot(output, rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        for path in output.iterdir():
            path.replace(args.output_dir / path.name)


if __name__ == "__main__":
    main()
