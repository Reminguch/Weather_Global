#!/usr/bin/env python3
"""Pool paired extreme-event sufficient statistics from a frozen V24 campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVENTS = ("hot", "cold", "wind", "rain")
THRESHOLDS = ("primary", "sensitivity")
DOMAINS = ("global", "land")
BRANCHES = ("baseline", "full")
STAT_NAMES = (
    "hit_area", "miss_area", "false_alarm_area", "correct_negative_area", "eligible_area",
    "hit_count", "miss_count", "false_alarm_count", "correct_negative_count", "eligible_count",
    "tail_absolute_error_sum", "tail_signed_error_sum", "tail_squared_error_sum",
    "tail_truth_sum", "tail_forecast_sum",
)
METRICS = (
    "csi", "recall", "precision", "false_alarm_ratio", "frequency_bias",
    "observed_event_frequency", "forecast_event_frequency", "tail_mae", "tail_bias",
    "tail_rmse", "tail_truth_mean", "tail_forecast_mean",
)
LIMITATIONS = [
    "Scores describe extremes represented on the sampled 1-degree ERA5 grid, not station or subgrid extremes.",
    "2023 was already explored; this expanded evaluation is not a pristine independent confirmation set.",
    "Paired date-block intervals describe forecast-sample uncertainty, not variability across training seeds.",
    "Intervals are pointwise descriptive 95% intervals, without a simultaneous preservation claim or numerical-noise gate.",
    "TP24 scores begin at lead 24 hours and sum four forecast TP6 intervals; earlier rain leads are undefined.",
    "Undefined metric denominators remain undefined; conditional intensity scores accompany full occurrence scores.",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def metrics_from_sums(values):
    """Only call after pooling cases/leads; zero exposure never means zero error."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != len(STAT_NAMES):
        raise ValueError("Unexpected extreme-statistic axis")
    h, m, f, c, area = np.moveaxis(values[..., :5], -1, 0)
    tail = h + m
    return np.stack((
        divide(h, h + m + f), divide(h, tail), divide(h, h + f), divide(f, h + f),
        divide(h + f, tail), divide(tail, area), divide(h + f, area),
        divide(values[..., 10], tail), divide(values[..., 11], tail),
        np.sqrt(divide(values[..., 12], tail)), divide(values[..., 13], tail),
        divide(values[..., 14], tail),
    ), axis=-1)


def circular_block_counts(n, *, block_size=5, replicates=2000, seed=20261106):
    if n < 1 or not 1 <= block_size <= n or replicates < 1:
        raise ValueError("Invalid date-block bootstrap shape")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(replicates, (n + block_size - 1) // block_size))
    indices = ((starts[..., None] + np.arange(block_size)) % n).reshape(replicates, -1)[:, :n]
    counts = np.zeros((replicates, n), dtype=np.int64)
    np.add.at(counts, (np.arange(replicates)[:, None], indices), 1)
    return counts


def interval(values):
    """Keep support explicit when a sparse bootstrap draw has no denominator."""
    values = np.asarray(values, dtype=float)
    flat = values.reshape(values.shape[0], -1)
    bounds = np.full((2, flat.shape[1]), np.nan)
    support = np.isfinite(flat).sum(axis=0)
    for i in np.flatnonzero(support):
        bounds[:, i] = np.quantile(flat[np.isfinite(flat[:, i]), i], [.025, .975])
    return bounds.reshape((2,) + values.shape[1:]), support.reshape(values.shape[1:])


def endpoint_statistics(per_case, counts):
    """Input case x event x threshold x domain x branch x sufficient statistic."""
    values = np.asarray(per_case, dtype=np.float64)
    point = metrics_from_sums(values.sum(axis=0))
    sampled = np.tensordot(counts, values, axes=(1, 0))
    draws = metrics_from_sums(sampled)
    bounds, support = interval(draws)
    delta = point[..., 1, :] - point[..., 0, :]
    delta_draws = draws[..., 1, :] - draws[..., 0, :]
    delta_bounds, delta_support = interval(delta_draws)
    return point, bounds, support, delta, delta_bounds, delta_support


def _csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing to publish an empty CSV")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def _check_record(path, spec, dates, expected_climatology_hash):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        for key in ("protocol_hash", "model_id", "checkpoint_sha256"):
            if metadata.get(key) != spec[key]:
                raise ValueError(f"Extreme record {key} mismatch: {path}")
        if metadata.get("baseline_checkpoint_sha256") != spec["baseline"]["checkpoint_sha256"]:
            raise ValueError(f"Extreme baseline identity mismatch: {path}")
        if metadata.get("status") != "complete" or metadata.get("initialization_time") not in dates:
            raise ValueError(f"Incomplete or unexpected extreme record: {path}")
        climate_record = metadata.get("provenance", {}).get("extremes_protocol", {}).get("climatology_metadata", {})
        if climate_record.get("sha256") != expected_climatology_hash:
            raise ValueError(f"Extreme climatology identity mismatch: {path}")
        stats = np.asarray(data["stats"], dtype=np.float64)
        valid = np.asarray(data["valid_step_event"], dtype=bool)
        if stats.shape != (40, 4, 2, 2, 2, 15) or valid.shape != (40, 4):
            raise ValueError(f"Extreme statistic shape mismatch: {path}")
        expected_valid = np.ones((40, 4), bool)
        expected_valid[:3, 3] = False
        if not np.array_equal(valid, expected_valid):
            raise ValueError(f"Unexpected extreme lead availability: {path}")
        if not np.isfinite(stats[valid]).all():
            raise ValueError(f"Nonfinite valid extreme sufficient statistics: {path}")
        valid_stats = stats[valid]
        if np.any(valid_stats[..., :11] < 0) or np.any(valid_stats[..., 12] < 0):
            raise ValueError(f"Negative contingency or absolute/squared error sum: {path}")
        if not np.equal(valid_stats[..., 5:10], np.floor(valid_stats[..., 5:10])).all():
            raise ValueError(f"Noninteger raw event counts: {path}")
        for key, expected in (("events", EVENTS), ("thresholds", THRESHOLDS), ("domains", DOMAINS),
                              ("branches", BRANCHES), ("statistics", STAT_NAMES)):
            if tuple(data[key].tolist()) != expected:
                raise ValueError(f"Extreme {key} differs: {path}")
        if not np.array_equal(data["lead_hours"], np.arange(1, 41) * 6):
            raise ValueError(f"Extreme lead coordinates differ: {path}")
        negative_counts = np.asarray(data["negative_tp6_counts"], dtype=np.int64)
        if negative_counts.shape != (40, 3) or np.any(negative_counts < 0):
            raise ValueError(f"Invalid negative-rain diagnostic: {path}")
        if tuple(data["negative_tp6_branches"].tolist()) != ("truth", "baseline", "full"):
            raise ValueError(f"Negative-rain branch labels differ: {path}")
        stats[~valid] = 0.
    return metadata["initialization_time"], stats, negative_counts


def load_campaign(root):
    from scripts.analyze_models.evaluate_v24_amse_regime import resolve_result
    root = Path(root)
    campaign = read(root / "campaign.json")
    dates = campaign["dates"]
    if len(dates) != 122 or len(set(dates)) != 122 or dates != sorted(dates):
        raise ValueError("Expected the complete ordered 122-start campaign")
    source_records = [dict(path=str(root / "campaign.json"), sha256=sha(root / "campaign.json"))]
    models = {}
    for mi, model in enumerate(campaign["models"]):
        records, losses, negative_counts = {}, {}, {}
        for shard in range(campaign["shards"]):
            manifest_path = root / f"shards/{shard:02d}/manifest.json"
            manifest = read(manifest_path)
            clim = manifest["extremes"]["climatology_metadata"]
            if sha(clim["path"]) != clim["sha256"]:
                raise ValueError("Climatology metadata changed after binding")
            directory, spec, result, _ = resolve_result(manifest_path, mi, "spectral")
            for filename in ("evaluation.json", "specification.json"):
                path = directory / filename
                source_records.append(dict(path=str(path), sha256=sha(path)))
            shard_dates = spec["initialization_times"]
            for row in result["original_graphcast_loss_per_initialization"]:
                stamp = row["initialization_time"]
                if stamp in losses:
                    raise ValueError("Duplicate loss initialization")
                losses[stamp] = np.stack((row["baseline_per_step"], row["full_per_step"]), axis=-1)
            for path in sorted((directory / "extremes").glob("*.npz")):
                stamp, stats, negative = _check_record(path, spec, shard_dates, clim["sha256"])
                if stamp in records:
                    raise ValueError("Duplicate extreme initialization")
                records[stamp] = stats
                negative_counts[stamp] = negative
                source_records.append(dict(path=str(path), sha256=sha(path)))
        if set(records) != set(dates) or set(losses) != set(dates):
            raise ValueError(f"Incomplete full-year extreme evaluation for {model['id']}")
        loss = np.stack([losses[t] for t in dates])
        if loss.shape != (122, 40, 2) or not np.isfinite(loss).all() or np.any(loss <= 0):
            raise ValueError("Invalid complete exact GC loss array")
        models[model["id"]] = dict(stats=np.stack([records[t] for t in dates]), loss=loss,
                                  negative_tp6_counts=np.stack([negative_counts[t] for t in dates]))
    return campaign, models, source_records


def plot_report(image_dir, endpoints, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    for ei, event in enumerate(EVENTS):
        for ri, metric in enumerate(("csi", "tail_mae")):
            ax = axes[ri, ei]
            for mi, model in enumerate(models):
                rows = [r for r in endpoints if r["model"] == model["id"] and r["event"] == event
                        and r["threshold"] == "primary" and r["domain"] == "global"
                        and r["metric"] == metric and r["endpoint"] != "rollout"]
                x = np.asarray([float(r["endpoint"][3:]) for r in rows]) + .07 * mi
                y = np.asarray([r["full_minus_baseline"] for r in rows])
                lo = np.asarray([r["difference_ci95_lower"] for r in rows])
                hi = np.asarray([r["difference_ci95_upper"] for r in rows])
                ax.plot(x, y, "o-", color=colors[mi % len(colors)], label=model["label"])
                ax.vlines(x, lo, hi, color=colors[mi % len(colors)])
            ax.axhline(0, color="gray", linewidth=.8)
            ax.set_title(f"{event.capitalize()}: {'CSI' if ri == 0 else 'tail MAE'}")
            ax.set_xticks([1, 5, 10]); ax.grid(alpha=.2)
            if ri == 1:
                ax.set_xlabel("Lead days")
            ax.set_ylabel("Model − GC" + (" (higher better)" if ri == 0 else " (lower better)"))
    fig.suptitle("2023 extremes: 122 paired starts; pointwise date-block 95% intervals")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=len(models))
    fig.tight_layout(rect=(0, .07, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(image_dir / f"extreme_skill.{suffix}", dpi=180)
    plt.close(fig)


def report(root):
    root = Path(root).resolve()
    campaign, arrays, sources = load_campaign(root)
    workspace = Path(campaign.get("workspace", __import__("os").environ.get("WEATHER_WORKSPACE", ROOT)))
    out = Path(campaign.get("output_data_dir", workspace / "plots/analyze_models/data/resolution_eval" / root.name))
    images = Path(campaign.get("output_image_dir", workspace / "plots/analyze_models/images/resolution_eval" / root.name))
    out.mkdir(parents=True, exist_ok=True)
    counts = circular_block_counts(122)
    lead_rows, endpoint_rows, summary, warnings = [], [], [], []
    reference = arrays["M1"]["loss"][..., 0]
    for model in campaign["models"]:
        model_id = model["id"]
        data = arrays[model_id]
        pooled = data["stats"].sum(axis=0)
        lead_metrics = metrics_from_sums(pooled)
        for lead, event, threshold, domain, branch in np.ndindex(40, 4, 2, 2, 2):
            available = not (event == 3 and lead < 3)
            row = dict(model=model_id, lead_hours=(lead + 1) * 6, event=EVENTS[event],
                       threshold=THRESHOLDS[threshold], domain=DOMAINS[domain], branch=BRANCHES[branch],
                       available=available)
            row.update(zip(STAT_NAMES, pooled[lead, event, threshold, domain, branch] if available else [None] * 15))
            row.update(zip(METRICS, lead_metrics[lead, event, threshold, domain, branch] if available else [None] * 12))
            lead_rows.append(row)
        for name, selection in (("day1", [3]), ("day5", [19]), ("day10", [39]), ("rollout", list(range(40)))):
            per_case = data["stats"][:, selection].sum(axis=1)
            point, bounds, support, delta, delta_bounds, delta_support = endpoint_statistics(per_case, counts)
            for event, threshold, domain, metric in np.ndindex(4, 2, 2, len(METRICS)):
                ix = (event, threshold, domain)
                endpoint_rows.append(dict(model=model_id, endpoint=name, event=EVENTS[event],
                    threshold=THRESHOLDS[threshold], domain=DOMAINS[domain], metric=METRICS[metric],
                    baseline=point[ix + (0, metric)], full=point[ix + (1, metric)],
                    full_minus_baseline=delta[ix + (metric,)],
                    baseline_ci95_lower=bounds[(0,) + ix + (0, metric)],
                    baseline_ci95_upper=bounds[(1,) + ix + (0, metric)],
                    full_ci95_lower=bounds[(0,) + ix + (1, metric)],
                    full_ci95_upper=bounds[(1,) + ix + (1, metric)],
                    difference_ci95_lower=delta_bounds[(0,) + ix + (metric,)],
                    difference_ci95_upper=delta_bounds[(1,) + ix + (metric,)],
                    baseline_finite_bootstrap_draws=int(support[ix + (0, metric)]),
                    full_finite_bootstrap_draws=int(support[ix + (1, metric)]),
                    difference_finite_bootstrap_draws=int(delta_support[ix + (metric,)])))
        baseline, full = data["loss"][..., 0], data["loss"][..., 1]
        summary.append(dict(model=model_id, exact_gc_loss=float(full.mean()),
            common_gc_loss=float(reference.mean()), paired_gc_loss=float(baseline.mean()),
            mse_reduction_rollout_pct=float(100 * (1 - full.mean() / reference.mean())),
            mse_reduction_day10_pct=float(100 * (1 - full[:, -1].mean() / reference[:, -1].mean())),
            paired_mse_reduction_rollout_pct=float(100 * (1 - full.mean() / baseline.mean()))))
        warnings.append(dict(model=model_id, kind="finite_baseline_repeatability", policy="advisory",
            loss_max_abs_difference=float(np.max(np.abs(baseline - reference))),
            loss_rollout_relative_difference=float(baseline.mean() / reference.mean() - 1),
            occurrence_statistics_max_abs_difference=float(np.max(np.abs(
                data["stats"][..., 0, :10] - arrays["M1"]["stats"][..., 0, :10])))))
        warnings.append(dict(model=model_id, kind="negative_precipitation_values", policy="retained without clipping",
            branches=["truth", "baseline", "full"],
            total_count=data["negative_tp6_counts"].sum(axis=(0, 1)).tolist(),
            count_per_lead=data["negative_tp6_counts"].sum(axis=0).tolist()))
    _csv(out / "per_lead.csv", lead_rows)
    _csv(out / "endpoints.csv", endpoint_rows)
    _csv(out / "accuracy.csv", summary)
    plot_report(images, endpoint_rows, campaign["models"])
    payload = dict(status="complete", starts=122, leads=40, models=campaign["models"], summary=summary,
        metric_definitions=dict(csi="H/(H+M+F)", recall="H/(H+M)", precision="H/(H+F)",
            false_alarm_ratio="F/(H+F)", frequency_bias="(H+F)/(H+M)",
            tail_mae="area-weighted absolute error on verifying threshold exceedances",
            pooling="Sum sufficient statistics over cases and requested valid leads before forming ratios",
            thresholds="Primary q98 hot/wind/rain and q2 cold; sensitivity q95 and q5, respectively",
            climatology="1991-2020, fixed local calendar-day +/-30-day windows, matching valid UTC hour; leap days excluded",
            rain_eligibility="Historical local TP24 q95 > 1 mm; the same mask applies to primary and sensitivity",
            weights="Exact spherical cell areas normalized globally; land mask restricts exposure",
            units={"hot": "K", "cold": "K", "wind": "m s-1", "rain": "mm per 24h"}),
        bootstrap=dict(method="paired circular moving blocks of initialization dates", block_starts=5,
            cadence_days=3, block_nominal_days=15, replicates=2000, seed=20261106,
            intervals="Pointwise percentile95; finite-draw counts reported for undefined sparse draws"),
        warnings=warnings, limitations=LIMITATIONS, sources=sources,
        output_data_dir=str(out), output_image_dir=str(images))
    write_json(out / "report.json", payload)
    lines = ["# Full 2023 extreme-event evaluation", "", "122 paired initialization dates; 40 six-hour leads through day 10.", "",
             "| Model | Exact GC loss reduction, rollout (%) | Day 10 (%) |", "| --- | ---: | ---: |"]
    lines.extend(f"| {r['model']} | {r['mse_reduction_rollout_pct']:.3f} | {r['mse_reduction_day10_pct']:.3f} |" for r in summary)
    lines += ["", "Extreme-event occurrence and conditional intensity scores, raw contingency counts, and paired uncertainty intervals are in `per_lead.csv` and `endpoints.csv`.",
              "Positive CSI differences favor the corrected model; negative tail-MAE differences favor it. Better tail errors alone do not establish better event detection.", ""]
    lines.extend("- " + item for item in LIMITATIONS)
    lines += ["", "Finite baseline repeatability differences are retained in `report.json` as advisory measurements.", ""]
    (out / "report.md").write_text("\n".join(lines))
    write_json(root / "complete.json", dict(status="complete", report=str(out / "report.json"),
                                           report_sha256=sha(out / "report.json"), starts=122, leads=40))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
