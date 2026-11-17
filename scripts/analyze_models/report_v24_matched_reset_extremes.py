#!/usr/bin/env python3
"""Compare frozen GraphCast, carry/carry and matched reset/reset extremes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_models.report_v24_extremes_2023 import (
    DOMAINS, EVENTS, METRICS, STAT_NAMES, THRESHOLDS, _check_record, _csv,
    circular_block_counts, endpoint_statistics, interval, metrics_from_sums,
    read, sha, write_json,
)

MODELS = (
    {"id": "GC", "label": "GraphCast"},
    {"id": "M1", "label": "Carry/carry (M1)"},
    {"id": "R_matched", "label": "Reset/reset"},
)
ENDPOINTS = (("day1", [3]), ("day5", [19]), ("day10", [39]),
             ("rollout", list(range(40))))
COMPARISONS = (("GC", "GC"), ("M1", "GC"), ("R_matched", "GC"),
               ("M1", "R_matched"))
LIMITATIONS = [
    "Scores describe extremes represented on the sampled 1-degree ERA5 grid, not station or subgrid extremes.",
    "These are fixed-climatological-threshold event scores, not gain-swept precision-recall curves.",
    "Paired date-block intervals describe forecast-sample uncertainty for fixed checkpoints, not variability across training seeds.",
    "Intervals are pointwise descriptive 95% intervals, without a simultaneous preservation claim or numerical-noise gate.",
    "TP24 scores begin at lead 24 hours and sum four forecast TP6 intervals; earlier rain leads are undefined.",
    "Undefined metric denominators remain undefined; conditional intensity scores accompany full occurrence scores.",
]


def checked_pin(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError(f"Pinned file changed: {path}")
    return path


def initialization_id(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _insert_losses(destination, result, expected_dates):
    rows = result["original_graphcast_loss_per_initialization"]
    if [row["initialization_time"] for row in rows] != list(expected_dates):
        raise ValueError("Loss initialization order differs from its pinned shard")
    for row in rows:
        stamp = row["initialization_time"]
        if stamp in destination:
            raise ValueError(f"Duplicate loss initialization: {stamp}")
        values = np.stack((row["baseline_per_step"], row["full_per_step"]), axis=-1)
        if values.shape != (40, 2) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Invalid original GraphCast loss: {stamp}")
        destination[stamp] = values


def _ordered_arrays(records, losses, negatives, dates):
    if any(set(values) != set(dates) for values in (records, losses, negatives)):
        raise ValueError("Incomplete or unexpected initialization coverage")
    return dict(stats=np.stack([records[t] for t in dates]),
                loss=np.stack([losses[t] for t in dates]),
                negative_tp6_counts=np.stack([negatives[t] for t in dates]))


def _load_reference(manifest, sources):
    """Read only the canonical M1 evaluation; its baseline supplies raw GC."""
    from scripts.analyze_models.evaluate_v24_amse_regime import resolve_result

    root = Path(manifest["reference_root"])
    campaign_path = checked_pin(manifest["reference_campaign"])
    complete_path = checked_pin(manifest["reference_complete"])
    campaign, complete = read(campaign_path), read(complete_path)
    if complete.get("status") != "complete" or campaign["dates"] != manifest["dates"]:
        raise ValueError("Canonical campaign completion or date coverage differs")
    report_pin = dict(path=complete["report"], sha256=complete["report_sha256"])
    old_report = read(checked_pin(report_pin))
    pinned_sources = {str(Path(r["path"]).resolve()): r for r in old_report["sources"]}
    sources.extend([manifest["reference_campaign"], manifest["reference_complete"], report_pin])
    selected = [i for i, model in enumerate(campaign["models"]) if model["id"] == "M1"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one canonical carry/carry model")
    index = selected[0]
    checkpoint = campaign["models"][index]["checkpoint_record"]
    checked_pin(checkpoint)
    sources.append(checkpoint)
    records, losses, negatives = {}, {}, {}
    for shard in range(campaign["shards"]):
        shard_path = root / f"shards/{shard:02d}/manifest.json"
        shard_manifest = read(shard_path)
        if shard_manifest["extremes"]["climatology_metadata"]["sha256"] != manifest["climatology_metadata"]["sha256"]:
            raise ValueError("Canonical and reset climatology metadata differ")
        directory, spec, result, _ = resolve_result(shard_path, index, "spectral")
        if spec["checkpoint_sha256"] != checkpoint["sha256"]:
            raise ValueError("Canonical carry checkpoint differs")
        if spec["baseline"]["checkpoint_sha256"] != manifest["baseline"]["checkpoint_sha256"]:
            raise ValueError("Canonical and reset baseline checkpoints differ")
        shard_dates = spec["initialization_times"]
        for path in [directory / "evaluation.json", directory / "specification.json",
                     *sorted((directory / "extremes").glob("*.npz"))]:
            pin = pinned_sources.get(str(path.resolve()))
            if pin is None:
                raise ValueError(f"Canonical output absent from completed report provenance: {path}")
            checked_pin(pin)
            sources.append(pin)
            if path.suffix != ".npz":
                continue
            stamp, stats, negative = _check_record(path, spec, shard_dates,
                                                  manifest["climatology_metadata"]["sha256"])
            if stamp in records:
                raise ValueError(f"Duplicate canonical extreme initialization: {stamp}")
            records[stamp], negatives[stamp] = stats, negative
        _insert_losses(losses, result, shard_dates)
    return _ordered_arrays(records, losses, negatives, manifest["dates"])


def _load_reset(manifest, sources):
    dates = manifest["dates"]
    records, losses, negatives = {}, {}, {}
    spec = dict(protocol_hash=manifest["manifest_hash"], model_id="R_matched",
                checkpoint_sha256=manifest["checkpoint"]["sha256"], baseline=manifest["baseline"])
    for shard in manifest["shards"]:
        checked_pin(shard["initializations"])
        output = Path(shard["output_dir"])
        complete_path = output / "complete.json"
        complete = read(complete_path)
        if complete.get("status") != "complete" or complete.get("reset_state_every_step") is not True:
            raise ValueError(f"Incomplete or non-reset worker: {output}")
        if complete.get("manifest_hash", manifest["manifest_hash"]) != manifest["manifest_hash"]:
            raise ValueError("Worker manifest identity differs")
        init_payload = read(shard["initializations"]["path"])
        shard_dates = init_payload.get("initialization_times")
        if shard_dates is None:
            shard_dates = [row["initialization_time"] if isinstance(row, dict) else row
                           for row in init_payload["initializations"]]
        if complete["initialization_ids"] != [initialization_id(t) for t in shard_dates] or len(shard_dates) != shard["count"]:
            raise ValueError("Worker dates differ from pinned initialization manifest")
        if complete.get("initialization_times", shard_dates) != shard_dates:
            raise ValueError("Worker ISO timestamps differ from pinned initialization manifest")
        if len(set(shard_dates)) != len(shard_dates) or not set(shard_dates).issubset(dates):
            raise ValueError("Unexpected or duplicate reset worker dates")
        metrics_path = checked_pin(complete["metrics"])
        if metrics_path.resolve() != (output / "metrics.json").resolve():
            raise ValueError("Worker metrics point outside expected output")
        metrics = read(metrics_path)
        if metrics.get("rs_reset_every_step") is not True or metrics.get("target_steps") != 40:
            raise ValueError("Worker did not evaluate the requested reset policy and horizon")
        if metrics.get("evaluation_status") != "complete" or metrics.get("evaluated_samples") != len(shard_dates):
            raise ValueError("Reset metrics are incomplete")
        if metrics.get("checkpoint_sha256") != spec["checkpoint_sha256"] or metrics.get("baseline_checkpoint_sha256") != spec["baseline"]["checkpoint_sha256"]:
            raise ValueError("Reset metric checkpoint identity differs")
        _insert_losses(losses, metrics, shard_dates)
        sources.extend([shard["initializations"], complete["metrics"],
                        dict(path=str(complete_path), sha256=sha(complete_path))])
        pins = complete["records"]
        if len(pins) != len(shard_dates):
            raise ValueError("Worker extreme-record count differs")
        shard_records = set()
        for pin in pins:
            path = checked_pin(pin)
            if path.parent.resolve() != (output / "extremes").resolve():
                raise ValueError("Worker extreme record outside expected output")
            stamp, stats, negative = _check_record(path, spec, shard_dates,
                                                  manifest["climatology_metadata"]["sha256"])
            if path.stem != initialization_id(stamp):
                raise ValueError("Extreme record filename differs from its initialization time")
            if stamp in records:
                raise ValueError(f"Duplicate reset extreme initialization: {stamp}")
            records[stamp], negatives[stamp] = stats, negative
            shard_records.add(stamp)
            sources.append(pin)
        if shard_records != set(shard_dates):
            raise ValueError("Worker extremes do not cover its loss initializations")
    return _ordered_arrays(records, losses, negatives, dates)


def validate_truth_exposure(reference, candidate):
    """Thresholds/truth masks must match; forecast differences are not gates."""
    a, b = np.asarray(reference), np.asarray(candidate)
    # Each statistic is independent of the forecast: eligible area/count,
    # verifying-event area/count, and area-weighted verifying tail intensity.
    def truth(values):
        return np.stack((values[..., 4], values[..., 9],
                         values[..., 0] + values[..., 1],
                         values[..., 5] + values[..., 6], values[..., 13]), axis=-1)
    for values in (a, b):
        if not np.allclose(truth(values[..., 0, :]), truth(values[..., 1, :]),
                           rtol=1e-12, atol=1e-14):
            raise ValueError("Truth-selected events or exposure differ between branches")
    if not np.allclose(truth(a[..., 0, :]), truth(b[..., 0, :]), rtol=1e-12, atol=1e-14):
        raise ValueError("Canonical and reset truth-selected events or exposure differ")


def load_campaign(root):
    from scripts.experiments.run_v24_matched_reset_extremes import verify

    manifest = verify(Path(root))
    dates = manifest["dates"]
    if len(dates) != 122 or dates != sorted(set(dates)):
        raise ValueError("Expected 122 unique chronological initialization dates")
    if manifest["initialization_ids"] != [initialization_id(stamp) for stamp in dates]:
        raise ValueError("Compact initialization IDs differ from canonical ISO timestamps")
    sources = [manifest[key] for key in
               ("checkpoint", "climatology_metadata", "land_mask", "initializations")]
    reference = _load_reference(manifest, sources)
    reset = _load_reset(manifest, sources)
    validate_truth_exposure(reference["stats"], reset["stats"])
    if not np.array_equal(reference["negative_tp6_counts"][..., 0], reset["negative_tp6_counts"][..., 0]):
        raise ValueError("Verifying precipitation diagnostics differ")
    warnings = [dict(kind="finite_baseline_repeatability", policy="advisory",
        loss_max_abs_difference=float(np.max(np.abs(reset["loss"][..., 0] - reference["loss"][..., 0]))),
        loss_rollout_relative_difference=float(reset["loss"][..., 0].mean() / reference["loss"][..., 0].mean() - 1),
        occurrence_statistics_max_abs_difference=float(np.max(np.abs(
            reset["stats"][..., 0, :10] - reference["stats"][..., 0, :10]))))]
    arrays = {
        "GC": dict(stats=reference["stats"][..., 0, :], loss=reference["loss"][..., 0]),
        "M1": dict(stats=reference["stats"][..., 1, :], loss=reference["loss"][..., 1]),
        "R_matched": dict(stats=reset["stats"][..., 1, :], loss=reset["loss"][..., 1]),
    }
    for model, data, branch in (("GC", reference, 1), ("M1", reference, 2), ("R_matched", reset, 2)):
        counts = data["negative_tp6_counts"][..., branch]
        warnings.append(dict(kind="negative_precipitation_values", model=model,
                             policy="retained without clipping", total_count=int(counts.sum()),
                             count_per_lead=counts.sum(axis=0).tolist()))
    return manifest, arrays, sources, warnings


def build_tables(arrays, counts):
    lead_rows, endpoint_rows, accuracy_rows = [], [], []
    for model in MODELS:
        pooled = arrays[model["id"]]["stats"].sum(axis=0)
        metrics = metrics_from_sums(pooled)
        for lead, event, threshold, domain in np.ndindex(40, 4, 2, 2):
            available = not (event == 3 and lead < 3)
            row = dict(model=model["id"], lead_hours=(lead + 1) * 6, event=EVENTS[event],
                       threshold=THRESHOLDS[threshold], domain=DOMAINS[domain], available=available)
            ix = (lead, event, threshold, domain)
            row.update(zip(STAT_NAMES, pooled[ix] if available else [None] * len(STAT_NAMES)))
            row.update(zip(METRICS, metrics[ix] if available else [None] * len(METRICS)))
            lead_rows.append(row)
    for model, reference in COMPARISONS:
        for endpoint, selection in ENDPOINTS:
            per_case = np.stack([arrays[m]["stats"][:, selection].sum(axis=1)
                                 for m in (reference, model)], axis=-2)
            point, bounds, support, delta, delta_bounds, delta_support = endpoint_statistics(per_case, counts)
            for event, threshold, domain, metric in np.ndindex(4, 2, 2, len(METRICS)):
                ix = (event, threshold, domain)
                endpoint_rows.append(dict(model=model, reference=reference, endpoint=endpoint,
                    event=EVENTS[event], threshold=THRESHOLDS[threshold], domain=DOMAINS[domain],
                    metric=METRICS[metric], value=point[ix + (1, metric)],
                    reference_value=point[ix + (0, metric)], difference=delta[ix + (metric,)],
                    ci95_lower=bounds[(0,) + ix + (1, metric)], ci95_upper=bounds[(1,) + ix + (1, metric)],
                    reference_ci95_lower=bounds[(0,) + ix + (0, metric)],
                    reference_ci95_upper=bounds[(1,) + ix + (0, metric)],
                    difference_ci95_lower=delta_bounds[(0,) + ix + (metric,)],
                    difference_ci95_upper=delta_bounds[(1,) + ix + (metric,)],
                    finite_bootstrap_draws=int(support[ix + (1, metric)]),
                    reference_finite_bootstrap_draws=int(support[ix + (0, metric)]),
                    difference_finite_bootstrap_draws=int(delta_support[ix + (metric,)])))
            loss = arrays[model]["loss"][:, selection].mean(axis=1)
            baseline = arrays[reference]["loss"][:, selection].mean(axis=1)
            draws = 100 * (1 - (counts @ loss) / (counts @ baseline))
            bounds, support = interval(draws)
            accuracy_rows.append(dict(model=model, reference=reference, endpoint=endpoint,
                exact_graphcast_loss=float(loss.mean()), reference_loss=float(baseline.mean()),
                loss_reduction_pct=float(100 * (1 - loss.mean() / baseline.mean())),
                loss_reduction_ci95_lower=float(bounds[0]), loss_reduction_ci95_upper=float(bounds[1]),
                finite_bootstrap_draws=int(support)))
    return lead_rows, endpoint_rows, accuracy_rows


def plot_report(images, endpoints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images.mkdir(parents=True, exist_ok=True)
    colors = {"GC": "#555555", "M1": "#0072B2", "R_matched": "#D55E00"}
    for memory in (False, True):
        fig, axes = plt.subplots(2, 4, figsize=(14.5, 7), sharex=True)
        for ei, event in enumerate(EVENTS):
            for ri, metric in enumerate(("csi", "tail_mae")):
                ax = axes[ri, ei]
                for mi, model in enumerate(MODELS[1:2] if memory else MODELS):
                    rows = [r for r in endpoints if r["model"] == model["id"]
                            and r["reference"] == ("R_matched" if memory else "GC")
                            and r["event"] == event and r["threshold"] == "primary"
                            and r["domain"] == "global" and r["metric"] == metric
                            and r["endpoint"] != "rollout"]
                    x = np.array([float(r["endpoint"][3:]) for r in rows]) + (0 if memory else .08 * (mi - 1))
                    key, ci = ("difference", "difference_") if memory else ("value", "")
                    y = np.array([r[key] for r in rows])
                    ax.plot(x, y, "o-", color=colors[model["id"]], label=model["label"], markersize=4)
                    ax.vlines(x, [r[ci + "ci95_lower"] for r in rows],
                              [r[ci + "ci95_upper"] for r in rows], color=colors[model["id"]])
                if memory:
                    ax.axhline(0, color="gray", linewidth=.8)
                ax.set_title(event.capitalize())
                ax.set_xticks([1, 5, 10]); ax.grid(alpha=.2)
                units = {"hot": "K", "cold": "K", "wind": "m s$^{-1}$", "rain": "mm / 24 h"}[event]
                label = "CSI" if ri == 0 else f"Tail MAE ({units})"
                ax.set_ylabel(("Carry − reset: " if memory else "") + label)
                if ri == 1:
                    ax.set_xlabel("Lead days")
        title = "Carry/carry − reset/reset" if memory else "GraphCast, carry/carry and reset/reset"
        fig.suptitle(title + ": 122 paired starts; pointwise date-block 95% intervals")
        if not memory:
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, .06 if not memory else 0, 1, .95))
        stem = "memory_effect" if memory else "extreme_skill"
        for suffix in ("pdf", "png"):
            fig.savefig(images / f"{stem}.{suffix}", dpi=180)
        plt.close(fig)


def report(root: Path):
    root = Path(root).resolve()
    manifest, arrays, sources, warnings = load_campaign(root)
    workspace = Path(manifest.get("workspace", ROOT))
    out = Path(manifest.get("output_data_dir", workspace / "plots/analyze_models/data/resolution_eval" / root.name))
    images = Path(manifest.get("output_image_dir", workspace / "plots/analyze_models/images/resolution_eval" / root.name))
    out.mkdir(parents=True, exist_ok=True)
    counts = circular_block_counts(122)
    leads, endpoints, accuracy = build_tables(arrays, counts)
    for name, rows in (("per_lead", leads), ("endpoints", endpoints), ("accuracy", accuracy)):
        _csv(out / f"{name}.csv", rows)
    plot_report(images, endpoints)
    payload = dict(status="complete", starts=122, leads=40, manifest_hash=manifest["manifest_hash"],
        models=MODELS, comparisons=[dict(model=m, reference=r) for m, r in COMPARISONS], summary=accuracy,
        metric_definitions=dict(csi="H/(H+M+F)", recall="H/(H+M)", precision="H/(H+F)",
            tail_mae="Area-weighted absolute error on verifying threshold exceedances",
            pooling="Sum sufficient statistics over cases and requested valid leads before forming ratios",
            loss_reduction_pct="100*(1-mean(model original GraphCast loss)/mean(reference original GraphCast loss))",
            differences="Model minus reference; positive CSI and negative tail MAE favor the model",
            thresholds="Primary q98 hot/wind/rain and q2 cold; sensitivity q95 and q5, respectively",
            climatology="1991-2020, fixed local calendar-day +/-30-day windows, matching valid UTC hour; leap days excluded",
            rain_eligibility="Historical local TP24 q95 > 1 mm; identical mask for both threshold sets",
            weights="Exact spherical cell areas normalized globally; land mask restricts exposure"),
        bootstrap=dict(method="Paired circular moving blocks of chronological initialization dates",
            block_starts=5, cadence_days=3, block_nominal_days=15, replicates=2000, seed=20261106,
            intervals="Pointwise percentile 95%; finite-draw support reported for undefined sparse draws"),
        warnings=warnings, limitations=LIMITATIONS, sources=sources,
        output_data_dir=str(out), output_image_dir=str(images))
    write_json(out / "report.json", payload)
    lines = ["# Matched reset/reset extreme-weather evaluation", "",
        "122 paired 2023 initialization dates; 40 six-hour leads through day 10. All models use the same frozen 1991–2020 event thresholds.", "",
        "| Model | Reference | Endpoint | Exact GC loss reduction (%) | 95% interval |",
        "| --- | --- | --- | ---: | ---: |"]
    labels = {model["id"]: model["label"] for model in MODELS}
    lines.extend(f"| {labels[row['model']]} | {labels[row['reference']]} | {row['endpoint']} | "
                 f"{row['loss_reduction_pct']:.3f} | [{row['loss_reduction_ci95_lower']:.3f}, {row['loss_reduction_ci95_upper']:.3f}] |"
                 for row in accuracy)
    lines += ["", "`endpoints.csv` reports absolute event scores and paired differences for carry/carry versus GraphCast, reset/reset versus GraphCast, and carry/carry versus reset/reset. `per_lead.csv` retains all event counts, areas and intensity statistics.", "",
        "Positive CSI differences favor the named model; negative tail-MAE differences favor it. Better conditional intensity errors alone do not establish better event detection.", "",
        "`extreme_skill.pdf` shows all three models; `memory_effect.pdf` shows the paired carry/carry minus reset/reset contrast. Both use primary thresholds and global exposure; tables also contain sensitivity thresholds and land-only results.", ""]
    lines.extend("- " + item for item in LIMITATIONS)
    lines += ["", "Finite repeated-GraphCast differences are retained as advisory measurements in `report.json`.", ""]
    (out / "report.md").write_text("\n".join(lines))
    write_json(root / "complete.json", dict(status="complete", starts=122, leads=40,
        manifest_hash=manifest["manifest_hash"], report=str(out / "report.json"),
        report_sha256=sha(out / "report.json")))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
