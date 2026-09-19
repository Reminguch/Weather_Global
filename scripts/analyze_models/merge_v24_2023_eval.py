#!/usr/bin/env python3
"""Audit fixed-date V24 shards and report paired date-block loss intervals.

The M1 execution supplies the canonical baseline for every comparison. Losses
are exact GraphCast objective contributions saved by the evaluator; this script
never reconstructs them from channel RMSE or averages per-case percentages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc(value: str) -> datetime:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError(f"Timestamp must explicitly use UTC: {value!r}")
    return stamp.astimezone(timezone.utc)


def _iso(stamp: datetime) -> str:
    return stamp.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def validate_protocol(protocol: dict[str, Any]) -> tuple[list[str], list[str], np.ndarray]:
    """Require the full prespecified year; incomplete years cannot produce reports."""
    times = protocol["initialization_times"]
    expected = [
        _iso(datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=day))
        for day in range(365)
    ]
    if times != expected:
        raise ValueError("Protocol must list all 365 daily 00 UTC starts in 2023, in order")
    leads = np.asarray(protocol["lead_hours"])
    if protocol["target_steps"] != 40 or not np.array_equal(leads, np.arange(6, 241, 6)):
        raise ValueError("Protocol must score all 40 six-hour leads through 240 hours")
    model_ids = [model["model_id"] for model in protocol["models"]]
    if model_ids != ["M1", "M2", "M3"]:
        raise ValueError("Protocol must contain the frozen M1, M2, M3 models in order")
    if protocol["baseline"]["canonical_model_id"] != "M1":
        raise ValueError("M1 must supply the canonical baseline")
    for key in ("repeatability_rtol", "repeatability_atol"):
        value = protocol["baseline"][key]
        if not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid baseline {key}")
    for key in ("protocol_hash", "initialization_manifest_sha256"):
        if not isinstance(protocol.get(key), str) or not protocol[key]:
            raise ValueError(f"Missing protocol {key}")
    return times, model_ids, leads.astype(np.int64)


def merge_shards(
    protocol: dict[str, Any], shard_paths: list[Path]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return [model, initialization, lead] losses after strict provenance checks."""
    times, model_ids, lead_hours = validate_protocol(protocol)
    if not shard_paths or len(set(path.resolve() for path in shard_paths)) != len(shard_paths):
        raise ValueError("Shard paths must be nonempty and unique")
    by_time = {value: index for index, value in enumerate(times)}
    identities = {model["model_id"]: model for model in protocol["models"]}
    baseline = np.full((len(model_ids), len(times), len(lead_hours)), np.nan)
    full = np.full_like(baseline, np.nan)
    seen: set[tuple[str, str]] = set()
    shard_ids: dict[str, set[int]] = {model: set() for model in model_ids}
    shard_counts: dict[str, int] = {}
    sources = []
    for path in shard_paths:
        payload = _load(path)
        model_id = payload.get("model_id")
        if model_id not in identities:
            raise ValueError(f"Unknown model_id in {path}: {model_id!r}")
        model_index = model_ids.index(model_id)
        required = {
            "protocol_hash": protocol["protocol_hash"],
            "initialization_manifest_sha256": protocol["initialization_manifest_sha256"],
            "checkpoint_sha256": identities[model_id]["checkpoint_sha256"],
            "baseline_checkpoint_sha256": protocol["baseline"]["checkpoint_sha256"],
            "global_initialization_times": times,
            "target_steps": 40,
            "evaluation_status": "complete",
            "metrics_complete": True,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(f"Shard {path} has unexpected {key}")
        shard_index, shard_count = payload.get("anchor_shard_index"), payload.get("anchor_shard_count")
        if type(shard_count) is not int or not 1 <= shard_count <= len(times):
            raise ValueError(f"Invalid anchor_shard_count in {path}")
        if type(shard_index) is not int or not 0 <= shard_index < shard_count:
            raise ValueError(f"Invalid anchor_shard_index in {path}")
        if model_id in shard_counts and shard_counts[model_id] != shard_count:
            raise ValueError(f"Inconsistent shard count for {model_id}")
        shard_counts[model_id] = shard_count
        if shard_index in shard_ids[model_id]:
            raise ValueError(f"Duplicate shard index for {model_id}: {shard_index}")
        shard_ids[model_id].add(shard_index)
        records = payload.get("original_graphcast_loss_per_initialization")
        if not isinstance(records, list) or not records:
            raise ValueError(f"No per-initialization exact losses in {path}")
        expected_times = times[shard_index::shard_count]
        actual_times = [record["initialization_time"] for record in records]
        if actual_times != expected_times:
            raise ValueError(f"Missing, duplicate, or misplaced initialization records in {path}")
        if payload.get("evaluated_samples") != len(records):
            raise ValueError(f"Inconsistent evaluated_samples in {path}")
        for record in records:
            init_time = record["initialization_time"]
            key = (model_id, init_time)
            if key in seen:
                raise ValueError(f"Duplicate initialization: {key}")
            seen.add(key)
            ordinal = by_time[init_time]
            stamp = _utc(init_time)
            expected_valid = [_iso(stamp + timedelta(hours=int(lead))) for lead in lead_hours]
            expected_input = [_iso(stamp - timedelta(hours=6)), init_time]
            expected_record = {
                "ordinal": ordinal,
                "initialization_id": stamp.strftime("%Y%m%dT%H%M%SZ"),
                "input_times": expected_input,
                "valid_times": expected_valid,
                "initialization_year": 2023,
                "valid_years": [_utc(value).year for value in expected_valid],
            }
            for field, expected in expected_record.items():
                if record.get(field) != expected:
                    raise ValueError(f"Invalid {field} for {key}")
            if "lead_hours" in record and record["lead_hours"] != lead_hours.tolist():
                raise ValueError(f"Invalid lead_hours for {key}")
            for field, destination in (("baseline_per_step", baseline), ("full_per_step", full)):
                values = np.asarray(record.get(field), dtype=np.float64)
                if values.shape != (40,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"Invalid or incomplete {field} for {key}")
                destination[model_index, ordinal] = values
        sources.append({"path": str(path), "sha256": _sha256(path), "model_id": model_id})
    for model_id in model_ids:
        if model_id not in shard_counts or shard_ids[model_id] != set(range(shard_counts[model_id])):
            raise ValueError(f"Missing shards for {model_id}")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(full)):
        raise ValueError("Incomplete model × initialization × lead coverage")
    canonical = baseline[0]
    if np.any(canonical <= 0):
        raise ValueError("Canonical baseline losses must be positive at every initialization and lead")
    baseline_cfg = protocol["baseline"]
    repeatability = {}
    for index, model_id in enumerate(model_ids):
        absolute = np.abs(baseline[index] - canonical)
        tolerance = baseline_cfg["repeatability_atol"] + baseline_cfg["repeatability_rtol"] * np.abs(canonical)
        repeatability[model_id] = {
            "max_absolute_difference": float(np.max(absolute)),
            "max_relative_difference": float(np.max(absolute / canonical)),
            "passed": bool(np.all(absolute <= tolerance)),
        }
        if not repeatability[model_id]["passed"]:
            raise ValueError(f"Baseline repeatability failed for {model_id}: {repeatability[model_id]}")
    return baseline, full, {"sources": sources, "baseline_repeatability": repeatability}


def moving_block_counts(
    times: list[str], *, block_days: int, replicates: int, seed: int, stratified: bool = True
) -> np.ndarray:
    """Bootstrap multiplicities, using non-wrapping blocks of complete daily starts.

    Each row resamples exactly the original size of every seasonal segment.
    Reusing these counts for every model and lead preserves all paired outcomes.
    """
    if block_days < 1 or replicates < 2:
        raise ValueError("block_days must be positive and replicates must be at least two")
    dates = [_utc(value) for value in times]
    if any(right - left != timedelta(days=1) for left, right in zip(dates, dates[1:])):
        raise ValueError("Moving-block bootstrap requires ordered, consecutive daily starts")
    if stratified:
        segments = [np.asarray([i for i, date in enumerate(dates) if date.month in months])
                    for months in ((1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12,))]
    else:
        segments = [np.arange(len(times))]
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(times)), dtype=np.float64)
    for segment in segments:
        size = len(segment)
        if not size:
            continue
        if block_days > size:
            raise ValueError(f"Block length {block_days} exceeds segment length {size}")
        starts = rng.integers(0, size - block_days + 1, size=(replicates, (size + block_days - 1) // block_days))
        local_indices = (starts[..., None] + np.arange(block_days)).reshape(replicates, -1)[:, :size]
        np.add.at(counts, (np.arange(replicates)[:, None], segment[local_indices]), 1.0)
    return counts


def loss_improvement(baseline: np.ndarray, full: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return [replicate, model, rollout-or-lead] reductions of pooled losses."""
    pooled_base = counts @ baseline
    pooled_full = np.einsum("bi,mil->bml", counts, full, optimize=True)
    if np.any(pooled_base <= 0):
        raise ValueError("Undefined loss ratio: nonpositive pooled baseline")
    leads = 100.0 * (1.0 - pooled_full / pooled_base[:, None, :])
    rollout = 100.0 * (1.0 - pooled_full.sum(axis=-1) / pooled_base.sum(axis=-1)[:, None])
    return np.concatenate((rollout[..., None], leads), axis=-1)


def bootstrap_analysis(
    times: list[str], baseline: np.ndarray, full: np.ndarray, *, block_days: int,
    replicates: int, seed: int, stratified: bool = True,
) -> dict[str, Any]:
    counts = moving_block_counts(times, block_days=block_days, replicates=replicates, seed=seed, stratified=stratified)
    point = loss_improvement(baseline, full, np.ones((1, len(times))))[0]
    draws = loss_improvement(baseline, full, counts)
    lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
    # The simultaneous family contains the three rollout endpoints only.
    rollout_sd = np.std(draws[:, :, 0], axis=0, ddof=1)
    varying = rollout_sd > 1e-12
    if np.any(varying):
        deviation = np.abs((draws[:, varying, 0] - point[varying, 0]) / rollout_sd[varying])
        critical = float(np.quantile(np.max(deviation, axis=1), 0.95))
    else:
        critical = 0.0
    half_width = np.where(varying, critical * rollout_sd, 0.0)
    return {
        "method": "paired non-circular moving-block percentile bootstrap",
        "segmentation": "Jan-Feb, Mar-May, Jun-Aug, Sep-Nov, December" if stratified else "unstratified annual",
        "block_days": block_days, "replicates": replicates, "seed": seed,
        "improvement_pct": point.tolist(),
        "pointwise_ci95_lower": lower.tolist(), "pointwise_ci95_upper": upper.tolist(),
        "rollout_simultaneous_ci95_lower": (point[:, 0] - half_width).tolist(),
        "rollout_simultaneous_ci95_upper": (point[:, 0] + half_width).tolist(),
        "rollout_simultaneous_method": "maximum absolute standardized bootstrap deviation across all model rollout endpoints",
        "rollout_simultaneous_critical_value": critical,
        "rollout_degenerate_endpoints": np.flatnonzero(~varying).tolist(),
    }


def analyze(protocol: dict[str, Any], baseline: np.ndarray, full: np.ndarray) -> dict[str, Any]:
    settings = protocol["bootstrap"]
    if settings.get("segmentation") != "five_seasons":
        raise ValueError("Primary bootstrap segmentation must be five_seasons")
    common = {"replicates": settings["replicates"], "seed": settings["seed"]}
    times = protocol["initialization_times"]
    canonical = baseline[0]
    primary = bootstrap_analysis(times, canonical, full, block_days=settings["block_days"], **common)
    sensitivities = [bootstrap_analysis(times, canonical, full, block_days=days, **common)
                     for days in settings["sensitivity_block_days"]]
    sensitivities.append(bootstrap_analysis(times, canonical, full, block_days=settings["block_days"], stratified=False, **common))
    return {
        "schema_version": 1, "protocol_hash": protocol["protocol_hash"],
        "metric": "exact original GraphCast weighted loss reduction: 100*(1-sum(model)/sum(canonical baseline))",
        "canonical_baseline_model_id": "M1",
        "initialization_count": len(times), "lead_hours": protocol["lead_hours"],
        "model_ids": [model["model_id"] for model in protocol["models"]],
        "canonical_baseline_per_step": canonical.mean(axis=0).tolist(),
        "canonical_baseline_rollout": float(canonical.mean()),
        "model_per_step": full.mean(axis=1).tolist(),
        "model_rollout": full.mean(axis=(1, 2)).tolist(),
        "primary": primary, "sensitivities": sensitivities,
        "limitations": [
            "Per-lead confidence intervals are pointwise, not simultaneous across forecast leads.",
            "Rollout simultaneous intervals cover the three model-versus-baseline endpoints, not model ranking.",
            "Seasonal block boundaries break some cross-season dependence; annual and block-length sensitivities are reported.",
            "These accuracy metrics do not establish preservation of spatial structure or extreme-event skill.",
        ],
    }


def write_outputs(
    protocol: dict[str, Any], result: dict[str, Any], baseline: np.ndarray,
    full: np.ndarray, output_dir: Path, image_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Retain every loss, including repeated baselines, for independent reanalysis.
    with (output_dir / "per_initialization_losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "initialization_time", "lead_hours", "canonical_baseline_loss", "paired_baseline_loss", "model_loss", "protocol_hash"])
        for m, model_id in enumerate(result["model_ids"]):
            for i, stamp in enumerate(protocol["initialization_times"]):
                for k, lead in enumerate(protocol["lead_hours"]):
                    writer.writerow([model_id, stamp, lead, baseline[0, i, k], baseline[m, i, k], full[m, i, k], protocol["protocol_hash"]])
    with (output_dir / "loss_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["analysis", "model_id", "endpoint", "lead_hours", "block_days", "baseline_loss", "model_loss", "improvement_pct", "pointwise_ci95_lower", "pointwise_ci95_upper", "rollout_simultaneous_ci95_lower", "rollout_simultaneous_ci95_upper"])
        analyses = [("primary", result["primary"])] + [(f"sensitivity_{i + 1}", analysis) for i, analysis in enumerate(result["sensitivities"])]
        for name, analysis in analyses:
            for m, model_id in enumerate(result["model_ids"]):
                for endpoint in range(41):
                    is_rollout = endpoint == 0
                    writer.writerow([
                        name, model_id, "rollout" if is_rollout else "lead", "" if is_rollout else result["lead_hours"][endpoint - 1], analysis["block_days"],
                        result["canonical_baseline_rollout"] if is_rollout else result["canonical_baseline_per_step"][endpoint - 1],
                        result["model_rollout"][m] if is_rollout else result["model_per_step"][m][endpoint - 1],
                        analysis["improvement_pct"][m][endpoint], analysis["pointwise_ci95_lower"][m][endpoint], analysis["pointwise_ci95_upper"][m][endpoint],
                        analysis["rollout_simultaneous_ci95_lower"][m] if is_rollout else "",
                        analysis["rollout_simultaneous_ci95_upper"][m] if is_rollout else "",
                    ])
    (output_dir / "loss_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    primary = result["primary"]
    lines = [
        "# Frozen-model 2023 exact GraphCast loss evaluation", "",
        f"All 365 daily 00 UTC initializations in 2023; 40 six-hour leads. Protocol `{protocol['protocol_hash']}`.", "",
        "Every model uses the canonical M1 GraphCast baseline. Positive loss reduction means improvement.", "",
        f"Paired non-wrapping {primary['block_days']}-day blocks, five calendar segments, {primary['replicates']:,} replicates, seed {primary['seed']}. Each draw keeps all leads and all models paired.", "",
        "| Model | Rollout reduction | Pointwise 95% CI | Simultaneous 95% CI (three rollouts) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for m, model_id in enumerate(result["model_ids"]):
        lines.append(f"| {model_id} | {primary['improvement_pct'][m][0]:.3f}% | [{primary['pointwise_ci95_lower'][m][0]:.3f}, {primary['pointwise_ci95_upper'][m][0]:.3f}] | [{primary['rollout_simultaneous_ci95_lower'][m]:.3f}, {primary['rollout_simultaneous_ci95_upper'][m]:.3f}] |")
    lines += ["", "All per-lead intervals, block-length/annual sensitivities, and baseline repeatability measurements are retained in `loss_summary.json`; `loss_summary.csv` and `per_initialization_losses.csv` allow independent reanalysis.", ""]
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if image_dir is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        image_dir.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(figsize=(8, 4.5))
        days = np.asarray(result["lead_hours"]) / 24.0
        for m, model_id in enumerate(result["model_ids"]):
            line, = axes.plot(days, primary["improvement_pct"][m][1:], label=model_id)
            axes.fill_between(days, primary["pointwise_ci95_lower"][m][1:], primary["pointwise_ci95_upper"][m][1:], alpha=0.18, color=line.get_color())
        axes.axhline(0, color="black", linewidth=0.8)
        axes.set(xlabel="Forecast lead (days)", ylabel="Exact GraphCast loss reduction (%)", title="2023 daily starts · pointwise 95% paired block intervals")
        axes.grid(alpha=0.2)
        axes.legend()
        figure.tight_layout()
        figure.savefig(image_dir / "exact_loss_reduction_2023.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shard-json", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-image-dir", type=Path)
    args = parser.parse_args()
    protocol = _load(args.protocol)
    baseline, full, audit = merge_shards(protocol, args.shard_json)
    result = analyze(protocol, baseline, full)
    result.update(audit)
    result["protocol_file_sha256"] = _sha256(args.protocol)
    write_outputs(protocol, result, baseline, full, args.output_dir, args.output_image_dir)
    print(f"Audited all 365 × 40 outcomes for three models; saved {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
