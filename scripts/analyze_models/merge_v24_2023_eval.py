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
from scripts.experiments.numerical_policy import is_advisory, numerical_comparison

MODEL_IDS = ("M1", "M2", "M3", "M1_reset", "C_carry", "C_reset", "R_reset")
VARIABLES = (
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr", "geopotential",
    "temperature", "specific_humidity", "u_component_of_wind",
    "v_component_of_wind", "vertical_velocity",
)
SECONDARY_ENDPOINTS = (
    "M1_vs_GC_day10", "M1_vs_M1_reset_rollout",
    "C_carry_vs_C_reset_rollout", "C_carry_vs_R_reset_rollout",
)


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
    is_advisory(protocol)
    times = protocol["initialization_times"]
    version = protocol.get("schema_version", 1)
    if version not in (1, 2):
        raise ValueError(f"Unsupported protocol schema_version: {version}")
    stride = protocol.get("initialization_stride_days", 1)
    if stride != (3 if version == 2 else 1):
        raise ValueError("Protocol cadence must be daily for schema1 or every third day for schema2")
    expected = [
        _iso(datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=day))
        for day in range(0, 365, stride)
    ]
    if times != expected:
        raise ValueError(f"Protocol must list all {len(expected)} prespecified 00 UTC starts in 2023, in order")
    leads = np.asarray(protocol["lead_hours"])
    if protocol["target_steps"] != 40 or not np.array_equal(leads, np.arange(6, 241, 6)):
        raise ValueError("Protocol must score all 40 six-hour leads through 240 hours")
    model_ids = [model["model_id"] for model in protocol["models"]]
    if model_ids[:3] != ["M1", "M2", "M3"] or len(set(model_ids)) != len(model_ids):
        raise ValueError("Protocol must start with unique frozen M1, M2, M3 model IDs")
    if version == 2:
        if model_ids != list(MODEL_IDS):
            raise ValueError("Schema2 requires the seven frozen model policies in order")
        analysis = protocol.get("analysis", {})
        if analysis.get("primary_endpoint") != "M1_vs_GC_rollout" or analysis.get("secondary_endpoints") != list(SECONDARY_ENDPOINTS):
            raise ValueError("Schema2 requires the frozen primary and four secondary endpoints")
        settings = protocol["bootstrap"]
        if (settings.get("segmentation") != "annual" or settings.get("block_starts") != 5
                or settings.get("sensitivity_block_starts") != [3, 7]
                or settings.get("quarter_sensitivity") is not True):
            raise ValueError("Schema2 requires annual five-start blocks, 3/7-start and quarterly sensitivities")
        exposure = protocol.get("prior_exposure", {})
        exposed = exposure.get("initialization_times", [])
        expected_exposed = [_iso(datetime(2023, month, 15, tzinfo=timezone.utc))
                            for month in (1, 2, 4, 5, 7, 8, 10, 11)]
        if exposed != expected_exposed or exposure.get("sampled_initialization_times") != [t for t in times if t in exposed]:
            raise ValueError("Schema2 must disclose the eight prior pilot dates and their three sampled intersections")
        if exposure.get("selection_influence") != "none_reported":
            raise ValueError("Unexpected prior-exposure selection-influence declaration")
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
    variable_data = None
    if protocol.get("schema_version", 1) == 2:
        variable_data = {
            "variables": list(VARIABLES), "metadata": None,
            "baseline": np.full((*baseline.shape, len(VARIABLES)), np.nan),
            "full": np.full((*baseline.shape, len(VARIABLES)), np.nan),
        }
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
        if "reset_state_every_step" in identities[model_id]:
            if payload.get("rs_reset_every_step") != identities[model_id]["reset_state_every_step"]:
                raise ValueError(f"Shard {path} has unexpected memory reset policy")
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
        if variable_data is not None:
            metadata = payload.get("variable_mse_metadata")
            if not isinstance(metadata, dict) or set(metadata) != set(VARIABLES):
                raise ValueError(f"Missing or incomplete variable_mse_metadata in {path}")
            for variable, entry in metadata.items():
                if (not isinstance(entry.get("units"), str) or not entry["units"]
                        or entry.get("spatial_weighting") != "normalized_cosine_latitude"
                        or entry.get("pressure_pooling") != ("surface" if variable in VARIABLES[:5] else "equal_levels")):
                    raise ValueError(f"Invalid variable MSE metadata for {variable} in {path}")
            if variable_data["metadata"] is None:
                variable_data["metadata"] = metadata
            elif variable_data["metadata"] != metadata:
                raise ValueError(f"Inconsistent variable MSE metadata in {path}")
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
            if variable_data is not None:
                variables = record.get("variable_mse")
                if not isinstance(variables, dict) or set(variables) != set(VARIABLES):
                    raise ValueError(f"Missing or incomplete variable_mse for {key}")
                for v, variable in enumerate(VARIABLES):
                    for branch in ("baseline", "full"):
                        values = np.asarray(variables[variable].get(f"{branch}_per_step"), dtype=np.float64)
                        if values.shape != (40,) or not np.all(np.isfinite(values)) or np.any(values < 0):
                            raise ValueError(f"Invalid variable_mse {variable}/{branch} for {key}")
                        variable_data[branch][model_index, ordinal, :, v] = values
        if variable_data is not None:
            ordinals = [by_time[t] for t in actual_times]
            exact_aggregate = payload.get("original_graphcast_loss", {})
            for branch, values in (("baseline", baseline), ("full", full)):
                pooled = values[model_index, ordinals].mean(axis=0)
                saved = np.asarray(exact_aggregate.get(f"{branch}_per_step"), dtype=np.float64)
                rollout = exact_aggregate.get(f"{branch}_rollout")
                if (saved.shape != (40,) or not np.all(np.isfinite(saved))
                        or not np.allclose(pooled, saved, rtol=2e-6, atol=0)
                        or not isinstance(rollout, (int, float)) or not np.isfinite(rollout)
                        or not np.isclose(float(pooled.mean()), rollout, rtol=2e-6, atol=0)):
                    raise ValueError(f"Per-initialization exact losses disagree with aggregate {branch} losses in {path}")
            aggregates = payload.get("per_variable_per_step", {})
            if set(aggregates) != set(VARIABLES):
                raise ValueError(f"Missing variable RMSE aggregates in {path}")
            for v, variable in enumerate(VARIABLES):
                for branch in ("baseline", "full"):
                    pooled = np.sqrt(variable_data[branch][model_index, ordinals, :, v].mean(axis=0))
                    saved = np.asarray(aggregates[variable].get(f"rmse_{branch}"), dtype=np.float64)
                    if saved.shape != (40,) or not np.all(np.isfinite(saved)) or not np.allclose(pooled, saved, rtol=2e-6, atol=0):
                        raise ValueError(f"Variable MSE disagrees with aggregate RMSE for {variable}/{branch} in {path}")
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
        repeatability[model_id] = numerical_comparison(
            baseline[index], canonical, rtol=baseline_cfg["repeatability_rtol"],
            atol=baseline_cfg["repeatability_atol"], label=f"Baseline repeatability failed for {model_id}",
            advisory=is_advisory(protocol))
    audit = {"sources": sources, "baseline_repeatability": repeatability,
             "numerical_status": "warning" if any(not row["passed"] for row in repeatability.values()) else "passed",
             "paired_baseline_sensitivity": paired_baseline_sensitivity(model_ids, baseline, full)}
    if variable_data is not None:
        if not all(np.all(np.isfinite(variable_data[b])) for b in ("baseline", "full")):
            raise ValueError("Incomplete variable MSE coverage")
        audit["variable_mse"] = variable_data
    return baseline, full, audit


def paired_baseline_sensitivity(model_ids, baseline, full):
    """Descriptive execution-noise sensitivity; preserve all canonical endpoints."""
    canonical = 100 * (1 - full.mean(axis=(1, 2)) / baseline[0].mean())
    paired = 100 * (1 - full.mean(axis=(1, 2)) / baseline.mean(axis=(1, 2)))
    canonical_order = np.argsort(-canonical, kind="stable").tolist()
    paired_order = np.argsort(-paired, kind="stable").tolist()
    return {
        "scope": "Descriptive denominator sensitivity, not a new model-selection or confirmatory endpoint.",
        "models": [{"model_id": model_id,
                    "canonical_improvement_pct_rollout": float(canonical[i]),
                    "paired_improvement_pct_rollout": float(paired[i]),
                    "difference_percentage_points": float(paired[i] - canonical[i]),
                    "paired_improvement_pct_per_step": (100 * (1 - full[i].mean(0) / baseline[i].mean(0))).tolist()}
                   for i, model_id in enumerate(model_ids)],
        "canonical_ranking": [model_ids[i] for i in canonical_order],
        "paired_ranking": [model_ids[i] for i in paired_order],
        "ranking_changed": canonical_order != paired_order,
    }


def numerical_summary_lines(result):
    sensitivity = result.get("paired_baseline_sensitivity")
    if sensitivity is None:
        return []
    lines = ["", f"Numerical diagnostics: **{result.get('numerical_status', 'unknown')}**. "
             "Canonical M1 endpoints above are unchanged; the following descriptive sensitivity uses each model's own repeated GraphCast baseline.", "",
             "| Model | Canonical reduction | Paired reduction | Difference (percentage points) |",
             "| --- | ---: | ---: | ---: |"]
    for row in sensitivity["models"]:
        lines.append(f"| {row['model_id']} | {row['canonical_improvement_pct_rollout']:.3f}% | "
                     f"{row['paired_improvement_pct_rollout']:.3f}% | {row['difference_percentage_points']:+.3f} |")
    lines.extend(["", "Canonical ranking: " + ", ".join(sensitivity["canonical_ranking"]) + ".",
                  "Paired-baseline ranking: " + ", ".join(sensitivity["paired_ranking"]) + ". "
                  "These rankings are descriptive, not model-selection decisions.", ""])
    for model_id, row in result.get("baseline_repeatability", {}).items():
        pooled = row.get("pooled_relative_difference")
        lines.append(f"- {model_id}: pooled baseline drift {100 * pooled:.4f}%; "
                     f"worst start/lead drift {100 * row['max_relative_difference']:.4f}%.")
    return lines


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


def cadence_block_counts(
    times: list[str], *, stride_days: int, block_starts: int,
    replicates: int, seed: int, segmentation: str = "annual",
) -> np.ndarray:
    """Resample the original regular date grid, including subsequently masked dates."""
    if block_starts < 1 or replicates < 2 or stride_days < 1 or not times:
        raise ValueError("Invalid cadence bootstrap dimensions")
    dates = [_utc(value) for value in times]
    if any(b - a != timedelta(days=stride_days) for a, b in zip(dates, dates[1:])):
        raise ValueError("Cadence bootstrap requires the complete, regularly spaced initialization grid")
    if segmentation == "annual":
        segments = [np.arange(len(times))]
    elif segmentation == "calendar_quarters":
        segments = [np.asarray([i for i, date in enumerate(dates) if (date.month - 1) // 3 == q])
                    for q in range(4)]
    else:
        raise ValueError(f"Unknown cadence bootstrap segmentation: {segmentation}")
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(times)), dtype=np.float64)
    for segment in segments:
        size = len(segment)
        if size < block_starts:
            raise ValueError("Bootstrap block exceeds calendar segment")
        starts = rng.integers(0, size - block_starts + 1,
                             size=(replicates, (size + block_starts - 1) // block_starts))
        indices = (starts[..., None] + np.arange(block_starts)).reshape(replicates, -1)[:, :size]
        np.add.at(counts, (np.arange(replicates)[:, None], segment[indices]), 1.0)
    return counts


def named_endpoint_draws(
    baseline: np.ndarray, full: np.ndarray, counts: np.ndarray, model_ids: list[str],
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    """Return paired effects and absolute reductions, pooling before every ratio.

    Contrasts use percentage points of GC-referenced loss reduction; a positive
    effect always favors the first arm. Counts may include a fixed exposure mask.
    """
    endpoints = [
        ("M1_vs_GC_rollout", "M1", "GC", "rollout", "primary"),
        ("M1_vs_GC_day10", "M1", "GC", "day10", "secondary"),
        ("M1_vs_M1_reset_rollout", "M1", "M1_reset", "rollout", "secondary"),
        ("C_carry_vs_C_reset_rollout", "C_carry", "C_reset", "rollout", "secondary"),
        ("C_carry_vs_R_reset_rollout", "C_carry", "R_reset", "rollout", "secondary"),
        ("C_reset_vs_R_reset_rollout", "C_reset", "R_reset", "rollout", "descriptive"),
    ]
    sample_counts = counts.sum(axis=1)
    if np.any(sample_counts <= 0):
        raise ValueError("No eligible initializations in a bootstrap draw")
    pooled_base = counts @ baseline
    pooled_full = np.einsum("bi,mil->bml", counts, full, optimize=True)
    effects, differences, definitions = [], [], []
    for name, first, second, endpoint, role in endpoints:
        lead = slice(None) if endpoint == "rollout" else slice(-1, None)
        a = pooled_full[:, model_ids.index(first), lead].sum(axis=1)
        b = (pooled_base[:, lead] if second == "GC"
             else pooled_full[:, model_ids.index(second), lead]).sum(axis=1)
        gc = pooled_base[:, lead].sum(axis=1)
        if np.any(gc <= 0):
            raise ValueError("Undefined contrast: nonpositive pooled canonical baseline")
        effects.append(100 * (b - a) / gc)
        differences.append((b - a) / (sample_counts * (baseline.shape[1] if endpoint == "rollout" else 1)))
        definitions.append(dict(endpoint_id=name, first_arm=first, second_arm=second,
                                aggregation=endpoint, role=role,
                                effect_units="percent" if second == "GC" else "percentage_points"))
    return definitions, np.stack(effects, axis=1), np.stack(differences, axis=1)


def cadence_bootstrap_analysis(
    protocol: dict[str, Any], baseline: np.ndarray, full: np.ndarray, *,
    block_starts: int, segmentation: str = "annual", eligible: np.ndarray | None = None,
    analysis_id: str = "primary",
) -> dict[str, Any]:
    settings = protocol["bootstrap"]
    times = protocol["initialization_times"]
    model_ids = [m["model_id"] for m in protocol["models"]]
    mask = np.ones(len(times), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if mask.shape != (len(times),) or not mask.any():
        raise ValueError("Invalid initialization eligibility mask")
    counts = cadence_block_counts(times, stride_days=protocol["initialization_stride_days"],
                                  block_starts=block_starts, segmentation=segmentation,
                                  replicates=settings["replicates"], seed=settings["seed"])
    counts *= mask[None, :]
    point_counts = mask.astype(float)[None, :]
    improvement = loss_improvement(baseline, full, point_counts)[0]
    improvement_draws = loss_improvement(baseline, full, counts)
    lower, upper = np.quantile(improvement_draws, [0.025, 0.975], axis=0)
    definitions, point, absolute = named_endpoint_draws(baseline, full, point_counts, model_ids)
    _, draws, absolute_draws = named_endpoint_draws(baseline, full, counts, model_ids)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    abs_lo, abs_hi = np.quantile(absolute_draws, [0.025, 0.975], axis=0)
    family = np.asarray([i for i, item in enumerate(definitions) if item["role"] == "secondary"])
    sd = np.std(draws[:, family], axis=0, ddof=1)
    varying = sd > 1e-12
    critical = (float(np.quantile(np.max(np.abs((draws[:, family[varying]] - point[0, family[varying]])
                                               / sd[varying]), axis=1), 0.95)) if varying.any() else 0.0)
    half_width = np.where(varying, critical * sd, 0.0)
    simultaneous = {int(i): (float(point[0, i] - half_width[j]), float(point[0, i] + half_width[j]))
                    for j, i in enumerate(family)}
    primary_passed = bool(lo[0] > 0)
    named = []
    for i, definition in enumerate(definitions):
        item = dict(definition, effect=float(point[0, i]), ci95_lower=float(lo[i]), ci95_upper=float(hi[i]),
                    absolute_loss_reduction=float(absolute[0, i]),
                    absolute_pointwise_ci95_lower=float(abs_lo[i]), absolute_pointwise_ci95_upper=float(abs_hi[i]))
        if i in simultaneous:
            item.update(simultaneous_ci95_lower=simultaneous[i][0], simultaneous_ci95_upper=simultaneous[i][1],
                        interval_supports_positive=simultaneous[i][0] > 0,
                        confirmatory_supported=primary_passed and simultaneous[i][0] > 0)
        else:
            item.update(interval_supports_positive=bool(lo[i] > 0),
                        confirmatory_supported=primary_passed if definition["role"] == "primary" else False)
        named.append(item)
    return dict(
        analysis_id=analysis_id, method="paired non-wrapping moving-block percentile bootstrap",
        segmentation=segmentation, block_starts=block_starts,
        block_elapsed_days=(block_starts - 1) * protocol["initialization_stride_days"],
        replicates=settings["replicates"], seed=settings["seed"],
        initialization_count=int(mask.sum()), excluded_initialization_times=[t for t, keep in zip(times, mask) if not keep],
        resampling_grid_initialization_count=len(times),
        bootstrap_eligible_count_min=int(counts.sum(axis=1).min()),
        bootstrap_eligible_count_max=int(counts.sum(axis=1).max()),
        canonical_baseline_rollout=float(baseline[mask].mean()),
        canonical_baseline_per_step=baseline[mask].mean(axis=0).tolist(),
        model_rollout=full[:, mask].mean(axis=(1, 2)).tolist(),
        model_per_step=full[:, mask].mean(axis=1).tolist(),
        improvement_pct=improvement.tolist(), pointwise_ci95_lower=lower.tolist(), pointwise_ci95_upper=upper.tolist(),
        named_endpoints=named, secondary_family=list(SECONDARY_ENDPOINTS),
        simultaneous_method="maximum absolute standardized bootstrap deviation across four prespecified secondary endpoints",
        simultaneous_critical_value=critical,
    )


def descriptive_diagnostics(
    protocol: dict[str, Any], baseline: np.ndarray, full: np.ndarray,
    variable_mse: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pool physical MSE before square root; no seasonal or variable significance claims."""
    dates = [_utc(t) for t in protocol["initialization_times"]]
    groups = {"annual": tuple(range(1, 13)), "DJF": (12, 1, 2), "MAM": (3, 4, 5),
              "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    exact, variables = [], []
    for season, months in groups.items():
        mask = np.asarray([d.month in months for d in dates])
        for m, model in enumerate(protocol["models"]):
            for endpoint in range(41):
                lead = slice(None) if endpoint == 0 else slice(endpoint - 1, endpoint)
                gc = float(baseline[mask, lead].mean())
                model_loss = float(full[m, mask, lead].mean())
                common = dict(season=season, initialization_count=int(mask.sum()), model_id=model["model_id"],
                              endpoint="rollout" if endpoint == 0 else "lead",
                              lead_hours=None if endpoint == 0 else protocol["lead_hours"][endpoint - 1])
                exact.append(dict(common, baseline_loss=gc, model_loss=model_loss,
                                  improvement_pct=100 * (1 - model_loss / gc)))
                if variable_mse is None:
                    continue
                for v, variable in enumerate(variable_mse["variables"]):
                    base_rmse = float(np.sqrt(variable_mse["baseline"][0, mask, lead, v].mean()))
                    model_rmse = float(np.sqrt(variable_mse["full"][m, mask, lead, v].mean()))
                    metadata = variable_mse["metadata"][variable]
                    variables.append(dict(common, variable=variable, **metadata, baseline_rmse=base_rmse,
                                          model_rmse=model_rmse, absolute_rmse_reduction=base_rmse - model_rmse,
                                          rmse_reduction_pct=100 * (1 - model_rmse / base_rmse) if base_rmse > 0 else None))
    return {"interpretation": "Descriptive pooled scores; no multiplicity-adjusted seasonal or variable claims.",
            "exact_loss": exact, "variable_rmse": variables}


def analyze_cadence(protocol, baseline, full, variable_mse=None):
    times, model_ids, _ = validate_protocol(protocol)
    if baseline.shape != full.shape or full.shape != (len(model_ids), len(times), 40):
        raise ValueError("Loss arrays do not match protocol coverage")
    if not np.all(np.isfinite([baseline, full])) or np.any(baseline <= 0) or np.any(full < 0):
        raise ValueError("Invalid losses for cadence analysis")
    canonical = baseline[0]
    primary = cadence_bootstrap_analysis(protocol, canonical, full, block_starts=5)
    sensitivities = [cadence_bootstrap_analysis(protocol, canonical, full, block_starts=b,
                                               analysis_id=f"annual_{b}_starts") for b in (3, 7)]
    sensitivities.append(cadence_bootstrap_analysis(protocol, canonical, full, block_starts=5,
                                                    segmentation="calendar_quarters", analysis_id="quarterly_5_starts"))
    exposed = set(protocol["prior_exposure"]["sampled_initialization_times"])
    sensitivities.append(cadence_bootstrap_analysis(protocol, canonical, full, block_starts=5,
                                                    eligible=np.asarray([t not in exposed for t in times]),
                                                    analysis_id="exclude_prior_pilot_starts"))
    claims = []
    for endpoint in primary["named_endpoints"]:
        if endpoint["role"] == "descriptive":
            continue
        checks = {a["analysis_id"]: next(e["confirmatory_supported"] for e in a["named_endpoints"]
                                          if e["endpoint_id"] == endpoint["endpoint_id"])
                  for a in [primary, *sensitivities]}
        claims.append(dict(endpoint_id=endpoint["endpoint_id"], supported_by_primary_analysis=endpoint["confirmatory_supported"],
                           supported_in_all_sensitivities=all(checks.values()), analysis_support=checks))
    return dict(
        schema_version=2, protocol_hash=protocol["protocol_hash"], initialization_count=len(times),
        initialization_stride_days=3, lead_hours=protocol["lead_hours"], model_ids=model_ids,
        metric="exact original GraphCast weighted loss reduction: 100*(1-sum(model)/sum(canonical baseline))",
        canonical_baseline_model_id="M1", canonical_baseline_per_step=canonical.mean(axis=0).tolist(),
        canonical_baseline_rollout=float(canonical.mean()), model_per_step=full.mean(axis=1).tolist(),
        model_rollout=full.mean(axis=(1, 2)).tolist(), primary=primary, sensitivities=sensitivities,
        claim_sensitivity=claims, prior_exposure=protocol["prior_exposure"],
        diagnostics=descriptive_diagnostics(protocol, canonical, full, variable_mse),
        limitations=[
            "This year-round every-third-day sample is not the complete daily 2023 forecast population.",
            "Uncertainty conditions on fixed checkpoints and this year; it does not measure training-seed or between-year variation.",
            "All leads and models remain paired; pointwise lead intervals and variable/seasonal diagnostics do not establish simultaneous skill.",
            "Secondary confirmation requires primary success and a positive simultaneous lower bound in the four-endpoint family.",
            "Non-wrapping blocks underweight boundary dates; quarterly blocks also break cross-quarter dependence. Sensitivities disclose inference stability.",
            "Excluding three sampled pilot starts does not undo prior exposure or remove all overlapping pilot forecast windows.",
            "M1 reset tests dependence on state carry; the separately trained V22 control supports V22, not a matched capacity-control claim for M1.",
            "Accuracy scores do not establish preservation of spatial structure or extreme-event skill.",
        ],
    )


def analyze(protocol: dict[str, Any], baseline: np.ndarray, full: np.ndarray,
            variable_mse: dict[str, Any] | None = None) -> dict[str, Any]:
    if protocol.get("schema_version", 1) == 2:
        return analyze_cadence(protocol, baseline, full, variable_mse)
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
            "Rollout simultaneous intervals cover all frozen model-versus-baseline endpoints, not model ranking.",
            "Seasonal block boundaries break some cross-season dependence; annual and block-length sensitivities are reported.",
            "These accuracy metrics do not establish preservation of spatial structure or extreme-event skill.",
        ],
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_cadence_outputs(protocol, result, baseline, full, output_dir, image_dir, variable_mse):
    output_dir.mkdir(parents=True, exist_ok=True)
    analyses = [result["primary"], *result["sensitivities"]]
    with (output_dir / "per_initialization_losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "initialization_time", "lead_hours", "canonical_baseline_loss",
                         "paired_baseline_loss", "model_loss", "protocol_hash"])
        for m, model_id in enumerate(result["model_ids"]):
            for i, stamp in enumerate(protocol["initialization_times"]):
                for k, lead in enumerate(protocol["lead_hours"]):
                    writer.writerow([model_id, stamp, lead, baseline[0, i, k], baseline[m, i, k],
                                     full[m, i, k], protocol["protocol_hash"]])
    if variable_mse is not None:
        with (output_dir / "per_initialization_variable_mse.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_id", "initialization_time", "lead_hours", "variable", "units",
                             "canonical_baseline_mse", "paired_baseline_mse", "model_mse", "protocol_hash"])
            for m, model_id in enumerate(result["model_ids"]):
                for i, stamp in enumerate(protocol["initialization_times"]):
                    for k, lead in enumerate(protocol["lead_hours"]):
                        for v, variable in enumerate(variable_mse["variables"]):
                            writer.writerow([model_id, stamp, lead, variable, variable_mse["metadata"][variable]["units"],
                                             variable_mse["baseline"][0, i, k, v], variable_mse["baseline"][m, i, k, v],
                                             variable_mse["full"][m, i, k, v], protocol["protocol_hash"]])
    rows = []
    for analysis in analyses:
        for m, model_id in enumerate(result["model_ids"]):
            for endpoint in range(41):
                rollout = endpoint == 0
                rows.append(dict(
                    analysis=analysis["analysis_id"], model_id=model_id,
                    endpoint="rollout" if rollout else "lead",
                    lead_hours=None if rollout else protocol["lead_hours"][endpoint - 1],
                    block_starts=analysis["block_starts"], segmentation=analysis["segmentation"],
                    initialization_count=analysis["initialization_count"],
                    baseline_loss=analysis["canonical_baseline_rollout"] if rollout else analysis["canonical_baseline_per_step"][endpoint - 1],
                    model_loss=analysis["model_rollout"][m] if rollout else analysis["model_per_step"][m][endpoint - 1],
                    improvement_pct=analysis["improvement_pct"][m][endpoint],
                    pointwise_ci95_lower=analysis["pointwise_ci95_lower"][m][endpoint],
                    pointwise_ci95_upper=analysis["pointwise_ci95_upper"][m][endpoint],
                ))
    _write_table(output_dir / "loss_summary.csv", rows)
    _write_table(output_dir / "named_endpoints.csv", [dict(analysis=a["analysis_id"], **e)
                                                     for a in analyses for e in a["named_endpoints"]])
    _write_table(output_dir / "seasonal_exact_loss.csv", result["diagnostics"]["exact_loss"])
    _write_table(output_dir / "variable_rmse.csv", result["diagnostics"]["variable_rmse"])
    _write_table(output_dir / "claim_sensitivity.csv", [
        dict(endpoint_id=c["endpoint_id"], analysis=name, confirmatory_supported=supported,
             supported_in_all_sensitivities=c["supported_in_all_sensitivities"])
        for c in result["claim_sensitivity"] for name, supported in c["analysis_support"].items()
    ])
    (output_dir / "loss_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    primary = result["primary"]
    main_endpoint = primary["named_endpoints"][0]
    lines = [
        "# Frozen-model 2023 verification", "",
        "122 initializations at 00 UTC every third day, January 1–December 30, 2023; 40 six-hour leads through day 10. Late-December forecasts use January 2024 truth.", "",
        f"Protocol `{protocol['protocol_hash']}`. Every loss reduction uses the canonical M1 execution of frozen GraphCast.", "",
        f"The prespecified M1 full-rollout loss reduction is {main_endpoint['effect']:.3f}% (paired 95% CI {main_endpoint['ci95_lower']:.3f} to {main_endpoint['ci95_upper']:.3f}%). Primary confirmation: {'supported' if main_endpoint['confirmatory_supported'] else 'not established'}.", "",
        f"Paired non-wrapping blocks of five initialization positions (12 elapsed days), {primary['replicates']:,} draws, seed {primary['seed']}. All models and leads share each draw.", "",
        "| Endpoint | Effect | Units | 95% interval | Confirmatory support |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for endpoint in primary["named_endpoints"]:
        lo = endpoint.get("simultaneous_ci95_lower", endpoint["ci95_lower"])
        hi = endpoint.get("simultaneous_ci95_upper", endpoint["ci95_upper"])
        support = "descriptive" if endpoint["role"] == "descriptive" else ("yes" if endpoint["confirmatory_supported"] else "no")
        lines.append(f"| {endpoint['endpoint_id']} | {endpoint['effect']:.3f} | {endpoint['effect_units'].replace('_', ' ')} | [{lo:.3f}, {hi:.3f}] | {support} |")
    lines += ["", "Primary and descriptive endpoint intervals are pointwise. The four secondary intervals form one simultaneous 95% family; secondary confirmation is gated by primary success. Positive paired contrasts favor the first arm. Their percentage-point effects compare GC-referenced reductions, not percentage reductions relative to the second arm.", "",
              "M2/M3, seasonal scores and variable RMSE are predefined descriptive diagnostics; the test set does not select a new winner.", "",
              "Sensitivity analyses use annual 3-/7-start blocks, calendar-quarter 5-start blocks, and the original 122-position grid with three prior-pilot starts masked from scores (119 retained).", "",
              "| Claim | Primary analysis | All prespecified sensitivities |", "| --- | --- | --- |"]
    for claim in result["claim_sensitivity"]:
        lines.append(f"| {claim['endpoint_id']} | {'supported' if claim['supported_by_primary_analysis'] else 'not established'} | {'supported' if claim['supported_in_all_sensitivities'] else 'not established in every analysis'} |")
    exposure = protocol["prior_exposure"]
    lines += ["", "Previously evaluated pilot dates: " + ", ".join(t[:10] for t in exposure["initialization_times"]) + ". The user reports no influence on model selection. This evaluation is not described as untouched.", "",
              "The sampled pilot dates removed only in sensitivity are " + ", ".join(t[:10] for t in exposure["sampled_initialization_times"]) + ".", "",
              "Exact losses, raw per-initialization variable MSE, endpoint intervals, seasonal diagnostics, and sensitivity checks are available in the accompanying JSON/CSV files. Variable RMSE pools cosine-latitude-weighted squared errors and equal pressure levels before taking a square root.", ""]
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    lines.extend(numerical_summary_lines(result))
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    if image_dir is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image_dir.mkdir(parents=True, exist_ok=True)

    def save(figure, stem):
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(image_dir / f"{stem}.{suffix}", dpi=180)
        plt.close(figure)

    days = np.asarray(protocol["lead_hours"]) / 24
    figure, axis = plt.subplots(figsize=(9, 5))
    for m, model_id in enumerate(result["model_ids"]):
        line, = axis.plot(days, primary["improvement_pct"][m][1:], label=model_id)
        axis.fill_between(days, primary["pointwise_ci95_lower"][m][1:], primary["pointwise_ci95_upper"][m][1:],
                          alpha=0.10, color=line.get_color())
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="Forecast lead (days)", ylabel="Exact GraphCast loss reduction (%)",
             title="2023 · 122 starts · pointwise 95% paired block intervals")
    axis.legend(ncol=2)
    save(figure, "exact_loss_reduction_2023")
    figure, axis = plt.subplots(figsize=(8, 3.8))
    comparisons = primary["named_endpoints"][2:5]
    for index, endpoint in enumerate(comparisons):
        axis.plot([endpoint["simultaneous_ci95_lower"], endpoint["simultaneous_ci95_upper"]], [index, index], color="C0")
        axis.plot(endpoint["effect"], index, "o", color="C0")
    axis.set_yticks(range(len(comparisons)), [e["endpoint_id"].removesuffix("_rollout").replace("_vs_", " vs ") for e in comparisons])
    axis.axvline(0, color="black", linewidth=.8)
    axis.set(xlabel="Difference in GC-referenced loss reduction (percentage points)",
             title="Memory comparisons · simultaneous intervals from four-endpoint family")
    save(figure, "memory_contrasts_2023")
    records = result["diagnostics"]["variable_rmse"]
    if records:
        figure, axes = plt.subplots(4, 3, figsize=(12, 11))
        for axis, variable in zip(axes.flat, VARIABLES):
            selected = [r for r in records if r["season"] == "annual" and r["model_id"] == "M1"
                        and r["variable"] == variable and r["endpoint"] == "lead"]
            axis.plot(days, [r["baseline_rmse"] for r in selected], label="GraphCast")
            axis.plot(days, [r["model_rmse"] for r in selected], label="M1")
            axis.set(title=variable, xlabel="Forecast lead (days)", ylabel=selected[0]["units"])
        axes.flat[-1].axis("off")
        axes.flat[0].legend()
        figure.suptitle("2023 annual physical-unit RMSE · descriptive")
        save(figure, "variable_rmse_2023")


def write_outputs(
    protocol: dict[str, Any], result: dict[str, Any], baseline: np.ndarray,
    full: np.ndarray, output_dir: Path, image_dir: Path | None = None,
    variable_mse: dict[str, Any] | None = None,
) -> None:
    if result.get("schema_version", 1) == 2:
        write_cadence_outputs(protocol, result, baseline, full, output_dir, image_dir, variable_mse)
        return
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
        "| Model | Rollout reduction | Pointwise 95% CI | Simultaneous 95% CI (all model rollouts) |",
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
    variable_mse = audit.pop("variable_mse", None)
    result = analyze(protocol, baseline, full, variable_mse)
    result.update(audit)
    result.update(status="complete_with_warnings" if audit["numerical_status"] == "warning" else "complete",
                  integrity_passed=True, numerical_policy=protocol.get("numerical_policy", "strict"))
    result["protocol_file_sha256"] = _sha256(args.protocol)
    write_outputs(protocol, result, baseline, full, args.output_dir, args.output_image_dir, variable_mse)
    print(f"Audited all {len(protocol['initialization_times'])} × 40 outcomes for {len(protocol['models'])} model policies; saved {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
