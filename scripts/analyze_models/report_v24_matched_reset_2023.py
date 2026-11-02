#!/usr/bin/env python3
"""Strict paired report for the selected V24 matched reset-training control.

Reuses the original frozen 122-start M1 and M1_reset forecasts without changing
their protocol or confirmatory family. All ratios pool exact GraphCast losses
before division; the new training-policy contrast is a separate planned analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODEL_IDS = ("M1", "M1_reset", "R_matched")
POLICY_LABELS = ("Carry train / carry eval", "Carry train / reset eval", "Reset train / reset eval")
OUTPUT_NAME = "v24_matched_reset_2023_20261018"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def protocol_digest(protocol: dict[str, Any]) -> str:
    body = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_file(record: dict[str, str], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"Missing or changed {label}: {path}")
    return path


def _import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import frozen helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_workflow(workflow: Path):
    """Verify immutable protocol, dependencies, checkpoint, and frozen helpers."""
    protocol = load_json(workflow / "protocol.json")
    if protocol.get("schema_version") != 1 or protocol.get("protocol_hash") != protocol_digest(protocol):
        raise ValueError("Invalid matched-reset protocol schema or self hash")
    reference_path = checked_file(protocol["reference_protocol"], "reference protocol")
    reference = load_json(reference_path)
    if reference.get("protocol_hash") != protocol_digest(reference):
        raise ValueError("Reference protocol self hash changed")
    if reference["protocol_hash"] != protocol.get("reference_protocol_hash"):
        raise ValueError("Reference protocol identity mismatch")
    campaign = Path(protocol["reference_campaign"])
    if not campaign.is_absolute() or reference_path.resolve() != (campaign / "protocol.json").resolve():
        raise ValueError("Reference campaign does not contain reference protocol")
    if (protocol.get("model_id") != "R_matched" or protocol.get("target_steps") != 40
            or protocol.get("anchor_shard_count") != 8):
        raise ValueError("Matched-reset protocol requires R_matched, 40 leads, and eight shards")
    for key in ("initialization_times", "bootstrap", "baseline", "forecast"):
        if protocol.get(key) != reference.get(key):
            raise ValueError(f"Matched-reset {key} differs from original campaign")
    manifest = checked_file(protocol["initialization_manifest"], "initialization manifest")
    original_manifest = checked_file(reference["initialization_manifest"], "reference initialization manifest")
    if manifest.read_bytes() != original_manifest.read_bytes():
        raise ValueError("Initialization manifest differs from reference")
    checked_file(protocol["checkpoint"], "matched-reset checkpoint")
    source_files = protocol.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("Missing extension source_files")
    for filename, digest in source_files.items():
        checked_file({"path": filename, "sha256": digest}, "extension source")
    source_hashes_path = checked_file(reference["source_hashes"], "reference frozen-source manifest")
    if source_hashes_path.resolve() != (campaign / "source_hashes.json").resolve():
        raise ValueError("Reference frozen-source manifest is outside original campaign")
    provenance = protocol.get("training_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("Missing matched-reset training provenance records")
    for record in provenance:
        checked_file(record, "matched-reset training provenance")
    source_hashes = load_json(source_hashes_path)
    source_root = Path(reference["source_root"])
    if not source_root.is_absolute() or source_root.resolve() != (campaign / "source").resolve():
        raise ValueError("Reference source root does not belong to original campaign")
    for relative, digest in source_hashes.items():
        path = source_root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or sha256(path) != digest:
            raise ValueError(f"Original frozen source changed: {relative}")
    # Import exact functions by path. Explicitly bind the frozen numerical policy
    # too, so pre-existing imports in a caller cannot select workspace policy.
    sys.path.insert(0, str(source_root))
    helpers = _import_file(source_root / "scripts/analyze_models/merge_v24_2023_eval.py", "_matched_reset_original_merge")
    numerical = _import_file(source_root / "scripts/experiments/numerical_policy.py", "_matched_reset_original_numerical")
    helpers.numerical_comparison = numerical.numerical_comparison
    helpers.validate_protocol(reference)
    return protocol, reference, helpers, sha256(source_hashes_path)


def _finite_leads(values, label: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (array.shape != (40,) or not np.all(np.isfinite(array))
            or np.any(array <= 0 if positive else array < 0)):
        raise ValueError(f"Invalid or incomplete {label}")
    return array


def validate_shard(payload, *, model_id, shard_index, protocol, reference, helpers,
                   source_digest, canonical_metadata=None):
    """Validate one complete shard; return ordered records and physical metadata."""
    times = reference["initialization_times"]
    identity = next((m for m in reference["models"] if m["model_id"] == model_id), None)
    new = model_id == "R_matched"
    expected = {
        "model_id": model_id,
        "protocol_hash": protocol["protocol_hash"] if new else reference["protocol_hash"],
        "checkpoint_sha256": protocol["checkpoint"]["sha256"] if new else identity["checkpoint_sha256"],
        "baseline_checkpoint_sha256": reference["baseline"]["checkpoint_sha256"],
        "initialization_manifest_sha256": reference["initialization_manifest_sha256"],
        "global_initialization_times": times, "target_steps": 40,
        "anchor_shard_count": 8, "anchor_shard_index": shard_index,
        "evaluation_status": "complete", "metrics_complete": True,
        "rs_reset_every_step": model_id != "M1", "eval_mode": "cold_full",
        "residual_state_init_resolved": "zero", "warmup_steps": 0,
        "eval_feedback": "full", "baseline_branch": "pure_baseline_self_rollout",
    }
    if new:
        expected.update(reference_protocol_hash=reference["protocol_hash"],
                        training_state_policy="reset_every_anchor", evaluator_source_digest=source_digest)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{model_id} shard {shard_index}: unexpected {key}")
    records = payload.get("original_graphcast_loss_per_initialization")
    expected_times = times[shard_index::8]
    if (not isinstance(records, list) or [r.get("initialization_time") for r in records] != expected_times
            or payload.get("evaluated_samples") != len(expected_times)):
        raise ValueError(f"{model_id} shard {shard_index}: missing, duplicate, or misplaced initializations")
    metadata = payload.get("variable_mse_metadata")
    if not isinstance(metadata, dict) or set(metadata) != set(helpers.VARIABLES):
        raise ValueError("Incomplete variable_mse_metadata")
    for variable, item in metadata.items():
        if (not isinstance(item.get("units"), str) or not item["units"]
                or item.get("spatial_weighting") != "normalized_cosine_latitude"
                or item.get("pressure_pooling") != ("surface" if variable in helpers.VARIABLES[:5] else "equal_levels")):
            raise ValueError(f"Invalid units or variable MSE semantics: {variable}")
    if canonical_metadata is not None and metadata != canonical_metadata:
        raise ValueError("Variable units/metadata differ between paired policies")
    leads = reference["lead_hours"]
    for ordinal, record in zip(range(shard_index, len(times), 8), records):
        stamp = helpers._utc(times[ordinal])
        expected_record = {
            "ordinal": ordinal, "initialization_id": stamp.strftime("%Y%m%dT%H%M%SZ"),
            "initialization_year": 2023,
            "input_times": [helpers._iso(stamp - timedelta(hours=6)), times[ordinal]],
            "valid_times": [helpers._iso(stamp + timedelta(hours=lead)) for lead in leads],
            "valid_years": [(stamp + timedelta(hours=lead)).year for lead in leads],
        }
        for field, value in expected_record.items():
            if record.get(field) != value:
                raise ValueError(f"{model_id}/{times[ordinal]}: invalid {field}")
        if "lead_hours" in record and record["lead_hours"] != leads:
            raise ValueError("Record lead hours differ from protocol")
        for branch in ("baseline", "full"):
            _finite_leads(record.get(f"{branch}_per_step"), f"{model_id}/{branch} loss", positive=branch == "baseline")
        variables = record.get("variable_mse")
        if not isinstance(variables, dict) or set(variables) != set(helpers.VARIABLES):
            raise ValueError("Missing per-initialization physical variable MSE")
        for variable in helpers.VARIABLES:
            for branch in ("baseline", "full"):
                _finite_leads(variables[variable].get(f"{branch}_per_step"), f"{variable}/{branch} MSE")
    aggregate = payload.get("original_graphcast_loss", {})
    for branch in ("baseline", "full"):
        pooled = np.mean([r[f"{branch}_per_step"] for r in records], axis=0)
        saved = _finite_leads(aggregate.get(f"{branch}_per_step"), f"aggregate {branch} loss")
        rollout = aggregate.get(f"{branch}_rollout")
        if (not np.allclose(pooled, saved, rtol=2e-6, atol=0)
                or not isinstance(rollout, (int, float)) or not np.isfinite(rollout)
                or not np.isclose(pooled.mean(), rollout, rtol=2e-6, atol=0)):
            raise ValueError(f"Per-initialization and aggregate {branch} losses disagree")
    aggregates = payload.get("per_variable_per_step", {})
    if set(aggregates) != set(helpers.VARIABLES):
        raise ValueError("Missing aggregate variable RMSE")
    for variable in helpers.VARIABLES:
        for branch in ("baseline", "full"):
            pooled = np.sqrt(np.mean([r["variable_mse"][variable][f"{branch}_per_step"] for r in records], axis=0))
            saved = _finite_leads(aggregates[variable].get(f"rmse_{branch}"), f"aggregate {variable}/{branch} RMSE")
            if not np.allclose(pooled, saved, rtol=2e-6, atol=0):
                raise ValueError(f"Per-initialization MSE and aggregate RMSE disagree for {variable}/{branch}")
    return records, metadata


def gather(workflow, protocol, reference, helpers, source_digest):
    times = reference["initialization_times"]
    baseline = np.full((3, len(times), 40), np.nan)
    full = np.full_like(baseline, np.nan)
    sources, metadata = [], None
    record_identity: dict[str, dict[str, Any]] = {}
    for model_index, model_id in enumerate(MODEL_IDS):
        directory = (workflow if model_id == "R_matched" else Path(protocol["reference_campaign"])) / "shards"
        expected = {directory / f"{model_id}_{index:02d}.json" for index in range(8)}
        # Retained unpublished/partial recovery artifacts are not final shards.
        # M1_reset files also share M1's prefix. Accept only its numeric finals.
        actual = {p for p in directory.glob(f"{model_id}_*.json")
                  if re.fullmatch(re.escape(model_id) + r"_\d+\.json", p.name)}
        if actual != expected:
            raise ValueError(f"Missing or unexpected shards for {model_id}: expected eight complete numeric shards")
        for index, path in enumerate(sorted(expected)):
            payload = load_json(path)
            records, metadata = validate_shard(payload, model_id=model_id, shard_index=index,
                protocol=protocol, reference=reference, helpers=helpers, source_digest=source_digest,
                canonical_metadata=metadata)
            for record in records:
                ordinal = record["ordinal"]
                if np.isfinite(baseline[model_index, ordinal]).any():
                    raise ValueError(f"Duplicate initialization for {model_id}")
                baseline[model_index, ordinal] = record["baseline_per_step"]
                full[model_index, ordinal] = record["full_per_step"]
                identity = {k: record[k] for k in ("initialization_time", "input_times", "valid_times", "dataset_index") if k in record}
                for optional in ("truth_sha256", "truth_hashes", "truth_units"):
                    if optional in record:
                        identity[optional] = record[optional]
                stamp = record["initialization_time"]
                if model_index == 0:
                    record_identity[stamp] = identity
                else:
                    canonical = record_identity[stamp]
                    for key in set(identity) & set(canonical):
                        if identity[key] != canonical[key]:
                            raise ValueError(f"Paired truth/input identity mismatch for {stamp}: {key}")
            sources.append(dict(path=str(path.resolve()), sha256=sha256(path), model_id=model_id))
    if not np.all(np.isfinite([baseline, full])):
        raise ValueError("Incomplete policy × initialization × lead coverage")
    cfg = reference["baseline"]
    repeatability = {
        model: helpers.numerical_comparison(baseline[i], baseline[0], rtol=cfg["repeatability_rtol"],
            atol=cfg["repeatability_atol"], label=f"Repeated baseline drift: {model}", advisory=True)
        for i, model in enumerate(MODEL_IDS)
    }
    return baseline, full, dict(sources=sources, variable_mse_metadata=metadata,
        baseline_repeatability=repeatability,
        numerical_status="warning" if any(not r["passed"] for r in repeatability.values()) else "passed")


def analysis(protocol, reference, helpers, baseline, full, *, block_starts=5,
             segmentation="annual", eligible=None, analysis_id="primary"):
    """Use original paired resampling helpers, keeping the original cadence grid."""
    times = reference["initialization_times"]
    if baseline.shape != full.shape or full.shape != (3, len(times), 40):
        raise ValueError("Loss arrays do not match three policies and full initialization/lead coverage")
    if not np.all(np.isfinite([baseline, full])) or np.any(baseline <= 0) or np.any(full < 0):
        raise ValueError("Invalid losses")
    settings = reference["bootstrap"]
    mask = np.ones(len(times), bool) if eligible is None else np.asarray(eligible, bool)
    if mask.shape != (len(times),) or not mask.any():
        raise ValueError("Invalid eligibility mask")
    counts = helpers.cadence_block_counts(times, stride_days=reference["initialization_stride_days"],
        block_starts=block_starts, segmentation=segmentation,
        replicates=settings["replicates"], seed=settings["seed"])
    counts *= mask[None, :]
    point_counts = mask.astype(float)[None, :]
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("Empty bootstrap draw after exposure mask")
    point = helpers.loss_improvement(baseline[0], full, point_counts)[0]
    draws = helpers.loss_improvement(baseline[0], full, counts)
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    contrasts = []
    for first, second, role in ((0, 2, "planned_primary"), (1, 2, "descriptive"), (0, 1, "descriptive_existing_control")):
        effect = point[first] - point[second]
        pair_draws = draws[:, first] - draws[:, second]
        lo, hi = np.quantile(pair_draws, [.025, .975], axis=0)
        contrasts.append(dict(contrast_id=f"{MODEL_IDS[first]}_vs_{MODEL_IDS[second]}",
            first_arm=MODEL_IDS[first], second_arm=MODEL_IDS[second], rollout_role=role,
            day10_role="planned_secondary" if (first, second) == (0, 2) else "descriptive",
            effect_units="percentage_points_of_canonical_GraphCast_loss_reduction",
            effect=effect.tolist(), pointwise_ci95_lower=lo.tolist(), pointwise_ci95_upper=hi.tolist()))
    own_point = np.stack([helpers.loss_improvement(baseline[i], full[i:i+1], point_counts)[0, 0] for i in range(3)])
    own_draws = np.stack([helpers.loss_improvement(baseline[i], full[i:i+1], counts)[:, 0] for i in range(3)], axis=1)
    own_lo, own_hi = np.quantile(own_draws, [.025, .975], axis=0)
    own_pair_draws = own_draws[:, 0] - own_draws[:, 2]
    own_pair_lo, own_pair_hi = np.quantile(own_pair_draws, [.025, .975], axis=0)
    return dict(analysis_id=analysis_id, method="paired non-wrapping moving-block percentile bootstrap",
        block_starts=block_starts, block_elapsed_days=(block_starts-1)*reference["initialization_stride_days"],
        segmentation=segmentation, replicates=settings["replicates"], seed=settings["seed"],
        initialization_count=int(mask.sum()), resampling_grid_initialization_count=len(times),
        excluded_initialization_times=[t for t, keep in zip(times, mask) if not keep],
        bootstrap_eligible_count_min=int(counts.sum(1).min()), bootstrap_eligible_count_max=int(counts.sum(1).max()),
        canonical_baseline_rollout=float(baseline[0, mask].mean()),
        canonical_baseline_per_step=baseline[0, mask].mean(0).tolist(),
        model_rollout=full[:, mask].mean((1, 2)).tolist(), model_per_step=full[:, mask].mean(1).tolist(),
        improvement_pct=point.tolist(), pointwise_ci95_lower=lower.tolist(), pointwise_ci95_upper=upper.tolist(),
        contrasts=contrasts, own_baseline_sensitivity=dict(scope="Descriptive numerical denominator sensitivity; canonical endpoints remain primary.",
            improvement_pct=own_point.tolist(), pointwise_ci95_lower=own_lo.tolist(), pointwise_ci95_upper=own_hi.tolist(),
            M1_vs_R_matched_effect=(own_point[0]-own_point[2]).tolist(),
            M1_vs_R_matched_ci95_lower=own_pair_lo.tolist(), M1_vs_R_matched_ci95_upper=own_pair_hi.tolist()))


def analyze(protocol, reference, helpers, baseline, full):
    primary = analysis(protocol, reference, helpers, baseline, full)
    sensitivities = [analysis(protocol, reference, helpers, baseline, full, block_starts=b,
        analysis_id=f"annual_{b}_starts") for b in (3, 7)]
    sensitivities.append(analysis(protocol, reference, helpers, baseline, full,
        segmentation="calendar_quarters", analysis_id="quarterly_5_starts"))
    exposed = set(reference["prior_exposure"]["sampled_initialization_times"])
    sensitivities.append(analysis(protocol, reference, helpers, baseline, full,
        eligible=[t not in exposed for t in reference["initialization_times"]], analysis_id="exclude_prior_pilot_starts"))
    return dict(schema_version=1, protocol_hash=protocol["protocol_hash"],
        reference_protocol_hash=reference["protocol_hash"], model_ids=list(MODEL_IDS), policy_labels=list(POLICY_LABELS),
        initialization_count=len(reference["initialization_times"]), lead_hours=reference["lead_hours"],
        metric="Exact original GraphCast weighted loss reduction: 100*(1-sum(model loss)/sum(canonical M1 GraphCast loss))",
        primary_endpoint="M1_vs_R_matched_rollout", secondary_endpoint="M1_vs_R_matched_day10",
        analysis_scope="Separate planned extension; does not inherit the original evaluation's prespecified confirmatory family.",
        training_source_limit=protocol.get("training_source_limit", ""),
        primary=primary, sensitivities=sensitivities, prior_exposure=reference["prior_exposure"],
        checkpoints={"M1": next(m["checkpoint"] for m in reference["models"] if m["model_id"] == "M1"),
                     "R_matched": protocol["checkpoint"]},
        limitations=[
            "Uncertainty conditions on these fixed checkpoints and the 2023 every-third-day sample; training-seed and between-year variation are unmeasured.",
            "The training-policy comparison is a separate planned analysis added after the original M1 evaluation, not part of its confirmatory endpoint family.",
            "Leadwise and contrast intervals are pointwise; no simultaneous significance across leads or contrast families is claimed.",
            "Non-wrapping blocks underweight boundary dates; quarter blocks break cross-quarter dependence.",
            "Excluding the three sampled pilot dates does not undo prior test-year exposure or overlapping forecast windows.",
            "A reset-trained control estimates this training-policy contrast for one training realization; it does not isolate all possible memory mechanisms.",
            "Forecast accuracy alone does not establish preservation of extremes or spatial structure.",
        ])


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty report table")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(workflow, protocol, reference, result, baseline, full):
    report_dir = workflow / "report"
    project = Path(reference["project_root"])
    data_dir = project / "plots/analyze_models/data/resolution_eval" / OUTPUT_NAME
    image_dir = project / "plots/analyze_models/images/resolution_eval" / OUTPUT_NAME
    for directory in (report_dir, data_dir, image_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows = [dict(model_id=model, initialization_time=stamp, lead_hours=lead,
                 canonical_baseline_loss=float(baseline[0, i, k]), paired_baseline_loss=float(baseline[m, i, k]),
                 model_loss=float(full[m, i, k]), protocol_hash=protocol["protocol_hash"])
            for m, model in enumerate(MODEL_IDS) for i, stamp in enumerate(reference["initialization_times"])
            for k, lead in enumerate(reference["lead_hours"])]
    _write_csv(report_dir / "per_initialization_losses.csv", rows)
    _write_csv(data_dir / "per_initialization_losses.csv", rows)
    loss_rows, contrast_rows = [], []
    for item in [result["primary"], *result["sensitivities"]]:
        for m, model in enumerate(MODEL_IDS):
            for endpoint in range(41):
                loss_rows.append(dict(analysis=item["analysis_id"], model_id=model,
                    endpoint="rollout" if endpoint == 0 else "lead", lead_hours=None if endpoint == 0 else reference["lead_hours"][endpoint-1],
                    initialization_count=item["initialization_count"], reduction_pct=item["improvement_pct"][m][endpoint],
                    pointwise_ci95_lower=item["pointwise_ci95_lower"][m][endpoint], pointwise_ci95_upper=item["pointwise_ci95_upper"][m][endpoint]))
        for contrast in item["contrasts"]:
            for endpoint in range(41):
                contrast_rows.append(dict(analysis=item["analysis_id"], contrast_id=contrast["contrast_id"],
                    endpoint="rollout" if endpoint == 0 else "lead", lead_hours=None if endpoint == 0 else reference["lead_hours"][endpoint-1],
                    effect_percentage_points=contrast["effect"][endpoint], pointwise_ci95_lower=contrast["pointwise_ci95_lower"][endpoint],
                    pointwise_ci95_upper=contrast["pointwise_ci95_upper"][endpoint]))
    _write_csv(data_dir / "loss_summary.csv", loss_rows)
    _write_csv(data_dir / "paired_contrasts.csv", contrast_rows)
    result["outputs"] = dict(report_dir=str(report_dir), data_dir=str(data_dir), image_dir=str(image_dir))
    (report_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    primary = result["primary"]
    lines = ["# Matched reset-training control: 2023 verification", "",
        "122 starts at 00 UTC every third day, January 1–December 30, 2023; 40 six-hour leads through day 10. Late-December verification uses January 2024 truth.", "",
        "M1 and M1_reset reuse the original frozen forecast files. The matched reset-trained checkpoint is evaluated from zero state, resetting its memory before every forecast step, with full forecast feedback.", "",
        f"Extension protocol: `{protocol['protocol_hash']}`. Reference protocol: `{reference['protocol_hash']}`.", "",
        "Every score uses exact original GraphCast loss, pooled over starts and leads before taking a ratio against the canonical M1 execution of GraphCast. No RMSE reconstruction or averaging of casewise percentages is used.", "",
        "| Training / evaluation policy | Rollout loss reduction (pointwise 95% CI) | Day-10 loss reduction (pointwise 95% CI) |",
        "| --- | ---: | ---: |"]
    for m, label in enumerate(POLICY_LABELS):
        cells = [f"{primary['improvement_pct'][m][k]:.3f}% [{primary['pointwise_ci95_lower'][m][k]:.3f}, {primary['pointwise_ci95_upper'][m][k]:.3f}]" for k in (0, 40)]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    lines += ["", "The planned primary contrast is carry train / carry eval versus matched reset train / reset eval, measured in percentage points of canonical GraphCast-referenced reduction. Day 10 is secondary. Positive effects favor the first arm. This extension does not inherit the original evaluation's prespecified confirmatory family; all intervals below are pointwise.", "",
        "| Paired contrast | Rollout effect, percentage points (95% CI) | Day-10 effect, percentage points (95% CI) |", "| --- | ---: | ---: |"]
    for contrast in primary["contrasts"]:
        cells = [f"{contrast['effect'][k]:+.3f} [{contrast['pointwise_ci95_lower'][k]:+.3f}, {contrast['pointwise_ci95_upper'][k]:+.3f}]" for k in (0, 40)]
        lines.append(f"| {contrast['contrast_id']} | {cells[0]} | {cells[1]} |")
    lines += ["", "Paired non-wrapping bootstrap: 10,000 replicates, seed 20260916, five consecutive initialization starts per block (12 elapsed days), resampling complete paired leads and policies.", "",
        "| Sensitivity | Eligible starts | Carry/carry versus matched reset rollout effect (95% CI), percentage points |", "| --- | ---: | ---: |"]
    for item in result["sensitivities"]:
        c = item["contrasts"][0]
        lines.append(f"| {item['analysis_id']} | {item['initialization_count']} | {c['effect'][0]:+.3f} [{c['pointwise_ci95_lower'][0]:+.3f}, {c['pointwise_ci95_upper'][0]:+.3f}] |")
    lines += ["", "Pilot exclusion keeps the original regular resampling grid and masks the three exposed starts after resampling; remaining cases are not compressed into artificial adjacency.", "",
        f"Numerical baseline diagnostics: **{result['audit']['numerical_status']}**. Finite repeated-baseline differences are advisory and do not change canonical endpoints.", "",
        "| Model | Pooled repeated GC drift | Largest start/lead relative drift | Own-baseline rollout reduction |", "| --- | ---: | ---: | ---: |"]
    for m, model in enumerate(MODEL_IDS):
        row = result["audit"]["baseline_repeatability"][model]
        own = primary["own_baseline_sensitivity"]["improvement_pct"][m][0]
        lines.append(f"| {model} | {100*row['pooled_relative_difference']:+.4f}% | {100*row['max_relative_difference']:.4f}% | {own:.3f}% |")
    own = primary["own_baseline_sensitivity"]
    lines += ["", f"Using each policy's own repeated GC denominator, the descriptive carry/carry versus matched reset rollout contrast is {own['M1_vs_R_matched_effect'][0]:+.3f} percentage points [{own['M1_vs_R_matched_ci95_lower'][0]:+.3f}, {own['M1_vs_R_matched_ci95_upper'][0]:+.3f}].", "",
        "Checkpoint provenance:", "", f"- Carry-trained M1: `{result['checkpoints']['M1']}`.",
        f"- Matched reset-trained: `{protocol['checkpoint']['path']}`; SHA256 `{protocol['checkpoint']['sha256']}`.", "", "Limitations:", ""]
    lines += [f"- {line}" for line in result["limitations"]]
    if result.get("training_source_limit"):
        lines += [f"- Training source provenance: {result['training_source_limit']}"]
    lines += ["", f"Source audit, full paired losses, bootstrap details, and figure paths: `{report_dir / 'summary.json'}`.", ""]
    (report_dir / "report.md").write_text("\n".join(lines))
    plot_figures(result, reference, image_dir)
    return report_dir / "report.md"


def plot_figures(result, reference, image_dir):
    """Render the saved analysis without repeating inference or bootstrap draws."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    primary = result["primary"]
    days = np.asarray(reference["lead_hours"]) / 24
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    for m, (label, color) in enumerate(zip(POLICY_LABELS, ("#245c9b", "#d18524", "#267d55"))):
        ax.plot(days, primary["improvement_pct"][m][1:], label=label, color=color, lw=2)
    ax.axhline(0, color="0.35", lw=.8)
    ax.set(xlabel="Forecast lead (days)", ylabel="Exact GraphCast loss reduction (%)", xlim=(.25, 10))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.text(.5, .01, "Canonical M1 GraphCast baseline.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(image_dir / f"exact_loss_reduction.{suffix}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    for contrast, label, color in zip(primary["contrasts"][:2],
            ("Carry/carry − matched reset/reset", "Carry/reset − matched reset/reset"), ("#245c9b", "#d18524")):
        ax.plot(days, contrast["effect"][1:], label=label, color=color, lw=2)
    ax.axhline(0, color="0.35", lw=.8)
    ax.set(xlabel="Forecast lead (days)", ylabel="Difference in loss reduction (percentage points)", xlim=(.25, 10))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.text(.5, .01, "Positive favors first arm.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(image_dir / f"paired_memory_contrasts.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    args = parser.parse_args()
    workflow = args.workflow.resolve()
    protocol, reference, helpers, source_digest = load_workflow(workflow)
    baseline, full, audit = gather(workflow, protocol, reference, helpers, source_digest)
    result = analyze(protocol, reference, helpers, baseline, full)
    result["audit"] = audit
    print(write_outputs(workflow, protocol, reference, result, baseline, full))


if __name__ == "__main__":
    main()
