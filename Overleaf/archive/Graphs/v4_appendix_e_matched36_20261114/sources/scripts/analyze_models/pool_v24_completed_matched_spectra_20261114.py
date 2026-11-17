#!/usr/bin/env python3
"""Pool a frozen completed-shard intersection without forecasting or fitting.

Only checked shards from the full matched-reset spectral campaign are used.
The independent four-date preview is never added. Existing campaign reports,
manifests, and plotting outputs are not modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.experiments import run_v24_matched_reset_spectra as full
from scripts.experiments.frozen_runtime import verify_source
from scripts.experiments.run_v24_mgda import check, pinned, read, seal, unseal
from src.analysis.smoothing_observer import atomic_npz
from src.analysis.spherical_smoothing import json_digest

RESET_ROOT = ROOT / "artifacts/evaluations/v24_matched_reset_spectra_20261113"
REFERENCE_ROOT = ROOT / "artifacts/evaluations/v24_optimal_smoothing_v4_20261112"
OUTPUT_ROOT = ROOT / "artifacts/evaluations/v24_matched_spectra_completed_20261114"
STATUS = "interim_matched_cohort"
POWER_KEYS = ("prediction_power", "cross_power", "error_power")
REFERENCE_SHAPES = {key: (40, 2, 3, 83, 180) for key in POWER_KEYS}
REFERENCE_SHAPES.update(truth_power=(40, 83, 180), exact_gc_loss=(40, 2, 3),
                        spectral_weighted_gc_loss=(40, 2, 3))
RESET_SHAPES = {key: (40, 2, 83, 180) for key in POWER_KEYS}
RESET_SHAPES.update(truth_power=(40, 83, 180), exact_gc_loss=(40, 2),
                    spectral_weighted_gc_loss=(40, 2), normalized_grid_mse=(40, 2, 83))


def save_pooled(path, values, metadata):
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["metadata"])) != metadata:
                raise ValueError(f"Existing output metadata differs: {path}")
            for key, value in values.items():
                np.testing.assert_array_equal(archive[key], value)
    else:
        atomic_npz(path, values, metadata)


def freeze_cohort(output):
    """Snapshot checked completion before any potentially lengthy validation."""
    cohort_path = output / "cohort.json"
    if cohort_path.exists():
        cohort = unseal(cohort_path)
        for record in cohort["sources"] + cohort["completed_checks"]:
            check(record)
        return cohort
    checked_paths = tuple(sorted((RESET_ROOT / "checks").glob("evaluate_[0-9][0-9].json")))
    if not checked_paths:
        raise ValueError("No checked reset shards have completed")
    snapshot_time = datetime.now(timezone.utc).isoformat()
    manifest = unseal(RESET_ROOT / "manifest.json")
    interim_path = REFERENCE_ROOT / "interim_20261113/report.json"
    interim = unseal(interim_path)
    by_reference = {Path(record["path"]).stem: record for record in interim["statistics"]}
    if len(by_reference) != len(interim["statistics"]) or len(by_reference) != 92:
        raise ValueError("Expected 92 unique pinned GC/M1 reference starts")
    completed, reset_records, protocol_by_id, completed_tasks = [], {}, {}, []
    for path in checked_paths:
        proof = unseal(path)
        task = int(path.stem.rsplit("_", 1)[1])
        shard = manifest["shards"]["evaluate"][task]
        check(shard["manifest"])
        ids = full.original.expected_ids(read(shard["manifest"]["path"]))
        if (proof["status"] != "complete" or proof["campaign_hash"] != manifest["record_hash"]
                or proof["count"] != shard["count"] or proof["initialization_ids"] != ids
                or proof["reset_state_every_step"] is not True
                or sorted(Path(record["path"]).stem for record in proof["statistics"]) != ids):
            raise ValueError(f"Invalid completed shard proof: {path}")
        check(proof["metrics"])
        completed.append(pinned(path))
        completed_tasks.append(task)
        for record in proof["statistics"]:
            stamp = Path(record["path"]).stem
            if stamp in reset_records or not Path(record["path"]).resolve().is_relative_to(RESET_ROOT):
                raise ValueError("Duplicate reset sample or source outside full campaign")
            reset_records[stamp] = record
            protocol_by_id[stamp] = shard["manifest"]["sha256"]
    ids = [stamp for stamp in manifest["initialization_ids"] if stamp in reset_records and stamp in by_reference]
    if not ids or len(ids) != len(reset_records):
        raise ValueError("Completed reset starts do not all have matching reference statistics")
    cohort = seal(cohort_path, {
        "format": "v24_completed_matched_spectral_cohort_v1", "status": STATUS,
        "snapshot_time_utc": snapshot_time, "count": len(ids), "planned_count": 92,
        "initialization_ids": ids, "first_initialization": ids[0], "last_initialization": ids[-1],
        "completed_tasks": completed_tasks, "completed_checks": completed,
        "reference_statistics": [by_reference[stamp] for stamp in ids],
        "reset_statistics": [reset_records[stamp] for stamp in ids],
        "reset_protocol_hash_by_id": protocol_by_id,
        "source_campaign_hash": manifest["record_hash"], "reference_campaign_hash": interim["campaign_hash"],
        "sources": [pinned(RESET_ROOT / "manifest.json"), pinned(REFERENCE_ROOT / "manifest.json"),
                    pinned(interim_path), pinned(REFERENCE_ROOT / "filters.json")],
        "scope": "Intersection frozen from completed checked full-campaign shards; no preview-four padding; not annual",
        "uncertainty": "Fixed-checkpoint point estimates; no new bootstrap or training-seed variation"})
    return cohort


def validate_campaigns(cohort):
    reset_manifest = full.verify(RESET_ROOT, frozen=False)
    reference_manifest = unseal(REFERENCE_ROOT / "manifest.json")
    filters = unseal(REFERENCE_ROOT / "filters.json")
    reference_report = unseal(REFERENCE_ROOT / "interim_20261113/report.json")
    check(reference_report["filter"])
    for record in reference_manifest["inputs"]:
        check(record)
    for record in (reference_manifest["checkpoint"], reset_manifest["checkpoint"]):
        check(record)
    source = verify_source(reference_manifest["source_root"])
    if (source["source_digest"] != reference_manifest["source_digest"]
            or filters["campaign_hash"] != reference_manifest["record_hash"]
            or filters["lead_hours"] != full.LEADS or filters["lmax"] != 179
            or filters["development_count"] != 32 or filters["fit_year"] != 2022
            or set(filters["branches"]) != {"GC", "M1"} or len(filters["layout"]["channels"]) != 83
            or reset_manifest["architecture"] != reference_manifest["architecture"]
            or reset_manifest["baseline"] != reference_manifest["baseline"]
            or cohort["source_campaign_hash"] != reset_manifest["record_hash"]
            or cohort["reference_campaign_hash"] != reference_manifest["record_hash"]):
        raise ValueError("Campaign, source, architecture, baseline or filter mismatch")
    return reference_manifest, reset_manifest, filters


def checked_arrays(record, manifest, expected_shapes, layout, *, reset, cohort, filter_digest):
    check(record)
    path = Path(record["path"])
    # Reuse the saved-statistics loader's campaign/date/status/finite checks.
    arrays, meta = full.original.load_statistics(path, manifest, "evaluate")
    expected_branches = ["GC", "R_matched"] if reset else ["GC", "M1"]
    expected_schema = "v24_raw_spectral_statistics_v1" if reset else "v24_optimal_filter_statistics_v1"
    if (meta["schema"] != expected_schema or meta["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]
            or meta["source_digest"] != manifest["source_digest"] or meta["layout"] != layout
            or meta["branches"] != expected_branches or meta["lead_hours"] != full.LEADS or meta["lmax"] != 179):
        raise ValueError(f"Saved spectral provenance differs: {path}")
    if reset:
        if (meta["reset_state_every_step"] is not True or meta["variants"] != ["raw"]
                or meta["protocol_hash"] != cohort["reset_protocol_hash_by_id"][path.stem]
                or set(arrays) != set(expected_shapes)):
            raise ValueError("Unexpected reset policy, shard protocol, variants or array keys")
    elif meta["filter_digest"] != filter_digest or meta["variants"] != ["raw", "projected", "optimal"]:
        raise ValueError("GC/M1 filter or variant mismatch")
    selected = {}
    for key, shape in expected_shapes.items():
        value = arrays[key]
        if value.shape != shape or not np.isfinite(value).all() or (key != "cross_power" and np.any(value < 0)):
            raise ValueError(f"Invalid saved shape or values: {path}:{key}")
        selected[key] = value
    return selected


def moment_closure(values, *, reset):
    truth = values["truth_power"][:, None] if reset else values["truth_power"][:, None, None]
    scale = values["prediction_power"] + truth
    delta = scale - 2 * values["cross_power"] - values["error_power"]
    maximum = float(np.max(np.abs(delta) / np.maximum(scale, np.finfo(float).tiny)))
    if maximum > 1e-10:
        raise ValueError("Corrupt spectral moment identity")
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output in (RESET_ROOT, REFERENCE_ROOT) or output.is_relative_to(RESET_ROOT) or output.is_relative_to(REFERENCE_ROOT):
        raise ValueError("Output must be separate from source campaigns")
    cohort = freeze_cohort(output)
    ids, count = cohort["initialization_ids"], cohort["count"]
    print(json.dumps({"frozen_count": count, "planned_count": 92, "first": ids[0], "last": ids[-1],
                      "checked_shards": cohort["completed_tasks"]}), flush=True)
    reference_manifest, reset_manifest, filters = validate_campaigns(cohort)
    layout, filter_digest = filters["layout"], json_digest(filters)
    aggregates = [{key: np.zeros(shape, dtype=np.float64) for key, shape in shapes.items()}
                  for shapes in (REFERENCE_SHAPES, RESET_SHAPES)]
    individuals = [[], []]
    closure = [0., 0.]
    truth_max_relative = 0.
    for ordinal, (reference_record, reset_record) in enumerate(zip(cohort["reference_statistics"], cohort["reset_statistics"])):
        pair = [checked_arrays(reference_record, reference_manifest, REFERENCE_SHAPES, layout,
                               reset=False, cohort=cohort, filter_digest=filter_digest),
                checked_arrays(reset_record, reset_manifest, RESET_SHAPES, layout,
                               reset=True, cohort=cohort, filter_digest=filter_digest)]
        np.testing.assert_allclose(pair[0]["truth_power"], pair[1]["truth_power"], rtol=1e-9, atol=1e-12)
        truth_max_relative = max(truth_max_relative, float(np.max(
            np.abs(pair[0]["truth_power"] - pair[1]["truth_power"]) /
            np.maximum(np.abs(pair[0]["truth_power"]), np.finfo(float).tiny))))
        for i, arrays in enumerate(pair):
            closure[i] = max(closure[i], moment_closure(arrays, reset=i == 1))
            for key, values in arrays.items():
                aggregates[i][key] += values
            individuals[i].append(arrays["exact_gc_loss"])
        if (ordinal + 1) % 12 == 0:
            print(f"Validated and pooled {ordinal + 1}/{count} paired dates", flush=True)
    pooled = [{key: values / count for key, values in total.items()} for total in aggregates]
    individuals = [np.asarray(values) for values in individuals]
    canonical = full.canonical_reset_losses({**reset_manifest, "initialization_ids": ids})
    diagnostics = {
        "canonical_reset_metric_drift": full.drift_diagnostic(individuals[1][:, :, 1], canonical, rtol=1e-3,
            label="completed-cohort reset forecast/archived reset checkpoint evaluation"),
        "repeated_gc_metric_drift": full.drift_diagnostic(individuals[1][:, :, 0], individuals[0][:, :, 0, 0],
            rtol=1e-3, label="reset-worker GC/reference GC evaluation"),
        "worker_grid_mse_roundoff": [unseal(record["path"])["grid_mse_roundoff"] for record in cohort["completed_checks"]]}
    common = {"status": STATUS, "count": count, "planned_count": 92, "initialization_ids": ids,
              "lead_hours": full.LEADS, "layout": layout, "lmax": 179,
              "scope": cohort["scope"], "uncertainty": cohort["uncertainty"],
              "cohort": pinned(output / "cohort.json"),
              "checks": {"same_truth_and_dates": True, "truth_power_maximum_relative_difference": truth_max_relative,
                         "spectral_identity_relative_max": closure, "finite_values_retained": True}}
    common_sources = [pinned(output / "cohort.json"), pinned(Path(__file__)),
                      pinned(Path(full.__file__)), pinned(Path(full.original.__file__))]
    ref_metadata = {**common, "format": "v24_completed_matched_reference_pooled_v1",
                    "campaign_hash": reference_manifest["record_hash"], "model_id": "M1",
                    "checkpoint": reference_manifest["checkpoint"],
                    "source_digest": reference_manifest["source_digest"],
                    "branches": ["GC", "M1"], "variants": ["raw", "projected", "optimal"],
                    "filter_digest": filter_digest, "filter": pinned(REFERENCE_ROOT / "filters.json"),
                    "fit_year": 2022, "fit_initializations": 32,
                    "sources": [*common_sources, *cohort["sources"], *cohort["reference_statistics"]]}
    reset_metadata = {**common, "format": "v24_completed_matched_reset_pooled_v1",
                      "campaign_hash": reset_manifest["record_hash"], "model_id": "R_matched",
                      "checkpoint": reset_manifest["checkpoint"], "checkpoint_sha256": full.RESET_SHA256,
                      "source_digest": reset_manifest["source_digest"], "reset_state_every_step": True,
                      "branches": ["GC", "R_matched"], "variants": ["raw"], "filter_fitting": False,
                      "numerical_diagnostics": diagnostics,
                      "sources": [*common_sources, *cohort["sources"], *cohort["completed_checks"], *cohort["reset_statistics"]]}
    summaries = []
    for i, (directory, metadata) in enumerate((("gc_m1_reference", ref_metadata), ("reset", reset_metadata))):
        target = output / directory
        target.mkdir(parents=True, exist_ok=True)
        save_pooled(target / "power_pooled.npz", pooled[i], metadata)
        individual_metadata = {"initialization_ids": ids, "lead_hours": full.LEADS,
                               "branches": metadata["branches"], "variants": metadata["variants"],
                               "axes": "initialization,lead,branch" + (",variant" if i == 0 else "")}
        save_pooled(target / "per_initialization_exact_gc_loss.npz", {"exact_gc_loss": individuals[i]}, individual_metadata)
        rows = [{"model": model, "variant": variant,
                 "exact_gc_rollout_loss": float((pooled[i]["exact_gc_loss"][:, bi, vi] if i == 0
                                                 else pooled[i]["exact_gc_loss"][:, bi]).mean())}
                for bi, model in enumerate(metadata["branches"])
                for vi, variant in enumerate(metadata["variants"])]
        summary = seal(target / "summary.json", {**metadata, "rows": rows,
            "exact_gc_loss_by_lead": pooled[i]["exact_gc_loss"].tolist(),
            "data": {name: pinned(target / name) for name in ("power_pooled.npz", "per_initialization_exact_gc_loss.npz")}})
        summaries.append(summary)
    # Reconfirm the frozen input snapshot after writing, without adopting new shards.
    for record in cohort["sources"] + cohort["completed_checks"]:
        check(record)
    print(json.dumps({"output": str(output), "count": count,
                      "reference_rows": summaries[0]["rows"], "reset_rows": summaries[1]["rows"],
                      "checks": common["checks"], "numerical_diagnostics": diagnostics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
