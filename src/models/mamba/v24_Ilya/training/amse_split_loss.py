"""AMSE component objectives and auditable, shared two-initialization scaling.

No division by truth power or independent objective normalization is used.
The source spherical loss owns all transforms, weights and regularization.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

STAT_FILES = ("stddev_by_level.nc", "mean_by_level.nc", "diffs_stddev_by_level.nc")
PROVENANCE_KEYS = ("baseline_checkpoint_fingerprint", "anchor_manifest_fingerprint",
                   "prepared_metadata_sha256", "stats_sha256")


def build_amse_split_objectives(full_field_loss):
    """Return raw weighted A/C and exact grid GC diagnostics from one transform."""
    def objectives(prediction, truth):
        _, diagnostics = full_field_loss(prediction, truth)
        return jnp.stack((diagnostics["amplitude"], diagnostics["pattern"])), {
            "gc_loss": diagnostics["grid_mse_fp32"],
        }
    return objectives


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def calibration_signature(config):
    """Scientific forward-pass identity, excluding output/optimizer calibration."""
    value = _json(config)
    architecture = dict(value["architecture"])
    architecture.pop("temporal_d_inner", None)
    objective = dict(value["objective"])
    objective.pop("amse_split_calibration", None)
    return {key: value[key] for key in ("data", "sequence", "distributed", "memory")} | {
        "architecture": architecture, "objective": objective,
        "seed": value["optimizer"]["seed"], "precision": value["optimizer"]["precision"],
    }


def _validate_raw(record):
    if not isinstance(record, Mapping) or record.get("version") != 1 or record.get("kind") != "amse_split_raw":
        raise ValueError("AMSE split raw calibration requires version=1 and kind=amse_split_raw")
    value = _json(dict(record))
    batches = value.get("batches")
    if type(batches) is not int or batches <= 0:
        raise ValueError("AMSE split calibration batches must be positive")
    raw = np.asarray(value.get("raw_values"), dtype=np.float64)
    if raw.shape != (batches, 2) or not np.all(np.isfinite(raw) & (raw >= 0)):
        raise ValueError("AMSE split raw_values must be finite nonnegative [batches, 2]")
    if len(value.get("batch_manifest", ())) != batches:
        raise ValueError("AMSE split calibration requires every batch identity")
    for key in (*PROVENANCE_KEYS, "init_from_sha256", "source_config_sha256", "source_config"):
        if key not in value:
            raise ValueError(f"AMSE split calibration missing provenance: {key}")
    for key in ("baseline_checkpoint_fingerprint", "anchor_manifest_fingerprint",
                "prepared_metadata_sha256", "init_from_sha256", "source_config_sha256"):
        digest = value[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"AMSE split calibration invalid SHA256: {key}")
    if set(value["stats_sha256"]) != set(STAT_FILES):
        raise ValueError("AMSE split calibration requires all statistics fingerprints")
    for digest in value["stats_sha256"].values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("AMSE split statistics fingerprints must be SHA256 values")
    source = value["source_config"]
    if source["objective"].get("objective_pair") != "amse_split":
        raise ValueError("AMSE split calibration source has another objective pair")
    calibration_signature(source)
    return value


def make_amse_split_calibration_record(raw_values, batch_manifest, **provenance):
    """Freeze a single forward-only pass; this record cannot train a model."""
    reserved = {"version", "kind", "batches", "raw_values", "batch_manifest"}
    if reserved & provenance.keys():
        raise ValueError("AMSE split provenance cannot replace calibration values")
    return _validate_raw(dict(version=1, kind="amse_split_raw", batches=len(raw_values),
                              raw_values=np.asarray(raw_values).tolist(),
                              batch_manifest=batch_manifest, **provenance))


def combine_amse_split_calibrations(records):
    """Pool matched di16/di32 forward runs; freeze one common scalar."""
    values = [_validate_raw(record) for record in records]
    if len(values) != 2:
        raise ValueError("AMSE split pooling requires exactly two initializations")
    values.sort(key=lambda record: record["source_config"]["architecture"]["temporal_d_inner"])
    widths = [record["source_config"]["architecture"]["temporal_d_inner"] for record in values]
    if widths != [16, 32]:
        raise ValueError("AMSE split pooling requires di16 and di32")
    first, second = values
    if first["init_from_sha256"] == second["init_from_sha256"]:
        raise ValueError("AMSE split pooling requires two distinct checkpoint identities")
    for key in ("batches", "batch_manifest", *PROVENANCE_KEYS):
        if first[key] != second[key]:
            raise ValueError(f"AMSE split calibration arms differ: {key}")
    if calibration_signature(first["source_config"]) != calibration_signature(second["source_config"]):
        raise ValueError("AMSE split calibration forward configurations differ beyond width")
    scale = float(np.mean([np.sum(record["raw_values"], axis=1) for record in values]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("AMSE split common_scale must be finite and positive")
    return dict(version=1, objective_pair="amse_split", common_scale=scale,
                batches_per_initialization=first["batches"], batch_manifest=first["batch_manifest"],
                provenance={key: first[key] for key in PROVENANCE_KEYS}, initializations=values)


def validate_amse_split_calibration(calibration):
    if not isinstance(calibration, Mapping) or calibration.get("version") != 1 or calibration.get("objective_pair") != "amse_split":
        raise ValueError("AMSE split calibration requires version=1 and objective_pair=amse_split")
    value = _json(dict(calibration))
    scale = value.get("common_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("AMSE split common_scale must be finite and positive")
    expected = combine_amse_split_calibrations(value.get("initializations", ()))
    if value != expected:
        raise ValueError("AMSE split common scale/provenance differs from pooled raw evidence")
    return value


def calibration_fingerprint(calibration):
    payload = validate_amse_split_calibration(calibration)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
