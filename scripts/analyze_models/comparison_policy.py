"""Explicit user exclusions from scientific comparisons, preserving raw evidence.

This is a presentation policy, never a numerical acceptance or training gate.
Frozen experiment snapshots and sealed historical reports are not modified.
"""
from __future__ import annotations

from pathlib import Path


POLICY = {
    "id": "full_gc_lr1em5_excluded_20261112",
    "date": "2026-11-12",
    "campaign": "res1_fresh_gc_mamba_gc_lambda_20261107",
    "model_family": "full_gc",
    "peak_learning_rate": 1e-5,
    "run_ids": (
        "gcft_s22_l000_lr0", "gcft_s22_l025", "gcft_s22_l050_profile",
        "gcft_s22_l075", "gcft_s22_l100",
    ),
    "reason": "User rejected full-GC peak-LR 1e-5 results for scientific comparisons; retain for provenance/debugging only.",
    "preferred_full_gc_peak_learning_rate": 3e-7,
}
PRESENTATION_SUFFIX = "_eligible_20261112_v2"


def _path(value):
    return value.get("path") if isinstance(value, dict) else value


def exclusion(model, campaign=None):
    """Match actual training identity, not lambda, a score, or a logical label."""
    if model.get("model_family", model.get("family")) != POLICY["model_family"]:
        return None
    paths = [_path(model.get(key)) for key in ("config", "run_config", "checkpoint")]
    identities = []
    for value in paths:
        if not value:
            continue
        parts = Path(value).parts
        if POLICY["campaign"] not in parts:
            continue
        index = parts.index(POLICY["campaign"])
        tail = parts[index + 1:]
        if len(tail) >= 2 and tail[0] in ("runs", "configs", "calibration_evaluation"):
            identities.append(Path(tail[1]).stem if tail[0] == "configs" else tail[1])
    if campaign and Path(campaign).name == POLICY["campaign"]:
        identities.append(model.get("training_run_id", model.get("id")))
    matches = set(identities).intersection(POLICY["run_ids"])
    if not matches:
        return None
    # A changed LR label or continuation does not rehabilitate rejected weights.
    # Fresh 3e-7 experiments must have a distinct run/campaign identity.
    return {"model_id": model.get("logical_id", model.get("id", model.get("model"))),
            "training_run_id": sorted(matches)[0], "policy_id": POLICY["id"],
            "campaign": POLICY["campaign"], "peak_learning_rate": POLICY["peak_learning_rate"],
            "reason": POLICY["reason"]}


def partition_models(models, campaign=None):
    included, excluded = [], []
    for model in models:
        record = exclusion(model, campaign)
        if record is None:
            included.append(model)
        else:
            excluded.append(record)
    return included, excluded


def presentation_directory(path, excluded):
    """Never overwrite an archived report with a filtered presentation."""
    path = Path(path)
    if excluded and not path.name.endswith(PRESENTATION_SUFFIX):
        return path.with_name(path.name + PRESENTATION_SUFFIX)
    return path
