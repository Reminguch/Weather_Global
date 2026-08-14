"""Native v23 checkpoints plus self-contained legacy warm-start parsing."""

from __future__ import annotations

import pickle
import hashlib
import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .config import ARCHITECTURE_ID, SCHEMA_VERSION


CHECKPOINT_FORMAT = "legacy_v22_pickle"
TRAINING_CHECKPOINT_FORMAT = "v23_Ilya_training_pickle"
SWA_CHECKPOINT_FORMAT = "v23_Ilya_swa_pickle"
LEGACY_ARCHITECTURE_IDS = {None, "v22_final", ARCHITECTURE_ID}
WARM_START_FORMATS = {
    CHECKPOINT_FORMAT,
    "v22_final_training_pickle",
    "v22_final_swa_pickle",
    TRAINING_CHECKPOINT_FORMAT,
    SWA_CHECKPOINT_FORMAT,
}



@dataclass(frozen=True)
class V23IlyaCheckpoint:
    residual_params: Mapping[str, Mapping[str, Any]]
    residual_state: Mapping[str, Mapping[str, Any]] | None
    checkpoint_format: str = CHECKPOINT_FORMAT
    checkpoint_kind: str = "legacy"

    @property
    def has_residual_state(self) -> bool:
        return bool(self.residual_state)


def load_v23_Ilya_checkpoint(path: Path) -> V23IlyaCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"v23_Ilya checkpoint not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"Could not read v23_Ilya pickle checkpoint {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed v23_Ilya checkpoint {path}: expected a mapping payload")
    architecture_id = payload.get("architecture_id")
    if architecture_id not in LEGACY_ARCHITECTURE_IDS:
        raise ValueError(
            f"Checkpoint {path} has architecture_id={architecture_id!r}, "
            "expected a v23-compatible residual checkpoint"
        )
    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint {path} has unsupported schema_version={schema_version!r}"
        )
    residual_params = payload.get("residual_params")
    if not isinstance(residual_params, Mapping) or not residual_params:
        raise ValueError(
            f"Malformed v23_Ilya checkpoint {path}: missing non-empty 'residual_params' mapping"
        )
    residual_state = payload.get("residual_state")
    if residual_state is not None and not isinstance(residual_state, Mapping):
        raise ValueError(
            f"Malformed v23_Ilya checkpoint {path}: 'residual_state' must be a mapping or None"
        )
    checkpoint_format = str(payload.get("checkpoint_format", CHECKPOINT_FORMAT))
    supported_formats = WARM_START_FORMATS
    if checkpoint_format not in supported_formats:
        raise ValueError(
            f"Checkpoint {path} has unsupported checkpoint_format={checkpoint_format!r}"
        )
    checkpoint_kind = str(payload.get("checkpoint_kind", "legacy"))
    return V23IlyaCheckpoint(
        residual_params=residual_params,
        residual_state=residual_state,
        checkpoint_format=checkpoint_format,
        checkpoint_kind=checkpoint_kind,
    )


@dataclass(frozen=True)
class V23IlyaTrainingCheckpoint:
    completed_step: int
    residual_params: Mapping[str, Mapping[str, Any]]
    residual_state: Mapping[str, Mapping[str, Any]]
    optimizer_state: Any
    rng_key: Any
    training_cursor: Mapping[str, int]
    resolved_training_config: Mapping[str, Any]
    baseline_checkpoint_path: str
    baseline_checkpoint_fingerprint: str
    anchor_manifest_fingerprint: str
    parameter_overlay_metadata: Mapping[str, Any]


def load_v23_Ilya_training_checkpoint(path: Path) -> V23IlyaTrainingCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"v23_Ilya training checkpoint not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"Could not read v23_Ilya training checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed v23_Ilya training checkpoint {path}: expected mapping")
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(f"{path} is not a {ARCHITECTURE_ID} checkpoint")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported v23_Ilya checkpoint schema {payload.get('schema_version')!r} in {path}"
        )
    if payload.get("checkpoint_format") != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an exact-resume training checkpoint")
    if payload.get("checkpoint_kind") != "training":
        raise ValueError(f"Cannot resume from checkpoint_kind={payload.get('checkpoint_kind')!r}")
    required = (
        "completed_step",
        "residual_params",
        "residual_state",
        "optimizer_state",
        "rng_key",
        "training_cursor",
        "resolved_training_config",
        "baseline_checkpoint_path",
        "baseline_checkpoint_fingerprint",
        "anchor_manifest_fingerprint",
        "parameter_overlay_metadata",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Malformed v23_Ilya training checkpoint {path}: missing {missing}")
    completed_step = int(payload["completed_step"])
    if completed_step < 0:
        raise ValueError(f"Malformed completed_step={completed_step} in {path}")
    cursor = payload["training_cursor"]
    if not isinstance(cursor, Mapping):
        raise ValueError(f"Malformed training_cursor in {path}")
    residual_params = payload["residual_params"]
    residual_state = payload["residual_state"]
    resolved_config = payload["resolved_training_config"]
    overlay_metadata = payload["parameter_overlay_metadata"]
    if not isinstance(residual_params, Mapping) or not residual_params:
        raise ValueError(f"Malformed residual_params in {path}")
    if not isinstance(residual_state, Mapping):
        raise ValueError(f"Malformed residual_state in {path}")
    if not isinstance(resolved_config, Mapping):
        raise ValueError(f"Malformed resolved_training_config in {path}")
    if not isinstance(overlay_metadata, Mapping):
        raise ValueError(f"Malformed parameter_overlay_metadata in {path}")
    if payload["optimizer_state"] is None:
        raise ValueError(f"Malformed optimizer_state in {path}")
    if payload["rng_key"] is None:
        raise ValueError(f"Malformed rng_key in {path}")
    for name in ("baseline_checkpoint_fingerprint", "anchor_manifest_fingerprint"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError(f"Malformed {name} in {path}")
    return V23IlyaTrainingCheckpoint(
        completed_step=completed_step,
        residual_params=residual_params,
        residual_state=residual_state,
        optimizer_state=payload["optimizer_state"],
        rng_key=payload["rng_key"],
        training_cursor=cursor,
        resolved_training_config=resolved_config,
        baseline_checkpoint_path=str(payload["baseline_checkpoint_path"]),
        baseline_checkpoint_fingerprint=str(payload["baseline_checkpoint_fingerprint"]),
        anchor_manifest_fingerprint=str(payload["anchor_manifest_fingerprint"]),
        parameter_overlay_metadata=overlay_metadata,
    )


def atomic_pickle_dump(payload: Mapping[str, Any], path: Path) -> None:
    """Write a host-backed pickle and publish it with an atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    host_payload = jax.device_get(payload)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(host_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    relative_paths = (
        Path("metadata.json"),
        Path("anchors/anchor_indices.npy"),
        Path("anchors/split_train.npy"),
        Path("anchors/split_val.npy"),
    )
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing anchor manifest file: {path}")
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def training_checkpoint_payload(
    *,
    completed_step: int,
    residual_params,
    residual_state,
    optimizer_state,
    rng_key,
    training_cursor: Mapping[str, int],
    resolved_training_config: Mapping[str, Any],
    baseline_checkpoint_path: str,
    baseline_checkpoint_fingerprint: str,
    anchor_manifest_fingerprint: str,
    parameter_overlay_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "architecture_id": ARCHITECTURE_ID,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
        "checkpoint_kind": "training",
        "completed_step": int(completed_step),
        "residual_params": residual_params,
        "residual_state": residual_state,
        "optimizer_state": optimizer_state,
        "rng_key": rng_key,
        "training_cursor": dict(training_cursor),
        "resolved_training_config": dict(resolved_training_config),
        "baseline_checkpoint_path": baseline_checkpoint_path,
        "baseline_checkpoint_fingerprint": baseline_checkpoint_fingerprint,
        "anchor_manifest_fingerprint": anchor_manifest_fingerprint,
        "parameter_overlay_metadata": dict(parameter_overlay_metadata),
    }


def build_v23_Ilya_swa(
    inputs: list[Path],
    source_steps: list[int],
    output: Path,
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("SWA requires at least two checkpoints")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing SWA output: {output}")
    if len(inputs) != len(source_steps):
        raise ValueError("source_steps must contain one entry per input checkpoint")
    if source_steps != sorted(source_steps) or len(set(source_steps)) != len(source_steps):
        raise ValueError("source_steps must be unique and sorted ascending")
    if any(step < 0 for step in source_steps):
        raise ValueError("source_steps must be non-negative")
    raw_checkpoints = []
    training_checkpoints = []
    for path, expected_step in zip(inputs, source_steps, strict=True):
        checkpoint = load_v23_Ilya_training_checkpoint(path)
        if checkpoint.completed_step != expected_step:
            raise ValueError(
                f"Checkpoint {path} has completed_step={checkpoint.completed_step}, "
                f"expected {expected_step}"
            )
        with path.open("rb") as handle:
            raw_checkpoints.append(pickle.load(handle))
        training_checkpoints.append(checkpoint)
    first = training_checkpoints[0]
    signature = jax.tree_util.tree_structure(first.residual_params)
    first_leaves = jax.tree_util.tree_leaves(first.residual_params)
    def swa_config_signature(value: Mapping[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(value))
        optimizer = normalized.get("optimizer", {})
        optimizer.pop("max_steps", None)
        optimizer.pop("checkpoint_every", None)
        normalized.pop("validation", None)
        architecture = normalized.get("architecture")
        if isinstance(architecture, dict):
            # Normalize checkpoints written before the no-op field was removed.
            architecture.pop("temporal_hidden_size", None)
            architecture.setdefault("temporal_bc_groups", 1)
        return normalized

    first_config_signature = swa_config_signature(first.resolved_training_config)
    for raw_checkpoint, path, checkpoint in zip(
        raw_checkpoints[1:], inputs[1:], training_checkpoints[1:], strict=True
    ):
        if jax.tree_util.tree_structure(checkpoint.residual_params) != signature:
            raise ValueError(f"Residual parameter tree differs in {path}")
        for leaf_index, (expected, candidate) in enumerate(
            zip(first_leaves, jax.tree_util.tree_leaves(checkpoint.residual_params), strict=True)
        ):
            if expected.shape != candidate.shape or expected.dtype != candidate.dtype:
                raise ValueError(
                    f"Residual parameter leaf {leaf_index} shape/dtype differs in {path}"
                )
        if swa_config_signature(checkpoint.resolved_training_config) != first_config_signature:
            raise ValueError(f"SWA checkpoints disagree on resolved_training_config in {path}")
        for field in ("baseline_checkpoint_fingerprint", "anchor_manifest_fingerprint"):
            if raw_checkpoints[0][field] != raw_checkpoint[field]:
                raise ValueError(f"SWA checkpoints disagree on {field}")

    def average_leaf(*leaves):
        first_leaf = np.asarray(leaves[0])
        if not np.issubdtype(first_leaf.dtype, np.floating):
            if not all(np.array_equal(first_leaf, np.asarray(leaf)) for leaf in leaves[1:]):
                raise ValueError("Non-floating SWA parameter leaves differ")
            return first_leaf
        accumulator = sum(np.asarray(leaf, dtype=np.float64) for leaf in leaves)
        return (accumulator / len(leaves)).astype(first_leaf.dtype)

    averaged = jax.tree_util.tree_map(
        average_leaf,
        *(checkpoint.residual_params for checkpoint in training_checkpoints),
    )
    payload = {
        "architecture_id": ARCHITECTURE_ID,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format": SWA_CHECKPOINT_FORMAT,
        "checkpoint_kind": "swa",
        "residual_params": averaged,
        "residual_state": first.residual_state,
        "swa_source_steps": list(source_steps),
        "swa_source_ckpts": [str(path) for path in inputs],
        "resolved_training_config": first.resolved_training_config,
        "baseline_checkpoint_path": first.baseline_checkpoint_path,
        "baseline_checkpoint_fingerprint": first.baseline_checkpoint_fingerprint,
        "anchor_manifest_fingerprint": first.anchor_manifest_fingerprint,
    }
    atomic_pickle_dump(payload, output)
    return payload


def validate_param_tree_compatible(
    initialized: Mapping[str, Mapping[str, Any]],
    loaded: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require identical Haiku module/parameter keys and leaf shapes."""

    initialized_modules = set(initialized)
    loaded_modules = set(loaded)
    missing_modules = sorted(initialized_modules - loaded_modules)
    extra_modules = sorted(loaded_modules - initialized_modules)
    missing_params: list[str] = []
    extra_params: list[str] = []
    mismatched_shapes: list[str] = []

    for module_name in sorted(initialized_modules & loaded_modules):
        initialized_module = initialized[module_name]
        loaded_module = loaded[module_name]
        if not isinstance(initialized_module, Mapping) or not isinstance(loaded_module, Mapping):
            raise ValueError(f"Invalid Haiku parameter module {module_name!r}: expected mappings")
        initialized_names = set(initialized_module)
        loaded_names = set(loaded_module)
        missing_params.extend(f"{module_name}/{name}" for name in sorted(initialized_names - loaded_names))
        extra_params.extend(f"{module_name}/{name}" for name in sorted(loaded_names - initialized_names))
        for name in sorted(initialized_names & loaded_names):
            expected_shape = getattr(initialized_module[name], "shape", None)
            loaded_shape = getattr(loaded_module[name], "shape", None)
            if expected_shape != loaded_shape:
                mismatched_shapes.append(
                    f"{module_name}/{name}: expected={expected_shape}, loaded={loaded_shape}"
                )

    if missing_modules or extra_modules or missing_params or extra_params or mismatched_shapes:
        details = []
        if missing_modules:
            details.append(f"missing_modules={missing_modules[:8]}")
        if extra_modules:
            details.append(f"extra_modules={extra_modules[:8]}")
        if missing_params:
            details.append(f"missing_params={missing_params[:8]}")
        if extra_params:
            details.append(f"extra_params={extra_params[:8]}")
        if mismatched_shapes:
            details.append(f"mismatched_shapes={mismatched_shapes[:8]}")
        raise ValueError("Incompatible v23_Ilya residual parameter tree: " + "; ".join(details))


def resolve_residual_state(
    requested: str,
    zero_state: Mapping[str, Mapping[str, Any]],
    checkpoint: V23IlyaCheckpoint,
) -> tuple[Mapping[str, Mapping[str, Any]], str]:
    if requested == "zero":
        return zero_state, "zero"
    if requested == "warm24":
        return zero_state, "zero_then_truth_warmup"
    if requested == "ckpt":
        if checkpoint.has_residual_state:
            assert checkpoint.residual_state is not None
            return checkpoint.residual_state, "checkpoint"
        warnings.warn(
            "Checkpoint state was requested but the checkpoint has no residual_state; "
            "falling back to zero state.",
            RuntimeWarning,
            stacklevel=2,
        )
        return zero_state, "zero_missing_checkpoint_state"
    raise ValueError(f"Unsupported residual state initialization mode: {requested!r}")
