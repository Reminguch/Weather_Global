"""Native-only v24 checkpoint serialization and validation."""

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
import jax.numpy as jnp
import numpy as np

from .config import ARCHITECTURE_ID, SCHEMA_VERSION


TRAINING_CHECKPOINT_FORMAT = "v24_Ilya_training_pickle"
DATA_PARALLEL_TRAINING_CHECKPOINT_FORMAT = "v24_Ilya_data_parallel_training_pickle"
SWA_CHECKPOINT_FORMAT = "v24_Ilya_swa_pickle"
NATIVE_CHECKPOINT_FORMATS = {
    TRAINING_CHECKPOINT_FORMAT,
    DATA_PARALLEL_TRAINING_CHECKPOINT_FORMAT,
    SWA_CHECKPOINT_FORMAT,
}


def _validate_floating_tree_fp32(tree: Any, label: str) -> None:
    """Reject quantized floating checkpoint boundaries while allowing integers."""

    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None:
            continue
        if jnp.issubdtype(dtype, jnp.inexact) and dtype != jnp.float32:
            raise ValueError(
                f"{label} floating leaf {index} must be float32 in v24_Ilya, got {dtype}"
            )



@dataclass(frozen=True)
class FrozenBaselineOverlayStats:
    """Audit information for loading a frozen baseline checkpoint."""

    copied: int
    ignored_source: tuple[str, ...]


def overlay_frozen_baseline_params(
    initialized: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], FrozenBaselineOverlayStats]:
    """Load every locally required baseline parameter and ignore source-only leaves.

    GraphCast checkpoints can contain auxiliary output heads that are not created by
    the local forward graph. Those source-only leaves are safe to ignore, but every
    parameter created by the local graph must be present with the exact same shape.
    """

    if not isinstance(initialized, Mapping) or not isinstance(source, Mapping):
        raise ValueError("Frozen GraphCast parameter trees must be mappings")

    loaded: dict[str, dict[str, Any]] = {}
    missing_modules: list[str] = []
    missing_params: list[str] = []
    mismatched_shapes: list[str] = []

    for module_name, initialized_module in initialized.items():
        if not isinstance(initialized_module, Mapping):
            raise ValueError(
                f"Invalid initialized GraphCast module {module_name!r}: expected a mapping"
            )
        source_module = source.get(module_name)
        if source_module is None:
            missing_modules.append(str(module_name))
            continue
        if not isinstance(source_module, Mapping):
            raise ValueError(
                f"Invalid checkpoint GraphCast module {module_name!r}: expected a mapping"
            )
        loaded_module: dict[str, Any] = {}
        for parameter_name, initialized_leaf in initialized_module.items():
            path = f"{module_name}/{parameter_name}"
            if parameter_name not in source_module:
                missing_params.append(path)
                continue
            source_leaf = source_module[parameter_name]
            initialized_shape = getattr(initialized_leaf, "shape", None)
            source_shape = getattr(source_leaf, "shape", None)
            if initialized_shape != source_shape:
                mismatched_shapes.append(
                    f"{path}: expected={initialized_shape}, loaded={source_shape}"
                )
                continue
            loaded_module[parameter_name] = source_leaf
        loaded[str(module_name)] = loaded_module

    if missing_modules or missing_params or mismatched_shapes:
        details = []
        if missing_modules:
            details.append(f"missing_modules={sorted(missing_modules)[:8]}")
        if missing_params:
            details.append(f"missing_params={sorted(missing_params)[:8]}")
        if mismatched_shapes:
            details.append(f"mismatched_shapes={sorted(mismatched_shapes)[:8]}")
        raise ValueError(
            "Frozen GraphCast checkpoint does not cover the initialized baseline: "
            + "; ".join(details)
        )

    ignored_source: list[str] = []
    for module_name, source_module in source.items():
        if not isinstance(source_module, Mapping):
            raise ValueError(
                f"Invalid checkpoint GraphCast module {module_name!r}: expected a mapping"
            )
        initialized_module = initialized.get(module_name)
        if initialized_module is None:
            ignored_source.extend(
                f"{module_name}/{parameter_name}" for parameter_name in source_module
            )
            continue
        ignored_source.extend(
            f"{module_name}/{parameter_name}"
            for parameter_name in source_module
            if parameter_name not in initialized_module
        )

    copied = sum(len(module) for module in loaded.values())
    return loaded, FrozenBaselineOverlayStats(
        copied=copied,
        ignored_source=tuple(sorted(ignored_source)),
    )


@dataclass(frozen=True)
class V24IlyaCheckpoint:
    residual_params: Mapping[str, Mapping[str, Any]]
    residual_state: Mapping[str, Mapping[str, Any]] | None
    checkpoint_format: str
    checkpoint_kind: str

    @property
    def has_residual_state(self) -> bool:
        return bool(self.residual_state)


def load_v24_Ilya_checkpoint(path: Path) -> V24IlyaCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"v24_Ilya checkpoint not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"Could not read v24_Ilya pickle checkpoint {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed v24_Ilya checkpoint {path}: expected a mapping payload")
    architecture_id = payload.get("architecture_id")
    if architecture_id != ARCHITECTURE_ID:
        raise ValueError(
            f"Checkpoint {path} has architecture_id={architecture_id!r}, "
            f"expected {ARCHITECTURE_ID!r}; v23/v22 checkpoints cannot be used "
            "with the FP32 physical-weather contract"
        )
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint {path} has unsupported schema_version={schema_version!r}"
        )
    residual_params = payload.get("residual_params")
    if not isinstance(residual_params, Mapping) or not residual_params:
        raise ValueError(
            f"Malformed v24_Ilya checkpoint {path}: missing non-empty 'residual_params' mapping"
        )
    residual_state = payload.get("residual_state")
    if residual_state is not None and not isinstance(residual_state, Mapping):
        raise ValueError(
            f"Malformed v24_Ilya checkpoint {path}: 'residual_state' must be a mapping or None"
        )
    _validate_floating_tree_fp32(residual_params, "residual_params")
    if residual_state is not None:
        _validate_floating_tree_fp32(residual_state, "residual_state")
    checkpoint_format = payload.get("checkpoint_format")
    if checkpoint_format not in NATIVE_CHECKPOINT_FORMATS:
        raise ValueError(
            f"Checkpoint {path} has unsupported checkpoint_format={checkpoint_format!r}; "
            "only native v24 training, data-parallel training, and SWA checkpoints "
            "are accepted"
        )
    checkpoint_kind = payload.get("checkpoint_kind")
    expected_kind = "swa" if checkpoint_format == SWA_CHECKPOINT_FORMAT else "training"
    if checkpoint_kind != expected_kind:
        raise ValueError(
            f"Checkpoint {path} has checkpoint_kind={checkpoint_kind!r}, "
            f"expected {expected_kind!r} for {checkpoint_format!r}"
        )
    return V24IlyaCheckpoint(
        residual_params=residual_params,
        residual_state=residual_state,
        checkpoint_format=str(checkpoint_format),
        checkpoint_kind=str(checkpoint_kind),
    )


@dataclass(frozen=True)
class V24IlyaTrainingCheckpoint:
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


def load_v24_Ilya_training_checkpoint(path: Path) -> V24IlyaTrainingCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"v24_Ilya training checkpoint not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"Could not read v24_Ilya training checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed v24_Ilya training checkpoint {path}: expected mapping")
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(f"{path} is not a {ARCHITECTURE_ID} checkpoint")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported v24_Ilya checkpoint schema {payload.get('schema_version')!r} in {path}"
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
        raise ValueError(f"Malformed v24_Ilya training checkpoint {path}: missing {missing}")
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
    _validate_floating_tree_fp32(residual_params, "residual_params")
    _validate_floating_tree_fp32(residual_state, "residual_state")
    _validate_floating_tree_fp32(payload["optimizer_state"], "optimizer_state")
    return V24IlyaTrainingCheckpoint(
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


@dataclass(frozen=True)
class V24IlyaDataParallelTrainingCheckpoint:
    completed_step: int
    residual_params: Mapping[str, Mapping[str, Any]]
    replica_states: Mapping[str, Mapping[str, Any]]
    optimizer_state: Any
    rng_key: Any
    replica_group_cursor: Mapping[str, int]
    active_segment_ids: tuple[int, ...]
    num_devices: int
    resolved_training_config: Mapping[str, Any]
    baseline_checkpoint_path: str
    baseline_checkpoint_fingerprint: str
    anchor_manifest_fingerprint: str
    parameter_overlay_metadata: Mapping[str, Any]


def load_v24_Ilya_data_parallel_training_checkpoint(
    path: Path,
) -> V24IlyaDataParallelTrainingCheckpoint:
    if not path.is_file():
        raise FileNotFoundError(f"v24_Ilya training checkpoint not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(
            f"Could not read v24_Ilya data-parallel checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed data-parallel checkpoint {path}: expected mapping")
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(f"{path} is not a {ARCHITECTURE_ID} checkpoint")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported v24_Ilya checkpoint schema "
            f"{payload.get('schema_version')!r} in {path}"
        )
    if payload.get("checkpoint_format") != DATA_PARALLEL_TRAINING_CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a data-parallel exact-resume checkpoint")
    if payload.get("checkpoint_kind") != "training":
        raise ValueError(f"Cannot resume from checkpoint_kind={payload.get('checkpoint_kind')!r}")
    required = (
        "completed_step",
        "residual_params",
        "replica_states",
        "optimizer_state",
        "rng_key",
        "replica_group_cursor",
        "active_segment_ids",
        "num_devices",
        "resolved_training_config",
        "baseline_checkpoint_path",
        "baseline_checkpoint_fingerprint",
        "anchor_manifest_fingerprint",
        "parameter_overlay_metadata",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Malformed data-parallel checkpoint {path}: missing {missing}")
    completed_step = int(payload["completed_step"])
    num_devices = int(payload["num_devices"])
    active_segment_ids = tuple(int(value) for value in payload["active_segment_ids"])
    if completed_step < 0:
        raise ValueError(f"Malformed completed_step={completed_step} in {path}")
    if num_devices < 2:
        raise ValueError(f"Malformed num_devices={num_devices} in {path}")
    if len(active_segment_ids) != num_devices:
        raise ValueError(
            f"Expected {num_devices} active_segment_ids, got {len(active_segment_ids)}"
        )
    for name in (
        "residual_params",
        "replica_states",
        "replica_group_cursor",
        "resolved_training_config",
        "parameter_overlay_metadata",
    ):
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"Malformed {name} in {path}")
    if not payload["residual_params"]:
        raise ValueError(f"Malformed residual_params in {path}")
    if payload["optimizer_state"] is None or payload["rng_key"] is None:
        raise ValueError(f"Malformed optimizer_state or rng_key in {path}")
    for name in ("baseline_checkpoint_fingerprint", "anchor_manifest_fingerprint"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError(f"Malformed {name} in {path}")
    _validate_floating_tree_fp32(payload["residual_params"], "residual_params")
    _validate_floating_tree_fp32(payload["replica_states"], "replica_states")
    _validate_floating_tree_fp32(payload["optimizer_state"], "optimizer_state")
    return V24IlyaDataParallelTrainingCheckpoint(
        completed_step=completed_step,
        residual_params=payload["residual_params"],
        replica_states=payload["replica_states"],
        optimizer_state=payload["optimizer_state"],
        rng_key=payload["rng_key"],
        replica_group_cursor=payload["replica_group_cursor"],
        active_segment_ids=active_segment_ids,
        num_devices=num_devices,
        resolved_training_config=payload["resolved_training_config"],
        baseline_checkpoint_path=str(payload["baseline_checkpoint_path"]),
        baseline_checkpoint_fingerprint=str(payload["baseline_checkpoint_fingerprint"]),
        anchor_manifest_fingerprint=str(payload["anchor_manifest_fingerprint"]),
        parameter_overlay_metadata=payload["parameter_overlay_metadata"],
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


def data_parallel_training_checkpoint_payload(
    *,
    completed_step: int,
    residual_params,
    replica_states,
    optimizer_state,
    rng_key,
    replica_group_cursor: Mapping[str, int],
    active_segment_ids: tuple[int, ...],
    num_devices: int,
    resolved_training_config: Mapping[str, Any],
    baseline_checkpoint_path: str,
    baseline_checkpoint_fingerprint: str,
    anchor_manifest_fingerprint: str,
    parameter_overlay_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "architecture_id": ARCHITECTURE_ID,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_format": DATA_PARALLEL_TRAINING_CHECKPOINT_FORMAT,
        "checkpoint_kind": "training",
        "completed_step": int(completed_step),
        "residual_params": residual_params,
        # No single lane state is canonical for generic evaluation/warm start.
        "residual_state": None,
        "replica_states": replica_states,
        "optimizer_state": optimizer_state,
        "rng_key": rng_key,
        "replica_group_cursor": dict(replica_group_cursor),
        "active_segment_ids": [int(value) for value in active_segment_ids],
        "num_devices": int(num_devices),
        "resolved_training_config": dict(resolved_training_config),
        "baseline_checkpoint_path": baseline_checkpoint_path,
        "baseline_checkpoint_fingerprint": baseline_checkpoint_fingerprint,
        "anchor_manifest_fingerprint": anchor_manifest_fingerprint,
        "parameter_overlay_metadata": dict(parameter_overlay_metadata),
    }


def build_v24_Ilya_swa(
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
    training_checkpoints: list[
        V24IlyaTrainingCheckpoint | V24IlyaDataParallelTrainingCheckpoint
    ] = []
    checkpoint_formats: list[str] = []
    for path, expected_step in zip(inputs, source_steps, strict=True):
        try:
            with path.open("rb") as handle:
                checkpoint_format = pickle.load(handle).get("checkpoint_format")
        except Exception as exc:
            raise ValueError(f"Could not inspect SWA checkpoint {path}: {exc}") from exc
        if checkpoint_format == TRAINING_CHECKPOINT_FORMAT:
            checkpoint = load_v24_Ilya_training_checkpoint(path)
        elif checkpoint_format == DATA_PARALLEL_TRAINING_CHECKPOINT_FORMAT:
            checkpoint = load_v24_Ilya_data_parallel_training_checkpoint(path)
        else:
            raise ValueError(
                f"SWA input {path} has unsupported checkpoint_format="
                f"{checkpoint_format!r}"
            )
        if checkpoint.completed_step != expected_step:
            raise ValueError(
                f"Checkpoint {path} has completed_step={checkpoint.completed_step}, "
                f"expected {expected_step}"
            )
        training_checkpoints.append(checkpoint)
        checkpoint_formats.append(str(checkpoint_format))
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
            architecture.setdefault("residual_width", None)
            architecture.setdefault("residual_initialization", "baseline_overlay")
            # Normalize checkpoints written before the no-op field was removed.
            architecture.pop("temporal_hidden_size", None)
            architecture.setdefault("temporal_bc_groups", 1)
        return normalized

    first_config_signature = swa_config_signature(first.resolved_training_config)
    for path, checkpoint in zip(inputs[1:], training_checkpoints[1:], strict=True):
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
            if getattr(first, field) != getattr(checkpoint, field):
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
        # Data-parallel checkpoints contain distinct lane states and therefore
        # have no canonical recurrent state for SWA evaluation or warm start.
        "residual_state": (
            first.residual_state
            if all(value == TRAINING_CHECKPOINT_FORMAT for value in checkpoint_formats)
            else None
        ),
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
        raise ValueError("Incompatible v24_Ilya residual parameter tree: " + "; ".join(details))


def resolve_residual_state(
    requested: str,
    zero_state: Mapping[str, Mapping[str, Any]],
    checkpoint: V24IlyaCheckpoint,
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
