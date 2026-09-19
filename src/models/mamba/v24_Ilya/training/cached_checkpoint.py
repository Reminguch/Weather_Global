"""Native-compatible cached checkpoints and content-addressed experiment gates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..checkpoint import (
    atomic_json_dump, atomic_pickle_dump, file_sha256,
    load_v24_Ilya_training_checkpoint, training_checkpoint_payload,
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def numerical_tree_sha256(tree: Any) -> str:
    """Hash structure, dtype, shape and exact array bytes without pickle metadata."""
    import jax
    import numpy as np
    leaves, structure = jax.tree_util.tree_flatten(jax.device_get(tree))
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        if array.dtype.hasobject:
            raise ValueError("Numerical checkpoint identity cannot contain object arrays")
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def numerical_checkpoint_identity(checkpoint) -> dict[str, str]:
    return {name: numerical_tree_sha256(getattr(checkpoint, name)) for name in
            ("residual_params", "residual_state", "optimizer_state", "rng_key", "training_cursor")}


def scientific_config(config) -> dict:
    """Identity independent of output location and operational stopping points."""
    value = copy.deepcopy(config.to_dict() if hasattr(config, "to_dict") else dict(config))
    value.pop("output", None)
    value.pop("validation", None)
    for key in ("max_steps", "checkpoint_every"):
        value["optimizer"].pop(key, None)
    # Only constant schedules are approved for the paired experiment: changing
    # max_steps must not change a cosine trajectory during smoke/projection.
    if value["optimizer"].get("learning_rate_schedule", "constant") != "constant":
        raise ValueError("Shared initialization requires a constant learning-rate schedule")
    return value


def code_fingerprints(repo_root: Path) -> dict[str, str]:
    """Pin all executed local model/training code, including maintained code."""
    directories = (
        "src/models/mamba/v24_Ilya", "src/models/mamba/modules", "src/models/mamba/training",
        "src/models/graphcast/training/core", "third_party/graphcast/graphcast",
    )
    paths = {path for directory in directories for path in (repo_root / directory).rglob("*.py")}
    for relative in (
        "scripts/training/train_v24_cached_open_loop.py",
        "scripts/training/train_v24_Ilya.py",
        "scripts/experiments/run_v24_cached_pair.py",
        "scripts/experiments/v24_cached_pair.slurm",
        "scripts/graphcast_env.sh",
    ):
        path = repo_root / relative
        if path.is_file():
            paths.add(path)
    return {str(path.relative_to(repo_root)): file_sha256(path) for path in sorted(paths)}


def build_provenance(config, shared_init_path: Path, cache_root: Path, repo_root: Path | None = None,
                     *, execution: Mapping[str, Any] | None = None) -> dict:
    repo_root = Path(repo_root or Path(__file__).resolve().parents[5])
    manifest = json.loads((Path(cache_root) / "manifest.json").read_text())
    ready = json.loads((Path(cache_root) / "READY.json").read_text())
    if ready.get("partial") or ready.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Production gates require the complete verified cache")
    import importlib.metadata
    return {
        "scientific_config_sha256": digest_json(scientific_config(config)),
        "shared_init_sha256": file_sha256(Path(shared_init_path)),
        "cache_manifest_sha256": manifest["manifest_sha256"],
        "code_sha256": code_fingerprints(repo_root),
        "execution": dict(execution or {}),
        "environment": {name: importlib.metadata.version(name)
                        for name in ("jax", "jaxlib", "dm-haiku", "optax")},
    }


def require_production_gates(parity_report: Path, paired_report: Path, provenance: Mapping[str, Any]) -> dict:
    reports = {}
    for kind, path in (("full_parity", Path(parity_report)), ("paired20", Path(paired_report))):
        report = json.loads(path.read_text())
        if report.get("kind") != kind or report.get("passed") is not True:
            raise ValueError(f"Required {kind} gate did not pass: {path}")
        if report.get("provenance") != dict(provenance):
            raise ValueError(f"Stale {kind} gate: code, initialization, cache or configuration changed")
        if kind == "paired20" and int(report.get("completed_steps", 0)) < 20:
            raise ValueError("Paired gate must complete at least twenty updates per backend")
        reports[kind] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return reports


def checkpoint_sidecar(path: Path) -> Path:
    return path.with_suffix(".cached.json")


def write_cached_checkpoint(path: Path, *, config, completed_step, residual_params,
                            residual_state, optimizer_state, rng_key, cursor,
                            baseline_fingerprint, anchor_fingerprint, overlay_metadata,
                            provenance: Mapping[str, Any], execution: Mapping[str, Any]) -> Path:
    payload = training_checkpoint_payload(
        completed_step=completed_step, residual_params=residual_params,
        residual_state=residual_state, optimizer_state=optimizer_state, rng_key=rng_key,
        training_cursor=cursor.to_dict(), resolved_training_config=config.to_dict(),
        baseline_checkpoint_path=str(config.baseline_checkpoint),
        baseline_checkpoint_fingerprint=baseline_fingerprint,
        anchor_manifest_fingerprint=anchor_fingerprint,
        parameter_overlay_metadata=overlay_metadata,
    )
    atomic_pickle_dump(payload, path)
    atomic_json_dump({"format": "v24_cached_checkpoint_v1", "checkpoint_sha256": file_sha256(path),
                      "completed_step": int(completed_step), "provenance": dict(provenance),
                      "execution": dict(execution)}, checkpoint_sidecar(path))
    atomic_json_dump({"completed_step": int(completed_step), "checkpoint": str(path)},
                     config.run_dir / "latest_checkpoint.json")
    return path


def validate_cached_checkpoint(path: Path, provenance: Mapping[str, Any], execution: Mapping[str, Any]):
    sidecar = json.loads(checkpoint_sidecar(path).read_text())
    if sidecar.get("format") != "v24_cached_checkpoint_v1":
        raise ValueError("Unsupported cached checkpoint sidecar")
    if sidecar.get("checkpoint_sha256") != file_sha256(path):
        raise ValueError("Cached checkpoint hash differs from sidecar")
    if sidecar.get("provenance") != dict(provenance) or sidecar.get("execution") != dict(execution):
        raise ValueError("Cached resume provenance/execution differs from the saved checkpoint")
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    if checkpoint.completed_step != sidecar.get("completed_step"):
        raise ValueError("Checkpoint step differs from sidecar")
    return checkpoint


def project_shared_initialization(shared_path: Path, config, *, provenance=None, execution=None) -> Path:
    """Copy identical numerical step-zero state into a run-local native checkpoint."""
    source = load_v24_Ilya_training_checkpoint(shared_path)
    if source.completed_step != 0 or source.training_cursor != {"epoch": 0, "segment_index": 0, "segment_offset": 0}:
        raise ValueError("Shared initialization must be canonical step zero")
    if scientific_config(source.resolved_training_config) != scientific_config(config):
        raise ValueError("Run configuration changes the shared scientific initialization")
    path = config.run_dir / "checkpoints/checkpoint_step00000000.pkl"
    metadata_path = config.run_dir / "shared_initialization.json"
    expected = {"shared_init": str(Path(shared_path).resolve()), "shared_init_sha256": file_sha256(shared_path),
                "config_sha256": digest_json(config.to_dict()),
                "numerical_sha256": numerical_checkpoint_identity(source)}
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Run already belongs to another shared initialization/configuration")
        if not path.is_file():
            raise FileNotFoundError(path)
        if saved.get("projected_checkpoint_sha256") != file_sha256(path):
            raise ValueError("Projected initialization checkpoint was modified")
        target = validate_cached_checkpoint(path, provenance or {}, execution or {})
        if numerical_checkpoint_identity(target) != expected["numerical_sha256"]:
            raise ValueError("Projected numerical initialization differs from the canonical source")
        return path
    if config.run_dir.exists() and any(config.run_dir.iterdir()):
        raise FileExistsError(f"Refusing to project into nonempty run directory: {config.run_dir}")
    from .data import TrainingCursor
    write_cached_checkpoint(path, config=config, completed_step=0,
                            residual_params=source.residual_params, residual_state=source.residual_state,
                            optimizer_state=source.optimizer_state, rng_key=source.rng_key,
                            cursor=TrainingCursor(), baseline_fingerprint=source.baseline_checkpoint_fingerprint,
                            anchor_fingerprint=source.anchor_manifest_fingerprint,
                            overlay_metadata=source.parameter_overlay_metadata,
                            provenance=provenance or {}, execution=execution or {})
    target = load_v24_Ilya_training_checkpoint(path)
    if numerical_checkpoint_identity(target) != expected["numerical_sha256"]:
        raise ValueError("Projection changed the canonical numerical initialization")
    atomic_json_dump({**config.to_dict(), "shared_init_sha256": expected["shared_init_sha256"]},
                     config.run_dir / "run_config.json")
    atomic_json_dump({**expected, "projected_checkpoint_sha256": file_sha256(path)}, metadata_path)
    return path


def latest_checkpoint_for_run(config) -> Path | None:
    """Discover completed checkpoints without silently restarting an existing run."""
    marker = config.run_dir / "latest_checkpoint.json"
    if not marker.exists():
        return None
    value = json.loads(marker.read_text())
    path = Path(value["checkpoint"])
    if path.resolve().parent.parent != config.run_dir.resolve() or not path.is_file():
        raise ValueError("Latest checkpoint pointer is missing or belongs to another run")
    checkpoint = load_v24_Ilya_training_checkpoint(path)
    if checkpoint.completed_step != int(value["completed_step"]):
        raise ValueError("Latest checkpoint pointer has an inconsistent step")
    return path


def reconcile_run_metrics(run_dir: Path, completed_step: int) -> dict[str, int]:
    """Archive uncheckpointed metric tails before exact replay appends new rows."""
    import os
    import tempfile
    result = {}
    for name in ("train_metrics.jsonl", "checkpoint_metrics.jsonl", "validation_metrics.jsonl"):
        path = Path(run_dir) / name
        if not path.is_file():
            continue
        lines = path.read_text().splitlines(keepends=True)
        retained, discarded = [], []
        for index, line in enumerate(lines):
            if not line.strip():
                retained.append(line)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError(f"Corrupted nonterminal metric record: {path}:{index + 1}")
                discarded.append(line)
                continue
            (retained if int(record["step"]) <= completed_step else discarded).append(line)
        if not discarded:
            continue
        # Keep discarded observations for diagnosis while the primary stream
        # records only the state trajectory represented by the resume point.
        recovery_dir = Path(run_dir) / "recovery"
        recovery_dir.mkdir(exist_ok=True)
        archive = recovery_dir / f"{name}.after_step{completed_step:08d}.{digest_json(discarded)[:12]}"
        if not archive.exists():
            archive.write_text("".join(discarded))
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("".join(retained))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        result[name] = len(discarded)
    return result
