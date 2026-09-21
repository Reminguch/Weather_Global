"""Exact numerical resume is distinct from intentional branch-only stage transfer."""
from __future__ import annotations

from pathlib import Path
import pickle
import json
import jax
from .io import sha256, write_pickle, write_json, read_json


def reconcile_metrics(path, completed_update):
    """Archive uncommitted log tails before exact checkpoint replay."""
    path = Path(path)
    if not path.exists():
        return
    from .io import atomic_bytes
    committed, tail = [], []
    for line in path.read_text().splitlines(keepends=True):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            tail.append(line)
            continue
        (committed if record["update"] <= completed_update else tail).append(line)
    if tail:
        archive = path.with_name(path.name + f".after_{completed_update}.archive")
        if archive.exists():
            archive = path.with_name(archive.name + "." + sha256(path)[:12])
        atomic_bytes(archive, "".join(tail).encode())
        atomic_bytes(path, "".join(committed).encode())


def save_checkpoint(path, *, stage, identities, params, optimizer, memory, rng, cursor):
    if stage not in ("pretrain", "finetune"):
        raise ValueError("Unknown training stage")
    if stage == "finetune" and not cursor.get("episode_boundary", False):
        raise ValueError("This runner saves only at episode boundaries")
    payload = jax.device_get({"format": "neuralgcm_exact_resume_v1", "stage": stage,
                             "identities": identities, "params": params, "optimizer": optimizer,
                             "memory": memory, "rng": rng, "cursor": cursor})
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve completed checkpoint: {path}")
    write_pickle(path, payload)
    write_json(path.with_suffix(".json"), {"sha256": sha256(path), "stage": stage,
               "identities": identities, "cursor": cursor}, immutable=True)
    return path


def load_checkpoint(path, *, stage, identities):
    path = Path(path)
    metadata = read_json(path.with_suffix(".json"))
    if sha256(path) != metadata["sha256"]:
        raise ValueError("Checkpoint hash differs")
    with path.open("rb") as f:
        state = pickle.load(f)
    if (state["format"] != "neuralgcm_exact_resume_v1" or state["stage"] != stage
            or state["identities"] != identities):
        raise ValueError("Resume requires identical stage, schema, data, normalization, source and architecture")
    return state


def transfer_parent(path, expected_compatibility):
    metadata = read_json(Path(path).with_suffix(".json"))
    if metadata["stage"] != "pretrain":
        raise ValueError("Fine-tuning requires a selected pretrained parent")
    for key, value in expected_compatibility.items():
        if metadata["identities"].get(key) != value:
            raise ValueError(f"Incompatible stage transfer: {key}")
    state = load_checkpoint(path, stage="pretrain", identities=metadata["identities"])
    return state["params"], {"operation": "pretrain_to_finetune_transfer", "path": str(Path(path).resolve()),
                             "sha256": metadata["sha256"], "reset": ["optimizer", "rng", "cursor", "memory"]}
