"""Validate and fingerprint consolidated local Zarr metadata before scheduling.

This is a metadata-only check: coordinate coverage and field/chunk validation
remain the responsibility of the workflow's scientific data preflight. No Zarr
client is started, so this check is safe to run on the submission host.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def frozen_zarr_metadata(path: str | Path) -> dict[str, str]:
    """Return the existing ``path``/``sha256`` identity for a v2 or v3 store.

    Prefer ``zarr.json`` whenever it exists, including when it is invalid: a
    stale v2 metadata file must not hide an invalid v3 store. Hash the original
    bytes without reserializing JSON to preserve existing v2 protocol hashes.
    """
    store = Path(path)
    if not store.is_dir():
        raise FileNotFoundError(f"Zarr dataset directory does not exist: {store}")
    version = 3 if (store / "zarr.json").exists() else 2
    metadata_path = store / ("zarr.json" if version == 3 else ".zmetadata")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing consolidated Zarr metadata in {store}; expected zarr.json "
            "(v3) or .zmetadata (v2)"
        )
    metadata_path = metadata_path.resolve(strict=True)
    payload = metadata_path.read_bytes()
    try:
        metadata = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid Zarr metadata JSON: {metadata_path}") from error
    valid = isinstance(metadata, dict)
    if version == 3:
        consolidated = metadata.get("consolidated_metadata") if valid else None
        valid = (valid and metadata.get("zarr_format") == 3
                 and isinstance(consolidated, dict)
                 and consolidated.get("kind") == "inline"
                 and isinstance(consolidated.get("metadata"), dict)
                 and bool(consolidated["metadata"]))
    else:
        valid = (valid and metadata.get("zarr_consolidated_format") == 1
                 and isinstance(metadata.get("metadata"), dict)
                 and bool(metadata["metadata"]))
    if not valid:
        raise ValueError(f"Expected consolidated Zarr v{version} metadata: {metadata_path}")
    return {"path": str(metadata_path), "sha256": hashlib.sha256(payload).hexdigest()}


def verify_frozen_zarr_metadata(path: str | Path, expected: dict[str, str]) -> dict[str, str]:
    """Reject metadata edits or a change in the format selected by the reader.

    Hashing only the pinned file misses a later ``zarr.json`` that supersedes a
    still-unchanged ``.zmetadata``. Re-select metadata from the dataset root.
    """
    current = frozen_zarr_metadata(path)
    if current != expected:
        raise ValueError(f"Evaluation dataset metadata changed: {path}; selected {current['path']}, "
                         f"expected {expected['path']}")
    return current
