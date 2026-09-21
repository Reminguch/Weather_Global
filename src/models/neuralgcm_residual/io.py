"""Content identities and atomic, immutable artifact utilities."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, value, *, immutable=False):
    path = Path(path)
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists():
        if read_json(path) != json.loads(serialized):
            raise FileExistsError(f"Immutable artifact differs: {path}")
        return
    atomic_bytes(path, serialized.encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_pickle(path, value):
    atomic_bytes(path, pickle.dumps(value, protocol=5))


def versions():
    result = {}
    for name in ("neuralgcm", "dinosaur", "jax", "jaxlib", "dm-haiku", "optax", "numpy", "xarray"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
