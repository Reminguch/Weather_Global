"""Freeze complete experiment code and execute the verified copy.

Data and outputs belong to the workspace; Python and subprocess entry points
belong to the source snapshot. Never refresh a snapshot from a changing workspace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_DIRS = ("src", "scripts", "third_party/graphcast")
SOURCE_SUFFIXES = {".py", ".sh", ".slurm"}
FORMAT = "weather_frozen_source_v1"
VENDORED_ENTRYPOINT = "third_party/graphcast/graphcast/graphcast.py"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_files(root):
    return sorted(p for name in SOURCE_DIRS for p in (root / name).rglob("*")
                  if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts)


def _digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_source(workspace, destination):
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    if (destination / "snapshot.json").exists():
        return verify_source(destination)
    if destination.exists():
        raise FileExistsError(f"Incomplete/legacy source directory; use a new snapshot: {destination}")
    for name in SOURCE_DIRS:
        if not (workspace / name).is_dir():
            raise FileNotFoundError(f"Required source directory is missing: {workspace / name}")
    if not (workspace / VENDORED_ENTRYPOINT).is_file():
        raise FileNotFoundError(f"Required vendored GraphCast is missing: {workspace / VENDORED_ENTRYPOINT}")
    if any(destination.is_relative_to(workspace / name) for name in SOURCE_DIRS):
        raise ValueError("Snapshot destination must be outside source directories")
    files = {}
    destination.mkdir(parents=True)
    for source in _source_files(workspace):
        relative = source.relative_to(workspace)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = file_sha256(source)
        shutil.copy2(source, target)
        if file_sha256(target) != before or file_sha256(source) != before:
            raise ValueError(f"Source changed while freezing: {source}")
        files[str(relative)] = before
    manifest = {"format": FORMAT, "workspace": str(workspace), "files": files,
                "source_digest": _digest(files)}
    temporary = destination / "snapshot.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "snapshot.json")
    return verify_source(destination)


def verify_source(source_root):
    source_root = Path(source_root).resolve()
    manifest_path = source_root / "snapshot.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No complete source snapshot: {manifest_path}; prepare a new experiment "
                                "before submission or use an explicit recovery for historical runs")
    manifest = json.loads(manifest_path.read_text())
    files = manifest["files"]
    if not files:
        raise ValueError("Empty source snapshot")
    for relative, expected in files.items():
        path = source_root / relative
        if not path.resolve().is_relative_to(source_root):
            raise ValueError(f"Source path escapes snapshot: {relative}")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    if manifest.get("format") == FORMAT:
        if VENDORED_ENTRYPOINT not in files:
            raise ValueError("Source snapshot lacks vendored GraphCast")
        if manifest["source_digest"] != _digest(files):
            raise ValueError("Source manifest digest differs")
        actual = {str(p.relative_to(source_root)) for p in _source_files(source_root)}
        if actual != set(files):
            raise ValueError("Snapshot contains unrecorded or missing executable source")
    return manifest


def source_environment(source_root, workspace):
    source_root, workspace = Path(source_root).resolve(), Path(workspace).resolve()
    environment = dict(os.environ)
    environment.update(PROJECT_ROOT=str(workspace), WEATHER_WORKSPACE=str(workspace),
                       WEATHER_SOURCE_ROOT=str(source_root),
                       PYTHONSAFEPATH="1",
                       PYTHONPATH=f"{source_root}:{source_root / 'third_party/graphcast'}")
    return environment


def frozen_command(source_root, workspace, script, args):
    source_root, workspace = Path(source_root).resolve(), Path(workspace).resolve()
    manifest = verify_source(source_root)
    if manifest.get('workspace') and Path(manifest['workspace']).resolve() != workspace:
        raise ValueError('Runtime workspace differs from the prepared snapshot')
    path = (source_root / script).resolve()
    if not path.is_relative_to(source_root) or str(path.relative_to(source_root)) not in manifest["files"]:
        raise ValueError(f"Entry point is not recorded in the snapshot: {script}")
    return ([sys.executable, "-u", str(path), *map(str, args)],
            {"cwd": workspace, "env": source_environment(source_root, workspace)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "verify", "exec"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--script")
    # Separate entry-point arguments so its options are never parsed here.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    forwarded = argv[split + 1:]
    if args.phase == "verify":
        verify_source(args.source)
    elif args.workspace is None:
        parser.error("--workspace is required for freeze and exec")
    elif args.phase == "freeze":
        freeze_source(args.workspace, args.source)
    elif args.script is None:
        parser.error("exec requires --script")
    else:
        command, options = frozen_command(args.source, args.workspace, args.script, forwarded)
        subprocess.run(command, check=True, **options)


if __name__ == "__main__":
    main()
