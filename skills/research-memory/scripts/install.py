#!/usr/bin/env python3
"""Install this portable skill without overwriting an existing different copy."""

import argparse
import os
from pathlib import Path
import shutil
import sys


def files_under(root):
    """Compare meaningful package files, excluding generated Python caches."""
    files = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("Skill package contains a symlink: {}".format(path))
        if path.is_file():
            files[relative] = path.read_bytes()
    return files


def main():
    codex_dir = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skills-dir", type=Path, default=codex_dir / "skills",
        help="User skills directory (default: CODEX_HOME/skills or ~/.codex/skills)",
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    destination = args.skills_dir.expanduser().absolute() / "research-memory"

    try:
        source_files = files_under(source)
        if Path("SKILL.md") not in source_files:
            raise ValueError("Source is missing SKILL.md")
        if destination.is_symlink():
            raise ValueError("Existing destination is a symlink; left unchanged: {}".format(destination))
        if destination.exists():
            if destination.is_dir() and files_under(destination) == source_files:
                print("Already installed: {}".format(destination))
                return 0
            raise ValueError("Existing destination differs; review it before replacing: {}".format(destination))
        if source in destination.resolve().parents:
            raise ValueError("Destination must be outside the source skill folder")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source, destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        if files_under(destination) != source_files:
            raise ValueError("Installed content did not match the source: {}".format(destination))
    except (OSError, ValueError) as exc:
        print("Installation failed: {}".format(exc), file=sys.stderr)
        return 1

    print("Installed: {}".format(destination))
    print("Restart Codex if research-memory does not appear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
