#!/usr/bin/env python3
"""Download public DeepMind GraphCast params and normalization stats."""

from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
from pathlib import Path


BUCKET = "dm_graphcast"
PREFIX = "graphcast"
DEFAULT_PARAMS_DIR = Path("data/graphcast/graphcast/params")
DEFAULT_STATS_DIR = Path("data/graphcast/graphcast/stats_graphcast_37")
STAT_FILENAMES = ("diffs_stddev_by_level.nc", "mean_by_level.nc", "stddev_by_level.nc")
GRAPHCAST37_RE = re.compile(
    r"^GraphCast\b.*resolution 0\.25\b.*pressure levels 37\b.*\.npz$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument(
        "--params-name",
        default=None,
        help="Exact params filename under dm_graphcast/graphcast/params/. If omitted, discover GraphCast37.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _https_url(blob_name: str) -> str:
    return f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(blob_name)}"


def _list_blobs(prefix: str) -> list[str]:
    try:
        from google.cloud import storage
    except Exception as exc:
        raise RuntimeError(
            "google-cloud-storage is required for discovery. Pass --params-name to use HTTPS fallback without listing."
        ) from exc
    client = storage.Client.create_anonymous_client()
    return [blob.name for blob in client.list_blobs(BUCKET, prefix=prefix)]


def _discover_graphcast37_params() -> str:
    prefix = f"{PREFIX}/params/"
    candidates = []
    for blob_name in _list_blobs(prefix):
        filename = blob_name.removeprefix(prefix)
        if GRAPHCAST37_RE.search(filename):
            candidates.append(filename)
    if not candidates:
        raise RuntimeError("Could not discover a GraphCast 0.25-degree, 37-level params file.")
    if len(candidates) > 1:
        candidates = sorted(candidates)
    return candidates[0]


def _download(blob_name: str, output: Path, *, overwrite: bool, dry_run: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        print(f"[download] exists: {output}")
        return
    url = _https_url(blob_name)
    if dry_run:
        print(f"[download] dry-run {url} -> {output}")
        return
    tmp_path = output.with_name(f".{output.name}.tmp")
    print(f"[download] {url} -> {output}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(16 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp_path.replace(output)


def main() -> None:
    args = parse_args()
    params_name = args.params_name or _discover_graphcast37_params()
    print(f"GraphCast37 params: {params_name}")
    _download(
        f"{PREFIX}/params/{params_name}",
        args.params_dir / params_name,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    for filename in STAT_FILENAMES:
        _download(
            f"{PREFIX}/stats/{filename}",
            args.stats_dir / filename,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
