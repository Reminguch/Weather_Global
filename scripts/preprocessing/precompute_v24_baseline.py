#!/usr/bin/env python3
"""Generate, inspect, or verify a v24 baseline-only trajectory cache."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v24_Ilya.baseline_cache import (
    build_manifest, generate_shard, load_context, verify_cache,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--max-chunks', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pilot-parity', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    context = load_context(args.config)
    manifest = build_manifest(context, args.num_shards, args.max_chunks)
    if args.verify:
        existing = json.loads((args.output_root / 'manifest.json').read_text())
        if existing != manifest:
            raise ValueError('Saved manifest differs from current source/configuration/code')
        if args.dry_run:
            raise ValueError('--verify and --dry-run cannot be combined')
        print(json.dumps(verify_cache(args.output_root), indent=2))
    elif args.dry_run:
        print(json.dumps({
            'compatibility_sha256': manifest['compatibility_sha256'],
            'manifest_sha256': manifest['manifest_sha256'],
            'chunks': len(manifest['chunks']),
            'chunks_by_split': {s: sum(c['split'] == s for c in manifest['chunks']) for s in ('train', 'val')},
            'chunks_by_shard': [sum(c['shard'] == i for c in manifest['chunks']) for i in range(args.num_shards)],
            'prediction_GiB': manifest['prediction_bytes'] / 2**30,
            'excluded_tail_anchors': manifest['excluded_tail_anchors'],
            'partial': manifest['partial'],
        }, indent=2))
    else:
        import jax
        if '--xla_gpu_deterministic_ops=true' not in os.environ.get('XLA_FLAGS', '').split():
            raise RuntimeError('Generation requires XLA_FLAGS=--xla_gpu_deterministic_ops=true, matching repository GPU parity tests')
        if not any(device.platform == 'gpu' for device in jax.devices()):
            raise RuntimeError('Baseline generation requires a GPU allocation')
        generate_shard(context, manifest, args.output_root, args.shard_index,
                       resume=args.resume, parity=args.pilot_parity)


if __name__ == '__main__':
    main()
