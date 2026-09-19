#!/usr/bin/env python3
"""Prepare shared initialization or train the cached stepwise residual branch."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shared-init", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--backend", choices=("cached_stepwise",), default="cached_stepwise")
    parser.add_argument("--prefetch-batches", type=int, choices=(0, 1), default=0)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--paired-report", type=Path)
    args = parser.parse_args()
    from src.models.mamba.v24_Ilya.training.cached_runner import prepare_shared_initialization, run_cached_training
    if args.stage == "prepare":
        if args.output_dir is None:
            parser.error("prepare requires --output-dir")
        print(prepare_shared_initialization(args.config, args.output_dir), flush=True)
    else:
        if args.shared_init is None or args.cache_root is None:
            parser.error("train requires --shared-init and --cache-root")
        print(run_cached_training(args.config, args.shared_init, cache_root=args.cache_root,
                                  resume=args.resume, max_steps=args.max_steps,
                                  backend=args.backend,
                                  prefetch_batches=args.prefetch_batches,
                                  expected_manifest_sha256=args.expected_manifest_sha256,
                                  parity_report=args.parity_report, paired_report=args.paired_report), flush=True)


if __name__ == "__main__":
    main()
