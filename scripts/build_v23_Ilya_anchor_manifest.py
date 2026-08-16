#!/usr/bin/env python3
"""Create a standalone anchor manifest for v23_Ilya."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v23_Ilya.anchor_manifest import build_anchor_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/graphcast/graphcast/dataset/prepared_stream/res1"),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path(
            "data/graphcast/graphcast/params/"
            "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
            "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/graphcast/graphcast/dataset/anchor_manifests/v23_Ilya_res1"
        ),
    )
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--validation-year", type=int, default=2022)
    parser.add_argument("--time-start")
    parser.add_argument("--time-end")
    parser.add_argument("--allow-incomplete-prepared-store", action="store_true")
    parser.add_argument("--allow-empty-validation", action="store_true")
    args = parser.parse_args()
    metadata = build_anchor_manifest(
        prepared_root=args.prepared_root,
        baseline_checkpoint=args.baseline_checkpoint,
        output_root=args.output_root,
        train_end_year=args.train_end_year,
        validation_year=args.validation_year,
        time_start=args.time_start,
        time_end=args.time_end,
        allow_incomplete_prepared_store=args.allow_incomplete_prepared_store,
        allow_empty_validation=args.allow_empty_validation,
    )
    print(
        f"wrote {metadata['n_anchors_total']} anchors "
        f"({metadata['n_anchors_train']} train, {metadata['n_anchors_val']} validation) "
        f"to {args.output_root}"
    )


if __name__ == "__main__":
    main()
