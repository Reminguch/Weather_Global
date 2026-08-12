#!/usr/bin/env python3
"""Build a schema-checked v22_final SWA checkpoint from explicit inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v22_final.checkpoint import build_v22_final_swa  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--source-steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_v22_final_swa(args.inputs, args.source_steps, args.output)
    print(f"Saved v22_final SWA checkpoint: {args.output}")


if __name__ == "__main__":
    main()
