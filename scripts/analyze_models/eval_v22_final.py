#!/usr/bin/env python3
"""Thin CLI for the maintained v22_final evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v22_final.config import parse_args  # noqa: E402
from src.models.mamba.v22_final.evaluation import evaluate_v22_final  # noqa: E402


def main() -> None:
    evaluate_v22_final(parse_args())


if __name__ == "__main__":
    main()
