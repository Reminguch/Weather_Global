#!/usr/bin/env python3
"""Thin CLI for the maintained v23_Ilya evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v23_Ilya.config import parse_args  # noqa: E402
from src.models.mamba.v23_Ilya.evaluation import evaluate_v23_Ilya  # noqa: E402


def main() -> None:
    evaluate_v23_Ilya(parse_args())


if __name__ == "__main__":
    main()
