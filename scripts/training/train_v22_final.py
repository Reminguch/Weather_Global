#!/usr/bin/env python3
"""Thin entry point for maintained v22_final training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v22_final.training.config import parse_cli  # noqa: E402
from src.models.mamba.v22_final.training.runner import run_training  # noqa: E402


def main() -> None:
    run_training(parse_cli())


if __name__ == "__main__":
    main()
