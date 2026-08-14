#!/usr/bin/env python3
"""Compatibility wrapper for the resumable GraphCast37 prepared-stream builder."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_operations.preprocessing.build_graphcast37_prepared_stream import main


if __name__ == "__main__":
    main()
