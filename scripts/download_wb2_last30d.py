#!/usr/bin/env python3
"""Compatibility wrapper for the WB2 download CLI."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_operations.download.download_wb2_last30d import main


if __name__ == "__main__":
    main()
