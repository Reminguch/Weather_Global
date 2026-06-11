#!/usr/bin/env python3
"""Compatibility wrapper for the DeepMind GraphCast asset downloader."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_operations.download.download_deepmind_graphcast_assets import main


if __name__ == "__main__":
    main()
