"""Residual-memory training entrypoints and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent.parent
training_dir_str = str(TRAINING_DIR)
if training_dir_str not in sys.path:
    sys.path.insert(0, training_dir_str)
