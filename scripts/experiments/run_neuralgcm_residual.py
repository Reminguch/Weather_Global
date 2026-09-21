#!/usr/bin/env python3
"""Prepare, submit, or execute the pinned NeuralGCM native residual experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# A frozen worker must import its own vendored API even when the caller sourced
# the live checkout's environment activation script.
sys.path.insert(0, str(ROOT / "third_party/neuralgcm"))
from src.models.neuralgcm_residual.numerics import configure_environment
configure_environment()
from src.models.neuralgcm_residual.launcher import main

if __name__ == "__main__":
    main()
