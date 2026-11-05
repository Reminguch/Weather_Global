"""Open-loop decoded-weather correction of frozen NeuralGCM (spec v2)."""
from .runtime import configure
configure()
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
# Reuse the repository's GraphCast and v24 implementations, never a pip fork.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/graphcast'))
