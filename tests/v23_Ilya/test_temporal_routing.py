from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))

from graphcast import graphcast as gc  # noqa: E402
from src.models.mamba.v23_Ilya.config import V23IlyaArchitectureConfig  # noqa: E402
from src.models.mamba.v23_Ilya.model import attach_temporal  # noqa: E402


def test_v22_uses_full_mamba_for_both_persistence_modes() -> None:
    for stateful in (False, True):
        predictor = object.__new__(gc.GraphCast)
        config = V23IlyaArchitectureConfig(
            temporal_d_inner=4,
            temporal_bc_groups=2,
            temporal_stateful=stateful,
            temporal_init_scheme="mamba1",
            temporal_dt_init="constant",
        )

        attach_temporal(predictor, config)

        assert predictor._temporal_stateful is stateful
        assert predictor._temporal_use_full_mamba is True
        assert predictor._temporal_bc_groups == 2
        assert predictor._temporal_init_scheme == "mamba1"
        assert predictor._temporal_dt_init == "constant"
        assert gc._get_temporal_block_cls(
            stateful,
            use_full_mamba=predictor._temporal_use_full_mamba,
        ) is gc._StatefulTemporalBlock


def test_legacy_stateless_routing_is_unchanged() -> None:
    assert gc._get_temporal_block_cls(False) is gc._StatelessTemporalBlock


def test_v22_defaults_to_one_bc_group() -> None:
    assert V23IlyaArchitectureConfig(temporal_d_inner=4).temporal_bc_groups == 1
