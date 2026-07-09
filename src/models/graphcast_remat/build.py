"""build_predictor_remat: same signature as src.models.graphcast.training.core.model.build_predictor
but constructs the rematerialization-enabled subclass.

This does NOT modify third_party graphcast.py; it just swaps in the subclass.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GC_THIRD_PARTY = _REPO_ROOT / "third_party" / "graphcast"
if str(_GC_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_GC_THIRD_PARTY))

from graphcast import (  # noqa: E402
    autoregressive,
    casting,
    normalization,
)
from src.models.graphcast_remat import GraphCast as GraphCastRemat  # noqa: E402


def build_predictor_remat(
    model_cfg,
    task_cfg,
    stats,
    *,
    use_bf16: bool,
    gradient_checkpointing: bool,
    temporal_backbone: str,
    temporal_location: str,
    temporal_hidden_size: int,
    temporal_d_inner,
    temporal_d_state: int,
    temporal_d_conv: int,
    temporal_dt_rank,
    temporal_bias: bool,
    temporal_conv_bias: bool,
    temporal_layers: int,
    temporal_dropout: float,
    temporal_stateful: bool = False,
    zero_init_temporal_out: bool = False,
    # New remat flags (default False -> behaves identically to base build_predictor).
    # When remat_processor_steps=True, the SAFE Mode-2 is used by default:
    # processor msg-passing in hk.remat, Mamba OUTSIDE remat. The UNSAFE
    # Mode-1 (processor+Mamba in same remat) is only reachable by setting
    # the test-only `remat_temporal_inside_processor=True` kwarg below.
    remat_processor_steps: bool = False,
    remat_mesh2grid: bool = False,
    remat_grid2mesh: bool = False,
    # TEST-ONLY: True -> opt into UNSAFE Mode-1 (Mamba inside remat boundary).
    # Failed gradient verification.  Do NOT set in production code.
    remat_temporal_inside_processor: bool = False,
):
    """Build a predictor whose inner GraphCast is the remat-enabled subclass.

    Matches src.models.graphcast.training.core.model.build_predictor signature
    except for the three new remat flags appended at the end.
    """
    predictor = GraphCastRemat(model_cfg, task_cfg)

    # Set _temporal_* on the inner GraphCast (subclass inherits these attrs).
    if hasattr(predictor, "_temporal_backbone"):
        predictor._temporal_backbone = temporal_backbone
        predictor._temporal_location = temporal_location
        predictor._temporal_stateful = temporal_stateful
        predictor._temporal_hidden_size = temporal_hidden_size
        predictor._temporal_d_inner = temporal_d_inner
        predictor._temporal_d_state = temporal_d_state
        predictor._temporal_d_conv = temporal_d_conv
        predictor._temporal_dt_rank = temporal_dt_rank
        predictor._temporal_bias = temporal_bias
        predictor._temporal_conv_bias = temporal_conv_bias
        predictor._temporal_layers = temporal_layers
        predictor._temporal_dropout = temporal_dropout
        predictor._temporal_zero_init_out = zero_init_temporal_out

    # Wire remat flags onto the inner GraphCast.
    predictor._remat_processor_steps = remat_processor_steps
    predictor._remat_mesh2grid = remat_mesh2grid
    predictor._remat_grid2mesh = remat_grid2mesh
    predictor._remat_temporal_inside_processor = remat_temporal_inside_processor

    if use_bf16:
        predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )
    predictor = autoregressive.Predictor(
        predictor, gradient_checkpointing=gradient_checkpointing)
    return predictor


__all__ = ["build_predictor_remat"]
