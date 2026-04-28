"""Legacy shared rollout helpers for analysis scripts."""

from __future__ import annotations

from pathlib import Path

from src.models.graphcast.runtime import (
    build_run_jitted as build_graphcast_run_jitted,
    infer_family,
    load_run_config,
    suppress_graphcast_future_warnings,
)
from src.models.mamba.gc_mamba.runtime import build_run_jitted as build_gc_mamba_run_jitted
from src.models.mamba.residual_mamba.runtime import (
    build_run_jitted as build_residual_mamba_run_jitted,
    build_truth_anchored_residual_runner,
)


def build_run_jitted(ckpt_obj, stats, ckpt_path: Path):
    family = infer_family(load_run_config(ckpt_path))
    if family == "graphcast":
        return build_graphcast_run_jitted(ckpt_obj, stats, ckpt_path)
    if family == "gc_mamba":
        return build_gc_mamba_run_jitted(ckpt_obj, stats, ckpt_path)
    if family == "residual_mamba":
        return build_residual_mamba_run_jitted(ckpt_obj, stats, ckpt_path)
    raise ValueError(f"Unknown inference family: {family}")

__all__ = [
    "build_run_jitted",
    "build_truth_anchored_residual_runner",
    "suppress_graphcast_future_warnings",
]
