"""Canonical inference engine for residual Mamba checkpoints."""

from __future__ import annotations

from pathlib import Path

from src.models.graphcast.runtime import infer_family, load_checkpoint_and_stats, load_run_config
from src.models.mamba.residual_mamba.runtime import (
    MODEL_NAME,
    build_run_jitted,
    build_truth_anchored_residual_runner,
)


class ResidualMambaInferenceEngine:
    family = MODEL_NAME

    def __init__(self, ckpt_path: str | Path, stats_dir: str | Path):
        self.ckpt_path = Path(ckpt_path)
        self.stats_dir = Path(stats_dir)
        self._runner = None
        self._truth_anchored = None
        self.task_cfg = None
        self.model_cfg = None
        self.run_cfg = None

    def load(self) -> "ResidualMambaInferenceEngine":
        ckpt_obj, stats = load_checkpoint_and_stats(self.ckpt_path, self.stats_dir)
        self.run_cfg = load_run_config(self.ckpt_path)
        family = infer_family(self.run_cfg)
        if family != self.family:
            raise ValueError(f"Checkpoint family mismatch: expected {self.family}, found {family}")
        self._runner, self.task_cfg, self.model_cfg, self.run_cfg = build_run_jitted(
            ckpt_obj, stats, self.ckpt_path
        )
        self._truth_anchored = build_truth_anchored_residual_runner(ckpt_obj, stats, self.ckpt_path)
        return self

    def rollout(self, *, rng, inputs, targets_template, forcings):
        if self._runner is None:
            self.load()
        return self._runner(
            rng=rng,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )

    def truth_anchored_runner(self):
        if self._truth_anchored is None:
            self.load()
        return self._truth_anchored
