"""Configuration and CLI contract for v22_final evaluation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ARCHITECTURE_ID = "v22_final"
SCHEMA_VERSION = 1

DEFAULT_DATA_PATH = "data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
DEFAULT_STATS_DIR = "data/graphcast/graphcast/stats"
DEFAULT_BASELINE_CHECKPOINT = (
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
    "mesh 2to5 - precipitation input and output.npz"
)

EVAL_MODES = (
    "cold_bp",
    "cold_full",
    "warm_bp",
    "warm_full",
    "warm_bp_reset_state",
    "warm_full_reset_state",
)
RESIDUAL_STATE_INIT_MODES = ("zero", "ckpt", "warm24")


@dataclass(frozen=True)
class V22FinalArchitectureConfig:
    """Architecture shared by v22_final training and evaluation."""

    resolution: float = 1.0
    mesh_size: int = 5
    width: int = 512
    baseline_msg_steps: int = 16
    residual_msg_steps: int = 2
    temporal_location: str = "mesh_processor_interleaved"
    temporal_hidden_size: int = 128
    temporal_d_inner: int | None = None
    temporal_d_state: int = 16
    temporal_d_conv: int = 4
    temporal_dt_rank: str = "auto"
    temporal_layers: int = 2
    temporal_conv_bias: bool = True
    temporal_stateful: bool = True
    temporal_zero_init_out: bool = True
    temporal_bias: bool = False
    temporal_dropout: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "mesh_size": self.mesh_size,
            "width": self.width,
            "baseline_msg_steps": self.baseline_msg_steps,
            "residual_msg_steps": self.residual_msg_steps,
            "temporal_hidden_size": self.temporal_hidden_size,
            "temporal_d_state": self.temporal_d_state,
            "temporal_d_conv": self.temporal_d_conv,
            "temporal_layers": self.temporal_layers,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.temporal_d_inner is not None and self.temporal_d_inner <= 0:
            raise ValueError(
                f"temporal_d_inner must be positive or None, got {self.temporal_d_inner}"
            )
        if self.temporal_location not in (
            "mesh_post_encoder",
            "mesh_processor_interleaved",
        ):
            raise ValueError(
                "temporal_location must be 'mesh_post_encoder' or "
                f"'mesh_processor_interleaved', got {self.temporal_location!r}"
            )
        if not 0.0 <= self.temporal_dropout < 1.0:
            raise ValueError(
                f"temporal_dropout must be in [0, 1), got {self.temporal_dropout}"
            )


@dataclass(frozen=True)
class V22FinalEvalConfig:
    """Complete, validated configuration for one v22_final evaluation."""

    ckpt: Path
    out_json: Path
    eval_mode: str
    data_path: str = DEFAULT_DATA_PATH
    stats_dir: Path = Path(DEFAULT_STATS_DIR)
    ckpt_in: Path = Path(DEFAULT_BASELINE_CHECKPOINT)
    resolution: float = 1.0
    mesh_size: int = 5
    width: int = 512
    baseline_msg_steps: int = 16
    residual_msg_steps: int = 2
    val_year: int = 2022
    train_start_year: int | None = 2020
    train_end_year: int | None = 2021
    input_duration: str | None = "12h"
    target_steps: int = 40
    warmup_steps: int = 24
    temporal_location: str = "mesh_processor_interleaved"
    temporal_hidden_size: int = 128
    temporal_d_inner: int | None = None
    temporal_d_state: int = 16
    temporal_d_conv: int = 4
    temporal_dt_rank: str = "auto"
    temporal_layers: int = 2
    temporal_conv_bias: bool = True
    temporal_stateful: bool = True
    temporal_zero_init_out: bool = True
    temporal_bias: bool = False
    temporal_dropout: float = 0.0
    n_samples: int = 32
    residual_state_init: str = "zero"
    seed: int = 0
    residual_alpha: float = 1.0
    force_idx: int | None = None

    def __post_init__(self) -> None:
        # Preserve the flat legacy evaluator interface while validating its
        # architecture through the object shared with training.
        self.architecture
        if self.eval_mode not in EVAL_MODES:
            raise ValueError(f"eval_mode must be one of {EVAL_MODES}, got {self.eval_mode!r}")
        if self.residual_state_init not in RESIDUAL_STATE_INIT_MODES:
            raise ValueError(
                "residual_state_init must be one of "
                f"{RESIDUAL_STATE_INIT_MODES}, got {self.residual_state_init!r}"
            )
        positive = {
            "target_steps": self.target_steps,
            "n_samples": self.n_samples,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if not math.isfinite(self.residual_alpha):
            raise ValueError(f"residual_alpha must be finite, got {self.residual_alpha}")
        if self.force_idx is not None and self.force_idx < 0:
            raise ValueError(f"force_idx must be non-negative, got {self.force_idx}")
        if (
            self.train_start_year is not None
            and self.train_end_year is not None
            and self.train_start_year > self.train_end_year
        ):
            raise ValueError(
                f"train_start_year={self.train_start_year} exceeds "
                f"train_end_year={self.train_end_year}"
            )

    @property
    def architecture(self) -> V22FinalArchitectureConfig:
        return V22FinalArchitectureConfig(
            resolution=self.resolution,
            mesh_size=self.mesh_size,
            width=self.width,
            baseline_msg_steps=self.baseline_msg_steps,
            residual_msg_steps=self.residual_msg_steps,
            temporal_location=self.temporal_location,
            temporal_hidden_size=self.temporal_hidden_size,
            temporal_d_inner=self.temporal_d_inner,
            temporal_d_state=self.temporal_d_state,
            temporal_d_conv=self.temporal_d_conv,
            temporal_dt_rank=self.temporal_dt_rank,
            temporal_layers=self.temporal_layers,
            temporal_conv_bias=self.temporal_conv_bias,
            temporal_stateful=self.temporal_stateful,
            temporal_zero_init_out=self.temporal_zero_init_out,
            temporal_bias=self.temporal_bias,
            temporal_dropout=self.temporal_dropout,
        )

    @property
    def is_warm(self) -> bool:
        return self.eval_mode.startswith("warm_")

    @property
    def reset_state_after_warmup(self) -> bool:
        return self.eval_mode.endswith("_reset_state")

    @property
    def is_full_feedback(self) -> bool:
        return self.eval_mode.removesuffix("_reset_state").endswith("_full")

    @property
    def effective_warmup_steps(self) -> int:
        return self.warmup_steps if self.is_warm else 0

    @property
    def total_rollout_steps(self) -> int:
        return self.effective_warmup_steps + self.target_steps

    @property
    def sample_total_steps(self) -> int:
        """Anchor horizon shared by cold and warm modes."""

        return self.warmup_steps + self.target_steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the frozen v22_final residual-Mamba model.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--stats-dir", type=Path, default=Path(DEFAULT_STATS_DIR))
    parser.add_argument("--ckpt-in", type=Path, default=Path(DEFAULT_BASELINE_CHECKPOINT))
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--mesh-size", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--baseline-msg-steps", type=int, default=16)
    parser.add_argument("--residual-msg-steps", type=int, default=2)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--train-start-year", type=int, default=2020)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--input-duration", default="12h")
    parser.add_argument("--target-steps", type=int, default=40, help="Metric horizon.")
    parser.add_argument("--warmup-steps", type=int, default=24)
    parser.add_argument("--eval-mode", choices=EVAL_MODES, required=True)
    parser.add_argument("--temporal-location", default="mesh_processor_interleaved")
    parser.add_argument("--temporal-hidden-size", type=int, default=128)
    parser.add_argument("--temporal-d-inner", type=int, default=None)
    parser.add_argument("--temporal-d-state", type=int, default=16)
    parser.add_argument("--temporal-d-conv", type=int, default=4)
    parser.add_argument("--temporal-dt-rank", default="auto")
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument(
        "--no-temporal-conv-bias",
        dest="temporal_conv_bias",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--temporal-stateful",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Carry Mamba SSM and convolution state between autoregressive calls.",
    )
    parser.add_argument(
        "--no-zero-init-out",
        dest="temporal_zero_init_out",
        action="store_false",
        default=True,
    )
    parser.add_argument("--temporal-bias", action="store_true", default=False)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument(
        "--residual-state-init",
        choices=RESIDUAL_STATE_INIT_MODES,
        default="zero",
        help="zero (default), ckpt, or legacy warm24 alias for zero followed by warmup.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-alpha", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--force-idx", type=int, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> V22FinalEvalConfig:
    namespace = build_arg_parser().parse_args(argv)
    return V22FinalEvalConfig(**vars(namespace))
