"""Configuration and CLI contract for v24_Ilya evaluation."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ARCHITECTURE_ID = "v24_Ilya"
SCHEMA_VERSION = 1

DEFAULT_DATA_PATH = "data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
DEFAULT_STATS_DIR = "data/graphcast/graphcast/stats"
DEFAULT_BASELINE_CHECKPOINT = (
    "data/graphcast/graphcast/params/"
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
class V24IlyaArchitectureConfig:
    """Architecture shared by v24_Ilya training and evaluation."""

    resolution: float = 1.0
    mesh_size: int = 5
    width: int = 512
    baseline_msg_steps: int = 16
    residual_msg_steps: int = 2
    temporal_location: str = "mesh_processor_interleaved"
    temporal_d_inner: int | None = None
    temporal_bc_groups: int = 1
    temporal_d_state: int = 16
    temporal_d_conv: int = 4
    temporal_dt_rank: str = "auto"
    temporal_init_scheme: str = "legacy_haiku"
    temporal_dt_init: str = "random"
    temporal_dt_min: float = 0.001
    temporal_dt_max: float = 0.1
    temporal_dt_scale: float = 1.0
    temporal_dt_init_floor: float = 1e-4
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
        if self.temporal_bc_groups <= 0:
            raise ValueError(
                f"temporal_bc_groups must be positive, got {self.temporal_bc_groups}"
            )
        if self.temporal_d_inner is None and self.temporal_bc_groups != 1:
            raise ValueError(
                "temporal_d_inner is required when temporal_bc_groups is not 1"
            )
        if self.temporal_d_inner is not None:
            if self.temporal_bc_groups > self.temporal_d_inner:
                raise ValueError(
                    "temporal_bc_groups must not exceed temporal_d_inner, got "
                    f"{self.temporal_bc_groups} > {self.temporal_d_inner}"
                )
            if self.temporal_d_inner % self.temporal_bc_groups:
                raise ValueError(
                    "temporal_d_inner must be divisible by temporal_bc_groups, got "
                    f"{self.temporal_d_inner} and {self.temporal_bc_groups}"
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
        if self.temporal_init_scheme not in ("legacy_haiku", "mamba1"):
            raise ValueError(
                "temporal_init_scheme must be 'legacy_haiku' or 'mamba1', got "
                f"{self.temporal_init_scheme!r}"
            )
        if self.temporal_dt_init not in ("random", "constant"):
            raise ValueError(
                "temporal_dt_init must be 'random' or 'constant', got "
                f"{self.temporal_dt_init!r}"
            )
        if not math.isfinite(self.temporal_dt_min) or self.temporal_dt_min <= 0:
            raise ValueError("temporal_dt_min must be positive and finite")
        if (
            not math.isfinite(self.temporal_dt_max)
            or self.temporal_dt_max < self.temporal_dt_min
        ):
            raise ValueError("temporal_dt_max must be finite and at least temporal_dt_min")
        if not math.isfinite(self.temporal_dt_scale) or self.temporal_dt_scale <= 0:
            raise ValueError("temporal_dt_scale must be positive and finite")
        if (
            not math.isfinite(self.temporal_dt_init_floor)
            or self.temporal_dt_init_floor <= 0
        ):
            raise ValueError("temporal_dt_init_floor must be positive and finite")
        if self.temporal_dt_init_floor > self.temporal_dt_max:
            raise ValueError("temporal_dt_init_floor must not exceed temporal_dt_max")


@dataclass(frozen=True)
class V24IlyaEvalConfig:
    """Complete, validated configuration for one v24_Ilya evaluation."""

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
    anchor_history_steps: int | None = None
    stream_block_steps: int = 4
    omit_rms_bias: bool = False
    temporal_location: str = "mesh_processor_interleaved"
    temporal_d_inner: int | None = None
    temporal_bc_groups: int = 1
    temporal_d_state: int = 16
    temporal_d_conv: int = 4
    temporal_dt_rank: str = "auto"
    temporal_init_scheme: str = "legacy_haiku"
    temporal_dt_init: str = "random"
    temporal_dt_min: float = 0.001
    temporal_dt_max: float = 0.1
    temporal_dt_scale: float = 1.0
    temporal_dt_init_floor: float = 1e-4
    temporal_layers: int = 2
    temporal_conv_bias: bool = True
    temporal_stateful: bool = True
    temporal_zero_init_out: bool = True
    temporal_bias: bool = False
    temporal_dropout: float = 0.0
    n_samples: int = 32
    residual_state_init: str = "zero"
    reset_state_every_step: bool = False
    seed: int = 0
    residual_alpha: float = 1.0
    force_idx: int | None = None
    anchor_shard_count: int = 1
    anchor_shard_index: int = 0
    merge_state_out: Path | None = None

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
            "anchor_shard_count": self.anchor_shard_count,
            "stream_block_steps": self.stream_block_steps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.anchor_history_steps is not None:
            if self.anchor_history_steps < 0:
                raise ValueError(
                    "anchor_history_steps must be non-negative or None, got "
                    f"{self.anchor_history_steps}"
                )
            if self.anchor_history_steps < self.effective_warmup_steps:
                raise ValueError(
                    "anchor_history_steps must be at least the effective warmup, got "
                    f"{self.anchor_history_steps} < {self.effective_warmup_steps}"
                )
        if not math.isfinite(self.residual_alpha):
            raise ValueError(f"residual_alpha must be finite, got {self.residual_alpha}")
        if self.force_idx is not None and self.force_idx < 0:
            raise ValueError(f"force_idx must be non-negative, got {self.force_idx}")
        if not 0 <= self.anchor_shard_index < self.anchor_shard_count:
            raise ValueError(
                "anchor_shard_index must be in [0, anchor_shard_count), got "
                f"{self.anchor_shard_index} for count {self.anchor_shard_count}"
            )
        if self.force_idx is not None and self.anchor_shard_count != 1:
            raise ValueError("force_idx cannot be combined with anchor sharding")
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
    def architecture(self) -> V24IlyaArchitectureConfig:
        return V24IlyaArchitectureConfig(
            resolution=self.resolution,
            mesh_size=self.mesh_size,
            width=self.width,
            baseline_msg_steps=self.baseline_msg_steps,
            residual_msg_steps=self.residual_msg_steps,
            temporal_location=self.temporal_location,
            temporal_d_inner=self.temporal_d_inner,
            temporal_bc_groups=self.temporal_bc_groups,
            temporal_d_state=self.temporal_d_state,
            temporal_d_conv=self.temporal_d_conv,
            temporal_dt_rank=self.temporal_dt_rank,
            temporal_init_scheme=self.temporal_init_scheme,
            temporal_dt_init=self.temporal_dt_init,
            temporal_dt_min=self.temporal_dt_min,
            temporal_dt_max=self.temporal_dt_max,
            temporal_dt_scale=self.temporal_dt_scale,
            temporal_dt_init_floor=self.temporal_dt_init_floor,
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

    @property
    def effective_anchor_history_steps(self) -> int:
        """History reserved when selecting matched scored forecast anchors."""

        if self.anchor_history_steps is None:
            return self.effective_warmup_steps
        return self.anchor_history_steps


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the frozen v24_Ilya residual-Mamba model.")
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
    parser.add_argument(
        "--anchor-history-steps",
        type=int,
        default=None,
        help=(
            "Select anchors by the scored H1 origin after reserving this many "
            "truth-history steps. Use the same value across a warmup sweep to "
            "score identical dates. Defaults to the effective warmup."
        ),
    )
    parser.add_argument(
        "--stream-block-steps",
        type=int,
        default=4,
        help="Load truth and forcing data in bounded blocks of this many steps.",
    )
    parser.add_argument(
        "--omit-rms-bias",
        action="store_true",
        default=False,
        help=(
            "Skip large spatial bias accumulators while retaining exact GraphCast "
            "loss and per-variable/per-level RMSE and MAE."
        ),
    )
    parser.add_argument("--eval-mode", choices=EVAL_MODES, required=True)
    parser.add_argument("--temporal-location", default="mesh_processor_interleaved")
    parser.add_argument("--temporal-d-inner", type=int, default=None)
    parser.add_argument("--temporal-bc-groups", type=int, default=1)
    parser.add_argument("--temporal-d-state", type=int, default=16)
    parser.add_argument("--temporal-d-conv", type=int, default=4)
    parser.add_argument("--temporal-dt-rank", default="auto")
    parser.add_argument(
        "--temporal-init-scheme",
        choices=("legacy_haiku", "mamba1"),
        default="legacy_haiku",
    )
    parser.add_argument(
        "--temporal-dt-init",
        choices=("random", "constant"),
        default="random",
    )
    parser.add_argument("--temporal-dt-min", type=float, default=0.001)
    parser.add_argument("--temporal-dt-max", type=float, default=0.1)
    parser.add_argument("--temporal-dt-scale", type=float, default=1.0)
    parser.add_argument("--temporal-dt-init-floor", type=float, default=1e-4)
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
    parser.add_argument(
        "--reset-state-every-step",
        action="store_true",
        default=False,
        help="Diagnostic: restore residual temporal state before every model step.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-alpha", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--force-idx", type=int, default=None)
    parser.add_argument(
        "--anchor-shard-count",
        type=int,
        default=1,
        help="Split the deterministically selected anchors into this many shards.",
    )
    parser.add_argument(
        "--anchor-shard-index",
        type=int,
        default=0,
        help="Zero-based round-robin anchor shard to evaluate.",
    )
    parser.add_argument(
        "--merge-state-out",
        type=Path,
        default=None,
        help="Optional host accumulator state used for exact cross-shard merging.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> V24IlyaEvalConfig:
    namespace = build_arg_parser().parse_args(argv)
    return V24IlyaEvalConfig(**vars(namespace))
