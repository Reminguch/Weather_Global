"""Versioned JSON and CLI contract for v24_Ilya training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import (
    ARCHITECTURE_ID,
    DEFAULT_STATS_DIR,
    SCHEMA_VERSION,
    V24IlyaArchitectureConfig,
)


FEEDBACK_MODES = ("baseline", "closed_loop_sg")
PRECISIONS = ("bf16", "fp32")
TEMPORAL_STATE_POLICIES = ("carry", "reset_every_anchor")
LOSS_MODES = ("last_step", "all_steps", "sparse_steps")
LEARNING_RATE_SCHEDULES = ("constant", "cosine")
WEATHER_TAPE_PRECISIONS = ("fp32",)
BPTT_BACKEND = "explicit_reverse_vjp"
DISTRIBUTED_MODES = ("single", "data_parallel")
WEIGHT_DECAY_POLICIES = ("all", "mamba_standard")


@dataclass(frozen=True)
class V24IlyaDistributedConfig:
    mode: str = "single"
    num_devices: int = 1
    per_device_batch_size: int = 1
    drop_incomplete_replica_group: bool = True

    def __post_init__(self) -> None:
        if self.mode not in DISTRIBUTED_MODES:
            raise ValueError(
                f"distributed.mode must be one of {DISTRIBUTED_MODES}, got {self.mode!r}"
            )
        if type(self.num_devices) is not int or self.num_devices <= 0:
            raise ValueError("distributed.num_devices must be a positive integer")
        if self.mode == "single" and self.num_devices != 1:
            raise ValueError("distributed.mode='single' requires num_devices=1")
        if self.mode == "data_parallel" and self.num_devices < 2:
            raise ValueError("distributed.mode='data_parallel' requires num_devices>=2")
        if type(self.per_device_batch_size) is not int or self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be a positive integer")
        if self.mode == "data_parallel" and self.per_device_batch_size != 1:
            raise ValueError(
                "v24_Ilya data parallelism requires per_device_batch_size=1"
            )
        if not isinstance(self.drop_incomplete_replica_group, bool):
            raise ValueError(
                "distributed.drop_incomplete_replica_group must be boolean"
            )
        if self.per_device_batch_size > 1 and not self.drop_incomplete_replica_group:
            raise ValueError("Single-GPU batching requires drop_incomplete_replica_group=true")
        if self.mode == "data_parallel" and not self.drop_incomplete_replica_group:
            raise ValueError(
                "v24_Ilya data parallelism currently requires "
                "drop_incomplete_replica_group=true"
            )

    @property
    def global_batch_size(self) -> int:
        return self.num_devices * self.per_device_batch_size

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class V24IlyaValidationConfig:
    enabled: bool = False
    every_steps: int = 2_000
    num_segments: int | None = 16
    final_num_segments: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"validation.enabled must be boolean, got {self.enabled!r}")
        if type(self.every_steps) is not int or self.every_steps <= 0:
            raise ValueError(
                f"validation.every_steps must be a positive integer, got "
                f"{self.every_steps!r}"
            )
        for name in ("num_segments", "final_num_segments"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(
                    f"validation.{name} must be a positive integer or null, "
                    f"got {value!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class V24IlyaTrainConfig:
    prepared_root: Path
    anchor_manifest_root: Path
    baseline_checkpoint: Path
    output_root: Path
    run_name: str
    architecture: V24IlyaArchitectureConfig
    stats_dir: Path = Path(DEFAULT_STATS_DIR)
    input_duration: str = "12h"
    target_steps: int = 1
    time_start: str | None = None
    time_end: str | None = None
    allow_incomplete_prepared_store: bool = False
    segment_steps: int = 64
    bptt_steps: int = 16
    ar_tail_k: int = 12
    feedback_mode: str = "baseline"
    temporal_state_policy: str = "carry"
    loss_mode: str = "last_step"
    supervised_horizons: tuple[int, ...] = ()
    supervised_weights: tuple[float, ...] = ()
    weather_tape_precision: str = "fp32"
    max_steps: int = 50_000
    checkpoint_every: int = 2_000
    learning_rate: float = 1e-4
    mamba_lr_multiplier: float = 1.0
    spatial_lr_multiplier: float = 1.0
    learning_rate_schedule: str = "constant"
    end_learning_rate: float | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-4
    weight_decay_policy: str = "all"
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 18
    precision: str = "bf16"
    distributed: V24IlyaDistributedConfig = field(
        default_factory=V24IlyaDistributedConfig
    )
    validation: V24IlyaValidationConfig = field(
        default_factory=V24IlyaValidationConfig
    )

    def __post_init__(self) -> None:
        for name in ("mamba_lr_multiplier", "spatial_lr_multiplier"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.run_name or self.run_name in {".", ".."}:
            raise ValueError("run_name must be a non-empty directory name")
        if Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must not contain directory separators")
        if self.target_steps != 1:
            raise ValueError("target_steps must be 1 for v24_Ilya BPTT training")
        if (self.time_start is None) != (self.time_end is None):
            raise ValueError("data.time_start and data.time_end must be provided together")
        for name in ("time_start", "time_end"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"data.{name} must be a non-empty string or null")
        if not isinstance(self.allow_incomplete_prepared_store, bool):
            raise ValueError("data.allow_incomplete_prepared_store must be boolean")
        if self.allow_incomplete_prepared_store and self.time_start is None:
            raise ValueError(
                "data.allow_incomplete_prepared_store=true requires data.time_start "
                "and data.time_end"
            )
        for name in ("segment_steps", "bptt_steps", "max_steps", "checkpoint_every"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.segment_steps % self.bptt_steps:
            raise ValueError(
                f"segment_steps={self.segment_steps} must be divisible by "
                f"bptt_steps={self.bptt_steps}"
            )
        if not 0 <= self.ar_tail_k <= self.bptt_steps - 1:
            raise ValueError(
                f"ar_tail_k must be in [0, {self.bptt_steps - 1}], got {self.ar_tail_k}"
            )
        if self.feedback_mode not in FEEDBACK_MODES:
            raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}")
        if self.temporal_state_policy not in TEMPORAL_STATE_POLICIES:
            raise ValueError(
                "temporal_state_policy must be one of "
                f"{TEMPORAL_STATE_POLICIES}, got {self.temporal_state_policy!r}"
            )
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if self.loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {LOSS_MODES}")
        if self.loss_mode == "sparse_steps":
            horizons = tuple(self.supervised_horizons)
            weights = tuple(self.supervised_weights)
            if not horizons:
                raise ValueError(
                    "sparse_steps requires at least one supervised_horizon"
                )
            if len(horizons) != len(weights):
                raise ValueError(
                    "supervised_horizons and supervised_weights must have "
                    "matching lengths"
                )
            if any(type(horizon) is not int for horizon in horizons):
                raise ValueError("supervised_horizons must contain integers")
            if any(horizon <= 0 for horizon in horizons):
                raise ValueError("supervised_horizons must be positive")
            if any(left >= right for left, right in zip(horizons, horizons[1:])):
                raise ValueError(
                    "supervised_horizons must be ordered and unique"
                )
            endpoint_horizon = self.ar_tail_k + 1
            if horizons[-1] != endpoint_horizon:
                raise ValueError(
                    "sparse_steps must supervise the final AR endpoint "
                    f"horizon {endpoint_horizon}"
                )
            if any(
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(float(weight))
                or float(weight) <= 0
                for weight in weights
            ):
                raise ValueError(
                    "supervised_weights must contain positive finite numbers"
                )
        elif self.supervised_horizons or self.supervised_weights:
            raise ValueError(
                "supervised_horizons and supervised_weights are only valid for "
                "loss_mode='sparse_steps'"
            )
        if self.weather_tape_precision != "fp32":
            raise ValueError(
                "weather_tape_precision must be 'fp32' in v24_Ilya; raw physical "
                "BF16 weather trajectories are unsupported because they quantize "
                "GraphCast inputs and autoregressive feedback before normalization"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.learning_rate_schedule not in LEARNING_RATE_SCHEDULES:
            raise ValueError(
                "learning_rate_schedule must be one of "
                f"{LEARNING_RATE_SCHEDULES}, got {self.learning_rate_schedule!r}"
            )
        if self.learning_rate_schedule == "constant":
            if self.end_learning_rate is not None:
                raise ValueError(
                    "constant learning_rate_schedule requires end_learning_rate "
                    "to be null or omitted"
                )
        else:
            if self.learning_rate <= 0:
                raise ValueError("cosine learning_rate_schedule requires learning_rate > 0")
            if self.warmup_steps >= self.max_steps:
                raise ValueError(
                    "cosine learning_rate_schedule requires warmup_steps < max_steps"
                )
            if self.end_learning_rate is None:
                raise ValueError(
                    "cosine learning_rate_schedule requires end_learning_rate"
                )
            if (
                not math.isfinite(float(self.end_learning_rate))
                or not 0 <= self.end_learning_rate <= self.learning_rate
            ):
                raise ValueError(
                    "end_learning_rate must be finite and between zero and "
                    f"learning_rate={self.learning_rate}, got {self.end_learning_rate}"
                )
        for name in ("adam_beta1", "adam_beta2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1), got {value}")
        if self.weight_decay_policy not in WEIGHT_DECAY_POLICIES:
            raise ValueError(
                "weight_decay_policy must be one of "
                f"{WEIGHT_DECAY_POLICIES}, got {self.weight_decay_policy!r}"
            )

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def truth_prefix_steps(self) -> int:
        return self.bptt_steps - self.ar_tail_k

    @property
    def supervised_step_indices(self) -> tuple[int, ...]:
        """Internal BPTT indices whose targets contribute to the objective."""

        if self.loss_mode == "last_step":
            return (self.bptt_steps - 1,)
        if self.loss_mode == "all_steps":
            return tuple(range(self.bptt_steps))
        return tuple(
            self.truth_prefix_steps + horizon - 2
            for horizon in self.supervised_horizons
        )

    @property
    def normalized_supervised_weights(self) -> tuple[float, ...]:
        """Loss coefficients in the same order as ``supervised_step_indices``."""

        if self.loss_mode == "last_step":
            return (1.0,)
        if self.loss_mode == "all_steps":
            return (1.0 / self.bptt_steps,) * self.bptt_steps
        total = math.fsum(float(weight) for weight in self.supervised_weights)
        return tuple(float(weight) / total for weight in self.supervised_weights)

    @property
    def supervised_horizon_labels(self) -> tuple[int, ...]:
        """Forecast-horizon labels corresponding to recorded component losses."""

        if self.loss_mode == "sparse_steps":
            return tuple(self.supervised_horizons)
        return tuple(
            index - self.truth_prefix_steps + 2
            for index in self.supervised_step_indices
        )

    def _objective_dict(self) -> dict[str, Any]:
        objective: dict[str, Any] = {"loss_mode": self.loss_mode}
        if self.loss_mode == "sparse_steps":
            objective.update(
                supervised_horizons=list(self.supervised_horizons),
                supervised_weights=list(self.supervised_weights),
            )
        return objective

    def _optimizer_dict(self) -> dict[str, Any]:
        optimizer = {
            "max_steps": self.max_steps,
            "checkpoint_every": self.checkpoint_every,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "weight_decay_policy": self.weight_decay_policy,
            "warmup_steps": self.warmup_steps,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "precision": self.precision,
        }
        if self.mamba_lr_multiplier != 1.0 or self.spatial_lr_multiplier != 1.0:
            optimizer.update(mamba_lr_multiplier=self.mamba_lr_multiplier,
                             spatial_lr_multiplier=self.spatial_lr_multiplier)
        legacy_defaults = (
            self.learning_rate_schedule == "constant"
            and self.end_learning_rate is None
            and self.adam_beta1 == 0.9
            and self.adam_beta2 == 0.999
        )
        if not legacy_defaults:
            optimizer.update(
                learning_rate_schedule=self.learning_rate_schedule,
                end_learning_rate=self.end_learning_rate,
                adam_beta1=self.adam_beta1,
                adam_beta2=self.adam_beta2,
            )
        return optimizer

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": ARCHITECTURE_ID,
            "schema_version": SCHEMA_VERSION,
            "data": {
                "prepared_root": str(self.prepared_root),
                "anchor_manifest_root": str(self.anchor_manifest_root),
                "baseline_checkpoint": str(self.baseline_checkpoint),
                "stats_dir": str(self.stats_dir),
                "input_duration": self.input_duration,
                "target_steps": self.target_steps,
                "time_start": self.time_start,
                "time_end": self.time_end,
                "allow_incomplete_prepared_store": self.allow_incomplete_prepared_store,
            },
            "architecture": dataclasses.asdict(self.architecture),
            "sequence": {
                "segment_steps": self.segment_steps,
                "bptt_steps": self.bptt_steps,
                "ar_tail_k": self.ar_tail_k,
                "feedback_mode": self.feedback_mode,
                "temporal_state_policy": self.temporal_state_policy,
            },
            "objective": self._objective_dict(),
            "memory": {
                "weather_tape_precision": self.weather_tape_precision,
                "bptt_backend": BPTT_BACKEND,
            },
            "optimizer": self._optimizer_dict(),
            "distributed": self.distributed.to_dict(),
            "validation": self.validation.to_dict(),
            "output": {
                "output_root": str(self.output_root),
                "run_name": self.run_name,
            },
        }


@dataclass(frozen=True)
class V24IlyaTrainInvocation:
    config: V24IlyaTrainConfig
    config_path: Path
    resume: Path | None = None
    init_from: Path | None = None
    dry_run: bool = False
    baseline_validation_only: bool = False
    validation_compare: Path | None = None


def _section(payload: Mapping[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Training config requires a {name!r} object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise ValueError(f"Unknown keys in {name}: {extra}")
    return dict(value)


def load_training_config(path: Path) -> V24IlyaTrainConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read training config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Training config must be a JSON object")
    extra_top = sorted(
        set(payload)
        - {
            "architecture_id",
            "schema_version",
            "data",
            "architecture",
            "sequence",
            "optimizer",
            "objective",
            "memory",
            "distributed",
            "validation",
            "output",
        }
    )
    if extra_top:
        raise ValueError(f"Unknown top-level training config keys: {extra_top}")
    if payload.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(f"architecture_id must be {ARCHITECTURE_ID!r}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    data = _section(
        payload,
        "data",
        {
            "prepared_root",
            "anchor_manifest_root",
            "baseline_checkpoint",
            "stats_dir",
            "input_duration",
            "target_steps",
            "time_start",
            "time_end",
            "allow_incomplete_prepared_store",
        },
    )
    architecture_values = _section(
        payload,
        "architecture",
        {field.name for field in dataclasses.fields(V24IlyaArchitectureConfig)},
    )
    sequence = _section(
        payload,
        "sequence",
        {
            "segment_steps",
            "bptt_steps",
            "ar_tail_k",
            "feedback_mode",
            "temporal_state_policy",
        },
    )
    optimizer = _section(
        payload,
        "optimizer",
        {
            "max_steps",
            "checkpoint_every",
            "learning_rate",
            "mamba_lr_multiplier",
            "spatial_lr_multiplier",
            "learning_rate_schedule",
            "end_learning_rate",
            "adam_beta1",
            "adam_beta2",
            "weight_decay",
            "weight_decay_policy",
            "warmup_steps",
            "grad_clip",
            "seed",
            "precision",
        },
    )
    objective = _section(
        payload,
        "objective",
        {"loss_mode", "supervised_horizons", "supervised_weights"},
    )
    memory = _section(
        payload,
        "memory",
        {"weather_tape_precision", "bptt_backend"},
    )
    backend = memory.pop("bptt_backend", BPTT_BACKEND)
    if backend != BPTT_BACKEND:
        raise ValueError(
            f"memory.bptt_backend must be {BPTT_BACKEND!r}, got {backend!r}"
        )
    if "distributed" in payload:
        distributed_values = _section(
            payload,
            "distributed",
            {
                "mode",
                "num_devices",
                "per_device_batch_size",
                "drop_incomplete_replica_group",
            },
        )
    else:
        distributed_values = {}
    if "validation" in payload:
        validation_values = _section(
            payload,
            "validation",
            {"enabled", "every_steps", "num_segments", "final_num_segments"},
        )
    else:
        validation_values = {}
    output = _section(payload, "output", {"output_root", "run_name"})
    required = {
        "data": (data, {"prepared_root", "anchor_manifest_root", "baseline_checkpoint"}),
        "architecture": (architecture_values, set()),
        "sequence": (sequence, set()),
        "optimizer": (optimizer, set()),
        "objective": (objective, {"loss_mode"}),
        "memory": (memory, {"weather_tape_precision"}),
        "output": (output, {"output_root", "run_name"}),
    }
    for section_name, (values, names) in required.items():
        missing = sorted(names - set(values))
        if missing:
            raise ValueError(f"Missing required keys in {section_name}: {missing}")

    path_keys = ("prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir")
    for name in path_keys:
        if name in data:
            data[name] = Path(data[name])
    output["output_root"] = Path(output["output_root"])
    for name in ("supervised_horizons", "supervised_weights"):
        if name in objective:
            value = objective[name]
            if not isinstance(value, list):
                raise ValueError(f"objective.{name} must be a JSON array")
            objective[name] = tuple(value)
    return V24IlyaTrainConfig(
        **data,
        architecture=V24IlyaArchitectureConfig(**architecture_values),
        distributed=V24IlyaDistributedConfig(**distributed_values),
        validation=V24IlyaValidationConfig(**validation_values),
        **sequence,
        **objective,
        **memory,
        **optimizer,
        **output,
    )


def apply_operational_overrides(
    config: V24IlyaTrainConfig,
    *,
    max_steps: int | None,
    checkpoint_every: int | None,
    output_root: Path | None,
    run_name: str | None,
) -> V24IlyaTrainConfig:
    replacements: dict[str, Any] = {}
    for name, value in (
        ("max_steps", max_steps),
        ("checkpoint_every", checkpoint_every),
        ("output_root", output_root),
        ("run_name", run_name),
    ):
        if value is not None:
            replacements[name] = value
    return dataclasses.replace(config, **replacements)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train maintained v24_Ilya residual-Mamba.")
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--init-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-validation-only",
        action="store_true",
        help=(
            "Run the fixed validation protocol once with freshly initialized "
            "zero-output residual parameters, write baseline_validation.json, "
            "and exit without an optimizer update."
        ),
    )
    parser.add_argument(
        "--validation-compare",
        type=Path,
        metavar="CHECKPOINT",
        help=(
            "Run fixed validation for both the fresh zero-output baseline and "
            "CHECKPOINT, write validation_comparison.json, and exit."
        ),
    )
    return parser


def parse_cli(argv: Sequence[str] | None = None) -> V24IlyaTrainInvocation:
    args = build_cli_parser().parse_args(argv)
    config = load_training_config(args.config)
    config = apply_operational_overrides(
        config,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    return V24IlyaTrainInvocation(
        config=config,
        config_path=args.config,
        resume=args.resume,
        init_from=args.init_from,
        dry_run=args.dry_run,
        baseline_validation_only=args.baseline_validation_only,
        validation_compare=args.validation_compare,
    )


def validate_resume_config(
    current: V24IlyaTrainConfig,
    saved: Mapping[str, Any],
    *,
    completed_step: int,
) -> None:
    candidate = current.to_dict()
    saved_copy = json.loads(json.dumps(saved))
    saved_copy.setdefault("distributed", V24IlyaDistributedConfig().to_dict())
    saved_architecture = saved_copy.get("architecture")
    saved_data = saved_copy.get("data")
    if isinstance(saved_data, dict):
        saved_data.setdefault("time_start", None)
        saved_data.setdefault("time_end", None)
        saved_data.setdefault("allow_incomplete_prepared_store", False)

    if isinstance(saved_architecture, dict):
        saved_architecture.setdefault("residual_width", None)
        saved_architecture.setdefault("residual_initialization", "baseline_overlay")
        # This no-op field was recorded by early v24_Ilya checkpoints but was
        # never consumed by the full-Mamba implementation.
        saved_architecture.pop("temporal_hidden_size", None)
        saved_architecture.setdefault("temporal_bc_groups", 1)
        saved_architecture.setdefault("temporal_init_scheme", "legacy_haiku")
        saved_architecture.setdefault("temporal_dt_init", "random")
        saved_architecture.setdefault("temporal_dt_min", 0.001)
        saved_architecture.setdefault("temporal_dt_max", 0.1)
        saved_architecture.setdefault("temporal_dt_scale", 1.0)
        saved_architecture.setdefault("temporal_dt_init_floor", 1e-4)
    saved_sequence = saved_copy.get("sequence")
    if isinstance(saved_sequence, dict):
        # Checkpoints written before the explicit state-policy field always
        # used the legacy carry behavior.  For stateless architectures this
        # carried an empty Haiku state tree and was therefore a no-op.
        saved_sequence.setdefault("temporal_state_policy", "carry")
    saved_optimizer = saved_copy.get("optimizer")
    if isinstance(saved_optimizer, dict):
        saved_optimizer.setdefault("weight_decay_policy", "all")
        saved_optimizer.setdefault("learning_rate_schedule", "constant")
        saved_optimizer.setdefault("end_learning_rate", None)
        saved_optimizer.setdefault("adam_beta1", 0.9)
        saved_optimizer.setdefault("adam_beta2", 0.999)
    # Validation is operational and must not perturb exact-resume training.
    # Older checkpoints predate this optional section, and resumed runs may
    # safely change its cadence or subset size.
    candidate.pop("validation", None)
    saved_copy.pop("validation", None)
    try:
        saved_max_steps = int(saved_copy["optimizer"]["max_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Resume checkpoint has a malformed resolved_training_config"
        ) from exc
    for payload in (candidate, saved_copy):
        optimizer = payload["optimizer"]
        optimizer.pop("max_steps", None)
        optimizer.pop("checkpoint_every", None)
    if candidate != saved_copy:
        raise ValueError(
            "Resume configuration differs from the checkpoint. Only max_steps and "
            "checkpoint_every may change."
        )
    if current.max_steps < saved_max_steps:
        raise ValueError(
            f"max_steps may only increase on resume: saved={saved_max_steps}, "
            f"requested={current.max_steps}"
        )
    if current.max_steps <= completed_step:
        raise ValueError(
            f"max_steps={current.max_steps} must exceed completed_step={completed_step}"
        )
