"""Versioned JSON and CLI contract for v23_Ilya training."""

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
    V23IlyaArchitectureConfig,
)


FEEDBACK_MODES = ("baseline", "closed_loop_sg")
PRECISIONS = ("bf16", "fp32")
TEMPORAL_STATE_POLICIES = ("carry", "reset_every_anchor")
LOSS_MODES = ("last_step", "all_steps")
WEATHER_TAPE_PRECISIONS = ("bf16", "fp32")
BPTT_BACKEND = "explicit_reverse_vjp"
DISTRIBUTED_MODES = ("single", "data_parallel")


@dataclass(frozen=True)
class V23IlyaDistributedConfig:
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
        if self.per_device_batch_size != 1:
            raise ValueError(
                "v23_Ilya data parallelism requires per_device_batch_size=1"
            )
        if not isinstance(self.drop_incomplete_replica_group, bool):
            raise ValueError(
                "distributed.drop_incomplete_replica_group must be boolean"
            )
        if self.mode == "data_parallel" and not self.drop_incomplete_replica_group:
            raise ValueError(
                "v23_Ilya data parallelism currently requires "
                "drop_incomplete_replica_group=true"
            )

    @property
    def global_batch_size(self) -> int:
        return self.num_devices * self.per_device_batch_size

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class V23IlyaValidationConfig:
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
class V23IlyaTrainConfig:
    prepared_root: Path
    anchor_manifest_root: Path
    baseline_checkpoint: Path
    output_root: Path
    run_name: str
    architecture: V23IlyaArchitectureConfig
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
    weather_tape_precision: str = "bf16"
    max_steps: int = 50_000
    checkpoint_every: int = 2_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 18
    precision: str = "bf16"
    distributed: V23IlyaDistributedConfig = field(
        default_factory=V23IlyaDistributedConfig
    )
    validation: V23IlyaValidationConfig = field(
        default_factory=V23IlyaValidationConfig
    )

    def __post_init__(self) -> None:
        if not self.run_name or self.run_name in {".", ".."}:
            raise ValueError("run_name must be a non-empty directory name")
        if Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must not contain directory separators")
        if self.target_steps != 1:
            raise ValueError("target_steps must be 1 for v23_Ilya BPTT training")
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
        if self.weather_tape_precision not in WEATHER_TAPE_PRECISIONS:
            raise ValueError(
                "weather_tape_precision must be one of "
                f"{WEATHER_TAPE_PRECISIONS}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def truth_prefix_steps(self) -> int:
        return self.bptt_steps - self.ar_tail_k

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
            "objective": {
                "loss_mode": self.loss_mode,
            },
            "memory": {
                "weather_tape_precision": self.weather_tape_precision,
                "bptt_backend": BPTT_BACKEND,
            },
            "optimizer": {
                "max_steps": self.max_steps,
                "checkpoint_every": self.checkpoint_every,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "grad_clip": self.grad_clip,
                "seed": self.seed,
                "precision": self.precision,
            },
            "distributed": self.distributed.to_dict(),
            "validation": self.validation.to_dict(),
            "output": {
                "output_root": str(self.output_root),
                "run_name": self.run_name,
            },
        }


@dataclass(frozen=True)
class V23IlyaTrainInvocation:
    config: V23IlyaTrainConfig
    config_path: Path
    resume: Path | None = None
    init_from: Path | None = None
    dry_run: bool = False


def _section(payload: Mapping[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Training config requires a {name!r} object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise ValueError(f"Unknown keys in {name}: {extra}")
    return dict(value)


def load_training_config(path: Path) -> V23IlyaTrainConfig:
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
        {field.name for field in dataclasses.fields(V23IlyaArchitectureConfig)},
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
        {"max_steps", "checkpoint_every", "learning_rate", "weight_decay", "warmup_steps", "grad_clip", "seed", "precision"},
    )
    objective = _section(payload, "objective", {"loss_mode"})
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
    return V23IlyaTrainConfig(
        **data,
        architecture=V23IlyaArchitectureConfig(**architecture_values),
        distributed=V23IlyaDistributedConfig(**distributed_values),
        validation=V23IlyaValidationConfig(**validation_values),
        **sequence,
        **objective,
        **memory,
        **optimizer,
        **output,
    )


def apply_operational_overrides(
    config: V23IlyaTrainConfig,
    *,
    max_steps: int | None,
    checkpoint_every: int | None,
    output_root: Path | None,
    run_name: str | None,
) -> V23IlyaTrainConfig:
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
    parser = argparse.ArgumentParser(description="Train maintained v23_Ilya residual-Mamba.")
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--init-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_cli(argv: Sequence[str] | None = None) -> V23IlyaTrainInvocation:
    args = build_cli_parser().parse_args(argv)
    config = load_training_config(args.config)
    config = apply_operational_overrides(
        config,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    return V23IlyaTrainInvocation(
        config=config,
        config_path=args.config,
        resume=args.resume,
        init_from=args.init_from,
        dry_run=args.dry_run,
    )


def validate_resume_config(
    current: V23IlyaTrainConfig,
    saved: Mapping[str, Any],
    *,
    completed_step: int,
) -> None:
    candidate = current.to_dict()
    saved_copy = json.loads(json.dumps(saved))
    saved_copy.setdefault("distributed", V23IlyaDistributedConfig().to_dict())
    saved_architecture = saved_copy.get("architecture")
    saved_data = saved_copy.get("data")
    if isinstance(saved_data, dict):
        saved_data.setdefault("time_start", None)
        saved_data.setdefault("time_end", None)
        saved_data.setdefault("allow_incomplete_prepared_store", False)

    if isinstance(saved_architecture, dict):
        # This no-op field was recorded by early v23_Ilya checkpoints but was
        # never consumed by the full-Mamba implementation.
        saved_architecture.pop("temporal_hidden_size", None)
        saved_architecture.setdefault("temporal_bc_groups", 1)
    saved_sequence = saved_copy.get("sequence")
    if isinstance(saved_sequence, dict):
        # Checkpoints written before the explicit state-policy field always
        # used the legacy carry behavior.  For stateless architectures this
        # carried an empty Haiku state tree and was therefore a no-op.
        saved_sequence.setdefault("temporal_state_policy", "carry")
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
