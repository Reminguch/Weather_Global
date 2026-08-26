"""Versioned JSON and CLI contract for v22_final training."""

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
    V22FinalArchitectureConfig,
)


FEEDBACK_MODES = ("baseline", "closed_loop_sg")
PRECISIONS = ("bf16", "fp32")
TEMPORAL_STATE_POLICIES = ("carry", "reset_every_anchor")
LEARNING_RATE_SCHEDULES = ("constant", "cosine")


@dataclass(frozen=True)
class V22FinalValidationConfig:
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
class V22FinalTrainConfig:
    prepared_root: Path
    anchor_manifest_root: Path
    baseline_checkpoint: Path
    output_root: Path
    run_name: str
    architecture: V22FinalArchitectureConfig
    stats_dir: Path = Path(DEFAULT_STATS_DIR)
    input_duration: str = "12h"
    target_steps: int = 1
    segment_steps: int = 64
    bptt_steps: int = 16
    ar_tail_k: int = 12
    feedback_mode: str = "baseline"
    temporal_state_policy: str = "carry"
    max_steps: int = 50_000
    checkpoint_every: int = 2_000
    learning_rate: float = 1e-4
    learning_rate_schedule: str = "constant"
    decay_start_step: int | None = None
    end_learning_rate: float | None = None
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 18
    precision: str = "bf16"
    validation: V22FinalValidationConfig = field(
        default_factory=V22FinalValidationConfig
    )

    def __post_init__(self) -> None:
        if not self.run_name or self.run_name in {".", ".."}:
            raise ValueError("run_name must be a non-empty directory name")
        if Path(self.run_name).name != self.run_name:
            raise ValueError("run_name must not contain directory separators")
        if self.target_steps != 1:
            raise ValueError("target_steps must be 1 for v22_final BPTT training")
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
        if self.learning_rate_schedule not in LEARNING_RATE_SCHEDULES:
            raise ValueError(
                "learning_rate_schedule must be one of "
                f"{LEARNING_RATE_SCHEDULES}, got {self.learning_rate_schedule!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.learning_rate_schedule == "constant":
            if self.decay_start_step is not None or self.end_learning_rate is not None:
                raise ValueError(
                    "constant learning_rate_schedule requires decay_start_step and "
                    "end_learning_rate to be null or omitted"
                )
        else:
            if self.learning_rate <= 0:
                raise ValueError("cosine learning_rate_schedule requires learning_rate > 0")
            if self.decay_start_step is None:
                raise ValueError("cosine learning_rate_schedule requires decay_start_step")
            if not self.warmup_steps <= self.decay_start_step < self.max_steps:
                raise ValueError(
                    "decay_start_step must be between warmup_steps and max_steps-1, "
                    f"got {self.decay_start_step}"
                )
            if self.end_learning_rate is None:
                raise ValueError("cosine learning_rate_schedule requires end_learning_rate")
            if (
                not math.isfinite(float(self.end_learning_rate))
                or not 0 <= self.end_learning_rate <= self.learning_rate
            ):
                raise ValueError(
                    "end_learning_rate must be finite and between zero and "
                    f"learning_rate={self.learning_rate}, got {self.end_learning_rate}"
                )

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
            },
            "architecture": dataclasses.asdict(self.architecture),
            "sequence": {
                "segment_steps": self.segment_steps,
                "bptt_steps": self.bptt_steps,
                "ar_tail_k": self.ar_tail_k,
                "feedback_mode": self.feedback_mode,
                "temporal_state_policy": self.temporal_state_policy,
            },
            "optimizer": {
                "max_steps": self.max_steps,
                "checkpoint_every": self.checkpoint_every,
                "learning_rate": self.learning_rate,
                "learning_rate_schedule": self.learning_rate_schedule,
                "decay_start_step": self.decay_start_step,
                "end_learning_rate": self.end_learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "grad_clip": self.grad_clip,
                "seed": self.seed,
                "precision": self.precision,
            },
            "validation": self.validation.to_dict(),
            "output": {
                "output_root": str(self.output_root),
                "run_name": self.run_name,
            },
        }


@dataclass(frozen=True)
class V22FinalTrainInvocation:
    config: V22FinalTrainConfig
    config_path: Path
    resume: Path | None = None
    init_from: Path | None = None
    zero_state_on_init: bool = False
    dry_run: bool = False


def _section(payload: Mapping[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Training config requires a {name!r} object")
    extra = sorted(set(value) - allowed)
    if extra:
        raise ValueError(f"Unknown keys in {name}: {extra}")
    return dict(value)


def load_training_config(path: Path) -> V22FinalTrainConfig:
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
        {"prepared_root", "anchor_manifest_root", "baseline_checkpoint", "stats_dir", "input_duration", "target_steps"},
    )
    architecture_values = _section(
        payload,
        "architecture",
        {field.name for field in dataclasses.fields(V22FinalArchitectureConfig)},
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
            "learning_rate_schedule",
            "decay_start_step",
            "end_learning_rate",
            "weight_decay",
            "warmup_steps",
            "grad_clip",
            "seed",
            "precision",
        },
    )
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
    return V22FinalTrainConfig(
        **data,
        architecture=V22FinalArchitectureConfig(**architecture_values),
        validation=V22FinalValidationConfig(**validation_values),
        **sequence,
        **optimizer,
        **output,
    )


def apply_operational_overrides(
    config: V22FinalTrainConfig,
    *,
    max_steps: int | None,
    checkpoint_every: int | None,
    output_root: Path | None,
    run_name: str | None,
    learning_rate: float | None = None,
) -> V22FinalTrainConfig:
    replacements: dict[str, Any] = {}
    for name, value in (
        ("max_steps", max_steps),
        ("checkpoint_every", checkpoint_every),
        ("output_root", output_root),
        ("run_name", run_name),
        ("learning_rate", learning_rate),
    ):
        if value is not None:
            replacements[name] = value
    return dataclasses.replace(config, **replacements)


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train maintained v22_final residual-Mamba.")
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--init-from", type=Path)
    parser.add_argument(
        "--zero-state-on-init",
        action="store_true",
        help="Use freshly initialized recurrent state when warm-starting parameters.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_cli(argv: Sequence[str] | None = None) -> V22FinalTrainInvocation:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if args.zero_state_on_init and args.init_from is None:
        parser.error("--zero-state-on-init requires --init-from")
    config = load_training_config(args.config)
    config = apply_operational_overrides(
        config,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        output_root=args.output_root,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
    )
    return V22FinalTrainInvocation(
        config=config,
        config_path=args.config,
        resume=args.resume,
        init_from=args.init_from,
        zero_state_on_init=args.zero_state_on_init,
        dry_run=args.dry_run,
    )


def validate_resume_config(
    current: V22FinalTrainConfig,
    saved: Mapping[str, Any],
    *,
    completed_step: int,
) -> None:
    candidate = current.to_dict()
    saved_copy = json.loads(json.dumps(saved))
    saved_architecture = saved_copy.get("architecture")
    if isinstance(saved_architecture, dict):
        # This no-op field was recorded by early v22_final checkpoints but was
        # never consumed by the full-Mamba implementation.
        saved_architecture.pop("temporal_hidden_size", None)
        saved_architecture.setdefault("temporal_bc_groups", 1)
    saved_sequence = saved_copy.get("sequence")
    if isinstance(saved_sequence, dict):
        # Checkpoints written before the explicit state-policy field always
        # used the legacy carry behavior.  For stateless architectures this
        # carried an empty Haiku state tree and was therefore a no-op.
        saved_sequence.setdefault("temporal_state_policy", "carry")
    saved_optimizer = saved_copy.get("optimizer")
    if isinstance(saved_optimizer, dict):
        # Checkpoints written before configurable decay used warmup followed by
        # a constant learning rate.
        saved_optimizer.setdefault("learning_rate_schedule", "constant")
        saved_optimizer.setdefault("decay_start_step", None)
        saved_optimizer.setdefault("end_learning_rate", None)
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
