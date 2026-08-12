"""Versioned JSON and CLI contract for v22_final training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass
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
    max_steps: int = 50_000
    checkpoint_every: int = 2_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 18
    precision: str = "bf16"

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
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
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
            },
            "architecture": dataclasses.asdict(self.architecture),
            "sequence": {
                "segment_steps": self.segment_steps,
                "bptt_steps": self.bptt_steps,
                "ar_tail_k": self.ar_tail_k,
                "feedback_mode": self.feedback_mode,
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
        set(payload) - {"architecture_id", "schema_version", "data", "architecture", "sequence", "optimizer", "output"}
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
        {"segment_steps", "bptt_steps", "ar_tail_k", "feedback_mode"},
    )
    optimizer = _section(
        payload,
        "optimizer",
        {"max_steps", "checkpoint_every", "learning_rate", "weight_decay", "warmup_steps", "grad_clip", "seed", "precision"},
    )
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
) -> V22FinalTrainConfig:
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
    parser = argparse.ArgumentParser(description="Train maintained v22_final residual-Mamba.")
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


def parse_cli(argv: Sequence[str] | None = None) -> V22FinalTrainInvocation:
    args = build_cli_parser().parse_args(argv)
    config = load_training_config(args.config)
    config = apply_operational_overrides(
        config,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    return V22FinalTrainInvocation(
        config=config,
        config_path=args.config,
        resume=args.resume,
        init_from=args.init_from,
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
