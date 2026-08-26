"""Fixed-subset validation for the v23_Ilya training objective."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from src.models.graphcast.training.core.eval_selection import select_eval_subset

from ..checkpoint import atomic_json_dump
from .config import V23IlyaTrainConfig
from .data import V23IlyaTrainingData


_VALIDATION_RNG_NAMESPACE = 0x563232


def validation_step_keys(
    *, seed: int, segment_id: int, chunk_index: int, bptt_steps: int
) -> jax.Array:
    """Return stable per-anchor keys without consuming the training RNG."""
    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, _VALIDATION_RNG_NAMESPACE)
    key = jax.random.fold_in(key, int(segment_id))
    key = jax.random.fold_in(key, int(chunk_index))
    _retained, *step_keys = jax.random.split(key, bptt_steps + 1)
    return jnp.stack(step_keys)


def select_validation_segment_ids(
    training_data: V23IlyaTrainingData, max_segments: int | None
) -> tuple[np.ndarray, str]:
    midpoint_times = []
    for segment in training_data.validation_segments:
        raw = training_data.anchor_indices[segment]
        midpoint_times.append(
            training_data.store.time.values[int(raw[len(raw) // 2])]
        )
    selection = select_eval_subset(
        np.arange(len(training_data.validation_segments), dtype=np.int64),
        max_segments,
        times=midpoint_times,
        policy="stratified_fixed",
        role="fixed_checkpoint",
        fold=0,
    )
    return selection.item_ids, selection.policy


def run_fixed_validation(
    *,
    validation_step,
    residual_params,
    zero_residual_state,
    training_data: V23IlyaTrainingData,
    task_config,
    config: V23IlyaTrainConfig,
    segment_ids: Sequence[int] | np.ndarray,
    step: int,
    role: str,
    subset_policy: str,
) -> dict[str, Any]:
    """Evaluate the training BPTT loss over complete validation segments."""
    selected = np.asarray(segment_ids, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("Validation requires at least one selected segment")
    if selected.min() < 0 or selected.max() >= len(training_data.validation_segments):
        raise ValueError("Validation segment ID is outside the available segment range")

    started = time.monotonic()
    total_loss = 0.0
    total_loss_components = np.zeros(
        len(config.supervised_horizon_labels), dtype=np.float64
    )
    total_anchors = 0
    chunks_per_segment = config.segment_steps // config.bptt_steps
    for segment_id_np in selected:
        segment_id = int(segment_id_np)
        segment = training_data.validation_segments[segment_id]
        validation_state = jax.tree_util.tree_map(
            jnp.zeros_like,
            zero_residual_state,
        )
        for chunk_index, offset in enumerate(
            range(0, config.segment_steps, config.bptt_steps)
        ):
            chunk = training_data.load_segment_chunk(
                segment,
                offset,
                config,
                task_config,
            )
            keys = validation_step_keys(
                seed=config.seed,
                segment_id=segment_id,
                chunk_index=chunk_index,
                bptt_steps=config.bptt_steps,
            )
            loss, validation_state, loss_components = validation_step(
                residual_params,
                validation_state,
                keys,
                chunk.input_frames,
                chunk.static_inputs,
                chunk.truths,
                chunk.forcings,
            )
            loss = jax.block_until_ready(loss)
            loss_components = jax.block_until_ready(loss_components)
            total_loss += float(jax.device_get(loss)) * config.bptt_steps
            total_loss_components += (
                np.asarray(jax.device_get(loss_components), dtype=np.float64)
                * config.bptt_steps
            )
            total_anchors += config.bptt_steps

    loss_by_horizon = {
        str(horizon): float(total_loss_components[position] / total_anchors)
        for position, horizon in enumerate(config.supervised_horizon_labels)
    }
    return {
        "step": int(step),
        "loss": total_loss / total_anchors,
        "loss_by_horizon": loss_by_horizon,
        "duration_seconds": time.monotonic() - started,
        "role": role,
        "available_segments": len(training_data.validation_segments),
        "selected_segments": int(selected.size),
        "segment_ids": [int(value) for value in selected.tolist()],
        "num_chunks": int(selected.size * chunks_per_segment),
        "num_anchors": int(total_anchors),
        "bptt_steps": config.bptt_steps,
        "ar_tail_k": config.ar_tail_k,
        "feedback_mode": config.feedback_mode,
        "subset_policy": subset_policy,
        "subset_fingerprint": training_data.fingerprint_validation_subset(selected),
    }


def load_validation_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed validation JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"Validation record at {path}:{line_number} is not an object")
        records.append(value)
    return records


def find_validation_record(
    records: Sequence[Mapping[str, Any]],
    *,
    step: int,
    role: str,
    subset_fingerprint: str | None = None,
) -> Mapping[str, Any] | None:
    matches = [
        record
        for record in records
        if int(record.get("step", -1)) == step
        and record.get("role") == role
        and (
            subset_fingerprint is None
            or record.get("subset_fingerprint") == subset_fingerprint
        )
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Duplicate validation records for step={step}, role={role}, "
            f"subset={subset_fingerprint}"
        )
    return matches[0] if matches else None


def append_validation_record(path: Path, record: Mapping[str, Any]) -> bool:
    """Append once; identical retries are no-ops and conflicts are rejected."""
    existing = load_validation_records(path)
    match = find_validation_record(
        existing,
        step=int(record["step"]),
        role=str(record["role"]),
        subset_fingerprint=str(record["subset_fingerprint"]),
    )
    if match is not None:
        if dict(match) != dict(record):
            raise ValueError(
                f"Conflicting validation record for step={record['step']}, "
                f"role={record['role']}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()
    return True


def update_best_validation(
    *,
    records: Sequence[Mapping[str, Any]],
    checkpoint_for_step,
    output_path: Path,
    subset_fingerprint: str | None = None,
) -> Mapping[str, Any] | None:
    fixed = [
        record
        for record in records
        if record.get("role") == "fixed_checkpoint"
        and (
            subset_fingerprint is None
            or record.get("subset_fingerprint") == subset_fingerprint
        )
    ]
    if not fixed:
        return None
    best = min(fixed, key=lambda record: (float(record["loss"]), int(record["step"])))
    payload = {
        "step": int(best["step"]),
        "loss": float(best["loss"]),
        "checkpoint": str(checkpoint_for_step(int(best["step"]))),
        "role": "fixed_checkpoint",
        "subset_fingerprint": best["subset_fingerprint"],
    }
    atomic_json_dump(payload, output_path)
    return payload


def update_train_validation_plot(
    *,
    train_metrics_path: Path,
    validation_metrics_path: Path,
    output_path: Path,
    subset_fingerprint: str | None = None,
) -> None:
    """Plot trailing-100 training loss and fixed validation loss atomically."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_records = []
    if train_metrics_path.is_file():
        train_records = [
            json.loads(line)
            for line in train_metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    validation_records = [
        record
        for record in load_validation_records(validation_metrics_path)
        if record.get("role") == "fixed_checkpoint"
        and (
            subset_fingerprint is None
            or record.get("subset_fingerprint") == subset_fingerprint
        )
    ]
    if not train_records and not validation_records:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_records:
        steps = np.asarray([int(record["step"]) for record in train_records])
        losses = np.asarray([float(record["loss"]) for record in train_records])
        width = min(100, losses.size)
        rolling = np.convolve(losses, np.ones(width) / width, mode="valid")
        ax.plot(steps[width - 1 :], rolling, label=f"train loss ({width}-step mean)")
    if validation_records:
        ax.plot(
            [int(record["step"]) for record in validation_records],
            [float(record["loss"]) for record in validation_records],
            marker="o",
            label="fixed validation loss",
        )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("BPTT residual loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=150)
    plt.close(fig)
    temporary.replace(output_path)
