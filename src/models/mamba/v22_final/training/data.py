"""Deterministic prepared-array and anchor-manifest access for v22_final."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.models.graphcast.training.core.batching import input_steps_from_duration
from src.models.graphcast.training.core.eval_selection import select_eval_subset
from src.models.graphcast.training.core.prepared_array import PreparedArrayStore

from ..checkpoint import manifest_sha256
from .config import V22FinalTrainConfig


@dataclass(frozen=True)
class TrainingCursor:
    """Location of the next BPTT chunk to consume."""

    epoch: int = 0
    segment_index: int = 0
    segment_offset: int = 0

    def __post_init__(self) -> None:
        for name in ("epoch", "segment_index", "segment_offset"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingCursor":
        expected = {"epoch", "segment_index", "segment_offset"}
        if set(value) != expected:
            raise ValueError(
                f"training_cursor requires exactly {sorted(expected)}, got {sorted(value)}"
            )
        return cls(**{name: int(value[name]) for name in expected})


@dataclass(frozen=True)
class BPTTChunk:
    truth_inputs: tuple[Any, ...]
    targets: tuple[Any, ...]
    forcings: tuple[Any, ...]
    raw_anchor_indices: np.ndarray


@dataclass(frozen=True)
class TrainingChunk:
    truth_inputs: tuple[Any, ...]
    targets: tuple[Any, ...]
    forcings: tuple[Any, ...]
    raw_anchor_indices: np.ndarray
    next_cursor: TrainingCursor


@dataclass(frozen=True)
class V22FinalTrainingData:
    store: PreparedArrayStore
    anchor_indices: np.ndarray
    train_split: np.ndarray
    val_split: np.ndarray
    segments: tuple[np.ndarray, ...]
    validation_segments: tuple[np.ndarray, ...]
    fixed_validation_segment_ids: np.ndarray
    validation_subset_policy: str
    validation_subset_fingerprint: str | None
    time_step: pd.Timedelta
    input_steps: int
    manifest_fingerprint: str

    def validate_cursor(self, cursor: TrainingCursor, config: V22FinalTrainConfig) -> None:
        if cursor.segment_index >= len(self.segments):
            raise ValueError(
                f"segment_index={cursor.segment_index} exceeds {len(self.segments)} segments"
            )
        if cursor.segment_offset >= config.segment_steps:
            raise ValueError(
                f"segment_offset={cursor.segment_offset} must be below {config.segment_steps}"
            )
        if cursor.segment_offset % config.bptt_steps:
            raise ValueError(
                f"segment_offset={cursor.segment_offset} is not aligned to "
                f"bptt_steps={config.bptt_steps}"
            )

    def load_segment_chunk(
        self,
        segment: np.ndarray,
        segment_offset: int,
        config: V22FinalTrainConfig,
        task_config,
    ) -> BPTTChunk:
        if segment_offset < 0 or segment_offset % config.bptt_steps:
            raise ValueError(
                f"segment_offset={segment_offset} must be a non-negative multiple "
                f"of bptt_steps={config.bptt_steps}"
            )
        stop = segment_offset + config.bptt_steps
        chunk_manifest_indices = np.asarray(segment[segment_offset:stop], dtype=np.int64)
        if chunk_manifest_indices.size != config.bptt_steps:
            raise ValueError("Segment offset points to an incomplete BPTT chunk")
        raw_anchor_indices = self.anchor_indices[chunk_manifest_indices]
        truth_inputs = []
        targets = []
        forcings = []
        for anchor_position, raw_anchor_index in enumerate(raw_anchor_indices):
            inputs, target, forcing = self.store.build_batch_from_indices(
                indices=[int(raw_anchor_index)],
                input_steps=self.input_steps,
                target_steps=config.target_steps,
                task_cfg=task_config,
                dt=self.time_step,
            )
            if anchor_position < config.truth_prefix_steps:
                truth_inputs.append(inputs)
            targets.append(target)
            forcings.append(forcing)
        return BPTTChunk(
            truth_inputs=tuple(truth_inputs),
            targets=tuple(targets),
            forcings=tuple(forcings),
            raw_anchor_indices=np.asarray(raw_anchor_indices, dtype=np.int64),
        )

    def build_chunk(
        self,
        cursor: TrainingCursor,
        config: V22FinalTrainConfig,
        task_config,
    ) -> TrainingChunk:
        self.validate_cursor(cursor, config)
        loaded = self.load_segment_chunk(
            self.segments[cursor.segment_index],
            cursor.segment_offset,
            config,
            task_config,
        )
        return TrainingChunk(
            truth_inputs=loaded.truth_inputs,
            targets=loaded.targets,
            forcings=loaded.forcings,
            raw_anchor_indices=loaded.raw_anchor_indices,
            next_cursor=advance_cursor(cursor, config, len(self.segments)),
        )

    def validation_segment_metadata(self, segment_ids: np.ndarray) -> list[dict[str, Any]]:
        metadata = []
        for segment_id in np.asarray(segment_ids, dtype=np.int64):
            segment = self.validation_segments[int(segment_id)]
            raw = self.anchor_indices[segment]
            midpoint = int(raw[len(raw) // 2])
            metadata.append(
                {
                    "segment_id": int(segment_id),
                    "first_raw_anchor": int(raw[0]),
                    "last_raw_anchor": int(raw[-1]),
                    "midpoint_time": str(pd.Timestamp(self.store.time.values[midpoint])),
                }
            )
        return metadata

    def fingerprint_validation_subset(self, segment_ids: np.ndarray) -> str:
        payload = []
        for segment_id in np.asarray(segment_ids, dtype=np.int64):
            segment = self.validation_segments[int(segment_id)]
            payload.append(
                {
                    "segment_id": int(segment_id),
                    "raw_anchor_indices": self.anchor_indices[segment].tolist(),
                }
            )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def build_segments(split_indices: np.ndarray, segment_steps: int) -> tuple[np.ndarray, ...]:
    split = np.asarray(split_indices, dtype=np.int64)
    if split.ndim != 1:
        raise ValueError(f"split indices must be one-dimensional, got {split.shape}")
    if segment_steps <= 0:
        raise ValueError("segment_steps must be positive")
    return tuple(
        np.asarray(split[start:start + segment_steps], dtype=np.int64)
        for start in range(0, split.size - segment_steps + 1, segment_steps)
    )


def advance_cursor(
    cursor: TrainingCursor,
    config: V22FinalTrainConfig,
    segment_count: int,
) -> TrainingCursor:
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    next_offset = cursor.segment_offset + config.bptt_steps
    if next_offset < config.segment_steps:
        return TrainingCursor(cursor.epoch, cursor.segment_index, next_offset)
    next_segment = cursor.segment_index + 1
    if next_segment < segment_count:
        return TrainingCursor(cursor.epoch, next_segment, 0)
    return TrainingCursor(cursor.epoch + 1, 0, 0)


def _load_index_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing anchor manifest file: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{path} must contain a one-dimensional integer array")
    return np.asarray(values, dtype=np.int64)


def open_training_data(
    config: V22FinalTrainConfig,
    task_config,
) -> V22FinalTrainingData:
    store = PreparedArrayStore(config.prepared_root, label="v22-final-training")
    store.validate(resolution=config.architecture.resolution, task_cfg=task_config)
    time_step = pd.Timedelta(
        np.diff(np.asarray(store.time.values).astype("datetime64[ns]"))[0]
    )
    input_steps = input_steps_from_duration(task_config.input_duration, time_step)

    metadata_path = config.anchor_manifest_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing anchor manifest metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not np.isclose(float(metadata.get("resolution", -1)), config.architecture.resolution):
        raise ValueError("Anchor manifest resolution does not match training architecture")
    if int(metadata.get("input_steps", -1)) != input_steps:
        raise ValueError("Anchor manifest input_steps do not match the configured task")
    if list(metadata.get("target_variables", [])) != list(task_config.target_variables):
        raise ValueError("Anchor manifest target_variables do not match the baseline task")

    anchors_root = config.anchor_manifest_root / "anchors"
    anchor_indices = _load_index_array(anchors_root / "anchor_indices.npy")
    train_split = _load_index_array(anchors_root / "split_train.npy")
    val_split = _load_index_array(anchors_root / "split_val.npy")
    for label, split in (("train", train_split), ("validation", val_split)):
        if split.size and (split.min() < 0 or split.max() >= anchor_indices.size):
            raise ValueError(f"{label} split contains indices outside anchor_indices")
    if anchor_indices.size == 0:
        raise ValueError("Anchor manifest contains no anchors")
    min_valid = input_steps - 1
    max_valid = store.sizes["time"] - config.target_steps - 1
    if anchor_indices.min() < min_valid or anchor_indices.max() > max_valid:
        raise ValueError(
            f"Anchor positions must lie in [{min_valid}, {max_valid}], got "
            f"{anchor_indices.min()}..{anchor_indices.max()}"
        )
    segments = build_segments(train_split, config.segment_steps)
    validation_segments = build_segments(val_split, config.segment_steps)
    if not segments:
        raise ValueError(
            f"Training split has {train_split.size} entries, fewer than one "
            f"segment of {config.segment_steps}"
        )
    fixed_validation_segment_ids = np.asarray([], dtype=np.int64)
    validation_subset_policy = "disabled"
    validation_subset_fingerprint = None
    if config.validation.enabled:
        if not validation_segments:
            raise ValueError(
                f"Validation split has {val_split.size} entries, fewer than one "
                f"segment of {config.segment_steps}"
            )
        midpoint_times = []
        for segment in validation_segments:
            raw = anchor_indices[segment]
            midpoint_times.append(store.time.values[int(raw[len(raw) // 2])])
        selection = select_eval_subset(
            np.arange(len(validation_segments), dtype=np.int64),
            config.validation.num_segments,
            times=midpoint_times,
            policy="stratified_fixed",
            role="fixed_checkpoint",
            fold=0,
        )
        fixed_validation_segment_ids = selection.item_ids
        validation_subset_policy = selection.policy
        fingerprint_payload = [
            {
                "segment_id": int(segment_id),
                "raw_anchor_indices": anchor_indices[
                    validation_segments[int(segment_id)]
                ].tolist(),
            }
            for segment_id in fixed_validation_segment_ids
        ]
        validation_subset_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    return V22FinalTrainingData(
        store=store,
        anchor_indices=anchor_indices,
        train_split=train_split,
        val_split=val_split,
        segments=segments,
        validation_segments=validation_segments,
        fixed_validation_segment_ids=fixed_validation_segment_ids,
        validation_subset_policy=validation_subset_policy,
        validation_subset_fingerprint=validation_subset_fingerprint,
        time_step=time_step,
        input_steps=input_steps,
        manifest_fingerprint=manifest_sha256(config.anchor_manifest_root),
    )
