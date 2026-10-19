"""Independent, padded sequence lanes over a verified one-step baseline cache.

The producer is opened and verified with its original single-lane configuration.
This adapter only changes the consumer's segment order and batch dimension; it
never rewrites the manifest, regenerates its identity, or relaxes verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Iterator, Mapping

import numpy as np
import xarray as xr

from .cached_data import CachedSequenceBatch, CachedTrainingData, owned_dataset
from .data import TrainingCursor


@dataclass(frozen=True)
class CachedK1Cursor:
    """The next complete optimizer batch; shuffling is derived from its epoch."""

    epoch: int = 0
    group_index: int = 0
    chunk_index: int = 0

    def __post_init__(self) -> None:
        for name in ("epoch", "group_index", "chunk_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CachedK1Cursor":
        if set(value) != {"epoch", "group_index", "chunk_index"}:
            raise ValueError("cached k1 cursor requires epoch, group_index, chunk_index")
        return cls(**dict(value))


@dataclass(frozen=True)
class CachedK1Batch:
    inputs: tuple[xr.Dataset, ...]
    targets: tuple[xr.Dataset, ...]
    forcings: tuple[xr.Dataset, ...]
    valid_mask: np.ndarray
    segment_ids: np.ndarray
    raw_anchor_indices: np.ndarray
    chunk_ids: np.ndarray
    timestamps: tuple[tuple[str, ...], ...]
    split: str
    chunk_index: int
    next_cursor: CachedK1Cursor | None
    truths: tuple[xr.Dataset, ...] | None = None
    baselines: tuple[xr.Dataset, ...] | None = None

    @property
    def reset_state(self) -> bool:
        return self.chunk_index == 0

    @property
    def residual_targets(self) -> tuple[xr.Dataset, ...]:
        return self.targets

    @property
    def valid_sequences(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def valid_weather_steps(self) -> int:
        return self.valid_sequences * len(self.inputs)


def _concatenate_lanes(datasets: list[xr.Dataset]) -> xr.Dataset:
    """Allocate independent buffers, preserving exact non-batch coordinates."""
    for dataset in datasets:
        if dataset.sizes.get("batch", 1) != 1:
            raise ValueError("Producer fields must have a singleton batch dimension")
    lanes = [dataset if "batch" in dataset.dims else dataset.expand_dims(batch=[0])
             for dataset in datasets]
    result = xr.concat(lanes, dim="batch", data_vars="all", coords="minimal",
                       compat="equals", join="exact", combine_attrs="override")
    result = result.assign_coords(batch=np.arange(len(lanes), dtype=np.int64))
    # In particular, B=1 and unchanged coordinates must not alias source buffers.
    return owned_dataset(result)


class CachedK1Data:
    """Batch complete segments without dropping a partial final group.

    Each lane keeps its segment for all four 24-step chunks. Padded lanes copy
    a real lane's finite physical fields, receive zero loss weight, and are
    explicitly marked with segment/anchor id -1. They never count as examples.
    """

    def __init__(self, verified: CachedTrainingData, *, batch_size: int,
                 seed: int, shuffle: bool = True):
        for name, value in (("batch_size", batch_size), ("seed", seed)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if batch_size == 0:
            raise ValueError("batch_size must be positive")
        self.verified = verified
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.common = verified.common
        self.manifest = verified.manifest
        compatibility = self.manifest["compatibility"]
        if (self.common.ar_tail_k != 0 or compatibility.get("ar_tail_k") != 0
                or compatibility.get("feedback_mode") != "baseline"
                or self.common.feedback_mode != "baseline"):
            raise ValueError("Cached k1 requires a genuine ar_tail_k=0 baseline cache")
        self.bptt_steps = int(self.common.bptt_steps)
        self.segment_steps = int(self.common.segment_steps)
        if self.bptt_steps != 24 or self.segment_steps != 96:
            raise ValueError("Cached k1 requires 96-step segments and 24-step chunks")
        if (compatibility.get("bptt_steps") != self.bptt_steps
                or compatibility.get("truth_prefix_steps") != self.bptt_steps):
            raise ValueError("Cached k1 requires observed input windows at every step")
        if self.common.distributed.global_batch_size != 1:
            raise ValueError("Open the verified producer with its original batch size 1")
        if self.manifest.get("partial", True):
            raise ValueError("Cached k1 requires a complete verified production cache")
        self.chunks_per_segment = self.segment_steps // self.bptt_steps
        self.num_segments = len(verified.training_data.segments)
        self.num_validation_segments = len(verified.training_data.validation_segments)
        if self.num_segments == 0:
            raise ValueError("Cached k1 requires at least one complete training segment")
        self.groups_per_epoch = (self.num_segments + self.batch_size - 1) // self.batch_size
        self.steps_per_epoch = self.groups_per_epoch * self.chunks_per_segment
        self.validation_groups = ((self.num_validation_segments + self.batch_size - 1)
                                  // self.batch_size)
        self._validate_coverage()
        self.identity = {**verified.identity, "consumer_batch_size": self.batch_size,
                         "sampler": "pcg64_seed_epoch_segment_permutation_v1",
                         "sampler_seed": self.seed, "shuffle_segments": self.shuffle,
                         "padded_final_group": True,
                         "steps_per_epoch": self.steps_per_epoch}

    def _validate_coverage(self) -> None:
        expected = {(split, segment, chunk * self.bptt_steps)
                    for split, count in (("train", self.num_segments),
                                         ("val", self.num_validation_segments))
                    for segment in range(count) for chunk in range(self.chunks_per_segment)}
        actual = [(item["split"], int(item["segment_id"]), int(item["segment_offset"]))
                  for item in self.manifest["chunks"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Cached k1 manifest lacks exact complete segment/chunk coverage")

    def segment_order(self, epoch: int) -> np.ndarray:
        CachedK1Cursor(epoch=epoch)
        order = np.arange(self.num_segments, dtype=np.int64)
        if self.shuffle:
            np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                [self.seed, int(epoch)]))).shuffle(order)
        return order

    def validate_cursor(self, cursor: CachedK1Cursor) -> None:
        if not isinstance(cursor, CachedK1Cursor):
            raise TypeError("Expected CachedK1Cursor")
        if cursor.group_index >= self.groups_per_epoch:
            raise ValueError("group_index exceeds the number of segment groups")
        if cursor.chunk_index >= self.chunks_per_segment:
            raise ValueError("chunk_index exceeds the number of chunks per segment")

    def completed_updates(self, cursor: CachedK1Cursor) -> int:
        self.validate_cursor(cursor)
        return (cursor.epoch * self.steps_per_epoch
                + cursor.group_index * self.chunks_per_segment + cursor.chunk_index)

    def cursor_for_update(self, completed_updates: int) -> CachedK1Cursor:
        if (isinstance(completed_updates, bool)
                or not isinstance(completed_updates, Integral) or completed_updates < 0):
            raise ValueError("completed_updates must be a non-negative integer")
        epoch, within_epoch = divmod(int(completed_updates), self.steps_per_epoch)
        group, chunk = divmod(within_epoch, self.chunks_per_segment)
        return CachedK1Cursor(epoch, group, chunk)

    def build_batch(self, cursor: CachedK1Cursor, *,
                    include_validation_fields: bool = False) -> CachedK1Batch:
        self.validate_cursor(cursor)
        start = cursor.group_index * self.batch_size
        segments = self.segment_order(cursor.epoch)[start:start + self.batch_size]
        following = self.cursor_for_update(self.completed_updates(cursor) + 1)
        return self._build(segments, cursor.chunk_index, "train", following,
                           include_validation_fields, epoch=cursor.epoch)

    def build_validation_batch(self, group_index: int, chunk_index: int, *,
                               include_validation_fields: bool = True) -> CachedK1Batch:
        cursor = CachedK1Cursor(group_index=group_index, chunk_index=chunk_index)
        if cursor.group_index >= self.validation_groups:
            raise ValueError("group_index exceeds validation groups")
        if cursor.chunk_index >= self.chunks_per_segment:
            raise ValueError("chunk_index exceeds chunks per segment")
        start = group_index * self.batch_size
        segments = np.arange(start, min(start + self.batch_size,
                                        self.num_validation_segments), dtype=np.int64)
        return self._build(segments, chunk_index, "val", None, include_validation_fields)

    def _build(self, segments: np.ndarray, chunk_index: int, split: str,
               following: CachedK1Cursor | None, include_fields: bool,
               *, epoch: int = 0) -> CachedK1Batch:
        offset = chunk_index * self.bptt_steps
        lanes: list[CachedSequenceBatch] = []
        for segment in segments:
            if split == "train":
                lane = self.verified.build_chunk(
                    TrainingCursor(epoch=epoch, segment_index=int(segment), segment_offset=offset),
                    include_validation_fields=include_fields)
            else:
                lane = self.verified.load_validation_chunk(
                    int(segment), offset, include_validation_fields=include_fields)
            for name in ("inputs", "targets", "forcings"):
                if len(getattr(lane, name)) != self.bptt_steps:
                    raise ValueError("Producer returned an incomplete cached chunk")
            lanes.append(lane)
        valid = len(lanes)
        lanes.extend([lanes[-1]] * (self.batch_size - valid))

        def stack(name: str) -> tuple[xr.Dataset, ...]:
            fields = [getattr(lane, name) for lane in lanes]
            if any(field is None or len(field) != self.bptt_steps for field in fields):
                raise ValueError(f"Producer did not return complete {name}")
            return tuple(_concatenate_lanes([field[step] for field in fields])
                         for step in range(self.bptt_steps))

        mask = np.zeros(self.batch_size, dtype=np.float32)
        mask[:valid] = 1
        segment_ids = np.full(self.batch_size, -1, dtype=np.int64)
        segment_ids[:valid] = segments
        anchors = np.full((self.batch_size, self.bptt_steps), -1, dtype=np.int64)
        chunks = np.full(self.batch_size, -1, dtype=np.int64)
        for lane_index in range(valid):
            anchors[lane_index] = lanes[lane_index].raw_anchor_indices
            chunks[lane_index] = lanes[lane_index].chunk_id
        return CachedK1Batch(
            inputs=stack("inputs"), targets=stack("targets"), forcings=stack("forcings"),
            valid_mask=mask, segment_ids=segment_ids, raw_anchor_indices=anchors,
            chunk_ids=chunks, timestamps=tuple(lane.timestamps if index < valid else ()
                                             for index, lane in enumerate(lanes)),
            split=split, chunk_index=chunk_index, next_cursor=following,
            truths=stack("truths") if include_fields else None,
            baselines=stack("baselines") if include_fields else None)

    def iter_batches(self, cursor: CachedK1Cursor, *,
                     max_batches: int) -> Iterator[CachedK1Batch]:
        if isinstance(max_batches, bool) or not isinstance(max_batches, Integral) or max_batches < 0:
            raise ValueError("max_batches must be a non-negative integer")
        self.validate_cursor(cursor)
        for _ in range(max_batches):
            batch = self.build_batch(cursor)
            yield batch
            assert batch.next_cursor is not None
            cursor = batch.next_cursor
