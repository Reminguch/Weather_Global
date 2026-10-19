"""Owned sequence batches from verified, immutable frozen-GC trajectories.

No baseline or residual predictor is initialized here. The existing cache reader
is the reconstruction oracle; all yielded weather arrays are copied before its
reusable source workspace can be used for another chunk.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import xarray as xr
import jax

from .. import baseline_cache
from .cached_config import CachedTrainingConfig
from .config import load_training_config
from .data import TrainingCursor, advance_cursor, concatenate_batch


def owned_dataset(dataset: xr.Dataset) -> xr.Dataset:
    """Copy data and coordinates, with physical floating fields always FP32."""
    result = dataset.copy(deep=True)
    for name, variable in result.data_vars.items():
        dtype = np.float32 if np.issubdtype(variable.dtype, np.inexact) else variable.dtype
        result[name].data = np.ascontiguousarray(variable.data, dtype=dtype)
    return result


def _source_spatial_coords(dataset: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    """Restore source coordinate dtypes lost by the cache's JSON metadata.

    The producer reader reconstructs latitude/longitude JSON lists as FP64.
    Original prepared coordinates can be FP32, and their dtype is part of JAX's
    xarray tree metadata. Reattach only equal spatial coordinates; each field
    keeps its own batch/time coordinates and its existing weather buffers.
    """
    replacements = {}
    for name in ("lat", "lon", "level"):
        if name not in dataset.coords or name not in template.coords:
            continue
        actual, expected = dataset.coords[name], template.coords[name]
        if actual.dims != expected.dims or not np.array_equal(actual.values, expected.values):
            raise ValueError(f"Cached {name} coordinates differ from source coordinates")
        replacements[name] = expected
    if not replacements:
        return dataset
    # assign_coords moves replaced names to the end. Preserve the maintained
    # reader's coordinate order as well as its non-spatial metadata: both are
    # visible in the exact xarray tree signature used by the parity gate.
    return xr.Dataset(
        data_vars={name: field.variable for name, field in dataset.data_vars.items()},
        coords={name: replacements.get(name, coord).variable
                for name, coord in dataset.coords.items()},
        attrs=dataset.attrs,
    )


@dataclass(frozen=True)
class CachedSequenceBatch:
    inputs: tuple[xr.Dataset, ...]
    targets: tuple[xr.Dataset, ...]
    forcings: tuple[xr.Dataset, ...]
    truths: tuple[xr.Dataset, ...] | None
    baselines: tuple[xr.Dataset, ...] | None
    chunk_id: int
    split: str
    segment_id: int
    segment_offset: int
    chunk_index: int
    raw_anchor_indices: np.ndarray
    timestamps: tuple[str, ...]
    next_cursor: TrainingCursor | None
    chunk_ids: tuple[int, ...] = ()
    segment_ids: tuple[int, ...] = ()

    @property
    def reset_state(self) -> bool:
        """Reset memory on new segments; physical windows restart every chunk."""
        return self.segment_offset == 0

    @property
    def residual_targets(self) -> tuple[xr.Dataset, ...]:
        return self.targets


@dataclass(frozen=True)
class CachedValidationStep:
    """Borrowed fields valid until the streaming iterator advances or closes."""

    step_index: int
    inputs: xr.Dataset
    target: xr.Dataset
    forcing: xr.Dataset
    truth: xr.Dataset
    baseline: xr.Dataset


class CachedTrainingData:
    """Cursor-compatible owned access to training and validation cache chunks."""

    def __init__(self, context, reader: baseline_cache.BaselineCacheReader):
        self.context = context
        self.common = context[0]
        self.training_data = context[4]
        self.reader = reader
        self.manifest = reader.manifest
        self._materialization_lock = threading.Lock()
        # The maintained reconstruction helper uses jnp.asarray internally.
        # Keep those temporary arrays on CPU instead of copying weather to the
        # training GPU and immediately back to host at every cached step.
        self._cpu_device = jax.devices("cpu")[0]
        self._items = {}
        for item in self.manifest["chunks"]:
            key = (item["split"], item["segment_id"], item["segment_offset"])
            if key in self._items:
                raise ValueError(f"Duplicate cache segment/offset: {key}")
            self._items[key] = item
        self.identity = {
            "manifest_sha256": self.manifest["manifest_sha256"],
            "compatibility_sha256": self.manifest["compatibility_sha256"],
            "cache_root": str(reader.root.resolve()),
            "format": self.manifest["format"],
            "num_shards": self.manifest["num_shards"],
            "chunks": len(self.manifest["chunks"]),
            "train_chunks": sum(item["split"] == "train" for item in self._items.values()),
            "validation_chunks": sum(item["split"] == "val" for item in self._items.values()),
            "checksum_policy": "READY provenance; shard schema/coverage on first read",
            "batch_size": self.common.distributed.per_device_batch_size,
            "dropped_training_segments": len(self.training_data.segments) % self.common.distributed.per_device_batch_size,
        }

    def _materialize(
        self,
        *,
        split: str,
        segment_id: int,
        segment_offset: int,
        next_cursor: TrainingCursor | None,
        include_validation_fields: bool,
    ) -> CachedSequenceBatch:
        key = (split, segment_id, segment_offset)
        if key not in self._items:
            raise ValueError(f"No cached complete chunk for {key}")
        item = self._items[key]
        # Validation and one-batch prefetch share the original workspace. Hold
        # the lock until every returned array owns its memory.
        with self._materialization_lock, jax.default_device(self._cpu_device):
            chunk = baseline_cache.load_chunk(self.context, item)
            inputs, targets, forcings = [], [], []
            truths = [] if include_validation_fields else None
            baselines = [] if include_validation_fields else None
            for index, (weather, baseline, target, forcing) in enumerate(
                self.reader.iter_chunk(item["id"], chunk, self.training_data.time_step)
            ):
                template = chunk.truths[index]
                weather = _source_spatial_coords(weather, template)
                baseline = _source_spatial_coords(baseline, template)
                target = _source_spatial_coords(target, template)
                forcing = _source_spatial_coords(forcing, template)
                inputs.append(owned_dataset(weather))
                targets.append(owned_dataset(target))
                forcings.append(owned_dataset(forcing))
                if include_validation_fields:
                    assert truths is not None and baselines is not None
                    truths.append(owned_dataset(chunk.truths[index]))
                    baselines.append(owned_dataset(baseline))
            if len(inputs) != self.common.bptt_steps:
                raise ValueError("Cached reader returned an incomplete sequence")
            return CachedSequenceBatch(
                inputs=tuple(inputs), targets=tuple(targets), forcings=tuple(forcings),
                truths=None if truths is None else tuple(truths),
                baselines=None if baselines is None else tuple(baselines),
                chunk_id=int(item["id"]), split=split,
                segment_id=int(segment_id), segment_offset=int(segment_offset),
                chunk_index=int(item["chunk_index"]),
                raw_anchor_indices=np.array(item["raw_anchor_indices"], dtype=np.int64, copy=True),
                timestamps=tuple(item["target_timestamps"]), next_cursor=next_cursor,
            )

    def build_chunk(
        self, cursor: TrainingCursor, *, include_validation_fields: bool = False,
    ) -> CachedSequenceBatch:
        self.training_data.validate_cursor(cursor, self.common)
        batch_size = self.common.distributed.per_device_batch_size
        if cursor.segment_index % batch_size or cursor.segment_index + batch_size > len(self.training_data.segments):
            raise ValueError("Batched cached cursor must address a complete aligned segment group")
        next_cursor = advance_cursor(cursor, self.common, len(self.training_data.segments))
        if batch_size > 1:
            lanes = tuple(self._materialize(
                split="train", segment_id=cursor.segment_index + lane,
                segment_offset=cursor.segment_offset, next_cursor=None,
                include_validation_fields=include_validation_fields,
            ) for lane in range(batch_size))

            def combine(name):
                if getattr(lanes[0], name) is None:
                    return None
                return tuple(concatenate_batch([getattr(lane, name)[step] for lane in lanes])
                             for step in range(self.common.bptt_steps))

            return CachedSequenceBatch(
                inputs=combine("inputs"), targets=combine("targets"), forcings=combine("forcings"),
                truths=combine("truths"), baselines=combine("baselines"),
                chunk_id=lanes[0].chunk_id, split="train", segment_id=cursor.segment_index,
                segment_offset=cursor.segment_offset, chunk_index=lanes[0].chunk_index,
                raw_anchor_indices=np.stack([lane.raw_anchor_indices for lane in lanes]),
                timestamps=tuple(timestamp for lane in lanes for timestamp in lane.timestamps),
                next_cursor=next_cursor, chunk_ids=tuple(lane.chunk_id for lane in lanes),
                segment_ids=tuple(lane.segment_id for lane in lanes),
            )
        return self._materialize(
            split="train", segment_id=cursor.segment_index,
            segment_offset=cursor.segment_offset,
            next_cursor=next_cursor,
            include_validation_fields=include_validation_fields,
        )

    def load_validation_chunk(
        self, segment_id: int, segment_offset: int, *, include_validation_fields: bool = True,
    ) -> CachedSequenceBatch:
        return self._materialize(
            split="val", segment_id=segment_id, segment_offset=segment_offset,
            next_cursor=None, include_validation_fields=include_validation_fields,
        )

    def iter_validation_steps(
        self, segment_id: int, segment_offset: int,
    ) -> Iterator[CachedValidationStep]:
        """Stream validation without copying complete prediction/truth tapes.

        The source workspace is locked until this generator is exhausted or
        closed. Consume a row synchronously before advancing. The CPU-device
        scope ends before each yield so the caller's model still runs on GPU.
        """
        key = ("val", segment_id, segment_offset)
        if key not in self._items:
            raise ValueError(f"No cached complete chunk for {key}")
        item = self._items[key]
        with self._materialization_lock:
            with jax.default_device(self._cpu_device):
                chunk = baseline_cache.load_chunk(self.context, item)
            rows = self.reader.iter_chunk(item["id"], chunk, self.training_data.time_step)
            try:
                for index in range(self.common.bptt_steps):
                    with jax.default_device(self._cpu_device):
                        try:
                            inputs, baseline, target, forcing = next(rows)
                        except StopIteration as exc:
                            raise ValueError("Cached reader returned an incomplete sequence") from exc
                    template = chunk.truths[index]
                    inputs = _source_spatial_coords(inputs, template)
                    baseline = _source_spatial_coords(baseline, template)
                    target = _source_spatial_coords(target, template)
                    forcing = _source_spatial_coords(forcing, template)
                    yield CachedValidationStep(index, inputs, target, forcing, chunk.truths[index], baseline)
                with jax.default_device(self._cpu_device):
                    if next(rows, None) is not None:
                        raise ValueError("Cached reader returned an oversized sequence")
            finally:
                rows.close()

    def iter_chunks(
        self, cursor: TrainingCursor, *, prefetch_batches: int = 0,
        max_chunks: int | None = None, include_validation_fields: bool = False,
    ) -> Iterator[CachedSequenceBatch]:
        """Yield ordered batches, holding at most one prefetched next batch.

        Prefetch cursors are speculative: only a completed optimizer update may
        commit ``batch.next_cursor`` to a training checkpoint. Close this
        generator when breaking out early to release its one worker promptly.
        """
        if type(prefetch_batches) is not int or prefetch_batches not in (0, 1):
            raise ValueError("prefetch_batches must be 0 or 1")
        if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 0):
            raise ValueError("max_chunks must be a non-negative integer or None")
        if max_chunks == 0:
            return
        if prefetch_batches == 0:
            count = 0
            while max_chunks is None or count < max_chunks:
                batch = self.build_chunk(cursor, include_validation_fields=include_validation_fields)
                assert batch.next_cursor is not None
                cursor = batch.next_cursor
                count += 1
                yield batch
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cached-weather") as executor:
            future = executor.submit(self.build_chunk, cursor, include_validation_fields=include_validation_fields)
            count = 0
            try:
                while future is not None:
                    batch = future.result()
                    assert batch.next_cursor is not None
                    count += 1
                    future = (
                        executor.submit(self.build_chunk, batch.next_cursor, include_validation_fields=include_validation_fields)
                        if max_chunks is None or count < max_chunks else None
                    )
                    yield batch
            finally:
                if future is not None:
                    future.cancel()


def validate_consumer_manifest(saved: dict, expected: dict, *, numerical_policy="strict"):
    """Retain original producer hashes while checking the consumer's raw inputs."""
    import copy
    if numerical_policy not in ("strict", "advisory"):
        raise ValueError("Unknown numerical policy")
    if numerical_policy == "strict":
        if saved != expected:
            raise ValueError("Cached manifest differs from current source/configuration/code identity")
        return
    baseline_cache.validate_manifest(saved)
    baseline_cache.validate_manifest(expected)
    left, right = copy.deepcopy(saved), copy.deepcopy(expected)
    for record in (left, right):
        record.pop("manifest_sha256")
        record.pop("compatibility_sha256")
        record["compatibility"].pop("generation_code_sha256")
    if left != right:
        raise ValueError("Cached raw data/model/statistics/schedule identity differs from consumer")


def open_cached_training_data(config: CachedTrainingConfig, *, numerical_policy="strict") -> CachedTrainingData:
    """Check actual producer/source identity and require complete production data.

    Legacy readers require the original producer code. Advisory recoveries pin
    the immutable producer manifest and separately validate current raw inputs.
    """
    if load_training_config(config.config_path) != config.common:
        raise ValueError("Cached common configuration differs from its source JSON")
    root = config.execution.cache_root
    saved = json.loads((root / "manifest.json").read_text())
    baseline_cache.validate_manifest(saved)
    if saved["partial"]:
        raise ValueError("Cached training requires a complete production manifest")
    pinned = config.execution.expected_manifest_sha256
    if pinned is not None and saved["manifest_sha256"] != pinned:
        raise ValueError("Cached manifest differs from expected_manifest_sha256")
    context = baseline_cache.load_context(config.config_path)
    expected = baseline_cache.build_manifest(context, num_shards=saved["num_shards"])
    if numerical_policy == "advisory" and pinned is None:
        raise ValueError("Recovery cache reuse requires its pinned original manifest hash")
    validate_consumer_manifest(saved, expected, numerical_policy=numerical_policy)
    reader = baseline_cache.BaselineCacheReader(root, saved["compatibility_sha256"])
    ready = json.loads((root / "READY.json").read_text())
    required_ready = {
        "manifest_sha256": saved["manifest_sha256"],
        "chunks": len(saved["chunks"]),
        "predictions": sum(len(item["raw_anchor_indices"]) for item in saved["chunks"]),
        "shards": saved["num_shards"],
        "partial": False,
    }
    if any(ready.get(key) != value for key, value in required_ready.items()):
        raise ValueError("READY marker coverage differs from the production manifest")
    result = CachedTrainingData(context, reader)
    result.identity.update(numerical_policy=numerical_policy,
                           producer_generation_code_sha256=saved["compatibility"].get("generation_code_sha256", {}),
                           consumer_generation_code_sha256=expected["compatibility"].get("generation_code_sha256", {}))
    return result
