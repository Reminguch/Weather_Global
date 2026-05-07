from __future__ import annotations

import concurrent.futures
import dataclasses
import time
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .model import gc
from .prepared_array import PreparedArrayBlock, PreparedArrayStore


@dataclasses.dataclass(frozen=True)
class PreparedBlockLoadStats:
    load_s: float
    cache_hits: int = 0
    cache_misses: int = 0
    loaded_gib: float = 0.0
    loader: str = "prepared-block"


class PreparedBlockBatchLoader:
    """Build per-step batches from cached contiguous prepared-array lane blocks."""

    def __init__(
        self,
        store: PreparedArrayStore,
        segments: list[np.ndarray],
        *,
        input_steps: int,
        target_steps: int,
        task_cfg: gc.TaskConfig,
        dt: pd.Timedelta,
        load_executor: concurrent.futures.Executor | None = None,
        max_workers: int = 1,
        label: str = "prepared-block",
    ) -> None:
        self._store = store
        self._segments = segments
        self._input_steps = int(input_steps)
        self._target_steps = int(target_steps)
        self._task_cfg = task_cfg
        self._dt = dt
        self._load_executor = load_executor
        self._max_workers = max(1, int(max_workers))
        self._label = label
        self._cache: dict[int, PreparedArrayBlock] = {}
        self._cache_segment_ids: dict[int, int] = {}
        self.last_stats = PreparedBlockLoadStats(load_s=0.0, loader=self._label)

    def _load_lane_block(self, segment_id: int) -> PreparedArrayBlock:
        segment = self._segments[int(segment_id)]
        block_start = int(segment[0]) - self._input_steps + 1
        block_stop = int(segment[-1]) + self._target_steps + 1
        return self._store.load_time_block(
            block_start,
            block_stop,
            task_cfg=self._task_cfg,
        )

    def _ensure_blocks(self, chunk: Any) -> PreparedBlockLoadStats:
        t0 = time.time()
        hits = 0
        misses: list[tuple[int, int]] = []
        for lane, segment_id in enumerate(chunk.lane_segment_ids):
            cached = self._cache.get(lane)
            if cached is not None and self._cache_segment_ids.get(lane) == int(segment_id):
                hits += 1
            else:
                misses.append((lane, int(segment_id)))

        loaded_bytes = 0
        if misses:
            if self._load_executor is not None and self._max_workers > 1:
                futures = [
                    self._load_executor.submit(self._load_lane_block, segment_id)
                    for _, segment_id in misses
                ]
                loaded = [future.result() for future in futures]
            else:
                loaded = [self._load_lane_block(segment_id) for _, segment_id in misses]
            for (lane, segment_id), block in zip(misses, loaded):
                self._cache[lane] = block
                self._cache_segment_ids[lane] = segment_id
                loaded_bytes += block.bytes_loaded

        stats = PreparedBlockLoadStats(
            load_s=time.time() - t0,
            cache_hits=hits,
            cache_misses=len(misses),
            loaded_gib=loaded_bytes / (1024**3),
            loader=self._label,
        )
        self.last_stats = stats
        return stats

    def build_step_batch(self, final_indices: np.ndarray) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        final_indices = np.asarray(final_indices, dtype=np.int64)
        blocks = [self._cache[lane] for lane in range(len(final_indices))]
        return self._store.build_step_from_blocks(
            blocks,
            final_indices,
            input_steps=self._input_steps,
            target_steps=self._target_steps,
            task_cfg=self._task_cfg,
            dt=self._dt,
        )

    def iter_chunk_batches(
        self,
        chunk: Any,
    ):
        stats = self._ensure_blocks(chunk)
        for step_indices in chunk.chunk_indices:
            yield self.build_step_batch(step_indices), stats

    def load_chunk(
        self,
        chunk: Any,
    ) -> tuple[tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], PreparedBlockLoadStats]:
        stats = self._ensure_blocks(chunk)
        inputs: list[xr.Dataset] = []
        targets: list[xr.Dataset] = []
        forcings: list[xr.Dataset] = []
        for step_indices in chunk.chunk_indices:
            batch_inputs, batch_targets, batch_forcings = self.build_step_batch(step_indices)
            inputs.append(batch_inputs)
            targets.append(batch_targets)
            forcings.append(batch_forcings)
        return tuple(inputs), tuple(targets), tuple(forcings), stats
