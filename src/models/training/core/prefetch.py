from __future__ import annotations

from collections import deque
import concurrent.futures
import dataclasses
import threading
import time
from typing import Any, Callable

import jax
import numpy as np
import xarray as xr

from .model import xarray_jax


@dataclasses.dataclass
class BatchRequest:
    indices: np.ndarray
    reset_state: bool
    new_epoch: bool


@dataclasses.dataclass
class PreparedBatch:
    request: BatchRequest
    inputs: xr.Dataset
    targets: xr.Dataset
    forcings: xr.Dataset
    host_build_time: float
    device_put_time: float = 0.0
    device_staged: bool = False
    device_slot_acquired: bool = False
    device_error: str | None = None


def _device_put_batch(
    batch: tuple[xr.Dataset, xr.Dataset, xr.Dataset],
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    return xarray_jax.tree_map_with_dims(lambda array, dims: jax.device_put(array), batch)


def _block_until_ready_tree(tree: Any) -> None:
    def _block(array, dims):
        if hasattr(array, "block_until_ready"):
            array.block_until_ready()
        return array

    xarray_jax.tree_map_with_dims(_block, tree)


class BatchPrefetcher:
    def __init__(
        self,
        *,
        request_fn: Callable[[], BatchRequest],
        build_fn: Callable[[np.ndarray], tuple[xr.Dataset, xr.Dataset, xr.Dataset]],
        max_workers: int,
        depth: int,
        device_depth: int,
    ) -> None:
        self._request_fn = request_fn
        self._build_fn = build_fn
        self._depth = max(1, depth)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._futures: deque[concurrent.futures.Future[PreparedBatch]] = deque()
        self._device_slots = threading.BoundedSemaphore(device_depth) if device_depth > 0 else None
        self._device_enabled = device_depth > 0
        self._lock = threading.Lock()
        self._device_warning_printed = False

    def start(self) -> None:
        self._fill()

    def _fill(self) -> None:
        while len(self._futures) < self._depth:
            request = self._request_fn()
            self._futures.append(self._executor.submit(self._build, request))

    def _build(self, request: BatchRequest) -> PreparedBatch:
        t0 = time.time()
        inputs, targets, forcings = self._build_fn(request.indices)
        host_build_time = time.time() - t0

        prepared = PreparedBatch(
            request=request,
            inputs=inputs,
            targets=targets,
            forcings=forcings,
            host_build_time=host_build_time,
        )

        slot_acquired = False
        if self._device_slots is not None and self._device_enabled:
            slot_acquired = self._device_slots.acquire(blocking=False)
        if slot_acquired:
            prepared.device_slot_acquired = True
            try:
                t_put = time.time()
                prepared.inputs, prepared.targets, prepared.forcings = _device_put_batch(
                    (prepared.inputs, prepared.targets, prepared.forcings)
                )
                _block_until_ready_tree((prepared.inputs, prepared.targets, prepared.forcings))
                prepared.device_put_time = time.time() - t_put
                prepared.device_staged = True
            except Exception as exc:
                prepared.device_error = repr(exc)
                prepared.device_slot_acquired = False
                self._device_slots.release()
                with self._lock:
                    self._device_enabled = False

        return prepared

    def get(self) -> tuple[PreparedBatch, float]:
        if not self._futures:
            self._fill()
        future = self._futures.popleft()
        t_wait = time.time()
        prepared = future.result()
        data_wait = time.time() - t_wait
        if prepared.device_error and not self._device_warning_printed:
            print(f"[prefetch] disabling GPU device staging after error: {prepared.device_error}")
            self._device_warning_printed = True
        self._fill()
        return prepared, data_wait

    def release_device_slot(self, prepared: PreparedBatch) -> None:
        if prepared.device_slot_acquired and self._device_slots is not None:
            self._device_slots.release()
            prepared.device_slot_acquired = False

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
