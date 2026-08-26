"""Dataset adapters for v24_Ilya evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import xarray as xr

from src.models.graphcast.training.core.batching import (
    build_batch_from_indices as _build_batch_from_indices,
    infer_time_step,
    input_steps_from_duration,
    valid_final_input_indices,
)
from src.data_operations.loaders.graphcast_dataset import open_graphcast_era5
from src.models.graphcast.training.core.dataset import (
    _ensure_datetime_coord,
    prepare_dataset_for_task,
    select_resolution,
)

from .config import V24IlyaEvalConfig


@dataclass(frozen=True)
class EvalDataset:
    dataset: xr.Dataset
    time_step: pd.Timedelta
    input_steps: int


def open_eval_dataset(config: V24IlyaEvalConfig, task_config) -> EvalDataset:
    # Evaluation only needs the validation slice.  Opening it directly also
    # supports compact validation-only stores (for example January 2022 at
    # 0.25 degrees), which cannot satisfy the training split helper's
    # requirement that at least one non-validation year be present.
    if "://" in config.data_path:
        raise ValueError(
            f"data_path must be local; remote URIs are disabled: {config.data_path}"
        )
    print(f"Opening local evaluation dataset: {config.data_path}")
    dataset = _ensure_datetime_coord(open_graphcast_era5(config.data_path))
    dataset, base_resolution, stride = select_resolution(dataset, config.resolution)
    time_index = pd.DatetimeIndex(pd.to_datetime(dataset.time.values))
    eval_positions = np.flatnonzero(time_index.year == config.val_year)
    if eval_positions.size == 0:
        years = sorted(set(time_index.year.astype(int).tolist()))
        raise ValueError(
            f"Requested val year {config.val_year} not present in evaluation data years: {years}"
        )
    eval_dataset = dataset.isel(time=eval_positions)
    print(
        "Evaluation data: "
        f"val_year={config.val_year}, val_time={eval_dataset.sizes['time']}, "
        f"base_res={base_resolution}, target_res={config.resolution}, stride={stride}"
    )
    eval_dataset = prepare_dataset_for_task(eval_dataset, task_config)
    time_step = infer_time_step(eval_dataset)
    input_steps = input_steps_from_duration(task_config.input_duration, time_step)
    return EvalDataset(dataset=eval_dataset, time_step=time_step, input_steps=input_steps)


def valid_eval_indices(eval_data: EvalDataset, target_steps: int) -> np.ndarray:
    return valid_final_input_indices(
        eval_data.dataset.sizes["time"],
        eval_data.input_steps,
        target_steps,
    )


def valid_scored_eval_indices(
    eval_data: EvalDataset,
    *,
    history_steps: int,
    target_steps: int,
) -> np.ndarray:
    """Return common H1 origins with bounded truth history and future targets.

    Each returned index is the final truth/input frame immediately before the
    first scored prediction. A rollout with ``warmup_steps`` begins at
    ``scored_index - warmup_steps``.
    """

    if history_steps < 0:
        raise ValueError(f"history_steps must be non-negative, got {history_steps}")
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    first = eval_data.input_steps - 1 + history_steps
    stop = eval_data.dataset.sizes["time"] - target_steps
    if stop <= first:
        return np.asarray([], dtype=np.int64)
    return np.arange(first, stop, dtype=np.int64)


def build_eval_batch(
    eval_data: EvalDataset,
    *,
    indices: Iterable[int],
    target_steps: int,
    task_config,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    return _build_batch_from_indices(
        eval_data.dataset,
        indices=indices,
        input_steps=eval_data.input_steps,
        target_steps=target_steps,
        task_cfg=task_config,
        dt=eval_data.time_step,
    )


def iter_eval_steps(
    eval_data: EvalDataset,
    *,
    final_input_idx: int,
    total_steps: int,
    block_steps: int,
    task_config,
) -> Iterator[tuple[xr.Dataset, xr.Dataset]]:
    """Yield one truth/forcing step while loading only a bounded host block."""

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if block_steps <= 0:
        raise ValueError(f"block_steps must be positive, got {block_steps}")
    step_ns = int(eval_data.time_step / pd.Timedelta(1, "ns"))
    for block_start in range(0, total_steps, block_steps):
        count = min(block_steps, total_steps - block_start)
        _inputs, targets, forcings = build_eval_batch(
            eval_data,
            indices=[final_input_idx + block_start],
            target_steps=count,
            task_config=task_config,
        )
        times = (
            block_start
            + 1
            + np.arange(count, dtype=np.int64)
        ) * np.timedelta64(step_ns, "ns")
        targets = targets.assign_coords(time=times)
        forcings = forcings.assign_coords(time=times)
        del _inputs
        for local_index in range(count):
            selection = {"time": slice(local_index, local_index + 1)}
            yield targets.isel(**selection), forcings.isel(**selection)
