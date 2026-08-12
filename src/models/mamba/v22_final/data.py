"""Dataset adapters for v22_final evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from src.models.graphcast.training.core.batching import (
    build_batch_from_indices as _build_batch_from_indices,
    infer_time_step,
    input_steps_from_duration,
    valid_final_input_indices,
)
from src.models.graphcast.training.core.dataset import _open_local_splits, prepare_dataset_for_task

from .config import V22FinalEvalConfig


@dataclass(frozen=True)
class EvalDataset:
    dataset: xr.Dataset
    time_step: pd.Timedelta
    input_steps: int


def open_eval_dataset(config: V22FinalEvalConfig, task_config) -> EvalDataset:
    split_config = SimpleNamespace(
        data_path=config.data_path,
        resolution=config.resolution,
        val_year=config.val_year,
        train_start_year=config.train_start_year,
        train_end_year=config.train_end_year,
    )
    _train_dataset, eval_dataset = _open_local_splits(split_config)
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
