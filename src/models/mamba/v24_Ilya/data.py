"""Dataset adapters for v24_Ilya evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
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
    if config.initialization_manifest is None:
        eval_positions = np.flatnonzero(time_index.year == config.val_year)
        if eval_positions.size == 0:
            years = sorted(set(time_index.year.astype(int).tolist()))
            raise ValueError(
                f"Requested val year {config.val_year} not present in evaluation data years: {years}"
            )
        eval_dataset = dataset.isel(time=eval_positions)
    else:
        # Validate against the unsliced store before bounding the context.
        # In particular, don't construct forcings for an entire multiyear
        # training store when a manifest requests a small development span.
        time_step = infer_time_step(dataset)
        input_steps = input_steps_from_duration(task_config.input_duration, time_step)
        records = resolve_initialization_manifest(
            EvalDataset(dataset, time_step, input_steps),
            config.initialization_manifest,
            target_steps=config.target_steps,
            history_steps=config.effective_anchor_history_steps,
        )
        first = records[0]["dataset_index"] - config.effective_anchor_history_steps - input_steps + 1
        stop = records[-1]["dataset_index"] + config.target_steps + 1
        eval_dataset = dataset.isel(time=slice(first, stop))
    print(
        "Evaluation data: "
        f"val_year={config.val_year}, val_time={eval_dataset.sizes['time']}, "
        f"base_res={base_resolution}, target_res={config.resolution}, stride={stride}"
    )
    eval_dataset = prepare_dataset_for_task(eval_dataset, task_config)
    time_step = infer_time_step(eval_dataset)
    input_steps = input_steps_from_duration(task_config.input_duration, time_step)
    return EvalDataset(dataset=eval_dataset, time_step=time_step, input_steps=input_steps)


def _utc_timestamp(value: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"Initialization timestamps must be UTC strings, got {value!r}")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"Initialization timestamp requires an explicit UTC offset: {value!r}")
    if timestamp.utcoffset() != pd.Timedelta(0):
        raise ValueError(f"Initialization timestamp must use UTC: {value!r}")
    return timestamp.tz_convert("UTC")


def utc_iso(timestamp: pd.Timestamp) -> str:
    """Canonical UTC text for manifest IDs and exported valid times."""

    return timestamp.isoformat().replace("+00:00", "Z")


def resolve_initialization_manifest(
    eval_data: EvalDataset,
    manifest_path: Path,
    *,
    target_steps: int,
    history_steps: int = 0,
) -> list[dict]:
    """Resolve every explicit start, rejecting incomplete or ambiguous windows.

    Input JSON is ``{"schema_version": 1, "initializations": [...]}``.
    Entries are UTC strings or objects with ``initialization_time``. Metadata
    is derived from actual timestamps so indices never define date identity.
    """

    if target_steps <= 0 or history_steps < 0:
        raise ValueError("target_steps must be positive and history_steps non-negative")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Initialization manifest requires schema_version=1")
    entries = payload.get("initializations")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Initialization manifest requires a non-empty initializations list")
    starts = pd.DatetimeIndex([
        _utc_timestamp(entry.get("initialization_time") if isinstance(entry, dict) else entry)
        for entry in entries
    ])
    if starts.has_duplicates:
        raise ValueError("Initialization manifest contains duplicate timestamps")
    if not starts.is_monotonic_increasing:
        raise ValueError("Initialization manifest timestamps must be chronological")
    if eval_data.time_step != pd.Timedelta("6h"):
        raise ValueError("Initialization-manifest evaluation requires six-hour data")
    if eval_data.input_steps != 2:
        raise ValueError("Initialization-manifest evaluation requires two input frames")
    time_index = pd.DatetimeIndex(pd.to_datetime(eval_data.dataset.time.values, utc=True))
    if time_index.hasnans or time_index.has_duplicates or not time_index.is_monotonic_increasing:
        raise ValueError("Evaluation data timestamps must be finite, unique, and chronological")
    result = []
    for ordinal, start in enumerate(starts):
        if start != start.floor("6h"):
            raise ValueError(f"Initialization timestamp must align with 00/06/12/18 UTC: {utc_iso(start)}")
        required = pd.date_range(
            start - (history_steps + eval_data.input_steps - 1) * eval_data.time_step,
            start + target_steps * eval_data.time_step,
            freq=eval_data.time_step,
        )
        positions = time_index.get_indexer(required)
        missing = required[positions < 0]
        if len(missing):
            raise ValueError(
                f"Initialization {utc_iso(start)} is missing required input/target timestamps: "
                + ", ".join(utc_iso(value) for value in missing[:8])
            )
        if np.any(np.diff(positions) != 1):
            raise ValueError(f"Initialization {utc_iso(start)} has non-six-hour intervening data")
        input_times = pd.date_range(
            start - (eval_data.input_steps - 1) * eval_data.time_step,
            start,
            freq=eval_data.time_step,
        )
        valid_times = pd.date_range(
            start + eval_data.time_step, periods=target_steps, freq=eval_data.time_step,
        )
        result.append({
            "ordinal": ordinal,
            "initialization_id": start.strftime("%Y%m%dT%H%M%SZ"),
            "initialization_time": utc_iso(start),
            "dataset_index": int(time_index.get_loc(start)),
            "input_times": [utc_iso(value) for value in input_times],
            "valid_times": [utc_iso(value) for value in valid_times],
            "lead_hours": [6 * step for step in range(1, target_steps + 1)],
            "initialization_year": int(start.year),
            "valid_years": [int(value.year) for value in valid_times],
            "season": ("DJF", "MAM", "JJA", "SON")[(start.month % 12) // 3],
        })
    return result


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
