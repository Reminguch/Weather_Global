from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .model import gc
from .prepared_array import PreparedArrayStore


class PreparedDataError(RuntimeError):
    """Prepared-array store is missing, incompatible, or invalid for a requested split."""


@dataclasses.dataclass(frozen=True)
class PreparedEvalWindow:
    store: PreparedArrayStore
    eval_start: str
    eval_end: str
    eval_year: int | float


def resolution_tag(resolution: float) -> str:
    value = float(resolution)
    if np.isclose(value, round(value), atol=1e-6):
        return f"res{int(round(value))}"
    return f"res{str(value).replace('.', 'p')}"


def prepared_store_path_from_root(prepared_data_root: str | Path, resolution: float | int) -> Path:
    return Path(prepared_data_root) / resolution_tag(float(resolution))


def open_prepared_store(
    prepared_data_root: str | Path,
    resolution: float | int,
    task_cfg: gc.TaskConfig,
    *,
    label: str = "prepared-array",
) -> PreparedArrayStore:
    store_path = prepared_store_path_from_root(prepared_data_root, resolution)
    if not store_path.exists():
        raise FileNotFoundError(f"Prepared array store not found: {store_path}")
    store = PreparedArrayStore(store_path, label=label)
    try:
        store.validate(resolution=float(resolution), task_cfg=task_cfg)
    except ValueError as exc:
        raise PreparedDataError(str(exc)) from exc
    return store


def split_prepared_store_by_year(
    store: PreparedArrayStore,
    val_year: int,
    *,
    train_start_year: int | None = None,
    train_end_year: int | None = None,
    label: str = "prepared_array",
) -> tuple[PreparedArrayStore, PreparedArrayStore, list[int]]:
    time_index = pd.DatetimeIndex(pd.to_datetime(store.time.values))
    years = sorted(set(time_index.year.astype(int).tolist()))
    if int(val_year) not in years:
        raise PreparedDataError(f"Requested val year {val_year} not present in {label} dataset years: {years}")
    if (train_start_year is None) != (train_end_year is None):
        raise PreparedDataError("Provide both train_start_year and train_end_year, or neither.")

    train_years = [year for year in years if year != int(val_year)]
    if train_start_year is not None:
        train_years = [year for year in train_years if int(train_start_year) <= year <= int(train_end_year)]
    if not train_years:
        raise PreparedDataError(f"No train years left in {label} data after year selection.")

    train_idx = np.where(np.isin(time_index.year, np.asarray(train_years)))[0]
    val_idx = np.where(time_index.year == int(val_year))[0]
    if train_idx.size == 0:
        raise PreparedDataError("Empty train split after year selection.")
    if val_idx.size == 0:
        raise PreparedDataError("Empty validation split after year selection.")
    return (
        store.split_by_time_indices(train_idx, label=f"{label}-train"),
        store.split_by_time_indices(val_idx, label=f"{label}-eval"),
        train_years,
    )


def _format_time(value: pd.Timestamp | np.datetime64) -> str:
    return str(pd.Timestamp(value).to_datetime64())


def select_prepared_eval_window(
    store: PreparedArrayStore,
    *,
    eval_start: str | None = None,
    eval_end: str | None = None,
    eval_year: int | None = None,
) -> PreparedEvalWindow:
    time_index = pd.DatetimeIndex(pd.to_datetime(store.time.values))
    if eval_start is not None or eval_end is not None:
        if eval_start is None or eval_end is None:
            raise PreparedDataError("Provide both eval_start and eval_end, or neither.")
        start = pd.Timestamp(eval_start)
        end = pd.Timestamp(eval_end)
        if end <= start:
            raise PreparedDataError(f"eval_end must be after eval_start, got {eval_start!r} to {eval_end!r}.")
        selected = np.where((time_index >= start) & (time_index < end))[0]
        selected_year: int | float = np.nan
    else:
        if eval_year is None:
            raise PreparedDataError("No eval year available. Pass eval_year or use checkpoints with val_year.")
        selected = np.where(time_index.year == int(eval_year))[0]
        selected_year = int(eval_year)

    if selected.size == 0:
        raise PreparedDataError("Prepared eval selection is empty.")
    if selected.size > 1 and not np.array_equal(np.diff(selected), np.ones(selected.size - 1, dtype=selected.dtype)):
        raise PreparedDataError("Prepared eval selection must be contiguous.")

    split = store.split_by_time_indices(selected, label=f"{store.label}-eval")
    split_time = pd.DatetimeIndex(pd.to_datetime(split.time.values))
    step = split_time[1] - split_time[0] if len(split_time) > 1 else pd.Timedelta(hours=6)
    return PreparedEvalWindow(
        store=split,
        eval_start=_format_time(split_time[0]),
        eval_end=_format_time(split_time[-1] + step),
        eval_year=selected_year,
    )


def load_prepared_metric_grid(
    prepared_data_root: str | Path,
    *,
    resolution: float | int = 18,
) -> tuple[xr.DataArray, xr.DataArray]:
    store_path = prepared_store_path_from_root(prepared_data_root, resolution)
    if not store_path.exists():
        raise FileNotFoundError(f"Prepared metric-grid store not found: {store_path}")
    store = PreparedArrayStore(store_path, label="prepared-metric-grid")
    return (
        xr.DataArray(np.asarray(store.coords["lat"]), dims=["lat"]),
        xr.DataArray(np.asarray(store.coords["lon"]), dims=["lon"]),
    )


def prepared_eval_metadata(
    store_path: str | Path,
    selected_window: PreparedEvalWindow,
    *,
    data_source: str = "prepared_array",
) -> dict[str, object]:
    return {
        "data_source": data_source,
        "prepared_store": str(store_path),
        "eval_start": selected_window.eval_start,
        "eval_end": selected_window.eval_end,
        "eval_year": selected_window.eval_year,
    }
