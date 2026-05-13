from __future__ import annotations

import dataclasses
import warnings
from typing import Sequence

import numpy as np
import pandas as pd


EVAL_SUBSET_STRATIFIED_FIXED = "stratified_fixed"
EVAL_SUBSET_STRATIFIED_ROTATING = "stratified_rotating"
EVAL_SUBSET_FIRST = "first"


@dataclasses.dataclass(frozen=True)
class EvalSubsetSelection:
    positions: np.ndarray
    item_ids: np.ndarray
    policy: str
    role: str
    fold: int | None
    available: int
    capped: bool

    def metadata(self, *, item_name: str) -> dict[str, object]:
        metadata: dict[str, object] = {
            "eval_subset_policy": self.policy,
            "eval_subset_role": self.role,
            "eval_subset_fold": None if self.fold is None else int(self.fold),
            f"eval_subset_available_{item_name}": int(self.available),
            f"eval_subset_selected_{item_name}": int(self.positions.size),
        }
        if self.capped:
            metadata[f"eval_subset_{item_name}_ids"] = [int(x) for x in self.item_ids.tolist()]
            metadata["eval_subset_positions"] = [int(x) for x in self.positions.tolist()]
        return metadata


def _safe_datetime_index(values: Sequence[object] | np.ndarray | None) -> pd.DatetimeIndex | None:
    if values is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            datetime_values = pd.to_datetime(values, errors="coerce")
        if pd.isna(datetime_values).any():
            return None
        return pd.DatetimeIndex(datetime_values)
    except (TypeError, ValueError):
        return None


def _groups_from_times(count: int, times: Sequence[object] | np.ndarray | None) -> list[np.ndarray]:
    datetime_index = _safe_datetime_index(times)
    if datetime_index is None or len(datetime_index) != count:
        return [np.arange(count, dtype=np.int64)]
    quarters = np.asarray(datetime_index.quarter, dtype=np.int64)
    return [np.where(quarters == quarter)[0].astype(np.int64) for quarter in sorted(np.unique(quarters))]


def _allocate_counts(group_sizes: list[int], count: int, max_items: int) -> list[int]:
    non_empty = [i for i, size in enumerate(group_sizes) if size > 0]
    if not non_empty:
        return [0 for _ in group_sizes]

    allocations = [0 for _ in group_sizes]
    if max_items >= len(non_empty):
        for i in non_empty:
            allocations[i] = 1
        remaining = max_items - len(non_empty)
    else:
        chosen = np.linspace(0, len(non_empty) - 1, max_items, dtype=int).tolist()
        for i in [non_empty[j] for j in chosen]:
            allocations[i] = 1
        remaining = 0

    while remaining > 0:
        best_i = max(
            non_empty,
            key=lambda i: (max_items * group_sizes[i] / count - allocations[i], group_sizes[i], -i),
        )
        allocations[best_i] += 1
        remaining -= 1

    return [min(alloc, group_sizes[i]) for i, alloc in enumerate(allocations)]


def _fill_unique(indices: list[int], size: int, target_count: int) -> list[int]:
    seen = set()
    unique = []
    for index in indices:
        normalized = int(index) % size
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    if len(unique) < target_count:
        for index in range(size):
            if index not in seen:
                unique.append(index)
                seen.add(index)
            if len(unique) == target_count:
                break
    return unique[:target_count]


def _select_from_group(group: np.ndarray, count: int, *, policy: str, fold: int) -> list[int]:
    if count <= 0:
        return []
    if count >= group.size:
        return [int(x) for x in group.tolist()]
    if policy == EVAL_SUBSET_STRATIFIED_ROTATING:
        raw = np.floor(np.arange(count, dtype=np.float64) * group.size / count + fold).astype(int)
    else:
        raw = np.rint(np.linspace(0, group.size - 1, count)).astype(int)
    local = _fill_unique(raw.tolist(), int(group.size), count)
    return [int(group[i]) for i in local]


def select_eval_subset(
    item_ids: Sequence[int] | np.ndarray,
    max_items: int | None,
    *,
    times: Sequence[object] | np.ndarray | None = None,
    policy: str = EVAL_SUBSET_STRATIFIED_FIXED,
    role: str = "fixed_checkpoint",
    fold: int | None = None,
) -> EvalSubsetSelection:
    item_ids_np = np.asarray(item_ids, dtype=np.int64)
    available = int(item_ids_np.size)
    if available == 0:
        raise ValueError("No eval items available.")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive or None.")
    if max_items is None or max_items >= available:
        positions = np.arange(available, dtype=np.int64)
        return EvalSubsetSelection(
            positions=positions,
            item_ids=item_ids_np,
            policy="all",
            role=role,
            fold=None,
            available=available,
            capped=False,
        )

    fold_value = int(fold or 0)
    if policy == EVAL_SUBSET_FIRST:
        positions = np.arange(max_items, dtype=np.int64)
    elif policy in {EVAL_SUBSET_STRATIFIED_FIXED, EVAL_SUBSET_STRATIFIED_ROTATING}:
        groups = _groups_from_times(available, times)
        allocations = _allocate_counts([int(group.size) for group in groups], available, int(max_items))
        selected: list[int] = []
        for group, alloc in zip(groups, allocations, strict=True):
            selected.extend(_select_from_group(group, alloc, policy=policy, fold=fold_value))
        positions = np.asarray(sorted(selected), dtype=np.int64)
    else:
        raise ValueError(
            f"Unknown eval subset policy {policy!r}; expected "
            f"{EVAL_SUBSET_FIRST!r}, {EVAL_SUBSET_STRATIFIED_FIXED!r}, or {EVAL_SUBSET_STRATIFIED_ROTATING!r}."
        )

    return EvalSubsetSelection(
        positions=positions,
        item_ids=item_ids_np[positions],
        policy=policy,
        role=role,
        fold=fold_value,
        available=available,
        capped=True,
    )
