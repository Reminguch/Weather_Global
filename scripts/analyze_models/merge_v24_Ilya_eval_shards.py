#!/usr/bin/env python3
"""Exactly merge independently evaluated v24_Ilya anchor shards."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.mamba.v24_Ilya.evaluation import (  # noqa: E402
    write_evaluation_snapshot,
)
from src.models.mamba.v24_Ilya.metrics import (  # noqa: E402
    METRIC_MERGE_STATE_FORMAT,
    V24IlyaMetricAccumulator,
)


_METRIC_SECTIONS = {
    "original_graphcast_loss",
    "per_variable_per_step",
    "per_channel_per_step",
    "residual_diagnostics_per_variable",
}

_LIBC = ctypes.CDLL(None)

_SHARED_METADATA_KEYS = (
    "architecture_id",
    "schema_version",
    "checkpoint_format",
    "resolution",
    "seed",
    "eval_mode",
    "warmup_steps",
    "warmup_feedback",
    "eval_feedback",
    "rs_reset_after_warmup",
    "rs_reset_every_step",
    "baseline_branch",
    "residual_state_init",
    "residual_state_init_requested",
    "residual_state_init_resolved",
    "target_steps",
    "sample_total_steps",
    "anchor_history_steps",
    "anchor_index_semantics",
    "stream_block_steps",
    "metric_profile",
    "metrics_omitted",
    "n_samples",
    "anchor_shard_count",
    "global_chosen_idx",
    "ckpt",
    "ckpt_in",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-json", type=Path, action="append", required=True)
    parser.add_argument("--merge-state", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument(
        "--delete-merge-states",
        action="store_true",
        help="Remove the large temporary accumulator states after a successful merge.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Shard JSON must contain an object: {path}")
    return payload


def _validate_shard_jsons(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], list[int]]:
    if not paths:
        raise ValueError("At least one shard JSON is required")
    payloads = [_load_json(path) for path in paths]
    reference = payloads[0]
    shard_count = reference.get("anchor_shard_count")
    if shard_count != len(paths):
        raise ValueError(
            f"Expected {shard_count} shard JSONs from metadata, got {len(paths)}"
        )
    global_indices = reference.get("global_chosen_idx")
    if not isinstance(global_indices, list) or not global_indices:
        raise ValueError("Shard metadata has no global_chosen_idx")

    by_index: dict[int, dict[str, Any]] = {}
    for path, payload in zip(paths, payloads, strict=True):
        for key in _SHARED_METADATA_KEYS:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"Shard metadata differs for {key!r}: {path}")
        if payload.get("evaluation_status") != "complete":
            raise ValueError(f"Shard is not complete: {path}")
        if payload.get("metrics_complete") is not True:
            raise ValueError(f"Shard does not contain complete metrics: {path}")
        shard_index = payload.get("anchor_shard_index")
        if not isinstance(shard_index, int) or not 0 <= shard_index < shard_count:
            raise ValueError(f"Invalid shard index in {path}: {shard_index!r}")
        if shard_index in by_index:
            raise ValueError(f"Duplicate shard index {shard_index}")
        expected_indices = global_indices[shard_index::shard_count]
        if payload.get("chosen_idx") != expected_indices:
            raise ValueError(f"Shard {shard_index} selected the wrong anchors")
        if payload.get("completed_chosen_idx") != expected_indices:
            raise ValueError(f"Shard {shard_index} did not complete every anchor")
        if payload.get("evaluated_samples") != len(expected_indices):
            raise ValueError(f"Shard {shard_index} sample count is inconsistent")
        by_index[shard_index] = payload

    expected_shards = set(range(shard_count))
    if set(by_index) != expected_shards:
        raise ValueError(
            f"Shard indices differ: got={sorted(by_index)}, "
            f"expected={sorted(expected_shards)}"
        )
    return [by_index[index] for index in range(shard_count)], global_indices


def _load_merge_state(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict):
        raise ValueError(f"Merge state must contain a mapping: {path}")
    return state


def _release_host_memory() -> None:
    gc.collect()
    malloc_trim = getattr(_LIBC, "malloc_trim", None)
    if malloc_trim is not None:
        malloc_trim(0)


def _state_sample_count(state: dict[str, Any]) -> int:
    counts_by_variable = state["arrays"]["n_per_var"]
    flattened = [np.asarray(value) for value in counts_by_variable.values()]
    if not flattened:
        raise ValueError("Merge state has no per-variable counts")
    reference = flattened[0]
    for counts in flattened[1:]:
        if not np.array_equal(counts, reference):
            raise ValueError("Merge-state per-variable sample counts differ")
    if np.any(reference <= 0) or np.any(reference != reference[0]):
        raise ValueError("Merge-state sample counts differ across leads")
    return int(reference[0])


def merge_v24_Ilya_eval_shards(
    *,
    shard_json_paths: list[Path],
    merge_state_paths: list[Path],
    out_json: Path,
    delete_merge_states: bool,
) -> dict[str, Any]:
    if len(shard_json_paths) != len(merge_state_paths):
        raise ValueError("The number of shard JSONs and merge states must match")
    if out_json.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out_json}")

    shard_payloads, global_indices = _validate_shard_jsons(shard_json_paths)
    state_by_shard: dict[int, Path] = {}
    for payload, path in zip(shard_payloads, merge_state_paths, strict=True):
        shard_index = int(payload["anchor_shard_index"])
        declared = payload.get("merge_state_path")
        if declared is not None and Path(declared) != path:
            raise ValueError(
                f"Shard {shard_index} declares merge state {declared}, got {path}"
            )
        state_by_shard[shard_index] = path

    merged: V24IlyaMetricAccumulator | None = None
    for shard_index in range(len(shard_payloads)):
        path = state_by_shard[shard_index]
        print(f"Loading metric merge state shard={shard_index}: {path}", flush=True)
        state = _load_merge_state(path)
        if state.get("format") != METRIC_MERGE_STATE_FORMAT:
            raise ValueError(f"Unexpected merge-state format in {path}")
        expected_count = len(global_indices[shard_index::len(shard_payloads)])
        actual_count = _state_sample_count(state)
        if actual_count != expected_count:
            raise ValueError(
                f"Shard {shard_index} merge state has {actual_count} samples; "
                f"expected {expected_count}"
            )
        if merged is None:
            merged = V24IlyaMetricAccumulator.from_merge_state(state)
        else:
            merged.merge_exported_state(state)
        del state
        _release_host_memory()

    assert merged is not None
    reference = shard_payloads[0]
    base_output = {
        key: value
        for key, value in reference.items()
        if key not in _METRIC_SECTIONS
        and key
        not in {
            "evaluation_status",
            "evaluated_samples",
            "remaining_samples",
            "chosen_idx",
            "completed_chosen_idx",
            "metrics_available",
            "metrics_complete",
            "intermediate_metrics_omitted",
        }
    }
    base_output.update(
        anchor_shard_index=None,
        merge_state_path=None,
        merged_anchor_shards=list(range(len(shard_payloads))),
        shard_json_paths=[str(path) for path in shard_json_paths],
        metric_merge_state_format=METRIC_MERGE_STATE_FORMAT,
    )
    output = write_evaluation_snapshot(
        out_json,
        base_output=base_output,
        metric_output=merged.finalize(),
        chosen_indices=global_indices,
        completed_samples=len(global_indices),
        status="complete",
    )
    print(f"Wrote exact merged evaluation: {out_json}", flush=True)

    if delete_merge_states:
        for path in merge_state_paths:
            path.unlink()
            print(f"Removed temporary merge state: {path}", flush=True)
    return output


def main() -> None:
    args = _parse_args()
    merge_v24_Ilya_eval_shards(
        shard_json_paths=args.shard_json,
        merge_state_paths=args.merge_state,
        out_json=args.out_json,
        delete_merge_states=args.delete_merge_states,
    )


if __name__ == "__main__":
    main()
