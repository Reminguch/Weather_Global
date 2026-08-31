#!/usr/bin/env python3
"""Compare uniform- and heavy-tail sparse-H20 res1 training trajectories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HORIZONS = (1, 4, 8, 12, 16, 20)
DEFAULT_HEAVY_TAIL = Path(
    "artifacts/checkpoints/v24_Ilya/res1_v22compat_init_ablation_20260828/"
    "v22compat_sparse_legacy_di16_bcg1_tzero_fp32_lr1em4_10k/train_metrics.jsonl"
)
DEFAULT_V22_EXACT = Path(
    "artifacts/checkpoints/v22_final/res1_dm_7yr_k20_di_bcg_20k_20260813/"
    "di16_bcg1_closed_sg_stateful_20k/eval/cold_full_zero_exact/"
    "swa_step02000-08000.json"
)


def _records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records:
        raise ValueError(f"No metrics in {path}")
    for record in records:
        if "loss_by_horizon" not in record:
            raise ValueError(f"Missing sparse horizon losses in {path}")
    return records


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("Cannot take the mean of no values")
    return sum(values) / len(values)


def _index(records: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    return {
        (int(record["epoch"]), int(record["segment_index"]), int(record["segment_offset"])): record
        for record in records
    }


def _epoch_report(label: str, records: list[dict[str, Any]]) -> None:
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_epoch[int(record["epoch"])].append(record)
    print(f"{label}: steps=1-{records[-1]['step']}")
    for epoch in sorted(by_epoch):
        group = by_epoch[epoch]
        components = ",".join(
            f"h{horizon}={_mean(float(row['loss_by_horizon'][str(horizon)]) for row in group):.4f}"
            for horizon in HORIZONS
        )
        print(
            f"  epoch={epoch} n={len(group)} loss={_mean(float(row['loss']) for row in group):.4f} "
            f"grad={_mean(float(row['gradient_norm']) for row in group):.4f} {components}"
        )

    indexed = _index(records)
    epochs = sorted(by_epoch)
    for left, right in zip(epochs, epochs[1:]):
        keys = [key for key in indexed if key[0] == left and (right, key[1], key[2]) in indexed]
        if not keys:
            continue
        composite = _mean(
            float(indexed[(right, key[1], key[2])]["loss"]) - float(indexed[key]["loss"])
            for key in keys
        )
        horizon_text = ",".join(
            f"h{horizon}={_mean(float(indexed[(right, key[1], key[2])]['loss_by_horizon'][str(horizon)]) - float(indexed[key]['loss_by_horizon'][str(horizon)]) for key in keys):+.4f}"
            for horizon in HORIZONS
        )
        print(
            f"  same_chunk epoch{left}->{right} n={len(keys)} loss_delta={composite:+.4f} {horizon_text}"
        )


def _cross_run_report(candidate: list[dict[str, Any]], heavy_tail: list[dict[str, Any]]) -> None:
    left, right = _index(candidate), _index(heavy_tail)
    keys = sorted(left.keys() & right.keys())
    if not keys:
        raise ValueError("No common (epoch, segment_index, segment_offset) chunks")
    components = ",".join(
        f"h{horizon}={_mean(float(left[key]['loss_by_horizon'][str(horizon)]) - float(right[key]['loss_by_horizon'][str(horizon)]) for key in keys):+.4f}"
        for horizon in HORIZONS
    )
    composite = _mean(float(left[key]["loss"]) - float(right[key]["loss"]) for key in keys)
    print(f"uniform_minus_heavy common_chunks={len(keys)} loss_delta={composite:+.4f} {components}")


def _eval_report(paths: list[Path], reference: Path) -> None:
    if not paths:
        return
    v22 = json.loads(reference.read_text())
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["chosen_idx"] != v22["chosen_idx"]:
            raise ValueError(f"Anchor mismatch: {path}")
        metric = payload["original_graphcast_loss"]
        baseline, model = metric["baseline_per_step"], metric["full_per_step"]
        print(
            f"{path.name}: through_day10={100 * (1 - sum(model) / sum(baseline)):.4f}% "
            f"day10={100 * (1 - model[39] / baseline[39]):.4f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-metrics", type=Path, required=True)
    parser.add_argument("--heavy-tail-metrics", type=Path, default=DEFAULT_HEAVY_TAIL)
    parser.add_argument("--v22-exact-reference", type=Path, default=DEFAULT_V22_EXACT)
    parser.add_argument("--uniform-eval", type=Path, action="append", default=[])
    args = parser.parse_args()

    uniform = _records(args.uniform_metrics)
    heavy_tail = _records(args.heavy_tail_metrics)
    _epoch_report("uniform_sparse", uniform)
    _epoch_report("heavy_tail_sparse", heavy_tail)
    _cross_run_report(uniform, heavy_tail)
    _eval_report(args.uniform_eval, args.v22_exact_reference)


if __name__ == "__main__":
    main()
