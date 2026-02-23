#!/usr/bin/env python3
"""Inference/evaluation entrypoint for one-batch runs.

Batch format (.npz):
- inputs.<name>
- targets.<name>       (required by current CanonicalBatch + adapter contracts)
- forcings.<name>      (optional)
- coords.<name>        (optional)
- metadata.<name>      (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.contracts import CanonicalBatch
from src.pipelines.evaluate import evaluate_batch, initialize_evaluation


def _load_config(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML config requires `pyyaml` installed.") from exc
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, Mapping):
            raise TypeError("YAML config root must be a mapping.")
        return loaded
    raise ValueError("Config file must end with .json, .yaml, or .yml")


def _split_prefix(mapping: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    head = f"{prefix}."
    return {key[len(head) :]: mapping[key] for key in mapping if key.startswith(head)}


def _load_batch(path: Path) -> CanonicalBatch:
    with np.load(path, allow_pickle=True) as npz:
        payload: dict[str, Any] = {key: npz[key] for key in npz.files}

    return CanonicalBatch(
        inputs=_split_prefix(payload, "inputs"),
        targets=_split_prefix(payload, "targets"),
        forcings=_split_prefix(payload, "forcings"),
        coords=_split_prefix(payload, "coords"),
        metadata=_split_prefix(payload, "metadata"),
    )


def _to_float(value: Any) -> float:
    return float(np.asarray(value))


def _save_predictions(path: Path, predictions: Mapping[str, Any]) -> None:
    out: dict[str, Any] = {}
    for key, value in predictions.items():
        out[f"predictions.{key}"] = np.asarray(value)
    np.savez(path, **out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-batch inference/evaluation.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--batch", type=Path, required=True, help="Path to .npz canonical batch.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--predictions-out", type=Path, help="Optional output .npz for predictions.")
    args = parser.parse_args()

    config = _load_config(args.config)
    batch = _load_batch(args.batch)
    rng = np.array([0, args.seed], dtype=np.uint32)
    model, state = initialize_evaluation(config, rng=rng, sample_batch=batch)
    metrics, predictions = evaluate_batch(model, state, batch)

    loss = _to_float(metrics.get("loss", 0.0))
    print(f"loss={loss:.6f}")
    print("prediction_keys=", sorted(predictions.keys()))

    if args.predictions_out:
        _save_predictions(args.predictions_out, predictions)
        print(f"saved predictions: {args.predictions_out}")


if __name__ == "__main__":
    main()
