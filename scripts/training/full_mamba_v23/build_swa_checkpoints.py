#!/usr/bin/env python3
"""Build a schema-checked uniform SWA checkpoint from explicit inputs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-type",
        choices=("residual", "graphcast"),
        required=True,
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-steps", type=int, nargs="+", required=True)
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if len(args.inputs) < 2:
        raise ValueError("SWA requires at least two checkpoints")
    if len(args.inputs) != len(args.source_steps):
        raise ValueError("--source-steps must contain one entry per input checkpoint")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SWA inputs: {missing}")


def build_residual(args: argparse.Namespace) -> None:
    checkpoints = []
    for path in args.inputs:
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if set(checkpoint) != {"residual_params", "residual_state"}:
            raise ValueError(f"Unexpected residual checkpoint keys in {path}: {sorted(checkpoint)}")
        checkpoints.append(checkpoint)

    first_params = checkpoints[0]["residual_params"]
    structure = jax.tree.structure(first_params)
    for path, checkpoint in zip(args.inputs[1:], checkpoints[1:]):
        if jax.tree.structure(checkpoint["residual_params"]) != structure:
            raise ValueError(f"Residual parameter tree differs in {path}")

    averaged = jax.tree.map(
        lambda *leaves: sum(leaves) / len(leaves),
        *(checkpoint["residual_params"] for checkpoint in checkpoints),
    )
    payload = {
        "residual_params": averaged,
        "residual_state": checkpoints[0]["residual_state"],
        "swa_source_steps": args.source_steps,
        "swa_source_ckpts": [str(path) for path in args.inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle)


def build_graphcast(args: argparse.Namespace) -> None:
    with np.load(args.inputs[0], allow_pickle=False) as first:
        first_arrays = {key: first[key] for key in first.files}
    signature = {key: (array.shape, array.dtype) for key, array in first_arrays.items()}

    for path in args.inputs[1:]:
        with np.load(path, allow_pickle=False) as checkpoint:
            candidate = {key: (checkpoint[key].shape, checkpoint[key].dtype) for key in checkpoint.files}
        if candidate != signature:
            raise ValueError(f"GraphCast checkpoint schema differs in {path}")

    output_arrays: dict[str, np.ndarray] = {}
    for key, first_array in first_arrays.items():
        if np.issubdtype(first_array.dtype, np.floating):
            accumulator = np.zeros(first_array.shape, dtype=np.float64)
            for path in args.inputs:
                with np.load(path, allow_pickle=False) as checkpoint:
                    accumulator += checkpoint[key].astype(np.float64, copy=False)
            output_arrays[key] = (accumulator / len(args.inputs)).astype(first_array.dtype)
            continue
        for path in args.inputs[1:]:
            with np.load(path, allow_pickle=False) as checkpoint:
                if not np.array_equal(first_array, checkpoint[key]):
                    raise ValueError(f"Non-floating GraphCast metadata differs for {key!r} in {path}")
        output_arrays[key] = first_array

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_arrays)
    provenance = args.output.with_suffix(args.output.suffix + ".provenance.json")
    provenance.write_text(
        json.dumps(
            {
                "checkpoint_type": "graphcast",
                "swa_source_steps": args.source_steps,
                "swa_source_ckpts": [str(path) for path in args.inputs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    if args.checkpoint_type == "residual":
        build_residual(args)
    else:
        build_graphcast(args)
    print(f"Saved SWA checkpoint: {args.output}")


if __name__ == "__main__":
    main()
