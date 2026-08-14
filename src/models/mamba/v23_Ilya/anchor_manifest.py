"""Build the anchor-only manifest used by standalone v23_Ilya training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.graphcast.training.core.batching import input_steps_from_duration
from src.models.graphcast.training.core.model import load_graphcast_checkpoint
from src.models.graphcast.training.core.prepared_array import PreparedArrayStore

from .config import ARCHITECTURE_ID


def build_anchor_manifest(
    *,
    prepared_root: Path,
    baseline_checkpoint: Path,
    output_root: Path,
    train_end_year: int = 2021,
    validation_year: int = 2022,
) -> dict:
    """Create deterministic contiguous anchors without residual-array placeholders."""

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite anchor manifest: {output_root}")
    checkpoint = load_graphcast_checkpoint(baseline_checkpoint)
    task_config = checkpoint.task_config
    model_config = checkpoint.model_config
    store = PreparedArrayStore(prepared_root, label="v23-Ilya-anchor-builder")
    store.validate(resolution=model_config.resolution, task_cfg=task_config)
    time_values = np.asarray(store.time.values).astype("datetime64[ns]")
    time_step = pd.Timedelta(time_values[1] - time_values[0])
    input_steps = input_steps_from_duration(task_config.input_duration, time_step)

    # Match the proven res1 anchor safety margin: one additional frame is kept
    # unused at each edge beyond the input/target requirement.
    anchors = np.arange(
        input_steps,
        store.sizes["time"] - 2,
        dtype=np.int64,
    )
    anchor_times = time_values[anchors]
    years = pd.DatetimeIndex(anchor_times).year.to_numpy()
    train_split = np.flatnonzero(years <= train_end_year).astype(np.int64)
    val_split = np.flatnonzero(years == validation_year).astype(np.int64)
    if train_split.size == 0 or val_split.size == 0:
        raise ValueError(
            f"Empty train/validation split: train={train_split.size}, val={val_split.size}"
        )
    if train_split[-1] + 1 != val_split[0]:
        raise ValueError("Train and validation anchors must form adjacent chronological splits")

    output_root.mkdir(parents=True)
    incomplete = output_root / ".incomplete"
    incomplete.write_text("building\n", encoding="utf-8")
    anchors_root = output_root / "anchors"
    anchors_root.mkdir()
    np.save(anchors_root / "anchor_indices.npy", anchors, allow_pickle=False)
    np.save(anchors_root / "anchor_times.npy", anchor_times, allow_pickle=False)
    np.save(anchors_root / "split_train.npy", train_split, allow_pickle=False)
    np.save(anchors_root / "split_val.npy", val_split, allow_pickle=False)
    metadata = {
        "format_version": 1,
        "manifest_kind": "anchor_only",
        "architecture_id": ARCHITECTURE_ID,
        "source_prepared_root": str(prepared_root),
        "baseline_checkpoint": str(baseline_checkpoint),
        "resolution": float(model_config.resolution),
        "mesh_size": int(model_config.mesh_size),
        "baseline_msg_steps": int(model_config.gnn_msg_steps),
        "input_duration": str(task_config.input_duration),
        "input_steps": int(input_steps),
        "target_steps": 1,
        "n_anchors_total": int(anchors.size),
        "n_anchors_train": int(train_split.size),
        "n_anchors_val": int(val_split.size),
        "train_end_year": int(train_end_year),
        "validation_year": int(validation_year),
        "pressure_levels": [int(value) for value in task_config.pressure_levels],
        "target_variables": list(task_config.target_variables),
        "anchor_indices_file": "anchors/anchor_indices.npy",
        "anchor_times_file": "anchors/anchor_times.npy",
        "anchor_split": {
            "train": "anchors/split_train.npy",
            "val": "anchors/split_val.npy",
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    incomplete.unlink()
    return metadata
