#!/usr/bin/env python
"""Merge staged GraphCast history into one directory with global step numbers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


PAIR_FILES = ("train_loss.json", "eval_loss.json", "step_times.json", "memory_gib.json")
STEP_OBJECT_FILES = ("eval_details.json", "actual_usage.json", "timing_details.json")
LIST_FILES = ("epoch_summary.json",)


def atomic_json_dump(path: Path, value: object, *, indent: int | None = None) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_json(path: Path) -> list[object]:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON array: {path}")
    return value


def parse_stage(value: str) -> tuple[str, Path, int]:
    try:
        label, directory, offset_text = value.split("|", 2)
        offset = int(offset_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stage must be LABEL|DIRECTORY|GLOBAL_STEP_OFFSET") from exc
    if not label or offset < 0:
        raise argparse.ArgumentTypeError(f"Invalid stage: {value}")
    return label, Path(directory), offset


def offset_pairs(records: list[object], offset: int, source: Path) -> list[list[object]]:
    result: list[list[object]] = []
    for record in records:
        if not isinstance(record, list) or not record or not isinstance(record[0], int):
            raise ValueError(f"Expected [step, value] record in {source}")
        result.append([record[0] + offset, *record[1:]])
    return result


def offset_step_objects(records: list[object], offset: int, source: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("step"), int):
            raise ValueError(f"Expected object with integer step in {source}")
        updated = dict(record)
        updated["step"] = int(updated["step"]) + offset
        result.append(updated)
    return result


def link_checkpoint(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to replace checkpoint alias: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--stage", action="append", required=True, type=parse_stage)
    args = parser.parse_args()

    target = args.target.resolve()
    stages = [(label, directory.resolve(), offset) for label, directory, offset in args.stage]
    lineage_path = target / "training_lineage.json"
    if lineage_path.exists():
        print(f"Global-step history already prepared: {lineage_path}")
        return
    if not target.is_dir():
        raise FileNotFoundError(f"Missing target directory: {target}")
    for _, directory, _ in stages:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing history stage: {directory}")

    backup_dir = target / "history_local_step_numbering"
    checkpoint_backup_dir = target / "checkpoints_local_step_numbering"
    if backup_dir.exists() or checkpoint_backup_dir.exists():
        raise FileExistsError("Found an incomplete history preparation; inspect backups before retrying")
    backup_dir.mkdir()
    checkpoint_backup_dir.mkdir()

    files = (*PAIR_FILES, *STEP_OBJECT_FILES, *LIST_FILES)
    for filename in files:
        source = target / filename
        if source.is_file():
            shutil.copy2(source, backup_dir / filename)

    # Load everything before replacing data in the target, which is also the
    # final stage in the ordinary continuation chain.
    merged_pairs = {filename: [] for filename in PAIR_FILES}
    merged_objects = {filename: [] for filename in STEP_OBJECT_FILES}
    merged_lists = {filename: [] for filename in LIST_FILES}
    lineage_stages: list[dict[str, object]] = []
    for label, directory, offset in stages:
        stage_info: dict[str, object] = {"label": label, "directory": str(directory), "global_step_offset": offset}
        for filename in PAIR_FILES:
            records = load_json(directory / filename)
            merged_pairs[filename].extend(offset_pairs(records, offset, directory / filename))
            if filename == "train_loss.json" and records:
                stage_info["local_step_range"] = [records[0][0], records[-1][0]]
                stage_info["global_step_range"] = [records[0][0] + offset, records[-1][0] + offset]
        for filename in STEP_OBJECT_FILES:
            merged_objects[filename].extend(offset_step_objects(load_json(directory / filename), offset, directory / filename))
        for filename in LIST_FILES:
            for record in load_json(directory / filename):
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object in {directory / filename}")
                merged_lists[filename].append({"stage": label, "global_step_offset": offset, **record})
        lineage_stages.append(stage_info)

    for filename, records in merged_pairs.items():
        atomic_json_dump(target / filename, records)
    for filename, records in merged_objects.items():
        atomic_json_dump(target / filename, records, indent=2)
    for filename, records in merged_lists.items():
        atomic_json_dump(target / filename, records, indent=2)

    # Local names collide with global aliases (e.g. local 50k vs global 50k),
    # so keep the originals before publishing canonical global-step aliases.
    for checkpoint in target.glob("ckpt_step*.npz"):
        shutil.move(str(checkpoint), checkpoint_backup_dir / checkpoint.name)
    for _, directory, offset in stages:
        checkpoint_dir = checkpoint_backup_dir if directory == target else directory
        for checkpoint in sorted(checkpoint_dir.glob("ckpt_step*.npz")):
            local_step = int(checkpoint.stem.removeprefix("ckpt_step"))
            link_checkpoint(checkpoint, target / f"ckpt_step{local_step + offset}.npz")

    final_step = max(record[0] for record in merged_pairs["train_loss.json"])
    atomic_json_dump(
        lineage_path,
        {
            "description": "Global-step history merged from staged vanilla GraphCast continuations.",
            "stages": lineage_stages,
            "resume_from_global_step": final_step,
            "local_history_backup": str(backup_dir),
            "local_checkpoint_backup": str(checkpoint_backup_dir),
        },
        indent=2,
    )
    print(f"Prepared global history through step {final_step}")


if __name__ == "__main__":
    main()
