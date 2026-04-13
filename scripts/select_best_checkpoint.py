#!/usr/bin/env python3
"""
Select best checkpoint per run directory from eval_loss.json and copy it
to a standardized filename (default: ckpt_best.npz).

By default, "best" means the minimum validation loss value in eval_loss.json.
If there is no checkpoint saved exactly at that step, the script uses the
latest checkpoint at or before that step ("previous" mode).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable


CKPT_RE = re.compile(r"^ckpt_step(\d+)\.npz$")


def _load_eval_pairs(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    out: list[tuple[int, float]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((int(item[0]), float(item[1])))
            except Exception:
                continue
    return out


def _list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in run_dir.iterdir():
        if not p.is_file():
            continue
        m = CKPT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out, key=lambda x: x[0])


def _choose_checkpoint(
    ckpts: list[tuple[int, Path]],
    target_step: int,
    match_mode: str,
) -> tuple[int, Path, str] | None:
    if not ckpts:
        return None

    exact = [(s, p) for s, p in ckpts if s == target_step]
    if exact:
        s, p = exact[0]
        return s, p, "exact"

    if match_mode == "previous":
        prev = [(s, p) for s, p in ckpts if s <= target_step]
        if prev:
            s, p = prev[-1]
            return s, p, "previous"
        s, p = ckpts[0]
        return s, p, "next"

    # nearest
    s, p = min(ckpts, key=lambda x: (abs(x[0] - target_step), x[0]))
    return s, p, "nearest"


def _discover_run_dirs(roots: Iterable[Path]) -> list[Path]:
    run_dirs: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            print(f"Not a directory: {root}", file=sys.stderr)
            continue
        for p in root.rglob("eval_loss.json"):
            run_dirs.add(p.parent)
        if (root / "eval_loss.json").exists():
            run_dirs.add(root)
    return sorted(run_dirs)


def select_and_copy_best(
    run_dir: Path,
    *,
    output_name: str,
    match_mode: str,
    dry_run: bool,
) -> tuple[bool, str]:
    eval_pairs = _load_eval_pairs(run_dir / "eval_loss.json")
    if not eval_pairs:
        return False, "missing/invalid eval_loss.json"

    ckpts = _list_checkpoints(run_dir)
    if not ckpts:
        return False, "no ckpt_step*.npz files"

    best_eval_step, best_eval_loss = min(eval_pairs, key=lambda x: (x[1], x[0]))
    chosen = _choose_checkpoint(ckpts, best_eval_step, match_mode)
    if chosen is None:
        return False, "no checkpoint could be selected"
    chosen_step, chosen_path, match_type = chosen

    dst = run_dir / output_name
    if not dry_run:
        shutil.copy2(chosen_path, dst)

        meta = {
            "best_eval_step": int(best_eval_step),
            "best_eval_loss": float(best_eval_loss),
            "selected_checkpoint": chosen_path.name,
            "selected_checkpoint_step": int(chosen_step),
            "match_type": match_type,
            "output_file": output_name,
            "match_mode": match_mode,
        }
        with (run_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    msg = (
        f"best_eval(step={best_eval_step}, loss={best_eval_loss:.6f}) "
        f"-> {chosen_path.name} ({match_type}) as {output_name}"
    )
    return True, msg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Root dirs to scan recursively (e.g. artifacts/checkpoints).",
    )
    parser.add_argument(
        "--output-name",
        default="ckpt_best.npz",
        help="Filename to copy best checkpoint to (default: ckpt_best.npz).",
    )
    parser.add_argument(
        "--match-mode",
        choices=["previous", "nearest"],
        default="previous",
        help=(
            "How to map best eval step to saved checkpoint if no exact step exists. "
            "'previous' (default) picks latest <= step; 'nearest' picks closest step."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be selected, without copying files.",
    )
    args = parser.parse_args()

    run_dirs = _discover_run_dirs(args.roots)
    if not run_dirs:
        print("No run directories with eval_loss.json found.")
        return

    ok = 0
    skipped = 0
    for run_dir in run_dirs:
        updated, msg = select_and_copy_best(
            run_dir,
            output_name=args.output_name,
            match_mode=args.match_mode,
            dry_run=args.dry_run,
        )
        if updated:
            print(f"Updated: {run_dir} | {msg}")
            ok += 1
        else:
            print(f"Skipped: {run_dir} | {msg}")
            skipped += 1

    print(
        f"Done. processed={len(run_dirs)} updated={ok} skipped={skipped} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()

