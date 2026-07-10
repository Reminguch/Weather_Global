"""Parameter helpers for inserting temporal Mamba into pretrained GraphCast."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OverlayStats:
    copied: int
    initialized: int


def is_temporal_param(module_name: str, param_name: str) -> bool:
    path = f"{module_name}/{param_name}".lower()
    return "temporal" in path or "mamba" in path


def is_encoder_or_decoder_param(module_name: str, param_name: str) -> bool:
    """True iff the param belongs to grid2mesh (encoder) or mesh2grid (decoder)."""
    path = f"{module_name}/{param_name}".lower()
    return ("grid2mesh" in path) or ("mesh2grid" in path)


def trainable_label(module_name: str, param_name: str, trainable_part: str) -> str:
    if trainable_part == "all":
        return "train"
    is_temporal = is_temporal_param(module_name, param_name)
    if trainable_part == "mamba":
        return "train" if is_temporal else "freeze"
    if trainable_part == "graphcast":
        return "freeze" if is_temporal else "train"
    if trainable_part == "proc_and_mamba":
        # Freeze grid2mesh (encoder) and mesh2grid (decoder); train processor
        # (mesh_gnn) + Mamba blocks. Used for res_mamba when GC2's g2m/m2g
        # are overlaid from GC1 and held fixed.
        return "freeze" if is_encoder_or_decoder_param(module_name, param_name) else "train"
    raise ValueError(f"Unsupported trainable_part={trainable_part!r}")


def build_trainable_labels(params: Mapping[str, Mapping[str, Any]], trainable_part: str) -> dict[str, dict[str, str]]:
    return {
        module_name: {
            param_name: trainable_label(module_name, param_name, trainable_part)
            for param_name in module_params
        }
        for module_name, module_params in params.items()
    }


def partition_params_by_trainable_part(
    params: Mapping[str, Mapping[str, Any]],
    trainable_part: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    trainable: dict[str, dict[str, Any]] = {}
    frozen: dict[str, dict[str, Any]] = {}
    for module_name, module_params in params.items():
        for param_name, leaf in module_params.items():
            target = (
                trainable
                if trainable_label(module_name, param_name, trainable_part) == "train"
                else frozen
            )
            target.setdefault(module_name, {})[param_name] = leaf
    return trainable, frozen


def merge_param_partitions(
    trainable_params: Mapping[str, Mapping[str, Any]],
    frozen_params: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = {
        module_name: dict(module_params)
        for module_name, module_params in frozen_params.items()
    }
    for module_name, module_params in trainable_params.items():
        merged.setdefault(module_name, {}).update(module_params)
    return merged


def overlay_matching_params(
    initial_params: Mapping[str, Mapping[str, Any]],
    source_params: Mapping[str, Mapping[str, Any]],
    strict: bool = True,
) -> tuple[dict[str, dict[str, Any]], OverlayStats]:
    """Copy source leaves into an initialized target tree when keys/shapes match.

    The source tree is expected to be the vanilla GraphCast parameter tree, and
    the initialized tree may include additional temporal Mamba leaves.

    With strict=True (default): raise if any source key is missing from target
        or has mismatched shape. Use this when target should contain source
        (e.g. same architecture).

    With strict=False: silently skip source keys missing from target. Use this
        when overlaying a LARGER baseline into a SMALLER target (e.g. GC1 mp=6
        into GC2 mp=2 — only steps 0..1 will be copied, steps 2..5 ignored).
    """
    merged = {
        module_name: dict(module_params)
        for module_name, module_params in initial_params.items()
    }
    missing: list[str] = []
    mismatched: list[str] = []
    copied = 0
    skipped = 0

    for module_name, source_module in source_params.items():
        target_module = merged.get(module_name)
        if target_module is None:
            if strict:
                missing.extend(f"{module_name}/{param_name}" for param_name in source_module)
            else:
                skipped += len(source_module)
            continue
        for param_name, source_leaf in source_module.items():
            if param_name not in target_module:
                if strict:
                    missing.append(f"{module_name}/{param_name}")
                else:
                    skipped += 1
                continue
            target_leaf = target_module[param_name]
            if getattr(source_leaf, "shape", None) != getattr(target_leaf, "shape", None):
                if strict:
                    mismatched.append(
                        f"{module_name}/{param_name}: "
                        f"source={getattr(source_leaf, 'shape', None)} "
                        f"target={getattr(target_leaf, 'shape', None)}"
                    )
                else:
                    skipped += 1
                continue
            target_module[param_name] = source_leaf
            copied += 1

    if strict and (missing or mismatched):
        detail = []
        if missing:
            detail.append(f"missing={missing[:8]}")
        if mismatched:
            detail.append(f"mismatched={mismatched[:8]}")
        raise ValueError(
            "Cannot overlay vanilla GraphCast params into GC-Mamba params; "
            "architecture keys/shapes do not match. "
            + "; ".join(detail)
        )

    initialized = sum(
        1
        for module_name, module_params in initial_params.items()
        for param_name in module_params
        if module_name not in source_params or param_name not in source_params[module_name]
    )
    return merged, OverlayStats(copied=copied, initialized=initialized)
