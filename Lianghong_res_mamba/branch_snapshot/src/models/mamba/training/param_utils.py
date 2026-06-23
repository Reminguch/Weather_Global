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


def trainable_label(module_name: str, param_name: str, trainable_part: str) -> str:
    if trainable_part == "all":
        return "train"
    is_temporal = is_temporal_param(module_name, param_name)
    if trainable_part == "mamba":
        return "train" if is_temporal else "freeze"
    if trainable_part == "graphcast":
        return "freeze" if is_temporal else "train"
    raise ValueError(f"Unsupported trainable_part={trainable_part!r}")


def build_trainable_labels(params: Mapping[str, Mapping[str, Any]], trainable_part: str) -> dict[str, dict[str, str]]:
    return {
        module_name: {
            param_name: trainable_label(module_name, param_name, trainable_part)
            for param_name in module_params
        }
        for module_name, module_params in params.items()
    }


def overlay_matching_params(
    initial_params: Mapping[str, Mapping[str, Any]],
    source_params: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], OverlayStats]:
    """Copy source leaves into an initialized target tree when keys/shapes match.

    The source tree is expected to be the vanilla GraphCast parameter tree, and
    the initialized tree may include additional temporal Mamba leaves.
    """
    merged = {
        module_name: dict(module_params)
        for module_name, module_params in initial_params.items()
    }
    missing: list[str] = []
    mismatched: list[str] = []
    copied = 0

    for module_name, source_module in source_params.items():
        target_module = merged.get(module_name)
        if target_module is None:
            missing.extend(f"{module_name}/{param_name}" for param_name in source_module)
            continue
        for param_name, source_leaf in source_module.items():
            if param_name not in target_module:
                missing.append(f"{module_name}/{param_name}")
                continue
            target_leaf = target_module[param_name]
            if getattr(source_leaf, "shape", None) != getattr(target_leaf, "shape", None):
                mismatched.append(
                    f"{module_name}/{param_name}: "
                    f"source={getattr(source_leaf, 'shape', None)} "
                    f"target={getattr(target_leaf, 'shape', None)}"
                )
                continue
            target_module[param_name] = source_leaf
            copied += 1

    if missing or mismatched:
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
