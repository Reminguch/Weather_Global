from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import RunConfig
from .dataset import prepared_store_path
from .model import checkpoint, gc


_STEP_CKPT_RE = re.compile(r"^ckpt_step(\d+)\.npz$")


def prune_old_step_checkpoints(out_dir: Path, *, keep_step: int) -> None:
    keep_name = f"ckpt_step{keep_step}.npz"
    removed = 0
    for path in out_dir.glob("ckpt_step*.npz"):
        match = _STEP_CKPT_RE.match(path.name)
        if match is None or path.name == keep_name:
            continue
        path.unlink()
        removed += 1
    if removed:
        print(f"pruned {removed} old step checkpoint(s), kept {out_dir / keep_name}")


def save_checkpoint(
    out_dir: Path,
    *,
    params: hk.Params,
    step: int,
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    description: str,
    license_text: str,
    filename: str | None = None,
) -> None:
    path = out_dir / (filename if filename is not None else f"ckpt_step{step}.npz")
    ckpt_out = gc.CheckPoint(
        params=params,
        model_config=model_cfg,
        task_config=task_cfg,
        description=description,
        license=license_text,
    )
    with path.open("wb") as f:
        checkpoint.dump(f, ckpt_out)
    print(f"saved checkpoint: {path}")
    if filename is None:
        prune_old_step_checkpoints(out_dir, keep_step=step)


def save_logs(
    out_dir: Path,
    train_losses: list[tuple[int, float]],
    eval_losses: list[tuple[int, float]],
    eval_details: list[dict[str, Any]],
    step_times: list[tuple[int, float]],
    timing_details: list[dict[str, Any]],
    mem_usage: list[tuple[int, float]],
    actual_usage: list[dict[str, Any]],
    epoch_summaries: list[dict[str, Any]],
) -> None:
    with (out_dir / "train_loss.json").open("w", encoding="utf-8") as f:
        json.dump(train_losses, f)
    with (out_dir / "eval_loss.json").open("w", encoding="utf-8") as f:
        json.dump(eval_losses, f)
    with (out_dir / "eval_details.json").open("w", encoding="utf-8") as f:
        json.dump(eval_details, f, indent=2)
    with (out_dir / "step_times.json").open("w", encoding="utf-8") as f:
        json.dump(step_times, f)
    with (out_dir / "timing_details.json").open("w", encoding="utf-8") as f:
        json.dump(timing_details, f, indent=2)
    with (out_dir / "memory_gib.json").open("w", encoding="utf-8") as f:
        json.dump(mem_usage, f)
    with (out_dir / "actual_usage.json").open("w", encoding="utf-8") as f:
        json.dump(actual_usage, f, indent=2)

    rss_vals = [float(x["proc_rss_gib"]) for x in actual_usage if x.get("proc_rss_gib") is not None]
    gpu_vals = [float(x["gpu_mem_gib"]) for x in actual_usage if x.get("gpu_mem_gib") is not None]
    with (out_dir / "actual_usage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rss_gib_peak": float(np.max(rss_vals)) if rss_vals else None,
                "rss_gib_avg": float(np.mean(rss_vals)) if rss_vals else None,
                "gpu_mem_gib_peak": float(np.max(gpu_vals)) if gpu_vals else None,
                "gpu_mem_gib_avg": float(np.mean(gpu_vals)) if gpu_vals else None,
                "samples": len(actual_usage),
            },
            f,
            indent=2,
        )
    with (out_dir / "epoch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(epoch_summaries, f, indent=2)


def build_batch_builder_metadata(
    *,
    requested_batch_builder: str,
    effective_train_batch_builder: str | None = None,
    effective_eval_batch_builder: str | None = None,
    numpy_cache_active: bool = False,
) -> dict[str, Any]:
    train_builder = effective_train_batch_builder or requested_batch_builder
    eval_builder = effective_eval_batch_builder or requested_batch_builder
    used_fallback = requested_batch_builder == "numpy" and eval_builder != "numpy"
    return {
        "requested_batch_builder": requested_batch_builder,
        "effective_train_batch_builder": train_builder,
        "effective_eval_batch_builder": eval_builder,
        "numpy_cache_active": bool(numpy_cache_active),
        "used_fallback": used_fallback,
    }


def _load_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _load_step_value_pairs(path: Path) -> list[tuple[int, float]]:
    data = _load_json_list(path)
    out: list[tuple[int, float]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((int(item[0]), float(item[1])))
    return out


def _load_train_losses(path: Path) -> list[tuple[int, float]]:
    data = _load_json_list(path)
    if not data:
        return []
    first = data[0]
    if isinstance(first, (list, tuple)):
        out: list[tuple[int, float]] = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((int(item[0]), float(item[1])))
        return out
    return [(i + 1, float(loss)) for i, loss in enumerate(data)]


def _filter_pairs_upto_step(data: list[tuple[int, float]], max_step: int) -> list[tuple[int, float]]:
    return [(int(step), float(value)) for step, value in data if int(step) <= max_step]


def _load_dict_series_upto_step(path: Path, max_step: int) -> list[dict[str, Any]]:
    data = _load_json_list(path)
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if step is None:
            continue
        if int(step) <= max_step:
            out.append(item)
    return out


def _read_proc_mem_gib() -> tuple[float | None, float | None]:
    """Return (current_rss_gib, peak_hwm_gib) from /proc/self/status."""
    try:
        with Path("/proc/self/status").open("r", encoding="utf-8") as f:
            lines = f.readlines()
        rss_kib: int | None = None
        hwm_kib: int | None = None
        for line in lines:
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kib = int(line.split()[1])
        rss_gib = float(rss_kib) / (1024**2) if rss_kib is not None else None
        hwm_gib = float(hwm_kib) / (1024**2) if hwm_kib is not None else None
        return rss_gib, hwm_gib
    except Exception:
        return None, None


def _read_gpu_mem_by_device() -> tuple[list[dict[str, float | int]], float | None]:
    """Return per-device GPU memory stats and total used GiB from nvidia-smi."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        devices: list[dict[str, float | int]] = []
        total_mib = 0.0
        for raw in proc.stdout.splitlines():
            row = raw.strip()
            if not row:
                continue
            parts = [x.strip() for x in row.split(",") if x.strip() != ""]
            if len(parts) < 3:
                continue
            index = int(parts[0])
            used_mib = float(parts[1])
            total_dev_mib = float(parts[2])
            total_mib += used_mib
            devices.append(
                {
                    "index": index,
                    "used_gib": used_mib / 1024.0,
                    "total_gib": total_dev_mib / 1024.0,
                }
            )
        return devices, (total_mib / 1024.0 if devices else None)
    except Exception:
        return [], None


def sample_actual_usage(step: int) -> dict[str, Any]:
    proc_rss_gib, proc_hwm_gib = _read_proc_mem_gib()
    gpu_devices, gpu_mem_gib = _read_gpu_mem_by_device()
    return {
        "step": step,
        "timestamp": time.time(),
        "proc_rss_gib": proc_rss_gib,
        "proc_hwm_gib": proc_hwm_gib,
        "gpu_mem_gib": gpu_mem_gib,
        "gpu_mem_total_gib": gpu_mem_gib,
        "gpu_devices": gpu_devices,
    }


def plot_loss_curves(
    out_dir: Path,
    train_losses: list[tuple[int, float]],
    eval_losses: list[tuple[int, float]],
) -> None:
    if not train_losses and not eval_losses:
        return

    plt.figure()
    y_vals: list[float] = []

    if train_losses:
        train_steps, train_vals = zip(*train_losses)
        plt.plot(train_steps, train_vals, label="train loss", alpha=0.6)
        y_vals.extend(train_vals)

    if eval_losses:
        eval_steps, eval_vals = zip(*eval_losses)
        plt.plot(eval_steps, eval_vals, marker="o", label="val loss")
        y_vals.extend(eval_vals)

    # Scale y-axis from validation-loss dynamics when available.
    # Train curve may be clipped by this range.
    y_ref = list(eval_vals) if eval_losses else y_vals
    if y_ref:
        y_min = min(y_ref)
        y_max = max(y_ref)
        y_span = y_max - y_min
        pad = 0.1 * y_span if y_span > 0 else max(1e-8, 0.1 * abs(y_max))
        lo = max(0.0, y_min - pad)
        hi = y_max + pad
        if hi > lo:
            plt.ylim(lo, hi)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Train and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_loss.png")
    plt.close()


def _write_run_config(
    out_dir: Path,
    cfg: RunConfig,
    model_cfg: gc.ModelConfig,
    task_cfg: gc.TaskConfig,
    *,
    numpy_cache_active: bool = False,
    train_cache_estimate_gib: float | None = None,
    effective_train_batch_builder: str | None = None,
    effective_eval_batch_builder: str | None = None,
) -> None:
    builder_metadata = build_batch_builder_metadata(
        requested_batch_builder=cfg.batch_builder,
        effective_train_batch_builder=effective_train_batch_builder,
        effective_eval_batch_builder=effective_eval_batch_builder,
        numpy_cache_active=numpy_cache_active,
    )
    payload = {
        "data_path": cfg.data_path,
        "data_source": cfg.data_source,
        "prepared_data_root": cfg.prepared_data_root,
        "prepared_store_path": str(prepared_store_path(cfg)) if cfg.data_source in {"prepared", "prepared_array"} else None,
        "val_year": cfg.val_year,
        "train_start_year": cfg.train_start_year,
        "train_end_year": cfg.train_end_year,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "max_steps": cfg.max_steps,
        "eval_every": cfg.eval_every,
        "eval_batch_size": cfg.eval_batch_size,
        "checkpoint_every": cfg.checkpoint_every,
        "seed": cfg.seed,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "precision": cfg.precision,
        "init_from_graphcast_ckpt": cfg.init_from_graphcast_ckpt,
        "trainable_part": cfg.trainable_part,
        "data_pipeline": {
            "data_source": cfg.data_source,
            "prepared_data_root": cfg.prepared_data_root,
            "prepared_store_path": str(prepared_store_path(cfg)) if cfg.data_source in {"prepared", "prepared_array"} else None,
            "data_cache_mode": cfg.data_cache_mode,
            "data_cache_max_gib": cfg.data_cache_max_gib,
            "batch_builder": cfg.batch_builder,
            "prefetch_workers": cfg.prefetch_workers,
            "prefetch_depth": cfg.prefetch_depth,
            "prefetch_device_depth": cfg.prefetch_device_depth,
            "usage_every": cfg.usage_every,
            **builder_metadata,
            "train_cache_estimate_gib": train_cache_estimate_gib,
        },
        "optimization": {
            "step_unit": "optimizer_updates",
            "microbatch_size": cfg.batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "effective_batch_size": cfg.batch_size * cfg.grad_accum_steps,
        },
        "temporal_config": {
            "backbone": cfg.temporal_backbone,
            "location": cfg.temporal_location,
            "stateful": cfg.temporal_stateful,
            "d_inner": cfg.temporal_d_inner,
            "d_state": cfg.temporal_d_state,
            "d_conv": cfg.temporal_d_conv,
            "dt_rank": cfg.temporal_dt_rank,
            "bias": cfg.temporal_bias,
            "conv_bias": cfg.temporal_conv_bias,
            "layers": cfg.temporal_layers,
            "dropout": cfg.temporal_dropout,
            "zero_init_output": cfg.zero_init_temporal_out,
        },
        "model_config": dataclasses.asdict(model_cfg),
        "task_config": dataclasses.asdict(task_cfg),
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
