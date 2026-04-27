#!/usr/bin/env python3
"""Train GraphCast/Mamba on residual targets over shuffled chronological segments."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CURRENT_DIR.parent
for path in (TRAINING_DIR,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xarray as xr

from graphcast_train.batching import (
    NumpyBatchCache,
    build_batch_from_indices,
    build_batch_from_indices_vectorized,
    infer_time_step,
    input_steps_from_duration,
)
from graphcast_train.dataset import _open_local_splits, _training_cache_decision, maybe_cache_training_data, prepare_dataset_for_task
from graphcast_train.logging import (
    _filter_pairs_upto_step,
    _load_dict_series_upto_step,
    _load_json_list,
    _load_step_value_pairs,
    _load_train_losses,
    build_batch_builder_metadata,
    plot_loss_curves,
    sample_actual_usage,
    save_checkpoint,
    save_logs,
)
from graphcast_train.model import load_graphcast_checkpoint, load_stats, scalarize_loss, validate_stats_coverage
from graphcast_train.segments import (
    SegmentBatchScheduler,
    _build_chunk_batches,
    _reset_temporal_state_lanes,
    _save_chunk_timing_logs,
    _stop_gradient_temporal_state,
    build_full_segments,
    valid_contiguous_final_input_indices,
)
from residual_memory.config import parse_args
from residual_memory.utils import (
    augment_run_config,
    build_eval_loss_transform,
    build_loss_transform,
    build_predict_transform,
    run_residual_eval,
)


def main() -> None:
    segment_cfg = parse_args()
    cfg = segment_cfg.base_cfg
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ckpt = load_graphcast_checkpoint(Path(segment_cfg.baseline_ckpt))
    resume_ckpt = load_graphcast_checkpoint(Path(segment_cfg.resume_ckpt)) if segment_cfg.resume_ckpt else None

    base_model_cfg = baseline_ckpt.model_config
    task_cfg = baseline_ckpt.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)

    model_cfg = dataclasses.replace(
        base_model_cfg,
        resolution=cfg.resolution,
        mesh_size=cfg.mesh_size,
        latent_size=cfg.width,
        gnn_msg_steps=cfg.processor_msg_steps,
        hidden_layers=1,
        mesh2grid_edge_normalization_factor=None,
    )

    norm_stats = load_stats(Path(cfg.stats_dir))
    validate_stats_coverage(task_cfg, norm_stats)
    train_ds, eval_ds = _open_local_splits(cfg)
    train_ds = prepare_dataset_for_task(train_ds, task_cfg)
    eval_ds = prepare_dataset_for_task(eval_ds, task_cfg)

    dt_train = infer_time_step(train_ds)
    dt_eval = infer_time_step(eval_ds)
    if dt_train != dt_eval:
        raise ValueError(f"Train/eval time step mismatch: train={dt_train}, eval={dt_eval}")

    input_steps = input_steps_from_duration(task_cfg.input_duration, dt_train)
    if input_steps < 2:
        raise ValueError("Residual segment training expects at least two input frames.")
    target_steps = cfg.target_steps

    train_final_indices = valid_contiguous_final_input_indices(
        train_ds,
        input_steps=input_steps,
        target_steps=target_steps,
        dt=dt_train,
    )
    eval_final_indices = valid_contiguous_final_input_indices(
        eval_ds,
        input_steps=input_steps,
        target_steps=target_steps,
        dt=dt_train,
    )
    segments = build_full_segments(train_final_indices, segment_cfg.len_segment)
    if not segments:
        raise ValueError(
            "No full training segments after timestamp-contiguous filtering. "
            f"len_segment={segment_cfg.len_segment}, valid_windows={len(train_final_indices)}"
        )
    print(
        "Prepared residual segment windows: "
        f"train_windows={len(train_final_indices)}, eval_windows={len(eval_final_indices)}, "
        f"segments={len(segments)}, len_segment={segment_cfg.len_segment}, "
        f"bptt_steps={segment_cfg.bptt_steps}, input_steps={input_steps}, target_steps={target_steps}"
    )

    should_cache_train, train_cache_estimate_gib = _training_cache_decision(train_ds, cfg, task_cfg)
    train_ds, eval_ds = maybe_cache_training_data(train_ds, eval_ds, cfg, task_cfg)

    numpy_cache_active = False
    train_numpy_cache: NumpyBatchCache | None = None
    eval_numpy_cache: NumpyBatchCache | None = None
    effective_train_batch_builder = cfg.batch_builder
    effective_eval_batch_builder = cfg.batch_builder
    if cfg.batch_builder == "numpy":
        if should_cache_train:
            train_numpy_cache = NumpyBatchCache(train_ds, task_cfg, label="residual-segment-train")
            eval_numpy_cache = NumpyBatchCache(eval_ds, task_cfg, label="residual-segment-eval")
            numpy_cache_active = True
        else:
            effective_train_batch_builder = "vectorized"
            effective_eval_batch_builder = "vectorized"
            print(
                "[numpy-cache] requested for residual segments but train split is not cached; "
                "falling back to vectorized builder. Use --data-cache-mode=always to force it."
            )

    if numpy_cache_active:
        assert train_numpy_cache is not None
        assert eval_numpy_cache is not None

        def train_batch_builder(
            _ds: xr.Dataset,
            *,
            indices: Iterable[int],
            input_steps: int,
            target_steps: int,
            task_cfg,
            dt: pd.Timedelta,
        ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
            return train_numpy_cache.build_batch_from_indices(
                indices=indices,
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt,
            )

        def eval_batch_builder(
            _ds: xr.Dataset,
            *,
            indices: Iterable[int],
            input_steps: int,
            target_steps: int,
            task_cfg,
            dt: pd.Timedelta,
        ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
            return eval_numpy_cache.build_batch_from_indices(
                indices=indices,
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt,
            )
    elif cfg.batch_builder == "legacy":
        train_batch_builder = build_batch_from_indices
        eval_batch_builder = build_batch_from_indices
    else:
        train_batch_builder = build_batch_from_indices_vectorized
        eval_batch_builder = build_batch_from_indices_vectorized

    residual_loss_transform = build_loss_transform(model_cfg, task_cfg, norm_stats, cfg)
    residual_eval_transform = build_eval_loss_transform(
        model_cfg,
        task_cfg,
        norm_stats,
        cfg,
        temporal_backbone=cfg.temporal_backbone,
        temporal_location=cfg.temporal_location,
        temporal_hidden_size=cfg.temporal_hidden_size,
        temporal_d_inner=cfg.temporal_d_inner,
        temporal_d_state=cfg.temporal_d_state,
        temporal_d_conv=cfg.temporal_d_conv,
        temporal_dt_rank=cfg.temporal_dt_rank,
        temporal_bias=cfg.temporal_bias,
        temporal_conv_bias=cfg.temporal_conv_bias,
        temporal_layers=cfg.temporal_layers,
        temporal_dropout=cfg.temporal_dropout,
        temporal_stateful=cfg.temporal_stateful,
    )
    baseline_predict_transform = build_predict_transform(
        base_model_cfg,
        task_cfg,
        norm_stats,
        cfg,
        temporal_backbone="none",
        temporal_location="mesh_post_encoder",
        temporal_hidden_size=cfg.temporal_hidden_size,
        temporal_d_inner=None,
        temporal_d_state=cfg.temporal_d_state,
        temporal_d_conv=cfg.temporal_d_conv,
        temporal_dt_rank="auto",
        temporal_bias=False,
        temporal_conv_bias=True,
        temporal_layers=1,
        temporal_dropout=0.0,
        temporal_stateful=False,
    )

    rng = jax.random.PRNGKey(cfg.seed)
    init_indices = [int(segments[lane % len(segments)][0]) for lane in range(cfg.batch_size)]
    sample_inputs, sample_targets, sample_forcings = train_batch_builder(
        train_ds,
        indices=init_indices,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
    )
    params, state = residual_loss_transform.init(rng, sample_inputs, sample_targets, sample_forcings, True)
    if cfg.resume_step is not None:
        assert resume_ckpt is not None
        params = resume_ckpt.params
        print(f"Resuming residual model from step {cfg.resume_step} ({segment_cfg.resume_ckpt})")
    else:
        print("Residual model uses fresh initialization; frozen baseline is used only for residual targets.")
        if cfg.temporal_backbone == "mamba":
            print("Residual Mamba fresh init uses zero-initialized temporal out_proj, so step-0 residual output is zero.")

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)
    augment_run_config(
        out_dir,
        segment_cfg=segment_cfg,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        numpy_cache_active=numpy_cache_active,
        train_cache_estimate_gib=train_cache_estimate_gib,
        effective_train_batch_builder=effective_train_batch_builder,
        effective_eval_batch_builder=effective_eval_batch_builder,
    )
    batch_builder_metadata = build_batch_builder_metadata(
        requested_batch_builder=cfg.batch_builder,
        effective_train_batch_builder=effective_train_batch_builder,
        effective_eval_batch_builder=effective_eval_batch_builder,
        numpy_cache_active=numpy_cache_active,
    )

    _, baseline_train_state = baseline_predict_transform.init(
        rng,
        sample_inputs,
        sample_targets,
        sample_forcings,
        False,
    )

    @functools.partial(jax.jit)
    def train_chunk(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        rng_key: jax.Array,
        chunk_inputs: tuple[xr.Dataset, ...],
        chunk_targets: tuple[xr.Dataset, ...],
        chunk_forcings: tuple[xr.Dataset, ...],
        reset_mask: jax.Array,
    ):
        state = _reset_temporal_state_lanes(state, reset_mask)

        def loss_fn(p, s, key):
            current_state = s
            losses = []
            keys = jax.random.split(key, segment_cfg.bptt_steps)
            for bptt_i in range(segment_cfg.bptt_steps):
                baseline_key, residual_key = jax.random.split(keys[bptt_i])
                baseline_preds, _ = baseline_predict_transform.apply(
                    baseline_ckpt.params,
                    baseline_train_state,
                    baseline_key,
                    chunk_inputs[bptt_i],
                    chunk_targets[bptt_i],
                    chunk_forcings[bptt_i],
                    False,
                )
                residual_targets = chunk_targets[bptt_i] - baseline_preds
                (loss_and_diag, current_state) = residual_loss_transform.apply(
                    p,
                    current_state,
                    residual_key,
                    chunk_inputs[bptt_i],
                    residual_targets,
                    chunk_forcings[bptt_i],
                    True,
                )
                losses.append(scalarize_loss(loss_and_diag[0]))
            return jnp.mean(jnp.stack(losses)), current_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, _stop_gradient_temporal_state(new_state), new_opt_state, loss

    step = cfg.resume_step if cfg.resume_step is not None else 0
    train_losses: list[tuple[int, float]] = []
    eval_losses: list[tuple[int, float]] = []
    eval_details: list[dict[str, Any]] = []
    step_times: list[tuple[int, float]] = []
    mem_usage: list[tuple[int, float]] = []
    actual_usage: list[dict[str, Any]] = []
    epoch_summaries: list[dict[str, Any]] = []
    chunk_timing: list[dict[str, Any]] = []

    if cfg.resume_step is not None:
        train_losses = _filter_pairs_upto_step(_load_train_losses(out_dir / "train_loss.json"), cfg.resume_step)
        eval_losses = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "eval_loss.json"), cfg.resume_step)
        step_times = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "step_times.json"), cfg.resume_step)
        mem_usage = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "memory_gib.json"), cfg.resume_step)
        eval_details = _load_dict_series_upto_step(out_dir / "eval_details.json", cfg.resume_step)
        actual_usage = _load_dict_series_upto_step(out_dir / "actual_usage.json", cfg.resume_step)
        epoch_summaries = _load_json_list(out_dir / "epoch_summary.json")
        chunk_timing = _load_dict_series_upto_step(out_dir / "chunk_timing.json", cfg.resume_step)

    best_eval_step: int | None = None
    best_eval_loss = float("inf")
    if eval_losses:
        best_eval_step, best_eval_loss = min(eval_losses, key=lambda x: (x[1], x[0]))
        print(f"[best:init] step {best_eval_step} val {best_eval_loss:.6f}")

    def maybe_save_best_checkpoint(eval_step: int, eval_total: float) -> None:
        nonlocal best_eval_step, best_eval_loss
        if eval_total >= best_eval_loss:
            return
        best_eval_step = int(eval_step)
        best_eval_loss = float(eval_total)
        save_checkpoint(
            out_dir,
            params=params,
            step=eval_step,
            model_cfg=model_cfg,
            task_cfg=task_cfg,
            description=baseline_ckpt.description,
            license_text=baseline_ckpt.license,
            filename="ckpt_best.npz",
        )
        with (out_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_eval_step": best_eval_step,
                    "best_eval_loss": best_eval_loss,
                    "best_checkpoint": "ckpt_best.npz",
                    "match_type": "full_forecast_equivalent",
                },
                f,
                indent=2,
            )
        print(f"[best] updated step {best_eval_step} val {best_eval_loss:.6f}")

    scheduler = SegmentBatchScheduler(
        segments,
        batch_size=cfg.batch_size,
        bptt_steps=segment_cfg.bptt_steps,
        seed=cfg.seed,
    )
    pass_start_step = step + 1
    pass_loss_accum: list[float] = []
    observed_epoch: int | None = None

    def save_all_logs() -> None:
        save_logs(
            out_dir,
            train_losses,
            eval_losses,
            eval_details,
            step_times,
            [],
            mem_usage,
            actual_usage,
            epoch_summaries,
        )
        _save_chunk_timing_logs(out_dir, chunk_timing)

    load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=segment_cfg.chunk_load_workers)
    prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def load_chunk_payload(
        chunk_indices: tuple[np.ndarray, ...],
        reset_mask_np: np.ndarray,
        chunk_epoch: int,
    ) -> tuple[tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], tuple[xr.Dataset, ...], np.ndarray, int]:
        chunk_inputs, chunk_targets, chunk_forcings = _build_chunk_batches(
            train_ds,
            chunk_indices,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt_train,
            batch_builder=train_batch_builder,
            chunk_load_workers=segment_cfg.chunk_load_workers,
            load_executor=load_executor,
        )
        return chunk_inputs, chunk_targets, chunk_forcings, reset_mask_np, chunk_epoch

    def submit_next_chunk() -> concurrent.futures.Future:
        chunk_indices, reset_mask_np = scheduler.next_chunk()
        return prefetch_executor.submit(
            load_chunk_payload,
            chunk_indices,
            reset_mask_np,
            scheduler.epoch,
        )

    pending_chunk = submit_next_chunk()

    try:
        while step < cfg.max_steps:
            iteration_t0 = time.time()
            t_data = time.time()
            chunk_inputs, chunk_targets, chunk_forcings, reset_mask_np, chunk_epoch = pending_chunk.result()
            data_wait_s = time.time() - t_data

            if observed_epoch is None:
                observed_epoch = chunk_epoch
            elif chunk_epoch != observed_epoch and pass_loss_accum:
                epoch_summaries.append(
                    {
                        "pass": observed_epoch,
                        "steps": step - pass_start_step + 1,
                        "train_loss_mean": float(np.mean(pass_loss_accum)),
                        "time_per_step_mean": float(
                            np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                        ),
                        "mem_gib_max": float(
                            np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])
                        ),
                    }
                )
                observed_epoch = chunk_epoch
                pass_start_step = step + 1
                pass_loss_accum = []

            rng, step_key = jax.random.split(rng)
            next_chunk = submit_next_chunk() if step + 1 < cfg.max_steps else None
            t0 = time.time()
            params, state, opt_state, loss = train_chunk(
                params,
                state,
                opt_state,
                step_key,
                chunk_inputs,
                chunk_targets,
                chunk_forcings,
                jnp.asarray(reset_mask_np),
            )

            loss_f = float(loss)
            gpu_train_s = time.time() - t0
            iteration_wall_s = time.time() - iteration_t0

            step += 1
            train_losses.append((step, loss_f))
            pass_loss_accum.append(loss_f)
            step_times.append((step, gpu_train_s))
            chunk_timing.append(
                {
                    "step": step,
                    "data_wait_s": data_wait_s,
                    "gpu_train_s": gpu_train_s,
                    "iteration_wall_s": iteration_wall_s,
                    "batch_size": cfg.batch_size,
                    "bptt_steps": segment_cfg.bptt_steps,
                    "chunk_load_workers": segment_cfg.chunk_load_workers,
                }
            )

            usage = sample_actual_usage(step=step)
            actual_usage.append(usage)
            if usage.get("gpu_mem_gib") is not None:
                mem_usage.append((step, float(usage["gpu_mem_gib"])))

            if step % 200 == 0:
                print(
                    f"step {step}/{cfg.max_steps} loss {loss_f:.6f} "
                    f"segment_epoch {chunk_epoch} reset_lanes {int(reset_mask_np.sum())} "
                    f"data_wait={data_wait_s:.3f}s gpu={gpu_train_s:.3f}s iter={iteration_wall_s:.3f}s"
                )

            if step % cfg.eval_every == 0:
                eval_metrics = run_residual_eval(
                    residual_eval_transform,
                    baseline_predict_transform,
                    params,
                    baseline_ckpt.params,
                    rng,
                    eval_ds,
                    eval_final_indices,
                    eval_batch_size=cfg.eval_batch_size,
                    input_steps=input_steps,
                    target_steps=target_steps,
                    task_cfg=task_cfg,
                    dt=dt_train,
                    progress_label=f"eval@step{step}",
                    batch_builder=eval_batch_builder,
                )
                eval_losses.append((step, eval_metrics["total"]))
                maybe_save_best_checkpoint(step, float(eval_metrics["total"]))
                eval_details.append({"step": step, **eval_metrics, **batch_builder_metadata})
                print(f"[eval] step {step} total {eval_metrics['total']:.6f}")
                plot_loss_curves(out_dir, train_losses, eval_losses)
                save_all_logs()

            if step % cfg.checkpoint_every == 0:
                save_checkpoint(
                    out_dir,
                    params=params,
                    step=step,
                    model_cfg=model_cfg,
                    task_cfg=task_cfg,
                    description=baseline_ckpt.description,
                    license_text=baseline_ckpt.license,
                )

            if next_chunk is not None:
                pending_chunk = next_chunk
    finally:
        prefetch_executor.shutdown(wait=False, cancel_futures=True)
        load_executor.shutdown(wait=False, cancel_futures=True)

    if pass_loss_accum:
        epoch_summaries.append(
            {
                "pass": observed_epoch,
                "steps": step - pass_start_step + 1,
                "train_loss_mean": float(np.mean(pass_loss_accum)),
                "time_per_step_mean": float(
                    np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                ),
                "mem_gib_max": float(np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])),
            }
        )

    final_eval = run_residual_eval(
        residual_eval_transform,
        baseline_predict_transform,
        params,
        baseline_ckpt.params,
        rng,
        eval_ds,
        eval_final_indices,
        eval_batch_size=cfg.eval_batch_size,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
        progress_label="eval@final",
        batch_builder=eval_batch_builder,
    )
    eval_losses.append((step, final_eval["total"]))
    maybe_save_best_checkpoint(step, float(final_eval["total"]))
    eval_details.append({"step": step, "final": True, **final_eval, **batch_builder_metadata})

    save_checkpoint(
        out_dir,
        params=params,
        step=step,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        description=baseline_ckpt.description,
        license_text=baseline_ckpt.license,
    )
    save_all_logs()
    plot_loss_curves(out_dir, train_losses, eval_losses)
    print(f"Done. Final eval total {final_eval['total']:.6f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
