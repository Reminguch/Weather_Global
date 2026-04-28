#!/usr/bin/env python3
"""Train GraphCast/Mamba on shuffled chronological segments with chunked BPTT."""

from __future__ import annotations

import concurrent.futures
import argparse
import dataclasses
import sys
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class SegmentRunConfig:
    base_cfg: object
    len_segment: int
    bptt_steps: int
    chunk_load_workers: int


def parse_gc_mamba_args(argv: list[str] | None = None) -> SegmentRunConfig:
    from src.models.graphcast.training.core.config import (
        DEFAULT_CKPT,
        DEFAULT_DATA_PATH,
        DEFAULT_STATS_DIR,
        RunConfig,
    )

    parser = argparse.ArgumentParser(
        description="Train GraphCast on shuffled chronological segments with chunked BPTT."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--mesh-size", type=int, default=4)
    parser.add_argument("--width", type=int, choices=[128, 256, 512, 1024], default=128)
    parser.add_argument("--processor-msg-steps", type=int, choices=[1, 2, 3, 4], default=1)
    parser.add_argument("--val-year", type=int, default=2021)
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--ckpt-in", default=DEFAULT_CKPT)
    parser.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", default="artifacts/checkpoints/graphcast_mamba_interleaved_segments")
    parser.add_argument("--run-name", default="segments_res2_m4_w128_mp1")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000, help="Optimizer updates, not forecast windows.")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--input-duration", default=None)
    parser.add_argument("--target-steps", type=int, default=1)
    parser.add_argument("--len-segment", type=int, default=30)
    parser.add_argument("--bptt-steps", type=int, default=6)
    parser.add_argument(
        "--chunk-load-workers",
        type=int,
        default=6,
        help="Parallel workers for loading the independent BPTT batches in each chunk.",
    )
    parser.add_argument("--temporal-backbone", choices=["none", "mamba"], default="none")
    parser.add_argument(
        "--temporal-location",
        choices=["mesh_post_encoder", "mesh_processor_interleaved", "mesh_post_encoder_residual"],
        default="mesh_post_encoder",
    )
    parser.add_argument("--temporal-hidden-size", type=int, default=128)
    parser.add_argument("--temporal-d-inner", type=int, default=None)
    parser.add_argument("--temporal-d-state", type=int, default=16)
    parser.add_argument("--temporal-d-conv", type=int, default=4)
    parser.add_argument("--temporal-dt-rank", default="auto")
    parser.add_argument("--temporal-bias", action="store_true", default=False)
    parser.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias", action="store_false", default=True)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)
    parser.add_argument("--temporal-stateful", action="store_true", default=False)
    parser.add_argument("--data-cache-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--data-cache-max-gib", type=float, default=48.0)
    parser.add_argument("--batch-builder", choices=["legacy", "vectorized", "numpy"], default="numpy")
    args = parser.parse_args(argv)

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.len_segment <= 0:
        raise ValueError("--len-segment must be > 0")
    if args.bptt_steps <= 0:
        raise ValueError("--bptt-steps must be > 0")
    if args.chunk_load_workers <= 0:
        raise ValueError("--chunk-load-workers must be > 0")
    if args.len_segment % args.bptt_steps != 0:
        raise ValueError("--bptt-steps must divide --len-segment")
    if args.target_steps != 1:
        raise ValueError("Segment BPTT training currently requires --target-steps 1.")
    if args.train_start_year is not None and args.train_end_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_end_year is not None and args.train_start_year is None:
        raise ValueError("Provide both --train-start-year and --train-end-year, or neither.")
    if args.train_start_year is not None and args.train_start_year > args.train_end_year:
        raise ValueError("--train-start-year must be <= --train-end-year")
    if args.resume_step is not None and args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if args.temporal_hidden_size <= 0:
        raise ValueError("--temporal-hidden-size must be > 0")
    if args.temporal_d_inner is not None and args.temporal_d_inner <= 0:
        raise ValueError("--temporal-d-inner must be > 0")
    if args.temporal_d_state <= 0:
        raise ValueError("--temporal-d-state must be > 0")
    if args.temporal_d_conv <= 0:
        raise ValueError("--temporal-d-conv must be > 0")
    if args.temporal_dt_rank != "auto" and int(args.temporal_dt_rank) <= 0:
        raise ValueError("--temporal-dt-rank must be 'auto' or a positive integer")
    if args.temporal_layers <= 0:
        raise ValueError("--temporal-layers must be > 0")
    if not (0.0 <= args.temporal_dropout < 1.0):
        raise ValueError("--temporal-dropout must be in [0, 1)")
    if args.data_cache_max_gib <= 0:
        raise ValueError("--data-cache-max-gib must be > 0")

    base_cfg = RunConfig(
        data_path=args.data_path,
        resolution=args.resolution,
        mesh_size=args.mesh_size,
        width=args.width,
        processor_msg_steps=args.processor_msg_steps,
        val_year=args.val_year,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        ckpt_in=args.ckpt_in,
        stats_dir=args.stats_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        eval_batch_size=args.eval_batch_size,
        checkpoint_every=args.checkpoint_every,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        precision=args.precision,
        resume_step=args.resume_step,
        input_duration=args.input_duration,
        temporal_backbone=args.temporal_backbone,
        temporal_location=args.temporal_location,
        temporal_hidden_size=args.temporal_hidden_size,
        temporal_d_inner=args.temporal_d_inner,
        temporal_d_state=args.temporal_d_state,
        temporal_d_conv=args.temporal_d_conv,
        temporal_dt_rank=args.temporal_dt_rank,
        temporal_bias=args.temporal_bias,
        temporal_conv_bias=args.temporal_conv_bias,
        temporal_layers=args.temporal_layers,
        temporal_dropout=args.temporal_dropout,
        temporal_stateful=args.temporal_stateful,
        target_steps=args.target_steps,
        sequential_segment_steps=None,
        data_cache_mode=args.data_cache_mode,
        data_cache_max_gib=args.data_cache_max_gib,
        batch_builder=args.batch_builder,
        prefetch_workers=0,
        prefetch_depth=0,
        prefetch_device_depth=0,
        usage_every=1,
        eval_only=False,
    )
    return SegmentRunConfig(
        base_cfg=base_cfg,
        len_segment=args.len_segment,
        bptt_steps=args.bptt_steps,
        chunk_load_workers=args.chunk_load_workers,
    )


def run_gc_mamba_training(segment_cfg: SegmentRunConfig) -> None:
    import dataclasses
    import functools
    import json
    import time
    from typing import Iterable

    import haiku as hk
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import pandas as pd
    import xarray as xr

    from src.models.graphcast.training.core.batching import (
        BatchBuilder,
        NumpyBatchCache,
        build_batch_from_indices,
        build_batch_from_indices_vectorized,
        infer_time_step,
        input_steps_from_duration,
    )
    from src.models.graphcast.training.core.dataset import (
        _open_local_splits,
        _training_cache_decision,
        maybe_cache_training_data,
        prepare_dataset_for_task,
    )
    from src.models.graphcast.training.core.logging import (
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
    from src.models.graphcast.training.core.model import (
        build_predictor,
        gc,
        load_graphcast_checkpoint,
        load_stats,
        scalarize_loss,
        validate_stats_coverage,
    )
    from src.models.graphcast.training.core.segments import (
        SegmentBatchScheduler,
        _build_chunk_batches,
        _reset_temporal_state_lanes,
        _save_chunk_timing_logs,
        _stop_gradient_temporal_state,
        _write_segment_run_config,
        build_full_segments,
        run_eval_fresh_state,
        valid_contiguous_final_input_indices,
    )

    cfg = segment_cfg.base_cfg
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_in = load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
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
        raise ValueError("Segment training expects at least two input frames.")
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
        "Prepared segment windows: "
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
            train_numpy_cache = NumpyBatchCache(train_ds, task_cfg, label="segment-train")
            eval_numpy_cache = NumpyBatchCache(eval_ds, task_cfg, label="segment-eval")
            numpy_cache_active = True
        else:
            effective_train_batch_builder = "vectorized"
            effective_eval_batch_builder = "vectorized"
            print(
                "[numpy-cache] requested for segments but train split is not cached; "
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
            task_cfg: gc.TaskConfig,
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
            task_cfg: gc.TaskConfig,
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

    def forward_fn(inputs, targets, forcings, is_training):
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=True,
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
        return predictor.loss(inputs, targets, forcings)

    transformed = hk.transform_with_state(forward_fn)
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
    params, state = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, True)
    if cfg.resume_step is not None:
        params = ckpt_in.params
        print(f"Resuming from step {cfg.resume_step} (params loaded from {cfg.ckpt_in})")

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)
    _write_segment_run_config(
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
                (loss_and_diag, current_state) = transformed.apply(
                    p,
                    current_state,
                    keys[bptt_i],
                    chunk_inputs[bptt_i],
                    chunk_targets[bptt_i],
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
            description=ckpt_in.description,
            license_text=ckpt_in.license,
            filename="ckpt_best.npz",
        )
        with (out_dir / "best_checkpoint.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_eval_step": best_eval_step,
                    "best_eval_loss": best_eval_loss,
                    "best_checkpoint": "ckpt_best.npz",
                    "match_type": "exact",
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

            next_chunk = submit_next_chunk() if step + 1 < cfg.max_steps else None
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
                eval_metrics = run_eval_fresh_state(
                    transformed,
                    params,
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
                eval_details.append({"step": step, "total": eval_metrics["total"], **batch_builder_metadata})
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
                    description=ckpt_in.description,
                    license_text=ckpt_in.license,
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

    final_eval = run_eval_fresh_state(
        transformed,
        params,
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
    eval_details.append({"step": step, "final": True, "total": final_eval["total"], **batch_builder_metadata})

    save_checkpoint(
        out_dir,
        params=params,
        step=step,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        description=ckpt_in.description,
        license_text=ckpt_in.license,
    )
    save_all_logs()
    plot_loss_curves(out_dir, train_losses, eval_losses)
    print(f"Done. Final eval total {final_eval['total']:.6f}. Outputs in {out_dir}")


def _extract_model_and_argv(argv: list[str] | None = None) -> tuple[str, list[str]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=["gc_mamba", "residual_mamba"], default="gc_mamba")
    args, remaining = parser.parse_known_args(raw_argv)
    return args.model, remaining


def main(argv: list[str] | None = None) -> None:
    model_name, remaining_argv = _extract_model_and_argv(argv)
    if model_name == "residual_mamba":
        from src.models.mamba.residual_mamba.training.config import parse_args as parse_residual_args

        segment_cfg = parse_residual_args(remaining_argv)
        from src.models.mamba.residual_mamba.training.runner import run_training as run_residual_training

        run_residual_training(segment_cfg=segment_cfg)
        return

    if any(arg in {"-h", "--help"} for arg in remaining_argv):
        parse_gc_mamba_args(remaining_argv)
        return

    segment_cfg = parse_gc_mamba_args(remaining_argv)
    run_gc_mamba_training(segment_cfg)


def train() -> None:
    main()


if __name__ == "__main__":
    main()
