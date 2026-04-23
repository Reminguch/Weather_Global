#!/usr/bin/env python3
"""Train GraphCast on local ERA5 data.

The heavy lifting lives in scripts/training/graphcast_train so this file stays
as the command-line orchestration layer.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import time
from pathlib import Path
from typing import Iterable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xarray as xr

from graphcast_train.batching import (
    BatchBuilder,
    NumpyBatchCache,
    build_batch_from_indices,
    build_batch_from_indices_vectorized,
    build_sequential_segments,
    infer_time_step,
    input_steps_from_duration,
    valid_final_input_indices,
)
from graphcast_train.config import (
    DEFAULT_CKPT,
    DEFAULT_DATA_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_STATS_DIR,
    GRAPHCAST_VARS,
    RunConfig,
    parse_args,
)
from graphcast_train.dataset import (
    _ensure_datetime_coord,
    _open_local_splits,
    _training_cache_decision,
    maybe_cache_training_data,
    prepare_dataset_for_task,
)
from graphcast_train.eval import run_eval
from graphcast_train.logging import (
    _filter_pairs_upto_step,
    _load_dict_series_upto_step,
    _load_json_list,
    _load_step_value_pairs,
    _load_train_losses,
    _write_run_config,
    plot_loss_curves,
    sample_actual_usage,
    save_checkpoint,
    save_logs,
)
from graphcast_train.model import (
    build_predictor,
    gc,
    load_graphcast_checkpoint,
    load_stats,
    scalarize_loss,
    validate_stats_coverage,
)
from graphcast_train.prefetch import BatchPrefetcher, BatchRequest, PreparedBatch


def main() -> None:
    cfg = parse_args()
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
    if cfg.temporal_backbone != "none" and input_steps < 2:
        raise ValueError("Temporal module requires at least 2 input steps.")
    if cfg.temporal_backbone != "none":
        print(
            "Temporal module configured "
            f"(backbone={cfg.temporal_backbone}, location={cfg.temporal_location}, "
            f"stateful={cfg.temporal_stateful}, hidden={cfg.temporal_hidden_size}, "
            f"d_inner={cfg.temporal_d_inner}, d_state={cfg.temporal_d_state}, "
            f"d_conv={cfg.temporal_d_conv}, dt_rank={cfg.temporal_dt_rank}, "
            f"layers={cfg.temporal_layers}, dropout={cfg.temporal_dropout})"
        )
    target_steps = cfg.target_steps

    train_final_indices = valid_final_input_indices(train_ds.sizes["time"], input_steps, target_steps)
    eval_final_indices = valid_final_input_indices(eval_ds.sizes["time"], input_steps, target_steps)
    if len(train_final_indices) == 0:
        raise ValueError("No train samples after applying input/target window requirements.")
    if len(eval_final_indices) == 0:
        raise ValueError("No eval samples after applying input/target window requirements.")

    print(
        "Prepared dataset windows: "
        f"train_time={train_ds.sizes['time']}, eval_time={eval_ds.sizes['time']}, "
        f"train_samples={len(train_final_indices)}, eval_samples={len(eval_final_indices)}, "
        f"input_steps={input_steps}, target_steps={target_steps}"
    )

    should_cache_train, train_cache_estimate_gib = _training_cache_decision(train_ds, cfg, task_cfg)
    train_ds, eval_ds = maybe_cache_training_data(train_ds, eval_ds, cfg, task_cfg)

    numpy_cache_active = False
    train_numpy_cache: NumpyBatchCache | None = None
    eval_numpy_cache: NumpyBatchCache | None = None
    if cfg.batch_builder == "numpy":
        if should_cache_train:
            train_numpy_cache = NumpyBatchCache(train_ds, task_cfg, label="train")
            eval_numpy_cache = NumpyBatchCache(eval_ds, task_cfg, label="eval")
            numpy_cache_active = True
        else:
            print(
                "[numpy-cache] requested but train split is not cached; "
                "falling back to vectorized builder. Use --data-cache-mode=always to force it."
            )

    batch_builder_fn: BatchBuilder
    eval_batch_builder_fn: BatchBuilder
    if numpy_cache_active:
        assert train_numpy_cache is not None
        assert eval_numpy_cache is not None

        def train_numpy_builder(
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

        def eval_numpy_builder(
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

        batch_builder_fn = train_numpy_builder
        eval_batch_builder_fn = eval_numpy_builder
    elif cfg.batch_builder == "legacy":
        batch_builder_fn = build_batch_from_indices
        eval_batch_builder_fn = build_batch_from_indices
    else:
        batch_builder_fn = build_batch_from_indices_vectorized
        eval_batch_builder_fn = build_batch_from_indices_vectorized

    def build_train_batch(indices: Iterable[int]) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        return batch_builder_fn(
            train_ds,
            indices=indices,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt_train,
        )

    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
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

    def predict_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg,
            task_cfg,
            norm_stats,
            use_bf16=(cfg.precision == "bf16"),
            gradient_checkpointing=False,
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
        return predictor(inputs, targets_template=targets, forcings=forcings)

    transformed = hk.transform_with_state(forward_fn)
    transformed_predict = hk.transform_with_state(predict_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    sample_inputs, sample_targets, sample_forcings = build_train_batch([int(train_final_indices[0])])

    params, state = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, True)
    if cfg.resume_step is not None:
        params = ckpt_in.params
        print(f"Resuming from step {cfg.resume_step} (params loaded from {cfg.ckpt_in})")
    elif cfg.eval_only:
        params = ckpt_in.params
        print(f"Eval-only: loaded params from {cfg.ckpt_in}")

    if cfg.eval_only:
        print("Eval-only mode: running eval on loaded checkpoint, no training.")
        eval_metrics = run_eval(
            transformed,
            params,
            state,
            rng,
            eval_ds,
            eval_final_indices,
            eval_batch_size=cfg.eval_batch_size,
            input_steps=input_steps,
            target_steps=target_steps,
            task_cfg=task_cfg,
            dt=dt_train,
            progress_label="eval-only",
            batch_builder=eval_batch_builder_fn,
            transformed_predict=transformed_predict,
        )
        print(f"[eval-only] total {eval_metrics['total']:.6f}")
        return

    opt = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    opt_state = opt.init(params)

    _write_run_config(
        out_dir,
        cfg,
        model_cfg,
        task_cfg,
        numpy_cache_active=numpy_cache_active,
        train_cache_estimate_gib=train_cache_estimate_gib,
    )

    def _map_temporal_state_leaves(state: hk.State, fn) -> hk.State:
        mutable_state = hk.data_structures.to_mutable_dict(state)
        for module_name, module_state in mutable_state.items():
            del module_name
            for state_name, leaf in module_state.items():
                if not isinstance(leaf, jax.Array):
                    continue
                if state_name.endswith("_ssm_state") or state_name.endswith("_conv_cache"):
                    module_state[state_name] = fn(leaf)
        return hk.data_structures.to_immutable_dict(mutable_state)

    def _reset_ssm_state(state: hk.State) -> hk.State:
        """Zero out temporal Mamba state so each sample starts fresh."""
        return _map_temporal_state_leaves(state, jnp.zeros_like)

    def _stop_grad_state(state: hk.State) -> hk.State:
        """Detach temporal Mamba state from the computation graph.

        The state values are preserved for the next sample, but gradients
        are cut so backprop only flows through the current sample's
        target_steps.
        """
        return _map_temporal_state_leaves(state, jax.lax.stop_gradient)

    use_sequential = cfg.sequential_segment_steps is not None

    @functools.partial(jax.jit, static_argnames=("reset_state",))
    def train_step(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        rng_key: jax.Array,
        inputs: xr.Dataset,
        targets: xr.Dataset,
        forcings: xr.Dataset,
        reset_state: bool = True,
    ):
        # reset_state=True: zero out SSM state (random sampling or segment boundary)
        # reset_state=False: carry state with stop_gradient (truncated BPTT within segment)
        if reset_state:
            state = _reset_ssm_state(state)
        else:
            state = _stop_grad_state(state)

        def loss_fn(p, s, key):
            (loss_and_diag, new_state) = transformed.apply(p, s, key, inputs, targets, forcings, True)
            loss = scalarize_loss(loss_and_diag[0])
            return loss, new_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, rng_key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, new_opt_state, loss

    step = cfg.resume_step if cfg.resume_step is not None else 0

    train_losses: list[tuple[int, float]] = []
    eval_losses: list[tuple[int, float]] = []
    eval_details: list[dict[str, Any]] = []
    step_times: list[tuple[int, float]] = []
    timing_details: list[dict[str, Any]] = []
    mem_usage: list[tuple[int, float]] = []
    actual_usage: list[dict[str, Any]] = []
    epoch_summaries: list[dict[str, Any]] = []

    if cfg.resume_step is not None:
        train_losses = _filter_pairs_upto_step(_load_train_losses(out_dir / "train_loss.json"), cfg.resume_step)
        eval_losses = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "eval_loss.json"), cfg.resume_step)
        step_times = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "step_times.json"), cfg.resume_step)
        timing_details = _load_dict_series_upto_step(out_dir / "timing_details.json", cfg.resume_step)
        mem_usage = _filter_pairs_upto_step(_load_step_value_pairs(out_dir / "memory_gib.json"), cfg.resume_step)
        eval_details = _load_dict_series_upto_step(out_dir / "eval_details.json", cfg.resume_step)
        actual_usage = _load_dict_series_upto_step(out_dir / "actual_usage.json", cfg.resume_step)
        epoch_summaries = _load_json_list(out_dir / "epoch_summary.json")
        print(
            "Loaded existing logs for resume: "
            f"train={len(train_losses)}, eval={len(eval_losses)}, "
            f"step_times={len(step_times)}, timings={len(timing_details)}, "
            f"mem={len(mem_usage)}, actual={len(actual_usage)}"
        )

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

    np_rng = np.random.default_rng(cfg.seed)

    # Build sampling structure
    if use_sequential:
        segments = build_sequential_segments(train_final_indices, cfg.sequential_segment_steps)
        np_rng.shuffle(segments)
        seg_idx = 0  # current segment
        seg_cursor = 0  # position within current segment
        print(
            f"Sequential sampling: {len(segments)} segments of ~{cfg.sequential_segment_steps} steps "
            f"({cfg.sequential_segment_steps * 6 / 24:.0f} days each)"
        )
    else:
        current_indices = train_final_indices.copy()
        np_rng.shuffle(current_indices)
    cursor = 0
    pass_idx = 1
    pass_start_step = step + 1
    pass_loss_accum: list[float] = []

    def next_random_request() -> BatchRequest:
        nonlocal cursor, current_indices
        new_epoch = cursor >= len(current_indices)
        if new_epoch:
            current_indices = train_final_indices.copy()
            np_rng.shuffle(current_indices)
            cursor = 0
        batch_idx = current_indices[cursor : cursor + cfg.batch_size]
        cursor += cfg.batch_size
        return BatchRequest(indices=np.asarray(batch_idx, dtype=np.int64), reset_state=True, new_epoch=new_epoch)

    prefetcher: BatchPrefetcher | None = None
    if not use_sequential and cfg.prefetch_workers > 0 and cfg.prefetch_depth > 0:
        prefetcher = BatchPrefetcher(
            request_fn=next_random_request,
            build_fn=build_train_batch,
            max_workers=cfg.prefetch_workers,
            depth=cfg.prefetch_depth,
            device_depth=cfg.prefetch_device_depth,
        )
        prefetcher.start()
        print(
            "[prefetch] enabled "
            f"workers={cfg.prefetch_workers}, depth={cfg.prefetch_depth}, "
            f"device_depth={cfg.prefetch_device_depth}"
        )
    elif not use_sequential:
        print("[prefetch] disabled for random sampling; batches will build synchronously.")

    while step < cfg.max_steps:
        loop_t0 = time.time()
        prepared_batch: PreparedBatch | None = None
        if use_sequential:
            # Sequential segment sampling keeps state-carry semantics explicit.
            _new_epoch = False
            if seg_idx >= len(segments):
                # All segments exhausted -> new epoch
                epoch_summaries.append(
                    {
                        "pass": pass_idx,
                        "steps": step - pass_start_step + 1,
                        "train_loss_mean": float(np.mean(pass_loss_accum)) if pass_loss_accum else float("nan"),
                        "time_per_step_mean": float(
                            np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                        ),
                        "mem_gib_max": float(
                            np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])
                        ),
                    }
                )
                pass_idx += 1
                pass_start_step = step + 1
                pass_loss_accum = []
                segments = build_sequential_segments(train_final_indices, cfg.sequential_segment_steps)
                np_rng.shuffle(segments)
                seg_idx = 0
                seg_cursor = 0
            elif seg_cursor >= len(segments[seg_idx]):
                # Current segment exhausted -> move to next segment (reset state)
                seg_idx += 1
                seg_cursor = 0
                if seg_idx >= len(segments):
                    continue  # will trigger new epoch on next iteration

            # At start of segment, reset state; otherwise carry with stop_gradient
            reset_state = (seg_cursor == 0)
            seg = segments[seg_idx]
            batch_idx = seg[seg_cursor : seg_cursor + cfg.batch_size]
            seg_cursor += cfg.batch_size
            t_build = time.time()
            batch_inputs, batch_targets, batch_forcings = build_train_batch(batch_idx)
            data_wait = 0.0
            host_build_time = time.time() - t_build
            device_put_time = 0.0
            device_staged = False
        else:
            if prefetcher is not None:
                prepared_batch, data_wait = prefetcher.get()
                batch_inputs = prepared_batch.inputs
                batch_targets = prepared_batch.targets
                batch_forcings = prepared_batch.forcings
                reset_state = prepared_batch.request.reset_state
                _new_epoch = prepared_batch.request.new_epoch
                host_build_time = prepared_batch.host_build_time
                device_put_time = prepared_batch.device_put_time
                device_staged = prepared_batch.device_staged
            else:
                request = next_random_request()
                _new_epoch = request.new_epoch
                reset_state = request.reset_state
                t_build = time.time()
                batch_inputs, batch_targets, batch_forcings = build_train_batch(request.indices)
                data_wait = 0.0
                host_build_time = time.time() - t_build
                device_put_time = 0.0
                device_staged = False

        rng, step_key = jax.random.split(rng)
        t_compute = time.time()
        params, state, opt_state, loss = train_step(
            params,
            state,
            opt_state,
            step_key,
            batch_inputs,
            batch_targets,
            batch_forcings,
            reset_state=reset_state,
        )
        loss.block_until_ready()
        compute_sync_time = time.time() - t_compute
        step_wall_time = time.time() - loop_t0
        if prefetcher is not None and prepared_batch is not None:
            prefetcher.release_device_slot(prepared_batch)

        step += 1
        loss_f = float(loss)
        train_losses.append((step, loss_f))
        pass_loss_accum.append(loss_f)
        step_times.append((step, step_wall_time))
        timing_details.append(
            {
                "step": step,
                "data_wait": data_wait,
                "host_build": host_build_time,
                "device_put": device_put_time,
                "device_staged": device_staged,
                "compute_sync": compute_sync_time,
                "step_wall": step_wall_time,
            }
        )

        # Random mode: log epoch summary after the last step of each epoch completes.
        if _new_epoch:
            epoch_summaries.append(
                {
                    "pass": pass_idx,
                    "steps": step - pass_start_step + 1,
                    "train_loss_mean": float(np.mean(pass_loss_accum)) if pass_loss_accum else float("nan"),
                    "time_per_step_mean": float(
                        np.mean([t for s, t in step_times if s >= pass_start_step] or [float("nan")])
                    ),
                    "mem_gib_max": float(
                        np.max([m for s, m in mem_usage if s >= pass_start_step] or [float("nan")])
                    ),
                }
            )
            pass_idx += 1
            pass_start_step = step + 1
            pass_loss_accum = []

        if cfg.usage_every > 0 and (step == 1 or step % cfg.usage_every == 0):
            usage = sample_actual_usage(step=step)
            actual_usage.append(usage)
            if usage.get("gpu_mem_gib") is not None:
                mem_usage.append((step, float(usage["gpu_mem_gib"])))

        if step % 10 == 0:
            print(
                f"step {step}/{cfg.max_steps} loss {loss_f:.6f} "
                f"data_wait={data_wait:.3f}s host_build={host_build_time:.3f}s "
                f"device_put={device_put_time:.3f}s compute_sync={compute_sync_time:.4f}s "
                f"step_wall={step_wall_time:.3f}s staged={int(device_staged)}"
            )

        if step % cfg.eval_every == 0:
            eval_metrics = run_eval(
                transformed,
                params,
                state,
                rng,
                eval_ds,
                eval_final_indices,
                eval_batch_size=cfg.eval_batch_size,
                input_steps=input_steps,
                target_steps=target_steps,
                task_cfg=task_cfg,
                dt=dt_train,
                progress_label=f"eval@step{step}",
                batch_builder=eval_batch_builder_fn,
                transformed_predict=transformed_predict,
            )
            eval_losses.append((step, eval_metrics["total"]))
            maybe_save_best_checkpoint(step, float(eval_metrics["total"]))
            eval_details.append(
                {
                    "step": step,
                    "total": eval_metrics["total"],
                }
            )
            print(f"[eval] step {step} total {eval_metrics['total']:.6f}")
            plot_loss_curves(out_dir, train_losses, eval_losses)
            save_logs(
                out_dir,
                train_losses,
                eval_losses,
                eval_details,
                step_times,
                timing_details,
                mem_usage,
                actual_usage,
                epoch_summaries,
            )

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

    if prefetcher is not None:
        prefetcher.shutdown()

    if pass_loss_accum:
        epoch_summaries.append(
            {
                "pass": pass_idx,
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

    final_eval = run_eval(
        transformed,
        params,
        state,
        rng,
        eval_ds,
        eval_final_indices,
        eval_batch_size=cfg.eval_batch_size,
        input_steps=input_steps,
        target_steps=target_steps,
        task_cfg=task_cfg,
        dt=dt_train,
        progress_label="eval@final",
        batch_builder=eval_batch_builder_fn,
        transformed_predict=transformed_predict,
    )
    eval_losses.append((step, final_eval["total"]))
    maybe_save_best_checkpoint(step, float(final_eval["total"]))
    eval_details.append(
        {
            "step": step,
            "final": True,
            "total": final_eval["total"],
        }
    )

    save_checkpoint(
        out_dir,
        params=params,
        step=step,
        model_cfg=model_cfg,
        task_cfg=task_cfg,
        description=ckpt_in.description,
        license_text=ckpt_in.license,
    )
    save_logs(
        out_dir,
        train_losses,
        eval_losses,
        eval_details,
        step_times,
        timing_details,
        mem_usage,
        actual_usage,
        epoch_summaries,
    )
    plot_loss_curves(out_dir, train_losses, eval_losses)

    rss_vals = [float(x["proc_rss_gib"]) for x in actual_usage if x.get("proc_rss_gib") is not None]
    gpu_vals = [float(x["gpu_mem_gib"]) for x in actual_usage if x.get("gpu_mem_gib") is not None]
    print(
        "Actual usage summary: "
        f"rss_peak={float(np.max(rss_vals)) if rss_vals else float('nan'):.3f} GiB, "
        f"rss_avg={float(np.mean(rss_vals)) if rss_vals else float('nan'):.3f} GiB, "
        f"gpu_peak={float(np.max(gpu_vals)) if gpu_vals else float('nan'):.3f} GiB, "
        f"gpu_avg={float(np.mean(gpu_vals)) if gpu_vals else float('nan'):.3f} GiB"
    )
    print(f"Done. Final eval total {final_eval['total']:.6f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
