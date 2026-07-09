"""gcmamba_v1: Ilya's GC-Mamba architecture on top of DeepMind GraphCast small.

Architecture (matches Ilya gc_mamba):
  - Single gc_mamba predictor (Mamba interleaved inside the mesh processor,
    location=mesh_processor_interleaved).
  - Mamba output zero-initialized -> at step 0 prediction == vanilla DeepMind GC.

Param treatment (fixing the inefficiency in Ilya's set_to_zero approach):
  - At init, params are split into two trees by Haiku module name:
      mamba_params : modules whose path contains 'temporal' or 'mamba' (trainable)
      gc_params    : everything else (frozen)
  - GC leaves are overlaid from DeepMind GraphCast small ckpt.
  - jax.value_and_grad(loss_fn)(mamba_params) only -> no GC gradient tree.

Training modes (controlled by --target-steps K, mirrors Ilya rolling_ar):
  - K = 1 (default): truth-fed BPTT over bptt_steps. Loss = mean of all
    per-step losses. Identical to Ilya's non-rolling branch.
  - K > 1: chunk-local autoregressive (BPTT with AR), matching Ilya rolling_ar
    and v22 production protocol:
      truth_prefix = bptt_steps - K
      first (truth_prefix) steps fed with truth from data;
      last K steps fed with self predictions via _advance_autoregressive_inputs;
      loss aggregated ONLY over the AR-tail steps.
    Default: full BPTT through AR chain (no stop_gradient on predictions),
    matching v22. Optional --feedback-stop-gradient enables v22closed (_sg).

Memory:
  - Each per-step forward wrapped in jax.checkpoint so BPTT activation
    memory stays linear-but-flat in bptt_steps (matches v22closed approach).

Ckpt:
  - Saves full params (merge mamba + gc) for eval pipeline compatibility.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from src.models.graphcast.training.core.model import (  # noqa: E402
    build_predictor,
    gc,
    load_graphcast_checkpoint,
    load_stats,
    scalarize_loss,
    validate_stats_coverage,
)
from src.models.graphcast_remat.build import build_predictor_remat  # noqa: E402
import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.mamba.training.param_utils import (  # noqa: E402
    is_temporal_param,
    overlay_matching_params,
)
from src.models.graphcast.training.core.batching import NumpyBatchCache  # noqa: E402
from src.data.prepared_array import PreparedArrayStore  # noqa: E402


# ---------- Param partition helpers ----------

def split_mamba_gc(params: hk.Params) -> tuple[hk.Params, hk.Params]:
    """Split flat-2-level Haiku params dict into (mamba_params, gc_params)."""
    mamba: dict = {}
    gc_: dict = {}
    for module_name, leaves in params.items():
        for leaf_name, value in leaves.items():
            tgt = mamba if is_temporal_param(module_name, leaf_name) else gc_
            tgt.setdefault(module_name, {})[leaf_name] = value
    return mamba, gc_


def merge_params(mamba: hk.Params, gc_: hk.Params) -> hk.Params:
    out: dict = {k: dict(v) for k, v in gc_.items()}
    for module_name, leaves in mamba.items():
        out.setdefault(module_name, {}).update(leaves)
    return out


def _stop_grad_state(state: hk.State) -> hk.State:
    return jax.tree_util.tree_map(jax.lax.stop_gradient, state)


# ---------- AR rollout helpers (mirror Ilya's segments.py) ----------

def _chunk_ar_truth_prefix(target_steps: int, bptt_steps: int) -> int:
    """Truth-fed prefix length for chunk-local AR training (Ilya's rolling_ar).

    target_steps == 1 means truth-fed everywhere (no AR feedback).
    target_steps > 1 means the LAST `target_steps` BPTT steps are AR-fed.
    """
    if target_steps <= 1:
        return bptt_steps
    if target_steps >= bptt_steps:
        raise ValueError(
            "Chunk-local AR requires --target-steps < --bptt-steps, "
            f"got target_steps={target_steps}, bptt_steps={bptt_steps}.")
    return bptt_steps - target_steps


def _constant_input_keys(inputs: xr.Dataset, predictions: xr.Dataset,
                         forcings: xr.Dataset) -> list[str]:
    """Variables in inputs that should NOT be replaced by predictions or forcings."""
    moving = set(predictions.data_vars) | set(forcings.data_vars)
    return [k for k in inputs.data_vars if k not in moving]


def _advance_autoregressive_inputs(inputs: xr.Dataset,
                                   predictions: xr.Dataset,
                                   forcings: xr.Dataset) -> xr.Dataset:
    """Build next-step inputs from current inputs + this step's prediction.

    Replaces the oldest time slice of rolling inputs with the new
    (prediction + forcings) frame; keeps inputs.time coords unchanged so
    the model still sees the expected `input_steps` window.
    """
    constant_keys = _constant_input_keys(inputs, predictions, forcings)
    constant_inputs = inputs[constant_keys]
    rolling_inputs = inputs.drop_vars(constant_keys)
    num_inputs = rolling_inputs.sizes["time"]
    next_frame = xr.merge([predictions, forcings])
    predicted_or_forced = next_frame[list(rolling_inputs.data_vars)]
    updated = (
        xr.concat([rolling_inputs, predicted_or_forced], dim="time")
        .tail(time=num_inputs)
        .assign_coords(time=rolling_inputs.coords["time"])
    )
    return xr.merge([constant_inputs, updated])


# ---------- Argparse ----------

def parse_args():
    p = argparse.ArgumentParser()
    # Data + ckpt
    p.add_argument("--data-path",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr",
                   help="(legacy) zarr path. Ignored if --prepared-root is set.")
    p.add_argument("--prepared-root",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/prepared_stream_2015_2022_v2/res1",
                   help="Ilya-style prepared array store (one .npy per var, memmap-loaded). "
                        "When set, supersedes --data-path. Avoids the 200 GiB zarr materialization "
                        "that OOMed run01 v1.")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
    p.add_argument("--ckpt-in",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
                           "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
                           "mesh 2to5 - precipitation input and output.npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", required=True)
    # Model (defaults match DeepMind small)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--processor-msg-steps", type=int, default=16)
    p.add_argument("--input-duration", default="12h")
    # Mamba
    p.add_argument("--temporal-location",
                   choices=["mesh_post_encoder", "mesh_processor_interleaved"],
                   default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=64)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--temporal-bias", action="store_true", default=False)
    p.add_argument("--no-temporal-conv-bias",
                   dest="temporal_conv_bias", action="store_false", default=True)
    p.add_argument("--temporal-stateful", action="store_true", default=True)
    p.add_argument("--no-temporal-stateful",
                   dest="temporal_stateful", action="store_false")
    # BPTT
    p.add_argument("--bptt-steps", type=int, default=8,
                   help="Total BPTT rollout depth per training iter (Mamba state carries through).")
    p.add_argument("--target-steps", type=int, default=1,
                   help="K = number of AR-tail steps with loss. K=1: truth-fed everywhere "
                        "(Ilya non-rolling). K>1: last K steps closed-loop (Ilya rolling_ar / v22).")
    p.add_argument("--feedback-stop-gradient", action="store_true", default=False,
                   help="If set, stop_gradient on predictions before AR feedback (v22closed _sg).")
    # Memory mode (CANONICAL entry point — aligned with Ilya's segments_training).
    # standard:     no remat (only Mamba param partition, which gcmamba_v1 always does).
    # conservative: same as standard for gcmamba_v1 (param partition is already on).
    # optimal:      conservative + processor-step remat (SAFE Mode-2) + mesh2grid remat.
    p.add_argument("--memory-mode",
                   choices=["standard", "conservative", "optimal"],
                   default="standard",
                   help="Training memory behavior. standard = no remat; "
                        "conservative = Mamba-only param partition (already always on "
                        "in gcmamba_v1); optimal = also rematerializes processor steps + "
                        "mesh2grid (SAFE Mode-2: Mamba runs OUTSIDE the remat boundary).")
    # Advanced/debug overrides — prefer --memory-mode in production.
    # NOTE: Mode-2 SAFE: hk.remat wraps ONLY the processor msg-passing step;
    # Mamba runs OUTSIDE the remat boundary. The reversed configuration
    # (Mamba INSIDE remat) failed gradient verification and is intentionally
    # NOT reachable from this CLI.
    p.add_argument("--remat-processor-steps", action="store_true", default=False,
                   help="Advanced/debug. hk.remat each frozen processor msg-passing step "
                        "(Mode-2). Prefer --memory-mode optimal.")
    p.add_argument("--remat-mesh2grid", action="store_true", default=False,
                   help="Advanced/debug. hk.remat the mesh2grid GNN call. "
                        "Prefer --memory-mode optimal.")
    p.add_argument("--remat-grid2mesh", action="store_true", default=False,
                   help="Advanced/debug. hk.remat the grid2mesh GNN call. "
                        "Not enabled by any --memory-mode preset.")
    p.add_argument("--len-segment", type=int, default=32,
                   help="Segment length (in 6h steps). Mamba state resets at segment start.")
    # Optimization
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    # Resume
    p.add_argument("--resume-from", default=None,
                   help="Path to pkl with {'params': merged_full_params, ...} to resume Mamba from.")
    p.add_argument("--start-step", type=int, default=0)
    # Splits
    p.add_argument("--train-start-year", type=int, default=2020)
    p.add_argument("--train-end-year", type=int, default=2021)
    p.add_argument("--val-year", type=int, default=2022)
    return p.parse_args()


# ---------- Main ----------

def main():
    cfg = parse_args()
    # --memory-mode -> remat flags (OR-merged with any explicit --remat-* flags).
    # "optimal" enables SAFE Mode-2 processor remat + mesh2grid remat.
    # grid2mesh remat is NOT part of any preset (per memory_optim.md priority).
    if cfg.memory_mode == "optimal":
        cfg.remat_processor_steps = cfg.remat_processor_steps or True
        cfg.remat_mesh2grid = cfg.remat_mesh2grid or True
    print(f"[gcmamba_v1] memory_mode={cfg.memory_mode}  "
          f"remat_processor_steps={cfg.remat_processor_steps}  "
          f"remat_mesh2grid={cfg.remat_mesh2grid}  "
          f"remat_grid2mesh={cfg.remat_grid2mesh}")
    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gcmamba_v1] out={out_dir}")

    # ----- 1) Load DeepMind ckpt for model_cfg + params -----
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
    )
    print(f"[gcmamba_v1] model_cfg: res={model_cfg.resolution}, mesh={model_cfg.mesh_size}, "
          f"width={model_cfg.latent_size}, mp={model_cfg.gnn_msg_steps}")

    norm_stats = load_stats(Path(cfg.stats_dir))
    validate_stats_coverage(task_cfg, norm_stats)

    # ----- 2) Data: PreparedArrayStore (Ilya-style memmap loader) -----
    # Replaces zarr + NumpyBatchCache, which materialized ~200 GiB into RAM
    # and OOMed run01. PreparedArrayStore uses np.load(mmap_mode="r") per
    # variable: only metadata in resident RAM, OS pages data on demand,
    # OS page cache amortizes repeated reads. Batch building is a numpy
    # gather over the memmaps -> sub-ms instead of ~95s per zarr fetch.
    store = PreparedArrayStore(cfg.prepared_root, label="gcmamba-v1-source")
    store.validate(resolution=cfg.resolution, task_cfg=task_cfg)
    all_times = pd.DatetimeIndex(pd.to_datetime(store.time.values))
    years = all_times.year.to_numpy()
    train_mask = (years != cfg.val_year)
    if cfg.train_start_year is not None:
        train_mask &= (years >= cfg.train_start_year) & (years <= cfg.train_end_year)
    val_mask = (years == cfg.val_year)
    if not train_mask.any():
        raise ValueError("Empty train split after applying train year bounds.")
    if not val_mask.any():
        raise ValueError(f"Empty val split: val_year={cfg.val_year} not present.")
    train_cache = store.split_by_time_indices(np.where(train_mask)[0], label="train")
    eval_cache  = store.split_by_time_indices(np.where(val_mask)[0],   label="eval")
    print(f"[gcmamba_v1] PreparedArrayStore: train_time={train_cache.sizes['time']}, "
          f"eval_time={eval_cache.sizes['time']}  (memmap, ~0 resident RAM)")
    dt = pd.Timedelta(np.diff(train_cache.time.values)[0])
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    if input_steps < 2:
        raise ValueError("Need at least 2 input frames (--input-duration >= 12h).")

    bptt = cfg.bptt_steps
    target_steps = cfg.target_steps
    rolling_ar = target_steps > 1
    truth_prefix_steps = _chunk_ar_truth_prefix(target_steps, bptt)
    seg_len = cfg.len_segment
    if seg_len % bptt != 0:
        raise ValueError("--len-segment must be divisible by --bptt-steps")
    chunks_per_segment = seg_len // bptt
    print(f"[gcmamba_v1] bptt={bptt} target_steps={target_steps} "
          f"rolling_ar={rolling_ar} truth_prefix={truth_prefix_steps} "
          f"feedback_stop_grad={cfg.feedback_stop_gradient}")
    print(f"[train_step mode] bptt={bptt}, target_steps={target_steps}, "
          f"truth steps=[0,{truth_prefix_steps}), "
          f"AR/loss steps=[{truth_prefix_steps},{bptt}), "
          f"feedback_sg={cfg.feedback_stop_gradient}")

    # Valid window indices = anchors where input_window + K target steps all exist
    train_indices = base_train.valid_final_input_indices(
        train_cache.sizes["time"], input_steps, target_steps=bptt)
    val_indices = base_train.valid_final_input_indices(
        eval_cache.sizes["time"], input_steps, target_steps=bptt)

    # Build segments: each segment is `seg_len` consecutive valid anchors.
    def _segments(indices, length):
        segs = []
        if len(indices) < length:
            return segs
        i = 0
        while i + length <= len(indices):
            seg = indices[i:i + length]
            # require strict-contiguous spacing == dt (no gaps)
            if np.all(np.diff(seg) == 1):
                segs.append(seg)
            i += length
        return segs

    train_segments = _segments(train_indices, seg_len)
    val_segments = _segments(val_indices, seg_len)
    if not train_segments:
        raise ValueError("No contiguous train segments built.")
    print(f"[gcmamba_v1] train_segments={len(train_segments)}  "
          f"val_segments={len(val_segments)}  bptt={bptt}  seg_len={seg_len}  "
          f"chunks/segment={chunks_per_segment}")

    # ----- 3) Build gc_mamba transform -----
    use_bf16 = True
    any_remat = (cfg.remat_processor_steps or cfg.remat_mesh2grid
                 or cfg.remat_grid2mesh)
    if any_remat:
        print(f"[gcmamba_v1] using GraphCastRemat: "
              f"processor={cfg.remat_processor_steps} "
              f"mesh2grid={cfg.remat_mesh2grid} "
              f"grid2mesh={cfg.remat_grid2mesh}")

    def forward_fn(inputs, targets, forcings, is_training):
        if any_remat:
            predictor = build_predictor_remat(
                model_cfg, task_cfg, norm_stats,
                use_bf16=use_bf16,
                gradient_checkpointing=False,
                temporal_backbone="mamba",
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
                zero_init_temporal_out=True,
                remat_processor_steps=cfg.remat_processor_steps,
                remat_mesh2grid=cfg.remat_mesh2grid,
                remat_grid2mesh=cfg.remat_grid2mesh,
                # Production never opts into Mode-1 (UNSAFE); kept False here.
                remat_temporal_inside_processor=False,
            )
        else:
            predictor = build_predictor(
                model_cfg, task_cfg, norm_stats,
                use_bf16=use_bf16,
                gradient_checkpointing=False,  # we wrap our own remat per AR step
                temporal_backbone="mamba",
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
                zero_init_temporal_out=True,
            )
        # build_predictor wraps the inner stack in autoregressive.Predictor,
        # but that wrapper doesn't implement .loss_and_predictions (base
        # predictor_base raises NotImplementedError).  Since we manage AR
        # feedback OURSELVES (one_step_ckpt per i), the autoregressive layer
        # is dead weight here — bypass it to reach the InputsAndResiduals
        # predictor that DOES implement loss_and_predictions.
        inner = predictor._predictor
        return inner.loss_and_predictions(inputs, targets, forcings)

    transformed = hk.transform_with_state(forward_fn)

    # ----- 4) Init params -----
    rng = jax.random.PRNGKey(cfg.seed)
    sample_anchor = int(train_segments[0][0])
    sample_inputs, sample_targets, sample_forcings = train_cache.build_batch_from_indices(
        indices=[sample_anchor],
        input_steps=input_steps,
        target_steps=1,
        task_cfg=task_cfg,
        dt=dt,
    )
    rng, k_init = jax.random.split(rng)
    params, state = transformed.init(
        k_init, sample_inputs, sample_targets, sample_forcings, True)

    # Overlay DeepMind weights into matching GC leaves (Mamba leaves untouched).
    params, overlay_stats = overlay_matching_params(params, ckpt_in.params, strict=False)
    print(f"[gcmamba_v1] overlay DeepMind: copied={overlay_stats.copied}, "
          f"fresh_init={overlay_stats.initialized} (expected: only Mamba/temporal leaves)")

    # SCOPE ASSERTION: verify every leaf NOT overlaid from DeepMind ckpt is in
    # a temporal/mamba module. If anything else is fresh-init, the Haiku scope
    # diverged from the base GraphCast class -> overlay incomplete -> step-0
    # forecast will NOT equal vanilla DeepMind GC.
    fresh_non_mamba = []
    for module_name, leaves in params.items():
        src_module = ckpt_in.params.get(module_name, {})
        for leaf_name in leaves:
            if leaf_name not in src_module:
                if not is_temporal_param(module_name, leaf_name):
                    fresh_non_mamba.append(f"{module_name}/{leaf_name}")
    if fresh_non_mamba:
        raise AssertionError(
            f"[gcmamba_v1] {len(fresh_non_mamba)} non-temporal leaves are "
            f"fresh-init (NOT overlaid from DeepMind ckpt). This means the "
            f"Haiku scope diverged from base GraphCast — overlay is incomplete "
            f"and step-0 forecast will not match vanilla GC. "
            f"First 10: {fresh_non_mamba[:10]}")
    print(f"[gcmamba_v1] scope check OK: all {overlay_stats.initialized} "
          f"fresh-init leaves live in temporal/mamba modules")

    # Partition into trainable Mamba and frozen GC.
    mamba_params, gc_params = split_mamba_gc(params)
    n_mamba = sum(int(np.prod(v.shape)) for ld in mamba_params.values() for v in ld.values())
    n_gc = sum(int(np.prod(v.shape)) for ld in gc_params.values() for v in ld.values())
    n_modules_mamba = len(mamba_params)
    n_modules_gc = len(gc_params)
    print(f"[gcmamba_v1] mamba: {n_modules_mamba} modules, {n_mamba:,} params (trainable)")
    print(f"[gcmamba_v1] gc:    {n_modules_gc} modules, {n_gc:,} params (frozen)")

    # Optional: resume Mamba from a prior pkl.
    if cfg.resume_from is not None:
        with Path(cfg.resume_from).open("rb") as f:
            ck = pickle.load(f)
        full_resume = ck["params"] if "params" in ck else ck["mamba_params"] if "mamba_params" in ck else None
        if full_resume is None:
            raise ValueError(f"Resume pkl missing 'params' or 'mamba_params': {cfg.resume_from}")
        # If it's the full tree, just re-split; if it's only mamba, keep gc from overlay.
        if any(is_temporal_param(m, "") for m in full_resume) and any(
                not is_temporal_param(m, "") for m in full_resume):
            mamba_params, gc_params_loaded = split_mamba_gc(full_resume)
            # Prefer loaded GC if it matches shape; else stick with DeepMind overlay.
            gc_params = gc_params_loaded
        else:
            mamba_params = full_resume
        print(f"[gcmamba_v1] resumed Mamba from {cfg.resume_from} at step {cfg.start_step}")

    # ----- 5) Optimizer (only on Mamba) -----
    if cfg.warmup_steps > 0:
        lr_schedule = optax.warmup_constant_schedule(
            init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup_steps)
    else:
        lr_schedule = cfg.lr
    opt_chain = []
    if cfg.grad_clip > 0:
        opt_chain.append(optax.clip_by_global_norm(cfg.grad_clip))
    opt_chain.append(optax.adamw(lr_schedule, weight_decay=cfg.weight_decay))
    opt = optax.chain(*opt_chain) if len(opt_chain) > 1 else opt_chain[0]
    opt_state = opt.init(mamba_params)
    print(f"[gcmamba_v1] opt initialized on {n_mamba:,} Mamba params only")

    # Save run config
    run_config = {
        "config": {k: getattr(cfg, k) for k in vars(cfg)},
        "model_cfg": dataclasses.asdict(model_cfg),
        "input_steps": int(input_steps),
        "n_train_segments": len(train_segments),
        "n_val_segments": len(val_segments),
        "n_mamba_params": int(n_mamba),
        "n_gc_params_frozen": int(n_gc),
        "deepmind_overlay_copied": int(overlay_stats.copied),
        "deepmind_overlay_fresh": int(overlay_stats.initialized),
        "bptt_steps": int(bptt),
        "target_steps": int(target_steps),
        "rolling_ar": bool(rolling_ar),
        "truth_prefix_steps": int(truth_prefix_steps),
        "feedback_stop_gradient": bool(cfg.feedback_stop_gradient),
        "memory_mode": str(cfg.memory_mode),
        "remat_processor_steps": bool(cfg.remat_processor_steps),
        "remat_mesh2grid": bool(cfg.remat_mesh2grid),
        "remat_grid2mesh": bool(cfg.remat_grid2mesh),
        "architecture": ("gcmamba_v1_FROZEN_DM_GC_PLUS_MAMBA"
                         + ("_AR_SG" if cfg.feedback_stop_gradient else
                            "_AR_BPTT" if rolling_ar else "_TRUTH_FED")),
    }
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # ----- 6) Train step -----
    feedback_stop_gradient = cfg.feedback_stop_gradient

    def _one_step_apply(mamba_p, gc_p, s, key, inp, tgt, frc):
        """Returns (loss, predictions, new_state)."""
        full = merge_params(mamba_p, gc_p)
        ((loss_da, _), preds), new_s = transformed.apply(
            full, s, key, inp, tgt, frc, True)
        return scalarize_loss(loss_da), preds, new_s

    _one_step_ckpt = jax.checkpoint(_one_step_apply)

    @jax.jit
    def train_step(mamba_p, gc_p, state_in, opt_st, rng_key,
                   chunk_inputs, chunk_targets, chunk_forcings):
        """BPTT over bptt steps. Truth-fed in [0, truth_prefix); AR-fed in
        [truth_prefix, bptt). Loss aggregated only on AR-tail (or all steps
        if rolling_ar is off).  Only mamba_p differentiated."""
        keys = jax.random.split(rng_key, bptt)

        def loss_fn(mp):
            s = state_in
            losses = []
            current_inputs = chunk_inputs[0]
            for i in range(bptt):
                # Use truth in truth-prefix, predictions otherwise
                if i < truth_prefix_steps:
                    current_inputs = chunk_inputs[i]
                loss_i, preds_i, s = _one_step_ckpt(
                    mp, gc_p, s, keys[i],
                    current_inputs, chunk_targets[i], chunk_forcings[i])
                # In rolling_ar: only accumulate loss on AR-tail steps
                if (not rolling_ar) or (i >= truth_prefix_steps):
                    losses.append(loss_i)
                # Build next input
                if i < bptt - 1:
                    if i + 1 < truth_prefix_steps:
                        current_inputs = chunk_inputs[i + 1]
                    else:
                        preds_for_feed = (
                            jax.tree_util.tree_map(jax.lax.stop_gradient, preds_i)
                            if feedback_stop_gradient else preds_i)
                        current_inputs = _advance_autoregressive_inputs(
                            current_inputs, preds_for_feed, chunk_forcings[i])
            return jnp.mean(jnp.stack(losses)), s

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(mamba_p)
        grad_norm = optax.global_norm(grads)
        updates, new_opt = opt.update(grads, opt_st, mamba_p)
        new_mp = optax.apply_updates(mamba_p, updates)
        # Stop_grad mamba_state at iter boundary (don't BPTT across train_steps)
        new_state = _stop_grad_state(new_state)
        return new_mp, new_state, new_opt, loss, grad_norm

    # ----- 7) Build per-chunk batches helper -----
    def build_chunk_batches(anchors_in_chunk):
        """For bptt consecutive anchors, build per-step (inputs, targets, forcings) tuples.

        Uses NumpyBatchCache — ~ms per batch instead of zarr fetch's seconds.
        """
        inputs_list, targets_list, forcings_list = [], [], []
        for a in anchors_in_chunk:
            inp, tgt, frc = train_cache.build_batch_from_indices(
                indices=[int(a)],
                input_steps=input_steps, target_steps=1,
                task_cfg=task_cfg, dt=dt)
            inputs_list.append(inp)
            targets_list.append(tgt)
            forcings_list.append(frc)
        return tuple(inputs_list), tuple(targets_list), tuple(forcings_list)

    # ----- 7b) AR feedback time-coord sanity check (rolling_ar mode only) -----
    # Catches the silent bug where chunk_forcings[i] has a different time
    # coord than the prediction at step i — AR feedback then writes the
    # wrong forcing into the next input frame.
    if rolling_ar:
        first_chunk_anchors = train_segments[0][:bptt]
        ci0, ct0, cf0 = build_chunk_batches(first_chunk_anchors)
        try:
            t_input = np.asarray(ci0[0]["time"].values)
            t_target = np.asarray(ct0[0]["time"].values)
            t_force = np.asarray(cf0[0]["time"].values)
            print(f"[gcmamba_v1] AR coord sanity (first chunk, step 0):")
            print(f"  input.time = {t_input}")
            print(f"  target.time= {t_target}")
            print(f"  force.time = {t_force}")
            if not np.array_equal(t_target, t_force):
                raise AssertionError(
                    f"target.time != force.time at step 0! "
                    f"AR feedback will silently corrupt inputs because "
                    f"_advance_autoregressive_inputs splices predictions "
                    f"(at target.time) and forcings (at force.time) into "
                    f"the same next-input frame.")
            # Also check truth_prefix transition: step truth_prefix-1 -> truth_prefix
            if truth_prefix_steps >= 1 and truth_prefix_steps < bptt:
                i_tp = truth_prefix_steps - 1
                t_tgt_last_truth = np.asarray(ct0[i_tp]["time"].values)
                t_inp_first_ar = np.asarray(ci0[i_tp + 1]["time"].values)[-1:]
                if not np.array_equal(t_tgt_last_truth, t_inp_first_ar):
                    print(f"  WARN: last truth target {t_tgt_last_truth} != "
                          f"first AR input's latest frame {t_inp_first_ar}. "
                          f"This may be OK if input-window construction inserts "
                          f"the previous step, but verify.")
            print(f"[gcmamba_v1] AR coord sanity OK")
        except KeyError as e:
            print(f"[gcmamba_v1] AR coord sanity SKIPPED (no time coord: {e})")
        # release the temporary chunk
        del ci0, ct0, cf0

    # ----- 8) Training loop -----
    train_losses = []
    eval_losses = []
    seg_iter = iter(train_segments)
    cur_segment = next(seg_iter)
    seg_pos = 0
    # Mamba state for current segment
    cur_state = state

    t0 = time.time()
    for step in range(cfg.start_step, cfg.max_steps + 1):
        # Advance segment if exhausted
        while seg_pos + bptt > len(cur_segment):
            try:
                cur_segment = next(seg_iter)
            except StopIteration:
                # reshuffle (deterministic per seed)
                rng, k_shuf = jax.random.split(rng)
                perm = np.array(jax.random.permutation(k_shuf, len(train_segments)))
                train_segments_shuf = [train_segments[int(i)] for i in perm]
                seg_iter = iter(train_segments_shuf)
                cur_segment = next(seg_iter)
            seg_pos = 0
            # Reset Mamba state at start of new segment.
            cur_state = jax.tree_util.tree_map(jnp.zeros_like, state)

        anchors_in_chunk = cur_segment[seg_pos:seg_pos + bptt]
        chunk_inputs, chunk_targets, chunk_forcings = build_chunk_batches(anchors_in_chunk)
        rng, step_key = jax.random.split(rng)
        t_step = time.time()
        mamba_params, cur_state, opt_state, loss, grad_norm = train_step(
            mamba_params, gc_params, cur_state, opt_state, step_key,
            chunk_inputs, chunk_targets, chunk_forcings)
        loss_f = float(loss)
        gn_f = float(grad_norm)
        dt_step = time.time() - t_step
        train_losses.append({
            "step": step, "loss": loss_f, "grad_norm": gn_f,
            "step_time": dt_step})
        seg_pos += bptt

        if step == cfg.start_step or step % 50 == 0:
            print(f"step {step}/{cfg.max_steps} loss {loss_f:.4f} "
                  f"gn {gn_f:.3f} step_time {dt_step:.2f}s")

        # Eval
        if step > 0 and step % cfg.eval_every == 0 and val_segments:
            t_eval = time.time()
            ev_losses = []
            # Just take first 4 val segments for speed; iterate K-chunks
            for seg in val_segments[:4]:
                e_state = jax.tree_util.tree_map(jnp.zeros_like, state)
                for j in range(0, len(seg) - bptt + 1, bptt):
                    chunk = seg[j:j + bptt]
                    inputs_l, targets_l, forcings_l = [], [], []
                    for a in chunk:
                        inp, tgt, frc = eval_cache.build_batch_from_indices(
                            indices=[int(a)],
                            input_steps=input_steps, target_steps=1,
                            task_cfg=task_cfg, dt=dt)
                        inputs_l.append(inp); targets_l.append(tgt); forcings_l.append(frc)
                    keys = jax.random.split(jax.random.PRNGKey(0), bptt)
                    # Eval is always truth-fed (Ilya convention)
                    for i in range(bptt):
                        loss_i, _preds, e_state = _one_step_apply(
                            mamba_params, gc_params, e_state, keys[i],
                            inputs_l[i], targets_l[i], forcings_l[i])
                        ev_losses.append(float(loss_i))
            mean_eval = float(np.mean(ev_losses))
            eval_losses.append({"step": step, "loss": mean_eval})
            print(f"[eval] step {step} mean_loss {mean_eval:.4f}  "
                  f"({time.time() - t_eval:.1f}s)")

        # Save logs
        if step % 200 == 0:
            with (out_dir / "train_log.json").open("w") as f:
                json.dump(train_losses, f)
            with (out_dir / "eval_log.json").open("w") as f:
                json.dump(eval_losses, f)

        # Save ckpt
        if step > 0 and step % cfg.checkpoint_every == 0:
            full = merge_params(mamba_params, gc_params)
            with (out_dir / f"gcmamba_v1_step{step}.pkl").open("wb") as f:
                pickle.dump({"params": full, "mamba_params": mamba_params,
                             "step": step, "model_cfg": dataclasses.asdict(model_cfg),
                             "task_cfg": dataclasses.asdict(task_cfg)}, f)
            print(f"[ckpt] saved step {step}")

    # Final save
    with (out_dir / "train_log.json").open("w") as f:
        json.dump(train_losses, f)
    with (out_dir / "eval_log.json").open("w") as f:
        json.dump(eval_losses, f)
    full = merge_params(mamba_params, gc_params)
    with (out_dir / f"gcmamba_v1_step{cfg.max_steps}.pkl").open("wb") as f:
        pickle.dump({"params": full, "mamba_params": mamba_params,
                     "step": cfg.max_steps,
                     "model_cfg": dataclasses.asdict(model_cfg),
                     "task_cfg": dataclasses.asdict(task_cfg)}, f)
    print(f"[done] {cfg.max_steps} steps in {(time.time() - t0)/3600:.2f}h. out={out_dir}")


if __name__ == "__main__":
    main()
