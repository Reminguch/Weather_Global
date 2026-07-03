"""Diagnostic: track Mamba SSM hidden-state h_t norm during inference rollout.

Loads K=22 step 6000 ckpt (best-per-lead winner), rolls out AR for `--target-steps`
plus a warmup, and at each step logs:
  - per-layer ssm_state L2 norm
  - per-layer ssm_state Δ from previous step
  - per-layer conv_cache L2 norm

Writes CSV + PNG to <out-dir>.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import pickle
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import DirectResidualNormalizer  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import (  # noqa: E402
    GCResidualWithZeroHead, _attach_temporal)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-in", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-msg-steps", type=int, default=16)
    p.add_argument("--residual-msg-steps", type=int, default=2)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--train-start-year", type=int, default=2015)
    p.add_argument("--train-end-year", type=int, default=2021)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--warmup-steps", type=int, default=24)
    p.add_argument("--anchor-date", default="2022-07-15T00:00:00")
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=128)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias",
                   action="store_false", default=True)
    p.add_argument("--no-zero-init-out", dest="temporal_zero_init_out",
                   action="store_false", default=True)
    p.add_argument("--temporal-bias", action="store_true", default=False)
    p.add_argument("--mode", choices=["cold_bp", "cold_full"], default="cold_full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def state_norms(state_dict):
    """Return dict from state key → L2 norm (float)."""
    out = {}
    def _walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                _walk(v, f"{path}/{k}" if path else k)
        else:
            arr = np.asarray(d).astype(np.float32)
            out[path] = float(np.linalg.norm(arr))
    _walk(state_dict)
    return out


def state_flat(state_dict):
    """Return concatenated flat array of only ssm_state entries."""
    arrs = []
    keys = []
    def _walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                _walk(v, f"{path}/{k}" if path else k)
        else:
            if "ssm_state" in path:
                arrs.append(np.asarray(d).astype(np.float32).flatten())
                keys.append(path)
    _walk(state_dict)
    if arrs:
        return np.concatenate(arrs), keys
    return np.array([]), []


def main():
    cfg = parse()
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.ckpt)
    K = cfg.target_steps; W = cfg.warmup_steps

    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg_baseline = dataclasses.replace(
        base_model_cfg, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.baseline_msg_steps)
    model_cfg_residual = dataclasses.replace(
        base_model_cfg, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.residual_msg_steps)
    norm_stats = base_train.load_stats(Path(cfg.stats_dir))

    class _S:
        data_path = cfg.data_path
        resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = cfg.train_start_year
        train_end_year = cfg.train_end_year
    _train_ds, eval_ds = base_train._open_local_splits(_S)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    target_time = np.datetime64(cfg.anchor_date)
    anchor_idx = int(np.argmin(np.abs(eval_ds.time.values - target_time)))
    print(f"[diag] anchor idx={anchor_idx} actual={eval_ds.time.values[anchor_idx]}")

    use_bf16 = True
    def _bp():
        p = gc.GraphCast(model_cfg_baseline, task_cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        return normalization.InputsAndResiduals(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
    def _rp():
        p = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
        _attach_temporal(p, cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        return DirectResidualNormalizer(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
    baseline_predict = hk.transform_with_state(
        lambda inp, tgt, frc: _bp()(inp, targets_template=tgt, forcings=frc))
    residual_predict = hk.transform_with_state(
        lambda inp, tgt, frc: _rp()(inp, targets_template=tgt, forcings=frc))

    total = W + K
    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[anchor_idx],
        input_steps=input_steps, target_steps=total,
        task_cfg=task_cfg, dt=dt)
    sample_targets_1 = sample_targets.isel(time=slice(0, 1))
    sample_forcings_1 = sample_forcings.isel(time=slice(0, 1))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, kb, kr = jax.random.split(rng, 3)
    baseline_params, baseline_state = baseline_predict.init(
        kb, sample_inputs, sample_targets_1, sample_forcings_1)
    baseline_params, _ = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    _rp_init, residual_state_init = residual_predict.init(
        kr, sample_inputs, sample_targets_1, sample_forcings_1)

    with ckpt_path.open("rb") as f:
        ck = pickle.load(f)
    residual_params = ck["residual_params"]

    print(f"[diag] state keys sample:")
    for k in list(residual_state_init.keys())[:10]:
        print(f"  {k}: {type(residual_state_init[k]).__name__}")

    @jax.jit
    def _bp_step(p, s, k, inp, tgt, frc):
        return baseline_predict.apply(p, s, k, inp, tgt, frc)
    @jax.jit
    def _rp_step(p, s, k, inp, tgt, frc):
        return residual_predict.apply(p, s, k, inp, tgt, frc)

    def _shift(prev_inputs, new_field_ds, forcings_next):
        target_time = prev_inputs.time.values[-1:] + dt
        ns = new_field_ds.assign_coords(time=target_time)
        fn = forcings_next.assign_coords(time=target_time)
        next_frame = xr.merge([ns, fn])
        if "datetime" in next_frame.coords:
            next_frame = next_frame.drop_vars("datetime")
        keys_in = [k for k in next_frame.data_vars if k in prev_inputs.data_vars]
        merged = xr.concat(
            [prev_inputs, next_frame[keys_in]],
            dim="time", data_vars="different")
        return merged.tail(time=input_steps)

    # Rollout: total = W + K steps. Cold_full means feedback with baseline+residual;
    # cold_bp means feedback with baseline only.
    cur = sample_inputs
    bs = baseline_state
    rs = residual_state_init

    trajectory = []
    prev_flat = None
    for i in range(total):
        tgt_k = sample_targets.isel(time=slice(i, i + 1))
        frc_k = sample_forcings.isel(time=slice(i, i + 1))
        rng, kb_i, kr_i = jax.random.split(rng, 3)
        bp, bs = _bp_step(baseline_params, bs, kb_i, cur, tgt_k, frc_k)
        rp, rs = _rp_step(residual_params, rs, kr_i, cur, tgt_k, frc_k)
        # State norm at this step (AFTER residual predict updated rs).
        rs_np = jax.tree_util.tree_map(np.asarray, rs)
        norms = state_norms(rs_np)
        flat, keys_seen = state_flat(rs_np)
        total_norm = float(np.linalg.norm(flat))
        dprev = float(np.linalg.norm(flat - prev_flat)) if prev_flat is not None else float("nan")
        prev_flat = flat.copy()
        # feedback
        if cfg.mode == "cold_full":
            full = jax.tree_util.tree_map(lambda a, b: a + b, bp, rp)
            cur = _shift(cur, full, frc_k)
        else:
            cur = _shift(cur, bp, frc_k)
        phase = "warmup" if i < W else "metric"
        step_idx = i if i < W else i - W
        trajectory.append({
            "step": i, "phase": phase, "step_in_phase": step_idx,
            "total_norm": total_norm, "delta_prev": dprev,
            "n_ssm_entries": len(keys_seen),
        })
        if i % 5 == 0:
            print(f"  {phase} step {i:>3d}  ‖h‖={total_norm:>10.3f}  Δprev={dprev:>10.3f}")

    # Save CSV
    csv_path = out_dir / f"ssm_state_trajectory_{cfg.mode}.csv"
    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory[0].keys()))
        writer.writeheader()
        for row in trajectory: writer.writerow(row)
    print(f"[diag] wrote {csv_path}")

    # Also dump final layer-by-layer norms
    layer_csv = out_dir / f"ssm_layer_norms_{cfg.mode}.csv"
    with layer_csv.open("w") as f:
        f.write("key,final_norm\n")
        for k, v in sorted(norms.items()):
            f.write(f"{k},{v:.6f}\n")
    print(f"[diag] wrote {layer_csv}")


if __name__ == "__main__":
    main()
