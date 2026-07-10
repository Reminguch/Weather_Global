"""Mamba SSM state tracer — single anchor, cold_bp rollout, dump ssm_state per step.
Output: npz with 'states' array shape (K+1, n_layers, n_mesh, d_inner, d_state),
        '|h|' per step (L2 norm), and pairwise cosine similarity matrix.
"""
from __future__ import annotations
import argparse, dataclasses, pickle, sys
from pathlib import Path

import haiku as hk
import jax, jax.numpy as jnp
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization
import scripts.training.train_graphcast as base_train
from src.models.graphcast.training.core.model import DirectResidualNormalizer
from src.models.mamba.training.param_utils import overlay_matching_params
from scripts.training.full_mamba_v9.train_mz_v9 import (
    GCResidualWithZeroHead, _attach_temporal,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-in", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--mesh-size", type=int, default=4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-msg-steps", type=int, default=6)
    p.add_argument("--residual-msg-steps", type=int, default=2)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--train-start-year", type=int, default=2015)
    p.add_argument("--train-end-year", type=int, default=2021)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=128)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias",
                   action="store_false", default=True)
    p.add_argument("--no-zero-init-out", dest="temporal_zero_init_out",
                   action="store_false", default=True)
    p.add_argument("--temporal-bias", action="store_true", default=False)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--anchor-idx", type=int, default=None,
                   help="If None, picks middle of val_indices")
    p.add_argument("--eval-mode", default="cold_bp", choices=["cold_bp", "cold_full"],
                   help="cold_bp: baseline self-rollout, residual instant. cold_full: corrected feedback (closed-loop)")
    p.add_argument("--residual-state-init", default="ckpt", choices=["ckpt", "zero", "warm24"],
                   help="ckpt: load from ckpt['residual_state']. zero: hk.init zero. warm24: 24 truth-fed steps to evolve state.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def extract_ssm_states(rs):
    """Walk through haiku state dict; collect all arrays under '~_state' keys."""
    found = []
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}/{k}" if prefix else k)
        elif hasattr(obj, '_fields'):  # NamedTuple
            for fname in obj._fields:
                walk(getattr(obj, fname), f"{prefix}/{fname}")
        else:
            if hasattr(obj, 'shape') and 'ssm_state' in prefix:
                found.append((prefix, np.asarray(obj)))
    walk(rs)
    return found


def main():
    cfg = parse_args()
    print(f"[trace] ckpt: {cfg.ckpt}")
    K = cfg.target_steps

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

    class _SplitCfg:
        data_path = cfg.data_path
        resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = cfg.train_start_year
        train_end_year = cfg.train_end_year
    _train_ds, eval_ds = base_train._open_local_splits(_SplitCfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    use_bf16 = True

    def _build_baseline():
        p = gc.GraphCast(model_cfg_baseline, task_cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p

    def _build_residual():
        p = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
        _attach_temporal(p, cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        p = DirectResidualNormalizer(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p

    def baseline_fn(inputs, targets, forcings):
        return _build_baseline()(inputs, targets_template=targets, forcings=forcings)
    def residual_fn(inputs, targets, forcings):
        return _build_residual()(inputs, targets_template=targets, forcings=forcings)
    baseline_predict = hk.transform_with_state(baseline_fn)
    residual_predict = hk.transform_with_state(residual_fn)

    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, K)
    print(f"[trace] {len(val_indices)} valid anchors; using idx={cfg.anchor_idx or val_indices[len(val_indices)//2]}")
    if cfg.anchor_idx is None:
        anchor_idx = int(val_indices[len(val_indices)//2])
    else:
        anchor_idx = cfg.anchor_idx

    inputs, all_targets, all_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[anchor_idx], input_steps=input_steps,
        target_steps=K, task_cfg=task_cfg, dt=dt)
    sample_t1 = all_targets.isel(time=slice(0, 1))
    sample_f1 = all_forcings.isel(time=slice(0, 1))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state_init = baseline_predict.init(
        k_b, inputs, sample_t1, sample_f1)
    baseline_params, _ = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    _, residual_state_init = residual_predict.init(
        k_r, inputs, sample_t1, sample_f1)

    with open(cfg.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    _hk_zero_state = residual_state_init  # hk.init returns true-zero ssm/conv state
    if cfg.residual_state_init == "ckpt":
        if "residual_state" in ckpt and ckpt["residual_state"]:
            residual_state_init = ckpt["residual_state"]
            print(f"[trace] residual_state: LOADED FROM CKPT (non-zero)")
        else:
            print(f"[trace] residual_state: zero (no key in ckpt)")
    elif cfg.residual_state_init == "zero":
        residual_state_init = _hk_zero_state
        print(f"[trace] residual_state: TRUE ZERO (hk.init)")
    elif cfg.residual_state_init == "warm24":
        residual_state_init = _hk_zero_state  # start zero, will warm below
        print(f"[trace] residual_state: WILL WARMUP 24 truth-fed steps from zero")
    # Print actual h_0 norms
    h0_states = extract_ssm_states(residual_state_init)
    for name, arr in h0_states:
        a = np.asarray(arr)
        print(f"  h_0 {name.split('/')[-2]}: norm={np.linalg.norm(a):.4f}")

    @jax.jit
    def _baseline_step(p, s, k, x, t, f):
        return baseline_predict.apply(p, s, k, x, t, f)

    @jax.jit
    def _residual_step(p, s, k, x, t, f):
        return residual_predict.apply(p, s, k, x, t, f)

    def _shift_inputs_with_field(prev_inputs, new_field_ds, forcings_next):
        target_time = prev_inputs.time.values[-1:] + dt
        ns = new_field_ds.assign_coords(time=target_time)
        fn = forcings_next.assign_coords(time=target_time)
        next_frame = xr.merge([ns, fn])
        if "datetime" in next_frame.coords:
            next_frame = next_frame.drop_vars("datetime")
        keys_in_next = [k for k in next_frame.data_vars if k in prev_inputs.data_vars]
        next_inputs_part = next_frame[keys_in_next]
        merged = xr.concat([prev_inputs, next_inputs_part], dim="time", data_vars="different")
        return merged.tail(time=input_steps)

    # Rollout — capture rs at each step
    cur = inputs
    bs = baseline_state_init
    rs = residual_state_init

    # WARM24 INIT: run 24 truth-fed steps (using all_targets[0:24]) to evolve rs.
    # All_targets has K=cfg.target_steps elements; we need at least 24+K for warm24,
    # so caller should pass --target-steps >= 24. We re-build batch with extra steps below.
    if cfg.residual_state_init == "warm24":
        # Re-build batch with extra 24 timesteps
        extra_inputs, extra_targets, extra_forcings = base_train.build_batch_from_indices(
            eval_ds, indices=[anchor_idx], input_steps=input_steps,
            target_steps=24 + K, task_cfg=task_cfg, dt=dt)
        # Warm up 24 truth-fed steps
        for w in range(24):
            tw = extra_targets.isel(time=slice(w, w+1))
            fw = extra_forcings.isel(time=slice(w, w+1))
            rng, kb, kr = jax.random.split(rng, 3)
            _, bs = _baseline_step(baseline_params, bs, kb, cur, tw, fw)
            _, rs = _residual_step(residual_params, rs, kr, cur, tw, fw)
            cur = _shift_inputs_with_field(cur, tw, fw)
        print(f"[trace] WARM24 done: 24 truth-fed steps applied")
        h_post_warm = extract_ssm_states(rs)
        for name, arr in h_post_warm:
            a = np.asarray(arr)
            print(f"  h_24 {name.split('/')[-2]}: norm={np.linalg.norm(a):.4f}")
        # Use the post-warmup targets/forcings for the metric rollout
        all_targets = extra_targets.isel(time=slice(24, 24+K))
        all_forcings = extra_forcings.isel(time=slice(24, 24+K))

    states_per_step = [extract_ssm_states(rs)]  # state at t=0 (after warmup if any)
    print(f"[trace] mode={cfg.eval_mode}  initial state arrays: {[(k, v.shape) for k, v in states_per_step[0]]}")

    is_full = cfg.eval_mode == "cold_full"
    if is_full:
        # Two branches diverge from step 0: baseline self-rollout vs corrected
        cur_b = cur; bs_b = bs        # pure baseline branch (state not traced)
        cur_f = cur; bs_f = bs        # full branch (state traced)

    for k in range(K):
        tgt_k = all_targets.isel(time=slice(k, k+1))
        frc_k = all_forcings.isel(time=slice(k, k+1))
        rng, kb, kr = jax.random.split(rng, 3)
        if is_full:
            bp_f, bs_f = _baseline_step(baseline_params, bs_f, kb, cur_f, tgt_k, frc_k)
            rp_f, rs = _residual_step(residual_params, rs, kr, cur_f, tgt_k, frc_k)
            full_pred = jax.tree_util.tree_map(lambda b, r: b + r, bp_f, rp_f)
            states_per_step.append(extract_ssm_states(rs))
            if k < K - 1:
                cur_f = _shift_inputs_with_field(cur_f, full_pred, frc_k)
        else:
            bp, bs = _baseline_step(baseline_params, bs, kb, cur, tgt_k, frc_k)
            rp, rs = _residual_step(residual_params, rs, kr, cur, tgt_k, frc_k)
            states_per_step.append(extract_ssm_states(rs))
            if k < K - 1:
                cur = _shift_inputs_with_field(cur, bp, frc_k)
        if (k + 1) % 5 == 0:
            print(f"[trace] step {k+1}/{K} done", flush=True)

    # Stack each layer's state into (T+1, ...) array
    n_layers = len(states_per_step[0])
    layer_names = [name for name, _ in states_per_step[0]]
    print(f"[trace] layers found: {layer_names}")

    out = {}
    for li, name in enumerate(layer_names):
        # Stack over time
        arrs = np.stack([states_per_step[t][li][1] for t in range(K + 1)], axis=0)
        # arrs shape: (T+1, B, n_mesh, d_inner, d_state)  — typical
        # Flatten everything except time
        flat = arrs.reshape(K + 1, -1)  # (T+1, D)
        # Norms
        norms = np.linalg.norm(flat, axis=1)
        # Pairwise cosine similarity (full T+1 × T+1)
        flat_n = flat / np.maximum(norms[:, None], 1e-12)
        cos = flat_n @ flat_n.T
        cos_step = np.array([1.0] + [cos[t, t-1] for t in range(1, K + 1)])

        layer_key = name.replace("/", "_")
        out[f"{layer_key}_norms"] = norms
        out[f"{layer_key}_cos_matrix"] = cos
        out[f"{layer_key}_cos_step"] = cos_step
        out[f"{layer_key}_state_shape"] = np.array(arrs.shape[1:])
        # NEW: dump normalized state vectors so we can compute cross-trajectory cos later
        out[f"{layer_key}_state_normalized_flat"] = flat_n.astype(np.float32)
        print(f"[trace] {name}: shape {arrs.shape}, norms[0..5] = {norms[:5]}, cos(t-1,t)[1..5] = {cos_step[1:6]}")

    np.savez(cfg.out_npz, **out)
    print(f"[trace] saved {cfg.out_npz}")


if __name__ == "__main__":
    main()
