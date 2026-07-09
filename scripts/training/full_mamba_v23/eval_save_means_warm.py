"""Save lat-weighted global mean traces under WARM_BP eval (Ilya-style).

24-step truth warmup → 40-step bp-feedback rollout. Mamba residual is computed
on the baseline trajectory at each step (open-loop, matches v22 training).
Saves per-channel (truth, baseline, full) means at each lead step for 83 channels.

Fixes vs eval_save_means.py:
  - No ckpt residual_state loading (zero init only)
  - 24-step truth warmup before metric phase (Mamba SSM state evolves)
  - Eval phase uses bp-feedback (open-loop, single trajectory)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    DirectResidualNormalizer,
)
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402

from scripts.training.full_mamba_v9.train_mz_v9 import (  # noqa: E402
    GCResidualWithZeroHead, _attach_temporal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", default=base_train.DEFAULT_DATA_PATH)
    p.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
    p.add_argument("--ckpt-in", default=(
        "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
        "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
        "mesh 2to5 - precipitation input and output.npz"))
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-msg-steps", type=int, default=16)
    p.add_argument("--residual-msg-steps", type=int, default=2)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--train-start-year", type=int, default=2020)
    p.add_argument("--train-end-year", type=int, default=2021)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--warmup-steps", type=int, default=24)
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=None)
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
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    ckpt_path = Path(cfg.ckpt)
    W = cfg.warmup_steps
    K = cfg.target_steps
    print(f"[warm-bp-means] ckpt: {ckpt_path}  warmup={W}  K={K}  mode=warm_bp")

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

    total_steps = W + K
    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, total_steps)
    print(f"[warm-bp-means] val_indices (need {total_steps} future steps): {len(val_indices)}")

    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(val_indices[0])],
        input_steps=input_steps, target_steps=total_steps,
        task_cfg=task_cfg, dt=dt)
    sample_targets_1step = sample_targets.isel(time=slice(0, 1))
    sample_forcings_1step = sample_forcings.isel(time=slice(0, 1))

    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state_init = baseline_predict.init(
        k_b, sample_inputs, sample_targets_1step, sample_forcings_1step)
    baseline_params, _ = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)

    # CLEAN init — no ckpt residual_state load.
    _residual_params_init, residual_state_init = residual_predict.init(
        k_r, sample_inputs, sample_targets_1step, sample_forcings_1step)

    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    # do NOT load ckpt["residual_state"]
    n_p = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[warm-bp-means] residual_params {n_p:,}, residual_state = zero init")

    @jax.jit
    def _baseline_step(params, state, key, inp, tgt, frc):
        out, new_state = baseline_predict.apply(params, state, key, inp, tgt, frc)
        return out, new_state
    @jax.jit
    def _residual_step(params, state, key, inp, tgt, frc):
        out, new_state = residual_predict.apply(params, state, key, inp, tgt, frc)
        return out, new_state

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

    def warm_bp_rollout(rng, inputs, all_targets, all_forcings):
        """warm_bp: 24 truth warmup → 40 bp-feedback rollout.
        Single trajectory (baseline self-rollout during eval phase)."""
        cur = inputs
        bs = baseline_state_init
        rs = residual_state_init

        # PHASE 1: warmup (truth feedback)
        for k in range(W):
            tgt_k = all_targets.isel(time=slice(k, k + 1))
            frc_k = all_forcings.isel(time=slice(k, k + 1))
            rng, kk_b, kk_r = jax.random.split(rng, 3)
            _bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
            _rp, rs = _residual_step(residual_params, rs, kk_r, cur, tgt_k, frc_k)
            cur = _shift_inputs_with_field(cur, tgt_k, frc_k)

        # PHASE 2: bp-feedback rollout, residual added on top per step
        baseline_chunks = []
        full_chunks = []
        for k in range(K):
            kk = W + k
            tgt_k = all_targets.isel(time=slice(kk, kk + 1))
            frc_k = all_forcings.isel(time=slice(kk, kk + 1))
            rng, kk_b, kk_r = jax.random.split(rng, 3)
            bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
            rp, rs = _residual_step(residual_params, rs, kk_r, cur, tgt_k, frc_k)
            full_pred = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp)
            baseline_chunks.append(bp)
            full_chunks.append(full_pred)
            if k < K - 1:
                cur = _shift_inputs_with_field(cur, bp, frc_k)

        baseline_pred = xr.concat(baseline_chunks, dim="time")
        full_pred_traj = xr.concat(full_chunks, dim="time")
        return baseline_pred, full_pred_traj

    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = sorted(rng_np.choice(val_indices, size=cfg.n_samples,
                                       replace=False).tolist())

    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_da = xr.DataArray(cos_lat / cos_lat.mean(), dims="lat")

    sum_truth, sum_b, sum_f, n_per = {}, {}, {}, {}

    def _add(key: str, arr_t, arr_b, arr_f):
        if key not in sum_truth:
            sum_truth[key] = np.zeros(K)
            sum_b[key] = np.zeros(K)
            sum_f[key] = np.zeros(K)
            n_per[key] = 0
        sum_truth[key] += arr_t
        sum_b[key] += arr_b
        sum_f[key] += arr_f
        n_per[key] += 1

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=total_steps, task_cfg=task_cfg, dt=dt)
        rng, rng_chain = jax.random.split(rng)
        baseline_pred, full_pred = warm_bp_rollout(rng_chain, inp, tgt, frc)

        # Metric phase: tgt at indices [W .. W+K-1]
        tgt_eval = tgt.isel(time=slice(W, W + K))

        for var in tgt_eval.data_vars:
            truth = tgt_eval[var].astype("float32")
            bp = baseline_pred[var].astype("float32")
            fp = full_pred[var].astype("float32")
            if "level" in truth.dims:
                for lev_i, lev in enumerate(truth["level"].values):
                    lev = int(lev)
                    key = f"{var}_level{lev}"
                    t_arr = np.zeros(K); b_arr = np.zeros(K); f_arr = np.zeros(K)
                    for k in range(K):
                        t_arr[k] = float((truth.isel(level=lev_i, time=k) * cos_lat_da).mean().values)
                        b_arr[k] = float((bp.isel(level=lev_i, time=k) * cos_lat_da).mean().values)
                        f_arr[k] = float((fp.isel(level=lev_i, time=k) * cos_lat_da).mean().values)
                    _add(key, t_arr, b_arr, f_arr)
            else:
                t_arr = np.zeros(K); b_arr = np.zeros(K); f_arr = np.zeros(K)
                for k in range(K):
                    t_arr[k] = float((truth.isel(time=k) * cos_lat_da).mean().values)
                    b_arr[k] = float((bp.isel(time=k) * cos_lat_da).mean().values)
                    f_arr[k] = float((fp.isel(time=k) * cos_lat_da).mean().values)
                _add(var, t_arr, b_arr, f_arr)

        if s_i < 3 or (s_i + 1) % 4 == 0:
            print(f"[warm-bp-means] sample {s_i+1}/{cfg.n_samples}", flush=True)

    out = {
        "eval_mode": "warm_bp",
        "warmup_steps": W,
        "target_steps": K,
        "n_samples": cfg.n_samples,
        "chosen_idx": chosen_idx,
        "ckpt": str(ckpt_path),
        "per_channel": {},
    }
    for key in sorted(n_per):
        n = n_per[key]
        out["per_channel"][key] = dict(
            truth=(sum_truth[key] / n).tolist(),
            baseline=(sum_b[key] / n).tolist(),
            full=(sum_f[key] / n).tolist(),
        )
    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"\n[warm-bp-means] wrote {cfg.out_json}  ({len(out['per_channel'])} channels)")


if __name__ == "__main__":
    main()
