"""v24-CLOSED MEANS — same closed-loop rollout as eval_v24_closed.py, but
ALSO accumulates lat-weighted GLOBAL MEAN traces of (truth, baseline_pred,
full_pred) per channel per lead step. For mean drift / climate diagnostic.

Output JSON adds `per_channel_means` field alongside the standard RMSE arrays.

ORIGINAL DOCSTRING (RMSE-only) follows:

v24-CLOSED eval — closed-loop AR rollout matching train_mz_v24_closed.py.

Logical design:
  step 1: bp_1 = G(input_0); rp_1 = R(input_0, s_0); full_1 = bp_1 + rp_1; metric vs truth_1
  step 2: cur_1 = shift(input_0, full_1); bp_2 = G(cur_1); ... full_2 = bp_2 + rp_2; metric vs truth_2
  step k: cur_{k-1} = shift(cur_{k-2}, full_{k-1}); ... full_k = bp_k + rp_k; metric vs truth_k

Two independent branches for fair baseline denominator:
  baseline branch: x_b_{k+1} = G(x_b_k)                — pure baseline self-rollout
  full branch:     x_f_{k+1} = G(x_f_k) + R(x_f_k, s_k) — closed-loop full model

Reports lat-weighted RMSE per lead step for both. Output JSON same shape as
eval_v22_clean.py's cold_full output for direct comparison.

Optional --warmup-steps W: prepend W truth-feedback steps to warm both branches
identically before the metric phase. Phase 2 starts at truth-anchor t0+W*6h
for both branches.

CLEAN: zero-init Mamba state at eval start (never load ckpt residual_state).
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
    p.add_argument("--ckpt", required=True,
                   help="v24_closed (or compatible) residual ckpt .pkl")
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
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Optional truth-feedback warmup before metric phase.")
    p.add_argument("--sample-pool-target-steps", type=int, default=None,
                   help="Override the future-steps requirement for anchor sampling. "
                        "Default = warmup_steps + target_steps. Pass the LARGEST value "
                        "(e.g., 240 for W=200+K=40) to ensure all warmups use SAME "
                        "anchor pool → apples-to-apples W ablation.")
    p.add_argument("--reset-state-after-warmup", action="store_true", default=False,
                   help="After truth warmup, reset Mamba SSM state (rs) to zero "
                        "before the metric phase. Isolates: 'is the warmed Mamba "
                        "state OOD?' Compare warm vs warm-with-reset_state.")
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    # Defaults set to v22closed small-Mamba (production K-scan). For v24
    # big-Mamba ckpts, explicitly override:
    #   --temporal-hidden-size 512 --temporal-layers 3 --temporal-d-conv 8
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
    print(f"[v24-closed-eval] ckpt: {ckpt_path}  warmup={W}  K={K}  feedback=FULL_PRED (closed-loop)")

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
    sample_pool_target_steps = (cfg.sample_pool_target_steps
                                 if cfg.sample_pool_target_steps is not None
                                 else total_steps)
    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, sample_pool_target_steps)
    print(f"[v24-closed-eval] val_indices (need {sample_pool_target_steps} future steps "
          f"for sample pool, actual W+K = {total_steps}): {len(val_indices)}")

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
    n_p = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[v24-closed-eval] residual_params {n_p:,}, residual_state = zero init")

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

    def closed_loop_rollout(rng, inputs, all_targets, all_forcings):
        """Two independent branches:
          baseline branch: x_b_{k+1} = G(x_b_k)
          full branch:     x_f_{k+1} = G(x_f_k) + R(x_f_k, s_k)
        Optional W-step truth warmup for both branches.
        """
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

        # Optional reset of Mamba SSM state — isolate "warmed state OOD?" effect.
        # Preserves the truth-warmed input window (cur) and baseline_state (bs),
        # only zeros out the Mamba state (rs) before metric phase.
        if cfg.reset_state_after_warmup:
            rs = residual_state_init

        # PHASE 2: closed-loop rollout, two branches
        cur_b = cur            # baseline branch
        bs_b = bs
        cur_f = cur            # full branch
        bs_f = bs
        rs_f = rs
        baseline_chunks = []
        full_chunks = []
        for k in range(K):
            kk = W + k
            tgt_k = all_targets.isel(time=slice(kk, kk + 1))
            frc_k = all_forcings.isel(time=slice(kk, kk + 1))
            rng, kk_base, kk_res = jax.random.split(rng, 3)
            # SHARED rng for both branches' baseline calls
            bp_b, bs_b = _baseline_step(baseline_params, bs_b, kk_base, cur_b, tgt_k, frc_k)
            bp_f, bs_f = _baseline_step(baseline_params, bs_f, kk_base, cur_f, tgt_k, frc_k)
            rp_f, rs_f = _residual_step(residual_params, rs_f, kk_res, cur_f, tgt_k, frc_k)
            full_pred = jax.tree_util.tree_map(lambda b, r: b + r, bp_f, rp_f)
            baseline_chunks.append(bp_b)
            full_chunks.append(full_pred)
            if k < K - 1:
                cur_b = _shift_inputs_with_field(cur_b, bp_b, frc_k)
                cur_f = _shift_inputs_with_field(cur_f, full_pred, frc_k)  # CLOSED-LOOP

        baseline_pred = xr.concat(baseline_chunks, dim="time")
        full_pred_traj = xr.concat(full_chunks, dim="time")
        return baseline_pred, full_pred_traj

    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = sorted(rng_np.choice(val_indices, size=cfg.n_samples,
                                       replace=False).tolist())

    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_da = xr.DataArray(cos_lat / cos_lat.mean(), dims="lat")

    sum_sq_b, sum_sq_f, n_per_var = {}, {}, {}
    sum_abs_b, sum_abs_f = {}, {}
    sum_sq_b_pl, sum_sq_f_pl, n_per_pl = {}, {}, {}
    sum_abs_b_pl, sum_abs_f_pl = {}, {}
    # Mean drift accumulators: per-channel lat-weighted GLOBAL MEAN at each lead
    sum_mean_t, sum_mean_b, sum_mean_f = {}, {}, {}
    sum_mean_t_pl, sum_mean_b_pl, sum_mean_f_pl = {}, {}, {}

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=total_steps, task_cfg=task_cfg, dt=dt)
        rng, rng_chain = jax.random.split(rng)
        baseline_pred, full_pred = closed_loop_rollout(rng_chain, inp, tgt, frc)
        tgt_eval = tgt.isel(time=slice(W, W + K))

        for var in tgt_eval.data_vars:
            truth = tgt_eval[var].astype("float32")
            bp = baseline_pred[var].astype("float32")
            fp = full_pred[var].astype("float32")
            if var not in sum_sq_b:
                sum_sq_b[var] = np.zeros(K); sum_sq_f[var] = np.zeros(K)
                sum_abs_b[var] = np.zeros(K); sum_abs_f[var] = np.zeros(K)
                sum_mean_t[var] = np.zeros(K); sum_mean_b[var] = np.zeros(K); sum_mean_f[var] = np.zeros(K)
                n_per_var[var] = 0
            for k in range(K):
                err_b_k = bp.isel(time=k) - truth.isel(time=k)
                err_f_k = fp.isel(time=k) - truth.isel(time=k)
                sum_sq_b[var][k] += float(((err_b_k ** 2) * cos_lat_da).mean().values)
                sum_sq_f[var][k] += float(((err_f_k ** 2) * cos_lat_da).mean().values)
                sum_abs_b[var][k] += float((np.abs(err_b_k) * cos_lat_da).mean().values)
                sum_abs_f[var][k] += float((np.abs(err_f_k) * cos_lat_da).mean().values)
                # Lat-weighted global mean (signed)
                sum_mean_t[var][k] += float((truth.isel(time=k) * cos_lat_da).mean().values)
                sum_mean_b[var][k] += float((bp.isel(time=k) * cos_lat_da).mean().values)
                sum_mean_f[var][k] += float((fp.isel(time=k) * cos_lat_da).mean().values)
            n_per_var[var] += 1

            if "level" in truth.dims:
                if var not in sum_sq_b_pl:
                    sum_sq_b_pl[var] = {}; sum_sq_f_pl[var] = {}
                    sum_abs_b_pl[var] = {}; sum_abs_f_pl[var] = {}
                    sum_mean_t_pl[var] = {}; sum_mean_b_pl[var] = {}; sum_mean_f_pl[var] = {}
                    n_per_pl[var] = {}
                for lev_i, lev in enumerate(truth["level"].values):
                    lev = int(lev)
                    if lev not in sum_sq_b_pl[var]:
                        sum_sq_b_pl[var][lev] = np.zeros(K); sum_sq_f_pl[var][lev] = np.zeros(K)
                        sum_abs_b_pl[var][lev] = np.zeros(K); sum_abs_f_pl[var][lev] = np.zeros(K)
                        sum_mean_t_pl[var][lev] = np.zeros(K)
                        sum_mean_b_pl[var][lev] = np.zeros(K)
                        sum_mean_f_pl[var][lev] = np.zeros(K)
                        n_per_pl[var][lev] = 0
                    for k in range(K):
                        eb = bp.isel(level=lev_i, time=k) - truth.isel(level=lev_i, time=k)
                        ef = fp.isel(level=lev_i, time=k) - truth.isel(level=lev_i, time=k)
                        sum_sq_b_pl[var][lev][k] += float(((eb ** 2) * cos_lat_da).mean().values)
                        sum_sq_f_pl[var][lev][k] += float(((ef ** 2) * cos_lat_da).mean().values)
                        sum_abs_b_pl[var][lev][k] += float((np.abs(eb) * cos_lat_da).mean().values)
                        sum_abs_f_pl[var][lev][k] += float((np.abs(ef) * cos_lat_da).mean().values)
                        sum_mean_t_pl[var][lev][k] += float((truth.isel(level=lev_i, time=k) * cos_lat_da).mean().values)
                        sum_mean_b_pl[var][lev][k] += float((bp.isel(level=lev_i, time=k) * cos_lat_da).mean().values)
                        sum_mean_f_pl[var][lev][k] += float((fp.isel(level=lev_i, time=k) * cos_lat_da).mean().values)
                    n_per_pl[var][lev] += 1

        if s_i < 3 or (s_i + 1) % 4 == 0:
            print(f"[v24-closed-eval] sample {s_i+1}/{cfg.n_samples}", flush=True)

    out = {
        "eval_mode": "v24_closed_loop",
        "warmup_steps": W,
        "warmup_feedback": "truth" if W > 0 else "none",
        "eval_feedback": "full_pred (closed-loop)",
        "reset_state_after_warmup": bool(cfg.reset_state_after_warmup),
        "baseline_branch": "pure_baseline_self_rollout",
        "residual_state_init": "zero_init_haiku_default_not_ckpt",
        "target_steps": K,
        "actual_total_steps": total_steps,
        "sample_pool_target_steps": sample_pool_target_steps,
        "n_samples": cfg.n_samples,
        "chosen_idx": chosen_idx,
        "ckpt": str(ckpt_path),
        "per_variable_per_step": {},
        "per_channel_per_step": {},
    }
    print(f"\n=== Per-step lat-weighted RMSE (closed-loop, warmup={W}, K={K}) ===")
    for var in sorted(n_per_var):
        n = n_per_var[var]
        r_b = np.sqrt(sum_sq_b[var] / n); r_f = np.sqrt(sum_sq_f[var] / n)
        a_b = sum_abs_b[var] / n; a_f = sum_abs_f[var] / n
        rel_rmse = (r_f - r_b) / np.maximum(r_b, 1e-12) * 100
        rel_mae = (a_f - a_b) / np.maximum(a_b, 1e-12) * 100
        # Mean traces (lat-weighted global mean, signed)
        m_t = (sum_mean_t[var] / n).tolist()
        m_b = (sum_mean_b[var] / n).tolist()
        m_f = (sum_mean_f[var] / n).tolist()
        out["per_variable_per_step"][var] = dict(
            rmse_baseline=r_b.tolist(), rmse_full=r_f.tolist(),
            mae_baseline=a_b.tolist(),  mae_full=a_f.tolist(),
            improvement_pct=(-rel_rmse).tolist(),
            improvement_pct_rmse=(-rel_rmse).tolist(),
            improvement_pct_mae=(-rel_mae).tolist(),
            # Mean traces for climate drift diagnostic
            mean_truth=m_t,
            mean_baseline=m_b,
            mean_full=m_f,
        )
        print(f"  {var:<30}", "RMSE", " ".join(f"K{k+1}{-rel_rmse[k]:+.2f}%" for k in range(min(K, 40))))
        if var in sum_sq_b_pl:
            for lev in sorted(sum_sq_b_pl[var]):
                nl = n_per_pl[var][lev]
                r_b_l = np.sqrt(sum_sq_b_pl[var][lev] / nl); r_f_l = np.sqrt(sum_sq_f_pl[var][lev] / nl)
                a_b_l = sum_abs_b_pl[var][lev] / nl; a_f_l = sum_abs_f_pl[var][lev] / nl
                imp_rmse = -((r_f_l - r_b_l) / np.maximum(r_b_l, 1e-12) * 100)
                imp_mae  = -((a_f_l - a_b_l) / np.maximum(a_b_l, 1e-12) * 100)
                key = f"{var}_level{lev}"
                m_t_l = (sum_mean_t_pl[var][lev] / nl).tolist()
                m_b_l = (sum_mean_b_pl[var][lev] / nl).tolist()
                m_f_l = (sum_mean_f_pl[var][lev] / nl).tolist()
                out["per_channel_per_step"][key] = dict(
                    rmse_baseline=r_b_l.tolist(), rmse_full=r_f_l.tolist(),
                    mae_baseline=a_b_l.tolist(),  mae_full=a_f_l.tolist(),
                    improvement_pct=imp_rmse.tolist(),
                    improvement_pct_rmse=imp_rmse.tolist(),
                    improvement_pct_mae=imp_mae.tolist(),
                    mean_truth=m_t_l,
                    mean_baseline=m_b_l,
                    mean_full=m_f_l,
                )
        else:
            out["per_channel_per_step"][var] = out["per_variable_per_step"][var]

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"\n[v24-closed-eval] wrote {cfg.out_json}")


if __name__ == "__main__":
    main()
