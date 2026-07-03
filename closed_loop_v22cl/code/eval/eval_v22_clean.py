"""v22 CLEAN eval — fixes 2 bugs in eval_v20_rollout.py and eval_v22_warmstart.py.

Bug 1 (was in warmstart only): single trajectory `current_inp` was fed to both
baseline and Mamba, then advanced by full_pred. From AR step 2 on, the reported
`rmse_baseline` was actually RMSE of `G(x_full_trajectory)`, not pure baseline
rollout. Fix: run TWO independent branches when feedback mode = full.

Bug 2 (was in BOTH cold v20 and warm v22 evals): loaded `residual_state` from
ckpt as init state. That is the leftover Mamba SSM state from training's final
batch — random and OOD for eval. Fix: always use zero / default init from
`residual_predict.init(...)`. The ckpt provides params only.

Parametric across 4 eval modes:
  --eval-mode cold_bp    : no warmup, eval-phase feedback = baseline   (= v20 legacy semantics, but with clean state init)
  --eval-mode cold_full  : no warmup, eval-phase feedback = full       (closed-loop, two-branch)
  --eval-mode warm_bp    : 24 truth warmup, eval-phase feedback = baseline
  --eval-mode warm_full  : 24 truth warmup, eval-phase feedback = full (closed-loop, two-branch)

For bp-feedback modes (cold_bp, warm_bp): ONE trajectory = baseline self-rollout.
Residual is computed on the baseline trajectory at each step; full_pred = bp + rp
is just an instantaneous correction, never fed back. Reported rmse_baseline =
RMSE of pure baseline rollout. Reported rmse_full = RMSE of (baseline + Mamba).

For full-feedback modes (cold_full, warm_full): TWO independent branches.
  baseline branch: x_b_{k+1} = G(x_b_k)
  full branch:     x_f_{k+1} = G(x_f_k) + R(x_f_k, s_k)
Reported rmse_baseline = RMSE of pure baseline rollout (left branch).
Reported rmse_full = RMSE of closed-loop full rollout (right branch).
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


EVAL_MODES = ["cold_bp", "cold_full", "warm_bp", "warm_full",
              "warm_bp_reset_state", "warm_full_reset_state"]


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
    p.add_argument("--target-steps", type=int, default=40, help="metric horizon (K)")
    p.add_argument("--warmup-steps", type=int, default=24,
                   help="truth-feedback warmup steps; ignored for cold_* modes")
    p.add_argument("--eval-mode", choices=EVAL_MODES, required=True)
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
    p.add_argument("--tail-calib-head", action="store_true",
                   help="Load v26 ckpt as GCResidualWithTailCalibrationHead.")
    p.add_argument("--residual-state-init", default="ckpt",
                   choices=["ckpt", "zero", "warm24"],
                   help="ckpt: load from ckpt['residual_state'] (default). "
                        "zero: use hk.init zero state. "
                        "warm24: pre-rollout 24 truth-fed steps to generate warm state.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--residual-alpha", type=float, default=1.0,
                   help="Scale applied to residual prediction before adding to baseline "
                        "(full_pred = baseline + alpha * residual). Default 1.0 = "
                        "original behaviour. Diagnostic tool for testing if late-ckpt "
                        "overfit is amplitude-driven.")
    p.add_argument("--out-json", required=True)
    p.add_argument("--force-idx", type=int, default=None,
                   help="DIAG: override chosen_idx to [force_idx], n_samples=1. "
                        "Used for same-anchor comparison with eval_city_trace.")
    return p.parse_args()


def main():
    cfg = parse_args()
    ckpt_path = Path(cfg.ckpt)
    is_warm = cfg.eval_mode.startswith("warm_")
    is_reset_state = cfg.eval_mode.endswith("_reset_state")
    # Strip _reset_state suffix to decide feedback direction.
    base_mode = cfg.eval_mode.replace("_reset_state", "")
    is_full_fb = base_mode.endswith("_full")
    W = cfg.warmup_steps if is_warm else 0
    K = cfg.target_steps
    print(f"[clean-eval] ckpt: {ckpt_path}")
    print(f"[clean-eval] mode={cfg.eval_mode}  warmup={W}  K={K}  "
          f"feedback={'full' if is_full_fb else 'baseline'}  reset_state={is_reset_state}")

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
        if cfg.tail_calib_head:
            from scripts.training.full_mamba_v26.tail_calibration_head import (
                GCResidualWithTailCalibrationHead)
            p = GCResidualWithTailCalibrationHead(model_cfg_residual, task_cfg)
        else:
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

    # CRITICAL: unify anchor pool across all modes. Sample from anchors that
    # support warmup_steps + target_steps future steps even if this mode has
    # W=0, so cold and warm always use the same chosen_idx.
    sample_total_steps = cfg.warmup_steps + cfg.target_steps

    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, sample_total_steps)
    print(f"[clean-eval] unified val_indices (need {sample_total_steps} future steps): {len(val_indices)}")

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

    # REVERTED: load residual_state from ckpt when available. The previous
    # "CRITICAL FIX" zero-initialised the SSM state, but empirically that
    # produces catastrophic deployment behaviour (cold_bp lead-1 RMSE ~2x
    # baseline, warm_full RMSE -1500%) because v22 is stateful — the trained
    # SSM expects to operate from the in-distribution state attained at the
    # end of training, not from s=0. The original eval_v22_warmstart loaded
    # ckpt["residual_state"]; restore that behaviour here.
    _residual_params_init, residual_state_init = residual_predict.init(
        k_r, sample_inputs, sample_targets_1step, sample_forcings_1step)

    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    _hk_zero_state = residual_state_init  # save the hk.init zero state
    if cfg.residual_state_init == "ckpt":
        if "residual_state" in ckpt and ckpt["residual_state"]:
            residual_state_init = ckpt["residual_state"]
            print(f"[clean-eval] residual_state_init = LOADED FROM CKPT")
        else:
            print(f"[clean-eval] residual_state_init = zero (no residual_state in ckpt)")
    elif cfg.residual_state_init == "zero":
        residual_state_init = _hk_zero_state
        print(f"[clean-eval] residual_state_init = ZERO (forced via --residual-state-init zero)")
    elif cfg.residual_state_init == "warm24":
        # warm24 init is applied per-sample below; defer until rollout
        residual_state_init = _hk_zero_state  # start from zero; warmup will fill
        print(f"[clean-eval] residual_state_init = WARM24 (will roll 24 truth-fed steps per sample)")
    _residual_state_init_mode = cfg.residual_state_init

    n_p = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[clean-eval] loaded residual_params {n_p:,}")

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

    def clean_rollout(rng, inputs, all_targets, all_forcings, *, debug=False):
        """Run eval rollout in the configured mode.

        For bp-feedback modes: ONE trajectory = pure baseline self-rollout.
        Residual is computed on the baseline trajectory at each step; full = bp + rp.
        baseline_chunks = bp (= pure baseline), full_chunks = bp + rp.

        For full-feedback modes: TWO independent branches.
        Same RNG key passed to baseline_step in both branches → bp_b is
        deterministic given cur_b, bp_f deterministic given cur_f. In bp-mode
        the branches stay perfectly in sync (sanity-checkable).
        """
        cur = inputs
        bs = baseline_state_init
        rs = residual_state_init

        # --- PHASE 1: WARMUP (only for warm_* modes; uses truth-feedback) ---
        for k in range(W):
            tgt_k = all_targets.isel(time=slice(k, k + 1))
            frc_k = all_forcings.isel(time=slice(k, k + 1))
            rng, kk_b, kk_r = jax.random.split(rng, 3)
            _bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
            _rp, rs = _residual_step(residual_params, rs, kk_r, cur, tgt_k, frc_k)
            cur = _shift_inputs_with_field(cur, tgt_k, frc_k)

        # --- OPTIONAL RESET: for *_reset_state modes, zero out Mamba SSM state
        # while keeping the truth-warmed input window cur and baseline_state bs.
        # This isolates: "is it the truth-warmed rs that's bad, vs the warmup
        # itself?" Compare warm_full vs warm_full_reset_state.
        if is_reset_state:
            rs = residual_state_init
            if debug:
                print(f"  [reset] rs reset to zero-init after warmup")

        # --- PHASE 2: METRIC ROLLOUT ---
        baseline_chunks = []
        full_chunks = []

        if is_full_fb:
            # TWO branches. cur and bs start identical to the warmed (or initial)
            # state; the branches diverge starting from step k=0.
            cur_b = cur            # baseline branch
            bs_b = bs
            cur_f = cur            # full branch
            bs_f = bs
            rs_f = rs
            for k in range(K):
                kk = W + k
                tgt_k = all_targets.isel(time=slice(kk, kk + 1))
                frc_k = all_forcings.isel(time=slice(kk, kk + 1))
                rng, kk_base, kk_res = jax.random.split(rng, 3)
                # SHARED RNG for both baseline calls — so bp_b vs bp_f
                # differ only because of cur_b vs cur_f / bs_b vs bs_f.
                bp_b, bs_b = _baseline_step(baseline_params, bs_b, kk_base, cur_b, tgt_k, frc_k)
                bp_f, bs_f = _baseline_step(baseline_params, bs_f, kk_base, cur_f, tgt_k, frc_k)
                rp_f, rs_f = _residual_step(residual_params, rs_f, kk_res, cur_f, tgt_k, frc_k)
                _alpha = cfg.residual_alpha
                full_pred = jax.tree_util.tree_map(lambda b, r: b + _alpha * r, bp_f, rp_f)
                baseline_chunks.append(bp_b)
                full_chunks.append(full_pred)
                if k < K - 1:
                    cur_b = _shift_inputs_with_field(cur_b, bp_b, frc_k)
                    cur_f = _shift_inputs_with_field(cur_f, full_pred, frc_k)
        else:
            # ONE branch — trajectory follows baseline (bp-feedback).
            for k in range(K):
                kk = W + k
                tgt_k = all_targets.isel(time=slice(kk, kk + 1))
                frc_k = all_forcings.isel(time=slice(kk, kk + 1))
                rng, kk_base, kk_res = jax.random.split(rng, 3)
                bp, bs = _baseline_step(baseline_params, bs, kk_base, cur, tgt_k, frc_k)
                rp, rs = _residual_step(residual_params, rs, kk_res, cur, tgt_k, frc_k)
                _alpha = cfg.residual_alpha
                full_pred = jax.tree_util.tree_map(lambda b, r: b + _alpha * r, bp, rp)
                baseline_chunks.append(bp)
                full_chunks.append(full_pred)
                if k < K - 1:
                    # bp-feedback: trajectory advances via baseline pred only.
                    cur = _shift_inputs_with_field(cur, bp, frc_k)

        baseline_pred = xr.concat(baseline_chunks, dim="time")
        full_pred_traj = xr.concat(full_chunks, dim="time")
        return baseline_pred, full_pred_traj

    rng_np = np.random.default_rng(cfg.seed)
    if cfg.force_idx is not None:
        assert cfg.force_idx in val_indices, (
            f"--force-idx {cfg.force_idx} not in val_indices "
            f"(range {int(val_indices.min())}..{int(val_indices.max())})")
        chosen_idx = [int(cfg.force_idx)]
        print(f"[clean-eval] DIAG: force_idx={cfg.force_idx}, running single anchor")
    else:
        chosen_idx = sorted(rng_np.choice(val_indices, size=cfg.n_samples,
                                           replace=False).tolist())

    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_da = xr.DataArray(cos_lat / cos_lat.mean(), dims="lat")

    sum_sq_b, sum_sq_f, n_per_var = {}, {}, {}
    sum_abs_b, sum_abs_f = {}, {}
    sum_sq_b_pl, sum_sq_f_pl, n_per_pl = {}, {}, {}
    sum_abs_b_pl, sum_abs_f_pl = {}, {}
    # Residual diagnostics (per variable, per lead).
    #   r_k    = fp_k - bp_k           (Mamba residual output)
    #   e_pre  = truth_k - bp_k        (what residual should fix)
    #   <r, e_pre>, ||r||^2, ||e_pre||^2 all lat-weighted, aggregated over field+samples.
    # Derived per-lead quantities:
    #   cos_t  = <r, e_pre> / (||r|| * ||e_pre||)        — alignment
    #   gain_t = ||r|| / ||e_pre||                       — magnitude ratio
    #   Δ_t    = E^pre - E^post = 2<r, e_pre> - ||r||^2 — instantaneous MSE improvement
    sum_dot_re, sum_norm_r_sq, sum_norm_e_sq = {}, {}, {}
    # RMSB additions (paper F.2): per-cell time-mean of error, then lat-w
    # RMS across (i,j). We accumulate sum of error per cell, divide by n,
    # square, lat-weight sum.
    sum_err_cell_b, sum_err_cell_f = {}, {}     # shape (K, n_lat, n_lon) per var
    sum_err_cell_b_pl, sum_err_cell_f_pl = {}, {}   # nested {var: {lev: (K, lat, lon)}}

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=total_steps, task_cfg=task_cfg, dt=dt)
        rng, rng_chain = jax.random.split(rng)
        baseline_pred, full_pred = clean_rollout(rng_chain, inp, tgt, frc, debug=(s_i == 0))

        # Metric is computed on the EVAL phase only (steps W..W+K-1 of tgt).
        tgt_eval = tgt.isel(time=slice(W, W + K))

        # DIAG: global mean / RMSE for surface variables at select leads (first sample)
        if s_i == 0:
            for k in [0, 19, 27, 39]:
                if k >= K: continue
                for var in ["2m_temperature", "total_precipitation_6hr"]:
                    if var not in tgt_eval.data_vars: continue
                    t_ = np.asarray(tgt_eval[var].isel(time=k).values).squeeze()
                    b_ = np.asarray(baseline_pred[var].isel(time=k).values).squeeze()
                    f_ = np.asarray(full_pred[var].isel(time=k).values).squeeze()
                    print(f"[diag idx={idx} k={k:2d} {var:<26s}] "
                          f"truth mean={t_.mean():+7.2f}  base mean={b_.mean():+7.2f}  full mean={f_.mean():+7.2f}  "
                          f"base RMSE={np.sqrt(((b_-t_)**2).mean()):7.3f}  full RMSE={np.sqrt(((f_-t_)**2).mean()):7.3f}")

        for var in tgt_eval.data_vars:
            truth = tgt_eval[var].astype("float32")
            bp = baseline_pred[var].astype("float32")
            fp = full_pred[var].astype("float32")
            if var not in sum_sq_b:
                sum_sq_b[var] = np.zeros(K); sum_sq_f[var] = np.zeros(K)
                sum_abs_b[var] = np.zeros(K); sum_abs_f[var] = np.zeros(K)
                sum_dot_re[var] = np.zeros(K)
                sum_norm_r_sq[var] = np.zeros(K)
                sum_norm_e_sq[var] = np.zeros(K)
                n_per_var[var] = 0
            for k in range(K):
                err_b_k = bp.isel(time=k) - truth.isel(time=k)
                err_f_k = fp.isel(time=k) - truth.isel(time=k)
                r_k     = fp.isel(time=k) - bp.isel(time=k)         # residual
                e_pre_k = -err_b_k                                  # = truth - bp
                sum_sq_b[var][k] += float(((err_b_k ** 2) * cos_lat_da).mean().values)
                sum_sq_f[var][k] += float(((err_f_k ** 2) * cos_lat_da).mean().values)
                sum_abs_b[var][k] += float((np.abs(err_b_k) * cos_lat_da).mean().values)
                sum_abs_f[var][k] += float((np.abs(err_f_k) * cos_lat_da).mean().values)
                sum_dot_re[var][k]    += float(((r_k * e_pre_k) * cos_lat_da).mean().values)
                sum_norm_r_sq[var][k] += float(((r_k * r_k)     * cos_lat_da).mean().values)
                sum_norm_e_sq[var][k] += float(((e_pre_k * e_pre_k) * cos_lat_da).mean().values)
            # === RMSB accumulator (per-cell bias) ===
            # err has dims (batch, time, lat, lon) for surface vars; we squeeze batch (=1)
            # and want (K=time, lat, lon).
            if "level" not in truth.dims:
                err_b_full = (bp - truth).transpose(..., "time", "lat", "lon").values.astype(np.float32)
                err_f_full = (fp - truth).transpose(..., "time", "lat", "lon").values.astype(np.float32)
                # drop leading batch dim if present (we eval with batch=1)
                while err_b_full.ndim > 3:
                    err_b_full = err_b_full.squeeze(0); err_f_full = err_f_full.squeeze(0)
                if var not in sum_err_cell_b:
                    sum_err_cell_b[var] = np.zeros_like(err_b_full)
                    sum_err_cell_f[var] = np.zeros_like(err_f_full)
                sum_err_cell_b[var] += err_b_full
                sum_err_cell_f[var] += err_f_full
            n_per_var[var] += 1

            if "level" in truth.dims:
                if var not in sum_sq_b_pl:
                    sum_sq_b_pl[var] = {}; sum_sq_f_pl[var] = {}
                    sum_abs_b_pl[var] = {}; sum_abs_f_pl[var] = {}
                    n_per_pl[var] = {}
                    sum_err_cell_b_pl[var] = {}; sum_err_cell_f_pl[var] = {}
                for lev_i, lev in enumerate(truth["level"].values):
                    lev = int(lev)
                    if lev not in sum_sq_b_pl[var]:
                        sum_sq_b_pl[var][lev] = np.zeros(K); sum_sq_f_pl[var][lev] = np.zeros(K)
                        sum_abs_b_pl[var][lev] = np.zeros(K); sum_abs_f_pl[var][lev] = np.zeros(K)
                        n_per_pl[var][lev] = 0
                    for k in range(K):
                        eb = bp.isel(level=lev_i, time=k) - truth.isel(level=lev_i, time=k)
                        ef = fp.isel(level=lev_i, time=k) - truth.isel(level=lev_i, time=k)
                        sum_sq_b_pl[var][lev][k] += float(((eb ** 2) * cos_lat_da).mean().values)
                        sum_sq_f_pl[var][lev][k] += float(((ef ** 2) * cos_lat_da).mean().values)
                        sum_abs_b_pl[var][lev][k] += float((np.abs(eb) * cos_lat_da).mean().values)
                        sum_abs_f_pl[var][lev][k] += float((np.abs(ef) * cos_lat_da).mean().values)
                    # RMSB per-level: per-cell errors → (K, lat, lon) after squeezing batch dim
                    eb_full = (bp.isel(level=lev_i) - truth.isel(level=lev_i)).transpose(..., "time", "lat", "lon").values.astype(np.float32)
                    ef_full = (fp.isel(level=lev_i) - truth.isel(level=lev_i)).transpose(..., "time", "lat", "lon").values.astype(np.float32)
                    while eb_full.ndim > 3:
                        eb_full = eb_full.squeeze(0); ef_full = ef_full.squeeze(0)
                    if lev not in sum_err_cell_b_pl[var]:
                        sum_err_cell_b_pl[var][lev] = np.zeros_like(eb_full)
                        sum_err_cell_f_pl[var][lev] = np.zeros_like(ef_full)
                    sum_err_cell_b_pl[var][lev] += eb_full
                    sum_err_cell_f_pl[var][lev] += ef_full
                    n_per_pl[var][lev] += 1

        if s_i < 3 or (s_i + 1) % 4 == 0:
            print(f"[clean-eval] sample {s_i+1}/{cfg.n_samples}", flush=True)

    print()
    print(f"=== Per-step lat-weighted RMSE (mode={cfg.eval_mode}, warmup={W}, K={K}) ===")
    out = {
        "eval_mode": cfg.eval_mode,
        "warmup_steps": W,
        "warmup_feedback": "truth" if W > 0 else "none",
        "eval_feedback": "full" if is_full_fb else "baseline",
        "rs_reset_after_warmup": is_reset_state,
        "baseline_branch": "pure_baseline_self_rollout",
        "residual_state_init": "loaded_from_ckpt_or_zero_default",
        "target_steps": K,
        "sample_total_steps": sample_total_steps,
        "n_samples": cfg.n_samples,
        "chosen_idx": chosen_idx,
        "ckpt": str(ckpt_path),
        "per_variable_per_step": {},
        "per_channel_per_step": {},
        "residual_diagnostics_per_variable": {},
    }
    # Lat-weighted RMS function for per-cell bias (sqrt of cos-lat-weighted mean of squared bias).
    cos_lat_2d = (cos_lat / cos_lat.mean()).astype(np.float32)  # shape (n_lat,)
    def lat_w_rms(bias_cell):  # bias_cell: (K, n_lat, n_lon)
        sq = bias_cell ** 2
        # weight along lat axis: weighted mean over (lat, lon)
        weighted_mean = (sq * cos_lat_2d[None, :, None]).mean(axis=(1, 2))
        return np.sqrt(weighted_mean)  # shape (K,)

    for var in sorted(n_per_var):
        n = n_per_var[var]
        r_b = np.sqrt(sum_sq_b[var] / n); r_f = np.sqrt(sum_sq_f[var] / n)
        a_b = sum_abs_b[var] / n; a_f = sum_abs_f[var] / n
        rel_rmse = (r_f - r_b) / np.maximum(r_b, 1e-12) * 100
        rel_mae = (a_f - a_b) / np.maximum(a_b, 1e-12) * 100
        entry = dict(
            rmse_baseline=r_b.tolist(), rmse_full=r_f.tolist(),
            mae_baseline=a_b.tolist(),  mae_full=a_f.tolist(),
            improvement_pct=(-rel_rmse).tolist(),
            improvement_pct_rmse=(-rel_rmse).tolist(),
            improvement_pct_mae=(-rel_mae).tolist(),
        )
        # RMSB (paper F.2): per-cell time-mean of error, then lat-w RMS.
        # Only available for surface vars at this top level (atmo vars get RMSB
        # in per_channel_per_step below per level).
        if var in sum_err_cell_b:
            bias_b = sum_err_cell_b[var] / n   # (K, n_lat, n_lon)
            bias_f = sum_err_cell_f[var] / n
            rmsb_b = lat_w_rms(bias_b)
            rmsb_f = lat_w_rms(bias_f)
            rel_rmsb = (rmsb_f - rmsb_b) / np.maximum(rmsb_b, 1e-12) * 100
            entry.update(
                rmsb_baseline=rmsb_b.tolist(),
                rmsb_full=rmsb_f.tolist(),
                improvement_pct_rmsb=(-rel_rmsb).tolist(),
            )
        out["per_variable_per_step"][var] = entry

        # Residual diagnostics: cos, gain, Δ_t (instantaneous MSE improvement).
        dot_re  = sum_dot_re[var] / n              # <r, e_pre> aggregated
        nr_sq   = sum_norm_r_sq[var] / n           # ||r||^2
        ne_sq   = sum_norm_e_sq[var] / n           # ||e_pre||^2 == MSE^pre
        rn      = np.sqrt(np.maximum(nr_sq, 0.0))
        en      = np.sqrt(np.maximum(ne_sq, 0.0))
        cos_t   = dot_re / np.maximum(rn * en, 1e-12)
        gain_t  = rn / np.maximum(en, 1e-12)
        delta_t = 2.0 * dot_re - nr_sq             # E^pre - E^post in lat-weighted MSE units
        out["residual_diagnostics_per_variable"][var] = dict(
            residual_cosine_by_lead=cos_t.tolist(),
            residual_gain_by_lead=gain_t.tolist(),
            residual_norm_by_lead=rn.tolist(),
            baseline_err_norm_by_lead=en.tolist(),
            dot_r_epre_by_lead=dot_re.tolist(),
            delta_mse_by_lead=delta_t.tolist(),
            E_pre_mse_by_lead=ne_sq.tolist(),
            E_post_mse_by_lead=(ne_sq - delta_t).tolist(),
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
                ch_entry = dict(
                    rmse_baseline=r_b_l.tolist(), rmse_full=r_f_l.tolist(),
                    mae_baseline=a_b_l.tolist(),  mae_full=a_f_l.tolist(),
                    improvement_pct=imp_rmse.tolist(),
                    improvement_pct_rmse=imp_rmse.tolist(),
                    improvement_pct_mae=imp_mae.tolist(),
                )
                # RMSB per (var, level)
                if var in sum_err_cell_b_pl and lev in sum_err_cell_b_pl[var]:
                    bias_b_pl = sum_err_cell_b_pl[var][lev] / nl
                    bias_f_pl = sum_err_cell_f_pl[var][lev] / nl
                    rmsb_b_l = lat_w_rms(bias_b_pl)
                    rmsb_f_l = lat_w_rms(bias_f_pl)
                    rel_rmsb_l = (rmsb_f_l - rmsb_b_l) / np.maximum(rmsb_b_l, 1e-12) * 100
                    ch_entry.update(
                        rmsb_baseline=rmsb_b_l.tolist(),
                        rmsb_full=rmsb_f_l.tolist(),
                        improvement_pct_rmsb=(-rel_rmsb_l).tolist(),
                    )
                out["per_channel_per_step"][key] = ch_entry
        else:
            out["per_channel_per_step"][var] = out["per_variable_per_step"][var]

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"\n[clean-eval] wrote {cfg.out_json}")


if __name__ == "__main__":
    main()
