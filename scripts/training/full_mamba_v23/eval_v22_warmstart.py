"""v22 warmstart eval — matches Ilya's `unified_resolution_eval.py` warm mode.

Two changes vs eval_v20_rollout.py (cold eval):

1. WARMUP PHASE: before measuring metrics, run `warmup_steps` (default 24) of
   truth-feedback steps. Mamba SSM state evolves through real ERA5 inputs
   instead of starting at zero — matches the training-time state distribution
   where state has been processed through many BPTT chunks within a segment.

2. AR FEEDBACK = FULL_PRED, NOT BASELINE. v22 training uses
   autoregressive.Predictor on the corrected predictor (GCResidualWithZeroHead
   returns baseline + residual), so during training AR feedback is full_pred.
   Cold eval (v20) used baseline-only feedback (v17/v18 legacy). We switch
   here to match training, matching Ilya's `truth_step` / `branch_rollout`.

Output JSON format identical to eval_v20_rollout.py for direct comparison.
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
    p.add_argument("--target-steps", type=int, default=40, help="metric horizon (K)")
    p.add_argument("--warmup-steps", type=int, default=24,
                   help="truth-feedback warmup steps before metric horizon (Ilya: 24 = 144h)")
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
    print(f"[warmstart-eval] ckpt: {ckpt_path}  K={cfg.target_steps}  warmup={cfg.warmup_steps}")

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

    # Total horizon = warmup + metric
    total_steps = cfg.warmup_steps + cfg.target_steps

    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, total_steps)
    print(f"[warmstart-eval] val_indices (need {total_steps} future steps): {len(val_indices)}")

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

    _residual_params_init, residual_state_init = residual_predict.init(
        k_r, sample_inputs, sample_targets_1step, sample_forcings_1step)

    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    if "residual_state" in ckpt and ckpt["residual_state"]:
        residual_state_init = ckpt["residual_state"]
    n_p = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[warmstart-eval] loaded residual_params {n_p:,}")

    @jax.jit
    def _baseline_step(params, state, key, inp, tgt, frc):
        out, new_state = baseline_predict.apply(params, state, key, inp, tgt, frc)
        return out, new_state
    @jax.jit
    def _residual_step(params, state, key, inp, tgt, frc):
        out, new_state = residual_predict.apply(params, state, key, inp, tgt, frc)
        return out, new_state

    def _shift_inputs_with_field(prev_inputs, new_field_ds, forcings_next):
        """Shift inputs window by 1 step. new_field_ds is a 1-step Dataset of
        atmospheric state variables (either truth or prediction)."""
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

    def warmstart_rollout(rng, inputs, all_targets, all_forcings, debug=False):
        """Two-phase rollout:
          Phase 1 (warmup_steps): truth-feedback, state evolves, no metric.
          Phase 2 (target_steps): full_pred-feedback (matches v22 training).
        Returns (baseline_traj [time=target_steps], full_traj [time=target_steps]).
        """
        W = cfg.warmup_steps
        K = cfg.target_steps
        current_inp = inputs
        bs = baseline_state_init
        rs = residual_state_init

        # ---- PHASE 1: WARMUP with truth feedback ----
        for k in range(W):
            tgt_k = all_targets.isel(time=slice(k, k + 1))
            frc_k = all_forcings.isel(time=slice(k, k + 1))
            rng, rng_b, rng_r = jax.random.split(rng, 3)
            _bp, bs = _baseline_step(baseline_params, bs, rng_b, current_inp, tgt_k, frc_k)
            _rp, rs = _residual_step(residual_params, rs, rng_r, current_inp, tgt_k, frc_k)
            # Shift: use TRUTH (tgt_k) as the next input, not predictions.
            current_inp = _shift_inputs_with_field(current_inp, tgt_k, frc_k)
            if debug and k < 3:
                print(f"  warmup k={k}: state shapes — residual leaves: "
                      f"{[l.shape for l in jax.tree_util.tree_leaves(rs)][:3]}")

        # ---- PHASE 2: METRIC ROLLOUT with full_pred feedback ----
        baseline_chunks = []
        full_chunks = []
        for k in range(K):
            kk = W + k
            tgt_k = all_targets.isel(time=slice(kk, kk + 1))
            frc_k = all_forcings.isel(time=slice(kk, kk + 1))
            rng, rng_b, rng_r = jax.random.split(rng, 3)
            bp, bs = _baseline_step(baseline_params, bs, rng_b, current_inp, tgt_k, frc_k)
            rp, rs = _residual_step(residual_params, rs, rng_r, current_inp, tgt_k, frc_k)
            full_pred = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp)
            baseline_chunks.append(bp)
            full_chunks.append(full_pred)
            if k < K - 1:
                # Feedback: FULL_PRED (matches v22 training autoregressive predictor)
                current_inp = _shift_inputs_with_field(current_inp, full_pred, frc_k)

        baseline_pred = xr.concat(baseline_chunks, dim="time")
        full_pred_traj = xr.concat(full_chunks, dim="time")
        return baseline_pred, full_pred_traj

    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = sorted(rng_np.choice(val_indices, size=cfg.n_samples,
                                       replace=False).tolist())

    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_da = xr.DataArray(cos_lat / cos_lat.mean(), dims="lat")

    K = cfg.target_steps
    sum_sq_b, sum_sq_f, n_per_var = {}, {}, {}
    sum_abs_b, sum_abs_f = {}, {}
    sum_sq_b_pl, sum_sq_f_pl, n_per_pl = {}, {}, {}
    sum_abs_b_pl, sum_abs_f_pl = {}, {}

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=total_steps, task_cfg=task_cfg, dt=dt)
        rng, rng_chain = jax.random.split(rng)
        baseline_pred, full_pred = warmstart_rollout(
            rng_chain, inp, tgt, frc, debug=(s_i == 0))

        # Metric is computed on the EVAL phase only (steps W..W+K-1 of tgt).
        tgt_eval = tgt.isel(time=slice(cfg.warmup_steps, cfg.warmup_steps + K))

        for var in tgt_eval.data_vars:
            truth = tgt_eval[var].astype("float32")
            bp = baseline_pred[var].astype("float32")
            fp = full_pred[var].astype("float32")
            if var not in sum_sq_b:
                sum_sq_b[var] = np.zeros(K); sum_sq_f[var] = np.zeros(K)
                sum_abs_b[var] = np.zeros(K); sum_abs_f[var] = np.zeros(K)
                n_per_var[var] = 0
            for k in range(K):
                err_b_k = bp.isel(time=k) - truth.isel(time=k)
                err_f_k = fp.isel(time=k) - truth.isel(time=k)
                sum_sq_b[var][k] += float(((err_b_k ** 2) * cos_lat_da).mean().values)
                sum_sq_f[var][k] += float(((err_f_k ** 2) * cos_lat_da).mean().values)
                sum_abs_b[var][k] += float((np.abs(err_b_k) * cos_lat_da).mean().values)
                sum_abs_f[var][k] += float((np.abs(err_f_k) * cos_lat_da).mean().values)
            n_per_var[var] += 1

            if "level" in truth.dims:
                if var not in sum_sq_b_pl:
                    sum_sq_b_pl[var] = {}; sum_sq_f_pl[var] = {}
                    sum_abs_b_pl[var] = {}; sum_abs_f_pl[var] = {}
                    n_per_pl[var] = {}
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
                    n_per_pl[var][lev] += 1

        if s_i < 3 or (s_i + 1) % 4 == 0:
            print(f"[warmstart-eval] sample {s_i+1}/{cfg.n_samples}", flush=True)

    print()
    print(f"=== Per-step lat-weighted RMSE & MAE (K={K}, warmup={cfg.warmup_steps}) ===")
    out = {
        "target_steps": K,
        "warmup_steps": cfg.warmup_steps,
        "n_samples": cfg.n_samples,
        "ckpt": str(ckpt_path),
        "per_variable_per_step": {},
        "per_channel_per_step": {},
    }
    for var in sorted(n_per_var):
        n = n_per_var[var]
        r_b = np.sqrt(sum_sq_b[var] / n); r_f = np.sqrt(sum_sq_f[var] / n)
        a_b = sum_abs_b[var] / n; a_f = sum_abs_f[var] / n
        rel_rmse = (r_f - r_b) / np.maximum(r_b, 1e-12) * 100
        rel_mae = (a_f - a_b) / np.maximum(a_b, 1e-12) * 100
        out["per_variable_per_step"][var] = dict(
            rmse_baseline=r_b.tolist(), rmse_full=r_f.tolist(),
            mae_baseline=a_b.tolist(),  mae_full=a_f.tolist(),
            improvement_pct=(-rel_rmse).tolist(),
            improvement_pct_rmse=(-rel_rmse).tolist(),
            improvement_pct_mae=(-rel_mae).tolist(),
        )
        print(f"  {var:<30}", "RMSE", " ".join(f"K{k+1}{-rel_rmse[k]:+.2f}%" for k in range(K)))
        if var in sum_sq_b_pl:
            for lev in sorted(sum_sq_b_pl[var]):
                nl = n_per_pl[var][lev]
                r_b_l = np.sqrt(sum_sq_b_pl[var][lev] / nl); r_f_l = np.sqrt(sum_sq_f_pl[var][lev] / nl)
                a_b_l = sum_abs_b_pl[var][lev] / nl; a_f_l = sum_abs_f_pl[var][lev] / nl
                imp_rmse = -((r_f_l - r_b_l) / np.maximum(r_b_l, 1e-12) * 100)
                imp_mae  = -((a_f_l - a_b_l) / np.maximum(a_b_l, 1e-12) * 100)
                key = f"{var}_level{lev}"
                out["per_channel_per_step"][key] = dict(
                    rmse_baseline=r_b_l.tolist(), rmse_full=r_f_l.tolist(),
                    mae_baseline=a_b_l.tolist(),  mae_full=a_f_l.tolist(),
                    improvement_pct=imp_rmse.tolist(),
                    improvement_pct_rmse=imp_rmse.tolist(),
                    improvement_pct_mae=imp_mae.tolist(),
                )
        else:
            out["per_channel_per_step"][var] = out["per_variable_per_step"][var]

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"\n[warmstart-eval] wrote {cfg.out_json}")


if __name__ == "__main__":
    main()
