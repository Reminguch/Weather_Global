"""v22 ablation eval — isolate causal role of Mamba memory in K=18 gain.

4 ablation modes (cold_bp, baseline-feedback only):
  none                  : standard cold_bp rollout, h_t evolves freely
  reset_ssm_every_step  : zero out ssm_state at the START of every rollout step (keep conv_cache)
  reset_all_every_step  : zero out BOTH ssm_state and conv_cache at every step (kill all temporal mem)
  stale_after_warmup    : run W truth-fed warmup → h_W ; then FREEZE h = h_W for all rollout steps

Optional: --warmup-steps W to first warm up h before the eval phase (used by `stale_after_warmup`
and any mode where we want a non-zero starting state). If W=0, init state = zero.

Per-step diagnostics dumped:
  - RMSE / MAE / improvement_pct by lead, per variable (same as eval_v22_clean.py)
  - residual_cosine_by_lead[var]         : cos(r_t, -err_t^bp) in normalized var space, lat-weighted
  - residual_gain_by_lead[var]           : ||r_t|| / ||err_t^bp|| (lat-weighted L2 ratio per var)
  - state_norm_by_lead[layer_key]        : ||h_t||_2 per layer (averaged across samples)
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
    GCResidualWithZeroHead, _attach_temporal,
)

ABLATION_MODES = ["none", "reset_ssm_every_step", "reset_all_every_step", "stale_after_warmup"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-in", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--out-json", required=True)
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
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="W truth-fed warmup steps before eval phase. For stale_after_warmup, "
                        "h is frozen at h_W during eval. For other modes, h starts from h_W.")
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
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablation-mode", choices=ABLATION_MODES, required=True)
    return p.parse_args()


def _zero_state(rs, *, zero_ssm=True, zero_conv=True):
    """Return a copy of rs with ssm_state / conv_cache zeroed selectively."""
    out = {}
    for top_k, top_v in rs.items():
        new_top = {}
        for k, v in top_v.items():
            if zero_ssm and "ssm_state" in k:
                new_top[k] = jnp.zeros_like(v)
            elif zero_conv and "conv_cache" in k:
                new_top[k] = jnp.zeros_like(v)
            else:
                new_top[k] = v
        out[top_k] = new_top
    return out


def _state_norms(rs):
    """L2 norm per leaf in rs.  Returns dict {leaf_path: float}."""
    out = {}
    for top_k, top_v in rs.items():
        for k, v in top_v.items():
            out[k] = float(np.linalg.norm(np.asarray(v)))
    return out


def main():
    cfg = parse_args()
    K = cfg.target_steps
    W = cfg.warmup_steps
    mode = cfg.ablation_mode
    print(f"[ablate] ckpt: {cfg.ckpt}")
    print(f"[ablate] mode={mode}  W={W}  K={K}  n_samples={cfg.n_samples}")

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
    # CRITICAL: keep anchor pool identical across all modes for fair comparison.
    # Match eval_v22_clean.py which uses W_max + K with W_max = 24 by default.
    sample_total_steps = max(W, 24) + K

    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, sample_total_steps)
    print(f"[ablate] {len(val_indices)} valid anchors "
          f"(need {sample_total_steps} future steps; per-rollout {total_steps})")

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

    with open(cfg.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    _hk_zero_state = residual_state_init  # TRUE zero hk.init state — use for everything

    @jax.jit
    def _baseline_step(params, state, key, inp, tgt, frc):
        return baseline_predict.apply(params, state, key, inp, tgt, frc)
    @jax.jit
    def _residual_step(params, state, key, inp, tgt, frc):
        return residual_predict.apply(params, state, key, inp, tgt, frc)

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

    # Cosine/gain computed per variable in PHYSICAL units (lat-weighted).
    # cosine is invariant under positive scalar normalization → no need to standardize.
    # gain |r|/|e| is also invariant, since both share the same units.
    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_norm = cos_lat / cos_lat.mean()
    cos_lat_da = xr.DataArray(cos_lat_norm, dims="lat")

    def clean_rollout(rng, inputs, all_targets, all_forcings):
        """cold_bp rollout (single trajectory follows baseline) with optional ablation."""
        cur = inputs
        bs = baseline_state_init
        rs = _hk_zero_state    # always start from TRUE zero
        rs_frozen = None       # populated only for stale_after_warmup

        # PHASE 1: WARMUP (truth-fed) — used to drive h to a non-zero state when W > 0.
        for k in range(W):
            tgt_k = all_targets.isel(time=slice(k, k + 1))
            frc_k = all_forcings.isel(time=slice(k, k + 1))
            rng, kk_b, kk_r = jax.random.split(rng, 3)
            _, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
            _, rs = _residual_step(residual_params, rs, kk_r, cur, tgt_k, frc_k)
            cur = _shift_inputs_with_field(cur, tgt_k, frc_k)

        if mode == "stale_after_warmup":
            rs_frozen = rs   # snapshot post-warmup state, will reuse every step

        # PHASE 2: K eval steps — baseline-feedback (cold_bp).
        baseline_chunks = []
        full_chunks = []
        residual_chunks = []
        # Track state norms per step (after each residual_step). state_norm_traj[k]
        # = dict of leaf -> sample-mean L2 norm at the END of step k.
        state_norm_traj = []

        for k in range(K):
            tgt_k = all_targets.isel(time=slice(W + k, W + k + 1))
            frc_k = all_forcings.isel(time=slice(W + k, W + k + 1))
            rng, kk_base, kk_res = jax.random.split(rng, 3)
            bp, bs = _baseline_step(baseline_params, bs, kk_base, cur, tgt_k, frc_k)

            # === Ablation pre-step state manipulation ===
            if mode == "reset_ssm_every_step":
                rs_in = _zero_state(rs, zero_ssm=True, zero_conv=False)
            elif mode == "reset_all_every_step":
                rs_in = _zero_state(rs, zero_ssm=True, zero_conv=True)
            elif mode == "stale_after_warmup":
                rs_in = rs_frozen
            else:  # none
                rs_in = rs

            rp, rs_new = _residual_step(residual_params, rs_in, kk_res, cur, tgt_k, frc_k)

            # === State propagation rule ===
            if mode == "stale_after_warmup":
                pass    # rs unchanged (still rs_frozen for next step's rs_in)
            elif mode == "reset_ssm_every_step":
                # Carry forward the updated state EXCEPT we just reset ssm before applying.
                # The residual_step has produced a new state from (rs with zeroed ssm). Keeping
                # this new state means conv_cache evolves with zeroed-ssm history. To strictly
                # kill SSM memory ACROSS steps, also overwrite rs to the just-reset+updated form.
                rs = rs_new
            elif mode == "reset_all_every_step":
                rs = rs_new   # but next step we'll reset everything again anyway
            else:
                rs = rs_new

            full_pred = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp)
            baseline_chunks.append(bp)
            full_chunks.append(full_pred)
            residual_chunks.append(rp)
            state_norm_traj.append(_state_norms(rs_new))    # log the just-updated rs

            if k < K - 1:
                cur = _shift_inputs_with_field(cur, bp, frc_k)

        baseline_pred = xr.concat(baseline_chunks, dim="time")
        full_pred_traj = xr.concat(full_chunks, dim="time")
        residual_pred = xr.concat(residual_chunks, dim="time")
        return baseline_pred, full_pred_traj, residual_pred, state_norm_traj

    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = sorted(rng_np.choice(val_indices, size=cfg.n_samples,
                                       replace=False).tolist())

    sum_sq_b, sum_sq_f, n_per_var = {}, {}, {}
    sum_abs_b, sum_abs_f = {}, {}
    # NEW: residual cosine + gain accumulators (per variable, per lead)
    sum_dot_re = {}     # <r, -err_b> per var, per lead
    sum_norm_r = {}     # ||r||^2 per var, per lead
    sum_norm_e = {}     # ||err_b||^2 per var, per lead
    sum_normr_l = {}    # ||r|| per var, per lead (mean of ratios across samples)
    sum_norme_l = {}    # ||err_b|| per var, per lead
    sum_cos    = {}     # cos(r, -err_b) per var, per lead — direct per-sample mean

    # State norms: accumulate sum across samples
    state_norm_acc = None
    state_keys = None

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=total_steps, task_cfg=task_cfg, dt=dt)
        rng, rng_chain = jax.random.split(rng)
        baseline_pred, full_pred, residual_pred, sn_traj = clean_rollout(
            rng_chain, inp, tgt, frc)

        # State-norm accumulate
        if state_norm_acc is None:
            state_keys = list(sn_traj[0].keys())
            state_norm_acc = {k: np.zeros(K) for k in state_keys}
        for k_step in range(K):
            for sk in state_keys:
                state_norm_acc[sk][k_step] += sn_traj[k_step][sk]

        # METRIC: same as eval_v22_clean, on phase-2 eval steps only.
        tgt_eval = tgt.isel(time=slice(W, W + K))

        for var in tgt_eval.data_vars:
            truth = tgt_eval[var].astype("float32")
            bp = baseline_pred[var].astype("float32")
            fp = full_pred[var].astype("float32")
            rp = residual_pred[var].astype("float32")

            err_b = bp - truth      # baseline error
            # residual is "correction towards truth" = -(err_b) ideally
            target_corr = -err_b    # cos & gain reference: signed error to correct
            # No normalization — cosine and gain are scale-invariant.
            rp_n  = rp
            tc_n  = target_corr

            if var not in sum_sq_b:
                sum_sq_b[var] = np.zeros(K); sum_sq_f[var] = np.zeros(K)
                sum_abs_b[var] = np.zeros(K); sum_abs_f[var] = np.zeros(K)
                n_per_var[var] = 0
                sum_dot_re[var]   = np.zeros(K)
                sum_norm_r[var]   = np.zeros(K)
                sum_norm_e[var]   = np.zeros(K)
                sum_normr_l[var]  = np.zeros(K)
                sum_norme_l[var]  = np.zeros(K)
                sum_cos[var]      = np.zeros(K)

            for k in range(K):
                err_b_k = bp.isel(time=k) - truth.isel(time=k)
                err_f_k = fp.isel(time=k) - truth.isel(time=k)
                sum_sq_b[var][k] += float(((err_b_k ** 2) * cos_lat_da).mean().values)
                sum_sq_f[var][k] += float(((err_f_k ** 2) * cos_lat_da).mean().values)
                sum_abs_b[var][k] += float((np.abs(err_b_k) * cos_lat_da).mean().values)
                sum_abs_f[var][k] += float((np.abs(err_f_k) * cos_lat_da).mean().values)

                # cosine & gain in normalized space, lat-weighted
                rn_k = rp_n.isel(time=k)
                tn_k = tc_n.isel(time=k)
                spatial_dims = [d for d in rn_k.dims if d not in ("batch",)]
                dot = float(((rn_k * tn_k) * cos_lat_da).mean(dim=spatial_dims).values.mean())
                nr  = float(((rn_k * rn_k) * cos_lat_da).mean(dim=spatial_dims).values.mean())
                ne  = float(((tn_k * tn_k) * cos_lat_da).mean(dim=spatial_dims).values.mean())
                sum_dot_re[var][k]  += dot
                sum_norm_r[var][k]  += nr
                sum_norm_e[var][k]  += ne
                nr_l = np.sqrt(max(nr, 0.0))
                ne_l = np.sqrt(max(ne, 0.0))
                sum_normr_l[var][k] += nr_l
                sum_norme_l[var][k] += ne_l
                cos_k = dot / max(nr_l * ne_l, 1e-12)
                sum_cos[var][k]     += cos_k
            n_per_var[var] += 1

        if s_i < 3 or (s_i + 1) % 4 == 0:
            print(f"[ablate] sample {s_i+1}/{cfg.n_samples}", flush=True)

    # Reduce
    out = {
        "ablation_mode": mode,
        "warmup_steps": W,
        "target_steps": K,
        "n_samples": cfg.n_samples,
        "chosen_idx": chosen_idx,
        "ckpt": cfg.ckpt,
        "per_variable_per_step": {},
        "residual_diagnostics_per_variable": {},
        "state_norm_per_layer": {},
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
            improvement_pct_rmse=(-rel_rmse).tolist(),
            improvement_pct_mae=(-rel_mae).tolist(),
        )
        # Residual diagnostics (sample-average each accumulator)
        cos_avg   = sum_cos[var] / n
        gain_avg  = sum_normr_l[var] / np.maximum(sum_norme_l[var], 1e-12)
        rnorm_avg = sum_normr_l[var] / n
        enorm_avg = sum_norme_l[var] / n
        out["residual_diagnostics_per_variable"][var] = dict(
            residual_cosine_by_lead=cos_avg.tolist(),
            residual_gain_by_lead=gain_avg.tolist(),
            residual_norm_by_lead=rnorm_avg.tolist(),
            baseline_err_norm_by_lead=enorm_avg.tolist(),
        )

    if state_norm_acc is not None:
        n_total = cfg.n_samples
        for sk in state_keys:
            out["state_norm_per_layer"][sk] = (state_norm_acc[sk] / n_total).tolist()

    # Console echo
    print()
    print(f"=== Summary (mode={mode}, W={W}, K={K}) ===")
    for var in ["2m_temperature", "10m_u_component_of_wind"]:
        if var in out["per_variable_per_step"]:
            imps = out["per_variable_per_step"][var]["improvement_pct_rmse"]
            coss = out["residual_diagnostics_per_variable"][var]["residual_cosine_by_lead"]
            gains = out["residual_diagnostics_per_variable"][var]["residual_gain_by_lead"]
            print(f"  {var}:")
            for ki in [0, 4, 9, 19, 29, 39]:
                if ki < K:
                    print(f"    L{ki+1:>2d}: imp={imps[ki]:+6.2f}%  cos={coss[ki]:+.3f}  gain={gains[ki]:.3f}")

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"[ablate] wrote {cfg.out_json}")


if __name__ == "__main__":
    main()
