"""Anchor-based rollout eval for residual_mamba checkpoints.

Supports BOTH cold-eval (open-loop, --feedback-mode baseline) and warm-up eval
(closed-loop, --feedback-mode gated) protocols.

Trajectories (per anchor):
  baseline:  pure GCv1 K=3 self-rollout (frozen GraphCast).
  resmamba:  residual_mamba predictor that internally produces
             full_pred = baseline + residual_correction (residual_output_head=True
             lives INSIDE the GraphCast predictor; InputsAndResiduals returns the
             full forecast directly). The "residual" injected into next inputs is
             computed manually as (rp_pred - bp_pred) for the closed-loop ('gated')
             mode; for open-loop ('baseline') mode we feed bp_pred into cur_r.

Warmup (W steps): truth-fed inputs for BOTH models (state evolves with truth).
Rollout (K steps): per step, both models forward; baseline rolls itself; residual
rolls per --feedback-mode.

Output: JSON with anchor_times + per_variable_per_lead lat-w RMSE / RMSB and
improvement_pct of resmamba over baseline.

Based on scripts/training/gcmamba_v2_diagnostics/eval_K1_3way.py — but only two
trajectories (baseline + resmamba) and uses build_predictor (with
residual_output_head=True) for the residual model, not GCResidualWithZeroHead.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import haiku as hk
import jax
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization, xarray_jax  # noqa: E402
import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    advance_residual_inputs,
    build_predictor,
    build_residual_correction_predictor,
    build_zero_residual_inputs,
    derive_model_config_from_checkpoint,
    load_graphcast_checkpoint,
    load_stats,
)
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from src.models.mamba.residual_mamba.feedback import (  # noqa: E402
    make_feedback_var_stddev_dict,
    FEEDBACK_CLIP_MODE_NONE,
    FEEDBACK_CLIP_MODE_TANH,
    FEEDBACK_CLIP_MODE_HARD,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-ckpt", required=True,
                   help="Frozen GCv1 K=3 ckpt (npz produced by graphcast checkpoint loader).")
    p.add_argument("--residual-ckpt", required=True,
                   help="residual_mamba step-N .npz checkpoint to eval.")
    p.add_argument("--data-path",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--mesh-size", type=int, default=4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-mp", type=int, default=6)
    p.add_argument("--residual-mp", type=int, default=2)
    p.add_argument("--n-anchors", type=int, default=240)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Truth-fed warmup steps (24 for closed-loop ckpts, 0 for open-loop).")
    p.add_argument("--target-steps", type=int, default=40,
                   help="Number of rollout lead steps to eval per anchor.")
    p.add_argument("--feedback-mode", choices=["baseline", "gated"], default="baseline",
                   help="'baseline' = open-loop (feed bp_pred); 'gated' = closed-loop.")
    p.add_argument("--feedback-lambda", type=float, default=1.0)
    p.add_argument("--feedback-clip-mode", choices=["none", "tanh", "hard"], default="none")
    p.add_argument("--feedback-clip-c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def _flat_npz_to_haiku(npz_path: Path) -> dict:
    """Convert flat 'params:<module>:<leaf>' npz keys back to nested haiku dict."""
    data = np.load(npz_path, allow_pickle=True)
    out: dict = {}
    for key in data.keys():
        if not key.startswith("params:"):
            continue
        rest = key[len("params:"):]
        if ":" not in rest:
            continue
        mod, leaf = rest.rsplit(":", 1)
        out.setdefault(mod, {})[leaf] = np.asarray(data[key])
    return out


def main():
    cfg = parse_args()
    W = cfg.warmup_steps
    K = cfg.target_steps

    # --- Load baseline ckpt for task/model config + frozen params ---
    ckpt_baseline = load_graphcast_checkpoint(Path(cfg.baseline_ckpt))
    task_cfg = dataclasses.replace(ckpt_baseline.task_config, input_duration=cfg.input_duration)
    norm_stats = load_stats(Path(cfg.stats_dir))

    mc_baseline = derive_model_config_from_checkpoint(
        ckpt_baseline.model_config,
        resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.baseline_mp, hidden_layers=1)
    mc_residual = derive_model_config_from_checkpoint(
        ckpt_baseline.model_config,
        resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.residual_mp, hidden_layers=1)

    use_bf16 = True

    # --- 1) baseline predictor: vanilla GC, no Mamba, no residual head ---
    def baseline_fn(inputs, targets, forcings):
        p = build_predictor(
            mc_baseline, task_cfg, norm_stats,
            use_bf16=use_bf16, gradient_checkpointing=False,
            temporal_backbone='none',
            temporal_location='mesh_post_encoder',
            temporal_d_inner=None,
            temporal_d_state=16,
            temporal_d_conv=4,
            temporal_dt_rank='auto',
            temporal_bias=False,
            temporal_conv_bias=True,
            temporal_layers=1,
            temporal_dropout=0.0,
            temporal_stateful=False,
            temporal_insert_count=None,
            zero_init_temporal_out=False,
            residual_output_head=False,
            autoregressive_loss_mode='none',
            memory_mode='standard',
        )
        return p(inputs, targets_template=targets, forcings=forcings)
    baseline_predict = hk.transform_with_state(baseline_fn)

    # --- 2) residual_mamba predictor: full-Mamba interleaved + residual head ---
    # CRITICAL: V-A / V-Bfix / K-curriculum ckpts are trained with
    # `build_residual_correction_predictor` which wraps the model with
    # DirectResidualNormalizer — the predictor's output is the RAW residual in
    # physical units (NOT the full forecast). Full forecast must be computed
    # manually as bp_pred + rp_pred.
    def residual_fn(inputs, targets, forcings):
        p = build_residual_correction_predictor(
            mc_residual, task_cfg, norm_stats,
            use_bf16=use_bf16, gradient_checkpointing=False,
            temporal_backbone='mamba',
            temporal_location='mesh_processor_interleaved',
            temporal_d_inner=128,
            temporal_d_state=16,
            temporal_d_conv=4,
            temporal_dt_rank='auto',
            temporal_bias=False,
            temporal_conv_bias=True,
            temporal_layers=1,
            temporal_dropout=0.0,
            temporal_stateful=True,
            temporal_insert_count=2,
            zero_init_temporal_out=True,
            residual_output_head=True,
            autoregressive_loss_mode='none',
            memory_mode='standard',
        )
        return p(inputs, targets_template=targets, forcings=forcings)
    residual_predict = hk.transform_with_state(residual_fn)

    # --- Data ---
    class _SplitCfg:
        data_path = cfg.data_path
        resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = 2015
        train_end_year = 2021
    _, eval_ds = base_train._open_local_splits(_SplitCfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    valid_idx = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, W + K)
    rng_np = np.random.default_rng(cfg.seed)
    anchor_indices = sorted(
        rng_np.choice(valid_idx, size=cfg.n_anchors, replace=False).tolist())

    # --- Init params ---
    sample_inp, sample_tgt, sample_frc = base_train.build_batch_from_indices(
        eval_ds, indices=[int(anchor_indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
    # DirectResidualNormalizer expects residual_inputs (prognostic vars are
    # residual values from previous steps, NOT absolute state). Zero-init at
    # anchor time, advance with predicted/truth residuals during rollout.
    sample_residual_inp = build_zero_residual_inputs(sample_inp, sample_tgt)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, k1, k2 = jax.random.split(rng, 3)

    bp_params, bp_state0 = baseline_predict.init(k1, sample_inp, sample_tgt, sample_frc)
    bp_params, bp_ov = overlay_matching_params(bp_params, ckpt_baseline.params, strict=True)
    n_b_loaded = sum(len(v) for v in ckpt_baseline.params.values())
    print(f"baseline params: overlaid {bp_ov.copied} / {n_b_loaded} GC leaves from {cfg.baseline_ckpt}")

    # residual_predict.init must use the residual-input dataset shape so the
    # network's input embedding matches the trained ckpt.
    rp_params, rp_state0 = residual_predict.init(k2, sample_residual_inp, sample_tgt, sample_frc)
    rp_trained = _flat_npz_to_haiku(Path(cfg.residual_ckpt))
    n_r_loaded = sum(len(v) for v in rp_trained.values())
    rp_params, rp_ov = overlay_matching_params(rp_params, rp_trained, strict=False)
    print(f"residual params: overlaid {rp_ov.copied} / {n_r_loaded} leaves from {cfg.residual_ckpt}")

    @jax.jit
    def _bp_step(p, s, key, inp, tgt, frc):
        return baseline_predict.apply(p, s, key, inp, tgt, frc)

    @jax.jit
    def _rp_step(p, s, key, inp, tgt, frc):
        return residual_predict.apply(p, s, key, inp, tgt, frc)

    # --- Var-stddev dict for normalized feedback clipping (gated mode only) ---
    var_stddevs = make_feedback_var_stddev_dict(
        norm_stats["stddev_by_level"],
        pressure_levels=tuple(task_cfg.pressure_levels),
    )

    def _to_numpy_dataset(ds):
        """Convert any JaxArrayWrapper inside an xarray Dataset to plain numpy."""
        new_vars = {}
        for var in ds.data_vars:
            da = ds[var]
            d = da.data
            if hasattr(d, '_data'):
                d = d._data
            arr_np = np.asarray(jax.device_get(d))
            new_vars[var] = (da.dims, arr_np)
        new_coords = {}
        for c in ds.coords:
            cv = ds[c].values
            if hasattr(cv, '_data'):
                cv = cv._data
            new_coords[c] = np.asarray(cv)
        return xr.Dataset(new_vars, coords=new_coords)

    def _shift_inputs(prev_inputs, new_field_ds, forcings_next):
        new_field_np = _to_numpy_dataset(new_field_ds)
        target_time = prev_inputs.time.values[-1:] + dt
        ns = new_field_np.assign_coords(time=target_time)
        fn = forcings_next.assign_coords(time=target_time)
        next_frame = xr.merge([ns, fn])
        if "datetime" in next_frame.coords:
            next_frame = next_frame.drop_vars("datetime")
        keys_in_next = [k for k in next_frame.data_vars if k in prev_inputs.data_vars]
        next_inputs_part = next_frame[keys_in_next]
        merged = xr.concat([prev_inputs, next_inputs_part], dim="time", data_vars="different")
        return merged.tail(time=input_steps)

    def _clip_per_var(arr: np.ndarray, var: str, mode: str, c: float) -> np.ndarray:
        """Per-variable normalized clipping (numpy version, applied in physical units)."""
        if mode == FEEDBACK_CLIP_MODE_NONE:
            return arr
        std = var_stddevs.get(var)
        if std is None:
            return arr
        std_np = np.asarray(std)
        # std shape can be scalar or (1,1,L,1,1) — must broadcast to arr shape.
        # arr at this point is shaped like the prediction Dataset's data_var:
        # (batch, time, [level,] lat, lon). We squeeze std appropriately.
        if std_np.ndim == 0:
            s = float(std_np)
            arr_norm = arr / s
            if mode == FEEDBACK_CLIP_MODE_TANH:
                clipped_norm = c * np.tanh(arr_norm / c)
            else:  # HARD
                clipped_norm = np.clip(arr_norm, -c, c)
            return clipped_norm * s
        # Level-dependent: reshape (1,1,L,1,1) to broadcast with arr's dims.
        # arr may have shape (B,T,L,Lat,Lon) for level vars.
        try:
            arr_norm = arr / std_np
        except ValueError:
            # If shapes don't match, fall back to skipping clip for this var.
            return arr
        if mode == FEEDBACK_CLIP_MODE_TANH:
            clipped_norm = c * np.tanh(arr_norm / c)
        else:
            clipped_norm = np.clip(arr_norm, -c, c)
        return clipped_norm * std_np

    # --- Lat-weighting ---
    lat_vals = eval_ds.lat.values
    cos_lat = np.cos(np.deg2rad(lat_vals))
    cos_lat_n = cos_lat / cos_lat.mean()
    target_vars = list(task_cfg.target_variables)

    # Accumulators: sum-of-squares (for RMSE) and sum-of-biases (for RMSB).
    sum_sq = {m: {v: np.zeros(K, dtype=np.float64) for v in target_vars}
              for m in ['baseline', 'resmamba']}
    sum_bias = {m: {v: np.zeros(K, dtype=np.float64) for v in target_vars}
                for m in ['baseline', 'resmamba']}
    anchor_times = []

    # First-anchor diagnostic for explicit time-indexing audit (pre-anchor warmup).
    _diag_idx = anchor_indices[0]
    _diag_warm_start_t = eval_ds.time.values[_diag_idx]
    _diag_anchor_t = eval_ds.time.values[_diag_idx + W]
    _diag_lead1_t = eval_ds.time.values[_diag_idx + W + 1]
    _diag_lead40_t = eval_ds.time.values[_diag_idx + W + K]
    print(f"[indexing] anchor[0]: warm_start={_diag_warm_start_t}, "
          f"forecast_anchor={_diag_anchor_t} (=warm_start+{W*6}h), "
          f"lead 1={_diag_lead1_t}, lead {K}={_diag_lead40_t}")

    for ai, idx in enumerate(anchor_indices):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=W + K, task_cfg=task_cfg, dt=dt)
        # Forecast anchor (= post-warmup input window's final time) is at idx+W.
        # Record THAT as anchor_time so consumers see the forecast start, not the
        # warmup start (which is W=24 steps earlier for closed-loop warm-up eval).
        anchor_time_idx = idx + W if W > 0 else idx
        anchor_times.append(str(pd.Timestamp(eval_ds.time.values[anchor_time_idx])))

        cur_b = inp
        cur_r = inp
        # Residual_inputs_r: prognostic vars start at zero (residual = 0 before any
        # step) + constant vars from inp. Advanced per-step with either truth_residual
        # (warmup) or predicted residual (rollout).
        residual_inputs_r = build_zero_residual_inputs(inp, tgt.isel(time=slice(0, 1)))
        sb = bp_state0
        sr = rp_state0
        sb_r = bp_state0  # separate baseline state for the "compute residual" branch
        rng_l = jax.random.PRNGKey(cfg.seed + 99 + ai)

        # Phase 1: warmup — truth-fed; state evolves but no error recorded.
        # cur_b / cur_r shift with truth. residual_inputs_r advances with truth
        # residual = truth - bp_pred_r (teacher-forced residual signal).
        for k in range(W):
            tgt_k = tgt.isel(time=slice(k, k + 1))
            frc_k = frc.isel(time=slice(k, k + 1))
            rng_l, *keys = jax.random.split(rng_l, 4)
            _b, sb = _bp_step(bp_params, sb, keys[0], cur_b, tgt_k, frc_k)
            bp_for_r, sb_r = _bp_step(bp_params, sb_r, keys[1], cur_r, tgt_k, frc_k)
            _r, sr = _rp_step(rp_params, sr, keys[2], residual_inputs_r, tgt_k, frc_k)
            # truth_residual = truth - bp_for_r (both physical units).
            bp_for_r_np = _to_numpy_dataset(bp_for_r)
            truth_residual_vars = {}
            for vname, da in bp_for_r_np.data_vars.items():
                if vname in tgt_k.data_vars:
                    truth_arr = np.asarray(tgt_k[vname].values)
                    bp_arr = np.asarray(da.values)
                    truth_residual_vars[vname] = (da.dims, truth_arr - bp_arr)
            truth_residual = xr.Dataset(truth_residual_vars, coords=bp_for_r_np.coords)
            residual_inputs_r = advance_residual_inputs(residual_inputs_r, truth_residual)
            cur_b = _shift_inputs(cur_b, tgt_k, frc_k)
            cur_r = _shift_inputs(cur_r, tgt_k, frc_k)

        # Phase 2: rollout — record per-step error and advance per --feedback-mode.
        for k in range(K):
            kk = W + k
            tgt_k = tgt.isel(time=slice(kk, kk + 1))
            frc_k = frc.isel(time=slice(kk, kk + 1))
            rng_l, *keys = jax.random.split(rng_l, 4)

            # Baseline self-rollout step.
            bp_pred, sb = _bp_step(bp_params, sb, keys[0], cur_b, tgt_k, frc_k)
            # Residual-branch baseline forward (state evolves on cur_r history).
            bp_pred_r, sb_r = _bp_step(bp_params, sb_r, keys[1], cur_r, tgt_k, frc_k)
            # Residual head: takes residual_inputs_r (zero-init + advanced
            # residuals), returns RAW residual in physical units. Full forecast =
            # bp_pred_r + rp_pred.
            rp_pred, sr = _rp_step(rp_params, sr, keys[2], residual_inputs_r, tgt_k, frc_k)

            # Record per-variable per-lead error.
            for var in target_vars:
                truth_da = tgt_k[var].astype("float32").transpose(..., "lat", "lon")
                bp_da = bp_pred[var].astype("float32").transpose(..., "lat", "lon")
                bp_r_da = bp_pred_r[var].astype("float32").transpose(..., "lat", "lon")
                rp_da = rp_pred[var].astype("float32").transpose(..., "lat", "lon")
                truth_v = np.asarray(truth_da.values).squeeze(0).squeeze(0)
                bp_v = np.asarray(jax.device_get(bp_da.values)).squeeze(0).squeeze(0)
                bp_r_v = np.asarray(jax.device_get(bp_r_da.values)).squeeze(0).squeeze(0)
                rp_v = np.asarray(jax.device_get(rp_da.values)).squeeze(0).squeeze(0)
                # Full forecast for the resmamba branch = baseline + residual.
                full_v = bp_r_v + rp_v
                if bp_v.ndim == 3:  # level var — mean over level
                    bp_v = bp_v.mean(axis=0)
                    full_v = full_v.mean(axis=0)
                    truth2 = truth_v.mean(axis=0)
                else:
                    truth2 = truth_v
                eb = bp_v - truth2
                er = full_v - truth2
                sum_sq['baseline'][var][k] += float(((eb ** 2) * cos_lat_n[:, None]).mean())
                sum_sq['resmamba'][var][k] += float(((er ** 2) * cos_lat_n[:, None]).mean())
                sum_bias['baseline'][var][k] += float((eb * cos_lat_n[:, None]).mean())
                sum_bias['resmamba'][var][k] += float((er * cos_lat_n[:, None]).mean())

            # Advance baseline trajectory: feed its own prediction.
            cur_b = _shift_inputs(cur_b, bp_pred, frc_k)

            # Advance residual trajectory + residual_inputs_r per feedback mode.
            # Stream consistency: cur_r and residual_inputs_r MUST track the same
            # residual ("what was physically injected on top of baseline").
            rp_np = _to_numpy_dataset(rp_pred)
            if cfg.feedback_mode == "baseline":
                # Open-loop: cur_r feeds bp_pred_r (NO residual injected physically).
                # residual_inputs_r advances with the model's own residual prediction
                # (= rp_pred) so the SSM state evolves consistently with v22's
                # open-loop semantics (Mamba sees what it predicted).
                cur_r = _shift_inputs(cur_r, bp_pred_r, frc_k)
                residual_inputs_r = advance_residual_inputs(residual_inputs_r, rp_np)
            else:
                # Gated closed-loop: next_input = bp_pred_r + λ * clip(rp).
                # residual_inputs_r advances with λ * clip(rp) (matches physical
                # injection — stream consistency).
                bp_r_np = _to_numpy_dataset(bp_pred_r)
                new_vars = {}
                injected_residual_vars = {}
                for vname, da in bp_r_np.data_vars.items():
                    b_arr = np.asarray(da.values)
                    if vname in rp_np.data_vars:
                        residual_corr = np.asarray(rp_np[vname].values)
                        residual_corr = _clip_per_var(
                            residual_corr, vname,
                            cfg.feedback_clip_mode, cfg.feedback_clip_c)
                        gated_residual = cfg.feedback_lambda * residual_corr
                        next_arr = b_arr + gated_residual
                        injected_residual_vars[vname] = (da.dims, gated_residual)
                    else:
                        next_arr = b_arr
                    new_vars[vname] = (da.dims, next_arr)
                next_ds = xr.Dataset(new_vars, coords=bp_r_np.coords)
                injected_ds = xr.Dataset(injected_residual_vars, coords=bp_r_np.coords)
                cur_r = _shift_inputs(cur_r, next_ds, frc_k)
                residual_inputs_r = advance_residual_inputs(residual_inputs_r, injected_ds)

        if (ai + 1) % 8 == 0 or ai == 0:
            print(f"  anchor {ai + 1}/{cfg.n_anchors}", flush=True)

    # --- Aggregate to RMSE / RMSB ---
    n = cfg.n_anchors
    per_var = {}
    for var in target_vars:
        rmse_b = np.sqrt(sum_sq['baseline'][var] / n)
        rmse_r = np.sqrt(sum_sq['resmamba'][var] / n)
        # RMSB = sqrt(mean_anchor(bias)^2) — but here we accumulate bias and
        # then square the mean. Spec: "rmsb[v][k] = sqrt(mean_anchor(bias[v][k])^2)"
        # = |mean_anchor(bias)|.
        rmsb_b = np.sqrt((sum_bias['baseline'][var] / n) ** 2)
        rmsb_r = np.sqrt((sum_bias['resmamba'][var] / n) ** 2)
        improvement = (1 - rmse_r / np.maximum(rmse_b, 1e-12)) * 100
        per_var[var] = dict(
            rmse_baseline=rmse_b.tolist(),
            rmse_resmamba=rmse_r.tolist(),
            improvement_pct=improvement.tolist(),
            rmsb_baseline=rmsb_b.tolist(),
            rmsb_resmamba=rmsb_r.tolist(),
        )

    out = {
        "residual_ckpt": str(cfg.residual_ckpt),
        "baseline_ckpt": str(cfg.baseline_ckpt),
        "eval_config": {
            "warmup_steps": W,
            "target_steps": K,
            "n_anchors": cfg.n_anchors,
            "feedback_mode": cfg.feedback_mode,
            "feedback_lambda": cfg.feedback_lambda,
            "feedback_clip_mode": cfg.feedback_clip_mode,
            "feedback_clip_c": cfg.feedback_clip_c,
            "lead_hours_per_step": 6,
        },
        "anchor_times": anchor_times,
        "target_variables": target_vars,
        "per_variable_per_lead": per_var,
    }

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {cfg.out_json}")

    print()
    print("=== Per-variable lead-mean RMSE improvement (resmamba vs baseline) ===")
    print(f"  {'variable':<32}{'rmse_base':>12}{'rmse_rm':>12}{'imp_%':>10}")
    for var in target_vars:
        v = per_var[var]
        b_mean = float(np.mean(v['rmse_baseline']))
        r_mean = float(np.mean(v['rmse_resmamba']))
        imp_mean = float(np.mean(v['improvement_pct']))
        print(f"  {var:<32}{b_mean:>12.4f}{r_mean:>12.4f}{imp_mean:>+9.2f}%")


if __name__ == "__main__":
    main()
