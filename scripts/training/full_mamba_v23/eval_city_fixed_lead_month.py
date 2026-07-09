"""City-level fixed-lead skill plot over a month.

For each anchor in a month, run a warm_bp rollout out to 8 days. At fixed
lead k in {2d, 4d, 6d, 8d}, extract truth / baseline / v22 temperature at
a given city. Plot 4 panels (one per lead) with valid_date on x-axis.

This averages out single-rollout noise and shows whether v22 actually
beats baseline at fixed lead across a month of init times.

Output: JSON with per-anchor predictions + matplotlib PNG.
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
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import DirectResidualNormalizer  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import GCResidualWithZeroHead, _attach_temporal  # noqa: E402


# City coordinates (lat, lon)
CITIES = {
    "NYC":     (40.7, 286.0),
    "Chicago": (41.9, 272.4),
    "LA":      (34.0, 241.7),
    "Paris":   (48.9, 2.4),
    "London":  (51.5, 359.9),
    "Tokyo":   (35.7, 139.7),
    "Beijing": (39.9, 116.4),
    "Shanghai":(31.2, 121.5),
    "Mumbai":  (19.1, 72.9),
    "Sydney":  (-33.9, 151.2),
    # Improvement-hotspot regions from bias-map analysis (lead=240h).
    "TibetHotspot":   (33.0, 80.0),    # v22 improves most (Δ -34 to -38%)
    "AlaskaHotspot":  (60.0, 244.0),   # v22 improves -17 to -19%
    "SHoceanTemp":    (-50.0, 180.0),  # SH temperate ocean (uniform ~-6.6%)
    "AntarcticCoast": (-70.0, 0.0),    # Antarctic coast region
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-v22", required=True,
                   help="v22 (open-loop trained) residual ckpt path.")
    p.add_argument("--ckpt-v22closed", default=None,
                   help="Optional v22closed (closed-loop trained) residual ckpt path. "
                        "If given, runs both models on the same anchors.")
    p.add_argument("--data-path",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
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
    p.add_argument("--input-duration", default="12h")
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
    p.add_argument("--cities", default="ALL",
                   help=f"Comma-separated city names, or 'ALL'. Options: {','.join(CITIES.keys())}")
    p.add_argument("--variables", default="2m_temperature,total_precipitation_6hr",
                   help="Comma-separated variable names to extract.")
    p.add_argument("--month", type=int, default=1,
                   help="Month index 1-12 (default Jan).")
    p.add_argument("--anchor-stride-hours", type=int, default=24,
                   help="Hours between consecutive anchor init times.")
    p.add_argument("--warmup-steps", type=int, default=24)
    p.add_argument("--metric-steps", type=int, default=56,
                   help="rollout length (14d = 56 steps at 6h).")
    p.add_argument("--leads-days", default="1,2,4,6,8,10,14",
                   help="Comma-separated lead-day list to extract.")
    p.add_argument("--feedback-mode", choices=["bp", "mixed", "full"], default="mixed",
                   help="bp: single bp-driven trajectory (baseline closed-loop, "
                        "residuals instantaneous correction). "
                        "mixed (RECOMMENDED): baseline + v22 share bp-driven trajectory "
                        "(matches their open-loop training); v22closed has own closed-loop "
                        "trajectory (matches its training). "
                        "full: each model gets its own closed-loop trajectory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True,
                   help="Directory where one JSON per city will be written.")
    return p.parse_args()


def main():
    cfg = parse_args()
    K = cfg.metric_steps
    W = cfg.warmup_steps

    # ---- Parse cities + variables ----
    if cfg.cities.upper() == "ALL":
        cities_to_run = list(CITIES.keys())
    else:
        cities_to_run = [c.strip() for c in cfg.cities.split(",") if c.strip()]
    unknown = [c for c in cities_to_run if c not in CITIES]
    if unknown:
        raise ValueError(f"Unknown cities: {unknown}. Options: {list(CITIES.keys())}")
    variables = [v.strip() for v in cfg.variables.split(",") if v.strip()]
    print(f"[city-month] cities ({len(cities_to_run)}): {cities_to_run}")
    print(f"[city-month] variables ({len(variables)}): {variables}")

    # ---- Build model ----
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
        train_start_year = 2020
        train_end_year = 2021
    _, eval_ds = base_train._open_local_splits(_SplitCfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    # ---- Find anchor indices: every anchor_stride_hours in given month ----
    times = pd.DatetimeIndex(eval_ds.time.values)
    anchor_mask = (times.month == cfg.month)
    candidate_indices = np.where(anchor_mask)[0]
    # Honor stride. Stride in hours -> in dt units = anchor_stride_hours / 6
    stride = max(1, cfg.anchor_stride_hours // 6)
    # Need each anchor to have W + K future steps
    candidate_indices = [i for i in candidate_indices
                         if i + W + K < eval_ds.sizes["time"] and i >= input_steps - 1]
    candidate_indices = candidate_indices[::stride]
    print(f"[city-month] month={cfg.month}  stride={cfg.anchor_stride_hours}h  "
          f"n_anchors={len(candidate_indices)}")

    # ---- Build apply fns ----
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

    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(candidate_indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state_init = baseline_predict.init(
        k_b, sample_inputs, sample_targets, sample_forcings)
    baseline_params, _ = overlay_matching_params(baseline_params, ckpt_in.params, strict=True)
    _residual_params_init, residual_state_init = residual_predict.init(
        k_r, sample_inputs, sample_targets, sample_forcings)
    # v22 (open-loop trained)
    with Path(cfg.ckpt_v22).open("rb") as f:
        ck_v22 = pickle.load(f)
    residual_params_v22 = ck_v22["residual_params"]
    print(f"[city-month] v22 ckpt: {cfg.ckpt_v22}")
    # Optional v22closed (closed-loop trained)
    residual_params_v22cl = None
    if cfg.ckpt_v22closed is not None:
        with Path(cfg.ckpt_v22closed).open("rb") as f:
            ck_v22cl = pickle.load(f)
        residual_params_v22cl = ck_v22cl["residual_params"]
        print(f"[city-month] v22closed ckpt: {cfg.ckpt_v22closed}")
    print(f"[city-month] both residual_state = zero init")

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

    # ---- Find nearest grid cell for each city ----
    lat_vals = eval_ds.lat.values
    lon_vals = eval_ds.lon.values
    city_idx_map = {}  # city -> (lat_idx, lon_idx, actual_lat, actual_lon)
    for city_name in cities_to_run:
        c_lat, c_lon = CITIES[city_name]
        li = int(np.argmin(np.abs(lat_vals - c_lat)))
        lo = int(np.argmin(np.abs(lon_vals - c_lon)))
        city_idx_map[city_name] = (li, lo, float(lat_vals[li]), float(lon_vals[lo]))
        print(f"  {city_name}: requested ({c_lat:.1f},{c_lon:.1f}) -> "
              f"grid ({lat_vals[li]:.1f},{lon_vals[lo]:.1f})")

    # ---- Lead steps to extract (in 6h steps) ----
    LEADS_DAYS = sorted({int(x.strip()) for x in cfg.leads_days.split(",") if x.strip()})
    LEADS_STEPS = [d * 4 for d in LEADS_DAYS]  # 6h step -> 4 per day
    print(f"[city-month] leads (days): {LEADS_DAYS}")
    for k_d, k_s in zip(LEADS_DAYS, LEADS_STEPS):
        if k_s > K:
            raise ValueError(f"Lead {k_d}d ({k_s} steps) exceeds metric_steps={K}.")

    # ---- Per-anchor rollout, extract per-(city, variable, lead) values ----
    has_v22cl = residual_params_v22cl is not None
    n = len(candidate_indices)
    # store[city][var][lead]['truth'|'base'|'v22'|'v22cl'] -> np.array of length n
    store = {
        c: {v: {k_d: {tag: np.zeros(n) for tag in
                       (["truth","base","v22","v22cl"] if has_v22cl
                        else ["truth","base","v22"])}
                for k_d in LEADS_DAYS} for v in variables}
        for c in cities_to_run
    }
    anchor_times = []
    valid_times_per_lead = {k_d: [] for k_d in LEADS_DAYS}

    fb_mode = cfg.feedback_mode
    is_full = (fb_mode == "full")
    is_mixed = (fb_mode == "mixed")
    mode_desc = {
        "bp": "single bp-driven trajectory (baseline closed-loop, residuals instantaneous)",
        "mixed": "baseline+v22 share bp-driven trajectory; v22closed has own closed-loop",
        "full": "each model has own closed-loop trajectory",
    }[fb_mode]
    print(f"[city-month] feedback-mode = {fb_mode}  ({mode_desc})")

    for s_i, anchor_idx in enumerate(candidate_indices):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(anchor_idx)],
            input_steps=input_steps, target_steps=W + K,
            task_cfg=task_cfg, dt=dt)
        anchor_time = pd.Timestamp(eval_ds.time.values[anchor_idx])
        anchor_times.append(str(anchor_time))

        # Phase 1 — truth warmup (state evolves for both residual models)
        cur = inp; bs = baseline_state_init
        rs_v22 = residual_state_init
        rs_v22cl = residual_state_init if has_v22cl else None
        rng, rng_chain = jax.random.split(rng)
        for k in range(W):
            tgt_k = tgt.isel(time=slice(k, k + 1))
            frc_k = frc.isel(time=slice(k, k + 1))
            rng_chain, kk_b, kk_r = jax.random.split(rng_chain, 3)
            _bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
            _rp_v22, rs_v22 = _residual_step(residual_params_v22, rs_v22, kk_r, cur, tgt_k, frc_k)
            if has_v22cl:
                _rp_cl, rs_v22cl = _residual_step(residual_params_v22cl, rs_v22cl, kk_r,
                                                  cur, tgt_k, frc_k)
            cur = _shift_inputs_with_field(cur, tgt_k, frc_k)

        # ---- Phase 2: rollout ----
        # In bp mode: single trajectory `cur` driven by baseline_pred. Residual
        # is computed on baseline-driven inputs (open-loop wrt residual).
        # In full mode: each model gets its own closed-loop trajectory:
        #   baseline branch: cur_b advanced by bp_b
        #   v22 branch:      cur_v22 advanced by (bp + rp_v22)
        #   v22closed branch: cur_v22cl advanced by (bp + rp_v22cl)
        # (All branches start identical after the truth warmup.)
        if is_full:
            cur_b = cur; bs_b = bs
            cur_v22 = cur; bs_v22 = bs
            cur_v22cl = cur; bs_v22cl = bs if has_v22cl else None
        elif is_mixed:
            # baseline + v22 share bp-driven trajectory (= old bp behavior)
            # v22closed has own closed-loop trajectory
            if has_v22cl:
                cur_cl = cur; bs_cl = bs
        # else bp: reuse the single cur/bs

        for k in range(K):
            kk = W + k
            tgt_k = tgt.isel(time=slice(kk, kk + 1))
            frc_k = frc.isel(time=slice(kk, kk + 1))
            rng_chain, kk_b, kk_r = jax.random.split(rng_chain, 3)

            if is_full:
                # 3 independent trajectories
                bp_b, bs_b = _baseline_step(baseline_params, bs_b, kk_b,
                                            cur_b, tgt_k, frc_k)
                bp_v22, bs_v22 = _baseline_step(baseline_params, bs_v22, kk_b,
                                                cur_v22, tgt_k, frc_k)
                rp_v22, rs_v22 = _residual_step(residual_params_v22, rs_v22, kk_r,
                                                cur_v22, tgt_k, frc_k)
                full_v22 = jax.tree_util.tree_map(lambda b, r: b + r, bp_v22, rp_v22)
                if has_v22cl:
                    bp_cl, bs_v22cl = _baseline_step(baseline_params, bs_v22cl, kk_b,
                                                     cur_v22cl, tgt_k, frc_k)
                    rp_cl, rs_v22cl = _residual_step(residual_params_v22cl, rs_v22cl, kk_r,
                                                     cur_v22cl, tgt_k, frc_k)
                    full_v22cl = jax.tree_util.tree_map(lambda b, r: b + r, bp_cl, rp_cl)
                bp_for_plot = bp_b
                full_v22_for_plot = full_v22
                full_v22cl_for_plot = full_v22cl if has_v22cl else None
            elif is_mixed:
                # baseline + v22 share bp-driven trajectory
                bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
                rp_v22, rs_v22 = _residual_step(residual_params_v22, rs_v22, kk_r,
                                                cur, tgt_k, frc_k)
                full_v22 = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp_v22)
                # v22closed has its own closed-loop trajectory
                if has_v22cl:
                    bp_cl, bs_cl = _baseline_step(baseline_params, bs_cl, kk_b,
                                                  cur_cl, tgt_k, frc_k)
                    rp_cl, rs_v22cl = _residual_step(residual_params_v22cl, rs_v22cl, kk_r,
                                                     cur_cl, tgt_k, frc_k)
                    full_v22cl = jax.tree_util.tree_map(lambda b, r: b + r, bp_cl, rp_cl)
                bp_for_plot = bp
                full_v22_for_plot = full_v22
                full_v22cl_for_plot = full_v22cl if has_v22cl else None
            else:
                # bp mode: single trajectory
                bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
                rp_v22, rs_v22 = _residual_step(residual_params_v22, rs_v22, kk_r,
                                                cur, tgt_k, frc_k)
                full_v22 = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp_v22)
                if has_v22cl:
                    rp_cl, rs_v22cl = _residual_step(residual_params_v22cl, rs_v22cl, kk_r,
                                                     cur, tgt_k, frc_k)
                    full_v22cl = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp_cl)
                bp_for_plot = bp
                full_v22_for_plot = full_v22
                full_v22cl_for_plot = full_v22cl if has_v22cl else None

            # If this is one of our LEADS, extract for each city × variable
            for k_d, k_s in zip(LEADS_DAYS, LEADS_STEPS):
                if k + 1 != k_s:
                    continue
                for var in variables:
                    if var not in tgt_k.data_vars:
                        continue
                    for city_name, (li, lo, _, _) in city_idx_map.items():
                        t_v = tgt_k[var].isel(time=0, lat=li, lon=lo).values
                        b_v = bp_for_plot[var].isel(time=0, lat=li, lon=lo).values
                        f22 = full_v22_for_plot[var].isel(time=0, lat=li, lon=lo).values
                        store[city_name][var][k_d]["truth"][s_i] = float(np.asarray(t_v).squeeze())
                        store[city_name][var][k_d]["base"][s_i] = float(np.asarray(b_v).squeeze())
                        store[city_name][var][k_d]["v22"][s_i] = float(np.asarray(f22).squeeze())
                        if has_v22cl:
                            fcl = full_v22cl_for_plot[var].isel(time=0, lat=li, lon=lo).values
                            store[city_name][var][k_d]["v22cl"][s_i] = float(np.asarray(fcl).squeeze())

            # Advance trajectory for next step
            if k < K - 1:
                if is_full:
                    cur_b = _shift_inputs_with_field(cur_b, bp_b, frc_k)
                    cur_v22 = _shift_inputs_with_field(cur_v22, full_v22, frc_k)
                    if has_v22cl:
                        cur_v22cl = _shift_inputs_with_field(cur_v22cl, full_v22cl, frc_k)
                elif is_mixed:
                    # baseline + v22 share bp-driven trajectory
                    cur = _shift_inputs_with_field(cur, bp, frc_k)
                    # v22closed has own closed-loop trajectory
                    if has_v22cl:
                        cur_cl = _shift_inputs_with_field(cur_cl, full_v22cl, frc_k)
                else:
                    cur = _shift_inputs_with_field(cur, bp, frc_k)

        # Compute valid times (anchor + lead)
        for k_d in LEADS_DAYS:
            valid_t = anchor_time + pd.Timedelta(days=k_d)
            valid_times_per_lead[k_d].append(str(valid_t))

        if (s_i + 1) % 5 == 0 or s_i == 0:
            print(f"[city-month] anchor {s_i+1}/{n} ({anchor_time})", flush=True)

    # ---- Save one JSON per city ----
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for city_name in cities_to_run:
        li, lo, city_lat_a, city_lon_a = city_idx_map[city_name]
        city_out = {
            "city": city_name,
            "city_lat": city_lat_a,
            "city_lon": city_lon_a,
            "variables": variables,
            "feedback_mode": cfg.feedback_mode,
            "month": cfg.month,
            "year": cfg.val_year,
            "anchor_stride_hours": cfg.anchor_stride_hours,
            "warmup_steps": W,
            "metric_steps": K,
            "leads_days": LEADS_DAYS,
            "n_anchors": n,
            "anchor_times": anchor_times,
            "valid_times_per_lead": {str(k): v for k, v in valid_times_per_lead.items()},
            "ckpt_v22": str(cfg.ckpt_v22),
            "ckpt_v22closed": str(cfg.ckpt_v22closed) if has_v22cl else None,
            "vars": {},
        }
        for var in variables:
            city_out["vars"][var] = {
                "truth_at_lead":    {str(k_d): store[city_name][var][k_d]["truth"].tolist()
                                     for k_d in LEADS_DAYS},
                "baseline_at_lead": {str(k_d): store[city_name][var][k_d]["base"].tolist()
                                     for k_d in LEADS_DAYS},
                "v22_at_lead":      {str(k_d): store[city_name][var][k_d]["v22"].tolist()
                                     for k_d in LEADS_DAYS},
                "v22closed_at_lead": (
                    {str(k_d): store[city_name][var][k_d]["v22cl"].tolist()
                     for k_d in LEADS_DAYS} if has_v22cl else None),
            }
        out_json = out_dir / f"{city_name}_month{cfg.month}_fixed_lead.json"
        out_json.write_text(json.dumps(city_out, indent=1))
        print(f"\n[city-month] saved {out_json}")

    # ---- Print summary stats per city per variable ----
    for var in variables:
        # Convert precip m → mm for display only
        scale = 1000.0 if var == "total_precipitation_6hr" else 1.0
        unit = "mm" if scale != 1.0 else "K"
        print(f"\n=== Summary: {var} ({unit}, n_anchors avg) ===")
        print(f"{'city':<10} {'lead':>4}  {'base_RMSE':>9} {'base_bias':>9}  "
              f"{'v22_RMSE':>9} {'v22_bias':>9}" +
              (f"  {'v22cl_RMSE':>10} {'v22cl_bias':>10}" if has_v22cl else ""))
        for city_name in cities_to_run:
            for k_d in LEADS_DAYS:
                s = store[city_name][var][k_d]
                t = np.array(s["truth"]) * scale
                b = np.array(s["base"]) * scale
                fv = np.array(s["v22"]) * scale
                rmse_b = float(np.sqrt(np.mean((b - t)**2)))
                rmse_v = float(np.sqrt(np.mean((fv - t)**2)))
                bias_b = float(np.mean(b - t))
                bias_v = float(np.mean(fv - t))
                row = (f"{city_name:<10} {k_d:>3}d  "
                       f"{rmse_b:>9.3f} {bias_b:>+9.3f}  {rmse_v:>9.3f} {bias_v:>+9.3f}")
                if has_v22cl:
                    fc = np.array(s["v22cl"]) * scale
                    rmse_c = float(np.sqrt(np.mean((fc - t)**2)))
                    bias_c = float(np.mean(fc - t))
                    row += f"  {rmse_c:>10.3f} {bias_c:>+10.3f}"
                print(row)


if __name__ == "__main__":
    main()
