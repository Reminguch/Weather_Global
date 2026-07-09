"""Baseline-only eval: run a frozen GraphCast checkpoint at its native
resolution on a small set of val anchors, compute per-channel lat-weighted
RMSE and MAE. Used to compare GraphCast_operational (0.25 deg) against
GraphCast_small (1.0 deg) on the same calendar dates.
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

from graphcast import casting, graphcast as gc, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402


def _make_lat_regrid_matrix(src_lat, tgt_lat):
    """Conservative regrid matrix W (n_tgt, n_src) for latitude.
    Weights = sin-space overlap = exact spherical cell-area overlap.
    """
    src_dlat = abs(src_lat[1] - src_lat[0]) / 2
    tgt_dlat = abs(tgt_lat[1] - tgt_lat[0]) / 2
    src_lo = np.clip(np.minimum(src_lat - src_dlat, src_lat + src_dlat), -90, 90)
    src_hi = np.clip(np.maximum(src_lat - src_dlat, src_lat + src_dlat), -90, 90)
    tgt_lo = np.clip(np.minimum(tgt_lat - tgt_dlat, tgt_lat + tgt_dlat), -90, 90)
    tgt_hi = np.clip(np.maximum(tgt_lat - tgt_dlat, tgt_lat + tgt_dlat), -90, 90)

    def sin_area(lo, hi):
        return np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo))

    W = np.zeros((len(tgt_lat), len(src_lat)), dtype=np.float64)
    for i in range(len(tgt_lat)):
        ov_lo = np.maximum(src_lo, tgt_lo[i])
        ov_hi = np.minimum(src_hi, tgt_hi[i])
        valid = ov_hi > ov_lo
        W[i, valid] = sin_area(ov_lo[valid], ov_hi[valid])
    row_sum = W.sum(axis=1, keepdims=True)
    W = np.where(row_sum > 0, W / np.maximum(row_sum, 1e-30), 0)
    return W


def _make_lon_regrid_matrix(src_lon, tgt_lon):
    """Conservative regrid matrix for longitude with 360-deg wrap-around."""
    src_dlon = abs(src_lon[1] - src_lon[0]) / 2
    tgt_dlon = abs(tgt_lon[1] - tgt_lon[0]) / 2
    src_lon_ext = np.concatenate([src_lon - 360, src_lon, src_lon + 360])
    src_lo = src_lon_ext - src_dlon
    src_hi = src_lon_ext + src_dlon
    tgt_lo = tgt_lon - tgt_dlon
    tgt_hi = tgt_lon + tgt_dlon
    n = len(src_lon)
    W_ext = np.zeros((len(tgt_lon), 3 * n), dtype=np.float64)
    for i in range(len(tgt_lon)):
        ov = np.maximum(0, np.minimum(src_hi, tgt_hi[i]) - np.maximum(src_lo, tgt_lo[i]))
        W_ext[i] = ov
    W = W_ext[:, :n] + W_ext[:, n:2 * n] + W_ext[:, 2 * n:]
    W /= W.sum(axis=1, keepdims=True)
    return W


def conservative_regrid_np(arr, src_lat, src_lon, tgt_lat, tgt_lon,
                            W_lat=None, W_lon=None):
    """Conservative regrid of array with shape (..., n_src_lat, n_src_lon).
    Returns shape (..., n_tgt_lat, n_tgt_lon). Pass precomputed W_lat/W_lon
    to avoid rebuilding matrices in a loop.
    """
    if W_lat is None:
        W_lat = _make_lat_regrid_matrix(src_lat, tgt_lat)
    if W_lon is None:
        W_lon = _make_lon_regrid_matrix(src_lon, tgt_lon)
    a = np.einsum("ij,...jk->...ik", W_lat.astype(arr.dtype), arr)
    a = np.einsum("ij,...kj->...ki", W_lon.astype(arr.dtype), a)
    return a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-in", required=True, help="GraphCast .npz checkpoint")
    p.add_argument("--data-path", required=True, help="local zarr with eval anchors")
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--mesh-size", type=int, required=True)
    p.add_argument("--baseline-msg-steps", type=int, default=16)
    p.add_argument("--anchor-times-npy", required=True,
                   help="datetime64 array of the t (final input time) for each anchor")
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=1)
    p.add_argument("--width", type=int, default=512,
                   help="GraphCast latent_size (this is the small default; "
                        "0.25 deg operational ckpt is 512 too)")
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    p.add_argument("--regrid-to-deg", type=float, default=None,
                   help="If set, also compute apples-to-apples metrics after "
                        "conservative regrid of pred AND truth to this resolution "
                        "(e.g. 1.0). Native metrics still reported.")
    p.add_argument("--sparse-stride-deg", type=float, default=None,
                   help="If set, also compute Ilya-style metric: nearest-sample "
                        "pred+truth onto a fixed coarse grid at this physical "
                        "stride in degrees (e.g. 18.0), then accumulate "
                        "GraphCast paper-style weighted loss there.")
    p.add_argument("--sparse-levels", type=int, nargs="+", default=None,
                   help="If set, restrict sparse-grid 3D-var accumulation to these "
                        "pressure levels (e.g. 50 100 ... 1000). Use to make "
                        "Method-4 apples-to-apples across ckpts with different "
                        "native level counts.")
    return p.parse_args()


def main():
    cfg = parse_args()
    print(f"[eval-base] ckpt: {cfg.ckpt_in}")
    print(f"[eval-base] data: {cfg.data_path}")

    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg = dataclasses.replace(
        base_model_cfg, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.baseline_msg_steps)

    print(f"[eval-base] model: res={model_cfg.resolution} mesh={model_cfg.mesh_size} "
          f"levels={len(task_cfg.pressure_levels)} msg_steps={model_cfg.gnn_msg_steps}")
    print(f"[eval-base] target vars: {list(task_cfg.target_variables)}")

    norm_stats = base_train.load_stats(Path(cfg.stats_dir))

    # Open local eval zarr; sort by time. The zarr contains sparse 3-timestep
    # windows per anchor (t-1, t, t+1), so we slice per-anchor below.
    ds = xr.open_zarr(cfg.data_path, consolidated=True)
    ds = ds.sortby("time")
    ds = base_train.prepare_dataset_for_task(ds, task_cfg)
    print(f"[eval-base] eval ds time dim: {ds.sizes.get('time', '?')}")
    # Force dt to 6 hours since infer_time_step would pick gaps between anchors.
    dt = pd.Timedelta(hours=6)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    print(f"[eval-base] forced dt=6h, input_steps={input_steps}")

    use_bf16 = True
    def _build():
        p = gc.GraphCast(model_cfg, task_cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p
    def predict_fn(inputs, targets, forcings):
        return _build()(inputs, targets_template=targets, forcings=forcings)
    predict = hk.transform_with_state(predict_fn)

    # Anchor times -> indices in ds.time
    anchor_times = np.load(cfg.anchor_times_npy)
    anchor_times = np.array(anchor_times, dtype="datetime64[ns]")
    ds_times = ds.time.values.astype("datetime64[ns]")
    # Find each anchor time t in ds_times; we use the LATER input time (t) as anchor index
    # Each "anchor index" in the original eval is the index of the LAST input timestep.
    # We need a list of (idx_in_ds, anchor_time) pairs.
    # In our downloaded ds, each anchor's 3 timesteps (t-1, t, t+1) are present.
    # We pick t such that t-1 and t+1 are also present.
    anchor_t = []
    ds_times_int = ds_times.astype("datetime64[ns]").astype("int64")
    ds_time_set = set(ds_times_int.tolist())
    for t in anchor_times:
        t = np.datetime64(t, "ns")
        t_prev = t - np.timedelta64(6, "h")
        t_next = t + np.timedelta64(6, "h")
        if (t_prev.astype("int64") in ds_time_set
                and t.astype("int64") in ds_time_set
                and t_next.astype("int64") in ds_time_set):
            anchor_t.append(t)
    print(f"[eval-base] anchors that have full window (t-1, t, t+1): {len(anchor_t)} / {len(anchor_times)}")

    # Per-anchor batch builder: slice (t-1, t, t+1) into a dense 3-timestep
    # subset then call build_single_sample and add batch dim manually.
    def build_anchor_batch(anchor_time):
        t = np.datetime64(anchor_time, "ns")
        t_m1 = t - np.timedelta64(6, "h")
        t_p1 = t + np.timedelta64(6, "h")
        sub = ds.sel(time=[t_m1, t, t_p1])
        inputs, targets, forcings = base_train.build_single_sample(
            sub, final_input_idx=1,
            input_steps=input_steps, target_steps=cfg.target_steps,
            task_cfg=task_cfg, dt=dt)
        # Add batch dim if missing (graphcast model expects it)
        if "batch" not in inputs.dims:
            inputs = inputs.expand_dims("batch", axis=0).assign_coords(batch=[0])
            targets = targets.expand_dims("batch", axis=0).assign_coords(batch=[0])
            forcings = forcings.expand_dims("batch", axis=0).assign_coords(batch=[0])
        return inputs.load(), targets.load(), forcings.load()

    sample_inputs, sample_targets, sample_forcings = build_anchor_batch(anchor_t[0])

    rng = jax.random.PRNGKey(cfg.seed)
    params, state = predict.init(rng, sample_inputs, sample_targets, sample_forcings)
    params, stats = overlay_matching_params(params, ckpt_in.params, strict=False)
    print(f"[eval-base] overlay stats: copied={stats.copied}")
    n_p = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"[eval-base] loaded {n_p:,} params from ckpt")

    @jax.jit
    def step_fn(p, s, key, inp, tgt, frc):
        out, _ = predict.apply(p, s, key, inp, tgt, frc)
        return out

    lat = ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_da = xr.DataArray(cos_lat / cos_lat.mean(), dims="lat")

    # If regrid eval requested, precompute regrid matrices once.
    do_regrid = cfg.regrid_to_deg is not None and abs(cfg.regrid_to_deg - cfg.resolution) > 1e-9
    if do_regrid:
        dst = float(cfg.regrid_to_deg)
        tgt_lat = np.arange(90.0, -90.0 - dst / 2, -dst)
        tgt_lon = np.arange(0.0, 360.0, dst)
        src_lat = ds["lat"].values.astype(np.float64)
        src_lon = ds["lon"].values.astype(np.float64)
        W_lat = _make_lat_regrid_matrix(src_lat, tgt_lat)
        W_lon = _make_lon_regrid_matrix(src_lon, tgt_lon)
        cos_lat_rg = np.cos(np.deg2rad(tgt_lat))
        cos_lat_rg = cos_lat_rg / cos_lat_rg.mean()  # shape (n_tgt_lat,)
        print(f"[eval-base] regrid {cfg.resolution} -> {dst}: "
              f"({len(src_lat)},{len(src_lon)}) -> ({len(tgt_lat)},{len(tgt_lon)})")
    else:
        W_lat = W_lon = None
        cos_lat_rg = None

    # Ilya-style sparse-grid metric setup.
    do_sparse = cfg.sparse_stride_deg is not None
    if do_sparse:
        ss = float(cfg.sparse_stride_deg)
        # Choose sparse target lat/lon at exact ss-degree spacing
        sp_lat_target = np.arange(90.0, -90.0 - ss / 2, -ss)
        sp_lon_target = np.arange(0.0, 360.0, ss)
        src_lat_full = ds["lat"].values.astype(np.float64)
        src_lon_full = ds["lon"].values.astype(np.float64)
        sp_lat_idx = np.array([int(np.abs(src_lat_full - t).argmin()) for t in sp_lat_target])
        sp_lon_idx = np.array([int(np.abs(src_lon_full - t).argmin()) for t in sp_lon_target])
        sp_lat = src_lat_full[sp_lat_idx]
        sp_lon = src_lon_full[sp_lon_idx]
        # cos(lat) weight at sparse points, normalized to mean 1
        cos_lat_sp = np.cos(np.deg2rad(sp_lat))
        cos_lat_sp = (cos_lat_sp / cos_lat_sp.mean()).astype(np.float64)
        # Load diffs_stddev_by_level for normalization
        diffs_stddev = xr.open_dataset(Path(cfg.stats_dir) / "diffs_stddev_by_level.nc")
        # GraphCast paper per-variable loss weights
        sparse_w_var = {
            "2m_temperature": 1.0,
            "10m_u_component_of_wind": 0.1, "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1, "total_precipitation_6hr": 0.1,
        }
        print(f"[eval-base] sparse metric: stride={ss}, "
              f"{len(sp_lat)} x {len(sp_lon)} = {len(sp_lat)*len(sp_lon)} pts")
    else:
        sp_lat_idx = sp_lon_idx = cos_lat_sp = diffs_stddev = sparse_w_var = None

    n_used = min(cfg.n_samples, len(anchor_t))
    rng_np = np.random.default_rng(cfg.seed)
    if len(anchor_t) > n_used:
        pick = sorted(rng_np.choice(len(anchor_t), size=n_used, replace=False).tolist())
        anchor_t = [anchor_t[i] for i in pick]

    sum_sq, sum_abs, n_per = {}, {}, {}
    sum_sq_pl, sum_abs_pl, n_per_pl = {}, {}, {}
    sum_sq_rg, sum_abs_rg, n_per_rg = {}, {}, {}
    sum_sq_rg_pl, sum_abs_rg_pl, n_per_rg_pl = {}, {}, {}
    # Ilya-style: per-variable accumulator of normalized weighted squared error
    # (sum over batch of mean over (level, lat, lon) of (err/scale)^2 * lat_w * level_w)
    sparse_per_var_sum, sparse_per_var_n = {}, {}

    for s_i, t in enumerate(anchor_t):
        inp, tgt, frc = build_anchor_batch(t)
        rng, key = jax.random.split(rng)
        pred = step_fn(params, state, key, inp, tgt, frc)

        for var in tgt.data_vars:
            truth = tgt[var].astype("float32")
            p_arr = pred[var].astype("float32")
            err = p_arr - truth
            if var not in sum_sq:
                sum_sq[var] = 0.0; sum_abs[var] = 0.0; n_per[var] = 0
            sum_sq[var] += float(((err ** 2) * cos_lat_da).mean().values)
            sum_abs[var] += float((np.abs(err) * cos_lat_da).mean().values)
            n_per[var] += 1
            if "level" in truth.dims:
                if var not in sum_sq_pl:
                    sum_sq_pl[var] = {}; sum_abs_pl[var] = {}; n_per_pl[var] = {}
                for lev_i, lev in enumerate(truth["level"].values):
                    lev = int(lev)
                    if lev not in sum_sq_pl[var]:
                        sum_sq_pl[var][lev] = 0.0; sum_abs_pl[var][lev] = 0.0; n_per_pl[var][lev] = 0
                    eb = err.isel(level=lev_i)
                    sum_sq_pl[var][lev] += float(((eb ** 2) * cos_lat_da).mean().values)
                    sum_abs_pl[var][lev] += float((np.abs(eb) * cos_lat_da).mean().values)
                    n_per_pl[var][lev] += 1

            if do_regrid:
                # Conservatively regrid BOTH pred and truth to coarse grid,
                # then compute lat-weighted RMSE/MAE there. WB2 paper convention.
                p_rg = conservative_regrid_np(
                    np.asarray(p_arr.values, dtype=np.float32),
                    src_lat, src_lon, tgt_lat, tgt_lon, W_lat, W_lon)
                t_rg = conservative_regrid_np(
                    np.asarray(truth.values, dtype=np.float32),
                    src_lat, src_lon, tgt_lat, tgt_lon, W_lat, W_lon)
                err_rg = p_rg - t_rg
                # Broadcast cos_lat over (... lat, lon); take mean over all dims.
                cl = cos_lat_rg.reshape((1,) * (err_rg.ndim - 2) + (-1, 1))
                if var not in sum_sq_rg:
                    sum_sq_rg[var] = 0.0; sum_abs_rg[var] = 0.0; n_per_rg[var] = 0
                sum_sq_rg[var] += float(((err_rg ** 2) * cl).mean())
                sum_abs_rg[var] += float((np.abs(err_rg) * cl).mean())
                n_per_rg[var] += 1
                if "level" in truth.dims:
                    if var not in sum_sq_rg_pl:
                        sum_sq_rg_pl[var] = {}; sum_abs_rg_pl[var] = {}; n_per_rg_pl[var] = {}
                    lev_axis = list(truth.dims).index("level")
                    for lev_i, lev in enumerate(truth["level"].values):
                        lev = int(lev)
                        if lev not in sum_sq_rg_pl[var]:
                            sum_sq_rg_pl[var][lev] = 0.0; sum_abs_rg_pl[var][lev] = 0.0; n_per_rg_pl[var][lev] = 0
                        eb_rg = np.take(err_rg, lev_i, axis=lev_axis)
                        cl_b = cos_lat_rg.reshape((1,) * (eb_rg.ndim - 2) + (-1, 1))
                        sum_sq_rg_pl[var][lev] += float(((eb_rg ** 2) * cl_b).mean())
                        sum_abs_rg_pl[var][lev] += float((np.abs(eb_rg) * cl_b).mean())
                        n_per_rg_pl[var][lev] += 1

            if do_sparse:
                # Ilya-style: nearest-sample BOTH pred and truth at fixed sparse grid,
                # compute GraphCast-style normalized weighted MSE there.
                # truth.dims is (batch, [level,] lat, lon); index last 2 dims.
                p_sp = np.asarray(p_arr.values, dtype=np.float32)[..., sp_lat_idx[:, None], sp_lon_idx[None, :]]
                t_sp = np.asarray(truth.values, dtype=np.float32)[..., sp_lat_idx[:, None], sp_lon_idx[None, :]]
                err_sp = (p_sp - t_sp).astype(np.float64)
                if "level" in truth.dims:
                    lev_axis = list(truth.dims).index("level")
                    avail = [int(l) for l in truth["level"].values]
                    if cfg.sparse_levels is not None:
                        keep_idx = np.array([avail.index(int(L)) for L in cfg.sparse_levels if int(L) in avail])
                        levels_sub = np.array([avail[i] for i in keep_idx], dtype=np.float64)
                        err_sp = np.take(err_sp, keep_idx, axis=lev_axis)
                    else:
                        levels_sub = np.array(avail, dtype=np.float64)
                    diffs_v = diffs_stddev[var].sel(level=levels_sub.tolist()).values.astype(np.float64)
                    level_w = levels_sub / levels_sub.mean()
                    sq_norm = (err_sp / diffs_v.reshape(
                        (1,) * lev_axis + (-1,) + (1, 1)
                    )) ** 2
                    sq_norm = sq_norm * level_w.reshape(
                        (1,) * lev_axis + (-1,) + (1, 1)
                    )
                    cl = cos_lat_sp.reshape((1,) * (sq_norm.ndim - 2) + (-1, 1))
                    sq_norm = sq_norm * cl
                    contrib = float(sq_norm.mean())
                else:
                    scale = float(diffs_stddev[var].values)
                    sq_norm = (err_sp / scale) ** 2
                    cl = cos_lat_sp.reshape((1,) * (sq_norm.ndim - 2) + (-1, 1))
                    sq_norm = sq_norm * cl
                    contrib = float(sq_norm.mean())
                if var not in sparse_per_var_sum:
                    sparse_per_var_sum[var] = 0.0; sparse_per_var_n[var] = 0
                sparse_per_var_sum[var] += contrib
                sparse_per_var_n[var] += 1

        if s_i < 3 or (s_i + 1) % 4 == 0:
            print(f"[eval-base] sample {s_i+1}/{len(anchor_t)} t={t}", flush=True)

    print()
    print(f"{'variable':<28} {'RMSE_latw':>12} {'MAE_latw':>12}")
    out = {"resolution": cfg.resolution, "mesh_size": cfg.mesh_size,
           "ckpt": cfg.ckpt_in, "n_samples": len(anchor_t),
           "per_variable": {}, "per_channel": {}}
    for var in sorted(n_per):
        n = n_per[var]
        rmse = float((sum_sq[var] / n) ** 0.5)
        mae = float(sum_abs[var] / n)
        print(f"{var:<28} {rmse:>12.6f} {mae:>12.6f}")
        out["per_variable"][var] = dict(rmse_latw=rmse, mae_latw=mae)
        if var in sum_sq_pl:
            for lev in sorted(sum_sq_pl[var]):
                nl = n_per_pl[var][lev]
                rmse_l = float((sum_sq_pl[var][lev] / nl) ** 0.5)
                mae_l = float(sum_abs_pl[var][lev] / nl)
                out["per_channel"][f"{var}_level{lev}"] = dict(rmse_latw=rmse_l, mae_latw=mae_l)
        else:
            out["per_channel"][var] = out["per_variable"][var]

    if do_regrid:
        out["regrid_to_deg"] = float(cfg.regrid_to_deg)
        out["per_variable_regridded"] = {}
        out["per_channel_regridded"] = {}
        print()
        print(f"[eval-base] === regridded to {cfg.regrid_to_deg} deg (apples-to-apples) ===")
        print(f"{'variable':<28} {'RMSE_latw':>12} {'MAE_latw':>12}")
        for var in sorted(n_per_rg):
            n = n_per_rg[var]
            rmse = float((sum_sq_rg[var] / n) ** 0.5)
            mae = float(sum_abs_rg[var] / n)
            print(f"{var:<28} {rmse:>12.6f} {mae:>12.6f}")
            out["per_variable_regridded"][var] = dict(rmse_latw=rmse, mae_latw=mae)
            if var in sum_sq_rg_pl:
                for lev in sorted(sum_sq_rg_pl[var]):
                    nl = n_per_rg_pl[var][lev]
                    rmse_l = float((sum_sq_rg_pl[var][lev] / nl) ** 0.5)
                    mae_l = float(sum_abs_rg_pl[var][lev] / nl)
                    out["per_channel_regridded"][f"{var}_level{lev}"] = dict(rmse_latw=rmse_l, mae_latw=mae_l)
            else:
                out["per_channel_regridded"][var] = out["per_variable_regridded"][var]

    if do_sparse:
        out["sparse_stride_deg"] = float(cfg.sparse_stride_deg)
        out["sparse_per_variable"] = {}
        total_weighted = 0.0
        for var in sorted(sparse_per_var_sum):
            n = sparse_per_var_n[var]
            mean_var = sparse_per_var_sum[var] / n  # mean (over batch) of normalized weighted sq err
            w_var = sparse_w_var.get(var, 1.0)
            contrib = w_var * mean_var
            total_weighted += contrib
            out["sparse_per_variable"][var] = dict(
                normalized_loss=float(mean_var),
                weight=float(w_var),
                contribution=float(contrib))
        out["sparse_weighted_allvars"] = float(total_weighted)
        print()
        print(f"[eval-base] === Ilya-style sparse-grid metric "
              f"(stride={cfg.sparse_stride_deg} deg) ===")
        print(f"{'variable':<28} {'norm_loss':>12} {'w_var':>8} {'contrib':>12}")
        for var in sorted(sparse_per_var_sum):
            v = out["sparse_per_variable"][var]
            print(f"{var:<28} {v['normalized_loss']:>12.6f} {v['weight']:>8.2f} {v['contribution']:>12.6f}")
        print(f"\n  SPARSE weighted_allvars total: {total_weighted:.6f}")

    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"\n[eval-base] wrote {cfg.out_json}")


if __name__ == "__main__":
    main()
