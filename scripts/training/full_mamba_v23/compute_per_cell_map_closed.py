"""Compute per-cell error map at lead 240h for v22closed in warm_full mode.

Closed-loop self-rollout: cur is advanced by full_pred = baseline + residual.
24-step truth warmup → 40-step closed-loop metric rollout.

Output NC has same schema as compute_per_cell_bias_map.py (bias/variance fields
for baseline + closed-loop model) so the existing K22 NC and this one can be
diffed directly for the open-vs-closed comparison.
"""
from __future__ import annotations

import argparse
import dataclasses
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
from src.models.graphcast.training.core.model import DirectResidualNormalizer  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import GCResidualWithZeroHead, _attach_temporal  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
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
    p.add_argument("--n-anchors", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=24)
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variable", default="2m_temperature")
    p.add_argument("--lead-step-index", type=int, default=39,
                   help="Which lead step to map (39 = 240h).")
    p.add_argument("--out-nc", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    K = cfg.target_steps
    W = cfg.warmup_steps
    n_anchors = cfg.n_anchors

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

    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, W + K)
    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = sorted(rng_np.choice(val_indices, size=n_anchors,
                                       replace=False).tolist())
    print(f"[closed-map] sampling {n_anchors} anchors from val_year={cfg.val_year}")

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
        eval_ds, indices=[int(chosen_idx[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state_init = baseline_predict.init(
        k_b, sample_inputs, sample_targets, sample_forcings)
    baseline_params, _ = overlay_matching_params(baseline_params, ckpt_in.params, strict=True)
    _residual_params_init, residual_state_init = residual_predict.init(
        k_r, sample_inputs, sample_targets, sample_forcings)
    with Path(cfg.ckpt).open("rb") as f:
        ck = pickle.load(f)
    residual_params = ck["residual_params"]

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

    lat = eval_ds["lat"].values
    lon = eval_ds["lon"].values
    n_lat = len(lat); n_lon = len(lon)
    err_b_all = np.zeros((n_anchors, n_lat, n_lon), dtype=np.float32)
    err_f_all = np.zeros((n_anchors, n_lat, n_lon), dtype=np.float32)

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)],
            input_steps=input_steps, target_steps=W + K,
            task_cfg=task_cfg, dt=dt)
        all_tgt = tgt; all_frc = frc

        # Phase 1: 24-step truth warmup (both branches share)
        cur = inp; bs = baseline_state_init; rs = residual_state_init
        rng, rng_chain = jax.random.split(rng)
        for k in range(W):
            tgt_k = all_tgt.isel(time=slice(k, k + 1))
            frc_k = all_frc.isel(time=slice(k, k + 1))
            rng_chain, kk_b, kk_r = jax.random.split(rng_chain, 3)
            _bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
            _rp, rs = _residual_step(residual_params, rs, kk_r, cur, tgt_k, frc_k)
            cur = _shift_inputs_with_field(cur, tgt_k, frc_k)

        # Phase 2: closed-loop self-rollout (TWO independent trajectories)
        # baseline branch: cur_b advanced by bp_b
        # closed-loop branch: cur_f advanced by full_pred = bp_f + rp_f
        cur_b = cur; bs_b = bs
        cur_f = cur; bs_f = bs

        for k in range(K):
            kk = W + k
            tgt_k = all_tgt.isel(time=slice(kk, kk + 1))
            frc_k = all_frc.isel(time=slice(kk, kk + 1))
            rng_chain, kk_b, kk_r = jax.random.split(rng_chain, 3)
            bp_b, bs_b = _baseline_step(baseline_params, bs_b, kk_b, cur_b, tgt_k, frc_k)
            bp_f, bs_f = _baseline_step(baseline_params, bs_f, kk_b, cur_f, tgt_k, frc_k)
            rp, rs = _residual_step(residual_params, rs, kk_r, cur_f, tgt_k, frc_k)
            full_pred = jax.tree_util.tree_map(lambda b, r: b + r, bp_f, rp)
            if k == cfg.lead_step_index:
                tgt_var = tgt_k[cfg.variable].astype("float32")
                bp_var = bp_b[cfg.variable].astype("float32")
                fp_var = full_pred[cfg.variable].astype("float32")
                err_b_da = (bp_var - tgt_var).squeeze().transpose("lat", "lon")
                err_f_da = (fp_var - tgt_var).squeeze().transpose("lat", "lon")
                err_b_all[s_i] = np.asarray(err_b_da.values)
                err_f_all[s_i] = np.asarray(err_f_da.values)
            if k < K - 1:
                cur_b = _shift_inputs_with_field(cur_b, bp_b, frc_k)
                cur_f = _shift_inputs_with_field(cur_f, full_pred, frc_k)
        if (s_i + 1) % 4 == 0 or s_i == 0:
            print(f"[closed-map] anchor {s_i+1}/{n_anchors}", flush=True)

    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat = cos_lat / cos_lat.mean()

    bias_b = err_b_all.mean(axis=0)
    bias_f = err_f_all.mean(axis=0)
    var_b = ((err_b_all - bias_b[None]) ** 2).mean(axis=0)
    var_f = ((err_f_all - bias_f[None]) ** 2).mean(axis=0)
    mse_b = bias_b**2 + var_b
    mse_f = bias_f**2 + var_f

    def lat_w_mean(field_2d):
        return float((field_2d * cos_lat[:, None]).mean())

    print(f"\n=== closed-loop self-rollout (warm_full), lead {(cfg.lead_step_index+1)*6}h ===")
    print(f"  baseline RMSE (lat-w) = {np.sqrt(lat_w_mean(mse_b)):.4f}")
    print(f"  closed   RMSE (lat-w) = {np.sqrt(lat_w_mean(mse_f)):.4f}")
    print(f"  improvement = {100*(1-np.sqrt(lat_w_mean(mse_f))/np.sqrt(lat_w_mean(mse_b))):+.2f}%")

    ds_out = xr.Dataset(
        data_vars=dict(
            bias_baseline=(["lat","lon"], bias_b),
            bias_closed=(["lat","lon"], bias_f),
            variance_baseline=(["lat","lon"], var_b),
            variance_closed=(["lat","lon"], var_f),
        ),
        coords=dict(lat=lat, lon=lon),
        attrs=dict(
            variable=cfg.variable,
            lead_step_index=cfg.lead_step_index,
            lead_hours=(cfg.lead_step_index+1)*6,
            n_anchors=n_anchors,
            ckpt=str(cfg.ckpt),
            eval_mode="warm_full",
            warmup_steps=W,
        ),
    )
    Path(cfg.out_nc).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(cfg.out_nc)
    print(f"[closed-map] saved {cfg.out_nc}")


if __name__ == "__main__":
    main()
