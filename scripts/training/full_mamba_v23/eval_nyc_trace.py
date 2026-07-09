"""NYC 2m_temperature trace eval: single anchor, 40 lead step trajectory,
saves temperature at NYC grid point for (truth, baseline GraphCast, full = baseline+Mamba).

Cold-bp mode (open-loop, matches v22 training): baseline self-rollout, Mamba
adds residual per step but does NOT feed back. This is v22's natural eval.

Output: JSON with 3 arrays (truth_nyc, baseline_nyc, full_nyc) at lead 6..240h.
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
from src.models.graphcast.training.core.model import DirectResidualNormalizer  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import GCResidualWithZeroHead, _attach_temporal  # noqa: E402


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
    p.add_argument("--anchor-date", default="2022-07-01T00:00:00",
                   help="Anchor (forecast start) datetime, e.g. 2022-07-01T00:00:00")
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nyc-lat", type=float, default=40.7,
                   help="NYC latitude (positive = N)")
    p.add_argument("--nyc-lon", type=float, default=286.0,
                   help="NYC longitude (0-360, NYC = -74°W = 286°E)")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    ckpt_path = Path(cfg.ckpt)
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

    # Find anchor index from anchor_date
    target_time = np.datetime64(cfg.anchor_date)
    time_vals = eval_ds.time.values
    diffs = np.abs(time_vals - target_time)
    anchor_idx = int(np.argmin(diffs))
    print(f"[nyc-trace] anchor_date={cfg.anchor_date} → idx={anchor_idx} actual time={time_vals[anchor_idx]}")

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
        eval_ds, indices=[anchor_idx],
        input_steps=input_steps, target_steps=K,
        task_cfg=task_cfg, dt=dt)
    sample_targets_1step = sample_targets.isel(time=slice(0, 1))
    sample_forcings_1step = sample_forcings.isel(time=slice(0, 1))

    rng = jax.random.PRNGKey(cfg.seed)
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
    print(f"[nyc-trace] loaded residual_params, residual_state=zero init")

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

    # COLD-BP rollout (v22 open-loop semantics): bp self-rollout, residual added per step
    cur = sample_inputs
    bs = baseline_state_init
    rs = residual_state_init
    baseline_traj = []
    full_traj = []
    rng_chain = rng
    for k in range(K):
        tgt_k = sample_targets.isel(time=slice(k, k + 1))
        frc_k = sample_forcings.isel(time=slice(k, k + 1))
        rng_chain, kk_b, kk_r = jax.random.split(rng_chain, 3)
        bp, bs = _baseline_step(baseline_params, bs, kk_b, cur, tgt_k, frc_k)
        rp, rs = _residual_step(residual_params, rs, kk_r, cur, tgt_k, frc_k)
        full_pred = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp)
        baseline_traj.append(bp)
        full_traj.append(full_pred)
        if k < K - 1:
            # cold-bp: feedback = baseline_pred ONLY (open-loop, v22 production semantics)
            cur = _shift_inputs_with_field(cur, bp, frc_k)
        if (k + 1) % 10 == 0:
            print(f"[nyc-trace] step {k+1}/{K}")

    baseline_pred = xr.concat(baseline_traj, dim="time")
    full_pred = xr.concat(full_traj, dim="time")

    # Find nearest grid point to NYC
    lat_vals = eval_ds.lat.values
    lon_vals = eval_ds.lon.values
    lat_idx = int(np.argmin(np.abs(lat_vals - cfg.nyc_lat)))
    lon_idx = int(np.argmin(np.abs(lon_vals - cfg.nyc_lon)))
    print(f"[nyc-trace] NYC nearest grid: lat={lat_vals[lat_idx]:.2f}, lon={lon_vals[lon_idx]:.2f} "
          f"(target {cfg.nyc_lat}, {cfg.nyc_lon})")

    # Extract NYC temperature at each lead step
    truth_nyc = []
    baseline_nyc = []
    full_nyc = []
    lead_times = []
    for k in range(K):
        truth_k = sample_targets["2m_temperature"].isel(time=k)
        bp_k = baseline_pred["2m_temperature"].isel(time=k)
        fp_k = full_pred["2m_temperature"].isel(time=k)
        truth_nyc.append(float(truth_k.isel(lat=lat_idx, lon=lon_idx).values))
        baseline_nyc.append(float(bp_k.isel(lat=lat_idx, lon=lon_idx).values))
        full_nyc.append(float(fp_k.isel(lat=lat_idx, lon=lon_idx).values))
        lead_times.append(str(sample_targets.time.isel(time=k).values))

    out = {
        "ckpt": str(ckpt_path),
        "anchor_date": cfg.anchor_date,
        "anchor_idx": anchor_idx,
        "anchor_actual_time": str(time_vals[anchor_idx]),
        "eval_mode": "cold_bp_open_loop_v22_semantics",
        "nyc_lat": float(lat_vals[lat_idx]),
        "nyc_lon": float(lon_vals[lon_idx]),
        "target_steps": K,
        "lead_times": lead_times,
        "truth_nyc_K": truth_nyc,
        "baseline_nyc_K": baseline_nyc,
        "full_nyc_K": full_nyc,
    }
    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"\n[nyc-trace] wrote {cfg.out_json}")
    print(f"Sample: lead 6h → truth={truth_nyc[0]:.2f}K baseline={baseline_nyc[0]:.2f}K full={full_nyc[0]:.2f}K")
    print(f"Sample: lead 240h → truth={truth_nyc[-1]:.2f}K baseline={baseline_nyc[-1]:.2f}K full={full_nyc[-1]:.2f}K")


if __name__ == "__main__":
    main()
