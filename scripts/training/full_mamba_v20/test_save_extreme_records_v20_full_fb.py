"""Save full global grids (2m_T, 10m_u, 10m_v) for ERA5 truth, DeepMind GC
baseline, and v18 across a sampled set of 2022 anchors at K=1..6.

This output feeds the 'extreme records' analysis: pointwise heat/cold/wind
forecast bias vs how much the event exceeds the 2015-2021 record.
"""
from __future__ import annotations
import argparse, dataclasses, pickle, sys, time
from pathlib import Path

import haiku as hk, jax, numpy as np, xarray as xr, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization  # noqa: E402
import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import DirectResidualNormalizer  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import GCResidualWithZeroHead, _attach_temporal  # noqa: E402


SURF_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--ckpt-in", default=(
        "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
        "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
        "mesh 2to5 - precipitation input and output.npz"))
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--baseline-msg-steps", type=int, default=16)
    p.add_argument("--residual-msg-steps", type=int, default=2)
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=None)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--temporal-bias", action="store_true", default=False)
    p.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias",
                   action="store_false", default=True)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--no-zero-init-out", dest="temporal_zero_init_out",
                   action="store_false", default=True)
    p.add_argument("--target-steps", type=int, default=6)
    p.add_argument("--anchor-stride-days", type=int, default=3,
                   help="anchor every N days in 2022 (default 3 → ~120 anchors)")
    p.add_argument("--out-nc", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    K = cfg.target_steps

    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = dataclasses.replace(ckpt_in.task_config, input_duration="12h")

    model_cfg_baseline = dataclasses.replace(
        base_model_cfg, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=512, gnn_msg_steps=cfg.baseline_msg_steps)
    model_cfg_residual = dataclasses.replace(
        base_model_cfg, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=512, gnn_msg_steps=cfg.residual_msg_steps)

    norm_stats = base_train.load_stats(Path(cfg.stats_dir))

    def baseline_fn(inp, tgt, frc):
        p = gc.GraphCast(model_cfg_baseline, task_cfg)
        p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(p,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p(inp, targets_template=tgt, forcings=frc)
    def residual_fn(inp, tgt, frc):
        p = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
        _attach_temporal(p, cfg)
        p = casting.Bfloat16Cast(p)
        p = DirectResidualNormalizer(p,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p(inp, targets_template=tgt, forcings=frc)
    baseline_predict = hk.transform_with_state(baseline_fn)
    residual_predict = hk.transform_with_state(residual_fn)

    print(f"Opening {cfg.data_path}")
    ds = xr.open_zarr(cfg.data_path, consolidated=True).sortby("time")
    ds = base_train.prepare_dataset_for_task(ds, task_cfg)
    dt = base_train.infer_time_step(ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    anchor_times = pd.date_range("2022-01-02 00:00", "2022-12-30 00:00",
                                 freq=f"{cfg.anchor_stride_days}D")
    valid_anchors = []
    times_ns = ds.time.values.astype("datetime64[ns]")
    for t in anchor_times:
        t_ns = np.datetime64(t, "ns")
        ok = True
        for k in range(-(input_steps-1), K+1):
            tk = t_ns + np.timedelta64(int(k*6*3600*1e9), "ns")
            if tk not in times_ns:
                ok = False; break
        if ok:
            valid_anchors.append(t_ns)
    print(f"valid_anchors: {len(valid_anchors)} (stride={cfg.anchor_stride_days}d)")

    rng = jax.random.PRNGKey(0)
    sample_t = valid_anchors[0]
    sample_idx = int(np.where(times_ns == sample_t)[0][0])
    sub = ds.isel(time=slice(sample_idx - input_steps + 1, sample_idx + 2))
    sample_inputs, sample_targets, sample_forcings = base_train.build_single_sample(
        sub, final_input_idx=input_steps-1, input_steps=input_steps,
        target_steps=1, task_cfg=task_cfg, dt=dt)
    if "batch" not in sample_inputs.dims:
        sample_inputs = sample_inputs.expand_dims("batch", axis=0).assign_coords(batch=[0])
        sample_targets = sample_targets.expand_dims("batch", axis=0).assign_coords(batch=[0])
        sample_forcings = sample_forcings.expand_dims("batch", axis=0).assign_coords(batch=[0])
    sample_inputs = sample_inputs.load()
    sample_targets = sample_targets.load()
    sample_forcings = sample_forcings.load()
    sample_targets_1 = sample_targets.isel(time=slice(0, 1))
    sample_forcings_1 = sample_forcings.isel(time=slice(0, 1))

    rng, kb = jax.random.split(rng)
    baseline_params, baseline_state = baseline_predict.init(
        kb, sample_inputs, sample_targets_1, sample_forcings_1)
    baseline_params, _ = overlay_matching_params(baseline_params, ckpt_in.params, strict=True)

    rng, kr = jax.random.split(rng)
    residual_params_init, residual_state = residual_predict.init(
        kr, sample_inputs, sample_targets_1, sample_forcings_1)
    with open(cfg.ckpt, "rb") as f:
        ck = pickle.load(f)
    residual_params = ck["residual_params"]
    if "residual_state" in ck and ck["residual_state"]:
        residual_state = ck["residual_state"]
    print("v18 ckpt loaded")

    @jax.jit
    def _baseline_step(p, s, key, inp, tgt, frc):
        out, _ = baseline_predict.apply(p, s, key, inp, tgt, frc)
        return out
    @jax.jit
    def _residual_step(p, s, key, inp, tgt, frc):
        out, new_s = residual_predict.apply(p, s, key, inp, tgt, frc)
        return out, new_s

    def _shift_inputs(prev_inputs, new_state, forcings_next):
        target_time = prev_inputs.time.values[-1:] + dt
        ns = new_state.assign_coords(time=target_time)
        fn = forcings_next.assign_coords(time=target_time)
        next_frame = xr.merge([ns, fn])
        if "datetime" in next_frame.coords:
            next_frame = next_frame.drop_vars("datetime")
        keys_in_next = [k for k in next_frame.data_vars if k in prev_inputs.data_vars]
        nip = next_frame[keys_in_next]
        merged = xr.concat([prev_inputs, nip], dim="time", data_vars="different")
        return merged.tail(time=input_steps)

    def build_anchor_batch(t):
        idx = int(np.where(times_ns == t)[0][0])
        sub = ds.isel(time=slice(idx - input_steps + 1, idx + K + 1))
        ins, tgts, frcs = base_train.build_single_sample(
            sub, final_input_idx=input_steps-1, input_steps=input_steps,
            target_steps=K, task_cfg=task_cfg, dt=dt)
        if "batch" not in ins.dims:
            ins = ins.expand_dims("batch", axis=0).assign_coords(batch=[0])
            tgts = tgts.expand_dims("batch", axis=0).assign_coords(batch=[0])
            frcs = frcs.expand_dims("batch", axis=0).assign_coords(batch=[0])
        return ins.load(), tgts.load(), frcs.load()

    lat = ds.lat.values
    lon = ds.lon.values
    nlat, nlon = lat.size, lon.size
    n_anc = len(valid_anchors)

    def _to_latlon(da2d):
        return da2d.transpose("lat", "lon").values.astype(np.float32)

    out_truth = {v: np.full((n_anc, K, nlat, nlon), np.nan, dtype=np.float32) for v in SURF_VARS}
    out_base  = {v: np.full((n_anc, K, nlat, nlon), np.nan, dtype=np.float32) for v in SURF_VARS}
    out_full  = {v: np.full((n_anc, K, nlat, nlon), np.nan, dtype=np.float32) for v in SURF_VARS}

    t_start = time.time()
    for i_anc, t_anc in enumerate(valid_anchors):
        inp, tgt, frc = build_anchor_batch(t_anc)
        rng_local, _ = jax.random.split(rng)
        rs = residual_state
        current_inp = inp
        for k in range(K):
            tgt_k = tgt.isel(time=slice(k, k+1))
            frc_k = frc.isel(time=slice(k, k+1))
            rng_local, rng_b, rng_r = jax.random.split(rng_local, 3)
            bp = _baseline_step(baseline_params, baseline_state, rng_b, current_inp, tgt_k, frc_k)
            rp, rs = _residual_step(residual_params, rs, rng_r, current_inp, tgt_k, frc_k)
            full = jax.tree_util.tree_map(lambda b, r: b + r, bp, rp)

            for v in SURF_VARS:
                if v in tgt_k.data_vars:
                    out_truth[v][i_anc, k] = _to_latlon(tgt_k[v].isel(time=0, batch=0))
                    out_base[v][i_anc, k]  = _to_latlon(bp[v].isel(time=0, batch=0))
                    out_full[v][i_anc, k]  = _to_latlon(full[v].isel(time=0, batch=0))

            if k < K - 1:
                next_frc = frc.isel(time=slice(k, k+1))  # v20 fix: forcings AT new time slot = forcings[k]
                current_inp = _shift_inputs(current_inp, full, next_frc)  # diagnostic: full feedback

        if (i_anc + 1) % 5 == 0 or i_anc < 3:
            elapsed = time.time() - t_start
            eta = elapsed / (i_anc + 1) * (n_anc - i_anc - 1)
            print(f"  anchor {i_anc+1}/{n_anc} t={t_anc} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    anc_times = np.array(valid_anchors, dtype="datetime64[ns]")
    leads_h = np.arange(1, K + 1) * 6

    coords = {
        "anchor_time": anc_times,
        "lead_h": leads_h,
        "lat": lat,
        "lon": lon,
    }
    out_ds = xr.Dataset(coords=coords)
    for v in SURF_VARS:
        out_ds[f"{v}_truth"]    = (("anchor_time", "lead_h", "lat", "lon"), out_truth[v])
        out_ds[f"{v}_baseline"] = (("anchor_time", "lead_h", "lat", "lon"), out_base[v])
        out_ds[f"{v}_v18"]      = (("anchor_time", "lead_h", "lat", "lon"), out_full[v])

    out_ds.attrs["ckpt_v18"] = cfg.ckpt
    out_ds.attrs["ckpt_baseline"] = cfg.ckpt_in
    out_ds.attrs["K"] = K
    out_ds.attrs["anchor_stride_days"] = cfg.anchor_stride_days

    encoding = {k: {"zlib": True, "complevel": 4} for k in out_ds.data_vars}
    out_ds.to_netcdf(cfg.out_nc, encoding=encoding)
    print(f"\nwrote {cfg.out_nc}")
    print(f"total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
