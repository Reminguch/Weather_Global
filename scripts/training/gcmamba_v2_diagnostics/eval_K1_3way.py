"""3-way deployment eval: GCv1 K=3 baseline vs gc_mamba K=1 vs res_mamba K=1.

All res=2 mesh=4 w=512.

Trajectories (per anchor):
  baseline:   pure GCv1 K=3 self-rollout. 1 model forward per step.
  gc_mamba:   GCv1 K=3 backbone with Mamba inserted at mesh_post_processor.
              1 model forward per step (Mamba inside).
  res_mamba:  GCv1 K=3 baseline forward + residual head forward.
              full_pred = baseline + residual. CLOSED-LOOP: cur ← full_pred.

Warmup: 24 truth-fed steps for ALL 3 models (state evolves).
Rollout: 40 closed-loop self-rollout steps. Compute per-cell error at each step.

Output:
  Per-anchor per-city per-lead values (for time-series plot)
  Per-variable per-lead lat-w RMSE (for improvement table)
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization, xarray_jax  # noqa: E402
import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    build_predictor, derive_model_config_from_checkpoint, load_graphcast_checkpoint,
    load_stats, DirectResidualNormalizer,
)
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import (  # noqa: E402
    GCResidualWithZeroHead, _attach_temporal,
)


CITIES = {
    "NYC":     (40.7, 286.0),
    "LA":      (34.0, 241.7),
    "Paris":   (48.9, 2.4),
    "London":  (51.5, 359.9),
    "Tokyo":   (35.7, 139.7),
    "Beijing": (39.9, 116.4),
    "Mumbai":  (19.1, 72.9),
    "Sydney":  (-33.9, 151.2),
    "TibetHotspot":   (33.0, 80.0),
    "AntarcticCoast": (-70.0, 0.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gcv1-k3-ckpt", required=True)
    p.add_argument("--gc-mamba-ckpt", required=True)
    p.add_argument("--res-mamba-ckpt", required=True)
    p.add_argument("--data-path",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--mesh-size", type=int, default=4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-mp", type=int, default=6)
    p.add_argument("--res-mamba-mp", type=int, default=2)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--gcm-temporal-d-inner", type=int, default=128)
    p.add_argument("--gcm-temporal-d-state", type=int, default=16)
    p.add_argument("--gcm-temporal-d-conv", type=int, default=4)
    p.add_argument("--gcm-temporal-layers", type=int, default=1)
    p.add_argument("--gcm-temporal-insert-count", type=int, default=1)
    p.add_argument("--rm-temporal-d-inner", type=int, default=128)
    p.add_argument("--rm-temporal-d-state", type=int, default=16)
    p.add_argument("--rm-temporal-d-conv", type=int, default=4)
    p.add_argument("--rm-temporal-layers", type=int, default=1)
    p.add_argument("--rm-temporal-insert-count", type=int, default=2)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--n-anchors", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=24)
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def _flat_npz_to_haiku(npz_path: Path) -> dict:
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


class _Cfg:
    """Minimal config wrapper for _attach_temporal compat (mirrors train_mz_v9 args)."""
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


def main():
    cfg = parse_args()
    W = cfg.warmup_steps
    K = cfg.target_steps

    # --- Load baseline ckpt (= GCv1 K=3) for task/model config + params ---
    ckpt_baseline = load_graphcast_checkpoint(Path(cfg.gcv1_k3_ckpt))
    task_cfg = dataclasses.replace(ckpt_baseline.task_config, input_duration=cfg.input_duration)
    norm_stats = load_stats(Path(cfg.stats_dir))

    mc_main = derive_model_config_from_checkpoint(
        ckpt_baseline.model_config, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.baseline_mp, hidden_layers=1)
    mc_residual = derive_model_config_from_checkpoint(
        ckpt_baseline.model_config, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.res_mamba_mp, hidden_layers=1)

    use_bf16 = True

    # --- 1) baseline forward: vanilla GC (no Mamba) ---
    def _build_baseline():
        p = gc.GraphCast(mc_main, task_cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p
    def baseline_fn(i, t, f): return _build_baseline()(i, targets_template=t, forcings=f)
    baseline_predict = hk.transform_with_state(baseline_fn)

    # --- 2) gc_mamba forward: build_predictor with Mamba inside, mp=6 ---
    # autoregressive_loss_mode='none' bypasses autoregressive.Predictor wrapper
    # (hk.scan in init() triggers JaxArrayWrapper inside xarray.concat).
    def gc_mamba_fn(inputs, targets, forcings):
        p = build_predictor(
            mc_main, task_cfg, norm_stats, use_bf16=True, gradient_checkpointing=False,
            temporal_backbone='mamba',
            temporal_location='mesh_post_processor',
            temporal_d_inner=cfg.gcm_temporal_d_inner,
            temporal_d_state=cfg.gcm_temporal_d_state,
            temporal_d_conv=cfg.gcm_temporal_d_conv,
            temporal_dt_rank='auto', temporal_bias=False, temporal_conv_bias=True,
            temporal_layers=cfg.gcm_temporal_layers, temporal_dropout=0.0,
            temporal_stateful=True, temporal_insert_count=cfg.gcm_temporal_insert_count,
            zero_init_temporal_out=True,
            autoregressive_loss_mode='none')
        return p(inputs, targets_template=targets, forcings=forcings)
    gc_mamba_predict = hk.transform_with_state(gc_mamba_fn)

    # --- 3) res_mamba residual head: GCResidualWithZeroHead + Mamba interleaved + DirectResidualNormalizer ---
    rm_cfg = _Cfg(
        temporal_location='mesh_processor_interleaved',
        temporal_hidden_size=128,
        temporal_d_inner=cfg.rm_temporal_d_inner,
        temporal_d_state=cfg.rm_temporal_d_state,
        temporal_d_conv=cfg.rm_temporal_d_conv,
        temporal_dt_rank='auto',
        temporal_bias=False, temporal_conv_bias=True,
        temporal_layers=cfg.rm_temporal_layers, temporal_dropout=0.0,
        temporal_insert_count=cfg.rm_temporal_insert_count,
        temporal_zero_init_out=True,
        temporal_stateful=True,
    )
    def _build_residual():
        p = GCResidualWithZeroHead(mc_residual, task_cfg)
        _attach_temporal(p, rm_cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        p = DirectResidualNormalizer(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p
    def residual_fn(i, t, f): return _build_residual()(i, targets_template=t, forcings=f)
    residual_predict = hk.transform_with_state(residual_fn)

    # --- Data ---
    class _SplitCfg:
        data_path = cfg.data_path; resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = 2015; train_end_year = 2021
    _, eval_ds = base_train._open_local_splits(_SplitCfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    valid_idx = base_train.valid_final_input_indices(eval_ds.sizes["time"], input_steps, W + K)
    rng_np = np.random.default_rng(cfg.seed)
    anchor_indices = sorted(rng_np.choice(valid_idx, size=cfg.n_anchors, replace=False).tolist())

    # --- Init params for shapes ---
    sample_inp, sample_tgt, sample_frc = base_train.build_batch_from_indices(
        eval_ds, indices=[int(anchor_indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, k1, k2, k3 = jax.random.split(rng, 4)

    bp_params, bp_state0 = baseline_predict.init(k1, sample_inp, sample_tgt, sample_frc)
    bp_params, _ = overlay_matching_params(bp_params, ckpt_baseline.params, strict=True)
    print(f"baseline params: overlaid {sum(len(v) for v in ckpt_baseline.params.values())} GC leaves from GCv1 K=3")

    gm_params, gm_state0 = gc_mamba_predict.init(k2, sample_inp, sample_tgt, sample_frc)
    gm_trained = _flat_npz_to_haiku(Path(cfg.gc_mamba_ckpt))
    gm_params, _ = overlay_matching_params(gm_params, gm_trained, strict=False)
    print(f"gc_mamba params: overlaid {sum(len(v) for v in gm_trained.values())} leaves from trained K=1 ckpt")

    rm_params, rm_state0 = residual_predict.init(k3, sample_inp, sample_tgt, sample_frc)
    rm_trained = _flat_npz_to_haiku(Path(cfg.res_mamba_ckpt))
    rm_params, rm_ov = overlay_matching_params(rm_params, rm_trained, strict=False)
    print(f"res_mamba residual params: overlaid {rm_ov.copied} / {sum(len(v) for v in rm_trained.values())} leaves")

    @jax.jit
    def _bp_step(p, s, key, inp, tgt, frc): return baseline_predict.apply(p, s, key, inp, tgt, frc)
    @jax.jit
    def _gm_step(p, s, key, inp, tgt, frc): return gc_mamba_predict.apply(p, s, key, inp, tgt, frc)
    @jax.jit
    def _rm_step(p, s, key, inp, tgt, frc): return residual_predict.apply(p, s, key, inp, tgt, frc)

    def _to_numpy_dataset(ds):
        """Aggressively convert any JaxArrayWrapper inside an xarray Dataset to plain numpy."""
        new_vars = {}
        for var in ds.data_vars:
            da = ds[var]
            d = da.data
            # JaxArrayWrapper has ._data attribute that holds the underlying jax array
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

    # --- City lat/lon indices ---
    lat_vals = eval_ds.lat.values
    lon_vals = eval_ds.lon.values
    city_idx = {}
    for nm, (clat, clon) in CITIES.items():
        li = int(np.argmin(np.abs(lat_vals - clat)))
        lo = int(np.argmin(np.abs(lon_vals - clon)))
        city_idx[nm] = (li, lo, float(lat_vals[li]), float(lon_vals[lo]))

    cos_lat = np.cos(np.deg2rad(lat_vals))
    cos_lat_n = cos_lat / cos_lat.mean()
    target_vars = list(task_cfg.target_variables)
    n_lat, n_lon = len(lat_vals), len(lon_vals)

    # Output containers
    sum_sq = {m: {v: np.zeros(K, dtype=np.float64) for v in target_vars}
              for m in ['baseline', 'gc_mamba', 'res_mamba']}
    # city_traces[model][city][var] = list of length n_anchors * K
    city_traces = {m: {c: {v: [] for v in target_vars} for c in CITIES}
                   for m in ['baseline', 'gc_mamba', 'res_mamba', 'truth']}
    anchor_times = []

    for ai, idx in enumerate(anchor_indices):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=W + K, task_cfg=task_cfg, dt=dt)
        anchor_times.append(str(pd.Timestamp(eval_ds.time.values[idx])))

        # Independent trajectories
        cur_b = inp; sb = bp_state0
        cur_g = inp; sg = gm_state0
        cur_r = inp; sb_r = bp_state0  # baseline branch for res_mamba
        sr = rm_state0                  # residual head state
        rng_l = jax.random.PRNGKey(cfg.seed + 99 + ai)

        # Phase 1: 24-step truth warmup
        for k in range(W):
            tgt_k = tgt.isel(time=slice(k, k+1))
            frc_k = frc.isel(time=slice(k, k+1))
            rng_l, *keys = jax.random.split(rng_l, 5)
            _b, sb = _bp_step(bp_params, sb, keys[0], cur_b, tgt_k, frc_k)
            _g, sg = _gm_step(gm_params, sg, keys[1], cur_g, tgt_k, frc_k)
            _bp_r, sb_r = _bp_step(bp_params, sb_r, keys[2], cur_r, tgt_k, frc_k)
            _r, sr = _rm_step(rm_params, sr, keys[3], cur_r, tgt_k, frc_k)
            # all 3 trajectories share truth feed during warmup
            cur_b = _shift_inputs(cur_b, tgt_k, frc_k)
            cur_g = _shift_inputs(cur_g, tgt_k, frc_k)
            cur_r = _shift_inputs(cur_r, tgt_k, frc_k)

        # Phase 2: 40-step closed-loop self-rollout
        for k in range(K):
            kk = W + k
            tgt_k = tgt.isel(time=slice(kk, kk+1))
            frc_k = frc.isel(time=slice(kk, kk+1))
            rng_l, *keys = jax.random.split(rng_l, 5)
            bp_pred, sb = _bp_step(bp_params, sb, keys[0], cur_b, tgt_k, frc_k)
            gm_pred, sg = _gm_step(gm_params, sg, keys[1], cur_g, tgt_k, frc_k)
            # res_mamba: baseline forward + residual forward → full = b + r
            rb_pred, sb_r = _bp_step(bp_params, sb_r, keys[2], cur_r, tgt_k, frc_k)
            r_pred, sr   = _rm_step(rm_params, sr, keys[3], cur_r, tgt_k, frc_k)
            full_pred = jax.tree_util.tree_map(lambda a, b: a + b, rb_pred, r_pred)

            for var in target_vars:
                # Force (..., lat, lon) order so cos_lat broadcasts properly.
                truth_da = tgt_k[var].astype("float32").transpose(..., "lat", "lon")
                bp_da = bp_pred[var].astype("float32").transpose(..., "lat", "lon")
                gm_da = gm_pred[var].astype("float32").transpose(..., "lat", "lon")
                fp_da = full_pred[var].astype("float32").transpose(..., "lat", "lon")
                truth_v = np.asarray(truth_da.values).squeeze(0).squeeze(0)
                bp_v = np.asarray(jax.device_get(bp_da.values)).squeeze(0).squeeze(0)
                gm_v = np.asarray(jax.device_get(gm_da.values)).squeeze(0).squeeze(0)
                fp_v = np.asarray(jax.device_get(fp_da.values)).squeeze(0).squeeze(0)
                if bp_v.ndim == 3:  # has level dim → mean over level
                    bp_v = bp_v.mean(axis=0); gm_v = gm_v.mean(axis=0); fp_v = fp_v.mean(axis=0)
                    truth2 = truth_v.mean(axis=0)
                else:
                    truth2 = truth_v
                eb = bp_v - truth2; eg = gm_v - truth2; er = fp_v - truth2
                sum_sq['baseline'][var][k]  += float(((eb**2) * cos_lat_n[:, None]).mean())
                sum_sq['gc_mamba'][var][k]  += float(((eg**2) * cos_lat_n[:, None]).mean())
                sum_sq['res_mamba'][var][k] += float(((er**2) * cos_lat_n[:, None]).mean())
                for nm, (li, lo, _, _) in city_idx.items():
                    city_traces['truth'][nm][var].append(float(truth2[li, lo]))
                    city_traces['baseline'][nm][var].append(float(bp_v[li, lo]))
                    city_traces['gc_mamba'][nm][var].append(float(gm_v[li, lo]))
                    city_traces['res_mamba'][nm][var].append(float(fp_v[li, lo]))

            cur_b = _shift_inputs(cur_b, bp_pred, frc_k)
            cur_g = _shift_inputs(cur_g, gm_pred, frc_k)
            cur_r = _shift_inputs(cur_r, full_pred, frc_k)

        if (ai + 1) % 4 == 0 or ai == 0:
            print(f"  anchor {ai+1}/{cfg.n_anchors}", flush=True)

    out = {
        'cities': CITIES, 'city_idx': city_idx,
        'anchor_times': anchor_times,
        'warmup_steps': W, 'target_steps': K, 'n_anchors': cfg.n_anchors,
        'lead_hours_per_step': 6,
        'target_variables': target_vars,
        'per_variable_per_lead': {},
        'city_traces': city_traces,
    }
    n = cfg.n_anchors
    for var in target_vars:
        rmse_b = np.sqrt(sum_sq['baseline'][var] / n)
        rmse_g = np.sqrt(sum_sq['gc_mamba'][var] / n)
        rmse_r = np.sqrt(sum_sq['res_mamba'][var] / n)
        out['per_variable_per_lead'][var] = dict(
            rmse_baseline=rmse_b.tolist(),
            rmse_gc_mamba=rmse_g.tolist(),
            rmse_res_mamba=rmse_r.tolist(),
            improvement_gc_mamba_pct=((1 - rmse_g / np.maximum(rmse_b, 1e-12)) * 100).tolist(),
            improvement_res_mamba_pct=((1 - rmse_r / np.maximum(rmse_b, 1e-12)) * 100).tolist(),
        )

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {cfg.out_json}")

    print()
    print("=== Per-variable lead-mean improvement ===")
    print(f"  {'variable':<32}{'rmse_base':>12}{'gc_mamba':>14}{'res_mamba':>14}")
    for var in target_vars:
        v = out['per_variable_per_lead'][var]
        b_mean = float(np.mean(v['rmse_baseline']))
        ig = float(np.mean(v['improvement_gc_mamba_pct']))
        ir = float(np.mean(v['improvement_res_mamba_pct']))
        print(f"  {var:<32}{b_mean:>12.4f}{ig:>+13.2f}%{ir:>+13.2f}%")


if __name__ == "__main__":
    main()
