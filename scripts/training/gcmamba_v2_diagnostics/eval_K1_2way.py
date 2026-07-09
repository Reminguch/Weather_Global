"""2-way deployment eval: baseline GC vs gc_mamba K=1 (the 20k postproc res=1 run).

For 32 anchors × (24 truth-warmup + 40 closed-loop self-rollout):
  - Run baseline GC self-rollout
  - Run gc_mamba self-rollout
  - Compute per-cell error vs truth at each lead step
  - Save city traces (truth, baseline, gc_mamba) + global lat-w RMSE
"""
from __future__ import annotations
import argparse, dataclasses, json, sys
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
from src.models.graphcast.training.core.model import (  # noqa: E402
    build_predictor, derive_model_config_from_checkpoint, load_graphcast_checkpoint, load_stats)
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402


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
    p.add_argument("--baseline-ckpt", required=True,
                   help="Vanilla GC ckpt (DeepMind small for res=1, GCv1 K=3 for res=2).")
    p.add_argument("--gc-mamba-ckpt", required=True,
                   help="Trained gc_mamba ckpt (npz with params: prefix).")
    p.add_argument("--data-path",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr")
    p.add_argument("--stats-dir",
                   default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--processor-msg-steps", type=int, default=16)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--temporal-location", default="mesh_post_processor")
    p.add_argument("--temporal-d-inner", type=int, default=128)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-layers", type=int, default=1)
    p.add_argument("--temporal-insert-count", type=int, default=1)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--n-anchors", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=24)
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--align-w-max", type=int, default=0,
                   help="if >0, sample anchors based on (W_max + K) and shift each anchor "
                        "forward by (W_max - W) so AR phase starts at the same calendar time "
                        "regardless of W. Lets you sweep warmup duration apples-to-apples.")
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


def main():
    cfg = parse_args()
    W = cfg.warmup_steps
    K = cfg.target_steps

    ck = load_graphcast_checkpoint(Path(cfg.baseline_ckpt))
    task_cfg = dataclasses.replace(ck.task_config, input_duration=cfg.input_duration)
    norm_stats = load_stats(Path(cfg.stats_dir))

    mc = derive_model_config_from_checkpoint(
        ck.model_config, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.processor_msg_steps, hidden_layers=1)

    # --- 1) baseline: vanilla GC, no Mamba ---
    def _build_baseline():
        p = gc.GraphCast(mc, task_cfg)
        p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p
    def baseline_fn(i,t,f): return _build_baseline()(i, targets_template=t, forcings=f)
    bp_t = hk.transform_with_state(baseline_fn)

    # --- 2) gc_mamba: build_predictor with Mamba inside ---
    # autoregressive_loss_mode='none' avoids wrapping with autoregressive.Predictor
    # (which uses hk.scan and trips xarray_jax's JaxArrayWrapper inside concat).
    # We do our own per-step rollout below.
    def gc_mamba_fn(i, t, f):
        p = build_predictor(
            mc, task_cfg, norm_stats, use_bf16=True, gradient_checkpointing=False,
            temporal_backbone='mamba',
            temporal_location=cfg.temporal_location,
            temporal_d_inner=cfg.temporal_d_inner,
            temporal_d_state=cfg.temporal_d_state,
            temporal_d_conv=cfg.temporal_d_conv,
            temporal_dt_rank='auto', temporal_bias=False, temporal_conv_bias=True,
            temporal_layers=cfg.temporal_layers, temporal_dropout=0.0,
            temporal_stateful=True, temporal_insert_count=cfg.temporal_insert_count,
            zero_init_temporal_out=True,
            autoregressive_loss_mode='none')
        return p(i, targets_template=t, forcings=f)
    gm_t = hk.transform_with_state(gc_mamba_fn)

    # --- Data ---
    class _SplitCfg:
        data_path = cfg.data_path; resolution = cfg.resolution
        val_year = cfg.val_year
        train_start_year = 2015; train_end_year = 2021
    _, eval_ds = base_train._open_local_splits(_SplitCfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    W_pool = cfg.align_w_max if cfg.align_w_max > 0 else W
    if cfg.align_w_max > 0 and W > cfg.align_w_max:
        raise ValueError(f"warmup-steps ({W}) must be <= align-w-max ({cfg.align_w_max})")
    valid_idx = base_train.valid_final_input_indices(eval_ds.sizes["time"], input_steps, W_pool + K)
    rng_np = np.random.default_rng(cfg.seed)
    anchors_raw = sorted(rng_np.choice(valid_idx, size=cfg.n_anchors, replace=False).tolist())
    if cfg.align_w_max > 0:
        # Shift each anchor forward by (W_max - W) so AR phase starts at the
        # same calendar time across all W tasks in the sweep.
        shift = cfg.align_w_max - W
        anchors = [a + shift for a in anchors_raw]
        print(f"[align_w_max] sampled {len(anchors)} anchors with W_max={cfg.align_w_max}, "
              f"shifting each by +{shift} so AR-start aligned (W={W} task)")
    else:
        anchors = anchors_raw

    # --- Init params ---
    sample_inp, sample_tgt, sample_frc = base_train.build_batch_from_indices(
        eval_ds, indices=[int(anchors[0])], input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, k1, k2 = jax.random.split(rng, 3)
    bp_params, bp_state0 = bp_t.init(k1, sample_inp, sample_tgt, sample_frc)
    bp_params, _ = overlay_matching_params(bp_params, ck.params, strict=True)
    print(f"baseline: overlaid {sum(len(v) for v in ck.params.values())} GC leaves")
    gm_params, gm_state0 = gm_t.init(k2, sample_inp, sample_tgt, sample_frc)
    gm_trained = _flat_npz_to_haiku(Path(cfg.gc_mamba_ckpt))
    gm_params, gm_ov = overlay_matching_params(gm_params, gm_trained, strict=False)
    print(f"gc_mamba: overlaid {gm_ov.copied} / {sum(len(v) for v in gm_trained.values())} leaves")

    @jax.jit
    def _bp_step(p,s,key,i,t,f): return bp_t.apply(p,s,key,i,t,f)
    @jax.jit
    def _gm_step(p,s,key,i,t,f): return gm_t.apply(p,s,key,i,t,f)

    def _to_numpy_dataset(ds):
        nv = {}
        for var in ds.data_vars:
            da = ds[var]; d = da.data
            if hasattr(d, '_data'): d = d._data
            nv[var] = (da.dims, np.asarray(jax.device_get(d)))
        nc = {}
        for c in ds.coords:
            cv = ds[c].values
            if hasattr(cv, '_data'): cv = cv._data
            nc[c] = np.asarray(cv)
        return xr.Dataset(nv, coords=nc)

    def _shift(prev, new_field, fnext):
        new_field = _to_numpy_dataset(new_field)
        target_time = prev.time.values[-1:] + dt
        ns = new_field.assign_coords(time=target_time)
        fn = fnext.assign_coords(time=target_time)
        nfr = xr.merge([ns, fn])
        if "datetime" in nfr.coords: nfr = nfr.drop_vars("datetime")
        kin = [k for k in nfr.data_vars if k in prev.data_vars]
        nip = nfr[kin]
        merged = xr.concat([prev, nip], dim="time", data_vars="different")
        return merged.tail(time=input_steps)

    lat_vals = eval_ds.lat.values
    lon_vals = eval_ds.lon.values
    city_idx = {nm: (int(np.argmin(np.abs(lat_vals - cl))),
                     int(np.argmin(np.abs(lon_vals - co))),
                     float(lat_vals[int(np.argmin(np.abs(lat_vals - cl)))]),
                     float(lon_vals[int(np.argmin(np.abs(lon_vals - co)))]))
                for nm,(cl,co) in CITIES.items()}
    cos_lat = np.cos(np.deg2rad(lat_vals)); cos_lat_n = cos_lat / cos_lat.mean()
    target_vars = list(task_cfg.target_variables)
    print(f"target vars ({len(target_vars)}): {target_vars}")

    sum_sq = {m: {v: np.zeros(K, dtype=np.float64) for v in target_vars}
              for m in ['baseline','gc_mamba']}
    city_traces = {m: {c: {v: [] for v in target_vars} for c in CITIES}
                   for m in ['baseline','gc_mamba','truth']}
    anchor_times = []

    for ai, idx in enumerate(anchors):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=W + K, task_cfg=task_cfg, dt=dt)
        anchor_times.append(str(pd.Timestamp(eval_ds.time.values[idx])))

        cur_b = inp; sb = bp_state0
        cur_g = inp; sg = gm_state0
        rng_l = jax.random.PRNGKey(cfg.seed + 99 + ai)

        # Phase 1: 24-step truth warmup
        for k in range(W):
            tgt_k = tgt.isel(time=slice(k, k+1))
            frc_k = frc.isel(time=slice(k, k+1))
            rng_l, k1, k2 = jax.random.split(rng_l, 3)
            _b, sb = _bp_step(bp_params, sb, k1, cur_b, tgt_k, frc_k)
            _g, sg = _gm_step(gm_params, sg, k2, cur_g, tgt_k, frc_k)
            cur_b = _shift(cur_b, tgt_k, frc_k)
            cur_g = _shift(cur_g, tgt_k, frc_k)

        # Phase 2: 40-step closed-loop self-rollout
        for k in range(K):
            kk = W + k
            tgt_k = tgt.isel(time=slice(kk, kk+1))
            frc_k = frc.isel(time=slice(kk, kk+1))
            rng_l, k1, k2 = jax.random.split(rng_l, 3)
            bp_pred, sb = _bp_step(bp_params, sb, k1, cur_b, tgt_k, frc_k)
            gm_pred, sg = _gm_step(gm_params, sg, k2, cur_g, tgt_k, frc_k)

            for var in target_vars:
                # Force (..., lat, lon) dim order so cos_lat broadcasts over lon.
                # Surface vars: (batch=1, time=1, lat, lon) → squeeze → (lat, lon).
                # Atmos vars: (batch=1, time=1, level, lat, lon) → squeeze → (level, lat, lon).
                truth_da = tgt_k[var].astype("float32").transpose(..., "lat", "lon")
                bp_da = bp_pred[var].astype("float32").transpose(..., "lat", "lon")
                gm_da = gm_pred[var].astype("float32").transpose(..., "lat", "lon")
                truth_v = np.asarray(truth_da.values).squeeze(0).squeeze(0)
                bp_v = np.asarray(jax.device_get(bp_da.values)).squeeze(0).squeeze(0)
                gm_v = np.asarray(jax.device_get(gm_da.values)).squeeze(0).squeeze(0)
                if bp_v.ndim == 3:
                    bp_v = bp_v.mean(axis=0); gm_v = gm_v.mean(axis=0)
                    truth2 = truth_v.mean(axis=0)
                else:
                    truth2 = truth_v
                eb = bp_v - truth2; eg = gm_v - truth2
                sum_sq['baseline'][var][k] += float(((eb**2) * cos_lat_n[:,None]).mean())
                sum_sq['gc_mamba'][var][k] += float(((eg**2) * cos_lat_n[:,None]).mean())
                for nm,(li,lo,_,_) in city_idx.items():
                    city_traces['truth'][nm][var].append(float(truth2[li,lo]))
                    city_traces['baseline'][nm][var].append(float(bp_v[li,lo]))
                    city_traces['gc_mamba'][nm][var].append(float(gm_v[li,lo]))

            cur_b = _shift(cur_b, bp_pred, frc_k)
            cur_g = _shift(cur_g, gm_pred, frc_k)

        if (ai+1) % 4 == 0 or ai == 0:
            print(f"  anchor {ai+1}/{cfg.n_anchors}", flush=True)

    out = {
        'cities': CITIES, 'city_idx': city_idx,
        'anchor_times': anchor_times,
        'warmup_steps': W, 'target_steps': K, 'n_anchors': cfg.n_anchors,
        'lead_hours_per_step': 6,
        'target_variables': target_vars,
        'per_variable_per_lead': {}, 'city_traces': city_traces,
    }
    n = cfg.n_anchors
    print()
    print(f"=== Per-variable lead-mean RMSE + improvement ===")
    print(f"  {'variable':<32}{'rmse_base':>12}{'rmse_gcm':>12}{'improvement':>14}")
    for var in target_vars:
        rmse_b = np.sqrt(sum_sq['baseline'][var]/n)
        rmse_g = np.sqrt(sum_sq['gc_mamba'][var]/n)
        imp = (1 - rmse_g / np.maximum(rmse_b, 1e-12)) * 100
        out['per_variable_per_lead'][var] = dict(
            rmse_baseline=rmse_b.tolist(),
            rmse_gc_mamba=rmse_g.tolist(),
            improvement_gc_mamba_pct=imp.tolist(),
        )
        print(f"  {var:<32}{float(rmse_b.mean()):>12.4f}{float(rmse_g.mean()):>12.4f}{float(imp.mean()):>+13.2f}%")

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {cfg.out_json}")


if __name__ == "__main__":
    main()
