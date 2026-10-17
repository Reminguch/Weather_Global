#!/usr/bin/env python3
"""Score GraphCast-operational 0.25deg on the SAME 32 anchors as FengWu/ours.

Pure GraphCast inference (NO Mamba, NO param overlay -> avoids the arch clash):
per anchor load the 2 prefetched 0.25deg input frames, build a 42-step time axis
(2 inputs + 40 nan targets), extract inputs/targets/forcings, run 40x6h AR rollout
with rollout.chunked_prediction, extract z500/t850/q700 each lead, conservative-
regrid 0.25->1.5deg (SAME weights as truth), lat-weighted RMSE vs the existing
truth15 npy (z500=ch0, t850=ch1, q700=ch2). Writes gc025_scores_wb2grid.json.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]   # repo root (v22clean tree)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))
import jax                                                          # noqa: E402
import haiku as hk                                                  # noqa: E402
from graphcast import (checkpoint, graphcast as gc, casting,        # noqa: E402
                       normalization, data_utils, rollout)

FW = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/fengwu")
SCRATCH = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global")
CKPT = SCRATCH / ("data/graphcast/graphcast/params/GraphCast_operational - "
                  "ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 "
                  "- mesh 2to6 - precipitation output only.npz")
STATS = SCRATCH / "data/graphcast/graphcast/stats"
ICS = FW / "gc025_ics"
NSTEPS = 40

# ---- conservative regrid 0.25->1.5, EXACT weights matching truth builder ----
def lat_edges(c): m = 0.5*(c[1:]+c[:-1]); return np.concatenate([[-90.0], m, [90.0]])
def lon_edges(c): s = c[1]-c[0]; return np.concatenate([c-s/2, [c[-1]+s/2]])
def wlat(src, tgt):
    se, te = lat_edges(src), lat_edges(tgt)
    slo, sup = np.sin(np.deg2rad(se[:-1])), np.sin(np.deg2rad(se[1:]))
    W = np.zeros((len(tgt), len(src)))
    for j in range(len(tgt)):
        tlo, tup = np.sin(np.deg2rad(te[j])), np.sin(np.deg2rad(te[j+1]))
        W[j] = np.maximum(np.minimum(sup, tup) - np.maximum(slo, tlo), 0)
    return W / W.sum(1, keepdims=True)
def wlon(src, tgt):
    se, te = lon_edges(src), lon_edges(tgt)
    W = np.zeros((len(tgt), len(src)))
    for j in range(len(tgt)):
        tlo, tup = te[j], te[j+1]
        for sh in (-360.0, 0.0, 360.0):
            W[j] += np.maximum(np.minimum(se[1:]+sh, tup) - np.maximum(se[:-1]+sh, tlo), 0)
    return W / W.sum(1, keepdims=True)
SRC_LAT = np.arange(-90, 90.001, 0.25); SRC_LON = np.arange(0, 360, 0.25)
TGT_LAT = np.arange(-90, 90.001, 1.5);  TGT_LON = np.arange(0, 360, 1.5)
WLA, WLO = wlat(SRC_LAT, TGT_LAT), wlon(SRC_LON, TGT_LON)
def regrid(a):   # (..., 721, 1440) ascending-lat -> (..., 121, 240) ascending
    t = np.tensordot(a, WLO.T, axes=([-1], [0]))
    t = np.tensordot(t, WLA.T, axes=([-2], [0]))
    return np.moveaxis(t, -1, -2)
COSW = np.cos(np.deg2rad(TGT_LAT)); COSW = COSW / COSW.mean()
def latw_rmse(a, b): return float(np.sqrt(((a - b)**2 * COSW[:, None]).mean()))

CH = {"z500": ("geopotential", 500), "t850": ("temperature", 850),
      "q700": ("specific_humidity", 700)}
CH_ORDER = ["z500", "t850", "q700"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); cfg = ap.parse_args()
    ck = checkpoint.load(open(CKPT, "rb"), gc.CheckPoint); mc, tc = ck.model_config, ck.task_config
    stats = {n: xr.load_dataset(STATS / f"{n}.nc").astype(np.float32)
             for n in ["diffs_stddev_by_level", "mean_by_level", "stddev_by_level"]}

    def build():
        p = gc.GraphCast(mc, tc); p = casting.Bfloat16Cast(p)
        return normalization.InputsAndResiduals(
            p, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
            mean_by_level=stats["mean_by_level"], stddev_by_level=stats["stddev_by_level"])
    @hk.transform_with_state
    def fwd(i, t, f): return build()(i, targets_template=t, forcings=f)
    params = ck.params
    # jit is REQUIRED: under jit the xarray_jax leaves become traced ndarrays so
    # dataset_to_stacked's jnp.concat works; eager JaxArrayWrapper fails on newer xarray.
    _apply = jax.jit(lambda rng, i, t, f: fwd.apply(params, {}, rng, i, t, f)[0])
    def predictor_fn(rng, inputs, targets_template, forcings):
        return _apply(rng, inputs, targets_template, forcings)
    rng = jax.random.PRNGKey(0)

    manifest = json.load(open(FW / "manifest.json"))
    if cfg.smoke: manifest = manifest[:1]
    sum_sq = {c: np.zeros(NSTEPS) for c in CH_ORDER}; n_done = 0
    for m in manifest:
        tag = m["tag"]; t0 = pd.Timestamp(m["t0"])
        nc = ICS / f"{tag}.nc"
        if not nc.exists(): print(f"MISSING {tag}, skip"); continue
        real = xr.open_dataset(nc).rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
        # build 42-step abs-time axis (2 real inputs + 40 nan targets)
        t_first = t0 - pd.Timedelta(hours=6)
        times42 = t_first + pd.to_timedelta(np.arange(2 + NSTEPS) * 6, "h")
        ds = real.reindex(time=times42)                 # 2 real, 40 nan (prognostic)
        # probe coord transform -> graphcast time/datetime layout
        ds = ds.rename({"time": "datetime"}).expand_dims(batch=1)
        ds = ds.assign_coords(time=("datetime", ds.datetime.values - ds.datetime.values[0]))
        ds = ds.swap_dims({"datetime": "time"}).drop_vars("datetime")
        ds = ds.assign_coords(datetime=(("batch", "time"),
                                        ds.time.values[None] + np.datetime64(t_first)))
        # operational model OUTPUTS total_precipitation_6hr (not an input); add a
        # placeholder target var so extract can build the targets template (we score
        # z500/t850/q700 only, and targets_template is *nan'd for the AR rollout).
        if "total_precipitation_6hr" not in ds:
            ds["total_precipitation_6hr"] = xr.zeros_like(ds["2m_temperature"])
        data_utils.add_tisr_var(ds)
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds, target_lead_times=slice("6h", f"{6*NSTEPS}h"),
            input_variables=tc.input_variables, target_variables=tc.target_variables,
            forcing_variables=tc.forcing_variables, pressure_levels=tc.pressure_levels,
            input_duration=tc.input_duration)
        t_a = time.time()
        preds = rollout.chunked_prediction(
            predictor_fn, rng, inputs, targets_template=targets * np.nan, forcings=forcings)
        # preds: (batch, time=40, level, lat, lon); lat ascending
        for ci, cname in enumerate(CH_ORDER):
            var, lvl = CH[cname]
            field = preds[var].sel(level=lvl).isel(batch=0).values      # (40,721,1440) asc lat
            # truth15 npy is stored N->S (descending); our regrid emits ascending,
            # so flip lat to match truth (verified: z500 6h rmse 1404 -> 19.4 after flip).
            f15 = regrid(field)[:, ::-1, :]                             # (40, 121, 240) -> N->S
            truth = np.load(FW / "truth" / f"{tag}_truth15.npy")        # (40,3,121,240)
            for k in range(NSTEPS):
                r = latw_rmse(f15[k], truth[k, ci])
                if cfg.smoke: print(f"[smoke] {cname} step{k+1}: RMSE {r:.4f}")
                else: sum_sq[cname][k] += r**2
        n_done += 1
        print(f"[{n_done}] {tag} done ({time.time()-t_a:.1f}s)", flush=True)
    if not cfg.smoke:
        out = {c: np.sqrt(sum_sq[c] / n_done).tolist() for c in CH_ORDER}
        out["n_anchors"] = n_done; out["model"] = "GraphCast_operational_0.25deg_same32anchors"
        json.dump(out, open(FW / "gc025_scores_wb2grid.json", "w"), indent=1)
        print("wrote gc025_scores_wb2grid.json  n_anchors =", n_done)
    print("GC025 ROLLOUT SCORE: DONE")

if __name__ == "__main__":
    main()
