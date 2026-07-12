#!/usr/bin/env python3
"""Phase 3B probe: residual head (2-msg GNN + interleaved Mamba) fwd+BWD at
0.25deg / mesh-6, ONE GPU. Remat ladder via REMAT env: none | whole.

Measures C_R (loss+grad step time) and peak memory — the deciding number for
whether a full train step fits one card. Uses the same real 3-step nc as 3A;
residual target = random (memory probe, not science).
"""
import os
import sys
import time
import types
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))
import jax                                            # noqa: E402
import haiku as hk                                    # noqa: E402
from graphcast import (checkpoint, graphcast as gc, casting,
                       data_utils)                    # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import (   # noqa: E402
    GCResidualWithZeroHead, _attach_temporal)
from src.models.graphcast.training.core.model import DirectResidualNormalizer, scalarize_loss  # noqa: E402
import dataclasses                                    # noqa: E402

REMAT = os.environ.get("REMAT", "none")
SCRATCH = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global")
CKPT = SCRATCH / ("data/graphcast/graphcast/params/GraphCast_operational - "
                  "ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 "
                  "- mesh 2to6 - precipitation output only.npz")
DATA = SCRATCH / "data/graphcast/graphcast/dataset/probe_025_3steps.nc"
STATS = SCRATCH / "data/graphcast/graphcast/stats"

with open(CKPT, "rb") as f:
    ck = checkpoint.load(f, gc.CheckPoint)
mc, tc = ck.model_config, ck.task_config

stats = {n: xr.load_dataset(STATS / f"{n}.nc").astype(np.float32)
         for n in ["diffs_stddev_by_level", "mean_by_level", "stddev_by_level"]}

ds = xr.load_dataset(DATA).rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
ds = ds.rename({"time": "datetime"}).expand_dims(batch=1)
ds = ds.assign_coords(time=("datetime", ds.datetime.values - ds.datetime.values[0]))
ds = ds.swap_dims({"datetime": "time"}).drop_vars("datetime")
ds = ds.assign_coords(datetime=(("batch", "time"),
                                ds.time.values[None] + np.datetime64("2022-01-01T00")))
data_utils.add_tisr_var(ds)
inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    ds, target_lead_times=slice("6h", "6h"),
    input_variables=tc.input_variables, target_variables=tc.target_variables,
    forcing_variables=tc.forcing_variables, pressure_levels=tc.pressure_levels,
    input_duration=tc.input_duration)

# random residual target with target shapes (normalized-space magnitude ~O(1)
# after DirectResidualNormalizer divides by diffs_std — raw ~ diffs_std scale)
rngnp = np.random.default_rng(0)
res_target = targets.copy(deep=True)
for v in res_target.data_vars:
    res_target[v].values = rngnp.normal(
        size=res_target[v].shape).astype(np.float32) * 1.0

cfg = types.SimpleNamespace(
    temporal_location="mesh_processor_interleaved", temporal_hidden_size=128,
    temporal_d_inner=128, temporal_d_state=16, temporal_d_conv=4,
    temporal_dt_rank="auto", temporal_layers=2, temporal_bias=True,
    temporal_conv_bias=True, temporal_dropout=0.0, temporal_zero_init_out=True)

mc_res = dataclasses.replace(mc, gnn_msg_steps=2)   # residual head: 2 msg steps

def build():
    p = GCResidualWithZeroHead(mc_res, tc)
    _attach_temporal(p, cfg)
    p = casting.Bfloat16Cast(p)
    p = DirectResidualNormalizer(
        p, stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"])
    return p

@hk.transform_with_state
def loss_fn(i, t, f):
    pred = build()
    if REMAT == "whole":
        def call(i_, t_, f_):
            l, _ = pred.loss(i_, t_, forcings=f_)
            return scalarize_loss(l)
        return hk.remat(call)(i, t, f)
    l, _ = pred.loss(i, t, forcings=f)
    return scalarize_loss(l)

rng = jax.random.PRNGKey(0)
params, state = loss_fn.init(rng, inputs, res_target, forcings)
n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"REMAT={REMAT}  residual params: {n_par/1e6:.2f}M")

def scalar_loss(p, i, t, f):
    l, _ = loss_fn.apply(p, state, rng, i, t, f)
    return l

vg = jax.jit(jax.value_and_grad(lambda p: scalar_loss(p, inputs, res_target, forcings)))
for it in range(3):
    t0 = time.time()
    l, g = vg(params)
    jax.block_until_ready(jax.tree_util.tree_leaves(g)[0])
    dt = time.time() - t0
    ms = jax.local_devices()[0].memory_stats()
    print(f"iter{it}: {dt:.2f}s  loss={float(l):.4f}  "
          f"peak={ms['peak_bytes_in_use']/2**30:.2f}GiB")
print("PHASE3B RESIDUAL BWD PROBE: DONE")
