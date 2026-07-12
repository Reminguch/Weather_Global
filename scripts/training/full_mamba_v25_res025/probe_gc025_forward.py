#!/usr/bin/env python3
"""Phase 3A probe: real GraphCast_operational 0.25deg forward on ONE GPU.

Measures: (a) it loads + runs at all, (b) steady-state forward time = C_G,
(c) per-GPU peak memory. Unsharded — this anchors the sharding budget.
Uses 3 real ERA5 timesteps (2 inputs + 1 target) extracted from WB2.
"""
import sys
import time
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))
import jax                                            # noqa: E402
import haiku as hk                                    # noqa: E402
from graphcast import (checkpoint, graphcast as gc, casting, normalization,
                       data_utils, xarray_jax)        # noqa: E402

SCRATCH = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global")
CKPT = SCRATCH / ("data/graphcast/graphcast/params/GraphCast_operational - "
                  "ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 "
                  "- mesh 2to6 - precipitation output only.npz")
DATA = SCRATCH / "data/graphcast/graphcast/dataset/probe_025_3steps.nc"
STATS = SCRATCH / "data/graphcast/graphcast/stats"

with open(CKPT, "rb") as f:
    ckpt = checkpoint.load(f, gc.CheckPoint)
mc, tc = ckpt.model_config, ckpt.task_config
print("model_config:", mc)

stats = {n: xr.load_dataset(STATS / f"{n}.nc").astype(np.float32)
         for n in ["diffs_stddev_by_level", "mean_by_level", "stddev_by_level"]}

ds = xr.load_dataset(DATA)
# WB2 full-res zarr uses latitude/longitude names -> graphcast wants lat/lon
ds = ds.rename({"latitude": "lat", "longitude": "lon"})
# WB2 layout -> graphcast layout: add batch dim, datetime coords, ascending lat
ds = ds.sortby("lat")
ds = ds.rename({"time": "datetime"})
ds = ds.expand_dims(batch=1)
ds = ds.assign_coords(time=("datetime", ds.datetime.values - ds.datetime.values[0]))
ds = ds.swap_dims({"datetime": "time"})
ds = ds.drop_vars("datetime")
ds = ds.assign_coords(datetime=(("batch", "time"), ds.time.values[None] + np.datetime64("2022-01-01T00")))
data_utils.add_tisr_var(ds)          # compute TISR from datetime (not in WB2 zarr)
# keep datetime: extract_inputs_targets_forcings calls add_derived_vars internally

inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    ds, target_lead_times=slice("6h", "6h"),
    input_variables=tc.input_variables, target_variables=tc.target_variables,
    forcing_variables=tc.forcing_variables, pressure_levels=tc.pressure_levels,
    input_duration=tc.input_duration)
print("inputs vars:", len(inputs.data_vars), " targets:", len(targets.data_vars),
      " forcings:", len(forcings.data_vars))

def build(mc, tc):
    p = gc.GraphCast(mc, tc)
    p = casting.Bfloat16Cast(p)
    p = normalization.InputsAndResiduals(
        p, diffs_stddev_by_level=stats["diffs_stddev_by_level"],
        mean_by_level=stats["mean_by_level"], stddev_by_level=stats["stddev_by_level"])
    return p

@hk.transform_with_state
def fwd(i, t, f):
    return build(mc, tc)(i, targets_template=t * np.nan, forcings=f)

params, state = ckpt.params, {}
apply_j = jax.jit(lambda i, t, f: fwd.apply(params, state, jax.random.PRNGKey(0), i, t, f))

for it in range(3):
    t0 = time.time()
    out, _ = apply_j(inputs, targets, forcings)
    jax.block_until_ready(xarray_jax.unwrap_data(out["2m_temperature"]))
    dt = time.time() - t0
    ms = jax.local_devices()[0].memory_stats()
    print(f"iter{it}: {dt:.2f}s  peak={ms['peak_bytes_in_use']/2**30:.2f}GiB "
          f"inuse={ms['bytes_in_use']/2**30:.2f}GiB")
print("t2m out mean:", float(out["2m_temperature"].mean()), "(should be ~270-290K)")
print("PHASE3A GC-FORWARD PROBE: DONE")
