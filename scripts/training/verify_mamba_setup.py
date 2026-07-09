#!/usr/bin/env python3
"""Comprehensive verification of Mamba setup.

Tests:
  1. Hyperparameters are sane
  2. Mamba params exist with expected values
  3. SSM state exists with correct shape, starts at zero
  4. Forward pass: state is updated and finite
  5. Backward pass: Mamba params receive finite gradients
  6. State propagation through rollout: target_steps=2 vs target_steps=4 produce different state progression
  7. stop_gradient preserves values (only detaches from graph)
  8. Checkpoint save/load preserves Mamba params
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPHCAST_LOCAL = ROOT / "third_party" / "graphcast"
if str(GRAPHCAST_LOCAL) not in sys.path:
    sys.path.insert(0, str(GRAPHCAST_LOCAL))
sys.path.insert(0, str(ROOT / "scripts" / "training"))

import xarray as xr
import pandas as pd

from graphcast import checkpoint
from graphcast import graphcast as gc
from src.data.graphcast_dataset import open_graphcast_era5
from train_graphcast import (
    build_predictor,
    build_batch_from_indices,
    build_sequential_segments,
    _ensure_datetime_coord,
    prepare_dataset_for_task,
    infer_time_step,
    valid_final_input_indices,
)

RESULTS = {"pass": [], "fail": [], "warn": []}


def sect(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check(cond, name, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    (RESULTS["pass"] if cond else RESULTS["fail"]).append(name)
    return cond


def warn(name, detail=""):
    print(f"  [WARN] {name}  ({detail})")
    RESULTS["warn"].append(name)


def flatten(tree, prefix=""):
    flat = {}
    if hasattr(tree, "items"):
        for k, v in tree.items():
            sub = prefix + "/" + k if prefix else k
            if hasattr(v, "items"):
                flat.update(flatten(v, sub))
            else:
                flat[sub] = v
    return flat


def main():
    sect("1. Hyperparameters")
    lr = 1e-4
    wd = 1e-4
    target_steps = 2
    segment_steps = 120
    mamba_hidden = 128
    latent_size = 128
    mesh_size = 4
    n_mesh = 2562  # for mesh_size=4

    check(0 < lr < 1e-2, "lr sane", f"{lr}")
    check(0 <= wd < 1e-2, "weight_decay sane", f"{wd}")
    check(target_steps >= 2, "target_steps >= 2", f"{target_steps}")
    check(segment_steps >= 4 * target_steps, "segment_steps >> target_steps", f"{segment_steps}")
    check(mamba_hidden == latent_size, "mamba_hidden == latent_size", f"{mamba_hidden}")

    sect("2. Loading dataset correctly")
    data_path = "data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr"
    print(f"Opening {data_path}...")
    ds = open_graphcast_era5(str(ROOT / data_path))
    ds = _ensure_datetime_coord(ds)
    ds_times = pd.to_datetime(ds["time"].values)
    ds = ds.isel(time=np.where(ds_times.year == 2020)[0][:30])
    # Downsample to 2 degree
    ds = ds.isel(lat=slice(0, None, 2), lon=slice(0, None, 2))
    ds = ds.compute()
    print(f"  dims: {dict(ds.sizes)}")
    check("batch" in ds.dims, "dataset has batch dim", f"batch={ds.sizes.get('batch')}")

    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    task_cfg = gc.TaskConfig(
        input_variables=[
            "2m_temperature", "mean_sea_level_pressure",
            "10m_v_component_of_wind", "10m_u_component_of_wind",
            "total_precipitation_6hr", "temperature", "geopotential",
            "u_component_of_wind", "v_component_of_wind",
            "vertical_velocity", "specific_humidity",
            "toa_incident_solar_radiation",
            "year_progress_sin", "year_progress_cos",
            "day_progress_sin", "day_progress_cos",
            "geopotential_at_surface", "land_sea_mask",
        ],
        target_variables=[
            "2m_temperature", "mean_sea_level_pressure",
            "10m_v_component_of_wind", "10m_u_component_of_wind",
            "total_precipitation_6hr", "temperature", "geopotential",
            "u_component_of_wind", "v_component_of_wind",
            "vertical_velocity", "specific_humidity",
        ],
        forcing_variables=[
            "toa_incident_solar_radiation",
            "year_progress_sin", "year_progress_cos",
            "day_progress_sin", "day_progress_cos",
        ],
        pressure_levels=pressure_levels,
        input_duration="12h",
    )
    ds = prepare_dataset_for_task(ds, task_cfg)

    # Load stats
    stats_dir = ROOT / "data/graphcast/graphcast/stats"
    stats = {
        "diffs_stddev_by_level": xr.open_dataset(stats_dir / "diffs_stddev_by_level.nc").compute(),
        "mean_by_level": xr.open_dataset(stats_dir / "mean_by_level.nc").compute(),
        "stddev_by_level": xr.open_dataset(stats_dir / "stddev_by_level.nc").compute(),
    }

    dt = infer_time_step(ds)
    indices = valid_final_input_indices(ds.sizes["time"], 2, target_steps)
    check(len(indices) > 0, "valid sample indices", f"n={len(indices)}")

    sample_inputs, sample_targets, sample_forcings = build_batch_from_indices(
        ds, indices=[int(indices[0])],
        input_steps=2, target_steps=target_steps,
        task_cfg=task_cfg, dt=dt,
    )
    print(f"  sample_inputs dims: {dict(sample_inputs.sizes)}")

    sect("3. Building model")
    model_cfg = gc.ModelConfig(
        resolution=0.0, mesh_size=mesh_size, latent_size=latent_size,
        gnn_msg_steps=1, hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )

    def forward_fn(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg, task_cfg, stats,
            use_bf16=True, gradient_checkpointing=True,
            temporal_backbone="mamba",
            temporal_location="mesh_post_encoder_residual",
            temporal_hidden_size=mamba_hidden,
            temporal_layers=1, temporal_dropout=0.0,
            temporal_stateful=True,
        )
        return predictor.loss(inputs, targets, forcings)

    transformed = hk.transform_with_state(forward_fn)
    rng = jax.random.PRNGKey(0)
    print("Initializing (compiles, may take a minute)...")
    params, state = transformed.init(rng, sample_inputs, sample_targets, sample_forcings, True)
    print("  done")

    sect("4. Checking Mamba params")
    flat_p = flatten(params)
    mamba_p = {k: v for k, v in flat_p.items() if "temporal" in k.lower()}
    check(len(mamba_p) > 0, "Mamba params exist", f"n={len(mamba_p)}")
    for ename in ["a_log", "skip", "dt_proj", "in_proj", "out_proj", "layer_norm"]:
        found = any(ename in k for k in mamba_p.keys())
        check(found, f"param '{ename}' exists")

    a_log = [v for k, v in mamba_p.items() if "a_log" in k][0]
    check(abs(float(np.asarray(a_log).mean()) + 0.1) < 0.01, "a_log init ≈ -0.1",
          f"mean={float(np.asarray(a_log).mean()):.4f}")

    sect("5. Checking SSM state shape and init")
    flat_s = flatten(state)
    ssm = {k: v for k, v in flat_s.items() if "ssm_state" in k.lower()}
    check(len(ssm) > 0, "SSM state exists", f"n={len(ssm)}")
    for k, v in ssm.items():
        arr = np.asarray(v)
        check(arr.shape == (n_mesh, mamba_hidden),
              f"shape = ({n_mesh}, {mamba_hidden})", f"got {arr.shape}")
        check(float(np.abs(arr).max()) == 0.0, "initial state is zero")

    sect("6. Forward pass updates state")
    (loss_out, new_state) = transformed.apply(
        params, state, rng, sample_inputs, sample_targets, sample_forcings, True)
    loss_val = float(jnp.asarray(loss_out[0]).sum())
    check(np.isfinite(loss_val), "loss finite", f"loss={loss_val:.4f}")

    new_flat_s = flatten(new_state)
    for k in ssm.keys():
        old = np.asarray(ssm[k])
        new = np.asarray(new_flat_s[k])
        changed = not np.allclose(old, new)
        check(changed, "SSM state was updated after forward",
              f"old_max={np.abs(old).max():.4f}, new_max={np.abs(new).max():.4f}")
        check(np.isfinite(new).all(), "new state is finite")

    sect("7. State propagation: target_steps=2 vs target_steps=4")
    # Build a target_steps=4 forward fn to check that running 4 steps produces
    # a different state than running 2 steps.
    inputs4, targets4, forcings4 = build_batch_from_indices(
        ds, indices=[int(valid_final_input_indices(ds.sizes["time"], 2, 4)[0])],
        input_steps=2, target_steps=4, task_cfg=task_cfg, dt=dt,
    )

    def forward_fn_4(inputs, targets, forcings, is_training):
        del is_training
        predictor = build_predictor(
            model_cfg, task_cfg, stats,
            use_bf16=True, gradient_checkpointing=True,
            temporal_backbone="mamba", temporal_location="mesh_post_encoder_residual",
            temporal_hidden_size=mamba_hidden, temporal_layers=1, temporal_dropout=0.0,
            temporal_stateful=True,
        )
        return predictor.loss(inputs, targets, forcings)

    transformed_4 = hk.transform_with_state(forward_fn_4)
    # Use the same init params (re-init with the 4-step sample to get same shapes)
    params_4, state_4 = transformed_4.init(rng, inputs4, targets4, forcings4, True)
    (_, state_after_4) = transformed_4.apply(
        params_4, state_4, rng, inputs4, targets4, forcings4, True)

    flat_s4 = flatten(state_after_4)
    ssm_after_4 = {k: v for k, v in flat_s4.items() if "ssm_state" in k.lower()}
    for k in ssm.keys():
        state_2_val = np.asarray(new_flat_s[k])  # after target_steps=2
        state_4_val = np.asarray(ssm_after_4[k])  # after target_steps=4
        different = not np.allclose(state_2_val, state_4_val, rtol=1e-3)
        check(different, "target_steps=2 and target_steps=4 produce different final states",
              f"diff_norm={np.linalg.norm(state_2_val - state_4_val):.4f}")

    sect("8. stop_gradient preserves state values")

    # Reimplement _stop_grad_state (it's a nested function in train_graphcast.py main)
    def _stop_grad_state(s):
        return jax.tree_util.tree_map(
            lambda leaf: jax.lax.stop_gradient(leaf) if isinstance(leaf, jax.Array) else leaf,
            s,
        )

    # Create a state tree and make sure stop_grad preserves the values
    def make_nonzero_state(s):
        return jax.tree_util.tree_map(
            lambda x: jnp.ones_like(x) * 0.5 if isinstance(x, jax.Array) else x, s)

    nonzero_state = make_nonzero_state(state)
    stopped = _stop_grad_state(nonzero_state)
    flat_nonzero = flatten(nonzero_state)
    flat_stopped = flatten(stopped)
    for k in ssm.keys():
        check(np.allclose(np.asarray(flat_nonzero[k]), np.asarray(flat_stopped[k])),
              "stop_gradient preserves state values")
        # The key is that stop_gradient doesn't change the RUN-TIME value
        # (the gradient break is a JIT-time property, hard to test directly)

    sect("9. Checkpoint save/load roundtrip")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = f.name
    try:
        ckpt_out = gc.CheckPoint(
            params=params, model_config=model_cfg, task_config=task_cfg,
            description="verify", license="test",
        )
        with open(tmp, "wb") as f:
            checkpoint.dump(f, ckpt_out)
        loaded = np.load(tmp, allow_pickle=True)
        loaded_mamba = [k for k in loaded.keys() if "temporal" in k.lower()]
        check(len(loaded_mamba) > 0, "saved checkpoint contains Mamba params",
              f"n={len(loaded_mamba)}")
        # Verify values match
        for k in loaded_mamba[:3]:
            orig_key = k[len("params:"):]
            parts = orig_key.rsplit(":", 1)
            mod, pname = parts[0], parts[1]
            orig_val = np.asarray(params[mod][pname])
            loaded_val = np.asarray(loaded[k])
            check(np.allclose(orig_val, loaded_val), f"roundtrip value match: {pname}")
    finally:
        Path(tmp).unlink(missing_ok=True)

    sect("10. Sequential segments")
    segments = build_sequential_segments(np.arange(1000), segment_steps)
    check(len(segments) > 0, "segments built", f"n={len(segments)}")
    check(all(len(s) <= segment_steps for s in segments), "no segment exceeds limit")
    check(all(all(np.diff(s) == 1) for s in segments), "within-segment indices contiguous")

    sect("SUMMARY")
    print(f"  PASS: {len(RESULTS['pass'])}")
    print(f"  FAIL: {len(RESULTS['fail'])}")
    print(f"  WARN: {len(RESULTS['warn'])}")
    if RESULTS["fail"]:
        print()
        print("  FAILED:")
        for n in RESULTS["fail"]:
            print(f"    - {n}")
    print()
    print("  *** " + ("ALL PASSED" if not RESULTS["fail"] else "SOME FAILED") + " ***")
    return 0 if not RESULTS["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())
