#!/usr/bin/env python3
"""Run a trained MZ model (meshed, simplified or FullMamba) forward on a few
contiguous eval segments and save baseline/corrected/truth tensors for later
plotting.

The run config (``run_config.json``) is inspected to decide whether to build
``MZResidualMeshedMamba`` (simplified) or ``MZResidualFullMambaMeshed``
(d_state>1). Works for any run that has ``meshed=True`` in the config.

Output: a single ``.npz`` file containing:
  baseline [seg, T, 1, lat, lon, F]
  corrected [seg, T, 1, lat, lon, F]
  truth [seg, T, 1, lat, lon, F]
  times [seg, T]   (numpy datetime64)
  lat [lat], lon [lon]
  feature_order (tuple of variable names)
  feature_slices (dict of var -> slice)
  pressure_levels
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
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_DIR = ROOT / "scripts" / "training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import train_graphcast as base_train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Directory with mz_residual_stepNNN.pkl + run_config.json")
    p.add_argument("--step", type=int, default=3000)
    p.add_argument("--num-segments", type=int, default=8)
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--locations", default="",
                   help="Comma-separated list of name:lat:lon triples to extract "
                        "time series for. e.g. 'Beijing:39.9:116.4,NYC:40.7:286.0'. "
                        "If empty, saves the full grid (caution: ~2 GB).")
    p.add_argument("--lat-stride", type=int, default=1,
                   help="Subsample latitude by this stride to shrink output size.")
    p.add_argument("--lon-stride", type=int, default=1,
                   help="Subsample longitude by this stride to shrink output size.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg_blob = json.load(open(run_dir / "run_config.json"))
    cfg_dict = cfg_blob["config"]

    # Build a simple namespace for attribute access.
    class _C:
        pass
    cfg = _C()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    print(f"[infer] run_dir = {run_dir}")
    print(f"[infer] meshed={cfg.meshed}  full_mamba={getattr(cfg, 'full_mamba', False)}  "
          f"d_state={getattr(cfg, 'd_state', None)}  hidden_size={cfg.hidden_size}  "
          f"layers={cfg.layers}")

    # Dispatch MZ module.
    if cfg.meshed and getattr(cfg, "full_mamba", False):
        from src.models.mz.full_mamba import (
            MZResidualFullMambaConfig, MZResidualFullMambaMeshed,
        )
        variant = "full_mamba"
    elif cfg.meshed:
        from src.models.mz.meshed_mamba import (
            MZResidualMeshedConfig, MZResidualMeshedMamba,
        )
        variant = "meshed_simplified"
    else:
        from src.models.mz.grid_mamba import MZResidualConfig, MZResidualMamba
        variant = "grid"
    from src.models.mz.meshed_mamba import build_grid_mesh_projections
    print(f"[infer] variant = {variant}")

    # Determine which variable set the run used.
    full_variables = getattr(cfg, "full_variables", None)
    if full_variables is None:
        # Fall back to resolved_variables dumped in run_config
        resolved_vars = cfg_blob.get("resolved_variables", None)
        full_variables = (resolved_vars is not None and len(resolved_vars) > 4)

    if full_variables:
        RESOLVED_VARIABLES = (
            "2m_temperature", "mean_sea_level_pressure",
            "10m_u_component_of_wind", "10m_v_component_of_wind",
            "total_precipitation_6hr",
            "temperature", "geopotential", "u_component_of_wind",
            "v_component_of_wind", "vertical_velocity", "specific_humidity",
        )
    else:
        RESOLVED_VARIABLES = (
            "mean_sea_level_pressure", "geopotential",
            "u_component_of_wind", "v_component_of_wind",
        )
    PRESSURE_LEVEL_VARS = {
        "geopotential", "u_component_of_wind", "v_component_of_wind",
        "temperature", "vertical_velocity", "specific_humidity",
    }
    print(f"[infer] full_variables={full_variables}  n_vars={len(RESOLVED_VARIABLES)}")

    # Rebuild the baseline predictor machinery (same pipeline as training).
    ckpt = base_train.load_graphcast_checkpoint(Path(cfg.baseline_ckpt))
    task_cfg = ckpt.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg = dataclasses.replace(
        ckpt.model_config, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
    )
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

    # Build feature layout.
    def resolved_feature_layout():
        layout, slices, cursor = [], {}, 0
        for name in RESOLVED_VARIABLES:
            width = len(task_cfg.pressure_levels) if name in PRESSURE_LEVEL_VARS else 1
            slices[name] = slice(cursor, cursor + width)
            layout.append(name)
            cursor += width
        return tuple(layout), slices, cursor

    feature_order, feature_slices, feature_dim = resolved_feature_layout()

    # Build normalization tensors from stats.
    def _stack(ds):
        chunks = []
        for name in feature_order:
            da = ds[name]
            if "level" in da.dims:
                arr = da.sel(level=list(task_cfg.pressure_levels)).values.astype(np.float32)
            else:
                arr = np.asarray([float(da.values)], dtype=np.float32)
            chunks.append(arr)
        return np.concatenate(chunks, axis=0)

    diffs_std_f = _stack(norm_stats["diffs_stddev_by_level"])
    mean_f = _stack(norm_stats["mean_by_level"])
    std_f = _stack(norm_stats["stddev_by_level"])

    input_mean_f = jnp.asarray(mean_f, dtype=jnp.float32)
    input_std_f = jnp.asarray(std_f, dtype=jnp.float32)
    output_denorm_f = jnp.asarray(diffs_std_f, dtype=jnp.float32)
    residual_input_std_f = jnp.asarray(diffs_std_f, dtype=jnp.float32)

    # Import filter / segment helpers. They live in train_mz_residual_memory_resume
    # (same structure across train scripts).
    sys.path.insert(0, str(TRAIN_DIR))
    import importlib
    import train_mz_residual_memory_resume as mz_train

    eval_indices_raw = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, 1
    )
    eval_indices = mz_train._filter_time_continuous_indices(
        eval_ds, eval_indices_raw, input_steps=input_steps, target_steps=1, dt=dt,
    )
    eval_segments = [
        seg for seg in mz_train._time_continuous_segments(
            eval_ds, eval_indices, cfg.segment_steps, dt,
        ) if len(seg) == cfg.segment_steps
    ]
    print(f"[infer] eval_segments available = {len(eval_segments)}")

    # Baseline predictor
    def baseline_predict_fn(inputs, targets, forcings, is_training):
        predictor = base_train.build_predictor(
            model_cfg, task_cfg, norm_stats,
            use_bf16=(cfg.baseline_precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone="none", temporal_location="mesh_post_encoder",
            temporal_hidden_size=model_cfg.latent_size,
            temporal_layers=1, temporal_dropout=0.0,
        )
        return predictor(inputs, targets_template=targets, forcings=forcings,
                         is_training=is_training)

    baseline_predict = hk.transform_with_state(baseline_predict_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    # Init baseline state.
    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(eval_indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt,
    )
    sample_inputs = mz_train._to_jax_dataset(sample_inputs)
    sample_targets = mz_train._to_jax_dataset(sample_targets)
    sample_forcings = mz_train._to_jax_dataset(sample_forcings)
    _, baseline_state = baseline_predict.init(
        rng, sample_inputs, sample_targets, sample_forcings, False,
    )
    baseline_params = ckpt.params

    # Build MZ model.
    lat_deg = np.asarray(eval_ds.lat.values, dtype=np.float64)
    lon_deg = np.asarray(eval_ds.lon.values, dtype=np.float64)
    proj_arrays, n_mesh = build_grid_mesh_projections(
        lat_deg=lat_deg, lon_deg=lon_deg,
        mesh_size=getattr(cfg, "mz_mesh_size", 5),
        n_grid_neighbors=getattr(cfg, "n_grid_neighbors", 6),
        n_mesh_neighbors=getattr(cfg, "n_mesh_neighbors", 3),
    )
    print(f"[infer] built mesh: n_mesh={n_mesh}")

    if variant == "full_mamba":
        mz_cfg = MZResidualFullMambaConfig(
            input_size=feature_dim * 2, output_size=feature_dim,
            hidden_size=cfg.hidden_size, d_state=cfg.d_state, expand=cfg.expand,
            layers=cfg.layers, dropout=getattr(cfg, "dropout", 0.0),
            a_log_init_min=cfg.a_log_init_min, a_log_init_max=cfg.a_log_init_max,
        )
        mk = lambda: MZResidualFullMambaMeshed(
            mz_cfg, n_mesh_nodes=n_mesh, **proj_arrays,
        )
    elif variant == "meshed_simplified":
        mz_cfg = MZResidualMeshedConfig(
            input_size=feature_dim * 2, output_size=feature_dim,
            hidden_size=cfg.hidden_size, layers=cfg.layers,
            dropout=getattr(cfg, "dropout", 0.0),
            a_log_init=getattr(cfg, "a_log_init", -0.1),
        )
        mk = lambda: MZResidualMeshedMamba(mz_cfg, n_mesh_nodes=n_mesh, **proj_arrays)
    else:
        raise ValueError(f"unsupported variant: {variant}")

    def residual_objective(seq_inputs, baseline_next, truth_next, is_training):
        current_state, prev_residual = jnp.split(seq_inputs, 2, axis=-1)
        current_state_n = (current_state - input_mean_f[None, None, None, None, :]) / input_std_f[None, None, None, None, :]
        prev_residual_n = prev_residual / residual_input_std_f[None, None, None, None, :]
        seq_inputs_n = jnp.concatenate([current_state_n, prev_residual_n], axis=-1)
        model = mk()
        pred_residual_n = model(seq_inputs_n, is_training=is_training)
        pred_residual = pred_residual_n * output_denorm_f[None, None, None, None, :]
        corrected = baseline_next + pred_residual
        return {"corrected": corrected, "pred_residual": pred_residual}

    residual_model = hk.transform(residual_objective)
    mem_params = pickle.load(open(run_dir / f"mz_residual_step{args.step}.pkl", "rb"))

    @jax.jit
    def infer_step(params, key, seq_inputs, baseline_next, truth_next):
        return residual_model.apply(params, key, seq_inputs, baseline_next, truth_next, False)

    num = min(args.num_segments, len(eval_segments))
    baseline_out, corrected_out, truth_out, times_out = [], [], [], []
    seg_first_indices = []
    time_coord = pd.DatetimeIndex(pd.to_datetime(eval_ds.time.values))

    for i in range(num):
        seg = eval_segments[i]
        rng, key = jax.random.split(rng)
        seq_inputs, baseline_next, truth_next, _tf_mask = mz_train._segment_to_tensors(
            baseline_predict, baseline_params, baseline_state, key,
            eval_ds, seg, input_steps=input_steps,
            task_cfg=task_cfg, dt=dt, feature_order=feature_order,
        )
        out = infer_step(mem_params, key, seq_inputs, baseline_next, truth_next)
        baseline_out.append(np.asarray(baseline_next, dtype=np.float32))
        corrected_out.append(np.asarray(out["corrected"], dtype=np.float32))
        truth_out.append(np.asarray(truth_next, dtype=np.float32))
        seg_arr = np.asarray(seg, dtype=np.int64)
        target_idx = seg_arr + 1
        times_out.append(time_coord[target_idx].values)
        seg_first_indices.append(int(seg_arr[0]))
        print(f"[infer] seg {i}: idx={int(seg_arr[0])} time_range={time_coord[target_idx[0]]} .. {time_coord[target_idx[-1]]}")

    baseline_arr = np.stack(baseline_out, axis=0)   # [seg, T, 1, lat, lon, F]
    corrected_arr = np.stack(corrected_out, axis=0)
    truth_arr = np.stack(truth_out, axis=0)
    times_arr = np.stack(times_out, axis=0)

    # ---- Optionally subsample / extract specific locations to keep file size
    # small (full grid is ~2.4 GB for 8 segments at 1° × 11 vars).
    save_lat = lat_deg.astype(np.float32)
    save_lon = lon_deg.astype(np.float32)
    if args.locations:
        # Parse "Name:lat:lon,Name:lat:lon,..." into arrays
        names, lats, lons = [], [], []
        for tok in args.locations.split(","):
            tok = tok.strip()
            if not tok:
                continue
            parts = tok.split(":")
            if len(parts) != 3:
                raise ValueError(f"--locations entry must be name:lat:lon, got {tok!r}")
            names.append(parts[0])
            lats.append(float(parts[1]))
            lons.append(float(parts[2]))
        # Map each requested location to nearest grid index
        lat_idxs = np.array([int(np.argmin(np.abs(lat_deg - L))) for L in lats], dtype=np.int32)
        lon_idxs = np.array([int(np.argmin(np.abs(lon_deg - L))) for L in lons], dtype=np.int32)
        # Extract: result shape [seg, T, 1, n_loc, F]
        baseline_arr = baseline_arr[:, :, :, lat_idxs, lon_idxs, :]
        corrected_arr = corrected_arr[:, :, :, lat_idxs, lon_idxs, :]
        truth_arr = truth_arr[:, :, :, lat_idxs, lon_idxs, :]
        save_lat = np.array([lat_deg[i] for i in lat_idxs], dtype=np.float32)
        save_lon = np.array([lon_deg[i] for i in lon_idxs], dtype=np.float32)
        loc_names = np.asarray(names)
        print(
            f"[infer] extracted {len(names)} locations: {list(zip(names, save_lat.tolist(), save_lon.tolist()))}"
        )
    elif args.lat_stride > 1 or args.lon_stride > 1:
        # Subsample the grid dimensions
        baseline_arr = baseline_arr[:, :, :, ::args.lat_stride, ::args.lon_stride, :]
        corrected_arr = corrected_arr[:, :, :, ::args.lat_stride, ::args.lon_stride, :]
        truth_arr = truth_arr[:, :, :, ::args.lat_stride, ::args.lon_stride, :]
        save_lat = save_lat[::args.lat_stride]
        save_lon = save_lon[::args.lon_stride]
        loc_names = np.asarray([])
        print(f"[infer] subsampled lat×lon by {args.lat_stride}×{args.lon_stride}")
    else:
        loc_names = np.asarray([])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        baseline=baseline_arr,
        corrected=corrected_arr,
        truth=truth_arr,
        times=times_arr,
        lat=save_lat,
        lon=save_lon,
        location_names=loc_names,
        feature_order=np.asarray(list(feature_order)),
        feature_slices=np.asarray([[sl.start, sl.stop] for _, sl in feature_slices.items()]),
        pressure_levels=np.asarray(list(task_cfg.pressure_levels)),
        seg_first_indices=np.asarray(seg_first_indices),
    )
    print(f"[infer] saved: {out_path}  baseline shape = {baseline_arr.shape}")


if __name__ == "__main__":
    main()
