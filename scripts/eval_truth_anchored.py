#!/usr/bin/env python3
"""Truth-anchored AR eval (Ilya-style) for v3/v4 residual Mamba ckpts.

Per chunk:
  1. Build inputs (last input_steps cycles), targets+forcings (W+S+L cycles ahead).
  2. WARMUP W steps: at each step, run baseline + residual one step,
     update rolling_inputs ← TRUTH (not model pred). Spins up SSM hidden +
     accumulates rolling state.
  3. TRUNK S anchors: at each anchor, FORK the context (clone rolling_inputs,
     prev_residual, h_states) and run a free AR rollout L steps where each
     step's rolling_inputs ← MODEL_FULL_PRED (not truth) — this is where
     baseline and residual are both AR self-fed, matching deployment.
     Accumulate per-lead metric vs branch-truth. Then advance trunk by 1
     truth-step.

Differs from scripts/eval_per_level.py in two essential ways:
  * baseline self-feeds with model prediction during branch rollouts (the
    eval_per_level.py path pre-computes baseline at every position from the
    truth-fed inputs; deployment-honest setup needs baseline AR).
  * 1 warmup window seeds S anchor branches → S× efficiency per warmup.

Output JSON matches eval_per_level.py layout (per_channel +
per_channel_per_leadtime) so existing comparison helpers work.
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
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_DIR = ROOT / "scripts" / "training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
TRAIN_V3_DIR = ROOT / "scripts" / "training" / "full_mamba_v3"
if str(TRAIN_V3_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_V3_DIR))

import train_graphcast as base_train  # noqa: E402
import train_mz_fullmamba_v3 as mz_train  # noqa: E402

N_INPUT_STEPS = 2  # GraphCast standard


def _update_inputs(inputs: xr.Dataset, next_frame: xr.Dataset) -> xr.Dataset:
    """Mirror Ilya's helper: append next_frame to inputs (only matching vars),
    keep last `time=num_inputs` cycles, preserve original time coord values."""
    num_inputs = inputs.sizes["time"]
    predicted_or_forced_inputs = next_frame[list(inputs.keys())]
    return (
        xr.concat([inputs, predicted_or_forced_inputs], dim="time")
        .tail(time=num_inputs)
        .assign_coords(time=inputs.coords["time"])
    )


def _constant_inputs(inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset) -> xr.Dataset:
    constant_inputs = inputs.drop_vars(targets_template.keys(), errors="ignore")
    constant_inputs = constant_inputs.drop_vars(forcings.keys(), errors="ignore")
    return constant_inputs


def block_to_xarray(block_jax: jax.Array, template_xr: xr.Dataset,
                    feature_order: tuple, feature_slices: dict,
                    pressure_levels: list, surface_names: set) -> xr.Dataset:
    """Convert [batch, lat, lon, F] residual/full-pred back to xarray.Dataset
    matching the structure of `template_xr` (output of baseline_predict.apply,
    which has time=1, batch, lat, lon[, level] per variable)."""
    block_np = np.asarray(block_jax, dtype=np.float32)  # [B, lat, lon, F]
    out_vars = {}
    for name in feature_order:
        sl = feature_slices[name]
        var_data = block_np[..., sl]  # [B, lat, lon, k]
        if name in surface_names:
            assert var_data.shape[-1] == 1, f"{name} surface but width {var_data.shape[-1]}"
            arr = var_data[..., 0]  # [B, lat, lon]
            arr = arr[None, ...]    # [time=1, B, lat, lon]
            out_vars[name] = xr.DataArray(
                arr, dims=("time", "batch", "lat", "lon"),
                coords={
                    "time": template_xr["time"].values,
                    "batch": template_xr["batch"].values,
                    "lat": template_xr["lat"].values,
                    "lon": template_xr["lon"].values,
                },
            )
        else:
            arr = var_data[None, ...]  # [time=1, B, lat, lon, level]
            out_vars[name] = xr.DataArray(
                arr, dims=("time", "batch", "lat", "lon", "level"),
                coords={
                    "time": template_xr["time"].values,
                    "batch": template_xr["batch"].values,
                    "lat": template_xr["lat"].values,
                    "lon": template_xr["lon"].values,
                    "level": pressure_levels,
                },
            )
    return xr.Dataset(out_vars)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--trunk-steps", type=int, default=8)
    p.add_argument("--max-lead", type=int, default=8)
    p.add_argument("--window-batch-size", type=int, default=4)
    p.add_argument("--num-chunks", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    W = int(args.warmup_steps)
    S = int(args.trunk_steps)
    L = int(args.max_lead)
    B = int(args.window_batch_size)
    N_CHUNKS = int(args.num_chunks)

    run_dir = Path(args.run_dir)
    cfg_blob = json.load(open(run_dir / "run_config.json"))
    cfg_dict = cfg_blob["config"]

    class _C: pass
    cfg = _C()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    print(f"[eval] run_dir={run_dir}  step={args.step}")
    print(f"[eval] truth_anchored AR: W={W}  S={S}  L={L}  B={B}  N_CHUNKS={N_CHUNKS}")

    if not (cfg.meshed and getattr(cfg, "full_mamba", False)):
        raise NotImplementedError("Only full_mamba meshed variants supported.")
    from src.models.mz.full_mamba import (
        MZResidualFullMambaConfig, MZResidualFullMambaMeshed,
    )
    from src.models.mz.meshed_mamba import build_grid_mesh_projections

    full_variables = getattr(cfg, "full_variables", None)
    if full_variables is None:
        full_variables = len(cfg_blob.get("resolved_variables", [])) > 4
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
    SURFACE_NAMES = {
        "2m_temperature", "mean_sea_level_pressure",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "total_precipitation_6hr",
    }

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
    assert input_steps == N_INPUT_STEPS, f"GraphCast expects N_INPUT_STEPS=2, got {input_steps}"

    lat_deg_eval = np.asarray(eval_ds.lat.values, dtype=np.float64)
    if np.any(np.isclose(np.abs(lat_deg_eval), 90.0)):
        d_lat = float(np.abs(lat_deg_eval[1] - lat_deg_eval[0]))
        pole_mask = np.isclose(np.abs(lat_deg_eval), 90.0)
        pole_w = 2.0 * np.sin(np.deg2rad(d_lat) / 4.0) ** 2
        non_pole_w = 2.0 * np.sin(np.deg2rad(d_lat) / 2.0) * np.cos(np.deg2rad(lat_deg_eval))
        lat_w = np.where(pole_mask, pole_w, non_pole_w)
    else:
        lat_w = np.cos(np.deg2rad(lat_deg_eval))
    lat_w = (lat_w / lat_w.mean()).astype(np.float32)

    pressure_levels = list(task_cfg.pressure_levels)

    def resolved_layout():
        layout, slices, cursor = [], {}, 0
        for name in RESOLVED_VARIABLES:
            width = len(pressure_levels) if name in PRESSURE_LEVEL_VARS else 1
            slices[name] = slice(cursor, cursor + width)
            layout.append(name)
            cursor += width
        return tuple(layout), slices, cursor
    feature_order, feature_slices, feature_dim = resolved_layout()

    def _stack(ds):
        chunks = []
        for name in feature_order:
            da = ds[name]
            if "level" in da.dims:
                arr = da.sel(level=pressure_levels).values.astype(np.float32)
            else:
                arr = np.asarray([float(da.values)], dtype=np.float32)
            chunks.append(arr)
        return np.concatenate(chunks, axis=0)
    diffs_std = _stack(norm_stats["diffs_stddev_by_level"])
    mean_f = _stack(norm_stats["mean_by_level"])
    std_f = _stack(norm_stats["stddev_by_level"])
    input_mean_f = jnp.asarray(mean_f, dtype=jnp.float32)
    input_std_f = jnp.asarray(std_f, dtype=jnp.float32)
    output_denorm_f = jnp.asarray(diffs_std, dtype=jnp.float32)
    residual_input_std_f = jnp.asarray(diffs_std, dtype=jnp.float32)

    # Find chunk starts
    chunk_size = N_INPUT_STEPS + W + S + L  # need this many consecutive cycles
    eval_n = eval_ds.sizes["time"]

    eval_indices_raw = base_train.valid_final_input_indices(eval_n, N_INPUT_STEPS, W + S + L)
    eval_indices = mz_train._filter_time_continuous_indices(
        eval_ds, eval_indices_raw, input_steps=N_INPUT_STEPS,
        target_steps=W + S + L, dt=dt)
    print(f"[eval] valid chunk-launch indices = {len(eval_indices)}")
    # Stride by S so chunks don't overlap their trunk regions
    chunk_launches = []
    last = -10**9
    for idx in eval_indices:
        if int(idx) - last >= S:
            chunk_launches.append(int(idx))
            last = int(idx)
        if len(chunk_launches) >= N_CHUNKS * B:
            break
    print(f"[eval] chunk_launches selected = {len(chunk_launches)}")

    # Baseline predictor (frozen GraphCast)
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
    rng = jax.random.PRNGKey(getattr(cfg, "seed", 0))

    # Init baseline state from a single-step sample
    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(eval_indices[0])],
        input_steps=N_INPUT_STEPS, target_steps=1, task_cfg=task_cfg, dt=dt)
    sample_inputs_j = mz_train._to_jax_dataset(sample_inputs)
    sample_targets_j = mz_train._to_jax_dataset(sample_targets)
    sample_forcings_j = mz_train._to_jax_dataset(sample_forcings)
    _, baseline_state = baseline_predict.init(
        rng, sample_inputs_j, sample_targets_j, sample_forcings_j, False)
    baseline_params = ckpt.params

    # Residual model
    lat_deg = np.asarray(eval_ds.lat.values, dtype=np.float64)
    lon_deg = np.asarray(eval_ds.lon.values, dtype=np.float64)
    proj, n_mesh = build_grid_mesh_projections(
        lat_deg=lat_deg, lon_deg=lon_deg,
        mesh_size=getattr(cfg, "mz_mesh_size", 5),
        n_grid_neighbors=getattr(cfg, "n_grid_neighbors", 6),
        n_mesh_neighbors=getattr(cfg, "n_mesh_neighbors", 3),
    )
    use_specialist_heads = bool(getattr(cfg, "use_specialist_heads", False))
    upper_idx_t, surface_idx_t = (), ()
    if use_specialist_heads:
        upper_idx, surface_idx = [], []
        for name in feature_order:
            sl = feature_slices[name]
            target = surface_idx if name in SURFACE_NAMES else upper_idx
            target.extend(range(sl.start, sl.stop))
        upper_idx_t = tuple(upper_idx)
        surface_idx_t = tuple(surface_idx)
    mz_cfg = MZResidualFullMambaConfig(
        input_size=feature_dim*2, output_size=feature_dim,
        hidden_size=cfg.hidden_size, d_state=cfg.d_state, expand=cfg.expand,
        layers=cfg.layers, dropout=getattr(cfg, "dropout", 0.0),
        a_log_init_min=cfg.a_log_init_min, a_log_init_max=cfg.a_log_init_max,
        use_specialist_heads=use_specialist_heads,
        upper_channel_indices=upper_idx_t,
        surface_channel_indices=surface_idx_t,
    )
    mk = lambda: MZResidualFullMambaMeshed(mz_cfg, n_mesh_nodes=n_mesh, **proj)

    residual_to_state_rescale_f = output_denorm_f / input_std_f

    def residual_step_fn(seq_inputs, baseline_next_real, h_in, is_training):
        """Single-step (T=1) residual head call. seq_inputs [T=1,B,lat,lon,2F]
        REAL units; baseline_next_real [T=1,B,lat,lon,F] REAL units; h_in list
        of arrays [B*M, D, N]. Returns (pred_residual_real [B,lat,lon,F], h_out)."""
        cs, pr = jnp.split(seq_inputs, 2, axis=-1)
        cs_n = (cs - input_mean_f[None,None,None,None,:]) / input_std_f[None,None,None,None,:]
        pr_n = pr / residual_input_std_f[None,None,None,None,:]
        baseline_abs_n = (
            baseline_next_real - input_mean_f[None,None,None,None,:]
        ) / input_std_f[None,None,None,None,:]
        m = mk()
        # Use rollout_ar with T=1, allow_tf_at_t0=True so prev_residual goes
        # through. tf_mask=[1] ensures the supplied prev_residual is used as
        # the residual input for this single step.
        T_one = jnp.ones((1,), dtype=cs.dtype)
        pred_residual_n, h_out = m.rollout_ar(
            cs_n,
            is_training=is_training,
            true_prev_residual_n_tblnf=pr_n,
            tf_mask_per_step=T_one,
            baseline_absolute_n_tblnf=baseline_abs_n,
            residual_to_state_rescale_f=residual_to_state_rescale_f,
            allow_tf_at_t0=True,
            initial_h_states=h_in,
            return_final_h_states=True,
        )
        pred_residual_real = pred_residual_n * output_denorm_f[None,None,None,None,:]
        return pred_residual_real[0], h_out  # drop T axis -> [B,lat,lon,F]

    residual_model = hk.transform(residual_step_fn)

    # Init residual model + load ckpt
    sample_cs = mz_train._extract_feature_block(
        sample_inputs, time_index=-1, task_cfg=task_cfg, feature_order=feature_order)
    sample_cs = sample_cs[None]  # [T=1, B, lat, lon, F]
    sample_pr = jnp.zeros_like(sample_cs)
    sample_seq = jnp.concatenate([sample_cs, sample_pr], axis=-1)  # [1,B,lat,lon,2F]
    # baseline_next_real for init
    sample_pred, _ = baseline_predict.apply(
        baseline_params, baseline_state, rng,
        sample_inputs_j, sample_targets_j, sample_forcings_j, False)
    sample_baseline = mz_train._extract_feature_block(
        sample_pred, time_index=0, task_cfg=task_cfg, feature_order=feature_order)
    sample_baseline = sample_baseline[None]  # [T=1, B, lat, lon, F]
    rng, kinit = jax.random.split(rng)
    mem_params = residual_model.init(kinit, sample_seq, sample_baseline, None, True)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(mem_params))
    print(f"[eval] residual model init: {n_params:,} params")
    mem_params = pickle.load(open(run_dir / f"mz_residual_step{args.step}.pkl", "rb"))

    # Layers/M for h state shape
    layers = cfg.layers
    M = n_mesh
    D_inner = cfg.hidden_size * cfg.expand
    N_state = cfg.d_state

    @jax.jit
    def baseline_step_fn(state, key, inputs, targets, forcings):
        return baseline_predict.apply(
            baseline_params, state, key, inputs, targets, forcings, False)

    @jax.jit
    def residual_step_jit(params, key, seq_inputs, baseline_next_real, h_in):
        return residual_model.apply(params, key, seq_inputs, baseline_next_real, h_in, False)

    # Per-channel + per-leadtime accumulators
    F = feature_dim
    sq_err_base = np.zeros((L, F), dtype=np.float64)
    sq_err_corr = np.zeros((L, F), dtype=np.float64)
    abs_err_base = np.zeros((L, F), dtype=np.float64)
    abs_err_corr = np.zeros((L, F), dtype=np.float64)
    sq_err_base_latw = np.zeros((L, F), dtype=np.float64)
    sq_err_corr_latw = np.zeros((L, F), dtype=np.float64)
    abs_err_base_latw = np.zeros((L, F), dtype=np.float64)
    abs_err_corr_latw = np.zeros((L, F), dtype=np.float64)
    count_per_lead = 0  # total (anchor × chunk × lat × lon) per lead bucket

    lat_w_blnf = lat_w[None, :, None, None]  # broadcast over (B,lat,lon,F)

    # Group chunks into batches of B
    n_groups = len(chunk_launches) // B
    print(f"[eval] processing {n_groups} chunk batches of size B={B}")

    import time
    t_start_total = time.time()

    for grp in range(n_groups):
        launches = chunk_launches[grp*B : (grp+1)*B]
        if len(launches) < B:
            break
        t0 = time.time()

        # Build inputs (last N_INPUT_STEPS cycles) + targets+forcings (W+S+L cycles ahead)
        inp_list, tgt_list, frc_list = [], [], []
        for li in launches:
            inp_one, tgt_one, frc_one = base_train.build_batch_from_indices(
                eval_ds, indices=[li],
                input_steps=N_INPUT_STEPS, target_steps=W + S + L,
                task_cfg=task_cfg, dt=dt)
            inp_list.append(inp_one.isel(batch=0))
            tgt_list.append(tgt_one.isel(batch=0))
            frc_list.append(frc_one.isel(batch=0))
        full_inputs = xr.concat(inp_list, dim="batch").assign_coords(batch=np.arange(B))
        targets_b = xr.concat(tgt_list, dim="batch").assign_coords(batch=np.arange(B))
        forcings_b = xr.concat(frc_list, dim="batch").assign_coords(batch=np.arange(B))
        # Split static (constant) inputs from time-varying ones. Rolling
        # inputs must only carry vars present in (targets ∪ forcings); static
        # fields like geopotential_at_surface stay constant and are merged
        # back at each baseline_predict.apply call.
        constant_inputs = _constant_inputs(full_inputs, targets_b, forcings_b)
        rolling_inputs = full_inputs.drop_vars(constant_inputs.keys())

        # SSM hidden states + prev_residual init
        h_states = [
            jnp.zeros((B * M, D_inner, N_state), dtype=jnp.float32)
            for _ in range(layers)
        ]
        prev_residual_real = jnp.zeros(
            (B, len(eval_ds.lat), len(eval_ds.lon), F), dtype=jnp.float32)

        # WARMUP loop W steps
        for w in range(W):
            tgt_step = targets_b.isel(time=slice(w, w+1))
            frc_step = forcings_b.isel(time=slice(w, w+1))
            tgt_template = tgt_step * np.nan  # no truth leak to baseline
            full_inp = xr.merge([constant_inputs, rolling_inputs])
            inp_j = mz_train._to_jax_dataset(full_inp)
            tgt_j = mz_train._to_jax_dataset(tgt_template)
            frc_j = mz_train._to_jax_dataset(frc_step)
            rng, k = jax.random.split(rng)
            baseline_pred_xr, _ = baseline_step_fn(baseline_state, k, inp_j, tgt_j, frc_j)
            baseline_block = mz_train._extract_feature_block(
                baseline_pred_xr, time_index=0,
                task_cfg=task_cfg, feature_order=feature_order)  # [B,lat,lon,F]
            cs_block = mz_train._extract_feature_block(
                full_inp, time_index=-1,
                task_cfg=task_cfg, feature_order=feature_order)
            seq_inp = jnp.concatenate(
                [cs_block[None], prev_residual_real[None]], axis=-1)  # [1,B,lat,lon,2F]
            baseline_for_res = baseline_block[None]
            rng, k2 = jax.random.split(rng)
            _, h_states = residual_step_jit(
                mem_params, k2, seq_inp, baseline_for_res, h_states)
            # prev_residual ← OBSERVED truth-residual (TRUTH, not model pred)
            truth_block = mz_train._extract_feature_block(
                tgt_step, time_index=0,
                task_cfg=task_cfg, feature_order=feature_order)
            prev_residual_real = truth_block - baseline_block
            # rolling_inputs ← TRUTH
            truth_frame = xr.merge([tgt_step, frc_step])
            rolling_inputs = _update_inputs(rolling_inputs, truth_frame)

        # TRUNK loop S anchors. Each anchor a forks a free L-step rollout.
        for a in range(S):
            # Clone trunk context for branch
            branch_rolling = rolling_inputs  # xarray immutable
            branch_h = [h for h in h_states]
            branch_prev_res = prev_residual_real
            # Branch lead-by-lead
            for lead in range(L):
                pos = W + a + lead
                tgt_step = targets_b.isel(time=slice(pos, pos+1))
                frc_step = forcings_b.isel(time=slice(pos, pos+1))
                tgt_template = tgt_step * np.nan
                full_branch = xr.merge([constant_inputs, branch_rolling])
                inp_j = mz_train._to_jax_dataset(full_branch)
                tgt_j = mz_train._to_jax_dataset(tgt_template)
                frc_j = mz_train._to_jax_dataset(frc_step)
                rng, k = jax.random.split(rng)
                baseline_pred_xr, _ = baseline_step_fn(baseline_state, k, inp_j, tgt_j, frc_j)
                baseline_block = mz_train._extract_feature_block(
                    baseline_pred_xr, time_index=0,
                    task_cfg=task_cfg, feature_order=feature_order)
                cs_block = mz_train._extract_feature_block(
                    full_branch, time_index=-1,
                    task_cfg=task_cfg, feature_order=feature_order)
                seq_inp = jnp.concatenate(
                    [cs_block[None], branch_prev_res[None]], axis=-1)
                baseline_for_res = baseline_block[None]
                rng, k2 = jax.random.split(rng)
                pred_residual, branch_h = residual_step_jit(
                    mem_params, k2, seq_inp, baseline_for_res, branch_h)
                full_pred_block = baseline_block + pred_residual

                # Accumulate metric at this lead vs branch-truth
                truth_block = mz_train._extract_feature_block(
                    tgt_step, time_index=0,
                    task_cfg=task_cfg, feature_order=feature_order)
                b_np = np.asarray(baseline_block, dtype=np.float32)
                c_np = np.asarray(full_pred_block, dtype=np.float32)
                t_np = np.asarray(truth_block, dtype=np.float32)
                diff_b = b_np - t_np
                diff_c = c_np - t_np
                ab_b = np.abs(diff_b); ab_c = np.abs(diff_c)
                sq_b = np.square(diff_b); sq_c = np.square(diff_c)
                # Sum over (B, lat, lon), keep F. Layout: [B, lat, lon, F]
                abs_err_base[lead] += ab_b.sum(axis=(0, 1, 2))
                abs_err_corr[lead] += ab_c.sum(axis=(0, 1, 2))
                sq_err_base[lead] += sq_b.sum(axis=(0, 1, 2))
                sq_err_corr[lead] += sq_c.sum(axis=(0, 1, 2))
                ab_b_w = ab_b * lat_w_blnf
                ab_c_w = ab_c * lat_w_blnf
                sq_b_w = sq_b * lat_w_blnf
                sq_c_w = sq_c * lat_w_blnf
                abs_err_base_latw[lead] += ab_b_w.sum(axis=(0, 1, 2))
                abs_err_corr_latw[lead] += ab_c_w.sum(axis=(0, 1, 2))
                sq_err_base_latw[lead] += sq_b_w.sum(axis=(0, 1, 2))
                sq_err_corr_latw[lead] += sq_c_w.sum(axis=(0, 1, 2))
                if lead == 0:
                    count_per_lead += B * b_np.shape[1] * b_np.shape[2]

                # AR self-feed: branch_rolling ← MODEL_PRED + forcings
                full_pred_xr = block_to_xarray(
                    full_pred_block, baseline_pred_xr,
                    feature_order, feature_slices, pressure_levels, SURFACE_NAMES)
                branch_rolling = _update_inputs(
                    branch_rolling, xr.merge([full_pred_xr, frc_step]))
                branch_prev_res = pred_residual
            # END BRANCH

            # Advance trunk by 1 truth-step (only if not the last anchor)
            if a < S - 1:
                pos = W + a
                tgt_step = targets_b.isel(time=slice(pos, pos+1))
                frc_step = forcings_b.isel(time=slice(pos, pos+1))
                tgt_template = tgt_step * np.nan
                full_inp = xr.merge([constant_inputs, rolling_inputs])
                inp_j = mz_train._to_jax_dataset(full_inp)
                tgt_j = mz_train._to_jax_dataset(tgt_template)
                frc_j = mz_train._to_jax_dataset(frc_step)
                rng, k = jax.random.split(rng)
                baseline_pred_xr, _ = baseline_step_fn(baseline_state, k, inp_j, tgt_j, frc_j)
                baseline_block = mz_train._extract_feature_block(
                    baseline_pred_xr, time_index=0,
                    task_cfg=task_cfg, feature_order=feature_order)
                cs_block = mz_train._extract_feature_block(
                    full_inp, time_index=-1,
                    task_cfg=task_cfg, feature_order=feature_order)
                seq_inp = jnp.concatenate(
                    [cs_block[None], prev_residual_real[None]], axis=-1)
                baseline_for_res = baseline_block[None]
                rng, k2 = jax.random.split(rng)
                _, h_states = residual_step_jit(
                    mem_params, k2, seq_inp, baseline_for_res, h_states)
                truth_block = mz_train._extract_feature_block(
                    tgt_step, time_index=0,
                    task_cfg=task_cfg, feature_order=feature_order)
                prev_residual_real = truth_block - baseline_block
                truth_frame = xr.merge([tgt_step, frc_step])
                rolling_inputs = _update_inputs(rolling_inputs, truth_frame)

        elapsed = time.time() - t0
        print(f"  chunk_batch {grp+1}/{n_groups}: launches={launches}  elapsed={elapsed:.1f}s")

    t_total = time.time() - t_start_total
    print(f"[eval] total elapsed: {t_total:.1f}s ({t_total/60:.1f} min)")

    # Compute metrics
    base_rmse_lead_latw = np.sqrt(sq_err_base_latw / count_per_lead)
    corr_rmse_lead_latw = np.sqrt(sq_err_corr_latw / count_per_lead)
    base_mae_lead_latw = abs_err_base_latw / count_per_lead
    corr_mae_lead_latw = abs_err_corr_latw / count_per_lead
    base_rmse_lead = np.sqrt(sq_err_base / count_per_lead)
    corr_rmse_lead = np.sqrt(sq_err_corr / count_per_lead)
    base_mae_lead = abs_err_base / count_per_lead
    corr_mae_lead = abs_err_corr / count_per_lead

    # Aggregate across leads for "overall" per-channel
    n_lead = L
    sq_base_overall = sq_err_base.sum(axis=0)
    sq_corr_overall = sq_err_corr.sum(axis=0)
    ab_base_overall = abs_err_base.sum(axis=0)
    ab_corr_overall = abs_err_corr.sum(axis=0)
    sq_base_overall_w = sq_err_base_latw.sum(axis=0)
    sq_corr_overall_w = sq_err_corr_latw.sum(axis=0)
    ab_base_overall_w = abs_err_base_latw.sum(axis=0)
    ab_corr_overall_w = abs_err_corr_latw.sum(axis=0)
    total_count = count_per_lead * n_lead
    base_rmse_overall = np.sqrt(sq_base_overall / total_count)
    corr_rmse_overall = np.sqrt(sq_corr_overall / total_count)
    base_mae_overall = ab_base_overall / total_count
    corr_mae_overall = ab_corr_overall / total_count
    base_rmse_overall_latw = np.sqrt(sq_base_overall_w / total_count)
    corr_rmse_overall_latw = np.sqrt(sq_corr_overall_w / total_count)
    base_mae_overall_latw = ab_base_overall_w / total_count
    corr_mae_overall_latw = ab_corr_overall_w / total_count

    # Per-channel breakdown
    per_channel = []
    for ch_idx, name in enumerate(feature_order):
        sl = feature_slices[name]
        is_surface = name in SURFACE_NAMES
        levels = [None] if is_surface else pressure_levels
        for li, lvl in enumerate(levels):
            cidx = sl.start + li
            entry = {
                "variable": name, "level": lvl, "channel_idx": cidx,
                "baseline_MAE": float(base_mae_overall[cidx]),
                "corrected_MAE": float(corr_mae_overall[cidx]),
                "baseline_RMSE": float(base_rmse_overall[cidx]),
                "corrected_RMSE": float(corr_rmse_overall[cidx]),
                "baseline_MAE_latw": float(base_mae_overall_latw[cidx]),
                "corrected_MAE_latw": float(corr_mae_overall_latw[cidx]),
                "baseline_RMSE_latw": float(base_rmse_overall_latw[cidx]),
                "corrected_RMSE_latw": float(corr_rmse_overall_latw[cidx]),
            }
            entry["delta_MAE_pct"] = (
                100*(entry["baseline_MAE"]-entry["corrected_MAE"])/entry["baseline_MAE"]
                if entry["baseline_MAE"]>0 else 0.0)
            entry["delta_MAE_pct_latw"] = (
                100*(entry["baseline_MAE_latw"]-entry["corrected_MAE_latw"])/entry["baseline_MAE_latw"]
                if entry["baseline_MAE_latw"]>0 else 0.0)
            entry["delta_RMSE_pct_latw"] = (
                100*(entry["baseline_RMSE_latw"]-entry["corrected_RMSE_latw"])/entry["baseline_RMSE_latw"]
                if entry["baseline_RMSE_latw"]>0 else 0.0)
            per_channel.append(entry)

    # Per-channel-per-leadtime
    per_channel_per_leadtime = []
    for ch_idx, name in enumerate(feature_order):
        sl = feature_slices[name]
        is_surface = name in SURFACE_NAMES
        levels = [None] if is_surface else pressure_levels
        for li, lvl in enumerate(levels):
            cidx = sl.start + li
            for lead_idx in range(L):
                entry = {
                    "variable": name, "level": lvl, "channel_idx": cidx,
                    "lead_time_h": (lead_idx + 1) * 6,
                    "lead_time_idx": lead_idx,
                    "baseline_MAE_latw": float(base_mae_lead_latw[lead_idx, cidx]),
                    "corrected_MAE_latw": float(corr_mae_lead_latw[lead_idx, cidx]),
                    "baseline_RMSE_latw": float(base_rmse_lead_latw[lead_idx, cidx]),
                    "corrected_RMSE_latw": float(corr_rmse_lead_latw[lead_idx, cidx]),
                }
                entry["delta_MAE_pct_latw"] = (
                    100*(entry["baseline_MAE_latw"]-entry["corrected_MAE_latw"])/entry["baseline_MAE_latw"]
                    if entry["baseline_MAE_latw"]>0 else 0.0)
                entry["delta_RMSE_pct_latw"] = (
                    100*(entry["baseline_RMSE_latw"]-entry["corrected_RMSE_latw"])/entry["baseline_RMSE_latw"]
                    if entry["baseline_RMSE_latw"]>0 else 0.0)
                per_channel_per_leadtime.append(entry)

    n_better_latw = sum(1 for ch in per_channel if ch["delta_MAE_pct_latw"] > 0)
    n_worse_latw = sum(1 for ch in per_channel if ch["delta_MAE_pct_latw"] < 0)
    n_targets_better_RMSE_latw = sum(
        1 for e in per_channel_per_leadtime if e["delta_RMSE_pct_latw"] > 0)

    out = {
        "run_dir": str(run_dir),
        "step": args.step,
        "horizon": L,                # max lead steps
        "mode": "truth_anchored_ar",
        "warmup_steps": W,
        "trunk_steps": S,
        "max_lead": L,
        "window_batch_size": B,
        "num_chunks": n_groups,
        "n_channels_total": len(per_channel),
        "n_channels_better_latw": n_better_latw,
        "n_channels_worse_latw": n_worse_latw,
        "n_lead_times": L,
        "n_targets_total": len(per_channel_per_leadtime),
        "n_targets_better_RMSE_latw": n_targets_better_RMSE_latw,
        "per_channel": per_channel,
        "per_channel_per_leadtime": per_channel_per_leadtime,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] saved {args.out}")
    print(f"  channels improved (latw MAE): {n_better_latw}/{len(per_channel)}")
    print(f"  targets improved (latw RMSE): {n_targets_better_RMSE_latw}/{len(per_channel_per_leadtime)}")


if __name__ == "__main__":
    main()
