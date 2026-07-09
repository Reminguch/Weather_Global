#!/usr/bin/env python3
"""Minimal MZ inference: load step400 MZ params + frozen GraphCast baseline,
run forward on the first few eval segments, save baseline / corrected / truth
tensors plus time stamps to npz for plotting.
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_graphcast as base_train  # noqa: E402
import train_mz_residual_memory as mz  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Directory with mz_residual_stepNNN.pkl + run_config.json")
    p.add_argument("--step", type=int, default=400)
    p.add_argument("--num-segments", type=int, default=4)
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument(
        "--legacy-v1",
        action="store_true",
        default=False,
        help="Use the pre-refactor v1 teacher-only module "
             "(src.models.mz_residual_mamba_v1_teacher) so that old "
             "checkpoints (e.g. mz_residual_step400.pkl from "
             "mz_r4_m3_i32_seg32_h16_fullnorm) with flat Haiku param "
             "names can be loaded. Current (v2) module has ~/-prefixed "
             "param names and is incompatible with those checkpoints.",
    )
    args = p.parse_args()

    # Dispatch the MZ module import based on the flag. Both modules expose
    # MZResidualConfig + MZResidualMamba with the same construction signature,
    # but their Haiku parameter trees have different naming conventions.
    if args.legacy_v1:
        from src.models.mz_residual_mamba_v1_teacher import (  # noqa: E402
            MZResidualConfig, MZResidualMamba,
        )
        print("[infer] using legacy v1 teacher-only module (flat param names)")
    else:
        from src.models.mz_residual_mamba import (  # noqa: E402
            MZResidualConfig, MZResidualMamba,
        )
        print("[infer] using v2 module (rollout_ar available; ~/-prefixed param names)")

    run_dir = Path(args.run_dir)
    cfg_blob = __import__("json").load(open(run_dir / "run_config.json"))
    cfg_dict = cfg_blob["config"]

    class _C:
        pass
    cfg = _C()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    # Reconstruct the same pipeline as train_mz_residual_memory.main(), but eval only
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
    _train_ds = base_train.prepare_dataset_for_task(_train_ds, task_cfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)

    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)

    eval_indices_raw = base_train.valid_final_input_indices(eval_ds.sizes["time"], input_steps, 1)
    eval_indices = mz._filter_time_continuous_indices(
        eval_ds, eval_indices_raw, input_steps=input_steps, target_steps=1, dt=dt
    )
    eval_segments = [
        seg for seg in mz._time_continuous_segments(eval_ds, eval_indices, cfg.segment_steps, dt)
        if len(seg) == cfg.segment_steps
    ]

    feature_order, feature_slices, feature_dim = mz._resolved_feature_layout(task_cfg)
    diffs_std_f_np = mz._build_diffs_stddev_vector(task_cfg, norm_stats, feature_order)
    mean_f_np, std_f_np = mz._build_mean_stddev_vectors(task_cfg, norm_stats, feature_order)

    input_mean_f = jnp.asarray(mean_f_np, dtype=jnp.float32)
    input_std_f = jnp.asarray(std_f_np, dtype=jnp.float32)
    output_denorm_f = jnp.asarray(diffs_std_f_np, dtype=jnp.float32)
    residual_input_std_f = jnp.asarray(diffs_std_f_np, dtype=jnp.float32)

    def baseline_predict_fn(inputs, targets, forcings, is_training):
        predictor = base_train.build_predictor(
            model_cfg, task_cfg, norm_stats,
            use_bf16=(cfg.baseline_precision == "bf16"),
            gradient_checkpointing=False,
            temporal_backbone="none", temporal_location="mesh_post_encoder",
            temporal_hidden_size=model_cfg.latent_size,
            temporal_layers=1, temporal_dropout=0.0,
        )
        return predictor(inputs, targets_template=targets, forcings=forcings, is_training=is_training)

    baseline_predict = hk.transform_with_state(baseline_predict_fn)
    rng = jax.random.PRNGKey(cfg.seed)

    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(eval_indices[0])],
        input_steps=input_steps, target_steps=1, task_cfg=task_cfg, dt=dt,
    )
    sample_inputs = mz._to_jax_dataset(sample_inputs)
    sample_targets = mz._to_jax_dataset(sample_targets)
    sample_forcings = mz._to_jax_dataset(sample_forcings)
    _, baseline_state = baseline_predict.init(rng, sample_inputs, sample_targets, sample_forcings, False)
    baseline_params = ckpt.params

    mz_cfg = MZResidualConfig(
        input_size=feature_dim * 2, output_size=feature_dim,
        hidden_size=cfg.hidden_size, layers=cfg.layers,
        dropout=cfg.dropout, a_log_init=cfg.a_log_init,
    )

    def residual_objective(seq_inputs, baseline_next, truth_next, is_training):
        current_state, prev_residual = jnp.split(seq_inputs, 2, axis=-1)
        current_state_n = (current_state - input_mean_f[None, None, None, None, :]) / input_std_f[None, None, None, None, :]
        prev_residual_n = prev_residual / residual_input_std_f[None, None, None, None, :]
        seq_inputs_n = jnp.concatenate([current_state_n, prev_residual_n], axis=-1)
        model = MZResidualMamba(mz_cfg)
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
        # _segment_to_tensors now returns a 4-tuple (the 4th is tf_mask used
        # only for target_rollout training). For K=1 inference we discard it.
        seq_inputs, baseline_next, truth_next, _tf_mask = mz._segment_to_tensors(
            baseline_predict, baseline_params, baseline_state, key,
            eval_ds, seg, input_steps=input_steps, task_cfg=task_cfg, dt=dt,
            feature_order=feature_order,
        )
        out = infer_step(mem_params, key, seq_inputs, baseline_next, truth_next)
        baseline_out.append(np.asarray(baseline_next, dtype=np.float32))
        corrected_out.append(np.asarray(out["corrected"], dtype=np.float32))
        truth_out.append(np.asarray(truth_next, dtype=np.float32))
        # Time stamps: each batch element's TARGET time = final_input_idx + 1
        seg_arr = np.asarray(seg, dtype=np.int64)
        target_idx = seg_arr + 1
        times_out.append(time_coord[target_idx].values)
        seg_first_indices.append(int(seg_arr[0]))
        print(f"segment {i}: seg_first={seg_arr[0]} target_times={time_coord[target_idx[0]]} .. {time_coord[target_idx[-1]]}")

    # Stack along segments axis -> shape [seg, T, B=1, lat, lon, F]
    baseline_arr = np.stack(baseline_out, axis=0)
    corrected_arr = np.stack(corrected_out, axis=0)
    truth_arr = np.stack(truth_out, axis=0)
    times_arr = np.stack(times_out, axis=0)

    lat = np.asarray(eval_ds.lat.values, dtype=np.float32)
    lon = np.asarray(eval_ds.lon.values, dtype=np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        baseline=baseline_arr,
        corrected=corrected_arr,
        truth=truth_arr,
        times=times_arr,
        lat=lat, lon=lon,
        feature_order=np.asarray(list(feature_order)),
        feature_slices=np.asarray([[sl.start, sl.stop] for _, sl in feature_slices.items()]),
        pressure_levels=np.asarray(list(task_cfg.pressure_levels)),
        seg_first_indices=np.asarray(seg_first_indices),
    )
    print(f"saved: {out}  shapes: baseline={baseline_arr.shape}")


if __name__ == "__main__":
    main()
