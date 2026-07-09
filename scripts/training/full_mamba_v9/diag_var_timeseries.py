"""Per-anchor lat-weighted MAE+RMSE for one variable, saved alongside
each anchor's datetime, for plotting error vs calendar time across the
val year. Generates time-series at every K (lead time)."""
from __future__ import annotations

import argparse
import dataclasses
import pickle
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

from graphcast import casting, graphcast as gc, normalization, rollout  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import DirectResidualNormalizer  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402

from scripts.training.full_mamba_v9.train_mz_v9 import (  # noqa: E402
    GCResidualWithZeroHead, _attach_temporal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--ckpt-in", default=(
        "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/params/"
        "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - "
        "mesh 2to5 - precipitation input and output.npz"))
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--mesh-size", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-msg-steps", type=int, default=16)
    p.add_argument("--residual-msg-steps", type=int, default=2)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--train-start-year", type=int, default=2020)
    p.add_argument("--train-end-year", type=int, default=2021)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=6)
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=None)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias",
                   action="store_false", default=True)
    p.add_argument("--no-zero-init-out", dest="temporal_zero_init_out",
                   action="store_false", default=True)
    p.add_argument("--temporal-bias", action="store_true", default=False)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--n-anchors", type=int, default=200,
                   help="Stride-sample this many anchors evenly across val_year")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--var", default="total_precipitation_6hr")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--out-npz", required=True)
    return p.parse_args()


def main():
    cfg = parse_args()
    ckpt_path = Path(cfg.ckpt)
    ckpt_in = base_train.load_graphcast_checkpoint(Path(cfg.ckpt_in))
    base_model_cfg = ckpt_in.model_config
    task_cfg = ckpt_in.task_config
    if cfg.input_duration is not None:
        task_cfg = dataclasses.replace(task_cfg, input_duration=cfg.input_duration)
    model_cfg_baseline = dataclasses.replace(
        base_model_cfg, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.baseline_msg_steps)
    model_cfg_residual = dataclasses.replace(
        base_model_cfg, resolution=cfg.resolution, mesh_size=cfg.mesh_size,
        latent_size=cfg.width, gnn_msg_steps=cfg.residual_msg_steps)
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

    use_bf16 = True
    def _build_baseline():
        p = gc.GraphCast(model_cfg_baseline, task_cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p
    def _build_residual():
        p = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
        _attach_temporal(p, cfg)
        if use_bf16: p = casting.Bfloat16Cast(p)
        p = DirectResidualNormalizer(
            p, stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"])
        return p
    def baseline_fn(inputs, targets, forcings):
        return _build_baseline()(inputs, targets_template=targets, forcings=forcings)
    def residual_fn(inputs, targets, forcings):
        return _build_residual()(inputs, targets_template=targets, forcings=forcings)
    baseline_predict = hk.transform_with_state(baseline_fn)
    residual_predict = hk.transform_with_state(residual_fn)

    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, cfg.target_steps)
    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(val_indices[0])],
        input_steps=input_steps, target_steps=cfg.target_steps,
        task_cfg=task_cfg, dt=dt)
    sample_targets_1step = sample_targets.isel(time=slice(0, 1))
    sample_forcings_1step = sample_forcings.isel(time=slice(0, 1))

    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state = baseline_predict.init(
        k_b, sample_inputs, sample_targets_1step, sample_forcings_1step)
    baseline_params, _ = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    _, residual_state = residual_predict.init(
        k_r, sample_inputs, sample_targets_1step, sample_forcings_1step)
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    if "residual_state" in ckpt and ckpt["residual_state"]:
        residual_state = ckpt["residual_state"]
    print(f"[diag-ts] loaded {ckpt_path.name}")

    @jax.jit
    def _baseline_step(params, state, key, inp, tgt, frc):
        out, _ = baseline_predict.apply(params, state, key, inp, tgt, frc)
        return out
    @jax.jit
    def _residual_step(params, state, key, inp, tgt, frc):
        out, new_state = residual_predict.apply(params, state, key, inp, tgt, frc)
        return out, new_state

    def baseline_predictor(rng, inputs, targets_template, forcings):
        return _baseline_step(baseline_params, baseline_state, rng, inputs, targets_template, forcings)
    def full_predictor(rng, inputs, targets_template, forcings):
        rng_b, rng_r = jax.random.split(rng)
        bp = _baseline_step(baseline_params, baseline_state, rng_b, inputs, targets_template, forcings)
        rp, _ = _residual_step(residual_params, residual_state, rng_r, inputs, targets_template, forcings)
        return jax.tree_util.tree_map(lambda b, r: b + r, bp, rp)

    # Stride-sample anchors evenly across val_year
    if cfg.n_anchors >= len(val_indices):
        chosen = list(val_indices)
    else:
        stride = max(1, len(val_indices) // cfg.n_anchors)
        chosen = list(val_indices[::stride])[:cfg.n_anchors]
    K = cfg.target_steps
    N = len(chosen)
    print(f"[diag-ts] N anchors = {N}, K = {K}")

    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_da = xr.DataArray(cos_lat / cos_lat.mean(), dims="lat")

    times_out = []
    rmse_b = np.zeros((N, K), dtype=np.float32)
    rmse_f = np.zeros((N, K), dtype=np.float32)
    mae_b  = np.zeros((N, K), dtype=np.float32)
    mae_f  = np.zeros((N, K), dtype=np.float32)
    # lat-weighted global spatial means of each field (scalar per anchor per K)
    mean_truth = np.zeros((N, K), dtype=np.float32)
    mean_base  = np.zeros((N, K), dtype=np.float32)
    mean_full  = np.zeros((N, K), dtype=np.float32)

    for i, idx in enumerate(chosen):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=K, task_cfg=task_cfg, dt=dt)
        rng, r_b, r_f = jax.random.split(rng, 3)
        b_chunks = list(rollout.chunked_prediction_generator(
            baseline_predictor, r_b, inp, tgt, frc, num_steps_per_chunk=1))
        f_chunks = list(rollout.chunked_prediction_generator(
            full_predictor, r_f, inp, tgt, frc, num_steps_per_chunk=1))
        bp = xr.concat(b_chunks, dim="time")
        fp = xr.concat(f_chunks, dim="time")

        def _v(ds):
            a = ds[cfg.var].astype("float32")
            if "level" in a.dims:
                a = a.sel(level=cfg.level)
            return a
        T = _v(tgt); B = _v(bp); F = _v(fp)

        # anchor time = last input time (the "issue time" of the forecast)
        anchor_t = pd.Timestamp(eval_ds["time"].values[int(idx)])
        times_out.append(np.datetime64(anchor_t))

        for k in range(K):
            err_b_k = B.isel(time=k) - T.isel(time=k)
            err_f_k = F.isel(time=k) - T.isel(time=k)
            rmse_b[i, k] = float(np.sqrt(((err_b_k ** 2) * cos_lat_da).mean().values))
            rmse_f[i, k] = float(np.sqrt(((err_f_k ** 2) * cos_lat_da).mean().values))
            mae_b[i, k]  = float((np.abs(err_b_k) * cos_lat_da).mean().values)
            mae_f[i, k]  = float((np.abs(err_f_k) * cos_lat_da).mean().values)
            mean_truth[i, k] = float((T.isel(time=k) * cos_lat_da).mean().values)
            mean_base[i, k]  = float((B.isel(time=k) * cos_lat_da).mean().values)
            mean_full[i, k]  = float((F.isel(time=k) * cos_lat_da).mean().values)

        if (i + 1) % 25 == 0 or i < 3:
            print(f"[diag-ts] {i+1}/{N}  idx={idx}  t={anchor_t}", flush=True)

    out = Path(cfg.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        anchor_time=np.array(times_out),
                        rmse_baseline=rmse_b, rmse_full=rmse_f,
                        mae_baseline=mae_b, mae_full=mae_f,
                        mean_truth=mean_truth, mean_baseline=mean_base, mean_full=mean_full,
                        var=cfg.var, level=cfg.level if cfg.level else -1)
    print(f"[diag-ts] wrote {out}  shape={rmse_b.shape}")


if __name__ == "__main__":
    main()
