"""v9 diagnostics: per-variable RMS(r*), RMS(r_pred), corr(r_pred, r*).

Diagnoses whether v9's residual model is producing useful signal or noise:
- r*     = truth - baseline_pred  (the residual the model SHOULD learn)
- r_pred = residual_predict(inputs)  (what the model actually outputs)
- If RMS(r_pred) grows but corr ≈ 0, the model is outputting noise.
- If RMS(r_pred) ≈ 0, the model is outputting nothing (zero-init head not moving).
- If RMS(r_pred) ≈ RMS(r*) AND corr > 0, the model is genuinely learning.

Usage:
  python diagnose_v9.py --ckpt <path/to/v9_residual_stepK.pkl> [--n-samples 16]
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
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

from graphcast import casting, graphcast as gc, normalization  # noqa: E402

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.training.core.model import (  # noqa: E402
    DirectResidualNormalizer,
)
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402

from scripts.training.full_mamba_v9.train_mz_v9 import (  # noqa: E402
    GCResidualWithZeroHead, _attach_temporal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-path", default=base_train.DEFAULT_DATA_PATH)
    p.add_argument("--stats-dir", default=base_train.DEFAULT_STATS_DIR)
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
    p.add_argument("--target-steps", type=int, default=1)
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=None)
    p.add_argument("--temporal-d-state", type=int, default=16)
    p.add_argument("--temporal-d-conv", type=int, default=4)
    p.add_argument("--temporal-dt-rank", default="auto")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--temporal-bias", action="store_true", default=False)
    p.add_argument("--no-temporal-conv-bias", dest="temporal_conv_bias",
                   action="store_false", default=True)
    p.add_argument("--temporal-dropout", type=float, default=0.0)
    p.add_argument("--no-zero-init-out", dest="temporal_zero_init_out",
                   action="store_false", default=True)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    cfg = parse_args()
    ckpt_path = Path(cfg.ckpt)
    print(f"[diag] ckpt: {ckpt_path}")

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
        if use_bf16:
            p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(
            p,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"],
        )
        return p

    def _build_residual():
        p = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
        _attach_temporal(p, cfg)
        if use_bf16:
            p = casting.Bfloat16Cast(p)
        p = DirectResidualNormalizer(
            p,
            stddev_by_level=norm_stats["stddev_by_level"],
            mean_by_level=norm_stats["mean_by_level"],
            diffs_stddev_by_level=norm_stats["diffs_stddev_by_level"],
        )
        return p

    def baseline_fn(inputs, targets, forcings):
        return _build_baseline()(inputs, targets_template=targets, forcings=forcings)

    def residual_fn(inputs, targets, forcings):
        return _build_residual()(inputs, targets_template=targets, forcings=forcings)

    baseline_predict = hk.transform_with_state(baseline_fn)
    residual_predict = hk.transform_with_state(residual_fn)

    # Init both for params/state plumbing.
    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, cfg.target_steps)
    print(f"[diag] val_indices: {len(val_indices)} valid samples in {cfg.val_year}")

    # Use first sample to init params.
    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(val_indices[0])],
        input_steps=input_steps, target_steps=cfg.target_steps,
        task_cfg=task_cfg, dt=dt)

    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state = baseline_predict.init(
        k_b, sample_inputs, sample_targets, sample_forcings)
    baseline_params, _ = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)

    residual_params_init, residual_state = residual_predict.init(
        k_r, sample_inputs, sample_targets, sample_forcings)
    residual_params_init, _ = overlay_matching_params(
        residual_params_init, ckpt_in.params, strict=False)

    # Load trained ckpt and overlay onto initialized residual_params.
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    if "residual_state" in ckpt and ckpt["residual_state"]:
        residual_state = ckpt["residual_state"]
    print(f"[diag] loaded residual_params from ckpt; "
          f"{sum(p.size for p in jax.tree_util.tree_leaves(residual_params)):,} params")

    @jax.jit
    def _baseline_step(params, state, key, inp, tgt, frc):
        out, _ = baseline_predict.apply(params, state, key, inp, tgt, frc)
        return out

    @jax.jit
    def _residual_step(params, state, key, inp, tgt, frc):
        out, new_state = residual_predict.apply(params, state, key, inp, tgt, frc)
        return out, new_state

    # Evenly sample val indices.
    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = rng_np.choice(val_indices, size=cfg.n_samples, replace=False)
    print(f"[diag] sampling {cfg.n_samples} val indices: {sorted(chosen_idx.tolist())[:5]} ...")

    # Accumulators per variable (raw target units).
    sum_r_star_sq = {}    # sum over samples of (r*)^2 mean over (lat,lon,level,time)
    sum_r_pred_sq = {}    # same for r_pred
    sum_r_cross = {}      # sum of (r* * r_pred) mean
    n_per_var = {}

    # Latitude weights.
    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat = cos_lat / cos_lat.mean()  # normalize
    cos_lat_da = xr.DataArray(cos_lat, dims="lat")

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=cfg.target_steps, task_cfg=task_cfg, dt=dt)
        rng, k1, k2 = jax.random.split(rng, 3)
        baseline_pred = _baseline_step(baseline_params, baseline_state, k1, inp, tgt, frc)
        r_pred, _ = _residual_step(residual_params, residual_state, k2, inp, tgt, frc)
        r_star = tgt - baseline_pred  # both xarray Datasets

        for var in r_star.data_vars:
            a = r_star[var].astype("float32")
            b = r_pred[var].astype("float32")
            # weight by cos(lat) on the lat dim.
            w_a = a * cos_lat_da
            w_b = b * cos_lat_da
            # average over batch, time, lat, lon (and level if present).
            mean_a_sq = float((w_a ** 2).mean().values)
            mean_b_sq = float((w_b ** 2).mean().values)
            mean_cross = float((w_a * w_b).mean().values)
            sum_r_star_sq[var] = sum_r_star_sq.get(var, 0.0) + mean_a_sq
            sum_r_pred_sq[var] = sum_r_pred_sq.get(var, 0.0) + mean_b_sq
            sum_r_cross[var] = sum_r_cross.get(var, 0.0) + mean_cross
            n_per_var[var] = n_per_var.get(var, 0) + 1

        if s_i < 3 or (s_i + 1) % 5 == 0:
            print(f"[diag] sample {s_i+1}/{cfg.n_samples} processed")

    print(f"\n[diag] Per-variable diagnostics (lat-weighted, averaged over {cfg.n_samples} val samples):\n")
    print(f"{'variable':<28} {'RMS(r*)':>12} {'RMS(r_pred)':>14} {'corr':>8} {'ratio':>8}")
    print("-" * 76)
    for var in sorted(n_per_var):
        n = n_per_var[var]
        rms_a = (sum_r_star_sq[var] / n) ** 0.5
        rms_b = (sum_r_pred_sq[var] / n) ** 0.5
        cross = sum_r_cross[var] / n
        denom = max(rms_a * rms_b, 1e-30)
        corr = cross / denom
        ratio = rms_b / max(rms_a, 1e-30)
        print(f"{var:<28} {rms_a:>12.4g} {rms_b:>14.4g} {corr:>8.3f} {ratio:>8.3f}")

    print("\nLegend:")
    print("  RMS(r*)     = residual the model SHOULD learn (raw units)")
    print("  RMS(r_pred) = residual the model actually outputs")
    print("  corr        = corr(r_pred, r*); 1=perfect, 0=noise/zero")
    print("  ratio       = RMS(r_pred)/RMS(r*); 1=ideal magnitude")


if __name__ == "__main__":
    main()
