"""v9 lead-1 eval: per-variable lat-weighted RMSE.

Mirrors eval_v11.py but uses v9's GCResidualWithZeroHead class (full DeepMind
GC2 structure with Mamba interleaved + zero-init residual head).
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
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main():
    cfg = parse_args()
    ckpt_path = Path(cfg.ckpt)
    print(f"[eval-v9] ckpt: {ckpt_path}")

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

    rng = jax.random.PRNGKey(cfg.seed)
    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, cfg.target_steps)
    print(f"[eval-v9] val_indices: {len(val_indices)} valid samples")

    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(val_indices[0])],
        input_steps=input_steps, target_steps=cfg.target_steps,
        task_cfg=task_cfg, dt=dt)

    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state = baseline_predict.init(
        k_b, sample_inputs, sample_targets, sample_forcings)
    baseline_params, _ = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)

    _residual_params_init, residual_state = residual_predict.init(
        k_r, sample_inputs, sample_targets, sample_forcings)

    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    if "residual_state" in ckpt and ckpt["residual_state"]:
        residual_state = ckpt["residual_state"]
    n_p = sum(p.size for p in jax.tree_util.tree_leaves(residual_params))
    print(f"[eval-v9] loaded residual_params {n_p:,}")

    @jax.jit
    def _baseline_step(params, state, key, inp, tgt, frc):
        out, _ = baseline_predict.apply(params, state, key, inp, tgt, frc)
        return out

    @jax.jit
    def _residual_step(params, state, key, inp, tgt, frc):
        out, new_state = residual_predict.apply(params, state, key, inp, tgt, frc)
        return out, new_state

    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = sorted(rng_np.choice(val_indices, size=cfg.n_samples,
                                       replace=False).tolist())

    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_da = xr.DataArray(cos_lat / cos_lat.mean(), dims="lat")

    sum_sq_baseline, sum_sq_full = {}, {}
    n_per_var, n_per_var_per_level = {}, {}
    sum_sq_baseline_pl, sum_sq_full_pl = {}, {}

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=cfg.target_steps, task_cfg=task_cfg, dt=dt)
        rng, k1, k2 = jax.random.split(rng, 3)
        baseline_pred = _baseline_step(baseline_params, baseline_state, k1, inp, tgt, frc)
        r_pred, _ = _residual_step(residual_params, residual_state, k2, inp, tgt, frc)
        full_pred = jax.tree_util.tree_map(lambda b, r: b + r, baseline_pred, r_pred)

        for var in tgt.data_vars:
            truth = tgt[var].astype("float32")
            bp = baseline_pred[var].astype("float32")
            fp = full_pred[var].astype("float32")
            err_b = bp - truth
            err_f = fp - truth
            mse_b = float(((err_b ** 2) * cos_lat_da).mean().values)
            mse_f = float(((err_f ** 2) * cos_lat_da).mean().values)
            sum_sq_baseline[var] = sum_sq_baseline.get(var, 0.0) + mse_b
            sum_sq_full[var] = sum_sq_full.get(var, 0.0) + mse_f
            n_per_var[var] = n_per_var.get(var, 0) + 1

            if "level" in truth.dims:
                if var not in sum_sq_baseline_pl:
                    sum_sq_baseline_pl[var] = {}
                    sum_sq_full_pl[var] = {}
                    n_per_var_per_level[var] = {}
                for lev_i, lev in enumerate(truth["level"].values):
                    lev = int(lev)
                    eb = err_b.isel(level=lev_i)
                    ef = err_f.isel(level=lev_i)
                    sum_sq_baseline_pl[var][lev] = sum_sq_baseline_pl[var].get(lev, 0.0) + float(((eb ** 2) * cos_lat_da).mean().values)
                    sum_sq_full_pl[var][lev] = sum_sq_full_pl[var].get(lev, 0.0) + float(((ef ** 2) * cos_lat_da).mean().values)
                    n_per_var_per_level[var][lev] = n_per_var_per_level[var].get(lev, 0) + 1

        if s_i < 3 or (s_i + 1) % 8 == 0:
            print(f"[eval-v9] sample {s_i+1}/{cfg.n_samples}")

    print(f"\n{'variable':<28} {'RMSE_baseline':>14} {'RMSE_full':>12} {'rel_change':>12}")
    print("-" * 72)
    out_dict = {"per_variable": {}, "per_channel": {}}
    n_chan_better = 0
    n_chan_total = 0
    for var in sorted(n_per_var):
        n = n_per_var[var]
        rmse_b = (sum_sq_baseline[var] / n) ** 0.5
        rmse_f = (sum_sq_full[var] / n) ** 0.5
        rel = (rmse_f - rmse_b) / max(rmse_b, 1e-12) * 100
        print(f"{var:<28} {rmse_b:>14.4g} {rmse_f:>12.4g} {rel:>+11.2f}%")
        out_dict["per_variable"][var] = dict(rmse_baseline=rmse_b, rmse_full=rmse_f, rel_change_pct=rel)
        if var in sum_sq_baseline_pl:
            for lev in sorted(sum_sq_baseline_pl[var]):
                nl = n_per_var_per_level[var][lev]
                rmse_b_l = (sum_sq_baseline_pl[var][lev] / nl) ** 0.5
                rmse_f_l = (sum_sq_full_pl[var][lev] / nl) ** 0.5
                rel_l = (rmse_f_l - rmse_b_l) / max(rmse_b_l, 1e-12) * 100
                key = f"{var}_level{lev}"
                out_dict["per_channel"][key] = dict(rmse_baseline=rmse_b_l, rmse_full=rmse_f_l, rel_change_pct=rel_l)
                if rmse_f_l < rmse_b_l:
                    n_chan_better += 1
                n_chan_total += 1
        else:
            out_dict["per_channel"][var] = dict(rmse_baseline=rmse_b, rmse_full=rmse_f, rel_change_pct=rel)
            if rmse_f < rmse_b:
                n_chan_better += 1
            n_chan_total += 1

    print()
    print(f"[eval-v9] channels improved: {n_chan_better} / {n_chan_total} "
          f"({100*n_chan_better/n_chan_total:.1f}%)")
    out_dict["channels_improved"] = n_chan_better
    out_dict["channels_total"] = n_chan_total
    out_dict["channels_improved_pct"] = 100 * n_chan_better / n_chan_total
    out_dict["n_samples"] = cfg.n_samples
    out_dict["ckpt"] = str(ckpt_path)
    if cfg.out_json:
        with open(cfg.out_json, "w") as f:
            json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    main()
