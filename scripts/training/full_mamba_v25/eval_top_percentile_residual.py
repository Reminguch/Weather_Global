"""Top-percentile residual RMSE eval for gated extreme residual head.

For each (variable, lead), rank cells by |y - bp| (baseline error), take the
top-α fraction (α ∈ {0.05, 0.01}), and compute RMSE_baseline / RMSE_full /
improvement on that subset only. This is the headline metric for whether the
gated extreme head actually corrects extreme cases.

Also dumps gate map statistics: mean gate by variable + extreme rate, false-
alarm rate (gate>0.5 on non-extreme cells).
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
from src.models.graphcast.training.core.model import DirectResidualNormalizer  # noqa: E402
from src.models.mamba.training.param_utils import overlay_matching_params  # noqa: E402
from scripts.training.full_mamba_v9.train_mz_v9 import _attach_temporal, GCResidualWithZeroHead  # noqa: E402
from scripts.training.full_mamba_v25.gated_extreme_head import (  # noqa: E402
    GCResidualWithGatedExtremeHead)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-in", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--mesh-size", type=int, default=4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--baseline-msg-steps", type=int, default=6)
    p.add_argument("--residual-msg-steps", type=int, default=2)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--train-start-year", type=int, default=2015)
    p.add_argument("--train-end-year", type=int, default=2021)
    p.add_argument("--input-duration", default="12h")
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--temporal-location", default="mesh_processor_interleaved")
    p.add_argument("--temporal-hidden-size", type=int, default=128)
    p.add_argument("--temporal-d-inner", type=int, default=128)
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
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-bias-init", type=float, default=-3.5)
    p.add_argument("--baseline-mode", action="store_true",
                   help="Eval a vanilla GCResidualWithZeroHead ckpt (no gate). "
                        "Gate-related diagnostics will be empty.")
    return p.parse_args()


def main():
    cfg = parse_args()
    K = cfg.target_steps
    print(f"[top-pct] ckpt: {cfg.ckpt}")

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
        if cfg.baseline_mode:
            p = GCResidualWithZeroHead(model_cfg_residual, task_cfg)
        else:
            p = GCResidualWithGatedExtremeHead(
                model_cfg_residual, task_cfg,
                gate_bias_init=cfg.gate_bias_init)
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

    val_indices = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, K)
    sample_inputs, sample_targets, sample_forcings = base_train.build_batch_from_indices(
        eval_ds, indices=[int(val_indices[0])],
        input_steps=input_steps, target_steps=K,
        task_cfg=task_cfg, dt=dt)
    sample_targets_1step = sample_targets.isel(time=slice(0, 1))
    sample_forcings_1step = sample_forcings.isel(time=slice(0, 1))

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_b, k_r = jax.random.split(rng, 3)
    baseline_params, baseline_state = baseline_predict.init(
        k_b, sample_inputs, sample_targets_1step, sample_forcings_1step)
    baseline_params, _ = overlay_matching_params(
        baseline_params, ckpt_in.params, strict=True)
    _, residual_state_init = residual_predict.init(
        k_r, sample_inputs, sample_targets_1step, sample_forcings_1step)

    with open(cfg.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    residual_params = ckpt["residual_params"]
    residual_state = residual_state_init  # fresh zero state for eval

    @jax.jit
    def _baseline_step(p, s, k, x, t, f):
        return baseline_predict.apply(p, s, k, x, t, f)
    @jax.jit
    def _residual_step(p, s, k, x, t, f):
        return residual_predict.apply(p, s, k, x, t, f)

    def _shift_inputs_with_field(prev, new, frc):
        ttime = prev.time.values[-1:] + dt
        ns = new.assign_coords(time=ttime); fn = frc.assign_coords(time=ttime)
        nxt = xr.merge([ns, fn])
        if "datetime" in nxt.coords: nxt = nxt.drop_vars("datetime")
        keys = [k for k in nxt.data_vars if k in prev.data_vars]
        nxtp = nxt[keys]
        return xr.concat([prev, nxtp], dim="time", data_vars="different").tail(time=input_steps)

    # Sample selection
    rng_np = np.random.default_rng(cfg.seed)
    chosen_idx = sorted(rng_np.choice(val_indices, size=cfg.n_samples,
                                       replace=False).tolist())

    lat = eval_ds["lat"].values
    cos_lat = np.cos(np.deg2rad(lat))
    cos_lat_norm = cos_lat / cos_lat.mean()

    # Accumulators: per (var, lead) we accumulate per-cell absolute errors and
    # baseline-error magnitudes across samples, then post-hoc rank to compute
    # top-percentile metrics.
    # To avoid memory blowup at res=1.0, we accumulate streaming sums per percentile
    # bin (e.g. 95th, 99th of |err_b| per (var,lead) per sample), then average.
    # Streaming approach: per sample, compute per-cell |err_b|, take top 5%/1%
    # cells from THIS sample, compute RMSE_b and RMSE_f on those cells, store.

    top5_per_var = {}   # var → list[(K,) arrays of RMSE_b, RMSE_f]
    top1_per_var = {}
    gate_stats_per_var = {}

    for s_i, idx in enumerate(chosen_idx):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=K, task_cfg=task_cfg, dt=dt)
        cur = inp
        bs = baseline_state
        rs = residual_state
        bp_chunks = []; full_chunks = []
        gate_state_per_step = []
        rng_l = jax.random.PRNGKey(cfg.seed + s_i)
        for k in range(K):
            tk = tgt.isel(time=slice(k, k+1))
            fk = frc.isel(time=slice(k, k+1))
            rng_l, kb, kr = jax.random.split(rng_l, 3)
            bp, bs = _baseline_step(baseline_params, bs, kb, cur, tk, fk)
            rp, rs = _residual_step(residual_params, rs, kr, cur, tk, fk)
            bp_chunks.append(bp); full_chunks.append(
                jax.tree_util.tree_map(lambda b, r: b + r, bp, rp))
            # Capture gate values from state (per variable)
            step_gate = {}
            for sk, val in rs.get("~", {}).items():
                if sk.startswith("gate__"):
                    var_name = sk.replace("gate__", "")
                    step_gate[var_name] = np.asarray(val)
            gate_state_per_step.append(step_gate)
            if k < K - 1:
                cur = _shift_inputs_with_field(cur, bp, fk)

        bp_traj = xr.concat(bp_chunks, dim="time")
        fp_traj = xr.concat(full_chunks, dim="time")

        # Per-var per-lead top-percentile RMSE
        for var in tgt.data_vars:
            truth = tgt[var].astype("float32")
            bp_v  = bp_traj[var].astype("float32")
            fp_v  = fp_traj[var].astype("float32")
            err_b = (bp_v - truth)  # (batch, time, ..., lat, lon)
            err_f = (fp_v - truth)
            # Make per-cell |err_b| for ranking, per lead k
            if var not in top5_per_var:
                top5_per_var[var] = {"rmse_b": np.zeros(K), "rmse_f": np.zeros(K), "n": np.zeros(K)}
                top1_per_var[var] = {"rmse_b": np.zeros(K), "rmse_f": np.zeros(K), "n": np.zeros(K)}
                gate_stats_per_var[var] = {"gate_mean": np.zeros(K), "gate_on_extreme": np.zeros(K),
                                            "false_alarm": np.zeros(K), "n": np.zeros(K)}
            for k in range(K):
                eb_k = np.abs(err_b.isel(time=k).values).reshape(-1)
                ef_k = np.abs(err_f.isel(time=k).values).reshape(-1)
                eb_sq = eb_k ** 2
                ef_sq = ef_k ** 2
                ncells = len(eb_k)
                k5 = max(1, int(ncells * 0.05))
                k1 = max(1, int(ncells * 0.01))
                # Top-5%: indices with largest |err_b|
                idx5 = np.argpartition(eb_k, -k5)[-k5:]
                idx1 = np.argpartition(eb_k, -k1)[-k1:]
                top5_per_var[var]["rmse_b"][k] += float(np.sqrt(np.mean(eb_sq[idx5])))
                top5_per_var[var]["rmse_f"][k] += float(np.sqrt(np.mean(ef_sq[idx5])))
                top5_per_var[var]["n"][k] += 1
                top1_per_var[var]["rmse_b"][k] += float(np.sqrt(np.mean(eb_sq[idx1])))
                top1_per_var[var]["rmse_f"][k] += float(np.sqrt(np.mean(ef_sq[idx1])))
                top1_per_var[var]["n"][k] += 1

                # Gate statistics if available
                if var in gate_state_per_step[k]:
                    g = gate_state_per_step[k][var].reshape(-1)
                    if g.size == ncells:
                        extreme_mask = np.zeros(ncells, dtype=bool)
                        extreme_mask[idx5] = True
                        gate_stats_per_var[var]["gate_mean"][k] += float(np.mean(g))
                        gate_stats_per_var[var]["gate_on_extreme"][k] += float(np.mean(g[extreme_mask]))
                        # False alarm: gate>0.5 on non-extreme
                        non_ext = ~extreme_mask
                        if non_ext.sum() > 0:
                            gate_stats_per_var[var]["false_alarm"][k] += float(np.mean(g[non_ext] > 0.5))
                        gate_stats_per_var[var]["n"][k] += 1

        if s_i < 3 or (s_i + 1) % 4 == 0:
            print(f"[top-pct] sample {s_i+1}/{cfg.n_samples}", flush=True)

    # Reduce
    out = {
        "ckpt": cfg.ckpt,
        "target_steps": K,
        "n_samples": cfg.n_samples,
        "top5_residual_per_variable": {},
        "top1_residual_per_variable": {},
        "gate_statistics_per_variable": {},
    }
    for var, d in top5_per_var.items():
        n = np.maximum(d["n"], 1.0)
        rb = d["rmse_b"] / n; rf = d["rmse_f"] / n
        imp = (rb - rf) / np.maximum(rb, 1e-12) * 100
        out["top5_residual_per_variable"][var] = dict(
            rmse_baseline=rb.tolist(), rmse_full=rf.tolist(),
            improvement_pct=imp.tolist())
    for var, d in top1_per_var.items():
        n = np.maximum(d["n"], 1.0)
        rb = d["rmse_b"] / n; rf = d["rmse_f"] / n
        imp = (rb - rf) / np.maximum(rb, 1e-12) * 100
        out["top1_residual_per_variable"][var] = dict(
            rmse_baseline=rb.tolist(), rmse_full=rf.tolist(),
            improvement_pct=imp.tolist())
    for var, d in gate_stats_per_var.items():
        n = np.maximum(d["n"], 1.0)
        out["gate_statistics_per_variable"][var] = dict(
            gate_mean=(d["gate_mean"] / n).tolist(),
            gate_on_extreme=(d["gate_on_extreme"] / n).tolist(),
            false_alarm_rate=(d["false_alarm"] / n).tolist(),
        )

    # Quick console summary
    print()
    print(f"=== Top-percentile residual RMSE summary (K={K}) ===")
    for var in ["2m_temperature", "10m_u_component_of_wind"]:
        if var in out["top5_residual_per_variable"]:
            d5 = out["top5_residual_per_variable"][var]
            d1 = out["top1_residual_per_variable"][var]
            g  = out.get("gate_statistics_per_variable", {}).get(var, {})
            print(f"\n  {var}")
            print(f"    {'lead':>5s}  {'top5_imp':>9s}  {'top1_imp':>9s}  {'gate':>6s}  {'gate@ext':>9s}")
            for k in [0, 9, 19, 29, 39]:
                if k < K:
                    g_mean = g.get('gate_mean', [0]*K)[k] if g else 0
                    g_ext  = g.get('gate_on_extreme', [0]*K)[k] if g else 0
                    print(f"    L{k+1:>3d}  {d5['improvement_pct'][k]:>+8.2f}%  "
                          f"{d1['improvement_pct'][k]:>+8.2f}%  {g_mean:>6.3f}  {g_ext:>9.3f}")

    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_json).write_text(json.dumps(out, indent=1))
    print(f"\n[top-pct] wrote {cfg.out_json}")


if __name__ == "__main__":
    main()
