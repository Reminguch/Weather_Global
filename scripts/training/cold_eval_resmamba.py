"""Cold-eval residual_mamba ckpts using the canonical inference-engine pattern.

Reads ALL architecture (mesh, width, mp, temporal config, baseline_ckpt path,
residual_ar_feedback mode) from the ckpt's saved run_config.json — no need
to re-specify on the CLI. Matches gc_mamba's inference engine convention.

Usage:
  python -u cold_eval_resmamba.py \\
      --residual-ckpt /path/to/resmamba_step20000.npz \\
      --data-path /path/to/wb2.zarr \\
      --stats-dir /path/to/stats \\
      --n-anchors 32 --target-steps 40 \\
      --out-json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "graphcast"))

import scripts.training.train_graphcast as base_train  # noqa: E402
from src.models.graphcast.runtime import (  # noqa: E402
    build_run_jitted as gc_build_run_jitted,
    load_checkpoint_and_stats,
    load_run_config,
)
from src.models.mamba.residual_mamba.inference.inference_engine import (  # noqa: E402
    ResidualMambaInferenceEngine,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--residual-ckpt", required=True,
                   help="residual_mamba ckpt; baseline ckpt path auto-read from its run_config.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--stats-dir", required=True)
    p.add_argument("--val-year", type=int, default=2022)
    p.add_argument("--n-anchors", type=int, default=32)
    p.add_argument("--target-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def _to_numpy_dataset(ds):
    """Convert any JaxArrayWrapper inside an xarray Dataset to plain numpy."""
    new_vars = {}
    for var in ds.data_vars:
        da = ds[var]
        d = da.data
        if hasattr(d, "_data"):
            d = d._data
        arr_np = np.asarray(jax.device_get(d))
        new_vars[var] = (da.dims, arr_np)
    new_coords = {}
    for c in ds.coords:
        cv = ds[c].values
        if hasattr(cv, "_data"):
            cv = cv._data
        new_coords[c] = np.asarray(cv)
    return xr.Dataset(new_vars, coords=new_coords)


def main():
    cfg = parse_args()

    # --- Load residual engine. Auto-reads baseline_ckpt + architecture from run_config. ---
    print(f"[cold-eval] loading residual engine from {cfg.residual_ckpt}", flush=True)
    residual_engine = ResidualMambaInferenceEngine(
        cfg.residual_ckpt, cfg.stats_dir,
    ).load()
    res_run_cfg = residual_engine.run_cfg
    task_cfg = residual_engine.task_cfg

    # --- Load baseline engine separately (graphcast self-rollout). ---
    baseline_ckpt_path = res_run_cfg.get("residual_training", {}).get("baseline_checkpoint")
    if not baseline_ckpt_path:
        raise ValueError("residual ckpt's run_config missing residual_training.baseline_checkpoint")
    print(f"[cold-eval] loading baseline engine from {baseline_ckpt_path}", flush=True)
    base_ckpt_obj, stats = load_checkpoint_and_stats(Path(baseline_ckpt_path), Path(cfg.stats_dir))
    baseline_run, base_task_cfg, _, _ = gc_build_run_jitted(
        base_ckpt_obj, stats, Path(baseline_ckpt_path),
    )

    # --- Data: open val_year split, anchor sample ---
    class _SplitCfg:
        data_path = cfg.data_path
        resolution = float(res_run_cfg.get("resolution") or task_cfg.resolution if hasattr(task_cfg, "resolution") else 2.0)
        val_year = cfg.val_year
        train_start_year = 2015
        train_end_year = 2021
    _, eval_ds = base_train._open_local_splits(_SplitCfg)
    eval_ds = base_train.prepare_dataset_for_task(eval_ds, task_cfg)
    dt = base_train.infer_time_step(eval_ds)
    input_steps = base_train.input_steps_from_duration(task_cfg.input_duration, dt)
    valid_idx = base_train.valid_final_input_indices(
        eval_ds.sizes["time"], input_steps, cfg.target_steps,
    )
    rng_np = np.random.default_rng(cfg.seed)
    anchor_indices = sorted(
        rng_np.choice(valid_idx, size=cfg.n_anchors, replace=False).tolist(),
    )

    # --- Lat weighting + accumulators ---
    lat_vals = eval_ds.lat.values
    cos_lat = np.cos(np.deg2rad(lat_vals))
    cos_lat_n = cos_lat / cos_lat.mean()
    target_vars = list(task_cfg.target_variables)
    K = cfg.target_steps
    sum_sq = {m: {v: np.zeros(K, dtype=np.float64) for v in target_vars}
              for m in ["baseline", "resmamba"]}
    sum_bias = {m: {v: np.zeros(K, dtype=np.float64) for v in target_vars}
                for m in ["baseline", "resmamba"]}
    anchor_times = []

    print(f"[cold-eval] {cfg.n_anchors} anchors × {K} leads", flush=True)
    for ai, idx in enumerate(anchor_indices):
        inp, tgt, frc = base_train.build_batch_from_indices(
            eval_ds, indices=[int(idx)], input_steps=input_steps,
            target_steps=K, task_cfg=task_cfg, dt=dt,
        )
        anchor_times.append(str(pd.Timestamp(eval_ds.time.values[idx])))

        rng = jax.random.PRNGKey(cfg.seed + 99 + ai)
        rng_b, rng_r = jax.random.split(rng, 2)

        # Each engine takes (rng, inputs, targets_template, forcings) and rolls
        # target_steps leads. Returns xr.Dataset with time dim = target_steps.
        bp_preds_full = baseline_run(rng=rng_b, inputs=inp, targets_template=tgt, forcings=frc)
        rp_preds_full = residual_engine.rollout(rng=rng_r, inputs=inp, targets_template=tgt, forcings=frc)

        bp_np = _to_numpy_dataset(bp_preds_full)
        rp_np = _to_numpy_dataset(rp_preds_full)

        for k in range(K):
            for var in target_vars:
                truth_da = tgt[var].isel(time=slice(k, k + 1)).astype("float32").transpose(..., "lat", "lon")
                bp_da = bp_np[var].isel(time=slice(k, k + 1)).astype("float32").transpose(..., "lat", "lon")
                rp_da = rp_np[var].isel(time=slice(k, k + 1)).astype("float32").transpose(..., "lat", "lon")
                truth_v = np.asarray(truth_da.values).squeeze(0).squeeze(0)
                bp_v = np.asarray(bp_da.values).squeeze(0).squeeze(0)
                rp_v = np.asarray(rp_da.values).squeeze(0).squeeze(0)
                if bp_v.ndim == 3:
                    bp_v = bp_v.mean(axis=0)
                    rp_v = rp_v.mean(axis=0)
                    truth2 = truth_v.mean(axis=0)
                else:
                    truth2 = truth_v
                eb = bp_v - truth2
                er = rp_v - truth2
                sum_sq["baseline"][var][k] += float(((eb ** 2) * cos_lat_n[:, None]).mean())
                sum_sq["resmamba"][var][k] += float(((er ** 2) * cos_lat_n[:, None]).mean())
                sum_bias["baseline"][var][k] += float((eb * cos_lat_n[:, None]).mean())
                sum_bias["resmamba"][var][k] += float((er * cos_lat_n[:, None]).mean())

        if (ai + 1) % 8 == 0 or ai == 0:
            print(f"  anchor {ai + 1}/{cfg.n_anchors}", flush=True)

    # --- Aggregate to RMSE / RMSB ---
    n = cfg.n_anchors
    per_var = {}
    for var in target_vars:
        rmse_b = np.sqrt(sum_sq["baseline"][var] / n)
        rmse_r = np.sqrt(sum_sq["resmamba"][var] / n)
        rmsb_b = np.sqrt((sum_bias["baseline"][var] / n) ** 2)
        rmsb_r = np.sqrt((sum_bias["resmamba"][var] / n) ** 2)
        improvement = (1 - rmse_r / np.maximum(rmse_b, 1e-12)) * 100
        per_var[var] = dict(
            rmse_baseline=rmse_b.tolist(),
            rmse_resmamba=rmse_r.tolist(),
            improvement_pct=improvement.tolist(),
            rmsb_baseline=rmsb_b.tolist(),
            rmsb_resmamba=rmsb_r.tolist(),
        )

    out = {
        "residual_ckpt": str(cfg.residual_ckpt),
        "baseline_ckpt": str(baseline_ckpt_path),
        "eval_config": {
            "n_anchors": cfg.n_anchors,
            "target_steps": K,
            "lead_hours_per_step": 6,
            "feedback_mode_from_run_cfg": res_run_cfg.get(
                "residual_training", {}
            ).get("autoregressive_feedback_mode"),
        },
        "anchor_times": anchor_times,
        "target_variables": target_vars,
        "per_variable_per_lead": per_var,
    }
    Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_json, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {cfg.out_json}", flush=True)

    print()
    print(f"=== Per-variable lead-mean RMSE improvement (resmamba vs baseline, n={n}) ===")
    print(f"  {'variable':<32}{'rmse_base':>12}{'rmse_rm':>12}{'imp_%':>10}")
    for var in target_vars:
        v = per_var[var]
        b_mean = float(np.mean(v["rmse_baseline"]))
        r_mean = float(np.mean(v["rmse_resmamba"]))
        imp = float(np.mean(v["improvement_pct"]))
        print(f"  {var:<32}{b_mean:>12.4f}{r_mean:>12.4f}{imp:>+9.2f}%")


if __name__ == "__main__":
    main()
