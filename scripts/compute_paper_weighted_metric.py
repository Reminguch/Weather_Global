#!/usr/bin/env python3
"""Post-hoc: compute a paper-style weighted-overall ΔTF from eval_log.json.

The raw ``overall_MAE`` reported by train_mz_* scripts is an unweighted
per-channel average, which is dominated by variables with large absolute
magnitudes (geopotential in m^2/s^2, MSLP in Pa) and essentially ignores
variables with small numeric units (specific_humidity in kg/kg, precip in m).

This script reconstructs a weighting closer to GraphCast's training loss:
  * per-variable weight w_var: MSLP/10m_u/10m_v/precip = 0.1, rest = 1.0.
  * normalise each variable's MAE by its level-averaged diffs_stddev,
    making variables comparable across physical units.
  * (level weights are NOT applied because eval_log only stores
    level-aggregated per-variable MAE; the proper level weighting lives
    inside training loss but cannot be recovered from per-variable MAE
    without the full tensor.)

This is not an exact replica of the training loss' overall metric, but it
is much closer to paper-style reporting than the raw overall MAE.

Usage:
    python scripts/compute_paper_weighted_metric.py <run_dir_or_glob>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# GraphCast per-variable loss weights (matches
# third_party/graphcast/graphcast/graphcast.py:477-490).
PER_VARIABLE_LOSS_WEIGHTS: dict[str, float] = {
    "2m_temperature": 1.0,
    "mean_sea_level_pressure": 0.1,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "total_precipitation_6hr": 0.1,
    "temperature": 1.0,
    "geopotential": 1.0,
    "u_component_of_wind": 1.0,
    "v_component_of_wind": 1.0,
    "vertical_velocity": 1.0,
    "specific_humidity": 1.0,
}

PRESSURE_LEVEL_VARS = {
    "temperature", "geopotential", "u_component_of_wind",
    "v_component_of_wind", "vertical_velocity", "specific_humidity",
}

DEFAULT_STATS_DIR = Path(
    "/scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/stats"
)


def load_diffs_std(stats_dir: Path, pressure_levels) -> dict[str, float]:
    """Load per-variable diffs_stddev (level-averaged for 3D vars)."""
    ds = xr.open_dataset(stats_dir / "diffs_stddev_by_level.nc")
    out: dict[str, float] = {}
    for name in PER_VARIABLE_LOSS_WEIGHTS.keys():
        if name not in ds:
            continue
        da = ds[name]
        if "level" in da.dims:
            arr = da.sel(level=list(pressure_levels)).values.astype(np.float64)
        else:
            arr = np.asarray([float(da.values)], dtype=np.float64)
        out[name] = float(np.mean(arr))
    return out


def compute_weighted(metrics: dict, which_vars: list[str],
                     diffs_std: dict[str, float]) -> dict[str, float]:
    """Compute weighted MAE / RMSE / ΔTF from per-variable MAEs in `metrics`.

    Returns a dict with {baseline_weighted_MAE, corrected_weighted_MAE,
    weighted_ΔTF_percent, ...}.
    """
    num = 0.0
    num_baseline = 0.0
    den = 0.0
    for v in which_vars:
        w = PER_VARIABLE_LOSS_WEIGHTS.get(v)
        std = diffs_std.get(v)
        if w is None or std is None:
            continue
        base = metrics.get(f"baseline_{v}_MAE")
        mz = metrics.get(f"corrected_{v}_MAE")
        if base is None or mz is None:
            continue
        num_baseline += w * (base / std)
        num += w * (mz / std)
        den += w
    result = {}
    if den > 0:
        result["baseline_weighted_MAE_norm"] = num_baseline / den
        result["corrected_weighted_MAE_norm"] = num / den
        result["weighted_delta_pct"] = (
            100.0 * (num_baseline - num) / num_baseline if num_baseline > 0 else 0.0
        )
    return result


def process_run(run_dir: Path, diffs_std: dict[str, float], vars_set: list[str]) -> None:
    eval_log = run_dir / "eval_log.json"
    if not eval_log.exists():
        return
    with eval_log.open() as f:
        log = json.load(f)
    if not log:
        return
    name = run_dir.name
    print(f"\n=== {name} ===")
    print(f"{'step':>4s}  {'raw_ΔTF':>8s}  {'paper-weighted_ΔTF':>20s}")
    for ev in log:
        s = ev["step"]
        base_ovr = ev["baseline_overall_MAE"]
        mz_ovr = ev["corrected_overall_MAE"]
        raw_imp = 100.0 * (base_ovr - mz_ovr) / base_ovr
        w = compute_weighted(ev, vars_set, diffs_std)
        w_imp = w.get("weighted_delta_pct", float("nan"))
        print(f"{s:>4d}  {raw_imp:>+7.3f}%  {w_imp:>+19.3f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    args = p.parse_args()

    # Hard-code the 13 pressure levels used by GraphCast_small.
    # (Matches task_cfg.pressure_levels in all our runs.)
    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    diffs_std = load_diffs_std(args.stats_dir, pressure_levels)

    print("diffs_stddev per variable (level-averaged):")
    for k, v in diffs_std.items():
        print(f"  {k:30s}  {v:.5g}")

    for d in args.run_dirs:
        # Determine which variable set the run used from run_config.
        rc = d / "run_config.json"
        if rc.exists():
            cfg = json.load(rc.open())
            vars_set = list(cfg.get("resolved_variables", list(PER_VARIABLE_LOSS_WEIGHTS.keys())))
        else:
            vars_set = list(PER_VARIABLE_LOSS_WEIGHTS.keys())
        process_run(d, diffs_std, vars_set)


if __name__ == "__main__":
    main()
