"""Compute residual alignment (cos), gain, and Δ_t = E^pre - E^post for any
completed eval JSON. For JSONs without raw <r,e_pre>/||r||^2 fields (i.e. v22_clean
JSONs before the diagnostic instrumentation), derives Δ_t directly from
RMSE_baseline^2 - RMSE_full^2 (exact) and uses ablate-JSON cos/gain when present.

Usage:
  python residual_alignment_check.py <json_path>

Reports per-lead:
  - RMSE_baseline, RMSE_full
  - Δ_t  (positive = residual helps that step; negative = hurts)
  - cos(r, e_pre)  if available
  - gain ||r||/||e_pre||  if available
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


def load_and_diag(json_path: Path, vars_of_interest=None):
    if vars_of_interest is None:
        vars_of_interest = [
            ("2m_temperature", "per_variable_per_step"),
            ("10m_u_component_of_wind", "per_variable_per_step"),
            ("geopotential_level500", "per_channel_per_step"),
            ("temperature_level850", "per_channel_per_step"),
        ]
    with json_path.open() as f:
        d = json.load(f)

    diag_src = d.get("residual_diagnostics_per_variable", {})

    print()
    print("=" * 100)
    print(f"{json_path.name}")
    print(f"  mode={d.get('eval_mode') or d.get('ablation_mode')}  W={d.get('warmup_steps')}  K={d.get('target_steps')}")
    print("=" * 100)

    leads_idx = [0, 4, 9, 14, 19, 24, 29, 39]   # L1, L5, L10, L15, L20, L25, L30, L40
    lead_str = ["L1", "L5", "L10", "L15", "L20", "L25", "L30", "L40"]

    for var, src_key in vars_of_interest:
        src = d.get(src_key, {})
        if var not in src:
            continue
        v = src[var]
        rb = np.asarray(v["rmse_baseline"], dtype=float)
        rf = np.asarray(v["rmse_full"],     dtype=float)
        Epre  = rb * rb
        Epost = rf * rf
        delta = Epre - Epost      # > 0 = residual helps, < 0 = hurts

        # cos / gain if instrumented
        diag = diag_src.get(var) if isinstance(diag_src, dict) else None
        cos  = np.asarray(diag.get("residual_cosine_by_lead",   []), dtype=float) if diag else None
        gain = np.asarray(diag.get("residual_gain_by_lead",     []), dtype=float) if diag else None
        delta_d = np.asarray(diag.get("delta_mse_by_lead",      []), dtype=float) if diag else None

        print(f"\n  {var}")
        header = f"    {'lead':>5s}  {'RMSE_b':>8s}  {'RMSE_f':>8s}  {'Δ_t':>10s}  {'imp%':>7s}"
        if cos is not None and cos.size > 0:
            header += f"  {'cos':>7s}  {'gain':>6s}"
        print(header)
        for ki, ls in zip(leads_idx, lead_str):
            if ki >= len(rb): continue
            line = f"    {ls:>5s}  {rb[ki]:>8.4f}  {rf[ki]:>8.4f}  {delta[ki]:>+10.4f}  {((rb[ki]-rf[ki])/rb[ki]*100):>+6.2f}%"
            if cos is not None and cos.size > ki:
                line += f"  {cos[ki]:>+7.3f}  {gain[ki]:>6.3f}"
            print(line)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: dump for all v22_clean K=18 step20000 ablation/warm jsons
        files = [
            "/home/lm8598/Weather_Global_experiments/results/2026-6-26-K18-mem-ablate/K18_step20000_ablate_normal_W0.json",
            "/home/lm8598/Weather_Global_experiments/results/2026-6-26-K18-mem-ablate/K18_step20000_ablate_reset_ssm_W0.json",
            "/home/lm8598/Weather_Global_experiments/results/2026-6-26-K18-mem-ablate/K18_step20000_ablate_reset_all_W0.json",
            "/home/lm8598/Weather_Global_experiments/results/2026-6-26-K18-mem-ablate/K18_step20000_ablate_stale_W24.json",
            "/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba/K18_step20000_truthwarm_zero_24h_full.json",
            "/home/lm8598/Weather_Global_experiments/results/2026-6-25-Res2Mamba/K18_step20000_truthwarm_zero_240h_full.json",
        ]
        for f in files:
            p = Path(f)
            if p.exists():
                load_and_diag(p)
            else:
                print(f"[skip] {p.name} not found")
    else:
        for arg in sys.argv[1:]:
            load_and_diag(Path(arg))
