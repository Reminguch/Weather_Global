"""Compare v22 K-scan zero-init (re-eval) vs ckpt-init (original) improvement_pct.

Reads:
  zero-init: /home/lm8598/.../2026-6-26-v22-kscan-zero-init/v22_K{K}_K40_zero_init.json
  ckpt-init: /home/lm8598/.../2026-05-23-v22/eval_jsons/v22_K{K}_K40.json

Reports per K, at lead 240h (L40), for 4 key vars: 2m_T, u10, z500@500, t850@850
  - ckpt_init improvement_pct_rmse
  - zero_init improvement_pct_rmse
  - delta = zero - ckpt  (negative = ckpt loaded warm state helps eval)
"""
from __future__ import annotations
import json
from pathlib import Path

CKPT_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-05-23-v22/eval_jsons")
ZERO_DIR = Path("/home/lm8598/Weather_Global_experiments/results/2026-6-26-v22-kscan-zero-init")

KS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]


def get_l40(d, var, lead_idx=39):
    pv = d.get('per_variable_per_step', {})
    pc = d.get('per_channel_per_step', {})
    if var.startswith('geopotential_level') or var.startswith('temperature_level'):
        src = pc
    else:
        src = pv
    if var not in src:
        return None
    imp = src[var].get('improvement_pct_rmse') or src[var].get('improvement_pct')
    return imp[lead_idx] if imp and len(imp) > lead_idx else None


def main():
    print("=" * 110)
    print("v22 K-scan: zero-init re-eval (10310646) vs ckpt-init (original)")
    print("at L40 (240h), improvement_pct_rmse")
    print("=" * 110)
    print(f"{'K':>3s} | "
          f"{'2m_T (ckpt/zero/Δ)':>28s} | "
          f"{'u10 (ckpt/zero/Δ)':>28s} | "
          f"{'z500 (ckpt/zero/Δ)':>28s} | "
          f"{'t850 (ckpt/zero/Δ)':>28s}")
    print("-" * 110)

    rows = []
    for K in KS:
        ckpt_p = CKPT_DIR / f"v22_K{K}_K40.json"
        zero_p = ZERO_DIR / f"v22_K{K}_K40_zero_init.json"
        if not zero_p.exists():
            print(f"K={K:<2d} | zero-init MISSING (waiting on job 10310646)")
            continue
        with ckpt_p.open() as f:
            dc = json.load(f)
        with zero_p.open() as f:
            dz = json.load(f)
        line = f"K={K:<2d} |"
        row = {'K': K}
        for var in ['2m_temperature', '10m_u_component_of_wind',
                    'geopotential_level500', 'temperature_level850']:
            c = get_l40(dc, var)
            z = get_l40(dz, var)
            if c is None or z is None:
                line += f"  {'n/a':>28s} |"
                continue
            delta = z - c
            line += f"  {c:>+7.2f}% / {z:>+7.2f}% / {delta:>+6.2f} |"
            row[var] = dict(ckpt=c, zero=z, delta=delta)
        print(line)
        rows.append(row)

    # Summary: how much of K-curriculum gain survives zero-init?
    if rows:
        print()
        print("=" * 110)
        print("Survival rate: zero / ckpt  (closer to 1.0 = improvement not init-dependent)")
        print("=" * 110)
        for r in rows:
            K = r['K']
            line = f"K={K:<2d} |"
            for var in ['2m_temperature', '10m_u_component_of_wind',
                        'geopotential_level500', 'temperature_level850']:
                if var not in r:
                    line += f"  {'n/a':>10s} |"
                    continue
                c = r[var]['ckpt']
                z = r[var]['zero']
                if abs(c) < 0.05:
                    line += f"  {'(noise)':>10s} |"
                else:
                    ratio = z / c
                    line += f"  {ratio:>+8.2f}x |"
            print(line)


if __name__ == "__main__":
    main()
