"""Build SWA ckpts for res=2 K=18 arch-screen (7 configs).

Uniform-average residual_params over ckpts step {3000, 4000, 5000, 6000, 7000, 8000}
per config. residual_state is dropped (eval uses zero-init anyway).
"""
from __future__ import annotations
from pathlib import Path
import pickle
import jax

ROOT = Path("/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22_res2_closedloop")
CONFIGS = ["inner128", "inner256", "inner512", "inner1024", "state32", "depth4", "conv8"]
STEPS = [3000, 4000, 5000, 6000, 7000, 8000]

for name in CONFIGS:
    cfg_dir = ROOT / f"K18_res2_{name}"
    ckpt_paths = [cfg_dir / f"v13_residual_step{s}.pkl" for s in STEPS]
    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        print(f"{name}: MISSING {[p.name for p in missing]}"); continue
    params_list = []
    residual_state_first = None
    for p in ckpt_paths:
        with p.open("rb") as f:
            ck = pickle.load(f)
        params_list.append(ck["residual_params"])
        if residual_state_first is None and "residual_state" in ck:
            residual_state_first = ck["residual_state"]
    n = len(params_list)
    swa_params = jax.tree.map(
        lambda *ps: sum(ps) / n, *params_list)
    out_p = cfg_dir / f"v13_residual_SWA_step3k-8k.pkl"
    with out_p.open("wb") as f:
        pickle.dump({"residual_params": swa_params,
                     "residual_state": residual_state_first,
                     "swa_source_steps": STEPS,
                     "swa_source_ckpts": [str(p) for p in ckpt_paths]}, f)
    print(f"{name}: SWA of {n} ckpts → {out_p}")

print("Done.")
