#!/usr/bin/env python3
"""Generic SWA builder: uniform-average ckpts over a step window.

Usage:
  build_swa_generic.py --run-glob '<base>/K{K}_*/*' --ks 4,8,12,16,20,22 \
      --steps 6000,8000,10000,12000,14000,16000 [--out-name v13_residual_SWA_step6k-16k.pkl]
Each resolved dir must contain v13_residual_step{s}.pkl for every step.
"""
import argparse
import glob
import pickle
from pathlib import Path
import jax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-glob", required=True,
                   help="glob with {K}, resolves to the ckpt dir per K")
    p.add_argument("--ks", default="4,8,12,16,20,22")
    p.add_argument("--steps", default="6000,8000,10000,12000,14000,16000")
    p.add_argument("--out-name", default="v13_residual_SWA_step6k-16k.pkl")
    cfg = p.parse_args()
    ks = [int(x) for x in cfg.ks.split(",")]
    steps = [int(x) for x in cfg.steps.split(",")]

    failures = []
    for K in ks:
        pat = cfg.run_glob.replace("{K}", str(K))
        dirs = sorted(glob.glob(pat))
        if not dirs:
            print(f"K={K}: NO dir matches {pat}"); failures.append(K); continue
        ckpt_dir = Path(dirs[0])
        paths = [ckpt_dir / f"v13_residual_step{s}.pkl" for s in steps]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            print(f"K={K}: MISSING {missing} in {ckpt_dir}"); failures.append(K); continue
        # rebuild fresh: never leave a stale SWA from a previous run in place
        out = ckpt_dir / cfg.out_name
        if out.exists():
            out.unlink()
        params_list, rs_first = [], None
        for pp in paths:
            with pp.open("rb") as f:
                ck = pickle.load(f)
            params_list.append(ck["residual_params"])
            if rs_first is None and "residual_state" in ck:
                rs_first = ck["residual_state"]
        n = len(params_list)
        swa = jax.tree.map(lambda *ps: sum(ps) / n, *params_list)
        out = ckpt_dir / cfg.out_name
        with out.open("wb") as f:
            pickle.dump({"residual_params": swa, "residual_state": rs_first,
                         "swa_source_steps": steps,
                         "swa_source_ckpts": [str(x) for x in paths]}, f)
        print(f"K={K:2d}: SWA of {n} ckpts -> {out}")
    if failures:
        import sys
        print(f"FAILED to build SWA for K={failures} (missing dir/ckpts). Exiting non-zero.")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
