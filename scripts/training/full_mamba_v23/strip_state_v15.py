#!/usr/bin/env python
"""Prepare the v15 step20000 ckpt for use as a v23 (d_conv=8) warm-start.

Background — the v15 ckpt has TWO d_conv=4-shape mismatches when resumed
into a d_conv=8 model:

1. `residual_state` contains `conv_cache` of shape (1, 10242, 128, 3).
   v23 expects (1, 10242, 128, 7). Mamba state is freshly zeroed at training
   start anyway, so this script drops residual_state entirely.

2. `residual_params` contains `mamba_block_*/conv1d/kernel` of shape (128, 4).
   v23 expects (128, 8). This script zero-pads the kernel along the time
   axis at the FRONT (positions 0..3) and preserves v15's learned weights at
   the BACK (positions 4..7). This is the causal-conv convention because in
   _DepthwiseCausalConv1D, `kernel[:, -1]` weights the current timestep and
   `kernel[:, 0]` weights the oldest. Front-padding means v23's "new past"
   positions start at zero and the model is initially identical to v15 on
   the receptive field that overlaps.

Other Mamba parameters (A_log, D, in_proj, out_proj, x_proj, dt_proj, conv1d.bias,
layer_norm) have shapes that don't depend on d_conv — they are passed through
unchanged.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in-ckpt",
        default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v15_v13arch_7yr_v2/v15_20k_v2/v13_residual_step20000.pkl",
    )
    p.add_argument(
        "--out-ckpt",
        default="/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v23_helpers/v15_step20000_for_dconv8.pkl",
    )
    p.add_argument("--new-d-conv", type=int, default=8, help="Target d_conv")
    args = p.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("rb") as f:
        ck = pickle.load(f)

    params = ck.get("residual_params", {})
    old_total = sum(p.size for kk in params.values() for p in kk.values())
    print(f"input: {in_path}")
    print(f"  modules: {len(params)}")
    print(f"  total params: {old_total:,}")
    print(f"  residual_state present? {'residual_state' in ck and bool(ck.get('residual_state'))}")
    print()

    # Pad conv1d.kernel from (D_inner, old_d_conv) to (D_inner, new_d_conv) by
    # prepending zeros.
    new_d_conv = args.new_d_conv
    padded = []
    for mod_name in list(params.keys()):
        if "conv1d" not in mod_name:
            continue
        if "kernel" not in params[mod_name]:
            continue
        old_kernel = params[mod_name]["kernel"]
        old_d_conv = int(old_kernel.shape[-1])
        if old_d_conv == new_d_conv:
            continue
        if old_d_conv > new_d_conv:
            raise ValueError(
                f"{mod_name}: old d_conv={old_d_conv} > new d_conv={new_d_conv}; "
                "shrinking conv kernels is not supported by this script."
            )
        pad_width = new_d_conv - old_d_conv
        new_kernel = np.concatenate(
            [
                np.zeros(old_kernel.shape[:-1] + (pad_width,), dtype=old_kernel.dtype),
                np.asarray(old_kernel),
            ],
            axis=-1,
        )
        params[mod_name]["kernel"] = new_kernel
        padded.append((mod_name, old_kernel.shape, new_kernel.shape))

    print(f"padded {len(padded)} conv1d.kernel(s):")
    for mod_name, old_s, new_s in padded:
        print(f"  {mod_name}: {tuple(old_s)} -> {tuple(new_s)}")
    print()

    new_total = sum(p.size for kk in params.values() for p in kk.values())
    delta = new_total - old_total

    out_ckpt = {
        "residual_params": params,
        "residual_state": {},  # drop -> training fresh-inits at new d_conv
    }
    with out_path.open("wb") as f:
        pickle.dump(out_ckpt, f)

    print(f"output: {out_path}")
    print(f"  total params: {new_total:,}  (delta: {delta:+,})")
    print(f"  residual_state: dropped (empty {{}})")
    print()
    print("Point v23_K1.slurm --resume-from at this file.")


if __name__ == "__main__":
    main()
