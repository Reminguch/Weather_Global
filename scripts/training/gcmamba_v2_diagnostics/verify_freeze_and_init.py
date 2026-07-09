"""Verify for both gc_mamba and res_mamba (post-K-launch ckpts):
  1. Init: GC params overlaid from GCv1 K=3 ckpt (copied >= expected count).
  2. Freeze: after N training steps, params in 'frozen' buckets are unchanged.

Loads a step-0 reference (either init or first saved ckpt) and a later step
ckpt, computes per-leaf max|Δp| / |p| relative change, and groups by module
class. Frozen modules MUST have ~0 change (bf16 noise floor ~1e-3).

Usage (after run produces step_500.npz or similar):
    python -m scripts.training.gcmamba_v2_diagnostics.verify_freeze_and_init \\
        --ckpt-early <step_N.npz>  --ckpt-late <step_M.npz>  --kind {gc_mamba,res_mamba}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _flat_npz_to_params(npz_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load segments_training-style flat npz → nested {module: {leaf: array}}."""
    data = np.load(npz_path, allow_pickle=True)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in data.keys():
        if not key.startswith("params:"):
            continue
        rest = key[len("params:"):]
        if ":" not in rest:
            continue
        module, leaf = rest.rsplit(":", 1)
        out.setdefault(module, {})[leaf] = np.asarray(data[key])
    return out


def classify_module(module_name: str, kind: str) -> str:
    lo = module_name.lower()
    if "temporal" in lo or "mamba" in lo: return "mamba"
    if "grid2mesh" in lo: return "g2m"
    if "mesh2grid" in lo: return "m2g"
    if "mesh_gnn" in lo or "mesh_processor" in lo: return "proc"
    return "other"


def expected_freeze_buckets(kind: str) -> set[str]:
    """Which buckets are expected to be frozen for each architecture."""
    if kind == "gc_mamba":
        # Only Mamba is trainable; g2m + proc + m2g + other are frozen.
        return {"g2m", "proc", "m2g", "other"}
    if kind == "res_mamba_proc_and_mamba":
        # proc + mamba trainable; g2m + m2g frozen.
        return {"g2m", "m2g"}
    if kind == "res_mamba_mamba_only":
        return {"g2m", "proc", "m2g", "other"}
    if kind == "res_mamba_all":
        # Nothing frozen in residual head (baseline GC1 is separate file).
        return set()
    raise ValueError(f"unknown kind={kind}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-early", required=True, help="Earlier-step ckpt (or step 0 init).")
    p.add_argument("--ckpt-late", required=True, help="Later-step ckpt.")
    p.add_argument("--kind", required=True,
                   choices=["gc_mamba", "res_mamba_proc_and_mamba",
                            "res_mamba_mamba_only", "res_mamba_all"])
    p.add_argument("--baseline-ckpt", default=None,
                   help="GCv1 K=3 ckpt to compare init overlay against (optional).")
    p.add_argument("--tol-frozen", type=float, default=1e-6,
                   help="Max |Δp|/|p| treated as 'frozen unchanged'.")
    return p.parse_args()


def main():
    cfg = parse_args()
    p_early = _flat_npz_to_params(Path(cfg.ckpt_early))
    p_late  = _flat_npz_to_params(Path(cfg.ckpt_late))
    print(f"Early ckpt: {cfg.ckpt_early}  ({len(p_early)} modules)")
    print(f"Late  ckpt: {cfg.ckpt_late}   ({len(p_late)} modules)")

    expected_frozen = expected_freeze_buckets(cfg.kind)
    print(f"Kind: {cfg.kind}  → expected frozen buckets: {sorted(expected_frozen)}")
    print()

    # === 1. Per-leaf max change, grouped by bucket ===
    buckets: dict[str, dict[str, float]] = {}  # bucket -> {leaf_path: max_rel_change}
    for mod in sorted(p_early.keys()):
        if mod not in p_late:
            print(f"  WARNING: {mod} in early but not late")
            continue
        bucket = classify_module(mod, cfg.kind)
        for leaf, v_early in p_early[mod].items():
            if leaf not in p_late[mod]: continue
            v_late = p_late[mod][leaf]
            if v_early.shape != v_late.shape: continue
            ae = np.asarray(v_early).astype(np.float32)
            al = np.asarray(v_late).astype(np.float32)
            denom = max(float(np.abs(ae).max()), 1e-12)
            max_abs = float(np.abs(al - ae).max())
            rel = max_abs / denom
            buckets.setdefault(bucket, {})[f"{mod}:{leaf}"] = rel

    print(f"=== Per-bucket relative change (max |Δp|/|p|, max across leaves) ===")
    print(f"{'bucket':<10}{'n_leaves':>10}{'max_rel_chg':>14}{'mean_rel_chg':>15}{'verdict':>16}")
    print('-' * 70)
    all_ok = True
    for bucket, leaves in sorted(buckets.items()):
        vals = list(leaves.values())
        max_rel = max(vals); mean_rel = np.mean(vals)
        is_expected_frozen = bucket in expected_frozen
        if is_expected_frozen:
            ok = max_rel < cfg.tol_frozen
            verdict = "FROZEN OK" if ok else "***LEAK!***"
            if not ok: all_ok = False
        else:
            verdict = "trainable (varies)"
        print(f'{bucket:<10}{len(vals):>10}{max_rel:>13.4e}{mean_rel:>14.4e}{verdict:>16}')
    print('-' * 70)
    print(f"OVERALL freeze check: {'PASS' if all_ok else 'FAIL'}")

    # === 2. Init overlay check (if baseline ckpt provided) ===
    if cfg.baseline_ckpt:
        print()
        print(f"=== Init overlay check vs baseline ({cfg.baseline_ckpt}) ===")
        baseline_params = _flat_npz_to_params(Path(cfg.baseline_ckpt))
        # For frozen modules in EARLY ckpt, expect they match baseline.
        match_n = 0; mismatch_n = 0; not_in_baseline = 0
        for mod in p_early:
            bucket = classify_module(mod, cfg.kind)
            if bucket not in expected_frozen: continue
            if mod not in baseline_params:
                not_in_baseline += 1; continue
            for leaf, ve in p_early[mod].items():
                vb = baseline_params[mod].get(leaf)
                if vb is None: continue
                if np.asarray(ve).shape != np.asarray(vb).shape: continue
                ae = np.asarray(ve).astype(np.float32)
                ab = np.asarray(vb).astype(np.float32)
                if np.allclose(ae, ab, atol=1e-5):
                    match_n += 1
                else:
                    mismatch_n += 1
        total_frozen_leaves = match_n + mismatch_n
        print(f"  frozen leaves matching baseline: {match_n}/{total_frozen_leaves}  "
              f"({100*match_n/max(total_frozen_leaves,1):.1f}%)")
        print(f"  frozen leaves diverged from baseline: {mismatch_n}")
        print(f"  frozen modules not in baseline (probably Mamba in g2m/m2g sense): {not_in_baseline}")
        init_ok = (mismatch_n == 0)
        print(f"  OVERALL init overlay: {'PASS' if init_ok else 'FAIL'}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
