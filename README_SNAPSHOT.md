# Frozen snapshot of the drifted tree (2026-07-09)

This branch (`AR-Training-Lianghong-H256frozen-DRIFTED`) captures the EXACT on-disk code state of
`/home/lm8598/Weather_Global_experiments/` on 2026-07-09 — the code that trained the
res=1 H=256 K-scan (`/scratch/.../results/v22closed/K*_H256_fresh_20k/`).
Use THIS tree for any eval of those ckpts. Original warning below.

# ⚠️ Original drifted-tree warning

Marker added **2026-07-09**. The directory was deliberately NOT renamed: the shared
conda env (`.conda/envs/graphcast311`) lives inside this tree and every slurm in both
trees hardcodes its absolute path, so a rename would break everything. This file is
the warning instead.

## Why this tree is dangerous

- `third_party/graphcast/graphcast/graphcast.py` and the Mamba modules here have
  **large staged + uncommitted local edits** (342 lines staged as of 2026-07-09).
  The on-disk state changes over time and is NOT reproducible from git history.
- Checkpoints trained with this tree's code must be eval'd with this tree's code,
  and vice versa. Loading the same ckpt in the clean tree silently succeeds
  (Haiku matches parameter names) but the forward pass can differ → garbage results.

## Documented incidents caused by mixing trees

1. **14.86K vs 2.80K RMSE @240h** — same v22 ckpt eval'd in the two trees (June 2026).
2. **2026-07-03 city traces**: `v22cl_r1_SWA_city_trace_coldfull*.slurm` had
   `cd v22clean` but invoked the eval script by **absolute path into this tree**
   → imports came from here → Beijing traces didn't match. The fixed slurm
   (`v22cl_SWA_city_v22cleantree.slurm`) uses a relative script path.
3. **2026-07-09 res=1 H=256 SWA eval (job 10889366)**: ckpts trained here were
   eval'd in the clean tree → -400% "improvements" (garbage). Controlled re-run
   (job 10894039, same ckpt/config, this tree) gave sane numbers — proving the
   trees are NOT functionally equivalent even though a static line-by-line diff
   suggested the differing paths were unused. Do not trust diff-based equivalence
   arguments; only trust train-tree == eval-tree.

## Root cause of the trap

All train/eval scripts anchor imports at the *script file's own location*:
```python
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
```
So `cd` does NOT decide which tree's code runs — the script path does.
A slurm with `cd <clean>` + `python /home/.../Weather_Global_experiments/...py`
runs THIS tree's code.

## What trained here (and therefore must be eval'd here)

- res=1 H=256 K-scan (`/scratch/.../results/v22closed/K*_H256_fresh_20k/`),
  trained via `scripts/training/full_mamba_v22closed/train_mz_v22closed.py`.

Everything else (res=1 H=128 / open-loop v22 / all res=2) lives in
`/home/lm8598/Weather_Global_experiments_v22clean/` — use that tree for those.

## Note on the conda env

`.conda/envs/graphcast311` inside this tree is shared by ALL experiments in both
trees. It is unaffected by code drift (it's just the Python+JAX runtime), but its
absolute path is baked into every slurm — hence the symlink at the old location.
