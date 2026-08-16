# Lianghong: please verify the recent `v23_Ilya` work

This note describes the current tree as of 2026-08-16. Lianghong, please compare it directly with your tree (in particular `/home/lm8598/Weather_Global_experiments`) and tell me if anything below is broken, wrong, or incompatible with your implementation/checkpoints. Please do not assume the directory unification preserved behavior: check the model construction, parameter names, sparse-loss definition, state carry/reset behavior, data splits, and resume behavior against yours.

## Start from the Git `minimalistic_code` branch

The intended Git branch for this work is `minimalistic_code` in `git@github.com:Reminguch/Weather_Global.git`.

For a new checkout:

```bash
git clone --branch minimalistic_code --single-branch \
  git@github.com:Reminguch/Weather_Global.git Weather_global
cd Weather_global
```

For an existing checkout, first make sure you do not have local changes that would be overwritten, then update and switch branches:

```bash
git status --short
git fetch origin
git switch minimalistic_code
git pull --ff-only origin minimalistic_code
```

Verify that the checkout is on the intended branch before using any command in this note:

```bash
git branch --show-current
git rev-parse --short HEAD
```

`git branch --show-current` must print `minimalistic_code`. The res0.25 sparse-loss implementation, configs, launchers, tests, analysis helper, and removal of the old `Ilya_code` gitlink are commit `752c11b` (`Add v23 res0.25 sparse-loss training`). Commit `4d4a5fe` and older do **not** contain everything documented below. Do not start the sparse run unless commit `752c11b` is an ancestor of your checkout:

```bash
git merge-base --is-ancestor 752c11b HEAD
```

That command must exit successfully. The branch also needs the later documentation commit containing this `LH.md`. If `origin/minimalistic_code` has not received those commits yet, ask me to push them before cloning or pulling. Verify that the checkout contains at least these files:

```text
LH.md
configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_smoke.json
configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k.json
scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr_sweep.slurm
scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr3em6_dp4_8k.slurm
```

Quick check:

```bash
test -f LH.md \
  && test -f configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k.json \
  && test -f scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr3em6_dp4_8k.slurm
```

Lianghong, please record the exact commit returned by `git rev-parse HEAD` when you verify this against your version so we do not compare two different working trees.

## What the res0.25 training does

The current res0.25 experiment is standalone `v23_Ilya` residual-Mamba training:

- GraphCast resolution: `0.25`, 37 pressure levels, mesh 6.
- Frozen GraphCast baseline plus a trainable residual GraphCast/Mamba branch.
- Stateful Mamba, `d_inner=16`, `bc_groups=1`, two temporal layers.
- A 120-step segment divided into 24-step BPTT chunks.
- Four teacher-forced transitions followed by 20 closed-loop stop-gradient transitions (`ar_tail_k=19`).
- Sparse loss at horizons `1, 4, 8, 12, 16, 20`, with weights `1, 1, 2, 2, 4, 8`. The weighted loss is normalized by the sum of the weights.
- BF16 model/weather tape, FP32 parameters, optimizer state, and recurrent-state boundaries.

The selected learning rate for the longer run is `3e-6`.

## Data paths

Repository root:

```text
/scratch/gpfs/DABANIN/iv9432/Weather_global
```

Prepared ERA5/GraphCast37 res0.25 store (2019-01-01 through 2022-12-31 at 6-hour spacing, about 5.1 TiB):

```text
/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/prepared_stream_graphcast37/res0p25
```

Anchor-only manifest (2019-2021 training, 2022 validation):

```text
/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/dataset/anchor_manifests/v23_Ilya_res0p25_2019_2022
```

Frozen GraphCast checkpoint:

```text
/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz
```

Normalization statistics:

```text
/scratch/gpfs/DABANIN/iv9432/Weather_global/data/graphcast/graphcast/stats
```

The prepared store was built from:

```text
gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr
```

## One-GPU run

The model and sparse objective fit on one A100-80GB. Four completed 200-update smokes each peaked at approximately `60,411-60,415 MiB` GPU memory. CPU RSS reached the 128 GiB allocation, so keep the existing `--mem=128G` request (or increase it); GPU memory is not the limiting issue.

From the repository root, run only the selected `3e-6` one-GPU smoke (array task 2):

```bash
sbatch --array=2-2 scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr_sweep.slurm
```

This uses:

```text
configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_smoke.json
```

It requests one A100-80GB and runs 200 updates. Running the Slurm script without `--array=2-2` launches all four independent one-GPU learning-rate smokes; that is a sweep, not four-GPU training.

## Four-GPU run

The longer configuration uses synchronous data parallelism across four A100-80GB GPUs, with one full model replica and batch size 1 on each GPU (global batch size 4). It is not spatial/model sharding. Consequently, it increases throughput/global batch size; it is not the same optimizer trajectory as a one-GPU run.

The 8,000-update job is split into two 4,000-update allocations because the estimated total duration exceeds one Slurm time window.

Stage 1:

```bash
sbatch scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr3em6_dp4_8k.slurm
```

After stage 1 finishes successfully, resume exactly from its step-4000 checkpoint:

```bash
RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k_20260816/di16_bcg1_sparse_h20_closed_sg_stateful_dp4_lr3em6_8k"
sbatch --export=ALL,MAX_STEPS=8000,RESUME="${RUN_DIR}/checkpoints/checkpoint_step00004000.pkl" \
  scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr3em6_dp4_8k.slurm
```

Before submitting either mode, paths and the resolved contract can be checked without starting a run:

```bash
source scripts/graphcast_env.sh
python scripts/training/train_v23_Ilya.py \
  --config configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k.json \
  --dry-run
```

## Directory unification

`v23_Ilya` is now organized inside the main `Weather_global` tree instead of being maintained as a second nested code tree:

- `src/models/mamba/v23_Ilya/` owns model construction, rollout/evaluation, checkpoint compatibility, data loading, sparse/endpoint reverse BPTT, validation, and single/data-parallel runners.
- `scripts/training/train_v23_Ilya.py`, `scripts/analyze_models/eval_v23_Ilya.py`, and `scripts/training/build_v23_Ilya_swa.py` are thin entry points into that package.
- Experiment JSON files live in `configs/experiments/v23_Ilya/`.
- Slurm launchers live in `scripts/experiments/`.
- Tests live in `tests/v23_Ilya/` and include a dependency-boundary test that rejects imports from `v22_final` and `full_mamba_v*` training packages.
- The canonical GraphCast code is the main tree's `third_party/graphcast/`; `v23_Ilya` uses it through the normal `graphcast` imports.

The intent is one canonical repository root, one GraphCast dependency, and a self-contained versioned `v23_Ilya` package, so training and evaluation construct the same model and do not silently pull code from an older training version.

Commit `752c11b` removes the old `Ilya_code` gitlink. However, `Il_LH_diff.md` still refers to `Ilya_code/...` and is therefore stale after unification. Lianghong, please explicitly verify the unified main-tree implementation against yours, especially:

1. Mamba placement/routing and full-Mamba selection.
2. The zero-initialized residual head and its parameter names.
3. Sparse-horizon indexing, weights, and normalization.
4. Stateful carry at segment and BPTT boundaries.
5. Frozen GraphCast behavior during explicit reverse VJP.
6. Single-GPU versus four-replica RNG/data cursor semantics.
7. Exact-resume checkpoint compatibility and whether changing `MAX_STEPS` from 4000 to 8000 is accepted as intended.
8. The 2019-2021 train / 2022 validation split and prepared-data variable conventions.

Please reply with any mismatch, even if it only changes names or checkpoint compatibility; those differences can otherwise look like successful training while producing a non-comparable model.
