# Lianghong Residual Mamba v22

This folder is a small, runnable wrapper around the maintained training pipeline for
the v22 Lianghong residual Mamba experiment. It does not vendor or replace the
active `src/` or `third_party/` packages; those shared pipelines stay the source
of truth for execution.

The original branch context is preserved under `branch_snapshot/` for audit and
comparison. Those files are not placed on `PYTHONPATH`.

## What v22 Runs

v22 is a frozen GraphCast-small backbone plus a trainable residual head with full
S6-style Mamba blocks interleaved inside the residual mesh processor.

Default launcher shape:

- resolution `1.0`, mesh `5`, width `512`
- frozen GraphCast-small baseline checkpoint
- residual processor message steps `2`
- `temporal_location=mesh_processor_interleaved`
- `temporal_insert_count=2`, `temporal_layers=2`
- `temporal_d_inner=128`, `temporal_d_state=16`, `temporal_d_conv=4`
- `target_steps=22`, `bptt_steps=24`, `len_segment=96`
- bf16, batch size `1`, learning rate `1e-4`, weight decay `1e-4`, seed `22`
- residual AR feedback defaults to `baseline_plus_residual`

## Prerequisites

Run from the repository root. The scripts source the GraphCast environment:

```bash
source scripts/graphcast_env.sh
```

Expected local assets:

- Official GraphCast-small checkpoint:
  `data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz`
- Prepared res1 store:
  `data/graphcast/graphcast/dataset/prepared_stream/res1`
- Normalization stats:
  `data/graphcast/graphcast/stats`

The Lianghong v22 README points to warm-start and final checkpoints under
`/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22/`. Those files were not
visible from this checkout when this folder was created, so exact resume requires
you to provide `RESUME_CKPT`.

## Exact Resume

Use this when you have the K=1 warm-start checkpoint from the Lianghong run.

```bash
RESUME_CKPT=/path/to/v13_residual_step23000.pkl \
  sbatch Lianghong_res_mamba/scripts/run_v22_resume.slurm
```

Defaults:

- `RESUME_STEP=23000`
- `MAX_STEPS=26000`
- `RUN_NAME=K22_from_v22K1_23k`

## Fresh Start

Use this when you want the same v22 architecture and hyperparameters, but with a
fresh residual head initialized from the official GraphCast-small baseline.

```bash
sbatch Lianghong_res_mamba/scripts/run_v22_fresh.slurm
```

Defaults:

- `MAX_STEPS=26000`
- `RUN_NAME=v22_fresh_residual_mamba_k22`

## Useful Overrides

Both launchers accept environment overrides:

```bash
OUT_DIR=artifacts/checkpoints/lianghong_res_mamba \
RUN_NAME=my_run \
MAX_STEPS=100 \
EVAL_EVERY=50 \
CHECKPOINT_EVERY=50 \
DRY_RUN=1 \
  sbatch Lianghong_res_mamba/scripts/run_v22_fresh.slurm
```

Common path overrides are `DATA_PATH`, `PREPARED_DATA_ROOT`, `STATS_DIR`, and
`OFFICIAL_CKPT`.

For command inspection without Slurm:

```bash
DRY_RUN=1 bash Lianghong_res_mamba/scripts/run_v22_fresh.slurm
RESUME_CKPT=/tmp/placeholder.pkl DRY_RUN=1 bash Lianghong_res_mamba/scripts/run_v22_resume.slurm
```

## K=1 Then K=22 Pair

Submit the paired current-code and Lianghong-labeled curriculum with:

```bash
bash Lianghong_res_mamba/scripts/submit_k1_then_k22_pair.sh
```

Inspect the four `sbatch` commands without submitting:

```bash
DRY_RUN=1 bash Lianghong_res_mamba/scripts/submit_k1_then_k22_pair.sh
```

The wrapper trains `*_k1` runs with `TARGET_STEPS=1` to `ckpt_step23000.npz`,
then submits dependent `*_k22_from_k1` runs with `TARGET_STEPS=22` and
`RESUME_STEP=23000`.

## Provenance

`branch_snapshot/` was extracted from `origin/AR-Training-Lianghong` at commit
`111e9977c2262e9f8e76310ec4565e73c6580ea7`. See
`branch_snapshot/MANIFEST.md` for the copied file list.
