# GraphCast Res2 Local Training

## Purpose
Train GraphCast from a local staged ERA5 dataset using:
- resolution `2.0` degrees
- mesh size `4`
- width sweep `{128, 256}`
- processor message steps sweep `{1, 2}`
- one-step horizon (`+6h`)

Main script: `scripts/training/train_graphcast_res2_stream.py`  
SLURM sweep: `scripts/training/train_graphcast_res2_stream.slurm`

## Local-Only Data Policy
This trainer is local-only.

Use:
- `--data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr`

Do not use `gs://...` in this script.

## Train/Validation Split
Default behavior:
- validation year: `2021` (`--val-year 2021`)
- train years: all years present in local dataset except `2021`

Optional train-year bounds:
- `--train-start-year`
- `--train-end-year`

If bounds are provided, train years are filtered to that range, and `val-year` is still excluded.

## Resolution Handling
- Script infers base grid resolution from local `lat/lon` coordinates.
- Target stride is computed as `resolution / base_resolution`.
- For local 1.0° data and `--resolution 2.0`, stride is `2`.

## How `max_steps=10000` Works
`max_steps` is optimizer-update count.

At each step:
1. Take one minibatch from shuffled valid train indices.
2. Run forward/backward.
3. Apply one optimizer update.

When one pass is exhausted, indices are reshuffled and training continues until `step == max_steps`.

## Single-Run Example
```bash
source scripts/graphcast_env.sh
python -u scripts/training/train_graphcast_res2_stream.py \
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --width 128 \
  --processor-msg-steps 1 \
  --run-name res2_m4_w128_mp1_h6_bs4
```

## SLURM Sweep Example
```bash
sbatch scripts/training/train_graphcast_res2_stream.slurm
```

Array mapping:
- `0 -> (width=128, msg=1)`
- `1 -> (width=128, msg=2)`
- `2 -> (width=256, msg=1)`
- `3 -> (width=256, msg=2)`

## Artifacts
Per run directory, outputs include:
- `run_config.json`
- `train_loss.json`
- `eval_loss.json`
- `eval_details.json`
- `step_times.json`
- `memory_gib.json`
- `actual_usage.json`
- `actual_usage_summary.json`
- `epoch_summary.json`
- `ckpt_step*.npz`
- `val_loss.png`

### Memory Semantics
`memory_gib.json` tracks total GPU memory used (GiB) from:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
```
