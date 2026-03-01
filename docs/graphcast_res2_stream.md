# GraphCast Res2 Streaming Training

## Purpose
Train GraphCast with streamed WeatherBench2 ERA5 data (no full local dataset download) using:
- resolution `2.0` degrees
- mesh size `4`
- width sweep `{128, 256}`
- processor message steps sweep `{1, 2}`
- one-step horizon (`+6h`)
- train split `1979-01-01` to `2020-12-31`
- validation split `2021-01-01` to `2021-12-31`

Main script: `scripts/training/train_graphcast_res2_stream.py`
SLURM sweep: `scripts/training/train_graphcast_res2_stream.slurm`

## How `max_steps=10000` Works
`max_steps` is the number of optimizer updates.

At each step:
1. Sample one minibatch of training windows (`batch_size=4`) from valid train indices.
2. Run forward/backward pass.
3. Apply one parameter update.

Sampling order is deterministic for a fixed `--seed`:
- Valid train indices are shuffled.
- Batches are consumed sequentially.
- When exhausted, indices are reshuffled and training continues.
- This repeats until `step == max_steps`.

So `max_steps=10000` means 10,000 updates (about 40,000 windows processed with `batch_size=4`), not a time interval.

## Streaming Requirements
- `gcsfs` installed in the environment (added to `requirements.txt`).
- Outbound access from cluster nodes to `gs://weatherbench2/...`.
- `xarray`, `zarr`, and `fsspec` available.

## Single-Run Example
```bash
source scripts/graphcast_env.sh
python -u scripts/training/train_graphcast_res2_stream.py \
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

## Cadence Profiles
- safe: `eval_every=500`, `checkpoint_every=1000`
- balanced (default): `eval_every=1000`, `checkpoint_every=2000`
- sparse: `eval_every=2000`, `checkpoint_every=4000`

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
`memory_gib.json` now tracks total GPU memory used (GiB) from:
```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
```
This replaces old allocator-only JAX memory stats.
