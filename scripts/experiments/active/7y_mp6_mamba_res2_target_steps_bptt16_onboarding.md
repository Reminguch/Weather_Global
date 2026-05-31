# Res2 MP6 Mamba Target-Step BPTT16 Runs

This note is an onboarding summary for the 12-run res2 experiment submitted on
2026-05-31 as Slurm array job `9025363`.

The experiment compares the two Mamba model families currently implemented in
this repo:

- `gc_mamba`: GraphCast initialized from a frozen vanilla GraphCast checkpoint,
  with trainable Mamba insertions.
- `residual_mamba`: a residual correction model trained against a frozen
  vanilla GraphCast baseline. This is the repo implementation behind the
  shorthand `res_mamba`.

The original request used `dg_mamba`; in this codebase that corresponds to
`gc_mamba`.

## Submitted Job

Submission script:

```text
scripts/experiments/active/7y_mp6_mamba_bptt16_res2_target_steps_20k.slurm
```

Submission command:

```bash
sbatch scripts/experiments/active/7y_mp6_mamba_bptt16_res2_target_steps_20k.slurm
```


## Experiment Matrix

All 12 tasks use this frozen baseline:

```text
artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k
```

The checkpoint file is:

```text
artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k/ckpt_best.npz
```

The run output root is:

```text
artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k_res2_target_steps_bptt16
```

Array mapping:

| Task | Model | Insert count | `d_inner` | `d_state` | Target steps | Run suffix |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `gc_mamba` | 2 | 64 | 32 | 4 | `gc_mamba_tc2_di64_ds32_20k_target_step4_bptt16` |
| 1 | `residual_mamba` | 1 | 64 | 32 | 4 | `residual_mamba_tc1_di64_ds32_20k_target_step4_bptt16` |
| 2 | `gc_mamba` | 2 | 128 | 64 | 4 | `gc_mamba_tc2_di128_ds64_20k_target_step4_bptt16` |
| 3 | `residual_mamba` | 1 | 128 | 64 | 4 | `residual_mamba_tc1_di128_ds64_20k_target_step4_bptt16` |
| 4 | `gc_mamba` | 2 | 64 | 32 | 8 | `gc_mamba_tc2_di64_ds32_20k_target_step8_bptt16` |
| 5 | `residual_mamba` | 1 | 64 | 32 | 8 | `residual_mamba_tc1_di64_ds32_20k_target_step8_bptt16` |
| 6 | `gc_mamba` | 2 | 128 | 64 | 8 | `gc_mamba_tc2_di128_ds64_20k_target_step8_bptt16` |
| 7 | `residual_mamba` | 1 | 128 | 64 | 8 | `residual_mamba_tc1_di128_ds64_20k_target_step8_bptt16` |
| 8 | `gc_mamba` | 2 | 64 | 32 | 12 | `gc_mamba_tc2_di64_ds32_20k_target_step12_bptt16` |
| 9 | `residual_mamba` | 1 | 64 | 32 | 12 | `residual_mamba_tc1_di64_ds32_20k_target_step12_bptt16` |
| 10 | `gc_mamba` | 2 | 128 | 64 | 12 | `gc_mamba_tc2_di128_ds64_20k_target_step12_bptt16` |
| 11 | `residual_mamba` | 1 | 128 | 64 | 12 | `residual_mamba_tc1_di128_ds64_20k_target_step12_bptt16` |

Full run names prepend the baseline directory name:

```text
vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k_<run suffix>
```

## Training Configuration

The script reads most fields from the baseline checkpoint's `run_config.json`.
Important inherited settings:

| Setting | Value |
| --- | --- |
| Data path | `data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr` |
| Data source | `prepared_array` |
| Prepared root | `data/graphcast/graphcast/dataset/prepared_stream` |
| Prepared store | `data/graphcast/graphcast/dataset/prepared_stream/res2` |
| Train years | `2015..2021` |
| Validation year | `2022` |
| Resolution | `2.0` degrees |
| Mesh size | `4` |
| Latent width | `512` |
| GraphCast message-passing steps | `6` |
| Input duration | `12h`, meaning two 6-hour input frames |
| Batch size | `8` |
| Eval batch size | `4` |
| Eval/checkpoint cadence | every `2000` optimizer updates |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Precision | `bf16` |

Experiment overrides:

| Setting | Value |
| --- | --- |
| Max steps | `20000` optimizer updates |
| Segment length | `32` time anchors |
| BPTT steps | `16` |
| Target steps | `4`, `8`, or `12` |
| Regular eval segments | `16` |
| Final eval segments | `all` |
| Chunk load workers | `6` |
| Temporal backbone | `mamba` |
| Temporal location | `mesh_processor_interleaved` |
| Temporal state | stateful |
| Mamba `d_inner` / `d_state` | `(64, 32)` and `(128, 64)` |
| Mamba `d_conv` | `4` |
| Mamba layers | `1` |
| Mamba dropout | `0.0` |

For target-step training, the training objective is the corrected
autoregressive tail inside each BPTT chunk. With `BPTT_STEPS=16`, the truth-fed
prefix lengths are:

| Target steps | Truth-fed prefix | Autoregressive scored tail |
| ---: | ---: | ---: |
| 4 | 12 | 4 |
| 8 | 8 | 8 |
| 12 | 4 | 12 |

`target_steps` must be smaller than `bptt_steps`, which is why the largest
target here is `12` under `BPTT_STEPS=16`.

## Model Architecture

### Vanilla GraphCast Baseline

The frozen baseline is a 7-year res2 GraphCast model:

```text
inputs at t-1,t
  -> grid2mesh encoder
  -> mesh processor, 6 message-passing steps
  -> mesh2grid decoder
  -> prediction at t+1
```

It is trained on `2015..2021`, validates on `2022`, uses a 12-hour input
window, and predicts one 6-hour target step per standard GraphCast forward
pass.

### `gc_mamba`

`gc_mamba` starts from the vanilla GraphCast checkpoint and inserts Mamba blocks
in the mesh processor path:

```text
GraphCast mesh latent
  -> Mamba temporal block
  -> GraphCast mesh processing / decoding
```

For these runs:

- the GraphCast parameters are initialized from `ckpt_best.npz`;
- `--trainable-part mamba` freezes the GraphCast baseline weights;
- `--zero-init-temporal-out` makes each inserted Mamba block initially behave
  like a no-op, so the run starts close to the frozen baseline;
- `temporal_insert_count=2`, so two Mamba insertions are used;
- Mamba state is carried across BPTT steps inside a segment.

Dimension vocabulary:

```text
width: GraphCast latent size, here 512
d_inner: Mamba internal channel count
d_state: SSM memory depth per internal channel
state shape: [batch, mesh_nodes, d_inner, d_state]
```

### `residual_mamba`

`residual_mamba` trains a fresh correction branch beside the frozen GraphCast
baseline:

```text
baseline inputs -> frozen GraphCast -> baseline_prediction

residual inputs
  -> grid2mesh
  -> mesh processor + Mamba
  -> mesh2grid
  -> zero-init residual output head
  -> residual_prediction

forecast = baseline_prediction + residual_prediction
training target = truth - baseline_prediction
```

For these runs:

- the baseline checkpoint is passed as `--baseline-ckpt`;
- baseline predictions are computed online during training and eval;
- the residual branch is fresh unless a resume checkpoint is supplied;
- `temporal_insert_count=1`;
- the final residual output head is zero-initialized, so a fresh run starts as
  `baseline_prediction + 0`.

Both Mamba families use the stateful Mamba module, which carries the SSM state
and the `d_conv - 1` causal convolution cache across BPTT steps.

## Data Pipeline

These runs use the repo's prepared-array pipeline. The goal is to avoid costly
raw xarray/Zarr indexing during training by converting the GraphCast task
variables into memmap-backed `.npy` arrays once, then streaming contiguous
blocks during training and eval.

Current res2 prepared store:

```text
data/graphcast/graphcast/dataset/prepared_stream/res2
```

The existing `res2/metadata.json` says:

| Field | Value |
| --- | --- |
| Source data | `data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr` |
| Time range | `2015-01-01 00:00:00` to `2022-12-31 18:00:00` |
| Time steps | `11688` |
| Cadence | 6-hourly |
| Grid | `91 x 180`, res2 |
| Pressure levels | 13 levels, `50..1000 hPa` |
| Format version | `prepared_array_format_version=1` |

The prepared store layout is:

```text
prepared_stream/res2/
  metadata.json
  validity.json
  coords/
    time.npy
    lat.npy
    lon.npy
    level.npy
  vars/
    2m_temperature.npy
    temperature.npy
    geopotential.npy
    ...
```

Training opens this store through `PreparedArrayStore`, validates that the
resolution, pressure levels, variables, and 6-hour time grid match the
checkpoint task config, then splits by year:

```text
train: years 2015..2021
eval:  year 2022
```

## How To Prepare Data

Always run Python through the GraphCast environment:

```bash
bash -lc 'source scripts/graphcast_env.sh && python --version'
```

### 1. Stage a local WeatherBench2/ERA5 Zarr, if needed

Use this when the local source Zarr is missing or needs to be rebuilt. This
step reads the public WB2 ERA5 Zarr, keeps GraphCast variables, downsamples to
1.0 degree, selects 13 pressure levels, and writes a local Zarr.

```bash
bash -lc 'source scripts/graphcast_env.sh && python scripts/stage_wb2_era5_yearly_append.py \
  --output data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr \
  --start-year 2015 \
  --end-year 2022'
```

This command needs network access. Existing docs for this stage live in:

```text
docs/wb2_local_staging.md
```

### 2. Build prepared-array stores

The preparation command reads the source Zarr and a checkpoint's task config,
then writes one `prepared_stream/resN` store per requested resolution.

For this experiment, res2 is sufficient:

```bash
bash -lc 'source scripts/graphcast_env.sh && python -m src.data_operations.preprocessing.prepare_graphcast_streaming_store \
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr \
  --ckpt-in artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k/ckpt_best.npz \
  --out-root data/graphcast/graphcast/dataset/prepared_stream \
  --resolutions 2'
```

If the store already exists and should be replaced, add:

```bash
--overwrite
```

There is also a Slurm wrapper:

```bash
RESOLUTIONS="2" \
DATA_PATH="data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr" \
CKPT_IN="artifacts/checkpoints/7_years/vanilla_gc/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream50k/ckpt_best.npz" \
OUT_ROOT="data/graphcast/graphcast/dataset/prepared_stream" \
sbatch scripts/analyze_models/run_prepare_graphcast_streaming_store.slurm
```

The wrapper requests `4` CPUs, `120G` CPU RAM, and `02:00:00`.

### 3. Train with the prepared store

Training scripts should use:

```text
--data-source prepared_array
--prepared-data-root data/graphcast/graphcast/dataset/prepared_stream
--batch-builder prepared_array
```

The run script in this experiment inherits those settings from the frozen
baseline `run_config.json`.

## Related Scripts

Primary script for this experiment:

```text
scripts/experiments/active/7y_mp6_mamba_bptt16_res2_target_steps_20k.slurm
```

Lineage and nearby scripts:

```text
scripts/experiments/active/7y_mp6_mamba_frozen_bptt16_res236_20k.slurm
scripts/experiments/active/7y_mp6_mamba_target_step4_res2_di128_ds64_20k.slurm
scripts/experiments/active/submit_7y_mp6_mamba_frozen_sweep_20k.sh
scripts/training/segments_training.slurm
scripts/analyze_models/run_prepare_graphcast_streaming_store.slurm
```

Training entrypoint:

```text
python -u -m src.models.mamba.training.segments_training
```

Important architecture docs:

```text
src/models/mamba/gc_mamba/gc_mamba.md
src/models/mamba/residual_mamba/residual_mamba.md
scripts/training/training_quickstart.md
```

## Outputs To Inspect After Completion

Each run directory should contain:

```text
run_config.json
train_loss.json
eval_loss.json
eval_details.json
ckpt_best.npz
ckpt_step20000.npz
best_checkpoint.json
```

Useful post-run checks:

```bash
sacct -j 9025363 --format=JobID,JobName%30,State,Elapsed,Timelimit,ReqMem,MaxRSS,AveRSS,AllocCPUS -P
tail -n 80 logs/7y_mp6_mamba_res2_target_bptt16_9025363_0.out
tail -n 80 logs/7y_mp6_mamba_res2_target_bptt16_9025363_0.err
```

For standalone resolution evaluation after the checkpoints exist, use:

```text
scripts/analyze_models/unified_resolution_eval.py
scripts/analyze_models/submit_resolution_eval_array.sh
scripts/analyze_models/run_resolution_eval_array.slurm
```

For long-rollout res2 checks, pass explicit lead steps such as:

```text
--lead-steps 1 4 8 16 24 32
```
