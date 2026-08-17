# `v23_Ilya` res0.25 training: technical handoff for Lianghong

This note describes the recent folder and workflow restructuring in `minimalistic_code` and compares it with [`AR-Training-Lianghong-v22clean-frozen`](https://github.com/Reminguch/Weather_Global/tree/AR-Training-Lianghong-v22clean-frozen) at commit `20d9c90`.

Lianghong: please compare this directly with your frozen tree and tell me if any model behavior, data convention, loss definition, recurrent-state boundary, checkpoint conversion, or evaluation mode is wrong or incompatible. The goal is behavioral agreement, not merely matching file names.

## Main differences from the frozen v22-clean workflow

| Area | `AR-Training-Lianghong-v22clean-frozen` | Current `v23_Ilya` |
| --- | --- | --- |
| Training code | One large script, `scripts/training/full_mamba_v20/train_mz_v20.py` | Maintained package under `src/models/mamba/v23_Ilya/`; `scripts/training/train_v23_Ilya.py` is a thin entry point |
| Configuration | Long CLI argument lists embedded in resolution-specific Slurm files | Strict JSON configs under `configs/experiments/v23_Ilya/` |
| Launchers | Mostly `slurm_pilots/`, with hard-coded user/tree paths | `scripts/experiments/`, rooted at the submitted checkout |
| Production resolutions | Matched res1 and res2 open/closed/SWA trios; res0.25 had partitioning probes, not this training path | Real res0.25 GraphCast-large training at 37 pressure levels, mesh 6 |
| Training data | Prepared inputs plus a `precomputed_residuals/...` root | GraphCast37 prepared truth store plus anchor-only manifest; residual targets are formed live |
| Loss | Mean of the loss at every one of the 24 BPTT steps | Sparse weighted loss at horizons 1, 4, 8, 12, 16, and 20 |
| Rematerialization | Ordinary `jax.value_and_grad` around the full 24-step loop, with `jax.checkpoint` on the complete one-step function | Explicit reverse VJP; detached weather/state boundaries are saved and one residual/Mamba step is replayed at a time |
| Frozen baseline in backward | Baseline prediction is inside the checkpointed one-step function | Frozen GraphCast is forward-only and is not replayed by the reverse rule |
| Recurrent state | Implicit carry within a sequential segment | Explicit `carry` or `reset_every_anchor` policy; checkpoint boundaries are FP32 |
| Parallelism | Single GPU | Single GPU is supported; synchronous four-GPU data parallelism is also implemented |
| Validation | No validation in the training loop; evaluation was post-hoc through `eval_v22_clean.py` | Fixed 2022 validation can run at checkpoint cadence and writes best-checkpoint metadata and a train/validation plot |
| Outputs | Mostly `results/...` | `artifacts/checkpoints/v23_Ilya/...` and `artifacts/evaluations/v23_Ilya/...` |

The comparison is not intended to claim numerical identity. The current res0.25 run also changes the baseline checkpoint, pressure-level count, grid, Mamba inner size, objective, learning rate, batch semantics, and data years.

## Directory and workflow restructuring

The current layout separates reusable implementation from experiment configuration and cluster launch logic:

```text
src/models/mamba/v23_Ilya/
  model.py                 shared training/evaluation model construction
  checkpoint.py            v23 checkpoints, legacy warm starts, exact resume
  rollout.py               cold/warm autoregressive rollout behavior
  evaluation.py            rollout metrics and JSON output
  training/
    config.py              strict training schema and resume validation
    data.py                segments, anchors, and selective reads
    frame_data.py          unique-frame and sparse-target materialization
    endpoint_step.py       explicit reverse VJP/rematerialization
    data_parallel.py       synchronous replicated four-GPU updates
    validation.py          fixed 2022 validation and best-checkpoint tracking
    runner.py              initialization, resume, train, validate, checkpoint

configs/experiments/v23_Ilya/   experiment definitions
scripts/training/               thin train/SWA entry points
scripts/analyze_models/         thin evaluator and analysis tools
scripts/experiments/            Slurm launchers
tests/v23_Ilya/                 config, data, remat, DP, checkpoint, rollout tests
artifacts/checkpoints/v23_Ilya/ run configs, metrics, checkpoints, validation
```

The old nested `Ilya_code` gitlink was removed. The canonical GraphCast dependency is now the main tree's `third_party/graphcast/`. `v23_Ilya` also has a dependency-boundary test that rejects imports from `v22_final` and `full_mamba_v*`, so the maintained package cannot silently borrow an older trainer.

## Current res0.25 experiment

The longer run is defined by:

```text
configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k.json
```

Its main settings are:

- GraphCast-large res0.25 baseline, 37 pressure levels, mesh 6, width 512, 16 baseline message-passing steps.
- Residual GraphCast with two message-passing steps and interleaved stateful Mamba.
- Mamba `d_inner=16`, `bc_groups=1`, `d_state=16`, `d_conv=4`, two temporal layers.
- A 120-step sequential segment, split into 24-step BPTT chunks.
- Four teacher-forced transitions followed by a 20-step stopped-gradient closed-loop trajectory.
- Recurrent Mamba state carried across chunks within the segment.
- BF16 model/weather tape; FP32 parameters, optimizer state, and recurrent-state boundaries.
- AdamW, peak learning rate `3e-6`, warmup 20, weight decay `1e-4`, gradient clipping at 1.
- Four replicated devices with per-device batch size 1, so global batch size 4.

### Sparse loss

The old v22-clean trainer appends a loss at every BPTT step and returns their uniform mean. The current sparse objective reads and supervises only these forecast horizons:

```text
horizon:  1  4  8  12  16  20
weight:   1  1  2   2   4   8
```

Weights are normalized by their sum, 18. The objective is therefore:

```text
(L1 + L4 + 2 L8 + 2 L12 + 4 L16 + 8 L20) / 18
```

`train_metrics.jsonl` records both the aggregate loss and `loss_by_horizon`. This makes long-horizon failure visible instead of hiding it inside a 24-step uniform mean.

### Different rematerialization rule

The v22-clean trainer uses a standard reverse pass through the jitted 24-step Python loop. Each complete AR step is wrapped in `jax.checkpoint`; that step contains the frozen baseline prediction, residual prediction, target construction, residual loss, and recurrent-state update.

`v23_Ilya` instead owns the reverse rule explicitly:

1. The forward pass saves the minimum detached boundary information: K+1 unique weather frames and K residual-state boundaries. It does not save overlapping two-frame windows or GraphCast activations.
2. The backward pass walks from step K to step 1.
3. It reconstructs each two-frame input window from adjacent saved frames.
4. It rematerializes only one residual/Mamba step and applies its VJP.
5. It accumulates parameter and recurrent-state cotangents across the reverse walk.
6. The frozen GraphCast baseline is never replayed in backward.

This is a memory strategy and a gradient-definition boundary. Lianghong, please verify especially that the stopped-gradient closed-loop feedback and state cotangent behavior match what you intend.

## Data locations

All paths below are under:

```text
/scratch/gpfs/DABANIN/iv9432/Weather_global
```

Training/validation prepared store, 2019-01-01 through 2022-12-31 at six-hour spacing (about 5.1 TiB):

```text
data/graphcast/graphcast/dataset/prepared_stream_graphcast37/res0p25
```

Anchor-only manifest, with 2019-2021 training anchors and 2022 validation anchors:

```text
data/graphcast/graphcast/dataset/anchor_manifests/v23_Ilya_res0p25_2019_2022
```

Frozen res0.25 GraphCast checkpoint:

```text
data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz
```

Normalization statistics:

```text
data/graphcast/graphcast/stats
```

The manifest stores only indices, times, deterministic splits, and source metadata. It does not contain residual tensors. Residual targets are computed during the forward pass as truth minus the stopped-gradient frozen GraphCast prediction.

## Run training

Run all commands from the repository root. First validate paths and the resolved contract without allocating model state or creating a run directory:

```bash
source scripts/graphcast_env.sh
python scripts/training/train_v23_Ilya.py \
  --config configs/experiments/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k.json \
  --dry-run
```

### One-GPU smoke

The selected `3e-6` 200-update smoke is array task 2:

```bash
sbatch --array=2-2 scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr_sweep.slurm
```

This exact model/update shape completed on one A100-80GB. The four learning-rate smokes used about 60.4 GiB sampled GPU memory and reached the 128 GiB CPU allocation; task 2 completed in 5:35:39. Thus the update fits on one A100-80GB, but the 200-step job has essentially no CPU-memory or six-hour walltime headroom. A long single-GPU trajectory must be split into exact-resume stages.

Running the launcher without `--array=2-2` starts four independent one-GPU learning-rate experiments. It is not four-GPU training.

### Four-GPU 8,000-update run

The maintained longer launcher uses synchronous data parallelism, not model or spatial sharding. Every GPU holds a full model replica and consumes a different segment lane.

Stage 1 runs through step 4000 by default:

```bash
sbatch scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr3em6_dp4_8k.slurm
```

After stage 1 completes successfully, resume the same run through step 8000:

```bash
RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k_20260816/di16_bcg1_sparse_h20_closed_sg_stateful_dp4_lr3em6_8k"

sbatch \
  --export=ALL,MAX_STEPS=8000,RESUME="${RUN_DIR}/checkpoints/checkpoint_step00004000.pkl" \
  scripts/experiments/train_v23_Ilya_res0p25_sparse_h20_lr3em6_dp4_8k.slurm
```

Exact resume restores parameters, optimizer, RNG, per-replica recurrent states, active segment IDs, and the replica cursor. Only `max_steps`, checkpoint cadence, and validation settings may change. Four-GPU and one-GPU checkpoints are not interchangeable for exact resume because their cursor and replica-state contracts differ.

### How to count four-GPU steps

The four GPUs are four synchronous data-parallel replicas, not four workers that
divide the numbered optimizer steps among themselves. At every optimizer step,
each GPU processes one different rollout segment, the four gradients are
averaged, and one shared parameter update is applied on all replicas:

```text
per-device batch size = 1
number of devices     = 4
global batch size     = 4 rollout segments per optimizer update
```

It is therefore useful to track two different quantities:

```text
optimizer updates = step
rollout exposure  = step * global_batch_size = step * 4
```

For example, step 500 in the DP4 run has made 500 optimizer updates while
processing 2,000 rollout segments. It matches the *data exposure* of step 2,000
in a one-GPU batch-one run, but it is not optimization-equivalent: the old run
made 2,000 parameter updates, whereas DP4 averages groups of four gradients and
makes only 500 updates.

This distinction changes how the experiment should be interpreted:

| DP4 optimizer step | Rollout segments processed | Batch-one exposure equivalent |
| ---: | ---: | ---: |
| 500 | 2,000 | step 2,000 |
| 1,000 | 4,000 | step 4,000 |
| 2,000 | 8,000 | step 8,000 |
| 4,000 | 16,000 | step 16,000 |
| 8,000 | 32,000 | step 32,000 |

Consequently, if the scientific target is the same rollout exposure as the old
8,000-step batch-one experiment, the corresponding DP4 endpoint is step 2,000,
not step 8,000. The currently configured 8,000-update DP4 trajectory processes
four times that exposure. Conversely, if the target is 8,000 optimizer updates,
then the current endpoint is intentional, but it should be reported as global
batch 4 and 32,000 processed rollout segments.

Checkpoint and evaluation cadence should state which clock it follows. To match
an old batch-one checkpoint every 2,000 processed segments, checkpoint/evaluate
every 500 DP4 updates. The current cadence of every 1,000 DP4 updates corresponds
to every 4,000 processed rollout segments. For plots, retain the optimizer step
in checkpoint names but add `rollout_segments_seen = step * 4` on the comparison
axis or in the accompanying table.

Walltime follows optimizer updates, not rollout exposure. Four-way data
parallelism processes four segments concurrently but does not turn 4,000
optimizer updates into 1,000 update rounds. With the measured one-GPU
steady-state time of about 73 seconds per update, 4,000 synchronized update
rounds are approximately 81 hours before compilation, checkpointing,
validation, and data-parallel synchronization overhead. The five-day stage
limit is therefore conservative but consistent with the current 4,000-update
stage definition.

## Find checkpoints, losses, and evaluations

For the four-GPU run, outputs are written below `RUN_DIR` from the command above:

```text
run_config.json                  resolved config, fingerprints, memory contract
train_metrics.jsonl             aggregate and per-horizon training losses
checkpoints/checkpoint_step*.pkl
latest_checkpoint.json
validation_metrics.jsonl        fixed 2022 sparse-objective validation
best_validation.json            lowest fixed-subset validation loss
train_validation_loss.png       training/validation loss plot
```

The four-GPU config validates every 1000 updates and at the final update. With `validation.num_segments=null`, it evaluates all complete validation segments from 2022. Validation uses deterministic keys and does not consume or modify the training RNG/cursor/state.

To summarize the completed one-GPU learning-rate sweep and regenerate its CSV/plot:

```bash
source scripts/graphcast_env.sh
python scripts/analyze_models/analyze_v23_sparse_lr_sweep.py \
  --root artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr_sweep_20260815
```

The analysis is written under that root's `analysis/` directory.

### Rollout evaluation status

The standalone maintained evaluator is:

```text
scripts/analyze_models/eval_v23_Ilya.py
```

It supports cold/warm initialization and baseline/full feedback, but its current data adapter expects one local, multi-year raw Zarr containing both training and validation years. The production res0.25 training data above is a prepared-array store, not that raw-Zarr interface. The only local raw res0.25 evaluation Zarr currently found is January 2022 alone:

```text
data/graphcast/graphcast/dataset/wb2_graphcast37_jan2022_0p25.zarr
```

That January-only store does not satisfy the evaluator's train/validation-year split. Do not pass the prepared-array root to `--data-path`, and do not present the January-only store as a full 2022 evaluation. Until the evaluator accepts prepared arrays or a multi-year raw res0.25 Zarr is provided, the supported res0.25 evaluation is the checkpoint validation produced by the training runner.

Once a compatible multi-year raw res0.25 Zarr exists, a warm full-feedback rollout has this form:

```bash
source scripts/graphcast_env.sh

RUN_DIR="artifacts/checkpoints/v23_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp4_8k_20260816/di16_bcg1_sparse_h20_closed_sg_stateful_dp4_lr3em6_8k"

python scripts/analyze_models/eval_v23_Ilya.py \
  --ckpt "${RUN_DIR}/checkpoints/checkpoint_step00008000.pkl" \
  --data-path /path/to/local_multi_year_graphcast37_res0p25.zarr \
  --ckpt-in "data/graphcast/graphcast/params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz" \
  --stats-dir data/graphcast/graphcast/stats \
  --resolution 0.25 --mesh-size 6 --width 512 \
  --baseline-msg-steps 16 --residual-msg-steps 2 \
  --temporal-d-inner 16 --temporal-bc-groups 1 \
  --target-steps 40 --warmup-steps 24 \
  --eval-mode warm_full --n-samples 32 \
  --out-json artifacts/evaluations/v23_Ilya/res0p25_dp4_step8000_warm_full.json
```

Use `cold_full` for zero-state evaluation without warmup, and use `warm_bp`/`cold_bp` when only the frozen baseline prediction is fed back instead of baseline plus residual.

## Direct verification request

Lianghong, please check these points against your frozen v22-clean implementation and report every mismatch:

1. Is the current interleaved Mamba placement and full-Mamba parameterization the intended successor to your v22 path?
2. Does `closed_loop_sg` stop gradients at the same feedback boundary as your trainer?
3. Is sparse-horizon indexing correct relative to the four teacher-forced transitions and 20-step closed-loop trajectory?
4. Should the horizon weights be normalized by 18, or should the scale match a different convention?
5. Does explicit residual-only reverse rematerialization produce the intended parameter and recurrent-state cotangents compared with your whole-step checkpointed loop?
6. Are FP32 recurrent-state boundaries and carry/reset points consistent with your checkpoints?
7. Are the 37-level variables, precipitation convention, 2019-2021/2022 split, and GraphCast-large checkpoint correct?
8. Are the one-GPU and four-replica data/RNG/cursor semantics comparable enough for the experiments we intend to report together?
9. Which cold/warm and baseline/full-feedback rollout modes should be the headline evaluation?

Please flag anything that is merely suspicious as well as anything definitely broken; a run can complete successfully while still being scientifically non-comparable.
