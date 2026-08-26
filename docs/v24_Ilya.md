# v24_Ilya: precision-correct physical trajectories

`v24_Ilya` is a standalone snapshot of the working-tree `v23_Ilya` model with
one intentional scientific change: every physical weather state remains FP32.
The architecture, sparse/full objectives, state carry, closed-loop training,
final zero residual head, and both legacy-Haiku and Mamba1 initialization
options are otherwise preserved.

## Precision contract

The required pipeline is:

```text
FP32 physical inputs and forcings
  -> FP32 GraphCast normalization
  -> BF16 GraphCast/Mamba neural compute
  -> FP32 normalized prediction
  -> FP32 unnormalization/residual addition
  -> FP32 autoregressive feedback and weather tape
```

This follows the wrapper ordering used by GraphCast: the BF16 cast wrapper is
inside the normalization/residual wrapper. BF16 therefore reduces neural
activation precision, but it does not permanently replace the physical state.
The normalized neural prediction is cast to the FP32 target-template dtype;
unnormalization and the residual connection then use the original FP32 input.

`memory.weather_tape_precision` remains in JSON metadata for auditability but
must be `"fp32"`. A raw BF16 setting is rejected. Teacher frames, predictions,
feedback frames, reverse-recomputation frames, recurrent-state boundaries,
parameters, optimizer state, residual targets, and losses use FP32 boundaries.

## Initialization

The final `temporal_residual_head` remains exactly zero-initialized, so the
fresh residual branch produces zero and the complete model begins at frozen
GraphCast. `architecture.temporal_zero_init_out` remains supported and is true
in the reference v24 configurations. `legacy_haiku` and `mamba1` temporal
initialization, including Mamba1 `dt_proj` configuration, are unchanged from
the v23 working-tree snapshot.

## Checkpoint boundary

Only the following native formats are accepted:

- `v24_Ilya_training_pickle`
- `v24_Ilya_data_parallel_training_pickle`
- `v24_Ilya_swa_pickle`

Resume is exact and native-v24-only. Evaluation, `--init-from`, validation
comparison, and SWA also reject v23/v22 architecture IDs and formats. This is
intentional: v23 checkpoints learned against a quantized raw-BF16 physical
trajectory and are not scientifically interchangeable with v24.

## Reference configurations and commands

The res1 five-update smoke uses sparse H20 supervision, legacy initialization,
the final zero head, FP32 weather, BF16 model compute, and LR `3e-6`:

```bash
source scripts/graphcast_env.sh
PYTHONPATH=. python scripts/training/train_v24_Ilya.py \
  --config configs/experiments/v24_Ilya/res1_sparse_h20_di16_bcg1_fp32_smoke.json \
  --dry-run
```

The controlled res0.25 DP2 run trains for 200 updates and checkpoints/validates
at steps 50, 100, 150, and 200:

```bash
source scripts/graphcast_env.sh
PYTHONPATH=. python scripts/training/train_v24_Ilya.py \
  --config configs/experiments/v24_Ilya/res0p25_2019_2022_sparse_h20_lr3em6_dp2_200.json \
  --dry-run
```

Prepared Slurm launchers are:

- `scripts/experiments/train_v24_Ilya_res1_fp32_smoke.slurm`
- `scripts/experiments/eval_v24_Ilya_res0p25_fixed_validation.slurm`
- `scripts/experiments/train_v24_Ilya_res0p25_sparse_h20_lr3em6_dp2_200.slurm`
- `scripts/experiments/eval_v24_Ilya_res0p25_sparse_h20_checkpoints.slurm`

No launcher is submitted automatically.

## Acceptance gate

The matched zero-residual res0.25 fixed validation must reproduce aggregate
GraphCast loss `5.307552` and matched per-horizon values within absolute
tolerance `1e-3`; it must not reproduce the raw-BF16 baseline `8.113585`.
The fixed-validation launcher enforces this automatically with
`scripts/analyze_models/verify_v24_Ilya_fp32_validation.py`.
Canonical evaluation reports exact original GraphCast loss reduction, computed
after aggregating model and baseline losses over the selected samples/leads.

Run the 200-step causal screen only after the smoke and baseline calibration.
Evaluate checkpoints 50/100/150/200. Continue beyond 200 only if matched
training validation and canonical evaluation agree on the sign and at least one
checkpoint beats the FP32 zero-residual GraphCast baseline.

The fixed-validation launcher requests two A100-80GB GPUs, 160 GB host memory,
and 1:20 walltime; the comparable completed job used about 110 GiB and 35
minutes. Longer training resources should be revised from the FP32 smoke's
recorded GPU and host-memory telemetry.
