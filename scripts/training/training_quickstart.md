# Training Quickstart

Short reference for the current GraphCast and Mamba training paths. Run from the
repo root after activating the environment:

```bash
source scripts/graphcast_env.sh
```

## Prepared Data

Build reusable memmap-backed streaming stores once per source/checkpoint task:

```bash
python -m src.data_operations.preprocessing.prepare_graphcast_streaming_store \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --ckpt-in "${CKPT_IN:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}" \
  --out-root "${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}" \
  --resolutions 1 2 3 4 9 18
```

Then train with `--data-source prepared_array --batch-builder prepared_array`. The store path is
derived from `--prepared-data-root` and `--resolution`, for example
`data/graphcast/graphcast/dataset/prepared_stream/res1`.

## Which Command?

### `gc`: vanilla GraphCast

Direct Python:

```bash
python -u -m src.models.graphcast.training.standard_training \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --data-source "${DATA_SOURCE:-raw}" \
  --prepared-data-root "${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}" \
  --stats-dir "${STATS_DIR:-data/graphcast/graphcast/stats}" \
  --ckpt-in "${CKPT_IN:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}" \
  --out-dir "${OUT_DIR:-artifacts/checkpoints/graphcast_manual}" \
  --run-name "${RUN_NAME:-gc_res4_m4_w128_mp1}" \
  --val-year "${VAL_YEAR:-2021}" \
  --resolution "${RESOLUTION:-4}" \
  --mesh-size "${MESH_SIZE:-4}" \
  --width "${WIDTH:-128}" \
  --processor-msg-steps "${MSG:-1}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --max-steps "${MAX_STEPS:-2000}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --batch-builder "${BATCH_BUILDER:-vectorized}" \
  --precision "${PRECISION:-bf16}"
```

SLURM wrapper:

```bash
RESOLUTION=4 MESH_SIZE=4 WIDTH=128 MSG=1 sbatch scripts/training/standard_training.slurm
```

### `gc_mamba`: GraphCast plus Mamba

```bash
python -u -m src.models.mamba.training.segments_training \
  --model gc_mamba \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --data-source "${DATA_SOURCE:-raw}" \
  --prepared-data-root "${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}" \
  --stats-dir "${STATS_DIR:-data/graphcast/graphcast/stats}" \
  --ckpt-in "${CKPT_IN:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}" \
  --out-dir "${OUT_DIR:-artifacts/checkpoints/gc_mamba_segments}" \
  --run-name "${RUN_NAME:-gc_mamba_res4_m4_w128_mp1}" \
  --val-year "${VAL_YEAR:-2021}" \
  --resolution "${RESOLUTION:-4}" \
  --mesh-size "${MESH_SIZE:-4}" \
  --width "${WIDTH:-128}" \
  --processor-msg-steps "${MSG:-1}" \
  --input-duration "${INPUT_DURATION:-12h}" \
  --target-steps "${TARGET_STEPS:-1}" \
  --len-segment "${LEN_SEGMENT:-32}" \
  --bptt-steps "${BPTT_STEPS:-8}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --max-steps "${MAX_STEPS:-2000}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --batch-builder "${BATCH_BUILDER:-direct}" \
  --precision "${PRECISION:-bf16}" \
  --temporal-backbone "${TEMPORAL_BACKBONE:-mamba}" \
  --temporal-location "${TEMPORAL_LOCATION:-mesh_processor_interleaved}" \
  --temporal-d-inner "${TEMPORAL_D_INNER:-16}" \
  --temporal-d-state "${TEMPORAL_D_STATE:-16}" \
  --temporal-stateful
```

SLURM wrapper:

```bash
MODEL=gc_mamba RESOLUTION=4 MESH_SIZE=4 WIDTH=128 MSG=1 sbatch scripts/training/segments_training.slurm
```

### `gc_mamba - frozen`: initialize from GraphCast, train only Mamba

Use this when `CKPT_IN` is a trained vanilla GraphCast checkpoint with the same
shape you want to augment.

```bash
python -u -m src.models.mamba.training.segments_training \
  --model gc_mamba \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --data-source "${DATA_SOURCE:-raw}" \
  --prepared-data-root "${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}" \
  --stats-dir "${STATS_DIR:-data/graphcast/graphcast/stats}" \
  --ckpt-in "${CKPT_IN}" \
  --init-from-graphcast-ckpt "${CKPT_IN}" \
  --trainable-part mamba \
  --zero-init-temporal-out \
  --out-dir "${OUT_DIR:-artifacts/checkpoints/gc_mamba_frozen_gc}" \
  --run-name "${RUN_NAME:-gc_mamba_frozen_gc_res4_m4_w128_mp1}" \
  --val-year "${VAL_YEAR:-2021}" \
  --resolution "${RESOLUTION:-4}" \
  --mesh-size "${MESH_SIZE:-4}" \
  --width "${WIDTH:-128}" \
  --processor-msg-steps "${MSG:-1}" \
  --input-duration "${INPUT_DURATION:-12h}" \
  --target-steps "${TARGET_STEPS:-1}" \
  --len-segment "${LEN_SEGMENT:-32}" \
  --bptt-steps "${BPTT_STEPS:-8}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --max-steps "${MAX_STEPS:-2000}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --batch-builder "${BATCH_BUILDER:-direct}" \
  --precision "${PRECISION:-bf16}" \
  --temporal-backbone mamba \
  --temporal-location "${TEMPORAL_LOCATION:-mesh_processor_interleaved}" \
  --temporal-d-inner "${TEMPORAL_D_INNER:-16}" \
  --temporal-d-state "${TEMPORAL_D_STATE:-16}" \
  --temporal-stateful
```

### `res_mamba`: residual Mamba

Residual Mamba needs a baseline checkpoint. The baseline is frozen and used to
build residual targets: `target - baseline_prediction`.

The residual branch is a fresh GraphCast-shaped correction model:

```text
frozen baseline -> baseline_prediction

residual inputs
  -> grid2mesh
  -> mesh processor + Mamba
  -> mesh2grid
  -> zero-init residual output head
  -> residual_prediction

forecast = baseline_prediction + residual_prediction
```

For GC-Mamba, zero-initializing the Mamba output projection makes the inserted
Mamba block initially behave like a no-op on top of a GraphCast forecast. For
residual Mamba, the Mamba no-op alone is not enough because the residual branch
is freshly initialized. The final residual output head is therefore also
zero-initialized, so a fresh residual run starts from `baseline_prediction + 0`.

Residual targets are computed online from the frozen baseline during training.
When `--temporal-stateful` is enabled, Mamba carries both SSM state and its
`d_conv - 1` causal convolution cache across BPTT steps.

```bash
python -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --data-source "${DATA_SOURCE:-raw}" \
  --prepared-data-root "${PREPARED_DATA_ROOT:-data/graphcast/graphcast/dataset/prepared_stream}" \
  --stats-dir "${STATS_DIR:-data/graphcast/graphcast/stats}" \
  --baseline-ckpt "${BASELINE_CKPT}" \
  --out-dir "${OUT_DIR:-artifacts/checkpoints/residual_mamba_segments}" \
  --run-name "${RUN_NAME:-res_mamba_res4_m4_w128_mp1}" \
  --val-year "${VAL_YEAR:-2021}" \
  --resolution "${RESOLUTION:-4}" \
  --mesh-size "${MESH_SIZE:-4}" \
  --width "${WIDTH:-128}" \
  --processor-msg-steps "${MSG:-1}" \
  --input-duration "${INPUT_DURATION:-12h}" \
  --target-steps "${TARGET_STEPS:-1}" \
  --len-segment "${LEN_SEGMENT:-32}" \
  --bptt-steps "${BPTT_STEPS:-8}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --max-steps "${MAX_STEPS:-2000}" \
  --lr "${LR:-1e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --batch-builder "${BATCH_BUILDER:-direct}" \
  --precision "${PRECISION:-bf16}" \
  --temporal-backbone "${TEMPORAL_BACKBONE:-mamba}" \
  --temporal-location "${TEMPORAL_LOCATION:-mesh_processor_interleaved}" \
  --temporal-d-inner "${TEMPORAL_D_INNER:-16}" \
  --temporal-d-state "${TEMPORAL_D_STATE:-16}" \
  --temporal-d-conv "${TEMPORAL_D_CONV:-4}" \
  --temporal-stateful
```

Fresh residual-Mamba runs enable the final residual output head by default. When
resuming, `--residual-output-head auto` preserves the existing run's
`run_config.json` setting so old checkpoints stay compatible. Use
`--residual-output-head enabled` only when intentionally migrating an old run to
the new head.

See `src/models/mamba/residual_mamba/residual_mamba.md` for the current
architecture notes.

SLURM wrapper:

```bash
MODEL=residual_mamba BASELINE_CKPT=artifacts/checkpoints/path/to/ckpt_best.npz \
  sbatch scripts/training/segments_training.slurm
```

## Eval During Training

Segment training runs validation inside the training job. The important controls are:

```bash
--eval-every 2000 \
--eval-batch-size 4 \
--eval-num-segments 16 \
--final-eval-num-segments all \
--checkpoint-every 2000
```

For `gc_mamba`, validation uses the same segment/BPTT semantics as training. Validation
segments are chronological, Mamba state is reset at the start of each segment, and state
is carried through the BPTT chunks inside that segment. When `--target-steps K > 1`,
each BPTT chunk uses the first `bptt_steps - K` anchors as a truth-fed prefix, then
feeds model predictions back for the final `K` anchors and averages loss only over that
corrected autoregressive tail. The physical input stream restarts from truth at the next
chunk; only Mamba/temporal state carries across chunks.

```text
src.models.mamba.training.segments_training.run_gc_mamba_training
  -> src.models.graphcast.training.core.segments.run_eval_segments
  -> transformed.apply(..., is_training=False)
```

For `res_mamba`, validation uses a frozen baseline checkpoint and scores the residual
model in training-equivalent form for one-step training:

```text
baseline_prediction = frozen_graphcast(inputs)
residual_target = target - baseline_prediction
loss = residual_model(residual_inputs, residual_target)
```

The full forecast equivalence is:

```text
loss(residual_pred, target - baseline_prediction)
  == loss(baseline_prediction + residual_pred, target)
```

With `--target-steps K > 1`, residual validation mirrors chunk-local corrected AR:
truth-prefix residual history is advanced with `target - baseline_prediction`, tail
residual history is advanced with the model residual prediction, and full GraphCast
inputs are advanced with `baseline_prediction + residual_prediction`. At the next chunk,
full inputs restart from truth while Mamba state carries detached.

```text
src.models.mamba.residual_mamba.training.runner.run_training
  -> src.models.mamba.residual_mamba.training.model.run_residual_eval
  -> baseline_predict_transform.apply(...)
  -> residual_eval_transform.apply(..., is_training=False)
```

Training outputs are written under `${OUT_DIR}/${RUN_NAME}`:

```text
ckpt_*.npz
ckpt_best.npz
best_checkpoint.json
run_config.json
train_loss.json
eval_loss.json
eval_details.json
chunk_timing.json
chunk_timing_summary.json
```

`ckpt_best.npz` is updated whenever the regular validation loss improves. The final eval
runs after training finishes and is appended to `eval_loss.json` and `eval_details.json`.

## Standalone Eval And Long Rollouts

Use `scripts/analyze_models/unified_resolution_eval.py` for checkpoint evaluation across
GraphCast, `gc_mamba`, and `residual_mamba`. It discovers checkpoints from
`ckpt_best.npz` files, reads each checkpoint's `run_config.json`, routes to the correct
runtime, and writes resolution-eval CSV rows under `plots/analyze_models/data/resolution_eval`.

Direct prepared-array eval:

```bash
python -u scripts/analyze_models/unified_resolution_eval.py \
  --families gc_mamba residual_mamba \
  --resolutions 2 4 9 \
  --data-source prepared_array \
  --checkpoint-roots artifacts/checkpoints/path/to/runs \
  --eval-year 2022 \
  --lead-days 1 2 4 \
  --metrics weighted_allvars per_variable \
  --eval-modes cold warm
```

Use `--lead-steps` for long rollouts or non-day-aligned lead times. One model step is 6h,
so these examples evaluate 6h, 1d, 2d, 4d, 6d, and 8d:

```bash
python -u scripts/analyze_models/unified_resolution_eval.py \
  --families gc_mamba residual_mamba \
  --resolutions 2 \
  --data-source prepared_array \
  --checkpoint-roots artifacts/checkpoints/path/to/runs \
  --eval-year 2022 \
  --lead-steps 1 4 8 16 24 32 \
  --metrics weighted_allvars per_variable \
  --eval-modes cold warm
```

Cold eval starts each forecast window from the checkpoint's initial model state. It rolls
the model forward autoregressively for the requested lead steps and feeds predictions
back into the next input window. For stateful Mamba checkpoints, this means Mamba memory
accumulates only within that forecast window.

Warm eval is truth-anchored before each scored branch. It first advances the model through
`--warmup-steps` using truth as feedback, then repeatedly scores a branch rollout of up to
the requested lead steps while continuing the trunk with truth feedback for
`--trunk-steps`. This is useful for asking whether stateful Mamba helps after memory has
already been warmed on a continuous trajectory.

Residual Mamba has two standalone eval semantics:

```bash
--residual-eval-semantics teacher_forced_training_equivalent
--residual-eval-semantics rollout
```

`teacher_forced_training_equivalent` matches one-step residual segment training/eval:
residual history is advanced with `target - baseline_prediction`, and full GraphCast
inputs are advanced with truth in the teacher-forced parts. `rollout` advances residual
history with the model's residual prediction and advances full inputs with the combined
forecast `baseline_prediction + residual_prediction`. Use `rollout` when measuring true
long autoregressive behavior; use `teacher_forced_training_equivalent` when comparing
against one-step teacher-forced objectives.

SLURM array eval discovers available `family:res` shards and then launches one array task
per shard:

```bash
FAMILIES="gc_mamba residual_mamba" \
RESOLUTIONS="2 4 9" \
CHECKPOINT_ROOTS="artifacts/checkpoints/path/to/runs" \
EVAL_YEAR=2022 \
EVAL_MODES="cold warm" \
METRICS="weighted_allvars per_variable" \
bash scripts/analyze_models/submit_resolution_eval_array.sh
```

For long-rollout array eval, the worker script accepts explicit lead steps through
`LEAD_STEPS`. The convenience submit wrapper currently discovers/submits the default
lead setup, so for explicit long-rollout shards submit the worker directly or extend the
wrapper export list.

```bash
export SHARD_SPECS="gc_mamba:2"
export CHECKPOINT_ROOTS="artifacts/checkpoints/path/to/runs"
export EVAL_YEAR=2022
export EVAL_MODES="cold warm"
export LEAD_STEPS="1 4 8 16 24 32"
export RESIDUAL_EVAL_SEMANTICS=rollout
sbatch --array=0-0 scripts/analyze_models/run_resolution_eval_array.slurm
```

The array worker is `scripts/analyze_models/run_resolution_eval_array.slurm`; the merge
job is `scripts/analyze_models/run_resolution_eval_merge.slurm`; the merge script calls
`scripts/analyze_models/merge_resolution_eval.py` to create combined CSVs and default
plots.

## Parameters And Options

| Group | Variable / flag | Meaning |
| --- | --- | --- |
| Data/checkpoints | `DATA_PATH`, `--data-path` | Local GraphCast-style ERA5 dataset, usually `.zarr`. |
| Data/checkpoints | `DATA_SOURCE`, `--data-source` | `raw` reads `DATA_PATH`; `prepared_array` reads `PREPARED_DATA_ROOT/res{RESOLUTION}`. |
| Data/checkpoints | `PREPARED_DATA_ROOT`, `--prepared-data-root` | Root for prepared-array memmap stores, default `data/graphcast/graphcast/dataset/prepared_stream`. |
| Data/checkpoints | `STATS_DIR`, `--stats-dir` | Normalization and weighting stats directory. |
| Data/checkpoints | `CKPT_IN`, `--ckpt-in` | GraphCast checkpoint used by `gc` and `gc_mamba`. |
| Data/checkpoints | `BASELINE_CKPT`, `--baseline-ckpt` | Required for `res_mamba`; frozen baseline used for residual targets. |
| Data/checkpoints | `RESUME_CKPT`, `--resume-ckpt` | Optional residual Mamba checkpoint to resume from. |
| Model shape | `RESOLUTION`, `--resolution` | Model grid resolution in degrees. |
| Model shape | `MESH_SIZE`, `--mesh-size` | GraphCast mesh refinement size. |
| Model shape | `WIDTH`, `--width` | GraphCast latent width. |
| Model shape | `MSG`, `--processor-msg-steps` | Number of GraphCast processor message-passing steps. |
| Segment training | `INPUT_DURATION`, `--input-duration` | Input history window, for example `12h`. |
| Segment training | `TARGET_STEPS`, `--target-steps` | Exact corrected AR tail length inside each BPTT chunk. `1` keeps one-step training; values `>1` must be less than `BPTT_STEPS`. |
| Segment training | `LEN_SEGMENT`, `--len-segment` | Chronological segment length. |
| Segment training | `BPTT_STEPS`, `--bptt-steps` | Truncated BPTT chunk length; must divide `LEN_SEGMENT`. |
| Eval in training | `EVAL_EVERY`, `--eval-every` | Optimizer-step interval for validation. |
| Eval in training | `EVAL_BATCH_SIZE`, `--eval-batch-size` | Number of validation segment lanes per eval chunk. |
| Eval in training | `EVAL_NUM_SEGMENTS`, `--eval-num-segments` | Number of validation segments for regular evals, or `all`. |
| Eval in training | `FINAL_EVAL_NUM_SEGMENTS`, `--final-eval-num-segments` | Number of validation segments for final eval, or `all`. |
| Eval in training | `CHECKPOINT_EVERY`, `--checkpoint-every` | Optimizer-step interval for regular `ckpt_*.npz` saves. |
| Optimization | `BATCH_SIZE`, `--batch-size` | Training batch size. |
| Optimization | `MAX_STEPS`, `--max-steps` | Optimizer update count. |
| Optimization | `LR`, `--lr` | Learning rate. |
| Optimization | `WEIGHT_DECAY`, `--weight-decay` | AdamW weight decay. |
| Optimization | `PRECISION`, `--precision` | One of `bf16`, `fp16`, or `fp32`. |
| Data pipeline | `BATCH_BUILDER`, `--batch-builder` | `legacy`, `vectorized`, `direct`, `numpy`, or `prepared_array`; `numpy` requires an active full-RAM cache. |
| Mamba | `TEMPORAL_BACKBONE`, `--temporal-backbone` | Use `mamba` to enable Mamba, `none` for no temporal module. |
| Mamba | `TEMPORAL_LOCATION`, `--temporal-location` | Usually `mesh_processor_interleaved`; also supports `mesh_post_encoder`. |
| Mamba | `TEMPORAL_STATEFUL`, `--temporal-stateful` | Carry Mamba state across autoregressive/segment steps. |
| Mamba | `TEMPORAL_D_INNER`, `--temporal-d-inner` | Mamba internal channel width. |
| Mamba | `TEMPORAL_D_STATE`, `--temporal-d-state` | Mamba SSM state size per internal channel. |

## Checkpoint Semantics

- `gc` uses `--ckpt-in` for initialization and resume-style loading.
- `gc_mamba` uses `--ckpt-in`, but GraphCast is not frozen by default.
- `gc_mamba - frozen` adds `--init-from-graphcast-ckpt`, `--trainable-part mamba`, and `--zero-init-temporal-out`.
- `res_mamba` requires `--baseline-ckpt`; the baseline GraphCast is frozen and used only to form residual targets.

For the full parser surface, use:

```bash
python -m src.models.graphcast.training.standard_training --help
python -m src.models.mamba.training.segments_training --help
```
