# Training Quickstart

Short reference for the current GraphCast and Mamba training paths. Run from the
repo root after activating the environment:

```bash
source scripts/graphcast_env.sh
```

## Prepared Data

Build reusable prepared stores once per source/checkpoint task:

```bash
python -m src.data_operations.preprocessing.prepare_graphcast_training_store \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --ckpt-in "${CKPT_IN:-data/graphcast/graphcast/params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz}" \
  --out-root "${PREPARED_DATA_ROOT:-prepared}" \
  --resolutions 1 2 4 9 15
```

Then train with `--data-source prepared --batch-builder direct`. The store path is derived from
`--prepared-data-root` and `--resolution`, for example `prepared/res1`.

## Which Command?

### `gc`: vanilla GraphCast

Direct Python:

```bash
python -u -m src.models.graphcast.training.standard_training \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --data-source "${DATA_SOURCE:-raw}" \
  --prepared-data-root "${PREPARED_DATA_ROOT:-prepared}" \
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
  --prepared-data-root "${PREPARED_DATA_ROOT:-prepared}" \
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
  --temporal-hidden-size "${TEMPORAL_HIDDEN_SIZE:-128}" \
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
  --prepared-data-root "${PREPARED_DATA_ROOT:-prepared}" \
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
  --temporal-hidden-size "${TEMPORAL_HIDDEN_SIZE:-128}" \
  --temporal-d-inner "${TEMPORAL_D_INNER:-16}" \
  --temporal-d-state "${TEMPORAL_D_STATE:-16}" \
  --temporal-stateful
```

### `res_mamba`: residual Mamba

Residual Mamba needs a baseline checkpoint. The baseline is frozen and used to
build residual targets: `target - baseline_prediction`.

```bash
python -u -m src.models.mamba.training.segments_training \
  --model residual_mamba \
  --data-path "${DATA_PATH:-data/graphcast/graphcast/dataset/wb2_res1_levels13_train.zarr}" \
  --data-source "${DATA_SOURCE:-raw}" \
  --prepared-data-root "${PREPARED_DATA_ROOT:-prepared}" \
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
  --temporal-hidden-size "${TEMPORAL_HIDDEN_SIZE:-128}" \
  --temporal-d-inner "${TEMPORAL_D_INNER:-16}" \
  --temporal-d-state "${TEMPORAL_D_STATE:-16}" \
  --temporal-stateful
```

SLURM wrapper:

```bash
MODEL=residual_mamba BASELINE_CKPT=artifacts/checkpoints/path/to/ckpt_best.npz \
  sbatch scripts/training/segments_training.slurm
```

## Parameters And Options

| Group | Variable / flag | Meaning |
| --- | --- | --- |
| Data/checkpoints | `DATA_PATH`, `--data-path` | Local GraphCast-style ERA5 dataset, usually `.zarr`. |
| Data/checkpoints | `DATA_SOURCE`, `--data-source` | `raw` reads `DATA_PATH`; `prepared` reads `PREPARED_DATA_ROOT/res{RESOLUTION}`. |
| Data/checkpoints | `PREPARED_DATA_ROOT`, `--prepared-data-root` | Root for prepared Zarr stores, default `prepared`. |
| Data/checkpoints | `STATS_DIR`, `--stats-dir` | Normalization and weighting stats directory. |
| Data/checkpoints | `CKPT_IN`, `--ckpt-in` | GraphCast checkpoint used by `gc` and `gc_mamba`. |
| Data/checkpoints | `BASELINE_CKPT`, `--baseline-ckpt` | Required for `res_mamba`; frozen baseline used for residual targets. |
| Data/checkpoints | `RESUME_CKPT`, `--resume-ckpt` | Optional residual Mamba checkpoint to resume from. |
| Model shape | `RESOLUTION`, `--resolution` | Model grid resolution in degrees. |
| Model shape | `MESH_SIZE`, `--mesh-size` | GraphCast mesh refinement size. |
| Model shape | `WIDTH`, `--width` | GraphCast latent width. |
| Model shape | `MSG`, `--processor-msg-steps` | Number of GraphCast processor message-passing steps. |
| Segment training | `INPUT_DURATION`, `--input-duration` | Input history window, for example `12h`. |
| Segment training | `TARGET_STEPS`, `--target-steps` | Autoregressive target steps; segmented Mamba training currently uses `1`. |
| Segment training | `LEN_SEGMENT`, `--len-segment` | Chronological segment length. |
| Segment training | `BPTT_STEPS`, `--bptt-steps` | Truncated BPTT chunk length; must divide `LEN_SEGMENT`. |
| Optimization | `BATCH_SIZE`, `--batch-size` | Training batch size. |
| Optimization | `MAX_STEPS`, `--max-steps` | Optimizer update count. |
| Optimization | `LR`, `--lr` | Learning rate. |
| Optimization | `WEIGHT_DECAY`, `--weight-decay` | AdamW weight decay. |
| Optimization | `PRECISION`, `--precision` | One of `bf16`, `fp16`, or `fp32`. |
| Data pipeline | `BATCH_BUILDER`, `--batch-builder` | `legacy`, `vectorized`, `direct`, or `numpy`; `numpy` requires an active full-RAM cache. |
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
