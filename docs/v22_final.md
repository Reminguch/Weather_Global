# v22_final

`v22_final` is the maintained, additive training and evaluation interface for
the final v22 residual-Mamba architecture. Its initial behavioral reference is commit
`cfe27d4`, especially:

- `scripts/training/full_mamba_v9/train_mz_v9.py` for the residual model and
  temporal attachment;
- `scripts/training/full_mamba_v23/eval_v22_clean.py` for rollout modes and
  metrics;
- `scripts/training/full_mamba_v20/train_mz_v20.py` for corrected forcing
  alignment, full-chunk BPTT loss, and open/closed-SG training behavior;
- the current modified `third_party/graphcast` integration.

The reference files remain unchanged. Production `v22_final` code does not
import from them; parity tests may import or execute them for comparison.

## Evaluation contract

The new entry point is `scripts/analyze_models/eval_v22_final.py`. It accepts
the legacy v22 evaluation arguments, except the v26-only tail-calibration head.
The default residual-state initialization is `zero`; `ckpt` and the legacy
`warm24` alias remain available explicitly.

The evaluator loads existing `.pkl` checkpoints without conversion, preserves
Haiku parameter names, and retains the legacy JSON metric sections. It adds
`architecture_id=v22_final`, schema, checkpoint-format, seed, resolution, and
requested/resolved state-initialization metadata.

Run Python and tests through the repository GraphCast environment:

```bash
source scripts/graphcast_env.sh
python scripts/analyze_models/eval_v22_final.py --help
python -m pytest -q tests/v22_final
```

## Training contract

The maintained trainer is `scripts/training/train_v22_final.py`. A versioned
JSON file is the canonical run definition; CLI overrides are limited to smoke
length/cadence, output location, and checkpoint initialization:

```bash
source scripts/graphcast_env.sh
python scripts/training/train_v22_final.py \
  --config configs/experiments/v22_final/res2_gc500k_k12_open.json \
  --dry-run
```

The two reference configurations reproduce the res2 GC500k AR-tail-K12 open
and closed-SG experiments. They use `PreparedArrayStore` for tensors. The
historical `precomputed_residuals/v22_res2_GCv1K3_synthetic` directory is an
anchor manifest: its metadata explicitly identifies `residuals/*.npy` as dummy
placeholders, and the trainer computes `truth - frozen_GraphCast(input)` live.

Training modes are deliberately distinct:

- A fresh run initializes residual GraphCast weights from the frozen baseline
  and initializes Mamba plus the residual head deterministically.
- `--init-from` loads legacy v20 weights and optional temporal state, then
  starts at update one with fresh optimizer, RNG, and data cursor.
- `--resume` accepts only a `v22_final_training_pickle`, verifies baseline and
  manifest fingerprints, and restores parameters, temporal state, Adam
  moments, RNG, cursor, and completed step exactly.

Every periodic checkpoint under `RUN/checkpoints/` is both exactly resumable
and directly readable by `eval_v22_final.py`. Checkpoints are written through
an atomic temporary-file rename; `latest_checkpoint.json` is updated only
after the checkpoint is complete. Training metrics are append-only JSONL.

Build uniform SWA from explicit, sorted checkpoints:

```bash
python scripts/training/build_v22_final_swa.py \
  --inputs RUN/checkpoints/checkpoint_step00042000.pkl \
           RUN/checkpoints/checkpoint_step00044000.pkl \
  --source-steps 42000 44000 \
  --output RUN/checkpoints/checkpoint_swa_step42000-44000.pkl
```

SWA files are evaluable warm starts but are not exact-resume checkpoints.

## Slurm workflow

The generic launcher requires `CONFIG` and accepts operational settings through
`MAX_STEPS`, `CHECKPOINT_EVERY`, `OUTPUT_ROOT`, `RUN_NAME`, `RESUME`, or
`INIT_FROM` environment variables. Its 32 GiB request is based on completed
v20 array job `11317907`, whose tasks peaked near 24 GiB. It enables XLA's
standard production kernels. The parity suite alone enables deterministic GPU
operations because GraphCast scatter reductions otherwise vary enough between
identical A100 runs to invalidate the `1e-6` comparison. Deterministic kernels
are a validation mode, not a production default: they are substantially slower
for this graph.

```bash
sbatch --export=ALL,CONFIG=configs/experiments/v22_final/res2_gc500k_k12_open.json \
  scripts/experiments/train_v22_final.slurm
```

The review-gate smoke submissions use distinct, initially absent run
directories and operational overrides only:

```bash
SMOKE_ROOT=artifacts/checkpoints/v22_final/training_migration_smoke
sbatch --time=00:30:00 \
  --export=ALL,CONFIG=configs/experiments/v22_final/res2_gc500k_k12_open.json,MAX_STEPS=20,CHECKPOINT_EVERY=10,OUTPUT_ROOT=${SMOKE_ROOT},RUN_NAME=open \
  scripts/experiments/train_v22_final.slurm
sbatch --time=00:30:00 \
  --export=ALL,CONFIG=configs/experiments/v22_final/res2_gc500k_k12_closed_sg.json,MAX_STEPS=20,CHECKPOINT_EVERY=10,OUTPUT_ROOT=${SMOKE_ROOT},RUN_NAME=closed_sg \
  scripts/experiments/train_v22_final.slurm
```

On a GPU allocation, evaluate either step-20 file with a two-step, one-sample
cold rollout. Use `cold_bp` for the open-feedback run and `cold_full` for the
closed-SG run:

```bash
python scripts/analyze_models/eval_v22_final.py \
  --ckpt ${SMOKE_ROOT}/open/checkpoints/checkpoint_step00000020.pkl \
  --ckpt-in artifacts/checkpoints/7_years/vanilla_gc_7y_res2_m4_w512_mp6_h6_bs8_accum1_stream500k/ckpt_step500000.npz \
  --data-path /scratch/gpfs/DABANIN/lm8598/Weather_Global/data/graphcast/graphcast/dataset/wb2_res1_levels13_2015_2022.zarr \
  --val-year 2022 --train-start-year 2015 --train-end-year 2021 \
  --resolution 2 --mesh-size 4 --width 512 \
  --baseline-msg-steps 6 --residual-msg-steps 2 \
  --target-steps 2 --warmup-steps 0 --eval-mode cold_bp \
  --temporal-d-inner 16 --temporal-d-state 16 --temporal-d-conv 4 \
  --temporal-layers 2 --no-temporal-stateful \
  --n-samples 1 --seed 18 --residual-state-init zero \
  --out-json ${SMOKE_ROOT}/open/eval/cold_bp_step20.json
```

No full 50k `v22_final` run should be submitted until the parity and 20-update
smoke gate has been reviewed. The opt-in A100 suite is:

```bash
XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true" \
JAX_COMPILATION_CACHE_DIR="${TMPDIR:-/tmp}/v22-final-jax-cache" \
V22_FINAL_TRAIN_GPU_TESTS=1 python -m pytest -q \
  tests/v22_final/test_training_integration_parity.py
```

GC-Mamba training and the v25/v26/v30 auxiliary heads are separate model
families and are not part of immutable `v22_final`. Legacy cleanup remains
deferred.
