# v23_Ilya endpoint-only residual-Mamba training

`v23_Ilya` is a standalone maintained GraphCast + residual-Mamba family. It does not import any previous training version. The frozen GraphCast baseline and the full residual branch use the same model construction as evaluation, while training uses an explicit reverse BPTT rule.

## Default res1 objective

The reference configuration is `configs/experiments/v23_Ilya/res1_endpoint_k24_di16_bcg1_closed_sg_stateful_20k.json`:

- 24 residual/Mamba transitions per update.
- Four teacher-forced input windows, represented by five unique weather frames.
- Twenty detached `closed_loop_sg` feedback transitions.
- Exactly one ERA5 target, at the final step.
- BF16 GraphCast/residual execution and a BF16 detached weather tape.
- FP32 recurrent-state boundaries, parameters, and optimizer state.
- Frozen GraphCast in the forward pass only; the reverse rule never replays it.

`objective.loss_mode` may be `last_step` (default) or `all_steps`. The latter loads all K targets and averages K losses. `memory.weather_tape_precision` may be `bf16` (default) or `fp32`. The only supported backend is `explicit_reverse_vjp`.

For K steps, endpoint mode saves K+1 unique dynamic input frames and K residual-state boundaries. It does not save overlapping two-frame windows, intermediate truth targets, or GraphCast activations. Reverse BPTT reconstructs each input window from adjacent frames and rematerializes one residual/Mamba step at a time.

## Data manifest

Build the independent anchor-only manifest once:

```bash
source scripts/graphcast_env.sh
python scripts/build_v23_Ilya_anchor_manifest.py
```

The generated `data/graphcast/graphcast/dataset/anchor_manifests/v23_Ilya_res1` manifest contains only indices, times, and deterministic splits. Training anchors end in 2021; validation anchors are in 2022. No precomputed or placeholder residual arrays are used.

Endpoint chunks read the four input windows as five unique frames, all 24 forcing frames, static fields once, and one final target. The selective-read counts and logical tape sizes are recorded in `run_config.json` under `derived.memory_contract`.

## Commands

Validate paths and the resolved contract without creating a run:

```bash
source scripts/graphcast_env.sh
python scripts/training/train_v23_Ilya.py \
  --config configs/experiments/v23_Ilya/res1_endpoint_k24_di16_bcg1_closed_sg_stateful_20k.json \
  --dry-run
```

Start training by removing `--dry-run`. Exact resume accepts only `v23_Ilya_training_pickle`; `--init-from` can parse compatible legacy residual, training, and SWA payloads using v23-owned code, then resets optimizer, RNG, step, and cursor.

Evaluation and SWA use the standalone entrypoints:

```bash
python scripts/analyze_models/eval_v23_Ilya.py --help
python scripts/training/build_v23_Ilya_swa.py --help
```

## Validation

```bash
source scripts/graphcast_env.sh
JAX_PLATFORMS=cpu PYTHONPATH=. pytest -q tests/v23_Ilya
```

The tests cover strict config parsing, selective truth reads, both loss modes, explicit-VJP parity with a naïve stateful unroll, JIT compilation, unique-frame tape accounting, checkpoint compatibility, rollout/model regressions, and dependency isolation.

A real accelerator memory profile is intentionally not launched by the implementation. Run it on the same device as the res1 reference with allocator preallocation disabled, then compare the observed peak against the previous 20–27 GiB range. Cross-device GraphCast mesh sharding and res0.25 configuration remain separate follow-up work.
