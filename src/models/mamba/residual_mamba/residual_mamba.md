# Residual Mamba Architecture

`residual_mamba` trains a fresh GraphCast-shaped correction model beside a
frozen GraphCast baseline.

```text
baseline_inputs -> frozen GraphCast -> baseline_prediction

residual_inputs
  -> grid2mesh
  -> mesh processor + Mamba
  -> mesh2grid
  -> zero-init residual output head
  -> residual_prediction

forecast = baseline_prediction + residual_prediction
target = truth - baseline_prediction
```

## Current Design

- The frozen baseline checkpoint is used online during training and evaluation
  to compute `truth - baseline_prediction`; residual targets are not
  precomputed.
- The residual branch uses fresh parameters unless `--resume-ckpt` is supplied.
- Mamba can be inserted in the mesh processor path and, when stateful, carries
  both SSM state and a causal convolution cache of width `d_conv - 1`.
- The final residual output head is a per-grid-node channel projection
  `Linear(num_outputs -> num_outputs)` applied after `mesh2grid` and before
  residual unnormalization.
- The residual output head is zero-initialized, so a fresh residual run starts
  from the exact frozen baseline forecast.

## No-Op Initialization

GC-Mamba and residual-Mamba need different no-op mechanisms.

- GC-Mamba starts from a pretrained GraphCast checkpoint; zero-initializing the
  Mamba output projection makes the inserted Mamba block initially do nothing.
- Residual-Mamba has a fresh correction branch; zero-initializing Mamba alone
  does not force the full residual correction to zero. The final residual output
  head provides the residual-specific no-op.

## Checkpoint Compatibility

New fresh residual-Mamba runs enable the output head by default. Runtime reads
`residual_training.output_head.enabled` from `run_config.json`; missing metadata
means the head is disabled so older checkpoints load with their original
architecture.
