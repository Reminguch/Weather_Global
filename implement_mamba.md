# Implement Mamba On Mesh Latent In GraphCast

## Goal

Add temporal modeling with Mamba **after grid->mesh encoding** and before GraphCast processor, so multi-history input (`T=2/4/6/8/12`) is handled in mesh latent space.

This document targets:
- minimal-risk first implementation;
- backward compatibility with current training/eval scripts;
- clear ablation path vs existing GraphCast baseline.

---

## Scope and Non-Goals

### In Scope
- Fork/modify GraphCast source used by this repo.
- Insert a temporal block at mesh latent stage.
- Keep old path available via config switch.
- Train/eval with same MAE-vs-lead pipeline.

### Out of Scope (v1)
- Rewriting full GraphCast processor.
- New loss functions.
- Mixed temporal modules in multiple layers at once.

---

## High-Level Design

Current conceptual path:
1. Inputs on grid (with history and forcings).
2. Grid features projected/encoded to mesh.
3. Mesh processor (message passing).
4. Mesh to grid decode.

Target v1 path:
1. Build per-time-step grid features.
2. Encode each step to mesh latent.
3. Apply `TemporalMeshBlock` (Mamba) over time dimension on mesh latent.
4. Aggregate/select temporal output as processor input.
5. Continue normal GraphCast processor and decoder.

Key idea:
- Temporal modeling should see mesh-structured latent, not only raw grid features.

---

## Repository Plan

## 1) Branching

Create dedicated branch in this repo:
- `exp/mamba-mesh-v1`

Optional split branches:
- `exp/mamba-mesh-v1-plumbing`
- `exp/mamba-mesh-v1-training`
- `exp/mamba-mesh-v1-eval`

## 2) GraphCast Source Strategy

Prefer one of:
- Vendor local fork under `third_party/graphcast_mamba/`
- Or patch existing graphcast import path with a local module overlay

Recommendation:
- `third_party/graphcast_mamba/` + explicit imports in training script.

Reason:
- clear diff and easier reproducibility.

---

## Implementation Steps

## Phase A: Plumbing and Config

1. Add new config flags in training entrypoint (`scripts/training/train_graphcast.py`):
- `--temporal-backbone {none,mamba}`
- `--temporal-location {mesh_post_encoder}`
- `--temporal-hidden-size`
- `--temporal-layers`
- `--temporal-dropout`

2. Add runtime checks:
- if `temporal-backbone=none`, exact old behavior.
- if `mamba`, require `input_steps >= 2`.

3. Persist all temporal config into `run_config.json`.

Acceptance:
- Existing command lines run unchanged.

## Phase B: Mesh-Latent Temporal Block

1. Add module file:
- `src/models/temporal_mesh_mamba.py`

2. Module interface:
- Input: mesh latent with explicit time axis (shape conceptually `[B, T, N_mesh, C]`)
- Output: same shape or reduced-to-last-step shape `[B, N_mesh, C]`

3. Start with simple policy:
- run Mamba over `T` independently for each mesh node;
- take last-step output as processor input.

4. Add optional residual + layer norm around temporal block.

Acceptance:
- unit test for shape consistency and deterministic forward pass.

## Phase C: Integrate Into GraphCast Path

1. Modify GraphCast internals at encoder->processor boundary to expose mesh latent with time semantics.
2. Insert `TemporalMeshBlock` before message passing.
3. Keep switchable old path (`temporal-backbone=none`).

Acceptance:
- `T=2` with `none` reproduces old numerics (within tolerance).
- `T=2` with `mamba` runs end-to-end train/eval.

## Phase D: Training and Stability

1. Two-stage training schedule:
- Stage 1: freeze GraphCast spatial core, train only temporal block.
- Stage 2: unfreeze all, joint finetune.

2. Suggested first sweep:
- history: `2,4,6,8`
- steps: `10k, 20k`
- seed: `0` first; add `1,2` after pipeline stable.

3. Keep same MAE-vs-lead eval script/arguments for fair comparison.

Acceptance:
- comparable MAE tables generated automatically.

---

## Files Likely To Touch

Primary:
- `scripts/training/train_graphcast.py`
- `scripts/training/hist_longrun_20k_local.sh`
- `scripts/training/hist_longrun_20k.slurm`
- `scripts/analyze_models/mae_vs_lead.py` (only if input-step handling needs updates)

New:
- `src/models/temporal_mesh_mamba.py`
- `tests/test_temporal_mesh_mamba.py`
- `docs/mamba_mesh_notes.md` (optional)

Forked GraphCast side:
- GraphCast module where grid->mesh encoding and processor handoff occur.

---

## Compatibility and Risk Notes

1. Pretrained compatibility:
- Adding temporal block changes internal activations.
- Use pretrained GraphCast weights for spatial core; temporal block random init.

2. Why previous long-history may degrade:
- Longer context without explicit temporal module often destabilizes rollout.
- Mesh-latent temporal module addresses this directly.

3. Main risks:
- wrong time-axis semantics at mesh stage;
- silent shape mismatches due xarray/jax wrappers;
- memory blow-up for larger `T`.

Mitigations:
- enforce shape assertions;
- start with small `T` and batch size 1;
- profile GPU memory after each integration step.

---

## Validation Checklist

1. `temporal-backbone=none` reproduces current baseline training behavior.
2. `temporal-backbone=mamba`, `T=2` trains without crash for 2k steps.
3. `T=4/6/8` runs complete at 10k+ steps.
4. MAE-vs-lead CSV/PNG generated for all runs.
5. Comparison table exported: `mean(lead1..24)`, `lead1`, `lead24`.

---

## Suggested Execution Order

1. Plumbing flags + no-op path.
2. Temporal block module + tests.
3. Mesh integration with switch.
4. Short smoke training (2k).
5. Long training (10k/20k).
6. Multi-seed.

---

## Definition of Done (v1)

Done means all conditions below hold:
- Code path switchable (`none` vs `mamba`).
- `T=2/4/6/8` training works.
- Fair MAE-vs-lead comparison is reproducible with one command path.
- Results and configs are stored with clear naming.

---

## Concrete Execution Plan

This section is the actual implementation order for this repo.

1. Preserve current baseline path.
- Keep `temporal-backbone=none` numerically identical to today's GraphCast path.
- Do not change existing checkpoints or eval scripts.

2. Add explicit temporal path in GraphCast internals.
- Today, time is folded into channels before `grid2mesh`.
- Add a second path that preserves time, so we can build
  `[time, num_grid_nodes, batch, channels]` before encoding to mesh.

3. Encode per time step to mesh latent.
- Reuse the same `grid2mesh_gnn` for each time step.
- Stack the outputs into `[time, num_mesh_nodes, batch, latent]`.

4. Insert temporal block in mesh latent space.
- v1 behavior for temporal path:
  - if backbone is `none`, reduce to last time step;
  - if backbone is `mamba`, run Mamba over time for each mesh node and return
    a `[num_mesh_nodes, batch, latent]` tensor.

5. Keep decoder contract unchanged.
- `mesh_gnn` still consumes `[num_mesh_nodes, batch, latent]`.
- `mesh2grid_gnn` and output formatting stay unchanged in v1.

6. Verify incrementally.
- verify old `none` path still imports/runs.
- verify time-preserving helper returns expected shapes.
- verify temporal path with last-step reduction runs end-to-end.
- only then replace reduction with real Mamba.

---

## Current Status

Implemented already:
- Training config flags for temporal experiments.
- Local vendored GraphCast source under `third_party/graphcast`.
- Temporal hook inserted between `grid2mesh` and `mesh_gnn`.
- Explicit time-preserving `grid -> mesh` path added for temporal mode.
- Minimal JAX/Haiku Mamba-style temporal block added and wired in.
- Smoke test passed through initialization and first train step for
  `temporal-backbone=mamba`, `input_duration=24h`.

Next coding steps:
1. Pass temporal hyperparameters cleanly through eval/inference paths.
2. Add small regression tests for temporal helper shapes.
3. Run controlled 200-step smoke comparisons for `none` vs `mamba`.
4. Measure GPU memory/runtime impact before longer runs.
5. Iterate on temporal block quality after baseline comparisons.
