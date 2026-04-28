# MZ-residual Mamba — v2 (Mod G + Mod A + Mod C)

Snapshot of the v2 codebase as of 2026-04-28. v1 paper-chain code under
`src/models/mz/full_mamba/` and `scripts/training/full_mamba/` is **not**
removed; this folder is a clean snapshot of v2-specific files for
auditability and reproducibility.

## Why v2

Per-variable diagnosis on v1 K-curriculum (see
`results/2026-04-27_train_eval_curves/per_variable_eval_trend.png`) showed:

1. **K=2/K=4/K=6 phases plateau within their own phase** — the
   apparent monotonic Δ% improvement is largely a horizon-shift artifact
   of changing eval h with K, not real per-step learning.
2. **Surface variables (MSLP, 10m winds, precip) degrade as K↑**, with
   MSLP collapsing from +1.31% (K=1) to -1.02% (K=6) under TF and
   to -4.46% under AR.
3. **Only K=1 and K=8 show real within-phase eval improvement**;
   the SSM memory hypothesised by the MZ formalism is not getting a clean
   training signal at intermediate K.

v2 introduces three orthogonal modifications to address the above:

### Mod G — specialist heads
Split the final residual head into two parallel `hk.Linear` projections —
one for the 78 upper-air channels (Z/T/q/u/v/w × 13 levels), one for the
5 surface channels (MSLP, 2m_T, 10m_u, 10m_v, precip). Surface gradients
no longer pull on upper-air capacity (and vice versa) at the output
layer. Implementation uses `concat + jnp.take` (one gather), not two
`.at[].set()` scatters, to keep GPU throughput close to v1 single-head.

### Mod A — anchor-as-batch
v1 K>1 flattens `S` anchors × `K` rollout steps into one Mamba
sequence of length `T = S·K`, which corrupts the SSM time axis with
repeated and reversed physical times across anchor boundaries. v2
transposes to `[T=K, B=S, lat, lon, F]` so each anchor is its own batch
sample and the SSM sees a clean monotonic K-step time series within
each anchor.

**Patch 1: K-aware fallback (`--anchor-as-batch-min-k`, default 2).**
At K=1 the legacy layout has a side-benefit of cross-anchor SSM hidden
state propagation (a "cycling regime" since the 16 anchors are
themselves on a 6h grid and SSM sees `[6h, 12h, ..., 96h]` monotonic).
Forcing `T=1, B=16` at K=1 cuts that signal without compensation. So
v2 only applies anchor-as-batch when `target_steps >= min_k`. K=1 falls
back to legacy.

### Mod C — lead-time weighted loss
`alpha_s = (s+1)^p / mean`. p=0 uniform (default, equivalent to v1).
p=1 linear, p≥1.5 emphasises long horizons. Only meaningful when Mod A
is effective (anchor-as-batch makes the time axis a clean lead axis).

### Patch 2 — `allow_tf_at_t0` in `rollout_ar`
Under anchor-as-batch every anchor's first step lands at `t=0`, where
v1 unconditionally used `prev_residual=0`. That suppressed the
`tf_mask[0]=1` truth-injection that legacy K>1 only ever applied at
intra-segment anchor starts (`t > 0`). Patch 2 adds an `allow_tf_at_t0`
flag (default False = v1 parity); v2 sets it to `effective_anchor_as_batch`
so each anchor's first step actually uses the observable previous-anchor
residual.

## Three Guards in `parse_args` (Patch 1+2 hardening)

1. `--specialist-heads` requires `--full-mamba`, `--meshed`,
   `--full-variables`, and is incompatible with `--atmos-only`.
2. Channel index split must be a clean permutation of 0..F-1 (no
   overlap, no gap).
3. Resuming a v1 single-`residual_head` ckpt into a v2 specialist-heads
   model raises unless `--allow-partial-resume` is also passed; warns
   that upper_head/surface_head will fresh zero-init.

## Folder layout

```
v2/
├── README.md                                (this file)
├── src/
│   └── mz_full_mamba_meshed_v2.py           (modified model: Mod G + Patch 2)
├── scripts/
│   ├── training/
│   │   └── train_mz_fullmamba_v2.py         (modified train script)
│   └── plots/
│       ├── plot_v1_vs_v2_K1.py
│       ├── plot_per_variable_eval_trend.py
│       └── plot_train_eval_curves.py
└── slurm/
    ├── mz_fullmamba_v2_GA_K1_4k.slurm        (real run)
    ├── mz_fullmamba_v2_GA_K2_8k.slurm        (chained on K=1)
    ├── mz_fullmamba_v2_GA_K1_smoke.slurm     (10-step bug-check)
    ├── mz_fullmamba_v2_GA_K2_smoke.slurm     (10-step bug-check)
    ├── mz_fullmamba_v2_smoke_v1parity.slurm  (legacy-equivalent smoke)
    ├── mz_fullmamba_v2_smoke_specialist.slurm
    └── mz_fullmamba_v2_smoke_anchorbatch.slurm
```

## How to use

The snapshot files in `v2/` are **for auditability, not direct
execution**. The actual live working copies are:

- `src/models/mz/full_mamba/mz_full_mamba_meshed.py` (v2 changes
  applied; flags default to v1 behaviour, so v1 paper-chain still works)
- `scripts/training/full_mamba/train_mz_fullmamba_v2.py`
- SLURM scripts in `/home/lm8598/Fermihubbardnumeric/slurm_pilots/`

### Recommended K-curriculum chain

```
v1-parity smoke   no --specialist-heads, no --anchor-as-batch
v2-G K=1 4k       --specialist-heads (no --anchor-as-batch)
v2-GA K=1 4k      --specialist-heads --anchor-as-batch
                  (effective_anchor_as_batch=False at K=1, Patch 1 fallback)
v2-GA K=2 4k      --specialist-heads --anchor-as-batch
                  resumed from v2-GA K=1 ckpt
                  (effective=True, allow_tf_at_t0=True at K>=2)
v2-GA K=4 4k      same flags + --target-steps 4 --segment-steps 8
v2-GAC K=4 4k     + --loss-lead-weight-power 1.0
```

## Status of the experimental chain

As of 2026-04-28:

| Run | flags | smoke | real | finding |
|---|---|---|---|---|
| v1-parity | none | ✓ pass (0 effective change) | n/a | v2 == v1 when flags off |
| v2-G smoke | --specialist-heads | ✓ pass (0 numerical change at 10 steps, expected for zero-init) | not yet | matches Run 0 to 1e-5 |
| v2-A smoke | --anchor-as-batch (min_k=1) | ✓ pass (loss differs by 1e-5, no NaN) | not yet | shape correct |
| v2-GA K=1 (pre-Patch-1+2) | both, no min_k | n/a | TIMEOUT @ step 2200 | trail v1 by 0.08pp Δ% (root-caused: Mod A cuts cross-anchor SSM at K=1) |
| **v2-GA K=1 (with Patch 1)** | + min_k=2 | pending | pending | predicted ≈ v1-parity at K=1 (fallback active) |
| **v2-GA K=2** | with Patch 1+2 | pending | pending | first real test of Mod A unlock memory |

## Reference

Per-variable trajectory plot: `results/2026-04-27_train_eval_curves/per_variable_eval_trend.png`
v1 vs v2 K=1 head-to-head: `results/2026-04-27_train_eval_curves/v1_vs_v2_K1.png`
Technical note: `results/2026-04-27_mz_mamba_technical_note.pdf`
