# Experiment Slurms

This directory keeps one-off and provenance-heavy Slurm entrypoints out of the
shared training and analysis script folders.

Use the repository environment for Python, training, preprocessing, and analysis
commands:

```bash
source scripts/graphcast_env.sh
```

## Layout

- `active/`: current or recently submitted experiment entrypoints that may need
  resubmission, monitoring, or continuation.
- `smoke/`: short validation jobs for model families and larger experiment
  templates.
- `archive/`: completed or superseded experiment definitions kept for
  reproducibility.

Do not delete archived Slurms only because the runs finished. A Slurm often
records the exact array geometry, checkpoint roots, run naming, and resource
budget that generated artifacts under `artifacts/checkpoints` or
`plots/analyze_models`.

## Active

| Script | Purpose | Main outputs |
| --- | --- | --- |
| `active/7y_mp6_vanilla_continue20k.slurm` | Continue matched 7-year MP6 vanilla baselines for another 20k steps. | `artifacts/checkpoints/7_years/vanilla_gc_mp6_continue20k` |
| `active/7y_mp6_mamba_frozen_sweep_20k.slurm` | Frozen GC-Mamba/residual-Mamba sweep from the MP6 vanilla continuation baselines. | `artifacts/checkpoints/7_years/mamba_frozen_from_vanilla_mp6_20k` |
| `active/submit_7y_mp6_mamba_frozen_sweep_20k.sh` | Submit the MP6 frozen Mamba sweep in per-baseline chunks with tuned resources. | Slurm job arrays |
| `active/7y_small_vanilla_w256_mp2_res236_100k.slurm` | Initial 7-year small vanilla W256 MP2 training for res2/res3/res6. | `artifacts/checkpoints/7_years/small_experiments` |
| `active/7y_small_vanilla_w256_mp2_continue_to_200k.slurm` | Continue the small vanilla runs from 150k to 200k. | `artifacts/checkpoints/7_years/small_experiments` |
| `active/7y_small_mamba_from_200k_staged.slurm` | Staged frozen/release Mamba runs from the 200k small vanilla checkpoints. | `artifacts/checkpoints/7_years/small_mamba_frozen_from_vanilla_200k_50k`, `artifacts/checkpoints/7_years/small_mamba_release_from_frozen50k_20k` |
| `active/submit_7y_small_200k_mamba_chain.sh` | Submit the small vanilla continuation and dependent staged Mamba jobs. | Slurm dependency chain |

## Smoke

| Script | Purpose |
| --- | --- |
| `smoke/vanilla_graphcast_res4_m4_w128_mp1_smoke.slurm` | Vanilla GraphCast res4 smoke test. |
| `smoke/gc_mamba_res4_m4_w128_mp1_dh16_ls32_smoke.slurm` | GC-Mamba res4 smoke test. |
| `smoke/residual_mamba_res4_m4_w128_mp1_dh16_ls32_smoke.slurm` | Residual-Mamba res4 smoke test. |
| `smoke/vanilla_graphcast_8y_w512_mp8_smoke.slurm` | Smoke version of the 8-year W512 MP8 vanilla pilot. |
| `smoke/graphcast_segments_8y_w512_mp8_smoke.slurm` | Smoke version of the 8-year W512 MP8 segment-training pilot. |

## Archive

| Folder | Contents | Why keep it |
| --- | --- | --- |
| `archive/2026-05_8y_w512_pilots/` | 8-year W512 vanilla and segment-training MP sweeps. | Provenance for pilot checkpoint roots and MP8 shortcut scripts. |
| `archive/2026-05_gc_mamba_sweeps/` | Early GC-Mamba resolution/mesh sweeps and Mamba-from-vanilla sweeps. | Records array mapping and continuation setup for older Mamba comparisons. |
| `archive/2026-05_resolution_grid/` | Vanilla W512 MP2/4/6 resolution-grid stream run. | Provenance for resolution-grid baseline artifacts. |

## Cleanup Rules

Before deleting a Slurm, check whether it produced or explains any checkpoint,
plot, log, or failed-run resource lesson. Prefer moving stale scripts into
`archive/YYYY-MM_topic/` with a README row over deleting them.

Deletion is reasonable when a script is a duplicate draft, never produced
artifacts, and is superseded by a documented script in this tree.
