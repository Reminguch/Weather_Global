# Weather_Global

This branch contains the GraphCast-based experiments for adding Mamba temporal
modeling at two different integration points:

- `mesh_post_encoder`: temporal mixing is applied after `grid2mesh` and before
  the original mesh processor.
- `mesh_processor_interleaved`: temporal mixing is applied inside the mesh
  processor, alternating with spatial message passing.

The implementation lives on branch `share/mamba-inside-mesh-v1`.

## Code Layout

- `scripts/training/train_graphcast.py`: main training entry point with temporal
  configuration flags.
- `scripts/analyze_models/mae_vs_lead.py`: rollout MAE evaluation.
- `src/models/temporal_mesh_mamba.py`: Mamba-style temporal block.
- `third_party/graphcast/graphcast/graphcast.py`: GraphCast integration.
- `scripts/training/inside_compare_2000.slurm`: inside-mesh comparison jobs.
- `scripts/training/inside_compare_hist2_2000.sh`: local hist2 comparison.
- `scripts/training/temporal_compare_2000.sh`: external temporal comparison.

## Shared Training Setup

Unless otherwise noted, all comparison runs use:

- Resolution: `2.0`
- Mesh size: `4`
- Width: `128`
- Processor message passing steps: `1`
- Batch size: `1`
- Precision: `bf16`
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- Train years: `2020-2021`
- Validation year: `2022`
- Evaluation metric files: `plots/analyze_models/nyc_mae_vs_lead_*.csv`

## External Temporal Results

These runs use `mesh_post_encoder`, where temporal mixing happens after
`grid2mesh` and before the original mesh processor.

| History | Model | Eval loss @500 | Eval loss @1000 | Eval loss @2000 | Mean MAE @2000 | Lead-1 MAE @2000 | Lead-24 MAE @2000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hist2 | none | 6.103 | 5.277 | 4.615 | 3.168 | 1.435 | 5.583 |
| hist2 | mamba | 6.229 | 5.405 | 4.769 | 5.746 | 1.370 | 6.714 |
| hist4 | none | 6.143 | 5.299 | 4.615 | 6.117 | 1.653 | 7.300 |
| hist4 | mamba | 6.272 | 5.412 | 4.768 | 5.962 | 1.416 | 5.410 |

Notes:

- For the external setup, MAE was reported from the final 2000-step checkpoint.
- The strongest final result in this group was `hist2 + none`.
- `hist4 + mamba` improved over `hist4 + none`, but did not beat `hist2 + none`.

## Inside-Mesh Interleaved Results

These runs use `mesh_processor_interleaved`, where each processor step
alternates between spatial message passing and temporal Mamba mixing on mesh
node latents.

| History | Model | Eval loss @500 | Eval loss @1000 | Eval loss @2000 | Mean MAE @500 | Mean MAE @1000 | Mean MAE @2000 | Lead-1 MAE @2000 | Lead-24 MAE @2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hist2 | none | 6.102 | 5.277 | 4.614 | 4.356 | 5.398 | 3.225 | 1.447 | 5.857 |
| hist2 | mamba | 6.254 | 5.437 | 4.799 | 4.670 | 5.217 | 7.616 | 1.349 | 10.892 |
| hist4 | none | 6.144 | 5.301 | 4.617 | 5.649 | 3.801 | 6.256 | 1.650 | 7.433 |
| hist4 | mamba | 6.300 | 5.424 | 4.778 | 8.119 | 3.715 | 5.965 | 1.522 | 8.038 |

Observations:

- Validation loss decreases smoothly for all runs.
- Rollout MAE is not monotonic with training step and does not always track
  validation loss.
- `hist2 + none` is still the strongest final inside-mesh result.
- `hist4 + mamba` showed its best behavior at the intermediate checkpoint
  (`step1000`), but the improvement was not sustained at `step2000`.

## Result Files

Selected result files:

- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist2_none_step500.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist2_none_step1000.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist2_none_step2000.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist2_mamba_step500.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist2_mamba_step1000.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist2_mamba_step2000.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist4_none_step500.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist4_none_step1000.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist4_none_step2000.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist4_mamba_step500.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist4_mamba_step1000.csv`
- `plots/analyze_models/nyc_mae_vs_lead_cmp2000_inside_hist4_mamba_step2000.csv`

## Current Conclusion

The inside-mesh temporal Mamba implementation is stable and trainable, but it
does not yet provide a consistent improvement over the baseline. The main
engineering result is that interleaving temporal Mamba inside the mesh
processor is feasible and reproducible on this branch.
