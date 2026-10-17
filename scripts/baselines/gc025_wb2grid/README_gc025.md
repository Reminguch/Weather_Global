# GraphCast 0.25° scored on the 1.5° WB2 grid (July 2022 pipeline)

These are the scripts behind our July 2026 GC025 check. They are copied unchanged from
`/scratch/gpfs/DABANIN/lm8598/Weather_Global/fengwu/`, except that paths under `/home` were replaced
with repository-relative or scratch paths.

| file | role |
|---|---|
| `prefetch_ics.py` | July truth builder (runs where there is internet access). It writes `truth/<tag>_truth15.npy`: ERA5 0.25° Z500, T850 and Q700 at the 40 valid times, conservatively regridded to 1.5°. Its anchor list (`JSON` path) is the same as `manifest.json`. |
| `prefetch_gc025_ics.py` | Pulls the two 0.25° input frames per anchor from the WB2 ERA5 zarr into `gc025_ics/`. |
| `gc025_rollout_score.py` | Runs pure GraphCast inference for 40 × 6 h (no Mamba), conservatively regrids 0.25° to 1.5°, and computes cos-latitude-weighted RMSE against the truth files. It writes `gc025_scores_wb2grid.json` to the data directory. |
| `gc025_score.slurm` | Launcher. Run `sbatch -A <account> scripts/baselines/gc025_wb2grid/gc025_score.slurm [--smoke]` from the repository root. |
| `manifest.json` | The 32 anchors from 2022 (mixed 00/06/12/18 UTC), the same as the K=22 32-start evaluation. |

Inputs, truth and outputs live in `/scratch/gpfs/DABANIN/lm8598/Weather_Global/fengwu` (`FW` in the
scripts). The `dabanin` group can read that directory.

## Scope and limits

- **Weights:** `GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13`,
  the HRES-fine-tuned operational model, initialized here from ERA5. This is not necessarily the
  variant behind the public WB2 `graphcast` scores. Check this first when comparing with WB2.
- **Coverage:** only Z500, T850 and Q700 for the 32 anchors in 2022.
- **2023:** the WB2 ERA5 zarr used for ICs and truth stops at 2023-01-10. A 2023 run needs ICs and
  truth from ARCO-ERA5 (`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`)
  or another full 2023 0.25° source.

## July result

`gc025_scores_wb2grid.json`, Z500 RMSE (m²/s²) on 32 anchors:

| lead | this pipeline | WB2 public `graphcast` 2022, full year 00/12 UTC |
|---|---|---|
| day 1 | 44.7 | 39.0 |
| day 10 | 775.3 | 732.7 |

Our run also comes out higher than the public numbers at short leads. Its init set, init hours and
model variant all differ from WB2's, so the size of each contribution is untested.
