# K=22 common-grid evaluation on arbitrary initializations (e.g. 2023)

This branch is `AR-Training-Lianghong-v22clean-frozen` plus three changes:

- `src/data/` (`graphcast_dataset.py`, `contracts.py`), which the frozen branch imported but never committed.
- `eval_v22_clean_wb2grid.py` gains `--init-range START END EVERY_HOURS` and `--per-init-npz`. The full-year flags (`--all-val`, `--idx-stride`, `--shard`) are included too.
- `slurm_pilots/v22cl_K22_wb2grid_initrange.slurm`: a one-command launcher.

Run the archived v22 checkpoints from this tree only. The current main tree has drifted. The same
checkpoint run there gives about 14.86K instead of 2.80K full RMSE at 240 h.

## Run

```bash
cd <this checkout>
sbatch -A <account> --export=ALL,DATA=/path/era5_1deg_2023.zarr,INIT_START=2023-01-01T00,INIT_END=2023-12-30T00,EVERY_H=72,OUT=/path/out/K22_2023 \
    slurm_pilots/v22cl_K22_wb2grid_initrange.slurm
```

- The init time is the last input frame (inputs at t-6 h and t).
- `DATA` must be a continuous 6-hourly, 1°, 13-level zarr. It must have the same variables as
  `wb2_res1_levels13_2015_2022.zarr` and cover `INIT_START - 6 h` to `INIT_END + 240 h`. For the
  122 inits above, that is 2022-12-31T18 to 2024-01-09T00. The script stops and lists any missing steps.
- The checkpoint, GraphCast-small parameters, normalization stats and Python environment all
  default to readable paths under `/scratch/gpfs/DABANIN/lm8598`.
- About 50 s per init on one 80 GB GPU (122 inits take roughly 2 h).

## Outputs

- `${OUT}.json`: RMSE, MAE and bias per variable, channel and lead. It has the same format as the
  2022 full-year shards behind the current common-grid figures, plus `init_times`.
- `${OUT}_per_init.npz`: `mse_baseline` and `mse_full`, each with shape (n_init, 40, n_channel).
  These are cos-latitude-weighted MSE on the 240x121 grid. The file also holds `init_times`,
  `channels` and `lead_hours`. RMSE = sqrt(mean over inits of MSE), which is how the JSON aggregates.

Forecasts and truth are both regridded conservatively from 1° to 1.5° before scoring, so the truth is
ERA5 at 1° regridded to 1.5°. Score GC025 against the same truth for a like-for-like comparison.

## Reproduction check (2022)

`INIT_START=INIT_END=2022-01-01T06` on `wb2_res1_levels13_2015_2022.zarr` re-runs the first init
of the July 2022 full-year run. The batches this branch builds (inputs, targets, forcings) are
bit-identical to the July code path. Results still differ slightly between GPU runs, by about
0.001 K at 6 h and about 2% at 240 h for this single init. This holds for the pure GraphCast
baseline too, so it comes from run-to-run GPU numerics. Expect these T2m diagnostic values, roughly:

```
k= 0  base RMSE ~0.43   full RMSE ~0.42
k=39  base RMSE ~2.66-2.74   full RMSE ~2.88-2.91
```

Values far outside this range (for example, full RMSE in the thousands) mean the wrong code tree is in use.

The July full-year run used inits at 06/18 UTC. WB2's GraphCast and ENS scores use 00/12 UTC, and
that offset causes the 12 h zigzag in our curves in the current Figures 5 and 9.
