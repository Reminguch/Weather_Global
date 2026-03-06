# WB2 Local Staging (1.0deg, 13 levels)

Use this when compute nodes cannot access `storage.googleapis.com`.

## 1) Build/maintain yearly local dataset

```bash
python scripts/stage_wb2_era5_yearly_append.py
```

Defaults:
- source URI: WB2 ERA5 public Zarr
- output: `data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr`
- if output exists: append one next missing year (`max(existing)+1`)
- if output does not exist: bootstrap `1979..2021`

Common commands:

```bash
# bootstrap (or append next year if dataset already exists)
python scripts/stage_wb2_era5_yearly_append.py

# append explicit range (only missing years are fetched)
python scripts/stage_wb2_era5_yearly_append.py --start-year 1979 --end-year 2021

# plan only, no writes
python scripts/stage_wb2_era5_yearly_append.py --dry-run

# rebuild from scratch
python scripts/stage_wb2_era5_yearly_append.py --overwrite
```

## 2) Download trailing last 30 days (local eval dataset)

```bash
python scripts/download_wb2_last30d.py
```

Defaults:
- source URI: WB2 ERA5 public Zarr
- output: `data/graphcast/graphcast/dataset/wb2_res1_levels13_last30d.zarr`
- rolling window: `30` days (`6h` cadence)

Common commands:

```bash
# default rolling 30-day dataset
python scripts/download_wb2_last30d.py

# custom trailing window (e.g. 14 days)
python scripts/download_wb2_last30d.py --days 14 --output data/graphcast/graphcast/dataset/wb2_res1_levels13_last14d.zarr

# overwrite existing output
python scripts/download_wb2_last30d.py --overwrite
```

## 3) Train from local data

Fine-tuning (train years + trailing validation window):

```bash
python -u scripts/training/finetune_graphcast.py \
  --train-path data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --train-start-year 1979 \
  --train-end-year 2021 \
  --val-days 30
```

Res2 training (train on all local years except 2021, validate on 2021):

```bash
python -u scripts/training/train_graphcast_res2_stream.py \
  --data-path data/graphcast/graphcast/dataset/wb2_res1_levels13_1979_2021.zarr \
  --val-year 2021
```

## Notes

- All training scripts are local-only (no remote URI input).
- Yearly stager and last-30d downloader should be run in a network/proxy-enabled context.
- Avoid concurrent writer+trainer on the same Zarr path.
