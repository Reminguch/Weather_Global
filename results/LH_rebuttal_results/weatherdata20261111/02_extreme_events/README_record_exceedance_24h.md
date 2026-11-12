# Forecast errors on record-breaking weather

Both figures compare forecasts against ERA5 on the native global 1° grid. They use the same 122 starts at 00 UTC every three days from January 1 to December 30, 2023.

- `record_exceedance_24h.pdf` shows the 24-hour forecasts, valid from January 2 to December 31, 2023.
- `record_exceedance_240h.pdf` shows the 10-day forecasts, valid from January 11, 2023 to January 9, 2024. The three forecasts valid in January 2024 use the historical January records.

## Models

- **Public GraphCast** uses the public 1° GraphCast Small checkpoint and its own rollout.
- **GraphCast–Mamba (raw step 2,000)** uses the seed 22 carry/carry checkpoint `checkpoint_step00002000.pkl`, saved after 500 warm-up updates and 2,000 main-stage updates, before weight averaging.

## Historical records and errors

Each grid cell and calendar month has its own record, built from all six-hourly ERA5 values during 2015–2021. Heat and cold use the highest and lowest 2 m temperatures, respectively. Wind uses the highest 10 m wind speed, calculated from the two horizontal wind components. [Local monthly record definition](https://arxiv.org/html/2508.15724v1)

ERA5 at the forecast valid time determines which cells broke a record and by how much. For heat and wind, exceedance is truth minus the historical maximum. For cold, it is the historical minimum minus truth. The bins have edges at 0, 1, 2, 3, 4, 5, 6 and 8 K or m/s, with a final bin for exceedance of 8 or more. Only strictly positive exceedances count.

Both models are scored on the same grid-cell and forecast-start pairs, with equal weight per pair. The numbers above each column give the sample count for each bin. The top row shows RMSE, where lower is better. The bottom row shows mean forecast minus ERA5, where values closer to zero indicate less bias.
