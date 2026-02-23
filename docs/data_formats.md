# GraphCast ERA5 dataset formats

Two dataset layouts exist; code that expects the **CDS NetCDF** layout can use the compatibility loader to read the **WB13 Zarr** as well.

## 1. CDS rolling NetCDF (`source-era5_cds_rolling-last30d_res-1.0_levels-13_steps-04.nc`)

- **Source:** `scripts/fetch_era5_rolling_window.py` (CDS API, rolling 30-day window).
- **Layout:**
  - **Batch dimension:** All variables have a leading `batch` dimension of size 1.
  - **Shapes:**
    - Surface: `(batch=1, time, lat, lon)` — e.g. `2m_temperature`, `mean_sea_level_pressure`, `land_sea_mask`, `geopotential_at_surface`.
    - Level: `(batch=1, time, level, lat, lon)` — e.g. `temperature`, `geopotential`, `u_component_of_wind`.
  - **Coordinates:** `batch`, `time` (datetime64), `lat`, `lon`, `level`.
  - **Static 2D:** Stored as `(1, time, lat, lon)` (repeated per time step).
- **Typical sizes:** `time=117` (6-hourly), `lat=181`, `lon=360`, `level=13`.

## 2. WB13 Zarr (`source-era5_wb13_latest-1y_res-1.0_levels-13_steps-all.zarr`)

- **Source:** `scripts/download_latest_era5.py` (WeatherBench2 wb13, downsampled to 1°).
- **Layout:**
  - **No batch dimension.** Variables are time-major or static 2D.
  - **Shapes:**
    - Surface: `(time, lat, lon)` — e.g. `2m_temperature`, `mean_sea_level_pressure`.
    - Level: `(time, level, lat, lon)`.
    - Static 2D: `(lat, lon)` — `land_sea_mask`, `geopotential_at_surface`.
  - **Coordinates:** `time` (hours since 1959-01-01, int64), `lat`, `lon`, `level`.
- **Typical sizes:** `time=1461` (e.g. 1 year of 6-hourly), `lat=181`, `lon=360`, `level=13`.

## Differences (summary)

| Aspect        | CDS NetCDF              | WB13 Zarr                 |
|---------------|-------------------------|----------------------------|
| Batch dim     | Yes `(1, time, ...)`    | No `(time, ...)`           |
| Static 2D     | `(1, time, lat, lon)`   | `(lat, lon)`               |
| Time encoding | datetime64              | hours since epoch (int64)  |
| File format   | NetCDF4                 | Zarr (directory store)     |

## Making Zarr compatible with the NetCDF layout

Use the loader in `src/data/graphcast_dataset.py`:

- **`open_graphcast_era5(path, time_slice=None)`**  
  Opens either a `.nc` or `.zarr` path and returns an xarray `Dataset` in the **CDS-compatible** layout:
  - Adds a `batch` dimension of size 1 to all variables.
  - Expands static 2D variables `(lat, lon)` to `(batch=1, time, lat, lon)` by broadcasting along the dataset’s `time` coordinate.
  - Optionally restricts to a time window via `time_slice` (e.g. `slice(-117, None)` for the last 117 steps).

This allows the same downstream code (e.g. exploration notebooks, training data iteration) to work on both datasets.
