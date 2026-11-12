# GraphCast–Mamba seed replication data

`data.json` contains the numerical values plotted in `forecast_skill_seed_replication_bd_std.pdf`, with panels labeled (a) and (b).

## Data and experiment

M1 denotes the selected 1° GraphCast–Mamba configuration from the [original paper](https://arxiv.org/abs/2610.28512v1). It combines a frozen GraphCast-small backbone with a trainable correction branch that carries temporal memory. We independently retrained it on ERA5 from 2015–2021 with seeds 22–26. Each run used 500 warm-up updates and 2,000 main-stage updates. Stochastic weight averaging used main-stage checkpoints at 500, 1,000, 1,500, and 2,000 updates.

Evaluation used 122 starts at 00 UTC every third day in 2023, with 40 six-hour leads through 240 hours. Memory starts at zero for each forecast and is carried between steps. Local evaluation used quality-controlled ERA5 verification fields. Published M1 values come from the archived paper curves.

## JSON fields

- `axes` lists forecast lead times in hours and days. All curve arrays use this order.
- `panels.a` contains exact GraphCast loss reductions. At each lead, losses are averaged over forecast starts before computing `100 × (1 − model / baseline)`.
- `panels.b` contains 2 m temperature RMSE reductions. Cosine-latitude-weighted squared errors are averaged over forecast starts before taking the square root and computing the reduction.
- Each local seed is scored against its paired GraphCast-small baseline. `mean_pct` and `sample_sd_pp` are the arithmetic mean and sample standard deviation across the five seed curves. The shaded band spans `mean_minus_1sd_pct` to `mean_plus_1sd_pct`.
- `pooled_40_step_graphcast_loss` contains the absolute losses annotated in panel (a). Each score averages the exact loss over 122 starts and 40 leads. The local M1 standard deviation is computed across the five pooled seed scores.
