# Additional fifteen-year ablations

`additional_ablations_15yr.pdf` / `.png` combine the two author-supplied
figures into (a) spatial depth, (b) memory/training feedback, and (c) Mamba
dimensions. All three panels have the same plotting area, type sizes, axis
styling, and right-side legends. Horizontal axes use optimizer updates in
units of $10^3$; panel (c) displays only 0–10 in these units.
The original PNG and PDF files are preserved in this directory.
Rebuild the combined figure from this directory with:

```bash
python plot_ablation_15yr.py
```

The generator uses `../../paper_style.py`: on its 6.5-inch export canvas,
axis labels, ticks and legends are 9 points, and titles are
11 points (before scaling to the ICLR text width; see
`../../FIGURE_TYPOGRAPHY.md`). The combined figure is 6.3 inches high; wider right-side legend
areas accommodate the larger type while retaining equal-sized stacked panels.
The archived original figures retain their original formatting.

The generator needs only NumPy and Matplotlib. `ablation_15yr_data.json`
contains the plotted values, evaluation and training-config paths and hashes,
metric definition, 32 evaluation indices, and the comparison against the
original PDF paths. All 461 monitoring points retain their original values
and endpoints in the data export; the displayed range in panel (c) ends at
10,000 updates. Later additions to the training logs are not included.
The initial zero point is the zero-initialized correction, not a measured
validation result. The 26 SWA markers are reconstructed from the original
channel-RMSE summaries and are not smoothed or rescaled.

## Interpretation

- These are separate ERA5 2007–2021 runs with seed 22, 1° resolution,
  96-step segments, 24-step gradient windows, and closed-loop training
  feedback. In the reset controls both SSM state and convolution cache are
  reset during training and forecasting. Reset models retain the two-frame
  meteorological input; they are not memory-free in that broader sense.
- Monitoring curves use four forecast initializations. SWA markers use 32
  fixed 2022 initializations shared across these ablations, **different** from
  the principal paper's `paper_matched32` evaluation. Forecasts start with
  zero temporal memory and use full-prediction feedback for 40 six-hour steps.
- The metric is the approximate reconstructed GraphCast loss reduction:
  squared latitude-weighted channel RMSE, per-level difference normalization,
  pressure weights, and variable weights. It is neither RMSE reduction nor
  the main experiment's exact GraphCast loss. The old source `CAPTION.md`
  describes an obsolete loss-ratio figure and is not used as the caption.
- Panel (a) changes the number of spatial graph message-passing steps while
  retaining two temporal-module insertion positions, inner dimension 128,
  state dimension 16, learning rate 1e-4, and training tail K=20. Filled stars
  average updates 6k–11k, open stars 2k–8k; small gray stars are the two-step
  graph's seeds 23/24 under the 6k–11k average. The depth ranking changes with
  the averaging window, so this figure does not establish an optimal depth.
- Panel (b) retains all twelve combinations of training/forecasting
  memory policy, feedback tail, and learning rate. The stored `ar_tail_K=0`
  cases are displayed as K=1 (teacher forcing), following the paper's
  historical convention. K=20 counts twenty feedback-conditioned steps.
  Learning rate affects the memory comparison: at 1e-4, carry/carry and
  reset/reset give 22.7669% and 14.1239%; at 3e-4 they give 5.5081% and 9.6114%.
- Panel (c) retains six runs but labels actual `d_inner / d_state`
  dimensions. Historical S128 and H256i128 have identical architecture;
  I256 and H256 also share architecture. Parenthetical (1)/(2) distinguish
  runs, not different widths or independent random seeds. All have seed 22.
  The unused `temporal_hidden_size` flag did not set the module's input
  dimension (512 mesh features). The saved checkpoint parameter shapes
  were compared directly; `size_parameter_shapes.json` records that audit.
- SWA markers are plotted at averaging-window midpoints (8,500 and 5,000),
  not at single checkpoint steps. The 6k–11k SWA window in panel (c) is
  unchanged by the displayed 0–10k range. No significance or uncertainty
  interval is inferred from the displayed aggregate values.

## Historical label mapping for the size comparison

| Original run label | Displayed dimensions | Run label |
|---|---|---|
| S16 | 16 / 16 | — |
| S128 | 128 / 16 | (1) |
| H256i128 | 128 / 16 | (2) |
| I256 | 256 / 16 | (1) |
| H256 | 256 / 16 | (2) |
| N64 | 128 / 64 | — |
