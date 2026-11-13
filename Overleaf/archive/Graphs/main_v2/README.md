Figures for `Overleaf/main_v2.tex`, rebuilt exclusively from saved evaluation artifacts.

From the repository root:

```bash
source scripts/graphcast_env.sh
python Overleaf/Graphs/main_v2/generate_figures.py
```

Add `--preview-dir /tmp/main_v2_figures` to create PNG previews. No evaluation or Slurm jobs are launched.

Both generators use `../paper_style.py`: axis labels are 9 pt, ticks and
legends 9 pt, and panel titles 11 pt on the 6.5-inch export canvas. See
`../FIGURE_TYPOGRAPHY.md` for sizes after scaling to the ICLR text width.
The layouts reserve room for the enlarged text; long labels and titles wrap,
and the six-panel memory comparison uses a shared forecast-lead label.

- `exact_gc_loss.pdf`: blue curves show the threefold Mamba-learning-rate configuration with SWA, `di16_fast_mamba`, SWA 500–2000; dashed gray curves in the absolute-error panels show its paired GraphCast baseline. The legend identifies the models as `GraphCast` and `GraphCast–Mamba`, and the footer specifies inner dimension 16, one B,C group, and the SWA window. It has the highest aggregate and day-ten exact-loss reductions among the reference and completed SWA configurations in the frozen optimizer and learning-rate reports (16.3434% and 23.6831%). The top row shows exact original GraphCast loss and its reduction on 32 matched cold starts with zero initial memory. The bottom row shows 2 m temperature RMSE (K) and RMSE reduction from `per_variable_per_step.2m_temperature`. Temperature RMSE uses cosine-latitude weighting and averages squared errors across starts before taking the square root. The displayed baseline and all reductions use the same saved evaluation. The paper's main table compares inner dimensions 16 and 32 under joint and threefold Mamba learning rates, with the full learning-rate sweep in the appendix.
- `memory_reset.pdf`: a single panel comparing carry/carry, carry/reset, and reset/reset state policies for di16 with BCG1, SWA 2k–8k. Carry/carry has the largest day-ten reduction (22.5235% versus 9.2419% and 12.9072%). The metric is explicitly approximate because it is reconstructed from saved channel RMSE. Negative values are retained in the displayed range.
- `memory_reset_all.pdf`: the same comparison for BCG1 and BCG4, both with di16, across SWA 2k–8k, 4k–10k, and 6k–12k. These overlapping checkpoint windows are not independent training repetitions.
- `optimizer_sensitivity.pdf`: the frozen optimizer report's 22 available SWA day-10 exact-loss reductions. Asterisks identify SWA 4k–8k instead of 2k–8k. Gray dash cells mean unavailable in that report, not zero improvement. Rows distinguish initialization and di/BCG; columns distinguish learning rate and optimizer moments.

`figure_sources.json` records hashes and protocol metadata. The generator checks the exact loss ratios, the selected model's 40-lead temperature RMSE reductions against its paired RMSEs, all completed SWA entries in both reports against evaluation JSON, the selected configuration's ranking, and all 720 memory-curve values against the existing repository reconstruction function and normalization statistics. It also verifies matched sample indices and both reports' archived source hashes. No confidence intervals or significance claims can be inferred from these aggregate artifacts alone.

The training-horizon appendix has a separate generator:

```bash
source scripts/graphcast_env.sh
python Overleaf/Graphs/main_v2/generate_appendix_horizon_figures.py
```

- `training_ar_horizon.pdf`: standalone archived v22 K-sweep, reconstructed from `results/v22/eval_jsons/`. The historical paper-formula weighted MSE uses variable-level RMSE and mean-level difference standard deviations. It is explicitly approximate and differs from the exact original GraphCast loss. Each K uses its saved baseline; forecast-start indices are absent from these JSONs, and their baseline RMSEs differ slightly.
- Both training-tail figures use the label `GraphCast loss reduction (%)` and the vertical range -12 to 28. The open-loop plot retains the curves from `results/2026-05-23-v22/plots/v22_vs_baseline.png`. Both plots mark the K=20 training-tail duration with a vertical dashed line at 20 × 6 = 120 h (5 days); the previous horizontal day-10 loss reference has been removed. `training_ar_horizon.reference.json` records the duration marker. This is the forecast-feedback tail for K=20, not the full training window or retained memory history; other K curves have their own tail durations. The sweeps differ in experimental settings and loss reconstruction, as stated in the v3 caption and Appendix A.2.
- `channels_improved_k22.pdf`: count of the 83 channels with strictly lower latitude-weighted RMSE than the paired GraphCast baseline for K22. The day-ten count is 82; all 11 variable groups improve at that lead. Counts describe the signs of aggregate RMSE differences, without a significance test.

`appendix_horizon_sources.json` records the source hashes, historical formula, per-K baseline differences, plotted values, channel counts, and day-ten variable-group reductions. The generator accepts `--preview-dir /tmp/main_v2_appendix_figures` for PNG previews.
