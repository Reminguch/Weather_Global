Figures for `Overleaf/main_v2.tex`, rebuilt exclusively from saved evaluation artifacts.

From the repository root:

```bash
source scripts/graphcast_env.sh
python Overleaf/Graphs/main_v2/generate_figures.py
```

Add `--preview-dir /tmp/main_v2_figures` to create PNG previews. No evaluation or Slurm jobs are launched.

- `exact_gc_loss.pdf`: blue curves show the threefold Mamba-learning-rate configuration with SWA, `di16_fast_mamba`, SWA 500–2000; dashed gray curves in the absolute-error panels show its paired GraphCast baseline. The legend identifies the training method as `3× Mamba LR + SWA`, and the footer specifies inner dimension 16, one B,C group, and the SWA window. It has the highest aggregate and day-ten exact-loss reductions among the reference and completed SWA configurations in the frozen optimizer and learning-rate reports (16.3434% and 23.6831%). The top row shows exact original GraphCast loss and its reduction on 32 matched cold starts with zero initial memory. The bottom row shows 2 m temperature RMSE (K) and RMSE reduction from `per_variable_per_step.2m_temperature`. Temperature RMSE uses cosine-latitude weighting and averages squared errors across starts before taking the square root. The displayed baseline and all reductions use the same saved evaluation. The paper's main table compares inner dimensions 16 and 32 under joint and threefold Mamba learning rates, with the full learning-rate sweep in the appendix.
- `memory_reset.pdf`: a single panel comparing carry/carry, carry/reset, and reset/reset state policies for di16 with BCG1, SWA 2k–8k. Carry/carry has the largest day-ten reduction (22.5235% versus 9.2419% and 12.9072%). The metric is explicitly approximate because it is reconstructed from saved channel RMSE. Negative values are retained in the displayed range.
- `memory_reset_all.pdf`: the same comparison for BCG1 and BCG4, both with di16, across SWA 2k–8k, 4k–10k, and 6k–12k. These overlapping checkpoint windows are not independent training repetitions.
- `optimizer_sensitivity.pdf`: the frozen optimizer report's 22 available SWA day-10 exact-loss reductions. Asterisks identify SWA 4k–8k instead of 2k–8k. Gray dash cells mean unavailable in that report, not zero improvement. Rows distinguish initialization and di/BCG; columns distinguish learning rate and optimizer moments.

`figure_sources.json` records hashes and protocol metadata. The generator checks the exact loss ratios, the selected model's 40-lead temperature RMSE reductions against its paired RMSEs, all completed SWA entries in both reports against evaluation JSON, the selected configuration's ranking, and all 720 memory-curve values against the existing repository reconstruction function and normalization statistics. It also verifies matched sample indices and both reports' archived source hashes. No confidence intervals or significance claims can be inferred from these aggregate artifacts alone.
