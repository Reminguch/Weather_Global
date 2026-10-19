# Strict >30 hPa residual-loss experiment

This is an isolated ablation requested on 2026-10-02. No changes are made to existing frozen experiments.

- All seven ERA5 fields train only on the 29 pressure levels greater than 30 hPa. The eight excluded levels are sliced out before every accuracy, spectrum and bias calculation. Their loss coefficients and output cotangents are exactly zero.
- Native model-state objectives are disabled because sigma-level losses cover the full column and do not implement the same fixed pressure mask. This is a data-only ablation, not a change to the physical solver or its state.
- The solver still advances the full atmosphere, and shared residual parameters may indirectly change upper levels. Zero upper loss does not freeze upper atmospheric state.
- Two fresh runs use the same initialization, seed 22, K=2, width 128, d_inner 16, batch 2, origins, optimizer and 2,000-update budget. Calibrated uses fixed train-only scales and field priorities (main fields 1, each cloud 0.05). Adaptive adds the existing EMA-window controller, window 100, probe every 20 updates, bounds 0.25–4. Bounds remain a possible limitation and are logged, not assumed solved.
- Calibration uses only selected layers. Means divide by 29 included layers. Relative to the prior experiment this removes both upper-pressure and native objectives, so the difference cannot be attributed to upper pressure alone.
- Model selection averages seven physical field RMSE ratios, aggregated over the selected levels and 6/12-hour leads. Every field must be within 5% of baseline; step zero is included. This gate does not guarantee no regression.
- All seven variables and all 37 levels remain in validation and independent 120-hour evaluation. Report the selected levels separately and use training compute time on plots as requested.
- Smoke gates: independent real-data pilot, exact zero excluded-output cotangents, invariance of loss and parameter gradients to excluded-output perturbations, explicit dynamic-coefficient JIT test, finite real updates, frozen backbone, exact separate-process resume including controller and optimizer.
- H200 jobs use ailab / gpu-test and one-hour allocations, with checkpoint/requeue near 55 minutes. Pilot and production artifacts remain separate on scratch.
