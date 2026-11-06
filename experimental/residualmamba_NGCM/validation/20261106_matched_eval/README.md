# Matched evaluation validation

Eight CPU regression tests passed. A separate aggregation regression on all 15 original K20 segments reproduces the existing total, baseline and per-step losses exactly. They cover selecting the actual training loader, the exact K1/K20 origin/lead conventions, prefix-plus-tail aggregation, and rejection of mixed checkpoints/protocols, incomplete segments and nonfinite contributions.

The first H200 real-batch check passed. Baseline contributions from the compiled scan and direct per-step computation agree exactly (maximum absolute difference 0). Full checkpoint evaluation follows this check; the smoke loss is not a complete validation result.

All jobs use ailab, gpu-test, one H200 and a one-hour limit. The controller runs at most three jobs, resumes completed segments after a soft deadline, regenerates separate matched curves after each checkpoint, and publishes the complete report automatically. The frozen original training source and checkpoints are not modified.

Primary results: [matched evaluation report](../../results/v2_matched_20261106/README.md).
