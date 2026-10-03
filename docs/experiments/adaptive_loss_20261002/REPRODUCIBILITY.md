# Evidence and reproduction

`history/` retains every saved short validation, all 7 fields x 37 levels x 2 leads, and cumulative optimizer-step compute seconds. `results/` retains every evaluated immutable checkpoint at 6–120 hours. `baselines/` retains the corresponding frozen-model reference. Origin lists, numerical source hashes, resource hashes, statistics hashes and checkpoint hashes are in these artifacts. No large model checkpoint or training dataset is committed.

`variable_improvement.csv` provides field aggregates. `pressure_level_detail.csv` provides all short-validation levels, not just favorable cells. `adaptive_probe_weights.csv` contains all recorded controller probes. It reports parameter-gradient norms, not Adam-update shares. Aggregation uses mean MSE before square root, without training loss weights. The 0.1% counting tolerance is not a significance test.

Plots can be regenerated with NumPy and Matplotlib using `python code/live_eval/report.py --root .` from this directory. This renderer copy resolves baseline paths relative to the package; numerical evaluation is unchanged. Running it regenerates REPORT.md, so preserve the authored DIAGNOSIS.md and publication introduction if needed.

`code/adaptive_loss/` copies the original frozen experiment implementation. Full runtime dependencies remain in Weather_Global and are identified by the per-file hashes; this is not a self-contained distribution of the dataset/model runtime. `code/weather_only/` documents the new, separately frozen strict-mask ablation. Smoke evidence for the completed original experiment is under `smoke/`; new strict-mask smoke evidence is separately retained under `weather_only/preflight_v1` and `weather_only/lifecycle_v1`; both passed before production submission.

The 16 long-evaluation origins do not overlap the 8 selection origins. Some were observed in prior pilot experiments, so this is validation rather than a pristine final test set. One seed and these sample sizes do not establish significance. The independent 2023 test split was not used.
