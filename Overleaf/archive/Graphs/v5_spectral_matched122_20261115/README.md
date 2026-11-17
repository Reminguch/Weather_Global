# v5 spectra on the complete 122-start evaluation

Figures 6 and 10–13 and the numerical exports here use all 122 forecast
initializations at 00 UTC every third day from January 1 through December 30,
2023, with 40 six-hour leads through day 10. Raw GraphCast, GraphCast with its
2022-fitted output filter, original M1, and matched R1 use the same ordered
dates and ERA5 truth. Both correction models retain their unfiltered outputs.

The completed GC/M1 optimal-filter report supplies its annual pooled moments.
The completed matched-reset report combines the original 92 starts with the
30 October–December starts. `summary.json` and `manifest.json` pin these inputs,
record the exact cohort and metric definitions, and retain numerical warnings.
The earlier 70-start archive remains unchanged in the neighboring directory.

At day 5, M1 has 2.85% lower exact GraphCast loss than R1, with aggregate
power mismatch 69.60% versus 69.10%. Across the ten-day rollout, the M1 loss
advantage over R1 is 5.76%, and at day 10 it is 9.26%. These are fixed-checkpoint
point estimates; no new confidence intervals or training-seed estimates were
computed. The separate principal memory evaluation and its intervals retain
their original numerical results.

`independent_numeric_validation.json` verifies 48 exported metric values
directly from the original pooled arrays. `reproduction.json` records exact
reproduction of plotted coordinates and layouts from the compact exports.
All five generated figures were visually inspected. No forecasts, filter fits,
training, or bootstrap draws were run for this manuscript refresh.

The reset rerun's pooled loss is 10.359790727, compared with the historical
10.359537610 (approximately 0.00244% higher). Its maximum individual relative
difference is 3.29457%, exceeding the recorded 0.1% reference bound. These
finite differences remain warnings, preserved in the summary diagnostics.

Reproduce the five plots using Python, NumPy, and Matplotlib:

```bash
python -B plot_v5_matched_spectra.py --export-source . --count 122 --output /tmp/v5-matched122-replot
```

Use a new output directory. Within Weather_global, first source
`scripts/graphcast_env.sh`. The command uses this archive's compact exports and
included rendering dependency, without checkpoints or original forecast files.
The active manuscript figure names end in `_v5_matched122.pdf`.
