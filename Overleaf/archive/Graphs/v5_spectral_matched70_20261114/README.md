# v5 spectral refresh on 70 matched starts

Figures 6 and 10–13, Table 1 and the active spectral discussion in both v5
sources use 70 initializations from January 1 through July 27, 2023, at all
40 six-hour leads. The same ordered starts are used for raw GraphCast,
GraphCast plus its 2022-fitted output filter, original M1 and matched R1.
This is a provisional subset of the standard 122-start annual protocol.
The user chose to update completed data now rather than wait for all 122.

The preceding 92-start campaign matched the GC/M1 spectra available on
November 13 (January 1–October 1). The remaining 30 October–December GC/M1
starts were queued separately; corresponding reset spectra were not part of
the 92-start campaign. No new forecasts, training, filter fitting or bootstrap
draws were performed for this refresh.

`summary.json` records exact sample IDs, model/source fingerprints, diagnostic
definitions, Table 1 scores and comparison values. The CSV and NPZ files retain
all plotted data, including all 83 channels at six hours, day 5 and day 10.
`independent_numeric_validation.json` checks all 48 exported metric values
against a separate calculation. `reproduction.json` confirms matching plotted
coordinates and layouts from these compact exports. `previous36/` preserves
the five replaced PDF assets; older numerical archives remain untouched.

Reproduce the five plots using Python with NumPy and Matplotlib (in the
Weather_global workspace, first source `scripts/graphcast_env.sh`):

```bash
python -B plot_v5_matched_spectra.py --export-source . --count 70 --output /tmp/v5-matched70-replot
```

Use a new output directory. This command uses only this archive's exports;
it does not need checkpoints, forecast data or the larger pooled moment files.
The historical rendering-math dependency is included beside the new script.
The manuscript retains one provisional-status note and the existing compact
layouts. No removed physical-power figures are restored.
