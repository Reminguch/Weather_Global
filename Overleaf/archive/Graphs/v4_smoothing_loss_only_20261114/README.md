# Single-panel smoothing loss figure

The user requested removing the right-hand "Loss reduction vs raw GC" panel
from the smoothing-accuracy figure. That figure was formerly numbered 13 and
is now Figure 12 following conversion of the earlier summary chart to Table 2.
Only the absolute exact-GraphCast-loss axis is retained, with all four models,
the same 36 dates, all forty leads, original colors and five-day training-tail
marker. All 160 plotted values match the frozen source CSV exactly.

`loss_by_lead.csv` is a byte-identical copy of the fixed 36-date numerical
export. `manifest.json` pins it, the plotting source and PDF/PNG outputs.
The original two-panel figures and fixed data archives remain unchanged.
The active PDF is `arxiv_pics/fig12_smoothing_loss_v4_matched36.pdf`.
The canvas remains 6.25 by 2.65 inches and the manuscript uses full linewidth.

To reproduce with NumPy and Matplotlib, run from this directory:

```sh
python plot_v4_smoothing_loss_only_20261114.py --data loss_by_lead.csv --output /tmp/smoothing-loss-only
```

No forecasts, filters, loss estimates or uncertainty calculations were rerun.
