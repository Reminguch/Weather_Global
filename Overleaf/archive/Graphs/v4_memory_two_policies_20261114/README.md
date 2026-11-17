# Figure 6: matched memory policies

November 14, 2026 presentation revision: carry/carry versus separately trained
reset/reset. The original matched SWA panel reuses paired date-bootstrap
intervals; the independent raw-checkpoint panel reuses five carry seeds and
one reset seed, with the saved carry mean and sample SD. Both panels retain
their own common GraphCast baseline and distinct verification provenance.

Every retained curve/interval is unchanged from the November 13 exports in
`../v4_seed_uncertainty_20261113/`. The mean/SD CSV is copied byte-for-byte.
The excluded carry/reset diagnostic, original figure and full numerical exports
remain in their historical locations. No forecasts, training, resampling or
new numerical estimates were performed for this presentation change.

The manuscript uses `arxiv_pics/fig6_memory_policies_v4_matched_only.pdf` at
80% text width, without a colored revision frame. The exported plot retains
its 6 x 4.05 inch canvas; its common y limits cover all retained curves/bands.
Reproduce from the parent Weather_global workspace with
`scripts/analyze_models/plot_v4_memory_two_policies_20261114.py` in the GraphCast
environment. `manifest.json` pins workspace sources and results;
`archive_index.json` pins the files in this archive and the manuscript PDF asset.
