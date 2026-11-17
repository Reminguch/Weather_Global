# Original carry/carry and reset/reset spectra

November 14, 2026 appendix revision. Figures 16–18 in `v4.tex` use the
same four forecast dates: 2023-01-01, 2023-04-01, 2023-07-03, 2023-10-01.
Carry/carry and reset/reset are the original predictions without amplitude
rescaling. These diagnostics are separate from the 92-date filtering cohort
and the 122-date memory-training comparison. No new forecast was run here.

The CSVs contain exact native-grid losses, weighted amplitude/coherence,
physical powers for T2m/Z500/Q700/TP6h, and all-channel summary scores.
The manifest pins source data, scripts and plotted artifacts and retains
finite rerun drift. It records definitions, model identities, units and scope.
Physical powers are shown separately for each field, with ERA5 reference;
degree zero is omitted and no climatology is subtracted.

Run `scripts/analyze_models/plot_v4_memory_spectra_appendix_20261114.py`
from the parent Weather_global workspace in its GraphCast environment.
The script imports `plot_v4_smoothing_paper_20261114.py`, also archived here;
large source moments remain at the workspace paths pinned in the manifest.
`archive_index.json` supplies hashes relative to this portable archive.
