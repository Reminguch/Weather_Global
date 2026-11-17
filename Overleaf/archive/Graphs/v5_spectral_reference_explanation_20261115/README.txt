Attenuation-reference interpretation, 15 November 2026

The combined-reference crossing table is derived from the already published
83-channel weighted amplitude curves in
../v5_variable_spectral_presentation_20261115/weighted_spectral_curves.csv.
The source digest and M1/R1 curve-difference diagnostics are in interpretation.json.

At a fixed forecast lead, take the first sampled integer degree whose weighted
mean amplitude ratio is strictly below sqrt(1 - deficit_reference). No
interpolation is used. The reference fraction corresponds to power loss only
for a single channel and degree; for the combined curve, compare mean amplitude.
Wavelengths use 40,030 km divided by the degree. They are illustrative crossing
scales, not exact physical resolution limits or forecast-skill thresholds.

The reference follows Subich et al. (2025), Section 3.1:
https://arxiv.org/html/2501.19374v2#S3.SS1
Their 25% choice is illustrative. The 20% and 30% alternatives preserve the
qualitative comparison: similar M1/R1 crossings at day 5, and attenuation
extending to coarser scales for M1 at day 10. The full amplitude and coherence
curves, rather than a shared crossing alone, support the day-5 similarity.

These are crossings across spatial wavenumber at a fixed lead. The separate
scale_crossings.csv in the earlier variable archive instead reports crossings
across forecast lead for individual channels at fixed wavenumbers.
