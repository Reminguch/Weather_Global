V5 variable spectral presentation, 15 November 2026

Figures 6 and 13 retain the original GraphCast variable/pressure-weighted
83-channel amplitude and coherence curves. Figures 11 and 12 show T2m,
Z500, U850, Q100 at days 5 and 10. Scalar metrics are not drawn on
coherence panels. Dotted line: amplitude matching / unit coherence;
dash-dot line on amplitude panels: sqrt(0.75). This is a 25% power
deficit only for an individual channel, not for the combined curve.

Native-grid RMSE contrasts: selected_native_grid_rmse.csv. All 83
channels and original metric weightings are retained in the source
CSVs. The two channel coherence exports agree to rtol=atol=1e-13.
Finite rerun warnings remain in channel_source_summary.json.

Replay with Python/NumPy/Matplotlib:
python plot_v5_variable_presentation_20261115.py --input-dir . --image-dir /tmp/v5-variable-replay
