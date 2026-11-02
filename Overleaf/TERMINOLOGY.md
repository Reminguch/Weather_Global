Approved terminology for the paper

The baseline is **GraphCast**, as requested by the author. Use the same name
in prose, tables, and figure legends. Specify its 1° output resolution in the
experimental setup; distinguish the higher-resolution reference as
**GraphCast (0.25°)**.

| Concept | Preferred term |
|---|---|
| Complete corrected model | GraphCast–Mamba (`GraphCast--Mamba` in LaTeX) |
| Frozen predictor G_theta | GraphCast backbone; GraphCast baseline when evaluated independently |
| Trainable branch R_phi | Mamba correction branch |
| Branch output c_t | Forecast correction |
| Complete output Y-hat_t | GraphCast–Mamba forecast |
| Mechanism retaining information between forecast steps | Temporal memory |
| Complete carried object m_t | Temporal memory state |
| Individual state-space model component h_t | SSM state |
| Stored causal-convolution inputs | Causal-convolution cache |
| Two-layer temporal block | Mamba module; distinguish its two Mamba layers |
| Spatial operation | Graph message-passing step |
| Reset experiment | Temporal-memory ablation; memory-reset controls |
| Policy order | Training/forecasting: carry/carry, carry/reset, reset/reset |
| Elapsed forecast time | Forecast lead time; day 10 = 240 h after initialization |
| Individual evaluation origin | Forecast initialization |
| Principal selection dataset | 2022 development set |
| Primary metric | Exact GraphCast loss (normalized weighted mean-squared error) |
| Reset-study diagnostic | Approximate reconstructed GraphCast loss |
| Historical horizon diagnostic | Approximate paper-formula weighted MSE |
| Eleven pooled outputs | Variable-group RMSE |
| Eighty-three individual outputs | Channel RMSE |
| Temporal-parameter optimization rate | Mamba learning rate; define LR before abbreviated table labels |
| Main experiment's full budget | 2,500 optimizer updates |
| Main experiment's SWA indices | Post-warm-up updates 500, 1,000, 1,500, 2,000 |
| Ground-truth input policy | Teacher forcing; teacher-forced prefix (ERA5 inputs) |
| Weather-variable labels | Specific humidity; mean sea-level pressure; pressure vertical velocity; six-hour accumulated precipitation |
| Compact weather-variable notation | T2m, U10, V10, U850, V850, Z500; define other variable/pressure-level combinations in captions |

Keep configuration qualifiers for distinct GraphCast–Mamba experiments. In
particular, the common-grid figure uses the archived K=22 configuration and
the city examples use a separate fifteen-year correction.

The historical training-tail labels are not a uniform count of predictions:
K=1 denotes teacher forcing with no forecast feedback, while K>1 denotes K
feedback-conditioned steps following the teacher-forced prefix. A chain with
K feedback steps includes K+1 predictions. Preserve the archived labels and
explain this convention rather than renumbering the experiments.

Temporal context and a physical memory kernel are distinct from the carried
temporal memory state. Exact loss, its approximate reconstructions, RMSE, and
MAE also remain distinct. A reset control retains the two-frame meteorological
input, so describe it as lacking persistent temporal memory.

The terminology pass preserves collaborator annotations and numerical results.
`tools/standardize_figure_labels.py` updates the archived figure labels from
preserved input PDFs and checks that no pixels outside the label regions change.
Its provenance is recorded in `tools/figure_label_changes.json`. The figure
PNGs and architecture SVGs are refreshed from the PDFs used in the paper.
The plotting generators retain the approved labels for subsequent regeneration.
