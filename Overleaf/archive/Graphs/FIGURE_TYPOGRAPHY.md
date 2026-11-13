# Paper figure typography

The maintained plotting sources share `paper_style.py`. On their 6.5-inch export
canvases, axis labels, tick labels and legends are 9 pt, and panel
titles are 11 pt. Heatmap numbers and annotations are 9 pt. The ICLR manuscript
scales these to its 5.5-inch text width, giving approximately 7.6 pt axis labels,
ticks and legends, and 9.3 pt titles. Mathematical subscripts and superscripts
retain their normal relative size.

Figures share a common export width. `main_v3.tex` displays the quantitative plots
at full text width, avoiding additional reductions to 80% or 90% of that width.
Long axis labels wrap, dense plots reserve space for their legends, and the
common-grid comparison uses shared axis labels. The diagrams retain the author's
editable presentation artwork.

## Regeneration

From the Weather_global repository root, use the GraphCast Python environment:

```bash
source scripts/graphcast_env.sh
python Overleaf/Graphs/main_v2/generate_figures.py --preview-dir Overleaf/Graphs/main_v2
python Overleaf/Graphs/main_v2/generate_appendix_horizon_figures.py
python Overleaf/Graphs/main_v3/generate_7yr.py
python Overleaf/Graphs/main_v3/generate_memory_comparison.py
python Overleaf/Graphs/main_v3/plot_training_ar_horizon_15yr.py
python Overleaf/Graphs/city_traces/generate_compact.py
python Overleaf/Graphs/main_v3/ablation_15yr/plot_ablation_15yr.py
python Overleaf/tools/rebuild_common_grid_figures.py
```

The common-grid generator additionally requires PyMuPDF. The existing evaluation
and city generators read their documented saved results; they do not run forecast
evaluation or training. The closed-loop horizon and extra-ablation generators use
frozen numerical exports stored beside them.

Then build with `bash Overleaf/build_v3.sh` and inspect the resulting
`Overleaf/main_v3.pdf` at page scale, especially the multi-panel comparisons and
the appendix figure with its long caption. Generated PDFs are the LaTeX assets;
PNGs are previews.

`tools/standardize_figure_labels.py` is the earlier terminology-only workflow.
It now skips these maintained plots by default, preserving the larger typography.
Its explicit `--include-regenerated-plots` option restores the historical layouts;
rerun the generators above afterward to restore the current typography.

## Loss labels

The v3 manuscript and maintained plots use `GraphCast loss` as the display name.
Appendix A.2 documents direct evaluation, channel-based reconstruction, and the
historical variable-group reconstruction. This terminology update changes no
plotted values or formulas; source metadata retains the original metric details.
Per-variable errors and reductions continue to use `RMSE`.
