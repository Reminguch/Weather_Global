# Non-Markovian Weather manuscript

The active manuscript is `arxiv_versoin.tex` (existing filename preserved).
Select this file as the main document in Overleaf and use pdfLaTeX.

## ICLR v5

The November 15 spectral refresh uses **122 matched starts, January 1–December
30, 2023**, at all 40 six-hour leads, for raw and filtered GraphCast, M1 and R1.
Figures 6 and 10–13, Table 6, and their numerical discussion in both v5 sources
now use this completed annual cohort. The 30-date reset extension and annual
merge completed successfully. No new forecasts were launched for this replot.
Earlier 36- and 70-date exports and figures remain archived. Updated data,
sources, and reproduction instructions are in
`archive/Graphs/v5_spectral_matched122_20261115/`; PDFs and build checks are in
the parent workspace's `artifacts/manuscripts/v5_spectral_matched122_20261115/build/`.

The subsequent authorized variable-wise revision keeps the combined 83-channel
spectra in Section 4.6 and moves the scalar comparison to Appendix C (Table 6).
Figures 11 and 12 now pair amplitude and coherence for T2m, Z500, U850, and Q100
at days 5 and 10. Table 7 gives native-grid RMSE reductions relative to R1.
The four fields illustrate varied behavior, including the adverse day-10 Q100
result; all 83 channels and both explicitly distinguished mismatch weightings
remain in the supporting data. The sqrt(0.75) line denotes a 25% power deficit
only for an individual channel and scale, and is a weighted-amplitude reference
on combined curves. Original exploratory T400 figures remain archived.
The portable renderer, selection rationale, input arrays, source hashes and
coordinate checks are in `archive/Graphs/v5_variable_spectral_presentation_20261115/`.
The corresponding response drafts are `replies/reply_v96w.txt` and
`replies/reply_v96w.tex`. The latest checked
PDFs and page previews are in the parent workspace's
`artifacts/manuscripts/v5_variable_spectral_revision_20261115/`.

The follow-up reference clarification cites Subich Section 3.1 in the main
text and Figure 6 caption, explains how crossing scales indicate attenuation,
and restricts the M1/R1 similarity statement to day 5. The sampled crossings
are 16/16 at day 5 and 7/12 at day 10; nearby reference levels preserve this
contrast. Derived crossing tables are in
`archive/Graphs/v5_spectral_reference_explanation_20261115/`, with the corresponding
builds in `artifacts/manuscripts/v5_spectral_reference_explanation_20261115/`
in the parent workspace.

Appendix C now interprets the per-variable ranges below the reference alongside
coherence and native-grid RMSE, including isolated T2m dips, the U850 tradeoff,
Z500's excess fine-scale power and the adverse Q100 result. The latest checked
builds are in the parent workspace's
`artifacts/manuscripts/v5_appendix_reference_interpretation_20261115/`.

`v5.tex` is now the **tracked review copy** of the pilot. Original wording
removed or replaced during compression is retained with `\sout{}`; replacements
are green. Moved equations, the training schematic, and the policy table remain
visible at their original locations and are labelled as moves to Appendix A.2.
Existing author comments and the earlier blue/red revision colors are retained.

Select **`v5_clean.tex`** in Overleaf for the standalone clean manuscript.
It contains the complete paper source: deleted text and review annotations are
physically removed, and new/revised text remains blue. It can be edited directly
and does not input `v5.tex`. The usual figure, bibliography, and style dependencies
remain in this project. **`v5.tex`** is retained as the longer tracked comparison.
The build script compiles the selected files without regenerating their sources.
The full original v4 introduction is preserved. The latest revision adds an
overview to Appendix A and condenses its training configuration, defines M1
as the principal selected model, and distinguishes its independent retrainings.
The matched reset-trained, reset-evaluated control is named **R1** in v5.
The independent local seed-22 reset control is identified as an R1 retraining,
distinct from the original matched SWA checkpoint. Text, tables, and active
figure legends use this naming; historical review wording remains struck.
All three Figure 1 panels are centered within their subfigures.
The former Appendix E now follows B as Appendix C; common-grid comparisons
and city examples become D and E. References use labels and update automatically.
Appendix C now gives its diagnostic protocol and spectral methods once, then
presents the supporting figures. Repeated training/SWA and bootstrap setup
across the appendices is replaced with references to A.1 and A.4; evaluation
cohorts, masks, and uncertainty scopes remain explicit where they differ.
The spectral comparison now uses the completed 122-initialization evaluation.
Outdated draft-status wording has been removed from both manuscript sources.
Numerical-repeatability details remain in source comments and data manifests;
the spectral scores remain fixed-checkpoint point estimates without new
uncertainty intervals. The active spectral figure assets use
`_v5_matched122` or `_v5_variables122` names. Figure 6 has a compact single-row legend below the panels.
Its height is reduced from 4.9 to 3.8 inches with the original label font sizes.
Figures 11–13 now follow the same compact style: a single-row legend below the
panels, no duplicate draft-status heading, and less unused vertical space.
Use this layout for subsequent power-spectrum and amplitude/coherence figures,
retaining readable label sizes, field/lead titles, annotations, and plotted data.
Earlier top-float placement lets Figure 6 share page 9 with its discussion;
the pre-conclusion float barrier is removed so text can fill the available space.
All manuscript prose, equations, tables, captions, and comments are retained.
The day-5/day-10 physical power-spectrum plots (former v5 Figures 14–15,
v4 Figures 16–17) have been removed at the author's request. The common-grid
and city plots are now Figures 14–15. Removed captions remain struck in the
review copy; their original source blocks and PDFs are preserved in
`archive/Graphs/v5_removed_power_figures_20261114/`.
With the variable-wise revision, the checked clean PDF has 10 main-text
pages and 30 total pages; the tracked review PDF has 39 pages.
Run `bash build_v5.sh /absolute/path/outside/Overleaf` to build both PDFs;
an optional second argument `review` or `clean` builds just that view.

The spectral score table (Table 6 in Appendix C) reports each error as a ratio to raw
GraphCast: 1.00 is baseline and lower is better, rounded to two decimals.
Both exact MSE and power mismatch use their own raw-GraphCast denominator.
The score definitions and normalization are unchanged; the latest values use
the matched 122-date cohort described above. The reviewer v96w response's quoted
spectral scores are synchronized with this update.

In the Weather_global workspace, updated PDFs and checks are stored in
`artifacts/manuscripts/v5_tracked_20261114/`, outside the manuscript repository.
The initial unmarked pilot is retained in `v5_pilot_20261114/` beside it.
The introduction-restored PDFs and comparison checks are in
`artifacts/manuscripts/v5_intro_restored_20261114/` in that workspace.
The clean PDF with blue revision text is in
`artifacts/manuscripts/v5_clean_blue_20261114/`; its content and page count
match the introduction-restored clean draft.
The standalone clean TeX and its checked PDF are in
`artifacts/manuscripts/v5_clean_tex_20261114/` in the Weather_global workspace.
The latest appendix/M1 revision and its checks are in
`artifacts/manuscripts/v5_appendix_revision_20261114/`. The subsequent compact
appendices and validation are in `artifacts/manuscripts/v5_appendix_compact_20261114/`.
The latest full-evaluation draft and checks are in
`artifacts/manuscripts/v5_pending122_20261114/`. Pending-result figure provenance
is in `archive/Graphs/v5_spectral_pending122_20261114/`.
The subsequent physical-spectrum removal build and checks are in
`artifacts/manuscripts/v5_remove_power_figures_20261114/`.
The R1 naming update is checked in `artifacts/manuscripts/v5_r1_naming_20261114/`;
figure-label sources and unchanged-coordinate checks are retained in
`archive/Graphs/v5_r1_labels_20261114/`.
The subsequent formatting-only build and page previews are in
`artifacts/manuscripts/v5_formatting_20261114/`; Figure 6 layout provenance is in
`archive/Graphs/v5_fig6_compact_20261114/`.
The matching Appendix C figure layouts are archived in
`archive/Graphs/v5_spectral_compact_20261114/`, with the compiled manuscript and
page previews in `artifacts/manuscripts/v5_spectral_style_20261114/`.
Figure-label source,
provenance, and numerical-equivalence checks are retained in
`archive/Graphs/v5_m1_labels_20261114/`.
The bundled, unmodified `ulem.sty` supports strikethrough on local TeX installs.
See `v5_pilot_notes.txt` for the change list and figure mapping.

## ICLR revision v4

`v4.tex` is the anonymous ICLR 2027 revision. It builds on the arXiv
scientific text, figures, and appendices, with the ICLR formatting and
AI-use, ethics, and reproducibility statements from `archive/main_v3.tex`.
Select `v4.tex` as the main document when building the review version.
The original arXiv manuscript remains available as `arxiv_versoin.tex`.

Changes relative to `archive/main_v3.tex` are marked in blue in `v4.tex`:
changed phrases, numerical values and genuinely new material. Preserved wording
remains black, including within revised paragraphs and captions. Figures use
plain `\includegraphics` without revision frames. The latest Overleaf sizing
is retained, including 80% text width for Figures 4 and 6.

The memory comparison now presents carry/carry against the separately trained
reset/reset control. Figure 6 retains the original matched SWA/date-interval
panel and the independent raw-checkpoint replication with five carry seeds and
one reset seed. The inference-only carry/reset diagnostic is omitted from the
active manuscript and response; its historical outputs remain archived.
The focused figure and its unchanged numerical curves are archived in
`archive/Graphs/v4_memory_two_policies_20261114/`.

The November 14 smoothing revision is marked in **red**, including new prose,
equations, table entries, captions and the AMSE bibliography entry.
Earlier changed text remains blue; figures have no colored revision frames.
Section 4.6 and Appendix E include raw GraphCast, filtered GraphCast,
carry/carry (M1) and matched reset/reset throughout. Their figures and Table 2
use the same fixed 36-date diagnostic cohort (1 January–16 April 2023), retained independently
of subsequent full-campaign completion. Filtered M1 is omitted. Both memory
policies retain their original unfiltered spectra, without amplitude matching.
These point estimates have no date/seed uncertainty bands; the principal
122-date memory comparison and its uncertainty remain unchanged.

The revised discussion separates the advantage over the tested GraphCast
filter from the lead-dependent tradeoff against reset/reset. At day 5,
carry/carry has 2.84% lower MSE with 0.50 percentage points higher aggregate
power mismatch. At day 10, its 10.20% loss advantage accompanies a larger
spectral difference. It does not claim equal spectra or uniform dominance
of reset/reset. Table 2 now reports relative MSE and power mismatch as
percentages of raw GC for rollout, day 5 and day 10 (baseline 100%, lower better).

Section 4.6 explains why power spectra provide an additional measure of forecast
quality: lower MSE can come at the cost of spatial detail. It cites GraphCast
and Subich et al. (AMSE) and motivates the development-fitted smoothing control.
It presents day-5 and day-10 amplitude/coherence side by side in Figure 7 and the four-model percentage
comparison in Table 2 (replacing the bar chart previously shown as Figure 8). The day-5 result supports the architecture's accuracy
benefit with a small aggregate spectral cost; day 10 is presented as a regime
with higher absolute errors and lower coherence, while retaining its relative
accuracy advantage. Appendix E keeps the equations, metric definitions,
per-variable/multilead diagnostics and physical power spectra. It references
the main-text table instead of repeating it. Figure 12 now shows only
the absolute GraphCast-loss curves; its redundant loss-reduction panel has
been removed. The four curves and fixed 36-date data are unchanged.

Figure numbers follow the new placement: training-horizon figures are 8–9,
all-variable common-grid comparison is 10, city traces 11, and detailed
spectral diagnostics 12–17. Figures 8–11 and 13–17 are byte-identical copies of
the existing figures with unchanged inclusion widths. The mapping is retained in
`archive/Graphs/v4_smoothing_table_20261114/`.
Figure 7 now uses two columns (day 5 and day 10) and two rows (amplitude ratio
and coherence), with common axes for each metric and all four models on the
same fixed 36 dates. Its source export, script and provenance are retained in
`archive/Graphs/v4_spectra_day5_day10_20261114/`.
The original arXiv source is unchanged. The fixed 36-date numerical exports,
all-channel curves, source metadata and plotting snapshots are archived in
`archive/Graphs/v4_appendix_e_matched36_20261114/`. Full 40-lead source pools
remain pinned and retained in the parent Weather_global workspace. Later
completed dates must use a new archive rather than overwrite this snapshot.
Earlier four-date and 92-date archives remain historical provenance and
are not mixed into the active Appendix E comparisons.

The unmodified ICLR style, bibliography style, and matching `fancyhdr.sty`
are copied from `archive/` to support a root-level Overleaf build.
To build locally, run `bash build_v4.sh /absolute/path/outside/Overleaf`.
Omit the argument to create a temporary build directory under `/tmp`.
The script writes `v4.pdf` and all auxiliary files to that directory.
At the user's request, a checked copy of `v4.pdf` is also kept in this folder.

The November 13 revision adds five fresh M1 training seeds to Figure 4 and
the appendix seed table, separates the original SWA and local raw-checkpoint
memory comparisons in Figure 6, and adds paired date-bootstrap intervals to
Figures 5/10 (originally 5/9) and a day-10 comparison table. New assets use `_v4_` filenames;
the original arXiv manuscript and its figures are unchanged. See
`arxiv_pics/README.md` for provenance and archived numerical exports.
All reviewer response drafts are in `replies/`. The OpenReview-compatible
text drafts are `replies/reply_Jvjy.txt`, `replies/reply_Jvjy_concise.txt`, and
`replies/reply_v96w.txt`. Their original W-first order is retained.
The additional draft `replies/reply_Jvjy_concise_qfirst.txt` uses Q1, Q2, W1, W2
order. `replies/reply_v96w_qfirst.txt` uses W1, W2, Q1, Q2 order despite its
retained filename. Both preserve the opening summaries and link the W
responses to the supporting details in Q.

`replies/reply.tex` combines the existing LaTeX reviewer responses. Select it
as the main document to compile both replies in Overleaf. For a local build,
run pdfLaTeX from the `Overleaf/` directory with `replies/reply.tex` as the
input and an output directory outside the project. The reviewer-specific
documents `replies/reply_Jvjy.tex` and `replies/reply_v96w.tex` can also be
compiled independently; their manuscript references use `v5_clean.tex`.
The text and LaTeX drafts are separate versions. The v96w response includes
the current v5 spectral values and the revised comparison with filtered
GraphCast. Keep its numerical values synchronized with subsequent v5 updates.
The earlier `replies/replies.tex` and `replies/replies.pdf` are retained as
the historical Jvjy response with v4 references. These drafts have not been
posted to OpenReview.

Author action retained in `v4.tex`: verify the inherited AI-use statement.
The new revision includes AI-assisted analysis implementation and plotting,
so its existing claim of no analytical assistance needs author correction.

## Active files

- `arxiv_versoin.tex`: manuscript, including appendices.
- `bib.bib` and `arxiv_references.bib`: both bibliography databases are required.
- `plainnat.bst`: the unchanged bibliography style used by the manuscript.
- `nicefrac.sty`: the unchanged local package previously bundled under `texmf/`.
- `arxiv_pics/`: original figure PDFs, v4 replacements and appendix spectral diagnostics, named by printed figure number and content. Its README maps each file to its source.
- `AGENTS.md` and `TERMINOLOGY.md`: editing instructions and terminology.

Figure 1 has three separate PDF panels. Figure 3 is drawn directly in LaTeX,
so there is no `fig3` PDF. The remaining external figures are numbered 2 and 4–10.

## Build

Run from this folder with a standard TeX Live installation. Keep generated PDFs
and auxiliary files outside the project root, for example:

```sh
build_dir="$(mktemp -d /private/tmp/weather-arxiv.XXXXXX)"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$build_dir" arxiv_versoin.tex
```

The compiled manuscript is written to `$build_dir/arxiv_versoin.pdf`.
The active manuscript no longer needs a custom `TEXMFHOME` setting.

## Archived work

`archive/` preserves earlier manuscripts, introduction drafts, ICLR and NeurIPS
styles, plotting scripts, editable figure sources, data, and unused figures.
Their original relative layout is preserved, including a copy of `bib.bib` for
older drafts. The old README and build scripts apply to that archived layout.
The official ICLR style files have not been modified.

Root-level compiled PDFs were removed during reorganization. Figure PDFs remain
in `arxiv_pics/` and in the historical source directories under `archive/`.

## Writing suggestions

User suggestions are recorded using `\suozhi{...}`. These annotations remain in
the source; the current arXiv manuscript suppresses them in the rendered PDF.
They are comments for discussion unless the user explicitly approves a prose change.
Preserve collaborator comments and original Chinese wording in adjacent source
comments. See `AGENTS.md`.
