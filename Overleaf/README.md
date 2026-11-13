# Non-Markovian Weather manuscript

The active manuscript is `arxiv_versoin.tex` (existing filename preserved).
Select this file as the main document in Overleaf and use pdfLaTeX.

## ICLR revision v4

`v4.tex` is the anonymous ICLR 2027 revision. It builds on the arXiv
scientific text, figures, and appendices, with the ICLR formatting and
AI-use, ethics, and reproducibility statements from `archive/main_v3.tex`.
Select `v4.tex` as the main document when building the review version.
The original arXiv manuscript remains available as `arxiv_versoin.tex`.

Changes relative to `archive/main_v3.tex` are marked in blue in `v4.tex`:
revised paragraphs, captions, headings, table values, and new material.
Replaced figure panels have blue frames; the plots retain their original
scientific color scales. Unchanged material remains black.

The unmodified ICLR style, bibliography style, and matching `fancyhdr.sty`
are copied from `archive/` to support a root-level Overleaf build.
To build locally, run `bash build_v4.sh /absolute/path/outside/Overleaf`.
Omit the argument to create a temporary build directory under `/tmp`.
The script writes `v4.pdf` and all auxiliary files to that directory.
At the user's request, a checked copy of `v4.pdf` is also kept in this folder.

The November 13 revision adds five fresh M1 training seeds to Figure 4 and
the appendix seed table, separates the original SWA and local raw-checkpoint
memory comparisons in Figure 6, and adds paired date-bootstrap intervals to
Figures 5/9 and a day-10 comparison table. New assets use `_v4_` filenames;
the original arXiv manuscript and its figures are unchanged. See
`arxiv_pics/README.md` for provenance and archived numerical exports.
`rebuttal_Jvjy.txt` is the updated OpenReview-compatible draft response, with
arXiv identifiers written as text. It has not been posted to OpenReview.

Author action retained in `v4.tex`: verify the inherited AI-use statement.
The new revision includes AI-assisted analysis implementation and plotting,
so its existing claim of no analytical assistance needs author correction.

## Active files

- `arxiv_versoin.tex`: manuscript, including appendices.
- `bib.bib` and `arxiv_references.bib`: both bibliography databases are required.
- `plainnat.bst`: the unchanged bibliography style used by the manuscript.
- `nicefrac.sty`: the unchanged local package previously bundled under `texmf/`.
- `arxiv_pics/`: the 11 original external figure PDFs and four v4 replacements, named by printed figure number and content. Its README maps each file to its source.
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
