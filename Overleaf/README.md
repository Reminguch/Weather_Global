# Non-Markovian Weather manuscript

Use the root `main_v2.tex` as the main document in Overleaf with pdfLaTeX.
`main.tex` and `Graphs/main_v2.tex` retain earlier drafts.

The active manuscript uses the official ICLR 2027 anonymous submission template
and bibliography style. `iclr2027_conference.sty`, `iclr2027_conference.bst`, and
the required `fancyhdr.sty` are unmodified copies from the
[official template archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip),
retrieved on September 24, 2026. See the
[author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

Keep the bundled `fancyhdr.sty` beside the main document. The conference template
depends on its header-setting behavior. Using a newer system version can leave
the submission header blank.

## Writing suggestions

User suggestions are recorded using `\suozhi{...}` and appear in blue. They are
comments for discussion, not approved changes to the manuscript. Original Chinese
wording is preserved in adjacent source comments. Do not rewrite the corresponding
passage unless the user explicitly requests it. See `AGENTS.md`.

The default PDF includes these comments. For a clean preview, define
`\hideauthorcomments` before loading the manuscript, for example

```latex
\def\hideauthorcomments{1}
\input{main_v2.tex}
```

Use pdfLaTeX, BibTeX, and two further pdfLaTeX passes to resolve the bibliography
and cross-references. Both previews are available locally in the ignored `build/`
directory. `main_v2.pdf` shows comments and `main_v2_clean.pdf` hides them.

The clean preview is not a complete submission package. An author TODO before the
bibliography records the ICLR 2027 requirement for a study-wide AI use statement.
The authors must supply their usage history and verification details.
