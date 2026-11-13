# Editable architecture figures

## Submission notation corrections (September 26)

The two-slide master and the PDFs in this directory are the active figures
included by `Overleaf/main_v3.tex`. The correction branch labels its two
Mamba modules explicitly, with two layers per module. The rollout preserves
the author's input/output convention: `X_{0:1}` produces the first forecast
`Yhat_1` with correction `c_1`. Its memory starts at `m_0 = 0` and advances
through `m_1`, `m_2`, and `m_3`. The manuscript abbreviates each two-frame
window `X_{t-1:t}` as `X_t` and indexes predictions by forecast call.

`Overleaf/tools/fix_architecture_notation.py` applies these label changes to
the edited master without replacing its artwork. `notation-fix-report.json`
records the backup, export crops, checksums, and validation. The parent-directory
PDFs and their previews remain the earlier exports used by `main_v2.tex`.

## Current edited version (September 25)

`02_correction_branch.pptx` is now the user's edited **two-slide master**:
slide 1 is the correction branch; slide 2 is the temporal rollout. The matching
single-figure PDFs are `02_correction_branch.pdf` and `03_temporal_rollout.pdf`.
Copies in the parent directory are the assets included in Figure 1 of
`Overleaf/main_v2.tex`. They were exported through LibreOffice, cropped without
rasterizing the vector content, and visually checked.

The latest pass standardizes the rollout's model labels, enlarges its memory
and correction labels, uses `c_1`, `c_2`, `c_3`, and enlarges the branch's upper
memory labels. Lower rollout objects and the branch's lower annotations are
unchanged. See `font-update-report.json` for sizes, checksums, and the backup.

The font changes have also been reapplied to the later PowerPoint save from
13:32 EDT, preserving its revised images and all 49 lower rollout objects.
Reopen `02_correction_branch.pptx` to load this update. Both slides were rendered
and visually checked with LibreOffice.

The input globe pairs keep their original spacing. Each rear globe and its mesh
overlay use 25% native picture transparency; the foreground globe is unchanged.
The white rims, backing circles, and added shadows have been removed. All three
pairs remain editable, and their original image pixels are unchanged.
Both copies of `03_temporal_rollout.pdf` were refreshed from this deck. See
`globe-separation-report.json` for the current checksum, validation, and backup;
`font-update-report.json` records the preceding font-only pass.

The combined deck and ZIP below are **earlier export snapshots**. The original
SVGs are also earlier versions. Do not regenerate over the edited master using
the original SVG exporter; that would discard manual presentation edits.

## Original export snapshots

Open these files in desktop PowerPoint or Keynote:

- `02_correction_branch.pptx`: one slide, matching the original figure's 16 × 3.4 inch canvas.
- `03_temporal_rollout.pptx`: one slide, matching the original figure's 16 × 6.9 inch canvas.
- `gc_mamba_editable_figures.pptx`: both figures on separate 16:9 slides.

## Editing

1. Install the four DejaVu Sans fonts from `fonts/` before opening the presentation
   to keep the original label metrics. On macOS, open the font files with Font
   Book and install them. Their redistribution license is included.
2. In Keynote, use File → Open and select the `.pptx`; save as a Keynote file if
   desired. In PowerPoint, open the `.pptx` directly.
3. Boxes, arrow paths, and labels are native objects. Select a group and use
   Ungroup when needed to edit its components. Mathematical labels use small
   grouped text boxes for subscripts and superscripts, rather than an equation
   editor. Shape outlines and labels are separate objects; select and group
   them together before moving a labeled box.
4. Arrowheads and shafts are grouped. Paths can be resized or edited, but are
   not attached connectors: moving a box requires adjusting its arrows.
5. The Earth illustrations and mesh overlays are separate movable, replaceable
   PNG pictures with circular crops. Their pixels are not editable vector shapes.

For LaTeX, edit the figure-sized single-slide deck and export it as PDF. Save
the edited `.pptx` or `.key` as the new editable source. The Python SVG generator
does not read edits made in presentation software.

## Verification and reproduction

The original SVG/PDF/PNG files were left unchanged. Shapes, label positions,
colors, and image layers were converted from the SVGs. Source SHA-256 hashes and
counts of editable objects appear in `export-report.json`.

The generated presentations were reopened and all nested objects checked with
python-pptx. A reconstruction from the serialized presentation geometry was
rendered for visual inspection. Native PowerPoint and Keynote were unavailable
on this Linux host, so their import/font rendering has not been directly tested.

Recreate the decks from the repository root:

```bash
python -m pip install --target .cache/presentation-python python-pptx==1.0.2 Pillow
PYTHONPATH=.cache/presentation-python python scripts/export_gc_mamba_pptx.py
```

The exporter uses the system DejaVu Sans fonts and their license under
`/usr/share/fonts/dejavu-sans-fonts` and `/usr/share/licenses/dejavu-sans-fonts`.

References:
- [Apple: open and edit PowerPoint presentations in Keynote](https://support.apple.com/guide/keynote/tan72232b56/mac)
- [Microsoft: SVG editing and conversion to shapes](https://support.microsoft.com/en-us/office/graphics-visuals/edit-svg-images-in-microsoft-365)
