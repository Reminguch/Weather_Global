#!/usr/bin/env python3
"""Reframe archived common-grid panels with legible, paper-size vector text.

The only available curve artwork is the bitmap embedded in the archived PDF.
This script copies that bitmap's panel crops without resampling or digitizing
curves. It supplies a shared vector legend. The appendix retains its original
in-panel legend, while the selected main-text panels retain the previous TeX
crop's exact white legend mask.
The archive remains untouched. No numerical evaluation is required.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Graphs"))
from paper_style import apply_style, FIGURE_WIDTH, LABEL_SIZE

SOURCE = ROOT / "tools/figure_label_sources/Graphs/add_mamba.pdf"
LABELS = ["Z500", "Z850", "T500", "T850", "T2m", "MSL",
          "Q700", "Q850", "U850", "V850", "U10", "V10"]
# Pixel positions read from the source raster's original tick marks. Keeping
# the raster coordinate system avoids estimating or changing any data values.
YTICKS = [
    [(50,477.5),(60,429.5),(80,332.5),(100,235.5),(120,139.5)],
    [(50,466.5),(60,419.5),(80,326.5),(100,232.5),(120,139.5)],
    [(50,446.5),(60,403.5),(80,316.5),(100,230.5),(120,143.5)],
    [(40,420.5),(60,347.5),(80,275.5),(100,202.5),(120,130.5)],
    [(50,947.5),(60,899.5),(80,804.5),(100,709.5),(120,614.5)],
    [(50,938.5),(60,894.5),(80,807.5),(100,719.5),(120,632.5)],
    [(40,912.5),(60,832.5),(80,752.5),(100,672.5),(120,593.5)],
    [(20,969.5),(40,891.5),(60,813.5),(80,735.5),(100,656.5)],
    [(40,1423.5),(60,1345.5),(80,1267.5),(100,1189.5),(120,1112.5)],
    [(40,1424.5),(60,1346.5),(80,1268.5),(100,1190.5),(120,1112.5)],
    [(40,1446.5),(60,1364.5),(80,1282.5),(100,1201.5),(120,1119.5)],
    [(40,1441.5),(60,1361.5),(80,1281.5),(100,1201.5),(120,1122.5)],
]
LEGEND_BOX = (105, 105, 431, 179)
ARCHIVAL_LABEL_BOXES = [(153,106,423,128), (153,132,423,154), (153,156,423,178)]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_panels(*, selected=False):
    with fitz.open(SOURCE) as doc:
        images = doc[0].get_images()
        assert len(images) == 1, "The frozen source must contain one plot bitmap"
        payload = doc.extract_image(images[0][0])["image"]
    original = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    assert original.shape == (1540, 2520, 3)
    artwork = original.copy()
    allowed = np.zeros(original.shape[:2], dtype=bool)
    if selected:
        x0, y0, x1, y1 = LEGEND_BOX
        # Exact mask formerly used by common_grid_three_panels.tex:
        # crop origin (12,64), picture height 476, put(93,361), size (326,74).
        # The source legend was baked into the bitmap over early peak tips;
        # do not infer the obscured data when relocating its readable legend.
        artwork[y0:y1, x0:x1] = 255
        allowed[y0:y1, x0:x1] = True
    else:
        # Retain the archival legend footprint and visible curve tips. Replace
        # text only within the already approved terminology-pass white boxes.
        inset = Image.fromarray(artwork)
        draw = ImageDraw.Draw(inset)
        font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"), 14)
        labels = [("GraphCast (0.25°)", "#1fa450"),
                  ("GraphCast–Mamba (K=22)", "#26609f"),
                  ("GraphCast", "#666666")]
        for box, (label, color) in zip(ARCHIVAL_LABEL_BOXES, labels, strict=True):
            x0, y0, x1, y1 = box
            draw.rectangle((x0,y0,x1-1,y1-1), fill="white")
            draw.text((x0,y0+1), label, font=font, fill=color)
            allowed[y0:y1, x0:x1] = True
        artwork = np.asarray(inset)
    outside = int(np.count_nonzero(np.any(artwork != original, axis=2) & ~allowed))
    assert outside == 0
    panels = []
    for i, label in enumerate(LABELS):
        row, col = divmod(i, 4)
        left = 98 + 618 * col
        top = 97 + 491 * row
        box = (left, top, left + 536, top + 386)
        crop = artwork[top:top+386, left:left+536].copy()
        panels.append({"label": label, "box": box, "pixels": crop,
                       "sha256": digest(crop.tobytes()), "index": i})
    return panels, outside


def plot_panel(ax, panel):
    left, top, right, bottom = panel["box"]
    ax.imshow(panel["pixels"], origin="upper", interpolation="none",
              extent=(left, right, bottom, top), aspect="auto")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    # Reinforce existing spine centers with vectors; thin raster borders can
    # otherwise disappear when the original image is reduced to paper width.
    for side, position in [("left",left+.5),("right",right-1.5),
                           ("top",top+.5),("bottom",bottom-1.5)]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_position(("data", position))
        ax.spines[side].set_linewidth(.6)
    ax.set_xticks([left + .5 + 534 * d / 10 for d in range(0, 11, 2)],
                  [str(d) for d in range(0, 11, 2)])
    ticks = [(value, pos) for value, pos in YTICKS[panel["index"]]
             if value % 20 == 0]
    ax.set_yticks([pos for _, pos in ticks], [str(v) for v, _ in ticks])
    ax.tick_params(length=2.5, width=.65, pad=2.5)
    ax.set_title(panel["label"], pad=5)


def shared_legend(fig, y):
    handles = [Line2D([], [], color="#26609f", lw=1.7),
               Line2D([], [], color="#8b919b", lw=1.5, ls="--"),
               Line2D([], [], color="#1fa450", lw=1.7)]
    return fig.legend(handles,
        ["GraphCast–Mamba ($K=22$)", "GraphCast", "GraphCast ($0.25°$)"],
        loc="upper center", bbox_to_anchor=(.52, y), ncol=3,
        frameon=False, handlelength=2.1, handletextpad=.5,
        columnspacing=1.1, borderaxespad=0)


def write_figure(path, panels, *, full):
    rows, cols, height = (3, 4, 5.3) if full else (1, 3, 2.35)
    fig, axes = plt.subplots(rows, cols, figsize=(FIGURE_WIDTH, height),
                             squeeze=False)
    if full:
        fig.subplots_adjust(left=.112, right=.988, bottom=.12, top=.845,
                            wspace=.39, hspace=.53)
        fig.suptitle("Full-year 2022 forecasts on the common 1.5° verification grid",
                     x=.52, y=.994)
        legend = shared_legend(fig, .937)
    else:
        fig.subplots_adjust(left=.112, right=.988, bottom=.265, top=.80,
                            wspace=.29)
        legend = shared_legend(fig, .985)
    for ax, panel in zip(axes.flat, panels, strict=True):
        plot_panel(ax, panel)
    fig.supxlabel("Forecast lead time (days)", x=.55,
                  y=.025 if full else .035, fontsize=LABEL_SIZE)
    fig.supylabel("RMSE vs ENS [%]", x=.015, y=.49, fontsize=LABEL_SIZE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    for text in [*fig.texts, *legend.get_texts(),
                 *(text for ax in axes.flat for text in
                   [ax.title, *ax.get_xticklabels(), *ax.get_yticklabels()])]:
        box = text.get_window_extent(renderer)
        assert (box.x0 >= figure_box.x0 - .5 and box.y0 >= figure_box.y0 - .5
                and box.x1 <= figure_box.x1 + .5 and box.y1 <= figure_box.y1 + .5), text.get_text()
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    # Verify PDF embeds exact crops, so no resampling or curve reconstruction
    # has silently entered the paper asset.
    expected = sorted(panel["sha256"] for panel in panels)
    with fitz.open(path) as doc:
        assert abs(doc[0].rect.width - 468) < .01
        actual = []
        for image in doc[0].get_images():
            payload = doc.extract_image(image[0])["image"]
            pixels = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
            # Matplotlib stores this top-origin image in bottom-origin PDF
            # order and uses an affine transform to restore its orientation.
            actual.append(digest(pixels[::-1].tobytes()))
        assert sorted(actual) == expected, "PDF changed one of the panel bitmaps"
        font_sizes = sorted({round(s["size"], 2)
            for b in doc[0].get_text("dict")["blocks"]
            for line in b.get("lines", []) for s in line["spans"]})
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes()),
            "pdf_width_pt": 468, "embedded_panel_bitmaps_match": True,
            "font_sizes_pt": font_sizes,
            "panels": [{k: v for k, v in p.items() if k != "pixels"} for p in panels]}


def main():
    apply_style()
    panels, outside = load_panels()
    selected, selected_outside = load_panels(selected=True)
    reports = [write_figure(ROOT / "Graphs/add_mamba.pdf", panels, full=True),
               write_figure(ROOT / "Graphs/main_v3/common_grid_three_panels.pdf",
                            [selected[i] for i in [0, 4, 8]], full=False)]
    report = {"source": str(SOURCE.relative_to(ROOT)),
              "source_sha256": digest(SOURCE.read_bytes()),
              "selected_previous_tex_legend_mask_pixels": LEGEND_BOX,
              "appendix_existing_terminology_boxes_pixels": ARCHIVAL_LABEL_BOXES,
              "changed_source_pixels_outside_label_regions": outside,
              "selected_changed_source_pixels_outside_previous_mask": selected_outside,
              "outputs": reports}
    (ROOT / "tools/common_grid_typography.json").write_text(
        json.dumps(report, indent=2) + "\n")
    for item in reports:
        print(item["path"], "— exact panel bitmaps verified")


if __name__ == "__main__":
    main()
