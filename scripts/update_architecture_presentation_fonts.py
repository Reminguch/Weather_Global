"""Adjust the user's two-slide architecture deck without regenerating its art.

Usage: python scripts/update_architecture_presentation_fonts.py INPUT.pptx OUTPUT.pptx
Requires python-pptx. This targets the manually edited September 25 deck, rather
than the earlier SVG export. Bottom labels and all diagram artwork are preserved.
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN


def remove(shape):
    element = shape._element
    element.getparent().remove(element)


def label(slide, name, x, y, width, height, parts, color="243447", align=PP_ALIGN.CENTER):
    shape = slide.shapes.add_textbox(Pt(x), Pt(y), Pt(width), Pt(height))
    shape.name = name
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = paragraph.space_after = Pt(0)
    for text, size, italic, subscript in parts:
        run = paragraph.add_run()
        run.text = text
        run.font.name = "DejaVu Sans"
        # DrawingML applies its 58% script scaling when baseline is nonzero.
        # Compensate so the requested subscript size is the rendered size.
        run.font.size = Pt(size / 0.58 if subscript else size)
        run.font.italic = italic
        run.font.color.rgb = RGBColor.from_string(color)
        props = run._r.get_or_add_rPr()
        props.set("lang", "en-US")
        props.set("dirty", "0")
        if subscript:
            props.set("baseline", "-18000")
    return shape


def update(source, target):
    prs = Presentation(source)
    assert len(prs.slides) == 2, "Expected the user's edited two-slide deck"
    branch, rollout = prs.slides

    # Preserve the branch's lower annotations and existing 14 pt box labels.
    branch_preserved = {
        s.shape_id: s._element.xml for s in branch.shapes
        if s.shape_id not in (60, 124, 128, 132)
    }
    for group in branch.shapes:
        if group.shape_id not in (60, 124, 128, 132):
            continue
        for shape in group.shapes:
            if not shape.has_text_frame:
                continue
            is_base = shape.text == "m"
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(18 if is_base else 10.8)
            if is_base:
                shape.top -= Pt(1)
                shape.width += Pt(3)
            else:
                shape.left += Pt(2)
                shape.width += Pt(4)

    original = {s.shape_id: s for s in rollout.shapes}
    bottom = {s.shape_id: s._element.xml for s in rollout.shapes if s.top >= Pt(145)}

    # Replace only model-label text. Artwork and box positions remain untouched.
    for ident in (158, 171, 226, 227, 228, 229, 230, 231, 232, 236):
        remove(original[ident])
    for index, (correction_box, graphcast_box) in enumerate(((85, 87), (104, 106), (123, 125)), 1):
        for box_id, word, symbol, sub in ((correction_box, "Correction ", "R", "φ"),
                                         (graphcast_box, "Frozen GraphCast ", "G", "θ")):
            box = original[box_id]
            x, y, w, h = [value / 12700 for value in (box.left, box.top, box.width, box.height)]
            label(rollout, f"Step {index}: {word.strip()}", x, y, w, h,
                  [(word, 7, False, False), (symbol, 7, True, False), (sub, 4.9, True, True)])

    # Orange memory labels grow to the same 7 pt main size as model labels.
    initial = original[77]
    remove(initial)
    label(rollout, "Initial temporal memory", initial.left / 12700, 22.5, 105, 12,
          [("Temporal memory  ", 7, False, False), ("m", 7, True, False),
           ("−1", 4.9, False, True), (" = 0", 7, False, False)],
          color="B45C16", align=PP_ALIGN.LEFT)
    for index, ident in enumerate((96, 115, 134)):
        group = original[ident]
        remove(group)
        label(rollout, f"Temporal memory m_{index}", group.left / 12700 - 1, 22.5, 15, 12,
              [("m", 7, True, False), (str(index), 4.9, False, True)], color="B45C16")

    # Match the user's one-based prediction labels: c_1, c_2, c_3.
    for index, ident in enumerate((91, 110, 129), 1):
        group = original[ident]
        remove(group)
        right = (group.left + group.width) / 12700
        label(rollout, f"Correction c_{index}", right - 15, group.top / 12700 - 1, 15, 12,
              [("c", 7, True, False), (str(index), 4.9, False, True)],
              color="246A9A", align=PP_ALIGN.RIGHT)

    for ident in (84, 103, 122):
        # These heading text boxes are inside their own unchanged groups.
        for group in rollout.shapes:
            if group.shape_type == 6:
                for shape in group.shapes:
                    if shape.shape_id == ident:
                        for paragraph in shape.text_frame.paragraphs:
                            for run in paragraph.runs:
                                run.font.size = Pt(7)

    for shape in branch.shapes:
        if shape.shape_id in branch_preserved:
            assert shape._element.xml == branch_preserved[shape.shape_id]
    after = {s.shape_id: s for s in rollout.shapes}
    for ident, expected in bottom.items():
        assert after[ident]._element.xml == expected, f"Bottom object {ident} changed"

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    prs.save(target)
    check = Presentation(target)
    assert len(check.slides) == 2
    print(f"Saved {target}; preserved {len(bottom)} lower-slide objects unchanged.")


if __name__ == "__main__":
    update(*sys.argv[1:])
