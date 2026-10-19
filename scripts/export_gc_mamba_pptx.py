"""Export the two architecture SVGs as editable PowerPoint objects.

Run: PYTHONPATH=.cache/presentation-python python scripts/export_gc_mamba_pptx.py
Dependencies: python-pptx, Pillow. The converter intentionally accepts only the
SVG primitives used by these two Matplotlib exports; unsupported input fails.
"""

from pathlib import Path
import base64
import hashlib
import io
import json
import re
import shutil
import xml.etree.ElementTree as ET

from PIL import ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Overleaf/Pics/gc_mamba_architecture"
OUT = SOURCE / "presentation"
STEMS = ("02_correction_branch", "03_temporal_rollout")
SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"
FONT_DIR = Path("/usr/share/fonts/dejavu-sans-fonts")
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"


def xml(tag, **attrs):
    node = OxmlElement(tag)
    for key, value in attrs.items():
        node.set(key, str(value))
    return node


def style(node, inherited):
    result = inherited.copy()
    for pair in node.get("style", "").split(";"):
        if ":" in pair:
            key, value = pair.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def transform(value, state):
    # Diagonal affine transforms suffice here (translations and image flips).
    sx, sy, tx, ty = state
    for kind, raw in re.findall(r"(\w+)\(([^)]*)\)", value):
        vals = list(map(float, re.findall(NUMBER, raw)))
        if kind == "translate":
            tx += sx * vals[0]
            ty += sy * (vals[1] if len(vals) > 1 else 0)
        elif kind == "scale":
            sx *= vals[0]
            sy *= vals[1] if len(vals) > 1 else vals[0]
        elif kind == "rotate" and vals[0] == 0:
            pass
        else:
            raise ValueError(f"Unsupported SVG transform: {kind} {vals}")
    return sx, sy, tx, ty


def path_commands(data):
    tokens = re.findall(NUMBER + r"|[A-Za-z]", data)
    sizes = {"M": 2, "L": 2, "Q": 4, "C": 6, "z": 0, "Z": 0}
    commands = []
    while tokens:
        cmd = tokens.pop(0)
        if cmd not in sizes:
            raise ValueError(f"Unsupported SVG path command: {cmd}")
        size = sizes[cmd]
        commands.append((cmd, list(map(float, tokens[:size]))))
        del tokens[:size]
    return commands


def bounds(commands):
    xs = [x for _, values in commands for x in values[::2]]
    ys = [y for _, values in commands for y in values[1::2]]
    return min(xs), min(ys), max(xs), max(ys)


def font_info(css):
    value = css["font"]
    size = float(re.search(r"(" + NUMBER + r")px", value)[1])
    bold = "700" in value or "bold" in value
    italic = "oblique" in value or "italic" in value
    suffix = "-BoldOblique" if bold and italic else "-Bold" if bold else "-Oblique" if italic else ""
    font = ImageFont.truetype(str(FONT_DIR / f"DejaVuSans{suffix}.ttf"), round(size * 100))
    return size, bold, italic, font


class Exporter:
    def __init__(self, slide, root, offset_y=0):
        self.slide = slide
        self.root = root
        self.offset_y = offset_y
        self.clips = {node.get("id"): node for node in root.iter(SVG + "clipPath")}
        self.counts = {"vector_shapes": 0, "text_boxes": 0, "pictures": 0}

    def path(self, node, shapes, css, state, name):
        commands = path_commands(node.get("d"))
        sx, sy, tx, ty = state
        commands = [(cmd, [v * (sx if i % 2 == 0 else sy) + (tx if i % 2 == 0 else ty)
                           for i, v in enumerate(values)]) for cmd, values in commands]
        x0, y0, x1, y1 = bounds(commands)
        w, h = max(x1 - x0, .01), max(y1 - y0, .01)
        shape = shapes.add_shape(MSO_SHAPE.RECTANGLE, Pt(x0), Pt(y0), Pt(w), Pt(h))
        shape.name = name
        props = shape._element.spPr
        props.remove(props.prstGeom)
        geometry = xml("a:custGeom")
        for tag in ("avLst", "gdLst", "ahLst", "cxnLst"):
            geometry.append(xml("a:" + tag))
        geometry.append(xml("a:rect", l="0", t="0", r="r", b="b"))
        paths = xml("a:pathLst")
        path = xml("a:path", w=round(w * 12700), h=round(h * 12700))
        tags = {"M": "moveTo", "L": "lnTo", "Q": "quadBezTo", "C": "cubicBezTo", "z": "close", "Z": "close"}
        for cmd, values in commands:
            segment = xml("a:" + tags[cmd])
            for x, y in zip(values[::2], values[1::2]):
                segment.append(xml("a:pt", x=round((x - x0) * 12700), y=round((y - y0) * 12700)))
            path.append(segment)
        paths.append(path)
        geometry.append(paths)
        props.insert(1, geometry)
        fill = css.get("fill", "#000000")
        if fill == "none":
            shape.fill.background()
        else:
            shape.fill.solid()
            shape.fill.fore_color.rgb = RGBColor.from_string(fill.lstrip("#"))
        stroke = css.get("stroke", "none")
        if stroke == "none":
            shape.line.fill.background()
        else:
            shape.line.color.rgb = RGBColor.from_string(stroke.lstrip("#"))
            sw = float(css.get("stroke-width", 1))
            shape.line.width = Pt(sw)
            line = props.get_or_add_ln()
            line.set("cap", "rnd" if css.get("stroke-linecap") == "round" else "flat")
            if "stroke-dasharray" in css:
                dash, gap = map(float, css["stroke-dasharray"].split(","))
                custom = xml("a:custDash")
                custom.append(xml("a:ds", d=round(dash / sw * 100000), sp=round(gap / sw * 100000)))
                line.append(custom)
            line.append(xml("a:round"))
        self.counts["vector_shapes"] += 1

    def textbox(self, text, x, y, css, shapes, name):
        size, bold, italic, font = font_info(css)
        width = font.getlength(text) / 100
        anchor = css.get("text-anchor", "start")
        if anchor == "middle":
            x -= width / 2
        elif anchor == "end":
            x -= width
        ascent = font.getmetrics()[0] / 100
        shape = shapes.add_textbox(Pt(x), Pt(y - ascent), Pt(max(width + 3, size)), Pt(size * 1.8))
        shape.name = name + " — " + text.strip()
        frame = shape.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.word_wrap = False
        frame.auto_size = MSO_AUTO_SIZE.NONE
        frame.vertical_anchor = MSO_ANCHOR.TOP
        para = frame.paragraphs[0]
        para.space_before = para.space_after = Pt(0)
        run = para.add_run()
        run.text = text
        run.font.name = "DejaVu Sans"
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = RGBColor.from_string(css.get("fill", "#243447").lstrip("#"))
        self.counts["text_boxes"] += 1

    def text(self, node, shapes, css, state, name):
        sx, sy, tx, ty = state
        assert sx == sy == 1, "Scaled text needs explicit font-size conversion"
        if not list(node):
            self.textbox(node.text or "", tx + float(node.get("x", 0)), ty + float(node.get("y", 0)), css, shapes, name)
            return
        # Preserve mathematical super/subscripts, including mixed baselines in a
        # single Matplotlib tspan. Normal words remain whole editable text boxes.
        for span in node:
            scss = style(span, css)
            size, _, _, font = font_info(scss)
            xs = list(map(float, span.get("x", "0").split()))
            ys = list(map(float, span.get("y", "0").split()))
            groups = []
            previous_x = previous_y = previous_width = None
            for i, char in enumerate(span.text or ""):
                x = xs[i] if i < len(xs) else previous_x + previous_width
                y = ys[i] if i < len(ys) else ys[-1]
                width = font.getlength(char) / 100
                if groups and abs(y - previous_y) < .01 and abs(x - previous_x - previous_width) < .6:
                    groups[-1][2] += char
                else:
                    groups.append([x, y, char])
                previous_x, previous_y, previous_width = x, y, width
            for x, y, text in groups:
                if text == "\u0302":
                    # A standalone combining accent is rendered inconsistently
                    # by presentation apps; use its spacing equivalent.
                    text = "\u02c6"
                    x -= font.getlength(text) / 100
                self.textbox(text, tx + x, ty + y, scss, shapes, name)

    def picture(self, node, shapes, state, clip, name):
        sx, sy, tx, ty = state
        x, y, w, h = [float(node.get(key)) for key in ("x", "y", "width", "height")]
        left, right = sorted((sx * x + tx, sx * (x + w) + tx))
        top, bottom = sorted((sy * y + ty, sy * (y + h) + ty))
        image = io.BytesIO(base64.b64decode(node.get(XLINK + "href").split(",", 1)[1]))
        crop = None
        if clip:
            clip_node = self.clips[re.search(r"#([^)]*)", clip)[1]]
            paths = clip_node.findall(SVG + "path")
            if paths:
                # These paths are circular clip regions around the globes.
                commands = path_commands(paths[0].get("d"))
                assert len([c for c, _ in commands if c == "C"]) == 8
                cx0, cy0, cx1, cy1 = bounds(commands)
                cy0 += self.offset_y
                cy1 += self.offset_y
                crop = ((cx0 - left) / (right - left), (cx1 - right) / (left - right),
                        (cy0 - top) / (bottom - top), (cy1 - bottom) / (top - bottom))
                left, right, top, bottom = cx0, cx1, cy0, cy1
        shape = shapes.add_picture(image, Pt(left), Pt(top), Pt(right - left), Pt(bottom - top))
        shape.name = name
        if sx < 0:
            shape._element.spPr.xfrm.set("flipH", "1")
        if sy < 0:
            shape._element.spPr.xfrm.set("flipV", "1")
        if crop:
            shape.crop_left, shape.crop_right = crop[:2]
            shape.crop_top, shape.crop_bottom = crop[2:] if sy > 0 else crop[2:][::-1]
            shape._element.spPr.prstGeom.set("prst", "ellipse")
        self.counts["pictures"] += 1

    def walk(self, node, shapes, css=None, state=(1, 1, 0, 0), clip=None, name=""):
        if not isinstance(node.tag, str):
            return
        tag = node.tag.removeprefix(SVG)
        if tag in ("defs", "metadata", "clipPath"):
            return
        css = style(node, css or {})
        state = transform(node.get("transform", ""), state)
        clip = node.get("clip-path", clip)
        name = node.get("id", name)
        if name == "patch_1":
            return  # The slide supplies the white background.
        if tag in ("svg", "g"):
            if name.startswith(("patch_", "text_")) and node.get("id"):
                group = shapes.add_group_shape()
                group.name = name
                comments = [child for child in node if child.tag is ET.Comment]
                if comments:
                    group.name = comments[0].text.strip()
                shapes = group.shapes
            for child in node:
                self.walk(child, shapes, css, state, clip, name)
        elif tag == "path":
            self.path(node, shapes, css, state, name)
        elif tag == "text":
            self.text(node, shapes, css, state, name)
        elif tag == "image":
            self.picture(node, shapes, state, clip, name)
        else:
            raise ValueError(f"Unsupported visible SVG element: {tag}")


def presentation(stems, filename, combined=False):
    prs = Presentation()
    prs.slide_width = Pt(1152)
    report = []
    for stem in stems:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        root = ET.parse(SOURCE / (stem + ".svg"), parser=parser).getroot()
        _, _, width, height = map(float, root.get("viewBox").split())
        assert width == 1152
        slide_height = 648 if combined else height
        prs.slide_height = Pt(slide_height)
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
        offset = (slide_height - height) / 2
        converter = Exporter(slide, root, offset)
        converter.walk(root, slide.shapes, state=(1, 1, 0, offset))
        slide.notes_slide.notes_text_frame.text = (
            f"Editable conversion of {stem}.svg. Labels and vector shapes are native "
            "PowerPoint objects. Use Ungroup to edit grouped labels or arrow parts. "
            "Earth imagery and its mesh overlay remain separate cropped PNG pictures. "
            "Arrows preserve SVG paths; they do not automatically reconnect when boxes move. "
            "Install the bundled DejaVu Sans fonts to preserve label spacing."
        )
        report.append({"source": stem + ".svg", "sha256": hashlib.sha256((SOURCE / (stem + '.svg')).read_bytes()).hexdigest(), **converter.counts})
    prs.core_properties.title = "Editable GraphCast–Mamba architecture figures"
    prs.core_properties.subject = "Native shapes, editable labels, and separate globe images"
    prs.save(OUT / filename)
    # Reopen the serialized artifact, including all nested groups and images.
    reopened = Presentation(OUT / filename)
    assert len(reopened.slides) == len(stems)
    for slide, expected in zip(reopened.slides, report):
        counts = {"vector_shapes": 0, "text_boxes": 0, "pictures": 0}
        def check(shapes):
            for shape in shapes:
                if shape.shape_type == 6:
                    check(shape.shapes)
                elif shape.shape_type == 13:
                    assert shape.image.blob
                    counts["pictures"] += 1
                elif shape.has_text_frame and shape.text:
                    counts["text_boxes"] += 1
                else:
                    assert shape._element.spPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}custGeom") is not None
                    counts["vector_shapes"] += 1
        check(slide.shapes)
        assert all(counts[k] == expected[k] for k in counts), (counts, expected)
    return {"file": filename, "slides": report}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = [presentation([stem], stem + ".pptx") for stem in STEMS]
    reports.append(presentation(STEMS, "gc_mamba_editable_figures.pptx", combined=True))
    fonts = OUT / "fonts"
    fonts.mkdir(exist_ok=True)
    for suffix in ("", "-Bold", "-Oblique", "-BoldOblique"):
        shutil.copy2(FONT_DIR / f"DejaVuSans{suffix}.ttf", fonts)
    shutil.copy2("/usr/share/licenses/dejavu-sans-fonts/LICENSE", fonts / "LICENSE.txt")
    (OUT / "export-report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
