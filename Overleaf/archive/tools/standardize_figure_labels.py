#!/usr/bin/env python3
"""Apply the approved terminology to archived paper figures, preserving curves.

Requires PyMuPDF and numpy. Run from any directory. The input PDFs in
figure_label_sources/ preserve the artwork used before this terminology pass.
Only the specified label regions may change; a pixel comparison enforces this.
Do not use regenerated PNG previews as a source for the paper's vector PDFs.

Maintained plots now get their labels and paper-scale typography directly from
their generators (see Graphs/FIGURE_TYPOGRAPHY.md). Skip these by default so a
terminology pass cannot restore the archived, smaller fonts. The optional
--include-regenerated-plots flag reproduces the historical label-only pass.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCES = Path(__file__).resolve().parent / 'figure_label_sources'
FONT = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'
INK = (36 / 255, 52 / 255, 71 / 255)
BLUE = (36 / 255, 106 / 255, 154 / 255)
ORANGE = (180 / 255, 92 / 255, 22 / 255)
FONT_OBJECT = fitz.Font(fontfile=FONT)
REGENERATED_PLOTS = {
    'Graphs/add_mamba.pdf',
    'Graphs/city_traces/cities_Paris_NYC_20220115_compact.pdf',
    'Graphs/main_v2/channels_improved_k22.pdf',
    'Graphs/main_v2/exact_gc_loss.pdf',
    'Graphs/main_v2/memory_reset.pdf',
    'Graphs/main_v2/memory_reset_all.pdf',
    'Graphs/main_v2/training_ar_horizon.pdf',
    'Graphs/main_v3/overview_7yr_coverage.pdf',
    'Graphs/main_v3/overview_7yr_heatmap.pdf',
    'Graphs/main_v3/training_ar_horizon_15yr_closedloop.pdf',
}


def lines(page):
    result = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            result.append({
                'text': ''.join(s['text'] for s in line['spans']).replace('\xa0', ' '),
                'rect': fitz.Rect(line['bbox']),
                'size': line['spans'][0]['size'],
                'rotate': 90 if line['dir'] == (0., -1.) else 0,
            })
    return result


class Labels:
    def __init__(self, relative):
        self.relative = relative
        self.source = SOURCES / relative
        self.doc = fitz.open(self.source)
        self.page = self.doc[0]
        self.original_lines = lines(self.page)
        self.regions = []
        self.pending = []
        self.before = self.page.get_pixmap(alpha=False)

    def replace(self, old, new, *, rect=None, size=None, align=1, color=(0, 0, 0)):
        found = [line for line in self.original_lines if line['text'] == old]
        if not found:
            raise ValueError(f'{self.relative}: missing label {old!r}')
        for line in found:
            old_rect = line['rect']
            self.page.add_redact_annot(old_rect, fill=False, cross_out=False)
            self.regions.append(list(old_rect))
            target = fitz.Rect(rect(old_rect) if callable(rect) else rect) if rect is not None else fitz.Rect(old_rect)
            self.pending.append((target, new, size or line['size'], line['rotate'], align, color))

    def region(self, rect, text, size, *, align=1, color=(0, 0, 0), raster=False):
        rect = fitz.Rect(rect)
        if raster:
            self.page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
        else:
            self.page.add_redact_annot(rect, fill=False, cross_out=False)
        self.regions.append(list(rect))
        self.pending.append((rect, text, size, 0, align, color))

    def finish(self):
        # Never erase plotted lines, patches, or raster image data.
        self.page.apply_redactions(images=0, graphics=0)
        for rect, text, size, rotate, align, color in self.pending:
            rect = fitz.Rect(rect)
            # Give the replacement room for the original font's ascenders.
            if rotate:
                rect.x0 -= 1.5
                rect.x1 += 1.5
            else:
                rect.y0 -= .7
                rect.y1 += 2.5
            logical_width = rect.height if rotate else rect.width
            longest = max(FONT_OBJECT.text_length(t, fontsize=size) for t in text.split('\n'))
            if longest > logical_width:
                size *= logical_width / longest * .99
            for _ in range(30):
                shape = self.page.new_shape()
                remaining = shape.insert_textbox(rect, text, fontname='TermSans',
                    fontfile=FONT, fontsize=size, color=color, align=align,
                    rotate=rotate, lineheight=1.10)
                if remaining >= 0:
                    shape.commit()
                    break
                size *= .96
            else:
                raise ValueError(f'Cannot fit label {text!r}')
            self.regions.append(list(rect))
        target = ROOT / self.relative
        self.doc.save(target, garbage=4, deflate=True)
        self.doc.close()
        after_doc = fitz.open(target)
        after = after_doc[0].get_pixmap(alpha=False)
        assert (self.before.width, self.before.height) == (after.width, after.height)
        old = np.frombuffer(self.before.samples, dtype=np.uint8).reshape(self.before.height, self.before.width, 3)
        new = np.frombuffer(after.samples, dtype=np.uint8).reshape(after.height, after.width, 3)
        allowed = np.zeros(old.shape[:2], dtype=bool)
        for x0, y0, x1, y1 in self.regions:
            allowed[max(0,int(y0)-4):min(after.height,int(y1)+5),
                    max(0,int(x0)-4):min(after.width,int(x1)+5)] = True
        difference = np.any(old != new, axis=2)
        outside = int(np.count_nonzero(difference & ~allowed))
        if outside:
            raise ValueError(f'{self.relative}: {outside} pixels changed outside label regions')
        # Refresh previews only where they already exist.
        png = target.with_suffix('.png')
        if png.exists():
            after_doc[0].get_pixmap(matrix=fitz.Matrix(2,2), alpha=False).save(png)
        if self.relative.startswith('Pics/gc_mamba_architecture/'):
            target.with_suffix('.svg').write_text(after_doc[0].get_svg_image(text_as_path=False))
        return {
            'path': self.relative,
            'input_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
            'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'changed_pixels_outside_label_regions': outside,
            'label_regions': self.regions,
        }


def ordinary_labels(editor):
    for line in editor.original_lines:
        old = line['text']
        rect = line['rect']
        if old in ('Forecast lead (days)', 'Forecast lead'):
            editor.replace(old, 'Forecast lead time (days)' if '(days)' in old else 'Forecast lead time',
                rect=lambda r: ((r.x0+r.x1)/2-63,r.y0,(r.x0+r.x1)/2+63,r.y1), size=9)
            break
    if any(l['text']=='Training state / evaluation state' for l in editor.original_lines):
        r=next(l['rect'] for l in editor.original_lines if l['text']=='Training state / evaluation state')
        editor.replace('Training state / evaluation state','Training/forecasting memory policy',
                       rect=(145,r.y0,323,r.y1),size=8)
    for old,new in [('Vertical velocity','Pressure vertical velocity'),
                    ('6 h precipitation','6 h accumulated precipitation')]:
        for line in editor.original_lines:
            if line['text']==old:
                r=line['rect']
                editor.replace(old,new,rect=(1,r.y0,r.x1,r.y1),align=2,size=8)
    for line in editor.original_lines:
        old=line['text']
        if 'BCG = ' in old:
            r=line['rect']
            editor.replace(old,old.replace('BCG = ','B,C groups: '),
                           rect=(r.x0,r.y0,min(r.x0+131,466),r.y1),align=0,size=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-regenerated-plots', action='store_true',
                        help='Restore archived plot layouts as part of the historical terminology pass.')
    args = parser.parse_args()
    reports=[]
    for source in sorted(SOURCES.rglob('*.pdf')):
        relative=source.relative_to(SOURCES).as_posix()
        if relative in REGENERATED_PLOTS and not args.include_regenerated_plots:
            print(f'Skipping maintained plot: {relative}')
            continue
        editor=Labels(relative)
        ordinary_labels(editor)
        if relative.endswith('02_correction_branch.pdf'):
            editor.region((4,53,148,77),'Xₜ → grid features',13,align=0,color=BLUE)
            editor.region((957,166,1059,191),'Forecast correction cₜ',11,align=2,color=BLUE)
            for rect,index in [((335,97,442,137),1),((657,97,764,137),2)]:
                editor.region(rect,f'Mamba module {index}\n(2 layers)',12,color=INK)
            for rect,index in [((166,98,293,140),1),((485,98,612,140),2)]:
                editor.region(rect,f'Graph message-\npassing step {index}',12,color=INK)
        elif relative.endswith('03_temporal_rollout.pdf'):
            editor.replace('GraphCast Mamba','GraphCast–Mamba',rect=(420,20,732,47),size=21,color=INK)
            # Retain mathematical subscripts with Unicode subscript glyphs.
            editor.region((50,90,239,107),'Temporal memory state m₋₁ = 0',11,align=0,color=ORANGE)
            for cx in (230.4,576,921.6):
                editor.region((cx-95,157,cx+95,190),'Mamba correction branch Rᵩ',12,color=INK)
        elif relative.endswith('exact_gc_loss.pdf'):
            editor.replace('GraphCast baseline','GraphCast',align=0)
            editor.replace('3× Mamba LR + SWA','GraphCast–Mamba',rect=(257,339.7,375,349),align=0,size=8)
            editor.replace('(c) 2 m temperature error','(c) 2 m temperature RMSE',align=0)
            editor.replace('(d) 2m temperature error reduction','(d) 2 m temperature RMSE reduction',align=0)
            editor.region((80,354,388,369),
                'Mamba inner dimension 16; one B,C group; SWA post-warm-up updates 500–2,000',7)
        elif relative.endswith('overview_7yr_heatmap.pdf'):
            editor.replace('RMSE reduction relative to GraphCast-small','RMSE reduction relative to GraphCast',align=0)
        elif relative.endswith('training_ar_horizon_15yr_closedloop.pdf'):
            editor.replace('GraphCast paper-formula weighted','Approx. paper-formula weighted MSE',size=8.5)
            editor.replace('MSE reduction (%)','reduction (%)',size=9)
        elif relative.endswith('training_ar_horizon.pdf'):
            editor.replace('Autoregressive training horizon','Training tail label K',align=0)
        elif relative.endswith('channels_improved_k22.pdf'):
            editor.replace('Breadth of long-range gains (K = 22)','Channel coverage (K = 22)',align=0)
        elif '/city_traces/' in relative:
            for line in editor.original_lines:
                old=line['text']; r=line['rect']
                if old=='T2M (°C)':
                    editor.replace(old,'T2m (°C)')
                elif old in ('6 h precip. (mm)','6 h precipitation (mm)'):
                    # A two-line vertical label fits the same row without entering the data panels.
                    cy=(r.y0+r.y1)/2
                    editor.replace(old,'6 h accumulated\nprecipitation (mm)',
                                   rect=(max(0,r.x0-3),cy-45,r.x1+1,cy+45),size=7)
                elif old=='Ours':
                    editor.replace(old,'GraphCast–Mamba',rect=(r.x0,r.y0,r.x0+83,r.y1),align=0,size=8)
        elif relative=='Graphs/add_mamba.pdf':
            # This archived PDF contains a raster plot. Cover existing label areas
            # with vector text; preserve every plot panel and its data pixels.
            editor.region((160,20,2390,58),'Full-year 2022 forecasts on the common 1.5° verification grid',22,raster=True)
            for rect,label,color in [
                ((153,106,423,128),'GraphCast (0.25°)',(0.12,0.64,0.31)),
                ((153,132,423,154),'GraphCast–Mamba (K=22)',(0.15,0.38,0.62)),
                ((153,156,423,178),'GraphCast',(0.4,0.4,0.4)),
            ]:
                editor.region(rect,label,14,align=0,raster=True)
            for x,label in [(1255,'U10'),(1875,'V10')]:
                editor.region((x+130,1046,x+455,1080),label,21,raster=True)
            for x in [0,620,1240,1860]:
                for y in [515,1006,1497]:
                    editor.region((x+242,y,x+446,y+30),'Forecast lead time (days)',14,raster=True)
        reports.append(editor.finish())
        print(relative)
    (ROOT/'tools'/'figure_label_changes.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    main()
