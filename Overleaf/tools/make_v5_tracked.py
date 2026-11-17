#!/usr/bin/env python3
"""Generate the initial reviewable v5 from its preserved before/after sources.

v5.tex is the editable canonical source after generation. v5_clean.tex hides
its deletions; it does not contain a second independently editable manuscript.
"""
from pathlib import Path
from difflib import SequenceMatcher
import json
import re

PAPER = Path(__file__).resolve().parents[1]
ARCHIVE = PAPER / 'archive/v5_pilot_20261114'
OLD = (ARCHIVE / 'v4_before_compression.tex').read_text()
NEW = (ARCHIVE / 'v5_clean_relative_errors.tex').read_text()
records = []


def group(text, start, opening='{', closing='}'):
    assert text[start] == opening
    depth = 1
    pos = start + 1
    while depth:
        if text[pos] == '\\':
            pos += 2
            continue
        depth += (text[pos] == opening) - (text[pos] == closing)
        pos += 1
    return text[start + 1:pos - 1], pos


def strike_text(text):
    """Remove long formatting groups, preserving wording and inline mathematics."""
    result = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '$':
            end = i + 1
            while end < len(text) and (text[end] != '$' or text[end - 1] == '\\'):
                end += 1
            result.append(text[i:end + 1])
            i = end + 1
        elif c == '{':
            body, i = group(text, i)
            result.append(strike_text(body))
        elif c == '\\':
            match = re.match(r'\\[A-Za-z]+\*?|\\.', text[i:])
            command = match[0]
            end = i + len(command)
            if command in (r'\vfourchange', r'\vfourred', r'\codexedit', r'\textbf', r'\textit', r'\emph'):
                body, i = group(text, end)
                result.append(strike_text(body))
                continue
            while end < len(text) and text[end] in '[{':
                _, end = group(text, end, text[end], ']' if text[end] == '[' else '}')
            item = text[i:end]
            if command.rstrip('*') in (r'\citep', r'\citet', r'\cite', r'\ref', r'\eqref'):
                item = r'\mbox{' + item + '}'
            result.append(item)
            i = end
        else:
            result.append(c)
            i += 1
    return ''.join(result).strip()


def caption_parts(text):
    start = text.index(r'\caption{')
    body, end = group(text, start + len(r'\caption'))
    return start, body, end


def tokens(text):
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        match = re.match(r'\s*\\begin\{(figure|table|align)\}', lines[i])
        if match:
            ending = r'\end{' + match[1] + '}'
            j = i
            while ending not in lines[j]:
                j += 1
            out.append(''.join(lines[i:j + 1]))
            i = j + 1
        else:
            out.append(lines[i])
            i += 1
    return out


def tracked_only(content):
    return '\\ifvfivechanges\n' + content + '\n\\fi\n'


def removed_paragraph(text):
    return tracked_only(r'\par\begingroup\color{vfiveDeleted}\noindent' + '\n'
                        + r'\sout{' + strike_text(text) + '}' + '\n'
                        + r'\par\endgroup')


def old_float(text, kind, moved=False):
    start, cap, end = caption_parts(text)
    content = text[:start] + r'\captionof*{' + kind + '}{\\sout{' + strike_text(cap) + '}}' + text[end:]
    content = re.sub(r'\\begin\{' + kind + r'\}(?:\[[^\]]*\])?', r'\\begin{center}', content)
    content = content.replace(r'\end{' + kind + '}', r'\end{center}')
    content = re.sub(r'\\label\{[^}]+\}', '', content)
    content = content.replace(r'\begingroup\color{red}', r'\begingroup')
    # Strike every data/header cell without changing table alignment or rules.
    if kind == 'table':
        rows = []
        for line in content.splitlines():
            if line.rstrip().endswith(r'\\'):
                cells = line.rstrip()[:-2].split('&')
                marked = []
                for cell in cells:
                    value = cell.strip()
                    if not value:
                        marked.append('')
                    elif value.startswith(r'\multicolumn'):
                        pos = len(r'\multicolumn')
                        _, pos = group(value, pos)
                        _, pos = group(value, pos)
                        body, end = group(value, pos)
                        marked.append(value[:pos] + '{\\sout{' + strike_text(body) + '}}' + value[end:])
                    elif value.startswith(r'\shortstack'):
                        body, end = group(value, len(r'\shortstack'))
                        marked.append(r'\shortstack{' + r'\\'.join(r'\sout{' + strike_text(part) + '}' for part in body.split(r'\\')) + '}' + value[end:])
                    else:
                        marked.append(r'\sout{' + strike_text(value) + '}')
                line = ' & '.join(marked) + r' \\'
            rows.append(line)
        content = '\n'.join(rows)
    note = ('Moved to Appendix A.2; original placement shown below.' if moved
            else 'Previous table: loss reductions (replaced by relative errors below).')
    return tracked_only(r'\par\begingroup\color{vfiveDeleted}' + '\n'
                        + r'\noindent\textit{' + note + '}\par\n' + content
                        + '\n\\par\\endgroup')


def old_math(text):
    body = text.split(r'\begin{align}', 1)[1].split(r'\end{align}', 1)[0]
    rows = [re.sub(r'\s+', ' ', row.replace('&', '')).strip()
            for row in body.split(r'\\') if row.strip()]
    lines = [r'\text{\sout{\ensuremath{' + row + '}}}' for row in rows]
    return tracked_only(r'\par\begingroup\color{vfiveDeleted}' + '\n'
                        + r'\noindent\textit{Detailed equations moved to Appendix A.2.}' + '\n'
                        + '\\begin{gather*}\n' + '\\\\\n'.join(lines)
                        + '\n\\end{gather*}\n\\endgroup')


def old_item(text):
    if not text.strip() or text.lstrip().startswith('%'):
        return ''
    if r'\begin{figure}' in text:
        return old_float(text, 'figure', moved=True)
    if r'\begin{table}' in text:
        return old_float(text, 'table', moved=r'\label{tab:memory_controls}' in text)
    if r'\begin{align}' in text:
        return old_math(text)
    if text.strip() == r'\FloatBarrier':
        return tracked_only(r'\par\noindent\textit{V5 layout change: removed the forced float barrier here.}\par')
    return removed_paragraph(text)


def new_item(text):
    if not text.strip() or text.lstrip().startswith('%'):
        return text
    if text.lstrip().startswith((r'\section', r'\subsection', r'\label', r'\paragraph')):
        return text
    if r'\begin{' in text:
        return text
    return r'\vfiveadded{' + text.rstrip('\n') + '}\n'


def changed_figure(old, new):
    _, oldcap, _ = caption_parts(old)
    start, newcap, end = caption_parts(new)
    if oldcap == newcap:
        return '% V5: byte-identical figure asset renamed to match its new figure number.\n' + new
    removed = tracked_only(r'\caption*{\color{vfiveDeleted}\sout{' + strike_text(oldcap) + '}}')
    return new[:start] + removed + r'\caption{\vfiveadded{' + newcap + '}}' + new[end:]


oldbody = OLD.split(r'\begin{document}', 1)[1]
preamble, newbody = NEW.split(r'\begin{document}', 1)
left, right = tokens(oldbody), tokens(newbody)
pieces = []
for tag, i, j, k, l in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
    if tag == 'equal':
        pieces.extend(right[k:l])
        continue
    olditems, newitems = left[i:j], right[k:l]
    records.append({'operation': tag, 'before': ''.join(olditems), 'after': ''.join(newitems)})
    def figure_label(item):
        if r'\begin{figure}' not in item:
            return None
        found = re.findall(r'\\label\{([^}]+)\}', item)
        return found[-1] if found else None
    oldfigures = {figure_label(item): item for item in olditems if figure_label(item)}
    newlabels = {figure_label(item) for item in newitems if figure_label(item)}
    paired = set(oldfigures) & newlabels
    pieces.extend(old_item(item) for item in olditems if figure_label(item) not in paired)
    if any(r'\subsection{Architecture and memory implementation}' in item for item in newitems):
        pieces.append(tracked_only(r'\par\noindent\textit{V5 relocation: the architecture details, training schematic, and memory-policy table below were moved from Sections 3 and 4.5.}\par'))
    pieces.extend(changed_figure(oldfigures[figure_label(item)], item)
                  if figure_label(item) in paired else new_item(item) for item in newitems)

support = r'''
% V5 review layer. v5.tex shows changes; v5_clean.tex hides this layer.
\usepackage[normalem]{ulem}
\definecolor{vfiveDeleted}{RGB}{150,45,45}
\definecolor{vfiveAdded}{RGB}{0,105,65}
\newif\ifvfivechanges
\ifdefined\vfiveclean\vfivechangesfalse\else\vfivechangestrue\fi
% Clean view: retain revision highlighting in blue, with all deletions hidden.
\ifvfivechanges\else\colorlet{red}{blue}\fi
\newcommand{\vfiveidentity}[1]{#1}
\DeclareRobustCommand{\vfiveadded}[1]{\ifvfivechanges{\color{vfiveAdded}%
  \let\vfourchange\vfiveidentity\let\vfourred\vfiveidentity#1}\else{\color{blue}%
  \let\vfourchange\vfiveidentity\let\vfourred\vfiveidentity#1}\fi}
\pdfstringdefDisableCommands{\def\vfiveadded#1{#1}\def\sout#1{#1}}
'''
result = preamble + support + '\n\\begin{document}' + ''.join(pieces)
# Explain the markup without changing the scientific abstract or title.
legend = tracked_only(r'\begin{center}\small\textit{V5 review copy: \sout{struck text} is removed or replaced; \textcolor{vfiveAdded}{green text} is the replacement. Moves to Appendix A.2 are labelled. Compile v5\_clean.tex for the page-limit view.}\end{center}')
result = result.replace(r'\maketitle', r'\maketitle' + '\n' + legend, 1)
(PAPER / 'v5.tex').write_text(result)
(PAPER / 'v5_clean.tex').write_text('% Clean view: deleted text hidden; retained revision text blue.\n\\def\\vfiveclean{1}\n\\input{v5.tex}\n')
(ARCHIVE / 'tracked_change_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'Generated v5.tex with {len(records)} tracked change blocks and a shared-source clean view.')
