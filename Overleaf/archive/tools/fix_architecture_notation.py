"""Correct labels in the manually edited two-slide architecture master.

Usage: python Overleaf/tools/fix_architecture_notation.py INPUT.pptx OUTPUT.pptx
Only the selected text shapes change; other XML and embedded artwork are kept.
Export with LibreOffice using the crops in presentation/notation-fix-report.json.
"""

from pathlib import Path
import re
import sys
from zipfile import ZipFile


def edit_shape(xml, ident, edit):
    pattern = re.compile(r'<p:sp>.*?</p:sp>', re.DOTALL)
    found = []

    def replace(match):
        shape = match.group()
        if re.search(rf'<p:cNvPr id="{ident}"(?:\s|/|>)', shape):
            found.append(ident)
            return edit(shape)
        return shape

    updated = pattern.sub(replace, xml)
    assert found == [ident], (ident, found)
    return updated


def set_texts(shape, texts):
    assert len(re.findall(r'<a:t>.*?</a:t>', shape)) == len(texts)
    values = iter(texts)
    return re.sub(r'<a:t>.*?</a:t>', lambda _: f'<a:t>{next(values)}</a:t>', shape)


def module_label(shape, index):
    # Three lines retain the original 14 pt type within the existing box.
    paragraph = re.search(r'<a:p>.*?</a:p>', shape).group()
    assert re.findall(r'<a:t>(.*?)</a:t>', paragraph) == ['Mamba']
    paragraphs = ''.join(set_texts(paragraph, [text]) for text in
                         ('Mamba', f'module {index}', '(2 layers)'))
    shape = re.sub(r'(<p:txBody>.*?<a:lstStyle/>).*?(</p:txBody>)',
                   lambda m: m[1] + paragraphs + m[2], shape)
    shape = re.sub(r'(<a:off x="\d+" y=")\d+("/>)',
                   lambda m: m[1] + str(99 * 12700) + m[2], shape, count=1)
    shape = re.sub(r'(<a:ext cx="\d+" cy=")\d+("/>)',
                   lambda m: m[1] + str(56 * 12700) + m[2], shape, count=1)
    return shape


def update(source, target):
    with ZipFile(source) as original:
        entries = {info.filename: (info, original.read(info.filename))
                   for info in original.infolist()}
    branch = entries['ppt/slides/slide1.xml'][1].decode()
    for ident, index in ((91, 1), (136, 2)):
        branch = edit_shape(branch, ident, lambda s, i=index: module_label(s, i))
    rollout = entries['ppt/slides/slide2.xml'][1].decode()
    # Preserve the author's X_{0:1} -> Y_1 convention and correction c_1.
    # Initial memory is therefore m_0, followed by m_1, m_2, and m_3.
    rollout = edit_shape(rollout, 232, lambda s: set_texts(
        s, ['Temporal memory  ', 'm', '0', ' = 0']))
    for index, ident in enumerate((233, 234, 235), 1):
        rollout = edit_shape(rollout, ident, lambda s, i=index: re.sub(
            r'name="Temporal memory m_\d+"', f'name="Temporal memory m_{i}"',
            set_texts(s, ['m', str(i)])))
    replacements = {'ppt/slides/slide1.xml': branch.encode(),
                    'ppt/slides/slide2.xml': rollout.encode()}
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, 'w') as output:
        for name, (info, data) in entries.items():
            output.writestr(info, replacements.get(name, data))
    with ZipFile(target) as output:
        for name, (_, data) in entries.items():
            if name not in replacements:
                assert output.read(name) == data, name
    print(f'Saved {target}; all non-slide package entries preserved.')


if __name__ == '__main__':
    update(*sys.argv[1:])
