"""Apply 25% transparency to rear input globes, preserving their layout.

Usage: python scripts/style_architecture_globe_pairs.py INPUT.pptx OUTPUT.pptx
Requires python-pptx (available under .cache/presentation-python).
Use the master with the original globe spacing and no white rim. The Earth
and mesh stay editable, with native picture transparency and unchanged pixels.
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.oxml.xmlchemy import OxmlElement


REAR_IDS = (101, 118, 119, 120, 144, 154)
OPACITY = 75000


def update(source, target):
    prs = Presentation(source)
    assert len(prs.slides) == 2, "Expected the edited two-slide master"
    slide = prs.slides[1]
    assert not any(s.name.startswith("Input globe separation ") for s in slide.shapes), (
        "Restore the original globe layout before applying transparency"
    )
    branch_xml = prs.slides[0]._element.xml
    preserved = {
        shape.shape_id: shape._element.xml
        for shape in slide.shapes if shape.shape_id not in REAR_IDS
    }
    shapes = {s.shape_id: s for s in slide.shapes}
    for ident in REAR_IDS:
        blip = shapes[ident]._element.blipFill.blip
        for effect in list(blip):
            if effect.tag == OxmlElement("a:alphaModFix").tag:
                blip.remove(effect)
        alpha = OxmlElement("a:alphaModFix")
        alpha.set("amt", str(OPACITY))
        # DrawingML places image effects before the optional extension list.
        blip.insert(0, alpha)

    assert prs.slides[0]._element.xml == branch_xml
    for shape in slide.shapes:
        if shape.shape_id in preserved:
            assert shape._element.xml == preserved[shape.shape_id], shape.shape_id
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    prs.save(target)
    check = Presentation(target)
    for shape in check.slides[1].shapes:
        if shape.shape_id in REAR_IDS:
            effect = shape._element.blipFill.blip.xpath("./a:alphaModFix")[0]
            assert effect.get("amt") == str(OPACITY)
    print(f"Saved {target}; rear globes 25% transparent; preserved {len(preserved)} other objects.")


if __name__ == "__main__":
    update(*sys.argv[1:])
