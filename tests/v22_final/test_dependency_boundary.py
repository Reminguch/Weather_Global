from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_MARKERS = (
    "scripts.training." + "full_mamba_",
    "scripts/training/" + "full_mamba_",
    "scripts.training." + "gcmamba_v2_" + "diagnostics",
    "scripts/training/" + "gcmamba_v2_" + "diagnostics",
)


def _maintained_v22_files() -> list[Path]:
    files = list((ROOT / "src/models/mamba/v22_final").rglob("*.py"))
    files.extend(
        [
            ROOT / "scripts/training/train_v22_final.py",
            ROOT / "scripts/training/build_v22_final_swa.py",
            ROOT / "scripts/analyze_models/eval_v22_final.py",
        ]
    )
    files.extend((ROOT / "scripts/analyze_models").glob("*v22_final*"))
    files.extend((ROOT / "scripts/experiments").glob("*v22_final*"))
    files.extend((ROOT / "tests/v22_final").rglob("*.py"))
    return sorted(set(files))


def test_maintained_v22_files_do_not_reference_archive_folders() -> None:
    violations = {}
    for path in _maintained_v22_files():
        text = path.read_text(encoding="utf-8")
        matches = [marker for marker in ARCHIVE_MARKERS if marker in text]
        if matches:
            violations[str(path.relative_to(ROOT))] = matches

    assert not violations, f"v22_final archive dependencies found: {violations}"
