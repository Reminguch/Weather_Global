from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture()
def stage_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "src.data_operations.staging.stage_wb2_era5_yearly_append"
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "xarray", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "zarr", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        types.SimpleNamespace(Timedelta=lambda **_: object()),
    )

    import importlib

    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def _report(stage_module, output: Path, *, write_mode: str = "create"):
    return stage_module.RunReport(
        uri="gs://example/source.zarr",
        output=str(output),
        start_year=2016,
        end_year=2016,
        chunk_time=120,
        include_tisr=True,
        overwrite=False,
        dry_run=False,
        source_year_min=1959,
        source_year_max=2023,
        existing_years=[],
        requested_years=[2016],
        skipped_years=[],
        appended_years=[2016],
        rebuild_required=False,
        write_mode=write_mode,
        started_at_unix=1.0,
        ended_at_unix=2.0,
        elapsed_sec=1.0,
    )


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_canonical_report_path_uses_stage_report_suffix(stage_module) -> None:
    output = Path("/tmp/wb2_res1_levels13_train.zarr")

    assert stage_module._canonical_report_path(output) == Path(
        "/tmp/wb2_res1_levels13_train.zarr.stage_report.json"
    )


def test_save_report_replaces_canonical_report(stage_module, tmp_path: Path) -> None:
    output = tmp_path / "wb2_res1_levels13_train.zarr"
    report_path = stage_module._canonical_report_path(output)
    report_path.write_text('{"old": true}\n', encoding="utf-8")

    stage_module._save_report(report_path, _report(stage_module, output, write_mode="append"))

    payload = _read_json(report_path)
    assert "old" not in payload
    assert payload["output"] == str(output)
    assert payload["write_mode"] == "append"
    assert not report_path.with_name(f".{report_path.name}.tmp").exists()


def test_save_report_merges_and_removes_auxiliary_same_output_reports(
    stage_module, tmp_path: Path
) -> None:
    output = tmp_path / "wb2_res1_levels13_train.zarr"
    report_path = stage_module._canonical_report_path(output)
    aux_report = tmp_path / "wb2_res1_levels13_train.zarr.repair_2016_2018_report.json"
    other_report = tmp_path / "wb2_res1_levels13_2018_redownload.zarr.stage_report.json"

    aux_report.write_text('{"target": "train", "years": [2016, 2018]}\n', encoding="utf-8")
    other_report.write_text('{"output": "other"}\n', encoding="utf-8")

    stage_module._save_report(report_path, _report(stage_module, output))

    payload = _read_json(report_path)
    assert payload["write_mode"] == "create"
    assert payload["superseded_reports"] == [
        {
            "path": str(aux_report),
            "report": {"target": "train", "years": [2016, 2018]},
        }
    ]
    assert not aux_report.exists()
    assert other_report.exists()


def test_save_report_preserves_existing_superseded_reports(stage_module, tmp_path: Path) -> None:
    output = tmp_path / "wb2_res1_levels13_train.zarr"
    report_path = stage_module._canonical_report_path(output)
    old_report = tmp_path / "old.repair_report.json"
    old_report.write_text('{"target": "old"}\n', encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "output": str(output),
                "superseded_reports": [
                    {
                        "path": str(old_report),
                        "report": {"target": "old"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stage_module._save_report(report_path, _report(stage_module, output, write_mode="append"))

    payload = _read_json(report_path)
    assert payload["write_mode"] == "append"
    assert payload["superseded_reports"] == [
        {
            "path": str(old_report),
            "report": {"target": "old"},
        }
    ]
    assert old_report.exists()
