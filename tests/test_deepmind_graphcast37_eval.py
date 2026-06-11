from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from scripts.analyze_models import unified_resolution_eval as ure
from scripts.analyze_models import plot_res1_bptt16_k_sweep_lead_curves as res1_plot
from src.data_operations.staging.stage_wb2_graphcast37_window import PRESSURE_LEVELS_37
from src.models.graphcast.evaluation.device_resolution_eval import build_metric_spec
from src.models.graphcast.training.core.prepared_array import PreparedArrayStore
from src.models.graphcast.training.core.prepared_data import resolution_tag


def test_resolution_tag_supports_quarter_degree() -> None:
    assert resolution_tag(0.25) == "res0p25"


def test_unified_resolution_eval_cli_accepts_float_resolution_and_metric_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "unified_resolution_eval.py",
            "--resolutions",
            "0.25",
            "--metric-grid-resolution",
            "1",
            "--metric-variables",
            "2m_temperature",
        ],
    )

    args = ure.parse_args()

    assert args.resolutions == [0.25]
    assert args.metric_grid_resolution == 1.0
    assert args.metric_variables == ["2m_temperature"]


def test_residual_eval_defaults_to_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["unified_resolution_eval.py"])

    args = ure.parse_args()

    assert args.residual_eval_semantics == "rollout"
    assert ure.RESIDUAL_EVAL_SEMANTICS == "rollout"


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "scripts/analyze_models/run_resolution_eval_array.slurm",
        ROOT / "scripts/analyze_models/submit_resolution_eval_array.sh",
        ROOT / "scripts/analyze_models/submit_small_mamba_freeze_release_eval.sh",
    ],
)
def test_slurm_wrappers_default_residual_eval_to_rollout(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    assert 'RESIDUAL_EVAL_SEMANTICS="${RESIDUAL_EVAL_SEMANTICS:-rollout}"' in text
    assert 'RESIDUAL_EVAL_SEMANTICS="${RESIDUAL_EVAL_SEMANTICS:-teacher_forced_training_equivalent}"' not in text


def test_res1_lead_plotter_rejects_teacher_forced_residual_rows(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "family": "residual_mamba",
                "variant": "official_res1_gcsmall_residual_mamba_tc2_di128_ds64_k1_bptt16_bs1_mp2_conservative",
                "res": 1,
                "lead_days": 0.25,
                "lead_steps": 1,
                "metric_kind": "weighted_allvars",
                "variable": "",
                "value": 0.9,
                "residual_eval_semantics": "teacher_forced_training_equivalent",
            }
        ]
    ).to_csv(shard_dir / "resolution_eval_residual_mamba_res1_warm_res1grid_k1_40.csv", index=False)

    with pytest.raises(ValueError, match="teacher-forced residual_mamba"):
        res1_plot._load_mamba_rows(tmp_path, "shards/*.csv")

    rows = res1_plot._load_mamba_rows(tmp_path, "shards/*.csv", allow_teacher_forced_residual=True)
    assert len(rows) == 1


def test_res1_lead_plotter_accepts_explicit_mamba_csvs(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "resolution_eval_gc_mamba_res1.csv"
    pd.DataFrame(
        [
            {
                "family": "gc_mamba",
                "variant": "official_res1_gcsmall_gc_mamba_tc2_di128_ds64_k1_bptt16_bs1_optimal",
                "res": 1,
                "lead_days": 0.25,
                "lead_steps": 1,
                "metric_kind": "weighted_allvars",
                "variable": "",
                "value": 1.0,
            }
        ]
    ).to_csv(csv_path, index=False)

    rows = res1_plot._load_mamba_rows(tmp_path / "unused", "missing/*.csv", mamba_csvs=[csv_path])

    assert len(rows) == 1
    assert rows.iloc[0]["curve"] == "GC-Mamba k1"


def test_checkpoint_resolution_parser_accepts_deepmind_graphcast37_filename() -> None:
    ckpt_path = Path(
        "data/graphcast/graphcast/params/"
        "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - "
        "mesh 2to6 - precipitation input and output.npz"
    )

    assert ure._parse_res_from_path(ckpt_path, {}) == 0.25


def test_metric_variables_filter_rmse_rows_to_2m_temperature() -> None:
    rows: list[dict] = []
    entry = ure.ModelEntry(
        family="graphcast",
        variant="GraphCast37",
        model_type="graphcast",
        di=None,
        res=0.25,
        ckpt_path=Path("GraphCast37.npz"),
        run_name="GraphCast37",
    )

    ure._append_metric_rows(
        rows,
        entry,
        lead_days=[1.0],
        lead_steps=[4],
        metrics=["rmse_k"],
        eval_mode="cold",
        weighted_by_day={1.0: 99.0},
        per_variable_by_day={"temperature": {1.0: 3.0}},
        rmse_by_day={"2m_temperature": {1.0: 2.0}},
        n_by_day={1.0: 7},
        warmup_steps=24,
        trunk_steps=32,
        eval_metadata={
            "metric_variables": "2m_temperature",
            "metric_grid_resolution": 1.0,
            "stats_dir": "stats_graphcast_37",
        },
    )

    assert [(row["metric_kind"], row["variable"], row["value"]) for row in rows] == [
        ("rmse_k", "2m_temperature", 2.0)
    ]
    assert rows[0]["metric_grid_resolution"] == 1.0
    assert rows[0]["metric_variables"] == "2m_temperature"


def test_device_metric_spec_filters_to_2m_temperature() -> None:
    coords = {
        "batch": np.arange(1),
        "time": np.arange(1),
        "lat": np.asarray([90.0, 89.0], dtype=np.float32),
        "lon": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    targets = xr.Dataset(
        {
            "2m_temperature": (("batch", "time", "lat", "lon"), np.ones((1, 1, 2, 2), dtype=np.float32)),
            "mean_sea_level_pressure": (
                ("batch", "time", "lat", "lon"),
                np.ones((1, 1, 2, 2), dtype=np.float32),
            ),
        },
        coords=coords,
    )

    spec = build_metric_spec(
        targets,
        stats={"diffs_stddev_by_level": xr.Dataset()},
        res_grid_lats=xr.DataArray(coords["lat"], dims=["lat"]),
        res_grid_lons=xr.DataArray(coords["lon"], dims=["lon"]),
        per_variable_weights={"2m_temperature": 1.0, "mean_sea_level_pressure": 1.0},
        max_lead_steps=1,
        metric_variables=("2m_temperature",),
        nyc_lat=40.7,
        nyc_lon=286.0,
        nyc_output_name="2m_temperature_nyc",
    )

    assert [var.output_name for var in spec.variables] == ["2m_temperature"]


def test_prepared_store_validation_accepts_37_pressure_levels(tmp_path: Path) -> None:
    store = tmp_path / "res0p25"
    (store / "coords").mkdir(parents=True)
    (store / "vars").mkdir()

    np.save(store / "coords" / "time.npy", np.asarray(["2022-01-01T00", "2022-01-01T06"], dtype="datetime64[ns]"))
    np.save(store / "coords" / "lat.npy", np.asarray([90.0, 89.75], dtype=np.float32))
    np.save(store / "coords" / "lon.npy", np.asarray([0.0, 0.25], dtype=np.float32))
    np.save(store / "coords" / "level.npy", np.asarray(PRESSURE_LEVELS_37, dtype=np.int64))
    np.save(store / "vars" / "2m_temperature.npy", np.ones((2, 2, 2), dtype=np.float32))
    metadata = {
        "prepared_array_format_version": 1,
        "resolution": 0.25,
        "pressure_levels": PRESSURE_LEVELS_37,
        "coords": {
            "time": {"shape": [2], "dtype": "datetime64[ns]"},
            "lat": {"shape": [2], "dtype": "float32"},
            "lon": {"shape": [2], "dtype": "float32"},
            "level": {"shape": [37], "dtype": "int64"},
        },
        "variables": {
            "2m_temperature": {
                "dims": ["time", "lat", "lon"],
                "shape": [2, 2, 2],
                "dtype": "float32",
            }
        },
    }
    (store / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    task_cfg = SimpleNamespace(
        pressure_levels=PRESSURE_LEVELS_37,
        input_variables=("2m_temperature",),
        target_variables=("2m_temperature",),
        forcing_variables=(),
    )

    PreparedArrayStore(store).validate(resolution=0.25, task_cfg=task_cfg)
