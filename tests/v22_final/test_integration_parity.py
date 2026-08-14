from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.models.mamba.v22_final.evaluation import compare_metric_outputs
from src.models.mamba.v22_final.metrics import V22FinalMetricAccumulator


FIXTURE_PATH = Path(__file__).parent / "fixtures/evaluation_metric_contract_v1.json"


def _dataset(values: list[float]) -> xr.Dataset:
    array = np.asarray(values, dtype=np.float32)[None, :, None, None]
    return xr.Dataset(
        {"x": (("batch", "time", "lat", "lon"), array)},
        coords={"batch": [0], "time": [0, 1], "lat": [0.0], "lon": [0.0]},
    )


def test_evaluation_metrics_match_frozen_v22_contract() -> None:
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    accumulator = V22FinalMetricAccumulator(target_steps=2, latitudes=np.asarray([0.0]))
    accumulator.update(
        _dataset([0.0, 0.0]),
        _dataset([1.0, 2.0]),
        _dataset([0.0, 1.0]),
    )
    candidate = {"chosen_idx": [4], **accumulator.finalize()}

    compare_metric_outputs(expected, candidate, rtol=0.0, atol=0.0)
