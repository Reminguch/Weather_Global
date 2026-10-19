"""Generated features may drift without weakening stored-data identity."""
import copy
import json

import numpy as np
import pytest
import xarray as xr

from scripts.experiments.run_v24_2023_eval import generated_input_variables, pilot_observation_identity
from scripts.experiments.v24_2023_qualification import NumericalObserver


TIMES = ["2022-01-15T00:00:00Z"]
DERIVED = ["toa_incident_solar_radiation"]


def observed(weather=1., solar=2., rng=0):
    observer = NumericalObserver(DERIVED)
    inputs = xr.Dataset({
        "temperature": (("lat",), np.array([weather], dtype=np.float32)),
        DERIVED[0]: (("lat",), np.array([solar], dtype=np.float32)),
    }, coords={"lat": np.array([0.], dtype=np.float32)})
    observer.initialization({"initialization_time": TIMES[0]}, inputs, np.array([rng], dtype=np.uint32))
    observer.data["setup"] = {key: {"sha256": key} for key in (
        "baseline_params", "baseline_state", "statistics", "baseline_model_config",
        "task_config", "residual_params", "residual_state")}
    observer.data["predictions"] = [dict(initialization_time=TIMES[0], step_index=i,
                                        truth="truth", baseline="baseline", full="full") for i in range(40)]
    return {"numerical_observation": observer.data}


def identity(result, advisory=True):
    return pilot_observation_identity(result, TIMES,
                                     advisory_derived_variables=DERIVED if advisory else ())


def test_generated_drift_advisory_but_strict_keeps_full_inputs():
    left, right = observed(), observed(solar=2.001)
    assert identity(left) == identity(right)
    assert identity(left, False) != identity(right, False)
    assert left["numerical_observation"]["initializations"][0]["derived_inputs"] != right["numerical_observation"]["initializations"][0]["derived_inputs"]


@pytest.mark.parametrize("change", [{"weather": 1.01}, {"rng": 1}])
def test_stored_weather_and_rng_remain_exact(change):
    assert identity(observed()) != identity(observed(**change))


def test_nonfinite_generated_inputs_fail():
    with pytest.raises(ValueError, match="Nonfinite"):
        observed(solar=np.nan)


@pytest.mark.parametrize("field", ["source_inputs", "derived_inputs", "derived_inputs_finite", "derived_input_variables"])
def test_missing_evidence_fails(field):
    result = copy.deepcopy(observed())
    del result["numerical_observation"]["initializations"][0][field]
    with pytest.raises(ValueError):
        identity(result)


def test_variables_present_in_source_cannot_be_exempted(tmp_path):
    (tmp_path / ".zmetadata").write_text(json.dumps({"metadata": {f"{DERIVED[0]}/.zarray": {}}}))
    variables = generated_input_variables({"pilot": {"data_path": str(tmp_path)}})
    assert DERIVED[0] not in variables
    assert "day_progress_sin" in variables
