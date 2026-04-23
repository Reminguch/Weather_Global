from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import xarray as xr

import haiku as hk
import jax

from src.models.mz_residual_mamba import MZResidualConfig
from src.models.mz_residual_mamba import MZResidualMamba
from src.models.mz_residual_mamba import shift_residual_history


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "training" / "train_mz_residual_memory.py"
_SPEC = importlib.util.spec_from_file_location("train_mz_residual_memory", SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_shift_residual_history_zeroes_first_step():
    residual = jnp.arange(2 * 1 * 2 * 1 * 3, dtype=jnp.float32).reshape(2, 1, 2, 1, 3)
    shifted = shift_residual_history(residual)
    np.testing.assert_allclose(np.asarray(shifted[0]), 0.0)
    np.testing.assert_allclose(np.asarray(shifted[1]), np.asarray(residual[0]))


def test_extract_feature_block_concatenates_selected_variables_in_order():
    batch = [0, 1]
    time = [0, 1]
    lat = [10.0]
    lon = [20.0]
    level = [100, 200]

    ds = xr.Dataset(
        {
            "mean_sea_level_pressure": xr.DataArray(
                np.array(
                    [
                        [[[1.0]], [[2.0]]],
                        [[[3.0]], [[4.0]]],
                    ],
                    dtype=np.float32,
                ),
                dims=("batch", "time", "lat", "lon"),
                coords={"batch": batch, "time": time, "lat": lat, "lon": lon},
            ),
            "geopotential": xr.DataArray(
                np.array(
                    [
                        [[[[10.0, 11.0]]], [[[12.0, 13.0]]]],
                        [[[[14.0, 15.0]]], [[[16.0, 17.0]]]],
                    ],
                    dtype=np.float32,
                ),
                dims=("batch", "time", "lat", "lon", "level"),
                coords={"batch": batch, "time": time, "lat": lat, "lon": lon, "level": level},
            ),
            "u_component_of_wind": xr.DataArray(
                np.array(
                    [
                        [[[[20.0, 21.0]]], [[[22.0, 23.0]]]],
                        [[[[24.0, 25.0]]], [[[26.0, 27.0]]]],
                    ],
                    dtype=np.float32,
                ),
                dims=("batch", "time", "lat", "lon", "level"),
                coords={"batch": batch, "time": time, "lat": lat, "lon": lon, "level": level},
            ),
            "v_component_of_wind": xr.DataArray(
                np.array(
                    [
                        [[[[30.0, 31.0]]], [[[32.0, 33.0]]]],
                        [[[[34.0, 35.0]]], [[[36.0, 37.0]]]],
                    ],
                    dtype=np.float32,
                ),
                dims=("batch", "time", "lat", "lon", "level"),
                coords={"batch": batch, "time": time, "lat": lat, "lon": lon, "level": level},
            ),
        }
    )

    class _TaskCfg:
        pressure_levels = (100, 200)

    feature_order, _feature_slices, feature_dim = _MODULE._resolved_feature_layout(_TaskCfg)
    block = _MODULE._extract_feature_block(
        ds,
        time_index=-1,
        task_cfg=_TaskCfg,
        feature_order=feature_order,
    )
    assert feature_dim == 7
    assert block.shape == (2, 1, 1, 7)
    np.testing.assert_allclose(
        np.asarray(block[:, 0, 0, :]),
        np.array(
            [
                [2.0, 12.0, 13.0, 22.0, 23.0, 32.0, 33.0],
                [4.0, 16.0, 17.0, 26.0, 27.0, 36.0, 37.0],
            ],
            dtype=np.float32,
        ),
    )


def test_mz_residual_mamba_forward_shape():
    cfg = MZResidualConfig(input_size=6, output_size=4, hidden_size=8, layers=1, a_log_init=-0.1)

    def forward(x):
        return MZResidualMamba(cfg)(x, is_training=False)

    transformed = hk.transform(forward)
    x = jnp.ones((5, 2, 3, 4, 6), dtype=jnp.float32)
    params = transformed.init(jax.random.PRNGKey(0), x)
    y = transformed.apply(params, None, x)
    assert y.shape == (5, 2, 3, 4, 4)
