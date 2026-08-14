from __future__ import annotations

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from src.models.mamba.v22_final.model import GCResidualWithZeroHead as FinalHead


EXPECTED_PARAMETER_SHAPES = {
    "temporal_residual_head": {"w": (3, 3), "b": (3,)},
}


def _transformed(residual_class):
    def forward(values):
        predictor = object.__new__(residual_class)
        predictor._temporal_backbone = "none"
        predictor._temporal_location = "mesh_processor_interleaved"
        predictor._maybe_init = lambda inputs: None
        predictor._inputs_to_grid_node_features = lambda inputs, forcings: inputs
        predictor._run_grid2mesh_gnn = lambda features: (features, features)
        predictor._run_mesh_gnn = lambda features, *, is_training: features
        predictor._run_mesh2grid_gnn = lambda mesh, grid: mesh
        predictor._grid_node_outputs_to_prediction = lambda output, targets: output
        return residual_class.__call__(predictor, values, None, None)

    return hk.transform(forward)


def _shapes(params) -> dict[str, dict[str, tuple[int, ...]]]:
    mutable = hk.data_structures.to_mutable_dict(params)
    return {
        module: {name: tuple(value.shape) for name, value in module_params.items()}
        for module, module_params in mutable.items()
    }


def test_zero_head_matches_frozen_v22_contract() -> None:
    values = jnp.ones((5, 1, 3), dtype=jnp.float32)
    key = jax.random.PRNGKey(0)
    final = _transformed(FinalHead)
    final_params = final.init(key, values)

    assert _shapes(final_params) == EXPECTED_PARAMETER_SHAPES
    for leaf in jax.tree_util.tree_leaves(final_params):
        np.testing.assert_array_equal(np.asarray(leaf), 0.0)
    np.testing.assert_array_equal(np.asarray(final.apply(final_params, key, values)), 0.0)
