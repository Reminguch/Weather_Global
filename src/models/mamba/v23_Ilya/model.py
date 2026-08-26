"""Authoritative v23_Ilya model construction."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import haiku as hk

from src.models.graphcast.training.core.model import DirectResidualNormalizer

from graphcast import casting, graphcast as gc, normalization

from .config import V23IlyaArchitectureConfig, V23IlyaEvalConfig


def attach_temporal(
    predictor: gc.GraphCast,
    config: V23IlyaArchitectureConfig,
) -> gc.GraphCast:
    predictor._temporal_backbone = "mamba"
    predictor._temporal_location = config.temporal_location
    predictor._temporal_d_inner = config.temporal_d_inner
    predictor._temporal_bc_groups = config.temporal_bc_groups
    predictor._temporal_d_state = config.temporal_d_state
    predictor._temporal_d_conv = config.temporal_d_conv
    predictor._temporal_dt_rank = config.temporal_dt_rank
    predictor._temporal_init_scheme = config.temporal_init_scheme
    predictor._temporal_dt_init = config.temporal_dt_init
    predictor._temporal_dt_min = config.temporal_dt_min
    predictor._temporal_dt_max = config.temporal_dt_max
    predictor._temporal_dt_scale = config.temporal_dt_scale
    predictor._temporal_dt_init_floor = config.temporal_dt_init_floor
    predictor._temporal_bias = config.temporal_bias
    predictor._temporal_conv_bias = config.temporal_conv_bias
    predictor._temporal_layers = config.temporal_layers
    predictor._temporal_dropout = config.temporal_dropout
    predictor._temporal_stateful = config.temporal_stateful
    # v23_Ilya uses one full Mamba implementation for both persistence modes.
    # False resets/discards memory instead of selecting the legacy
    # lightweight temporal architecture.
    predictor._temporal_use_full_mamba = True
    predictor._temporal_zero_init_out = config.temporal_zero_init_out
    return predictor


class GCResidualWithZeroHead(gc.GraphCast):
    """GraphCast residual predictor with the legacy zero-initialized output head."""

    def __call__(self, inputs, targets_template, forcings, is_training=False):
        self._maybe_init(inputs)
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)
        latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(grid_node_features)
        if self._temporal_backbone != "none" and self._temporal_location == "mesh_post_encoder":
            latent_mesh_nodes = self._run_temporal_mesh_block(
                latent_mesh_nodes,
                is_training=is_training,
            )
        updated_latent_mesh_nodes = self._run_mesh_gnn(
            latent_mesh_nodes,
            is_training=is_training,
        )
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes,
            latent_grid_nodes,
        )
        residual_head = hk.Linear(
            output_grid_nodes.shape[-1],
            w_init=hk.initializers.Constant(0.0),
            b_init=hk.initializers.Constant(0.0),
            name="temporal_residual_head",
        )
        residual_output = residual_head(output_grid_nodes)
        return self._grid_node_outputs_to_prediction(residual_output, targets_template)


@dataclass(frozen=True)
class V23IlyaPredictors:
    baseline: hk.TransformedWithState
    residual: hk.TransformedWithState


@dataclass(frozen=True)
class V23IlyaModelConfigs:
    baseline: gc.ModelConfig
    residual: gc.ModelConfig


def build_model_configs(
    base_model_config: gc.ModelConfig,
    architecture: V23IlyaArchitectureConfig,
) -> V23IlyaModelConfigs:
    return V23IlyaModelConfigs(
        baseline=dataclasses.replace(
            base_model_config,
            resolution=architecture.resolution,
            mesh_size=architecture.mesh_size,
            latent_size=architecture.width,
            gnn_msg_steps=architecture.baseline_msg_steps,
        ),
        residual=dataclasses.replace(
            base_model_config,
            resolution=architecture.resolution,
            mesh_size=architecture.mesh_size,
            latent_size=architecture.width,
            gnn_msg_steps=architecture.residual_msg_steps,
        ),
    )


def make_baseline_predictor(
    model_config: gc.ModelConfig,
    task_config: gc.TaskConfig,
    stats,
    *,
    use_bf16: bool,
):
    predictor = gc.GraphCast(model_config, task_config)
    if use_bf16:
        predictor = casting.Bfloat16Cast(predictor)
    return normalization.InputsAndResiduals(
        predictor,
        stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )


def make_residual_predictor(
    model_config: gc.ModelConfig,
    task_config: gc.TaskConfig,
    stats,
    architecture: V23IlyaArchitectureConfig,
    *,
    use_bf16: bool,
):
    predictor = GCResidualWithZeroHead(model_config, task_config)
    attach_temporal(predictor, architecture)
    if use_bf16:
        predictor = casting.Bfloat16Cast(predictor)
    return DirectResidualNormalizer(
        predictor,
        stddev_by_level=stats["stddev_by_level"],
        mean_by_level=stats["mean_by_level"],
        diffs_stddev_by_level=stats["diffs_stddev_by_level"],
    )


def build_predictors(
    base_model_config,
    task_config,
    stats,
    config: V23IlyaEvalConfig,
) -> V23IlyaPredictors:
    architecture = config.architecture
    model_configs = build_model_configs(base_model_config, architecture)

    def build_baseline():
        return make_baseline_predictor(
            model_configs.baseline,
            task_config,
            stats,
            use_bf16=True,
        )

    def build_residual():
        return make_residual_predictor(
            model_configs.residual,
            task_config,
            stats,
            architecture,
            use_bf16=True,
        )

    def baseline_fn(inputs, targets, forcings):
        return build_baseline()(inputs, targets_template=targets, forcings=forcings)

    def residual_fn(inputs, targets, forcings):
        return build_residual()(inputs, targets_template=targets, forcings=forcings)

    return V23IlyaPredictors(
        baseline=hk.transform_with_state(baseline_fn),
        residual=hk.transform_with_state(residual_fn),
    )
