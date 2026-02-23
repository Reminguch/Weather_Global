"""GraphCast adapter with compact runtime implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.data.contracts import CanonicalBatch, validate_canonical_batch
from src.models.base import ForecastModel, LossAndPredictions, TrainState


def _as_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"`{field_name}` must be a mapping, got {type(value)!r}.")
    return dict(value)


def _parse_graphcast_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_section = config.get("model", {})
    if not isinstance(model_section, Mapping):
        raise TypeError("`config['model']` must be a mapping when provided.")

    graphcast_section = model_section.get("graphcast", {})
    if not isinstance(graphcast_section, Mapping):
        raise TypeError("`config['model']['graphcast']` must be a mapping when provided.")

    backend = str(graphcast_section.get("backend", model_section.get("backend", "graphcast"))).strip()
    backend = backend.lower()
    if backend not in {"graphcast", "stub"}:
        raise ValueError(f"Unknown GraphCast backend {backend!r}. Expected `graphcast` or `stub`.")

    parsed = {
        "backend": backend,
        "use_bfloat16": bool(graphcast_section.get("use_bfloat16", False)),
        "use_normalization": bool(graphcast_section.get("use_normalization", True)),
        "gradient_checkpointing": bool(graphcast_section.get("gradient_checkpointing", True)),
        "model_config": _as_dict(graphcast_section.get("model_config", {}), field_name="model_config"),
        "task_config": _as_dict(graphcast_section.get("task_config", {}), field_name="task_config"),
        "normalization_stats": _as_dict(
            graphcast_section.get("normalization_stats", {}),
            field_name="normalization_stats",
        ),
    }

    if backend == "graphcast" and not parsed["model_config"]:
        raise ValueError("GraphCast backend requires `config['model']['graphcast']['model_config']`.")
    if backend == "graphcast" and not parsed["task_config"]:
        raise ValueError("GraphCast backend requires `config['model']['graphcast']['task_config']`.")
    return parsed


class GraphCastAdapter(ForecastModel):
    """Single-file adapter that supports stub and real GraphCast backends."""

    def __init__(self, config: Mapping[str, Any]):
        self._config = _parse_graphcast_config(config)
        self._transformed = None

    @staticmethod
    def _predict_stub(batch: CanonicalBatch) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        for key, target in batch.targets.items():
            target_arr = np.asarray(target)
            candidate = np.asarray(batch.inputs[key]) if key in batch.inputs else None
            predictions[key] = candidate if candidate is not None and candidate.shape == target_arr.shape else np.zeros_like(target_arr)
        return predictions

    @staticmethod
    def _to_dataset(mapping: Mapping[str, Any], *, field_name: str, allow_empty: bool = False):
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError("GraphCast backend requires xarray.") from exc

        if allow_empty and not mapping:
            return xr.Dataset()

        for dataset_key in ("dataset", "__dataset__"):
            if dataset_key in mapping:
                dataset = mapping[dataset_key]
                if isinstance(dataset, xr.Dataset):
                    return dataset
                raise TypeError(f"`{field_name}.{dataset_key}` must be an xarray.Dataset.")

        if mapping and all(isinstance(value, xr.DataArray) for value in mapping.values()):
            return xr.Dataset(dict(mapping))

        raise TypeError(
            f"GraphCast backend cannot convert `{field_name}` mapping to xarray.Dataset."
        )

    def _ensure_real_backend(self):
        if self._transformed is not None:
            return self._transformed

        try:
            import haiku as hk
            from graphcast import autoregressive
            from graphcast import casting
            from graphcast import graphcast as graphcast_module
            from graphcast import normalization
        except ImportError as exc:
            raise ImportError(
                "GraphCast backend requires installed dependencies: `graphcast`, `jax`, and `haiku`."
            ) from exc

        config = self._config

        def forward(inputs, targets, forcings, is_training: bool):
            del is_training
            predictor = graphcast_module.GraphCast(
                graphcast_module.ModelConfig(**config["model_config"]),
                graphcast_module.TaskConfig(**config["task_config"]),
            )

            if config["use_bfloat16"]:
                predictor = casting.Bfloat16Cast(predictor)

            if config["use_normalization"]:
                stats = config["normalization_stats"]
                required = ("stddev_by_level", "mean_by_level", "diffs_stddev_by_level")
                missing = [key for key in required if key not in stats]
                if missing:
                    missing_csv = ", ".join(missing)
                    raise ValueError(
                        f"Missing normalization statistics: {missing_csv}. "
                        "Set `use_normalization=False` for smoke runs."
                    )
                predictor = normalization.InputsAndResiduals(
                    predictor,
                    stddev_by_level=stats["stddev_by_level"],
                    mean_by_level=stats["mean_by_level"],
                    diffs_stddev_by_level=stats["diffs_stddev_by_level"],
                )

            predictor = autoregressive.Predictor(
                predictor,
                gradient_checkpointing=config["gradient_checkpointing"],
            )
            return predictor.loss_and_predictions(inputs, targets, forcings)

        self._transformed = hk.transform_with_state(forward)
        return self._transformed

    def _convert_real_batch(self, batch: CanonicalBatch):
        return (
            self._to_dataset(batch.inputs, field_name="inputs"),
            self._to_dataset(batch.targets, field_name="targets"),
            self._to_dataset(batch.forcings, field_name="forcings", allow_empty=True),
        )

    def init(self, rng: Any, sample_batch: CanonicalBatch) -> TrainState:
        validate_canonical_batch(sample_batch)
        if self._config["backend"] == "stub":
            return TrainState(
                params={"backend": "stub"},
                model_state={},
                optimizer_state=None,
                step=0,
                rng=rng,
            )

        transformed = self._ensure_real_backend()
        inputs, targets, forcings = self._convert_real_batch(sample_batch)
        params, model_state = transformed.init(rng, inputs, targets, forcings, True)
        return TrainState(
            params=params,
            model_state=model_state,
            optimizer_state=None,
            step=0,
            rng=rng,
        )

    def loss_and_predictions(
        self, train_state: TrainState, batch: CanonicalBatch, *, is_training: bool
    ) -> LossAndPredictions:
        validate_canonical_batch(batch)
        if self._config["backend"] == "stub":
            predictions = self._predict_stub(batch)
            per_target = []
            for key, target in batch.targets.items():
                diff = np.asarray(predictions[key]) - np.asarray(target)
                per_target.append(np.mean(np.square(diff)))
            loss = np.array(float(np.mean(per_target)) if per_target else 0.0, dtype=np.float32)
            return LossAndPredictions(
                loss=loss,
                predictions=predictions,
                metrics={"loss": float(loss), "num_targets": len(predictions)},
                next_model_state=train_state.model_state,
            )

        transformed = self._ensure_real_backend()
        inputs, targets, forcings = self._convert_real_batch(batch)
        (loss, predictions), next_model_state = transformed.apply(
            train_state.params,
            train_state.model_state,
            train_state.rng,
            inputs,
            targets,
            forcings,
            is_training,
        )
        return LossAndPredictions(
            loss=loss,
            predictions={"graphcast": predictions},
            metrics={"loss": float(np.asarray(loss))},
            next_model_state=next_model_state,
        )

    def predict(self, train_state: TrainState, batch: CanonicalBatch) -> Mapping[str, Any]:
        if self._config["backend"] == "stub":
            validate_canonical_batch(batch)
            return self._predict_stub(batch)
        return self.loss_and_predictions(train_state, batch, is_training=False).predictions


def build_graphcast(config: Mapping[str, Any]) -> ForecastModel:
    """Build GraphCast adapter from generic config mapping."""

    return GraphCastAdapter(config)
