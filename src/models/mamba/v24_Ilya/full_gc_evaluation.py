"""Direct full-field GraphCast predictors for the shared paired evaluator."""
from __future__ import annotations

import haiku as hk

from .model import V24IlyaPredictors, build_model_configs, make_baseline_predictor


def build_full_gc_predictors(base_model_config, task_config, stats, config):
    model_config = build_model_configs(base_model_config, config.architecture).baseline

    def forecast(inputs, targets, forcings):
        predictor = make_baseline_predictor(model_config, task_config, stats, use_bf16=True)
        return predictor(inputs, targets_template=targets, forcings=forcings)

    # The same architecture/normalizer, with independently supplied public and
    # fine-tuned parameter trees. The residual slot returns a FULL prediction.
    return V24IlyaPredictors(baseline=hk.transform_with_state(forecast),
                            residual=hk.transform_with_state(forecast))
