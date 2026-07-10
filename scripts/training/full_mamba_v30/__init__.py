"""v30 Contrastive Event Adapter Head.

Skeleton — Phase 0/1/2 training not yet wired.
"""
from .event_adapter_head import (
    GCResidualWithEventAdapter,
    gather_event_patch,
    info_nce_loss,
)
from .event_tube_sampler import (
    EventTubeBatch,
    PerStepEventSampler,
    sample_event_anchors,
    build_truth_tube,
    sample_climate_matched_negatives,
)

__all__ = [
    "GCResidualWithEventAdapter",
    "gather_event_patch",
    "info_nce_loss",
    "EventTubeBatch",
    "PerStepEventSampler",
    "sample_event_anchors",
    "build_truth_tube",
    "sample_climate_matched_negatives",
]
