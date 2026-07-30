"""Token, cost, and latency tracking for metrics computation."""

from __future__ import annotations

from typing import Any, Dict

from src.tracking import UsageSummary, get_tracker

METRICS_FEATURES = (
    "metrics_finding_rubric",
    "metrics_diversity_embedding",
)


def capture_metrics_usage_start() -> Dict[str, UsageSummary]:
    tracker = get_tracker()
    return {
        feature: tracker.by_feature.get(feature, UsageSummary()).copy()
        for feature in METRICS_FEATURES
    }


def summarize_metrics_usage(start: Dict[str, UsageSummary]) -> Dict[str, Any]:
    """Return metrics-only usage since ``capture_metrics_usage_start()``."""
    tracker = get_tracker()
    total = UsageSummary()
    by_feature: Dict[str, Any] = {}
    for feature in METRICS_FEATURES:
        end = tracker.by_feature.get(feature, UsageSummary())
        begin = start.get(feature, UsageSummary())
        delta = end.subtract(begin)
        if delta.n_calls > 0:
            by_feature[feature] = delta.to_dict()
            total.merge(delta)
    return {
        "total": total.to_dict(),
        "by_feature": by_feature,
    }
