"""Helpers for reading and slimming LLM rubric payloads."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

RUBRIC_COMPONENTS = ("grounded", "relevance", "distinctness", "report_usefulness")


def extract_rubric_scores(rubric: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Return per-dimension scores from in-memory or on-disk rubric shapes."""
    if not rubric:
        return {key: None for key in RUBRIC_COMPONENTS}
    components = rubric.get("components") or {}
    if components:
        return {
            key: (components.get(key) or {}).get("mean")
            for key in RUBRIC_COMPONENTS
        }
    aggregated = rubric.get("aggregated") or {}
    if aggregated:
        return {
            key: (aggregated.get(key) or {}).get("mean")
            if isinstance(aggregated.get(key), dict)
            else aggregated.get(key)
            for key in RUBRIC_COMPONENTS
        }
    judges = rubric.get("judges") or []
    if judges:
        scores = judges[0].get("scores") or {}
        return {key: scores.get(key) for key in RUBRIC_COMPONENTS}
    return {key: None for key in RUBRIC_COMPONENTS}


def slim_rubric_for_metrics_json(rubric: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop redundant rubric fields before writing metrics.json."""
    if not rubric:
        return None
    judges = rubric.get("judges") or []
    if not judges:
        return None
    slim: Dict[str, Any] = {
        "judges": [
            {
                "provider": judge.get("provider"),
                "model": judge.get("model"),
                "scores": judge.get("scores") or {},
                "reasoning": judge.get("reasoning") or {},
            }
            for judge in judges
        ]
    }
    if len(judges) > 1:
        components = rubric.get("components") or {}
        aggregated: Dict[str, Any] = {}
        for key in RUBRIC_COMPONENTS:
            component = components.get(key) or {}
            entry: Dict[str, Any] = {"mean": component.get("mean")}
            variance = component.get("variance")
            if variance is not None and float(variance) > 0:
                entry["variance"] = variance
            aggregated[key] = entry
        slim["aggregated"] = aggregated
    return slim
