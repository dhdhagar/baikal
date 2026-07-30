"""Human-readable metrics_summary.json layout."""

from __future__ import annotations

from typing import Any, Dict, List

from src.metrics.summary_blocks import (
    format_aggregate_headlines_for_disk,
    slim_operational_block,
    slim_research_quality_agg_block,
    slim_retrieval_block,
)


def _format_per_query(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for entry in entries:
        row: Dict[str, Any] = {
            "query_id": entry.get("query_id"),
            "coverage": entry.get("coverage"),
        }
        research_quality = slim_research_quality_agg_block(entry.get("research_quality"))
        if research_quality is not None:
            row["research_quality"] = research_quality
        retrieval = slim_retrieval_block(entry.get("retrieval"))
        if retrieval is not None:
            row["retrieval"] = retrieval
        operational = slim_operational_block(entry.get("operational"))
        if operational is not None:
            row["operational"] = operational
        formatted.append(row)
    return formatted


def format_per_coverage_for_disk(
    per_coverage: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        bucket: {
            "n_queries": block.get("n_queries"),
            **format_aggregate_headlines_for_disk(block),
        }
        for bucket, block in per_coverage.items()
    }


def format_metrics_summary_for_disk(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build the on-disk metrics_summary.json payload."""
    formatted: Dict[str, Any] = {
        "n_queries": summary.get("n_queries"),
        "overall": format_aggregate_headlines_for_disk(summary.get("overall") or {}),
    }
    per_coverage = summary.get("per_coverage") or {}
    if per_coverage:
        formatted["per_coverage"] = format_per_coverage_for_disk(per_coverage)
    per_query = summary.get("per_query")
    if isinstance(per_query, list):
        formatted["per_query"] = _format_per_query(per_query)
    return formatted
