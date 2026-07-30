"""Shared headline metric blocks for result.json and run aggregation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

RETRIEVAL_SUMMARY_KEYS = (
    "table_gt_in_top_k",
    "table_gt_reachable",
    "table_recall",
    "table_precision",
    "lake_coverage",
    "passage_gt_in_top_k",
    "passage_gt_reachable",
    "passage_recall",
    "passage_precision",
)

OPERATIONAL_SUMMARY_KEYS = (
    "sql_success_rate",
    "cluster_attrition_rate",
    "diversity_mean",
    "n_findings",
)

# Run aggregates omit per-query budget (lives in args / metrics.json).
RESEARCH_QUALITY_AGG_KEYS = (
    "report_score",
    "finding_scores_sum",
    "n_findings_valid",
    "rubric_means",
)


def _block_has_values(block: Dict[str, Any]) -> bool:
    return any(value is not None for value in block.values())


def _slim_dict_block(
    block: Optional[Dict[str, Any]],
    keys: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if not block:
        return None
    slim = {key: block.get(key) for key in keys if block.get(key) is not None}
    return slim or None


def slim_research_quality_agg_block(
    block: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    return _slim_dict_block(block, RESEARCH_QUALITY_AGG_KEYS)


def slim_retrieval_block(block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _slim_dict_block(block, RETRIEVAL_SUMMARY_KEYS)


def slim_operational_block(block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return _slim_dict_block(block, OPERATIONAL_SUMMARY_KEYS)


def format_aggregate_headlines_for_disk(block: Dict[str, Any]) -> Dict[str, Any]:
    """Format nested research_quality / retrieval / operational for on-disk JSON."""
    formatted: Dict[str, Any] = {}
    research_quality = slim_research_quality_agg_block(block.get("research_quality"))
    if research_quality is not None:
        formatted["research_quality"] = research_quality
    retrieval = slim_retrieval_block(block.get("retrieval"))
    if retrieval is not None:
        formatted["retrieval"] = retrieval
    operational = slim_operational_block(block.get("operational"))
    if operational is not None:
        formatted["operational"] = operational
    return formatted


def research_quality_summary_block(
    metrics: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not metrics:
        return None
    research_quality = metrics.get("research_quality") or {}
    if not research_quality:
        return None
    block = {
        "report_score": research_quality.get("report_score"),
        "finding_scores_sum": research_quality.get("finding_scores_sum"),
        "n_findings_valid": research_quality.get("n_findings_valid"),
        "budget": research_quality.get("budget"),
    }
    rubric_means = research_quality.get("rubric_means")
    if rubric_means:
        block["rubric_means"] = rubric_means
    return block


def retrieval_summary_block(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metrics:
        return None
    retrieval = metrics.get("retrieval") or {}
    cumulative = retrieval.get("cumulative") or {}
    block = {
        "table_gt_in_top_k": retrieval.get("table_gt_in_top_k"),
        "table_gt_reachable": retrieval.get("table_gt_reachable"),
        "table_recall": cumulative.get("table_recall"),
        "table_precision": cumulative.get("table_precision"),
        "lake_coverage": cumulative.get("lake_coverage"),
        "passage_gt_in_top_k": retrieval.get("passage_gt_in_top_k"),
        "passage_gt_reachable": retrieval.get("passage_gt_reachable"),
        "passage_recall": cumulative.get("passage_recall"),
        "passage_precision": cumulative.get("passage_precision"),
    }
    if not _block_has_values(block):
        return None
    return block


def operational_summary_block(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metrics:
        return None
    operational = (metrics.get("operational") or {}).get("cumulative") or {}
    diversity = operational.get("diversity") or {}
    block = {
        "sql_success_rate": operational.get("sql_success_rate"),
        "cluster_attrition_rate": operational.get("cluster_attrition_rate"),
        "diversity_mean": diversity.get("mean"),
        "n_findings": operational.get("n_findings"),
    }
    if not _block_has_values(block):
        return None
    return block
