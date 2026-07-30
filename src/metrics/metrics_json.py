"""Human-readable metrics.json layout and artifact formatting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.metrics.rubric_utils import slim_rubric_for_metrics_json
from src.metrics.summary_blocks import research_quality_summary_block

_RETRIEVAL_HEADLINE_KEYS = (
    "table_gt_in_top_k",
    "table_gt_reachable",
    "passage_gt_in_top_k",
    "passage_gt_reachable",
)

_PER_STEP_RETRIEVAL_DROP_KEYS = frozenset({"tables_used", "passages_cited"})


def _slim_retrieval_block(retrieval: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not retrieval:
        return None
    block: Dict[str, Any] = {}
    for key in _RETRIEVAL_HEADLINE_KEYS:
        if key in retrieval:
            block[key] = retrieval.get(key)
    cumulative = retrieval.get("cumulative")
    if cumulative:
        block["cumulative"] = cumulative
    return block or None


def _slim_per_step_retrieval(retrieval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in retrieval.items()
        if key not in _PER_STEP_RETRIEVAL_DROP_KEYS
    }


def _slim_per_step(per_step: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slimmed: List[Dict[str, Any]] = []
    for step_metrics in per_step:
        research_quality = dict(step_metrics.get("research_quality") or {})
        rubric = research_quality.pop("rubric", None)
        slim_rubric = slim_rubric_for_metrics_json(rubric)
        if slim_rubric is not None:
            research_quality["rubric"] = slim_rubric
        slimmed.append(
            {
                "step": step_metrics.get("step"),
                "retrieval": _slim_per_step_retrieval(step_metrics.get("retrieval") or {}),
                "operational": step_metrics.get("operational") or {},
                "research_quality": research_quality,
            }
        )
    return slimmed


def format_metrics_for_disk(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Build the on-disk metrics.json payload (summaries first, per_step last)."""
    formatted: Dict[str, Any] = {}

    research_quality = research_quality_summary_block(metrics)
    if research_quality is not None:
        formatted["research_quality"] = research_quality

    retrieval = _slim_retrieval_block(metrics.get("retrieval"))
    if retrieval is not None:
        formatted["retrieval"] = retrieval

    operational = metrics.get("operational")
    if operational:
        formatted["operational"] = operational

    judge_usage = metrics.get("judge_usage")
    if judge_usage is None:
        judge_usage = metrics.get("usage")
    if judge_usage:
        formatted["judge_usage"] = judge_usage

    for key in (
        "budget_steps_completed",
        "budget_steps_with_iteration",
        "initial_candidate_clusters",
        "total_lake_tables",
    ):
        if metrics.get(key) is not None:
            formatted[key] = metrics[key]

    per_step = metrics.get("per_step")
    if isinstance(per_step, list):
        formatted["per_step"] = _slim_per_step(per_step)

    return formatted
