"""Aggregate per-query metrics into run-level summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.metrics.common import round4
from src.metrics.rubric_utils import RUBRIC_COMPONENTS
from src.metrics.summary_blocks import (
    OPERATIONAL_SUMMARY_KEYS,
    RESEARCH_QUALITY_AGG_KEYS,
    RETRIEVAL_SUMMARY_KEYS,
    operational_summary_block,
    research_quality_summary_block,
    retrieval_summary_block,
)
_COVERAGE_BUCKETS = ("low", "medium", "high")


def _normalize_coverage(coverage: Optional[str]) -> Optional[str]:
    if coverage is None:
        return None
    bucket = str(coverage).lower()
    return bucket if bucket in _COVERAGE_BUCKETS else None


def _mean(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round4(sum(nums) / len(nums))


def _mean_int(values: List[Optional[int]]) -> Optional[float]:
    nums = [int(v) for v in values if v is not None]
    if not nums:
        return None
    return round4(sum(nums) / len(nums))


def _aggregate_rubric_means(
    per_query: List[Dict[str, Any]],
) -> Optional[Dict[str, Optional[float]]]:
    means: Dict[str, Optional[float]] = {}
    for key in RUBRIC_COMPONENTS:
        values = [
            (item.get("research_quality") or {}).get("rubric_means", {}).get(key)
            for item in per_query
        ]
        means[key] = _mean(values)
    if not any(value is not None for value in means.values()):
        return None
    return means


def _aggregate_section(
    per_query: List[Dict[str, Any]],
    section: str,
    keys: List[str],
) -> Dict[str, Optional[float]]:
    return {
        key: _mean([(item.get(section) or {}).get(key) for item in per_query])
        for key in keys
    }


def _build_research_quality_aggregate(
    per_query: List[Dict[str, Any]],
) -> Dict[str, Any]:
    block: Dict[str, Any] = {}
    for key in RESEARCH_QUALITY_AGG_KEYS:
        if key == "rubric_means":
            continue
        if key == "n_findings_valid":
            value = _mean_int(
                [
                    (item.get("research_quality") or {}).get(key)
                    for item in per_query
                ]
            )
        else:
            value = _mean(
                [
                    (item.get("research_quality") or {}).get(key)
                    for item in per_query
                ]
            )
        if value is not None:
            block[key] = value
    rubric_means = _aggregate_rubric_means(per_query)
    if rubric_means is not None:
        block["rubric_means"] = rubric_means
    return block


def extract_query_summary(
    query_id: str,
    metrics: Dict[str, Any],
    *,
    coverage: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract headline scalar metrics for one query."""
    research_quality = research_quality_summary_block(metrics) or {}
    retrieval = retrieval_summary_block(metrics) or {}
    operational = operational_summary_block(metrics) or {}

    summary: Dict[str, Any] = {
        "query_id": query_id,
        "coverage": _normalize_coverage(coverage),
    }
    if research_quality:
        slim_rq = {
            key: research_quality.get(key)
            for key in (*RESEARCH_QUALITY_AGG_KEYS, "rubric_means")
            if research_quality.get(key) is not None
        }
        if slim_rq:
            summary["research_quality"] = slim_rq
    if retrieval:
        summary["retrieval"] = retrieval
    if operational:
        summary["operational"] = operational
    return summary


def _build_aggregate_block(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    block: Dict[str, Any] = {}
    research_quality = _build_research_quality_aggregate(per_query)
    if research_quality:
        block["research_quality"] = research_quality
    retrieval = _aggregate_section(
        per_query, "retrieval", list(RETRIEVAL_SUMMARY_KEYS)
    )
    if any(value is not None for value in retrieval.values()):
        block["retrieval"] = retrieval
    operational = _aggregate_section(
        per_query, "operational", list(OPERATIONAL_SUMMARY_KEYS)
    )
    if any(value is not None for value in operational.values()):
        block["operational"] = operational
    return block


def _build_per_coverage(
    per_query: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in per_query:
        bucket = item.get("coverage")
        if bucket is not None:
            grouped[bucket].append(item)

    return {
        bucket: {
            "n_queries": len(items),
            **_build_aggregate_block(items),
        }
        for bucket in _COVERAGE_BUCKETS
        if (items := grouped.get(bucket))
    }


def _per_query_sort_key(item: Dict[str, Any]) -> tuple[float, str]:
    report_score = (item.get("research_quality") or {}).get("report_score")
    score = float(report_score) if report_score is not None else float("-inf")
    return (-score, str(item.get("query_id") or ""))


def aggregate_run_metrics(
    outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a run-level metrics summary from per-query metric payloads.

    Each output item should include ``query_id``, ``metrics``, and optionally
    ``coverage``.
    """
    per_query = [
        extract_query_summary(
            item["query_id"],
            item["metrics"],
            coverage=item.get("coverage"),
        )
        for item in outputs
    ]
    per_query.sort(key=_per_query_sort_key)

    return {
        "n_queries": len(per_query),
        "overall": _build_aggregate_block(per_query),
        "per_coverage": _build_per_coverage(per_query),
        "per_query": per_query,
    }


def _format_aggregate_block(
    lines: List[str],
    label: str,
    block: Dict[str, Any],
) -> None:
    retrieval = block.get("retrieval") or {}
    for key in (
        "table_recall",
        "table_precision",
        "table_gt_in_top_k",
        "table_gt_reachable",
        "passage_recall",
        "passage_precision",
        "passage_gt_in_top_k",
        "passage_gt_reachable",
    ):
        value = retrieval.get(key)
        if value is not None:
            lines.append(f"{label} avg {key}: {value:.4f}")
    if retrieval.get("lake_coverage") is not None:
        lines.append(f"{label} avg lake_coverage: {retrieval['lake_coverage']:.4f}")
    operational = block.get("operational") or {}
    if operational.get("sql_success_rate") is not None:
        lines.append(
            f"{label} avg sql_success_rate: {operational['sql_success_rate']:.4f}"
        )
    report_score = (block.get("research_quality") or {}).get("report_score")
    if report_score is not None:
        lines.append(f"{label} avg report_score: {report_score:.4f}")


def format_run_metrics_summary(summary: Dict[str, Any]) -> str:
    """Render a human-readable overall, per-coverage, and per-query metrics summary."""
    lines: List[str] = []
    lines.append(f"Metrics summary ({summary.get('n_queries', 0)} queries)")
    lines.append("")

    overall = summary.get("overall") or {}
    lines.append("Overall")
    _format_aggregate_block(lines, "Overall", overall)
    lines.append("")

    per_coverage = summary.get("per_coverage") or {}
    if per_coverage:
        lines.append("Per coverage")
        for coverage, block in per_coverage.items():
            n_queries = block.get("n_queries", 0)
            lines.append(f"  {coverage} ({n_queries} queries)")
            coverage_lines: List[str] = []
            _format_aggregate_block(coverage_lines, coverage.capitalize(), block)
            lines.extend(f"    {line}" for line in coverage_lines)
        lines.append("")

    lines.append("Per query")
    lines.append(f"{'Query':<24} {'Report score':>14}")
    lines.append(f"{'-' * 24} {'-' * 14}")
    for item in summary.get("per_query") or []:
        query_id = str(item.get("query_id") or "?")
        score = (item.get("research_quality") or {}).get("report_score")
        score_text = f"{score:.4f}" if score is not None else "n/a"
        lines.append(f"{query_id:<24} {score_text:>14}")

    return "\n".join(lines)


def outputs_from_results(
    results: List[Dict[str, Any]],
    *,
    query_dirs: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Build aggregate inputs from per-query result.json entries."""
    from src.result_json import load_query_metrics

    outputs: List[Dict[str, Any]] = []
    for result in results:
        metrics = None
        if query_dirs:
            query_id = str(result.get("query_id") or "")
            query_dir = query_dirs.get(query_id)
            if query_dir:
                metrics = load_query_metrics(query_dir, result)
        if not metrics:
            continue
        outputs.append(
            {
                "query_id": result.get("query_id") or "unknown",
                "metrics": metrics,
                "coverage": result.get("coverage"),
            }
        )
    return outputs
