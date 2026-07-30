"""Human-readable run-level results_all.json layout."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.metrics.run_summary_json import format_per_coverage_for_disk
from src.metrics.summary_blocks import format_aggregate_headlines_for_disk
from src.utils import load_json, save_json

RESULTS_ALL_FILENAME = "results_all.json"
METRICS_SUMMARY_FILENAME = "metrics_summary.json"


def slim_query_index_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a slim per-query row for the run index."""
    query_id = str(result.get("query_id") or "")
    research_quality = (result.get("summary") or {}).get("research_quality") or {}
    entry: Dict[str, Any] = {
        "query_id": query_id,
        "coverage": result.get("coverage"),
        "method": result.get("method"),
        "result_path": f"{query_id}/result.json",
    }
    report_score = research_quality.get("report_score")
    if report_score is not None:
        entry["report_score"] = report_score
    return entry


def _metrics_headlines_from_summary(
    metrics_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Slim run-level metric blocks for results_all.json (matches metrics_summary.json)."""
    headlines: Dict[str, Any] = {}
    overall = format_aggregate_headlines_for_disk(metrics_summary.get("overall") or {})
    if overall:
        headlines["summary"] = overall
    per_coverage = metrics_summary.get("per_coverage") or {}
    if per_coverage:
        headlines["per_coverage"] = format_per_coverage_for_disk(per_coverage)
    n_queries = metrics_summary.get("n_queries")
    if n_queries is not None:
        headlines["n_queries_with_metrics"] = n_queries
    if overall or per_coverage:
        headlines["metrics_summary_path"] = METRICS_SUMMARY_FILENAME
    return headlines


def build_results_all_payload(
    results: List[Dict[str, Any]],
    *,
    time_taken: float,
    usage: Dict[str, Any],
    metrics_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the on-disk results_all.json payload."""
    queries = sorted(
        [slim_query_index_entry(result) for result in results],
        key=lambda item: str(item.get("query_id") or ""),
    )
    payload: Dict[str, Any] = {
        "n_completed": len(queries),
        "time_taken": round(time_taken, 3),
        "usage": usage,
    }
    if metrics_summary:
        payload.update(_metrics_headlines_from_summary(metrics_summary))
    payload["queries"] = queries
    return payload


def patch_results_all_metrics(
    log_dir: str,
    metrics_summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    """Refresh run-level metric headlines in an existing results_all.json."""
    path = os.path.join(log_dir, RESULTS_ALL_FILENAME)
    if not os.path.isfile(path):
        return

    payload = load_json(path)
    for key in ("summary", "per_coverage", "n_queries_with_metrics", "metrics_summary_path"):
        payload.pop(key, None)
    payload.update(_metrics_headlines_from_summary(metrics_summary))

    scores_by_id = {
        str(result.get("query_id") or ""): (
            (result.get("summary") or {}).get("research_quality") or {}
        ).get("report_score")
        for result in results
    }
    for entry in payload.get("queries") or []:
        query_id = str(entry.get("query_id") or "")
        report_score = scores_by_id.get(query_id)
        if report_score is not None:
            entry["report_score"] = report_score
        else:
            entry.pop("report_score", None)

    save_json(path, payload)
