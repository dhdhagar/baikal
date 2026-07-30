"""Human-readable per-query result.json schema and artifact sidecars."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.metrics.rubric_utils import extract_rubric_scores
from src.metrics.metrics_json import format_metrics_for_disk
from src.metrics.summary_blocks import (
    operational_summary_block,
    research_quality_summary_block,
    retrieval_summary_block,
)
from src.utils import load_json, save_json

GROUND_TRUTH_FILENAME = "ground_truth.json"
TOPK_FILENAME = "topk.json"
METRICS_FILENAME = "metrics.json"
INITIAL_RETRIEVAL_FILENAME = "initial_retrieval.json"
INITIAL_CLUSTERS_FILENAME = "initial_clusters.json"
CLUSTER_EXPANDED_PASSAGES_FILENAME = "cluster_expanded_passages.json"

_TOPK_PREVIEW = 5
_OPENCODE_LOG_KEYS = frozenset({"stdout", "stderr"})
_OPENCODE_ROUND_DROP_KEYS = frozenset({"message_preview"})


def _ground_truth_counts(ground_truth: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(ground_truth, dict):
        return None
    counts: Dict[str, Any] = {}
    for key in ("n_table", "n_text", "n_synth_text"):
        if key in ground_truth:
            counts[key] = ground_truth[key]
    for list_key, count_key in (
        ("table", "n_table"),
        ("text", "n_text"),
        ("synth_text", "n_synth_text"),
    ):
        if count_key not in counts and list_key in ground_truth:
            value = ground_truth[list_key]
            if isinstance(value, list):
                counts[count_key] = len(value)
    return counts or None


def _slim_finding_rubric(
    research_quality: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    rubric = research_quality.get("rubric")
    if not rubric:
        return None
    scores = extract_rubric_scores(rubric)
    if not any(value is not None for value in scores.values()):
        return None
    return {
        "finding_score": research_quality.get("finding_score"),
        "scores": scores,
    }


def _build_findings(
    iterations: List[Dict[str, Any]],
    per_step: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    per_step_by_step = {
        int(step["step"]): step for step in (per_step or []) if step.get("step") is not None
    }
    findings: List[Dict[str, Any]] = []
    for iteration in iterations:
        if iteration.get("finding_idx") is None:
            continue
        step = int(iteration["step"])
        research_quality = (
            (per_step_by_step.get(step) or {}).get("research_quality") or {}
        )
        findings.append(
            {
                "finding_idx": iteration["finding_idx"],
                "step": step,
                "sub_question": iteration.get("sub_question") or "",
                "answer": iteration.get("answer") or "",
                "tables_used": iteration.get("tables_used") or [],
                "passages_cited": iteration.get("passages_cited") or [],
                "rubric": _slim_finding_rubric(research_quality),
            }
        )
    findings.sort(key=lambda item: int(item.get("finding_idx") or 0))
    return findings


def _build_status_block(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    if metrics is not None and metrics.get("budget_steps_completed") is not None:
        status["budget_steps_completed"] = metrics["budget_steps_completed"]
    return status


def slim_opencode_meta(
    opencode_meta: Optional[Dict[str, Any]],
    query_dir: str,
) -> Optional[Dict[str, Any]]:
    if not opencode_meta:
        return None
    slim = {
        key: value
        for key, value in opencode_meta.items()
        if key not in _OPENCODE_LOG_KEYS and key != "usage"
    }
    rounds = slim.get("rounds")
    if isinstance(rounds, list):
        slim["rounds"] = [
            {
                key: value
                for key, value in round_item.items()
                if key not in _OPENCODE_ROUND_DROP_KEYS
            }
            for round_item in rounds
            if isinstance(round_item, dict)
        ]
    for path_key in ("stdout_path", "stderr_path", "prompt_path"):
        path_value = slim.get(path_key)
        if isinstance(path_value, str) and path_value:
            slim[path_key] = os.path.relpath(path_value, query_dir)
    return slim


def _artifact_paths(
    query_dir: str,
    *,
    opencode_meta: Optional[Dict[str, Any]],
    initial_retrieval_path: Optional[str],
    initial_clusters_path: Optional[str],
    include_ground_truth: bool,
) -> Dict[str, str]:
    artifacts = {
        "metrics": METRICS_FILENAME,
        "iterations": "iteration_{step:03d}.json",
        "topk": TOPK_FILENAME,
    }
    if include_ground_truth:
        artifacts["ground_truth"] = GROUND_TRUTH_FILENAME
    if opencode_meta:
        artifacts["opencode_stdout"] = "opencode_stdout.txt"
        artifacts["opencode_stderr"] = "opencode_stderr.txt"
        prompt_path = opencode_meta.get("prompt_path")
        if isinstance(prompt_path, str) and prompt_path:
            artifacts["opencode_prompt"] = os.path.relpath(prompt_path, query_dir)
    if initial_retrieval_path and os.path.isfile(initial_retrieval_path):
        artifacts["initial_retrieval"] = os.path.relpath(
            initial_retrieval_path,
            query_dir,
        )
    if initial_clusters_path and os.path.isfile(initial_clusters_path):
        artifacts["initial_clusters"] = os.path.relpath(
            initial_clusters_path,
            query_dir,
        )
    expanded_path = os.path.join(query_dir, CLUSTER_EXPANDED_PASSAGES_FILENAME)
    if os.path.isfile(expanded_path):
        artifacts["cluster_expanded_passages"] = CLUSTER_EXPANDED_PASSAGES_FILENAME
    return artifacts


def build_result_json(
    *,
    query_id: str,
    user_query: str,
    coverage: Optional[str],
    method: str,
    answer: str,
    iterations: List[Dict[str, Any]],
    ground_truth: Optional[Dict[str, Any]],
    topk_table_ids: List[str],
    topk_passage_ids: List[str],
    total_inference_clusters: int,
    retained_clusters: int,
    time_taken: float,
    usage: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    opencode_meta: Optional[Dict[str, Any]] = None,
    opencode_exec: bool = False,
    query_dir: Optional[str] = None,
    initial_retrieval_path: Optional[str] = None,
    initial_clusters_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the human-readable result payload (insertion order preserved)."""
    per_step = (metrics or {}).get("per_step")
    findings = _build_findings(iterations, per_step)
    summary: Dict[str, Any] = {
        "time_taken": time_taken,
        "usage": usage,
    }
    research_quality = research_quality_summary_block(metrics)
    if research_quality is not None:
        summary["research_quality"] = research_quality
    retrieval = retrieval_summary_block(metrics)
    if retrieval is not None:
        summary["retrieval"] = retrieval
    operational = operational_summary_block(metrics)
    if operational is not None:
        summary["operational"] = operational
    status = _build_status_block(metrics)
    if status:
        summary["status"] = status

    run: Dict[str, Any] = {
        "total_inference_clusters": total_inference_clusters,
        "retained_clusters": retained_clusters,
        "topk": {
            "n_tables": len(topk_table_ids),
            "n_passages": len(topk_passage_ids),
            "table_ids_preview": topk_table_ids[:_TOPK_PREVIEW],
            "passage_ids_preview": topk_passage_ids[:_TOPK_PREVIEW],
        },
    }
    if query_dir:
        run["artifacts"] = _artifact_paths(
            query_dir,
            opencode_meta=opencode_meta,
            initial_retrieval_path=initial_retrieval_path,
            initial_clusters_path=initial_clusters_path,
            include_ground_truth=ground_truth is not None,
        )

    result: Dict[str, Any] = {
        "query_id": query_id,
        "user_query": user_query,
        "coverage": coverage,
        "method": method,
        "answer": answer,
        "summary": summary,
        "findings": findings,
        "run": run,
    }
    ground_truth_counts = _ground_truth_counts(ground_truth)
    if ground_truth_counts is not None:
        result["ground_truth"] = ground_truth_counts
    if opencode_exec:
        result["opencode_exec"] = True
    if opencode_meta and query_dir:
        slim_opencode = slim_opencode_meta(opencode_meta, query_dir)
        if slim_opencode:
            result["opencode"] = slim_opencode
    if metrics is not None:
        result["metrics_path"] = METRICS_FILENAME
    return result


def save_query_result(
    query_dir: str,
    result: Dict[str, Any],
    *,
    ground_truth: Optional[Dict[str, Any]] = None,
    topk_table_ids: Optional[List[str]] = None,
    topk_passage_ids: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist result.json and machine-readable sidecar artifacts."""
    if ground_truth is not None:
        save_json(os.path.join(query_dir, GROUND_TRUTH_FILENAME), ground_truth)
    if topk_table_ids is not None or topk_passage_ids is not None:
        save_json(
            os.path.join(query_dir, TOPK_FILENAME),
            {
                "topk_table_ids": list(topk_table_ids or []),
                "topk_passage_ids": list(topk_passage_ids or []),
            },
        )
    if metrics is not None:
        save_json(
            os.path.join(query_dir, METRICS_FILENAME),
            format_metrics_for_disk(metrics),
        )
    result_path = os.path.join(query_dir, "result.json")
    save_json(result_path, result)
    return result_path


def load_query_ground_truth(query_dir: str) -> Optional[Dict[str, Any]]:
    sidecar_path = os.path.join(query_dir, GROUND_TRUTH_FILENAME)
    if os.path.isfile(sidecar_path):
        return load_json(sidecar_path)
    return None


def load_query_topk(query_dir: str) -> tuple[List[str], List[str]]:
    sidecar_path = os.path.join(query_dir, TOPK_FILENAME)
    if not os.path.isfile(sidecar_path):
        return [], []
    payload = load_json(sidecar_path)
    return (
        list(payload.get("topk_table_ids") or []),
        list(payload.get("topk_passage_ids") or []),
    )


def load_query_metrics(query_dir: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metrics_path = result.get("metrics_path") or METRICS_FILENAME
    if not os.path.isabs(metrics_path):
        metrics_path = os.path.join(query_dir, metrics_path)
    if os.path.isfile(metrics_path):
        return load_json(metrics_path)
    return None


def load_iterations_from_artifacts(
    query_dir: str,
    budget: int,
) -> List[Dict[str, Any]]:
    iterations: List[Dict[str, Any]] = []
    for step in range(1, budget + 1):
        path = os.path.join(query_dir, f"iteration_{step:03d}.json")
        if not os.path.isfile(path):
            continue
        payload = load_json(path)
        if payload.get("skipped"):
            continue
        iterations.append(payload)
    return iterations


def result_budget(result: Dict[str, Any], default: int = 0) -> int:
    if default > 0:
        return default
    research_quality = (result.get("summary") or {}).get("research_quality") or {}
    budget = research_quality.get("budget")
    if budget:
        return int(budget)
    run_status = (result.get("summary") or {}).get("status") or {}
    completed = run_status.get("budget_steps_completed")
    if completed:
        return int(completed)
    findings = result.get("findings")
    if isinstance(findings, list) and findings:
        return max(int(item.get("step") or 0) for item in findings)
    return 0


def result_retained_clusters(result: Dict[str, Any]) -> int:
    run = result.get("run") or {}
    return int(run.get("retained_clusters") or 0)


def result_time_taken(result: Dict[str, Any]) -> float:
    summary = result.get("summary") or {}
    return float(summary.get("time_taken") or 0.0)


def result_usage(result: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.get("summary") or {}
    return dict(summary.get("usage") or {})


def query_dirs_for_results(
    log_dir: str,
    results: List[Dict[str, Any]],
) -> Dict[str, str]:
    return {
        str(result.get("query_id")): os.path.join(log_dir, str(result.get("query_id")))
        for result in results
        if result.get("query_id") is not None
    }
