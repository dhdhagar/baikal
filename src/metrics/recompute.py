"""Recompute evaluation metrics from saved run artifacts."""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

from src.embedding_client import get_embedding_client
from src.metrics.aggregate import aggregate_run_metrics
from src.metrics.run_summary_json import format_metrics_summary_for_disk
from src.results_all import patch_results_all_metrics
from src.metrics.common import (
    assign_finding_indices,
    count_lake_tables,
    extract_gt_passage_ids,
    extract_gt_table_ids,
    load_uid_to_passage_id_mapping,
    load_uid_to_table_id_mapping,
    reorder_iteration_payload,
    sync_finding_indices_to_query_dir,
)
from src.metrics.progress import metrics_reporter
from src.metrics.research_quality import build_judge_clients
from src.metrics.tracker import MetricsTracker
from src.result_json import (
    METRICS_FILENAME,
    build_result_json,
    load_query_ground_truth,
    load_query_topk,
    result_budget,
    result_retained_clusters,
    result_usage,
    save_query_result,
)
from src.tracking import reset_tracker
from src.utils import (
    load_json,
    load_passage_descriptions_for_metrics,
    resolve_default_paths,
    save_json,
)


def _find_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()
        cur = parent


def _resolve_path(path: Optional[str], project_root: str) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _prepare_run_args(results_dir: str, args: Namespace) -> Namespace:
    """Resolve relative paths from args.json against the project root."""
    run_dir = results_dir if os.path.isfile(os.path.join(results_dir, "args.json")) else os.path.dirname(
        os.path.abspath(results_dir)
    )
    project_root = _find_project_root(run_dir)
    for attr in (
        "data_dir",
        "inference_clusters_path",
        "table_embeddings_path",
        "passage_embeddings_path",
        "corpus_path",
        "uid_to_passage_id_path",
        "uid_to_table_id_path",
        "passage_descriptions_path",
        "tables_lake_dir",
    ):
        value = getattr(args, attr, None)
        if value:
            setattr(args, attr, _resolve_path(value, project_root))
    resolve_default_paths(args)
    return args


def _load_run_args(results_dir: str) -> Namespace:
    args_path = os.path.join(results_dir, "args.json")
    if not os.path.isfile(args_path):
        parent = os.path.dirname(os.path.abspath(results_dir))
        args_path = os.path.join(parent, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(
            f"Missing args.json near {results_dir!r}; cannot infer run configuration."
        )
    raw = load_json(args_path)
    args = Namespace(**raw)
    return _prepare_run_args(os.path.dirname(args_path), args)


def discover_query_dirs(results_dir: str) -> List[str]:
    """Return query output directories that contain a result.json file."""
    result_path = os.path.join(results_dir, "result.json")
    if os.path.isfile(result_path):
        return [results_dir]

    query_dirs: List[str] = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "result.json")):
            query_dirs.append(path)
    return query_dirs


def _clusters_excluded_step(iteration_payload: Optional[dict]) -> int:
    if not iteration_payload:
        return 0
    excluded = iteration_payload.get("excluded_clusters")
    if isinstance(excluded, list):
        return len(excluded)
    return 0


def _load_budget_steps(query_dir: str, budget: int) -> List[Tuple[int, Optional[dict], int]]:
    steps: List[Tuple[int, Optional[dict], int]] = []
    for step in range(1, budget + 1):
        path = os.path.join(query_dir, f"iteration_{step:03d}.json")
        if not os.path.isfile(path):
            steps.append((step, None, 0))
            continue
        payload = load_json(path)
        if payload.get("skipped"):
            steps.append((step, None, _clusters_excluded_step(payload)))
        else:
            steps.append((step, payload, _clusters_excluded_step(payload)))
    return steps


def _sync_iterations_to_disk(query_dir: str, iterations: List[dict]) -> None:
    """Write recomputed iteration fields (citations, metrics, finding_idx) to disk."""
    for iteration in iterations:
        step = iteration.get("step")
        if not step:
            continue
        path = os.path.join(query_dir, f"iteration_{int(step):03d}.json")
        if not os.path.isfile(path):
            continue
        save_json(path, reorder_iteration_payload(dict(iteration)))


def _load_prior_query_metrics(query_dir: str) -> Optional[Dict[str, Any]]:
    metrics_path = os.path.join(query_dir, METRICS_FILENAME)
    if os.path.isfile(metrics_path):
        return load_json(metrics_path)
    return None


def _prior_has_judge_metrics(prior_metrics: Optional[Dict[str, Any]]) -> bool:
    if not prior_metrics:
        return False
    for step_metrics in prior_metrics.get("per_step") or []:
        research_quality = step_metrics.get("research_quality") or {}
        if research_quality.get("rubric"):
            return True
    return False


def _per_step_by_step(per_step: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_step: Dict[int, Dict[str, Any]] = {}
    for step_metrics in per_step or []:
        step = step_metrics.get("step")
        if step is not None:
            by_step[int(step)] = step_metrics
    return by_step


def _merge_preserved_research_quality(
    new_metrics: Dict[str, Any],
    prior_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep prior judge scores when recomputing without the LLM judge."""
    merged = dict(new_metrics)
    prior_research_quality = prior_metrics.get("research_quality")
    if prior_research_quality:
        merged["research_quality"] = dict(prior_research_quality)
    prior_judge_usage = prior_metrics.get("judge_usage")
    if prior_judge_usage:
        merged["judge_usage"] = prior_judge_usage

    prior_by_step = _per_step_by_step(prior_metrics.get("per_step") or [])
    merged_per_step: List[Dict[str, Any]] = []
    for step_metrics in merged.get("per_step") or []:
        step = step_metrics.get("step")
        if step is None:
            merged_per_step.append(step_metrics)
            continue
        prior_step = prior_by_step.get(int(step)) or {}
        prior_research_quality_step = prior_step.get("research_quality")
        if prior_research_quality_step:
            updated = dict(step_metrics)
            updated["research_quality"] = dict(prior_research_quality_step)
            merged_per_step.append(updated)
        else:
            merged_per_step.append(step_metrics)
    merged["per_step"] = merged_per_step
    return merged


def _apply_per_step_to_iterations(
    iterations: List[dict],
    per_step: List[Dict[str, Any]],
) -> None:
    by_step = _per_step_by_step(per_step)
    for iteration in iterations:
        step = iteration.get("step")
        if step is None:
            continue
        step_metrics = by_step.get(int(step))
        if step_metrics is not None:
            iteration["metrics"] = step_metrics


def recompute_query_metrics(
    query_dir: str,
    args: Namespace,
    *,
    inference_clusters: List[dict],
    research_quality_enabled: bool = True,
    judge_models: Optional[str] = None,
    compute_embed_diversity: Optional[bool] = None,
    reporter: Optional[Any] = None,
) -> Dict[str, Any]:
    result_path = os.path.join(query_dir, "result.json")
    result = load_json(result_path)
    prior_metrics = _load_prior_query_metrics(query_dir)
    user_query = result.get("user_query") or ""
    ground_truth = load_query_ground_truth(query_dir)
    topk_table_ids, topk_passage_ids = load_query_topk(query_dir)
    budget = int(getattr(args, "budget", 0) or 0) or result_budget(result)

    reset_tracker()

    uid_to_table_id = load_uid_to_table_id_mapping(getattr(args, "uid_to_table_id_path", ""))
    gt_tables = extract_gt_table_ids(ground_truth, uid_to_table_id=uid_to_table_id)
    gt_passages = None
    if getattr(args, "use_passages", False):
        uid_to_passage_id = load_uid_to_passage_id_mapping(
            getattr(args, "uid_to_passage_id_path", "")
        )
        gt_passages = extract_gt_passage_ids(
            ground_truth,
            uid_to_passage_id=uid_to_passage_id,
            passage_type=getattr(args, "passage_type", "synth"),
        )

    embed_diversity = (
        compute_embed_diversity
        if compute_embed_diversity is not None
        else bool(getattr(args, "compute_embed_diversity", False))
    )
    embedder = None
    if embed_diversity:
        embedder = get_embedding_client(
            provider=args.embedding_provider,
            model=args.embedding_model,
            gpu=args.gpu,
        )
    judge_llms = []
    if research_quality_enabled:
        models = judge_models if judge_models is not None else getattr(args, "judge_models", "gpt-5-mini")
        judge_llms = build_judge_clients(args.llm_provider, models)
    passage_descriptions = load_passage_descriptions_for_metrics(args, query_dir=query_dir)

    tracker = MetricsTracker(
        user_query=user_query,
        budget=budget,
        gt_tables=gt_tables,
        total_lake_tables=count_lake_tables(args.tables_lake_dir),
        topk_table_ids=topk_table_ids,
        inference_clusters=inference_clusters,
        initial_candidate_clusters=result_retained_clusters(result),
        k_relevant_tables=int(getattr(args, "k_relevant_tables", 0) or 0),
        embedder=embedder,
        compute_embed_diversity=embed_diversity,
        gt_passages=gt_passages,
        topk_passage_ids=topk_passage_ids,
        k_relevant_passages=int(getattr(args, "k_relevant_passages", 0) or 0),
        use_passages=bool(getattr(args, "use_passages", False)),
        judge_llms=judge_llms,
        judge_temperature=float(getattr(args, "judge_temperature", 1.0) or 1.0),
        research_quality_enabled=research_quality_enabled,
        reporter=reporter,
        passage_descriptions=passage_descriptions,
    )

    iterations: List[dict] = []
    for step, iteration, clusters_excluded_step in _load_budget_steps(query_dir, budget):
        if iteration is None:
            tracker.record_empty_step(step, clusters_excluded_step)
        else:
            tracker.record_iteration(iteration, step, clusters_excluded_step)
            iterations.append(iteration)

    final_report = result.get("answer") or ""
    assign_finding_indices(iterations)
    sync_finding_indices_to_query_dir(query_dir, iterations)
    metrics = tracker.finalize(final_report, iterations)

    if not research_quality_enabled and _prior_has_judge_metrics(prior_metrics):
        metrics = _merge_preserved_research_quality(metrics, prior_metrics)
        _apply_per_step_to_iterations(iterations, metrics.get("per_step") or [])

    _sync_iterations_to_disk(query_dir, iterations)

    run_block = result.get("run") or {}
    time_taken = result.get("time_taken")
    if time_taken is None:
        time_taken = (result.get("summary") or {}).get("time_taken") or 0.0
    total_inference_clusters = result.get("total_inference_clusters")
    if total_inference_clusters is None:
        total_inference_clusters = int(run_block.get("total_inference_clusters") or 0)

    updated = build_result_json(
        query_id=str(result.get("query_id") or os.path.basename(query_dir.rstrip(os.sep))),
        user_query=user_query,
        coverage=result.get("coverage"),
        method=str(result.get("method") or "pipeline"),
        answer=final_report,
        iterations=iterations,
        ground_truth=ground_truth,
        topk_table_ids=topk_table_ids,
        topk_passage_ids=topk_passage_ids,
        total_inference_clusters=total_inference_clusters,
        retained_clusters=result_retained_clusters(result),
        time_taken=float(time_taken),
        usage=result_usage(result),
        metrics=metrics,
        opencode_meta=result.get("opencode"),
        query_dir=query_dir,
    )
    save_query_result(
        query_dir,
        updated,
        ground_truth=ground_truth,
        topk_table_ids=topk_table_ids,
        topk_passage_ids=topk_passage_ids,
        metrics=metrics,
    )
    return metrics


def _summary_output_path(results_dir: str) -> str:
    return os.path.join(results_dir, "metrics_summary.json")


def _query_id_from_dir(query_dir: str, result: Dict[str, Any]) -> str:
    return str(result.get("query_id") or os.path.basename(query_dir.rstrip(os.sep)))


def recompute_results_dir(
    results_dir: str,
    *,
    research_quality_enabled: bool = True,
    judge_models: Optional[str] = None,
    compute_embed_diversity: Optional[bool] = None,
    args_override: Optional[Namespace] = None,
    silent: bool = False,
    rich_cli: bool = True,
) -> Dict[str, Any]:
    args = args_override or _load_run_args(results_dir)
    if args_override is not None:
        args = _prepare_run_args(results_dir, args)
    inference_clusters = load_json(args.inference_clusters_path)
    query_dirs = discover_query_dirs(results_dir)
    outputs: List[Dict[str, Any]] = []
    for query_dir in query_dirs:
        result_path = os.path.join(query_dir, "result.json")
        result = load_json(result_path)
        budget = int(getattr(args, "budget", 0) or 0) or result_budget(result)
        with metrics_reporter(
            budget,
            mode="bar",
            silent=silent,
            rich_cli=rich_cli,
            desc=f"Metrics {os.path.basename(query_dir)}",
        ) as reporter:
            metrics = recompute_query_metrics(
                query_dir,
                args,
                inference_clusters=inference_clusters,
                research_quality_enabled=research_quality_enabled,
                judge_models=judge_models,
                compute_embed_diversity=compute_embed_diversity,
                reporter=reporter,
            )
        outputs.append(
            {
                "query_id": _query_id_from_dir(query_dir, result),
                "query_dir": query_dir,
                "metrics": metrics,
                "coverage": result.get("coverage"),
            }
        )

    summary = aggregate_run_metrics(outputs)
    summary_path = _summary_output_path(results_dir)
    save_json(summary_path, format_metrics_summary_for_disk(summary))
    results = [
        load_json(os.path.join(item["query_dir"], "result.json"))
        for item in outputs
        if item.get("query_dir")
    ]
    patch_results_all_metrics(results_dir, summary, results)
    return {
        "summary_path": summary_path,
        "summary": summary,
        "queries": outputs,
    }
