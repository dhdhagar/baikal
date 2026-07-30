"""Pipeline orchestration for sequential query runs."""

from __future__ import annotations

import os
import time
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.cli_ui import init_ui
from src.metrics.aggregate import aggregate_run_metrics, format_run_metrics_summary, outputs_from_results
from src.metrics.run_summary_json import format_metrics_summary_for_disk
from src.results_all import METRICS_SUMMARY_FILENAME, build_results_all_payload
from src.opencode_usage import merge_nested_usage_dicts
from src.queries import resolve_queries
from src.query_runner import QueryRunner
from src.tracking import UsageSummary, format_usage_line, get_tracker, reset_tracker
from src.result_json import query_dirs_for_results, result_time_taken, result_usage
from src.utils import load_json, log, resolve_default_paths, save_json

_PATH_ATTRS = (
    "data_dir",
    "output_dir",
    "inference_clusters_path",
    "table_embeddings_path",
    "passage_embeddings_path",
    "corpus_path",
    "uid_to_passage_id_path",
    "uid_to_table_id_path",
    "passage_descriptions_path",
    "tables_lake_dir",
    "query_file",
)

RUN_QUERIES_FILENAME = "run_queries.json"


def make_log_dir(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        os.path.join(args.output_dir, timestamp)
        if args.use_timestamp_dir
        else args.output_dir
    )
    os.makedirs(log_dir, exist_ok=True)
    return os.path.abspath(log_dir)


def find_latest_run_dir(output_dir: str) -> Optional[str]:
    """Return the most recently modified run dir containing run_queries.json."""
    root = os.path.abspath(output_dir)
    if not os.path.isdir(root):
        return None
    candidates = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, RUN_QUERIES_FILENAME))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def absolutize_path_attrs(args) -> None:
    """Resolve configured paths to absolute paths (safe for Slurm workers)."""
    for attr in _PATH_ATTRS:
        value = getattr(args, attr, None)
        if value:
            setattr(args, attr, os.path.abspath(value))


def save_run_args(args, log_dir: str) -> str:
    path = os.path.join(log_dir, "args.json")
    save_json(path, vars(args))
    return path


def save_run_queries(log_dir: str, queries: List[Dict[str, Any]]) -> None:
    save_json(os.path.join(log_dir, RUN_QUERIES_FILENAME), queries)


def load_run_queries(log_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(log_dir, RUN_QUERIES_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{RUN_QUERIES_FILENAME} not found in {log_dir}; "
            "only --submitit runs can be merged (re-run with --submitit)."
        )
    return load_json(path)


def args_from_dict(raw: Dict[str, Any]) -> Namespace:
    args = Namespace(**raw)
    resolve_default_paths(args)
    absolutize_path_attrs(args)
    return args


def merge_usage_dicts(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if any(isinstance(u, dict) and "total" in u for u in usages if u):
        return merge_nested_usage_dicts(usages)

    total = UsageSummary()
    for usage in usages:
        if usage:
            total.merge(UsageSummary(**usage))
    return total.to_dict()


def build_results_payload(
    results: List[Dict[str, Any]],
    *,
    compute_metrics: bool,
    run_time_taken: float,
    usage: Optional[Dict[str, Any]] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    query_dirs = query_dirs_for_results(log_dir, results) if log_dir else None
    metrics_summary = None
    if compute_metrics:
        metric_outputs = outputs_from_results(results, query_dirs=query_dirs)
        if metric_outputs:
            metrics_summary = aggregate_run_metrics(metric_outputs)
    payload = build_results_all_payload(
        results,
        time_taken=run_time_taken,
        usage=usage or merge_usage_dicts([result_usage(r) for r in results]),
        metrics_summary=metrics_summary,
    )
    if metrics_summary is not None:
        payload["metrics_summary"] = metrics_summary
    return payload


def write_results_all(log_dir: str, payload: Dict[str, Any]) -> str:
    path = os.path.join(log_dir, "results_all.json")
    metrics_summary = payload.get("metrics_summary")
    results_all = {
        key: value for key, value in payload.items() if key != "metrics_summary"
    }
    save_json(path, results_all)
    if metrics_summary:
        save_json(
            os.path.join(log_dir, METRICS_SUMMARY_FILENAME),
            format_metrics_summary_for_disk(metrics_summary),
        )
    return path


def print_run_completion(
    *,
    n_queries: int,
    results_path: str,
    usage: UsageSummary,
    metrics_summary: Optional[Dict[str, Any]] = None,
    silent: bool = False,
) -> None:
    """Print optional metrics summary, then the standard run footer (always last)."""
    if silent:
        return
    if metrics_summary:
        print()
        print(format_run_metrics_summary(metrics_summary))
    print(f"\nMerged {n_queries} queries into {results_path}")
    print(f"Run usage: {format_usage_line(usage)} ({usage.n_calls} API calls)")


def queries_missing_results(
    log_dir: str,
    queries: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Return query ids and records that lack result.json under log_dir."""
    missing_ids: List[str] = []
    for rec in queries:
        qid = str(rec["query_id"])
        path = os.path.join(log_dir, qid, "result.json")
        if not os.path.isfile(path):
            missing_ids.append(qid)
    missing_set = set(missing_ids)
    missing_queries = [q for q in queries if str(q["query_id"]) in missing_set]
    return missing_ids, missing_queries


def format_missing_results_error(missing_ids: List[str]) -> str:
    """Human-readable error for queries without result.json."""
    preview = ", ".join(missing_ids[:5]) + (" …" if len(missing_ids) > 5 else "")
    return f"Missing result.json for {len(missing_ids)} queries: {preview}"


def merge_results_from_disk(
    log_dir: str,
    queries: List[Dict[str, Any]],
    *,
    compute_metrics: bool = False,
    run_time_taken: Optional[float] = None,
) -> Dict[str, Any]:
    """Load per-query result.json files and build a run-level payload."""
    missing_ids, _ = queries_missing_results(log_dir, queries)
    if missing_ids:
        raise FileNotFoundError(format_missing_results_error(missing_ids))

    results: List[Dict[str, Any]] = []
    for rec in queries:
        qid = str(rec["query_id"])
        results.append(load_json(os.path.join(log_dir, qid, "result.json")))

    if run_time_taken is None:
        times = [result_time_taken(r) for r in results]
        run_time_taken = max(times) if times else 0.0

    return build_results_payload(
        results,
        compute_metrics=compute_metrics,
        run_time_taken=run_time_taken,
        log_dir=log_dir,
    )


def run_queries_sequential(
    args,
    log_dir: str,
    queries: List[Dict[str, Any]],
    inference_clusters: List[dict],
    *,
    ui,
) -> List[Dict[str, Any]]:
    runner = QueryRunner(args, inference_clusters, log_dir)
    results: List[Dict[str, Any]] = []

    if ui.rich_cli:
        for i, query in enumerate(queries, start=1):
            results.append(
                runner.run(query, query_index=i, query_total=len(queries))
            )
            ui.finish_query()
    else:
        for query in tqdm(queries, desc="Queries", disable=args.silent):
            results.append(runner.run(query))

    return results


def run_pipeline(args, log_dir: Optional[str] = None) -> str:
    """Run the full sequential pipeline and return results_all.json path."""
    if log_dir is None:
        log_dir = make_log_dir(args)

    ui = init_ui(silent=args.silent, rich_cli=args.rich_cli)
    reset_tracker()
    run_started = time.perf_counter()

    with ui.session():
        if ui.rich_cli:
            ui.configure_queries(1)
            ui.set_phase("Setup")

        from src.clustering import ensure_inference_clusters

        inference_clusters = ensure_inference_clusters(args)
        if args.use_clustering:
            log(
                f"Inference clusters: {len(inference_clusters)}",
                silent=args.silent,
            )
        else:
            log(
                "Clustering disabled; one top-k cluster per query.",
                silent=args.silent,
            )

        if args.expand_inference_clusters:
            log(
                "expand_inference_clusters=True is not implemented yet.",
                silent=args.silent,
            )

        queries = resolve_queries(args)
        if ui.rich_cli:
            ui.configure_queries(len(queries))

        results = run_queries_sequential(
            args, log_dir, queries, inference_clusters, ui=ui
        )
        tracker = get_tracker()
        payload = build_results_payload(
            results,
            compute_metrics=bool(args.compute_metrics),
            run_time_taken=time.perf_counter() - run_started,
            usage=tracker.to_dict(),
            log_dir=log_dir,
        )
        results_path = write_results_all(log_dir, payload)

        ui.set_results_summary(
            results_path,
            n_queries=len(queries),
            n_completed=len(results),
            usage_summary=format_usage_line(tracker.total),
        )

    print_run_completion(
        n_queries=len(results),
        results_path=results_path,
        usage=tracker.total,
        metrics_summary=payload.get("metrics_summary"),
        silent=args.silent,
    )

    return results_path
