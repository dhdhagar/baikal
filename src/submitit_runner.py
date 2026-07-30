"""Parallel query execution via submitit (Slurm or local process pool)."""

from __future__ import annotations

import os
import sys
import time
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.cli_ui import init_ui
from src.pipeline import (
    args_from_dict,
    find_latest_run_dir,
    format_missing_results_error,
    load_run_queries,
    merge_results_from_disk,
    print_run_completion,
    queries_missing_results,
    save_run_queries,
    write_results_all,
)
from src.queries import resolve_queries
from src.query_runner import QueryRunner
from src.tracking import UsageSummary, reset_tracker
from src.utils import load_json, set_seed

SUBMITIT_ARG_NAMES = frozenset(
    {
        "submitit",
        "local",
        "max_workers",
        "slurm_partition",
        "timeout_min",
        "cpus_per_task",
        "gpus_per_node",
        "mem_gb",
        "merge_only",
        "retry_missing",
        "run_dir",
    }
)

OPTIONAL_EXECUTOR_KEYS = frozenset(
    {"slurm_partition", "timeout_min", "cpus_per_task", "gpus_per_node", "mem_gb"}
)
CLI_EXECUTOR_DEFAULTS = {"local": False, "max_workers": 2}


def worker_args(args) -> Dict[str, Any]:
    """Pickle-friendly args for submitit workers (no Rich UI, no submitit flags)."""
    raw = {k: v for k, v in vars(args).items() if k not in SUBMITIT_ARG_NAMES}
    raw["rich_cli"] = False
    raw["silent"] = True
    raw["use_timestamp_dir"] = False
    return raw


def resolve_run_dir(args) -> Optional[str]:
    if args.run_dir:
        return os.path.abspath(args.run_dir)
    if args.merge_only:
        return find_latest_run_dir(args.output_dir)
    return None


def merge_only_display_args(cli_args) -> Dict[str, Any]:
    """Build args dict for logging during --merge_only (saved run + CLI overrides)."""
    log_dir = resolve_run_dir(cli_args)
    display: Dict[str, Any] = {
        "merge_only": True,
        "retry_missing": bool(getattr(cli_args, "retry_missing", False)),
        "recompute_metrics": bool(getattr(cli_args, "recompute_metrics", False)),
        "output_dir": getattr(cli_args, "output_dir", None),
        "run_dir": log_dir,
    }
    if not log_dir:
        display["_note"] = (
            "No run directory found; pass --run_dir or ensure --output_dir "
            "contains a prior --submitit run with run_queries.json."
        )
        return display

    args_path = os.path.join(log_dir, "args.json")
    if os.path.isfile(args_path):
        saved = load_json(args_path)
        display.update(merge_executor_settings(saved, cli_args))
    else:
        display["_note"] = f"args.json not found in {log_dir}"
    return display


def run_query_job(
    args_dict: Dict[str, Any],
    query_record: Dict[str, Any],
    log_dir: str,
    inference_clusters_path: str,
) -> Dict[str, Any]:
    """Submitit worker entrypoint: execute one query in an isolated process."""
    args = args_from_dict(args_dict)
    set_seed(args.seed)
    init_ui(silent=True, rich_cli=False)
    reset_tracker()

    if args.use_clustering:
        inference_clusters = load_json(inference_clusters_path)
    else:
        inference_clusters = []
    runner = QueryRunner(args, inference_clusters, log_dir)
    return runner.run(query_record)


def _usage_from_payload(payload: Dict[str, Any]) -> UsageSummary:
    usage_raw = payload["usage"]
    if "total" in usage_raw:
        return UsageSummary(**usage_raw["total"])
    return UsageSummary(**usage_raw)


def report_merged_results(
    log_dir: str,
    queries: List[Dict[str, Any]],
    *,
    compute_metrics: bool,
    run_time_taken: Optional[float] = None,
) -> str:
    payload = merge_results_from_disk(
        log_dir,
        queries,
        compute_metrics=compute_metrics,
        run_time_taken=run_time_taken,
    )
    results_path = write_results_all(log_dir, payload)
    usage = _usage_from_payload(payload)
    print_run_completion(
        n_queries=payload.get("n_completed", len(payload.get("queries") or [])),
        results_path=results_path,
        usage=usage,
        metrics_summary=payload.get("metrics_summary"),
    )
    return results_path


def _argv_has(flag: str) -> bool:
    return flag in sys.argv


def merge_executor_settings(saved: Dict[str, Any], cli_args) -> Dict[str, Any]:
    """Apply CLI submitit overrides; prefer saved args.json when flags are omitted."""
    merged = dict(saved)
    for key in OPTIONAL_EXECUTOR_KEYS:
        cli_val = getattr(cli_args, key, None)
        if cli_val is not None:
            merged[key] = cli_val
    if _argv_has("--local"):
        merged["local"] = True
    elif _argv_has("--no-local"):
        merged["local"] = False
    elif "local" not in merged:
        merged["local"] = CLI_EXECUTOR_DEFAULTS["local"]
    if _argv_has("--max_workers"):
        merged["max_workers"] = cli_args.max_workers
    elif "max_workers" not in merged:
        merged["max_workers"] = CLI_EXECUTOR_DEFAULTS["max_workers"]
    return merged


def load_run_args_for_retry(log_dir: str, cli_args) -> Namespace:
    """Rebuild run Namespace from args.json, with CLI submitit executor overrides."""
    args_path = os.path.join(log_dir, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(
            f"args.json not found in {log_dir}; --retry_missing requires a saved "
            "--submitit run."
        )
    saved = load_json(args_path)
    return args_from_dict(merge_executor_settings(saved, cli_args))


def run_retry_missing_queries(
    cli_args,
    log_dir: str,
    missing_queries: List[Dict[str, Any]],
) -> int:
    """Submit jobs for queries missing result.json into an existing run directory."""
    if not missing_queries:
        return 0

    run_args = load_run_args_for_retry(log_dir, cli_args)
    missing_ids = [str(q["query_id"]) for q in missing_queries]
    preview = ", ".join(missing_ids[:5]) + (" …" if len(missing_ids) > 5 else "")
    print(f"Retrying {len(missing_queries)} missing queries: {preview}")

    submitit_folder = os.path.join(log_dir, "submitit_logs")
    os.makedirs(submitit_folder, exist_ok=True)
    print(f"Submitit logs: {submitit_folder}\n")

    method = getattr(run_args, "method", "dpr_discovery")
    if method == "opencode":
        return _retry_missing_opencode(
            run_args,
            log_dir,
            missing_queries,
            submitit_folder,
        )
    return _retry_missing_pipeline(
        run_args,
        log_dir,
        missing_queries,
        submitit_folder,
    )


def _run_retry_jobs(
    run_args,
    submitit_folder: str,
    queries: List[Dict[str, Any]],
    submit_job: Callable[[Any, Dict[str, Any]], Tuple[str, Any]],
) -> int:
    executor = build_submitit_executor(submitit_folder, run_args)

    def submit_one(query: Dict[str, Any]) -> Tuple[str, Any]:
        return submit_job(executor, query)

    _completed, failures = submit_and_wait_batched(
        queries,
        local=bool(run_args.local),
        max_workers=run_args.max_workers,
        submit_one=submit_one,
    )
    if failures:
        print(f"\n{len(failures)} retry job(s) failed.", file=sys.stderr)
        return 1
    return 0


def _retry_missing_pipeline(
    run_args,
    log_dir: str,
    queries: List[Dict[str, Any]],
    submitit_folder: str,
) -> int:
    from src.clustering import ensure_inference_clusters

    inference_clusters = ensure_inference_clusters(run_args)
    if run_args.use_clustering:
        print(f"Inference clusters: {len(inference_clusters)}")
    else:
        print("Clustering disabled; one top-k cluster per query.")

    args_dict = worker_args(run_args)
    clusters_path = run_args.inference_clusters_path

    def submit_job(executor, query: Dict[str, Any]) -> Tuple[str, Any]:
        job = executor.submit(
            run_query_job,
            args_dict,
            query,
            log_dir,
            clusters_path,
        )
        return str(query["query_id"]), job

    return _run_retry_jobs(run_args, submitit_folder, queries, submit_job)


def _retry_missing_opencode(
    run_args,
    log_dir: str,
    queries: List[Dict[str, Any]],
    submitit_folder: str,
) -> int:
    from src.opencode import (
        _log_opencode_cluster_setup,
        _remove_materialized_sqlite,
        materialize_lake_sqlite,
        run_opencode_job,
        setup_opencode_inference_clusters,
    )
    from src.utils import log

    sqlite_db_path = materialize_lake_sqlite(
        run_args.tables_lake_dir,
        os.path.join(log_dir, "data_lake.sqlite"),
    )
    print(f"Materialized data lake SQLite: {sqlite_db_path}")

    try:
        inference_clusters = setup_opencode_inference_clusters(run_args)
        _log_opencode_cluster_setup(run_args, inference_clusters, use_print=True)

        args_dict = worker_args(run_args)
        clusters_path = run_args.inference_clusters_path

        def submit_job(executor, query: Dict[str, Any]) -> Tuple[str, Any]:
            job = executor.submit(
                run_opencode_job,
                args_dict,
                query,
                log_dir,
                sqlite_db_path,
                clusters_path,
            )
            return str(query["query_id"]), job

        return _run_retry_jobs(run_args, submitit_folder, queries, submit_job)
    finally:
        _remove_materialized_sqlite(sqlite_db_path)
        log(
            f"Removed materialized SQLite: {sqlite_db_path}",
            silent=run_args.silent,
        )


def run_merge_only(args) -> int:
    log_dir = resolve_run_dir(args)
    if not log_dir:
        print(
            "No run directory found; pass --run_dir or ensure --output_dir "
            "contains a prior --submitit run with run_queries.json.",
            file=sys.stderr,
        )
        return 1
    if not os.path.isdir(log_dir):
        print(f"Not a directory: {log_dir}", file=sys.stderr)
        return 1

    args_path = os.path.join(log_dir, "args.json")
    if getattr(args, "recompute_metrics", False):
        from src.metrics.recompute import recompute_results_dir

        recompute_results_dir(
            log_dir,
            research_quality_enabled=not getattr(args, "no_llm_judge", False),
            judge_models=getattr(args, "judge_models", "gpt-5-mini"),
            compute_embed_diversity=getattr(args, "compute_embed_diversity", False),
            silent=getattr(args, "silent", False),
            rich_cli=getattr(args, "rich_cli", True),
        )
        compute_metrics = True
    elif os.path.isfile(args_path):
        saved = load_json(args_path)
        compute_metrics = bool(saved.get("compute_metrics", args.compute_metrics))
    else:
        compute_metrics = bool(args.compute_metrics)

    queries = load_run_queries(log_dir)
    missing_ids, missing_queries = queries_missing_results(log_dir, queries)
    retry_rc = 0
    if missing_queries:
        if args.retry_missing:
            try:
                retry_rc = run_retry_missing_queries(args, log_dir, missing_queries)
            except FileNotFoundError as exc:
                print(exc, file=sys.stderr)
                return 1
        else:
            print(format_missing_results_error(missing_ids), file=sys.stderr)
            return 1

    try:
        report_merged_results(log_dir, queries, compute_metrics=compute_metrics)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return retry_rc


def run_submitit_pipeline(args, log_dir: str) -> int:
    submitit_folder = os.path.join(log_dir, "submitit_logs")
    os.makedirs(submitit_folder, exist_ok=True)

    print(f"Run directory: {log_dir}")
    print(f"Submitit logs: {submitit_folder}\n")

    from src.clustering import ensure_inference_clusters

    inference_clusters = ensure_inference_clusters(args)
    if args.use_clustering:
        print(f"Inference clusters: {len(inference_clusters)}")
    else:
        print("Clustering disabled; one top-k cluster per query.")

    queries = resolve_queries(args)
    if len(queries) <= 1:
        print("Warning: only one query; --submitit adds overhead vs sequential mode")

    save_run_queries(log_dir, queries)

    args_dict = worker_args(args)
    clusters_path = args.inference_clusters_path
    executor = build_submitit_executor(submitit_folder, args)

    run_started = time.perf_counter()

    def submit_one(query: Dict[str, Any]) -> Tuple[str, Any]:
        job = executor.submit(
            run_query_job,
            args_dict,
            query,
            log_dir,
            clusters_path,
        )
        return str(query["query_id"]), job

    completed, failures = submit_and_wait_batched(
        queries,
        local=bool(args.local),
        max_workers=args.max_workers,
        submit_one=submit_one,
    )

    if failures:
        print(
            f"\n{len(failures)} job(s) failed; merging {len(completed)} successful result(s).",
            file=sys.stderr,
        )
        if not completed:
            return 1
        completed_set = set(completed)
        queries = [q for q in queries if str(q["query_id"]) in completed_set]

    report_merged_results(
        log_dir,
        queries,
        compute_metrics=bool(args.compute_metrics),
        run_time_taken=time.perf_counter() - run_started,
    )
    return 1 if failures else 0


DEFAULT_LOCAL_SUBMITIT_TIMEOUT_MIN = 60


def build_submitit_executor(folder: str, args):
    import submitit

    if args.local:
        executor = submitit.LocalExecutor(folder=folder)
        timeout_min = (
            args.timeout_min
            if args.timeout_min is not None
            else DEFAULT_LOCAL_SUBMITIT_TIMEOUT_MIN
        )
        executor.update_parameters(timeout_min=timeout_min)
        return executor

    executor = submitit.AutoExecutor(folder=folder)
    slurm: Dict[str, Any] = {}
    if args.slurm_partition:
        slurm["slurm_partition"] = args.slurm_partition
    if args.timeout_min is not None:
        slurm["timeout_min"] = args.timeout_min
    if args.cpus_per_task is not None:
        slurm["cpus_per_task"] = args.cpus_per_task
    if args.gpus_per_node is not None:
        slurm["gpus_per_node"] = args.gpus_per_node
    if args.mem_gb is not None:
        slurm["mem_gb"] = args.mem_gb
    if slurm:
        executor.update_parameters(**slurm)
    return executor


def submit_and_wait_batched(
    queries: List[Dict[str, Any]],
    *,
    local: bool,
    max_workers: int,
    submit_one: Callable[[Dict[str, Any]], Tuple[str, Any]],
) -> Tuple[List[str], List[Tuple[str, Exception]]]:
    """Submit query jobs and wait for completion, batching when running locally."""
    if local:
        batch_size = max(1, max_workers)
        print(f"\nSubmitting {len(queries)} jobs (max {batch_size} concurrent) …")
    else:
        batch_size = len(queries) or 1
        print(f"\nSubmitting {len(queries)} jobs …")

    completed: List[str] = []
    failures: List[Tuple[str, Exception]] = []
    for batch_start in range(0, len(queries), batch_size):
        batch = queries[batch_start : batch_start + batch_size]
        jobs: List[Tuple[str, Any]] = []
        for query in batch:
            query_id, job = submit_one(query)
            jobs.append((query_id, job))
            print(f"  submitted {query_id} → job {job.job_id}")
        if batch_start == 0:
            print("\nWaiting for jobs …")
        batch_completed, batch_failures = wait_for_submitit_jobs(jobs)
        completed.extend(batch_completed)
        failures.extend(batch_failures)
    return completed, failures


def wait_for_submitit_jobs(
    jobs: List[Tuple[str, Any]],
) -> Tuple[List[str], List[Tuple[str, Exception]]]:
    completed: List[str] = []
    failures: List[Tuple[str, Exception]] = []
    for query_id, job in jobs:
        try:
            job.result()
            completed.append(query_id)
            print(f"  finished {query_id} (job {job.job_id})")
        except Exception as exc:
            failures.append((query_id, exc))
            print(f"  FAILED {query_id} (job {job.job_id}): {exc}", file=sys.stderr)
    return completed, failures
