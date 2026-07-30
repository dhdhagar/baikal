"""Per-query budget loop: cluster selection, sub-questions, SQL, answers."""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, StateGraph

from src.cluster_prior import elicit_cluster_prior
from src.cluster_selection import (
    ClusterSelectionState,
    bandit_reward_from_iteration,
    select_cluster,
)
from tqdm import tqdm

from src.cli_ui import get_ui
from src.embeddings import (
    cluster_similarity_rank_summary,
    filter_clusters_by_tables_passages,
    get_passage_similarity_ranks,
    get_table_similarity_ranks,
    top_k_ids_from_ranks,
    top_k_table_ids_from_ranks,
)
from src.metrics.progress import make_metrics_reporter
from src.embedding_client import get_embedding_client
from src.llm import get_llm_client
from src.metrics import (
    MetricsTracker,
    assign_finding_indices,
    build_judge_clients,
    count_lake_tables,
    extract_gt_passage_ids,
    extract_gt_table_ids,
    load_uid_to_passage_id_mapping,
    load_uid_to_table_id_mapping,
    is_report_eligible_iteration,
    is_successful_iteration,
    sync_finding_indices_to_query_dir,
)
from src.report import generate_final_report
from src.result_json import build_result_json, save_query_result
from src.sql_db import (
    build_cluster_sqlite,
    execute_sql,
    fetch_sample_rows,
    load_cluster_table_metas,
)
from src.sql_qa import (
    answer_from_context,
    answer_from_sql_results,
    decide_sql_needed,
    generate_sql,
    refine_sql_with_error,
)
from src.passage_expansion import (
    EXPANSION_SUBQUESTIONS_K,
    ExpansionMode,
    decide_passage_expansion,
    grep_passage_ids,
    passages_from_ids,
)
from src.opencode_usage import combine_query_usage, format_nested_usage_line, merge_usage_summaries
from src.opencode_exec import (
    expansion_grep_keywords,
    opencode_expansion_trigger,
    run_opencode_subquestion,
)
from src.result_json import slim_opencode_meta
from src.subquestions import (
    filter_semantically_duplicate_subquestions,
    generate_subquestions,
    select_subquestion,
)
from src.tracking import format_usage_line, get_tracker
from src.utils import ensure_dir, load_json, log, save_json


class SQLLoopState(TypedDict, total=False):
    attempt_count: int
    max_attempts: int
    refine_strategy: str
    empty_result: bool
    last_sql: str
    last_error: Optional[str]
    last_row_count: int
    attempts_log: List[Dict[str, Any]]


class SubquestionExecResult(TypedDict, total=False):
    needs_sql: bool
    sql: str
    exec_result: Dict[str, Any]
    sql_attempts: List[Dict[str, Any]]
    answer: str
    opencode_meta: Optional[Dict[str, Any]]
    opencode_usage: Optional[Dict[str, Any]]
    opencode_failure: Optional[Dict[str, Any]]
    cluster_expansion: Optional[Dict[str, Any]]
    rank_summary: dict


def _pick_refine_strategy(
    attempt_idx: int,
    last_ok: bool,
    last_row_count: int,
    last_error: Optional[str],
) -> str:
    """
    Inspired by Stage-3's reason_node in pipelinenew_query.py.
    We keep the same *shape* of the retry phases without reproducing every heuristic.
    """
    if attempt_idx <= 0:
        return "generate"
    if attempt_idx == 1:
        return "simplify_drop_join"
    if last_error:
        return "debug"
    if last_ok and last_row_count == 0:
        return "discovery"
    return "debug"


def _run_subquestion_sql_loop_langgraph(
    *,
    llm,
    conn,
    sub_question: str,
    schema_string: str,
    samples_string: str,
    temperature: float,
    max_attempts: int,
) -> Dict[str, Any]:
    """
    LangGraph SQL loop in the style of Stage-3's _run_subquestion_sql_loop_langgraph:
    reason -> propose_sql -> execute_sql with conditional retries.

    Attempt 0 generates SQL; subsequent attempts refine based on observed tool output.
    """
    ui = get_ui()
    rich = ui.rich_cli

    def reason_node(state: SQLLoopState) -> SQLLoopState:
        attempt_idx = int(state.get("attempt_count", 0))
        attempts = state.get("attempts_log") or []
        last_ok = bool(attempts[-1].get("ok")) if attempts else False
        last_row_count = int(attempts[-1].get("row_count", 0) or 0) if attempts else 0
        last_error = attempts[-1].get("error") if attempts else None

        empty = bool(last_ok) and last_row_count == 0 and not last_error
        strategy = _pick_refine_strategy(
            attempt_idx, last_ok, last_row_count, last_error
        )
        return {
            "empty_result": empty,
            "refine_strategy": strategy,
            "last_error": last_error,
            "last_row_count": last_row_count,
        }

    def propose_sql_node(state: SQLLoopState) -> SQLLoopState:
        attempt_idx = int(state.get("attempt_count", 0))
        last_sql = state.get("last_sql", "")
        last_error = state.get("last_error")
        empty_result = bool(state.get("empty_result", False))

        if attempt_idx == 0 or not last_sql:
            sql = generate_sql(
                llm,
                sub_question,
                schema_string,
                samples_string,
                temperature,
            )
        else:
            sql = refine_sql_with_error(
                llm,
                sub_question,
                schema_string,
                samples_string,
                previous_sql=last_sql,
                error_message=last_error,
                empty_result=empty_result,
                temperature=temperature,
                attempts_log=state.get("attempts_log") or [],
            )
        ui.set_sql_result(sql)
        if rich:
            ui.set_status(
                f"SQL loop (attempt {attempt_idx + 1}/{max_attempts})"
            )
        return {"last_sql": sql}

    def execute_sql_node(state: SQLLoopState) -> SQLLoopState:
        attempt_idx = int(state.get("attempt_count", 0))
        sql = state.get("last_sql", "")
        result = execute_sql(conn, sql)
        ui.set_sql_result(sql, result)
        if rich:
            ui.set_status(
                f"SQL loop (attempt {attempt_idx + 1}/{max_attempts})"
            )

        attempts = list(state.get("attempts_log") or [])
        attempts.append(
            {
                "attempt": attempt_idx + 1,
                "sql": sql,
                "ok": bool(result.get("ok")),
                "row_count": int(result.get("row_count", 0) or 0),
                "error": result.get("error"),
            }
        )

        return {
            "attempt_count": attempt_idx + 1,
            "attempts_log": attempts,
            "last_sql": sql,
            "last_error": result.get("error"),
            "last_row_count": int(result.get("row_count", 0) or 0),
        }

    def should_continue(state: SQLLoopState) -> str:
        attempts = state.get("attempts_log") or []
        if attempts:
            last = attempts[-1]
            if last.get("ok") and int(last.get("row_count", 0) or 0) > 0:
                return "done"
        if int(state.get("attempt_count", 0)) >= int(
            state.get("max_attempts", max_attempts)
        ):
            return "done"
        return "retry"

    graph = StateGraph(SQLLoopState)
    graph.add_node("reason", reason_node)
    graph.add_node("propose_sql", propose_sql_node)
    graph.add_node("execute_sql", execute_sql_node)

    graph.set_entry_point("reason")
    graph.add_edge("reason", "propose_sql")
    graph.add_edge("propose_sql", "execute_sql")
    graph.add_conditional_edges(
        "execute_sql",
        should_continue,
        {"retry": "reason", "done": END},
    )

    app = graph.compile()
    final: SQLLoopState = app.invoke(
        {
            "attempt_count": 0,
            "max_attempts": max_attempts,
            "attempts_log": [],
            "last_sql": "",
            "last_error": None,
            "last_row_count": 0,
            "empty_result": False,
            "refine_strategy": "generate",
        }
    )

    attempts_log = final.get("attempts_log") or []
    last_sql = final.get("last_sql") or ""
    exec_result = (
        execute_sql(conn, last_sql)
        if last_sql
        else {"ok": False, "error": "no sql", "row_count": 0, "rows": []}
    )
    return {"sql": last_sql, "execution": exec_result, "attempts_log": attempts_log}


class QueryRunner:
    """Runs the budget loop for one user query."""

    def __init__(self, args, inference_clusters: List[dict], log_dir: str):
        self.args = args
        self.inference_clusters = inference_clusters
        self.log_dir = log_dir
        self.rng = random.Random(args.seed)
        self.llm = get_llm_client(args.llm_provider, args.llm_model)
        self.embedder = get_embedding_client(
            args.embedding_provider,
            args.embedding_model,
            gpu=args.gpu,
        )

        # Per-query state across budget iterations
        self.answered_subquestions: List[str] = []
        # cluster_id -> untried sub-questions for that cluster (reused across budget)
        self.pending_subquestions: Dict[str, List[str]] = {}
        # cluster_id -> sub-questions already tried for that cluster
        self.cluster_answered_subquestions: Dict[str, List[str]] = {}
        self.cluster_selection_state = ClusterSelectionState()
        self.iterations: List[Dict[str, Any]] = []
        self.topk_table_ids: List[str] = []
        self.topk_passage_ids: List[str] = []
        self.table_rank_by_id: Dict[str, int] = {}
        self.passage_rank_by_id: Dict[str, int] = {}
        self.candidate_clusters: List[dict] = []
        self.failed_clusters: List[Dict[str, Any]] = []
        self._current_query_out_dir: Optional[str] = None
        self._passage_descriptions_cache: Optional[Dict[str, dict]] = None
        self._opencode_usages: List[Dict[str, Any]] = []

    def _query_output_dir(self, query_id: str) -> str:
        return ensure_dir(os.path.join(self.log_dir, str(query_id)))

    def run(
        self,
        query_record: Dict[str, str],
        *,
        query_index: int = 1,
        query_total: int = 1,
    ) -> Dict[str, Any]:
        query_id = query_record["query_id"]
        user_query = query_record["query_text"]
        out_dir = self._query_output_dir(query_id)
        self._current_query_out_dir = out_dir
        ui = get_ui()
        rich = ui.rich_cli

        ui.begin_query(query_index, query_id, user_query, self.args.budget)
        query_started = time.perf_counter()
        tracker = get_tracker()
        usage_at_query_start = tracker.snapshot()
        usage_at_step_start = usage_at_query_start

        self.answered_subquestions = []
        self.pending_subquestions = {}
        self.cluster_answered_subquestions = {}
        self.cluster_selection_state.reset()
        self.iterations = []
        self._passage_descriptions_cache = None
        self._opencode_usages = []

        def compute_relevance() -> None:
            self.table_rank_by_id = get_table_similarity_ranks(
                user_query,
                self.args.table_embeddings_path,
                self.args.embedding_model,
                embedding_provider=self.args.embedding_provider,
                embedder=self.embedder,
            )
            self.topk_table_ids = top_k_table_ids_from_ranks(
                self.table_rank_by_id,
                self.args.k_relevant_tables,
            )
            if self.args.use_passages:
                self.passage_rank_by_id = get_passage_similarity_ranks(
                    user_query,
                    self.args.passage_embeddings_path,
                    self.args.embedding_model,
                    embedding_provider=self.args.embedding_provider,
                    embedder=self.embedder,
                )
                self.topk_passage_ids = top_k_ids_from_ranks(
                    self.passage_rank_by_id,
                    self.args.k_relevant_passages,
                )
            else:
                self.passage_rank_by_id = {}
                self.topk_passage_ids = []

        if rich:
            ui.set_status("relevance")
            compute_relevance()
        else:
            with tqdm(
                total=1,
                desc=f"{query_id}: relevance",
                disable=self.args.silent,
                leave=False,
            ) as pbar:
                compute_relevance()
                pbar.update(1)

        if not self.args.use_clustering:
            from src.inference_clusters import build_topk_inference_cluster

            self.inference_clusters = build_topk_inference_cluster(
                self.topk_table_ids,
                self.topk_passage_ids,
                tables_lake_dir=self.args.tables_lake_dir,
                passage_descriptions_path=(
                    self.args.passage_descriptions_path
                    if self.args.use_passages
                    else None
                ),
            )

        self.failed_clusters = []
        self.candidate_clusters = filter_clusters_by_tables_passages(
            self.inference_clusters,
            self.topk_table_ids,
            topk_passage_ids=self.topk_passage_ids if self.args.use_passages else None,
        )
        total_inference_clusters = len(self.inference_clusters)
        retained_clusters = len(self.candidate_clusters)
        initial_candidate_clusters = retained_clusters
        ui.set_cluster_pool_summary(retained_clusters, total_inference_clusters)
        ground_truth = query_record.get("ground_truth")
        compute_metrics = bool(self.args.compute_metrics)

        metrics_tracker: Optional[MetricsTracker] = None
        if compute_metrics:
            uid_to_table_id = load_uid_to_table_id_mapping(self.args.uid_to_table_id_path)
            gt_tables = extract_gt_table_ids(
                ground_truth,
                uid_to_table_id=uid_to_table_id,
            )
            gt_passages = None
            if self.args.use_passages:
                uid_to_passage_id = load_uid_to_passage_id_mapping(
                    self.args.uid_to_passage_id_path
                )
                gt_passages = extract_gt_passage_ids(
                    ground_truth,
                    uid_to_passage_id=uid_to_passage_id,
                    passage_type=self.args.passage_type,
                )
            metrics_tracker = MetricsTracker(
                user_query=user_query,
                budget=self.args.budget,
                gt_tables=gt_tables,
                total_lake_tables=count_lake_tables(self.args.tables_lake_dir),
                topk_table_ids=self.topk_table_ids,
                inference_clusters=self.inference_clusters,
                initial_candidate_clusters=initial_candidate_clusters,
                k_relevant_tables=self.args.k_relevant_tables,
                embedder=self.embedder if self.args.compute_embed_diversity else None,
                compute_embed_diversity=bool(self.args.compute_embed_diversity),
                gt_passages=gt_passages,
                topk_passage_ids=self.topk_passage_ids,
                k_relevant_passages=self.args.k_relevant_passages,
                use_passages=self.args.use_passages,
                judge_llms=(
                    build_judge_clients(
                        self.args.llm_provider,
                        self.args.judge_models,
                    )
                    if not self.args.no_llm_judge
                    else []
                ),
                judge_temperature=1.0,
                research_quality_enabled=not self.args.no_llm_judge,
                reporter=make_metrics_reporter(
                    self.args.budget,
                    mode="auto",
                    silent=self.args.silent,
                    rich_cli=rich,
                ),
                passage_descriptions=self._passage_descriptions_for_metrics(),
            )

        if not self.candidate_clusters:
            log(
                "No clusters overlap relevant tables/passages; skipping this query.",
                silent=self.args.silent,
            )
            self.iterations = []
        elif self.args.use_llm_priors:
            self._elicit_cluster_priors(user_query, out_dir)

        budget_range = range(1, self.args.budget + 1)
        if rich:
            step_iter = budget_range
        else:
            step_iter = tqdm(
                budget_range,
                desc=f"{query_id}: budget",
                disable=self.args.silent,
                leave=False,
            )

        for step in step_iter:
            ui.set_budget_step(step)
            if not rich and not self.args.silent:
                step_iter.set_postfix_str(f"step {step}/{self.args.budget}")
            n_failed_before = len(self.failed_clusters)
            step_started = time.perf_counter()
            iteration = self._run_one_iteration(user_query, step)
            step_time_taken = round(time.perf_counter() - step_started, 3)
            clusters_excluded_step = len(self.failed_clusters) - n_failed_before
            if iteration:
                if metrics_tracker:
                    metrics_tracker.record_iteration(
                        iteration, step, clusters_excluded_step
                    )
                    cluster_id = (iteration.get("selected_cluster") or {}).get("id")
                    if cluster_id:
                        reward = bandit_reward_from_iteration(
                            iteration, self.args.bandit_reward
                        )
                        if self.args.use_llm_priors:
                            self.cluster_selection_state.update_posterior(
                                cluster_id,
                                reward,
                                weight=self.args.posterior_evidence_weight,
                            )
                        else:
                            self.cluster_selection_state.record_outcome(
                                cluster_id, reward
                            )
                usage_now = tracker.snapshot()
                pipeline_usage = usage_now.subtract(usage_at_step_start).to_dict()
                opencode_usage = iteration.pop("opencode_usage", None)
                if opencode_usage:
                    iteration["usage"] = combine_query_usage(
                        pipeline_usage, opencode_usage
                    )
                else:
                    iteration["usage"] = pipeline_usage
                iteration["time_taken"] = step_time_taken
                usage_at_step_start = usage_now
                self.iterations.append(iteration)
                path = os.path.join(out_dir, f"iteration_{step:03d}.json")
                save_json(path, iteration)
                if rich:
                    ui.set_status(f"saved {os.path.basename(path)}")
            elif metrics_tracker:
                step_metrics = metrics_tracker.record_empty_step(
                    step, clusters_excluded_step
                )
                usage_now = tracker.snapshot()
                skipped_payload = {
                    "step": step,
                    "skipped": True,
                    "metrics": step_metrics,
                    "usage": usage_now.subtract(usage_at_step_start).to_dict(),
                    "time_taken": step_time_taken,
                }
                usage_at_step_start = usage_now
                path = os.path.join(out_dir, f"iteration_{step:03d}.json")
                save_json(path, skipped_payload)
                if rich:
                    ui.set_status(f"saved {os.path.basename(path)} (skipped)")
            else:
                usage_at_step_start = tracker.snapshot()
            if rich:
                ui.complete_budget_step(step)

        if rich:
            ui.set_status("final report")
        report_iterations = [
            it for it in self.iterations if is_report_eligible_iteration(it)
        ]
        assign_finding_indices(self.iterations)
        sync_finding_indices_to_query_dir(out_dir, self.iterations)
        report = generate_final_report(
            self.llm,
            user_query,
            report_iterations,
            self.args.temperature,
        )

        query_usage = tracker.snapshot().subtract(usage_at_query_start)
        usage_dict = query_usage.to_dict()
        if self.args.opencode_exec and self._opencode_usages:
            usage_dict = combine_query_usage(
                usage_dict,
                merge_usage_summaries(self._opencode_usages).to_dict(),
            )
        query_time_taken = round(time.perf_counter() - query_started, 3)
        query_metrics = None
        if metrics_tracker:
            query_metrics = metrics_tracker.finalize(report, self.iterations)
        result = build_result_json(
            query_id=query_id,
            user_query=user_query,
            coverage=query_record.get("coverage"),
            method="pipeline",
            answer=report,
            iterations=self.iterations,
            ground_truth=ground_truth,
            topk_table_ids=self.topk_table_ids,
            topk_passage_ids=self.topk_passage_ids,
            total_inference_clusters=total_inference_clusters,
            retained_clusters=retained_clusters,
            time_taken=query_time_taken,
            usage=usage_dict,
            metrics=query_metrics,
            opencode_exec=bool(self.args.opencode_exec),
            query_dir=out_dir,
        )
        result_path = save_query_result(
            out_dir,
            result,
            ground_truth=ground_truth,
            topk_table_ids=self.topk_table_ids,
            topk_passage_ids=self.topk_passage_ids,
            metrics=query_metrics,
        )
        usage_summary = (
            format_nested_usage_line(usage_dict)
            if usage_dict.get("opencode")
            else format_usage_line(query_usage)
        )
        ui.set_final_result(
            report,
            query_id=query_id,
            n_iterations=len(result.get("findings") or []),
            result_path=result_path,
            usage_summary=usage_summary,
        )
        return result

    def _elicit_cluster_priors(self, user_query: str, out_dir: str) -> None:
        n_clusters = len(self.candidate_clusters)
        max_workers = self.args.llm_prior_max_workers
        parallel = max_workers > 1 and n_clusters > 1
        log(
            f"Eliciting LLM priors for {n_clusters} clusters "
            f"({self.args.llm_prior_n_samples} samples each"
            f"{f', max_workers={max_workers}' if parallel else ''}).",
            silent=self.args.silent,
        )
        priors: Dict[str, Dict[str, float]] = {}

        def _elicit_one(
            cluster: dict,
        ) -> tuple[str, float, float, Dict[str, int], str, str]:
            cluster_id = cluster.get("cluster_id") or "unknown"
            alpha, beta, counts, prompt, system_prompt = elicit_cluster_prior(
                self.llm,
                user_query=user_query,
                cluster=cluster,
                bandit_reward=self.args.bandit_reward,
                n_samples=self.args.llm_prior_n_samples,
                temperature=self.args.temperature,
                silent=self.args.silent,
            )
            return cluster_id, alpha, beta, counts, prompt, system_prompt

        def _record_prior(
            cluster_id: str,
            alpha: float,
            beta: float,
            counts: Dict[str, int],
            prompt: str,
            system_prompt: str,
        ) -> None:
            self.cluster_selection_state.set_prior(cluster_id, alpha, beta)
            mean = alpha / (alpha + beta)
            priors[cluster_id] = {
                "alpha": alpha,
                "beta": beta,
                "mean": mean,
                "counts": counts,
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
            log(
                f"  {cluster_id}: Beta({alpha:.2f}, {beta:.2f}), mean={mean:.3f}",
                silent=self.args.silent,
            )

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_elicit_one, cluster)
                    for cluster in self.candidate_clusters
                ]
                for future in as_completed(futures):
                    cluster_id, alpha, beta, counts, prompt, system_prompt = future.result()
                    _record_prior(
                        cluster_id, alpha, beta, counts, prompt, system_prompt
                    )
        else:
            for cluster in self.candidate_clusters:
                cluster_id, alpha, beta, counts, prompt, system_prompt = _elicit_one(
                    cluster
                )
                _record_prior(
                    cluster_id, alpha, beta, counts, prompt, system_prompt
                )

        save_json(
            os.path.join(out_dir, "cluster_priors.json"),
            dict(sorted(priors.items(), key=lambda item: item[1]["mean"], reverse=True)),
        )

    def _exclude_cluster(self, cluster: dict, step: int, reason: str) -> None:
        """Remove a cluster from future selection and record the failure."""
        cluster_id = cluster.get("cluster_id") or "unknown"
        entry = {
            "cluster_id": cluster_id,
            "description": cluster.get("description", ""),
            "step": step,
            "reason": reason,
        }
        self.failed_clusters.append(entry)
        self._log_failed_cluster(entry)
        self.candidate_clusters = [
            c
            for c in self.candidate_clusters
            if (c.get("cluster_id") or "unknown") != cluster_id
        ]
        self.pending_subquestions.pop(cluster_id, None)
        self.cluster_answered_subquestions.pop(cluster_id, None)

    def _log_failed_cluster(self, entry: Dict[str, Any]) -> None:
        log(
            f"Cluster {entry['cluster_id']} excluded ({entry['reason']})",
            silent=self.args.silent,
        )
        if not self._current_query_out_dir:
            return
        path = os.path.join(self._current_query_out_dir, "failed_clusters.json")
        save_json(path, self.failed_clusters)

    def _run_one_iteration(self, user_query: str, step: int) -> Dict[str, Any]:
        if not self.candidate_clusters:
            return {}

        cluster: Optional[dict] = None
        pool: List[str] = []
        generated_subquestions: List[str] = []
        excluded_this_iteration: List[Dict[str, Any]] = []

        while self.candidate_clusters:
            cluster = select_cluster(
                self.candidate_clusters,
                self.args.cluster_selection_method,
                self.rng,
                self.cluster_selection_state,
                llm=self.llm,
                user_query=user_query,
                temperature=self.args.temperature,
                epsilon=self.args.cluster_selection_epsilon,
                ucb_c=self.args.cluster_selection_ucb_c,
                bandit_reward=self.args.bandit_reward,
                use_llm_priors=self.args.use_llm_priors,
                llm_prior_tau=self.args.llm_prior_tau,
                silent=self.args.silent,
            )
            cluster_id = cluster.get("cluster_id") or "unknown"
            cluster_description = cluster.get("description", "")
            table_ids = [t["table_id"] for t in cluster.get("tables", [])]
            passage_ids = [p["passage_id"] for p in cluster.get("passages", [])]
            rank_summary = cluster_similarity_rank_summary(
                table_ids,
                self.table_rank_by_id,
                passage_ids,
                self.passage_rank_by_id,
                include_passages=self.args.use_passages,
            )
            meta = (
                f"ranks best={rank_summary['best_rank']} "
                f"avg={rank_summary['avg_rank']} tables={len(table_ids)} "
                f"passages={len(passage_ids)}"
            )
            if self.args.use_passages and passage_ids:
                meta += (
                    f" passage_ranks best={rank_summary.get('passage_best_rank')} "
                    f"avg={rank_summary.get('passage_avg_rank')}"
                )
            ui = get_ui()
            if ui.rich_cli:
                ui.set_cluster(cluster_id, cluster_description, meta)
            rank_log = (
                f"  similarity ranks — best={rank_summary['best_rank']}, "
                f"avg={rank_summary['avg_rank']}, tables={len(table_ids)}, "
                f"passages={len(passage_ids)}"
            )
            if self.args.use_passages and passage_ids:
                rank_log += (
                    f", passage_best={rank_summary.get('passage_best_rank')}, "
                    f"passage_avg={rank_summary.get('passage_avg_rank')}"
                )
            log(
                f"Selected cluster {cluster_id}:\n"
                f"  {cluster_description}\n"
                f"{rank_log}",
                silent=self.args.silent,
            )

            pool = self.pending_subquestions.get(cluster_id, [])
            if pool:
                break

            _n_retries = 3
            generation_failed = False
            for _retry in range(_n_retries):
                gen = generate_subquestions(
                    self.llm,
                    user_query,
                    cluster,
                    self.args.k_subquestions,
                    self.args.temperature,
                    previous_subquestions=self.cluster_answered_subquestions.get(
                        cluster_id, []
                    ),
                    allow_empty=self.args.use_clustering,
                    silent=self.args.silent,
                )
                pool = gen.subquestions

                if not pool:
                    if gen.parse_failed:
                        log(
                            f"Attempt {_retry + 1}/{_n_retries}: sub-question "
                            "JSON parse failed; retrying.",
                            silent=self.args.silent,
                        )
                        continue
                    if self.args.use_clustering:
                        self._exclude_cluster(
                            cluster, step, "empty_subquestion_generation"
                        )
                        excluded_this_iteration.append(self.failed_clusters[-1])
                        cluster = None
                        generation_failed = True
                        break
                    log(
                        f"Attempt {_retry + 1}/{_n_retries}: empty sub-question "
                        "generation; retrying.",
                        silent=self.args.silent,
                    )
                    continue

                filtered_pool = filter_semantically_duplicate_subquestions(
                    self.llm,
                    self.answered_subquestions,
                    pool,
                    self.args.temperature,
                )
                n_filtered = len(pool) - len(filtered_pool)
                log(
                    f"Removed {n_filtered} semantically duplicate sub-questions.",
                    silent=self.args.silent,
                )
                if filtered_pool:
                    generated_subquestions = list(filtered_pool)
                    self._save_cluster_subquestions(
                        cluster_id,
                        cluster_description,
                        step,
                        generated_subquestions,
                    )
                    self.pending_subquestions[cluster_id] = filtered_pool
                    pool = filtered_pool
                    break

                pool = []
                log(
                    f"Attempt {_retry + 1}/{_n_retries}: all generated sub-questions "
                    "were semantically duplicate; retrying generation.",
                    silent=self.args.silent,
                )

            if generation_failed:
                continue
            if pool:
                break

            log(
                "Failed to generate new sub-questions after all retries.",
                silent=self.args.silent,
            )
            return {}

        if not cluster or not pool:
            log(
                "No candidate clusters remain after excluding clusters with "
                "no relevant sub-questions.",
                silent=self.args.silent,
            )
            return {}

        n_visits = self.cluster_selection_state.record_visit(cluster_id)

        # Select a sub-question from the pool
        sub_question = select_subquestion(
            pool,
            self.args.subquestion_selection_method,
            self.rng,
            llm=self.llm,
            user_query=user_query,
            cluster=cluster,
            previous_subquestions=self.answered_subquestions,
            temperature=self.args.temperature,
            silent=self.args.silent,
        )
        pool.remove(sub_question)
        self.pending_subquestions[cluster_id] = pool  # update the pool
        self.answered_subquestions.append(sub_question)  # add to the answered list
        self.cluster_answered_subquestions.setdefault(cluster_id, []).append(
            sub_question
        )
        ui = get_ui()
        ui.set_sub_question(sub_question)
        passages = list(cluster.get("passages", []))

        if self.args.opencode_exec:
            exec_out = self._execute_subquestion_opencode(
                user_query=user_query,
                sub_question=sub_question,
                cluster=cluster,
                cluster_id=cluster_id,
                table_ids=table_ids,
                step=step,
                rank_summary=rank_summary,
            )
        else:
            exec_out = self._execute_subquestion_pipeline(
                user_query=user_query,
                sub_question=sub_question,
                cluster=cluster,
                cluster_id=cluster_id,
                table_ids=table_ids,
                step=step,
                rank_summary=rank_summary,
                passages=passages,
            )

        needs_sql = bool(exec_out.get("needs_sql"))
        sql = exec_out.get("sql") or ""
        exec_result = exec_out.get("exec_result") or {
            "ok": True,
            "row_count": 0,
            "rows": [],
            "error": None,
        }
        sql_attempts = list(exec_out.get("sql_attempts") or [])
        answer = exec_out.get("answer") or ""
        opencode_meta = exec_out.get("opencode_meta")
        opencode_usage = exec_out.get("opencode_usage")
        cluster_expansion = exec_out.get("cluster_expansion")
        rank_summary = exec_out.get("rank_summary", rank_summary)

        failed_sql_attempts = [
            attempt
            for attempt in sql_attempts
            if not attempt.get("ok")
            or int(attempt.get("row_count", 0) or 0) == 0
        ]

        iteration: Dict[str, Any] = {
            "step": step,
            "selected_cluster": {
                "id": cluster.get("cluster_id"),
                "description": cluster.get("description"),
                "n_visits": n_visits,
                "table_ids": [
                    (t["table_id"], t.get("title", ""))
                    for t in cluster.get("tables", [])
                ],
                "passage_ids": [
                    (p["passage_id"], p.get("title", ""))
                    for p in cluster.get("passages", [])
                ],
                "similarity_ranks": rank_summary,
            },
            "sub_question": sub_question,
            "needs_sql": needs_sql,
            "answer": answer,
            "sql": (
                sql
                if needs_sql and is_successful_iteration(exec_result)
                else None
            ),
            "execution": exec_result if needs_sql else None,
            "failed_sql_attempts": failed_sql_attempts if needs_sql else None,
            "untried_subquestions": pool,
        }
        if cluster_expansion is not None:
            iteration["cluster_expansion"] = cluster_expansion
        if opencode_meta is not None and self._current_query_out_dir:
            workdir = os.path.join(
                self._current_query_out_dir, f"opencode_step_{step:03d}"
            )
            slim = slim_opencode_meta(opencode_meta, workdir)
            if slim:
                iteration["opencode"] = slim
        opencode_failure = exec_out.get("opencode_failure")
        if opencode_failure:
            iteration["opencode_failure"] = opencode_failure
        if excluded_this_iteration:
            iteration["excluded_clusters"] = excluded_this_iteration
        if opencode_usage:
            iteration["opencode_usage"] = opencode_usage
        return iteration

    def _opencode_failure_result(
        self,
        *,
        opencode_meta: Optional[Dict[str, Any]],
        opencode_usage: Optional[Dict[str, Any]],
        reason: str,
        rank_summary: dict,
        sql_attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> SubquestionExecResult:
        empty_exec: Dict[str, Any] = {
            "ok": True,
            "row_count": 0,
            "rows": [],
            "error": None,
        }
        return {
            "needs_sql": False,
            "sql": "",
            "exec_result": empty_exec,
            "sql_attempts": list(sql_attempts or []),
            "answer": "",
            "opencode_meta": opencode_meta,
            "opencode_usage": opencode_usage,
            "opencode_failure": {"reason": reason},
            "cluster_expansion": None,
            "rank_summary": rank_summary,
        }

    def _execute_subquestion_opencode(
        self,
        *,
        user_query: str,
        sub_question: str,
        cluster: dict,
        cluster_id: str,
        table_ids: List[str],
        step: int,
        rank_summary: dict,
    ) -> SubquestionExecResult:
        empty_exec: Dict[str, Any] = {
            "ok": True,
            "row_count": 0,
            "rows": [],
            "error": None,
        }
        if not self._current_query_out_dir:
            return self._opencode_failure_result(
                opencode_meta=None,
                opencode_usage=None,
                reason="missing_query_dir",
                rank_summary=rank_summary,
            )

        ui = get_ui()
        opencode_out = run_opencode_subquestion(
            self.args,
            user_query=user_query,
            sub_question=sub_question,
            cluster=cluster,
            table_ids=table_ids,
            query_dir=self._current_query_out_dir,
            step=step,
            passage_rank_by_id=self.passage_rank_by_id,
        )
        if not opencode_out:
            return self._opencode_failure_result(
                opencode_meta=None,
                opencode_usage=None,
                reason="no_commit",
                rank_summary=rank_summary,
            )

        opencode_meta = opencode_out.get("opencode_meta")
        opencode_usage = (opencode_meta or {}).get("usage")
        if opencode_usage:
            self._opencode_usages.append(opencode_usage)

        if opencode_out.get("commit_failed") or not opencode_out.get("answer"):
            return self._opencode_failure_result(
                opencode_meta=opencode_meta,
                opencode_usage=opencode_usage,
                reason=str(opencode_out.get("failure_reason") or "no_commit"),
                rank_summary=rank_summary,
                sql_attempts=list(opencode_out.get("failed_sql_attempts") or []),
            )

        cluster_expansion = None
        if self.args.expand_cluster and self.args.use_passages:
            if ui.rich_cli:
                ui.set_status("cluster passage expansion")
            cluster_expansion = self._sync_opencode_cluster_expansion(
                cluster=cluster,
                cluster_id=cluster_id,
                sub_question=sub_question,
                user_query=user_query,
                step=step,
                lake_state=opencode_out.get("lake_state") or {},
            )
            if cluster_expansion and cluster_expansion.get("added_passage_ids"):
                rank_summary = self._refresh_cluster_rank_summary(
                    cluster, table_ids
                )

        answer = opencode_out["answer"]
        ui.show_sql_answer(answer)
        return {
            "needs_sql": bool(opencode_out.get("needs_sql")),
            "sql": opencode_out.get("sql") or "",
            "exec_result": opencode_out.get("execution") or empty_exec,
            "sql_attempts": list(opencode_out.get("failed_sql_attempts") or []),
            "answer": answer,
            "opencode_meta": opencode_meta,
            "opencode_usage": opencode_usage,
            "cluster_expansion": cluster_expansion,
            "rank_summary": rank_summary,
        }

    def _execute_subquestion_pipeline(
        self,
        *,
        user_query: str,
        sub_question: str,
        cluster: dict,
        cluster_id: str,
        table_ids: List[str],
        step: int,
        rank_summary: dict,
        passages: List[dict],
    ) -> SubquestionExecResult:
        ui = get_ui()
        needs_sql = False
        sql = ""
        exec_result: Dict[str, Any] = {
            "ok": True,
            "row_count": 0,
            "rows": [],
            "error": None,
        }
        sql_attempts: List[Dict[str, Any]] = []
        cluster_expansion = None

        if table_ids:
            table_metas = load_cluster_table_metas(self.args.tables_lake_dir, table_ids)
            conn, schema_string = build_cluster_sqlite(table_metas)
            try:
                if self.args.use_passages:
                    needs_sql = decide_sql_needed(
                        self.llm,
                        sub_question,
                        passages,
                        schema_string,
                        self.args.temperature,
                    )
                else:
                    needs_sql = True
                if needs_sql:
                    if ui.rich_cli:
                        ui.set_status("SQL loop")
                    samples = fetch_sample_rows(conn, table_ids[:10])
                    loop_out = _run_subquestion_sql_loop_langgraph(
                        llm=self.llm,
                        conn=conn,
                        sub_question=sub_question,
                        schema_string=schema_string,
                        samples_string=samples,
                        temperature=self.args.temperature,
                        max_attempts=self.args.max_sql_attempts,
                    )
                    sql = loop_out["sql"]
                    exec_result = loop_out["execution"]
                    sql_attempts = loop_out["attempts_log"]
                else:
                    log(
                        "Answering from passages without SQL.",
                        silent=self.args.silent,
                    )
            finally:
                conn.close()
        elif passages:
            needs_sql = False
            log(
                "No tables in cluster; answering from passages only.",
                silent=self.args.silent,
            )

        sql_expansion = needs_sql and is_successful_iteration(exec_result)
        passage_expansion = not needs_sql and bool(passages)
        if (
            self.args.expand_cluster
            and self.args.use_passages
            and (sql_expansion or passage_expansion)
        ):
            if ui.rich_cli:
                ui.set_status("cluster passage expansion")
            cluster_expansion = self._maybe_expand_cluster(
                cluster=cluster,
                cluster_id=cluster_id,
                sub_question=sub_question,
                user_query=user_query,
                step=step,
                expansion_mode="sql" if sql_expansion else "passages",
                sql=sql if sql_expansion else None,
                exec_result=exec_result if sql_expansion else None,
            )
            passages = list(cluster.get("passages", []))
            if cluster_expansion and cluster_expansion.get("added_passage_ids"):
                rank_summary = self._refresh_cluster_rank_summary(
                    cluster, table_ids
                )

        if ui.rich_cli:
            ui.set_status("generating answer")
        else:
            log("Generating answer…", silent=self.args.silent)
        if self.args.use_passages or passages:
            answer = answer_from_context(
                self.llm,
                sub_question,
                passages,
                sql=sql if needs_sql else None,
                execution_result=exec_result if needs_sql else None,
                temperature=self.args.temperature,
            )
        else:
            answer = answer_from_sql_results(
                self.llm,
                sub_question,
                sql,
                exec_result,
                self.args.temperature,
            )
        ui.show_sql_answer(answer)
        return {
            "needs_sql": needs_sql,
            "sql": sql,
            "exec_result": exec_result,
            "sql_attempts": sql_attempts,
            "answer": answer,
            "cluster_expansion": cluster_expansion,
            "rank_summary": rank_summary,
        }

    def _passage_descriptions_for_metrics(self) -> Optional[Dict[str, dict]]:
        if not self.args.use_passages:
            return None
        if self._passage_descriptions_cache is None:
            from src.utils import load_passage_descriptions_for_metrics

            self._passage_descriptions_cache = (
                load_passage_descriptions_for_metrics(self.args) or {}
            )
        return self._passage_descriptions_cache or None

    def _load_passage_descriptions(self) -> Dict[str, dict]:
        if self._passage_descriptions_cache is None:
            self._passage_descriptions_for_metrics()
        return self._passage_descriptions_cache or {}

    def _refresh_cluster_rank_summary(
        self,
        cluster: dict,
        table_ids: List[str],
    ) -> dict:
        passage_ids = [p["passage_id"] for p in cluster.get("passages", [])]
        return cluster_similarity_rank_summary(
            table_ids,
            self.table_rank_by_id,
            passage_ids,
            self.passage_rank_by_id,
            include_passages=self.args.use_passages,
        )

    def _queue_expansion_subquestions(
        self,
        *,
        user_query: str,
        cluster_id: str,
        cluster: dict,
        new_passages: List[dict],
        meta: Dict[str, Any],
    ) -> None:
        """Generate and queue follow-up sub-questions from newly added passages."""
        expansion_cluster = {
            "cluster_id": cluster_id,
            "description": cluster.get("description", ""),
            "tables": cluster.get("tables", []),
            "passages": new_passages,
        }
        previous = list(
            self.cluster_answered_subquestions.get(cluster_id, [])
        ) + list(self.pending_subquestions.get(cluster_id, []))
        gen = generate_subquestions(
            self.llm,
            user_query,
            expansion_cluster,
            EXPANSION_SUBQUESTIONS_K,
            self.args.temperature,
            previous_subquestions=previous,
            allow_empty=True,
            silent=self.args.silent,
        )
        raw_subquestions = gen.subquestions
        if not raw_subquestions:
            return
        filtered = filter_semantically_duplicate_subquestions(
            self.llm,
            self.answered_subquestions,
            raw_subquestions,
            self.args.temperature,
        )
        if not filtered:
            return
        self.pending_subquestions.setdefault(cluster_id, []).extend(filtered)
        meta["generated_subquestions"] = filtered
        log(
            f"Cluster passage expansion: queued {len(filtered)} "
            f"new sub-question(s) for cluster {cluster_id}.",
            silent=self.args.silent,
        )

    def _cluster_passage_ids(self, cluster: dict) -> set[str]:
        return {
            p.get("passage_id")
            for p in cluster.get("passages", [])
            if p.get("passage_id")
        }

    def _apply_passage_expansion(
        self,
        *,
        cluster: dict,
        passage_ids: List[str],
        passage_descriptions: Dict[str, dict],
    ) -> tuple[List[dict], Dict[str, Any]]:
        """Add new passages to the cluster; return added records and rank metadata."""
        existing_ids = self._cluster_passage_ids(cluster)
        new_ids = [pid for pid in passage_ids if pid not in existing_ids]
        partial: Dict[str, Any] = {"added_passage_ids": [], "added_passage_ranks": {}}
        if not new_ids:
            return [], partial

        new_passages = passages_from_ids(passage_descriptions, new_ids)
        if not new_passages:
            return [], partial

        cluster.setdefault("passages", []).extend(new_passages)
        partial["added_passage_ids"] = [
            (p["passage_id"], p.get("title", "")) for p in new_passages
        ]
        partial["added_passage_ranks"] = {
            p["passage_id"]: self.passage_rank_by_id.get(p["passage_id"])
            for p in new_passages
        }
        return new_passages, partial

    def _sync_opencode_cluster_expansion(
        self,
        *,
        cluster: dict,
        cluster_id: str,
        sub_question: str,
        user_query: str,
        step: int,
        lake_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Merge passages retrieved by OpenCode grep/passage loads into the cluster."""
        existing_ids = self._cluster_passage_ids(cluster)
        retrieved_ids = list(lake_state.get("retrieved_passage_ids") or [])
        new_ids = [pid for pid in retrieved_ids if pid not in existing_ids]
        if not new_ids:
            return None

        meta: Dict[str, Any] = {
            "trigger": opencode_expansion_trigger(lake_state),
            "grep_queries": lake_state.get("grep_queries") or [],
            "added_passage_ids": [],
            "added_passage_ranks": {},
            "generated_subquestions": [],
        }

        passage_descriptions = self._load_passage_descriptions()
        new_passages, partial = self._apply_passage_expansion(
            cluster=cluster,
            passage_ids=new_ids,
            passage_descriptions=passage_descriptions,
        )
        if not new_passages:
            self._save_cluster_expanded_passages(cluster_id, step, sub_question, meta)
            return meta

        meta.update(partial)
        log(
            f"OpenCode cluster expansion: added {len(new_passages)} passage(s) "
            f"to cluster {cluster_id}.",
            silent=self.args.silent,
        )
        self._queue_expansion_subquestions(
            user_query=user_query,
            cluster_id=cluster_id,
            cluster=cluster,
            new_passages=new_passages,
            meta=meta,
        )
        self._save_cluster_expanded_passages(cluster_id, step, sub_question, meta)
        return meta

    def _maybe_expand_cluster(
        self,
        *,
        cluster: dict,
        cluster_id: str,
        sub_question: str,
        user_query: str,
        step: int,
        expansion_mode: ExpansionMode,
        sql: Optional[str] = None,
        exec_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Grep passage descriptions and optionally grow the cluster."""
        decision = decide_passage_expansion(
            self.llm,
            cluster,
            sub_question,
            self.args.temperature,
            mode=expansion_mode,
            sql=sql,
            execution_result=exec_result,
        )
        meta: Dict[str, Any] = {
            "trigger": expansion_mode,
            "llm_decision": decision,
            "added_passage_ids": [],
            "added_passage_ranks": {},
            "generated_subquestions": [],
        }

        if not decision.get("expand") or not decision.get("grep_keywords"):
            log(
                "Cluster passage expansion skipped (LLM chose not to expand).",
                silent=self.args.silent,
            )
            self._save_cluster_expanded_passages(cluster_id, step, sub_question, meta)
            return meta

        existing_ids = self._cluster_passage_ids(cluster)
        passage_descriptions = self._load_passage_descriptions()
        matched_ids = grep_passage_ids(
            self.args.passage_descriptions_path,
            decision["grep_keywords"],
            exclude_ids=existing_ids,
            rank_by=self.passage_rank_by_id,
        )
        new_passages, partial = self._apply_passage_expansion(
            cluster=cluster,
            passage_ids=matched_ids,
            passage_descriptions=passage_descriptions,
        )
        if not new_passages:
            log(
                "Cluster passage expansion: grep returned no new passages.",
                silent=self.args.silent,
            )
            self._save_cluster_expanded_passages(cluster_id, step, sub_question, meta)
            return meta

        meta.update(partial)
        log(
            f"Cluster passage expansion: added {len(new_passages)} passage(s) "
            f"to cluster {cluster_id}.",
            silent=self.args.silent,
        )

        if decision.get("generate_new_subquestions"):
            self._queue_expansion_subquestions(
                user_query=user_query,
                cluster_id=cluster_id,
                cluster=cluster,
                new_passages=new_passages,
                meta=meta,
            )

        self._save_cluster_expanded_passages(cluster_id, step, sub_question, meta)
        return meta

    def _save_cluster_expanded_passages(
        self,
        cluster_id: str,
        step: int,
        sub_question: str,
        expansion_meta: Dict[str, Any],
    ) -> None:
        if not self._current_query_out_dir:
            return

        path = os.path.join(
            self._current_query_out_dir, "cluster_expanded_passages.json"
        )
        existing: Dict[str, Any] = {}
        if os.path.isfile(path):
            existing = load_json(path) if os.path.getsize(path) else {}
        if not isinstance(existing, dict):
            existing = {}

        entry = existing.setdefault(
            cluster_id,
            {"steps": [], "all_added_passage_ids": []},
        )
        step_record = {
            "step": step,
            "sub_question": sub_question,
            "trigger": expansion_meta.get("trigger"),
            "grep_keywords": expansion_grep_keywords(expansion_meta),
            "grep_queries": expansion_meta.get("grep_queries") or [],
            "added_passage_ids": expansion_meta.get("added_passage_ids") or [],
            "added_passage_ranks": expansion_meta.get("added_passage_ranks") or {},
            "generated_subquestions": expansion_meta.get("generated_subquestions")
            or [],
        }
        entry["steps"].append(step_record)

        seen = {tuple(item) for item in entry.get("all_added_passage_ids") or []}
        for item in step_record["added_passage_ids"]:
            key = tuple(item)
            if key not in seen:
                entry.setdefault("all_added_passage_ids", []).append(item)
                seen.add(key)

        save_json(path, existing)

    def _save_cluster_subquestions(
        self,
        cluster_id: str,
        cluster_description: str,
        step: int,
        subquestions: List[str],
    ) -> None:
        """Print and persist the sub-question set generated for a cluster."""
        log(
            f"\nSub-questions for cluster {cluster_id} (n={len(subquestions)}):",
            silent=self.args.silent,
        )
        for i, sq in enumerate(subquestions, start=1):
            log(f"  {i}. {sq}", silent=self.args.silent)

        if not self._current_query_out_dir:
            return

        path = os.path.join(self._current_query_out_dir, "cluster_subquestions.json")
        existing: Dict[str, Any] = {}
        if os.path.isfile(path):
            existing = load_json(path) if os.path.getsize(path) else {}
        if not isinstance(existing, dict):
            existing = {}

        existing[cluster_id] = {
            "step": step,
            "description": cluster_description,
            "subquestions": subquestions,
        }
        save_json(path, existing)
