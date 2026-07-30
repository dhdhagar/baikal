"""Orchestrates retrieval, operational, and research-quality metrics during a run."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.embedding_client import EmbeddingClient
from src.llm import LLMClient
from src.metrics.common import (
    extract_passages_from_answer,
    extract_tables_from_answer,
    is_finding_iteration,
    is_successful_iteration,
    normalize_passage_ids,
    normalize_table_ids,
    round4,
)
from src.metrics.operational import (
    DiversityTracker,
    build_operational_cumulative_metrics,
    build_operational_step_metrics,
)
from src.metrics.research_quality import (
    build_research_quality_step,
    build_research_quality_summary,
    judge_finding_rubric,
)
from src.metrics.retrieval import (
    build_retrieval_cumulative_metrics,
    build_retrieval_step_metrics,
    build_retrieval_summary,
    compute_lake_coverage,
)
from src.metrics.progress import MetricsReporter, NullMetricsReporter
from src.metrics.usage import capture_metrics_usage_start, summarize_metrics_usage
from src.sql_db import extract_tables_from_sql


class MetricsTracker:
    """Accumulates per-step and query-level metrics during a query run."""

    def __init__(
        self,
        *,
        user_query: str,
        budget: int,
        gt_tables: Optional[List[str]],
        total_lake_tables: int,
        topk_table_ids: List[str],
        inference_clusters: List[dict],
        initial_candidate_clusters: int,
        k_relevant_tables: int,
        embedder: Optional[EmbeddingClient] = None,
        compute_embed_diversity: bool = False,
        gt_passages: Optional[List[str]] = None,
        topk_passage_ids: Optional[List[str]] = None,
        k_relevant_passages: int = 0,
        use_passages: bool = False,
        judge_llms: Optional[List[LLMClient]] = None,
        judge_temperature: float = 1.0,
        research_quality_enabled: bool = True,
        reporter: Optional[MetricsReporter] = None,
        passage_descriptions: Optional[Dict[str, Any]] = None,
    ) -> None:
        from src.metrics.common import normalize_table_ids

        self.user_query = user_query
        self.budget = budget
        self.gt_tables = normalize_table_ids(gt_tables)
        self.gt_passages = normalize_passage_ids(gt_passages)
        self.total_lake_tables = total_lake_tables
        self.topk_table_ids = list(topk_table_ids)
        self.topk_passage_ids = list(topk_passage_ids or [])
        self.inference_clusters = inference_clusters
        self.initial_candidate_clusters = initial_candidate_clusters
        self.k_relevant_tables = k_relevant_tables
        self.k_relevant_passages = k_relevant_passages
        self.use_passages = use_passages
        self.compute_embed_diversity = compute_embed_diversity
        if compute_embed_diversity and embedder is None:
            raise ValueError("embedder is required when compute_embed_diversity is enabled")
        self.judge_llms = list(judge_llms or [])
        self.judge_temperature = judge_temperature
        self.research_quality_enabled = research_quality_enabled and bool(self.judge_llms)
        self.reporter = reporter or NullMetricsReporter()
        self.passage_descriptions = passage_descriptions

        self.tables_used_cumulative: Set[str] = set()
        self.passages_cited_cumulative: Set[str] = set()
        self.findings: List[Dict[str, str]] = []
        self.finding_scores: List[float] = []
        self.steps_completed = 0
        self.steps_with_iteration = 0
        self.sql_attempts = 0
        self.sql_successes = 0
        self.clusters_excluded_cumulative = 0
        self.per_step_history: List[Dict[str, Any]] = []
        self.diversity_tracker = (
            DiversityTracker(embedder) if compute_embed_diversity else None
        )
        self._usage_start = capture_metrics_usage_start()

        self.retrieval = build_retrieval_summary(
            gt_tables=self.gt_tables,
            gt_passages=self.gt_passages,
            topk_table_ids=self.topk_table_ids,
            topk_passage_ids=self.topk_passage_ids,
            inference_clusters=self.inference_clusters,
            k_relevant_tables=k_relevant_tables,
            k_relevant_passages=k_relevant_passages,
            use_passages=use_passages,
        )

    def _diversity_snapshot(self) -> Dict[str, Optional[float]]:
        if self.diversity_tracker is None:
            return {"mean": None, "max": None}
        return self.diversity_tracker.snapshot()

    def _record_step(
        self,
        *,
        step: int,
        iteration: Optional[Dict[str, Any]],
        clusters_excluded_step: int,
        sql_success_step: bool,
        tables_step: Set[str],
        passages_step: Set[str],
        is_finding: bool,
        rubric: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        lake_cov = compute_lake_coverage(
            self.tables_used_cumulative,
            self.total_lake_tables,
        )
        retrieval_step = build_retrieval_step_metrics(
            tables_step=tables_step,
            passages_step=passages_step,
            tables_used_cumulative=self.tables_used_cumulative,
            passages_cited_cumulative=self.passages_cited_cumulative,
            gt_tables=self.gt_tables,
            gt_passages=self.gt_passages,
            use_passages=self.use_passages,
        )
        operational_step = build_operational_step_metrics(
            step=step,
            sql_success_step=sql_success_step,
            sql_attempts=self.sql_attempts,
            sql_successes=self.sql_successes,
            clusters_excluded_step=clusters_excluded_step,
            clusters_excluded_cumulative=self.clusters_excluded_cumulative,
            initial_candidate_clusters=self.initial_candidate_clusters,
            lake_coverage=lake_cov,
            diversity=self._diversity_snapshot(),
            n_findings=len(self.findings),
        )
        research_quality_step = build_research_quality_step(
            is_finding=is_finding,
            rubric=rubric,
        )
        finding_score = float(research_quality_step["finding_score"] or 0.0)
        self.finding_scores.append(finding_score)

        step_metrics = {
            "step": step,
            "retrieval": retrieval_step,
            "operational": operational_step,
            "research_quality": research_quality_step,
        }
        if iteration is not None:
            iteration["metrics"] = step_metrics
        self.per_step_history.append(step_metrics)
        self.reporter.step_done(step)
        return step_metrics

    def record_empty_step(self, step: int, clusters_excluded_step: int) -> Dict[str, Any]:
        """Record metrics when a budget step produces no iteration."""
        self.reporter.step_start(step, self.budget, "skipped step")
        self.steps_completed = step
        self.clusters_excluded_cumulative += clusters_excluded_step
        return self._record_step(
            step=step,
            iteration=None,
            clusters_excluded_step=clusters_excluded_step,
            sql_success_step=False,
            tables_step=set(),
            passages_step=set(),
            is_finding=False,
            rubric=None,
        )

    def record_iteration(
        self,
        iteration: Dict[str, Any],
        step: int,
        clusters_excluded_step: int,
    ) -> Dict[str, Any]:
        """Annotate an iteration with metrics; update cumulative state."""
        self.reporter.step_start(step, self.budget, "recording")
        self.steps_completed = step
        self.steps_with_iteration += 1
        self.clusters_excluded_cumulative += clusters_excluded_step

        sql = iteration.get("sql") or ""
        answer = iteration.get("answer") or ""
        tables_step = normalize_table_ids(
            list(extract_tables_from_sql(sql) | extract_tables_from_answer(answer))
        )
        iteration["tables_used"] = sorted(tables_step)

        passages_step = normalize_passage_ids(list(extract_passages_from_answer(answer)))
        iteration["passages_cited"] = sorted(passages_step)

        sql_success_step = False
        if iteration.get("needs_sql"):
            self.sql_attempts += 1
            sql_success_step = is_successful_iteration(iteration.get("execution"))
            if sql_success_step:
                self.sql_successes += 1

        is_finding = is_finding_iteration(iteration)

        self.tables_used_cumulative |= tables_step
        self.passages_cited_cumulative |= passages_step

        rubric: Optional[Dict[str, Any]] = None
        if is_finding:
            finding = {
                "sub_question": iteration.get("sub_question") or "",
                "answer": answer,
            }
            if self.research_quality_enabled:
                self.reporter.step_detail("LLM rubric")
                rubric = judge_finding_rubric(
                    self.judge_llms,
                    user_query=self.user_query,
                    iteration=iteration,
                    prior_findings=self.findings,
                    temperature=self.judge_temperature,
                    on_detail=self.reporter.step_detail,
                    passage_descriptions=self.passage_descriptions,
                )
            self.findings.append(finding)
            if self.compute_embed_diversity:
                self.reporter.step_detail("diversity embedding")
                assert self.diversity_tracker is not None
                self.diversity_tracker.add_finding(
                    finding["sub_question"],
                    finding["answer"],
                )

        return self._record_step(
            step=step,
            iteration=iteration,
            clusters_excluded_step=clusters_excluded_step,
            sql_success_step=sql_success_step,
            tables_step=tables_step,
            passages_step=passages_step,
            is_finding=is_finding,
            rubric=rubric,
        )

    def finalize(self, final_report: str, iterations: List[dict]) -> Dict[str, Any]:
        """Compute query-level summary metrics."""
        del final_report, iterations

        retrieval_cumulative = build_retrieval_cumulative_metrics(
            tables_used_cumulative=self.tables_used_cumulative,
            passages_cited_cumulative=self.passages_cited_cumulative,
            gt_tables=self.gt_tables,
            gt_passages=self.gt_passages,
            total_lake_tables=self.total_lake_tables,
            use_passages=self.use_passages,
        )
        operational_cumulative = build_operational_cumulative_metrics(
            sql_attempts=self.sql_attempts,
            sql_successes=self.sql_successes,
            clusters_excluded_cumulative=self.clusters_excluded_cumulative,
            initial_candidate_clusters=self.initial_candidate_clusters,
            diversity=self._diversity_snapshot(),
            n_findings=len(self.findings),
        )
        research_quality_summary = build_research_quality_summary(
            finding_scores=self.finding_scores,
            budget=self.budget,
            per_step=self.per_step_history,
        )
        self.reporter.finish(research_quality_summary)

        return {
            "retrieval": {
                **self.retrieval,
                "cumulative": retrieval_cumulative,
            },
            "operational": {
                "cumulative": operational_cumulative,
            },
            "research_quality": research_quality_summary,
            "per_step": self.per_step_history,
            "judge_usage": summarize_metrics_usage(self._usage_start),
            "budget_steps_completed": self.steps_completed,
            "budget_steps_with_iteration": self.steps_with_iteration,
            "initial_candidate_clusters": self.initial_candidate_clusters,
            "total_lake_tables": self.total_lake_tables,
        }
