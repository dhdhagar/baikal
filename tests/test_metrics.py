"""Tests for evaluation metrics."""

import unittest
from unittest.mock import patch

import numpy as np

from src.metrics.common import (
    assign_finding_indices,
    reorder_iteration_payload,
    extract_gt_passage_ids,
    extract_gt_table_ids,
    extract_passages_from_answer,
    extract_tables_from_answer,
    is_finding_iteration,
    is_report_eligible_iteration,
    is_successful_iteration,
    map_gt_passage_id,
    map_gt_table_id,
    snap_binary_score,
    snap_ordinal_score,
)
from src.metrics.operational import compute_diversity
from src.metrics.research_quality import (
    build_research_quality_summary,
    compute_finding_score,
    judge_finding_rubric,
    parse_judge_model_specs,
)
from src.metrics.retrieval import (
    compute_table_gt_in_top_k,
    compute_table_gt_reachable,
    compute_lake_coverage,
    compute_passage_gt_in_top_k,
    compute_passage_gt_reachable,
    compute_passage_precision,
    compute_passage_recall,
    compute_table_precision,
    compute_table_recall,
)
from src.sql_db import extract_tables_from_sql


class TestMetrics(unittest.TestCase):
    def test_extract_tables_from_sql(self):
        sql = """
        SELECT Stadium, MAX(cap_num) AS Capacity
        FROM (
          SELECT Stadium, cap_num FROM T1480
          UNION ALL
          SELECT Stadium, cap_num FROM T3527
        )
        JOIN T7130 ON T7130.id = T1480.id
        """
        tables = extract_tables_from_sql(sql)
        self.assertEqual(tables, {"T1480", "T3527", "T7130"})

    def test_extract_tables_from_sql_quoted_identifiers(self):
        self.assertEqual(
            extract_tables_from_sql("SELECT * FROM T809 LIMIT 10"),
            {"T809"},
        )
        self.assertEqual(
            extract_tables_from_sql('SELECT * FROM "T809" LIMIT 10'),
            {"T809"},
        )
        self.assertEqual(
            extract_tables_from_sql("SELECT * FROM 'T809' LIMIT 10"),
            {"T809"},
        )
        self.assertEqual(
            extract_tables_from_sql("SELECT a FROM `T809` JOIN `T810` ON a.id = b.id"),
            {"T809", "T810"},
        )

    def test_table_recall_and_precision(self):
        gt = {"T1", "T2", "T3", "T4"}
        used = {"T1", "T2", "T99"}
        self.assertEqual(compute_table_recall(used, gt), 0.5)
        self.assertEqual(compute_table_precision(used, gt), 2 / 3)

    def test_extract_gt_table_ids(self):
        self.assertEqual(
            extract_gt_table_ids({"tables": ["T1", "T2"]}),
            ["T1", "T2"],
        )
        self.assertEqual(
            extract_gt_table_ids({"table": ["T3", "T4"]}),
            ["T3", "T4"],
        )
        self.assertEqual(
            extract_gt_table_ids({"tables": ["T1"], "table": ["T2"]}),
            ["T1"],
        )
        self.assertIsNone(extract_gt_table_ids(None))
        self.assertIsNone(extract_gt_table_ids({}))

    def test_map_gt_table_ids(self):
        uid_to_table = {"22fe9439-57f6-5815-b223-8fa1d650485b": "T5060"}
        self.assertEqual(map_gt_table_id("T5060"), "T5060")
        self.assertEqual(
            map_gt_table_id("22fe9439-57f6-5815-b223-8fa1d650485b", uid_to_table),
            "T5060",
        )
        self.assertEqual(
            extract_gt_table_ids(
                {"table": ["22fe9439-57f6-5815-b223-8fa1d650485b", "T1"]},
                uid_to_table_id=uid_to_table,
            ),
            ["T5060", "T1"],
        )

    def test_map_gt_passage_ids(self):
        uid_to_passage = {"abc_0-5": "P42"}
        self.assertEqual(map_gt_passage_id("P42"), "P42")
        self.assertEqual(map_gt_passage_id("abc_0-5", uid_to_passage), "P42")
        self.assertEqual(
            extract_gt_passage_ids(
                {"synth_text": ["abc_0-5", "P7"]},
                uid_to_passage_id=uid_to_passage,
            ),
            ["P42", "P7"],
        )
        self.assertEqual(
            extract_gt_passage_ids(
                {"text": ["raw-uid-1"]},
                uid_to_passage_id={"raw-uid-1": "P99"},
                passage_type="raw",
            ),
            ["P99"],
        )

    def test_extract_passages_from_answer(self):
        answer = "Road density is higher in urban areas [P42], especially near [p7]."
        self.assertEqual(
            extract_passages_from_answer(answer),
            {"P42", "P7"},
        )

    def test_extract_passages_from_answer_bare_ids(self):
        answer = "Road density is higher in urban areas P42, especially near p7."
        self.assertEqual(
            extract_passages_from_answer(answer),
            {"P42", "P7"},
        )

    def test_extract_passages_from_answer_mixed_formats(self):
        answer = "See [P1] and also P2 for details."
        self.assertEqual(
            extract_passages_from_answer(answer),
            {"P1", "P2"},
        )

    def test_extract_tables_from_answer(self):
        answer = "Capacity is 50,000 from T809 and also [T42]."
        self.assertEqual(
            extract_tables_from_answer(answer),
            {"T809", "T42"},
        )

    def test_tracker_normalizes_table_ids_from_answer(self):
        from src.metrics.tracker import MetricsTracker

        tracker = MetricsTracker(
            user_query="q",
            budget=1,
            gt_tables=None,
            total_lake_tables=1,
            topk_table_ids=[],
            inference_clusters=[],
            initial_candidate_clusters=0,
            k_relevant_tables=0,
            judge_llms=[],
            research_quality_enabled=False,
        )
        iteration = {
            "step": 1,
            "sub_question": "sq",
            "answer": "From t809.",
            "needs_sql": False,
        }
        tracker.record_iteration(iteration, 1, 0)
        self.assertEqual(iteration["tables_used"], ["T809"])

    def test_answered_with_passage_evidence_bare_citation(self):
        iteration = {
            "answer": "Urban density is higher in the core P42.",
            "needs_sql": True,
            "execution": {"ok": False, "row_count": 0},
        }
        self.assertTrue(is_report_eligible_iteration(iteration))

    def test_passage_recall_and_precision(self):
        gt = {"P1", "P2", "P3", "P4"}
        cited = {"P1", "P2", "P99"}
        self.assertEqual(compute_passage_recall(cited, gt), 0.5)
        self.assertEqual(compute_passage_precision(cited, gt), 2 / 3)

    def test_passage_gt_retrieval_metrics(self):
        gt = {"P1", "P2", "P3"}
        top_k = ["P1", "P5", "P6"]
        clusters = [
            {"passages": [{"passage_id": "P1"}, {"passage_id": "P7"}]},
            {"passages": [{"passage_id": "P8"}]},
            {"passages": [{"passage_id": "P2"}, {"passage_id": "P5"}]},
        ]
        self.assertEqual(compute_passage_gt_in_top_k(gt, top_k), 1 / 3)
        self.assertEqual(compute_passage_gt_reachable(gt, clusters, top_k), 2 / 3)

    def test_lake_coverage(self):
        self.assertEqual(compute_lake_coverage({"T1", "T2"}, 1000), 0.002)

    def test_gt_retrieval_metrics(self):
        gt = {"T1", "T2", "T3"}
        top_k = ["T1", "T5", "T6"]
        clusters = [
            {"tables": [{"table_id": "T1"}, {"table_id": "T7"}]},
            {"tables": [{"table_id": "T8"}]},
            {"tables": [{"table_id": "T2"}, {"table_id": "T5"}]},
        ]
        self.assertEqual(compute_table_gt_in_top_k(gt, top_k), 1 / 3)
        self.assertEqual(compute_table_gt_reachable(gt, clusters, top_k), 2 / 3)

    def test_is_successful_iteration(self):
        self.assertTrue(is_successful_iteration({"ok": True, "row_count": 1}))
        self.assertFalse(is_successful_iteration({"ok": True, "row_count": 0}))
        self.assertFalse(is_successful_iteration({"ok": False, "row_count": 5}))

    def test_is_report_eligible_iteration(self):
        self.assertTrue(
            is_report_eligible_iteration(
                {
                    "needs_sql": True,
                    "execution": {"ok": True, "row_count": 3},
                }
            )
        )
        self.assertFalse(
            is_report_eligible_iteration(
                {
                    "needs_sql": True,
                    "execution": {"ok": False, "row_count": 0},
                    "answer": "Could not determine from SQL.",
                }
            )
        )
        self.assertTrue(
            is_report_eligible_iteration(
                {
                    "needs_sql": True,
                    "execution": {"ok": False, "row_count": 0},
                    "answer": "Urban density is higher [P42].",
                }
            )
        )
        self.assertFalse(
            is_report_eligible_iteration(
                {
                    "needs_sql": True,
                    "execution": {"ok": True, "row_count": 0},
                }
            )
        )
        self.assertTrue(
            is_report_eligible_iteration(
                {"needs_sql": False, "answer": "from passages"}
            )
        )

    def test_is_finding_iteration(self):
        self.assertFalse(
            is_finding_iteration(
                {
                    "needs_sql": True,
                    "execution": {"ok": False, "row_count": 0},
                    "answer": "failed",
                }
            )
        )
        self.assertTrue(
            is_finding_iteration(
                {
                    "needs_sql": True,
                    "execution": {"ok": False, "row_count": 0},
                    "answer": "Answer from passages [P7].",
                }
            )
        )
        self.assertTrue(
            is_finding_iteration(
                {
                    "needs_sql": False,
                    "answer": "from passages",
                }
            )
        )
        self.assertFalse(
            is_finding_iteration(
                {
                    "needs_sql": False,
                    "answer": "   ",
                }
            )
        )

    def test_reorder_iteration_payload(self):
        payload = {
            "step": 1,
            "sub_question": "q",
            "finding_idx": 2,
            "answer": "a",
        }
        ordered = reorder_iteration_payload(payload)
        self.assertEqual(list(ordered.keys())[:2], ["step", "finding_idx"])

    def test_assign_finding_indices(self):
        iterations = [
            {
                "step": 1,
                "needs_sql": True,
                "execution": {"ok": True, "row_count": 2},
            },
            {
                "step": 2,
                "needs_sql": True,
                "execution": {"ok": False, "row_count": 0},
                "answer": "Could not determine from SQL.",
            },
            {
                "step": 3,
                "needs_sql": True,
                "execution": {"ok": False, "row_count": 0},
                "answer": "Urban density is higher [P42].",
            },
            {"step": 4, "needs_sql": False, "answer": "from passages"},
        ]
        assign_finding_indices(iterations)
        self.assertEqual(iterations[0]["finding_idx"], 1)
        self.assertIsNone(iterations[1]["finding_idx"])
        self.assertEqual(iterations[2]["finding_idx"], 2)
        self.assertEqual(iterations[3]["finding_idx"], 3)

    def test_snap_scores(self):
        self.assertEqual(snap_binary_score("yes"), 1.0)
        self.assertEqual(snap_binary_score("no"), 0.0)
        self.assertEqual(snap_ordinal_score("partial"), 0.5)
        self.assertEqual(snap_ordinal_score("full"), 1.0)
        self.assertEqual(snap_ordinal_score(0.24), 0.25)

    def test_compute_finding_score(self):
        score = compute_finding_score(
            {
                "grounded": 1.0,
                "relevance": 0.75,
                "distinctness": 0.5,
                "report_usefulness": 1.0,
            }
        )
        self.assertEqual(score, 0.375)

    def test_report_score_aggregation(self):
        summary = build_research_quality_summary(
            finding_scores=[0.0, 0.5, 0.0, 1.0],
            budget=4,
            per_step=[
                {
                    "research_quality": {
                        "rubric": {
                            "components": {
                                "grounded": {"mean": 1.0},
                                "relevance": {"mean": 0.5},
                                "distinctness": {"mean": 1.0},
                                "report_usefulness": {"mean": 1.0},
                            }
                        }
                    }
                },
                {"research_quality": {"rubric": None}},
            ],
        )
        self.assertEqual(summary["finding_scores_sum"], 1.5)
        self.assertEqual(summary["report_score"], 0.375)
        self.assertEqual(summary["n_findings_valid"], 2)
        self.assertNotIn("finding_scores", summary)
        self.assertEqual(summary["rubric_means"]["grounded"], 0.5)

    def test_summarize_metrics_usage(self):
        from src.metrics.usage import capture_metrics_usage_start, summarize_metrics_usage
        from src.tracking import get_tracker, reset_tracker

        reset_tracker()
        start = capture_metrics_usage_start()
        get_tracker().record(
            call_type="completion",
            feature="metrics_finding_rubric",
            provider="openai",
            model="gpt-5-mini",
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=0.001,
            time_taken=0.5,
        )
        usage = summarize_metrics_usage(start)
        self.assertEqual(usage["total"]["total_tokens"], 120)
        self.assertEqual(usage["total"]["n_calls"], 1)
        self.assertIn("metrics_finding_rubric", usage["by_feature"])

    def test_parse_judge_model_specs(self):
        self.assertEqual(
            parse_judge_model_specs("gpt-5-mini"),
            [("openai", "gpt-5-mini")],
        )
        self.assertEqual(
            parse_judge_model_specs("gpt-5-mini, gpt-4o"),
            [("openai", "gpt-5-mini"), ("openai", "gpt-4o")],
        )

    def test_parse_judge_model_specs_mixed_providers(self):
        specs = parse_judge_model_specs(
            "openai:gpt-5-mini,litellm:azure/gpt-4o,gpt-4o",
            default_provider="openai",
        )
        self.assertEqual(
            specs,
            [
                ("openai", "gpt-5-mini"),
                ("litellm", "azure/gpt-4o"),
                ("openai", "gpt-4o"),
            ],
        )

    def test_judge_finding_rubric_multi_judge(self):
        class _FakeLLM:
            def __init__(self, model: str):
                self.provider = "openai"
                self.model = model

        fake_response = """
        {
          "grounded_reasoning": "Supported by SQL.",
          "grounded": "yes",
          "relevance_reasoning": "Mostly relevant.",
          "relevance": "substantial",
          "distinctness_reasoning": "Some overlap.",
          "distinctness": "partial",
          "report_usefulness_reasoning": "Useful insight.",
          "report_usefulness": "full"
        }
        """

        with patch("src.metrics.research_quality.chat", return_value=fake_response):
            result = judge_finding_rubric(
                [_FakeLLM("judge-a"), _FakeLLM("judge-b")],
                user_query="test query",
                iteration={
                    "sub_question": "sq",
                    "answer": "ans",
                    "needs_sql": True,
                    "sql": "SELECT 1",
                    "execution": {"row_count": 1, "rows": [{"x": 1}]},
                    "tables_used": ["T1"],
                    "selected_cluster": {"passage_ids": [["P1", "title"]]},
                },
                prior_findings=[],
                temperature=1.0,
            )

        self.assertEqual(len(result["judges"]), 2)
        self.assertEqual(result["components"]["grounded"]["mean"], 1.0)
        self.assertEqual(result["components"]["relevance"]["mean"], 0.75)
        self.assertEqual(result["components"]["relevance"]["variance"], 0.0)
        self.assertEqual(result["finding_score"], 0.375)

    def test_judge_prompt_includes_answer_citations(self):
        class _FakeLLM:
            provider = "openai"
            model = "judge-a"

        fake_response = """
        {
          "grounded_reasoning": "Supported.",
          "grounded": "yes",
          "relevance_reasoning": "Relevant.",
          "relevance": "full",
          "distinctness_reasoning": "New.",
          "distinctness": "full",
          "report_usefulness_reasoning": "Useful.",
          "report_usefulness": "full"
        }
        """
        captured: dict = {}

        def _capture_chat(_llm, prompt, **_kwargs):
            captured["prompt"] = prompt
            return fake_response

        with patch("src.metrics.research_quality.chat", side_effect=_capture_chat):
            judge_finding_rubric(
                [_FakeLLM()],
                user_query="test query",
                iteration={
                    "sub_question": "sq",
                    "answer": "Density is higher P42.",
                    "needs_sql": False,
                    "tables_used": ["T809"],
                    "passages_cited": ["P42"],
                },
                prior_findings=[],
                passage_descriptions={
                    "P42": {
                        "title": "Urban density",
                        "text": "Urban areas are denser.",
                    }
                },
            )

        prompt = captured["prompt"]
        self.assertIn("Passages cited in answer: P42", prompt)
        self.assertIn("[P42] Urban density", prompt)
        self.assertIn("Tables cited (SQL + answer): T809", prompt)
        self.assertIn("absence-only", prompt)
        self.assertIn('grounded: "no"', prompt)

    def test_diversity(self):
        class _FakeEmbedder:
            def encode(self, texts, show_progress_bar=False, feature="embedding"):
                out = []
                for i, _ in enumerate(texts):
                    v = np.zeros(4, dtype=np.float64)
                    v[i % 4] = 1.0
                    out.append(v)
                return np.asarray(out, dtype=np.float64)

        texts = ["a", "b", "c"]
        div = compute_diversity(texts, _FakeEmbedder())
        self.assertIsNotNone(div["mean"])
        self.assertIsNotNone(div["max"])
        self.assertGreater(div["mean"], 0.5)

        single = compute_diversity(["only"], _FakeEmbedder())
        self.assertIsNone(single["mean"])

    def test_sql_success_rate_uses_attempts_only(self):
        from src.metrics.operational import build_operational_cumulative_metrics

        metrics = build_operational_cumulative_metrics(
            sql_attempts=2,
            sql_successes=1,
            clusters_excluded_cumulative=0,
            initial_candidate_clusters=10,
            diversity={"mean": None, "max": None},
            n_findings=1,
        )
        self.assertEqual(metrics["sql_success_rate"], 0.5)
        self.assertEqual(metrics["n_sql_attempts"], 2)

        no_attempts = build_operational_cumulative_metrics(
            sql_attempts=0,
            sql_successes=0,
            clusters_excluded_cumulative=0,
            initial_candidate_clusters=10,
            diversity={"mean": None, "max": None},
            n_findings=0,
        )
        self.assertIsNone(no_attempts["sql_success_rate"])

    def test_aggregate_run_metrics(self):
        from src.metrics.aggregate import aggregate_run_metrics, format_run_metrics_summary

        outputs = [
            {
                "query_id": "q1",
                "coverage": "low",
                "metrics": {
                    "research_quality": {
                        "report_score": 0.5,
                        "finding_scores_sum": 2.0,
                        "n_findings_valid": 2,
                        "rubric_means": {
                            "grounded": 0.8,
                            "relevance": 0.6,
                            "distinctness": 0.4,
                            "report_usefulness": 0.5,
                        },
                    },
                    "retrieval": {
                        "table_gt_in_top_k": 1.0,
                        "table_gt_reachable": 0.5,
                        "passage_gt_in_top_k": 0.8,
                        "passage_gt_reachable": 0.6,
                        "cumulative": {
                            "table_recall": 0.8,
                            "table_precision": 0.6,
                            "passage_recall": 0.7,
                            "passage_precision": 0.5,
                            "lake_coverage": 0.1,
                        },
                    },
                    "operational": {
                        "cumulative": {
                            "sql_success_rate": 0.75,
                            "cluster_attrition_rate": 0.2,
                            "diversity": {"mean": 0.9, "max": 1.0},
                            "n_findings": 2,
                        }
                    },
                },
            },
            {
                "query_id": "q2",
                "coverage": "high",
                "metrics": {
                    "research_quality": {
                        "report_score": 0.25,
                        "finding_scores_sum": 1.0,
                        "n_findings_valid": 1,
                        "rubric_means": {
                            "grounded": 0.4,
                            "relevance": 0.8,
                            "distinctness": 0.2,
                            "report_usefulness": 0.3,
                        },
                    },
                    "retrieval": {
                        "table_gt_in_top_k": 0.0,
                        "table_gt_reachable": 1.0,
                        "passage_gt_in_top_k": 0.2,
                        "passage_gt_reachable": 0.4,
                        "cumulative": {
                            "table_recall": 0.4,
                            "table_precision": 0.2,
                            "passage_recall": 0.3,
                            "passage_precision": 0.1,
                            "lake_coverage": 0.05,
                        },
                    },
                    "operational": {
                        "cumulative": {
                            "sql_success_rate": 0.5,
                            "cluster_attrition_rate": 0.4,
                            "diversity": {"mean": 0.7, "max": 0.8},
                            "n_findings": 1,
                        }
                    },
                },
            },
        ]
        summary = aggregate_run_metrics(outputs)
        self.assertEqual(summary["n_queries"], 2)
        self.assertEqual(list(summary.keys()), ["n_queries", "overall", "per_coverage", "per_query"])
        overall_rq = summary["overall"]["research_quality"]
        self.assertEqual(overall_rq["report_score"], 0.375)
        self.assertEqual(overall_rq["rubric_means"]["relevance"], 0.7)
        self.assertEqual(summary["overall"]["retrieval"]["table_recall"], 0.6)
        self.assertEqual(summary["overall"]["retrieval"]["passage_recall"], 0.5)
        self.assertEqual(summary["overall"]["operational"]["sql_success_rate"], 0.625)
        self.assertEqual(summary["overall"]["operational"]["n_findings"], 1.5)
        self.assertEqual(
            summary["per_coverage"]["low"]["research_quality"]["report_score"],
            0.5,
        )
        self.assertEqual(
            summary["per_coverage"]["high"]["research_quality"]["report_score"],
            0.25,
        )
        self.assertEqual(summary["per_query"][0]["query_id"], "q1")
        self.assertEqual(summary["per_query"][1]["query_id"], "q2")
        text = format_run_metrics_summary(summary)
        self.assertLess(text.index("Overall avg report_score:"), text.index("q1"))
        self.assertIn("Per coverage", text)
        self.assertIn("low (1 queries)", text)
        self.assertIn("high (1 queries)", text)
        self.assertIn("q1", text)
        self.assertIn("Overall avg table_recall:", text)
        self.assertIn("Overall avg passage_recall:", text)
        self.assertIn("Overall avg report_score: 0.3750", text)


if __name__ == "__main__":
    unittest.main()
