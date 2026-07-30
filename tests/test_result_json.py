"""Tests for human-readable result.json schema helpers."""

import json
import os
import tempfile
import unittest

from src.metrics.metrics_json import format_metrics_for_disk
from src.metrics.research_quality import compute_rubric_means_from_per_step
from src.result_json import (
    INITIAL_CLUSTERS_FILENAME,
    build_result_json,
    load_query_ground_truth,
    load_query_metrics,
    load_query_topk,
    result_budget,
    result_retained_clusters,
    result_time_taken,
    result_usage,
    save_query_result,
)


class TestResultJson(unittest.TestCase):
    def test_build_result_json_orders_human_fields_first(self) -> None:
        metrics = {
            "research_quality": {
                "report_score": 0.5,
                "finding_scores_sum": 1.0,
                "n_findings_valid": 1,
                "budget": 2,
                "rubric_means": {
                    "grounded": 1.0,
                    "relevance": 0.75,
                    "distinctness": 1.0,
                    "report_usefulness": 0.75,
                },
            },
            "retrieval": {
                "table_gt_in_top_k": 0.25,
                "cumulative": {"table_recall": 0.1},
            },
            "operational": {
                "cumulative": {
                    "sql_success_rate": 1.0,
                    "n_findings": 1,
                }
            },
            "budget_steps_completed": 2,
            "per_step": [
                {
                    "step": 1,
                    "research_quality": {
                        "finding_score": 0.5625,
                        "rubric": {
                            "judges": [
                                {
                                    "provider": "openai",
                                    "model": "gpt-test",
                                    "scores": {
                                        "grounded": 1.0,
                                        "relevance": 0.75,
                                        "distinctness": 1.0,
                                        "report_usefulness": 0.75,
                                    },
                                }
                            ],
                        },
                    },
                }
            ],
        }
        iterations = [
            {
                "step": 1,
                "finding_idx": 1,
                "sub_question": "Q?",
                "answer": "A.",
                "tables_used": ["T1"],
                "passages_cited": [],
            }
        ]
        result = build_result_json(
            query_id="1",
            user_query="User?",
            coverage="medium",
            method="pipeline",
            answer="Final answer.",
            iterations=iterations,
            ground_truth={"n_table": 1, "table": ["t1"]},
            topk_table_ids=["T1", "T2"],
            topk_passage_ids=[],
            total_inference_clusters=10,
            retained_clusters=3,
            time_taken=12.5,
            usage={"total_tokens": 100},
            metrics=metrics,
        )
        keys = list(result.keys())
        self.assertEqual(keys[:5], [
            "query_id",
            "user_query",
            "coverage",
            "method",
            "answer",
        ])
        self.assertEqual(result["findings"][0]["rubric"]["scores"]["grounded"], 1.0)
        self.assertEqual(result["summary"]["research_quality"]["rubric_means"]["relevance"], 0.75)
        self.assertEqual(result["ground_truth"], {"n_table": 1})
        self.assertEqual(result["metrics_path"], "metrics.json")
        self.assertNotIn("iterations", result)
        self.assertNotIn("metrics", result)

    def test_save_and_load_sidecars(self) -> None:
        tmp = tempfile.mkdtemp()
        query_dir = os.path.join(tmp, "q1")
        os.makedirs(query_dir)
        ground_truth = {"n_table": 1, "table": ["t1"]}
        result = build_result_json(
            query_id="q1",
            user_query="Q",
            coverage=None,
            method="opencode",
            answer="A",
            iterations=[],
            ground_truth=ground_truth,
            topk_table_ids=["T1"],
            topk_passage_ids=["P1"],
            total_inference_clusters=0,
            retained_clusters=0,
            time_taken=1.0,
            usage={},
        )
        metrics = {"research_quality": {"report_score": 0.1}}
        save_query_result(
            query_dir,
            result,
            ground_truth=ground_truth,
            topk_table_ids=["T1"],
            topk_passage_ids=["P1"],
            metrics=metrics,
        )
        with open(os.path.join(query_dir, "result.json"), encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(load_query_ground_truth(query_dir), ground_truth)
        self.assertEqual(load_query_topk(query_dir), (["T1"], ["P1"]))
        self.assertEqual(load_query_metrics(query_dir, loaded), format_metrics_for_disk(metrics))
        self.assertEqual(result_usage(loaded), {})
        self.assertEqual(result_time_taken(loaded), 1.0)
        self.assertEqual(result_retained_clusters(loaded), 0)

    def test_findings_sorted_by_finding_idx(self) -> None:
        metrics = {
            "per_step": [
                {
                    "step": 2,
                    "research_quality": {"rubric": None},
                }
            ]
        }
        iterations = [
            {"step": 2, "finding_idx": 2, "sub_question": "B", "answer": "b"},
            {"step": 1, "finding_idx": 1, "sub_question": "A", "answer": "a"},
        ]
        result = build_result_json(
            query_id="1",
            user_query="Q",
            coverage=None,
            method="pipeline",
            answer="Final",
            iterations=iterations,
            ground_truth=None,
            topk_table_ids=[],
            topk_passage_ids=[],
            total_inference_clusters=0,
            retained_clusters=0,
            time_taken=1.0,
            usage={},
            metrics=metrics,
        )
        self.assertEqual(
            [finding["finding_idx"] for finding in result["findings"]],
            [1, 2],
        )

    def test_artifacts_omit_ground_truth_when_absent(self) -> None:
        tmp = tempfile.mkdtemp()
        query_dir = os.path.join(tmp, "q1")
        os.makedirs(query_dir)
        result = build_result_json(
            query_id="q1",
            user_query="Q",
            coverage=None,
            method="pipeline",
            answer="A",
            iterations=[],
            ground_truth=None,
            topk_table_ids=[],
            topk_passage_ids=[],
            total_inference_clusters=0,
            retained_clusters=0,
            time_taken=1.0,
            usage={},
            query_dir=query_dir,
        )
        self.assertNotIn("ground_truth", result["run"]["artifacts"])

    def test_artifacts_include_ground_truth_when_present(self) -> None:
        tmp = tempfile.mkdtemp()
        query_dir = os.path.join(tmp, "q1")
        os.makedirs(query_dir)
        result = build_result_json(
            query_id="q1",
            user_query="Q",
            coverage=None,
            method="pipeline",
            answer="A",
            iterations=[],
            ground_truth={"n_table": 1, "table": ["t1"]},
            topk_table_ids=[],
            topk_passage_ids=[],
            total_inference_clusters=0,
            retained_clusters=0,
            time_taken=1.0,
            usage={},
            query_dir=query_dir,
        )
        self.assertEqual(result["run"]["artifacts"]["ground_truth"], "ground_truth.json")

    def test_artifacts_include_initial_clusters_when_present(self) -> None:
        tmp = tempfile.mkdtemp()
        query_dir = os.path.join(tmp, "q1")
        os.makedirs(query_dir)
        clusters_path = os.path.join(query_dir, INITIAL_CLUSTERS_FILENAME)
        with open(clusters_path, "w", encoding="utf-8") as f:
            json.dump({"clusters": []}, f)
        result = build_result_json(
            query_id="q1",
            user_query="Q",
            coverage=None,
            method="opencode",
            answer="A",
            iterations=[],
            ground_truth=None,
            topk_table_ids=["T1"],
            topk_passage_ids=[],
            total_inference_clusters=1,
            retained_clusters=1,
            time_taken=1.0,
            usage={},
            query_dir=query_dir,
            initial_clusters_path=clusters_path,
        )
        self.assertEqual(
            result["run"]["artifacts"]["initial_clusters"],
            INITIAL_CLUSTERS_FILENAME,
        )

    def test_load_query_ground_truth_missing_sidecar(self) -> None:
        query_dir = os.path.join(tempfile.mkdtemp(), "q1")
        os.makedirs(query_dir)
        self.assertIsNone(load_query_ground_truth(query_dir))

    def test_summary_status_excludes_opencode(self) -> None:
        result = build_result_json(
            query_id="1",
            user_query="Q",
            coverage=None,
            method="opencode",
            answer="A",
            iterations=[],
            ground_truth=None,
            topk_table_ids=[],
            topk_passage_ids=[],
            total_inference_clusters=0,
            retained_clusters=0,
            time_taken=1.0,
            usage={},
            metrics={"budget_steps_completed": 5},
            opencode_meta={"status": "completed", "returncode": 0},
            query_dir=tempfile.mkdtemp(),
        )
        self.assertEqual(result["summary"]["status"], {"budget_steps_completed": 5})
        self.assertEqual(result["opencode"]["status"], "completed")

    def test_result_budget_prefers_configured_budget(self) -> None:
        result = {
            "summary": {
                "research_quality": {"budget": 50},
                "status": {"budget_steps_completed": 50},
            },
            "findings": [{"step": 10, "finding_idx": 1}],
        }
        self.assertEqual(result_budget(result), 50)

    def test_compute_rubric_means_from_per_step(self) -> None:
        means = compute_rubric_means_from_per_step(
            [
                {
                    "research_quality": {
                        "rubric": {
                            "components": {
                                "grounded": {"mean": 1.0},
                                "relevance": {"mean": 0.5},
                                "distinctness": {"mean": 1.0},
                                "report_usefulness": {"mean": 0.5},
                            }
                        }
                    }
                },
                {"research_quality": {"rubric": None}},
            ]
        )
        self.assertEqual(means["grounded"], 0.5)
        self.assertEqual(means["relevance"], 0.25)

    def test_build_result_json_flags_opencode_exec(self) -> None:
        result = build_result_json(
            query_id="1",
            user_query="User?",
            coverage="medium",
            method="pipeline",
            answer="Final answer.",
            iterations=[],
            ground_truth=None,
            topk_table_ids=[],
            topk_passage_ids=[],
            total_inference_clusters=0,
            retained_clusters=0,
            time_taken=1.0,
            usage={"total_tokens": 0},
            opencode_exec=True,
        )
        self.assertTrue(result.get("opencode_exec"))


if __name__ == "__main__":
    unittest.main()
