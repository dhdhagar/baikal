"""Tests for metrics.json formatting."""

import unittest

from src.metrics.metrics_json import format_metrics_for_disk
from src.metrics.research_quality import (
    build_research_quality_summary,
    compute_rubric_means_from_per_step,
)
from src.metrics.rubric_utils import extract_rubric_scores, slim_rubric_for_metrics_json


class TestMetricsJson(unittest.TestCase):
    def test_format_metrics_drops_redundant_fields(self) -> None:
        metrics = {
            "research_quality": {
                "report_score": 0.5,
                "finding_scores_sum": 1.0,
                "n_findings_valid": 1,
                "budget": 2,
                "finding_scores": [0.5, 0.5],
                "rubric_means": {"grounded": 1.0},
            },
            "retrieval": {
                "table_gt_in_top_k": 0.25,
                "k_relevant_tables": 10,
                "n_gt_tables": 4,
                "cumulative": {"table_recall": 0.1},
            },
            "operational": {"cumulative": {"n_findings": 1}},
            "usage": {"total": {"n_calls": 2}},
            "budget_steps_completed": 2,
            "per_step": [
                {
                    "step": 1,
                    "retrieval": {
                        "tables_used": ["T1"],
                        "table_recall": 0.0,
                        "passages_cited": ["P1"],
                    },
                    "operational": {"sql_success": 1},
                    "research_quality": {
                        "is_finding": True,
                        "finding_score": 0.5,
                        "rubric": {
                            "finding_score": 0.5,
                            "components": {
                                "grounded": {"mean": 1.0},
                                "relevance": {"mean": 0.5},
                                "distinctness": {"mean": 1.0},
                                "report_usefulness": {"mean": 0.5},
                            },
                            "judges": [
                                {
                                    "provider": "openai",
                                    "model": "gpt-test",
                                    "scores": {
                                        "grounded": 1.0,
                                        "relevance": 0.5,
                                        "distinctness": 1.0,
                                        "report_usefulness": 0.5,
                                    },
                                    "reasoning": {"grounded": "ok"},
                                }
                            ],
                        },
                    },
                }
            ],
        }
        formatted = format_metrics_for_disk(metrics)
        keys = list(formatted.keys())
        self.assertEqual(
            keys,
            [
                "research_quality",
                "retrieval",
                "operational",
                "judge_usage",
                "budget_steps_completed",
                "per_step",
            ],
        )
        self.assertNotIn("finding_scores", formatted["research_quality"])
        self.assertNotIn("k_relevant_tables", formatted["retrieval"])
        self.assertNotIn("n_gt_tables", formatted["retrieval"])
        self.assertEqual(formatted["judge_usage"]["total"]["n_calls"], 2)
        rubric = formatted["per_step"][0]["research_quality"]["rubric"]
        self.assertNotIn("components", rubric)
        self.assertNotIn("finding_score", rubric)
        self.assertEqual(rubric["judges"][0]["scores"]["grounded"], 1.0)
        per_step_retrieval = formatted["per_step"][0]["retrieval"]
        self.assertNotIn("tables_used", per_step_retrieval)
        self.assertNotIn("passages_cited", per_step_retrieval)
        self.assertEqual(per_step_retrieval["table_recall"], 0.0)

    def test_extract_rubric_scores_from_judges(self) -> None:
        scores = extract_rubric_scores(
            {
                "judges": [
                    {
                        "scores": {
                            "grounded": 1.0,
                            "relevance": 0.5,
                            "distinctness": 1.0,
                            "report_usefulness": 0.5,
                        }
                    }
                ]
            }
        )
        self.assertEqual(scores["grounded"], 1.0)
        self.assertEqual(scores["relevance"], 0.5)

    def test_slim_rubric_keeps_aggregated_for_multi_judge(self) -> None:
        rubric = {
            "judges": [
                {"provider": "a", "model": "m1", "scores": {"grounded": 1.0}},
                {"provider": "b", "model": "m2", "scores": {"grounded": 0.0}},
            ],
            "components": {
                "grounded": {"mean": 0.5, "variance": 0.25},
                "relevance": {"mean": 1.0, "variance": 0.0},
            },
        }
        slim = slim_rubric_for_metrics_json(rubric)
        self.assertIsNotNone(slim)
        assert slim is not None
        self.assertEqual(len(slim["judges"]), 2)
        self.assertEqual(slim["aggregated"]["grounded"]["mean"], 0.5)
        self.assertEqual(slim["aggregated"]["grounded"]["variance"], 0.25)
        self.assertNotIn("variance", slim["aggregated"]["relevance"])

    def test_build_research_quality_summary_omits_finding_scores_list(self) -> None:
        summary = build_research_quality_summary(
            finding_scores=[0.0, 0.5],
            budget=2,
            per_step=[],
        )
        self.assertNotIn("finding_scores", summary)

    def test_compute_rubric_means_from_judges(self) -> None:
        means = compute_rubric_means_from_per_step(
            [
                {
                    "research_quality": {
                        "rubric": {
                            "judges": [
                                {
                                    "scores": {
                                        "grounded": 1.0,
                                        "relevance": 0.5,
                                        "distinctness": 1.0,
                                        "report_usefulness": 0.5,
                                    }
                                }
                            ]
                        }
                    }
                }
            ]
        )
        self.assertEqual(means["grounded"], 1.0)
        self.assertEqual(means["relevance"], 0.5)


if __name__ == "__main__":
    unittest.main()
