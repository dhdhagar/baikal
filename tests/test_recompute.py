"""Tests for offline metrics recomputation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from argparse import Namespace

from src.metrics.recompute import recompute_query_metrics
from src.utils import load_json


class TestRecompute(unittest.TestCase):
    def test_recompute_persists_bare_passage_citations(self) -> None:
        fixture_lake = os.path.join(
            os.path.dirname(__file__), "fixtures", "opencode_lake"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            query_dir = os.path.join(run_dir, "q1")
            os.makedirs(query_dir)

            clusters_path = os.path.join(run_dir, "clusters.json")
            with open(clusters_path, "w", encoding="utf-8") as f:
                json.dump([], f)

            args_path = os.path.join(run_dir, "args.json")
            with open(args_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "budget": 1,
                        "use_passages": False,
                        "compute_metrics": True,
                        "no_llm_judge": True,
                        "llm_provider": "openai",
                        "inference_clusters_path": clusters_path,
                        "tables_lake_dir": fixture_lake,
                        "k_relevant_tables": 0,
                        "k_relevant_passages": 0,
                    },
                    f,
                )

            iteration = {
                "step": 1,
                "sub_question": "What about density?",
                "answer": "Urban density is higher in the core P42.",
                "needs_sql": False,
                "passages_cited": [],
                "tables_used": [],
            }
            with open(
                os.path.join(query_dir, "iteration_001.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(iteration, f)

            with open(os.path.join(query_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query_id": "q1",
                        "user_query": "Research question?",
                        "answer": "Final report.",
                        "method": "pipeline",
                    },
                    f,
                )

            args = Namespace(**load_json(args_path))
            recompute_query_metrics(
                query_dir,
                args,
                inference_clusters=[],
                research_quality_enabled=False,
            )

            result = load_json(os.path.join(query_dir, "result.json"))
            findings = result.get("findings") or []
            self.assertEqual(len(findings), 1)
            self.assertEqual(findings[0]["passages_cited"], ["P42"])

            artifact = load_json(os.path.join(query_dir, "iteration_001.json"))
            self.assertEqual(artifact["passages_cited"], ["P42"])
            self.assertIn("metrics", artifact)

    def test_recompute_preserves_prior_judge_metrics_without_llm_judge(self) -> None:
        fixture_lake = os.path.join(
            os.path.dirname(__file__), "fixtures", "opencode_lake"
        )
        prior_rubric = {
            "judges": [
                {
                    "provider": "openai",
                    "model": "gpt-5-mini",
                    "scores": {
                        "grounded": 0.9,
                        "relevance": 0.8,
                        "distinctness": 0.7,
                        "report_usefulness": 0.6,
                    },
                    "reasoning": {},
                }
            ]
        }
        prior_metrics = {
            "research_quality": {
                "report_score": 0.42,
                "finding_scores_sum": 0.42,
                "n_findings_valid": 1,
                "budget": 1,
                "rubric_means": {
                    "grounded": 0.9,
                    "relevance": 0.8,
                    "distinctness": 0.7,
                    "report_usefulness": 0.6,
                },
            },
            "judge_usage": {"total": {"n_calls": 1, "cost_usd": 0.01}},
            "per_step": [
                {
                    "step": 1,
                    "retrieval": {},
                    "operational": {},
                    "research_quality": {
                        "is_finding": True,
                        "finding_score": 0.42,
                        "rubric": prior_rubric,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            query_dir = os.path.join(run_dir, "q1")
            os.makedirs(query_dir)

            clusters_path = os.path.join(run_dir, "clusters.json")
            with open(clusters_path, "w", encoding="utf-8") as f:
                json.dump([], f)

            args_path = os.path.join(run_dir, "args.json")
            with open(args_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "budget": 1,
                        "use_passages": False,
                        "compute_metrics": True,
                        "no_llm_judge": True,
                        "llm_provider": "openai",
                        "inference_clusters_path": clusters_path,
                        "tables_lake_dir": fixture_lake,
                        "k_relevant_tables": 0,
                        "k_relevant_passages": 0,
                    },
                    f,
                )

            iteration = {
                "step": 1,
                "sub_question": "What about density?",
                "answer": "Urban density is higher in the core P42.",
                "needs_sql": False,
                "passages_cited": [],
                "tables_used": [],
            }
            with open(
                os.path.join(query_dir, "iteration_001.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(iteration, f)

            with open(os.path.join(query_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(prior_metrics, f)

            with open(os.path.join(query_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query_id": "q1",
                        "user_query": "Research question?",
                        "answer": "Final report.",
                        "method": "pipeline",
                    },
                    f,
                )

            args = Namespace(**load_json(args_path))
            recompute_query_metrics(
                query_dir,
                args,
                inference_clusters=[],
                research_quality_enabled=False,
            )

            metrics = load_json(os.path.join(query_dir, "metrics.json"))
            self.assertEqual(metrics["research_quality"]["report_score"], 0.42)
            self.assertEqual(
                metrics["per_step"][0]["research_quality"]["finding_score"],
                0.42,
            )
            self.assertEqual(
                metrics["per_step"][0]["research_quality"]["rubric"],
                prior_rubric,
            )
            self.assertEqual(metrics["judge_usage"], prior_metrics["judge_usage"])

            result = load_json(os.path.join(query_dir, "result.json"))
            findings = result.get("findings") or []
            self.assertEqual(findings[0]["passages_cited"], ["P42"])
            self.assertEqual(findings[0]["rubric"]["finding_score"], 0.42)
            self.assertEqual(findings[0]["rubric"]["scores"]["grounded"], 0.9)

            artifact = load_json(os.path.join(query_dir, "iteration_001.json"))
            self.assertEqual(artifact["passages_cited"], ["P42"])
            self.assertEqual(
                artifact["metrics"]["research_quality"]["finding_score"],
                0.42,
            )

    def test_load_passage_descriptions_prefers_query_local(self) -> None:
        from src.utils import load_passage_descriptions_for_metrics

        fixture_passages = os.path.join(
            os.path.dirname(__file__), "fixtures", "opencode_passages.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            query_dir = os.path.join(tmp, "q1")
            os.makedirs(query_dir)
            local_path = os.path.join(query_dir, "passage_descriptions.json")
            shutil.copy(fixture_passages, local_path)

            args = Namespace(
                use_passages=True,
                passage_type="synth",
                passage_descriptions_path=os.path.join(tmp, "missing.json"),
            )
            loaded = load_passage_descriptions_for_metrics(args, query_dir=query_dir)
            self.assertIsNotNone(loaded)
            self.assertIn("P1", loaded)


if __name__ == "__main__":
    unittest.main()
