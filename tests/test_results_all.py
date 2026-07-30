"""Tests for run-level results_all.json helpers."""

import json
import os
import tempfile
import unittest

from src.metrics.aggregate import aggregate_run_metrics
from src.metrics.run_summary_json import format_metrics_summary_for_disk
from src.pipeline import build_results_payload, write_results_all
from src.results_all import (
    METRICS_SUMMARY_FILENAME,
    build_results_all_payload,
    patch_results_all_metrics,
    slim_query_index_entry,
)


class TestResultsAll(unittest.TestCase):
    def _sample_result(self, query_id: str, report_score: float) -> dict:
        return {
            "query_id": query_id,
            "coverage": "low",
            "method": "pipeline",
            "answer": "Final answer.",
            "summary": {
                "research_quality": {"report_score": report_score},
            },
            "metrics_path": "metrics.json",
        }

    def _sample_metrics(self, report_score: float) -> dict:
        return {
            "research_quality": {
                "report_score": report_score,
                "finding_scores_sum": report_score * 2,
                "n_findings_valid": 1,
            },
            "retrieval": {
                "table_gt_in_top_k": 1.0,
                "cumulative": {"table_recall": 0.5},
            },
            "operational": {
                "cumulative": {
                    "sql_success_rate": 1.0,
                    "n_findings": 1,
                }
            },
        }

    def test_slim_query_index_entry(self) -> None:
        entry = slim_query_index_entry(self._sample_result("49", 0.42))
        self.assertEqual(entry["query_id"], "49")
        self.assertEqual(entry["report_score"], 0.42)
        self.assertEqual(entry["result_path"], "49/result.json")

    def test_build_results_all_payload_without_metrics(self) -> None:
        results = [self._sample_result("2", 0.1), self._sample_result("1", 0.2)]
        payload = build_results_all_payload(
            results,
            time_taken=12.5,
            usage={"total_tokens": 1},
        )
        self.assertEqual(payload["n_completed"], 2)
        self.assertEqual(list(payload.keys())[-1], "queries")
        self.assertEqual([q["query_id"] for q in payload["queries"]], ["1", "2"])
        self.assertNotIn("summary", payload)
        self.assertNotIn("n_queries_with_metrics", payload)
        self.assertNotIn("metrics_summary_path", payload)

    def test_build_results_payload_writes_sidecars(self) -> None:
        tmp = tempfile.mkdtemp()
        log_dir = os.path.join(tmp, "run")
        os.makedirs(log_dir)
        qdir = os.path.join(log_dir, "1")
        os.makedirs(qdir)
        result = self._sample_result("1", 0.5)
        with open(os.path.join(qdir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f)
        with open(os.path.join(qdir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self._sample_metrics(0.5), f)

        payload = build_results_payload(
            [result],
            compute_metrics=True,
            run_time_taken=3.0,
            usage={"total_tokens": 10},
            log_dir=log_dir,
        )
        write_results_all(log_dir, payload)

        with open(os.path.join(log_dir, "results_all.json"), encoding="utf-8") as f:
            results_all = json.load(f)
        self.assertNotIn("results", results_all)
        self.assertNotIn("metrics_summary", results_all)
        self.assertEqual(results_all["n_queries_with_metrics"], 1)
        self.assertEqual(results_all["metrics_summary_path"], METRICS_SUMMARY_FILENAME)
        self.assertIn("summary", results_all)
        self.assertEqual(list(results_all.keys())[-1], "queries")
        self.assertEqual(results_all["queries"][0]["query_id"], "1")
        self.assertTrue(os.path.isfile(os.path.join(log_dir, METRICS_SUMMARY_FILENAME)))

        with open(os.path.join(log_dir, METRICS_SUMMARY_FILENAME), encoding="utf-8") as f:
            metrics_summary = json.load(f)
        self.assertEqual(metrics_summary["overall"]["research_quality"]["report_score"], 0.5)

    def test_patch_results_all_metrics(self) -> None:
        tmp = tempfile.mkdtemp()
        log_dir = os.path.join(tmp, "run")
        os.makedirs(log_dir)
        old_result = self._sample_result("1", 0.1)
        payload = build_results_all_payload(
            [old_result],
            time_taken=1.0,
            usage={},
            metrics_summary=aggregate_run_metrics(
                [
                    {
                        "query_id": "1",
                        "coverage": "low",
                        "metrics": self._sample_metrics(0.1),
                    }
                ]
            ),
        )
        with open(os.path.join(log_dir, "results_all.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

        updated_result = self._sample_result("1", 0.9)
        summary = aggregate_run_metrics(
            [
                {
                    "query_id": "1",
                    "coverage": "low",
                    "metrics": self._sample_metrics(0.9),
                }
            ]
        )
        patch_results_all_metrics(log_dir, summary, [updated_result])

        with open(os.path.join(log_dir, "results_all.json"), encoding="utf-8") as f:
            patched = json.load(f)
        self.assertEqual(
            patched["summary"]["research_quality"]["report_score"],
            0.9,
        )
        self.assertEqual(patched["queries"][0]["report_score"], 0.9)

    def test_format_metrics_summary_for_disk(self) -> None:
        summary = aggregate_run_metrics(
            [
                {
                    "query_id": "q1",
                    "coverage": "low",
                    "metrics": self._sample_metrics(0.5),
                }
            ]
        )
        formatted = format_metrics_summary_for_disk(summary)
        self.assertEqual(list(formatted.keys()), ["n_queries", "overall", "per_coverage", "per_query"])
        self.assertIn("research_quality", formatted["overall"])


if __name__ == "__main__":
    unittest.main()
