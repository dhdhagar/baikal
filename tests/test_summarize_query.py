"""Tests for scripts/summarize_query.py."""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

from scripts.summarize_query import summarize_query


def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class TestSummarizeQuery(unittest.TestCase):
    def test_summarize_query_renders_full_iteration_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _save_json(
                os.path.join(tmpdir, "result.json"),
                {
                    "query_id": "41",
                    "user_query": "Example research question?",
                    "method": "pipeline",
                    "coverage": "medium",
                    "answer": "Final synthesized report.",
                    "summary": {
                        "research_quality": {"budget": 1},
                    },
                },
            )
            _save_json(
                os.path.join(tmpdir, "iteration_001.json"),
                {
                    "step": 1,
                    "selected_cluster": {
                        "id": "c1",
                        "description": "cluster one",
                        "table_ids": [["T1", "Table One"]],
                        "passage_ids": [["P1", "Passage One"]],
                    },
                    "sub_question": "How many rows?",
                    "needs_sql": True,
                    "sql": "SELECT 1 AS value;",
                    "execution": {
                        "ok": True,
                        "error": None,
                        "row_count": 1,
                        "rows": [{"value": 1}],
                    },
                    "failed_sql_attempts": [
                        {
                            "attempt": 1,
                            "sql": "SELECT bad;",
                            "ok": False,
                            "row_count": 0,
                            "error": "syntax error",
                            "rows": [],
                        }
                    ],
                    "answer": "There is one row.",
                },
            )

            markdown = summarize_query(tmpdir)

        self.assertIn("## Research question", markdown)
        self.assertIn("Example research question?", markdown)
        self.assertIn("## Final report", markdown)
        self.assertIn("Final synthesized report.", markdown)
        self.assertIn("## Step 1", markdown)
        self.assertIn("### Sub-question", markdown)
        self.assertIn("How many rows?", markdown)
        self.assertIn("### SQL attempts", markdown)
        self.assertIn("SELECT bad;", markdown)
        self.assertIn("syntax error", markdown)
        self.assertIn("### Final SQL", markdown)
        self.assertIn("SELECT 1 AS value;", markdown)
        self.assertIn("### Final SQL response", markdown)
        self.assertIn('"value": 1', markdown)
        self.assertIn("### Answer", markdown)
        self.assertIn("There is one row.", markdown)
        self.assertIn("T1 (Table One)", markdown)
        self.assertIn("P1 (Passage One)", markdown)


if __name__ == "__main__":
    unittest.main()
