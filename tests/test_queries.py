"""Tests for query loading."""

import json
import tempfile
import unittest

from src.queries import _resolve_query_id, dedupe_ground_truth, load_queries_from_file


class TestLoadQueries(unittest.TestCase):
    def test_dedupes_ground_truth_table_and_passage_ids(self):
        payload = [
            {
                "query_id": "q1",
                "query_text": "example query",
                "ground_truth": {
                    "n_table": 3,
                    "table": ["T1", "T2", "T1"],
                    "n_text": 4,
                    "text": ["P1", "P2", "P1", "P3"],
                },
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(payload, f)
            f.flush()
            records = load_queries_from_file(f.name)

        gt = records[0]["ground_truth"]
        self.assertEqual(gt["table"], ["T1", "T2"])
        self.assertEqual(gt["text"], ["P1", "P2", "P3"])
        self.assertEqual(gt["n_table"], 2)
        self.assertEqual(gt["n_text"], 3)

    def test_dedupe_ground_truth_updates_counts(self):
        gt = dedupe_ground_truth(
            {
                "n_table": 2,
                "table": ["T1", "T1"],
                "n_text": 3,
                "text": ["P1", "P2", "P1"],
            }
        )
        self.assertEqual(gt["table"], ["T1"])
        self.assertEqual(gt["text"], ["P1", "P2"])
        self.assertEqual(gt["n_table"], 1)
        self.assertEqual(gt["n_text"], 2)

    def test_preserves_ground_truth_without_id_lists(self):
        payload = [
            {
                "query_id": "q1",
                "query_text": "example query",
                "ground_truth": {"answer": "42"},
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(payload, f)
            f.flush()
            records = load_queries_from_file(f.name)

        self.assertEqual(records[0]["ground_truth"], {"answer": "42"})

    def test_integer_zero_query_id_is_preserved(self):
        payload = [
            {
                "query_id": 0,
                "query_text": "How have TORM's return on invested capital changed?",
            },
            {"query_id": 1, "query_text": "Second query"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(payload, f)
            f.flush()
            records = load_queries_from_file(f.name)

        self.assertEqual(records[0]["query_id"], "0")
        self.assertEqual(records[1]["query_id"], "1")

    def test_resolve_query_id_falls_back_to_slugify(self):
        text = "Some long analytical question?"
        self.assertEqual(
            _resolve_query_id({}, text),
            "some_long_analytical_question",
        )

    def test_resolve_query_id_uses_qid_when_query_id_missing(self):
        self.assertEqual(
            _resolve_query_id({"qid": 0}, "ignored"),
            "0",
        )


if __name__ == "__main__":
    unittest.main()
