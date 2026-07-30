"""Tests for shared summary block helpers."""

import unittest

from src.metrics.summary_blocks import (
    format_aggregate_headlines_for_disk,
    slim_retrieval_block,
)


class TestSummaryBlocks(unittest.TestCase):
    def test_format_aggregate_headlines_drops_null_retrieval_keys(self) -> None:
        formatted = format_aggregate_headlines_for_disk(
            {
                "research_quality": {"report_score": 0.5, "budget": 10},
                "retrieval": {
                    "table_recall": 0.8,
                    "table_precision": None,
                    "passage_recall": None,
                },
            }
        )
        self.assertEqual(formatted["research_quality"]["report_score"], 0.5)
        self.assertNotIn("budget", formatted["research_quality"])
        self.assertEqual(formatted["retrieval"], {"table_recall": 0.8})

    def test_slim_retrieval_block(self) -> None:
        slim = slim_retrieval_block(
            {"table_recall": 0.5, "passage_recall": None}
        )
        self.assertEqual(slim, {"table_recall": 0.5})


if __name__ == "__main__":
    unittest.main()
