"""Integration tests for QueryRunner cluster passage expansion."""

import json
import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from src.query_runner import QueryRunner


def _write_passage_descriptions(path: str, descriptions: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(descriptions, handle, indent=2)


def _minimal_args(passage_descriptions_path: str) -> Namespace:
    return Namespace(
        seed=42,
        silent=True,
        rich_cli=False,
        temperature=0.0,
        use_passages=True,
        passage_descriptions_path=passage_descriptions_path,
        tables_lake_dir="/tmp",
        llm_provider="openai",
        llm_model="gpt-5-mini",
        embedding_provider="local",
        embedding_model="test-model",
        gpu=False,
    )


class TestQueryRunnerExpandCluster(unittest.TestCase):
    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    def test_passage_mode_expands_cluster(
        self,
        mock_get_llm,
        mock_get_embedder,
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()

        descriptions = {
            "P1": {
                "passage_id": "P1",
                "uid": "u1",
                "title": "Overview",
                "text": "Brief venue summary",
            },
            "P2": {
                "passage_id": "P2",
                "uid": "u2",
                "title": "Capacity records",
                "text": "Stadium capacity details",
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            path = tmp.name
        try:
            _write_passage_descriptions(path, descriptions)
            runner = QueryRunner(_minimal_args(path), [], tempfile.mkdtemp())
            runner._current_query_out_dir = tempfile.mkdtemp()
            runner.passage_rank_by_id = {"P2": 3}

            cluster = {
                "cluster_id": "c1",
                "description": "Venues",
                "tables": [],
                "passages": [
                    {
                        "passage_id": "P1",
                        "uid": "u1",
                        "title": "Overview",
                        "text": "Brief venue summary",
                    }
                ],
            }

            with patch(
                "src.query_runner.decide_passage_expansion",
                return_value={
                    "expand": True,
                    "grep_keywords": ["capacity"],
                    "generate_new_subquestions": False,
                },
            ), patch(
                "src.query_runner.grep_passage_ids",
                return_value=["P2"],
            ) as mock_grep:
                meta = runner._maybe_expand_cluster(
                    cluster=cluster,
                    cluster_id="c1",
                    sub_question="What is the stadium capacity?",
                    user_query="Venue research",
                    step=1,
                    expansion_mode="passages",
                )

            mock_grep.assert_called_once()
            self.assertEqual(meta["trigger"], "passages")
            self.assertEqual(len(meta["added_passage_ids"]), 1)
            self.assertEqual(meta["added_passage_ids"][0][0], "P2")
            self.assertEqual(meta["added_passage_ranks"], {"P2": 3})
            self.assertEqual(len(cluster["passages"]), 2)
            self.assertEqual(cluster["passages"][-1]["passage_id"], "P2")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
