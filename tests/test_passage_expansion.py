"""Tests for cluster passage expansion."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.passage_expansion import (
    MAX_GREP_KEYWORDS,
    decide_passage_expansion,
    format_sql_results_block,
    grep_passage_ids,
    passages_from_ids,
)


def _write_passage_descriptions(path: str, descriptions: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(descriptions, handle, indent=2)


class TestFormatSqlResultsBlock(unittest.TestCase):
    def test_ok_execution_includes_rows(self) -> None:
        block = format_sql_results_block(
            'SELECT * FROM "T1"',
            {"ok": True, "row_count": 2, "rows": [{"a": 1}, {"a": 2}]},
            max_rows=10,
        )
        self.assertIn("T1", block)
        self.assertIn("2 rows", block)


class TestGrepPassageIds(unittest.TestCase):
    def setUp(self) -> None:
        self.descriptions = {
            "P1": {
                "passage_id": "P1",
                "title": "Olympic history",
                "text": "Medal counts over time",
            },
            "P2": {
                "passage_id": "P2",
                "title": "Stadium guide",
                "text": "Venue capacity records",
            },
            "P3": {
                "passage_id": "P3",
                "title": "Cooking tips",
                "text": "Pasta recipes",
            },
            "P4": {
                "passage_id": "P4",
                "title": "Medal trivia",
                "text": "Gold silver bronze facts",
            },
        }
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        self.path = self.tmp.name
        self.tmp.close()
        _write_passage_descriptions(self.path, self.descriptions)
        self.rank_by = {"P1": 5, "P2": 2, "P4": 1}

    def tearDown(self) -> None:
        os.unlink(self.path)

    def test_or_semantics_any_keyword(self) -> None:
        ids = grep_passage_ids(
            self.path,
            ["stadium", "cooking"],
            exclude_ids=set(),
            max_results=5,
        )
        self.assertEqual(set(ids), {"P2", "P3"})

    def test_excludes_existing_passages(self) -> None:
        ids = grep_passage_ids(
            self.path,
            ["medal"],
            exclude_ids={"P1"},
            max_results=5,
        )
        self.assertEqual(ids, ["P4"])

    def test_ranks_by_embedding_rank(self) -> None:
        ids = grep_passage_ids(
            self.path,
            ["medal"],
            exclude_ids=set(),
            max_results=5,
            rank_by=self.rank_by,
        )
        self.assertEqual(ids, ["P4", "P1"])

    def test_caps_at_max_results(self) -> None:
        ids = grep_passage_ids(
            self.path,
            ["medal", "stadium", "cooking"],
            exclude_ids=set(),
            max_results=2,
            rank_by=self.rank_by,
        )
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], "P4")

    def test_empty_keywords_returns_empty(self) -> None:
        self.assertEqual(
            grep_passage_ids(self.path, [], exclude_ids=set()),
            [],
        )


class TestPassagesFromIds(unittest.TestCase):
    def test_builds_cluster_shape(self) -> None:
        descriptions = {
            "P9": {
                "uid": "u9",
                "title": "Title",
                "text": "Body",
            }
        }
        passages = passages_from_ids(descriptions, ["P9", "P999"])
        self.assertEqual(len(passages), 1)
        self.assertEqual(
            passages[0],
            {
                "passage_id": "P9",
                "uid": "u9",
                "title": "Title",
                "text": "Body",
            },
        )


class TestDecidePassageExpansion(unittest.TestCase):
    @patch("src.passage_expansion.chat")
    def test_parses_sql_mode_json(self, mock_chat) -> None:
        mock_chat.return_value = """{
            "expand": true,
            "grep_keywords": ["Berlin", "1936"],
            "generate_new_subquestions": true
        }"""
        cluster = {
            "description": "Sports",
            "tables": [],
            "passages": [{"passage_id": "P1", "title": "t", "text": "x"}],
        }
        result = decide_passage_expansion(
            llm=object(),
            cluster=cluster,
            sub_question="Who hosted?",
            temperature=0.0,
            mode="sql",
            sql="SELECT host FROM T1",
            execution_result={"ok": True, "row_count": 1, "rows": [{"host": "Berlin"}]},
        )
        self.assertTrue(result["expand"])
        self.assertEqual(result["grep_keywords"], ["Berlin", "1936"])
        self.assertTrue(result["generate_new_subquestions"])
        mock_chat.assert_called_once()
        self.assertEqual(
            mock_chat.call_args.kwargs.get("feature"),
            "passage_expansion_decide_sql",
        )

    @patch("src.passage_expansion.chat")
    def test_parses_passages_mode_json(self, mock_chat) -> None:
        mock_chat.return_value = """{
            "expand": true,
            "grep_keywords": ["stadium capacity"],
            "generate_new_subquestions": false
        }"""
        cluster = {
            "description": "Venues",
            "tables": [{"table_id": "T1", "title": "t", "columns": [], "description": ""}],
            "passages": [{"passage_id": "P1", "title": "Guide", "text": "Venue overview"}],
        }
        result = decide_passage_expansion(
            llm=object(),
            cluster=cluster,
            sub_question="What is the largest stadium?",
            temperature=0.0,
            mode="passages",
        )
        self.assertTrue(result["expand"])
        self.assertEqual(result["grep_keywords"], ["stadium capacity"])
        mock_chat.assert_called_once()
        self.assertEqual(
            mock_chat.call_args.kwargs.get("feature"),
            "passage_expansion_decide_passages",
        )
        prompt = mock_chat.call_args.args[1]
        self.assertIn("CURRENT PASSAGES", prompt)
        self.assertNotIn("SQL EXECUTION", prompt)
        self.assertNotIn("Passages:\n", prompt)

    @patch("src.passage_expansion.chat")
    def test_caps_keywords_at_max(self, mock_chat) -> None:
        keywords = [f"kw{i}" for i in range(10)]
        mock_chat.return_value = json.dumps(
            {
                "expand": True,
                "grep_keywords": keywords,
                "generate_new_subquestions": False,
            }
        )
        result = decide_passage_expansion(
            llm=object(),
            cluster={"description": "x", "tables": [], "passages": []},
            sub_question="q",
            temperature=0.0,
            mode="sql",
            sql="SELECT 1",
            execution_result={"ok": True, "row_count": 1, "rows": []},
        )
        self.assertEqual(len(result["grep_keywords"]), MAX_GREP_KEYWORDS)

    @patch("src.passage_expansion.chat")
    def test_garbage_json_skips_expansion(self, mock_chat) -> None:
        mock_chat.return_value = "not json at all {{{"
        result = decide_passage_expansion(
            llm=object(),
            cluster={"description": "x", "tables": [], "passages": []},
            sub_question="q",
            temperature=0.0,
            mode="passages",
        )
        self.assertFalse(result["expand"])
        self.assertEqual(result["grep_keywords"], [])


if __name__ == "__main__":
    unittest.main()
