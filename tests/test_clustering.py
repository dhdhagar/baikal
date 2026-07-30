"""Tests for inference cluster construction."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.inference_clusters import build_topk_inference_cluster
from src.subquestions import generate_subquestions


class TestBuildTopkInferenceCluster(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tables_lake_dir = os.path.join(self.tmp, "lake")
        os.makedirs(self.tables_lake_dir)
        self.schema_path = os.path.join(self.tables_lake_dir, "schema_descriptions.json")
        with open(self.schema_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "T1": {
                        "title": "Stadiums",
                        "domain": "sports",
                        "columns": ["name", "capacity"],
                        "description": "Stadium records",
                    },
                    "T2": {
                        "title": "Teams",
                        "domain": "sports",
                        "columns": ["team"],
                        "description": "Team records",
                    },
                },
                f,
            )
        self.passage_descriptions_path = os.path.join(
            self.tmp, "passage_descriptions.json"
        )
        with open(self.passage_descriptions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "P1": {"uid": "u1", "title": "Alpha", "text": "First passage"},
                    "P9": {"uid": "u9", "title": "Beta", "text": "Ninth passage"},
                },
                f,
            )

    def test_builds_single_cluster_with_topk_items(self):
        clusters = build_topk_inference_cluster(
            ["T2", "T99", "T1"],
            ["P9", "P1", "P99"],
            tables_lake_dir=self.tables_lake_dir,
            passage_descriptions_path=self.passage_descriptions_path,
        )

        self.assertEqual(len(clusters), 1)
        cluster = clusters[0]
        self.assertEqual(cluster["cluster_id"], "topk")
        self.assertEqual(
            [t["table_id"] for t in cluster["tables"]],
            ["T2", "T1"],
        )
        self.assertEqual(cluster["tables"][0]["title"], "Teams")
        self.assertEqual(
            [p["passage_id"] for p in cluster["passages"]],
            ["P9", "P1"],
        )
        self.assertEqual(cluster["passages"][0]["text"], "Ninth passage")

    def test_tables_only_when_no_passage_path(self):
        clusters = build_topk_inference_cluster(
            ["T1"],
            ["P1"],
            tables_lake_dir=self.tables_lake_dir,
            passage_descriptions_path=None,
        )

        self.assertEqual(len(clusters[0]["tables"]), 1)
        self.assertEqual(clusters[0]["passages"], [])


class TestGenerateSubquestionsAllowEmpty(unittest.TestCase):
    @patch("src.subquestions.chat")
    def test_omits_empty_list_rule_when_not_allowed(self, mock_chat):
        mock_chat.return_value = '{"subquestions": ["How many stadiums?"]}'
        cluster = {
            "description": "test",
            "tables": [{"table_id": "T1", "title": "Stadiums", "columns": [], "description": ""}],
            "passages": [],
        }

        generate_subquestions(
            llm=object(),
            user_query="How many stadiums exist?",
            cluster=cluster,
            k=3,
            temperature=0.0,
            allow_empty=False,
        )

        prompt = mock_chat.call_args[0][1]
        self.assertNotIn("return an empty list", prompt)

    @patch("src.subquestions.chat")
    def test_includes_empty_list_rule_by_default(self, mock_chat):
        mock_chat.return_value = '{"subquestions": ["How many stadiums?"]}'
        cluster = {
            "description": "test",
            "tables": [{"table_id": "T1", "title": "Stadiums", "columns": [], "description": ""}],
            "passages": [],
        }

        generate_subquestions(
            llm=object(),
            user_query="How many stadiums exist?",
            cluster=cluster,
            k=3,
            temperature=0.0,
        )

        prompt = mock_chat.call_args[0][1]
        self.assertIn("return an empty list", prompt)


if __name__ == "__main__":
    unittest.main()
