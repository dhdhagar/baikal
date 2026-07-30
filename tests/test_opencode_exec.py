"""Tests for --opencode_exec pipeline integration."""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from src.opencode_exec import (
    build_opencode_subquestion_prompt,
    expansion_grep_keywords,
    opencode_expansion_trigger,
    run_opencode_subquestion,
)
from src.opencode_lake_tool import cmd_grep_passages, init_state, load_state, write_config
from src.opencode_usage import combine_query_usage, merge_usage_summaries
from src.query_runner import QueryRunner
from src.sql_db import materialize_cluster_sqlite


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
LAKE_DIR = os.path.join(FIXTURES, "opencode_lake")
PASSAGES_PATH = os.path.join(FIXTURES, "opencode_passages.json")


class TestOpenCodeExecValidation(unittest.TestCase):
    def test_opencode_exec_rejected_with_opencode_method(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--method",
            "opencode",
            "--opencode_exec",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_opencode_exec_expand_cluster_requires_passages(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--opencode_exec",
            "--expand_cluster",
            "--no-use_passages",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)


class TestOpenCodeSubquestionPrompt(unittest.TestCase):
    def test_prompt_includes_subquestion_and_cluster(self) -> None:
        cluster = {
            "cluster_id": "c1",
            "description": "Stadiums",
            "tables": [{"table_id": "T1", "title": "Venues", "columns": ["name"]}],
            "passages": [{"passage_id": "P1", "title": "Urban", "text": "Density."}],
        }
        prompt = build_opencode_subquestion_prompt(
            user_query="What is the capacity trend?",
            sub_question="Which stadium is largest?",
            cluster=cluster,
            cluster_schema_path="/tmp/schema.txt",
            sqlite_db_path="/tmp/cluster.sqlite",
            query_dir="/tmp/work",
            max_sql_attempts=3,
            use_passages=True,
            passage_descriptions_path="/tmp/passages.json",
            expand_cluster=False,
        )
        self.assertIn("Which stadium is largest?", prompt)
        self.assertIn("Stadiums", prompt)
        self.assertIn("exactly ONE finding", prompt)
        self.assertNotIn("grep-passages", prompt)

    def test_prompt_includes_expansion_after_sql(self) -> None:
        prompt = build_opencode_subquestion_prompt(
            user_query="Q",
            sub_question="SQ",
            cluster={"cluster_id": "c1", "description": "d", "tables": [], "passages": []},
            cluster_schema_path="/tmp/schema.txt",
            sqlite_db_path="/tmp/cluster.sqlite",
            query_dir="/tmp/work",
            max_sql_attempts=3,
            use_passages=True,
            passage_descriptions_path="/tmp/passages.json",
            expand_cluster=True,
        )
        self.assertIn("grep-passages", prompt)
        self.assertIn("inspect SQL results FIRST", prompt)


class TestExpansionHelpers(unittest.TestCase):
    def test_expansion_grep_keywords_from_grep_queries(self) -> None:
        keywords = expansion_grep_keywords(
            {
                "grep_queries": [
                    {"keywords": ["alpha", "beta"], "matched_ids": ["P2"]},
                    {"keywords": ["beta", "gamma"], "matched_ids": []},
                ]
            }
        )
        self.assertEqual(keywords, ["alpha", "beta", "gamma"])

    def test_expansion_grep_keywords_prefers_llm_decision(self) -> None:
        keywords = expansion_grep_keywords(
            {
                "llm_decision": {"grep_keywords": ["from-llm"]},
                "grep_queries": [{"keywords": ["from-opencode"], "matched_ids": []}],
            }
        )
        self.assertEqual(keywords, ["from-llm"])

    def test_opencode_expansion_trigger_after_sql(self) -> None:
        trigger = opencode_expansion_trigger(
            {
                "grep_queries": [{"keywords": ["stadium"], "matched_ids": ["P2"]}],
                "sql_attempts": {"1": [{"ok": True, "row_count": 2}]},
            }
        )
        self.assertEqual(trigger, "opencode_grep_after_sql")

    def test_opencode_expansion_trigger_passages_only(self) -> None:
        trigger = opencode_expansion_trigger(
            {
                "grep_queries": [{"keywords": ["detail"], "matched_ids": ["P2"]}],
                "sql_attempts": {},
            }
        )
        self.assertEqual(trigger, "opencode_grep_passages")


class TestGrepPassagesLakeTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.workdir = os.path.join(self.tmp, "step")
        os.makedirs(self.workdir)
        write_config(
            self.workdir,
            {
                "sqlite_path": os.path.join(self.tmp, "x.sqlite"),
                "passage_descriptions_path": PASSAGES_PATH,
                "budget": 1,
                "max_sql_attempts": 3,
                "allow_passage_grep": True,
                "cluster_passage_ids": ["P1"],
                "passage_rank_by_id": {},
            },
        )
        state = init_state()
        from src.opencode_lake_tool import save_state

        save_state(self.workdir, state)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grep_disabled_without_flag(self) -> None:
        write_config(
            self.workdir,
            {
                "sqlite_path": os.path.join(self.tmp, "x.sqlite"),
                "passage_descriptions_path": PASSAGES_PATH,
                "budget": 1,
                "max_sql_attempts": 3,
                "allow_passage_grep": False,
                "cluster_passage_ids": [],
                "passage_rank_by_id": {},
            },
        )
        with patch("sys.stdout", io.StringIO()) as out:
            rc = cmd_grep_passages(self.workdir, ["urban"])
        self.assertEqual(rc, 1)
        payload = json.loads(out.getvalue())
        self.assertFalse(payload["ok"])

    @patch("src.opencode_lake_tool.grep_passage_ids", return_value=["P99"])
    def test_grep_tracks_retrieved_passages(self, _mock_grep) -> None:
        with patch("sys.stdout", io.StringIO()) as out:
            rc = cmd_grep_passages(self.workdir, ["density"])
        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["matched_ids"], ["P99"])
        state = load_state(self.workdir)
        self.assertIn("P99", state["retrieved_passage_ids"])
        self.assertEqual(state["grep_queries"][0]["keywords"], ["density"])


class TestMaterializeClusterSqlite(unittest.TestCase):
    def test_cluster_sqlite_contains_only_requested_tables(self) -> None:
        tmp = tempfile.mkdtemp()
        try:
            dest = os.path.join(tmp, "cluster.sqlite")
            materialize_cluster_sqlite(LAKE_DIR, ["T1"], dest)
            import sqlite3

            conn = sqlite3.connect(dest)
            try:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                conn.close()
            self.assertEqual(tables, {"T1"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestQueryRunnerOpenCodeExec(unittest.TestCase):
    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    @patch("src.query_runner.run_opencode_subquestion")
    def test_opencode_exec_skips_langgraph_and_expand_llm(
        self,
        mock_run_subq,
        mock_get_llm,
        mock_get_embedder,
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()

        mock_run_subq.return_value = {
            "answer": "Alpha Stadium is largest [T1].",
            "needs_sql": True,
            "sql": 'SELECT name FROM "T1"',
            "execution": {"ok": True, "row_count": 1, "rows": [["Alpha"]], "error": None},
            "failed_sql_attempts": [],
            "opencode_meta": {
                "status": "completed",
                "returncode": 0,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.01,
                    "time_taken": 0.0,
                    "n_calls": 1,
                },
            },
            "lake_state": {"retrieved_passage_ids": [], "grep_queries": []},
        }

        args = Namespace(
            seed=42,
            silent=True,
            rich_cli=False,
            temperature=0.0,
            opencode_exec=True,
            expand_cluster=False,
            use_passages=False,
            passage_descriptions_path=None,
            passage_type="synth",
            tables_lake_dir=LAKE_DIR,
            llm_provider="openai",
            llm_model="gpt-5-mini",
            embedding_provider="local",
            embedding_model="test-model",
            gpu=False,
            max_sql_attempts=3,
            cluster_selection_method="random",
            subquestion_selection_method="random",
            k_subquestions=3,
            use_clustering=True,
        )
        runner = QueryRunner(args, [], tempfile.mkdtemp())
        runner._current_query_out_dir = tempfile.mkdtemp()
        runner.candidate_clusters = [
            {
                "cluster_id": "c1",
                "description": "Venues",
                "tables": [{"table_id": "T1", "title": "Stadiums", "columns": ["name"]}],
                "passages": [],
            }
        ]
        runner.pending_subquestions = {"c1": ["Which stadium is largest?"]}
        runner.table_rank_by_id = {"T1": 1}
        runner.passage_rank_by_id = {}

        with patch.object(runner, "_maybe_expand_cluster") as mock_expand:
            iteration = runner._run_one_iteration("What is the largest stadium?", 1)

        mock_expand.assert_not_called()
        mock_run_subq.assert_called_once()
        self.assertIsNotNone(iteration)
        assert iteration is not None
        self.assertEqual(iteration["answer"], "Alpha Stadium is largest [T1].")
        self.assertIn("opencode", iteration)
        self.assertEqual(iteration["opencode_usage"]["total_tokens"], 15)

    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    @patch("src.query_runner.run_opencode_subquestion")
    def test_opencode_commit_failure_still_writes_iteration(
        self,
        mock_run_subq,
        mock_get_llm,
        mock_get_embedder,
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()
        mock_run_subq.return_value = {
            "answer": "",
            "needs_sql": False,
            "sql": None,
            "execution": None,
            "failed_sql_attempts": [],
            "commit_failed": True,
            "failure_reason": "no_commit",
            "opencode_meta": {
                "status": "completed",
                "returncode": 0,
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                    "cost_usd": 0.005,
                    "time_taken": 0.0,
                    "n_calls": 1,
                },
            },
            "lake_state": {"findings": [], "retrieved_passage_ids": []},
        }

        args = Namespace(
            seed=42,
            silent=True,
            rich_cli=False,
            temperature=0.0,
            opencode_exec=True,
            expand_cluster=False,
            use_passages=False,
            passage_descriptions_path=None,
            passage_type="synth",
            tables_lake_dir=LAKE_DIR,
            llm_provider="openai",
            llm_model="gpt-5-mini",
            embedding_provider="local",
            embedding_model="test-model",
            gpu=False,
            max_sql_attempts=3,
            cluster_selection_method="random",
            subquestion_selection_method="random",
            k_subquestions=3,
            use_clustering=True,
        )
        runner = QueryRunner(args, [], tempfile.mkdtemp())
        runner._current_query_out_dir = tempfile.mkdtemp()
        runner.candidate_clusters = [
            {
                "cluster_id": "c1",
                "description": "Venues",
                "tables": [{"table_id": "T1", "title": "Stadiums", "columns": ["name"]}],
                "passages": [],
            }
        ]
        runner.pending_subquestions = {"c1": ["Which stadium is largest?"]}
        runner.table_rank_by_id = {"T1": 1}
        runner.passage_rank_by_id = {}

        iteration = runner._run_one_iteration("What is the largest stadium?", 1)

        self.assertIsNotNone(iteration)
        assert iteration is not None
        self.assertEqual(iteration["answer"], "")
        self.assertEqual(iteration["sub_question"], "Which stadium is largest?")
        self.assertEqual(iteration["opencode_failure"], {"reason": "no_commit"})
        self.assertIn("opencode", iteration)
        self.assertEqual(runner.pending_subquestions["c1"], [])
        self.assertEqual(runner.answered_subquestions, ["Which stadium is largest?"])

    @patch("src.query_runner.generate_subquestions")
    @patch("src.query_runner.filter_semantically_duplicate_subquestions")
    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    def test_opencode_expansion_queues_subquestions(
        self,
        mock_get_llm,
        mock_get_embedder,
        mock_filter,
        mock_generate,
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()
        mock_generate.return_value = MagicMock(
            subquestions=["Follow-up about extra passage?"]
        )
        mock_filter.return_value = ["Follow-up about extra passage?"]

        descriptions = {
            "P1": {
                "passage_id": "P1",
                "uid": "u1",
                "title": "Overview",
                "text": "Brief",
            },
            "P2": {
                "passage_id": "P2",
                "uid": "u2",
                "title": "Extra",
                "text": "More detail",
            },
        }
        runner = QueryRunner(
            Namespace(
                seed=42,
                silent=True,
                temperature=0.0,
                tables_lake_dir="/tmp",
                llm_provider="openai",
                llm_model="gpt-5-mini",
                embedding_provider="local",
                embedding_model="test-model",
                gpu=False,
            ),
            [],
            tempfile.mkdtemp(),
        )
        runner._current_query_out_dir = tempfile.mkdtemp()
        runner.passage_rank_by_id = {"P2": 2}
        runner._passage_descriptions_cache = descriptions
        cluster = {
            "cluster_id": "c1",
            "description": "Venues",
            "tables": [],
            "passages": [
                {
                    "passage_id": "P1",
                    "uid": "u1",
                    "title": "Overview",
                    "text": "Brief",
                }
            ],
        }
        meta = runner._sync_opencode_cluster_expansion(
            cluster=cluster,
            cluster_id="c1",
            sub_question="What details exist?",
            user_query="Main research question?",
            step=1,
            lake_state={
                "retrieved_passage_ids": ["P2"],
                "grep_queries": [{"keywords": ["detail"], "matched_ids": ["P2"]}],
            },
        )
        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta["generated_subquestions"], ["Follow-up about extra passage?"])
        self.assertEqual(
            runner.pending_subquestions["c1"],
            ["Follow-up about extra passage?"],
        )

    @patch("src.query_runner.generate_subquestions")
    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    def test_sync_opencode_cluster_expansion(
        self, mock_get_llm, mock_get_embedder, mock_generate
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()
        mock_generate.return_value = MagicMock(subquestions=[])

        descriptions = {
            "P1": {
                "passage_id": "P1",
                "uid": "u1",
                "title": "Overview",
                "text": "Brief",
            },
            "P2": {
                "passage_id": "P2",
                "uid": "u2",
                "title": "Extra",
                "text": "More detail",
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            path = tmp.name
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(descriptions, handle)
            args = Namespace(
                seed=42,
                silent=True,
                temperature=0.0,
                passage_descriptions_path=path,
                tables_lake_dir="/tmp",
                llm_provider="openai",
                llm_model="gpt-5-mini",
                embedding_provider="local",
                embedding_model="test-model",
                gpu=False,
            )
            runner = QueryRunner(args, [], tempfile.mkdtemp())
            runner._current_query_out_dir = tempfile.mkdtemp()
            runner.passage_rank_by_id = {"P2": 2}
            runner._passage_descriptions_cache = descriptions

            cluster = {
                "cluster_id": "c1",
                "description": "Venues",
                "tables": [],
                "passages": [
                    {
                        "passage_id": "P1",
                        "uid": "u1",
                        "title": "Overview",
                        "text": "Brief",
                    }
                ],
            }
            meta = runner._sync_opencode_cluster_expansion(
                cluster=cluster,
                cluster_id="c1",
                sub_question="What details exist?",
                user_query="Research question?",
                step=1,
                lake_state={
                    "retrieved_passage_ids": ["P2"],
                    "grep_queries": [{"keywords": ["detail"], "matched_ids": ["P2"]}],
                },
            )
            self.assertIsNotNone(meta)
            assert meta is not None
            self.assertEqual(meta["trigger"], "opencode_grep_passages")
            self.assertEqual(len(meta["added_passage_ids"]), 1)
            self.assertEqual(len(cluster["passages"]), 2)

            artifact_path = os.path.join(
                runner._current_query_out_dir, "cluster_expanded_passages.json"
            )
            with open(artifact_path, encoding="utf-8") as handle:
                artifact = json.load(handle)
            step_record = artifact["c1"]["steps"][0]
            self.assertEqual(step_record["grep_keywords"], ["detail"])
            self.assertEqual(step_record["grep_queries"][0]["keywords"], ["detail"])
        finally:
            os.unlink(path)

    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    def test_sync_opencode_skips_when_no_new_passages(
        self, mock_get_llm, mock_get_embedder
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()
        runner = QueryRunner(
            Namespace(
                seed=42,
                silent=True,
                temperature=0.0,
                tables_lake_dir="/tmp",
                llm_provider="openai",
                llm_model="gpt-5-mini",
                embedding_provider="local",
                embedding_model="test-model",
                gpu=False,
            ),
            [],
            tempfile.mkdtemp(),
        )
        cluster = {
            "cluster_id": "c1",
            "description": "Venues",
            "tables": [],
            "passages": [{"passage_id": "P1", "title": "T", "text": "x"}],
        }
        meta = runner._sync_opencode_cluster_expansion(
            cluster=cluster,
            cluster_id="c1",
            sub_question="Q",
            user_query="Research?",
            step=1,
            lake_state={"retrieved_passage_ids": [], "grep_queries": []},
        )
        self.assertIsNone(meta)

    def test_query_usage_merges_opencode_steps(self) -> None:
        pipeline = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.1,
            "time_taken": 1.0,
            "n_calls": 3,
        }
        opencode_total = merge_usage_summaries(
            [
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.01,
                    "time_taken": 0.5,
                    "n_calls": 1,
                }
            ]
        ).to_dict()
        combined = combine_query_usage(pipeline, opencode_total)
        self.assertIn("opencode", combined)
        self.assertEqual(combined["total"]["total_tokens"], 165)


if __name__ == "__main__":
    unittest.main()
