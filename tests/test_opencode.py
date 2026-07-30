"""Tests for OpenCode baseline lake tool and result adapter."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from src.cli_ui import init_ui
from src.opencode import (
    INITIAL_RETRIEVAL_FILENAME,
    _LakeUISeen,
    _apply_lake_state_to_ui,
    _budget_remaining,
    _build_initial_clusters_payload,
    _build_initial_retrieval_payload,
    _ensure_opencode_credentials,
    _load_resume_workspace_context,
    _opencode_subprocess_env,
    load_opencode_inference_clusters,
    _remove_materialized_sqlite,
    _setup_query_workspace,
    _watch_opencode_lake_state,
    build_continuation_prompt,
    build_opencode_prompt,
    finalize_query_from_workspace,
    findings_to_iterations,
    invoke_opencode,
    partial_workspace_status,
    run_opencode_job,
    run_opencode_query,
    setup_lake_workspace,
    write_iteration_artifacts,
)
from src.result_json import INITIAL_CLUSTERS_FILENAME
from src.submitit_runner import worker_args
from src.utils import (
    passage_descriptions_filename,
    query_passage_descriptions_path,
    resolve_passage_descriptions_source,
    save_json,
)
from src.opencode_lake_tool import cmd_commit, cmd_passage, cmd_sql, load_state, write_config
from src.sql_db import materialize_lake_sqlite


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
LAKE_DIR = os.path.join(FIXTURES, "opencode_lake")
PASSAGES_PATH = os.path.join(FIXTURES, "opencode_passages.json")
OPENCODE_JSON_SESSION_LINE = (
    '{"type":"text","timestamp":1,"sessionID":"ses_test123","part":{}}\n'
)
QUERY_RECORD = {
    "query_id": "q1",
    "query_text": "Test question?",
    "coverage": "low",
}


def _finalize_test_args(**overrides) -> Namespace:
    defaults = {
        "budget": 2,
        "compute_metrics": False,
        "temperature": 1.0,
        "llm_provider": "openai",
        "llm_model": "gpt-5-mini",
        "tables_lake_dir": LAKE_DIR,
        "use_passages": False,
        "passage_type": "synth",
        "uid_to_table_id_path": "",
        "uid_to_passage_id_path": "",
        "k_relevant_tables": 10,
        "k_relevant_passages": 10,
        "compute_embed_diversity": False,
        "embedding_provider": "local",
        "embedding_model": "test",
        "gpu": False,
        "judge_models": "gpt-5-mini",
        "no_llm_judge": True,
        "opencode_skip_retrieval": True,
        "opencode_skip_clustering": True,
        "use_clustering": True,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _run_tool(workdir: str, *args: str) -> dict:
    cmd = [sys.executable, "-m", "src.opencode_lake_tool", "--workdir", workdir, *args]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    payload = json.loads(completed.stdout or "{}")
    payload["_returncode"] = completed.returncode
    payload["_stderr"] = completed.stderr
    return payload


class TestOpenCodeLakeTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.workdir = os.path.join(self.tmp, "query_q1")
        os.makedirs(self.workdir)
        self.sqlite_path = materialize_lake_sqlite(
            LAKE_DIR,
            os.path.join(self.tmp, "lake.sqlite"),
        )
        write_config(
            self.workdir,
            {
                "sqlite_path": self.sqlite_path,
                "passage_descriptions_path": PASSAGES_PATH,
                "budget": 2,
                "max_sql_attempts": 2,
            },
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sql_and_commit_enforce_budget(self) -> None:
        first = _run_tool(
            self.workdir,
            "sql",
            'SELECT name, capacity FROM "T1" ORDER BY capacity DESC',
        )
        self.assertTrue(first["ok"])
        self.assertEqual(first["execution"]["row_count"], 2)

        commit1 = _run_tool(
            self.workdir,
            "commit",
            "--sub-question",
            "Which stadium is largest?",
            "--answer",
            "Alpha Stadium has the highest capacity.",
        )
        self.assertTrue(commit1["ok"])
        self.assertEqual(commit1["finding"]["step"], 1)

        fail_sql = _run_tool(self.workdir, "sql", 'SELECT bad_col FROM "T1"')
        self.assertTrue(fail_sql["ok"])
        self.assertFalse(fail_sql["execution"]["ok"])

        retry = _run_tool(
            self.workdir,
            "sql",
            'SELECT name FROM "T1" LIMIT 1',
        )
        self.assertTrue(retry["ok"])

        commit2 = _run_tool(
            self.workdir,
            "commit",
            "--sub-question",
            "Name one stadium",
            "--answer",
            "Alpha Stadium appears in the data.",
        )
        self.assertTrue(commit2["ok"])

        blocked = _run_tool(
            self.workdir,
            "sql",
            'SELECT name FROM "T1"',
        )
        self.assertFalse(blocked["ok"])
        self.assertIn("Budget exhausted", blocked["error"])

    def test_max_sql_attempts_per_step(self) -> None:
        for _ in range(2):
            out = _run_tool(self.workdir, "sql", 'SELECT bad_col FROM "T1"')
            self.assertTrue(out["ok"])
            self.assertFalse(out["execution"]["ok"])

        blocked = _run_tool(self.workdir, "sql", 'SELECT name FROM "T1"')
        self.assertFalse(blocked["ok"])
        self.assertIn("Max SQL attempts", blocked["error"])

    def test_passage_lookup(self) -> None:
        out = _run_tool(self.workdir, "passage", "P1")
        self.assertTrue(out["ok"])
        self.assertEqual(out["passage_id"], "P1")
        self.assertIn("Urban areas", out["passage"]["text"])

    def test_passage_only_commit(self) -> None:
        out = _run_tool(
            self.workdir,
            "commit",
            "--sub-question",
            "Road density?",
            "--answer",
            "Urban areas are denser [P1].",
            "--no-sql",
        )
        self.assertTrue(out["ok"])
        self.assertFalse(out["finding"]["needs_sql"])
        self.assertIsNone(out["finding"]["sql"])


class TestOpenCodeAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.query_dir = os.path.join(self.tmp, "q1")
        os.makedirs(self.query_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_findings_to_iterations_and_artifacts(self) -> None:
        findings = [
            {
                "step": 1,
                "sub_question": "Q1",
                "answer": "A1",
                "needs_sql": True,
                "sql": "SELECT 1",
                "execution": {"ok": True, "row_count": 1, "rows": [], "error": None},
                "failed_sql_attempts": None,
            }
        ]
        iterations = findings_to_iterations(findings)
        recorded = write_iteration_artifacts(self.query_dir, iterations, budget=3)
        self.assertEqual(len(recorded), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.query_dir, "iteration_001.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.query_dir, "iteration_002.json")))
        with open(
            os.path.join(self.query_dir, "iteration_002.json"), encoding="utf-8"
        ) as f:
            skipped = json.load(f)
        self.assertTrue(skipped["skipped"])

    @patch("src.opencode.get_llm_client")
    @patch("src.opencode.generate_final_report")
    def test_finalize_query_from_workspace(self, mock_report, mock_llm) -> None:
        mock_report.return_value = "Synthetic report"
        sqlite_path = materialize_lake_sqlite(
            LAKE_DIR,
            os.path.join(self.tmp, "lake.sqlite"),
        )
        write_config(
            self.query_dir,
            {
                "sqlite_path": sqlite_path,
                "passage_descriptions_path": None,
                "budget": 2,
                "max_sql_attempts": 3,
            },
        )
        sql_out = cmd_sql(self.query_dir, 'SELECT name FROM "T1"')
        self.assertEqual(sql_out, 0)
        cmd_commit(
            self.query_dir,
            sub_question="Names?",
            answer="Alpha Stadium",
            sql=None,
            needs_sql=True,
        )
        with open(
            os.path.join(self.query_dir, "agent_output.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({"answer": "Agent-authored report"}, f)

        result = finalize_query_from_workspace(
            _finalize_test_args(),
            QUERY_RECORD,
            self.query_dir,
            topk_table_ids=["T1"],
            topk_passage_ids=[],
            inference_clusters=[],
        )
        self.assertEqual(result["query_id"], "q1")
        self.assertEqual(result["method"], "opencode")
        self.assertEqual(result["answer"], "Synthetic report")
        self.assertEqual(len(result["findings"]), 1)
        mock_report.assert_called_once()
        self.assertTrue(os.path.isfile(os.path.join(self.query_dir, "result.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.query_dir, "topk.json")))

    @patch("src.opencode.get_llm_client")
    @patch("src.opencode.generate_final_report")
    def test_finalize_stores_opencode_usage_only_once(self, mock_report, mock_llm) -> None:
        mock_report.return_value = "Synthetic report"
        result = finalize_query_from_workspace(
            _finalize_test_args(),
            QUERY_RECORD,
            self.query_dir,
            topk_table_ids=["T1"],
            topk_passage_ids=[],
            inference_clusters=[],
            opencode_meta={
                "session_id": "ses_test123",
                "status": "completed",
                "usage": {
                    "session_id": "ses_test123",
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost_usd": 0.2,
                    "time_taken": 0.0,
                    "n_calls": 1,
                },
            },
        )
        usage = result["summary"]["usage"]
        self.assertIn("opencode", usage)
        self.assertNotIn("usage", result.get("opencode") or {})
        self.assertEqual(usage["opencode"]["session_id"], "ses_test123")


class TestOpenCodePrompt(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.query_dir = os.path.join(self.tmp, "q1")
        os.makedirs(self.query_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_prompt_mentions_budget_and_passages(self) -> None:
        passage_path = query_passage_descriptions_path(self.query_dir, "synth")
        prompt = build_opencode_prompt(
            query_id="q1",
            user_query="What is X?",
            schema_descriptions_path="/tmp/schema.json",
            sqlite_db_path="/tmp/lake.sqlite",
            query_dir=self.query_dir,
            budget=5,
            max_sql_attempts=3,
            use_passages=True,
            passage_descriptions_path=passage_path,
        )
        self.assertIn("at most 5 findings", prompt)
        self.assertIn("up to 3 SQL attempts", prompt)
        self.assertIn("Citation rules (required in every --answer)", prompt)
        self.assertIn("[T809]", prompt)
        self.assertIn("[P42]", prompt)
        self.assertIn(passage_path, prompt)
        self.assertIn("Passage descriptions:", prompt)
        self.assertNotIn("agent_output.json", prompt)
        self.assertIn("opencode_lake_tool", prompt)
        self.assertIn("PYTHONPATH=", prompt)
        self.assertIn("fully unattended", prompt)
        self.assertIn("Never ask questions", prompt)

    def test_prompt_includes_retrieval_when_provided(self) -> None:
        retrieval_path = os.path.join(self.query_dir, INITIAL_RETRIEVAL_FILENAME)
        prompt = build_opencode_prompt(
            query_id="q1",
            user_query="What is X?",
            schema_descriptions_path="/tmp/schema.json",
            sqlite_db_path="/tmp/lake.sqlite",
            query_dir=self.query_dir,
            budget=5,
            max_sql_attempts=3,
            use_passages=False,
            passage_descriptions_path=None,
            initial_retrieval_path=retrieval_path,
        )
        self.assertIn(retrieval_path, prompt)
        self.assertIn("Initial retrieval candidates:", prompt)
        self.assertIn("table ids and titles", prompt)
        self.assertNotIn("passage metadata", prompt)

    def test_prompt_retrieval_mentions_passages_when_enabled(self) -> None:
        retrieval_path = os.path.join(self.query_dir, INITIAL_RETRIEVAL_FILENAME)
        prompt = build_opencode_prompt(
            query_id="q1",
            user_query="What is X?",
            schema_descriptions_path="/tmp/schema.json",
            sqlite_db_path="/tmp/lake.sqlite",
            query_dir=self.query_dir,
            budget=5,
            max_sql_attempts=3,
            use_passages=True,
            passage_descriptions_path="/tmp/passages.json",
            initial_retrieval_path=retrieval_path,
        )
        self.assertIn("table/passage ids and titles", prompt)
        self.assertIn("passage metadata", prompt)

    def test_prompt_includes_clusters_when_provided(self) -> None:
        clusters_path = os.path.join(self.query_dir, INITIAL_CLUSTERS_FILENAME)
        prompt = build_opencode_prompt(
            query_id="q1",
            user_query="What is X?",
            schema_descriptions_path="/tmp/schema.json",
            sqlite_db_path="/tmp/lake.sqlite",
            query_dir=self.query_dir,
            budget=5,
            max_sql_attempts=3,
            use_passages=False,
            passage_descriptions_path=None,
            initial_clusters_path=clusters_path,
        )
        self.assertIn(clusters_path, prompt)
        self.assertIn("Initial inference clusters:", prompt)
        self.assertIn("cluster_id", prompt)
        self.assertIn("and tables.", prompt)
        self.assertNotIn("and passages.", prompt)

    def test_prompt_clusters_mentions_passages_when_enabled(self) -> None:
        clusters_path = os.path.join(self.query_dir, INITIAL_CLUSTERS_FILENAME)
        prompt = build_opencode_prompt(
            query_id="q1",
            user_query="What is X?",
            schema_descriptions_path="/tmp/schema.json",
            sqlite_db_path="/tmp/lake.sqlite",
            query_dir=self.query_dir,
            budget=5,
            max_sql_attempts=3,
            use_passages=True,
            passage_descriptions_path="/tmp/passages.json",
            initial_clusters_path=clusters_path,
        )
        self.assertIn("table/passage clusters", prompt)
        self.assertIn("tables, and passages.", prompt)

    def test_prompt_includes_finding_quality_guardrails(self) -> None:
        prompt = build_opencode_prompt(
            query_id="q1",
            user_query="What is X?",
            schema_descriptions_path="/tmp/schema.json",
            sqlite_db_path="/tmp/lake.sqlite",
            query_dir=self.query_dir,
            budget=5,
            max_sql_attempts=3,
            use_passages=False,
            passage_descriptions_path=None,
        )
        self.assertIn("Finding quality rules (required):", prompt)
        self.assertIn("SELECT COUNT(*)", prompt)
        self.assertIn("sqlite_master", prompt)
        self.assertIn("substantive analysis", prompt)
        self.assertNotIn(
            "Start from the initial retrieval candidates above",
            prompt,
        )

    def test_prompt_retrieval_guardrail_when_candidates_provided(self) -> None:
        retrieval_path = os.path.join(self.query_dir, INITIAL_RETRIEVAL_FILENAME)
        prompt = build_opencode_prompt(
            query_id="q1",
            user_query="What is X?",
            schema_descriptions_path="/tmp/schema.json",
            sqlite_db_path="/tmp/lake.sqlite",
            query_dir=self.query_dir,
            budget=5,
            max_sql_attempts=3,
            use_passages=False,
            passage_descriptions_path=None,
            initial_retrieval_path=retrieval_path,
        )
        self.assertIn(
            "Start from the initial retrieval candidates above",
            prompt,
        )

    def test_setup_query_workspace_materializes_passage_descriptions(self) -> None:
        for passage_type, explicit_path in (("synth", True), ("raw", False)):
            with self.subTest(passage_type=passage_type):
                query_dir = os.path.join(self.tmp, f"q_{passage_type}")
                os.makedirs(query_dir, exist_ok=True)
                data_dir = os.path.join(self.tmp, f"data_{passage_type}")
                os.makedirs(data_dir, exist_ok=True)
                source = os.path.join(self.tmp, passage_descriptions_filename(passage_type))
                shutil.copy(PASSAGES_PATH, source)

                if explicit_path:
                    args = Namespace(
                        use_passages=True,
                        passage_descriptions_path=source,
                        passage_type=passage_type,
                        budget=2,
                        max_sql_attempts=3,
                    )
                else:
                    resolved = os.path.join(
                        data_dir, passage_descriptions_filename(passage_type)
                    )
                    self.assertEqual(
                        resolve_passage_descriptions_source(
                            Namespace(
                                use_passages=True,
                                passage_descriptions_path=None,
                                passage_type=passage_type,
                                data_dir=data_dir,
                            )
                        ),
                        resolved,
                    )
                    shutil.copy(source, resolved)
                    args = Namespace(
                        use_passages=True,
                        passage_descriptions_path=None,
                        passage_type=passage_type,
                        data_dir=data_dir,
                        budget=2,
                        max_sql_attempts=3,
                    )

                _setup_query_workspace(args, query_dir, "/tmp/lake.sqlite")
                dest = query_passage_descriptions_path(query_dir, passage_type)
                self.assertTrue(os.path.lexists(dest))
                with open(dest, encoding="utf-8") as f:
                    data = json.load(f)
                self.assertIn("P1", data)


class TestOpenCodeInferenceClusterLoading(unittest.TestCase):
    def test_load_clusters_when_artifacts_enabled_and_use_clustering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clusters_path = os.path.join(tmp, "clusters.json")
            with open(clusters_path, "w", encoding="utf-8") as f:
                json.dump([{"cluster_id": "c1"}], f)
            args = Namespace(
                opencode_skip_clustering=False,
                use_clustering=True,
                inference_clusters_path=clusters_path,
            )
            loaded = load_opencode_inference_clusters(
                args,
                inference_clusters_path=clusters_path,
            )
            self.assertEqual(loaded, [{"cluster_id": "c1"}])

    def test_skip_load_when_artifacts_enabled_without_use_clustering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clusters_path = os.path.join(tmp, "clusters.json")
            with open(clusters_path, "w", encoding="utf-8") as f:
                json.dump([{"cluster_id": "c1"}], f)
            args = Namespace(
                opencode_skip_clustering=False,
                use_clustering=False,
                inference_clusters_path=clusters_path,
            )
            self.assertEqual(
                load_opencode_inference_clusters(
                    args,
                    inference_clusters_path=clusters_path,
                ),
                [],
            )

    def test_load_cached_clusters_when_artifacts_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clusters_path = os.path.join(tmp, "clusters.json")
            with open(clusters_path, "w", encoding="utf-8") as f:
                json.dump([{"cluster_id": "cached"}], f)
            args = Namespace(
                opencode_skip_clustering=True,
                use_clustering=True,
                inference_clusters_path=clusters_path,
            )
            self.assertEqual(
                load_opencode_inference_clusters(
                    args,
                    inference_clusters_path=clusters_path,
                ),
                [{"cluster_id": "cached"}],
            )


class TestOpenCodeRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.log_dir = self.tmp
        self.sqlite_path = materialize_lake_sqlite(
            LAKE_DIR,
            os.path.join(self.tmp, "lake.sqlite"),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_initial_retrieval_payload(self) -> None:
        schema_path = os.path.join(LAKE_DIR, "schema_descriptions.json")
        payload = _build_initial_retrieval_payload(
            ["T1"],
            ["P1"],
            schema_descriptions_path=schema_path,
            passage_descriptions_path=PASSAGES_PATH,
        )
        self.assertEqual(
            payload,
            {
                "tables": [{"id": "T1", "name": "Stadiums"}],
                "passages": [{"id": "P1", "name": "Urban density"}],
            },
        )

    def test_build_initial_clusters_payload_filters_by_topk(self) -> None:
        inference_clusters = [
            {
                "cluster_id": "c1",
                "description": "stadiums",
                "tables": [{"table_id": "T1", "title": "Stadiums"}],
                "passages": [],
            },
            {
                "cluster_id": "c2",
                "description": "other",
                "tables": [{"table_id": "T99", "title": "Other"}],
                "passages": [],
            },
        ]
        args = Namespace(
            use_clustering=True,
            use_passages=False,
            tables_lake_dir=LAKE_DIR,
        )
        payload = _build_initial_clusters_payload(
            inference_clusters,
            ["T1"],
            [],
            args=args,
            passage_descriptions_path=None,
        )
        self.assertEqual(payload["n_total_inference_clusters"], 2)
        self.assertEqual(payload["n_retained_clusters"], 1)
        self.assertEqual(payload["clusters"][0]["cluster_id"], "c1")

    def test_build_initial_clusters_payload_topk_cluster_when_disabled(self) -> None:
        args = Namespace(
            use_clustering=False,
            use_passages=False,
            tables_lake_dir=LAKE_DIR,
        )
        payload = _build_initial_clusters_payload(
            [],
            ["T1"],
            [],
            args=args,
            passage_descriptions_path=None,
        )
        self.assertEqual(payload["n_total_inference_clusters"], 1)
        self.assertEqual(payload["n_retained_clusters"], 1)
        self.assertEqual(payload["clusters"][0]["cluster_id"], "topk")
        self.assertEqual(
            [t["table_id"] for t in payload["clusters"][0]["tables"]],
            ["T1"],
        )

    @patch("src.opencode.generate_final_report")
    @patch("src.opencode.get_llm_client")
    @patch("src.opencode._compute_relevance")
    def test_skip_retrieval_by_default(
        self, mock_relevance, mock_llm, mock_report
    ) -> None:
        mock_report.return_value = "Synthetic report"
        mock_relevance.return_value = (["T1"], [])
        init_ui(silent=True, rich_cli=False)
        args = Namespace(
            opencode_skip_retrieval=True,
            budget=2,
            max_sql_attempts=3,
            use_passages=False,
            passage_type="synth",
            tables_lake_dir=LAKE_DIR,
            silent=True,
            llm_provider="openai",
            llm_model="gpt-5-mini",
            temperature=1.0,
            compute_metrics=False,
        )
        run_opencode_query(
            args,
            QUERY_RECORD,
            self.log_dir,
            self.sqlite_path,
            [],
            embedder=None,
            skip_agent=True,
        )
        mock_relevance.assert_not_called()
        query_dir = os.path.join(self.log_dir, "q1")
        self.assertFalse(
            os.path.isfile(os.path.join(query_dir, INITIAL_RETRIEVAL_FILENAME))
        )
        with open(os.path.join(query_dir, "prompt.txt"), encoding="utf-8") as f:
            prompt = f.read()
        self.assertNotIn("Initial retrieval candidates:", prompt)

    @patch("src.opencode.generate_final_report")
    @patch("src.opencode.get_llm_client")
    @patch("src.opencode._compute_relevance")
    def test_retrieval_writes_artifact_and_prompt_reference(
        self, mock_relevance, mock_llm, mock_report
    ) -> None:
        mock_report.return_value = "Synthetic report"
        mock_relevance.return_value = (["T1"], [])
        init_ui(silent=True, rich_cli=False)
        args = Namespace(
            opencode_skip_retrieval=False,
            budget=2,
            max_sql_attempts=3,
            use_passages=False,
            passage_type="synth",
            tables_lake_dir=LAKE_DIR,
            silent=True,
            llm_provider="openai",
            llm_model="gpt-5-mini",
            temperature=1.0,
            compute_metrics=False,
            embedding_provider="local",
            embedding_model="test",
            gpu=False,
        )
        run_opencode_query(
            args,
            QUERY_RECORD,
            self.log_dir,
            self.sqlite_path,
            [],
            embedder=object(),
            skip_agent=True,
        )
        mock_relevance.assert_called_once()
        query_dir = os.path.join(self.log_dir, "q1")
        retrieval_path = os.path.join(query_dir, INITIAL_RETRIEVAL_FILENAME)
        self.assertTrue(os.path.isfile(retrieval_path))
        with open(retrieval_path, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["tables"], [{"id": "T1", "name": "Stadiums"}])
        self.assertEqual(payload["passages"], [])
        with open(os.path.join(query_dir, "prompt.txt"), encoding="utf-8") as f:
            prompt = f.read()
        self.assertIn(retrieval_path, prompt)
        self.assertIn("Initial retrieval candidates:", prompt)

    @patch("src.opencode.generate_final_report")
    @patch("src.opencode.get_llm_client")
    @patch("src.opencode._compute_relevance")
    def test_clustering_writes_artifact_when_enabled(
        self, mock_relevance, mock_llm, mock_report
    ) -> None:
        mock_report.return_value = "Synthetic report"
        mock_relevance.return_value = (["T1"], [])
        init_ui(silent=True, rich_cli=False)
        args = Namespace(
            opencode_skip_retrieval=False,
            opencode_skip_clustering=False,
            use_clustering=False,
            budget=2,
            max_sql_attempts=3,
            use_passages=False,
            passage_type="synth",
            tables_lake_dir=LAKE_DIR,
            silent=True,
            llm_provider="openai",
            llm_model="gpt-5-mini",
            temperature=1.0,
            compute_metrics=False,
            embedding_provider="local",
            embedding_model="test",
            gpu=False,
        )
        run_opencode_query(
            args,
            QUERY_RECORD,
            self.log_dir,
            self.sqlite_path,
            [],
            embedder=object(),
            skip_agent=True,
        )
        query_dir = os.path.join(self.log_dir, "q1")
        clusters_path = os.path.join(query_dir, INITIAL_CLUSTERS_FILENAME)
        self.assertTrue(os.path.isfile(clusters_path))
        with open(clusters_path, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["n_retained_clusters"], 1)
        with open(os.path.join(query_dir, "prompt.txt"), encoding="utf-8") as f:
            prompt = f.read()
        self.assertIn(clusters_path, prompt)
        self.assertIn("Initial inference clusters:", prompt)


class TestRemoveMaterializedSqlite(unittest.TestCase):
    def test_removes_db_and_journal_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "data_lake.sqlite")
            for suffix in ("", "-wal", "-shm"):
                with open(f"{db_path}{suffix}", "w", encoding="utf-8") as f:
                    f.write("x")
            _remove_materialized_sqlite(db_path)
            for suffix in ("", "-wal", "-shm"):
                self.assertFalse(os.path.isfile(f"{db_path}{suffix}"))


class TestRunOpenCodeJob(unittest.TestCase):
    @patch("src.opencode.run_opencode_query")
    @patch("src.opencode.get_embedding_client")
    def test_run_opencode_job_calls_query_runner(self, mock_embedder, mock_run_query) -> None:
        mock_run_query.return_value = {"query_id": "q1", "method": "opencode"}
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "run")
            os.makedirs(log_dir)
            sqlite_path = os.path.join(log_dir, "data_lake.sqlite")
            with open(sqlite_path, "w", encoding="utf-8") as f:
                f.write("")
            clusters_path = os.path.join(tmp, "clusters.json")
            with open(clusters_path, "w", encoding="utf-8") as f:
                json.dump([], f)

            args = Namespace(
                seed=0,
                silent=True,
                rich_cli=False,
                embedding_provider="openai",
                embedding_model="text-embedding-3-small",
                gpu=False,
                tables_lake_dir=LAKE_DIR,
                inference_clusters_path=clusters_path,
                table_embeddings_path=None,
                data_dir=tmp,
                output_dir=tmp,
                opencode_skip_retrieval=True,
            )
            result = run_opencode_job(
                worker_args(args),
                QUERY_RECORD,
                log_dir,
                sqlite_path,
                clusters_path,
            )

        self.assertEqual(result["query_id"], "q1")
        mock_embedder.assert_not_called()
        mock_run_query.assert_called_once()
        call_args = mock_run_query.call_args.args
        self.assertEqual(call_args[2], log_dir)
        self.assertEqual(call_args[3], sqlite_path)


class TestOpenCodeContinuationHelpers(unittest.TestCase):
    def test_build_continuation_prompt_references_prior_output(self) -> None:
        stdout = "Explored T809 but stopped before committing."
        prompt = build_continuation_prompt(stdout)
        self.assertIn("best judgment", prompt)
        self.assertIn("Explored T809 but stopped before committing.", prompt)


class TestInvokeOpenCode(unittest.TestCase):
    def _write_lake_tool_config(self, workdir: str, *, budget: int = 2) -> None:
        write_config(
            workdir,
            {
                "sqlite_path": os.path.join(workdir, "lake.sqlite"),
                "passage_descriptions_path": None,
                "budget": budget,
                "max_sql_attempts": 3,
            },
        )
        save_json(
            os.path.join(workdir, "lake_tool_state.json"),
            {"step": 1, "attempts_this_step": 0, "findings": [], "sql_attempts": {}},
        )

    def test_opencode_data_home_links_shared_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared_dir = os.path.join(tmp, "shared", "opencode")
            os.makedirs(shared_dir)
            auth_path = os.path.join(shared_dir, "auth.json")
            with open(auth_path, "w", encoding="utf-8") as f:
                f.write('{"openai": {"type": "api"}}')

            data_home = os.path.join(tmp, "query", ".opencode_data")
            with patch.dict(os.environ, {"OPENCODE_SHARED_DATA_HOME": os.path.join(tmp, "shared")}):
                _ensure_opencode_credentials(data_home)

            linked_auth = os.path.join(data_home, "opencode", "auth.json")
            self.assertTrue(os.path.lexists(linked_auth))
            if os.path.islink(linked_auth):
                self.assertEqual(os.path.realpath(linked_auth), os.path.realpath(auth_path))
            else:
                with open(linked_auth, encoding="utf-8") as f:
                    self.assertEqual(json.load(f), {"openai": {"type": "api"}})

    @patch("src.opencode.fetch_all_opencode_session_usage", return_value=None)
    @patch("src.opencode.subprocess.run")
    def test_invoke_opencode_uses_devnull_stdin(self, mock_run, _mock_fetch) -> None:
        mock_run.return_value = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "done", "stderr": ""},
        )()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = os.path.join(tmp, "prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("test prompt")
            cwd = os.path.join(tmp, "query")
            os.makedirs(cwd)

            result = invoke_opencode(
                model="openai/gpt-4o-mini",
                prompt_path=prompt_path,
                cwd=cwd,
            )

        mock_run.assert_called_once()
        kwargs = mock_run.call_args.kwargs
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(
            kwargs["env"]["XDG_DATA_HOME"],
            os.path.join(cwd, ".opencode_data"),
        )
        cmd = mock_run.call_args.args[0]
        self.assertEqual(
            cmd,
            [
                "opencode",
                "run",
                "--model",
                "openai/gpt-4o-mini",
                "--dir",
                cwd,
                "--format",
                "json",
                "test prompt",
            ],
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["continuations"], 0)

    @patch("src.opencode.fetch_all_opencode_session_usage", return_value=None)
    @patch("src.opencode.subprocess.run")
    def test_invoke_opencode_continues_when_budget_remains(self, mock_run, _mock_fetch) -> None:
        mock_run.side_effect = [
            type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": OPENCODE_JSON_SESSION_LINE
                    + "Explored T809 but stopped before committing.",
                    "stderr": "",
                },
            )(),
            type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "Committed finding.", "stderr": ""},
            )(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = os.path.join(tmp, "prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("test prompt")
            cwd = os.path.join(tmp, "query")
            os.makedirs(cwd)
            self._write_lake_tool_config(cwd, budget=2)

            result = invoke_opencode(
                model="openai/gpt-4o-mini",
                prompt_path=prompt_path,
                cwd=cwd,
            )

            self.assertEqual(mock_run.call_count, 2)
            first_cmd = mock_run.call_args_list[0].args[0]
            second_cmd = mock_run.call_args_list[1].args[0]
            self.assertIn("--format", first_cmd)
            self.assertIn("json", first_cmd)
            self.assertNotIn("--continue", second_cmd)
            self.assertIn("-s", second_cmd)
            self.assertIn("ses_test123", second_cmd)
            self.assertIn("best judgment", second_cmd[-1])
            self.assertIn("Explored T809", second_cmd[-1])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["continuations"], 1)
            self.assertEqual(result["session_id"], "ses_test123")
            self.assertEqual(_budget_remaining(cwd), 2)

    @patch("src.opencode.fetch_all_opencode_session_usage", return_value=None)
    @patch("src.opencode.subprocess.run")
    def test_invoke_opencode_falls_back_to_continue_without_session_id(
        self, mock_run, _mock_fetch
    ) -> None:
        mock_run.side_effect = [
            type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": "Explored T809 but stopped before committing.",
                    "stderr": "",
                },
            )(),
            type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "Committed finding.", "stderr": ""},
            )(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = os.path.join(tmp, "prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("test prompt")
            cwd = os.path.join(tmp, "query")
            os.makedirs(cwd)
            self._write_lake_tool_config(cwd, budget=2)

            result = invoke_opencode(
                model="openai/gpt-4o-mini",
                prompt_path=prompt_path,
                cwd=cwd,
            )

        self.assertEqual(mock_run.call_count, 2)
        first_cmd = mock_run.call_args_list[0].args[0]
        second_cmd = mock_run.call_args_list[1].args[0]
        self.assertIn("--format", first_cmd)
        self.assertIn("--continue", second_cmd)
        self.assertNotIn("-s", second_cmd)
        self.assertIsNone(result.get("session_id"))
        self.assertEqual(result["continuations"], 1)

    @patch("src.opencode.fetch_all_opencode_session_usage", return_value=None)
    @patch("src.opencode.subprocess.run")
    def test_invoke_opencode_stops_after_two_stalled_rounds(self, mock_run, _mock_fetch) -> None:
        stalled = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": OPENCODE_JSON_SESSION_LINE + "Still exploring tables.",
                "stderr": "",
            },
        )()
        mock_run.side_effect = [stalled, stalled, stalled]
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = os.path.join(tmp, "prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("test prompt")
            cwd = os.path.join(tmp, "query")
            os.makedirs(cwd)
            self._write_lake_tool_config(cwd, budget=2)

            result = invoke_opencode(
                model="openai/gpt-4o-mini",
                prompt_path=prompt_path,
                cwd=cwd,
            )

        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(result["continuations"], 1)

    @patch("src.opencode.fetch_all_opencode_session_usage", return_value=None)
    @patch("src.opencode.subprocess.run")
    def test_invoke_opencode_resume_uses_session_and_resume_prompt(
        self, mock_run, _mock_fetch
    ) -> None:
        mock_run.return_value = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "Committed finding.", "stderr": ""},
        )()
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = os.path.join(tmp, "prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("full initial prompt")
            cwd = os.path.join(tmp, "query")
            os.makedirs(cwd)
            self._write_lake_tool_config(cwd, budget=50)
            save_json(
                os.path.join(cwd, "lake_tool_state.json"),
                {
                    "step": 35,
                    "attempts_this_step": 0,
                    "findings": [{"step": i} for i in range(1, 35)],
                    "sql_attempts": {},
                },
            )

            invoke_opencode(
                model="openai/gpt-4o-mini",
                prompt_path=prompt_path,
                cwd=cwd,
                resume_session_id="ses_resume123",
                resume_agent=True,
            )

        cmd = mock_run.call_args.args[0]
        self.assertIn("-s", cmd)
        self.assertIn("ses_resume123", cmd)
        self.assertNotIn("--format", cmd)
        message = cmd[-1]
        self.assertIn("34/50", message)
        self.assertNotIn("full initial prompt", message)

    @patch("src.opencode.fetch_all_opencode_session_usage", return_value=None)
    @patch("src.opencode.subprocess.run")
    def test_invoke_opencode_timeout_coerces_bytes_stdout(
        self, mock_run, _mock_fetch
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["opencode"],
            timeout=1,
            output=b"partial output",
            stderr=b"partial err",
        )
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = os.path.join(tmp, "prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("test prompt")
            cwd = os.path.join(tmp, "query")
            os.makedirs(cwd)

            result = invoke_opencode(
                model="openai/gpt-4o-mini",
                prompt_path=prompt_path,
                cwd=cwd,
            )

        self.assertEqual(result["status"], "timeout")
        self.assertIn("partial output", result["stdout"])
        self.assertIn("partial err", result["stderr"])


class TestPartialWorkspaceResume(unittest.TestCase):
    def test_partial_workspace_status_resume_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query_dir = os.path.join(tmp, "41")
            os.makedirs(query_dir)
            save_json(
                os.path.join(query_dir, "lake_tool_state.json"),
                {
                    "step": 35,
                    "findings": [{"step": i} for i in range(1, 35)],
                },
            )
            status = partial_workspace_status(query_dir, budget=50)
            self.assertTrue(status.resume_agent)
            self.assertFalse(status.finalize_only)
            self.assertEqual(status.n_findings, 34)

    def test_partial_workspace_status_finalize_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query_dir = os.path.join(tmp, "41")
            os.makedirs(query_dir)
            save_json(
                os.path.join(query_dir, "lake_tool_state.json"),
                {"findings": [{"step": i} for i in range(1, 51)]},
            )
            status = partial_workspace_status(query_dir, budget=50)
            self.assertFalse(status.resume_agent)
            self.assertTrue(status.finalize_only)
            self.assertEqual(status.n_findings, 50)

    def test_setup_lake_workspace_preserves_findings_when_not_resetting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = os.path.join(tmp, "query")
            os.makedirs(workdir)
            existing = {
                "step": 3,
                "attempts_this_step": 0,
                "findings": [{"step": 1, "answer": "kept"}],
                "sql_attempts": {},
            }
            save_json(os.path.join(workdir, "lake_tool_state.json"), existing)
            args = Namespace(
                use_passages=False,
                passage_type="synth",
                data_dir=tmp,
                budget=50,
                max_sql_attempts=3,
            )
            setup_lake_workspace(
                args,
                workdir,
                os.path.join(tmp, "lake.sqlite"),
                reset_state=False,
            )
            state = load_state(workdir)
            self.assertEqual(len(state["findings"]), 1)
            self.assertEqual(state["findings"][0]["answer"], "kept")

    def test_load_resume_workspace_context_reads_initial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            query_dir = os.path.join(tmp, "41")
            os.makedirs(query_dir)
            save_json(
                os.path.join(query_dir, INITIAL_RETRIEVAL_FILENAME),
                {
                    "tables": [{"id": "T1", "name": "A"}],
                    "passages": [{"id": "P2", "name": "B"}],
                },
            )
            save_json(
                os.path.join(query_dir, INITIAL_CLUSTERS_FILENAME),
                {
                    "n_retained_clusters": 2,
                    "n_total_inference_clusters": 5,
                    "clusters": [{"cluster_id": 0}],
                },
            )
            ctx = _load_resume_workspace_context(query_dir)
            self.assertEqual(ctx["topk_table_ids"], ["T1"])
            self.assertEqual(ctx["topk_passage_ids"], ["P2"])
            self.assertEqual(ctx["retained_clusters"], 2)
            self.assertEqual(ctx["total_inference_clusters"], 5)
            self.assertEqual(len(ctx["metrics_inference_clusters"]), 1)


class TestRunOpenCodeQueryResume(unittest.TestCase):
    @patch("src.opencode.finalize_query_from_workspace")
    @patch("src.opencode.invoke_opencode")
    def test_run_opencode_query_resumes_partial_workspace(
        self, mock_invoke, mock_finalize
    ) -> None:
        mock_invoke.return_value = {"status": "completed", "stdout": "", "stderr": ""}
        mock_finalize.return_value = {
            "query_id": "41",
            "answer": "done",
            "findings": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "run")
            query_dir = os.path.join(log_dir, "41")
            os.makedirs(query_dir)
            save_json(
                os.path.join(query_dir, "lake_tool_state.json"),
                {
                    "step": 35,
                    "attempts_this_step": 0,
                    "findings": [{"step": i, "answer": "a"} for i in range(1, 35)],
                    "sql_attempts": {},
                },
            )
            with open(os.path.join(query_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write("saved prompt")
            sqlite_path = materialize_lake_sqlite(
                LAKE_DIR,
                os.path.join(tmp, "lake.sqlite"),
            )
            init_ui(silent=True, rich_cli=False)
            args = _finalize_test_args(budget=50)
            run_opencode_query(
                args,
                {"query_id": "41", "query_text": "Resume me?", "coverage": "medium"},
                log_dir,
                sqlite_path,
                [],
                skip_agent=False,
            )
            state = load_state(query_dir)
            self.assertEqual(len(state["findings"]), 34)
            mock_invoke.assert_called_once()
            invoke_kwargs = mock_invoke.call_args.kwargs
            self.assertTrue(invoke_kwargs["resume_agent"])
            mock_finalize.assert_called_once()


class _RecordingUI:
    rich_cli = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def set_budget_step(self, step: int, detail: str = "") -> None:
        self.calls.append(("set_budget_step", (step, detail), {}))

    def set_sub_question(self, sub_question: str) -> None:
        self.calls.append(("set_sub_question", (sub_question,), {}))

    def set_sql_result(self, sql: str = "", execution=None) -> None:
        self.calls.append(("set_sql_result", (sql, execution), {}))

    def show_sql_answer(self, answer: str, *, pause_seconds: float = 2.0) -> None:
        self.calls.append(
            ("show_sql_answer", (answer,), {"pause_seconds": pause_seconds})
        )

    def complete_budget_step(self, step: int) -> None:
        self.calls.append(("complete_budget_step", (step,), {}))

    def set_status(self, status: str) -> None:
        self.calls.append(("set_status", (status,), {}))

    def log(self, msg: str, silent: bool = False) -> None:
        self.calls.append(("log", (msg,), {"silent": silent}))


class TestLakeStateUISync(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.workdir = os.path.join(self.tmp, "query_q1")
        os.makedirs(self.workdir)
        self.sqlite_path = materialize_lake_sqlite(
            LAKE_DIR,
            os.path.join(self.tmp, "lake.sqlite"),
        )
        write_config(
            self.workdir,
            {
                "sqlite_path": self.sqlite_path,
                "passage_descriptions_path": PASSAGES_PATH,
                "budget": 2,
                "max_sql_attempts": 2,
            },
        )
        self.ui = _RecordingUI()
        self.seen = _LakeUISeen()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sync(self) -> None:
        _apply_lake_state_to_ui(
            self.workdir,
            self.ui,
            budget=2,
            seen=self.seen,
        )

    def test_sync_reports_sql_attempts_before_commit(self) -> None:
        _run_tool(
            self.workdir,
            "sql",
            'SELECT name FROM "T1" LIMIT 1',
        )
        self._sync()

        self.assertEqual(self.seen.n_attempts, 1)
        self.assertEqual(self.ui.calls[0][0], "set_budget_step")
        self.assertEqual(self.ui.calls[0][1][0], 1)
        self.assertIn("SQL attempt", self.ui.calls[1][1][0])
        self.assertEqual(self.ui.calls[2][0], "set_sql_result")

    def test_sync_reports_committed_finding(self) -> None:
        _run_tool(
            self.workdir,
            "sql",
            'SELECT name FROM "T1" LIMIT 1',
        )
        _run_tool(
            self.workdir,
            "commit",
            "--sub-question",
            "Which stadium?",
            "--answer",
            "Alpha Stadium [T1].",
        )
        self._sync()

        self.assertEqual(self.seen.n_findings, 1)
        names = [call[0] for call in self.ui.calls]
        self.assertIn("set_sub_question", names)
        self.assertIn("show_sql_answer", names)
        self.assertIn("complete_budget_step", names)
        statuses = [call[1][0] for call in self.ui.calls if call[0] == "set_status"]
        self.assertIn("finding 1/2", statuses)
        answers = [
            call[1][0]
            for call in self.ui.calls
            if call[0] == "show_sql_answer"
        ]
        self.assertIn("Alpha Stadium [T1].", answers)

    def test_sync_reports_multiple_new_findings(self) -> None:
        save_json(
            os.path.join(self.workdir, "lake_tool_state.json"),
            {
                "step": 3,
                "attempts_this_step": 0,
                "findings": [
                    {
                        "step": 1,
                        "sub_question": "Q1",
                        "answer": "A1",
                        "needs_sql": False,
                        "sql": None,
                        "execution": None,
                    },
                    {
                        "step": 2,
                        "sub_question": "Q2",
                        "answer": "A2",
                        "needs_sql": False,
                        "sql": None,
                        "execution": None,
                    },
                ],
                "sql_attempts": {},
            },
        )
        self._sync()

        self.assertEqual(self.seen.n_findings, 2)
        completed = [
            call[1][0]
            for call in self.ui.calls
            if call[0] == "complete_budget_step"
        ]
        self.assertEqual(completed, [1, 2])
        statuses = [call[1][0] for call in self.ui.calls if call[0] == "set_status"]
        self.assertIn("finding 2/2", statuses)

    def test_sync_is_idempotent(self) -> None:
        _run_tool(
            self.workdir,
            "commit",
            "--sub-question",
            "Q1",
            "--answer",
            "A1",
            "--no-sql",
        )
        self._sync()
        first_count = len(self.ui.calls)
        self._sync()
        self.assertEqual(len(self.ui.calls), first_count)

    def test_sync_with_descending_findings_order(self) -> None:
        save_json(
            os.path.join(self.workdir, "lake_tool_state.json"),
            {
                "step": 3,
                "attempts_this_step": 0,
                "findings": [
                    {
                        "step": 2,
                        "sub_question": "Q2",
                        "answer": "A2",
                        "needs_sql": False,
                        "sql": None,
                        "execution": None,
                    },
                    {
                        "step": 1,
                        "sub_question": "Q1",
                        "answer": "A1",
                        "needs_sql": False,
                        "sql": None,
                        "execution": None,
                    },
                ],
                "sql_attempts": {},
            },
        )
        self._sync()

        self.assertEqual(self.seen.n_findings, 2)
        completed = [
            call[1][0]
            for call in self.ui.calls
            if call[0] == "complete_budget_step"
        ]
        self.assertEqual(completed, [1, 2])

    def test_incremental_sync_with_save_state_sorting(self) -> None:
        _run_tool(
            self.workdir,
            "commit",
            "--sub-question",
            "Q1",
            "--answer",
            "A1",
            "--no-sql",
        )
        state = load_state(self.workdir)
        self.assertEqual([int(f["step"]) for f in state["findings"]], [1])

        self._sync()
        self.assertEqual(self.seen.n_findings, 1)

        _run_tool(
            self.workdir,
            "commit",
            "--sub-question",
            "Q2",
            "--answer",
            "A2",
            "--no-sql",
        )
        state = load_state(self.workdir)
        self.assertEqual([int(f["step"]) for f in state["findings"]], [2, 1])

        self._sync()
        self.assertEqual(self.seen.n_findings, 2)
        completed = [
            call[1][0]
            for call in self.ui.calls
            if call[0] == "complete_budget_step"
        ]
        self.assertEqual(completed, [1, 2])


class TestWatchOpenCodeLakeState(unittest.TestCase):
    @patch("src.opencode.threading.Thread")
    @patch("src.opencode.get_ui")
    def test_skips_polling_without_rich_cli(
        self, mock_get_ui, mock_thread
    ) -> None:
        ui = _RecordingUI()
        ui.rich_cli = False
        mock_get_ui.return_value = ui

        with _watch_opencode_lake_state("/tmp/query", budget=2):
            pass

        mock_thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
