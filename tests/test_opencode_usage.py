"""Tests for OpenCode session usage capture."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.opencode_usage import (
    build_opencode_run_usage,
    combine_query_usage,
    fetch_opencode_session_usage,
    format_nested_usage_line,
    is_valid_session_id,
    load_latest_opencode_session_id,
    load_opencode_session_ids,
    fetch_all_opencode_session_usage,
    merge_nested_usage_dicts,
    normalize_opencode_db_row,
    parse_session_id_from_json_stdout,
)
from src.pipeline import merge_usage_dicts
from src.tracking import UsageSummary, UsageTracker


class TestSessionIdParsing(unittest.TestCase):
    def test_parse_session_id_from_json_stdout(self) -> None:
        stdout = "\n".join(
            [
                '{"type":"step_start","sessionID":"ses_abc123","timestamp":1}',
                '{"type":"text","sessionID":"ses_abc123","timestamp":2}',
            ]
        )
        self.assertEqual(parse_session_id_from_json_stdout(stdout), "ses_abc123")

    def test_parse_session_id_ignores_invalid_lines(self) -> None:
        stdout = "not json\n" '{"sessionID":"bad id with spaces"}' "\n"
        self.assertIsNone(parse_session_id_from_json_stdout(stdout))

    def test_is_valid_session_id(self) -> None:
        self.assertTrue(is_valid_session_id("ses_abc123"))
        self.assertTrue(is_valid_session_id("ses_152edf3bbffePl8AmprS4jEqf9"))
        self.assertFalse(is_valid_session_id("ses_bad id"))
        self.assertFalse(is_valid_session_id("msg_abc"))


class TestLoadLatestOpenCodeSessionId(unittest.TestCase):
    def test_load_opencode_session_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_home = os.path.join(tmp, ".opencode_data")
            db_dir = os.path.join(data_home, "opencode")
            os.makedirs(db_dir)
            db_path = os.path.join(db_dir, "opencode.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT)"
            )
            conn.execute(
                "INSERT INTO session (id, directory) VALUES (?, ?)",
                ("ses_old", "/tmp/q1"),
            )
            conn.execute(
                "INSERT INTO session (id, directory) VALUES (?, ?)",
                ("ses_new", "/tmp/q1"),
            )
            conn.commit()
            conn.close()

            self.assertEqual(
                load_opencode_session_ids(xdg_data_home=data_home),
                ["ses_old", "ses_new"],
            )

    def test_load_latest_opencode_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_home = os.path.join(tmp, ".opencode_data")
            db_dir = os.path.join(data_home, "opencode")
            os.makedirs(db_dir)
            db_path = os.path.join(db_dir, "opencode.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT)"
            )
            conn.execute(
                "INSERT INTO session (id, directory) VALUES (?, ?)",
                ("ses_old", "/tmp/q1"),
            )
            conn.execute(
                "INSERT INTO session (id, directory) VALUES (?, ?)",
                ("ses_new", "/tmp/q1"),
            )
            conn.commit()
            conn.close()

            session_id = load_latest_opencode_session_id(xdg_data_home=data_home)
            self.assertEqual(session_id, "ses_new")

    def test_load_latest_opencode_session_id_missing_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                load_latest_opencode_session_id(
                    xdg_data_home=os.path.join(tmp, ".opencode_data")
                )
            )


class TestFetchAllOpenCodeSessionUsage(unittest.TestCase):
    @patch("src.opencode_usage.fetch_opencode_session_usage")
    def test_fetch_all_opencode_session_usage_merges_sessions(
        self, mock_fetch
    ) -> None:
        mock_fetch.side_effect = [
            {
                "session_id": "ses_old",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost_usd": 0.1,
                "time_taken": 0.0,
                "n_calls": 1,
            },
            {
                "session_id": "ses_new",
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
                "cost_usd": 0.05,
                "time_taken": 0.0,
                "n_calls": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            data_home = os.path.join(tmp, ".opencode_data")
            db_dir = os.path.join(data_home, "opencode")
            os.makedirs(db_dir)
            db_path = os.path.join(db_dir, "opencode.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT)"
            )
            conn.execute(
                "INSERT INTO session (id, directory) VALUES (?, ?)",
                ("ses_old", "/tmp/q1"),
            )
            conn.execute(
                "INSERT INTO session (id, directory) VALUES (?, ?)",
                ("ses_new", "/tmp/q1"),
            )
            conn.commit()
            conn.close()

            usage = fetch_all_opencode_session_usage(xdg_data_home=data_home)

        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage["source"], "opencode_db")
        self.assertEqual(usage["session_ids"], ["ses_old", "ses_new"])
        self.assertEqual(usage["session_id"], "ses_new")
        self.assertEqual(usage["total_tokens"], 180)
        self.assertEqual(usage["cost_usd"], 0.15)
        self.assertEqual(mock_fetch.call_count, 2)


class TestNormalizeOpenCodeUsage(unittest.TestCase):
    def test_normalize_opencode_db_row(self) -> None:
        row = {
            "id": "ses_test",
            "cost": 0.1234,
            "tokens_input": 100,
            "tokens_output": 50,
            "tokens_reasoning": 10,
            "tokens_cache_read": 20,
            "tokens_cache_write": 5,
            "model": {"providerID": "openai", "id": "gpt-4o"},
            "title": "Test",
            "directory": "/tmp/q1",
        }
        usage = normalize_opencode_db_row(row)
        self.assertEqual(usage["session_id"], "ses_test")
        self.assertEqual(usage["model"], "openai/gpt-4o")
        self.assertEqual(usage["prompt_tokens"], 125)
        self.assertEqual(usage["completion_tokens"], 60)
        self.assertEqual(usage["total_tokens"], 185)
        self.assertEqual(usage["cost_usd"], 0.1234)


class TestUsageMerge(unittest.TestCase):
    def test_combine_query_usage_without_opencode(self) -> None:
        pipeline = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.01,
            "time_taken": 1.0,
            "n_calls": 1,
        }
        combined = combine_query_usage(pipeline, None)
        self.assertEqual(combined["pipeline"], pipeline)
        self.assertEqual(combined["total"], pipeline)
        self.assertNotIn("opencode", combined)

    def test_combine_query_usage_with_opencode(self) -> None:
        pipeline = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.01,
            "time_taken": 1.0,
            "n_calls": 1,
        }
        opencode = {
            "session_id": "ses_test",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost_usd": 0.2,
            "time_taken": 0.0,
            "n_calls": 1,
        }
        combined = combine_query_usage(pipeline, opencode)
        self.assertEqual(combined["total"]["total_tokens"], 135)
        self.assertAlmostEqual(combined["total"]["cost_usd"], 0.21)

    def test_format_nested_usage_line_shows_breakdown(self) -> None:
        usage = combine_query_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost_usd": 0.01,
                "time_taken": 1.0,
                "n_calls": 1,
            },
            {
                "session_id": "ses_test",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost_usd": 0.2,
                "time_taken": 0.0,
                "n_calls": 1,
            },
        )
        line = format_nested_usage_line(usage)
        self.assertIn("agent:", line)
        self.assertIn("pipeline:", line)
        self.assertNotIn("agent $", line)

    def test_merge_nested_usage_dicts(self) -> None:
        usages = [
            combine_query_usage(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.01,
                    "time_taken": 1.0,
                    "n_calls": 1,
                },
                {
                    "session_id": "ses_a",
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost_usd": 0.2,
                    "time_taken": 0.0,
                    "n_calls": 1,
                },
            ),
            combine_query_usage(
                {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                    "cost_usd": 0.02,
                    "time_taken": 0.5,
                    "n_calls": 1,
                },
                None,
            ),
        ]
        merged = merge_nested_usage_dicts(usages)
        self.assertEqual(merged["total"]["total_tokens"], 145)
        self.assertAlmostEqual(merged["total"]["cost_usd"], 0.23)
        self.assertEqual(merged["pipeline"]["total_tokens"], 25)
        self.assertEqual(merged["opencode"]["total_tokens"], 120)

    def test_build_opencode_run_usage(self) -> None:
        tracker = UsageTracker()
        tracker.record(
            call_type="completion",
            feature="final_report",
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=0.01,
        )
        results = [
            {
                "summary": {
                    "usage": combine_query_usage(
                        {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                            "cost_usd": 0.01,
                            "time_taken": 1.0,
                            "n_calls": 1,
                        },
                        {
                            "session_id": "ses_a",
                            "prompt_tokens": 1000,
                            "completion_tokens": 200,
                            "total_tokens": 1200,
                            "cost_usd": 0.5,
                            "time_taken": 0.0,
                            "n_calls": 1,
                        },
                    )
                }
            }
        ]
        run_usage = build_opencode_run_usage(results, tracker)
        self.assertEqual(run_usage["total"]["total_tokens"], 1320)
        self.assertAlmostEqual(run_usage["total"]["cost_usd"], 0.51)
        self.assertIn("pipeline", run_usage)
        self.assertIn("opencode", run_usage)

    def test_merge_usage_dicts_nested(self) -> None:
        merged = merge_usage_dicts(
            [
                combine_query_usage(
                    {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost_usd": 0.01,
                        "time_taken": 1.0,
                        "n_calls": 1,
                    },
                    {
                        "session_id": "ses_a",
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                        "cost_usd": 0.2,
                        "time_taken": 0.0,
                        "n_calls": 1,
                    },
                )
            ]
        )
        self.assertEqual(merged["total"]["total_tokens"], 135)
        self.assertIn("opencode", merged)


class TestFetchOpenCodeSessionUsage(unittest.TestCase):
    @patch("src.opencode_usage.subprocess.run")
    def test_fetch_opencode_session_usage(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "id": "ses_test",
                        "cost": 0.5,
                        "tokens_input": 1000,
                        "tokens_output": 200,
                        "tokens_reasoning": 0,
                        "tokens_cache_read": 0,
                        "tokens_cache_write": 0,
                        "model": {"providerID": "openai", "id": "gpt-4o-mini"},
                        "title": "Run",
                        "directory": "/tmp/q1",
                    }
                ]
            ),
        )
        usage = fetch_opencode_session_usage("ses_test")
        assert usage is not None
        self.assertEqual(usage["session_id"], "ses_test")
        self.assertEqual(usage["cost_usd"], 0.5)

    @patch("src.opencode_usage.subprocess.run")
    def test_fetch_opencode_session_usage_isolates_xdg_data_home(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="[]",
        )
        fetch_opencode_session_usage("ses_test", xdg_data_home="/tmp/q1/.opencode_data")
        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["XDG_DATA_HOME"], "/tmp/q1/.opencode_data")

    @patch("src.opencode_usage.subprocess.run")
    def test_fetch_rejects_invalid_session_id(self, mock_run) -> None:
        self.assertIsNone(fetch_opencode_session_usage("bad"))
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
