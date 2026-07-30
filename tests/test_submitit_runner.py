"""Tests for submitit batching helpers."""

import builtins
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from src.pipeline import format_missing_results_error, queries_missing_results
from src.submitit_runner import (
    build_submitit_executor,
    load_run_args_for_retry,
    merge_executor_settings,
    merge_only_display_args,
    run_merge_only,
    submit_and_wait_batched,
)


class _FakeJob:
    def __init__(self, job_id: str, *, fail: bool = False):
        self.job_id = job_id
        self._fail = fail

    def result(self):
        if self._fail:
            raise RuntimeError(f"job {self.job_id} failed")
        return {"job_id": self.job_id}


class SubmitAndWaitBatchedTests(unittest.TestCase):
    def test_local_batches_submissions(self):
        queries = [{"query_id": f"q{i}"} for i in range(5)]
        submit_calls: list[str] = []
        active_batches: list[int] = []
        current_batch = 0

        def submit_one(query):
            nonlocal current_batch
            submit_calls.append(str(query["query_id"]))
            active_batches.append(current_batch)
            return str(query["query_id"]), _FakeJob(str(query["query_id"]))

        original_wait = submit_and_wait_batched.__globals__["wait_for_submitit_jobs"]

        def track_batches(jobs):
            nonlocal current_batch
            current_batch += 1
            return original_wait(jobs)

        with patch(
            "src.submitit_runner.wait_for_submitit_jobs",
            side_effect=track_batches,
        ):
            completed, failures = submit_and_wait_batched(
                queries,
                local=True,
                max_workers=2,
                submit_one=submit_one,
            )

        self.assertEqual(completed, ["q0", "q1", "q2", "q3", "q4"])
        self.assertEqual(failures, [])
        self.assertEqual(submit_calls, ["q0", "q1", "q2", "q3", "q4"])
        self.assertEqual(active_batches, [0, 0, 1, 1, 2])

    def test_slurm_submits_all_before_waiting(self):
        queries = [{"query_id": f"q{i}"} for i in range(3)]
        submit_calls: list[str] = []

        def submit_one(query):
            submit_calls.append(str(query["query_id"]))
            return str(query["query_id"]), _FakeJob(str(query["query_id"]))

        completed, failures = submit_and_wait_batched(
            queries,
            local=False,
            max_workers=2,
            submit_one=submit_one,
        )

        self.assertEqual(completed, ["q0", "q1", "q2"])
        self.assertEqual(failures, [])
        self.assertEqual(submit_calls, ["q0", "q1", "q2"])

    def test_collects_failures_across_batches(self):
        queries = [{"query_id": f"q{i}"} for i in range(3)]

        def submit_one(query):
            query_id = str(query["query_id"])
            fail = query_id == "q1"
            return query_id, _FakeJob(query_id, fail=fail)

        completed, failures = submit_and_wait_batched(
            queries,
            local=True,
            max_workers=1,
            submit_one=submit_one,
        )

        self.assertEqual(completed, ["q0", "q2"])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "q1")


_HEAVY_CLUSTERING_MODULES = frozenset(
    {"umap", "pynndescent", "bertopic", "hdbscan", "src.clustering"}
)


def _reload_without_heavy_clustering(module_name: str) -> set[str]:
    """Reload module and return any heavy clustering deps that were imported."""
    imported: set[str] = set()
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in _HEAVY_CLUSTERING_MODULES or name in _HEAVY_CLUSTERING_MODULES:
            imported.add(name)
        return real_import(name, globals, locals, fromlist, level)

    to_drop = [
        key
        for key in list(sys.modules)
        if key == module_name or key.startswith(f"{module_name}.")
    ]
    for key in to_drop:
        sys.modules.pop(key, None)

    with patch("builtins.__import__", side_effect=tracking_import):
        importlib.import_module(module_name)
    return imported


class WorkerImportTests(unittest.TestCase):
    def test_submitit_runner_does_not_import_clustering_stack(self) -> None:
        imported = _reload_without_heavy_clustering("src.submitit_runner")
        self.assertEqual(imported, set())

    def test_query_runner_does_not_import_clustering_stack(self) -> None:
        imported = _reload_without_heavy_clustering("src.query_runner")
        self.assertEqual(imported, set())

    def test_pipeline_does_not_import_clustering_stack(self) -> None:
        imported = _reload_without_heavy_clustering("src.pipeline")
        self.assertEqual(imported, set())


class BuildSubmititExecutorTests(unittest.TestCase):
    def test_local_executor_ignores_max_workers(self):
        args = Namespace(
            local=True,
            max_workers=4,
            slurm_partition=None,
            timeout_min=None,
            cpus_per_task=None,
            gpus_per_node=None,
            mem_gb=None,
        )
        fake_executor = MagicMock()
        fake_local = MagicMock(return_value=fake_executor)
        fake_module = MagicMock()
        fake_module.LocalExecutor = fake_local

        with patch.dict("sys.modules", {"submitit": fake_module}):
            executor = build_submitit_executor("/tmp/submitit", args)

        self.assertEqual(executor, fake_executor)
        fake_local.assert_called_once_with(folder="/tmp/submitit")
        fake_executor.update_parameters.assert_called_once_with(timeout_min=60)


class QueriesMissingResultsTests(unittest.TestCase):
    def test_detects_missing_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "1"))
            os.makedirs(os.path.join(tmp, "2"))
            with open(os.path.join(tmp, "1", "result.json"), "w", encoding="utf-8") as f:
                json.dump({"query_id": "1"}, f)
            queries = [{"query_id": "1"}, {"query_id": "2"}]
            missing_ids, missing_queries = queries_missing_results(tmp, queries)
        self.assertEqual(missing_ids, ["2"])
        self.assertEqual(missing_queries, [{"query_id": "2"}])

    def test_format_missing_results_error(self) -> None:
        message = format_missing_results_error(["1", "2", "3"])
        self.assertEqual(message, "Missing result.json for 3 queries: 1, 2, 3")


class MergeExecutorSettingsTests(unittest.TestCase):
    def test_prefers_saved_executor_settings_when_cli_omits_flags(self) -> None:
        saved = {"local": True, "max_workers": 4, "slurm_partition": "gpu"}
        cli_args = Namespace(
            local=False,
            max_workers=2,
            slurm_partition=None,
            timeout_min=None,
            cpus_per_task=None,
            gpus_per_node=None,
            mem_gb=None,
        )
        with patch.object(sys, "argv", ["run"]):
            merged = merge_executor_settings(saved, cli_args)
        self.assertTrue(merged["local"])
        self.assertEqual(merged["max_workers"], 4)
        self.assertEqual(merged["slurm_partition"], "gpu")

    def test_cli_flags_override_saved_settings(self) -> None:
        saved = {"local": True, "max_workers": 4, "slurm_partition": "gpu"}
        cli_args = Namespace(
            local=False,
            max_workers=2,
            slurm_partition="cpu",
            timeout_min=90,
            cpus_per_task=None,
            gpus_per_node=None,
            mem_gb=None,
        )
        argv = ["run", "--no-local", "--max_workers", "2", "--slurm_partition", "cpu", "--timeout_min", "90"]
        with patch.object(sys, "argv", argv):
            merged = merge_executor_settings(saved, cli_args)
        self.assertFalse(merged["local"])
        self.assertEqual(merged["max_workers"], 2)
        self.assertEqual(merged["slurm_partition"], "cpu")
        self.assertEqual(merged["timeout_min"], 90)

    def test_load_run_args_for_retry_uses_saved_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "args.json"), "w", encoding="utf-8") as f:
                json.dump({"local": True, "max_workers": 3}, f)
            cli_args = Namespace(
                local=False,
                max_workers=2,
                slurm_partition=None,
                timeout_min=None,
                cpus_per_task=None,
                gpus_per_node=None,
                mem_gb=None,
            )
            with (
                patch.object(sys, "argv", ["run"]),
                patch(
                    "src.submitit_runner.args_from_dict",
                    side_effect=lambda raw: Namespace(**raw),
                ),
            ):
                run_args = load_run_args_for_retry(tmp, cli_args)
        self.assertTrue(run_args.local)
        self.assertEqual(run_args.max_workers, 3)


class MergeOnlyDisplayArgsTests(unittest.TestCase):
    def test_uses_saved_run_args_with_executor_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "run_queries.json"), "w", encoding="utf-8") as f:
                json.dump([{"query_id": "1", "query_text": "q"}], f)
            with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "data_dir": "data/tatqa",
                        "passage_type": "raw",
                        "compute_metrics": True,
                        "cpus_per_task": 4,
                        "mem_gb": 16,
                    },
                    f,
                )

            cli_args = Namespace(
                output_dir=tmp,
                run_dir=run_dir,
                merge_only=True,
                retry_missing=True,
                recompute_metrics=False,
                slurm_partition="cpu-preempt",
                timeout_min=240,
                cpus_per_task=8,
                mem_gb=32,
                gpus_per_node=None,
                local=False,
                max_workers=2,
            )

            display = merge_only_display_args(cli_args)

        self.assertTrue(display["merge_only"])
        self.assertTrue(display["retry_missing"])
        self.assertEqual(display["data_dir"], "data/tatqa")
        self.assertEqual(display["passage_type"], "raw")
        self.assertTrue(display["compute_metrics"])
        self.assertEqual(display["cpus_per_task"], 8)
        self.assertEqual(display["mem_gb"], 32)
        self.assertEqual(display["slurm_partition"], "cpu-preempt")


class RunMergeOnlyTests(unittest.TestCase):
    def test_retry_missing_runs_jobs_before_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run_dir, "5"))
            queries = [{"query_id": "5", "query_text": "q"}]
            with open(os.path.join(run_dir, "run_queries.json"), "w", encoding="utf-8") as f:
                json.dump(queries, f)
            with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "method": "dpr_discovery",
                        "compute_metrics": False,
                        "use_clustering": False,
                        "local": True,
                        "max_workers": 1,
                        "inference_clusters_path": "/tmp/clusters.json",
                        "seed": 0,
                    },
                    f,
                )

            cli_args = Namespace(
                output_dir=tmp,
                run_dir=run_dir,
                merge_only=True,
                retry_missing=True,
                recompute_metrics=False,
                compute_metrics=False,
                no_llm_judge=False,
                judge_models="gpt-5-mini",
                compute_embed_diversity=False,
                silent=False,
                rich_cli=True,
                local=True,
                max_workers=1,
                slurm_partition=None,
                timeout_min=None,
                cpus_per_task=None,
                gpus_per_node=None,
                mem_gb=None,
            )

            with (
                patch(
                    "src.submitit_runner.run_retry_missing_queries",
                    return_value=0,
                ) as retry_mock,
                patch(
                    "src.submitit_runner.report_merged_results",
                    return_value="/tmp/results_all.json",
                ) as merge_mock,
            ):
                rc = run_merge_only(cli_args)

            self.assertEqual(rc, 0)
            retry_mock.assert_called_once()
            merge_mock.assert_called_once()

    def test_without_retry_missing_fails_on_missing_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            os.makedirs(os.path.join(run_dir, "5"))
            queries = [{"query_id": "5", "query_text": "q"}]
            with open(os.path.join(run_dir, "run_queries.json"), "w", encoding="utf-8") as f:
                json.dump(queries, f)

            cli_args = Namespace(
                output_dir=tmp,
                run_dir=run_dir,
                merge_only=True,
                retry_missing=False,
                recompute_metrics=False,
                compute_metrics=False,
            )

            stderr_buf = io.StringIO()
            with patch("sys.stderr", stderr_buf):
                rc = run_merge_only(cli_args)

            self.assertEqual(rc, 1)
            self.assertIn("Missing result.json for 1 queries: 5", stderr_buf.getvalue())

    def test_recompute_metrics_runs_before_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            qdir = os.path.join(run_dir, "5")
            os.makedirs(qdir)
            queries = [{"query_id": "5", "query_text": "q"}]
            with open(os.path.join(run_dir, "run_queries.json"), "w", encoding="utf-8") as f:
                json.dump(queries, f)
            with open(os.path.join(qdir, "result.json"), "w", encoding="utf-8") as f:
                json.dump({"query_id": "5", "user_query": "q", "answer": "report"}, f)
            with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
                json.dump({"compute_metrics": False}, f)

            cli_args = Namespace(
                output_dir=tmp,
                run_dir=run_dir,
                merge_only=True,
                retry_missing=False,
                recompute_metrics=True,
                compute_metrics=False,
                no_llm_judge=True,
                judge_models="gpt-5-mini",
                compute_embed_diversity=False,
                silent=True,
                rich_cli=False,
            )

            with (
                patch(
                    "src.metrics.recompute.recompute_results_dir",
                    return_value={"summary": {}, "queries": [], "summary_path": ""},
                ) as recompute_mock,
                patch(
                    "src.submitit_runner.report_merged_results",
                    return_value="/tmp/results_all.json",
                ) as merge_mock,
            ):
                rc = run_merge_only(cli_args)

            self.assertEqual(rc, 0)
            recompute_mock.assert_called_once()
            merge_mock.assert_called_once()
            self.assertTrue(merge_mock.call_args.kwargs["compute_metrics"])


if __name__ == "__main__":
    unittest.main()
