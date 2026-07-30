"""Tests for CLI entrypoint validation."""

import io
import sys
import unittest
from unittest.mock import patch


class TestRunValidation(unittest.TestCase):
    def test_opencode_clustering_requires_retrieval(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--method",
            "opencode",
            "--no-opencode_skip_clustering",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_retry_missing_requires_merge_only(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--retry_missing",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_cluster_selection_ucb_requires_judge(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--cluster_selection_method",
            "ucb",
            "--no_llm_judge",
            "--compute_metrics",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_cluster_selection_ucb_requires_compute_metrics(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--cluster_selection_method",
            "ucb",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_use_llm_priors_requires_ucb_or_epsilon(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--use_llm_priors",
            "--cluster_selection_method",
            "llm",
            "--compute_metrics",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_use_llm_priors_with_random_rejected(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--use_llm_priors",
            "--cluster_selection_method",
            "random",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_llm_prior_max_workers_must_be_positive(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--use_llm_priors",
            "--cluster_selection_method",
            "ucb",
            "--compute_metrics",
            "--llm_prior_max_workers",
            "0",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)

    def test_posterior_evidence_weight_must_be_positive(self) -> None:
        from src.run import main

        argv = [
            "run",
            "--use_llm_priors",
            "--cluster_selection_method",
            "ucb",
            "--compute_metrics",
            "--posterior_evidence_weight",
            "0",
            "--output_dir",
            "results",
        ]
        with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main()
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
