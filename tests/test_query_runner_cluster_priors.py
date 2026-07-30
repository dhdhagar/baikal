"""Tests for parallel LLM prior elicitation in QueryRunner."""

import json
import os
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from src.query_runner import QueryRunner


def _empty_counts(**overrides) -> dict:
    counts = {
        "definitely_yes": 0,
        "maybe_yes": 0,
        "uncertain": 0,
        "maybe_not": 0,
        "definitely_not": 0,
        "cannot_comment": 0,
    }
    counts.update(overrides)
    return counts


def _elicit_result(
    alpha: float,
    beta: float,
    counts: dict | None = None,
    *,
    prompt: str = "user prompt",
    system_prompt: str = "system prompt",
) -> tuple[float, float, dict, str, str]:
    return alpha, beta, counts or _empty_counts(), prompt, system_prompt


def _prior_args(**overrides) -> Namespace:
    defaults = dict(
        seed=42,
        silent=True,
        rich_cli=False,
        temperature=1.0,
        use_passages=False,
        tables_lake_dir="/tmp",
        llm_provider="openai",
        llm_model="gpt-5-mini",
        embedding_provider="local",
        embedding_model="test-model",
        gpu=False,
        use_llm_priors=True,
        llm_prior_n_samples=2,
        llm_prior_max_workers=2,
        bandit_reward="finding",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestQueryRunnerClusterPriors(unittest.TestCase):
    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    @patch("src.query_runner.elicit_cluster_prior")
    def test_elicit_cluster_priors_runs_in_parallel(
        self,
        mock_elicit,
        mock_get_llm,
        mock_get_embedder,
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()
        active = 0
        peak = 0
        lock = threading.Lock()

        def _elicit_side_effect(*_args, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return _elicit_result(
                1.5,
                0.5,
                _empty_counts(maybe_yes=1),
                prompt="prompt for cluster",
            )

        mock_elicit.side_effect = _elicit_side_effect
        clusters = [
            {"cluster_id": f"c{i}", "description": "x", "tables": [], "passages": []}
            for i in range(3)
        ]
        runner = QueryRunner(_prior_args(llm_prior_max_workers=2), clusters, tempfile.mkdtemp())
        runner.candidate_clusters = clusters
        with tempfile.TemporaryDirectory() as out_dir:
            runner._elicit_cluster_priors("Q?", out_dir)
            self.assertEqual(mock_elicit.call_count, 3)
            self.assertGreaterEqual(peak, 2)
            priors_path = os.path.join(out_dir, "cluster_priors.json")
            with open(priors_path, encoding="utf-8") as handle:
                priors = json.load(handle)
            self.assertEqual(set(priors), {"c0", "c1", "c2"})
            self.assertAlmostEqual(priors["c0"]["mean"], 1.5 / 2.0)
            self.assertEqual(priors["c0"]["prompt"], "prompt for cluster")

    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    @patch("src.query_runner.elicit_cluster_prior")
    def test_elicit_cluster_priors_sequential_when_max_workers_one(
        self,
        mock_elicit,
        mock_get_llm,
        mock_get_embedder,
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()
        active = 0
        peak = 0
        lock = threading.Lock()

        def _elicit_side_effect(*_args, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return _elicit_result(0.5, 0.5)

        mock_elicit.side_effect = _elicit_side_effect
        clusters = [
            {"cluster_id": f"c{i}", "description": "x", "tables": [], "passages": []}
            for i in range(2)
        ]
        runner = QueryRunner(_prior_args(llm_prior_max_workers=1), clusters, tempfile.mkdtemp())
        runner.candidate_clusters = clusters
        with tempfile.TemporaryDirectory() as out_dir:
            runner._elicit_cluster_priors("Q?", out_dir)

        self.assertEqual(mock_elicit.call_count, 2)
        self.assertEqual(peak, 1)

    @patch("src.query_runner.get_embedding_client")
    @patch("src.query_runner.get_llm_client")
    @patch("src.query_runner.elicit_cluster_prior")
    def test_cluster_priors_json_sorted_by_mean_descending(
        self,
        mock_elicit,
        mock_get_llm,
        mock_get_embedder,
    ) -> None:
        mock_get_llm.return_value = MagicMock()
        mock_get_embedder.return_value = MagicMock()

        def _elicit_side_effect(_llm, *, cluster, **_kwargs):
            by_cluster = {
                "low": _elicit_result(1.0, 1.0),
                "mid": _elicit_result(1.5, 0.5),
                "high": _elicit_result(2.0, 0.5),
            }
            return by_cluster[cluster["cluster_id"]]

        mock_elicit.side_effect = _elicit_side_effect
        clusters = [
            {"cluster_id": "low", "description": "x", "tables": [], "passages": []},
            {"cluster_id": "high", "description": "x", "tables": [], "passages": []},
            {"cluster_id": "mid", "description": "x", "tables": [], "passages": []},
        ]
        runner = QueryRunner(_prior_args(), clusters, tempfile.mkdtemp())
        runner.candidate_clusters = clusters
        with tempfile.TemporaryDirectory() as out_dir:
            runner._elicit_cluster_priors("Q?", out_dir)
            with open(
                os.path.join(out_dir, "cluster_priors.json"),
                encoding="utf-8",
            ) as handle:
                priors = json.load(handle)
        self.assertEqual(list(priors), ["high", "mid", "low"])
        means = [entry["mean"] for entry in priors.values()]
        self.assertEqual(means, sorted(means, reverse=True))


if __name__ == "__main__":
    unittest.main()
