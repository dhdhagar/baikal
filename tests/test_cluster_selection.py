import random
import unittest
from unittest.mock import patch

from src.cluster_selection import (
    ClusterSelectionState,
    bandit_reward_from_iteration,
    select_cluster,
)

SAMPLE_CLUSTERS = [
    {"cluster_id": "c1", "description": "Sports stadiums", "tables": [], "passages": []},
    {"cluster_id": "c2", "description": "Team records", "tables": [], "passages": []},
]


def _iteration_with_rubric(
    *,
    finding_score: float = 0.5,
    grounded: float = 1.0,
    relevance: float = 0.75,
    distinctness: float = 0.5,
    usefulness: float = 0.25,
) -> dict:
    return {
        "metrics": {
            "research_quality": {
                "finding_score": finding_score,
                "rubric": {
                    "judges": [
                        {
                            "scores": {
                                "grounded": grounded,
                                "relevance": relevance,
                                "distinctness": distinctness,
                                "report_usefulness": usefulness,
                            }
                        }
                    ]
                },
            }
        }
    }


class TestBanditReward(unittest.TestCase):
    def test_finding_reward(self) -> None:
        it = _iteration_with_rubric(finding_score=0.42)
        self.assertEqual(bandit_reward_from_iteration(it, "finding"), 0.42)

    def test_grounded_component_rewards(self) -> None:
        it = _iteration_with_rubric(
            grounded=1.0, relevance=0.75, distinctness=0.5, usefulness=0.25
        )
        self.assertEqual(bandit_reward_from_iteration(it, "relevance"), 0.75)
        self.assertEqual(bandit_reward_from_iteration(it, "distinctness"), 0.5)
        self.assertEqual(bandit_reward_from_iteration(it, "usefulness"), 0.25)

    def test_ungrounded_component_rewards_are_zero(self) -> None:
        it = _iteration_with_rubric(grounded=0.0, relevance=0.75)
        self.assertEqual(bandit_reward_from_iteration(it, "relevance"), 0.0)

    def test_missing_metrics_returns_zero(self) -> None:
        self.assertEqual(bandit_reward_from_iteration({}, "finding"), 0.0)


class TestClusterSelectionState(unittest.TestCase):
    def test_record_visit_and_outcome(self) -> None:
        state = ClusterSelectionState()
        self.assertEqual(state.record_visit("c1"), 1)
        state.record_outcome("c1", 0.8)
        state.record_visit("c1")
        state.record_outcome("c1", 0.4)
        self.assertEqual(state.visits["c1"], 2)
        self.assertAlmostEqual(state.avg_reward("c1"), 0.6)


class TestSelectCluster(unittest.TestCase):
    def test_random_is_deterministic_with_seed(self) -> None:
        rng = random.Random(42)
        first = select_cluster(SAMPLE_CLUSTERS, "random", rng)
        rng = random.Random(42)
        second = select_cluster(SAMPLE_CLUSTERS, "random", rng)
        self.assertEqual(first, second)

    def test_empty_candidates_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_cluster([], "random", random.Random(0))

    def test_ucb_requires_state(self) -> None:
        with self.assertRaises(ValueError):
            select_cluster(SAMPLE_CLUSTERS, "ucb", random.Random(0))

    def test_ucb_prefers_unvisited(self) -> None:
        state = ClusterSelectionState()
        state.record_visit("c1")
        state.record_outcome("c1", 1.0)
        chosen = select_cluster(
            SAMPLE_CLUSTERS, "ucb", random.Random(0), state, ucb_c=1.0
        )
        self.assertEqual(chosen["cluster_id"], "c2")

    def test_ucb_prefers_high_reward_after_visits(self) -> None:
        state = ClusterSelectionState()
        for _ in range(3):
            state.record_visit("c1")
            state.record_outcome("c1", 0.1)
        for _ in range(3):
            state.record_visit("c2")
            state.record_outcome("c2", 0.9)
        chosen = select_cluster(
            SAMPLE_CLUSTERS, "ucb", random.Random(0), state, ucb_c=0.01
        )
        self.assertEqual(chosen["cluster_id"], "c2")

    def test_epsilon_zero_is_greedy(self) -> None:
        state = ClusterSelectionState()
        state.record_visit("c1")
        state.record_outcome("c1", 0.2)
        state.record_visit("c2")
        state.record_outcome("c2", 0.9)
        chosen = select_cluster(
            SAMPLE_CLUSTERS, "epsilon-greedy", random.Random(0), state, epsilon=0.0
        )
        self.assertEqual(chosen["cluster_id"], "c2")

    def test_epsilon_one_is_random(self) -> None:
        state = ClusterSelectionState()
        state.record_visit("c2")
        state.record_outcome("c2", 1.0)
        rng = random.Random(1)
        chosen = select_cluster(
            SAMPLE_CLUSTERS, "epsilon-greedy", rng, state, epsilon=1.0
        )
        self.assertIn(chosen["cluster_id"], {"c1", "c2"})

    def test_bayes_ucb_prefers_higher_posterior(self) -> None:
        state = ClusterSelectionState()
        state.set_prior("c1", 1.0, 9.0)
        state.set_prior("c2", 9.0, 1.0)
        state.record_visit("c1")
        state.record_visit("c2")
        chosen = select_cluster(
            SAMPLE_CLUSTERS,
            "ucb",
            random.Random(0),
            state,
            use_llm_priors=True,
        )
        self.assertEqual(chosen["cluster_id"], "c2")

    def test_epsilon_greedy_with_priors_is_greedy(self) -> None:
        state = ClusterSelectionState()
        state.set_prior("c1", 1.0, 9.0)
        state.set_prior("c2", 9.0, 1.0)
        chosen = select_cluster(
            SAMPLE_CLUSTERS,
            "epsilon-greedy",
            random.Random(0),
            state,
            epsilon=0.0,
            use_llm_priors=True,
        )
        self.assertEqual(chosen["cluster_id"], "c2")

    def test_epsilon_greedy_with_priors_explore_softmax(self) -> None:
        state = ClusterSelectionState()
        state.set_prior("c1", 1.0, 9.0)
        state.set_prior("c2", 9.0, 1.0)
        chosen = select_cluster(
            SAMPLE_CLUSTERS,
            "epsilon-greedy",
            random.Random(42),
            state,
            epsilon=1.0,
            use_llm_priors=True,
            llm_prior_tau=10.0,
        )
        self.assertEqual(chosen["cluster_id"], "c2")

    def test_posterior_update(self) -> None:
        state = ClusterSelectionState()
        state.set_prior("c1", 1.0, 1.0)
        state.update_posterior("c1", 0.8, weight=1.0)
        self.assertAlmostEqual(state.alpha["c1"], 1.8)
        self.assertAlmostEqual(state.beta["c1"], 1.2)

    def test_posterior_update_with_evidence_weight(self) -> None:
        state = ClusterSelectionState()
        state.set_prior("c1", 1.0, 1.0)
        state.update_posterior("c1", 0.8, weight=5.0)
        self.assertAlmostEqual(state.alpha["c1"], 1.0 + 5.0 * 0.8)
        self.assertAlmostEqual(state.beta["c1"], 1.0 + 5.0 * 0.2)

    def test_llm_requires_client(self) -> None:
        state = ClusterSelectionState()
        with self.assertRaises(ValueError):
            select_cluster(SAMPLE_CLUSTERS, "llm", random.Random(0), state)

    @patch("src.cluster_selection.chat")
    def test_llm_returns_selected_cluster(self, mock_chat) -> None:
        mock_chat.return_value = '{"selected_index": 2}'
        state = ClusterSelectionState()
        chosen = select_cluster(
            SAMPLE_CLUSTERS,
            "llm",
            random.Random(0),
            state,
            llm=object(),
            user_query="Research topic?",
            bandit_reward="relevance",
        )
        self.assertEqual(chosen["cluster_id"], "c2")
        prompt = mock_chat.call_args[0][1]
        self.assertIn("Research topic?", prompt)
        self.assertIn("avg_grounded_relevance", prompt)
        self.assertEqual(mock_chat.call_args.kwargs["feature"], "cluster_selection")

    @patch("src.cluster_selection.chat")
    def test_llm_invalid_index_falls_back_to_random(self, mock_chat) -> None:
        mock_chat.return_value = '{"selected_index": 99}'
        state = ClusterSelectionState()
        rng = random.Random(7)
        chosen = select_cluster(
            SAMPLE_CLUSTERS,
            "llm",
            rng,
            state,
            llm=object(),
            user_query="Q",
        )
        self.assertIn(chosen["cluster_id"], {"c1", "c2"})


if __name__ == "__main__":
    unittest.main()
