import unittest
from unittest.mock import patch

from src.cluster_prior import (
    CANNOT_COMMENT_KEY,
    UNINFORMED_PRIOR_ALPHA,
    UNINFORMED_PRIOR_BETA,
    categorical_counts_to_beta_params,
    elicit_cluster_prior,
    parse_belief_label,
)


class TestCategoricalPrior(unittest.TestCase):
    def test_parse_belief_label(self) -> None:
        self.assertEqual(
            parse_belief_label('{"belief": "definitely yes"}'),
            "definitely yes",
        )
        self.assertEqual(
            parse_belief_label('{"belief": "cannot comment"}'),
            "cannot comment",
        )
        self.assertIsNone(parse_belief_label('{"belief": "invalid"}'))

    def test_categorical_counts_to_beta_params(self) -> None:
        alpha, beta = categorical_counts_to_beta_params(
            {
                "definitely_yes": 2,
                "maybe_yes": 0,
                "uncertain": 0,
                "maybe_not": 0,
                "definitely_not": 0,
                CANNOT_COMMENT_KEY: 5,
            }
        )
        self.assertAlmostEqual(alpha, UNINFORMED_PRIOR_ALPHA + 2.0)
        self.assertAlmostEqual(beta, UNINFORMED_PRIOR_BETA)

    def test_mixed_counts(self) -> None:
        alpha, beta = categorical_counts_to_beta_params(
            {
                "definitely_yes": 1,
                "maybe_yes": 1,
                "uncertain": 1,
                "maybe_not": 0,
                "definitely_not": 0,
                CANNOT_COMMENT_KEY: 0,
            }
        )
        self.assertAlmostEqual(alpha, UNINFORMED_PRIOR_ALPHA + 2.25)
        self.assertAlmostEqual(beta, UNINFORMED_PRIOR_BETA + 0.75)

    @patch("src.cluster_prior.chat_many")
    def test_elicit_cluster_prior(self, mock_chat_many) -> None:
        mock_chat_many.return_value = [
            '{"belief": "maybe yes"}',
            '{"belief": "maybe yes"}',
        ]
        alpha, beta, counts, prompt, system_prompt = elicit_cluster_prior(
            object(),
            user_query="Q?",
            cluster={"cluster_id": "c1", "description": "x", "tables": [], "passages": []},
            bandit_reward="finding",
            n_samples=2,
            temperature=1.0,
            silent=True,
        )
        self.assertAlmostEqual(alpha, UNINFORMED_PRIOR_ALPHA + 2 * 0.75)
        self.assertAlmostEqual(beta, UNINFORMED_PRIOR_BETA + 0.5)
        self.assertEqual(counts["maybe_yes"], 2)
        self.assertIn("RESEARCH QUESTION:", prompt)
        self.assertIn("c1", prompt)
        self.assertIn("research scientist", system_prompt)
        self.assertEqual(mock_chat_many.call_args.kwargs["n"], 2)
        self.assertEqual(mock_chat_many.call_args.kwargs["feature"], "cluster_prior")

    @patch("src.cluster_prior.chat_many")
    def test_elicit_cluster_prior_tracks_cannot_comment(self, mock_chat_many) -> None:
        mock_chat_many.return_value = [
            '{"belief": "maybe yes"}',
            '{"belief": "cannot comment"}',
            '{"belief": "cannot comment"}',
        ]
        alpha, beta, counts, _, _ = elicit_cluster_prior(
            object(),
            user_query="Q?",
            cluster={"cluster_id": "c1", "description": "x", "tables": [], "passages": []},
            bandit_reward="finding",
            n_samples=3,
            temperature=1.0,
            silent=True,
        )
        self.assertAlmostEqual(alpha, UNINFORMED_PRIOR_ALPHA + 0.75)
        self.assertAlmostEqual(beta, UNINFORMED_PRIOR_BETA + 0.25)
        self.assertEqual(counts["maybe_yes"], 1)
        self.assertEqual(counts[CANNOT_COMMENT_KEY], 2)

    @patch("src.cluster_prior.chat_many")
    def test_elicit_cluster_prior_all_invalid_uses_uninformed_prior(
        self, mock_chat_many
    ) -> None:
        mock_chat_many.return_value = ['{"belief": "cannot comment"}'] * 3
        alpha, beta, counts, _, _ = elicit_cluster_prior(
            object(),
            user_query="Q?",
            cluster={"cluster_id": "c1", "description": "x", "tables": [], "passages": []},
            bandit_reward="finding",
            n_samples=3,
            temperature=1.0,
            silent=True,
        )
        self.assertAlmostEqual(alpha, UNINFORMED_PRIOR_ALPHA)
        self.assertAlmostEqual(beta, UNINFORMED_PRIOR_BETA)
        self.assertEqual(counts[CANNOT_COMMENT_KEY], 3)


if __name__ == "__main__":
    unittest.main()
