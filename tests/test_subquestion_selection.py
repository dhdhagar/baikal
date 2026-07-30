import random
import unittest
from unittest.mock import patch

from src.subquestions import select_subquestion

SAMPLE_CLUSTER = {
    "description": "Stadium data",
    "tables": [
        {
            "table_id": "T1",
            "title": "Stadiums",
            "columns": ["name", "capacity"],
            "description": "Stadium records",
        }
    ],
    "passages": [],
}


class TestSelectSubquestion(unittest.TestCase):
    def test_random_is_deterministic_with_seed(self) -> None:
        candidates = ["Q1", "Q2", "Q3"]
        rng = random.Random(42)
        first = select_subquestion(candidates, "random", rng)
        rng = random.Random(42)
        second = select_subquestion(candidates, "random", rng)
        self.assertEqual(first, second)
        self.assertIn(first, candidates)

    def test_single_candidate_skips_selection(self) -> None:
        self.assertEqual(
            select_subquestion(["Only one"], "llm", random.Random(0)),
            "Only one",
        )

    def test_empty_candidates_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_subquestion([], "random", random.Random(0))

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_subquestion(["Q1", "Q2"], "ucb", random.Random(0))

    def test_llm_requires_client(self) -> None:
        with self.assertRaises(ValueError):
            select_subquestion(["Q1", "Q2"], "llm", random.Random(0))

    @patch("src.subquestions.chat")
    def test_llm_returns_selected_candidate(self, mock_chat) -> None:
        mock_chat.return_value = '{"selected_index": 2}'
        candidates = ["Q1", "Q2", "Q3"]
        result = select_subquestion(
            candidates,
            "llm",
            random.Random(0),
            llm=object(),
            user_query="What is the research topic?",
            cluster=SAMPLE_CLUSTER,
            previous_subquestions=["Prior Q"],
            temperature=0.0,
        )
        self.assertEqual(result, "Q2")
        prompt = mock_chat.call_args[0][1]
        self.assertIn("What is the research topic?", prompt)
        self.assertIn("SELECTED CLUSTER CONTEXT:", prompt)
        self.assertIn("Stadiums", prompt)
        self.assertIn("Prior Q", prompt)
        self.assertIn("Q2", prompt)
        self.assertEqual(mock_chat.call_args.kwargs["feature"], "subquestion_selection")

    @patch("src.subquestions.chat")
    def test_llm_invalid_index_falls_back_to_random(self, mock_chat) -> None:
        mock_chat.return_value = '{"selected_index": 99}'
        candidates = ["Q1", "Q2"]
        rng = random.Random(7)
        result = select_subquestion(
            candidates,
            "llm",
            rng,
            llm=object(),
            user_query="Research?",
            temperature=0.0,
        )
        self.assertEqual(result, "Q2")

    @patch("src.subquestions.chat")
    def test_llm_parse_error_falls_back_to_random(self, mock_chat) -> None:
        mock_chat.return_value = "not json"
        candidates = ["Q1", "Q2"]
        rng = random.Random(7)
        result = select_subquestion(
            candidates,
            "llm",
            rng,
            llm=object(),
            user_query="Research?",
            temperature=0.0,
        )
        self.assertEqual(result, "Q2")

    @patch("src.subquestions.chat")
    @patch("src.subquestions.log")
    def test_llm_logs_success(self, mock_log, mock_chat) -> None:
        mock_chat.return_value = '{"selected_index": 1}'
        select_subquestion(
            ["Q1", "Q2"],
            "llm",
            random.Random(0),
            llm=object(),
            user_query="Research?",
            temperature=0.0,
        )
        mock_log.assert_called_once_with(
            "LLM selected sub-question 1 of 2.",
            silent=False,
        )

    @patch("src.subquestions.chat")
    @patch("src.subquestions.log")
    def test_llm_logs_fallback(self, mock_log, mock_chat) -> None:
        mock_chat.return_value = '{"selected_index": 99}'
        select_subquestion(
            ["Q1", "Q2"],
            "llm",
            random.Random(7),
            llm=object(),
            user_query="Research?",
            temperature=0.0,
        )
        mock_log.assert_called_once()
        self.assertIn("falling back to random", mock_log.call_args[0][0])

    @patch("src.subquestions.chat")
    def test_llm_no_prior_subquestions_shows_none(self, mock_chat) -> None:
        mock_chat.return_value = '{"selected_index": 1}'
        select_subquestion(
            ["Q1", "Q2"],
            "llm",
            random.Random(0),
            llm=object(),
            user_query="Research?",
            temperature=0.0,
        )
        prompt = mock_chat.call_args[0][1]
        self.assertIn("(none)", prompt)


if __name__ == "__main__":
    unittest.main()
