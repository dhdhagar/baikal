"""Tests for SQL QA helpers."""

import unittest
from unittest.mock import patch

from src.sql_qa import decide_sql_needed


class TestDecideSqlNeeded(unittest.TestCase):
    @patch("src.sql_qa.chat")
    def test_defaults_to_sql_on_parse_failure(self, mock_chat) -> None:
        mock_chat.return_value = "not json at all {{{"
        needs_sql = decide_sql_needed(
            llm=object(),
            sub_question="How many rows?",
            passages=[],
            schema_string='CREATE TABLE "T1" (id INT);',
            temperature=0.0,
        )
        self.assertTrue(needs_sql)


if __name__ == "__main__":
    unittest.main()
