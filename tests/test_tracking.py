"""Tests for LLM token/cost tracking."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from src.tracking import (
    UsageSummary,
    UsageTracker,
    format_usage_line,
    get_tracker,
    reset_tracker,
    track_api_response,
)


@dataclass
class _MockUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: Optional[int] = None


@dataclass
class _MockResponse:
    usage: _MockUsage
    choices: List[Any]
    _hidden_params: Optional[Dict[str, Any]] = None


class TestUsageSummary(unittest.TestCase):
    def test_add_and_subtract(self):
        a = UsageSummary(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            time_taken=12.5,
            n_calls=2,
        )
        b = UsageSummary(
            prompt_tokens=40,
            completion_tokens=10,
            total_tokens=50,
            cost_usd=0.004,
            time_taken=4.2,
            n_calls=1,
        )
        delta = a.subtract(b)
        self.assertEqual(delta.prompt_tokens, 60)
        self.assertEqual(delta.completion_tokens, 40)
        self.assertEqual(delta.total_tokens, 100)
        self.assertAlmostEqual(delta.cost_usd, 0.006)
        self.assertAlmostEqual(delta.time_taken, 8.3)
        self.assertEqual(delta.n_calls, 1)


class TestUsageTracker(unittest.TestCase):
    def setUp(self) -> None:
        reset_tracker()

    def test_record_and_aggregate(self):
        tracker = get_tracker()
        tracker.record(
            call_type="completion",
            feature="sql_generate",
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=0.001,
            time_taken=1.25,
        )
        tracker.record(
            call_type="embedding",
            feature="query_embedding",
            provider="openai",
            model="text-embedding-3-small",
            prompt_tokens=50,
            completion_tokens=0,
            cost_usd=0.0002,
            time_taken=0.5,
        )

        self.assertEqual(tracker.total.prompt_tokens, 150)
        self.assertEqual(tracker.total.completion_tokens, 20)
        self.assertEqual(tracker.total.total_tokens, 170)
        self.assertAlmostEqual(tracker.total.cost_usd, 0.0012)
        self.assertAlmostEqual(tracker.total.time_taken, 1.75)
        self.assertEqual(tracker.total.n_calls, 2)
        self.assertEqual(tracker.by_feature["sql_generate"].n_calls, 1)
        self.assertEqual(tracker.by_call_type["embedding"].prompt_tokens, 50)

    def test_snapshot_delta(self):
        tracker = get_tracker()
        start = tracker.snapshot()
        tracker.record(
            call_type="completion",
            feature="final_report",
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.0005,
        )
        delta = tracker.snapshot().subtract(start)
        self.assertEqual(delta.total_tokens, 15)
        self.assertAlmostEqual(delta.cost_usd, 0.0005)

    def test_to_dict_shape(self):
        tracker = get_tracker()
        tracker.record(
            call_type="completion",
            feature="test",
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.0,
        )
        payload = tracker.to_dict()
        self.assertIn("total", payload)
        self.assertIn("by_feature", payload)
        self.assertIn("by_call_type", payload)
        self.assertIn("test", payload["by_feature"])
        self.assertIn("time_taken", payload["total"])


class TestTrackApiResponse(unittest.TestCase):
    def setUp(self) -> None:
        reset_tracker()

    def test_extracts_usage_from_openai_style_response(self):
        resp = _MockResponse(
            usage=_MockUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            choices=[],
        )
        rec = track_api_response(
            resp,
            provider="openai",
            model="gpt-4o-mini",
            call_type="completion",
            feature="subquestion_generation",
            time_taken=2.5,
        )
        self.assertEqual(rec.prompt_tokens, 200)
        self.assertEqual(rec.completion_tokens, 30)
        self.assertEqual(rec.total_tokens, 230)
        self.assertAlmostEqual(rec.time_taken, 2.5)
        self.assertEqual(get_tracker().total.n_calls, 1)

    def test_uses_hidden_response_cost_when_present(self):
        resp = _MockResponse(
            usage=_MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[],
            _hidden_params={"response_cost": 0.042},
        )
        rec = track_api_response(
            resp,
            provider="litellm",
            model="gpt-4o-mini",
            call_type="completion",
            feature="completion",
        )
        self.assertAlmostEqual(rec.cost_usd, 0.042)

    def test_falls_back_to_litellm_cost_per_token(self):
        mock_litellm = MagicMock()
        mock_litellm.completion_cost.side_effect = RuntimeError("no response cost")
        mock_litellm.cost_per_token.return_value = (0.001, 0.002)
        resp = _MockResponse(
            usage=_MockUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            choices=[],
        )
        with patch.dict(sys.modules, {"litellm": mock_litellm}):
            rec = track_api_response(
                resp,
                provider="openai",
                model="gpt-4o-mini",
                call_type="completion",
                feature="completion",
            )
        self.assertAlmostEqual(rec.cost_usd, 0.003)


def _import_llm():
    try:
        import src.llm as llm_mod
    except ImportError as e:
        raise unittest.SkipTest(f"llm dependencies unavailable: {e}") from e
    return llm_mod


def _import_embedding_client():
    try:
        import src.embedding_client as embedding_mod
    except ImportError as e:
        raise unittest.SkipTest(f"embedding dependencies unavailable: {e}") from e
    return embedding_mod


class TestLLMClientIntegration(unittest.TestCase):
    def setUp(self) -> None:
        reset_tracker()

    def test_complete_records_usage(self):
        llm_mod = _import_llm()
        with patch.object(llm_mod, "OpenAI") as mock_openai_cls:
            from src.llm import LLMClient

            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = _MockResponse(
                usage=_MockUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
                choices=[MagicMock(message=MagicMock(content="hello"))],
            )

            llm = LLMClient(
                provider="openai",
                model="gpt-4o-mini",
                api_key="test-key",
                api_base="https://api.openai.com/v1",
            )
            out = llm.complete(
                [{"role": "user", "content": "hi"}],
                feature="sql_generate",
            )

            self.assertEqual(out, "hello")
            tracker = get_tracker()
            self.assertEqual(tracker.total.n_calls, 1)
            self.assertEqual(tracker.total.total_tokens, 20)
            self.assertEqual(tracker.by_feature["sql_generate"].n_calls, 1)


class TestEmbeddingClientIntegration(unittest.TestCase):
    def setUp(self) -> None:
        reset_tracker()

    def test_encode_records_embedding_usage(self):
        embedding_mod = _import_embedding_client()
        with patch.object(embedding_mod, "OpenAI") as mock_openai_cls:
            from src.embedding_client import EmbeddingClient

            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            row = MagicMock()
            row.index = 0
            row.embedding = [0.1, 0.2]
            mock_client.embeddings.create.return_value = MagicMock(
                data=[row],
                usage=_MockUsage(prompt_tokens=64, completion_tokens=0, total_tokens=64),
            )

            client = EmbeddingClient(
                provider="openai",
                model="text-embedding-3-small",
                api_key="test-key",
                api_base="https://api.openai.com/v1",
            )
            out = client.encode(["hello world"], feature="query_embedding")

            self.assertEqual(out.shape, (1, 2))
            tracker = get_tracker()
            self.assertEqual(tracker.total.n_calls, 1)
            self.assertEqual(tracker.total.prompt_tokens, 64)
            self.assertEqual(tracker.by_call_type["embedding"].n_calls, 1)


class TestFormatting(unittest.TestCase):
    def test_format_usage_line(self):
        summary = UsageSummary(
            prompt_tokens=1000,
            completion_tokens=250,
            total_tokens=1250,
            cost_usd=0.0345,
            time_taken=42.7,
            n_calls=3,
        )
        line = format_usage_line(summary)
        self.assertIn("1,250 tokens", line)
        self.assertIn("1,000 in", line)
        self.assertIn("250 out", line)
        self.assertIn("$0.0345", line)
        self.assertIn("42.7s", line)


if __name__ == "__main__":
    unittest.main()
