"""Tests for LLM JSON parsing and sub-question generation resilience."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.llm import (
    LLMClient,
    _ensure_embedding_env_vars,
    _has_embedding_credentials,
    _resolve_credentials,
    chat_many,
    parse_json_dict_from_llm,
    parse_json_from_llm,
    strip_code_fence,
)
from src.tracking import get_tracker, reset_tracker
from src.subquestions import (
    SubquestionGenerationResult,
    filter_semantically_duplicate_subquestions,
    generate_subquestions,
)


class TestResolveCredentials(unittest.TestCase):
    _LLM_ENV = {
        "LLM_API_KEY": "test-key",
        "LLM_API_BASE": "https://api.example.com/v1",
    }

    def _clear_llm_env(self):
        return patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "",
                "LLM_API_BASE": "",
                "EMBEDDING_API_KEY": "",
                "EMBEDDING_API_BASE": "",
                "ANTHROPIC_API_KEY": "",
            },
            clear=False,
        )

    def test_shell_exports_without_dotenv(self):
        with self._clear_llm_env():
            with patch("src.llm._ENV_FILE") as env_file:
                env_file.is_file.return_value = False
                with patch.dict(os.environ, self._LLM_ENV, clear=False):
                    api_key, api_base = _resolve_credentials("openai")
        self.assertEqual(api_key, "test-key")
        self.assertEqual(api_base, "https://api.example.com/v1")

    def test_missing_exports_and_dotenv_raises(self):
        with self._clear_llm_env():
            with patch("src.llm._ENV_FILE") as env_file:
                env_file.is_file.return_value = False
                with self.assertRaises(RuntimeError) as ctx:
                    _resolve_credentials("openai")
        self.assertIn("export", str(ctx.exception).lower())

    def test_loads_credentials_from_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "LLM_API_KEY=from-file\nLLM_API_BASE=https://file.example.com/v1\n",
                encoding="utf-8",
            )
            with self._clear_llm_env():
                with patch("src.llm._ENV_FILE", env_file):
                    api_key, api_base = _resolve_credentials("openai")
        self.assertEqual(api_key, "from-file")
        self.assertEqual(api_base, "https://file.example.com/v1")

    def test_shell_exports_override_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "LLM_API_KEY=from-file\nLLM_API_BASE=https://file.example.com/v1\n",
                encoding="utf-8",
            )
            with patch("src.llm._ENV_FILE", env_file):
                with patch.dict(os.environ, self._LLM_ENV, clear=False):
                    api_key, api_base = _resolve_credentials("openai")
        self.assertEqual(api_key, "test-key")
        self.assertEqual(api_base, "https://api.example.com/v1")

    def test_vllm_uses_defaults_without_credentials(self):
        with self._clear_llm_env():
            with patch("src.llm._ENV_FILE") as env_file:
                env_file.is_file.return_value = False
                api_key, api_base = _resolve_credentials("vllm")
        self.assertEqual(api_key, "EMPTY")
        self.assertEqual(api_base, "http://127.0.0.1:8000/v1")

    def test_normalize_empty_api_base_stays_empty(self):
        from src.llm import _normalize_openai_api_base

        self.assertEqual(_normalize_openai_api_base(""), "")
        self.assertEqual(_normalize_openai_api_base("https://api.example.com"), "https://api.example.com/v1")

    def test_get_llm_client_anthropic_uses_anthropic_key(self):
        from src.llm import get_llm_client

        with self._clear_llm_env():
            with patch("src.llm._ENV_FILE") as env_file:
                env_file.is_file.return_value = False
                with patch.dict(
                    os.environ,
                    {"ANTHROPIC_API_KEY": "anthropic-secret"},
                    clear=False,
                ):
                    client = get_llm_client(
                        "litellm", "anthropic/claude-sonnet-4-20250514"
                    )
        self.assertEqual(client.provider, "litellm")
        self.assertEqual(client.api_key, "anthropic-secret")
        self.assertEqual(client.api_base, "")

    def test_get_llm_client_anthropic_missing_key_raises(self):
        from src.llm import get_llm_client

        with self._clear_llm_env():
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
                with patch("src.llm._ENV_FILE") as env_file:
                    env_file.is_file.return_value = False
                    with self.assertRaises(RuntimeError) as ctx:
                        get_llm_client("litellm", "anthropic/claude-sonnet-4-20250514")
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_embedding_exports_without_dotenv(self):
        embedding_env = {
            "EMBEDDING_API_KEY": "embed-key",
            "EMBEDDING_API_BASE": "https://embed.example.com/v1",
        }
        with self._clear_llm_env():
            with patch("src.llm._ENV_FILE") as env_file:
                env_file.is_file.return_value = False
                with patch.dict(os.environ, embedding_env, clear=False):
                    _ensure_embedding_env_vars()
                    self.assertTrue(_has_embedding_credentials())
                    self.assertEqual(os.environ["EMBEDDING_API_KEY"], "embed-key")


class TestCompleteMany(unittest.TestCase):
    def setUp(self) -> None:
        reset_tracker()

    def _openai_llm(self, model: str = "gpt-4o-mini") -> LLMClient:
        return LLMClient(
            provider="openai",
            model=model,
            api_key="test-key",
            api_base="https://api.example.com/v1",
        )

    @patch("src.llm.OpenAI")
    def test_complete_many_returns_multiple_choices(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            usage=MagicMock(
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18,
            ),
            choices=[
                MagicMock(message=MagicMock(content='{"belief": "maybe yes"}')),
                MagicMock(message=MagicMock(content='{"belief": "definitely yes"}')),
            ],
        )
        llm = self._openai_llm()
        texts = llm.complete_many(
            [{"role": "user", "content": "hi"}],
            n=2,
            feature="cluster_prior",
        )
        self.assertEqual(len(texts), 2)
        self.assertIn("maybe yes", texts[0])
        create_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(create_kwargs["n"], 2)
        self.assertEqual(get_tracker().total.n_calls, 1)

    @patch("src.llm.OpenAI")
    def test_complete_many_n_one_omits_n_kwarg(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[MagicMock(message=MagicMock(content="hello"))],
        )
        llm = self._openai_llm()
        texts = llm.complete_many([{"role": "user", "content": "hi"}], n=1)
        self.assertEqual(texts, ["hello"])
        create_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertNotIn("n", create_kwargs)

    @patch("src.llm.OpenAI")
    def test_complete_delegates_to_complete_many(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[MagicMock(message=MagicMock(content="one"))],
        )
        llm = self._openai_llm()
        self.assertEqual(
            llm.complete([{"role": "user", "content": "hi"}]),
            "one",
        )

    @patch("src.llm.OpenAI")
    def test_gpt5_complete_many_uses_batched_n(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            usage=MagicMock(prompt_tokens=3, completion_tokens=3, total_tokens=6),
            choices=[
                MagicMock(message=MagicMock(content="a")),
                MagicMock(message=MagicMock(content="b")),
                MagicMock(message=MagicMock(content="c")),
            ],
        )
        llm = self._openai_llm(model="gpt-5-mini")
        texts = llm.complete_many([{"role": "user", "content": "hi"}], n=3)
        self.assertEqual(texts, ["a", "b", "c"])
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        self.assertEqual(
            mock_client.chat.completions.create.call_args.kwargs["n"], 3
        )

    @patch("litellm.completion")
    def test_litellm_complete_many(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            usage=MagicMock(prompt_tokens=4, completion_tokens=4, total_tokens=8),
            choices=[
                MagicMock(message=MagicMock(content="a")),
                MagicMock(message=MagicMock(content="b")),
            ],
        )
        llm = LLMClient(
            provider="litellm",
            model="azure/gpt-4o",
            api_key="key",
            api_base="https://api.example.com/v1",
        )
        texts = llm.complete_many([{"role": "user", "content": "hi"}], n=2)
        self.assertEqual(texts, ["a", "b"])
        self.assertEqual(mock_completion.call_args.kwargs["n"], 2)

    @patch("litellm.get_supported_openai_params", create=True)
    @patch("litellm.completion")
    def test_litellm_falls_back_when_model_does_not_support_n(
        self,
        mock_completion,
        mock_supported_params,
    ) -> None:
        mock_supported_params.return_value = ["temperature", "max_tokens"]
        mock_completion.side_effect = [
            MagicMock(
                usage=MagicMock(
                    prompt_tokens=2,
                    completion_tokens=1,
                    total_tokens=3,
                ),
                choices=[MagicMock(message=MagicMock(content=text))],
            )
            for text in ("a", "b", "c")
        ]
        llm = LLMClient(
            provider="litellm",
            model="bedrock/google.gemma-3-4b-it",
            api_key="key",
            api_base="",
        )

        texts = llm.complete_many([{"role": "user", "content": "hi"}], n=3)

        self.assertEqual(texts, ["a", "b", "c"])
        self.assertEqual(mock_completion.call_count, 3)
        self.assertTrue(
            all("n" not in call.kwargs for call in mock_completion.call_args_list)
        )
        self.assertEqual(get_tracker().total.n_calls, 3)

    @patch("litellm.get_supported_openai_params", create=True)
    @patch("litellm.completion")
    def test_litellm_retries_separately_when_n_is_rejected(
        self,
        mock_completion,
        mock_supported_params,
    ) -> None:
        unsupported_params_error = type(
            "UnsupportedParamsError", (Exception,), {}
        )
        mock_supported_params.return_value = ["n", "temperature"]
        mock_completion.side_effect = [
            unsupported_params_error(
                "bedrock does not support parameters: ['n']"
            ),
            MagicMock(
                usage=MagicMock(
                    prompt_tokens=2,
                    completion_tokens=1,
                    total_tokens=3,
                ),
                choices=[MagicMock(message=MagicMock(content="a"))],
            ),
            MagicMock(
                usage=MagicMock(
                    prompt_tokens=2,
                    completion_tokens=1,
                    total_tokens=3,
                ),
                choices=[MagicMock(message=MagicMock(content="b"))],
            ),
        ]
        llm = LLMClient(
            provider="litellm",
            model="custom/model",
            api_key="key",
            api_base="",
        )

        texts = llm.complete_many([{"role": "user", "content": "hi"}], n=2)

        self.assertEqual(texts, ["a", "b"])
        self.assertEqual(mock_completion.call_count, 3)
        self.assertEqual(mock_completion.call_args_list[0].kwargs["n"], 2)
        self.assertTrue(
            all(
                "n" not in call.kwargs
                for call in mock_completion.call_args_list[1:]
            )
        )
        self.assertEqual(get_tracker().total.n_calls, 2)

    @patch("litellm.completion")
    def test_litellm_anthropic_skips_api_base_and_json_mode(self, mock_completion) -> None:
        mock_completion.return_value = MagicMock(
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            choices=[MagicMock(message=MagicMock(content='{"ok": true}'))],
        )
        llm = LLMClient(
            provider="litellm",
            model="anthropic/claude-sonnet-4-20250514",
            api_key="anthropic-secret",
            api_base="",
        )
        texts = llm.complete_many(
            [{"role": "user", "content": "hi"}],
            n=1,
            json_mode=True,
        )
        self.assertEqual(texts, ['{"ok": true}'])
        kwargs = mock_completion.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "anthropic-secret")
        self.assertNotIn("api_base", kwargs)
        self.assertNotIn("response_format", kwargs)

    @patch("src.llm.OpenAI")
    def test_chat_many(self, mock_openai_cls) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            usage=MagicMock(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            choices=[
                MagicMock(message=MagicMock(content="first")),
                MagicMock(message=MagicMock(content="second")),
            ],
        )
        llm = self._openai_llm()
        texts = chat_many(llm, "prompt", n=2, feature="cluster_prior")
        self.assertEqual(texts, ["first", "second"])
        messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[1]["content"], "prompt")


class TestParseJsonFromLlm(unittest.TestCase):
    def test_parses_valid_json(self):
        data = parse_json_from_llm('{"subquestions": ["q1", "q2"]}')
        self.assertEqual(data["subquestions"], ["q1", "q2"])

    def test_strips_code_fence(self):
        raw = '```json\n{"needs_sql": false}\n```'
        self.assertFalse(parse_json_from_llm(raw)["needs_sql"])

    def test_repairs_malformed_json(self):
        raw = '{"subquestions": ["What is the "best" stadium?", "q2"]}'
        data = parse_json_from_llm(raw)
        self.assertIn("subquestions", data)
        self.assertEqual(len(data["subquestions"]), 2)

    def test_raises_on_empty(self):
        with self.assertRaises(ValueError):
            parse_json_from_llm("   ")

    def test_strip_code_fence_only(self):
        self.assertEqual(strip_code_fence("```\n{}\n```"), "{}")


class TestParseJsonDictFromLlm(unittest.TestCase):
    def test_parses_dict(self):
        data = parse_json_dict_from_llm('{"expand": false}')
        self.assertFalse(data["expand"])

    def test_raises_on_garbage(self):
        with self.assertRaises(ValueError):
            parse_json_dict_from_llm("not json at all {{{")

    def test_raises_on_list(self):
        with self.assertRaises(ValueError):
            parse_json_dict_from_llm("[1, 2, 3]")


class TestGenerateSubquestionsParseFailure(unittest.TestCase):
    @patch("src.subquestions.chat")
    def test_returns_parse_failed_on_unparseable_json(self, mock_chat):
        mock_chat.return_value = "not json at all {{{"
        cluster = {
            "description": "test",
            "tables": [{"table_id": "T1", "title": "Stadiums", "columns": [], "description": ""}],
            "passages": [],
        }

        result = generate_subquestions(
            llm=object(),
            user_query="How many stadiums exist?",
            cluster=cluster,
            k=3,
            temperature=0.0,
        )

        self.assertEqual(result, SubquestionGenerationResult([], parse_failed=True))
        self.assertTrue(mock_chat.call_args.kwargs.get("json_mode"))

    @patch("src.subquestions.chat")
    def test_parse_failed_false_on_intentional_empty(self, mock_chat):
        mock_chat.return_value = '{"subquestions": []}'
        cluster = {
            "description": "test",
            "tables": [{"table_id": "T1", "title": "Stadiums", "columns": [], "description": ""}],
            "passages": [],
        }

        result = generate_subquestions(
            llm=object(),
            user_query="How many stadiums exist?",
            cluster=cluster,
            k=3,
            temperature=0.0,
        )

        self.assertEqual(result, SubquestionGenerationResult([]))

    @patch("src.subquestions.chat")
    def test_parse_failed_on_string_subquestions_field(self, mock_chat):
        mock_chat.return_value = '{"subquestions": "only one question"}'
        cluster = {
            "description": "test",
            "tables": [{"table_id": "T1", "title": "Stadiums", "columns": [], "description": ""}],
            "passages": [],
        }

        result = generate_subquestions(
            llm=object(),
            user_query="How many stadiums exist?",
            cluster=cluster,
            k=3,
            temperature=0.0,
        )

        self.assertEqual(result, SubquestionGenerationResult([], parse_failed=True))


class TestFilterSemanticallyDuplicateSubquestions(unittest.TestCase):
    @patch("src.subquestions.chat")
    def test_keeps_candidates_on_invalid_remove_indices_type(self, mock_chat):
        mock_chat.return_value = '{"remove_indices": 1}'
        candidates = ["q1", "q2", "q3"]
        result = filter_semantically_duplicate_subquestions(
            llm=object(),
            answered=["already asked"],
            candidates=candidates,
            temperature=0.0,
        )
        self.assertEqual(result, candidates)

    @patch("src.subquestions.chat")
    def test_skips_non_numeric_indices(self, mock_chat):
        mock_chat.return_value = '{"remove_indices": [1, "x", 2]}'
        result = filter_semantically_duplicate_subquestions(
            llm=object(),
            answered=["already asked"],
            candidates=["q1", "q2", "q3"],
            temperature=0.0,
        )
        self.assertEqual(result, ["q3"])

    @patch("src.subquestions.chat")
    def test_keeps_candidates_on_parse_failure(self, mock_chat):
        mock_chat.return_value = "not json"
        candidates = ["q1", "q2"]
        result = filter_semantically_duplicate_subquestions(
            llm=object(),
            answered=["already asked"],
            candidates=candidates,
            temperature=0.0,
        )
        self.assertEqual(result, candidates)


if __name__ == "__main__":
    unittest.main()
