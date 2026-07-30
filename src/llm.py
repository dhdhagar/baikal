"""LLM client wrapper supporting OpenAI SDK, LiteLLM, and vLLM."""

import os
import re
import time
from dataclasses import dataclass, field
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import json_repair
from dotenv import dotenv_values
from openai import OpenAI

from src.tracking import track_api_response

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _REPO_ROOT / ".env"

SUPPORTED_LLM_PROVIDERS = ("openai", "litellm", "vllm")
OPENAI_COMPAT_PROVIDERS = ("openai", "vllm")
JSON_MODE_PROVIDERS = ("openai", "litellm")

DEFAULT_VLLM_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_API_KEY = "EMPTY"


class _NullCookieJar(CookieJar):
    """Ignore Set-Cookie so proxy cookies don't grow across reused API calls."""

    def extract_cookies(self, response, request):  # noqa: ARG002
        pass

    def set_cookie(self, cookie):  # noqa: ARG002
        pass


def _openai_http_client() -> httpx.Client:
    return httpx.Client(cookies=_NullCookieJar())


def _normalize_openai_api_base(api_base: str) -> str:
    """Ensure base URL works with the OpenAI SDK (…/v1)."""
    base = (api_base or "").rstrip("/")
    if not base:
        return ""
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _is_anthropic_litellm_model(model: str) -> bool:
    return (model or "").strip().lower().startswith("anthropic/")


def _has_anthropic_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _ensure_anthropic_env_vars() -> None:
    """Require ANTHROPIC_API_KEY for LiteLLM Anthropic models."""
    if _has_anthropic_credentials():
        return
    if not _ENV_FILE.is_file():
        raise RuntimeError(
            "Anthropic credentials missing. Either export ANTHROPIC_API_KEY "
            f"in your shell, or create {_ENV_FILE} with that variable."
        )
    _apply_dotenv_for_unset_keys()
    if not _has_anthropic_credentials():
        raise RuntimeError(
            f"ANTHROPIC_API_KEY must be set in {_ENV_FILE} "
            "when not exported in the shell."
        )


def _has_llm_credentials() -> bool:
    return bool(
        os.environ.get("LLM_API_KEY", "").strip()
        and os.environ.get("LLM_API_BASE", "").strip()
    )


def _has_embedding_credentials() -> bool:
    api_key = (
        os.environ.get("EMBEDDING_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )
    api_base = (
        os.environ.get("EMBEDDING_API_BASE", "").strip()
        or os.environ.get("LLM_API_BASE", "").strip()
    )
    return bool(api_key and api_base)


def _apply_dotenv_for_unset_keys() -> None:
    """Fill missing or empty env vars from repo-root .env without overriding exports."""
    if not _ENV_FILE.is_file():
        return
    for key, value in dotenv_values(_ENV_FILE).items():
        if value is not None and not os.environ.get(key, "").strip():
            os.environ[key] = value


def _ensure_llm_env_vars() -> None:
    """Use shell exports when present; otherwise require repo-root .env."""
    if _has_llm_credentials():
        return
    if not _ENV_FILE.is_file():
        raise RuntimeError(
            "LLM credentials missing. Either export LLM_API_KEY and LLM_API_BASE "
            f"in your shell, or create {_ENV_FILE} with those variables."
        )
    _apply_dotenv_for_unset_keys()
    if not _has_llm_credentials():
        raise RuntimeError(
            f"LLM_API_KEY and LLM_API_BASE must be set in {_ENV_FILE} "
            "when not exported in the shell."
        )


def _ensure_embedding_env_vars() -> None:
    """Use shell exports when present; otherwise require repo-root .env."""
    if _has_embedding_credentials():
        return
    if not _ENV_FILE.is_file():
        raise RuntimeError(
            "Embedding credentials missing. Either export LLM_API_KEY and LLM_API_BASE "
            "(or EMBEDDING_API_KEY and EMBEDDING_API_BASE) in your shell, "
            f"or create {_ENV_FILE} with those variables."
        )
    _apply_dotenv_for_unset_keys()
    if not _has_embedding_credentials():
        raise RuntimeError(
            f"Set LLM_API_KEY and LLM_API_BASE in {_ENV_FILE} "
            "(or EMBEDDING_API_KEY / EMBEDDING_API_BASE overrides) "
            "when not exported in the shell."
        )


def _resolve_credentials(provider: str) -> Tuple[str, str]:
    """Return (api_key, api_base) for the given provider."""
    if provider == "vllm":
        api_base = (
            os.environ.get("VLLM_API_BASE", "").strip()
            or os.environ.get("LLM_API_BASE", "").strip()
            or DEFAULT_VLLM_API_BASE
        )
        api_key = (
            os.environ.get("VLLM_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
            or DEFAULT_VLLM_API_KEY
        )
        return api_key, _normalize_openai_api_base(api_base)

    _ensure_llm_env_vars()
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    api_base = os.environ.get("LLM_API_BASE", "").strip()
    return api_key, _normalize_openai_api_base(api_base)


@dataclass
class LLMClient:
    """Configured LLM backend; model from CLI args, credentials from exports or .env."""

    provider: str
    model: str
    api_key: str
    api_base: str
    _openai_client: Optional[OpenAI] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.api_base = _normalize_openai_api_base(self.api_base)
        if self.provider in OPENAI_COMPAT_PROVIDERS:
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=_openai_http_client(),
            )

    def _supports_openai_json_mode(self) -> bool:
        """Whether ``response_format=json_object`` is safe for this backend/model."""
        if self.provider == "openai":
            return True
        if self.provider == "litellm":
            # Anthropic (and other native provider routes) reject OpenAI json_object.
            return not _is_anthropic_litellm_model(self.model)
        return False

    def _completion_kwargs(
        self,
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: bool,
        n: int,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if max_tokens:
            key = (
                "max_completion_tokens"
                if self.provider == "openai"
                else "max_tokens"
            )
            kwargs[key] = max_tokens
        if temperature:
            kwargs["temperature"] = temperature
        if json_mode and self.provider in JSON_MODE_PROVIDERS and self._supports_openai_json_mode():
            kwargs["response_format"] = {"type": "json_object"}
        if n > 1:
            kwargs["n"] = n
        return kwargs

    @staticmethod
    def _choice_texts(response: Any) -> List[str]:
        choices = getattr(response, "choices", None) or []
        return [
            (choice.message.content or "").strip()
            for choice in choices
            if getattr(choice.message, "content", None) is not None
        ]

    def _litellm_supports_n(self, litellm: Any) -> bool:
        """Return LiteLLM's advertised support for batched completions."""
        get_supported_params = getattr(
            litellm, "get_supported_openai_params", None
        )
        if not callable(get_supported_params):
            return True
        try:
            supported_params = get_supported_params(model=self.model)
        except Exception:
            # Unknown/custom models may not have capability metadata. Let the
            # request run and use the UnsupportedParamsError fallback below.
            return True
        return supported_params is None or "n" in supported_params

    @staticmethod
    def _is_unsupported_litellm_n_error(exc: Exception) -> bool:
        """Whether LiteLLM rejected the OpenAI ``n`` request parameter."""
        if exc.__class__.__name__ != "UnsupportedParamsError":
            return False
        message = str(exc).lower()
        return bool(
            re.search(
                r"(?:parameters?|params?).{0,100}\[[^\]]*['\"]n['\"][^\]]*\]",
                message,
            )
        )

    def complete_many(
        self,
        messages: List[Dict[str, str]],
        *,
        n: int,
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = None,
        feature: str = "completion",
        json_mode: bool = False,
    ) -> List[str]:
        if n < 1:
            raise ValueError("n must be >= 1")
        return self._dispatch_completion(
            messages,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            feature=feature,
            json_mode=json_mode,
        )

    def _dispatch_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        n: int,
        temperature: Optional[float],
        max_tokens: Optional[int],
        feature: str,
        json_mode: bool,
    ) -> List[str]:
        kwargs = self._completion_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            n=n,
        )

        if self.provider in OPENAI_COMPAT_PROVIDERS:
            started = time.perf_counter()
            resp = self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            track_api_response(
                resp,
                provider=self.provider,
                model=self.model,
                call_type="completion",
                feature=feature,
                time_taken=time.perf_counter() - started,
            )
            texts = self._choice_texts(resp)
            if not texts:
                raise RuntimeError("LLM returned no completion choices")
            return texts

        if self.provider == "litellm":
            try:
                import litellm
            except ImportError as e:
                raise RuntimeError(
                    "llm_provider=litellm requires the litellm package. "
                    "Install with: pip install litellm"
                ) from e
            completion_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                **kwargs,
            }
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            # Native provider routes (e.g. anthropic/...) must not inherit an
            # OpenAI-compatible api_base from LLM_API_BASE.
            if self.api_base and not _is_anthropic_litellm_model(self.model):
                completion_kwargs["api_base"] = self.api_base

            def _complete_once(call_kwargs: Dict[str, Any]) -> List[str]:
                started = time.perf_counter()
                resp = litellm.completion(**call_kwargs)
                track_api_response(
                    resp,
                    provider=self.provider,
                    model=self.model,
                    call_type="completion",
                    feature=feature,
                    time_taken=time.perf_counter() - started,
                )
                texts = self._choice_texts(resp)
                if not texts:
                    raise RuntimeError("LLM returned no completion choices")
                return texts

            def _complete_separately() -> List[str]:
                single_kwargs = dict(completion_kwargs)
                single_kwargs.pop("n", None)
                return [_complete_once(single_kwargs)[0] for _ in range(n)]

            if n > 1 and not self._litellm_supports_n(litellm):
                return _complete_separately()

            try:
                return _complete_once(completion_kwargs)
            except Exception as exc:
                # Capability metadata can be absent or stale. If LiteLLM
                # rejects only `n`, preserve the requested sample count by
                # issuing independent single-completion calls.
                if n > 1 and self._is_unsupported_litellm_n_error(exc):
                    return _complete_separately()
                raise

        raise ValueError(
            f"Unknown llm_provider {self.provider!r}; "
            f"expected one of {SUPPORTED_LLM_PROVIDERS}"
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = 1.0,
        max_tokens: Optional[int] = None,
        feature: str = "completion",
        json_mode: bool = False,
    ) -> str:
        return self.complete_many(
            messages,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            feature=feature,
            json_mode=json_mode,
        )[0]


def get_llm_client(provider: str, model: str) -> LLMClient:
    model = (model or "").strip()
    provider = (provider or "openai").strip().lower()

    if not model:
        raise RuntimeError("Set --llm_model (model name is configured via CLI, not .env)")
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unknown llm_provider {provider!r}; "
            f"expected one of {SUPPORTED_LLM_PROVIDERS}"
        )

    if provider == "litellm" and _is_anthropic_litellm_model(model):
        _ensure_anthropic_env_vars()
        return LLMClient(
            provider=provider,
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip(),
            api_base="",
        )

    api_key, api_base = _resolve_credentials(provider)
    return LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
    )


def strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def chat(
    llm: LLMClient,
    prompt: str,
    system_prompt: str = "You are a skilled data analyst.",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    feature: str = "completion",
    json_mode: bool = False,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return llm.complete(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        feature=feature,
        json_mode=json_mode,
    )


def chat_many(
    llm: LLMClient,
    prompt: str,
    *,
    n: int,
    system_prompt: str = "You are a skilled data analyst.",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    feature: str = "completion",
    json_mode: bool = False,
) -> List[str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return llm.complete_many(
        messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        feature=feature,
        json_mode=json_mode,
    )


def _json_parse_error(raw: str, *, detail: str = "") -> ValueError:
    suffix = f" ({detail})" if detail else ""
    return ValueError(
        f"Could not parse JSON from LLM response{suffix}: {raw[:200]}..."
    )


def _load_json_from_text(text: str, raw: str) -> Any:
    if not text:
        raise _json_parse_error(raw, detail="empty")
    try:
        result = json_repair.loads(text)
    except (ValueError, TypeError) as exc:
        raise _json_parse_error(raw) from exc
    if result == "":
        raise _json_parse_error(raw)
    return result


def parse_json_from_llm(raw: str) -> Any:
    """Parse JSON from an LLM response, repairing minor formatting issues."""
    return _load_json_from_text(strip_code_fence(raw), raw)


def parse_json_dict_from_llm(raw: str) -> dict:
    """Parse a JSON object from an LLM response; raises ValueError if not a dict."""
    result = parse_json_from_llm(raw)
    if not isinstance(result, dict):
        raise _json_parse_error(raw, detail="expected object")
    return result
