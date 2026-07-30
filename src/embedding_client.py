"""Embedding client supporting local SentenceTransformer, OpenAI, LiteLLM, and vLLM."""

import functools
import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from src.llm import (
    DEFAULT_VLLM_API_BASE,
    DEFAULT_VLLM_API_KEY,
    _ensure_embedding_env_vars,
    _normalize_openai_api_base,
    _openai_http_client,
)
from src.tracking import track_api_response

SUPPORTED_EMBEDDING_PROVIDERS = ("local", "openai", "litellm", "vllm")
OPENAI_COMPAT_EMBEDDING_PROVIDERS = ("openai", "vllm")


def resolve_local_device(gpu: bool) -> str:
    """Return SentenceTransformer device: cpu by default, or cuda/mps when --gpu is set."""
    if not gpu:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _resolve_embedding_credentials(provider: str) -> Tuple[str, str]:
    """Return (api_key, api_base) for remote embedding providers."""
    if provider == "vllm":
        api_base = (
            os.environ.get("VLLM_EMBED_API_BASE", "").strip()
            or os.environ.get("VLLM_API_BASE", "").strip()
            or os.environ.get("LLM_API_BASE", "").strip()
            or DEFAULT_VLLM_API_BASE
        )
        api_key = (
            os.environ.get("VLLM_EMBED_API_KEY", "").strip()
            or os.environ.get("VLLM_API_KEY", "").strip()
            or os.environ.get("LLM_API_KEY", "").strip()
            or DEFAULT_VLLM_API_KEY
        )
        return api_key, _normalize_openai_api_base(api_base)

    _ensure_embedding_env_vars()
    api_key = (
        os.environ.get("EMBEDDING_API_KEY", "").strip()
        or os.environ.get("LLM_API_KEY", "").strip()
    )
    api_base = (
        os.environ.get("EMBEDDING_API_BASE", "").strip()
        or os.environ.get("LLM_API_BASE", "").strip()
    )
    return api_key, _normalize_openai_api_base(api_base)


def _extract_embedding_rows(response: Any) -> List[Any]:
    data = response.get("data") if isinstance(response, dict) else response.data
    if not data:
        raise RuntimeError("Embedding API returned no data")
    return sorted(
        data,
        key=lambda row: row["index"] if isinstance(row, dict) else row.index,
    )


def _row_embedding(row: Any) -> List[float]:
    embedding = row["embedding"] if isinstance(row, dict) else row.embedding
    return list(map(float, embedding))


@dataclass
class EmbeddingClient:
    """Configured embedding backend; model name comes from CLI args."""

    provider: str
    model: str
    api_key: str = ""
    api_base: str = ""
    gpu: bool = False
    _openai_client: Optional[OpenAI] = field(default=None, repr=False)
    _local_model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.provider = self.provider.strip().lower()
        if self.provider in OPENAI_COMPAT_EMBEDDING_PROVIDERS:
            self.api_base = _normalize_openai_api_base(self.api_base)
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                http_client=_openai_http_client(),
            )

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        feature: str = "embedding",
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float64)

        if self.provider == "local":
            return self._encode_local(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
            )
        if self.provider in OPENAI_COMPAT_EMBEDDING_PROVIDERS:
            return self._encode_openai_compat(
                texts, batch_size=batch_size, feature=feature
            )
        if self.provider == "litellm":
            return self._encode_litellm(texts, batch_size=batch_size, feature=feature)

        raise ValueError(
            f"Unknown embedding_provider {self.provider!r}; "
            f"expected one of {SUPPORTED_EMBEDDING_PROVIDERS}"
        )

    def encode_one(self, text: str, *, feature: str = "embedding") -> np.ndarray:
        return self.encode([text], show_progress_bar=False, feature=feature)[0]

    def _encode_local(
        self,
        texts: List[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray:
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            device = resolve_local_device(self.gpu)
            self._local_model = SentenceTransformer(
                self.model, device=device, trust_remote_code=True
            )
        embs = self._local_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(embs, dtype=np.float64)

    def _encode_openai_compat(
        self,
        texts: List[str],
        *,
        batch_size: int,
        feature: str = "embedding",
    ) -> np.ndarray:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            started = time.perf_counter()
            resp = self._openai_client.embeddings.create(model=self.model, input=chunk)
            track_api_response(
                resp,
                provider=self.provider,
                model=self.model,
                call_type="embedding",
                feature=feature,
                time_taken=time.perf_counter() - started,
            )
            rows = _extract_embedding_rows(resp)
            if len(rows) != len(chunk):
                raise RuntimeError(
                    f"Expected {len(chunk)} embeddings, got {len(rows)}"
                )
            vectors.extend(_row_embedding(row) for row in rows)
        return np.asarray(vectors, dtype=np.float64)

    def _encode_litellm(
        self,
        texts: List[str],
        *,
        batch_size: int,
        feature: str = "embedding",
    ) -> np.ndarray:
        try:
            import litellm
        except ImportError as e:
            raise RuntimeError(
                "embedding_provider=litellm requires the litellm package. "
                "Install with: pip install litellm"
            ) from e

        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            started = time.perf_counter()
            resp = litellm.embedding(
                model=self.model,
                input=chunk,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            track_api_response(
                resp,
                provider=self.provider,
                model=self.model,
                call_type="embedding",
                feature=feature,
                time_taken=time.perf_counter() - started,
            )
            rows = _extract_embedding_rows(resp)
            if len(rows) != len(chunk):
                raise RuntimeError(
                    f"Expected {len(chunk)} embeddings, got {len(rows)}"
                )
            vectors.extend(_row_embedding(row) for row in rows)
        return np.asarray(vectors, dtype=np.float64)


@functools.lru_cache(maxsize=8)
def get_embedding_client(provider: str, model: str, gpu: bool = False) -> EmbeddingClient:
    model = (model or "").strip()
    provider = (provider or "local").strip().lower()

    if not model:
        raise RuntimeError(
            "Set --embedding_model (model name is configured via CLI, not .env)"
        )
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unknown embedding_provider {provider!r}; "
            f"expected one of {SUPPORTED_EMBEDDING_PROVIDERS}"
        )

    if provider == "local":
        return EmbeddingClient(provider=provider, model=model, gpu=gpu)

    api_key, api_base = _resolve_embedding_credentials(provider)
    return EmbeddingClient(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
    )
