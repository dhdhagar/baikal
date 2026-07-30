"""Operational run metrics (SQL success, cluster attrition, diversity)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.embedding_client import EmbeddingClient
from src.metrics.common import round4


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normed = embeddings / norms
    return normed @ normed.T


def diversity_from_embeddings(embeddings: np.ndarray) -> Dict[str, Optional[float]]:
    if embeddings.shape[0] < 2:
        return {"mean": None, "max": None}

    sims = _cosine_similarity_matrix(embeddings)
    n = sims.shape[0]
    per_item: List[float] = []
    for i in range(n):
        others = [sims[i, j] for j in range(n) if j != i]
        per_item.append(1.0 - float(np.mean(others)))
    return {
        "mean": round4(float(np.mean(per_item))),
        "max": round4(float(np.max(per_item))),
    }


class DiversityTracker:
    """Incrementally track finding diversity without re-embedding all findings."""

    def __init__(self, embedder: EmbeddingClient) -> None:
        self.embedder = embedder
        self._embeddings: Optional[np.ndarray] = None

    def add_finding(self, sub_question: str, answer: str) -> None:
        text = f"Sub-question: {sub_question}\nAnswer: {answer}"
        embedding = self.embedder.encode(
            [text],
            show_progress_bar=False,
            feature="metrics_diversity_embedding",
        )
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = vector
        else:
            self._embeddings = np.vstack([self._embeddings, vector])

    @property
    def n_findings(self) -> int:
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[0])

    def snapshot(self) -> Dict[str, Optional[float]]:
        if self._embeddings is None:
            return {"mean": None, "max": None}
        return diversity_from_embeddings(self._embeddings)


def compute_diversity(
    texts: List[str],
    embedder: EmbeddingClient,
) -> Dict[str, Optional[float]]:
    """
    Diversity over findings.

    For each item i, diversity_i = 1 - mean cosine sim to all other items.
    Returns mean and max of diversity_i across items.
    """
    if len(texts) < 2:
        return {"mean": None, "max": None}

    embeddings = embedder.encode(
        texts, show_progress_bar=False, feature="metrics_diversity_embedding"
    )
    return diversity_from_embeddings(np.asarray(embeddings, dtype=np.float64))


def build_operational_cumulative_metrics(
    *,
    sql_attempts: int,
    sql_successes: int,
    clusters_excluded_cumulative: int,
    initial_candidate_clusters: int,
    diversity: Dict[str, Optional[float]],
    n_findings: int,
) -> dict:
    sql_rate = sql_successes / sql_attempts if sql_attempts else None
    attrition_rate = (
        clusters_excluded_cumulative / initial_candidate_clusters
        if initial_candidate_clusters
        else None
    )
    return {
        "sql_success_rate": round4(sql_rate),
        "n_sql_attempts": sql_attempts,
        "cluster_attrition_rate": round4(attrition_rate),
        "n_clusters_excluded": clusters_excluded_cumulative,
        "diversity": diversity,
        "n_findings": n_findings,
    }


def build_operational_step_metrics(
    *,
    step: int,
    sql_success_step: bool,
    sql_attempts: int,
    sql_successes: int,
    clusters_excluded_step: int,
    clusters_excluded_cumulative: int,
    initial_candidate_clusters: int,
    lake_coverage: Optional[float],
    diversity: Dict[str, Optional[float]],
    n_findings: int,
) -> dict:
    return {
        "sql_success_step": sql_success_step,
        "sql_success_rate_cumulative": round4(
            sql_successes / sql_attempts if sql_attempts else None
        ),
        "cluster_attrition_step": clusters_excluded_step,
        "cluster_attrition_cumulative": clusters_excluded_cumulative,
        "cluster_attrition_rate_cumulative": round4(
            clusters_excluded_cumulative / initial_candidate_clusters
            if initial_candidate_clusters
            else None
        ),
        "lake_coverage": round4(lake_coverage),
        "diversity": diversity,
        "n_findings": n_findings,
    }
