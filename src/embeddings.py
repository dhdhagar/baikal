"""Embedding-based table and passage relevance for a user query."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.embedding_client import EmbeddingClient, get_embedding_client
from src.utils import cosine_topk, load_json


def load_embedding_index(path: str) -> Tuple[List[str], np.ndarray]:
    """Load id -> embedding dict; return ids and matrix in stable key order."""
    embed = load_json(path)
    ids = sorted(embed.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x)
    matrix = np.array([embed[iid] for iid in ids], dtype=np.float64)
    return ids, matrix


def load_table_embedding_index(path: str) -> Tuple[List[str], np.ndarray]:
    """Load table_id -> embedding dict; return ids and matrix in stable key order."""
    tables_embed = load_json(path)
    ids = sorted(tables_embed.keys())
    matrix = np.array([tables_embed[tid] for tid in ids], dtype=np.float64)
    return ids, matrix


def encode_query(
    query_text: str,
    embedding_model: str,
    embedding_provider: str = "local",
    embedder: Optional[EmbeddingClient] = None,
) -> np.ndarray:
    """Embed a single query string."""
    client = embedder or get_embedding_client(embedding_provider, embedding_model)
    return client.encode_one(query_text, feature="query_embedding")


def get_similarity_ranks(
    query_text: str,
    embeddings_path: str,
    embedding_model: str,
    embedding_provider: str = "local",
    embedder: Optional[EmbeddingClient] = None,
) -> Dict[str, int]:
    """Rank every item by cosine similarity to the query (1 = most similar)."""
    item_ids, matrix = load_embedding_index(embeddings_path)
    q_vec = encode_query(
        query_text,
        embedding_model,
        embedding_provider=embedding_provider,
        embedder=embedder,
    )
    idxs = cosine_topk(q_vec, matrix, len(matrix))
    return {item_ids[i]: rank + 1 for rank, i in enumerate(idxs)}


def get_table_similarity_ranks(
    query_text: str,
    embeddings_path: str,
    embedding_model: str,
    embedding_provider: str = "local",
    embedder: Optional[EmbeddingClient] = None,
) -> Dict[str, int]:
    """
    Rank every table by cosine similarity to the query.

    Returns table_id -> 1-based rank (1 = most similar).
    """
    table_ids, matrix = load_table_embedding_index(embeddings_path)
    q_vec = encode_query(
        query_text,
        embedding_model,
        embedding_provider=embedding_provider,
        embedder=embedder,
    )
    idxs = cosine_topk(q_vec, matrix, len(matrix))
    return {table_ids[i]: rank + 1 for rank, i in enumerate(idxs)}


def get_passage_similarity_ranks(
    query_text: str,
    embeddings_path: str,
    embedding_model: str,
    embedding_provider: str = "local",
    embedder: Optional[EmbeddingClient] = None,
) -> Dict[str, int]:
    """Rank every passage by cosine similarity to the query."""
    return get_similarity_ranks(
        query_text,
        embeddings_path,
        embedding_model,
        embedding_provider=embedding_provider,
        embedder=embedder,
    )


def top_k_ids_from_ranks(rank_by_id: Dict[str, int], k: int) -> List[str]:
    """Ids with the best (lowest) similarity ranks."""
    if k <= 0:
        return []
    return [
        iid
        for iid, _ in sorted(rank_by_id.items(), key=lambda item: item[1])[:k]
    ]


def top_k_table_ids_from_ranks(rank_by_table: Dict[str, int], k: int) -> List[str]:
    """Table ids with the best (lowest) similarity ranks."""
    return top_k_ids_from_ranks(rank_by_table, k)


def get_relevant_table_ids(
    query_text: str,
    embeddings_path: str,
    k: int,
    embedding_model: str,
    embedding_provider: str = "local",
    embedder: Optional[EmbeddingClient] = None,
) -> List[str]:
    """Top-k table ids by cosine similarity between query and table embeddings."""
    ranks = get_table_similarity_ranks(
        query_text,
        embeddings_path,
        embedding_model,
        embedding_provider=embedding_provider,
        embedder=embedder,
    )
    return top_k_table_ids_from_ranks(ranks, k)


def _cluster_item_rank_summary(
    item_ids: List[str],
    rank_by_id: Dict[str, int],
    ranks_key: str,
) -> Dict[str, Any]:
    """Best (min) and average 1-based similarity ranks for items in a cluster."""
    ranks = [rank_by_id[iid] for iid in item_ids if iid in rank_by_id]
    if not ranks:
        return {"best_rank": None, "avg_rank": None, ranks_key: {}}
    return {
        "best_rank": min(ranks),
        "avg_rank": round(sum(ranks) / len(ranks), 2),
        ranks_key: {iid: rank_by_id[iid] for iid in item_ids if iid in rank_by_id},
    }


def cluster_table_rank_summary(
    table_ids: List[str],
    rank_by_table: Dict[str, int],
) -> Dict[str, Any]:
    """Best (min) and average 1-based similarity ranks for tables in a cluster."""
    return _cluster_item_rank_summary(table_ids, rank_by_table, "table_ranks")


def cluster_passage_rank_summary(
    passage_ids: List[str],
    rank_by_passage: Dict[str, int],
) -> Dict[str, Any]:
    """Best (min) and average 1-based similarity ranks for passages in a cluster."""
    return _cluster_item_rank_summary(passage_ids, rank_by_passage, "passage_ranks")


def cluster_similarity_rank_summary(
    table_ids: List[str],
    rank_by_table: Dict[str, int],
    passage_ids: Optional[List[str]] = None,
    rank_by_passage: Optional[Dict[str, int]] = None,
    *,
    include_passages: bool = False,
) -> Dict[str, Any]:
    """Similarity rank summary for tables and, when enabled, passages in a cluster."""
    summary = cluster_table_rank_summary(table_ids, rank_by_table)
    if include_passages and passage_ids:
        passage_summary = cluster_passage_rank_summary(
            passage_ids,
            rank_by_passage or {},
        )
        summary["passage_best_rank"] = passage_summary["best_rank"]
        summary["passage_avg_rank"] = passage_summary["avg_rank"]
        summary["passage_ranks"] = passage_summary["passage_ranks"]
    return summary


def filter_clusters_by_tables_passages(
    inference_clusters: List[dict],
    topk_table_ids: List[str],
    topk_passage_ids: Optional[List[str]] = None,
) -> List[dict]:
    """Keep clusters that contain at least one top-k table or passage."""
    rel_tables = set(topk_table_ids)
    rel_passages = set(topk_passage_ids or [])
    out = []
    for cluster in inference_clusters:
        tids = {t["table_id"] for t in cluster.get("tables", [])}
        pids = {p["passage_id"] for p in cluster.get("passages", [])}
        if (tids & rel_tables) or (pids & rel_passages):
            out.append(cluster)
    return out
