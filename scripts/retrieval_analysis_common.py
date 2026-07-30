#!/usr/bin/env python3
"""Shared utilities for standalone top-k retrieval analyses."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_client import get_embedding_client
from src.metrics.common import (
    extract_gt_passage_ids,
    extract_gt_table_ids,
    load_uid_to_passage_id_mapping,
    load_uid_to_table_id_mapping,
    normalize_passage_ids,
    normalize_table_ids,
)
from src.metrics.retrieval import build_retrieval_summary
from src.queries import load_queries_from_file, sample_queries

METRIC_KEYS = (
    "table_gt_in_top_k",
    "table_gt_reachable",
    "passage_gt_in_top_k",
    "passage_gt_reachable",
)


def dataset_paths(data_dir: Path, passage_type: str = "raw") -> dict[str, Path]:
    suffix = "_raw" if passage_type == "raw" else ""
    cluster_suffix = f"-{passage_type}"
    return {
        "tables_lake_dir": data_dir / "lake",
        "corpus_path": data_dir / "corpus.jsonl",
        "table_embeddings_path": data_dir / "table_embeddings.json",
        "passage_embeddings_path": data_dir / f"passage_embeddings{suffix}.json",
        "passage_descriptions_path": data_dir
        / f"passage_descriptions{suffix}.json",
        "uid_to_table_id_path": data_dir / "dpdisc_uid_to_table_id.json",
        "uid_to_passage_id_path": data_dir
        / f"dpdisc_uid_to_passage_id{suffix}.json",
        "inference_clusters_path": data_dir
        / f"inference_clusters_tables-passages{cluster_suffix}.json",
    }


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))


def load_queries(
    query_file: Path,
    *,
    n_queries: int | None,
    seed: int,
    stratified: bool,
) -> list[dict[str, Any]]:
    queries = load_queries_from_file(str(query_file))
    if not queries:
        raise ValueError(f"No queries found in {query_file}")
    return sample_queries(queries, n_queries, stratified, seed)


def _id_sort_key(item_id: str) -> tuple[int, int | str]:
    suffix = item_id[1:]
    return (0, int(suffix)) if suffix.isdigit() else (1, item_id)


def load_normalized_embedding_index(path: Path) -> tuple[list[str], np.ndarray]:
    """Load an id->vector JSON and L2-normalize rows once for all queries."""
    with path.open(encoding="utf-8") as handle:
        embeddings = json.load(handle)
    if not isinstance(embeddings, dict) or not embeddings:
        raise ValueError(f"No embeddings found in {path}")
    item_ids = sorted((str(item_id) for item_id in embeddings), key=_id_sort_key)
    matrix = np.asarray([embeddings[item_id] for item_id in item_ids], dtype=np.float32)
    del embeddings
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix in {path}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    return item_ids, matrix


def top_ids(
    item_ids: Sequence[str],
    normalized_matrix: np.ndarray,
    query_vector: np.ndarray,
    k: int,
) -> list[str]:
    if k <= 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.shape[0] != normalized_matrix.shape[1]:
        raise ValueError(
            "Query and item embedding dimensions differ: "
            f"{query.shape[0]} vs {normalized_matrix.shape[1]}"
        )
    norm = float(np.linalg.norm(query))
    if norm == 0:
        return list(item_ids[: min(k, len(item_ids))])
    scores = normalized_matrix @ (query / norm)
    k = min(k, len(scores))
    indices = np.argpartition(-scores, k - 1)[:k]
    ordered = indices[np.argsort(-scores[indices], kind="stable")]
    return [item_ids[int(index)] for index in ordered]


def evaluate_retrieval(
    *,
    queries: Sequence[dict[str, Any]],
    embedding_model: str,
    embedding_provider: str,
    gpu: bool,
    table_embeddings_path: Path,
    passage_embeddings_path: Path,
    inference_clusters_path: Path,
    uid_to_table_id_path: Path,
    uid_to_passage_id_path: Path,
    passage_type: str,
    ks: Sequence[int],
) -> list[dict[str, Any]]:
    require_files(
        (
            table_embeddings_path,
            passage_embeddings_path,
            inference_clusters_path,
            uid_to_table_id_path,
            uid_to_passage_id_path,
        )
    )
    table_ids, table_matrix = load_normalized_embedding_index(table_embeddings_path)
    passage_ids, passage_matrix = load_normalized_embedding_index(
        passage_embeddings_path
    )
    with inference_clusters_path.open(encoding="utf-8") as handle:
        inference_clusters = json.load(handle)
    if not isinstance(inference_clusters, list):
        raise ValueError(
            f"Expected a list of inference clusters in {inference_clusters_path}"
        )

    uid_to_table_id = load_uid_to_table_id_mapping(str(uid_to_table_id_path))
    uid_to_passage_id = load_uid_to_passage_id_mapping(
        str(uid_to_passage_id_path)
    )
    embedder = get_embedding_client(
        embedding_provider,
        embedding_model,
        gpu=gpu,
    )
    max_k = max(ks)
    output: list[dict[str, Any]] = []

    for query_index, query in enumerate(queries, start=1):
        print(
            f"[{query_index}/{len(queries)}] embedding query {query['query_id']}",
            flush=True,
        )
        query_vector = embedder.encode_one(
            query["query_text"], feature="retrieval_analysis_query"
        )
        ranked_tables = top_ids(table_ids, table_matrix, query_vector, max_k)
        ranked_passages = top_ids(passage_ids, passage_matrix, query_vector, max_k)
        ground_truth = query.get("ground_truth")
        gt_tables = normalize_table_ids(
            extract_gt_table_ids(
                ground_truth,
                uid_to_table_id=uid_to_table_id,
            )
            or []
        )
        gt_passages = normalize_passage_ids(
            extract_gt_passage_ids(
                ground_truth,
                uid_to_passage_id=uid_to_passage_id,
                passage_type=passage_type,
            )
            or []
        )

        for k in ks:
            metrics = build_retrieval_summary(
                gt_tables=gt_tables,
                gt_passages=gt_passages,
                topk_table_ids=ranked_tables[:k],
                topk_passage_ids=ranked_passages[:k],
                inference_clusters=inference_clusters,
                k_relevant_tables=k,
                k_relevant_passages=k,
                use_passages=True,
            )
            output.append(
                {
                    "query_id": query["query_id"],
                    "coverage": query.get("coverage", "unknown"),
                    "k": k,
                    **metrics,
                }
            )
    return output


def mean_and_bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(np.mean(array))
    if n_bootstrap <= 0 or len(array) == 1:
        return {"mean": mean, "ci_low": None, "ci_high": None, "n": len(array)}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "mean": mean,
        "ci_low": float(low),
        "ci_high": float(high),
        "n": len(array),
    }


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    summary = []
    for group_index, group in enumerate(sorted(groups)):
        item: dict[str, Any] = {group_key: group, "n_queries": len(groups[group])}
        for metric_index, metric in enumerate(METRIC_KEYS):
            values = [
                float(row[metric])
                for row in groups[group]
                if row.get(metric) is not None
            ]
            item[metric] = mean_and_bootstrap_ci(
                values,
                n_bootstrap=n_bootstrap,
                seed=seed + group_index * 100 + metric_index,
            )
        summary.append(item)
    return summary


def write_analysis_outputs(
    *,
    output_path: Path,
    payload: dict[str, Any],
    summary: Sequence[dict[str, Any]],
    per_query: Sequence[dict[str, Any]],
    group_key: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["summary"] = list(summary)
    payload["per_query"] = list(per_query)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = output_path.with_suffix(".csv")
    fields = [group_key, "n_queries"]
    for metric in METRIC_KEYS:
        fields.extend((metric, f"{metric}_ci_low", f"{metric}_ci_high", f"{metric}_n"))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            row: dict[str, Any] = {
                group_key: item[group_key],
                "n_queries": item["n_queries"],
            }
            for metric in METRIC_KEYS:
                result = item[metric]
                row[metric] = result["mean"]
                row[f"{metric}_ci_low"] = result["ci_low"]
                row[f"{metric}_ci_high"] = result["ci_high"]
                row[f"{metric}_n"] = result["n"]
            writer.writerow(row)
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")


def parse_positive_ints(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected a comma-separated list of positive integers.")
    return values


def print_summary(summary: Sequence[dict[str, Any]], group_key: str) -> None:
    header = (
        f"{group_key:<34} {'Table@k':>10} {'Table reach':>12} "
        f"{'Passage@k':>11} {'Passage reach':>13}"
    )
    print(header)
    print("-" * len(header))
    for item in summary:
        means = [
            item[metric]["mean"]
            for metric in METRIC_KEYS
        ]
        formatted = [f"{value:.4f}" if value is not None else "n/a" for value in means]
        print(
            f"{str(item[group_key]):<34} {formatted[0]:>10} {formatted[1]:>12} "
            f"{formatted[2]:>11} {formatted[3]:>13}"
        )
