"""Ground-truth retrieval and hit-quality metrics."""

from __future__ import annotations

from typing import List, Optional, Set

from src.metrics.common import normalize_passage_ids, normalize_table_ids, round4


def compute_table_recall(used_tables: Set[str], gt_tables: Set[str]) -> Optional[float]:
    if not gt_tables:
        return None
    if not used_tables:
        return 0.0
    return len(used_tables & gt_tables) / len(gt_tables)


def compute_table_precision(used_tables: Set[str], gt_tables: Set[str]) -> Optional[float]:
    if not used_tables:
        return None
    if not gt_tables:
        return None
    return len(used_tables & gt_tables) / len(used_tables)


def compute_lake_coverage(used_tables: Set[str], total_lake_tables: int) -> Optional[float]:
    if total_lake_tables <= 0:
        return None
    return len(used_tables) / total_lake_tables


def compute_table_gt_in_top_k(
    gt_tables: Set[str],
    topk_table_ids: List[str],
) -> Optional[float]:
    if not gt_tables:
        return None
    top_k = normalize_table_ids(topk_table_ids)
    return len(gt_tables & top_k) / len(gt_tables)


def compute_table_gt_reachable(
    gt_tables: Set[str],
    inference_clusters: List[dict],
    topk_table_ids: List[str],
) -> Optional[float]:
    if not gt_tables:
        return None
    rel = set(topk_table_ids)
    reachable: Set[str] = set()
    for cluster in inference_clusters:
        cluster_tables = {
            t["table_id"] for t in cluster.get("tables", []) if t.get("table_id")
        }
        if cluster_tables & rel:
            reachable |= cluster_tables
    return len(gt_tables & reachable) / len(gt_tables)


def compute_passage_recall(
    cited_passages: Set[str],
    gt_passages: Set[str],
) -> Optional[float]:
    return compute_table_recall(cited_passages, gt_passages)


def compute_passage_precision(
    cited_passages: Set[str],
    gt_passages: Set[str],
) -> Optional[float]:
    return compute_table_precision(cited_passages, gt_passages)


def compute_passage_gt_in_top_k(
    gt_passages: Set[str],
    topk_passage_ids: List[str],
) -> Optional[float]:
    if not gt_passages:
        return None
    top_k = normalize_passage_ids(topk_passage_ids)
    return len(gt_passages & top_k) / len(gt_passages)


def compute_passage_gt_reachable(
    gt_passages: Set[str],
    inference_clusters: List[dict],
    topk_passage_ids: List[str],
) -> Optional[float]:
    if not gt_passages:
        return None
    rel = normalize_passage_ids(topk_passage_ids)
    reachable: Set[str] = set()
    for cluster in inference_clusters:
        cluster_passages = normalize_passage_ids(
            [
                p["passage_id"]
                for p in cluster.get("passages", [])
                if p.get("passage_id")
            ]
        )
        if cluster_passages & rel:
            reachable |= cluster_passages
    return len(gt_passages & reachable) / len(gt_passages)


def build_retrieval_summary(
    *,
    gt_tables: Set[str],
    gt_passages: Set[str],
    topk_table_ids: List[str],
    topk_passage_ids: List[str],
    inference_clusters: List[dict],
    k_relevant_tables: int,
    k_relevant_passages: int,
    use_passages: bool,
) -> dict:
    summary = {
        "table_gt_in_top_k": round4(
            compute_table_gt_in_top_k(gt_tables, topk_table_ids)
        ),
        "table_gt_reachable": round4(
            compute_table_gt_reachable(
                gt_tables, inference_clusters, topk_table_ids
            )
        ),
        "k_relevant_tables": k_relevant_tables,
        "n_gt_tables": len(gt_tables),
    }
    if use_passages:
        summary.update(
            {
                "passage_gt_in_top_k": round4(
                    compute_passage_gt_in_top_k(gt_passages, topk_passage_ids)
                ),
                "passage_gt_reachable": round4(
                    compute_passage_gt_reachable(
                        gt_passages,
                        inference_clusters,
                        topk_passage_ids,
                    )
                ),
                "k_relevant_passages": k_relevant_passages,
                "n_gt_passages": len(gt_passages),
            }
        )
    return summary


def build_retrieval_step_metrics(
    *,
    tables_step: Set[str],
    passages_step: Set[str],
    tables_used_cumulative: Set[str],
    passages_cited_cumulative: Set[str],
    gt_tables: Set[str],
    gt_passages: Set[str],
    use_passages: bool,
) -> dict:
    metrics = {
        "table_recall_at_budget": round4(
            compute_table_recall(tables_used_cumulative, gt_tables)
        ),
        "table_recall_marginal": round4(compute_table_recall(tables_step, gt_tables)),
        "table_precision_step": round4(compute_table_precision(tables_step, gt_tables)),
        "table_precision_cumulative": round4(
            compute_table_precision(tables_used_cumulative, gt_tables)
        ),
    }
    if not use_passages:
        return metrics
    metrics.update(
        {
            "passage_recall_at_budget": round4(
                compute_passage_recall(passages_cited_cumulative, gt_passages)
            ),
            "passage_recall_marginal": round4(
                compute_passage_recall(passages_step, gt_passages)
            ),
            "passage_precision_step": round4(
                compute_passage_precision(passages_step, gt_passages)
            ),
            "passage_precision_cumulative": round4(
                compute_passage_precision(passages_cited_cumulative, gt_passages)
            ),
        }
    )
    return metrics


def build_retrieval_cumulative_metrics(
    *,
    tables_used_cumulative: Set[str],
    passages_cited_cumulative: Set[str],
    gt_tables: Set[str],
    gt_passages: Set[str],
    total_lake_tables: int,
    use_passages: bool,
) -> dict:
    metrics = {
        "table_recall": round4(
            compute_table_recall(tables_used_cumulative, gt_tables)
        ),
        "table_precision": round4(
            compute_table_precision(tables_used_cumulative, gt_tables)
        ),
        "lake_coverage": round4(
            compute_lake_coverage(tables_used_cumulative, total_lake_tables)
        ),
        "n_tables_used": len(tables_used_cumulative),
    }
    if not use_passages:
        return metrics
    metrics.update(
        {
            "passage_recall": round4(
                compute_passage_recall(passages_cited_cumulative, gt_passages)
            ),
            "passage_precision": round4(
                compute_passage_precision(passages_cited_cumulative, gt_passages)
            ),
            "n_passages_cited": len(passages_cited_cumulative),
        }
    )
    return metrics
