"""Lightweight inference-cluster helpers (no BERTopic/UMAP dependencies)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from src.utils import load_json


def load_tables_metadata(tables_lake_dir: str) -> Dict[str, dict]:
    """Load extra metadata (title, domain) from lake JSON files."""
    metadata = {}
    if not tables_lake_dir or not os.path.isdir(tables_lake_dir):
        return metadata
    if os.path.exists(os.path.join(tables_lake_dir, "schema_descriptions.json")):
        metadata = load_json(os.path.join(tables_lake_dir, "schema_descriptions.json"))
    else:
        raise FileNotFoundError(f"schema_descriptions.json not found in {tables_lake_dir}")
    return metadata


def load_passages_metadata(passage_descriptions_path: str) -> Dict[str, dict]:
    if not os.path.isfile(passage_descriptions_path):
        raise FileNotFoundError(
            f"passage_descriptions.json not found at {passage_descriptions_path}"
        )
    return load_json(passage_descriptions_path)


def build_topk_inference_cluster(
    topk_table_ids: List[str],
    topk_passage_ids: List[str],
    *,
    tables_lake_dir: str,
    passage_descriptions_path: Optional[str] = None,
    cluster_id: str = "topk",
) -> List[dict]:
    """Build a single inference cluster from per-query top-k tables and passages."""
    tables_meta = load_tables_metadata(tables_lake_dir)
    tables = [
        {
            "table_id": tid,
            "title": tables_meta[tid].get("title", ""),
            "domain": tables_meta[tid].get("domain", ""),
            "columns": tables_meta[tid].get("columns", []),
            "description": tables_meta[tid].get("description", ""),
        }
        for tid in topk_table_ids
        if tid in tables_meta
    ]
    passages: List[dict] = []
    if topk_passage_ids and passage_descriptions_path:
        passages_meta = load_passages_metadata(passage_descriptions_path)
        passages = [
            {
                "passage_id": pid,
                "uid": passages_meta[pid].get("uid", ""),
                "title": passages_meta[pid].get("title", ""),
                "text": passages_meta[pid].get("text", ""),
            }
            for pid in topk_passage_ids
            if pid in passages_meta
        ]
    return [
        {
            "cluster_id": cluster_id,
            "description": "All top-k relevant tables and passages",
            "tables": tables,
            "passages": passages,
        }
    ]
