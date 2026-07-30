"""Shared helpers for the DPR discovery pipeline."""

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def log(msg: str, silent: bool = False) -> None:
    from src.cli_ui import log as cli_log

    cli_log(msg, silent=silent)


def slugify(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return (s[:max_len] or "query").rstrip("_")


def cosine_topk(
    query_vec: np.ndarray,
    matrix: np.ndarray,
    k: int,
) -> List[int]:
    """Return indices of top-k rows in matrix by cosine similarity to query_vec."""
    if k <= 0:
        return []
    q = query_vec.astype(np.float64)
    m = matrix.astype(np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return list(range(min(k, len(m))))
    row_norms = np.linalg.norm(m, axis=1)
    row_norms[row_norms == 0] = 1e-12
    scores = (m @ q) / (row_norms * q_norm)
    k = min(k, len(scores))
    top_idxs = np.argpartition(-scores, k - 1)[:k]
    return sorted(top_idxs.tolist(), key=lambda i: -scores[i])


def default_inference_clusters_filename(
    use_passages: bool = False,
    passage_type: str = "synth",
) -> str:
    if not use_passages:
        return "inference_clusters_tables-only.json"
    passage_type = passage_type or "synth"
    return f"inference_clusters_tables-passages-{passage_type}.json"


def passage_path_suffix(passage_type: str = "synth") -> str:
    passage_type = passage_type or "synth"
    return "" if passage_type == "synth" else f"_{passage_type}"


def passage_descriptions_filename(passage_type: str = "synth") -> str:
    return f"passage_descriptions{passage_path_suffix(passage_type)}.json"


def default_passage_descriptions_path(
    data_dir: str,
    passage_type: str = "synth",
) -> str:
    return os.path.join(data_dir, passage_descriptions_filename(passage_type))


def query_passage_descriptions_path(query_dir: str, passage_type: str = "synth") -> str:
    return os.path.join(query_dir, passage_descriptions_filename(passage_type))


def resolve_passage_descriptions_source(args) -> str:
    """Return passage descriptions JSON path for the configured passage_type."""
    if getattr(args, "passage_descriptions_path", None):
        return args.passage_descriptions_path
    data_dir = getattr(args, "data_dir", None)
    if not data_dir:
        raise ValueError(
            "passage_descriptions_path or data_dir is required when use_passages=True"
        )
    passage_type = getattr(args, "passage_type", "synth") or "synth"
    return default_passage_descriptions_path(data_dir, passage_type)


def load_passage_descriptions_for_metrics(
    args,
    *,
    query_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load passage descriptions for metrics judging (args path, then query-local)."""
    if not getattr(args, "use_passages", False):
        return None
    passage_type = getattr(args, "passage_type", "synth") or "synth"
    candidates: List[str] = []
    path = getattr(args, "passage_descriptions_path", "") or ""
    if path:
        candidates.append(path)
    if query_dir:
        local = query_passage_descriptions_path(query_dir, passage_type)
        if local not in candidates:
            candidates.append(local)
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return load_json(candidate)
    return None


def resolve_default_paths(args) -> None:
    """Fill in path defaults that depend on data_dir."""
    use_passages = bool(getattr(args, "use_passages", False))
    passage_type = getattr(args, "passage_type", "synth") or "synth"
    if not args.inference_clusters_path:
        args.inference_clusters_path = os.path.join(
            args.data_dir,
            default_inference_clusters_filename(use_passages, passage_type),
        )
    if not args.table_embeddings_path:
        args.table_embeddings_path = os.path.join(args.data_dir, "table_embeddings.json")
    if not args.tables_lake_dir:
        args.tables_lake_dir = os.path.join(args.data_dir, "lake")
    passage_suffix = passage_path_suffix(passage_type)
    if not getattr(args, "passage_embeddings_path", None):
        args.passage_embeddings_path = os.path.join(
            args.data_dir, f"passage_embeddings{passage_suffix}.json"
        )
    if not getattr(args, "corpus_path", None):
        args.corpus_path = os.path.join(args.data_dir, "corpus.jsonl")
    if not getattr(args, "uid_to_passage_id_path", None):
        args.uid_to_passage_id_path = os.path.join(
            args.data_dir, f"dpdisc_uid_to_passage_id{passage_suffix}.json"
        )
    if not getattr(args, "uid_to_table_id_path", None):
        args.uid_to_table_id_path = os.path.join(
            args.data_dir, "dpdisc_uid_to_table_id.json"
        )
    if not getattr(args, "passage_descriptions_path", None):
        args.passage_descriptions_path = resolve_passage_descriptions_source(args)
    # Default query file only when neither single-query nor file was given
    if not getattr(args, "query_file", None) and not getattr(args, "user_query", None):
        args.query_file = os.path.join(args.data_dir, "queries.json")
