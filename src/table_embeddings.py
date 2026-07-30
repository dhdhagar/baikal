"""Table embedding generation (placeholder)."""

import os

from typing import Dict, List

from src.embedding_client import get_embedding_client
from src.utils import load_json, log, save_json


def _build_schema_text(table_info: dict) -> str:
    title = str(table_info.get("title") or "").strip()
    columns = table_info.get("columns") or []
    if not isinstance(columns, list):
        columns = []
    columns_str = " | ".join(str(c) for c in columns)
    desc = str(table_info.get("description") or "").strip()
    return f"{title}. Columns: {columns_str}. {desc}"


def ensure_table_embeddings(
    path: str,
    tables_lake_dir: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    embedding_provider: str = "local",
    gpu: bool = False,
    silent: bool = False,
) -> str:
    """
    Return path to table embeddings JSON.

    If the file already exists, skip generation. Otherwise generate embeddings from
    `schema_descriptions.json` in the lake directory and write `{table_id: embedding}`.
    """
    if os.path.isfile(path):
        log(f"Using existing table embeddings: {path}", silent=silent)
        return path

    schema_path = os.path.join(tables_lake_dir, "schema_descriptions.json")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(
            f"Table embeddings not found at {path} and schema_descriptions.json not found at "
            f"{schema_path}. Provide table_embeddings.json or the HybridQA lake dir."
        )

    metas: Dict[str, dict] = load_json(schema_path)
    if not metas:
        raise ValueError(f"No table metadata found in {schema_path}.")

    table_ids = sorted(metas.keys())
    docs = [_build_schema_text(metas[tid]) for tid in table_ids]

    log(
        f"Generating embeddings for {len(table_ids)} tables with "
        f"{embedding_provider}/{embedding_model!r}…",
        silent=silent,
    )

    embedder = get_embedding_client(embedding_provider, embedding_model, gpu=gpu)
    embs = embedder.encode(
        docs,
        batch_size=32,
        show_progress_bar=not silent,
        feature="table_embedding_setup",
    )
    if hasattr(embs, "tolist"):
        embs = embs.tolist()
    if not isinstance(embs, list) or len(embs) != len(table_ids):
        raise RuntimeError("Unexpected embeddings output from embedding client.")

    out: Dict[str, List[float]] = {tid: list(map(float, embs[i])) for i, tid in enumerate(table_ids)}
    save_json(path, out, indent=2)
    log(f"Wrote table embeddings -> {path}", silent=silent)
    return path
