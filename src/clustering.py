"""One-time BERTopic clustering + filtering for inference clusters."""

import os
from collections import defaultdict
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from tqdm import tqdm

from src.embedding_client import get_embedding_client
from src.inference_clusters import load_passages_metadata, load_tables_metadata
from src.passage_embeddings import build_passage_text, ensure_passage_embeddings
from src.table_embeddings import ensure_table_embeddings
from src.cli_ui import get_ui
from src.utils import load_json, log, save_json


def build_schema_text(table_info: dict) -> str:
    title = table_info.get("title", "")
    columns = " | ".join(table_info.get("columns", []))
    desc = table_info.get("description", "")
    return f"{title}. Columns: {columns}. {desc}"


def _combine_embeddings_and_docs(
    tables_embed: Dict[str, List[float]],
    tables_meta: Dict[str, dict],
    passages_embed: Optional[Dict[str, List[float]]] = None,
    passages_meta: Optional[Dict[str, dict]] = None,
) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """Merge table and passage items for joint clustering."""
    item_ids: List[str] = []
    item_types: List[str] = []
    docs: List[str] = []
    embeddings: List[List[float]] = []

    for tid in sorted(tables_meta.keys()):
        item_ids.append(tid)
        item_types.append("table")
        docs.append(tables_meta[tid].get("description", ""))
        embeddings.append(tables_embed[tid])

    if passages_embed and passages_meta:
        for pid in sorted(passages_meta.keys(), key=lambda x: int(x[1:])):
            item_ids.append(pid)
            item_types.append("passage")
            docs.append(build_passage_text(passages_meta[pid]))
            embeddings.append(passages_embed[pid])

    return item_ids, item_types, docs, np.array(embeddings)


def run_bertopic_clustering(
    tables_embed: Dict[str, List[float]],
    tables_meta: Dict[str, dict],
    passages_embed: Optional[Dict[str, List[float]]] = None,
    passages_meta: Optional[Dict[str, dict]] = None,
    umap_n_neighbors: int = 10,
    umap_min_dist: float = 0.1,
    umap_n_components: int = 10,
    hdbscan_min_cluster_size: int = 3,
    min_topic_size: int = 3,
    vectorizer_min_df: int = 2,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, List[dict]]], Any]:
    """Cluster pre-computed embeddings with BERTopic; return topic_id -> {tables, passages}."""

    item_ids, item_types, docs, embeddings = _combine_embeddings_and_docs(
        tables_embed,
        tables_meta,
        passages_embed=passages_embed,
        passages_meta=passages_meta,
    )

    n_items = len(item_ids)
    eff_neighbors = min(umap_n_neighbors, max(2, n_items - 1))
    eff_components = min(umap_n_components, max(2, n_items - 2))
    eff_min_cluster = min(hdbscan_min_cluster_size, max(2, n_items // 3))
    eff_min_topic = min(min_topic_size, max(2, n_items // 3))

    umap_model = UMAP(
        n_neighbors=eff_neighbors,
        min_dist=umap_min_dist,
        n_components=eff_components,
        metric="cosine",
        random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=eff_min_cluster,
        metric="euclidean",
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.0,
        prediction_data=True,
    )
    eff_min_df = min(vectorizer_min_df, max(1, n_items // 5))
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=eff_min_df,
        ngram_range=(1, 2),
    )
    mmr = MaximalMarginalRelevance(diversity=0.3)
    representation_models = [mmr]

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=eff_min_topic,
        top_n_words=10,
        verbose=False,
        low_memory=False,
        representation_model=representation_models,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    clustered = defaultdict(lambda: {"tables": [], "passages": []})
    for idx, topic in enumerate(topics):
        item_id = item_ids[idx]
        item_type = item_types[idx]
        if item_type == "table":
            meta = tables_meta.get(item_id, {})
            clustered[str(topic)]["tables"].append(
                {
                    "table_id": item_id,
                    "title": meta.get("title", ""),
                    "domain": meta.get("domain", ""),
                    "columns": meta.get("columns", []),
                    "description": meta.get("description", ""),
                }
            )
        else:
            meta = (passages_meta or {}).get(item_id, {})
            clustered[str(topic)]["passages"].append(
                {
                    "passage_id": item_id,
                    "uid": meta.get("uid", ""),
                    "title": meta.get("title", ""),
                    "text": meta.get("text", ""),
                }
            )

    return dict(clustered), topic_model


def _topic_description(topic_model, topic_id: str) -> str:
    """Short natural-language cluster label from BERTopic top words."""
    if topic_id == "-1":
        return "Unclustered noise items"
    try:
        words = topic_model.get_topic(int(topic_id))
        if words:
            return ", ".join(w for w, _ in words[:12])
    except Exception:
        pass
    return f"Cluster {topic_id}"


def _item_embed_text(item: dict) -> str:
    if "table_id" in item:
        return build_schema_text(item)
    return build_passage_text(item)


def filter_clusters(
    raw_clusters: Dict[str, Dict[str, List[dict]]],
    topic_model=None,
    min_tables: int = 2,
    max_tables: int = 30,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_provider: str = "local",
    gpu: bool = False,
) -> List[dict]:
    """
    Drop noise/small clusters; split oversized ones with KMeans on item text.
    Output matches inference cluster schema used downstream.
    """
    embedder = None
    filtered = []
    dropped_noise = 0
    dropped_small = 0
    split_clusters = 0
    added_clusters = 0

    for cid, group in raw_clusters.items():
        tables = group.get("tables") or []
        passages = group.get("passages") or []
        items = tables + passages
        if cid == "-1":
            dropped_noise = len(items)
            continue
        if len(items) < min_tables:
            dropped_small += len(items)
            continue

        chunks = [(cid, tables, passages)]
        if len(items) > max_tables:
            if embedder is None:
                embedder = get_embedding_client(embedding_provider, embedding_model, gpu=gpu)
            texts = [_item_embed_text(item) for item in items]
            embeds = embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                feature="clustering_embedding",
            )
            n_sub = ceil(len(items) / max_tables)
            labels = KMeans(n_clusters=n_sub, random_state=42, n_init="auto").fit_predict(
                embeds
            )
            sub_tables: Dict[int, List[dict]] = defaultdict(list)
            sub_passages: Dict[int, List[dict]] = defaultdict(list)
            for lbl, item in zip(labels, items):
                if "table_id" in item:
                    sub_tables[lbl].append(item)
                else:
                    sub_passages[lbl].append(item)
            chunk_labels = sorted(set(sub_tables) | set(sub_passages))
            chunks = [
                (f"{cid}_sub{lbl}", sub_tables[lbl], sub_passages[lbl])
                for lbl in chunk_labels
            ]
            split_clusters += 1
            added_clusters += len(chunks) - 1

        for cluster_key, tbl_group, ps_group in chunks:
            group_items = tbl_group + ps_group
            if len(group_items) < min_tables:
                dropped_small += len(group_items)
                continue
            base_topic = cluster_key.split("_sub")[0]
            desc = (
                _topic_description(topic_model, base_topic)
                if topic_model is not None
                else f"Items grouped as topic {base_topic}"
            )
            filtered.append(
                {
                    "cluster_id": cluster_key,
                    "description": desc,
                    "tables": tbl_group,
                    "passages": ps_group,
                }
            )

    log(
        f"Cluster filter: dropped_noise={dropped_noise}, dropped_small={dropped_small}, "
        f"split={split_clusters} clusters, added={added_clusters} new clusters; total kept={len(filtered)} clusters"
    )
    return filtered


def build_inference_clusters(
    embeddings_path: str,
    tables_lake_dir: str,
    output_path: str,
    embedding_model: str,
    embedding_provider: str = "local",
    gpu: bool = False,
    silent: bool = False,
    use_passages: bool = False,
    passage_embeddings_path: Optional[str] = None,
    passage_descriptions_path: Optional[str] = None,
    bertopic_kwargs: Optional[dict] = None,
    filter_kwargs: Optional[dict] = None,
) -> List[dict]:
    """Run clustering + filtering and write inference_clusters.json."""
    tables_embed = load_json(embeddings_path)
    tables_meta = load_tables_metadata(tables_lake_dir)

    passages_embed = None
    passages_meta = None
    if use_passages:
        if not passage_embeddings_path or not passage_descriptions_path:
            raise ValueError(
                "passage_embeddings_path and passage_descriptions_path are required when use_passages=True"
            )
        passages_embed = load_json(passage_embeddings_path)
        passages_meta = load_passages_metadata(passage_descriptions_path)
        log(
            f"Clustering {len(tables_embed)} tables and {len(passages_embed)} passages…",
            silent=silent,
        )
    else:
        log(f"Clustering {len(tables_embed)} tables...", silent=silent)

    ui = get_ui()
    if ui.rich_cli:
        ui.set_phase("Clustering tables" if not use_passages else "Clustering tables and passages")

    ui_active = ui.rich_cli
    with tqdm(
        total=2,
        desc="Clustering",
        disable=silent or ui_active,
        leave=False,
    ) as pbar:
        raw_clusters, topic_model = run_bertopic_clustering(
            tables_embed,
            tables_meta,
            passages_embed=passages_embed,
            passages_meta=passages_meta,
            **(bertopic_kwargs or {}),
        )
        if ui_active:
            ui.set_status("BERTopic clustering")
        else:
            pbar.set_postfix_str("BERTopic")
        pbar.update(1)
        inference_clusters = filter_clusters(
            raw_clusters,
            topic_model=topic_model,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            gpu=gpu,
            **(filter_kwargs or {}),
        )
        if ui_active:
            ui.set_status("Filtering clusters")
        else:
            pbar.set_postfix_str("filter")
        pbar.update(1)
    save_json(output_path, inference_clusters)
    log(f"Wrote inference clusters -> {output_path}", silent=silent)
    return inference_clusters


def ensure_inference_clusters(args) -> List[dict]:
    """Ensure embeddings exist; build or load inference clusters unless disabled."""
    ui = get_ui()
    if ui.rich_cli:
        ui.set_phase("Embeddings")
    ui_active = ui.rich_cli
    embed_steps = 2 if args.use_passages else 1
    with tqdm(
        total=embed_steps,
        desc="Embeddings",
        disable=args.silent or ui_active,
        leave=False,
    ) as pbar:
        if ui_active:
            ui.set_status("Table embeddings")
        ensure_table_embeddings(
            args.table_embeddings_path,
            args.tables_lake_dir,
            embedding_model=args.embedding_model,
            embedding_provider=args.embedding_provider,
            gpu=args.gpu,
            silent=args.silent,
        )
        pbar.update(1)
        if args.use_passages:
            if ui_active:
                ui.set_status("Passage embeddings")
            ensure_passage_embeddings(
                args.passage_embeddings_path,
                args.data_dir,
                args.corpus_path,
                args.uid_to_passage_id_path,
                args.passage_descriptions_path,
                passage_type=args.passage_type,
                embedding_model=args.embedding_model,
                embedding_provider=args.embedding_provider,
                gpu=args.gpu,
                silent=args.silent,
            )
            pbar.update(1)

    if not args.use_clustering:
        log(
            "Clustering disabled; per-query top-k cluster built in QueryRunner.",
            silent=args.silent,
        )
        return []

    if os.path.isfile(args.inference_clusters_path):
        log(
            f"Using existing inference clusters: {args.inference_clusters_path}",
            silent=args.silent,
        )
        return load_json(args.inference_clusters_path)

    return build_inference_clusters(
        embeddings_path=args.table_embeddings_path,
        tables_lake_dir=args.tables_lake_dir,
        output_path=args.inference_clusters_path,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        gpu=args.gpu,
        silent=args.silent,
        use_passages=args.use_passages,
        passage_embeddings_path=args.passage_embeddings_path if args.use_passages else None,
        passage_descriptions_path=args.passage_descriptions_path if args.use_passages else None,
        bertopic_kwargs={
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_n_components": args.umap_n_components,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "min_topic_size": args.min_topic_size,
            "vectorizer_min_df": args.vectorizer_min_df,
        },
        filter_kwargs={
            "min_tables": args.cluster_min_tables,
            "max_tables": args.cluster_max_tables,
        },
    )
