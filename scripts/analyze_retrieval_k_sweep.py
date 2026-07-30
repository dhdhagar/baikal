#!/usr/bin/env python3
"""Measure top-k GT recall and cluster reachability over a sweep of k."""

from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_analysis_common import (
    dataset_paths,
    evaluate_retrieval,
    load_queries,
    parse_positive_ints,
    print_summary,
    summarize_rows,
    write_analysis_outputs,
)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument(
        "--ks",
        default="10,25,50,100,200,500",
        help="Comma-separated k values applied to both tables and passages.",
    )
    parser.add_argument("--n-queries", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the pipeline's coverage-stratified query sampling.",
    )
    parser.add_argument("--passage-type", choices=("raw", "synth"), default="raw")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-provider", default="local")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--table-embeddings-path", type=Path)
    parser.add_argument("--passage-embeddings-path", type=Path)
    parser.add_argument("--inference-clusters-path", type=Path)
    parser.add_argument("--uid-to-table-id-path", type=Path)
    parser.add_argument("--uid-to-passage-id-path", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = parse_positive_ints(args.ks)
    if args.n_queries is not None and args.n_queries <= 0:
        raise ValueError("--n-queries must be positive")
    if args.n_bootstrap < 0:
        raise ValueError("--n-bootstrap must be non-negative")

    defaults = dataset_paths(args.data_dir, args.passage_type)
    table_embeddings_path = (
        args.table_embeddings_path or defaults["table_embeddings_path"]
    )
    passage_embeddings_path = (
        args.passage_embeddings_path or defaults["passage_embeddings_path"]
    )
    inference_clusters_path = (
        args.inference_clusters_path or defaults["inference_clusters_path"]
    )
    uid_to_table_id_path = (
        args.uid_to_table_id_path or defaults["uid_to_table_id_path"]
    )
    uid_to_passage_id_path = (
        args.uid_to_passage_id_path or defaults["uid_to_passage_id_path"]
    )
    queries = load_queries(
        args.query_file,
        n_queries=args.n_queries,
        seed=args.seed,
        stratified=args.stratified,
    )
    rows = evaluate_retrieval(
        queries=queries,
        embedding_model=args.embedding_model,
        embedding_provider=args.embedding_provider,
        gpu=args.gpu,
        table_embeddings_path=table_embeddings_path,
        passage_embeddings_path=passage_embeddings_path,
        inference_clusters_path=inference_clusters_path,
        uid_to_table_id_path=uid_to_table_id_path,
        uid_to_passage_id_path=uid_to_passage_id_path,
        passage_type=args.passage_type,
        ks=ks,
    )
    summary = summarize_rows(
        rows,
        group_key="k",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print_summary(summary, "k")
    write_analysis_outputs(
        output_path=args.output,
        payload={
            "analysis": "retrieval_k_sweep",
            "data_dir": str(args.data_dir),
            "query_file": str(args.query_file),
            "n_queries": len(queries),
            "query_ids": [query["query_id"] for query in queries],
            "passage_type": args.passage_type,
            "embedding_model": args.embedding_model,
            "embedding_provider": args.embedding_provider,
            "ks": ks,
            "artifacts": {
                "table_embeddings": str(table_embeddings_path),
                "passage_embeddings": str(passage_embeddings_path),
                "inference_clusters": str(inference_clusters_path),
            },
            "bootstrap": {
                "n_resamples": args.n_bootstrap,
                "seed": args.seed,
                "unit": "query",
            },
        },
        summary=summary,
        per_query=rows,
        group_key="k",
    )


if __name__ == "__main__":
    main()
