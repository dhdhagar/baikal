#!/usr/bin/env python3
"""Compare embedding models at fixed k using an existing cluster artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import re
from pathlib import Path

from retrieval_analysis_common import (
    dataset_paths,
    evaluate_retrieval,
    load_queries,
    print_summary,
    require_files,
    summarize_rows,
    write_analysis_outputs,
)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Embedding model names, separated by spaces.",
    )
    parser.add_argument("--embedding-provider", default="local")
    parser.add_argument("--default-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--n-queries", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the pipeline's coverage-stratified query sampling.",
    )
    parser.add_argument("--passage-type", choices=("raw", "synth"), default="raw")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Model-specific embeddings (default: DATA_DIR/retrieval_model_artifacts).",
    )
    parser.add_argument(
        "--inference-clusters-path",
        type=Path,
        help="Existing cluster JSON (default: DATA_DIR raw/synth inference clusters).",
    )
    parser.add_argument(
        "--build-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate missing table/passage embeddings. Clusters are never generated.",
    )
    parser.add_argument(
        "--reuse-default-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse DATA_DIR defaults for --default-model instead of rebuilding them.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def artifact_slug(provider: str, model: str) -> str:
    readable = re.sub(r"[^a-zA-Z0-9]+", "-", f"{provider}-{model}").strip("-").lower()
    digest = hashlib.sha1(f"{provider}:{model}".encode()).hexdigest()[:8]
    return f"{readable[:64].rstrip('-')}-{digest}"


def model_artifacts(
    *,
    args: argparse.Namespace,
    defaults: dict[str, Path],
    model: str,
) -> dict[str, Path]:
    use_defaults = (
        args.reuse_default_artifacts
        and args.embedding_provider == "local"
        and model == args.default_model
    )
    if use_defaults:
        table_embeddings = defaults["table_embeddings_path"]
        passage_embeddings = defaults["passage_embeddings_path"]
    else:
        model_dir = args.artifact_dir / artifact_slug(args.embedding_provider, model)
        suffix = "_raw" if args.passage_type == "raw" else ""
        table_embeddings = model_dir / "table_embeddings.json"
        passage_embeddings = model_dir / f"passage_embeddings{suffix}.json"
    return {
        "table_embeddings": table_embeddings,
        "passage_embeddings": passage_embeddings,
    }


def ensure_model_artifacts(
    *,
    args: argparse.Namespace,
    defaults: dict[str, Path],
    model: str,
    artifacts: dict[str, Path],
) -> None:
    from src.passage_embeddings import ensure_passage_embeddings
    from src.table_embeddings import ensure_table_embeddings

    if not args.build_artifacts:
        require_files(
            (
                artifacts["table_embeddings"],
                artifacts["passage_embeddings"],
            )
        )
        return

    artifacts["table_embeddings"].parent.mkdir(parents=True, exist_ok=True)
    ensure_table_embeddings(
        str(artifacts["table_embeddings"]),
        str(defaults["tables_lake_dir"]),
        embedding_model=model,
        embedding_provider=args.embedding_provider,
        gpu=args.gpu,
    )
    ensure_passage_embeddings(
        str(artifacts["passage_embeddings"]),
        str(args.data_dir),
        str(defaults["corpus_path"]),
        str(defaults["uid_to_passage_id_path"]),
        str(defaults["passage_descriptions_path"]),
        passage_type=args.passage_type,
        embedding_model=model,
        embedding_provider=args.embedding_provider,
        gpu=args.gpu,
    )


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.n_queries is not None and args.n_queries <= 0:
        raise ValueError("--n-queries must be positive")
    if args.n_bootstrap < 0:
        raise ValueError("--n-bootstrap must be non-negative")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models contains duplicates")

    defaults = dataset_paths(args.data_dir, args.passage_type)
    args.artifact_dir = (
        args.artifact_dir or args.data_dir / "retrieval_model_artifacts"
    )
    inference_clusters_path = (
        args.inference_clusters_path or defaults["inference_clusters_path"]
    )
    require_files((inference_clusters_path,))
    print(f"Using fixed inference clusters: {inference_clusters_path}")

    queries = load_queries(
        args.query_file,
        n_queries=args.n_queries,
        seed=args.seed,
        stratified=args.stratified,
    )
    all_rows = []
    artifacts_by_model = {}
    for model_index, model in enumerate(args.models, start=1):
        print(f"\n=== Model {model_index}/{len(args.models)}: {model} ===", flush=True)
        artifacts = model_artifacts(
            args=args,
            defaults=defaults,
            model=model,
        )
        ensure_model_artifacts(
            args=args,
            defaults=defaults,
            model=model,
            artifacts=artifacts,
        )
        rows = evaluate_retrieval(
            queries=queries,
            embedding_model=model,
            embedding_provider=args.embedding_provider,
            gpu=args.gpu,
            table_embeddings_path=artifacts["table_embeddings"],
            passage_embeddings_path=artifacts["passage_embeddings"],
            inference_clusters_path=inference_clusters_path,
            uid_to_table_id_path=defaults["uid_to_table_id_path"],
            uid_to_passage_id_path=defaults["uid_to_passage_id_path"],
            passage_type=args.passage_type,
            ks=(args.k,),
        )
        for row in rows:
            row["model"] = model
        all_rows.extend(rows)
        artifacts_by_model[model] = {
            key: str(value)
            for key, value in artifacts.items()
        }
        gc.collect()

    summary = summarize_rows(
        all_rows,
        group_key="model",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print()
    print_summary(summary, "model")
    write_analysis_outputs(
        output_path=args.output,
        payload={
            "analysis": "retrieval_embedding_model_comparison",
            "data_dir": str(args.data_dir),
            "query_file": str(args.query_file),
            "n_queries": len(queries),
            "query_ids": [query["query_id"] for query in queries],
            "passage_type": args.passage_type,
            "embedding_provider": args.embedding_provider,
            "models": args.models,
            "k": args.k,
            "inference_clusters": str(inference_clusters_path),
            "artifacts_by_model": artifacts_by_model,
            "bootstrap": {
                "n_resamples": args.n_bootstrap,
                "seed": args.seed,
                "unit": "query",
            },
        },
        summary=summary,
        per_query=all_rows,
        group_key="model",
    )


if __name__ == "__main__":
    main()
