#!/usr/bin/env python3
"""Sample HybridQA DPRs, dedupe ground truth, and write dpdisc_dr_queries JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm import LLMClient, chat, get_llm_client
from src.queries import dedupe_ground_truth
from src.utils import load_json, save_json

SYSTEM_PROMPT = """You convert data product requests into deep research queries.

A data product request (DPR) asks someone to compile, assemble, or build structured datasets with specific fields and schemas.

A deep research query is a natural-language question that requests high-level analysis to obtain insights — such as trends, comparisons, patterns, relationships, or explanatory findings — rather than raw data compilation.

Rules:
- Output exactly one question sentence.
- Do not mention datasets, tables, schemas, columns, or compilation tasks.
- Preserve the topical domain and underlying information need from the DPR.
- Frame the question as an analytical request suitable for a research assistant.
- Return only the final question text, with no preamble or explanation."""

CONVERSION_PROMPT = """Convert the following data product request into one deep research query.

DATA PRODUCT REQUEST:
{dpr_text}

Deep research query:"""

DEFAULT_TRAIN_PATH = REPO_ROOT / "data/hybridqa/HybridQA_train.jsonl"
DEFAULT_DPR_PATH = REPO_ROOT / "data/hybridqa/dpdisc_dpr_100.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data/hybridqa/dpdisc_dr_queries_100.json"

MIN_TABLE_THRESHOLD = 20
COVERAGE_BUCKETS = {
    "low": "[20,25]",
    "medium": "(25,35]",
    "high": "(35,72]",
}
BUCKET_BINS = [MIN_TABLE_THRESHOLD - 1, 25, 35, 72]
BUCKET_SAMPLE_SIZES = {"1": 34, "2": 33, "3": 33}
COVERAGE_LABELS = {"1": "low", "2": "medium", "3": "high"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dpdisc_dr_queries_100.json by sampling HybridQA_train.jsonl, "
            "deduplicating ground-truth ids, and optionally generating query text."
        ),
    )
    parser.add_argument(
        "--train-path",
        default=str(DEFAULT_TRAIN_PATH),
        help="HybridQA_train.jsonl source path.",
    )
    parser.add_argument(
        "--dpr-output",
        default=str(DEFAULT_DPR_PATH),
        help="Where to write the sampled dpdisc_dpr_100.json file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for dpdisc_dr_queries_100.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified DPR sampling (default: 42).",
    )
    parser.add_argument(
        "--reuse-queries",
        default=None,
        help=(
            "Existing dpdisc_dr_queries JSON; reuse query text by dpr_id instead of "
            "calling an LLM."
        ),
    )
    parser.add_argument(
        "--skip-dpr-output",
        action="store_true",
        help="Do not write dpdisc_dpr_100.json (only rebuild queries output).",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="openai",
        choices=["openai", "litellm", "vllm"],
        help="LLM backend when generating new query text.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-5-mini",
        help="LLM model name when generating new query text.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM sampling temperature when generating new query text.",
    )
    return parser.parse_args()


def load_hybridqa_train(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _ground_truth_from_train_record(record: Dict[str, Any]) -> Dict[str, Any]:
    gt = record.get("ground_truth") or {}
    return dedupe_ground_truth(
        {
            "n_table": len(gt.get("table") or []),
            "table": list(gt.get("table") or []),
            "n_text": len(gt.get("text") or []),
            "text": list(gt.get("text") or []),
            "n_synth_text": len(gt.get("synth_text") or []),
            "synth_text": list(gt.get("synth_text") or []),
        }
    )


def sample_dprs(train_path: str, *, seed: int) -> pd.DataFrame:
    records = load_hybridqa_train(train_path)
    rows = []
    for record in records:
        gt = _ground_truth_from_train_record(record)
        rows.append(
            {
                "dpr_id": record["dpr_id"],
                "dpr": record["DPR"],
                "n_gt_tables": len(gt.get("table") or []),
                "n_gt_texts": len(gt.get("text") or []),
                "n_gt_synth_texts": len(gt.get("synth_text") or []),
                "gt_tables": gt.get("table") or [],
                "gt_texts": gt.get("text") or [],
                "gt_synth_texts": gt.get("synth_text") or [],
            }
        )

    dpr_df = pd.DataFrame(rows)
    dpr_df = dpr_df[dpr_df["n_gt_tables"] >= MIN_TABLE_THRESHOLD].copy()
    dpr_df["bucket"] = pd.cut(
        dpr_df["n_gt_tables"],
        bins=BUCKET_BINS,
        labels=["1", "2", "3"],
    )

    samples = []
    for bucket, count in BUCKET_SAMPLE_SIZES.items():
        pool = dpr_df[dpr_df["bucket"] == bucket]
        samples.append(pool.sample(count, random_state=seed))
    sampled = pd.concat(samples).reset_index(drop=True)
    sampled["coverage"] = sampled["bucket"].map(COVERAGE_LABELS)
    return sampled


def build_dpr_payload(sampled: pd.DataFrame) -> Dict[str, Any]:
    dprs = []
    for row in sampled.to_dict(orient="records"):
        ground_truth = dedupe_ground_truth(
            {
                "n_table": len(row["gt_tables"]),
                "table": list(row["gt_tables"]),
                "n_text": len(row["gt_texts"]),
                "text": list(row["gt_texts"]),
                "n_synth_text": len(row["gt_synth_texts"]),
                "synth_text": list(row["gt_synth_texts"]),
            }
        )
        dprs.append(
            {
                "dpr_id": row["dpr_id"],
                "dpr": row["dpr"],
                "coverage": row["coverage"],
                "ground_truth": ground_truth,
            }
        )

    return {
        "dataset": "hybridqa",
        "split": "train",
        "n_samples": len(dprs),
        "coverage_buckets": COVERAGE_BUCKETS,
        "dprs": dprs,
    }


def convert_dpr_to_query(
    llm: LLMClient,
    dpr_text: str,
    *,
    temperature: float,
) -> str:
    prompt = CONVERSION_PROMPT.format(dpr_text=dpr_text.strip())
    query = chat(llm, prompt, system_prompt=SYSTEM_PROMPT, temperature=temperature)
    return query.strip().strip('"').strip("'")


def build_output_record(
    dpr: Dict[str, Any],
    query: str,
    qid: int,
) -> Dict[str, Any]:
    dpr_id = str(dpr.get("dpr_id") or f"dpr_{qid}")
    ground_truth = dpr.get("ground_truth")
    if isinstance(ground_truth, dict):
        ground_truth = dedupe_ground_truth(ground_truth)
    return {
        "qid": qid,
        "dpr_id": dpr_id,
        "coverage": dpr.get("coverage", "unknown"),
        "query": query,
        "ground_truth": ground_truth,
    }


def generate_queries(
    dprs: List[Dict[str, Any]],
    llm: LLMClient,
    *,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for qid, dpr in enumerate(tqdm(dprs, desc="Generating deep research queries")):
        dpr_text = (dpr.get("dpr") or dpr.get("DPR") or "").strip()
        if not dpr_text:
            failures.append({"qid": qid, "dpr_id": dpr.get("dpr_id"), "error": "missing DPR text"})
            continue
        try:
            query = convert_dpr_to_query(llm, dpr_text, temperature=temperature)
            if not query:
                raise ValueError("LLM returned an empty query")
            results.append(build_output_record(dpr, query, qid))
        except Exception as exc:
            failures.append({"qid": qid, "dpr_id": dpr.get("dpr_id"), "error": str(exc)})

    return results, failures


def _load_query_text_by_dpr(path: str) -> Dict[str, str]:
    records = load_json(path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    by_id: Dict[str, str] = {}
    for record in records:
        dpr_id = str(record.get("dpr_id") or "")
        query = (record.get("query") or record.get("query_text") or "").strip()
        if dpr_id and query:
            by_id[dpr_id] = query
    return by_id


def build_query_records(
    dprs: List[Dict[str, Any]],
    *,
    reuse_queries_path: Optional[str],
    llm_provider: str,
    llm_model: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    if reuse_queries_path:
        query_by_dpr = _load_query_text_by_dpr(reuse_queries_path)
        records = []
        missing = []
        for qid, dpr in enumerate(dprs):
            dpr_id = str(dpr["dpr_id"])
            query = query_by_dpr.get(dpr_id)
            if not query:
                missing.append(dpr_id)
                continue
            records.append(build_output_record(dpr, query, qid))
        if missing:
            raise ValueError(
                "Missing query text for DPR id(s): " + ", ".join(missing)
            )
        return records

    llm = get_llm_client(llm_provider, llm_model)
    results, failures = generate_queries(dprs, llm, temperature=temperature)
    if failures:
        details = ", ".join(f"{item['dpr_id']}: {item['error']}" for item in failures)
        raise RuntimeError(f"Failed to generate queries for: {details}")
    return results


def count_duplicate_text_queries(records: List[Dict[str, Any]]) -> int:
    dup_queries = 0
    for record in records:
        gt = record.get("ground_truth") or {}
        texts = gt.get("text")
        if isinstance(texts, list) and len(texts) != len(set(map(str, texts))):
            dup_queries += 1
    return dup_queries


def main() -> None:
    args = parse_args()
    sampled = sample_dprs(args.train_path, seed=args.seed)
    dpr_payload = build_dpr_payload(sampled)

    if not args.skip_dpr_output:
        save_json(args.dpr_output, dpr_payload)
        print(f"Wrote {len(dpr_payload['dprs'])} sampled DPRs to {args.dpr_output}")

    query_records = build_query_records(
        dpr_payload["dprs"],
        reuse_queries_path=args.reuse_queries,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        temperature=args.temperature,
    )
    save_json(args.output, query_records)

    dup_queries = count_duplicate_text_queries(query_records)
    print(f"Wrote {len(query_records)} deep research queries to {args.output}")
    print(f"Queries with duplicate text ids: {dup_queries}")


if __name__ == "__main__":
    main()
