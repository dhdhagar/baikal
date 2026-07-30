"""Load and sample user queries."""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.utils import load_json, log, slugify

_GT_TABLE_KEYS = ("tables", "table")
_GT_TEXT_KEYS = ("passages", "texts", "text")
_GT_SYNTH_TEXT_KEYS = ("synth_text",)
_GT_PASSAGE_KEYS = _GT_TEXT_KEYS + _GT_SYNTH_TEXT_KEYS


def _unique_preserve_order(items: List[Any]) -> List[Any]:
    seen: set[str] = set()
    out: List[Any] = []
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def dedupe_ground_truth(gt: Dict[str, Any]) -> Dict[str, Any]:
    """Drop duplicate table/passage ids while preserving first-seen order."""
    gt = dict(gt)
    for key in _GT_TABLE_KEYS + _GT_PASSAGE_KEYS:
        value = gt.get(key)
        if isinstance(value, list):
            gt[key] = _unique_preserve_order(value)

    if "n_table" in gt:
        for key in _GT_TABLE_KEYS:
            value = gt.get(key)
            if isinstance(value, list):
                gt["n_table"] = len(value)
                break

    if "n_text" in gt:
        for key in _GT_TEXT_KEYS:
            value = gt.get(key)
            if isinstance(value, list):
                gt["n_text"] = len(value)
                break

    if "n_synth_text" in gt:
        for key in _GT_SYNTH_TEXT_KEYS:
            value = gt.get(key)
            if isinstance(value, list):
                gt["n_synth_text"] = len(value)
                break

    return gt


def _resolve_query_id(item: Dict[str, Any], text: str) -> str:
    """Return query_id from record fields, falling back to slugified query text."""
    qid = item.get("query_id")
    if qid is None:
        qid = item.get("qid")
    if qid is None:
        qid = slugify(text)
    return str(qid)


def _normalize_record(item: Any, idx: int) -> Optional[Dict[str, Any]]:
    """Accept several query JSON shapes used in this repo."""
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {
            "query_id": f"Q{idx}",
            "query_text": text,
            "coverage": "unknown",
        }

    if not isinstance(item, dict):
        return None

    text = (
        item.get("query_text")
        or item.get("query")
        or item.get("user_query")
        or ""
    ).strip()
    if not text:
        return None

    qid = _resolve_query_id(item, text)
    coverage = (
        item.get("coverage")
        or item.get("coverage_bucket")
        or "unknown"
    ).lower()
    rec: Dict[str, Any] = {
        "query_id": qid,
        "query_text": text,
        "coverage": coverage,
    }
    gt = item.get("ground_truth")
    if isinstance(gt, dict):
        rec["ground_truth"] = dedupe_ground_truth(gt)
    return rec


def load_queries_from_file(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    records = []

    if isinstance(data, dict):
        # e.g. {"evaluation_queries": [...]}
        if "evaluation_queries" in data:
            data = data["evaluation_queries"]
        else:
            data = list(data.values())

    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            rec = _normalize_record(item, idx)
            if rec:
                records.append(rec)
    return records


def stratified_sample(
    queries: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Balance low / medium / high coverage buckets when possible."""
    buckets = defaultdict(list)
    for q in queries:
        cov = q.get("coverage", "unknown").lower()
        if cov in ("low", "medium", "high"):
            buckets[cov].append(q)
        else:
            buckets["other"].append(q)

    target_buckets = ["low", "medium", "high"]
    active = [b for b in target_buckets if buckets[b]]
    if not active:
        return rng.sample(queries, min(n, len(queries)))

    per_bucket = max(1, n // len(active))
    selected = []
    for b in active:
        pool = buckets[b][:]
        rng.shuffle(pool)
        selected.extend(pool[:per_bucket])

    # Fill remainder from any leftover queries
    chosen_ids = {q["query_id"] for q in selected}
    remainder = [q for q in queries if q["query_id"] not in chosen_ids]
    rng.shuffle(remainder)
    for q in remainder:
        if len(selected) >= n:
            break
        selected.append(q)

    rng.shuffle(selected)
    return selected[:n]


def _parse_query_ids(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    return ids or None


def filter_queries_by_id(
    queries: List[Dict[str, Any]],
    query_ids: List[str],
) -> List[Dict[str, Any]]:
    by_id = {q["query_id"]: q for q in queries}
    missing = [qid for qid in query_ids if qid not in by_id]
    if missing:
        raise ValueError(f"Query ID(s) not found: {', '.join(missing)}")
    return [by_id[qid] for qid in query_ids]


def sample_queries(
    queries: List[Dict[str, Any]],
    n_queries: Optional[int],
    stratified: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    if not n_queries or n_queries >= len(queries):
        return queries

    rng = random.Random(seed)
    if stratified:
        return stratified_sample(queries, n_queries, rng)
    return rng.sample(queries, n_queries)


def resolve_queries(args) -> List[Dict[str, Any]]:
    query_ids = _parse_query_ids(getattr(args, "query_id", None))

    if args.user_query and args.query_file:
        raise ValueError("Provide either --user_query or --query_file, not both.")

    if args.user_query:
        if query_ids:
            raise ValueError("--query_id cannot be used with --user_query.")
        return [
            {
                "query_id": "user_query",
                "query_text": args.user_query.strip(),
                "coverage": "unknown",
            }
        ]

    if not args.query_file:
        raise ValueError("Provide --user_query or --query_file.")

    all_queries = load_queries_from_file(args.query_file)
    if not all_queries:
        raise ValueError(f"No queries found in {args.query_file}")

    queries = all_queries
    if query_ids:
        queries = filter_queries_by_id(all_queries, query_ids)

    sampled = sample_queries(
        queries,
        args.n_queries,
        args.stratified_sampling,
        args.seed,
    )
    log(
        f"Loaded {len(all_queries)} queries from {args.query_file}; "
        f"processing {len(sampled)}",
        silent=args.silent,
    )
    return sampled
