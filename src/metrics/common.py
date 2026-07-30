"""Shared helpers for evaluation metrics."""

from __future__ import annotations

import functools
import os
import re
from typing import Any, Dict, List, Optional, Set

from src.utils import load_json, save_json

TABLE_ID_RE = re.compile(r"^T\d+$", re.IGNORECASE)
PASSAGE_ID_RE = re.compile(r"^P\d+$", re.IGNORECASE)
PASSAGE_CITATION_RE = re.compile(r"\[P(\d+)\]|\bP(\d+)\b", re.IGNORECASE)
TABLE_CITATION_RE = re.compile(r"\[T(\d+)\]|\bT(\d+)\b", re.IGNORECASE)

ORDINAL_SCORE_LABELS: Dict[str, float] = {
    "none": 0.0,
    "minimal": 0.25,
    "partial": 0.5,
    "substantial": 0.75,
    "full": 1.0,
}
ORDINAL_SCORE_VALUES = tuple(ORDINAL_SCORE_LABELS.values())


def round4(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def normalize_table_ids(table_ids: Optional[List[str]]) -> Set[str]:
    if not table_ids:
        return set()
    return {str(t).upper() for t in table_ids}


def normalize_passage_ids(passage_ids: Optional[List[str]]) -> Set[str]:
    return normalize_table_ids(passage_ids)


def snap_ordinal_score(value: Any) -> float:
    """Map an LLM label or numeric value to {0, 0.25, 0.5, 0.75, 1.0}."""
    if isinstance(value, str):
        label = value.strip().lower()
        if label in ORDINAL_SCORE_LABELS:
            return ORDINAL_SCORE_LABELS[label]
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(ORDINAL_SCORE_VALUES, key=lambda s: abs(s - numeric))


def snap_binary_score(value: Any) -> float:
    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"yes", "true", "1", "grounded"}:
            return 1.0
        if label in {"no", "false", "0", "not_grounded", "ungrounded"}:
            return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 if numeric >= 0.5 else 0.0


@functools.lru_cache(maxsize=8)
def load_uid_to_table_id_mapping(path: str) -> Dict[str, str]:
    """Load HybridQA table uid -> T<id> mapping from JSON."""
    if not path or not os.path.isfile(path):
        return {}
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("uid_to_table_id"), dict):
        mapping = data["uid_to_table_id"]
    elif isinstance(data, dict):
        mapping = data
    else:
        return {}
    return {str(uid): str(table_id) for uid, table_id in mapping.items()}


@functools.lru_cache(maxsize=8)
def load_uid_to_passage_id_mapping(path: str) -> Dict[str, str]:
    """Load HybridQA passage uid -> P<id> mapping from JSON."""
    if not path or not os.path.isfile(path):
        return {}
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    return {str(uid): str(passage_id) for uid, passage_id in data.items()}


def map_gt_table_id(item_id: str, uid_to_table_id: Optional[Dict[str, str]] = None) -> str:
    """Normalize a ground-truth table id to canonical ``T<id>`` form."""
    value = str(item_id).strip()
    if TABLE_ID_RE.match(value):
        return value.upper()
    mapping = uid_to_table_id or {}
    mapped = mapping.get(value)
    if mapped:
        return mapped.upper()
    return value.upper()


def map_gt_passage_id(
    item_id: str,
    uid_to_passage_id: Optional[Dict[str, str]] = None,
) -> str:
    """Normalize a ground-truth passage id to canonical ``P<id>`` form."""
    value = str(item_id).strip()
    if PASSAGE_ID_RE.match(value):
        return value.upper()
    mapping = uid_to_passage_id or {}
    mapped = mapping.get(value)
    if mapped:
        return mapped.upper()
    return value.upper()


def _map_gt_ids(
    item_ids: List[Any],
    mapper,
    uid_mapping: Optional[Dict[str, str]] = None,
) -> List[str]:
    return [mapper(str(item_id), uid_mapping) for item_id in item_ids]


def extract_gt_table_ids(
    ground_truth: Optional[Dict[str, Any]],
    *,
    uid_to_table_id: Optional[Dict[str, str]] = None,
) -> Optional[List[str]]:
    """Return ground-truth table IDs from either ``tables`` or ``table`` key."""
    if not isinstance(ground_truth, dict):
        return None
    for key in ("tables", "table"):
        value = ground_truth.get(key)
        if isinstance(value, list):
            return _map_gt_ids(value, map_gt_table_id, uid_to_table_id)
    return None


def extract_gt_passage_ids(
    ground_truth: Optional[Dict[str, Any]],
    *,
    uid_to_passage_id: Optional[Dict[str, str]] = None,
    passage_type: str = "synth",
) -> Optional[List[str]]:
    """Return ground-truth passage IDs from passage/text keys in ``ground_truth``."""
    if not isinstance(ground_truth, dict):
        return None
    if passage_type == "raw":
        keys = ("passages", "texts", "text")
    else:
        keys = ("passages", "texts", "synth_text")
    for key in keys:
        value = ground_truth.get(key)
        if isinstance(value, list):
            return _map_gt_ids(value, map_gt_passage_id, uid_to_passage_id)
    return None


@functools.lru_cache(maxsize=8)
def count_lake_tables(tables_lake_dir: str) -> int:
    """Number of tables in the data lake (from schema_descriptions.json)."""
    schema_path = os.path.join(tables_lake_dir, "schema_descriptions.json")
    if not os.path.isfile(schema_path):
        return 0
    metadata = load_json(schema_path)
    return len(metadata)


def is_successful_iteration(execution: Optional[dict]) -> bool:
    if not execution:
        return False
    return bool(execution.get("ok")) and int(execution.get("row_count", 0) or 0) > 0


def answered_with_passage_evidence(iteration: dict) -> bool:
    """True when the answer cites at least one passage."""
    answer = (iteration.get("answer") or "").strip()
    if not answer:
        return False
    return bool(extract_passages_from_answer(answer))


def is_report_eligible_iteration(iteration: dict) -> bool:
    """Include successful SQL runs, passage-only answers, and passage fallback when SQL fails."""
    if iteration.get("needs_sql"):
        if is_successful_iteration(iteration.get("execution")):
            return True
        return answered_with_passage_evidence(iteration)
    return True


def is_finding_iteration(iteration: dict) -> bool:
    """A budget step that produced a report-eligible finding."""
    if not is_report_eligible_iteration(iteration):
        return False
    return bool((iteration.get("answer") or "").strip())


_ITERATION_PAYLOAD_PRIORITY = ("step", "finding_idx", "skipped")


def reorder_iteration_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Place high-signal fields first for human-readable iteration JSON."""
    ordered: Dict[str, Any] = {}
    for key in _ITERATION_PAYLOAD_PRIORITY:
        if key in payload:
            ordered[key] = payload[key]
    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def assign_finding_indices(iterations: List[dict]) -> None:
    """Set finding_idx on each iteration (1-based report index, or None)."""
    idx = 0
    for it in iterations:
        if is_report_eligible_iteration(it):
            idx += 1
            it["finding_idx"] = idx
        else:
            it["finding_idx"] = None


def sync_finding_indices_to_query_dir(query_dir: str, iterations: List[dict]) -> None:
    """Write finding_idx from iterations into per-step iteration JSON files."""
    for it in iterations:
        step = it.get("step")
        if not step:
            continue
        path = os.path.join(query_dir, f"iteration_{step:03d}.json")
        if not os.path.isfile(path):
            continue
        payload = load_json(path)
        if payload.get("skipped"):
            continue
        payload["finding_idx"] = it.get("finding_idx")
        save_json(path, reorder_iteration_payload(payload))


def _extract_ids(text: str, pattern: re.Pattern[str], prefix: str) -> Set[str]:
    if not text:
        return set()
    ids: Set[str] = set()
    for match in pattern.finditer(text):
        num = next(group for group in match.groups() if group is not None)
        ids.add(f"{prefix}{num}")
    return ids


def extract_passages_from_answer(text: str) -> Set[str]:
    """Return passage IDs cited in an answer (``[P<id>]`` or bare ``P<id>``)."""
    return _extract_ids(text, PASSAGE_CITATION_RE, "P")


def extract_tables_from_answer(text: str) -> Set[str]:
    """Return table IDs cited in an answer (``[T<id>]`` or bare ``T<id>``)."""
    return _extract_ids(text, TABLE_CITATION_RE, "T")
