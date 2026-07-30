"""OpenCode session usage capture and cost aggregation."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from typing import Any, Dict, List, Optional

from src.tracking import UsageSummary, UsageTracker, format_usage_line

# OpenCode session ids look like ``ses_152edf3bbffePl8AmprS4jEqf9``.
SESSION_ID_RE = re.compile(r"^ses_[A-Za-z0-9_-]+$")

_OPENCODE_DB_COLUMNS = (
    "id",
    "cost",
    "tokens_input",
    "tokens_output",
    "tokens_reasoning",
    "tokens_cache_read",
    "tokens_cache_write",
    "model",
    "title",
    "directory",
)


def is_valid_session_id(session_id: str) -> bool:
    return bool(session_id and SESSION_ID_RE.match(session_id))


def load_opencode_session_ids(*, xdg_data_home: str) -> List[str]:
    """Return OpenCode session ids from a per-workspace data home, oldest first."""
    db_path = os.path.join(xdg_data_home, "opencode", "opencode.db")
    if not os.path.isfile(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT id FROM session ORDER BY rowid").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [
        session_id
        for row in rows
        if isinstance((session_id := row[0]), str) and is_valid_session_id(session_id)
    ]


def load_latest_opencode_session_id(*, xdg_data_home: str) -> Optional[str]:
    """Return the most recent OpenCode session id from a per-workspace data home."""
    session_ids = load_opencode_session_ids(xdg_data_home=xdg_data_home)
    if not session_ids:
        return None
    return session_ids[-1]


def parse_session_id_from_json_stdout(stdout: str) -> Optional[str]:
    """Return the first sessionID found in OpenCode ``--format json`` NDJSON stdout."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = payload.get("sessionID")
        if isinstance(session_id, str) and is_valid_session_id(session_id):
            return session_id
    return None


def _usage_summary_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(data.get("prompt_tokens") or 0),
        "completion_tokens": int(data.get("completion_tokens") or 0),
        "total_tokens": int(data.get("total_tokens") or 0),
        "cost_usd": float(data.get("cost_usd") or 0.0),
        "time_taken": float(data.get("time_taken") or 0.0),
        "n_calls": int(data.get("n_calls") or 0),
    }


def normalize_opencode_db_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map an OpenCode ``session`` table row to a usage dict."""
    tokens_input = int(row.get("tokens_input") or 0)
    tokens_output = int(row.get("tokens_output") or 0)
    tokens_reasoning = int(row.get("tokens_reasoning") or 0)
    tokens_cache_read = int(row.get("tokens_cache_read") or 0)
    tokens_cache_write = int(row.get("tokens_cache_write") or 0)

    prompt_tokens = tokens_input + tokens_cache_read + tokens_cache_write
    completion_tokens = tokens_output + tokens_reasoning
    total_tokens = prompt_tokens + completion_tokens

    model_info = row.get("model")
    provider = ""
    model = ""
    if isinstance(model_info, dict):
        provider = str(model_info.get("providerID") or "")
        model = str(model_info.get("id") or "")
    model_label = f"{provider}/{model}" if provider and model else model or provider

    return {
        "session_id": row.get("id"),
        "source": "opencode_db",
        "model": model_label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(float(row.get("cost") or 0.0), 6),
        "time_taken": 0.0,
        "n_calls": 1,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_reasoning": tokens_reasoning,
        "tokens_cache_read": tokens_cache_read,
        "tokens_cache_write": tokens_cache_write,
        "title": row.get("title"),
        "directory": row.get("directory"),
    }


def fetch_opencode_session_usage(
    session_id: str,
    *,
    xdg_data_home: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Read token/cost totals for one OpenCode session from the local DB."""
    if not is_valid_session_id(session_id):
        return None

    columns = ", ".join(_OPENCODE_DB_COLUMNS)
    query = f"SELECT {columns} FROM session WHERE id = '{session_id}'"
    env = os.environ.copy()
    if xdg_data_home:
        env["XDG_DATA_HOME"] = xdg_data_home
    try:
        completed = subprocess.run(
            ["opencode", "db", query, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    try:
        rows = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not rows:
        return None
    return normalize_opencode_db_row(rows[0])


def fetch_all_opencode_session_usage(
    *,
    xdg_data_home: str,
) -> Optional[Dict[str, Any]]:
    """Sum token/cost totals across all OpenCode sessions in a workspace DB."""
    session_ids = load_opencode_session_ids(xdg_data_home=xdg_data_home)
    if not session_ids:
        return None

    usages = [
        usage
        for session_id in session_ids
        if (usage := fetch_opencode_session_usage(session_id, xdg_data_home=xdg_data_home))
    ]
    if not usages:
        return None

    merged = merge_usage_summaries(usages).to_dict()
    merged["source"] = "opencode_db"
    merged["session_ids"] = [
        str(item["session_id"])
        for item in usages
        if item.get("session_id")
    ]
    if merged["session_ids"]:
        merged["session_id"] = merged["session_ids"][-1]
    return merged


def combine_query_usage(
    pipeline: Dict[str, Any],
    opencode: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build nested per-query usage with pipeline, optional opencode, and total."""
    pipeline_summary = UsageSummary(**_usage_summary_fields(pipeline or {}))
    total = pipeline_summary.copy()
    combined: Dict[str, Any] = {
        "pipeline": pipeline_summary.to_dict(),
        "total": pipeline_summary.to_dict(),
    }
    if opencode:
        opencode_summary = UsageSummary(**_usage_summary_fields(opencode))
        total.merge(opencode_summary)
        combined["opencode"] = opencode
        combined["total"] = total.to_dict()
    return combined


def merge_usage_summaries(usages: List[Dict[str, Any]]) -> UsageSummary:
    total = UsageSummary()
    for usage in usages:
        if usage:
            total.merge(UsageSummary(**_usage_summary_fields(usage)))
    return total


def merge_nested_usage_dicts(usages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-query usage dicts (flat legacy or nested) into a run-level summary."""
    pipeline_parts: List[Dict[str, Any]] = []
    opencode_parts: List[Dict[str, Any]] = []

    for usage in usages:
        if not usage:
            continue
        if "total" in usage:
            if usage.get("pipeline"):
                pipeline_parts.append(usage["pipeline"])
            if usage.get("opencode"):
                opencode_parts.append(usage["opencode"])
        else:
            pipeline_parts.append(usage)

    pipeline_total = merge_usage_summaries(pipeline_parts)
    opencode_total = merge_usage_summaries(opencode_parts)
    grand_total = pipeline_total.copy()
    grand_total.merge(opencode_total)

    merged: Dict[str, Any] = {"total": grand_total.to_dict()}
    if pipeline_parts:
        merged["pipeline"] = pipeline_total.to_dict()
    if opencode_parts:
        merged["opencode"] = opencode_total.to_dict()
    return merged


def build_opencode_run_usage(
    results: List[Dict[str, Any]],
    tracker: UsageTracker,
) -> Dict[str, Any]:
    """Combine global pipeline tracker usage with per-query OpenCode agent usage."""
    from src.result_json import result_usage

    opencode_parts = [
        result_usage(r).get("opencode")
        for r in results
        if result_usage(r).get("opencode")
    ]
    opencode_total = merge_usage_summaries(opencode_parts)
    grand_total = tracker.total.copy()
    grand_total.merge(opencode_total)

    payload: Dict[str, Any] = {
        "pipeline": tracker.to_dict(),
        "total": grand_total.to_dict(),
    }
    if opencode_parts:
        payload["opencode"] = opencode_total.to_dict()
    return payload


def _pipeline_usage_dict(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Return flat pipeline usage from a per-query or run-level usage dict."""
    pipeline = usage.get("pipeline") or {}
    if isinstance(pipeline, dict) and "total" in pipeline:
        return dict(pipeline.get("total") or {})
    return dict(pipeline) if isinstance(pipeline, dict) else {}


def format_nested_usage_line(usage: Dict[str, Any]) -> str:
    """Format nested usage for console output."""
    opencode = usage.get("opencode")
    if not opencode:
        total = usage.get("total") or usage
        return format_usage_line(UsageSummary(**_usage_summary_fields(total)))

    pipeline = _pipeline_usage_dict(usage)
    agent_line = format_usage_line(UsageSummary(**_usage_summary_fields(opencode)))
    pipe_line = format_usage_line(UsageSummary(**_usage_summary_fields(pipeline)))
    return f"agent: {agent_line} · pipeline: {pipe_line}"
