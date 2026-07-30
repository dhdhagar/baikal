"""Budgeted SQL and passage access for the OpenCode baseline."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from src.passage_expansion import MAX_ADDED_PASSAGES, grep_passage_ids
from src.sql_db import execute_sql
from src.utils import load_json, save_json

CONFIG_FILENAME = "lake_tool_config.json"
STATE_FILENAME = "lake_tool_state.json"


def _config_path(workdir: str) -> str:
    return os.path.join(workdir, CONFIG_FILENAME)


def _state_path(workdir: str) -> str:
    return os.path.join(workdir, STATE_FILENAME)


def _state_lock_path(state_path: str) -> str:
    return f"{state_path}.lock"


def _normalize_findings(state: Dict[str, Any]) -> None:
    findings = list(state.get("findings") or [])
    findings.sort(key=lambda item: int(item.get("step") or 0), reverse=True)
    state["findings"] = findings


def _load_state_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if not text.strip():
        return init_state()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        payload, end = decoder.raw_decode(text.lstrip())
        if text.lstrip()[end:].strip():
            # Recover the first valid JSON object from a corrupted multi-write file.
            pass
    if not isinstance(payload, dict):
        raise ValueError(f"Expected lake tool state object in {path}")
    return payload


def _write_state_payload_atomic(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@contextmanager
def _state_file_lock(state_path: str, *, exclusive: bool) -> Iterator[None]:
    lock_path = _state_lock_path(state_path)
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lock_handle:
        fcntl.flock(
            lock_handle.fileno(),
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_state(workdir: str) -> Iterator[Dict[str, Any]]:
    """Load, mutate, and atomically persist lake tool state under an exclusive lock."""
    path = _state_path(workdir)
    with _state_file_lock(path, exclusive=True):
        if os.path.isfile(path):
            state = _load_state_payload(path)
        else:
            state = init_state()
        yield state
        _normalize_findings(state)
        _write_state_payload_atomic(path, state)


def load_config(workdir: str) -> Dict[str, Any]:
    path = _config_path(workdir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing lake tool config: {path}")
    return load_json(path)


def init_state() -> Dict[str, Any]:
    return {
        "step": 1,
        "attempts_this_step": 0,
        "findings": [],
        "sql_attempts": {},
        "grep_queries": [],
        "retrieved_passage_ids": [],
    }


def load_state(workdir: str) -> Dict[str, Any]:
    path = _state_path(workdir)
    if not os.path.isfile(path):
        return init_state()
    with _state_file_lock(path, exclusive=False):
        return _load_state_payload(path)


def save_state(workdir: str, state: Dict[str, Any]) -> None:
    path = _state_path(workdir)
    with _state_file_lock(path, exclusive=True):
        _normalize_findings(state)
        _write_state_payload_atomic(path, state)


def write_config(workdir: str, config: Dict[str, Any]) -> str:
    path = _config_path(workdir)
    save_json(path, config)
    return path


def _emit(payload: Dict[str, Any], *, ok: bool = True) -> int:
    payload = {"ok": ok, **payload}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if ok else 1


def _cluster_passage_ids(config: Dict[str, Any]) -> set[str]:
    return {str(pid) for pid in (config.get("cluster_passage_ids") or [])}


def _track_retrieved_passage(
    state: Dict[str, Any],
    config: Dict[str, Any],
    passage_id: str,
) -> None:
    pid = passage_id.strip().upper()
    if not pid.startswith("P"):
        pid = f"P{pid}"
    if pid in _cluster_passage_ids(config):
        return
    retrieved = list(state.get("retrieved_passage_ids") or [])
    if pid not in retrieved:
        retrieved.append(pid)
    state["retrieved_passage_ids"] = retrieved


def _passage_rank_by_id(config: Dict[str, Any]) -> Optional[Dict[str, int]]:
    rank_by = config.get("passage_rank_by_id")
    if not rank_by:
        return None
    return {str(k): int(v) for k, v in rank_by.items()}


def _budget_exhausted(state: Dict[str, Any], budget: int) -> bool:
    return int(state.get("step") or 1) > budget


def cmd_status(workdir: str) -> int:
    config = load_config(workdir)
    state = load_state(workdir)
    budget = int(config["budget"])
    return _emit(
        {
            "step": state["step"],
            "budget": budget,
            "attempts_this_step": state["attempts_this_step"],
            "max_sql_attempts": int(config["max_sql_attempts"]),
            "findings_recorded": len(state["findings"]),
            "budget_remaining": max(0, budget - len(state["findings"])),
        }
    )


def cmd_sql(workdir: str, sql: str) -> int:
    config = load_config(workdir)
    budget = int(config["budget"])
    max_attempts = int(config["max_sql_attempts"])

    with _exclusive_state(workdir) as state:
        step = int(state["step"])
        if _budget_exhausted(state, budget):
            return _emit(
                {
                    "error": f"Budget exhausted ({budget} findings).",
                    "step": step,
                },
                ok=False,
            )

        if int(state["attempts_this_step"]) >= max_attempts:
            return _emit(
                {
                    "error": (
                        f"Max SQL attempts ({max_attempts}) reached for step {step}. "
                        "Commit the finding before starting another step."
                    ),
                    "step": step,
                },
                ok=False,
            )

        sqlite_path = config["sqlite_path"]
        conn = sqlite3.connect(sqlite_path)
        try:
            execution = execute_sql(conn, sql)
        finally:
            conn.close()

        attempt = {
            "attempt": int(state["attempts_this_step"]) + 1,
            "sql": sql.strip(),
            "ok": bool(execution.get("ok")),
            "row_count": int(execution.get("row_count", 0) or 0),
            "error": execution.get("error"),
            "rows": execution.get("rows") or [],
        }
        step_key = str(step)
        step_attempts = list((state.get("sql_attempts") or {}).get(step_key) or [])
        step_attempts.append(attempt)
        state["sql_attempts"] = dict(state.get("sql_attempts") or {})
        state["sql_attempts"][step_key] = step_attempts
        state["attempts_this_step"] = int(state["attempts_this_step"]) + 1

        return _emit(
            {
                "step": step,
                "attempt": attempt["attempt"],
                "sql": attempt["sql"],
                "execution": execution,
            }
        )


def _execution_from_attempt(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": bool(item.get("ok")),
        "error": item.get("error"),
        "row_count": int(item.get("row_count", 0) or 0),
        "rows": item.get("rows") or [],
    }


def _pick_sql_for_commit(
    step_attempts: List[Dict[str, Any]],
    sql_override: Optional[str],
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if sql_override:
        for item in reversed(step_attempts):
            if item.get("sql") == sql_override:
                return sql_override, _execution_from_attempt(item)
        return sql_override, None

    for item in reversed(step_attempts):
        if item.get("ok") and int(item.get("row_count", 0) or 0) > 0:
            return item.get("sql"), _execution_from_attempt(item)

    if step_attempts:
        last = step_attempts[-1]
        return last.get("sql"), _execution_from_attempt(last)
    return None, None


def cmd_commit(
    workdir: str,
    *,
    sub_question: str,
    answer: str,
    sql: Optional[str],
    needs_sql: bool,
) -> int:
    config = load_config(workdir)
    budget = int(config["budget"])

    with _exclusive_state(workdir) as state:
        step = int(state["step"])
        if _budget_exhausted(state, budget):
            return _emit(
                {"error": f"Budget exhausted ({budget} findings).", "step": step},
                ok=False,
            )

        step_key = str(step)
        step_attempts = list((state.get("sql_attempts") or {}).get(step_key) or [])
        sql_value, execution = _pick_sql_for_commit(step_attempts, sql)

        if needs_sql and not sql_value:
            return _emit(
                {
                    "error": "SQL-backed finding requires at least one sql attempt.",
                    "step": step,
                },
                ok=False,
            )

        finding = {
            "step": step,
            "sub_question": sub_question,
            "answer": answer,
            "needs_sql": needs_sql,
            "sql": sql_value if needs_sql else None,
            "execution": execution if needs_sql else None,
            "failed_sql_attempts": [
                a
                for a in step_attempts
                if not a.get("ok") or int(a.get("row_count", 0) or 0) == 0
            ]
            if needs_sql
            else None,
        }
        findings = list(state.get("findings") or [])
        findings.append(finding)
        state["findings"] = findings
        state["step"] = step + 1
        state["attempts_this_step"] = 0

        return _emit({"finding": finding, "next_step": state["step"]})


def cmd_passage(workdir: str, passage_id: str) -> int:
    config = load_config(workdir)
    path = config.get("passage_descriptions_path")
    if not path:
        return _emit({"error": "Passages are disabled for this run."}, ok=False)

    descriptions = load_json(path)
    pid = passage_id.strip().upper()
    if not pid.startswith("P"):
        pid = f"P{pid}"

    record = descriptions.get(pid)
    if not record:
        return _emit({"error": f"Passage not found: {pid}"}, ok=False)

    with _exclusive_state(workdir) as state:
        _track_retrieved_passage(state, config, pid)
        return _emit({"passage_id": pid, "passage": record})


def cmd_grep_passages(workdir: str, keywords: List[str]) -> int:
    config = load_config(workdir)

    if not config.get("allow_passage_grep"):
        return _emit(
            {"error": "Passage grep is disabled for this run."},
            ok=False,
        )

    path = config.get("passage_descriptions_path")
    if not path:
        return _emit({"error": "Passages are disabled for this run."}, ok=False)

    cleaned = [str(k).strip() for k in keywords if str(k).strip()]
    if not cleaned:
        return _emit({"error": "At least one keyword is required."}, ok=False)

    with _exclusive_state(workdir) as state:
        exclude_ids = _cluster_passage_ids(config)
        exclude_ids.update(state.get("retrieved_passage_ids") or [])

        matched_ids = grep_passage_ids(
            path,
            cleaned,
            exclude_ids=exclude_ids,
            max_results=MAX_ADDED_PASSAGES,
            rank_by=_passage_rank_by_id(config),
        )

        grep_queries = list(state.get("grep_queries") or [])
        grep_queries.append({"keywords": cleaned, "matched_ids": matched_ids})
        state["grep_queries"] = grep_queries

        for pid in matched_ids:
            _track_retrieved_passage(state, config, pid)

        return _emit(
            {
                "keywords": cleaned,
                "matched_ids": matched_ids,
                "hint": (
                    "Load full passage text with the passage command before citing. "
                    "When SQL was used, prefer keywords derived from SQL result values."
                ),
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenCode data-lake tool")
    parser.add_argument("--workdir", required=True, help="Per-query working directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show budget and step status")

    sql_p = sub.add_parser("sql", help="Run a budgeted SELECT query")
    sql_p.add_argument("query", help="SQL SELECT statement")

    commit_p = sub.add_parser("commit", help="Record a finding and advance the budget step")
    commit_p.add_argument("--sub-question", required=True)
    commit_p.add_argument("--answer", required=True)
    commit_p.add_argument("--sql", default=None, help="SQL used for this finding")
    commit_p.add_argument(
        "--no-sql",
        action="store_true",
        help="Passage-only finding (no SQL evidence)",
    )

    passage_p = sub.add_parser("passage", help="Load a passage by id (e.g. P42)")
    passage_p.add_argument("passage_id")

    grep_p = sub.add_parser(
        "grep-passages",
        help="Search passage descriptions for keywords (OR semantics)",
    )
    grep_p.add_argument(
        "--keywords",
        nargs="+",
        required=True,
        help="One or more search keywords/phrases",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = os.path.abspath(args.workdir)

    if args.command == "status":
        return cmd_status(workdir)
    if args.command == "sql":
        return cmd_sql(workdir, args.query)
    if args.command == "commit":
        return cmd_commit(
            workdir,
            sub_question=args.sub_question,
            answer=args.answer,
            sql=args.sql,
            needs_sql=not args.no_sql,
        )
    if args.command == "passage":
        return cmd_passage(workdir, args.passage_id)
    if args.command == "grep-passages":
        return cmd_grep_passages(workdir, args.keywords)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
