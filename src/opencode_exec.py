"""OpenCode execution for dpr_discovery sub-questions (--opencode_exec)."""

from __future__ import annotations

import os
import subprocess
from argparse import Namespace
from typing import Any, Dict, List, Optional

from src.cli_ui import get_ui
from src.opencode import (
    _lake_tool_cmd,
    _watch_opencode_lake_state,
    invoke_opencode,
    setup_lake_workspace,
)
from src.opencode_lake_tool import load_state
from src.passage_expansion import MAX_ADDED_PASSAGES
from src.sql_db import build_cluster_sqlite_artifact, load_cluster_table_metas
from src.subquestions import format_cluster_context
from src.utils import ensure_dir, log, query_passage_descriptions_path


def opencode_expansion_trigger(lake_state: Dict[str, Any]) -> Optional[str]:
    """Classify how OpenCode expanded cluster evidence, if at all."""
    grep_queries = list(lake_state.get("grep_queries") or [])
    if grep_queries:
        had_successful_sql = _lake_state_had_successful_sql(lake_state)
        if had_successful_sql:
            return "opencode_grep_after_sql"
        return "opencode_grep_passages"
    if lake_state.get("retrieved_passage_ids"):
        if _lake_state_had_successful_sql(lake_state):
            return "opencode_passage_after_sql"
        return "opencode_passage_load"
    return None


def _lake_state_had_successful_sql(lake_state: Dict[str, Any]) -> bool:
    for attempts in (lake_state.get("sql_attempts") or {}).values():
        for attempt in attempts or []:
            if attempt.get("ok") and int(attempt.get("row_count", 0) or 0) > 0:
                return True
    return False


def expansion_grep_keywords(expansion_meta: Dict[str, Any]) -> List[str]:
    """Keywords from LLM expansion or OpenCode grep-passages calls."""
    llm_decision = expansion_meta.get("llm_decision") or {}
    keywords = [str(k).strip() for k in (llm_decision.get("grep_keywords") or []) if str(k).strip()]
    if keywords:
        return keywords
    seen: List[str] = []
    for query in expansion_meta.get("grep_queries") or []:
        for keyword in query.get("keywords") or []:
            text = str(keyword).strip()
            if text and text not in seen:
                seen.append(text)
    return seen


def build_opencode_subquestion_prompt(
    *,
    user_query: str,
    sub_question: str,
    cluster: dict,
    cluster_schema_path: str,
    sqlite_db_path: str,
    query_dir: str,
    max_sql_attempts: int,
    use_passages: bool,
    passage_descriptions_path: Optional[str],
    expand_cluster: bool,
) -> str:
    cluster_block = format_cluster_context(cluster)
    table_ids = [t.get("table_id") for t in cluster.get("tables", []) if t.get("table_id")]
    tables_line = ", ".join(table_ids) if table_ids else "(none — passage-only cluster)"

    passage_block = ""
    if use_passages and passage_descriptions_path:
        passage_block = f"""
Passage descriptions index: {passage_descriptions_path}
Load a passage by id: {_lake_tool_cmd(query_dir, "passage", "P42")}
"""

    expansion_block = ""
    if expand_cluster and use_passages and passage_descriptions_path:
        expansion_block = f"""
Optional cluster expansion (before you commit):
- If cluster passages are insufficient, you MAY search for additional passages.
- When tables exist and you run SQL: inspect SQL results FIRST, then grep using
  concrete terms from returned values (names, places, events — not generic words).
- When answering passage-only without SQL: grep from the sub-question and gaps
  in current passage coverage.
- Search (max {MAX_ADDED_PASSAGES} new passages per grep, OR keyword matching):
  {_lake_tool_cmd(query_dir, "grep-passages", "--keywords", '"keyword"', '"another"')}
- Load promising hits with passage before citing them in your answer.
- Do not grep before reviewing SQL results when SQL was used.
"""

    commit_sql_example = _lake_tool_cmd(
        query_dir,
        "commit",
        "--sub-question",
        f'"{sub_question[:80]}..."',
        "--answer",
        '"Answer with citations [T809] or [P42]."',
    )
    commit_passage_example = _lake_tool_cmd(
        query_dir,
        "commit",
        "--sub-question",
        f'"{sub_question[:80]}..."',
        "--answer",
        '"Passage-only answer [P42]."',
        "--no-sql",
    )

    citation_rules = """Citation rules (required in every --answer):
- Ground every factual claim in lake evidence only.
- Cite table IDs in brackets for SQL-backed claims (e.g., [T809]).
- Cite passage IDs in brackets for passage claims (e.g., [P42])."""

    return f"""You are a research agent answering ONE sub-question for a deep-research report.

Research question:
{user_query}

Sub-question to answer (use this exact text in commit --sub-question):
{sub_question}

SELECTED CLUSTER (tables and passages in scope):
{cluster_block}

Cluster schema (SQLite contains ONLY these tables: {tables_line}):
{cluster_schema_path}
SQLite database: {sqlite_db_path}
Working directory: {query_dir}
{passage_block}{expansion_block}
You have budget for exactly ONE finding. Run up to {max_sql_attempts} SQL attempts, then commit.

Run SQL (table names may be unquoted or double-quoted):
{_lake_tool_cmd(query_dir, "sql", '"SELECT * FROM T809 LIMIT 10"')}

Record your finding (required — use the sub-question above verbatim):
{commit_sql_example}
{commit_passage_example}

{citation_rules}

Check status:
{_lake_tool_cmd(query_dir, "status")}

Use only lake evidence. Do not use outside knowledge. Do not modify the lake.

Autonomous execution (required):
- You are running fully unattended; no human is available.
- Never ask questions or wait for user input.
- Commit exactly one finding for the sub-question above, then stop.
"""


def run_opencode_subquestion(
    args: Namespace,
    *,
    user_query: str,
    sub_question: str,
    cluster: dict,
    table_ids: List[str],
    query_dir: str,
    step: int,
    passage_rank_by_id: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run OpenCode to answer one pipeline sub-question (budget=1).

    Returns finding fields plus opencode_meta and lake_state. On commit failure,
    returns commit_failed=True with empty answer and failure_reason set.
    """
    workdir = ensure_dir(os.path.join(query_dir, f"opencode_step_{step:03d}"))
    table_metas = (
        load_cluster_table_metas(args.tables_lake_dir, table_ids) if table_ids else {}
    )
    sqlite_path = os.path.join(workdir, "cluster.sqlite")
    schema_string = build_cluster_sqlite_artifact(table_metas, sqlite_path)
    schema_path = os.path.join(workdir, "cluster_schema.txt")
    with open(schema_path, "w", encoding="utf-8") as handle:
        handle.write(schema_string)

    cluster_passage_ids = [
        p.get("passage_id")
        for p in cluster.get("passages", [])
        if p.get("passage_id")
    ]
    allow_grep = bool(
        args.expand_cluster and args.use_passages and args.passage_descriptions_path
    )
    setup_lake_workspace(
        args,
        workdir,
        sqlite_path,
        budget=1,
        cluster_passage_ids=cluster_passage_ids,
        allow_passage_grep=allow_grep,
        passage_rank_by_id=passage_rank_by_id,
    )

    passage_path = (
        query_passage_descriptions_path(workdir, args.passage_type)
        if args.use_passages
        else None
    )

    prompt = build_opencode_subquestion_prompt(
        user_query=user_query,
        sub_question=sub_question,
        cluster=cluster,
        cluster_schema_path=schema_path,
        sqlite_db_path=sqlite_path,
        query_dir=workdir,
        max_sql_attempts=args.max_sql_attempts,
        use_passages=args.use_passages,
        passage_descriptions_path=passage_path,
        expand_cluster=allow_grep,
    )
    prompt_path = os.path.join(workdir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)

    ui = get_ui()
    if ui.rich_cli:
        ui.set_status("Running OpenCode")

    model = f"{args.llm_provider}/{args.llm_model}"
    log(
        f"[opencode_exec] Step {step}: running sub-question with {model}",
        silent=args.silent,
    )

    opencode_meta: Optional[Dict[str, Any]] = None
    try:
        with _watch_opencode_lake_state(workdir, budget=1):
            opencode_meta = invoke_opencode(
                model=model,
                prompt_path=prompt_path,
                cwd=workdir,
                silent=args.silent,
            )
    except subprocess.TimeoutExpired as exc:
        opencode_meta = {
            "model": model,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "status": "timeout",
            "prompt_path": prompt_path,
        }

    if opencode_meta:
        stdout_path = os.path.join(workdir, "opencode_stdout.txt")
        stderr_path = os.path.join(workdir, "opencode_stderr.txt")
        with open(stdout_path, "w", encoding="utf-8") as handle:
            handle.write(opencode_meta.get("stdout") or "")
        with open(stderr_path, "w", encoding="utf-8") as handle:
            handle.write(opencode_meta.get("stderr") or "")
        opencode_meta["stdout_path"] = stdout_path
        opencode_meta["stderr_path"] = stderr_path
        opencode_meta["prompt_path"] = prompt_path

    state = load_state(workdir)
    findings = list(state.get("findings") or [])
    if not findings:
        failure_reason = "no_commit"
        if opencode_meta:
            status = opencode_meta.get("status")
            if status == "timeout":
                failure_reason = "timeout"
            elif status == "failed":
                failure_reason = "opencode_failed"
        log(
            f"[opencode_exec] Step {step}: OpenCode did not commit a finding "
            f"({failure_reason}).",
            silent=args.silent,
        )
        return {
            "answer": "",
            "needs_sql": False,
            "sql": None,
            "execution": None,
            "failed_sql_attempts": [],
            "opencode_meta": opencode_meta,
            "lake_state": state,
            "commit_failed": True,
            "failure_reason": failure_reason,
        }

    finding = findings[0]
    return {
        "answer": finding.get("answer") or "",
        "needs_sql": bool(finding.get("needs_sql")),
        "sql": finding.get("sql"),
        "execution": finding.get("execution"),
        "failed_sql_attempts": finding.get("failed_sql_attempts"),
        "opencode_meta": opencode_meta,
        "lake_state": state,
        "commit_failed": False,
    }
