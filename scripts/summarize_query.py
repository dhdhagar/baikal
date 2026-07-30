#!/usr/bin/env python3
"""Render a collaborator-facing markdown report for a single query run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULT_FILENAME = "result.json"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _result_budget(result: Dict[str, Any]) -> int:
    research_quality = (result.get("summary") or {}).get("research_quality") or {}
    budget = research_quality.get("budget")
    if budget:
        return int(budget)
    run_status = (result.get("summary") or {}).get("status") or {}
    completed = run_status.get("budget_steps_completed")
    if completed:
        return int(completed)
    findings = result.get("findings")
    if isinstance(findings, list) and findings:
        return max(int(item.get("step") or 0) for item in findings)
    return 0


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _md_codeblock(language: str, text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}"


def _format_sql_attempt(attempt: Dict[str, Any]) -> str:
    lines = [f"#### Attempt {attempt.get('attempt', '?')}"]
    sql = attempt.get("sql")
    if sql:
        lines.append(_md_codeblock("sql", str(sql)))
    payload = {
        "ok": attempt.get("ok"),
        "row_count": attempt.get("row_count"),
        "error": attempt.get("error"),
        "rows": attempt.get("rows"),
    }
    lines.append("**Response:**")
    lines.append(_md_codeblock("json", _json_block(payload)))
    return "\n\n".join(lines)


def _format_execution(execution: Dict[str, Any]) -> str:
    return _md_codeblock("json", _json_block(execution))


def _format_cluster(cluster: Optional[Dict[str, Any]]) -> str:
    if not cluster:
        return "_No cluster metadata._"
    lines = [
        f"- **Cluster ID:** {cluster.get('id')}",
        f"- **Description:** {cluster.get('description') or ''}",
    ]
    table_ids = cluster.get("table_ids") or []
    if table_ids:
        tables = [
            f"{table_id} ({title})" if title else str(table_id)
            for table_id, title in table_ids
        ]
        lines.append(f"- **Tables:** {', '.join(tables)}")
    passage_ids = cluster.get("passage_ids") or []
    if passage_ids:
        passages = [
            f"{passage_id} ({title})" if title else str(passage_id)
            for passage_id, title in passage_ids
        ]
        lines.append(f"- **Passages:** {', '.join(passages)}")
    return "\n".join(lines)


def _format_iteration(step_payload: Dict[str, Any]) -> str:
    step = step_payload.get("step", "?")
    if step_payload.get("skipped"):
        return f"## Step {step}\n\n_Skipped._\n"

    lines = [f"## Step {step}", ""]
    lines.append("### Selected cluster")
    lines.append(_format_cluster(step_payload.get("selected_cluster")))
    lines.append("")
    lines.append("### Sub-question")
    lines.append(str(step_payload.get("sub_question") or "").strip() or "_Empty._")
    lines.append("")
    lines.append(f"**Needs SQL:** `{step_payload.get('needs_sql')}`")

    failed_attempts = step_payload.get("failed_sql_attempts") or []
    if failed_attempts:
        lines.append("")
        lines.append("### SQL attempts")
        lines.extend(_format_sql_attempt(attempt) for attempt in failed_attempts)

    sql = step_payload.get("sql")
    if sql:
        lines.append("")
        lines.append("### Final SQL")
        lines.append(_md_codeblock("sql", str(sql)))

    execution = step_payload.get("execution")
    if execution:
        lines.append("")
        lines.append("### Final SQL response")
        lines.append(_format_execution(execution))

    lines.append("")
    lines.append("### Answer")
    lines.append(str(step_payload.get("answer") or "").strip() or "_Empty._")
    lines.append("")
    return "\n".join(lines)


def _discover_steps(query_dir: str, budget: int) -> List[int]:
    if budget > 0:
        return list(range(1, budget + 1))
    steps: List[int] = []
    for name in os.listdir(query_dir):
        if not name.startswith("iteration_") or not name.endswith(".json"):
            continue
        try:
            steps.append(int(name[len("iteration_") : -len(".json")]))
        except ValueError:
            continue
    return sorted(steps)


def summarize_query(query_dir: str) -> str:
    """Build a markdown report for a query artifact directory."""
    query_dir = os.path.abspath(query_dir)
    result_path = os.path.join(query_dir, RESULT_FILENAME)
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f"Missing {RESULT_FILENAME} in {query_dir}")

    result = _load_json(result_path)
    budget = _result_budget(result)
    steps = _discover_steps(query_dir, budget)

    lines = [
        "# Query investigation report",
        "",
        f"**Path:** `{query_dir}`",
        f"**Query ID:** {result.get('query_id', os.path.basename(query_dir))}",
        f"**Method:** {result.get('method', '')}",
        f"**Coverage:** {result.get('coverage', '')}",
        f"**Budget:** {budget or len(steps)}",
        "",
        "## Research question",
        "",
        str(result.get("user_query") or "").strip(),
        "",
        "## Final report",
        "",
        str(result.get("answer") or "").strip(),
        "",
        "## Trajectory",
        "",
    ]

    if not steps:
        lines.append("_No iteration artifacts found._")
        lines.append("")
        return "\n".join(lines)

    for step in steps:
        iteration_path = os.path.join(query_dir, f"iteration_{step:03d}.json")
        if not os.path.isfile(iteration_path):
            lines.append(f"## Step {step}")
            lines.append("")
            lines.append("_Missing iteration artifact._")
            lines.append("")
            continue
        lines.append(_format_iteration(_load_json(iteration_path)))

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a markdown report for a single query run directory."
    )
    parser.add_argument(
        "query_dir",
        type=str,
        help="Query artifact directory containing result.json and iteration_*.json files.",
    )
    args = parser.parse_args()

    query_dir = os.path.abspath(args.query_dir)
    if not os.path.isdir(query_dir):
        print(f"Not a directory: {query_dir}", file=sys.stderr)
        return 1

    try:
        print(summarize_query(query_dir))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
